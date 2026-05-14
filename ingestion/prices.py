"""
ingestion/prices.py — Incremental OHLCV and FX ingestion from FMP into Postgres.

This module powers two nightly jobs:

* :func:`run_price_ingestion` — Update the ``prices`` table with the
  most recent bars for a list of tickers. Skips dates already present,
  so re-running the job is cheap.
* :func:`run_fx_ingestion` — Same idea for the ``fx_rates`` table,
  using EUR-quoted currency pairs derived from ``transactions.csv``.

When run as a script (``python -m ingestion.prices``), the module
refreshes every ticker / FX pair already present in the database.

Logging
-------
The module logs at INFO level. In production, gunicorn captures
stdout/stderr and the platform's log viewer surfaces these entries
with their timestamps and severity.
"""

import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

import config
from db.connection import Session, engine
from db.models import FxRate, IngestionLog, Price, create_all_tables
from ingestion.fmp_client import fetch_fx_history, fetch_prices, fetch_prices_batch

logger = logging.getLogger(__name__)

# Number of tickers per parallel HTTP batch. FMP allows ~10 requests/s
# on the standard plan; 5 balances throughput vs back-pressure.
BATCH_SIZE: int = 5

# Maximum concurrent worker threads for the FMP fetch stage.
MAX_WORKERS: int = 8

# Minimum valid price. Filters out delisted penny stocks whose adjusted
# close decays to absurd values like 5.68e-10 — those rows would crash
# PostgreSQL's numeric type checks downstream.
MIN_PRICE: float = 0.0001


def _clean_rows(rows: list[dict]) -> list[dict]:
    """Filter out rows with anomalous prices that would crash PostgreSQL."""
    clean = []
    for r in rows:
        try:
            close = float(r.get("close") or 0)
            if close < MIN_PRICE:
                continue
            # Also cap volume at PostgreSQL bigint max
            if r.get("volume") and int(r["volume"]) > 9_223_372_036_854_775_807:
                r["volume"] = 0
            clean.append(r)
        except (TypeError, ValueError):
            continue
    return clean


def _upsert_prices(rows: list[dict], session) -> None:
    """Bulk upsert via PostgreSQL ON CONFLICT DO UPDATE."""
    rows = _clean_rows(rows)
    if not rows:
        return
    stmt = insert(Price).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={
            "open":      stmt.excluded.open,
            "high":      stmt.excluded.high,
            "low":       stmt.excluded.low,
            "close":     stmt.excluded.close,
            "adj_close": stmt.excluded.adj_close,
            "volume":    stmt.excluded.volume,
        },
    )
    session.execute(stmt)


def _latest_dates(tickers: list[str], session) -> dict:
    if not tickers:
        return {}
    result = session.execute(
        text(
            "SELECT ticker, MAX(date)::text FROM prices "
            "WHERE ticker = ANY(:t) GROUP BY ticker"
        ),
        {"t": tickers},
    ).fetchall()
    return {row[0]: row[1] for row in result}


def run_price_ingestion(
    tickers: list[str],
    years_back: int | None = None,
) -> tuple[int, int]:
    create_all_tables(engine)
    years         = years_back or config.YEARS_BACK
    default_start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end           = datetime.today().strftime("%Y-%m-%d")
    t0            = time.time()
    success = failed = 0
    total   = len(tickers)

    logger.info(
        "[prices] ingesting %d tickers from %s to %s",
        total, default_start, end,
    )

    with Session() as s:
        latest = _latest_dates(tickers, s)

    ticker_starts: dict[str, str] = {}
    for t in tickers:
        ld = latest.get(t)
        if ld:
            next_day = (
                datetime.strptime(ld, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            ticker_starts[t] = next_day
        else:
            ticker_starts[t] = default_start

    by_start: dict[str, list[str]] = defaultdict(list)
    for t, s in ticker_starts.items():
        if s <= end:
            by_start[s].append(t)

    completed = 0

    for start_date, group in by_start.items():
        batches = [group[i : i + BATCH_SIZE] for i in range(0, len(group), BATCH_SIZE)]

        def _fetch(batch: list[str]) -> tuple[list[dict], int, int]:
            try:
                rows = fetch_prices_batch(batch, start_date, end)
                if not rows:
                    rows = []
                    for t in batch:
                        rows.extend(fetch_prices(t, start_date, end))
                fetched = {r["ticker"] for r in rows}
                ok = len([t for t in batch if t in fetched])
                err = len(batch) - ok
                return rows, ok, err
            except Exception as exc:
                logger.exception("[prices] batch error: %s", exc)
                return [], 0, len(batch)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(_fetch, b): b for b in batches}
            with Session() as s:
                for fut in as_completed(futs):
                    rows, ok, err = fut.result()
                    success += ok
                    failed  += err
                    try:
                        _upsert_prices(rows, s)
                    except Exception as e:
                        logger.exception(
                            "[prices] upsert error (skipping batch): %s", e
                        )
                        s.rollback()
                    completed += len(futs[fut])
                    # Inline progress percentages stay on stdout for
                    # interactive runs; production logs receive a single
                    # completion line below.
                    pct = int(completed / max(total, 1) * 100)
                    print(f"\r  [prices] {pct}%  ", end="", flush=True)
                s.commit()

    print()  # newline after the carriage-return progress display
    duration = time.time() - t0

    with Session() as s:
        s.add(IngestionLog(
            job_type="prices",
            tickers_attempted=total,
            tickers_success=success,
            tickers_failed=failed,
            duration_seconds=round(duration, 1),
            notes=f"{default_start} → {end}",
        ))
        s.commit()

    logger.info(
        "[prices] done in %.0fs — %d ok / %d failed",
        duration, success, failed,
    )
    return success, failed


def run_fx_ingestion(
    currency_pairs: list[str],
    years_back: int | None = None,
) -> None:
    create_all_tables(engine)
    years = years_back or config.YEARS_BACK
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    with Session() as s:
        for pair in currency_pairs:
            rows = fetch_fx_history(pair, start, end)
            if not rows:
                logger.warning("[fx] no data returned for %s", pair)
                continue
            stmt = insert(FxRate).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["pair", "date"],
                set_={"rate": stmt.excluded.rate},
            )
            s.execute(stmt)
        s.commit()

    logger.info("[fx] done — pairs: %s", currency_pairs)


if __name__ == "__main__":
    # Bootstrap a basic logging config so script runs without app.py
    # still produce timestamped output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cli_tickers = sys.argv[1:] or None

    if cli_tickers:
        run_price_ingestion(cli_tickers)
    else:
        with Session() as s:
            result = s.execute(
                text("SELECT DISTINCT ticker FROM prices")
            ).fetchall()
            db_tickers = [r[0] for r in result]

        if not db_tickers:
            logger.error("No tickers in DB yet. Run setup.py first.")
            sys.exit(1)

        run_price_ingestion(db_tickers)

        with Session() as s:
            result = s.execute(
                text("SELECT DISTINCT pair FROM fx_rates")
            ).fetchall()
            pairs = [r[0] for r in result]
        if pairs:
            run_fx_ingestion(pairs)
