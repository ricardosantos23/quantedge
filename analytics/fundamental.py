"""
analytics/fundamental.py — Cached fundamental scores backed by Postgres.

Two-tier flow
-------------
1. :func:`load_fundamentals_from_db` — Instant read of every cached row.
2. :func:`refresh_fundamentals`     — Fetch stale or missing tickers
   from the FMP API, score them, and upsert the result back into the
   ``fundamentals`` table.
3. :func:`compute_fundamentals`     — Public API used by ``app.py``;
   wraps the refresh step and returns display-formatted column names.

Cache freshness
---------------
Fundamentals are refreshed at most once per ``stale_hours`` (default
24h). They do not change intraday, so this keeps FMP usage low while
guaranteeing the dashboard never serves stale data older than one
business day.

Composite score
---------------
The ``fundamental_score`` column is a NaN-safe weighted average of five
quality signals, each cross-sectionally percentile-ranked:

* 0.25 ROIC
* 0.20 FCF margin
* 0.20 Revenue 3-year CAGR
* 0.20 Gross margin
* 0.15 Interest coverage

Missing signals contribute zero weight rather than dragging the score
down, so a company with strong ROIC/FCF but missing growth data still
ranks consistently.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from db.connection import Session
from db.models import Fundamental
from ingestion.fmp_client import fetch_fundamentals_one

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  SCORING  (identical to original fundamental.py — do not modify)
# ════════════════════════════════════════════════════════════════════

def _score_col(s: pd.Series, invert: bool = False) -> pd.Series:
    r = s.rank(pct=True, na_option="keep")
    return (1 - r) if invert else r


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    _numeric = [
        "roic", "fcf_margin", "rev_cagr", "gross_margin", "interest_coverage",
        "ev_ebitda", "pe_ratio", "pb_ratio", "net_margin", "operating_margin",
        "fcf_yield", "dcf_upside", "dcf_value", "dcf_price_val", "piotroski",
        "share_dilution", "capex_pct_rev",
    ]
    _integer = ["insider_buys", "insider_sells", "insider_signal"]

    for c in _numeric + _integer:
        if c not in df.columns:
            df[c] = np.nan

    # Force numeric dtype on every score-related column. Without this,
    # columns that come back from FMP as mixed None / NaN / float — or
    # columns we just filled with np.nan above — stay as `object` dtype.
    # pandas .rank() raises "No matching signature found" on object
    # dtype under Python 3.14 because the Cython algos.rank_1d lookup
    # has no signature for object Series.
    for c in _numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in _integer:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    df["roic"]              = df["roic"].clip(-0.3, 0.6)
    df["fcf_margin"]        = df["fcf_margin"].clip(-0.5, 0.6)
    df["rev_cagr"]          = df["rev_cagr"].clip(-0.3, 1.0)
    df["gross_margin"]      = df["gross_margin"].clip(0.0, 1.0)
    df["interest_coverage"] = df["interest_coverage"].clip(0.0, 50.0)
    df["capex_pct_rev"]     = df["capex_pct_rev"].clip(0.0, 0.5)

    df["s_roic"]     = _score_col(df["roic"])
    df["s_fcf"]      = _score_col(df["fcf_margin"])
    df["s_growth"]   = _score_col(df["rev_cagr"])
    df["s_gross"]    = _score_col(df["gross_margin"])
    df["s_solvency"] = _score_col(df["interest_coverage"])

    _w = {"s_roic": 0.25, "s_fcf": 0.20, "s_growth": 0.20,
          "s_gross": 0.20, "s_solvency": 0.15}

    def _wavg(row):
        tw = tv = 0.0
        for col, w in _w.items():
            v = row.get(col, np.nan)
            if pd.notna(v):
                tw += w
                tv += w * v
        return round(tv / tw, 2) if tw > 0 else np.nan

    df["fundamental_score"] = df.apply(_wavg, axis=1)

    # ── Pre-format display strings ────────────────────────────────────
    def pct(x, d: int = 1) -> str:
        try:
            v = float(x)
            return f"{v * 100:+.{d}f}%" if np.isfinite(v) else "N/A"
        except Exception:
            return "N/A"

    def num(x, d: int = 1) -> str:
        try:
            v = float(x)
            return f"{v:.{d}f}" if np.isfinite(v) else "N/A"
        except Exception:
            return "N/A"

    df["dcf_upside_fmt"]    = df["dcf_upside"].apply(pct)
    df["roic_fmt"]          = df["roic"].apply(pct)
    df["fcf_margin_fmt"]    = df["fcf_margin"].apply(pct)
    df["fcf_yield_fmt"]     = df["fcf_yield"].apply(pct)
    df["op_margin_fmt"]     = df["operating_margin"].apply(pct)
    df["net_margin_fmt"]    = df["net_margin"].apply(pct)
    df["gross_margin_fmt"]  = df["gross_margin"].apply(pct)
    df["rev_cagr_fmt"]      = df["rev_cagr"].apply(pct)
    df["capex_pct_rev_fmt"] = df["capex_pct_rev"].apply(lambda x: pct(x, 1))
    df["share_dilution_fmt"]= df["share_dilution"].apply(
        lambda x: f"{float(x)*100:+.1f}%/yr"
        if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    )
    df["ev_ebitda_fmt"]     = df["ev_ebitda"].apply(num)
    df["pe_ratio_fmt"]      = df["pe_ratio"].apply(num)
    df["piotroski_fmt"]     = df["piotroski"].apply(
        lambda x: f"{int(round(float(x)))}/9"
        if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    )
    df["insider_signal_fmt"]= df["insider_signal"].apply(
        lambda x: (f"+{int(x)} buys" if x > 0
                   else (f"{abs(int(x))} sells" if x < 0 else "neutral"))
    )
    return df


# ════════════════════════════════════════════════════════════════════
#  DB READ / WRITE
# ════════════════════════════════════════════════════════════════════

def load_fundamentals_from_db(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Read cached fundamentals from DB. Returns empty DataFrame if none found."""
    with Session() as s:
        q = s.query(Fundamental)
        if tickers:
            q = q.filter(Fundamental.ticker.in_(tickers))
        rows = q.all()
        records = [
            {c.name: getattr(r, c.name) for c in Fundamental.__table__.columns}
            for r in rows
        ]

    if not records:
        return pd.DataFrame(columns=["ticker", "fundamental_score"])

    return pd.DataFrame(records)


def _upsert_fundamentals(df: pd.DataFrame) -> None:
    """Write scored fundamentals back to DB."""
    now = datetime.utcnow()

    _num_cols = [
        "roic", "fcf_margin", "rev_cagr", "gross_margin", "interest_coverage",
        "ev_ebitda", "pe_ratio", "pb_ratio", "net_margin", "operating_margin",
        "fcf_yield", "dcf_upside", "dcf_value", "dcf_price_val", "piotroski",
        "share_dilution", "capex_pct_rev", "fundamental_score",
    ]
    _int_cols = ["insider_buys", "insider_sells", "insider_signal"]
    _fmt_cols = [
        "dcf_upside_fmt", "roic_fmt", "fcf_margin_fmt", "fcf_yield_fmt",
        "op_margin_fmt", "net_margin_fmt", "gross_margin_fmt", "rev_cagr_fmt",
        "capex_pct_rev_fmt", "share_dilution_fmt", "ev_ebitda_fmt",
        "pe_ratio_fmt", "piotroski_fmt", "insider_signal_fmt",
    ]

    rows = []
    for _, row in df.iterrows():
        r: dict = {"ticker": row["ticker"], "updated_at": now}
        for c in _num_cols:
            v = row.get(c)
            r[c] = float(v) if pd.notna(v) else None
        for c in _int_cols:
            v = row.get(c, 0)
            r[c] = int(v) if pd.notna(v) else 0
        for c in _fmt_cols:
            r[c] = str(row.get(c, "N/A"))
        rows.append(r)

    if not rows:
        return

    update_cols = _num_cols + _int_cols + _fmt_cols + ["updated_at"]

    with Session() as s:
        stmt = insert(Fundamental).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        s.execute(stmt)
        s.commit()


# ════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════

def refresh_fundamentals(
    tickers: list[str],
    max_workers: int = 4,
    stale_hours: int = 24,
) -> pd.DataFrame:
    """
    Refresh stale / missing fundamentals from FMP, write to DB.
    Tickers updated within stale_hours are skipped (served from cache).
    Returns DataFrame for all requested tickers.
    """
    now    = datetime.utcnow()
    cutoff = now - timedelta(hours=stale_hours)

    with Session() as s:
        existing = {
            r.ticker: r.updated_at
            for r in s.query(Fundamental.ticker, Fundamental.updated_at)
                      .filter(Fundamental.ticker.in_(tickers))
        }

    to_refresh = [
        t for t in tickers
        if t not in existing or (existing[t] and existing[t] < cutoff)
    ]

    if not to_refresh:
        return load_fundamentals_from_db(tickers)

    logger.info("[fundamentals] refreshing %d tickers", len(to_refresh))
    results = []
    done = 0
    total = len(to_refresh)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_fundamentals_one, t): t for t in to_refresh}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
            done += 1
            # Inline progress on stdout for interactive runs; one
            # completion line goes to the logger for production.
            print(f"\r  [fundamentals] {int(done / total * 100)}%  ", end="", flush=True)
    print()

    if results:
        scored = _compute_scores(
            pd.DataFrame(results).replace([np.inf, -np.inf], np.nan)
        )
        _upsert_fundamentals(scored)
        logger.info("[fundamentals] refresh complete — %d tickers updated", len(results))

    return load_fundamentals_from_db(tickers)


def compute_fundamentals(
    tickers: list[str],
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Drop-in replacement for original compute_fundamentals().
    Returns DataFrame with display-friendly column names.
    """
    df = refresh_fundamentals(tickers, max_workers=max_workers)

    return df.rename(columns={
        "dcf_upside_fmt":    "DCF Upside",
        "roic_fmt":          "ROIC",
        "fcf_margin_fmt":    "FCF Margin",
        "fcf_yield_fmt":     "FCF Yield",
        "op_margin_fmt":     "Op Margin",
        "net_margin_fmt":    "Net Margin",
        "gross_margin_fmt":  "Gross Margin",
        "rev_cagr_fmt":      "Rev CAGR 3yr",
        "capex_pct_rev_fmt": "CapEx % Rev",
        "share_dilution_fmt":"Share Dilution",
        "ev_ebitda_fmt":     "EV/EBITDA",
        "pe_ratio_fmt":      "P/E",
        "piotroski_fmt":     "Piotroski",
        "insider_signal_fmt":"Insider (90d)",
    })
