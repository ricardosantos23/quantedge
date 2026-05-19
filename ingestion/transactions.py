"""
ingestion/transactions.py — Persist the user's trade history in Postgres.

Why this exists
---------------
``transactions.csv`` holds personal trades and is deliberately
git-ignored (the repo is public), so the file never reaches the
deployed container. The Portfolio tab therefore rendered permanently
empty in production even though the app no longer crashed (see
PR #7 / #8). Storing trades in the ``transactions`` table — the same
pattern as ``watchlist`` / ``alerts`` — fixes that and survives
redeploys, because the Railway Postgres database is persistent.

Seeding
-------
On first boot, if the ``transactions`` table is empty, it is
auto-seeded from:

1. the ``TRANSACTIONS_CSV`` environment variable (raw CSV text *or*
   base64-encoded CSV — auto-detected), or
2. a local ``transactions.csv`` file (development convenience).

After the first seed the data lives in the database; the environment
variable can be removed. To update trades later (the CSV grows as you
trade), edit your CSV / variable and re-run the ``sync`` command — the
auto-seed only ever fires against an *empty* table.

Standalone usage::

    python -m ingestion.transactions show               # count + preview
    python -m ingestion.transactions seed [FILE]        # seed only if empty
    python -m ingestion.transactions sync FILE          # replace from a CSV
    python -m ingestion.transactions sync --from-env    # replace from env var
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text as sql_text

from db.connection import Session, engine
from db.models import Transaction

logger = logging.getLogger(__name__)

#: Application-wide constant for pg_advisory_xact_lock — serialises the
#: first-boot seed across concurrent gunicorn workers (see seed_if_empty).
_SEED_LOCK_KEY = 0x51454447  # ascii "QEDG"

#: Canonical DataFrame schema every downstream consumer expects. Kept
#: identical to ``analytics.portfolio._TX_COLUMNS`` (the legacy CSV
#: header) so the rest of the portfolio code is untouched.
TX_COLUMNS = ["user", "ticker", "date", "quantity", "price", "currency", "type"]

#: Default environment variable holding the seed CSV (raw or base64).
ENV_VAR = "TRANSACTIONS_CSV"

# create_all() is cheap but does a metadata round-trip; load_transactions
# runs on every Portfolio page render, so guard the DDL to once/process.
_table_ready = False


# ════════════════════════════════════════════════════════════════════
#  TABLE BOOTSTRAP
# ════════════════════════════════════════════════════════════════════

def _ensure_table() -> None:
    """Create the ``transactions`` table if it does not exist.

    ``app.py`` (the web process) does not call ``create_all_tables``,
    so a freshly-added model would not exist on an already-provisioned
    Railway database until ``setup.py`` is re-run. ``checkfirst=True``
    makes this idempotent and safe to call from any entry point.
    """
    global _table_ready
    if _table_ready:
        return
    try:
        Transaction.__table__.create(bind=engine, checkfirst=True)
    except Exception:
        # Concurrent workers can race the CREATE; the table simply
        # existing is the desired end state, so a duplicate-create
        # error here is benign.
        logger.debug("[transactions] table-create race ignored",
                      exc_info=True)
    _table_ready = True


# ════════════════════════════════════════════════════════════════════
#  CSV PARSING
# ════════════════════════════════════════════════════════════════════

def parse_csv_text(text: str) -> pd.DataFrame:
    """Parse CSV text into the canonical, normalised transactions frame.

    Raises ``ValueError`` if a required column is missing or no rows
    survive parsing.
    """
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in TX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"transactions CSV is missing column(s): {', '.join(missing)}"
        )
    df = df[TX_COLUMNS].copy()
    df = df.dropna(how="all")
    df["ticker"]   = df["ticker"].astype(str).str.strip().str.upper()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["type"]     = df["type"].astype(str).str.strip().str.lower()
    df["user"]     = df["user"].astype(str).str.strip()
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"]    = pd.to_numeric(df["price"], errors="coerce")
    # Drop structurally invalid rows (blank trailing lines, bad dates).
    df = df.dropna(subset=["ticker", "date", "quantity", "price", "type"])
    df = df[df["ticker"] != ""]
    if df.empty:
        raise ValueError("transactions CSV contained no valid rows")
    return df.reset_index(drop=True)


def _decode_env_csv(raw: str) -> str:
    """Return CSV text from an env-var value (raw CSV or base64).

    Railway/Heroku variable editors can mangle multi-line values, so a
    base64-encoded blob is also accepted and auto-detected.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(f"{ENV_VAR} is empty")
    # Heuristic: a raw CSV starts with the header line.
    head = raw.lstrip().splitlines()[0].replace(" ", "").lower()
    if head.startswith("user,") and "ticker" in head:
        return raw
    # Otherwise assume base64; validate it really decodes to a CSV.
    try:
        decoded = base64.b64decode(raw, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"{ENV_VAR} is neither a raw CSV (missing 'user,ticker,...' "
            f"header) nor valid base64: {exc}"
        ) from exc
    return decoded


# ════════════════════════════════════════════════════════════════════
#  DB READ / WRITE
# ════════════════════════════════════════════════════════════════════

def count_transactions() -> int:
    _ensure_table()
    with Session() as s:
        return int(s.query(Transaction).count())


def load_transactions_df() -> pd.DataFrame:
    """Read every stored trade as the canonical transactions frame.

    Returns an empty frame with :data:`TX_COLUMNS` when the table is
    empty, preserving the "no portfolio yet" contract relied on by the
    rest of the app.
    """
    _ensure_table()
    with Session() as s:
        rows = (
            s.query(
                Transaction.user_name,
                Transaction.ticker,
                Transaction.date,
                Transaction.quantity,
                Transaction.price,
                Transaction.currency,
                Transaction.tx_type,
            )
            .order_by(Transaction.date, Transaction.id)
            .all()
        )

    if not rows:
        return pd.DataFrame(columns=TX_COLUMNS)

    df = pd.DataFrame(rows, columns=TX_COLUMNS)
    df["date"]     = pd.to_datetime(df["date"])
    df["ticker"]   = df["ticker"].astype(str).str.upper()
    df["currency"] = df["currency"].astype(str).str.upper()
    df["type"]     = df["type"].astype(str).str.lower()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"]    = pd.to_numeric(df["price"], errors="coerce")
    return df


def _write_all(session, df: pd.DataFrame) -> int:
    """Replace every row using an existing session (no commit here).

    The caller's :class:`Session` context manager owns the
    commit/rollback, which keeps the delete+insert atomic and lets the
    seed path hold an advisory lock across the whole operation.
    """
    records = [
        Transaction(
            user_name = str(r["user"]).strip() or "default",
            ticker    = str(r["ticker"]).strip().upper(),
            date      = pd.Timestamp(r["date"]).date(),
            quantity  = float(r["quantity"]),
            price     = float(r["price"]),
            currency  = str(r["currency"]).strip().upper() or "EUR",
            tx_type   = str(r["type"]).strip().lower(),
        )
        for _, r in df.iterrows()
    ]
    session.query(Transaction).delete()
    session.add_all(records)
    return len(records)


def replace_all(df: pd.DataFrame) -> int:
    """Atomically replace the whole table with ``df`` (already parsed).

    Used by the ``sync`` maintenance command. Returns the number of
    rows written.
    """
    _ensure_table()
    with Session() as s:
        n = _write_all(s, df)
    logger.info("[transactions] wrote %d rows to the transactions table", n)
    return n


# ════════════════════════════════════════════════════════════════════
#  SEED / SYNC
# ════════════════════════════════════════════════════════════════════

def _resolve_csv_path(csv_path: str | None) -> Path | None:
    """Resolve a (possibly relative) CSV path robustly.

    Callbacks pass the bare name ``"transactions.csv"``; under gunicorn
    the CWD is not guaranteed, so also try the project root (the parent
    of this package).
    """
    if not csv_path:
        return None
    p = Path(csv_path)
    if p.is_file():
        return p
    anchored = Path(__file__).resolve().parents[1] / csv_path
    return anchored if anchored.is_file() else None


def _seed_text(csv_path: str | None) -> tuple[str, str] | None:
    """Return ``(source_label, csv_text)`` for the first available seed."""
    env_raw = os.environ.get(ENV_VAR)
    if env_raw and env_raw.strip():
        return f"${ENV_VAR}", _decode_env_csv(env_raw)
    resolved = _resolve_csv_path(csv_path)
    if resolved is not None:
        return str(resolved), resolved.read_text(encoding="utf-8")
    return None


def seed_if_empty(csv_path: str | None = "transactions.csv") -> int:
    """Populate the table from a seed source, but only when it is empty.

    Idempotent: a non-empty table is never touched, so this is safe to
    call on every startup / page load. Returns the number of rows
    seeded (0 if the table was already populated or no source exists).
    """
    # Cheap pre-check avoids taking the lock on the common (already
    # seeded) path — i.e. every page load after the first boot.
    if count_transactions() > 0:
        return 0
    try:
        seed = _seed_text(csv_path)
    except ValueError as exc:
        logger.error("[transactions] cannot seed — %s", exc)
        return 0
    if seed is None:
        logger.warning(
            "[transactions] table empty and no seed source "
            "(set the %s env var or provide transactions.csv) — "
            "Portfolio tab will be empty until trades are loaded.",
            ENV_VAR,
        )
        return 0
    label, csv = seed
    try:
        df = parse_csv_text(csv)
    except ValueError as exc:
        logger.error("[transactions] seed source %s is invalid: %s",
                     label, exc)
        return 0

    _ensure_table()
    with Session() as s:
        # Serialise concurrent gunicorn workers: only the first to grab
        # this transaction-scoped advisory lock seeds; the lock releases
        # automatically on commit. The others then see a non-empty table
        # via the double-check below and no-op (no double-counting).
        s.execute(sql_text("SELECT pg_advisory_xact_lock(:k)"),
                  {"k": _SEED_LOCK_KEY})
        if s.query(Transaction).count() > 0:
            logger.info("[transactions] another worker already seeded — "
                        "skipping")
            return 0
        n = _write_all(s, df)
    logger.info("[transactions] auto-seeded %d trades from %s", n, label)
    return n


def ensure_seeded_and_load(csv_path: str | None = "transactions.csv"
                           ) -> pd.DataFrame:
    """Ensure the table exists, seed it if empty, return all trades.

    This is the single entry point used by
    :func:`analytics.portfolio.load_transactions`.
    """
    seed_if_empty(csv_path)
    return load_transactions_df()


def sync_transactions(source: str) -> int:
    """Replace the whole table from ``source``.

    ``source`` is either a path to a CSV file or the literal
    ``"--from-env"`` to read the :data:`ENV_VAR` variable. Use this to
    update trades after the initial seed.
    """
    if source == "--from-env":
        env_raw = os.environ.get(ENV_VAR, "")
        text = _decode_env_csv(env_raw)
        label = f"${ENV_VAR}"
    else:
        resolved = _resolve_csv_path(source)
        if resolved is None:
            raise FileNotFoundError(f"CSV not found: {source}")
        text = resolved.read_text(encoding="utf-8")
        label = str(resolved)
    df = parse_csv_text(text)
    n = replace_all(df)
    logger.info("[transactions] synced %d trades from %s", n, label)
    return n


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cmd = argv[0] if argv else "show"

    if cmd == "show":
        n = count_transactions()
        print(f"transactions table: {n} row(s)")
        if n:
            df = load_transactions_df()
            print(df.to_string(index=False, max_rows=20))
        return 0

    if cmd == "seed":
        path = argv[1] if len(argv) > 1 else "transactions.csv"
        n = seed_if_empty(path)
        print(f"seeded {n} row(s)"
              if n else "nothing seeded (table not empty or no source)")
        return 0

    if cmd == "sync":
        if len(argv) < 2:
            print("usage: python -m ingestion.transactions sync "
                  "<FILE | --from-env>", file=sys.stderr)
            return 2
        n = sync_transactions(argv[1])
        print(f"synced {n} row(s)")
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
