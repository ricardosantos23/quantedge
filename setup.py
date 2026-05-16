"""
setup.py — One-time database bootstrap for QuantEdge.

Run this ONCE before starting app.py for the first time.
Safe to re-run — all operations are idempotent (won't duplicate data).

Configuration (in .env or environment)
---------------------------------------
SCREENER_UNIVERSE controls which tickers get price + fundamental data:

  SCREENER_UNIVERSE=all
      Every actively traded stock on FMP (~48,000 tickers).
      Time: ~20-25 hours. Only do this overnight.

  SCREENER_UNIVERSE=500
      Top 500 stocks by market cap.
      Time: ~20 minutes. Good default.

  SCREENER_UNIVERSE=1000000000
      All stocks with market cap >= $1B (~2,500 tickers).
      Time: ~1-2 hours. Best balance of coverage vs speed.

  SCREENER_UNIVERSE=AAPL,MSFT,NVDA,...
      Exact list of tickers.

MIN_MARKET_CAP_FILTER (set below in this file)
-----------------------------------------------
  If SCREENER_UNIVERSE is "all" or a large number, you can still
  apply a minimum market cap filter to avoid tiny/illiquid stocks.
  Set to 0 to disable the filter entirely.

Usage
-----
    python setup.py
"""
import sys
import time
from datetime import datetime

import config

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these two values to control what gets ingested
# ════════════════════════════════════════════════════════════════════

# Minimum market cap filter (in USD).
# Applied on top of SCREENER_UNIVERSE when SCREENER_UNIVERSE="all".
# Set to 0 to ingest everything with no market cap filter.
# Examples:
#   0               → no filter (all ~48,000 tickers)
#   500_000_000     → $500M+ (~4,000 tickers,  ~2-3h)
#   1_000_000_000   → $1B+   (~2,500 tickers,  ~1-2h)  ← recommended
#   10_000_000_000  → $10B+  (~600 tickers,    ~20min)
MIN_MARKET_CAP_FILTER = 0   # $1B — change to 0 for truly all stocks

# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  QuantEdge — Database Setup")
print("=" * 60)
print(f"  DB:      {config.DB_URL[:40]}...")
print(f"  FMP key: {'✓ set' if config.FMP_API_KEY else '✗ NOT SET — aborting'}")

if not config.FMP_API_KEY:
    print("\n  Set FMP_API_KEY in your environment and re-run.")
    sys.exit(1)

t0 = time.time()

# ── 1. Create tables ──────────────────────────────────────────────
print("\n[1/7] Creating DB tables...")
from db.connection import engine
from db.models import create_all_tables
create_all_tables(engine)
print("  ✓ Tables ready")

# ── 2. Stock universe ─────────────────────────────────────────────
print("\n[2/7] Syncing stock universe from FMP...")
from ingestion.universe import run_universe_sync
n = run_universe_sync(enrich_profiles=False)
print(f"  ✓ {n} tickers in universe (profiles enriched separately below)")

# ── 3. Read transactions ──────────────────────────────────────────
print("\n[3/7] Reading transactions.csv...")
try:
    from analytics.portfolio import load_transactions
    tx = load_transactions("transactions.csv")
    portfolio_tickers = list(tx["ticker"].unique())
    currencies = [c for c in tx["currency"].unique() if c != "EUR"]
    start_date = tx["date"].min().strftime("%Y-%m-%d")
    print(f"  ✓ {len(portfolio_tickers)} portfolio tickers | currencies: {currencies}")
except FileNotFoundError:
    print("  ⚠ transactions.csv not found — skipping portfolio setup")
    portfolio_tickers = []
    currencies        = []
    start_date        = None

# ── 4. FX rates ───────────────────────────────────────────────────
if currencies:
    print(f"\n[4/7] Ingesting FX rates: {currencies}...")
    from ingestion.prices import run_fx_ingestion
    pairs = [f"EUR{c}" for c in currencies]
    run_fx_ingestion(pairs, years_back=config.YEARS_BACK)
    print("  ✓ FX rates ingested")
else:
    print("\n[4/7] No non-EUR currencies — skipping FX ingestion")

# ── 5. Price history ──────────────────────────────────────────────
print("\n[5/7] Ingesting price history...")

# ── Determine screener ticker list ────────────────────────────────
# Sourced from the FMP company-screener endpoint, which returns
# tickers already ranked by market cap WITH sector/market_cap in the
# same payload. This fixes the previous chicken-and-egg bug: ranking
# the universe by market cap used to require market caps that were
# never fetched (company_info.market_cap was NULL for ~all rows on a
# fresh DB), so "top 500" silently collapsed to just the portfolio.
from datetime import datetime as _dt

from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.connection import Session
from db.models import CompanyInfo
from ingestion.fmp_client import fetch_screener_universe

screener_universe = None  # list[dict] when sourced from the FMP screener

if isinstance(config.SCREENER_UNIVERSE, str) and config.SCREENER_UNIVERSE.lower() == "all":
    screener_universe = fetch_screener_universe(min_market_cap=MIN_MARKET_CAP_FILTER)
    top_n = [d["ticker"] for d in screener_universe]
    if MIN_MARKET_CAP_FILTER > 0:
        label = (f"all stocks with market cap >= "
                 f"${MIN_MARKET_CAP_FILTER/1e9:.1f}B ({len(top_n)} tickers)")
    else:
        label = f"ALL {len(top_n)} investable tickers"

elif isinstance(config.SCREENER_UNIVERSE, int):
    screener_universe = fetch_screener_universe(limit=config.SCREENER_UNIVERSE)
    top_n = [d["ticker"] for d in screener_universe]
    label = f"top {config.SCREENER_UNIVERSE} by market cap"

elif isinstance(config.SCREENER_UNIVERSE, list):
    top_n = [t.upper() for t in config.SCREENER_UNIVERSE]
    label = f"custom list of {len(top_n)} tickers"

else:
    screener_universe = fetch_screener_universe(limit=500)
    top_n = [d["ticker"] for d in screener_universe]
    label = "top 500 by market cap (fallback)"

# Persist the screener universe into company_info WITH market_cap and
# sector up front, so the app can rank/filter the screener immediately
# instead of waiting on a separate enrichment pass.
if screener_universe:
    with Session() as s:
        stmt = pg_insert(CompanyInfo).values(screener_universe)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "company_name": stmt.excluded.company_name,
                "sector":       stmt.excluded.sector,
                "industry":     stmt.excluded.industry,
                "market_cap":   stmt.excluded.market_cap,
                "exchange":     stmt.excluded.exchange,
                "country":      stmt.excluded.country,
                "updated_at":   _dt.utcnow(),
            },
        )
        s.execute(stmt)
        s.commit()
    print(f"  ✓ {len(screener_universe)} companies enriched (market cap, sector)")

# Always include portfolio tickers and SPY
tickers_to_ingest = list(set(top_n) | set(portfolio_tickers) | {"SPY"})

print(f"  Universe: {label}")
print(f"  Ingesting {len(tickers_to_ingest)} tickers × {config.YEARS_BACK} years...")

if len(tickers_to_ingest) > 5000:
    hours = len(tickers_to_ingest) / 48000 * 25
    print(f"  ⚠  Estimated time: {hours:.0f}-{hours*1.3:.0f} hours — safe to run overnight")

from ingestion.prices import run_price_ingestion
ok, failed = run_price_ingestion(tickers_to_ingest, years_back=config.YEARS_BACK)
print(f"  ✓ Prices: {ok} ok / {failed} failed")

# ── 5b. Enrich company profiles (sector, market cap) ─────────────
# Only for tickers we actually ingested prices for — not all 48k
print(f"\n  Enriching company profiles for {len(tickers_to_ingest)} tickers...")
from ingestion.fmp_client import fetch_company_profiles
from datetime import datetime as _dt
from sqlalchemy.dialects.postgresql import insert as pg_insert
from db.connection import Session
from db.models import CompanyInfo

profiles = fetch_company_profiles(tickers_to_ingest)
if profiles:
    with Session() as s:
        stmt = pg_insert(CompanyInfo).values(profiles)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "company_name": stmt.excluded.company_name,
                "sector":       stmt.excluded.sector,
                "industry":     stmt.excluded.industry,
                "market_cap":   stmt.excluded.market_cap,
                "exchange":     stmt.excluded.exchange,
                "country":      stmt.excluded.country,
                "updated_at":   _dt.utcnow(),
            },
        )
        s.execute(stmt)
        s.commit()
    print(f"  ✓ {len(profiles)} company profiles enriched (sector, market cap, exchange)")

# ── 6. Fundamentals ───────────────────────────────────────────────
print(f"\n[6/7] Refreshing fundamentals for {len(tickers_to_ingest)} tickers...")
from analytics.fundamental import refresh_fundamentals
df_fund = refresh_fundamentals(tickers_to_ingest, max_workers=4, stale_hours=0)
print(f"  ✓ {len(df_fund)} fundamentals cached")

# ── 7. Earnings calendar ──────────────────────────────────────────
print("\n[7/7] Syncing earnings calendar (next 90 days)...")
from ingestion.scheduler import run_earnings_sync
n_earn = run_earnings_sync(days_ahead=90)
print(f"  ✓ {n_earn} earnings events synced")

# ── Done ──────────────────────────────────────────────────────────
total = time.time() - t0
print("\n" + "=" * 60)
print(f"  Setup complete in {total/60:.1f} minutes")
print("  Now run:  python app.py")
print("=" * 60)
