"""
ingestion/universe.py — Sync FMP stock universe → company_info table.

Replaces stocks_all_pages.csv entirely.

Run on setup (once), then weekly via the scheduler.
The company_info table is the single source of truth for:
  - available tickers
  - company names
  - sectors / industries
  - market caps
  - exchanges / countries

Standalone usage
----------------
    python -m ingestion.universe
"""
import time
from datetime import datetime

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from db.connection import Session, engine
from db.models import CompanyInfo, IngestionLog, create_all_tables
from ingestion.fmp_client import fetch_stock_list, fetch_company_profiles


def run_universe_sync(enrich_profiles: bool = True) -> int:
    """
    1. Fetch full traded list from FMP /available-traded/list
    2. Upsert into company_info (ticker, exchange, is_etf)
    3. Optionally enrich with sector / market cap from /profile

    Returns total number of tickers upserted.
    """
    create_all_tables(engine)
    t0 = time.time()

    print("\n  [universe] fetching stock list from FMP...")
    stocks = fetch_stock_list()
    if not stocks:
        print("  [universe] ✗ no data returned — check FMP_API_KEY and plan.")
        return 0

    # ── Step 1: upsert basic info ─────────────────────────────────────
    with Session() as s:
        stmt = insert(CompanyInfo).values(stocks)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={
                "company_name":        stmt.excluded.company_name,
                "exchange":            stmt.excluded.exchange,
                "is_etf":              stmt.excluded.is_etf,
                "is_actively_trading": stmt.excluded.is_actively_trading,
                "updated_at":          datetime.utcnow(),
            },
        )
        s.execute(stmt)
        s.commit()

    print(f"  [universe] {len(stocks)} tickers upserted (basic info)")

    # ── Step 2: enrich stocks (not ETFs) with profile data ────────────
    if enrich_profiles:
        stock_tickers = [s["ticker"] for s in stocks if not s.get("is_etf")]
        print(f"  [universe] enriching {len(stock_tickers)} profiles (sector, market cap)...")
        profiles = fetch_company_profiles(stock_tickers)

        if profiles:
            with Session() as s:
                stmt2 = insert(CompanyInfo).values(profiles)
                stmt2 = stmt2.on_conflict_do_update(
                    index_elements=["ticker"],
                    set_={
                        "company_name": stmt2.excluded.company_name,
                        "sector":       stmt2.excluded.sector,
                        "industry":     stmt2.excluded.industry,
                        "market_cap":   stmt2.excluded.market_cap,
                        "exchange":     stmt2.excluded.exchange,
                        "country":      stmt2.excluded.country,
                        "updated_at":   datetime.utcnow(),
                    },
                )
                s.execute(stmt2)
                s.commit()
            print(f"  [universe] {len(profiles)} profiles enriched")

    duration = time.time() - t0
    with Session() as s:
        s.add(IngestionLog(
            job_type="universe",
            tickers_attempted=len(stocks),
            tickers_success=len(stocks),
            tickers_failed=0,
            duration_seconds=round(duration, 1),
        ))
        s.commit()

    print(f"  [universe] done in {duration:.0f}s")
    return len(stocks)


def get_universe_df(
    stocks_only: bool = False,
    active_only: bool = True,
    with_sector: bool = False,
) -> pd.DataFrame:
    with Session() as s:
        q = s.query(CompanyInfo)
        if active_only:
            q = q.filter(CompanyInfo.is_actively_trading == True)
        if stocks_only:
            q = q.filter(CompanyInfo.is_etf == False)
        if with_sector:
            q = q.filter(CompanyInfo.sector.isnot(None))
        rows = q.all()
        records = [{
            "Symbol":       r.ticker,
            "Company Name": r.company_name,
            "Market Cap":   r.market_cap,
            "Sector":       r.sector,
            "Industry":     r.industry,
            "Exchange":     r.exchange,
            "Country":      r.country,
        } for r in rows]

    if not records:
        return pd.DataFrame(columns=["Symbol", "Company Name", "Market Cap",
                                     "Sector", "Industry", "Exchange", "Country"])

    return pd.DataFrame(records)


if __name__ == "__main__":
    run_universe_sync()
