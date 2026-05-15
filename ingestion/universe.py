"""
ingestion/universe.py — Sync the FMP stock universe into ``company_info``.

The ``company_info`` table is the single source of truth for available
tickers and their metadata (name, sector, industry, market cap,
exchange, country). This module fully replaces the legacy
``stocks_all_pages.csv`` file.

Cadence
-------
* On first setup, run once via :func:`run_universe_sync` to populate
  every ticker.
* Re-run weekly via the scheduler (Sunday 23:00 UTC) to pick up new
  listings, delistings, and metadata changes.

Standalone usage::

    python -m ingestion.universe
"""

import logging
import time
from datetime import datetime

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from db.connection import Session, engine
from db.models import CompanyInfo, IngestionLog, create_all_tables
from ingestion.fmp_client import fetch_company_profiles, fetch_stock_list

logger = logging.getLogger(__name__)


def run_universe_sync(enrich_profiles: bool = True) -> int:
    """
    1. Fetch full traded list from FMP /available-traded/list
    2. Upsert into company_info (ticker, exchange, is_etf)
    3. Optionally enrich with sector / market cap from /profile

    Returns total number of tickers upserted.
    """
    create_all_tables(engine)
    t0 = time.time()

    logger.info("[universe] fetching stock list from FMP...")
    stocks = fetch_stock_list()
    if not stocks:
        logger.error("[universe] no data returned — check FMP_API_KEY and plan")
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

    logger.info("[universe] %d tickers upserted (basic info)", len(stocks))

    # ── Step 2: enrich stocks (not ETFs) with profile data ────────────
    if enrich_profiles:
        stock_tickers = [s["ticker"] for s in stocks if not s.get("is_etf")]
        logger.info(
            "[universe] enriching %d profiles (sector, market cap)...",
            len(stock_tickers),
        )
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
            logger.info("[universe] %d profiles enriched", len(profiles))

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

    logger.info("[universe] done in %.0fs", duration)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_universe_sync()
