"""
db/models.py — SQLAlchemy ORM models for every table in QuantEdge.

Tables
------
prices            — daily OHLCV (replaces yfinance on-the-fly calls)
fx_rates          — daily EUR → X exchange rates (replaces yfinance FX)
company_info      — ticker metadata (replaces stocks_all_pages.csv)
fundamentals      — cached FMP fundamental scores + raw metrics
watchlist         — user watchlist entries
alerts            — price / stop-loss alert rules
earnings_calendar — upcoming earnings events
ingestion_log     — audit trail for each nightly job run
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, BigInteger,
    Date, DateTime, Boolean, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ── Market data ───────────────────────────────────────────────────────────────

class Price(Base):
    __tablename__ = "prices"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticker    = Column(String(20), nullable=False)
    date      = Column(Date,       nullable=False)
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float)
    adj_close = Column(Float)
    volume    = Column(BigInteger)
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
        Index("ix_prices_ticker_date", "ticker", "date"),
    )


class FxRate(Base):
    """
    Daily FX close rate stored as 'units of quote per 1 EUR'.
    e.g. pair='EURUSD', rate=1.08 → 1 EUR = 1.08 USD.
    portfolio.py inverts this to get EUR per 1 unit of foreign currency.
    """
    __tablename__ = "fx_rates"
    id   = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(10), nullable=False)   # e.g. "EURUSD"
    date = Column(Date,       nullable=False)
    rate = Column(Float,      nullable=False)
    __table_args__ = (
        UniqueConstraint("pair", "date", name="uq_fx_pair_date"),
    )


# ── Universe ──────────────────────────────────────────────────────────────────

class CompanyInfo(Base):
    """Drop-in replacement for stocks_all_pages.csv."""
    __tablename__ = "company_info"
    ticker              = Column(String(20),  primary_key=True)
    company_name        = Column(String(500))
    sector              = Column(String(200))
    industry            = Column(String(200))
    market_cap          = Column(BigInteger)
    exchange            = Column(String(50))
    country             = Column(String(100))
    is_etf              = Column(Boolean, default=False)
    is_actively_trading = Column(Boolean, default=True)
    updated_at          = Column(DateTime, default=datetime.utcnow)


# ── Fundamentals cache ────────────────────────────────────────────────────────

class Fundamental(Base):
    """
    All fundamental metrics cached from FMP.
    Updated by the nightly scheduler (or on first request).
    Includes both raw numeric values and pre-formatted display strings
    so the app never needs to reformat them at render time.
    """
    __tablename__ = "fundamentals"
    ticker             = Column(String(20), primary_key=True)
    fundamental_score  = Column(Float)
    # ── Raw numeric metrics ───────────────────────────────────────────
    roic               = Column(Float)
    fcf_margin         = Column(Float)
    rev_cagr           = Column(Float)
    gross_margin       = Column(Float)
    interest_coverage  = Column(Float)
    ev_ebitda          = Column(Float)
    pe_ratio           = Column(Float)
    pb_ratio           = Column(Float)
    net_margin         = Column(Float)
    operating_margin   = Column(Float)
    fcf_yield          = Column(Float)
    dcf_upside         = Column(Float)
    dcf_value          = Column(Float)
    dcf_price_val      = Column(Float)
    piotroski          = Column(Float)
    share_dilution     = Column(Float)
    capex_pct_rev      = Column(Float)
    insider_signal     = Column(Integer, default=0)
    insider_buys       = Column(Integer, default=0)
    insider_sells      = Column(Integer, default=0)
    # ── Pre-formatted display strings ─────────────────────────────────
    dcf_upside_fmt     = Column(String(50))
    roic_fmt           = Column(String(50))
    fcf_margin_fmt     = Column(String(50))
    fcf_yield_fmt      = Column(String(50))
    op_margin_fmt      = Column(String(50))
    net_margin_fmt     = Column(String(50))
    gross_margin_fmt   = Column(String(50))
    rev_cagr_fmt       = Column(String(50))
    capex_pct_rev_fmt  = Column(String(50))
    share_dilution_fmt = Column(String(50))
    ev_ebitda_fmt      = Column(String(50))
    pe_ratio_fmt       = Column(String(50))
    piotroski_fmt      = Column(String(50))
    insider_signal_fmt = Column(String(50))
    updated_at         = Column(DateTime, default=datetime.utcnow)


# ── User features ─────────────────────────────────────────────────────────────

class Watchlist(Base):
    __tablename__ = "watchlist"
    id        = Column(Integer,  primary_key=True, autoincrement=True)
    user_name = Column(String(100), nullable=False, index=True)
    ticker    = Column(String(20),  nullable=False)
    notes     = Column(Text)
    added_at  = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint("user_name", "ticker", name="uq_watch_user_ticker"),
    )


class Alert(Base):
    """
    alert_type values:
        "price_above"  — fire when adj_close >= threshold
        "price_below"  — fire when adj_close <= threshold
        "stop_loss"    — fire when adj_close <= threshold (same logic, different label)
    """
    __tablename__ = "alerts"
    id           = Column(Integer,   primary_key=True, autoincrement=True)
    user_name    = Column(String(100), nullable=False, index=True)
    ticker       = Column(String(20),  nullable=False)
    alert_type   = Column(String(50),  nullable=False)
    threshold    = Column(Float,       nullable=False)
    email        = Column(String(200))
    triggered    = Column(Boolean, default=False)
    triggered_at = Column(DateTime)
    created_at   = Column(DateTime, default=datetime.utcnow)
    is_active    = Column(Boolean, default=True)


# ── Earnings ──────────────────────────────────────────────────────────────────

class EarningsCalendar(Base):
    __tablename__ = "earnings_calendar"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), nullable=False, index=True)
    earnings_date    = Column(Date,       nullable=False)
    eps_estimate     = Column(Float)
    eps_actual       = Column(Float)
    revenue_estimate = Column(Float)
    revenue_actual   = Column(Float)
    updated_at       = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint("ticker", "earnings_date", name="uq_earn_ticker_date"),
    )


# ── Ops ───────────────────────────────────────────────────────────────────────

class IngestionLog(Base):
    __tablename__ = "ingestion_log"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    run_at            = Column(DateTime, default=datetime.utcnow)
    job_type          = Column(String(50))  # "prices"|"universe"|"fundamentals"|"earnings"|"fx"
    tickers_attempted = Column(Integer)
    tickers_success   = Column(Integer)
    tickers_failed    = Column(Integer)
    duration_seconds  = Column(Float)
    notes             = Column(Text)


def create_all_tables(engine):
    """Idempotent — safe to call on every startup."""
    Base.metadata.create_all(engine)
