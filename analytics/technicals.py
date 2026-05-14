"""
analytics/technicals.py — Technical feature computation from DB prices.

Changes from original
---------------------
- Reads prices from PostgreSQL (no yfinance calls)
- load_prices_from_db() is the new entry point for data loading
- compute_technical_features() is unchanged — same signals, same weights
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import Session


# ════════════════════════════════════════════════════════════════════
#  DATA LOADING FROM DB
# ════════════════════════════════════════════════════════════════════

def load_prices_from_db(
    tickers: list[str] | None = None,
    years_back: int = 3,
) -> pd.DataFrame:
    """
    Load OHLCV + adj_close from the prices table.

    Parameters
    ----------
    tickers   : list of tickers to load; None = all tickers in DB
    years_back: how far back to load (default 3 — enough for all indicators)

    Returns
    -------
    DataFrame with columns: ticker, date, open, high, low, close, adj_close, volume
    """
    since = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")

    with Session() as s:
        if tickers:
            rows = s.execute(
                text("""
                    SELECT ticker, date, open, high, low, close, adj_close, volume
                    FROM prices
                    WHERE ticker = ANY(:t) AND date >= :since
                    ORDER BY ticker, date
                """),
                {"t": tickers, "since": since},
            ).fetchall()
        else:
            rows = s.execute(
                text("""
                    SELECT ticker, date, open, high, low, close, adj_close, volume
                    FROM prices
                    WHERE date >= :since
                    ORDER BY ticker, date
                """),
                {"since": since},
            ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high",
                                      "low", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ════════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION  (identical signals to original technicals.py)
# ════════════════════════════════════════════════════════════════════

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical signals + composite score for every ticker/date.

    Input  : DataFrame with [ticker, date, open, high, low, close, adj_close, volume]
    Output : Same DataFrame with additional columns:
               return_1d, return_20d, momentum_12m, vol_60d,
               max_drawdown_12m, ema_200, trend_strength,
               rsi_14, ytd, macd,
               mom_12m_z, ret_6m_z, vol_60d_z, dd_12m_z, macd_z, trend_z,
               raw_score, score, rank
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    price = df.groupby("ticker")["adj_close"]

    # ── Returns ───────────────────────────────────────────────────────
    df["return_1d"]    = price.pct_change()
    df["return_20d"]   = price.pct_change(20)
    df["momentum_12m"] = price.pct_change(252)

    # ── Volatility ────────────────────────────────────────────────────
    df["vol_60d"] = df.groupby("ticker")["return_1d"].transform(
        lambda x: x.rolling(60).std() * np.sqrt(252)
    )

    # ── Drawdown ──────────────────────────────────────────────────────
    def _drawdown(x):
        return x / x.cummax() - 1

    df["max_drawdown_12m"] = (
        df.groupby("ticker")["adj_close"]
          .transform(_drawdown)
          .groupby(df["ticker"])
          .transform(lambda x: x.rolling(252).min())
    )

    # ── EMA trend ─────────────────────────────────────────────────────
    df["ema_200"]        = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.ewm(span=200).mean()
    )
    df["trend_strength"] = (df["adj_close"] / df["ema_200"]) - 1

    # ── RSI ───────────────────────────────────────────────────────────
    def _rsi(x):
        delta = x.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-9)))

    df["rsi_14"] = df.groupby("ticker")["adj_close"].transform(_rsi)

    # ── YTD ───────────────────────────────────────────────────────────
    df["year"] = df["date"].dt.year
    df["ytd"]  = (
        df["adj_close"]
        / df.groupby(["ticker", "year"])["adj_close"].transform("first")
    ) - 1
    df = df.drop(columns=["year"])

    # ── MACD ──────────────────────────────────────────────────────────
    df["macd"] = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.ewm(span=12).mean() - x.ewm(span=26).mean()
    )

    # ── Drop warmup rows (need 252 bars minimum per ticker) ───────────
    df["_row_num"] = df.groupby("ticker").cumcount()
    df = df[df["_row_num"] >= 252].drop(columns=["_row_num"])
    df = df.dropna()

    if df.empty:
        return df

    # ── Cross-sectional Z-scores ──────────────────────────────────────
    def _zscore(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    for raw_col, z_col in [
        ("momentum_12m",    "mom_12m_z"),
        ("return_20d",      "ret_6m_z"),
        ("vol_60d",         "vol_60d_z"),
        ("max_drawdown_12m","dd_12m_z"),
        ("macd",            "macd_z"),
        ("trend_strength",  "trend_z"),
    ]:
        df[z_col] = df.groupby("date")[raw_col].transform(_zscore)

    # ── Composite score ───────────────────────────────────────────────
    df["raw_score"] = (
        0.30 * df["mom_12m_z"]
        + 0.20 * df["ret_6m_z"]
        + 0.15 * df["trend_z"]
        + 0.10 * df["macd_z"]
        - 0.20 * df["vol_60d_z"]
        - 0.15 * df["dd_12m_z"]
    )
    df["score"] = df.groupby("date")["raw_score"].rank(pct=True)
    df["rank"]  = df.groupby("date")["score"].rank(ascending=False)

    return df
