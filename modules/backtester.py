"""
backtester.py
=============
Simulates a monthly DCA strategy guided by the composite score.

Flow:
  1.  Each month, rank all stocks by weighted composite score
  2.  Allocate monthly_budget across top N stocks (equal weight)
  3.  Buy fractional shares at month-open price
  4.  Track portfolio value daily
  5.  Optionally rebalance every R months

Score weights are tunable → used by the optimiser to find the
parameter set that maximises Sharpe ratio (or total return).

Returns:
  - equity_curve  : daily portfolio value
  - trade_log     : every buy/sell with rationale
  - summary       : CAGR, Sharpe, Max DD, Win Rate, Profit Factor
"""

import numpy as np
import pandas as pd
from itertools import product


# ─────────────────────────────────────────
# DEFAULT WEIGHT VECTOR
# ─────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "mom_12m_z"  : 0.30,
    "ret_6m_z"   : 0.20,
    "trend_z"    : 0.15,
    "macd_z"     : 0.10,
    "vol_60d_z"  : -0.20,   # penalise high vol
    "dd_12m_z"   : -0.15,   # penalise deep drawdowns
}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _recompute_score(df_tech: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Re-scores the technical dataframe with a custom weight vector.
    Returns df with 'custom_score' column.
    """
    df = df_tech.copy()
    z_cols = list(weights.keys())
    available = [c for c in z_cols if c in df.columns]

    df["raw_score"] = sum(
        weights.get(c, 0) * df[c] for c in available
    )
    df["custom_score"] = df.groupby("date")["raw_score"].rank(pct=True)
    return df


def _monthly_dates(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end, freq="MS")   # month-start


# ─────────────────────────────────────────
# CORE BACKTEST
# ─────────────────────────────────────────
def run_backtest(
    df_tech      : pd.DataFrame,
    prices       : pd.DataFrame,
    start_date   : str  = "2015-01-01",
    end_date     : str  = None,
    monthly_budget: float = 1000.0,
    top_n        : int  = 5,
    rebalance_months: int = 1,
    weights      : dict = None,
    score_col    : str  = None,       # use pre-computed if provided
) -> dict:
    """
    Parameters
    ----------
    df_tech          : output of compute_technical_features()
    prices           : daily OHLCV with columns [ticker, date, close]
    start_date       : backtest start (ISO string)
    end_date         : backtest end  (ISO string or None → today)
    monthly_budget   : euros added to portfolio each month
    top_n            : how many stocks to hold at any one time
    rebalance_months : how often to rebalance (1=monthly, 3=quarterly)
    weights          : override weight dict for optimisation
    score_col        : column name to use as score (default: 'score')
    """

    weights   = weights or DEFAULT_WEIGHTS
    end_date  = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_ts  = pd.Timestamp(start_date)
    end_ts    = pd.Timestamp(end_date)

    # ── Recompute scores if weights provided ────────────────────────
    if score_col is None:
        df_scored = _recompute_score(df_tech, weights)
        score_col = "custom_score"
    else:
        df_scored = df_tech.copy()

    df_scored["date"] = pd.to_datetime(df_scored["date"])
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    # ── Pivot prices for fast lookup ────────────────────────────────
    price_pivot = prices.pivot_table(
        index="date", columns="ticker", values="close"
    ).sort_index()
    price_pivot = price_pivot.ffill()

    # Restrict to backtest window
    price_pivot = price_pivot.loc[
        (price_pivot.index >= start_ts) &
        (price_pivot.index <= end_ts)
    ]

    if price_pivot.empty:
        return {"error": "No price data in the requested date range."}

    # ── State ───────────────────────────────────────────────────────
    cash       = 0.0
    holdings   = {}        # {ticker: shares}
    trade_log  = []
    equity_curve = []

    monthly_dates = _monthly_dates(start_ts, end_ts)
    rebalance_set = set(monthly_dates[::rebalance_months])

    prev_month = None

    for date, price_row in price_pivot.iterrows():

        # Portfolio value
        port_val = cash + sum(
            price_row.get(t, np.nan) * q
            for t, q in holdings.items()
            if not np.isnan(price_row.get(t, np.nan))
        )
        equity_curve.append({"date": date, "value": port_val, "cash": cash})

        month_start = date.replace(day=1)
        if month_start == prev_month:
            continue
        prev_month = month_start

        if month_start not in rebalance_set:
            # Still add monthly budget even if not rebalancing
            cash += monthly_budget
            continue

        # ── Monthly injection ───────────────────────────────────────
        cash += monthly_budget

        # ── Latest scores up to this date ──────────────────────────
        scores_now = (
            df_scored[df_scored["date"] <= date]
            .sort_values("date")
            .groupby("ticker")
            .tail(1)
            [[  "ticker", score_col]]
            .dropna()
        )

        if scores_now.empty:
            continue

        # Only consider tickers with a valid price today
        valid_prices = price_row.dropna()
        scores_now   = scores_now[scores_now["ticker"].isin(valid_prices.index)]

        if scores_now.empty:
            continue

        # Top N
        top = (
            scores_now
            .sort_values(score_col, ascending=False)
            .head(top_n)
            ["ticker"]
            .tolist()
        )

        # ── Sell positions NOT in top N ────────────────────────────
        to_sell = [t for t in list(holdings.keys()) if t not in top]
        for t in to_sell:
            px = price_row.get(t, np.nan)
            if np.isnan(px):
                continue
            proceeds = holdings.pop(t) * px
            cash += proceeds
            trade_log.append({
                "date": date, "ticker": t, "action": "SELL",
                "price": round(px, 2),
                "value": round(proceeds, 2),
                "reason": "Dropped from top-N",
            })

        # ── Buy / top-up top N ─────────────────────────────────────
        alloc_per   = cash / max(top_n, 1)
        for t in top:
            px = price_row.get(t, np.nan)
            if np.isnan(px) or px <= 0:
                continue
            shares = alloc_per / px
            holdings[t] = holdings.get(t, 0) + shares
            cash -= shares * px
            trade_log.append({
                "date": date, "ticker": t, "action": "BUY",
                "price": round(px, 2),
                "shares": round(shares, 4),
                "value": round(shares * px, 2),
                "score": round(
                    scores_now.set_index("ticker")[score_col].get(t, np.nan), 3
                ),
            })

    # ── Equity curve ─────────────────────────────────────────────
    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {"error": "Empty equity curve — check date range."}

    eq = eq.set_index("date")

    # ── Summary metrics ────────────────────────────────────────────
    summary = _compute_summary(eq["value"], monthly_budget,
                                len(monthly_dates), pd.DataFrame(trade_log))

    return {
        "equity_curve": eq.reset_index(),
        "trade_log"   : pd.DataFrame(trade_log),
        "summary"     : summary,
        "final_holdings": holdings,
    }


# ─────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────
def _compute_summary(equity: pd.Series, monthly_budget: float,
                      n_months: int, trades: pd.DataFrame) -> dict:
    if equity.empty or equity.iloc[-1] <= 0:
        return {}

    total_invested = monthly_budget * n_months
    final_value    = equity.iloc[-1]
    total_return   = (final_value - total_invested) / total_invested

    # CAGR
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    if n_years > 0 and total_invested > 0:
        cagr = (final_value / total_invested) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Daily returns
    daily_ret = equity.pct_change().dropna()
    sharpe    = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)

    # Max drawdown
    peak  = equity.cummax()
    dd    = (equity - peak) / peak
    max_dd = dd.min()

    # Trade stats
    if not trades.empty and "action" in trades.columns:
        sells = trades[trades["action"] == "SELL"]
        buys  = trades[trades["action"] == "BUY"]
        n_trades = len(sells)
    else:
        n_trades = 0

    return {
        "total_invested"  : round(total_invested, 2),
        "final_value"     : round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct"        : round(cagr * 100, 2),
        "sharpe_ratio"    : round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades"        : n_trades,
        "years"           : round(n_years, 1),
    }


# ─────────────────────────────────────────
# WEIGHT OPTIMISER (Grid Search)
# ─────────────────────────────────────────
def optimise_weights(
    df_tech      : pd.DataFrame,
    prices       : pd.DataFrame,
    start_date   : str   = "2015-01-01",
    monthly_budget: float = 1000.0,
    top_n        : int   = 5,
    metric       : str   = "sharpe_ratio",   # or "cagr_pct" / "total_return_pct"
) -> pd.DataFrame:
    """
    Grid search over weight combinations.
    Returns a DataFrame with all tested combos ranked by `metric`.
    This is intentionally lightweight — a coarse search for the UI tuner.
    """
    weight_options = {
        "mom_12m_z": [0.20, 0.30, 0.40],
        "ret_6m_z" : [0.10, 0.20, 0.30],
        "trend_z"  : [0.10, 0.20],
        "vol_60d_z": [-0.15, -0.25],
        "dd_12m_z" : [-0.10, -0.20],
    }
    # macd_z gets the remainder to sum ≈ 1
    keys   = list(weight_options.keys())
    combos = list(product(*weight_options.values()))

    results = []
    for combo in combos:
        w = dict(zip(keys, combo))
        # Normalise so |weights| sum to 1
        total = sum(abs(v) for v in w.values())
        w = {k: v / total for k, v in w.items()}

        bt = run_backtest(
            df_tech, prices, start_date=start_date,
            monthly_budget=monthly_budget, top_n=top_n, weights=w
        )
        if "error" in bt:
            continue

        row = {"weights": w}
        row.update(bt["summary"])
        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(metric, ascending=False)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# BENCHMARK (Buy-and-hold SPY)
# ─────────────────────────────────────────
def benchmark_buyhold(prices: pd.DataFrame,
                       start_date: str,
                       monthly_budget: float = 1000.0,
                       benchmark_ticker: str = "SPY") -> pd.DataFrame:
    """
    Simulates DCA into a single benchmark ticker.
    Returns equity curve DataFrame.
    """
    df = prices[prices["ticker"] == benchmark_ticker].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    start_ts = pd.Timestamp(start_date)
    df = df[df.index >= start_ts]

    if df.empty:
        return pd.DataFrame()

    cash   = 0.0
    shares = 0.0
    equity = []

    prev_month = None
    for date, row in df.iterrows():
        month_start = date.replace(day=1)
        if month_start != prev_month:
            prev_month  = month_start
            cash       += monthly_budget
            price       = row["close"]
            new_shares  = cash / price
            shares     += new_shares
            cash        = 0.0

        equity.append({"date": date, "value": shares * row["close"]})

    return pd.DataFrame(equity)
