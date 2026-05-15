"""
modules/stop_loss.py — Per-ticker stop-loss strategy comparison and sizing.

For every active position in the portfolio, this module backtests three
families of stop-loss strategies and recommends the configuration that
maximises expectancy. The recommendation surfaces on the Portfolio tab
of the dashboard.

Strategies
----------
1. **ATR stop**       — ``price - N * ATR(14)``. Adapts to realised
   volatility; tighter in calm markets, wider in noisy ones.
2. **Trailing stop**  — ``peak_so_far * (1 - pct)``. Locks in gains
   without giving back more than a fixed fraction from the high.
3. **Fixed-percent**  — ``entry_price * (1 - pct)``. Simple,
   predictable, but does not adapt to volatility.

Metrics computed per (strategy, parameter) combination:

* Total return         — Compounded across all simulated trades.
* Maximum drawdown     — On the cumulative-return equity curve.
* Win rate             — Fraction of trades with positive return.
* Profit factor        — ``(avg_win * n_wins) / (avg_loss * n_losses)``.
* Expectancy           — ``W * avg_win - (1 - W) * avg_loss``. **The
  optimisation target**: positive expectancy is the necessary
  condition for a strategy to be worth running.

Public API
----------
* :func:`compute_stop_recommendations` — Top-level entry used by
  ``app.py`` to build the recommendations table.
* :func:`find_optimal_stop`           — Best configuration for a single
  ticker.
* :func:`kelly_criterion`             — Position-sizing helper that
  caps at 25% to keep half-Kelly under 12.5%.
"""

from itertools import product

import numpy as np
import pandas as pd


# ─────────────────────────────────────────
# ATR
# ─────────────────────────────────────────
def _atr(df, period=14):
    hi = df["high"]
    lo = df["low"]
    cl = df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────
def _simulate(price_series, stop_series, entry_interval=5, max_hold=90):
    """
    Enters a long position every `entry_interval` bars.
    Exits when EITHER:
      (a) price crosses below the stop level, OR
      (b) max_hold bars have elapsed (time-based exit — prevents
          infinite open trades and the 100% win-rate artefact where
          a bull-run stop never triggers and the outer loop breaks
          after a single "still open" trade).

    Returns DataFrame of (return_pct, holding_bars) per trade.
    """
    prices = price_series.values
    stops  = stop_series.values
    n      = len(prices)
    trades = []

    i = 0
    while i < n - 1:
        entry = prices[i]
        if entry == 0:
            i += entry_interval
            continue

        exit_idx = min(i + max_hold, n - 1)   # hard time-stop

        j = i + 1
        while j <= exit_idx:
            if prices[j] <= stops[j]:          # stop triggered
                break
            j += 1

        j = min(j, n - 1)    # clamp — j overshoots by 1 when loop exhausts
        ret = (prices[j] - entry) / entry
        trades.append({"return": ret, "bars": j - i})
        i = j + entry_interval

    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["return", "bars"])


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
def _metrics(trades):
    if trades.empty:
        return dict(total_return=np.nan, max_drawdown=np.nan,
                    win_rate=np.nan, profit_factor=np.nan, expectancy=np.nan)

    rets = trades["return"].values
    wins  = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate = len(wins) / len(rets)
    avg_win  = wins.mean()  if len(wins)  else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 1e-9

    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses) + 1e-9)
    expectancy    = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Equity curve for drawdown
    equity = np.cumprod(1 + rets)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    return dict(
        total_return  = float(np.prod(1 + rets) - 1),
        max_drawdown  = float(max_dd),
        win_rate      = float(win_rate),
        profit_factor = float(profit_factor),
        expectancy    = float(expectancy),
        n_trades      = len(rets),
    )


# ─────────────────────────────────────────
# SINGLE TICKER BACKTEST
# ─────────────────────────────────────────
def backtest_stops_single(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    df_ticker must have columns: date, open, high, low, close
    Returns DataFrame with one row per (strategy, param) combo.
    """
    df = df_ticker.copy().sort_values("date").reset_index(drop=True)

    if len(df) < 60:
        return pd.DataFrame()

    atr14 = _atr(df)
    close = df["close"]

    results = []

    # ── 1. ATR-based ──────────────────────────────────
    for mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
        stop_series = close - mult * atr14
        # Only start where ATR is valid
        valid = atr14.notna()
        if valid.sum() < 30:
            continue
        trades = _simulate(
            close[valid].reset_index(drop=True),
            stop_series[valid].reset_index(drop=True)
        )
        m = _metrics(trades)
        m.update({"strategy": "ATR", "param": mult, "param_label": f"ATR×{mult}"})
        results.append(m)

    # ── 2. Trailing stop ──────────────────────────────
    for pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        peak = close.expanding().max()
        stop_series = peak * (1 - pct)
        trades = _simulate(close, stop_series)
        m = _metrics(trades)
        m.update({"strategy": "Trailing", "param": pct,
                  "param_label": f"Trail {pct*100:.0f}%"})
        results.append(m)

    # ── 3. Fixed % stop ───────────────────────────────
    for pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        # rolling fixed stop: entry - pct (simplified: rolling 20-bar entry)
        entries = close.shift(20)
        stop_series = entries * (1 - pct)
        stop_series = stop_series.bfill()
        trades = _simulate(close, stop_series)
        m = _metrics(trades)
        m.update({"strategy": "Fixed%", "param": pct,
                  "param_label": f"Fixed {pct*100:.0f}%"})
        results.append(m)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────
# OPTIMAL STOP PER TICKER
# ─────────────────────────────────────────
def find_optimal_stop(df_ticker: pd.DataFrame) -> dict:
    """
    Returns the best stop configuration maximising expectancy.
    """
    results = backtest_stops_single(df_ticker)

    if results.empty or results["expectancy"].isna().all():
        return {
            "strategy": "N/A", "param_label": "N/A",
            "expectancy": np.nan, "win_rate": np.nan,
            "profit_factor": np.nan, "max_drawdown": np.nan,
            "stop_price": np.nan,
        }

    best = results.sort_values("expectancy", ascending=False).iloc[0]

    # Compute current stop price from latest bar
    last = df_ticker.sort_values("date").iloc[-1]
    strategy = best["strategy"]
    param    = best["param"]

    if strategy == "ATR":
        atr_val = _atr(df_ticker.sort_values("date"))
        atr_now = atr_val.dropna().iloc[-1] if not atr_val.dropna().empty else np.nan
        stop_px = last["close"] - param * atr_now if not np.isnan(atr_now) else np.nan
    elif strategy == "Trailing":
        peak    = df_ticker["close"].max()
        stop_px = peak * (1 - param)
    else:  # Fixed%
        stop_px = last["close"] * (1 - param)

    return {
        "strategy"     : strategy,
        "param_label"  : best["param_label"],
        "expectancy"   : round(best["expectancy"], 4),
        "win_rate"     : round(best["win_rate"], 4),
        "profit_factor": round(best["profit_factor"], 3),
        "max_drawdown" : round(best["max_drawdown"], 4),
        "stop_price"   : round(stop_px, 2) if not np.isnan(stop_px) else np.nan,
        "current_price": round(last["close"], 2),
        "stop_pct_away": round((stop_px - last["close"]) / last["close"], 4)
                         if not np.isnan(stop_px) else np.nan,
    }


# ─────────────────────────────────────────
# KELLY CRITERION (optimal risk per trade)
# ─────────────────────────────────────────
def kelly_criterion(win_rate: float, profit_factor: float) -> float:
    """
    Full Kelly fraction.  Use half-Kelly in practice.
    Kelly % = W - (1-W)/R   where R = profit_factor
    """
    if profit_factor <= 0 or np.isnan(win_rate) or np.isnan(profit_factor):
        return np.nan
    k = win_rate - (1 - win_rate) / profit_factor
    return max(0.0, min(k, 0.25))  # cap at 25%


# ─────────────────────────────────────────
# PORTFOLIO STOP RECOMMENDATIONS
# ─────────────────────────────────────────
def compute_stop_recommendations(prices: pd.DataFrame,
                                  transactions: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker currently held, run the stop backtest and
    return a consolidated recommendations table.
    """
    # Active positions (net qty > 0)
    pos = (
        transactions.copy()
        .assign(signed_qty=lambda d: d.apply(
            lambda r: r["quantity"] if r["type"] == "buy" else -r["quantity"], axis=1))
        .groupby("ticker")["signed_qty"]
        .sum()
    )
    active_tickers = pos[pos > 0].index.tolist()

    rows = []
    for ticker in active_tickers:
        df_t = prices[prices["ticker"] == ticker].copy()
        if df_t.empty:
            continue
        rec = find_optimal_stop(df_t)
        rec["ticker"] = ticker
        rec["quantity"] = int(pos[ticker])

        # Kelly position sizing
        k = kelly_criterion(rec["win_rate"], rec["profit_factor"])
        rec["kelly_fraction"]  = round(k, 4) if not np.isnan(k) else np.nan
        rec["half_kelly"]      = round(k / 2, 4) if not np.isnan(k) else np.nan

        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols_order = [
        "ticker", "current_price", "stop_price", "stop_pct_away",
        "strategy", "param_label", "expectancy", "win_rate",
        "profit_factor", "max_drawdown", "kelly_fraction",
        "half_kelly", "quantity"
    ]
    return df[[c for c in cols_order if c in df.columns]]
