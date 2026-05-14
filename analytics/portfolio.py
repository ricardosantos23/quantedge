"""
analytics/portfolio.py — Portfolio P&L and holdings analytics.

Changes from original
---------------------
- get_fx_data()           reads from DB fx_rates table (not yfinance)
- load_portfolio_prices() reads from DB prices table
- All other logic (build_positions, compute_pnl, compute_holdings_table,
  compute_portfolio_kpis) is unchanged.
"""
import numpy as np
import pandas as pd
from sqlalchemy import text

from db.connection import Session

import yfinance as _yf   # kept only for stock split lookup; no price calls

_splits_cache: dict = {}


# ════════════════════════════════════════════════════════════════════
#  TRANSACTIONS
# ════════════════════════════════════════════════════════════════════

def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    for col in ["user", "ticker", "date", "quantity", "price", "currency", "type"]:
        if col not in df.columns:
            raise ValueError(f"transactions.csv is missing column: {col}")
    df["ticker"]   = df["ticker"].str.upper()
    df["currency"] = df["currency"].str.upper()
    df["type"]     = df["type"].str.lower()
    return df


# ════════════════════════════════════════════════════════════════════
#  FX  —  now served from DB
# ════════════════════════════════════════════════════════════════════

def get_fx_data(currencies: list[str], start_date) -> dict:
    """
    Load daily FX rates from the fx_rates table.
    Returns {currency: DataFrame[date, fx]} where fx = EUR per 1 foreign unit.

    FMP stores EURUSD = 1.08 (USD per EUR).
    We invert: 1/1.08 = 0.926 EUR per USD.
    This matches the original yfinance behaviour exactly.
    """
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    fx_data: dict = {}

    with Session() as s:
        for c in currencies:
            if c == "EUR":
                continue
            pair = f"EUR{c}"
            rows = s.execute(
                text("""
                    SELECT date, rate FROM fx_rates
                    WHERE pair = :pair AND date >= :since
                    ORDER BY date
                """),
                {"pair": pair, "since": start_str},
            ).fetchall()

            if rows:
                df = pd.DataFrame(rows, columns=["date", "fx"])
                df["date"] = pd.to_datetime(df["date"])
                df["fx"]   = 1.0 / df["fx"]   # EUR per 1 foreign unit
                fx_data[c] = df
            else:
                print(f"  [portfolio] WARNING: no FX data for {pair} in DB. "
                      f"Run: python -m ingestion.prices (will include FX)")
                fx_data[c] = pd.DataFrame({
                    "date": [pd.Timestamp.today()],
                    "fx":   [1.0],
                })

    return fx_data


# ════════════════════════════════════════════════════════════════════
#  PRICES  —  now served from DB
# ════════════════════════════════════════════════════════════════════

def load_portfolio_prices(tickers: list[str], start_date) -> pd.DataFrame:
    """Load OHLCV for portfolio tickers from DB."""
    if not tickers:
        return pd.DataFrame()
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")

    with Session() as s:
        rows = s.execute(
            text("""
                SELECT ticker, date, open, high, low, close, adj_close, volume
                FROM prices
                WHERE ticker = ANY(:t) AND date >= :since
                ORDER BY ticker, date
            """),
            {"t": tickers, "since": start_str},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ticker","date","open","high",
                                      "low","close","adj_close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ════════════════════════════════════════════════════════════════════
#  STOCK SPLITS  (still uses yfinance — splits are rare, low traffic)
# ════════════════════════════════════════════════════════════════════

def _get_splits(ticker: str) -> pd.Series:
    if ticker in _splits_cache:
        return _splits_cache[ticker]
    try:
        s = _yf.Ticker(ticker).splits
        if s is not None and not s.empty:
            idx = pd.to_datetime(s.index)
            s.index = idx.tz_convert(None) if idx.tz is not None else idx
        else:
            s = pd.Series(dtype=float)
    except Exception:
        s = pd.Series(dtype=float)
    _splits_cache[ticker] = s
    return s


def _split_factor(ticker: str, after_date) -> float:
    """Cumulative split ratio for all splits after after_date."""
    s  = _get_splits(ticker)
    if s.empty:
        return 1.0
    dt = pd.Timestamp(after_date)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None).replace(tzinfo=None)
    future = s[s.index > dt]
    return float(future.prod()) if not future.empty else 1.0


# ════════════════════════════════════════════════════════════════════
#  FX HELPERS
# ════════════════════════════════════════════════════════════════════

def _norm_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _fx_rate(fx_data: dict, currency: str, date) -> float:
    if not fx_data or currency == "EUR" or currency not in fx_data:
        return 1.0
    df  = fx_data[currency].copy()
    df["date"] = _norm_dates(df["date"])
    df  = df.set_index("date").sort_index()
    dt  = pd.Timestamp(date)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None).replace(tzinfo=None)
    avail = df.index[df.index <= dt]
    return float(df["fx"].iloc[0]) if avail.empty else float(df.loc[avail[-1], "fx"])


def _fx_latest(fx_data: dict, currency: str) -> float:
    if not fx_data or currency == "EUR" or currency not in fx_data:
        return 1.0
    df = fx_data[currency].copy()
    df["date"] = _norm_dates(df["date"])
    return float(df.sort_values("date")["fx"].iloc[-1])


# ════════════════════════════════════════════════════════════════════
#  BUILD POSITIONS  (unchanged logic)
# ════════════════════════════════════════════════════════════════════

def build_positions(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    fx_data: dict,
) -> pd.DataFrame:
    transactions = transactions.sort_values("date")
    price_col    = "adj_close" if "adj_close" in prices.columns else "close"
    positions    = []

    for user in transactions["user"].unique():
        user_tx = transactions[transactions["user"] == user]
        for ticker in user_tx["ticker"].unique():
            tx         = user_tx[user_tx["ticker"] == ticker]
            first_date = tx["date"].min()

            price_df = prices[
                (prices["ticker"] == ticker) & (prices["date"] >= first_date)
            ].copy()

            if price_col not in price_df.columns:
                if "close" in price_df.columns:
                    price_df = price_df.rename(columns={"close": "price"})
                else:
                    continue
            else:
                price_df = price_df.rename(columns={price_col: "price"})

            price_df = price_df.sort_values("date")
            qty, history = 0, []
            for _, row in tx.iterrows():
                qty += row["quantity"] if row["type"] == "buy" else -row["quantity"]
                history.append((row["date"], qty))

            hist_df = pd.DataFrame(history, columns=["date", "quantity"])
            merged  = price_df.merge(hist_df, on="date", how="left")
            merged["quantity"] = merged["quantity"].ffill().fillna(0)

            currency = tx["currency"].iloc[0]
            if currency != "EUR" and currency in fx_data:
                fx = fx_data[currency]
                merged = merged.merge(fx, on="date", how="left")
                merged["fx"] = merged["fx"].ffill()
                merged["price_eur"] = merged["price"] * merged["fx"]
            else:
                merged["price_eur"] = merged["price"]

            merged["user"]   = user
            merged["ticker"] = ticker
            positions.append(merged)

    if not positions:
        return pd.DataFrame()
    return pd.concat(positions, ignore_index=True)


# ════════════════════════════════════════════════════════════════════
#  P&L  (unchanged logic)
# ════════════════════════════════════════════════════════════════════

def convert_price_to_eur(row, fx_data: dict) -> float:
    if row["currency"] == "EUR":
        return row["price"]
    return row["price"] * _fx_rate(fx_data, row["currency"], row["date"])


def compute_pnl(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    fx_data: dict,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()

    transactions = transactions.sort_values("date").copy()
    transactions["price_eur"] = transactions.apply(
        lambda r: convert_price_to_eur(r, fx_data), axis=1
    )

    results = []
    for user in transactions["user"].unique():
        user_tx = transactions[transactions["user"] == user]
        for ticker in user_tx["ticker"].unique():
            tx     = user_tx[user_tx["ticker"] == ticker]
            pos_df = positions[
                (positions["user"] == user) & (positions["ticker"] == ticker)
            ].copy()

            if pos_df.empty:
                continue

            qty = cost = 0
            pnl_series, tx_idx = [], 0
            tx_list = tx.to_dict("records")

            for _, row in pos_df.iterrows():
                date = row["date"]
                while tx_idx < len(tx_list) and tx_list[tx_idx]["date"] <= date:
                    t = tx_list[tx_idx]
                    if t["type"] == "buy":
                        qty  += t["quantity"]
                        cost += t["quantity"] * t["price_eur"]
                    elif t["type"] == "sell" and qty > 0:
                        avg_cost = cost / qty
                        cost    -= avg_cost * t["quantity"]
                        qty     -= t["quantity"]
                    tx_idx += 1

                if qty > 0:
                    pnl_series.append((row["price_eur"] - cost / qty) * qty)
                else:
                    pnl_series.append(0)

            pos_df["pnl"] = pnl_series
            results.append(pos_df[["date", "user", "ticker", "pnl"]])

    if not results:
        return pd.DataFrame()

    df = pd.concat(results)
    return df.groupby(["user", "date"])["pnl"].sum().reset_index()


# ════════════════════════════════════════════════════════════════════
#  HOLDINGS TABLE  (unchanged logic — reads prices from DB via caller)
# ════════════════════════════════════════════════════════════════════

def compute_holdings_table(
    transactions, prices, df_tech, df_fund, stocks_df, fx_data=None
):
    """Same logic as original — returns (open_df, closed_df)."""
    tx        = transactions.copy()
    tx["date"]= pd.to_datetime(tx["date"])
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    usd_rate  = _fx_latest(fx_data, "USD")

    open_rows, closed_rows = [], []

    for ticker in tx["ticker"].unique():
        t = tx[tx["ticker"] == ticker].sort_values("date")
        qty = cost = realised_pnl = 0.0
        first_buy = last_sell = None

        for _, row in t.iterrows():
            adj = _split_factor(ticker, row["date"])
            if row["type"] == "buy":
                if first_buy is None:
                    first_buy = row["date"]
                adj_qty = row["quantity"] * adj
                qty    += adj_qty
                rate    = _fx_rate(fx_data, row["currency"], row["date"])
                cost   += row["quantity"] * row["price"] * rate
            elif row["type"] == "sell" and qty > 0:
                adj_qty      = row["quantity"] * adj
                avg_c        = cost / qty
                sell_rate    = _fx_rate(fx_data, row["currency"], row["date"])
                sell_px_eur  = row["price"] * sell_rate
                realised_pnl+= adj_qty * (sell_px_eur - avg_c)
                cost        -= avg_c * adj_qty
                qty         -= adj_qty
                last_sell    = row["date"]

        info     = stocks_df[stocks_df["Symbol"] == ticker]
        company  = info["Company Name"].iloc[0] if not info.empty else ticker
        mkt_cap  = info["Market Cap"].iloc[0]   if not info.empty else np.nan
        sector   = info["Sector"].iloc[0]        if not info.empty else "N/A"
        exchange = info["Exchange"].iloc[0]      if not info.empty and "Exchange" in info.columns else "N/A"

        try:
            mc = float(mkt_cap)
            if mc >= 1e12:   cap_str = f"${mc/1e12:.1f}T"
            elif mc >= 1e9:  cap_str = f"${mc/1e9:.0f}B"
            else:            cap_str = f"${mc/1e6:.0f}M"
        except Exception:
            cap_str = "N/A"

        tech = (
            df_tech[df_tech["ticker"] == ticker].sort_values("date").tail(1)
            if df_tech is not None and not df_tech.empty else pd.DataFrame()
        )

        def tv(col, scale=1.0):
            if not tech.empty and col in tech.columns:
                try:
                    v = float(tech[col].iloc[0])
                    return round(v * scale, 1) if np.isfinite(v) else None
                except Exception:
                    pass
            return None

        fund = (
            df_fund[df_fund["ticker"] == ticker]
            if df_fund is not None and not df_fund.empty else pd.DataFrame()
        )
        fund_score = (
            round(float(fund["fundamental_score"].iloc[0]), 2)
            if not fund.empty and pd.notna(fund["fundamental_score"].iloc[0])
            else None
        )

        if qty >= 0.01:
            avg_entry = cost / qty
            tp        = prices[prices["ticker"] == ticker].sort_values("date")
            current_px     = float(tp[price_col].iloc[-1]) if not tp.empty else np.nan
            current_px_eur = current_px * usd_rate if not np.isnan(current_px) else np.nan
            pos_val        = current_px * qty if not np.isnan(current_px) else 0.0
            unreal_val     = (current_px_eur - avg_entry) * qty if not np.isnan(current_px_eur) else np.nan
            unreal_pct     = (current_px_eur / avg_entry - 1.0) * 100 if avg_entry > 0 and not np.isnan(current_px_eur) else np.nan
            unreal_str     = (
                f"€{unreal_val:+,.0f}  ({unreal_pct:+.1f}%)"
                if unreal_val is not None and not np.isnan(unreal_val) else "N/A"
            )
            open_rows.append({
                "Ticker":      ticker, "Exchange": exchange, "Company": company,
                "Qty":         round(qty, 4), "Entry €": round(avg_entry, 2),
                "Price €":     round(current_px_eur, 2) if not np.isnan(current_px_eur) else None,
                "Unreal. P/L": unreal_str, "_unreal_pct": unreal_pct,
                "Weight %":    None,
                "YTD %":       tv("ytd", 100.0), "Vol 60d %": tv("vol_60d", 100.0),
                "Drawdown %":  tv("max_drawdown_12m", 100.0),
                "Trend":       "▲ Above EMA200" if (tv("trend_strength", 1.0) or 0) > 0 else "▼ Below EMA200",
                "Tech Score":  tv("score", 1.0), "Fund Score": fund_score,
                "Mkt Cap":     cap_str, "Sector": sector,
                "_pos_val":    pos_val, "_first_buy": first_buy,
            })
        else:
            closed_rows.append({
                "Ticker":       ticker, "Exchange": exchange, "Company": company,
                "First Buy":    first_buy.strftime("%Y-%m-%d") if first_buy else "N/A",
                "Last Sell":    last_sell.strftime("%Y-%m-%d") if last_sell else "N/A",
                "Realised P/L": f"€{realised_pnl:+,.0f}" if realised_pnl else "N/A",
                "_realised":    realised_pnl,
                "Tech Score":   tv("score", 1.0), "Fund Score": fund_score,
                "Mkt Cap":      cap_str, "Sector": sector,
            })

    if open_rows:
        open_df   = pd.DataFrame(open_rows)
        total_val = open_df["_pos_val"].sum()
        open_df["Weight %"] = (open_df["_pos_val"] / total_val * 100).round(1) if total_val > 0 else 0.0
        open_df = open_df.sort_values("Weight %", ascending=False).reset_index(drop=True)
    else:
        open_df = pd.DataFrame()

    return open_df, pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame()


def compute_portfolio_kpis(pnl_df, holdings_df, transactions, fx_data=None) -> dict:
    default = dict(total_pnl=0, day_pnl=0, wallet=0,
                   total_return_pct=0, cagr_pct=0, sharpe=0, sortino=0)
    if pnl_df is None or pnl_df.empty:
        return default

    pnl_s     = pnl_df.groupby("date")["pnl"].sum().sort_index()
    total_pnl = float(pnl_s.iloc[-1])
    day_pnl   = float(pnl_s.diff().iloc[-1]) if len(pnl_s) >= 2 else 0.0
    wallet    = float(holdings_df["_pos_val"].sum()) if holdings_df is not None and not holdings_df.empty and "_pos_val" in holdings_df.columns else 0.0

    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    buys  = tx[tx["type"] == "buy"]
    sells = tx[tx["type"] == "sell"]
    invested = max(
        float((buys["quantity"] * buys["price"]).sum()) -
        float((sells["quantity"] * sells["price"]).sum()),
        1.0,
    )

    total_return_pct = (wallet - invested) / invested * 100
    n_years          = (pd.Timestamp.today() - tx["date"].min()).days / 365.25
    cagr_pct         = ((wallet / invested) ** (1.0 / max(n_years, 0.01)) - 1) * 100 if wallet > 0 else 0.0

    daily_ret = pnl_s.diff().dropna() / (wallet + 1e-9)
    sharpe    = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252))
    down      = daily_ret[daily_ret < 0]
    sortino   = float(daily_ret.mean() / (down.std() + 1e-9) * np.sqrt(252)) if len(down) > 1 else 0.0

    return dict(total_pnl=total_pnl, day_pnl=day_pnl, wallet=wallet,
                total_return_pct=total_return_pct, cagr_pct=cagr_pct,
                sharpe=sharpe, sortino=sortino)


def build_portfolio_analytics(transactions_path: str, prices: pd.DataFrame) -> pd.DataFrame:
    """Drop-in replacement for original build_portfolio_analytics()."""
    tx         = load_transactions(transactions_path)
    currencies = [c for c in tx["currency"].unique() if c != "EUR"]
    fx_data    = get_fx_data(currencies, tx["date"].min())
    positions  = build_positions(tx, prices, fx_data)
    return compute_pnl(positions, tx, fx_data)
