"""
app.py — QuantEdge Dashboard (Dash + Plotly).

Main entry point for the QuantEdge quantitative analytics web app. Wires
together the analytics, modules, and ingestion packages into a single
Dash application served by Flask (and gunicorn in production).

The app exposes three tabs:

* **Portfolio**   — Real-time P&L, holdings, FX, sector allocation, and
  stop-loss recommendations for the user's actual trades.
* **Screener**    — Multi-factor ranking table combining technical,
  fundamental, and ML scores, plus the current market-regime banner.
* **Strategy Lab** — Interactive backtest of a DCA strategy with live
  weight sliders and a weight optimiser.

Run locally:

    python app.py

Run in production:

    gunicorn --config gunicorn_config.py app:server

Open the UI at http://127.0.0.1:8050 (or the platform-assigned URL).
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

from modules.stop_loss       import compute_stop_recommendations
from modules.backtester      import run_backtest, optimise_weights, benchmark_buyhold, DEFAULT_WEIGHTS
from modules.ml_scorer       import compute_ml_scores
from modules.regime_detector import compute_regime
from analytics.technicals    import compute_technical_features, load_prices_from_db
from analytics.fundamental   import compute_fundamentals
from analytics.portfolio     import build_portfolio_analytics, load_transactions, get_fx_data
from ingestion.universe      import get_universe_df


# ── Logging configuration ────────────────────────────────────────────────────
# Configured once at the application entry point. Library modules use
# logging.getLogger(__name__) and inherit this configuration automatically.
# In production (gunicorn on Railway/Render/Heroku) the logs stream to
# stdout and are captured by the platform's log viewer.

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quantedge.app")


# ════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS — Forest Light
# ════════════════════════════════════════════════════════════════════
ACCENT   = "#2D6A4F"
ACCENT2  = "#1B4332"
BG       = "#FAFAF8"
SURFACE  = "#F0EDE6"
CARD     = "#FFFFFF"
BORDER   = "#D4CFC6"
TEXT     = "#1A1A18"
MUTED    = "#7A7670"
GREEN    = "#2D6A4F"
RED      = "#C0392B"
BLUE     = "#2471A3"
GOLD     = ACCENT
GOLD_DIM = ACCENT2

FONT_DISPLAY = "'Libre Baskerville', Georgia, serif"
FONT_BODY    = "'IBM Plex Mono', 'Courier New', monospace"
FONT_UI      = "'DM Sans', system-ui, sans-serif"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color=TEXT, size=12),
    xaxis=dict(gridcolor="#E8E4DC", linecolor=BORDER, tickcolor=MUTED, zeroline=False),
    yaxis=dict(gridcolor="#E8E4DC", linecolor=BORDER, tickcolor=MUTED, zeroline=False),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER,
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=50, r=30, t=40, b=50),
    colorway=[ACCENT, BLUE, "#E67E22", "#8E44AD", "#1ABC9C", RED],
)


# ════════════════════════════════════════════════════════════════════
#  COMPONENT HELPERS
# ════════════════════════════════════════════════════════════════════
def metric_card(label, value, sub=None, positive=None):
    val_color = GREEN if positive is True else RED if positive is False else ACCENT
    children = [
        html.Div(label, style={
            "color": MUTED, "fontSize": "10px", "letterSpacing": "1.2px",
            "textTransform": "uppercase", "fontFamily": FONT_UI, "marginBottom": "8px",
        }),
        html.Div(value, style={
            "color": val_color, "fontSize": "20px",
            "fontFamily": FONT_DISPLAY, "fontWeight": "700", "lineHeight": "1",
        }),
    ]
    if sub:
        children.append(html.Div(sub, style={
            "color": MUTED, "fontSize": "10px", "fontFamily": FONT_UI, "marginTop": "4px",
        }))
    return html.Div(children, style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderTop": f"3px solid {ACCENT}", "borderRadius": "8px",
        "padding": "16px 18px", "flex": "1", "minWidth": "110px",
    })


def section_header(title, subtitle=None):
    return html.Div([
        html.Span(title, style={
            "fontFamily": FONT_DISPLAY, "fontSize": "17px", "color": TEXT, "fontWeight": "700",
        }),
        html.Span(f"  {subtitle}", style={
            "fontFamily": FONT_UI, "fontSize": "12px", "color": MUTED,
        }) if subtitle else None,
    ], style={"borderBottom": f"1px solid {BORDER}", "paddingBottom": "10px", "marginBottom": "18px"})


def page_header(title, subtitle):
    return html.Div([
        html.Div(title, style={
            "fontFamily": FONT_DISPLAY, "fontSize": "24px",
            "fontWeight": "700", "color": TEXT, "marginBottom": "4px",
        }),
        html.Div(subtitle, style={"fontFamily": FONT_UI, "fontSize": "12px", "color": MUTED}),
    ], style={"marginBottom": "28px"})


def primary_btn(label, btn_id):
    return html.Button(label, id=btn_id, style={
        "background": ACCENT, "color": "white", "border": "none",
        "borderRadius": "6px", "padding": "9px 22px",
        "fontFamily": FONT_UI, "fontWeight": "600", "fontSize": "13px", "cursor": "pointer",
    })


def outline_btn(label, btn_id):
    return html.Button(label, id=btn_id, style={
        "background": "transparent", "color": ACCENT,
        "border": f"1px solid {ACCENT}", "borderRadius": "6px",
        "padding": "9px 22px", "fontFamily": FONT_UI,
        "fontWeight": "600", "fontSize": "13px", "cursor": "pointer",
    })


def card(children, mb="20px", p="20px 24px"):
    return html.Div(children, style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "10px", "padding": p, "marginBottom": mb,
    })


def styled_table(df, table_id, color_cols=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cond = [{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"}]
    if color_cols:
        for col, (lo, hi) in color_cols.items():
            cond += [
                {"if": {"filter_query": f"{{{col}}} >= {hi}", "column_id": col},
                 "color": GREEN, "fontWeight": "500"},
                {"if": {"filter_query": f"{{{col}}} <= {lo}", "column_id": col},
                 "color": RED, "fontWeight": "500"},
            ]
    return dash_table.DataTable(
        id=table_id,
        data=df.round(4).to_dict("records"),
        columns=[{"name": c.replace("_", " ").upper(), "id": c,
                  "type": "numeric" if c in numeric_cols else "text"} for c in df.columns],
        sort_action="native", filter_action="native", page_size=20,
        style_table={"overflowX": "auto", "borderRadius": "8px", "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": SURFACE, "color": ACCENT,
                      "fontFamily": FONT_UI, "fontSize": "10px", "fontWeight": "700",
                      "letterSpacing": "1px", "textTransform": "uppercase",
                      "borderBottom": f"1px solid {BORDER}", "padding": "10px 14px"},
        style_cell={"backgroundColor": CARD, "color": TEXT,
                    "fontFamily": FONT_BODY, "fontSize": "12px",
                    "padding": "9px 14px", "border": f"1px solid {BORDER}", "textAlign": "right"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                 for c in df.columns if df[c].dtype == object],
        style_data_conditional=cond,
    )


# ════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ════════════════════════════════════════════════════════════════════
def build_pnl_chart(pnl_df):
    fig = go.Figure()
    for user in pnl_df["user"].unique():
        u = pnl_df[pnl_df["user"] == user].sort_values("date")
        fig.add_trace(go.Scatter(
            x=u["date"], y=u["pnl"], name=user, mode="lines",
            line=dict(width=2.5, color=ACCENT),
            fill="tozeroy", fillcolor="rgba(45,106,79,0.07)",
        ))
    fig.update_layout(**PLOTLY_THEME,
        title=dict(text="Cumulative P&L (€)", font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
        hovermode="x unified", yaxis_tickprefix="€")
    return fig


def build_stock_vs_spy_chart(prices, ticker, first_buy_date):
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    prices = prices.copy()
    _dates = pd.to_datetime(prices["date"])
    prices["date"] = _dates.dt.tz_convert(None) if _dates.dt.tz is not None else _dates

    t_df = prices[prices["ticker"] == ticker].sort_values("date").copy()
    s_df = prices[prices["ticker"] == "SPY"].sort_values("date").copy()

    if t_df.empty:
        return go.Figure().update_layout(**PLOTLY_THEME)

    # x-axis starts at acquisition date (first_buy_date) if available,
    # otherwise falls back to the first available historical date
    if first_buy_date is not None:
        axis_start = pd.to_datetime(first_buy_date).tz_localize(None) \
                     if pd.to_datetime(first_buy_date).tzinfo is None \
                     else pd.to_datetime(first_buy_date).tz_convert(None)
    else:
        axis_start = pd.to_datetime(t_df["date"].iloc[0])

    # Filter ticker prices from acquisition date onwards
    t_df = t_df[t_df["date"] >= axis_start].set_index("date")
    if t_df.empty:
        return go.Figure().update_layout(**PLOTLY_THEME)

    t_pct = (t_df[price_col] / t_df[price_col].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_pct.index, y=t_pct.values, name=ticker,
                             mode="lines", line=dict(width=2.5, color=ACCENT)))

    if not s_df.empty:
        s_df = s_df[s_df["date"] >= axis_start].set_index("date")
        if not s_df.empty and s_df[price_col].iloc[0] > 0:
            s_pct = (s_df[price_col] / s_df[price_col].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=s_pct.index, y=s_pct.values, name="S&P 500",
                                     mode="lines", line=dict(width=1.5, color=MUTED, dash="dot")))

    if first_buy_date is not None:
        entry_ts = pd.to_datetime(first_buy_date).value  # nanoseconds → int, always works with datetime axes
        fig.add_vline(x=entry_ts,
                      line=dict(color=RED, dash="dash", width=1.5),
                      annotation_text="  My entry",
                      annotation_font=dict(color=RED, size=11, family=FONT_UI),
                      annotation_position="top right")

    fig.update_layout(**PLOTLY_THEME,
        title=dict(text=f"{ticker} vs S&P 500  —  % return since entry",
                   font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
        yaxis_title="Return (%)", yaxis_ticksuffix="%", hovermode="x unified")
    return fig


def build_equity_chart(equity, benchmark=None):
    fig = go.Figure()
    if equity is not None and not equity.empty:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["value"], name="Strategy",
                                 mode="lines", line=dict(width=2.5, color=ACCENT)))
    if benchmark is not None and not benchmark.empty and equity is not None and not equity.empty:
        scale = equity["value"].iloc[0] / benchmark["value"].iloc[0] \
                if benchmark["value"].iloc[0] > 0 else 1.0
        fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["value"] * scale,
                                 name="Benchmark (SPY)", mode="lines",
                                 line=dict(width=1.5, color=MUTED, dash="dot")))
    fig.update_layout(**PLOTLY_THEME,
        title=dict(text="Equity Curve", font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
        yaxis_tickprefix="€", hovermode="x unified")
    return fig


def build_drawdown_chart(equity):
    eq = equity.set_index("date")["value"]
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", mode="lines",
                               line=dict(color=RED, width=1.5),
                               fillcolor="rgba(192,57,43,0.10)"))
    fig.update_layout(**PLOTLY_THEME,
        title=dict(text="Drawdown (%)", font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
        yaxis_ticksuffix="%")
    return fig


def build_stop_chart(df_ticker, ticker, stop_price):
    df = df_ticker.sort_values("date").tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker,
        increasing_line_color=GREEN, decreasing_line_color=RED))
    try:
        sp = float(stop_price)
        if not np.isnan(sp):
            fig.add_hline(y=sp, line=dict(color=RED, dash="dash", width=1.5),
                          annotation_text=f"  Stop  €{sp:.2f}",
                          annotation_font=dict(color=RED, size=11, family=FONT_UI))
    except (TypeError, ValueError):
        pass
    fig.update_layout(**PLOTLY_THEME, xaxis_rangeslider_visible=False,
        title=dict(text=f"{ticker}  —  last 180 days + optimal stop",
                   font=dict(family=FONT_DISPLAY, size=15, color=TEXT)))
    return fig


def build_score_scatter(factor_df):
    df = factor_df.dropna(subset=["technical_score", "fundamental_score"])
    if df.empty:
        return go.Figure().update_layout(**PLOTLY_THEME)
    fig = px.scatter(df, x="technical_score", y="fundamental_score", text="ticker",
                     color="technical_score",
                     color_continuous_scale=[[0, "#D4CFC6"], [0.5, "#52B788"], [1, "#2D6A4F"]],
                     hover_data=["rating", "growth"] if "rating" in df.columns else None)
    fig.update_traces(textposition="top center", textfont=dict(size=9, color=TEXT))
    fig.update_layout(**PLOTLY_THEME,
        title=dict(text="Opportunity Map — Technical vs Fundamental",
                   font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
        xaxis_title="Technical Score", yaxis_title="Fundamental Score", coloraxis_showscale=False)
    return fig


def build_backtest_compare(results_df):
    if results_df.empty:
        return go.Figure()
    top    = results_df.head(10)
    labels = [f"#{i+1}" for i in range(len(top))]
    colors = [ACCENT if i == 0 else "#C8E6D8" for i in range(len(top))]
    fig    = make_subplots(rows=1, cols=3,
                           subplot_titles=["CAGR %", "Sharpe Ratio", "Max Drawdown %"])
    for col, metric in enumerate(["cagr_pct", "sharpe_ratio", "max_drawdown_pct"], 1):
        fig.add_trace(go.Bar(x=labels, y=top[metric], marker_color=colors,
                             showlegend=False), row=1, col=col)
    fig.update_layout(**PLOTLY_THEME,
        title=dict(text="Optimisation Results — Top 10 Weight Configurations",
                   font=dict(family=FONT_DISPLAY, size=15, color=TEXT)))
    return fig


# ════════════════════════════════════════════════════════════════════
#  PORTFOLIO DATA HELPERS
# ════════════════════════════════════════════════════════════════════

def build_allocation_charts(holdings_df):
    """Side-by-side donut charts: by asset and by sector."""
    if holdings_df is None or holdings_df.empty:
        return go.Figure().update_layout(**PLOTLY_THEME)

    PALETTE = [
        "#2D6A4F", "#52B788", "#95D5B2", "#B7E4C7",
        "#1B4332", "#74C69D", "#40916C", "#D8F3DC",
        "#2471A3", "#E67E22", "#8E44AD", "#C0392B",
    ]

    # Asset slice
    asset_df = holdings_df[["Ticker", "_pos_val"]].copy()
    asset_df = asset_df[asset_df["_pos_val"] > 0]

    # Sector slice
    sector_df = holdings_df[["Sector", "_pos_val"]].copy()
    sector_df = sector_df[sector_df["_pos_val"] > 0]
    sector_df = sector_df.groupby("Sector", as_index=False)["_pos_val"].sum()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["By Asset", "By Sector"],
    )

    fig.add_trace(go.Pie(
        labels=asset_df["Ticker"],
        values=asset_df["_pos_val"].round(2),
        hole=0.52,
        marker=dict(colors=PALETTE[:len(asset_df)],
                    line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, family=FONT_UI),
        hovertemplate="%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>",
        name="Asset",
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=sector_df["Sector"],
        values=sector_df["_pos_val"].round(2),
        hole=0.52,
        marker=dict(colors=PALETTE[2:2+len(sector_df)],
                    line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, family=FONT_UI),
        hovertemplate="%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>",
        name="Sector",
    ), row=1, col=2)

    pie_theme = {**PLOTLY_THEME,
                 "legend": dict(orientation="v", x=1.02, y=0.5,
                                font=dict(size=11, family=FONT_UI)),
                 "margin": dict(l=20, r=140, t=40, b=20),
                 "showlegend": True}
    fig.update_layout(**pie_theme, height=380)
    return fig


# ── FX helpers ───────────────────────────────────────────────────────
def _norm_dates(s):
    """Return tz-naive datetime series."""
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _fx_rate(fx_data, currency, date):
    """EUR per 1 unit of `currency` on `date`, forward-filled from daily table."""
    if not fx_data or currency == "EUR" or currency not in fx_data:
        return 1.0
    df = fx_data[currency].copy()
    df["date"] = _norm_dates(df["date"])
    df = df.set_index("date").sort_index()
    dt = pd.Timestamp(date)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None).replace(tzinfo=None)
    avail = df.index[df.index <= dt]
    return float(df["fx"].iloc[0]) if avail.empty else float(df.loc[avail[-1], "fx"])


def _fx_latest(fx_data, currency):
    """Most recent EUR per 1 unit of `currency`."""
    if not fx_data or currency == "EUR" or currency not in fx_data:
        return 1.0
    df = fx_data[currency].copy()
    df["date"] = _norm_dates(df["date"])
    return float(df.sort_values("date")["fx"].iloc[-1])


# ── Stock split helpers ──────────────────────────────────────────────
import yfinance as _yf

_splits_cache: dict = {}   # ticker -> pd.Series of split ratios (tz-naive index)


def _get_splits(ticker: str) -> "pd.Series":
    """Fetch and cache yfinance split history for ticker (tz-naive dates)."""
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
    """Cumulative split ratio for all splits that occurred AFTER after_date.

    Example: SMCI 10-for-1 split on 2024-09-30.
      Buy on 2024-01-01 -> factor = 10  (adjust qty up, price down)
      Buy on 2024-10-01 -> factor = 1   (split already reflected in transaction)
    """
    s = _get_splits(ticker)
    if s.empty:
        return 1.0
    dt = pd.Timestamp(after_date)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert(None).replace(tzinfo=None)
    future = s[s.index > dt]
    return float(future.prod()) if not future.empty else 1.0


def compute_holdings_table(transactions, prices, df_tech, df_fund, stocks_df, fx_data=None):
    """
    Devolve dois DataFrames: (open_df, closed_df).
    open_df   — positions with qty > 0 (Open Positions)
    closed_df — fully closed positions (Closed Positions)
    """
    tx        = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    price_col  = "adj_close" if "adj_close" in prices.columns else "close"
    usd_rate   = _fx_latest(fx_data, "USD")

    open_rows   = []
    closed_rows = []

    for ticker in tx["ticker"].unique():
        t = tx[tx["ticker"] == ticker].sort_values("date")
        qty, cost, first_buy, last_sell = 0.0, 0.0, None, None
        realised_pnl = 0.0

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
                # Use FX rate on the actual sell date, not the current rate
                sell_rate    = _fx_rate(fx_data, row["currency"], row["date"])
                sell_px_eur  = row["price"] * sell_rate
                realised_pnl += adj_qty * (sell_px_eur - avg_c)
                cost     -= avg_c * adj_qty
                qty      -= adj_qty
                last_sell = row["date"]

        # ── Shared info ──────────────────────────────────────────────
        info     = stocks_df[stocks_df["Symbol"] == ticker]
        company  = info["Company Name"].iloc[0] if not info.empty else ticker
        mkt_cap  = info["Market Cap"].iloc[0]   if not info.empty else np.nan
        sector   = info["Sector"].iloc[0]        if not info.empty else "N/A"
        exchange = info["Exchange"].iloc[0]      if not info.empty and "Exchange" in info.columns else "N/A"

        if not (isinstance(mkt_cap, float) and np.isnan(mkt_cap)):
            try:
                mc = float(mkt_cap)
                if   mc >= 1e12: cap_str = f"${mc/1e12:.1f}T"
                elif mc >= 1e9:  cap_str = f"${mc/1e9:.0f}B"
                else:             cap_str = f"${mc/1e6:.0f}M"
            except:
                cap_str = "N/A"
        else:
            cap_str = "N/A"

        tech = df_tech[df_tech["ticker"] == ticker].sort_values("date").tail(1) \
               if df_tech is not None and not df_tech.empty else pd.DataFrame()

        def tv(col, scale=1.0):
            if not tech.empty and col in tech.columns:
                v = tech[col].iloc[0]
                try:
                    return round(float(v) * scale, 1) if not np.isnan(float(v)) else None
                except:
                    return None
            return None

        fund = df_fund[df_fund["ticker"] == ticker] \
               if df_fund is not None and not df_fund.empty else pd.DataFrame()
        fund_score = round(float(fund["fundamental_score"].iloc[0]), 2)                      if not fund.empty and pd.notna(fund["fundamental_score"].iloc[0])                      else None

        # ── OPEN POSITION ────────────────────────────────────────────
        if qty >= 0.01:
            avg_entry = cost / qty
            tp        = prices[prices["ticker"] == ticker].sort_values("date")
            current_px     = float(tp[price_col].iloc[-1]) if not tp.empty else np.nan
            current_px_eur = current_px * usd_rate if not np.isnan(current_px) else np.nan
            pos_val        = current_px * qty if not np.isnan(current_px) else 0.0
            unreal_val     = (current_px_eur - avg_entry) * qty if not np.isnan(current_px_eur) else np.nan
            unreal_pct     = (current_px_eur / avg_entry - 1.0) * 100 \
                             if avg_entry > 0 and not np.isnan(current_px_eur) else np.nan
            unreal_str     = f"€{unreal_val:+,.0f}  ({unreal_pct:+.1f}%)" \
                             if unreal_val is not None and not np.isnan(unreal_val) else "N/A"

            open_rows.append({
                "Ticker":      ticker,
                "Exchange":    exchange,
                "Company":     company,
                "Qty":         round(qty, 4),
                "Entry €":     round(avg_entry, 2),
                "Price €":     round(current_px_eur, 2) if not np.isnan(current_px_eur) else None,
                "Unreal. P/L": unreal_str,
                "_unreal_pct": unreal_pct,
                "Weight %":    None,
                "YTD %":       tv("ytd",              100.0),
                "Vol 60d %":   tv("vol_60d",          100.0),
                "Drawdown %":  tv("max_drawdown_12m", 100.0),
                "Trend":       "▲ Above EMA200" if (tv("trend_strength", 1.0) or 0) > 0 else "▼ Below EMA200",
                "Tech Score":  tv("score",              1.0),
                "Fund Score":  fund_score,
                "Mkt Cap":     cap_str,
                "Sector":      sector,
                "_pos_val":    pos_val,
                "_first_buy":  first_buy,
            })

        # ── CLOSED POSITION ──────────────────────────────────────────
        else:
            # For closed positions show the realised P/L
            realised_str = f"€{realised_pnl:+,.0f}" if realised_pnl != 0 else "N/A"
            closed_rows.append({
                "Ticker":        ticker,
                "Exchange":      exchange,
                "Company":       company,
                "First Buy":     first_buy.strftime("%Y-%m-%d") if first_buy else "N/A",
                "Last Sell":     last_sell.strftime("%Y-%m-%d") if last_sell else "N/A",
                "Realised P/L":  realised_str,
                "_realised":     realised_pnl,
                "Tech Score":    tv("score", 1.0),
                "Fund Score":    fund_score,
                "Mkt Cap":       cap_str,
                "Sector":        sector,
            })

    # ── Open positions — compute Weight % ───────────────────────────
    if open_rows:
        open_df   = pd.DataFrame(open_rows)
        total_val = open_df["_pos_val"].sum()
        open_df["Weight %"] = (open_df["_pos_val"] / total_val * 100).round(1) if total_val > 0 else 0.0
        open_df = open_df.sort_values("Weight %", ascending=False).reset_index(drop=True)
    else:
        open_df = pd.DataFrame()

    closed_df = pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame()

    return open_df, closed_df


def compute_portfolio_kpis(pnl_df, holdings_df, transactions, fx_data=None):
    default = dict(total_pnl=0, day_pnl=0, wallet=0,
                   total_return_pct=0, cagr_pct=0, sharpe=0, sortino=0)
    if pnl_df is None or pnl_df.empty:
        return default

    pnl_s     = pnl_df.groupby("date")["pnl"].sum().sort_index()
    total_pnl = float(pnl_s.iloc[-1])
    day_pnl   = float(pnl_s.diff().iloc[-1]) if len(pnl_s) >= 2 else 0.0
    wallet    = float(holdings_df["_pos_val"].sum()) \
                if holdings_df is not None and not holdings_df.empty \
                   and "_pos_val" in holdings_df.columns else 0.0

    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    # Use raw transaction prices — same basis as _pos_val (raw current_px)
    # so total_return_pct and CAGR remain internally consistent
    buys  = tx[tx["type"] == "buy"]
    sells = tx[tx["type"] == "sell"]
    invested = max(
        float((buys["quantity"]  * buys["price"]).sum()) -
        float((sells["quantity"] * sells["price"]).sum()),
        1.0
    )

    total_return_pct = (wallet - invested) / invested * 100
    n_years  = (pd.Timestamp.today() - tx["date"].min()).days / 365.25
    cagr_pct = ((wallet / invested) ** (1.0 / max(n_years, 0.01)) - 1) * 100 \
               if wallet > 0 else 0.0

    daily_ret = pnl_s.diff().dropna() / (wallet + 1e-9)
    sharpe    = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252))
    down      = daily_ret[daily_ret < 0]
    sortino   = float(daily_ret.mean() / (down.std() + 1e-9) * np.sqrt(252)) \
                if len(down) > 1 else 0.0

    return dict(total_pnl=total_pnl, day_pnl=day_pnl, wallet=wallet,
                total_return_pct=total_return_pct, cagr_pct=cagr_pct,
                sharpe=sharpe, sortino=sortino)


# ════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ════════════════════════════════════════════════════════════════════
# The cache is shared across every Dash callback. Without a lock, multiple
# callbacks firing concurrently on the first request would all see an
# empty cache and each kick off the full FMP refresh in parallel — which
# multiplied API calls 3-6x and caused cascading rate-limit failures.
# The lock + double-checked pattern guarantees the expensive load runs
# exactly once; subsequent callers simply read the populated dict.
import threading as _threading

_cache: dict = {}
_cache_lock = _threading.Lock()

# ════════════════════════════════════════════════════════════════════
#  SCREENER UNIVERSE — controls which tickers are loaded into the
#  screener and Strategy Lab. Pulled from config.SCREENER_UNIVERSE,
#  which itself is sourced from the SCREENER_UNIVERSE environment
#  variable (or .env). Set the variable to one of:
#
#    "all"                    → every active ticker in company_info
#    500                      → first N tickers from the company_info table
#    "AAPL,MSFT,NVDA,..."     → comma-separated explicit list
#
#  Your portfolio tickers are always included regardless of this setting.
#  First load with "all" (~7000 tickers) takes 20–40 min; subsequent
#  loads are instant thanks to the in-process cache.
# ════════════════════════════════════════════════════════════════════
from config import SCREENER_UNIVERSE


def load_all_data(transactions_path, years_back=10):
    # Fast path: cache already populated, no lock contention.
    if "data" in _cache:
        return _cache["data"]

    # Slow path: only one thread does the heavy load; the rest wait and
    # then return the populated cache.
    with _cache_lock:
        if "data" in _cache:
            return _cache["data"]
        return _load_all_data_locked(transactions_path, years_back)


def _load_all_data_locked(transactions_path, years_back):
    """Internal helper. Caller MUST hold ``_cache_lock``."""
    tx         = load_transactions(transactions_path)
    tickers_tx = tx["ticker"].unique().tolist()
    # Pull the stock universe from the company_info table (replaces the
    # legacy stocks_all_pages.csv file).
    stocks_df  = get_universe_df(active_only=True)

    # ── Build screener ticker list from SCREENER_UNIVERSE ────────────
    all_tickers = (
        stocks_df["Symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    if isinstance(SCREENER_UNIVERSE, str) and SCREENER_UNIVERSE.lower() == "all":
        tickers_screen = all_tickers
        label = "all"
    elif isinstance(SCREENER_UNIVERSE, int):
        # Rank by market cap and take the top N, mirroring how setup.py
        # (via fetch_screener_universe) chooses which tickers to ingest.
        # Only the screener universe has market_cap populated in
        # company_info, so this selects exactly the tickers that have
        # price/fundamental data — instead of an arbitrary first-N slice
        # that left ~20% of the screener blank.
        ranked = (
            stocks_df.dropna(subset=["Market Cap"])
            .sort_values("Market Cap", ascending=False)
        )
        tickers_screen = (
            ranked["Symbol"].dropna().astype(str).str.upper()
            .head(SCREENER_UNIVERSE).tolist()
        )
        label = f"top {SCREENER_UNIVERSE} by market cap"
    elif isinstance(SCREENER_UNIVERSE, list):
        tickers_screen = [t.upper() for t in SCREENER_UNIVERSE]
        label = f"custom list ({len(tickers_screen)})"
    else:
        tickers_screen = all_tickers
        label = "all (fallback)"

    # Always include portfolio tickers
    tickers_screen = list(set(tickers_screen) | set(tickers_tx))

    logger.info("Loading %d tickers (%s)", len(tickers_screen), label)

    prices_port   = load_prices_from_db(tickers_tx,    years_back=years_back)
    prices_spy    = load_prices_from_db(["SPY"],        years_back=years_back)
    prices_screen = load_prices_from_db(tickers_screen, years_back=years_back)

    prices_all = pd.concat(
        [p for p in [prices_port, prices_spy, prices_screen] if p is not None],
        ignore_index=True,
    ).drop_duplicates(subset=["ticker", "date"])

    df_tech = compute_technical_features(prices_all)

    # Compute fundamentals for all tickers in the screener universe.
    # With FMP this covers the full SCREENER_UNIVERSE set.
    # For SCREENER_UNIVERSE="all" (~7000 tickers) this will take a while;
    # set SCREENER_UNIVERSE to a number or list for faster development runs.
    all_fund_tickers = list(set(tickers_screen) | set(tickers_tx))
    logger.info("Fundamentals for %d tickers...", len(all_fund_tickers))
    df_fund = compute_fundamentals(all_fund_tickers)

    df_tech_last = (
        df_tech.sort_values("date").groupby("ticker").tail(1)
        [["ticker", "score"]].rename(columns={"score": "technical_score"})
    )
    # factor_table carries technical score only; fundamentals are in df_fund
    # and merged into the screener table at render time
    factor_table = df_tech_last.copy()

    if prices_port is not None:
        # Pass the raw price frame; build_positions() internally selects
        # adj_close (preferred) or close and renames it to "price". A
        # pre-rename here would create two columns named "price" and
        # crash with "Cannot set a DataFrame with multiple columns to
        # the single column price_eur".
        pnl = build_portfolio_analytics(transactions_path, prices_port)
    else:
        pnl = pd.DataFrame()

    stops = compute_stop_recommendations(prices_port, tx) \
            if prices_port is not None else pd.DataFrame()

    currencies = [c for c in tx["currency"].unique() if c != "EUR"]
    fx_data    = get_fx_data(currencies, tx["date"].min()) if currencies else {}

    # ── ML Scores + Regime (usa dados já carregados) ─────────────
    ml_scores = compute_ml_scores(prices_all)
    regime    = compute_regime(prices_all)

    _cache["data"] = dict(
        transactions=tx, prices=prices_all, prices_port=prices_port,
        df_tech=df_tech, df_fund=df_fund, factor_table=factor_table,
        pnl=pnl, stops=stops, stocks_df=stocks_df,
        fx_data=fx_data, ml_scores=ml_scores, regime=regime,
    )
    return _cache["data"]


# ════════════════════════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════════════════════════
GOOGLE_FONTS = html.Link(
    rel="stylesheet",
    href=("https://fonts.googleapis.com/css2?"
          "family=Libre+Baskerville:wght@700"
          "&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700"
          "&family=IBM+Plex+Mono:wght@400;500"
          "&family=Lora:wght@700"
          "&family=Inter:wght@400;500;700"
          "&display=swap"),
)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    suppress_callback_exceptions=True,
    title="QuantEdge",
)

# WSGI entry point for gunicorn. The Procfile / Dockerfile start the
# app as `gunicorn ... app:server`, so the Flask server underneath
# Dash must be exposed at module level. Without this, gunicorn aborts
# with "Failed to find attribute 'server' in 'app'". `python app.py`
# (which calls app.run further down) never needed it, which is why
# this only surfaced on the first real gunicorn deployment.
server = app.server


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
def nav_btn(label, icon_cls, btn_id, active=False):
    return html.Button(
        [html.I(className=icon_cls), html.Span(label, className="nav-label")],
        id=btn_id,
        className="nav-item active" if active else "nav-item",
        n_clicks=0,
    )


sidebar = html.Div([
    html.Div([
        html.Div("Q", className="sidebar-logo-icon"),
        html.Div("QuantEdge", className="sidebar-logo-text"),
    ], className="sidebar-logo"),
    nav_btn("Portfolio", "bi bi-graph-up-arrow",  "nav-portfolio", active=True),
    nav_btn("Screener",  "bi bi-search",          "nav-screener"),
    nav_btn("Watchlist", "bi bi-bookmark-star",   "nav-watchlist"),
    nav_btn("Alerts",    "bi bi-bell",            "nav-alerts"),
    html.Div([
        html.Div(className="sidebar-footer-dot"),
        html.Div("Live data", className="sidebar-footer-text"),
    ], className="sidebar-footer"),
], id="sidebar", className="sidebar")


# ════════════════════════════════════════════════════════════════════
#  PAGE LAYOUTS
# ════════════════════════════════════════════════════════════════════
def layout_portfolio():
    return html.Div([
        page_header("Portfolio", "Performance · Holdings · Risk Management"),

        html.Div(id="portfolio-kpis",
                 style={"display": "flex", "gap": "12px",
                        "flexWrap": "wrap", "marginBottom": "24px"}),

        card([dcc.Graph(id="pnl-chart", config={"displayModeBar": False})]),

        card([
            section_header("Open Positions"),
            html.Div(id="holdings-table"),
        ]),

        card([
            section_header("Closed Positions"),
            html.Div(id="closed-positions-table"),
        ]),

        card([
            section_header("Allocation"),
            dcc.Graph(id="allocation-charts", config={"displayModeBar": False}),
        ]),

        card([
            html.Div([
                html.Label("Compare vs S&P 500:", style={
                    "fontFamily": FONT_UI, "fontSize": "12px",
                    "color": MUTED, "marginRight": "12px",
                }),
                dcc.Dropdown(id="spy-ticker-selector",
                             style={"minWidth": "160px", "fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
            dcc.Graph(id="stock-spy-chart", config={"displayModeBar": False}),
        ]),

        card([
            section_header("Stop Loss Recommendations", "optimal exit strategy per position"),
            html.P(
                "ATR-based, trailing, and fixed-% stops backtested on full history. "
                "The configuration with highest expectancy is selected per ticker. "
                "Kelly fractions indicate the theoretically optimal risk per trade.",
                style={"color": MUTED, "fontSize": "12px",
                       "fontFamily": FONT_UI, "marginBottom": "16px"},
            ),
            html.Div(id="stop-table"),
        ]),

        card([
            html.Div([
                html.Label("Stop chart:", style={
                    "fontFamily": FONT_UI, "fontSize": "12px",
                    "color": MUTED, "marginRight": "12px",
                }),
                dcc.Dropdown(id="stop-ticker-selector",
                             style={"minWidth": "160px", "fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
            dcc.Graph(id="stop-candle-chart", config={"displayModeBar": False}),
        ], mb="0"),
    ])


def layout_screener():
    return html.Div([
        page_header("Screener", "Technical + Fundamental + ML scoring"),

        # ── Regime Banner ────────────────────────────────────────────
        html.Div(id="regime-banner", style={"marginBottom": "20px"}),

        # ── Filters ──────────────────────────────────────────────────
        card([
            html.Div([
                html.Div([
                    html.Label("Min Technical Score",
                               style={"color": MUTED, "fontSize": "12px", "fontFamily": FONT_UI}),
                    dcc.Slider(id="min-tech-score", min=0, max=1, step=0.05, value=0,
                               marks={0: "0", 0.5: "0.5", 1: "1"},
                               tooltip={"always_visible": True, "placement": "bottom"}),
                ], style={"flex": "1", "paddingRight": "24px"}),
                html.Div([
                    html.Label("Sector Filter",
                               style={"color": MUTED, "fontSize": "12px", "fontFamily": FONT_UI}),
                    dcc.Dropdown(id="sector-filter", multi=True,
                                 placeholder="All sectors", style={"fontSize": "13px"}),
                ], style={"flex": "1", "paddingRight": "24px"}),
                html.Div([
                    html.Label("Top N to display",
                               style={"color": MUTED, "fontSize": "12px", "fontFamily": FONT_UI}),
                    dcc.Slider(id="screen-topn", min=10, max=100, step=10, value=50,
                               marks={10: "10", 50: "50", 100: "100"},
                               tooltip={"always_visible": True, "placement": "bottom"}),
                ], style={"flex": "1", "paddingRight": "24px"}),
                html.Div([
                    primary_btn("Apply", "apply-screen-btn"),
                    html.Button("Clear", id="clear-screen-btn", n_clicks=0, style={
                        "background": "transparent", "color": MUTED,
                        "border": f"1px solid {BORDER}", "borderRadius": "6px",
                        "padding": "9px 18px", "fontFamily": FONT_UI,
                        "fontWeight": "500", "fontSize": "13px",
                        "cursor": "pointer", "marginLeft": "8px",
                    }),
                ], style={"display": "flex", "alignItems": "flex-end"}),
            ], style={"display": "flex", "alignItems": "flex-end", "gap": "12px"}),
        ]),

        # ── Metric selector + logo grid ──────────────────────────────
        card([
            html.Div([
                html.Label("Rank by:", style={
                    "fontFamily": FONT_UI, "fontSize": "12px",
                    "color": MUTED, "marginRight": "12px",
                }),
                dcc.Dropdown(
                    id="screen-metric",
                    options=[
                        {"label": "Technical Score", "value": "technical_score"},
                        {"label": "ML Score", "value": "ml_score"},
                        {"label": "Fundamental Score", "value": "fundamental_score"},
                        {"label": "YTD %", "value": "YTD %"},
                    ],
                    value="technical_score", clearable=False,
                    style={"minWidth": "180px", "fontSize": "13px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
            html.Div(id="logo-grid"),
        ]),

        card([
            section_header("Sector Strength",
                           "average technical score by sector"),
            dcc.Graph(id="sector-chart", config={"displayModeBar": False}),
        ]),

        # ── Factor Table ──────────────────────────────────────────────
        card([
            section_header("Factor Table",
                           "all scored tickers · N/A = no analyst coverage"),
            html.Div(id="factor-table"),
        ], mb="0"),
    ])


def layout_strategy():
    return html.Div([
        page_header("Strategy Lab", "Backtest · Optimise · Simulate"),
        card([
            html.Div([
                html.Div([
                    html.Label("Start Date", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Input(id="bt-start", type="text", value="2015-01-01",
                              style={"background": SURFACE, "color": TEXT,
                                     "border": f"1px solid {BORDER}", "borderRadius": "6px",
                                     "padding": "8px 12px", "width": "140px",
                                     "fontFamily": FONT_BODY}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Monthly Budget (€)", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Input(id="bt-budget", type="number", value=1000,
                              style={"background": SURFACE, "color": TEXT,
                                     "border": f"1px solid {BORDER}", "borderRadius": "6px",
                                     "padding": "8px 12px", "width": "140px",
                                     "fontFamily": FONT_BODY}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Top N stocks", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Slider(id="bt-topn", min=3, max=20, step=1, value=5,
                               marks={3: "3", 10: "10", 20: "20"},
                               tooltip={"always_visible": True, "placement": "bottom"}),
                ], style={"flex": "2", "padding": "0 12px"}),
                html.Div([
                    html.Label("Rebalance (months)", style={"color": MUTED, "fontSize": "12px"}),
                    dcc.Slider(id="bt-rebal", min=1, max=12, step=1, value=1,
                               marks={1: "1", 3: "Q", 6: "6M", 12: "1Y"},
                               tooltip={"always_visible": True, "placement": "bottom"}),
                ], style={"flex": "2", "padding": "0 12px"}),
            ], style={"display": "flex", "gap": "24px", "alignItems": "flex-end",
                       "marginBottom": "24px"}),
            html.Div([
                html.Div("SCORE WEIGHTS", style={"fontFamily": FONT_UI, "fontSize": "10px",
                                                  "color": ACCENT, "letterSpacing": "2px",
                                                  "marginBottom": "16px"}),
                html.Div([
                    _weight_slider("Momentum 12M",   "w-mom",    0.30),
                    _weight_slider("Return 6M",      "w-ret",    0.20),
                    _weight_slider("Trend Strength", "w-trend",  0.15),
                    _weight_slider("MACD",           "w-macd",   0.10),
                    _weight_slider("Volatility (−)", "w-vol",   -0.20),
                    _weight_slider("Drawdown (−)",   "w-dd",    -0.15),
                ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
            ], style={"background": SURFACE, "borderRadius": "8px",
                       "border": f"1px solid {BORDER}", "padding": "20px", "marginBottom": "20px"}),
            html.Div([
                primary_btn("▶  Run Backtest", "run-bt-btn"),
                outline_btn("⚡  Auto-Optimise Weights", "run-opt-btn"),
            ], style={"display": "flex", "gap": "12px"}),
        ]),
        html.Div(id="bt-kpis", style={"display": "flex", "gap": "12px",
                                       "flexWrap": "wrap", "marginBottom": "20px"}),
        card([dcc.Graph(id="bt-equity-chart", config={"displayModeBar": False})]),
        card([dcc.Graph(id="bt-dd-chart", config={"displayModeBar": False})]),
        html.Div([
            section_header("Optimisation Results", "grid search over weight combinations"),
            dcc.Graph(id="opt-compare-chart", config={"displayModeBar": False}),
            html.Div(id="opt-table"),
        ], id="opt-section", style={"background": CARD, "border": f"1px solid {BORDER}",
                                     "borderRadius": "10px", "padding": "24px",
                                     "marginBottom": "20px", "display": "none"}),
        card([section_header("Trade Log"), html.Div(id="trade-log-table")], mb="0"),
    ])


def _weight_slider(label, slider_id, default):
    return html.Div([
        html.Div([
            html.Span(label, style={"fontFamily": FONT_UI, "fontSize": "12px", "color": TEXT}),
            html.Span(id=f"{slider_id}-out",
                      style={"fontFamily": FONT_BODY, "fontSize": "12px", "color": ACCENT}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"}),
        dcc.Slider(id=slider_id, min=-0.5, max=0.5, step=0.05, value=default,
                   marks={-0.5: "-0.5", 0: "0", 0.5: "0.5"},
                   tooltip={"always_visible": False}),
    ])


# ════════════════════════════════════════════════════════════════════
#  WATCHLIST + ALERTS — supporting layouts and callbacks
# ════════════════════════════════════════════════════════════════════
# Single-user mode: the Watchlist and Alert tables in db.models are
# keyed by user_name so the platform can support multiple users in the
# future. Until proper authentication is added, every row is stored
# under DEFAULT_USER.
DEFAULT_USER = os.environ.get("QUANTEDGE_USER", "ricardo")

from sqlalchemy import select, delete as sa_delete
from db.connection import Session as DBSession
from db.models import Watchlist as WatchlistRow, Alert as AlertRow


def _watchlist_rows():
    """Return the current user's watchlist as a list of dicts."""
    with DBSession() as s:
        rows = s.query(WatchlistRow).filter(
            WatchlistRow.user_name == DEFAULT_USER
        ).order_by(WatchlistRow.added_at.desc()).all()
        return [
            {"ticker": r.ticker, "notes": r.notes or "",
             "added": r.added_at.strftime("%Y-%m-%d") if r.added_at else ""}
            for r in rows
        ]


def _active_alert_rows():
    """Return active (non-triggered) alerts for the current user."""
    with DBSession() as s:
        rows = s.query(AlertRow).filter(
            AlertRow.user_name == DEFAULT_USER,
            AlertRow.is_active == True,         # noqa: E712
            AlertRow.triggered == False,        # noqa: E712
        ).order_by(AlertRow.created_at.desc()).all()
        return [
            {"id": r.id, "ticker": r.ticker, "alert_type": r.alert_type,
             "threshold": float(r.threshold), "email": r.email or "—",
             "created": r.created_at.strftime("%Y-%m-%d") if r.created_at else ""}
            for r in rows
        ]


def _triggered_alert_rows(days_back: int = 30):
    """Return alerts triggered in the last `days_back` days."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    with DBSession() as s:
        rows = s.query(AlertRow).filter(
            AlertRow.user_name == DEFAULT_USER,
            AlertRow.triggered == True,         # noqa: E712
            AlertRow.triggered_at >= cutoff,
        ).order_by(AlertRow.triggered_at.desc()).all()
        return [
            {"ticker": r.ticker, "alert_type": r.alert_type,
             "threshold": float(r.threshold),
             "triggered_at": r.triggered_at.strftime("%Y-%m-%d %H:%M")
                             if r.triggered_at else "—"}
            for r in rows
        ]


def _watchlist_table_component(rows):
    if not rows:
        return html.Div(
            "No tickers in your watchlist yet. Add one above.",
            style={"color": MUTED, "fontStyle": "italic", "padding": "12px 0"},
        )
    return dash_table.DataTable(
        id="watchlist-dt",
        columns=[
            {"name": "Ticker", "id": "ticker"},
            {"name": "Notes",  "id": "notes"},
            {"name": "Added",  "id": "added"},
        ],
        data=rows,
        row_deletable=True,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": FONT_BODY, "fontSize": "13px", "padding": "10px 12px",
                    "textAlign": "left", "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": SURFACE, "fontWeight": "600",
                      "color": TEXT, "borderBottom": f"2px solid {BORDER}"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"}],
    )


def _alerts_table_component(rows, table_id, deletable=True, include_triggered_col=False):
    if not rows:
        return html.Div(
            "Nothing to show here yet.",
            style={"color": MUTED, "fontStyle": "italic", "padding": "12px 0"},
        )
    columns = [
        {"name": "Ticker",    "id": "ticker"},
        {"name": "Type",      "id": "alert_type"},
        {"name": "Threshold", "id": "threshold", "type": "numeric",
         "format": {"specifier": ".2f"}},
    ]
    if include_triggered_col:
        columns.append({"name": "Triggered at", "id": "triggered_at"})
    else:
        columns += [
            {"name": "Email",   "id": "email"},
            {"name": "Created", "id": "created"},
        ]
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=rows,
        row_deletable=deletable,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": FONT_BODY, "fontSize": "13px", "padding": "10px 12px",
                    "textAlign": "left", "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": SURFACE, "fontWeight": "600",
                      "color": TEXT, "borderBottom": f"2px solid {BORDER}"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"}],
    )


def layout_watchlist():
    return html.Div([
        page_header("Watchlist", "Track tickers you want to keep an eye on"),
        card([
            section_header("Add ticker"),
            html.Div([
                dcc.Input(
                    id="wl-ticker", type="text", placeholder="e.g. NVDA",
                    className="qe-input",
                    style={"flex": "0 0 160px", "textTransform": "uppercase"},
                ),
                dcc.Input(
                    id="wl-notes", type="text", placeholder="Notes (optional)",
                    className="qe-input", style={"flex": "1"},
                ),
                primary_btn("Add", "wl-add-btn"),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div(id="wl-status"),
        ]),
        card([
            section_header("Your watchlist", "Delete a row by clicking the × on the left"),
            html.Div(id="wl-table-wrap"),
        ]),
    ])


def layout_alerts():
    return html.Div([
        page_header("Alerts", "Get notified when a price crosses your threshold"),
        card([
            section_header("Create alert"),
            html.Div([
                dcc.Input(
                    id="al-ticker", type="text", placeholder="Ticker",
                    className="qe-input",
                    style={"flex": "0 0 130px", "textTransform": "uppercase"},
                ),
                dcc.Dropdown(
                    id="al-type",
                    options=[
                        {"label": "Price above", "value": "price_above"},
                        {"label": "Price below", "value": "price_below"},
                        {"label": "Stop loss",   "value": "stop_loss"},
                    ],
                    placeholder="Trigger type",
                    clearable=False,
                    className="qe-dropdown",
                    style={"flex": "0 0 170px"},
                ),
                dcc.Input(
                    id="al-threshold", type="number", placeholder="Threshold (price)",
                    className="qe-input", style={"flex": "0 0 150px"},
                ),
                dcc.Input(
                    id="al-email", type="email", placeholder="Email (optional)",
                    className="qe-input", style={"flex": "1"},
                ),
                primary_btn("Create", "al-create-btn"),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center",
                      "flexWrap": "wrap"}),
            html.Div(id="al-status"),
        ]),
        card([
            section_header(
                "Active alerts",
                "Auto-checked every 15 min by the scheduler. Delete with the ×.",
            ),
            html.Div(id="al-active-wrap"),
        ]),
        card([
            section_header("Recently triggered", "Last 30 days"),
            html.Div(id="al-triggered-wrap"),
        ]),
    ])


# ── Watchlist callbacks ──────────────────────────────────────────────

@app.callback(
    Output("wl-table-wrap", "children"),
    Output("wl-status",     "children"),
    Output("wl-ticker",     "value"),
    Output("wl-notes",      "value"),
    Input("wl-add-btn", "n_clicks"),
    State("wl-ticker",  "value"),
    State("wl-notes",   "value"),
    prevent_initial_call=False,
)
def watchlist_add(n_clicks, ticker, notes):
    status = None
    if n_clicks and ticker:
        ticker_norm = ticker.strip().upper()
        if ticker_norm:
            try:
                with DBSession() as s:
                    existing = s.query(WatchlistRow).filter_by(
                        user_name=DEFAULT_USER, ticker=ticker_norm,
                    ).first()
                    if existing:
                        status = html.Div(
                            f"{ticker_norm} is already on your watchlist.",
                            className="qe-toast error",
                        )
                    else:
                        s.add(WatchlistRow(
                            user_name=DEFAULT_USER, ticker=ticker_norm,
                            notes=(notes or "").strip() or None,
                        ))
                        s.commit()
                        status = html.Div(
                            f"Added {ticker_norm} to your watchlist.",
                            className="qe-toast success",
                        )
            except Exception as e:
                logger.exception("Watchlist add failed: %s", e)
                status = html.Div(f"Failed to add: {e}", className="qe-toast error")
    return _watchlist_table_component(_watchlist_rows()), status, "", ""


@app.callback(
    Output("wl-table-wrap", "children", allow_duplicate=True),
    Input("watchlist-dt", "data"),
    State("watchlist-dt", "data_previous"),
    prevent_initial_call=True,
)
def watchlist_handle_delete(current, previous):
    if previous is None:
        return dash.no_update
    current_tickers  = {r["ticker"] for r in (current or [])}
    previous_tickers = {r["ticker"] for r in (previous or [])}
    removed = previous_tickers - current_tickers
    if not removed:
        return dash.no_update
    try:
        with DBSession() as s:
            for t in removed:
                s.query(WatchlistRow).filter_by(
                    user_name=DEFAULT_USER, ticker=t,
                ).delete()
            s.commit()
    except Exception as e:
        logger.exception("Watchlist delete failed: %s", e)
    return _watchlist_table_component(_watchlist_rows())


# ── Alerts callbacks ──────────────────────────────────────────────

@app.callback(
    Output("al-active-wrap",    "children"),
    Output("al-triggered-wrap", "children"),
    Output("al-status",         "children"),
    Output("al-ticker",         "value"),
    Output("al-type",           "value"),
    Output("al-threshold",      "value"),
    Output("al-email",          "value"),
    Input("al-create-btn", "n_clicks"),
    State("al-ticker",    "value"),
    State("al-type",      "value"),
    State("al-threshold", "value"),
    State("al-email",     "value"),
    prevent_initial_call=False,
)
def alerts_create(n_clicks, ticker, alert_type, threshold, email):
    status = None
    if n_clicks:
        if not ticker or not alert_type or threshold is None:
            status = html.Div(
                "Ticker, type and threshold are required.",
                className="qe-toast error",
            )
        else:
            try:
                with DBSession() as s:
                    s.add(AlertRow(
                        user_name=DEFAULT_USER,
                        ticker=ticker.strip().upper(),
                        alert_type=alert_type,
                        threshold=float(threshold),
                        email=(email or "").strip() or None,
                        is_active=True,
                        triggered=False,
                    ))
                    s.commit()
                status = html.Div(
                    f"Created {alert_type} alert for "
                    f"{ticker.strip().upper()} at {threshold:.2f}.",
                    className="qe-toast success",
                )
            except Exception as e:
                logger.exception("Alert create failed: %s", e)
                status = html.Div(f"Failed to create: {e}", className="qe-toast error")
    active = _alerts_table_component(_active_alert_rows(), "al-active-dt", deletable=True)
    triggered = _alerts_table_component(
        _triggered_alert_rows(), "al-triggered-dt",
        deletable=False, include_triggered_col=True,
    )
    return active, triggered, status, "", None, None, ""


@app.callback(
    Output("al-active-wrap", "children", allow_duplicate=True),
    Input("al-active-dt", "data"),
    State("al-active-dt", "data_previous"),
    prevent_initial_call=True,
)
def alerts_handle_delete(current, previous):
    if previous is None:
        return dash.no_update
    current_ids  = {r["id"] for r in (current or [])}
    previous_ids = {r["id"] for r in (previous or [])}
    removed = previous_ids - current_ids
    if not removed:
        return dash.no_update
    try:
        with DBSession() as s:
            for alert_id in removed:
                s.query(AlertRow).filter_by(id=alert_id).delete()
            s.commit()
    except Exception as e:
        logger.exception("Alert delete failed: %s", e)
    return _alerts_table_component(_active_alert_rows(), "al-active-dt", deletable=True)


# ════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    GOOGLE_FONTS,
    dcc.Store(id="active-tab", data="portfolio"),
    sidebar,
    html.Div(id="page-content", className="main-content"),
], style={"fontFamily": FONT_UI, "background": BG})


# ════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ════════════════════════════════════════════════════════════════════

@app.callback(
    Output("active-tab",    "data"),
    Output("nav-portfolio", "className"),
    Output("nav-screener",  "className"),
    Output("nav-watchlist", "className"),
    Output("nav-alerts",    "className"),
    Input("nav-portfolio",  "n_clicks"),
    Input("nav-screener",   "n_clicks"),
    Input("nav-watchlist",  "n_clicks"),
    Input("nav-alerts",     "n_clicks"),
    prevent_initial_call=True,
)
def update_nav(_p, _s, _w, _a):
    ctx = callback_context
    if not ctx.triggered:
        return "portfolio", "nav-item active", "nav-item", "nav-item", "nav-item"
    tab = ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-", "")
    def _cls(t): return "nav-item active" if tab == t else "nav-item"
    return tab, _cls("portfolio"), _cls("screener"), _cls("watchlist"), _cls("alerts")


@app.callback(Output("page-content", "children"), Input("active-tab", "data"))
def render_page(tab):
    tab = tab or "portfolio"
    if tab == "portfolio": return layout_portfolio()
    if tab == "screener":  return layout_screener()
    if tab == "watchlist": return layout_watchlist()
    if tab == "alerts":    return layout_alerts()
    return layout_portfolio()


@app.callback(
    Output("portfolio-kpis", "children"),
    Output("pnl-chart", "figure"),
    Input("active-tab", "data"),
)
def update_portfolio_header(tab):
    if tab != "portfolio":
        return dash.no_update, dash.no_update
    data     = load_all_data("transactions.csv")
    prices_p = data["prices_port"] if data["prices_port"] is not None else pd.DataFrame()
    open_df, _ = compute_holdings_table(data["transactions"], prices_p,
                                       data["df_tech"], data["df_fund"], data["stocks_df"],
                                       fx_data=data.get("fx_data", {}))
    kpi = compute_portfolio_kpis(data["pnl"], open_df, data["transactions"], fx_data=data.get("fx_data", {}))

    cards = [
        metric_card("Total P/L",    f"€{kpi['total_pnl']:+,.0f}",   positive=kpi["total_pnl"] >= 0),
        metric_card("Day P/L",      f"€{kpi['day_pnl']:+,.0f}",     positive=kpi["day_pnl"] >= 0),
        metric_card("Wallet",       f"€{kpi['wallet']:,.0f}"),
        metric_card("Total Return", f"{kpi['total_return_pct']:+.1f}%", positive=kpi["total_return_pct"] >= 0),
        metric_card("CAGR",         f"{kpi['cagr_pct']:+.1f}%",     positive=kpi["cagr_pct"] >= 0),
        metric_card("Sharpe",       f"{kpi['sharpe']:.2f}",         sub="Risk-adjusted", positive=kpi["sharpe"] >= 1),
        metric_card("Sortino",      f"{kpi['sortino']:.2f}",        sub="Downside risk", positive=kpi["sortino"] >= 1),
    ]

    fig = build_pnl_chart(data["pnl"]) \
          if data["pnl"] is not None and not data["pnl"].empty \
          else go.Figure().update_layout(**PLOTLY_THEME)
    fig.update_layout(**PLOTLY_THEME)
    return cards, fig


def _make_holdings_datatable(df, table_id, pct_col="_unreal_pct"):
    """Constrói DataTable para Open ou Closed positions."""
    def news_url(q):
        return "https://news.google.com/search?q=" + q.replace(" ", "+")

    df = df.copy()
    df["Ticker Link"]  = df.apply(lambda r: f"[{r['Ticker']}]({news_url(r['Ticker'])})", axis=1)
    df["Company Link"] = df.apply(lambda r: f"[{r['Company']}]({news_url(r['Company'])})", axis=1)

    # Columns to display depending on open vs closed table
    if pct_col == "_unreal_pct":
        show = ["Ticker Link", "Exchange", "Company Link", "Qty", "Entry €", "Price €",
                "Weight %", "Unreal. P/L", "YTD %", "Vol 60d %",
                "Drawdown %", "Trend", "Tech Score", "Fund Score", "Mkt Cap", "Sector"]
    else:
        show = ["Ticker Link", "Exchange", "Company Link", "First Buy", "Last Sell",
                "Realised P/L", "Tech Score", "Fund Score", "Mkt Cap", "Sector"]

    show = [c for c in show if c in df.columns]
    disp = df[show].copy()
    num_cols = disp.select_dtypes(include=[np.number]).columns.tolist()

    # Colour coding
    row_styles = []
    if pct_col in df.columns:
        for i, pct in enumerate(df[pct_col].fillna(0)):
            if   pct >= 20:  bg, fg = "#D5F5E3", "#1B4332"
            elif pct >= 10:  bg, fg = "#EAFAF1", "#2D6A4F"
            elif pct >= 0:   bg, fg = "#F2FBF6", "#52B788"
            elif pct >= -10: bg, fg = "#FEF9E7", "#B7770D"
            else:            bg, fg = "#FDEDEC", "#C0392B"
            col_id = "Unreal. P/L" if pct_col == "_unreal_pct" else "Realised P/L"
            if col_id in show:
                row_styles.append({"if": {"row_index": i, "column_id": col_id},
                                   "backgroundColor": bg, "color": fg, "fontWeight": "500"})

    col_defs = []
    for c in show:
        d = {"name": c.replace(" Link", ""), "id": c}
        if c in ("Ticker Link", "Company Link"):
            d["presentation"] = "markdown"
        elif c in num_cols:
            d["type"] = "numeric"
        col_defs.append(d)

    left_cols = {"Ticker Link", "Exchange", "Company Link", "Trend",
                 "Sector", "Mkt Cap", "Unreal. P/L", "Realised P/L",
                 "First Buy", "Last Sell"}

    return dash_table.DataTable(
        id=table_id,
        data=disp.to_dict("records"),
        columns=col_defs,
        sort_action="native", page_size=30,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": SURFACE, "color": ACCENT,
                      "fontFamily": FONT_UI, "fontSize": "10px", "fontWeight": "700",
                      "letterSpacing": "0.8px", "textTransform": "uppercase",
                      "borderBottom": f"1px solid {BORDER}", "padding": "10px 12px"},
        style_cell={"backgroundColor": CARD, "color": TEXT,
                    "fontFamily": FONT_BODY, "fontSize": "12px",
                    "padding": "9px 12px", "border": f"1px solid {BORDER}",
                    "textAlign": "right", "whiteSpace": "nowrap"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"} for c in left_cols if c in show
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"},
        ] + row_styles,
        markdown_options={"html": False, "link_target": "_blank"},
    )


@app.callback(
    Output("holdings-table",         "children"),
    Output("closed-positions-table", "children"),
    Output("allocation-charts",      "figure"),
    Output("spy-ticker-selector",    "options"),
    Output("spy-ticker-selector",    "value"),
    Output("stop-ticker-selector",   "options"),
    Output("stop-ticker-selector",   "value"),
    Input("active-tab", "data"),
)
def update_holdings(tab):
    if tab != "portfolio":
        return dash.no_update, dash.no_update, dash.no_update, [], None, [], None
    data     = load_all_data("transactions.csv")
    prices_p = data["prices_port"] if data["prices_port"] is not None else pd.DataFrame()
    open_df, closed_df = compute_holdings_table(
        data["transactions"], prices_p,
        data["df_tech"], data["df_fund"], data["stocks_df"],
        fx_data=data.get("fx_data", {}))

    empty_pie = go.Figure().update_layout(**PLOTLY_THEME)
    no_data   = html.P("No data available.", style={"color": MUTED, "fontFamily": FONT_UI})

    # ── Open Positions ───────────────────────────────────────────────
    if open_df.empty:
        open_tbl = no_data
        pie_fig  = empty_pie
        opts, first = [], None
    else:
        open_tbl = _make_holdings_datatable(open_df, "holdings-dt", "_unreal_pct")
        pie_fig  = build_allocation_charts(open_df)
        tickers  = open_df["Ticker"].tolist()
        opts     = [{"label": t, "value": t} for t in tickers]
        first    = tickers[0]

    # ── Closed Positions ─────────────────────────────────────────────
    if closed_df.empty:
        closed_tbl = html.P("No closed positions.", style={"color": MUTED, "fontFamily": FONT_UI})
    else:
        closed_tbl = _make_holdings_datatable(closed_df, "closed-dt", "_realised")

    return open_tbl, closed_tbl, pie_fig, opts, first, opts, first


@app.callback(Output("stock-spy-chart", "figure"), Input("spy-ticker-selector", "value"))
def update_spy_chart(ticker):
    if not ticker:
        return go.Figure().update_layout(**PLOTLY_THEME)
    data  = load_all_data("transactions.csv")
    tx_t  = data["transactions"]
    tx_t  = tx_t[(tx_t["ticker"] == ticker) & (tx_t["type"] == "buy")]
    first = pd.to_datetime(tx_t["date"].min()) if not tx_t.empty else None
    return build_stock_vs_spy_chart(data["prices"], ticker, first)


@app.callback(Output("stop-table", "children"), Input("active-tab", "data"))
def update_stop_table(tab):
    if tab != "portfolio":
        return dash.no_update
    data  = load_all_data("transactions.csv")
    stops = data["stops"]
    if stops is None or stops.empty:
        return html.P("No stop data.", style={"color": MUTED, "fontFamily": FONT_UI})
    disp = stops.copy()
    for col in ["stop_pct_away", "win_rate", "kelly_fraction", "half_kelly"]:
        if col in disp.columns:
            disp[col] = (disp[col] * 100).round(1).astype(str) + "%"
    return styled_table(disp, "stops-dt",
                        color_cols={"expectancy": (0, 0.05), "profit_factor": (0, 1)})


@app.callback(Output("stop-candle-chart", "figure"), Input("stop-ticker-selector", "value"))
def update_candle(ticker):
    if not ticker:
        return go.Figure().update_layout(**PLOTLY_THEME)
    data   = load_all_data("transactions.csv")
    prices = data["prices_port"]
    stops  = data["stops"]
    df_t   = prices[prices["ticker"] == ticker] if prices is not None else pd.DataFrame()
    stop_px = np.nan
    if stops is not None and not stops.empty and ticker in stops["ticker"].values:
        stop_px = stops[stops["ticker"] == ticker]["stop_price"].iloc[0]
    return build_stop_chart(df_t, ticker, stop_px)


# Populate sector dropdown options from stocks_df
@app.callback(
    Output("sector-filter", "options"),
    Input("active-tab", "data"),
)
def populate_sectors(tab):
    if tab != "screener":
        return []
    data    = load_all_data("transactions.csv")
    sdf     = data["stocks_df"]
    if "Sector" not in sdf.columns:
        return []
    sectors = sorted(sdf["Sector"].dropna().unique().tolist())
    return [{"label": s, "value": s} for s in sectors]


# Clear filters callback
@app.callback(
    Output("sector-filter",   "value"),
    Output("min-tech-score",  "value"),
    Input("clear-screen-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_):
    return [], 0


@app.callback(
    Output("regime-banner",   "children"),
    Output("logo-grid",       "children"),
    Output("sector-chart",    "figure"),
    Output("factor-table",    "children"),
    Input("apply-screen-btn", "n_clicks"),
    Input("screen-metric",    "value"),
    State("min-tech-score",   "value"),
    State("sector-filter",    "value"),
    State("screen-topn",      "value"),
    prevent_initial_call=False,
)
def update_screener(n, metric, min_tech, sectors, topn):
    data    = load_all_data("transactions.csv")
    ft      = data["factor_table"].copy()
    sdf     = data["stocks_df"].copy()
    topn    = int(topn or 50)

    # ── Enrich factor_table with stocks_df metadata ───────────────────
    merge_cols = [c for c in ["Symbol", "Company Name", "Sector",
                               "Market Cap", "Exchange"] if c in sdf.columns]
    if merge_cols:
        sdf_m = sdf[merge_cols].rename(columns={"Symbol": "ticker",
                                                  "Company Name": "Company",
                                                  "Market Cap": "Mkt Cap Raw"})
        ft = ft.merge(sdf_m, on="ticker", how="left")

    # ── Merge YTD + technical components from df_tech ───────────────
    df_tech = data["df_tech"]
    if df_tech is not None and not df_tech.empty:
        tech_cols = ["ticker","ytd","momentum_12m","return_20d","trend_strength",
                     "macd","vol_60d","max_drawdown_12m"]
        avail = [c for c in tech_cols if c in df_tech.columns]
        last = df_tech.sort_values("date").groupby("ticker").tail(1)[avail].copy()
        if "ytd" in last.columns:
            last["YTD %"] = (last["ytd"] * 100).round(1)
        ft = ft.merge(last, on="ticker", how="left")

    # ── Format Market Cap ─────────────────────────────────────────────
    if "Mkt Cap Raw" in ft.columns:
        def fmt_cap(x):
            try:
                x = float(x)
                if x >= 1e12: return f"${x/1e12:.1f}T"
                if x >= 1e9:  return f"${x/1e9:.0f}B"
                if x >= 1e6:  return f"${x/1e6:.0f}M"
                return f"${x:,.0f}"
            except:
                return "N/A"
        ft["Mkt Cap"] = ft["Mkt Cap Raw"].apply(fmt_cap)

    # ── Filters ───────────────────────────────────────────────────────
    if "technical_score" in ft.columns:
        ft = ft[ft["technical_score"] >= (min_tech or 0)]
    if sectors and "Sector" in ft.columns:
        ft = ft[ft["Sector"].isin(sectors)]

    ft = ft.sort_values("technical_score", ascending=False).reset_index(drop=True)

    # ── 1. Logo grid — ranked by selected metric ─────────────────────
    metric   = metric or "technical_score"
    sort_col = metric if metric in ft.columns else "technical_score"
    ranked   = ft.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(topn)

    fmp_logo = "https://financialmodelingprep.com/image-stock/{ticker}.png"

    logo_items = []
    for _, r in ranked.iterrows():
        tkr    = r["ticker"]
        co     = r.get("Company", tkr) if "Company" in ranked.columns else tkr
        sec    = r.get("Sector", "N/A")   if "Sector"  in ranked.columns else "N/A"
        score  = round(r[sort_col], 2) if pd.notna(r.get(sort_col)) else "N/A"
        ytd    = f'{r["YTD %"]:+.1f}%' if "YTD %" in ranked.columns and pd.notna(r.get("YTD %")) else "N/A"
        logo   = fmp_logo.format(ticker=tkr)

        safe_id = tkr.replace(".", "-").replace("/", "-")
        logo_items.append(html.Div([
            html.Div(tkr[:4], id=f"logo-fb-{safe_id}", style={
                "display": "flex", "width": "44px", "height": "44px",
                "borderRadius": "8px", "border": f"1px solid {BORDER}",
                "background": SURFACE, "alignItems": "center",
                "justifyContent": "center", "fontSize": "8px",
                "fontFamily": FONT_UI, "fontWeight": "700", "color": ACCENT,
                "position": "absolute", "top": "0", "left": "0", "zIndex": "0",
            }),
            html.Img(
                src=logo, id=f"logo-img-{safe_id}",
                title=f"{co} | {sec} | Score: {score} | YTD: {ytd}",
                style={
                    "width": "44px", "height": "44px",
                    "objectFit": "contain", "borderRadius": "8px",
                    "border": f"1px solid {BORDER}", "background": CARD,
                    "padding": "3px", "position": "relative", "zIndex": "1",
                },
            ),
            html.Div(tkr, style={
                "fontSize": "10px", "color": MUTED,
                "fontFamily": FONT_BODY, "marginTop": "3px",
                "textAlign": "center", "width": "44px",
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }),
        ], style={
            "display": "flex", "flexDirection": "column",
            "alignItems": "center", "position": "relative",
        }))

    metric_label = {
        "technical_score":   "Technical Score",
        "fundamental_score": "Fundamental Score",
        "YTD %":             "YTD %",
    }.get(metric, metric)

    logo_grid = html.Div([
        html.Div(
            f"Top {len(ranked)} by {metric_label} — hover for details",
            style={"fontFamily": FONT_UI, "fontSize": "11px",
                   "color": MUTED, "marginBottom": "12px"},
        ),
        html.Div(logo_items, style={
            "display": "flex", "flexWrap": "wrap", "gap": "10px",
        }),
    ])

    # ── 3. Sector strength chart ──────────────────────────────────────
    if "Sector" in ft.columns and ft["Sector"].notna().any():
        sec = (ft.groupby("Sector")["technical_score"]
               .mean().sort_values(ascending=True).reset_index())
        sec_fig = go.Figure(go.Bar(
            x=sec["technical_score"].round(3),
            y=sec["Sector"],
            orientation="h",
            marker_color=ACCENT,
        ))
        sec_fig.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Avg Technical Score by Sector",
                       font=dict(family=FONT_DISPLAY, size=15, color=TEXT)),
            xaxis_title="Avg Technical Score",
            height=340,
        )
    else:
        sec_fig = go.Figure().update_layout(**PLOTLY_THEME)

    # ── 4. Factor table ───────────────────────────────────────────────
    # Merge fundamental data (FMP columns) into factor table for display
    df_fund = data.get("df_fund")
    if df_fund is not None and not df_fund.empty:
        fund_cols = [c for c in [
            "ticker", "fundamental_score",
            "DCF Upside", "ROIC", "FCF Margin", "FCF Yield",
            "Op Margin", "Net Margin", "Gross Margin",
            "Rev CAGR 3yr", "CapEx % Rev", "Share Dilution",
            "EV/EBITDA", "P/E", "Piotroski",
        ] if c in df_fund.columns]
        ft = ft.merge(df_fund[fund_cols], on="ticker", how="left")

    # ── ML Scores merge ──────────────────────────────────────────────
    ml_scores = data.get("ml_scores")
    if ml_scores is not None and not ml_scores.empty:
        ml_merge = ml_scores[["ticker", "ml_score", "ml_q10", "ml_q90",
                               "ml_uncert", "ml_decile"]].copy()
        ft = ft.merge(ml_merge, on="ticker", how="left")
        # Formatar colunas de display
        ft["ML Score"] = ft["ml_score"].round(3)
        ft["ML Band"]  = ft.apply(
            lambda r: f"[{r['ml_q10']:.2f} — {r['ml_q90']:.2f}]"
            if pd.notna(r.get("ml_q10")) else "N/A", axis=1)
        ft["ML Decile"] = ft["ml_decile"].apply(
            lambda x: f"{int(x)}/10" if pd.notna(x) else "N/A")

    # ── Google News hyperlinks ────────────────────────────────────────
    def news_url(q):
        return "https://news.google.com/search?q=" + str(q).replace(" ", "+")

    ft["Ticker Link"]  = ft["ticker"].apply(
        lambda t: f"[{t}]({news_url(t)})")
    ft["Company Link"] = ft.apply(
        lambda r: f"[{r.get('Company', r['ticker'])}]({news_url(r.get('Company', r['ticker']))})"
        if "Company" in ft.columns else f"[{r['ticker']}]({news_url(r['ticker'])})", axis=1)

    # Component columns for JS tooltip — included in data but hidden from view
    _hidden_tech = [c for c in ["momentum_12m","return_20d","trend_strength",
                                 "macd","vol_60d","max_drawdown_12m"] if c in ft.columns]
    _hidden_fund = [c for c in ["roic","fcf_margin","rev_cagr","gross_margin",
                                 "interest_coverage"] if c in ft.columns]
    _all_hidden  = _hidden_tech + _hidden_fund

    show_cols = [c for c in [
        "Ticker Link", "Exchange", "Company Link", "Sector", "Mkt Cap", "YTD %",
        "technical_score", "ML Score", "ML Band", "ML Decile",
        "fundamental_score",
        "DCF Upside", "ROIC", "FCF Margin", "FCF Yield",
        "Op Margin", "Net Margin", "Gross Margin",
        "Rev CAGR 3yr", "CapEx % Rev", "Share Dilution",
        "EV/EBITDA", "P/E", "Piotroski",
    ] + _all_hidden if c in ft.columns]

    disp = ft[show_cols].copy()
    if "technical_score" in disp.columns:
        disp["technical_score"] = disp["technical_score"].round(2)
    if "fundamental_score" in disp.columns:
        disp["fundamental_score"] = disp["fundamental_score"].round(2)

    num_cols = ft[[c for c in show_cols if c in ft.columns]].select_dtypes(
        include=[np.number]).columns.tolist()

    col_defs = []
    for c in show_cols:
        d = {"name": c.replace(" Link", ""), "id": c}
        if c in ("Ticker Link", "Company Link"):
            d["presentation"] = "markdown"
        elif c in num_cols:
            d["type"] = "numeric"
        col_defs.append(d)

    left_cols = {"Ticker Link", "Company Link", "Exchange",
                 "Sector", "Mkt Cap", "ML Band", "ML Decile",
                 "Insider (90d)", "Share Dilution",
                 "DCF Upside", "ROIC", "FCF Margin", "FCF Yield",
                 "Op Margin", "Net Margin", "Gross Margin",
                 "Rev CAGR 3yr", "EV/EBITDA", "P/E", "Piotroski"}

    tbl = dash_table.DataTable(
        id="factor-dt",
        data=disp.to_dict("records"),
        columns=col_defs,
        sort_action="native", filter_action="native", page_size=25,
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": SURFACE, "color": ACCENT,
                      "fontFamily": FONT_UI, "fontSize": "10px", "fontWeight": "700",
                      "letterSpacing": "0.8px", "textTransform": "uppercase",
                      "borderBottom": f"1px solid {BORDER}", "padding": "10px 12px"},
        style_cell={"backgroundColor": CARD, "color": TEXT,
                    "fontFamily": FONT_BODY, "fontSize": "12px",
                    "padding": "9px 12px", "border": f"1px solid {BORDER}",
                    "textAlign": "right", "whiteSpace": "nowrap"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"}
            for c in left_cols if c in show_cols
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFAF8"},
            {"if": {"filter_query": "{technical_score} >= 0.7",
                    "column_id": "technical_score"},
             "color": GREEN, "fontWeight": "600"},
            {"if": {"filter_query": "{technical_score} <= 0.3",
                    "column_id": "technical_score"},
             "color": RED},
            {"if": {"filter_query": "{fundamental_score} >= 0.7",
                    "column_id": "fundamental_score"},
             "color": GREEN, "fontWeight": "600"},
            {"if": {"filter_query": "{fundamental_score} <= 0.3",
                    "column_id": "fundamental_score"},
             "color": RED},
            {"if": {"filter_query": "{ML Score} >= 0.7",
                    "column_id": "ML Score"},
             "color": GREEN, "fontWeight": "600"},
            {"if": {"filter_query": "{ML Score} <= 0.3",
                    "column_id": "ML Score"},
             "color": RED},
        ],
        markdown_options={"html": False, "link_target": "_blank"},
        hidden_columns=_all_hidden,
        tooltip_header={
            "Exchange":          "Stock exchange (NYSE, NASDAQ…).",
            "Sector":            "Business sector from stocks_all_pages.csv.",
            "Mkt Cap":           "Market cap = shares x price. T=trillion, B=billion.",
            "YTD %":             "Year-to-date return: (price_today / price_jan1) - 1.",
            "technical_score":   "Composite 0-1. Signals: momentum 12M (+30%), return 20d (+20%), trend vs EMA200 (+15%), MACD (+10%), minus volatility 60d (-20%) and drawdown 12M (-15%). Z-scored per date.",
            "ML Score":          "LightGBM ranking model (0-1). Trained on 10 technical features. Predicts 3-month relative rank — re-evaluate monthly, scores drift as new data arrives.",
            "ML Band":           "80% confidence band [q10 — q90]. Wide band = high uncertainty on this stock; treat the score as noisy.",
            "ML Decile":         "Decile ranking 1-10. Decile 10 = top 10% predicted performers. Decile 1 = bottom 10%.",
            "fundamental_score": "Quality 0-1. ROIC 25% + FCF margin 20% + Rev CAGR 20% + Gross margin 20% + Interest coverage 15%. NaN-safe weighted avg.",
            "DCF Upside":        "(DCF_value - price) / price. Positive = undervalued. FMP /discounted-cash-flow.",
            "ROIC":              "Return on Invested Capital: EBIT*(1-tax) / (equity+net_debt). >15% = competitive moat. Better than ROE.",
            "FCF Margin":        "FCF / revenue. Cash after capex. Software: 25-40%. Industrials: 3-8%.",
            "FCF Yield":         "FCF / market_cap. Cash return on purchase price. >5% = attractive.",
            "Op Margin":         "EBIT / revenue. Core operating efficiency before interest and taxes.",
            "Net Margin":        "Net income / revenue. Final profitability after all costs.",
            "Gross Margin":      "(Revenue - COGS) / revenue. Pricing power and moat proxy.",
            "Rev CAGR 3yr":      "Revenue CAGR 3 years from audited reports. Historical fact, not analyst estimate.",
            "CapEx % Rev":       "|capex| / revenue. Asset-light (software): <5%. Capital-intensive: 15-30%.",
            "Share Dilution":    "Annual change in shares outstanding. Negative = buybacks. Positive = dilution.",
            "EV/EBITDA":         "Enterprise Value / EBITDA. Independent of capital structure. Best cross-company multiple.",
            "P/E":               "Price / earnings TTM. Limited - distorted by non-recurring items.",
            "Piotroski":         "F-Score 0-9. Points for profitability, leverage, efficiency improvements. >6 = solid.",
        },
        tooltip_delay=0,
        tooltip_duration=None,
    )

    # ── 5. Regime banner ─────────────────────────────────────────────
    # Render the current market regime detected by modules.regime_detector.
    # Defaults are unified so a partial regime dict still produces a
    # coherent banner instead of mixing "0%" with "?" placeholders.
    regime = data.get("regime")
    if regime:
        regime_color = regime.get("color", ACCENT)
        stay_prob = regime.get("stay_probability")
        stay_prob_str = f"{stay_prob:.0%}" if isinstance(stay_prob, (int, float)) else "N/A"
        expected_dur = regime.get("expected_duration_days")
        expected_dur_str = (
            f"~{expected_dur} trading days"
            if isinstance(expected_dur, (int, float))
            else "N/A"
        )
        banner = html.Div([
            html.Div([
                html.Span(regime.get("stars", ""), style={
                    "fontSize": "18px", "marginRight": "12px",
                }),
                html.Span(f"Regime: {regime.get('regime', 'N/A')}", style={
                    "fontFamily": FONT_DISPLAY, "fontSize": "16px",
                    "fontWeight": "700", "color": regime_color,
                }),
                html.Span(f"  —  {regime.get('recommendation', '')}", style={
                    "fontFamily": FONT_UI, "fontSize": "12px",
                    "color": MUTED, "marginLeft": "12px",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span(f"P(stay in regime): {stay_prob_str}", style={
                    "fontFamily": FONT_BODY, "fontSize": "11px", "color": MUTED,
                }),
                html.Span(f"  ·  Expected duration: {expected_dur_str}", style={
                    "fontFamily": FONT_BODY, "fontSize": "11px", "color": MUTED,
                }),
                html.Span(f"  ·  Updated: {regime.get('date', 'N/A')}", style={
                    "fontFamily": FONT_BODY, "fontSize": "11px", "color": MUTED,
                }),
            ], style={"marginTop": "6px"}),
        ], style={
            "background": CARD, "border": f"1px solid {BORDER}",
            "borderLeft": f"4px solid {regime_color}",
            "borderRadius": "8px", "padding": "14px 20px",
        })
    else:
        # Regime detection returned None — most likely missing SPY history
        # or insufficient observations. See logs from modules.regime_detector
        # for the exact reason.
        banner = html.Div([
            html.Span("Regime: ", style={"fontFamily": FONT_UI, "color": MUTED}),
            html.Span(
                "Not available — see logs for diagnostic details.",
                style={"fontFamily": FONT_UI, "color": MUTED, "fontSize": "12px"},
            ),
        ], style={
            "background": SURFACE, "borderRadius": "8px",
            "padding": "12px 18px", "border": f"1px solid {BORDER}",
        })

    return banner, logo_grid, sec_fig, tbl


@app.callback(
    Output("w-mom-out",   "children"), Output("w-ret-out",   "children"),
    Output("w-trend-out", "children"), Output("w-macd-out",  "children"),
    Output("w-vol-out",   "children"), Output("w-dd-out",    "children"),
    Input("w-mom",   "value"), Input("w-ret",   "value"),
    Input("w-trend", "value"), Input("w-macd",  "value"),
    Input("w-vol",   "value"), Input("w-dd",    "value"),
)
def show_weights(mom, ret, trend, macd, vol, dd):
    return [f" {v:+.2f}" for v in (mom, ret, trend, macd, vol, dd)]


@app.callback(
    Output("bt-kpis",         "children"),
    Output("bt-equity-chart", "figure"),
    Output("bt-dd-chart",     "figure"),
    Output("trade-log-table", "children"),
    Input("run-bt-btn",       "n_clicks"),
    State("bt-start",  "value"), State("bt-budget", "value"),
    State("bt-topn",   "value"), State("bt-rebal",  "value"),
    State("w-mom",     "value"), State("w-ret",     "value"),
    State("w-trend",   "value"), State("w-macd",    "value"),
    State("w-vol",     "value"), State("w-dd",      "value"),
    prevent_initial_call=True,
)
def run_bt(n, start, budget, topn, rebal, w_mom, w_ret, w_trend, w_macd, w_vol, w_dd):
    data    = load_all_data("transactions.csv")
    weights = {"mom_12m_z": w_mom or 0.30, "ret_6m_z": w_ret or 0.20,
               "trend_z": w_trend or 0.15, "macd_z": w_macd or 0.10,
               "vol_60d_z": w_vol or -0.20, "dd_12m_z": w_dd or -0.15}
    result = run_backtest(df_tech=data["df_tech"], prices=data["prices"],
                          start_date=start or "2015-01-01",
                          monthly_budget=float(budget or 1000),
                          top_n=int(topn or 5), rebalance_months=int(rebal or 1),
                          weights=weights)
    empty_fig = go.Figure().update_layout(**PLOTLY_THEME)
    if "error" in result:
        return [], empty_fig, empty_fig, html.P(result["error"], style={"color": RED})

    s     = result["summary"]
    eq    = result["equity_curve"]
    bench = benchmark_buyhold(data["prices"], start or "2015-01-01", float(budget or 1000))
    kpis  = [
        metric_card("Invested",     f"€{s.get('total_invested',0):,.0f}"),
        metric_card("Final Value",  f"€{s.get('final_value',0):,.0f}",
                    positive=s.get("total_return_pct", 0) >= 0),
        metric_card("Total Return", f"{s.get('total_return_pct',0):.1f}%",
                    positive=s.get("total_return_pct", 0) >= 0),
        metric_card("CAGR",         f"{s.get('cagr_pct',0):.1f}%",
                    positive=s.get("cagr_pct", 0) >= 0),
        metric_card("Sharpe",       f"{s.get('sharpe_ratio',0):.2f}",
                    positive=s.get("sharpe_ratio", 0) >= 1),
        metric_card("Max Drawdown", f"{s.get('max_drawdown_pct',0):.1f}%", positive=False),
    ]
    eq_fig = build_equity_chart(eq, bench)
    eq_fig.update_layout(**PLOTLY_THEME)
    dd_fig = build_drawdown_chart(eq)
    dd_fig.update_layout(**PLOTLY_THEME)
    trades    = result["trade_log"]
    trade_tbl = styled_table(trades.tail(100), "trade-dt") \
                if trades is not None and not trades.empty \
                else html.P("No trades generated.", style={"color": MUTED})
    return kpis, eq_fig, dd_fig, trade_tbl


@app.callback(
    Output("opt-section",       "style"),
    Output("opt-compare-chart", "figure"),
    Output("opt-table",         "children"),
    Input("run-opt-btn",        "n_clicks"),
    State("bt-start",  "value"), State("bt-budget", "value"),
    State("bt-topn",   "value"),
    prevent_initial_call=True,
)
def run_optimiser(n, start, budget, topn):
    data    = load_all_data("transactions.csv")
    results = optimise_weights(df_tech=data["df_tech"], prices=data["prices"],
                               start_date=start or "2015-01-01",
                               monthly_budget=float(budget or 1000),
                               top_n=int(topn or 5), metric="sharpe_ratio")
    visible = {"background": CARD, "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "24px",
               "marginBottom": "20px", "display": "block"}
    if results.empty:
        return visible, go.Figure(), html.P("No results.", style={"color": MUTED})
    fig  = build_backtest_compare(results)
    fig.update_layout(**PLOTLY_THEME)
    dcols = [c for c in ["cagr_pct", "sharpe_ratio", "max_drawdown_pct",
                          "total_return_pct", "n_trades"] if c in results.columns]
    tbl   = styled_table(results[dcols].head(20), "opt-dt",
                         color_cols={"sharpe_ratio": (0, 1.5), "cagr_pct": (0, 10)})
    return visible, fig, tbl





# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
