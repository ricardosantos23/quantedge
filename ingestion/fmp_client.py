"""
ingestion/fmp_client.py — All FMP API calls.
Updated for new FMP /stable/ API (August 2025+).
Old /api/v3/ endpoints no longer work.
"""
import json
import time
import threading
import urllib.request as _req
import urllib.error as _err
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config

_lock = threading.Lock()


# ════════════════════════════════════════════════════════
#  CORE HTTP
# ════════════════════════════════════════════════════════

def get(endpoint: str, params: dict = None, retries: int = None) -> Any:
    if not config.FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set.")

    n_ret = retries if retries is not None else config.FMP_RETRIES
    all_params = {"apikey": config.FMP_API_KEY}
    if params:
        all_params.update(params)
    qs  = "&".join(f"{k}={v}" for k, v in all_params.items())
    url = f"{config.FMP_BASE}/{endpoint}?{qs}"

    for attempt in range(n_ret + 1):
        try:
            req = _req.Request(url, headers={"User-Agent": "QuantEdge/2.0"})
            with _req.urlopen(req, timeout=config.FMP_TIMEOUT) as r:
                data = json.loads(r.read())
                if isinstance(data, dict) and "Error Message" in data:
                    return None
                return data
        except _err.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(f"FMP API auth error ({e.code}). Check FMP_API_KEY.")
            if e.code == 429:
                time.sleep(2 ** (attempt + 1))
            elif attempt < n_ret:
                time.sleep(1)
        except Exception:
            if attempt < n_ret:
                time.sleep(1)
    return None


def _first(resp: Any) -> dict:
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        return resp[0]
    if isinstance(resp, dict):
        return resp
    return {}


def _safe(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _pick(src: dict, *keys) -> float:
    for k in keys:
        v = src.get(k)
        if v is not None:
            f = _safe(v)
            if not np.isnan(f):
                return f
    return np.nan


# ════════════════════════════════════════════════════════
#  PRICE HISTORY
#  New endpoint: /stable/historical-price-eod/full
# ════════════════════════════════════════════════════════

def fetch_prices(ticker: str, start: str, end: str) -> list:
    data = get("historical-price-eod/full", {"symbol": ticker, "from": start, "to": end})
    if not data or not isinstance(data, list):
        return []
    rows = []
    for h in data:
        try:
            rows.append({
                "ticker":    ticker,
                "date":      h["date"],
                "open":      h.get("open"),
                "high":      h.get("high"),
                "low":       h.get("low"),
                "close":     h.get("close"),
                "adj_close": h.get("adjClose") or h.get("close"),
                "volume":    int(h.get("volume") or 0),
            })
        except (KeyError, TypeError):
            continue
    return rows


def fetch_prices_batch(tickers: list, start: str, end: str) -> list:
    """Fetch prices for multiple tickers in parallel (one request per ticker)."""
    all_rows = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_prices, t, start, end): t for t in tickers}
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
    return all_rows


# ════════════════════════════════════════════════════════
#  FX RATES
#  Same price endpoint, just pass currency pair as symbol
# ════════════════════════════════════════════════════════

def fetch_fx_history(pair: str, start: str, end: str) -> list:
    data = get("historical-price-eod/full", {"symbol": pair, "from": start, "to": end})
    if not data or not isinstance(data, list):
        return []
    rows = []
    for h in data:
        try:
            rows.append({
                "pair": pair,
                "date": h["date"],
                "rate": float(h.get("close") or h.get("adjClose") or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


# ════════════════════════════════════════════════════════
#  STOCK UNIVERSE
#  New endpoint: /stable/stock-list
# ════════════════════════════════════════════════════════

def fetch_stock_list() -> list:
    data = get("stock-list") or []
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t = str(item.get("symbol") or "").upper().strip()
        if not t:
            continue
        rows.append({
            "ticker":              t,
            "company_name":        str(item.get("name") or ""),
            "exchange":            str(item.get("exchange") or item.get("exchangeShortName") or ""),
            "is_etf":              str(item.get("type", "")).lower() == "etf",
            "is_actively_trading": True,
        })
    return rows


def fetch_company_profiles(tickers: list, batch_size: int = 1) -> list:
    """Enrich tickers with sector/market cap via /stable/profile."""
    results = []
    for ticker in tickers:
        data = get("profile", {"symbol": ticker})
        if not data:
            continue
        item = _first(data)
        if not isinstance(item, dict):
            continue
        t  = str(item.get("symbol") or ticker).upper().strip()
        mc = item.get("marketCap") or item.get("mktCap")
        results.append({
            "ticker":       t,
            "company_name": str(item.get("companyName") or ""),
            "sector":       str(item.get("sector") or ""),
            "industry":     str(item.get("industry") or ""),
            "market_cap":   int(mc) if mc else None,
            "exchange":     str(item.get("exchange") or item.get("exchangeShortName") or ""),
            "country":      str(item.get("country") or ""),
        })
        time.sleep(0.05)
    return results


# ════════════════════════════════════════════════════════
#  EARNINGS CALENDAR
#  New endpoint: /stable/earnings-calendar
# ════════════════════════════════════════════════════════

def fetch_earnings_calendar(start: str, end: str) -> list:
    data = get("earnings-calendar", {"from": start, "to": end}) or []
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t = str(item.get("symbol") or "").upper().strip()
        if not t:
            continue
        def _f(k):
            v = item.get(k)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        rows.append({
            "ticker":           t,
            "earnings_date":    item.get("date"),
            "eps_estimate":     _f("epsEstimated"),
            "eps_actual":       _f("eps"),
            "revenue_estimate": _f("revenueEstimated"),
            "revenue_actual":   _f("revenue"),
        })
    return rows


# ════════════════════════════════════════════════════════
#  FUNDAMENTALS
#  All endpoints now use ?symbol= as query param
# ════════════════════════════════════════════════════════

def fetch_fundamentals_one(ticker: str) -> dict:
    base = {"ticker": ticker}

    # Key metrics TTM
    km = _first(get("key-metrics-ttm", {"symbol": ticker}) or [])
    base["roic"]           = _pick(km, "roicTTM", "returnOnInvestedCapitalTTM")
    base["fcf_yield"]      = _pick(km, "freeCashFlowYieldTTM", "fcfYieldTTM")
    base["ev_ebitda"]      = _pick(km, "evToEbitdaTTM", "enterpriseValueOverEBITDATTM")
    base["pe_ratio"]       = _pick(km, "peRatioTTM", "priceEarningsRatioTTM")
    base["pb_ratio"]       = _pick(km, "pbRatioTTM", "priceToBookRatioTTM")
    base["debt_ebitda"]    = _pick(km, "netDebtToEBITDATTM")
    base["dividend_yield"] = _pick(km, "dividendYieldTTM")

    # Ratios TTM
    rt = _first(get("ratios-ttm", {"symbol": ticker}) or [])
    base["gross_margin"]      = _pick(rt, "grossProfitMarginTTM", "grossProfitMargin")
    base["net_margin"]        = _pick(rt, "netProfitMarginTTM",   "netProfitMargin")
    base["operating_margin"]  = _pick(rt, "operatingProfitMarginTTM")
    base["interest_coverage"] = _pick(rt, "interestCoverageTTM",  "interestCoverage")
    base["current_ratio"]     = _pick(rt, "currentRatioTTM",      "currentRatio")
    base["roe"]               = _pick(rt, "returnOnEquityTTM",    "returnOnEquity")

    # Income statement
    inc = get("income-statement", {"symbol": ticker, "limit": 4, "period": "annual"}) or []
    if len(inc) >= 2:
        rev_new = _safe(inc[0].get("revenue"))
        rev_old = _safe(inc[-1].get("revenue"))
        n_yr    = len(inc) - 1
        base["rev_cagr"] = (
            (rev_new / rev_old) ** (1 / n_yr) - 1
            if not np.isnan(rev_new) and rev_old and rev_old > 0 else np.nan
        )
        def _sh(row):
            for k in ["weightedAverageShsOut", "weightedAverageShsOutDil"]:
                v = _safe(row.get(k))
                if not np.isnan(v):
                    return v
            return np.nan
        sh_new, sh_old = _sh(inc[0]), _sh(inc[-1])
        base["share_dilution"] = (
            (sh_new / sh_old) ** (1 / n_yr) - 1
            if not np.isnan(sh_new) and sh_old and sh_old > 0 else np.nan
        )
    else:
        base["rev_cagr"] = base["share_dilution"] = np.nan

    # Cash flow
    cf    = _first(get("cash-flow-statement", {"symbol": ticker, "limit": 1, "period": "annual"}) or [])
    fcf   = _pick(cf, "freeCashFlow")
    capex = _pick(cf, "capitalExpenditure", "capitalExpenditures")
    if not np.isnan(capex):
        capex = abs(capex)
    rev_cf = _pick(cf, "revenue")
    if np.isnan(rev_cf) and inc:
        rev_cf = _safe(inc[0].get("revenue"))
    base["fcf_margin"]    = fcf / rev_cf   if not np.isnan(fcf) and rev_cf else np.nan
    base["capex_pct_rev"] = capex / rev_cf if not np.isnan(capex) and rev_cf else np.nan

    # Financial growth
    fg = _first(get("financial-growth", {"symbol": ticker, "limit": 1, "period": "annual"}) or [])
    base["fcf_growth"] = _pick(fg, "freeCashFlowGrowth")
    base["eps_growth"]  = _pick(fg, "epsgrowth", "epsGrowth")

    # DCF
    dcf       = _first(get("discounted-cash-flow", {"symbol": ticker}) or [])
    dcf_val   = _pick(dcf, "dcf", "DCF")
    dcf_price = _pick(dcf, "stockPrice", "Stock Price", "price")
    base["dcf_upside"]    = (
        (dcf_val - dcf_price) / dcf_price
        if not np.isnan(dcf_val) and dcf_price > 0 else np.nan
    )
    base["dcf_value"]     = dcf_val
    base["dcf_price_val"] = dcf_price

    # Piotroski
    fs = _first(get("financial-score", {"symbol": ticker}) or [])
    base["piotroski"] = _pick(fs, "piotroskiScore", "piotroski")

    # Insider trading (90d)
    ins    = get("insider-trading", {"symbol": ticker, "limit": 50}) or []
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)
    buys = sells = 0
    for t in (ins if isinstance(ins, list) else []):
        dt_str = t.get("transactionDate") or t.get("filingDate") or ""
        dt = pd.to_datetime(dt_str, errors="coerce")
        if pd.isna(dt) or dt < cutoff:
            continue
        tt = str(t.get("transactionType", "")).upper()
        if "PURCHASE" in tt or tt in ("P", "P-"):
            buys += 1
        elif "SALE" in tt or tt in ("S", "S-"):
            sells += 1
    base["insider_buys"]   = buys
    base["insider_sells"]  = sells
    base["insider_signal"] = buys - sells

    return base
