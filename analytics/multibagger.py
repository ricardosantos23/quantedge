"""
analytics/multibagger.py — Opportunity Score for multi-bagger detection.

Three sub-scores:
  1. Earnings Surprise Velocity (40%) — beat estimates 2-4 quarters in a row + accelerating
  2. Analyst Revision Momentum  (35%) — analysts raising targets vs cutting
  3. Sector Rotation Signal     (25%) — sector relative strength vs market (from prices in DB)

All data cached in DB table `opportunity_scores`.
Called nightly by scheduler and on-demand from screener.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

import config
from db.connection import Session, engine
from ingestion.fmp_client import get, _safe


# ════════════════════════════════════════════════════════════════════
#  FMP DATA FETCHERS
# ════════════════════════════════════════════════════════════════════

def fetch_earnings_surprises(ticker: str) -> list[dict]:
    """Last 6 quarters of earnings surprises."""
    data = get("earnings-surprises", {"symbol": ticker, "limit": 6}) or []
    rows = []
    for item in (data if isinstance(data, list) else []):
        if not isinstance(item, dict):
            continue
        actual   = _safe(item.get("actualEarningResult") or item.get("actual"))
        estimate = _safe(item.get("estimatedEarning") or item.get("estimated"))
        if np.isnan(actual) or np.isnan(estimate) or estimate == 0:
            continue
        surprise_pct = (actual - estimate) / abs(estimate) * 100
        rows.append({
            "date":         item.get("date", ""),
            "actual":       actual,
            "estimate":     estimate,
            "surprise_pct": surprise_pct,
        })
    return rows


def fetch_analyst_estimates(ticker: str) -> dict:
    """Analyst EPS estimate revisions and price target summary."""
    result = {"target_consensus": np.nan, "target_high": np.nan,
              "num_analysts": 0, "recent_upgrades": 0, "recent_downgrades": 0}

    # Price target summary
    pt = get("price-target-summary", {"symbol": ticker})
    if pt and isinstance(pt, dict):
        result["target_consensus"] = _safe(pt.get("targetConsensus") or pt.get("priceTargetAverage"))
        result["target_high"]      = _safe(pt.get("targetHigh") or pt.get("priceTargetHigh"))
        result["num_analysts"]     = int(pt.get("numberOfAnalysts") or pt.get("numOfAnalysts") or 0)

    # Recent upgrades/downgrades (last 90 days)
    ud = get("upgrades-downgrades", {"symbol": ticker, "limit": 20}) or []
    if isinstance(ud, list):
        cutoff = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        for item in ud:
            if not isinstance(item, dict):
                continue
            dt = str(item.get("publishedDate") or item.get("date") or "")[:10]
            if dt < cutoff:
                continue
            action = str(item.get("newGrade") or item.get("action") or "").lower()
            prev   = str(item.get("previousGrade") or "").lower()
            # Upgrade signals
            if any(w in action for w in ["buy", "outperform", "overweight", "strong buy"]):
                if any(w in prev for w in ["hold", "neutral", "underperform", "sell", ""]):
                    result["recent_upgrades"] += 1
            # Downgrade signals
            if any(w in action for w in ["sell", "underperform", "underweight"]):
                result["recent_downgrades"] += 1

    return result


# ════════════════════════════════════════════════════════════════════
#  SCORE COMPUTATION
# ════════════════════════════════════════════════════════════════════

def _earnings_surprise_score(surprises: list[dict]) -> float:
    """
    Score 0-1 based on:
    - Consecutive beats (most important)
    - Acceleration of surprise magnitude
    - Consistency
    """
    if len(surprises) < 2:
        return 0.0

    # Sort chronologically (oldest first)
    surprises = sorted(surprises, key=lambda x: x["date"])
    pcts = [s["surprise_pct"] for s in surprises]

    # Consecutive beats score
    beats = [1 if p > 0 else 0 for p in pcts]
    # Count trailing consecutive beats
    consecutive = 0
    for b in reversed(beats):
        if b == 1:
            consecutive += 1
        else:
            break

    consec_score = min(consecutive / 4.0, 1.0)  # max at 4 consecutive

    # Acceleration: is surprise getting bigger over time?
    if len(pcts) >= 3:
        recent_avg = np.mean(pcts[-2:])
        older_avg  = np.mean(pcts[:-2])
        acceleration = 1.0 if recent_avg > older_avg else 0.0
    else:
        acceleration = 1.0 if pcts[-1] > pcts[0] else 0.0

    # Magnitude: average surprise of last 2 quarters
    recent_magnitude = np.mean(pcts[-2:]) if len(pcts) >= 2 else pcts[-1]
    mag_score = min(max(recent_magnitude / 20.0, 0.0), 1.0)  # 20% surprise → score 1.0

    return 0.50 * consec_score + 0.30 * mag_score + 0.20 * acceleration


def _analyst_revision_score(analyst_data: dict, current_price: float) -> float:
    """
    Score 0-1 based on:
    - Upgrade vs downgrade ratio (last 90 days)
    - Price target upside vs current price
    - Number of analysts covering (more = less hidden)
    """
    ups   = analyst_data.get("recent_upgrades", 0)
    downs = analyst_data.get("recent_downgrades", 0)
    total_actions = ups + downs

    # Upgrade momentum
    if total_actions > 0:
        upgrade_ratio = ups / total_actions
    elif ups == 0 and downs == 0:
        upgrade_ratio = 0.5  # neutral
    else:
        upgrade_ratio = 0.0

    # Price target upside
    consensus = analyst_data.get("target_consensus", np.nan)
    if not np.isnan(consensus) and current_price > 0:
        upside = (consensus - current_price) / current_price
        upside_score = min(max(upside / 0.30, 0.0), 1.0)  # 30% upside → score 1.0
    else:
        upside_score = 0.5

    # Analyst coverage (fewer analysts = more hidden opportunity)
    n = analyst_data.get("num_analysts", 0)
    if n == 0:
        coverage_score = 0.5
    elif n <= 5:
        coverage_score = 1.0    # very underfollowed
    elif n <= 15:
        coverage_score = 0.6    # moderate coverage
    else:
        coverage_score = 0.2    # well covered, less hidden

    return 0.45 * upgrade_ratio + 0.35 * upside_score + 0.20 * coverage_score


def _sector_rotation_score(ticker: str, sector: str, prices_df: pd.DataFrame) -> float:
    """
    Score 0-1 based on:
    - Sector relative strength vs SPY (last 20 and 40 trading days)
    - Stock relative strength vs its sector
    """
    if prices_df.empty or ticker not in prices_df["ticker"].values:
        return 0.5

    price_col = "adj_close" if "adj_close" in prices_df.columns else "close"

    def _rel_strength(t1: str, t2: str, days: int) -> float:
        """Return t1 return / t2 return over last `days` rows."""
        d1 = prices_df[prices_df["ticker"] == t1].sort_values("date").tail(days)
        d2 = prices_df[prices_df["ticker"] == t2].sort_values("date").tail(days)
        if len(d1) < 5 or len(d2) < 5:
            return 0.0
        r1 = float(d1[price_col].iloc[-1]) / float(d1[price_col].iloc[0]) - 1
        r2 = float(d2[price_col].iloc[-1]) / float(d2[price_col].iloc[0]) - 1
        return r1 - r2  # excess return

    # Sector tickers in DB (same sector)
    sector_tickers = prices_df[prices_df.get("sector", pd.Series()) == sector]["ticker"].unique() \
        if "sector" in prices_df.columns else []

    # Stock vs SPY
    vs_spy_20 = _rel_strength(ticker, "SPY", 20)
    vs_spy_40 = _rel_strength(ticker, "SPY", 40)

    # Average sector vs SPY (sector momentum)
    if len(sector_tickers) > 2:
        sector_vs_spy = np.nanmean([
            _rel_strength(t, "SPY", 20)
            for t in sector_tickers
            if t != ticker and t in prices_df["ticker"].values
        ])
    else:
        sector_vs_spy = 0.0

    # Combine: sector in rotation AND stock outperforming sector
    sector_rotation = min(max((sector_vs_spy + 0.05) / 0.10, 0.0), 1.0)
    stock_momentum  = min(max((vs_spy_20 + 0.05) / 0.15, 0.0), 1.0)
    accel           = 1.0 if vs_spy_20 > vs_spy_40 else 0.3

    return 0.40 * stock_momentum + 0.40 * sector_rotation + 0.20 * accel


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def compute_opportunity_scores(
    tickers: list[str],
    prices_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    max_workers: int = 4,
    min_market_cap: float = 300_000_000,
) -> pd.DataFrame:
    """
    Compute opportunity scores for all tickers.
    Returns DataFrame with columns:
        ticker, earnings_surprise_score, analyst_revision_score,
        sector_rotation_score, opportunity_score
    """
    if not tickers:
        return pd.DataFrame()

    price_col = "adj_close" if "adj_close" in prices_df.columns else "close"

    # Filter by market cap if stocks_df has it
    if not stocks_df.empty and "Market Cap" in stocks_df.columns:
        eligible = stocks_df[
            stocks_df["Market Cap"].fillna(0) >= min_market_cap
        ]["Symbol"].tolist()
        tickers = [t for t in tickers if t in eligible]

    if not tickers:
        return pd.DataFrame()

    # Get current prices
    current_prices = {}
    if not prices_df.empty:
        latest = prices_df.sort_values("date").groupby("ticker").tail(1)
        for _, row in latest.iterrows():
            current_prices[row["ticker"]] = float(row[price_col])

    # Get sector mapping
    sector_map = {}
    if not stocks_df.empty and "Sector" in stocks_df.columns:
        for _, row in stocks_df.iterrows():
            sector_map[row.get("Symbol", "")] = row.get("Sector", "")

    # Sector column on prices_df for rotation calculation
    if not prices_df.empty and "sector" not in prices_df.columns and sector_map:
        prices_df = prices_df.copy()
        prices_df["sector"] = prices_df["ticker"].map(sector_map)

    def _score_one(ticker: str) -> dict:
        base = {
            "ticker": ticker,
            "earnings_surprise_score": None,
            "analyst_revision_score":  None,
            "sector_rotation_score":   None,
            "opportunity_score":       None,
            "consecutive_beats":       0,
            "recent_surprise_pct":     None,
            "analyst_upgrades_90d":    0,
            "analyst_downgrades_90d":  0,
            "price_target_upside":     None,
            "updated_at":              datetime.utcnow(),
        }
        try:
            # 1. Earnings surprise
            surprises = fetch_earnings_surprises(ticker)
            es_score  = _earnings_surprise_score(surprises)
            base["earnings_surprise_score"] = round(es_score, 3)
            if surprises:
                sorted_s = sorted(surprises, key=lambda x: x["date"])
                base["recent_surprise_pct"] = round(sorted_s[-1]["surprise_pct"], 1)
                base["consecutive_beats"]   = sum(
                    1 for s in sorted_s[-4:] if s["surprise_pct"] > 0
                )

            # 2. Analyst revision
            cur_px = current_prices.get(ticker, 0.0)
            analyst = fetch_analyst_estimates(ticker)
            ar_score = _analyst_revision_score(analyst, cur_px)
            base["analyst_revision_score"] = round(ar_score, 3)
            base["analyst_upgrades_90d"]   = analyst.get("recent_upgrades", 0)
            base["analyst_downgrades_90d"] = analyst.get("recent_downgrades", 0)
            consensus = analyst.get("target_consensus", np.nan)
            if not np.isnan(consensus) and cur_px > 0:
                base["price_target_upside"] = round((consensus - cur_px) / cur_px * 100, 1)

            # 3. Sector rotation (from prices — no API call)
            sector = sector_map.get(ticker, "")
            sr_score = _sector_rotation_score(ticker, sector, prices_df)
            base["sector_rotation_score"] = round(sr_score, 3)

            # Final weighted score
            opp = 0.40 * es_score + 0.35 * ar_score + 0.25 * sr_score
            base["opportunity_score"] = round(opp, 3)

            time.sleep(0.1)  # gentle rate limiting
        except Exception as e:
            pass  # leave as None — don't crash the whole batch
        return base

    results = []
    total = len(tickers)
    done  = 0
    print(f"\n  [opportunity] scoring {total} tickers...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_score_one, t): t for t in tickers}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            pct = int(done / total * 100)
            print(f"\r  [opportunity] {pct}%  ", end="", flush=True)
    print()

    df = pd.DataFrame(results)
    return df


def load_opportunity_scores_from_db(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load cached opportunity scores from DB."""
    try:
        with Session() as s:
            if tickers:
                rows = s.execute(text("""
                    SELECT ticker, opportunity_score, earnings_surprise_score,
                           analyst_revision_score, sector_rotation_score,
                           consecutive_beats, recent_surprise_pct,
                           analyst_upgrades_90d, analyst_downgrades_90d,
                           price_target_upside, updated_at
                    FROM opportunity_scores
                    WHERE ticker = ANY(:t)
                """), {"t": tickers}).fetchall()
            else:
                rows = s.execute(text("""
                    SELECT ticker, opportunity_score, earnings_surprise_score,
                           analyst_revision_score, sector_rotation_score,
                           consecutive_beats, recent_surprise_pct,
                           analyst_upgrades_90d, analyst_downgrades_90d,
                           price_target_upside, updated_at
                    FROM opportunity_scores
                """)).fetchall()

        if not rows:
            return pd.DataFrame()

        cols = ["ticker","opportunity_score","earnings_surprise_score",
                "analyst_revision_score","sector_rotation_score",
                "consecutive_beats","recent_surprise_pct",
                "analyst_upgrades_90d","analyst_downgrades_90d",
                "price_target_upside","updated_at"]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()


def save_opportunity_scores_to_db(df: pd.DataFrame) -> None:
    """Upsert opportunity scores into DB."""
    if df.empty:
        return
    records = df.to_dict("records")
    try:
        with Session() as s:
            # Create table if not exists
            s.execute(text("""
                CREATE TABLE IF NOT EXISTS opportunity_scores (
                    ticker                  VARCHAR(20) PRIMARY KEY,
                    opportunity_score       FLOAT,
                    earnings_surprise_score FLOAT,
                    analyst_revision_score  FLOAT,
                    sector_rotation_score   FLOAT,
                    consecutive_beats       INTEGER DEFAULT 0,
                    recent_surprise_pct     FLOAT,
                    analyst_upgrades_90d    INTEGER DEFAULT 0,
                    analyst_downgrades_90d  INTEGER DEFAULT 0,
                    price_target_upside     FLOAT,
                    updated_at              TIMESTAMP DEFAULT NOW()
                )
            """))
            s.commit()

            for batch_start in range(0, len(records), 100):
                batch = records[batch_start:batch_start+100]
                batch = [{k: v for k, v in r.items()
                          if k in ["ticker","opportunity_score","earnings_surprise_score",
                                   "analyst_revision_score","sector_rotation_score",
                                   "consecutive_beats","recent_surprise_pct",
                                   "analyst_upgrades_90d","analyst_downgrades_90d",
                                   "price_target_upside","updated_at"]}
                         for r in batch]
                stmt = insert(text("opportunity_scores")).values(batch)
                s.execute(text("""
                    INSERT INTO opportunity_scores
                        (ticker, opportunity_score, earnings_surprise_score,
                         analyst_revision_score, sector_rotation_score,
                         consecutive_beats, recent_surprise_pct,
                         analyst_upgrades_90d, analyst_downgrades_90d,
                         price_target_upside, updated_at)
                    VALUES
                        (:ticker, :opportunity_score, :earnings_surprise_score,
                         :analyst_revision_score, :sector_rotation_score,
                         :consecutive_beats, :recent_surprise_pct,
                         :analyst_upgrades_90d, :analyst_downgrades_90d,
                         :price_target_upside, :updated_at)
                    ON CONFLICT (ticker) DO UPDATE SET
                        opportunity_score       = EXCLUDED.opportunity_score,
                        earnings_surprise_score = EXCLUDED.earnings_surprise_score,
                        analyst_revision_score  = EXCLUDED.analyst_revision_score,
                        sector_rotation_score   = EXCLUDED.sector_rotation_score,
                        consecutive_beats       = EXCLUDED.consecutive_beats,
                        recent_surprise_pct     = EXCLUDED.recent_surprise_pct,
                        analyst_upgrades_90d    = EXCLUDED.analyst_upgrades_90d,
                        analyst_downgrades_90d  = EXCLUDED.analyst_downgrades_90d,
                        price_target_upside     = EXCLUDED.price_target_upside,
                        updated_at              = EXCLUDED.updated_at
                """), batch)
            s.commit()
    except Exception as e:
        print(f"  [opportunity] DB save error: {e}")
