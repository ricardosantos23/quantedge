"""
modules/ml_scorer.py — LightGBM-based stock ranking model.

This module exposes a single entry point, :func:`compute_ml_scores`, that
consumes the price history already loaded by the Dash application and
returns a per-ticker ML score, an 80% confidence interval, and a decile
rank used to populate the screener factor table.

Design principles
-----------------
* The module never fetches data on its own — it operates exclusively on
  the DataFrame passed in by the caller.
* Training uses an explicit walk-forward split: the most recent
  ``HORIZON_DAYS + 5`` observations are excluded from the training set
  to prevent target leakage.
* All status output goes through :mod:`logging`, so production deployments
  can route it to structured log sinks (stdout, files, Sentry, etc.).

Public API
----------
* :func:`compute_ml_scores` — Train the model and score the latest bar
  for each ticker.

Used by ``app.py`` to merge ML factors into the screener output.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Model configuration ──────────────────────────────────────────────────────

#: Forward window (in trading days) used to build the supervised target.
#: ~63 trading days ≈ 3 calendar months. Live scores predict the relative
#: ranking of each ticker over this horizon, so they should be refreshed
#: roughly at the same cadence.
HORIZON_DAYS: int = 63

#: Fraction of the price history used for training. The remainder is held
#: out as a validation buffer to prevent target leakage at inference time.
TRAIN_FRACTION: float = 0.70

#: Minimum number of feature rows required to attempt model training.
#: Below this threshold the function returns ``None`` and the screener
#: simply omits the ML columns from its output.
MIN_FEATURE_ROWS: int = 10_000

#: Ordered list of feature columns fed to the LightGBM models.
#: Order must remain stable across train and predict calls — changing it
#: invalidates any persisted model checkpoint.
FEATURE_COLS: list[str] = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "mom_12m_ex1m",
    "vol_21d",
    "vol_63d",
    "trend_50",
    "trend_200",
    "drawdown_12m",
    "vol_surge",
]

#: Required price-history columns. Volume is mandatory because the
#: ``vol_surge`` feature divides dollar volume by its rolling mean —
#: a missing column would silently degrade scores to noise.
REQUIRED_COLUMNS: set[str] = {"ticker", "date", "volume"}

#: LightGBM hyperparameters shared by the point and quantile models.
_LGBM_PARAMS: dict = dict(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=31,
    min_child_samples=50,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)


# ── Feature engineering ──────────────────────────────────────────────────────


def _build_ml_features(
    prices: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Derive the 10 ranking features from raw OHLCV prices.

    The feature set is intentionally different from the technicals module
    (which uses z-scored composites, MACD, RSI). Here the goal is to
    expose the underlying return-and-volatility structure to LightGBM
    in raw form so that the model can discover its own non-linear
    interactions.

    Args:
        prices: Raw price DataFrame with columns ``ticker``, ``date``,
            ``close`` (or ``adj_close``), and ``volume``.

    Returns:
        A two-tuple ``(features_df, feature_cols)``. ``features_df`` is
        ``None`` if the input lacks a usable close column.
    """
    df = prices.copy().sort_values(["ticker", "date"])

    # Accept either column name; downstream logic only knows "close".
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    if "close" not in df.columns:
        return None, FEATURE_COLS

    g = df.groupby("ticker")["close"]

    # Multi-horizon momentum
    df["ret_1d"] = g.pct_change(1)
    df["ret_5d"] = g.pct_change(5)
    df["ret_21d"] = g.pct_change(21)
    df["ret_63d"] = g.pct_change(63)

    # 12-month-minus-1-month momentum (classic Jegadeesh-Titman residual)
    ret_252 = g.pct_change(252)
    ret_21 = g.pct_change(21)
    df["mom_12m_ex1m"] = (1 + ret_252) / (1 + ret_21) - 1

    # Realised volatility (annualised) at two horizons
    df["vol_21d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(21).std() * np.sqrt(252)
    )
    df["vol_63d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(63).std() * np.sqrt(252)
    )

    # Trend strength relative to two long moving averages
    df["sma_50"] = g.transform(lambda x: x.rolling(50).mean())
    df["sma_200"] = g.transform(lambda x: x.rolling(200).mean())
    df["trend_50"] = df["close"] / df["sma_50"] - 1
    df["trend_200"] = df["close"] / df["sma_200"] - 1

    # Distance from rolling 12-month peak (always <= 0)
    df["drawdown_12m"] = g.transform(
        lambda x: x / x.rolling(252, min_periods=20).max() - 1
    )

    # Dollar volume surge: relative spike vs 21-day mean
    df["dollar_vol"] = df["close"] * df["volume"]
    df["dollar_vol_ma"] = df.groupby("ticker")["dollar_vol"].transform(
        lambda x: x.rolling(21).mean()
    )
    df["vol_surge"] = df["dollar_vol"] / df["dollar_vol_ma"] - 1

    # Supervised target: percentile rank of forward return across the
    # cross-section on each date. Using ranks instead of raw returns
    # stabilises the loss across volatility regimes.
    df["fwd_return"] = g.transform(lambda x: x.shift(-HORIZON_DAYS) / x - 1)
    df["target_rank"] = df.groupby("date")["fwd_return"].rank(pct=True)

    df = df.dropna(subset=FEATURE_COLS + ["target_rank"])
    return df, FEATURE_COLS


# ── Training ─────────────────────────────────────────────────────────────────


def _train_models(
    train: pd.DataFrame,
) -> tuple[lgb.LGBMRegressor, dict[float, lgb.LGBMRegressor]]:
    """Train one point regressor plus three quantile regressors.

    The point model targets the conditional mean rank; the three quantile
    models (q10, q50, q90) form an empirical 80% prediction interval
    surfaced to the UI as a confidence band.

    Args:
        train: Training slice containing :data:`FEATURE_COLS` and
            ``target_rank``.

    Returns:
        ``(point_model, {0.1: q10_model, 0.5: q50_model, 0.9: q90_model})``.

    Raises:
        RuntimeError: If LightGBM fails to fit (e.g. feature/target
            shape mismatch). The original exception is chained.
    """
    X = train[FEATURE_COLS]
    y = train["target_rank"]

    try:
        point_model = lgb.LGBMRegressor(objective="regression", **_LGBM_PARAMS)
        point_model.fit(X, y)

        quantile_models: dict[float, lgb.LGBMRegressor] = {}
        for q in (0.1, 0.5, 0.9):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **_LGBM_PARAMS)
            m.fit(X, y)
            quantile_models[q] = m
    except Exception as exc:
        raise RuntimeError(
            f"LightGBM training failed on {len(X):,} rows with "
            f"{len(FEATURE_COLS)} features"
        ) from exc

    return point_model, quantile_models


# ── Public API ───────────────────────────────────────────────────────────────


def compute_ml_scores(prices: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Train the ML ranker and score the latest bar for each ticker.

    The function performs four steps:

    1. Validate that the price DataFrame carries the required columns.
    2. Build feature rows and the supervised target from raw prices.
    3. Split temporally and train one point + three quantile models.
    4. Predict on the most recent bar per ticker and return a ranking.

    Args:
        prices: DataFrame containing the full price history already
            loaded by the dashboard. Must include the columns listed in
            :data:`REQUIRED_COLUMNS` plus either ``close`` or
            ``adj_close``.

    Returns:
        A DataFrame indexed by row position with columns
        ``[ticker, date, ml_score, ml_q10, ml_q90, ml_uncert, ml_decile]``
        sorted by ``ml_score`` descending. Returns ``None`` when the
        input is empty, lacks required columns, or yields fewer than
        :data:`MIN_FEATURE_ROWS` usable rows after feature engineering.

    Notes:
        Live scores use a model trained on a 3-month forward return that
        is, by construction, not yet observable. Treat the quantile band
        as a noisy proxy for model uncertainty rather than a strict
        confidence interval.
    """
    if prices is None or prices.empty:
        logger.info("ML Scorer: empty price input, skipping")
        return None

    missing = REQUIRED_COLUMNS - set(prices.columns)
    if missing:
        logger.error(
            "ML Scorer: missing required columns %s — cannot compute features",
            sorted(missing),
        )
        return None

    logger.info("ML Scorer: building features from %d price rows", len(prices))
    df, feature_cols = _build_ml_features(prices)
    if df is None:
        logger.warning("ML Scorer: no close/adj_close column available")
        return None
    if len(df) < MIN_FEATURE_ROWS:
        logger.warning(
            "ML Scorer: insufficient feature rows (%d < %d), skipping",
            len(df), MIN_FEATURE_ROWS,
        )
        return None

    # Temporal split — exclude the most recent HORIZON_DAYS+5 to make sure
    # the forward target is fully observable on every training row.
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRACTION)]
    train = df[df["date"] < cutoff - pd.Timedelta(days=HORIZON_DAYS + 5)]

    logger.info(
        "ML Scorer: training on %d observations (%s → %s)",
        len(train),
        train["date"].min().date(),
        train["date"].max().date(),
    )

    try:
        point_model, quantile_models = _train_models(train)
    except RuntimeError:
        logger.exception("ML Scorer: model training failed, returning None")
        return None

    # Score the most recent bar of each ticker.
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    X = latest[FEATURE_COLS]

    latest["ml_score"] = point_model.predict(X)
    latest["ml_q10"] = quantile_models[0.1].predict(X)
    latest["ml_q90"] = quantile_models[0.9].predict(X)
    latest["ml_uncert"] = latest["ml_q90"] - latest["ml_q10"]
    latest["ml_decile"] = (
        pd.qcut(latest["ml_score"], 10, labels=False, duplicates="drop") + 1
    )

    result = (
        latest[
            ["ticker", "date", "ml_score", "ml_q10", "ml_q90", "ml_uncert", "ml_decile"]
        ]
        .sort_values("ml_score", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("ML Scorer: scored %d tickers", len(result))
    return result
