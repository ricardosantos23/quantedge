"""
modules/regime_detector.py — Market regime classifier (Gaussian Mixture).

This module exposes a single entry point, :func:`compute_regime`, that
fits a Gaussian Mixture Model on SPY daily features and classifies the
*current* market state into one of four labelled regimes:

* **BULL STRONG** — Persistent uptrend, low realised volatility.
* **BULL CALM** — Uptrend with rising dispersion.
* **UNCERTAIN** — Sideways or transitional state.
* **STRESS** — Drawdown phase, elevated volatility.

The classification is rendered as a banner on the screener tab so that
the user can interpret model output (e.g. ML ranking confidence) in the
appropriate macro context.

Design principles
-----------------
* The module never fetches data on its own — it operates exclusively on
  the price DataFrame passed in by the caller.
* The GMM is refit on every call to keep the model in sync with the
  most recent observations. Fitting takes ~2-3 seconds on a 10-year
  SPY history; this is fast enough to run inside the dashboard cache
  warm-up.
* The labelling step uses a composite score (trend, return, volatility,
  drawdown) so regime names remain stable even when the underlying GMM
  components are permuted between fits.

Public API
----------
* :func:`compute_regime` — Train the GMM, smooth state probabilities,
  and return the current regime dict.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Model configuration ──────────────────────────────────────────────────────

#: Number of Gaussian components in the mixture model. Four maps cleanly
#: onto BULL STRONG / BULL CALM / UNCERTAIN / STRESS without leaving
#: many tickers in unlabeled tails.
N_REGIMES: int = 4

#: Rolling window (trading days) for smoothing the raw component
#: probabilities. Reduces jitter at regime transitions without losing
#: signal at well-defined turning points.
SMOOTH_WINDOW: int = 10

#: Minimum dwell time (trading days) below which short regime "blips"
#: get absorbed into the previous regime. Prevents single-day flickers
#: from being labelled as transitions.
MIN_REGIME_DAYS: int = 5

#: Minimum number of SPY observations required to attempt classification.
#: ~500 trading days ≈ 2 years, which is enough for the GMM to identify
#: stable component clusters.
MIN_SPY_OBSERVATIONS: int = 500

#: Number of random initialisations used to select the best GMM fit.
#: Picks the seed with the highest log-likelihood on the training data.
GMM_INIT_SEEDS: int = 20

#: Feature columns used by the GMM. Order is fixed for reproducibility.
GMM_FEATURE_COLS: list[str] = [
    "ret_21d",
    "ret_63d",
    "vol_21d",
    "vol_63d",
    "trend_200",
    "drawdown",
]

#: Required price-history columns for SPY extraction.
REQUIRED_COLUMNS: set[str] = {"ticker", "date"}

#: Regime definitions ordered from most bullish to most bearish.
#: Each tuple holds (label, star-rating, hex color, recommendation text).
_REGIME_DEFINITIONS: list[tuple[str, str, str, str]] = [
    (
        "BULL FORTE",
        "★★★★★",
        "#2D6A4F",
        "Model is at peak performance. Trust the ranking.",
    ),
    (
        "BULL CALMO",
        "★★★★",
        "#95D5B2",
        "Model works well. Diversify across the top 20.",
    ),
    (
        "INCERTO",
        "★★",
        "#E67E22",
        "Uncertain regime. Model may underperform — reduce conviction.",
    ),
    (
        "STRESS",
        "★",
        "#C0392B",
        "Market stress. Protect capital, raise cash, tighten stops.",
    ),
]


# ── Feature engineering ──────────────────────────────────────────────────────


def _prepare_spy_features(spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Derive long-horizon features used to fit the GMM.

    Args:
        spy_prices: SPY-only slice of the master price DataFrame.

    Returns:
        DataFrame with one row per trading day, NaN rows dropped.
    """
    df = spy_prices.copy().sort_values("date")

    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    df["ret_1d"] = df["close"].pct_change()
    df["ret_21d"] = df["close"].pct_change(21)
    df["ret_63d"] = df["close"].pct_change(63)
    df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["ret_1d"].rolling(63).std() * np.sqrt(252)
    df["sma_200"] = df["close"].rolling(200).mean()
    df["trend_200"] = df["close"] / df["sma_200"] - 1
    df["drawdown"] = df["close"] / df["close"].cummax() - 1
    return df.dropna()


# ── State post-processing ────────────────────────────────────────────────────


def _filter_short(states: np.ndarray, min_days: int) -> np.ndarray:
    """Absorb regime spells shorter than ``min_days`` into the previous one.

    The first spell (index 0) is never absorbed because there is no
    previous regime to absorb it into.

    Args:
        states: 1-D array of integer state labels in chronological order.
        min_days: Minimum dwell time below which a spell is absorbed.

    Returns:
        New array with short spells rewritten.
    """
    result = states.copy()
    i = 0
    while i < len(result):
        cur = result[i]
        j = i
        while j < len(result) and result[j] == cur:
            j += 1
        if j - i < min_days and i > 0:
            prev = result[i - 1]
            for k in range(i, j):
                result[k] = prev
        i = j
    return result


def _classify(
    df: pd.DataFrame, states: np.ndarray, n_regimes: int
) -> tuple[dict, dict, dict, dict, pd.DataFrame]:
    """Assign human-readable labels to GMM components.

    Components are ranked by a composite score combining trend, mean
    return, volatility, and drawdown depth. This keeps labels stable
    across runs even when the GMM permutes its internal component order.

    Args:
        df: Feature frame including a ``state`` column added by caller.
        states: Smoothed state labels (must align with ``df`` rows).
        n_regimes: Number of components in the mixture.

    Returns:
        Five-tuple ``(labels, stars, colors, recommendations, stats_df)``.
    """
    df = df.copy()
    df["state"] = states
    rows = []
    for s in range(n_regimes):
        mask = df["state"] == s
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "state": s,
                "mean_ret": df.loc[mask, "ret_1d"].mean(),
                "mean_vol": df.loc[mask, "vol_21d"].mean(),
                "mean_trend": df.loc[mask, "trend_200"].mean(),
                "mean_dd": df.loc[mask, "drawdown"].mean(),
                "pct": mask.mean() * 100,
            }
        )

    stats = pd.DataFrame(rows)

    def _z(series: pd.Series) -> pd.Series:
        sd = series.std()
        return (series - series.mean()) / sd if sd > 0 else series * 0

    # Composite ranking — weights chosen to surface the most informative
    # dimensions (trend dominates, then return; volatility/drawdown
    # penalise downside states).
    stats["composite"] = (
        _z(stats["mean_trend"]) * 0.35
        + _z(stats["mean_ret"]) * 0.20
        + _z(stats["mean_vol"]) * (-0.20)
        + _z(stats["mean_dd"].abs()) * (-0.25)
    )

    stats = stats.sort_values("composite", ascending=False)
    ranked_states = stats["state"].tolist()

    labels: dict = {}
    stars: dict = {}
    colors: dict = {}
    recommendations: dict = {}
    for idx, state in enumerate(ranked_states):
        if idx < len(_REGIME_DEFINITIONS):
            label, star, color, rec = _REGIME_DEFINITIONS[idx]
        else:
            label, star, color, rec = f"R{idx}", "?", "gray", "?"
        labels[state] = label
        stars[state] = star
        colors[state] = color
        recommendations[state] = rec

    return labels, stars, colors, recommendations, stats


def _compute_durations(states: np.ndarray) -> dict[int, float]:
    """Compute the mean dwell time of each state.

    Args:
        states: Smoothed state labels in chronological order.

    Returns:
        Mapping ``{state: mean_spell_length_in_days}``.
    """
    durations: dict[int, list[int]] = {}
    cur, length = states[0], 1
    for i in range(1, len(states)):
        if states[i] == cur:
            length += 1
        else:
            durations.setdefault(cur, []).append(length)
            cur, length = states[i], 1
    durations.setdefault(cur, []).append(length)
    return {s: float(np.mean(d)) for s, d in durations.items()}


# ── Public API ───────────────────────────────────────────────────────────────


def compute_regime(prices: pd.DataFrame) -> Optional[dict]:
    """Fit the GMM on SPY history and classify the current market regime.

    Args:
        prices: Full price DataFrame already loaded by the dashboard.
            Must include the columns listed in :data:`REQUIRED_COLUMNS`
            plus either ``close`` or ``adj_close``, and must contain SPY
            with at least :data:`MIN_SPY_OBSERVATIONS` rows.

    Returns:
        A dict describing the current regime, or ``None`` if SPY data
        is missing or insufficient. Keys:

        * ``date`` (str)            — Date of the latest observation
        * ``regime`` (str)          — Label (e.g. ``"BULL FORTE"``)
        * ``stars`` (str)           — 1-5 star quality rating
        * ``color`` (str)           — Hex color for the UI banner
        * ``recommendation`` (str)  — Suggested action for the regime
        * ``composite`` (float)     — Composite ranking score
        * ``stay_probability`` (float)        — P(remain in regime tomorrow)
        * ``expected_duration_days`` (int)    — Expected days until transition
        * ``probabilities`` (dict[str, float]) — Probability of each regime

    Notes:
        The returned dict is always either ``None`` or fully populated
        with every key listed above. Callers may render defensively
        with ``regime.get(...)`` but they will never observe a partial
        dict in normal operation.
    """
    if prices is None or prices.empty:
        logger.info("Regime: empty price input, skipping")
        return None

    missing = REQUIRED_COLUMNS - set(prices.columns)
    if missing:
        logger.error(
            "Regime: missing required columns %s — cannot compute features",
            sorted(missing),
        )
        return None

    spy = prices[prices["ticker"] == "SPY"].copy()
    if spy.empty:
        logger.warning("Regime: SPY not found in price data, skipping")
        return None

    logger.info("Regime: preparing SPY features (%d rows)", len(spy))
    df = _prepare_spy_features(spy)
    if len(df) < MIN_SPY_OBSERVATIONS:
        logger.warning(
            "Regime: insufficient SPY history (%d < %d), skipping",
            len(df), MIN_SPY_OBSERVATIONS,
        )
        return None

    # Standardise features before fitting — GMM components are not
    # scale-invariant, and the feature set mixes returns (~0.001) with
    # vol/drawdown (~0.2).
    X = df[GMM_FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        "Regime: fitting GMM with %d components on %d observations",
        N_REGIMES, len(X_scaled),
    )
    best_model: Optional[GaussianMixture] = None
    best_score = -np.inf
    for seed in range(GMM_INIT_SEEDS):
        try:
            gmm = GaussianMixture(
                n_components=N_REGIMES,
                covariance_type="full",
                n_init=3,
                max_iter=300,
                random_state=seed,
            )
            gmm.fit(X_scaled)
        except Exception:
            logger.debug("Regime: GMM seed %d failed to converge", seed)
            continue
        score = gmm.score(X_scaled)
        if score > best_score:
            best_score, best_model = score, gmm

    if best_model is None:
        logger.error("Regime: all %d GMM seeds failed", GMM_INIT_SEEDS)
        return None

    # Smooth component probabilities to reduce day-to-day jitter, then
    # collapse to hard state labels and filter out micro-spells.
    raw_probs = best_model.predict_proba(X_scaled)
    smoothed = np.zeros_like(raw_probs)
    for s in range(N_REGIMES):
        smoothed[:, s] = (
            pd.Series(raw_probs[:, s])
            .rolling(SMOOTH_WINDOW, min_periods=1)
            .mean()
            .values
        )
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    states = smoothed.argmax(axis=1)
    states = _filter_short(states, MIN_REGIME_DAYS)

    # Empirical transition matrix from the post-smoothed state sequence.
    transitions = np.zeros((N_REGIMES, N_REGIMES))
    for t in range(len(states) - 1):
        transitions[states[t], states[t + 1]] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero
    transitions /= row_sums

    labels, stars_map, colors, recommendations, stats = _classify(
        df, states, N_REGIMES
    )
    durations = _compute_durations(states)

    cur_state = int(states[-1])
    cur_probs = smoothed[-1]
    cur_date = df["date"].iloc[-1]
    stay_prob = float(transitions[cur_state, cur_state])
    expected_duration = 1.0 / (1.0 - stay_prob + 1e-9)

    result = {
        "date": cur_date.strftime("%Y-%m-%d")
        if hasattr(cur_date, "strftime")
        else str(cur_date),
        "regime": labels[cur_state],
        "stars": stars_map[cur_state],
        "color": colors[cur_state],
        "recommendation": recommendations[cur_state],
        "composite": round(
            float(stats[stats["state"] == cur_state]["composite"].iloc[0]), 2
        ),
        "stay_probability": round(stay_prob, 3),
        "expected_duration_days": int(round(expected_duration)),
        "probabilities": {
            labels[s]: round(float(cur_probs[s]), 3) for s in range(N_REGIMES)
        },
    }

    logger.info(
        "Regime: %s %s (stay %.0f%%, duration ~%dd)",
        labels[cur_state],
        stars_map[cur_state],
        stay_prob * 100,
        int(round(expected_duration)),
    )

    return result
