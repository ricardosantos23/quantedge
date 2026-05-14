"""
modules/ml_scorer.py — ML Stock Picker
========================================
Recebe os preços já carregados pelo dashboard e gera ML scores.
NÃO descarrega dados — usa o que já existe no pipeline.

Uso no app.py:
    from modules.ml_scorer import compute_ml_scores
    ml_scores = compute_ml_scores(prices_all)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

HORIZON_DAYS = 63
TRAIN_FRACTION = 0.70
FEATURE_COLS = ["ret_5d", "ret_21d", "ret_63d", "mom_12m_ex1m",
                "vol_21d", "vol_63d", "trend_50", "trend_200",
                "drawdown_12m", "vol_surge"]


def _build_ml_features(prices):
    """
    Constrói as 10 features do modelo ML a partir dos preços brutos.
    Diferentes das features do technicals.py (que usa z-scores, MACD, RSI).
    """
    df = prices.copy().sort_values(["ticker", "date"])

    # Precisa de coluna 'close' — adaptar se vier com outro nome
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    if "close" not in df.columns:
        return None, None

    g = df.groupby("ticker")["close"]

    df["ret_1d"]  = g.pct_change(1)
    df["ret_5d"]  = g.pct_change(5)
    df["ret_21d"] = g.pct_change(21)
    df["ret_63d"] = g.pct_change(63)

    ret_252 = g.pct_change(252)
    ret_21  = g.pct_change(21)
    df["mom_12m_ex1m"] = (1 + ret_252) / (1 + ret_21) - 1

    df["vol_21d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(21).std() * np.sqrt(252))
    df["vol_63d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(63).std() * np.sqrt(252))

    df["sma_50"]  = g.transform(lambda x: x.rolling(50).mean())
    df["sma_200"] = g.transform(lambda x: x.rolling(200).mean())
    df["trend_50"]  = df["close"] / df["sma_50"] - 1
    df["trend_200"] = df["close"] / df["sma_200"] - 1

    df["drawdown_12m"] = g.transform(
        lambda x: x / x.rolling(252, min_periods=20).max() - 1)

    df["dollar_vol"] = df["close"] * df["volume"]
    df["dollar_vol_ma"] = df.groupby("ticker")["dollar_vol"].transform(
        lambda x: x.rolling(21).mean())
    df["vol_surge"] = df["dollar_vol"] / df["dollar_vol_ma"] - 1

    # Target: forward return rank (para treino)
    df["fwd_return"] = g.transform(
        lambda x: x.shift(-HORIZON_DAYS) / x - 1)
    df["target_rank"] = df.groupby("date")["fwd_return"].rank(pct=True)

    df = df.dropna(subset=FEATURE_COLS + ["target_rank"])
    return df, FEATURE_COLS


def _train_models(train):
    """Treina 1 model point + 3 quantile."""
    X = train[FEATURE_COLS]
    y = train["target_rank"]
    params = dict(n_estimators=500, learning_rate=0.02, max_depth=5,
                  num_leaves=31, min_child_samples=50, colsample_bytree=0.8,
                  subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)

    model = lgb.LGBMRegressor(objective="regression", **params)
    model.fit(X, y)

    qmodels = {}
    for q in [0.1, 0.5, 0.9]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        m.fit(X, y)
        qmodels[q] = m

    return model, qmodels


def compute_ml_scores(prices):
    """
    Entry point principal. Recebe preços brutos, retorna DataFrame de scores.

    Parameters
    ----------
    prices : DataFrame com colunas [ticker, date, close/adj_close, volume]
             Estes são os preços que o dashboard já carregou.

    Returns
    -------
    DataFrame com colunas:
        ticker, date, ml_score, ml_q10, ml_q90, ml_uncert, ml_decile
    Ou None se não houver dados suficientes.
    """
    if prices is None or prices.empty:
        return None

    print("  ML Scorer: a construir features...")
    df, feature_cols = _build_ml_features(prices)
    if df is None or len(df) < 10000:
        print("  ML Scorer: dados insuficientes")
        return None

    # Split temporal
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRACTION)]
    train = df[df["date"] < cutoff - pd.Timedelta(days=HORIZON_DAYS + 5)]

    print(f"  ML Scorer: treino com {len(train):,} obs "
          f"({train['date'].min().date()} → {train['date'].max().date()})")

    # Treinar
    model, qmodels = _train_models(train)

    # Scoring: última data de cada ticker
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    X = latest[FEATURE_COLS]

    latest["ml_score"]  = model.predict(X)
    latest["ml_q10"]    = qmodels[0.1].predict(X)
    latest["ml_q90"]    = qmodels[0.9].predict(X)
    latest["ml_uncert"] = latest["ml_q90"] - latest["ml_q10"]
    latest["ml_decile"] = pd.qcut(
        latest["ml_score"], 10, labels=False, duplicates="drop"
    ) + 1

    result = latest[["ticker", "date", "ml_score", "ml_q10",
                      "ml_q90", "ml_uncert", "ml_decile"]].copy()
    result = result.sort_values("ml_score", ascending=False).reset_index(drop=True)

    print(f"  ML Scorer: {len(result)} tickers scored")
    return result
