"""
modules/regime_detector.py — Detector de regime de mercado (GMM)
================================================================
Recebe os preços SPY já carregados pelo dashboard.
NÃO descarrega dados — usa o que já existe no pipeline.

Uso no app.py:
    from modules.regime_detector import compute_regime
    regime = compute_regime(prices_all)
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

N_REGIMES       = 4
SMOOTH_WINDOW   = 10
MIN_REGIME_DAYS = 5


def _prepare_spy_features(spy_prices):
    """Features de longo prazo para o GMM a partir dos preços do SPY."""
    df = spy_prices.copy().sort_values("date")

    # Adaptar coluna de preço
    if "close" not in df.columns and "adj_close" in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    df["ret_1d"]    = df["close"].pct_change()
    df["ret_21d"]   = df["close"].pct_change(21)
    df["ret_63d"]   = df["close"].pct_change(63)
    df["vol_21d"]   = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"]   = df["ret_1d"].rolling(63).std() * np.sqrt(252)
    df["sma_200"]   = df["close"].rolling(200).mean()
    df["trend_200"] = df["close"] / df["sma_200"] - 1
    df["drawdown"]  = df["close"] / df["close"].cummax() - 1
    return df.dropna()


def _filter_short(states, min_days):
    """Remove regimes mais curtos que min_days, absorve pelo anterior."""
    result = states.copy()
    i = 0
    while i < len(result):
        cur = result[i]; j = i
        while j < len(result) and result[j] == cur:
            j += 1
        if j - i < min_days and i > 0:
            prev = result[i - 1]
            for k in range(i, j):
                result[k] = prev
        i = j
    return result


def _classify(df, states, n_regimes):
    """Classifica estados por composite score multi-dimensional."""
    df = df.copy()
    df["state"] = states
    rows = []
    for s in range(n_regimes):
        m = df["state"] == s
        if m.sum() == 0:
            continue
        rows.append({
            "state":      s,
            "mean_ret":   df.loc[m, "ret_1d"].mean(),
            "mean_vol":   df.loc[m, "vol_21d"].mean(),
            "mean_trend": df.loc[m, "trend_200"].mean(),
            "mean_dd":    df.loc[m, "drawdown"].mean(),
            "pct":        m.mean() * 100,
        })

    stats = pd.DataFrame(rows)

    def _z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0

    stats["composite"] = (
        _z(stats["mean_trend"]) * 0.35 +
        _z(stats["mean_ret"])   * 0.20 +
        _z(stats["mean_vol"])   * (-0.20) +
        _z(stats["mean_dd"].abs()) * (-0.25)
    )

    stats = stats.sort_values("composite", ascending=False)
    ranked = stats["state"].tolist()

    defs = [
        ("BULL FORTE",  "★★★★★", "#2D6A4F", "Modelo no seu melhor. Confia no ranking."),
        ("BULL CALMO",  "★★★★",  "#95D5B2", "Modelo funciona bem. Diversifica (top 20)."),
        ("INCERTO",     "★★",    "#E67E22", "Regime incerto. Modelo pode subperformar."),
        ("STRESS",      "★",     "#C0392B", "Mercado sob stress. Proteger capital."),
    ]

    labels, stars, colors, recs = {}, {}, {}, {}
    for i, s in enumerate(ranked):
        if i < len(defs):
            labels[s], stars[s], colors[s], recs[s] = defs[i]
        else:
            labels[s], stars[s], colors[s], recs[s] = f"R{i}", "?", "gray", "?"

    return labels, stars, colors, recs, stats


def _compute_durations(states):
    """Duração média de cada regime em dias úteis."""
    durs = {}
    cur, length = states[0], 1
    for i in range(1, len(states)):
        if states[i] == cur:
            length += 1
        else:
            durs.setdefault(cur, []).append(length)
            cur, length = states[i], 1
    durs.setdefault(cur, []).append(length)
    return {s: np.mean(d) for s, d in durs.items()}


def compute_regime(prices):
    """
    Entry point principal. Extrai SPY dos preços, treina GMM, classifica regime.

    Parameters
    ----------
    prices : DataFrame com colunas [ticker, date, close/adj_close, volume]
             Deve incluir ticker 'SPY'.

    Returns
    -------
    dict com regime actual, probabilidades, transições.
    Ou None se SPY não estiver nos dados.
    """
    if prices is None or prices.empty:
        return None

    # Extrair SPY
    spy = prices[prices["ticker"] == "SPY"].copy()
    if spy.empty:
        print("  Regime: SPY não encontrado nos preços")
        return None

    print("  Regime: a preparar features SPY...")
    df = _prepare_spy_features(spy)
    if len(df) < 500:
        print("  Regime: histórico SPY insuficiente")
        return None

    # Treinar GMM
    feature_cols = ["ret_21d", "ret_63d", "vol_21d", "vol_63d",
                    "trend_200", "drawdown"]
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print(f"  Regime: a treinar GMM ({len(X)} obs, {N_REGIMES} estados)...")
    best_model, best_score = None, -np.inf
    for seed in range(20):
        gmm = GaussianMixture(n_components=N_REGIMES, covariance_type="full",
                               n_init=3, max_iter=300, random_state=seed)
        gmm.fit(X_s)
        s = gmm.score(X_s)
        if s > best_score:
            best_score, best_model = s, gmm

    # Probabilidades suavizadas
    raw_probs = best_model.predict_proba(X_s)
    smoothed = np.zeros_like(raw_probs)
    for s in range(N_REGIMES):
        smoothed[:, s] = pd.Series(raw_probs[:, s]).rolling(
            SMOOTH_WINDOW, min_periods=1).mean().values
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    states = smoothed.argmax(axis=1)
    states = _filter_short(states, MIN_REGIME_DAYS)

    # Transições empíricas
    trans = np.zeros((N_REGIMES, N_REGIMES))
    for t in range(len(states) - 1):
        trans[states[t], states[t+1]] += 1
    rs = trans.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    trans /= rs

    # Classificar
    labels, stars_map, colors, recs, stats = _classify(df, states, N_REGIMES)
    durs = _compute_durations(states)

    # Estado actual
    cur = states[-1]
    cur_probs = smoothed[-1]
    cur_date = df["date"].iloc[-1]
    stay_prob = trans[cur, cur]
    expected_dur = 1 / (1 - stay_prob + 1e-9)

    result = {
        "date":                  cur_date.strftime("%Y-%m-%d") if hasattr(cur_date, "strftime") else str(cur_date),
        "regime":                labels[cur],
        "stars":                 stars_map[cur],
        "color":                 colors[cur],
        "recommendation":       recs[cur],
        "composite":             round(float(stats[stats["state"] == cur]["composite"].iloc[0]), 2),
        "stay_probability":      round(float(stay_prob), 3),
        "expected_duration_days": round(float(expected_dur)),
        "probabilities": {
            labels[s]: round(float(cur_probs[s]), 3) for s in range(N_REGIMES)
        },
    }

    print(f"  Regime: {labels[cur]} {stars_map[cur]} "
          f"(manter: {stay_prob:.0%}, duração ~{expected_dur:.0f}d)")

    return result
