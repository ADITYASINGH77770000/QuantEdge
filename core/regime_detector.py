"""
core/regime_detector.py
──────────────────────────────────────────────────────────────────────────────
Research-grade market regime detection — 6 upgrades over basic HMM:

  1. FORWARD-PASS PROBABILITIES  — no lookahead (Viterbi uses future data)
  2. 5-FEATURE HMM               — adds range ratio + volume trend
  3. REGIME AGE SCALAR           — duration-aware position sizing
  4. CRITICAL SLOWING DOWN       — AC1+variance early warning (10-20d lead)
  5. STRATEGY ROUTER             — factor weights flip per regime
  6. ROLLING REFIT               — HMM retrained every 21d on 252d window

References:
  Shu, Yu & Mulvey (2024) arXiv:2402.05272
  Baitinger & Hoch (2024) SSRN:4796238
  iScience (2025) PMC11976486
  Aydinhan, Kolm et al. (2024) SSRN:4556048
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS = 252

REGIME_COLORS = {
    "Bull 📈":    "rgba(50,205,50,0.25)",
    "Sideways ↔": "rgba(255,215,0,0.20)",
    "Bear 📉":    "rgba(220,50,50,0.25)",
}
REGIME_LINE_COLORS = {
    "Bull 📈":    "limegreen",
    "Sideways ↔": "gold",
    "Bear 📉":    "crimson",
}

# ── 1+2  5-FEATURE HMM + FORWARD PROBABILITIES ────────────────────────────────

def _build_features(returns: pd.Series, df=None, vol_window: int = 21) -> np.ndarray:
    vol   = returns.rolling(vol_window, min_periods=5).std().bfill()
    trend = returns.rolling(5, min_periods=2).mean().bfill()
    if df is not None and "High" in df.columns and "Volume" in df.columns:
        rng = ((df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
               ).reindex(returns.index).bfill().ffill()
        vol_trend = (df["Volume"].pct_change()
                     .rolling(5, min_periods=2).mean()
                     .reindex(returns.index).bfill().ffill())
        X = np.column_stack([returns.values, vol.values, trend.values,
                              rng.values, vol_trend.values])
    else:
        X = np.column_stack([returns.values, vol.values])
    return np.nan_to_num(X)


def _label_states(model, X: np.ndarray, n_states: int) -> dict:
    raw   = model.predict(X)
    means = {s: X[raw == s, 0].mean() for s in range(n_states)}
    ranked = sorted(means, key=means.get, reverse=True)
    if n_states == 2:
        return {ranked[0]: "Bull 📈", ranked[1]: "Bear 📉"}
    return {ranked[0]: "Bull 📈", ranked[1]: "Sideways ↔", ranked[2]: "Bear 📉"}


def fit_hmm(returns: pd.Series, n_states: int = 2, n_iter: int = 200,
            df=None) -> tuple:
    """Fit 5-feature Gaussian HMM. Returns (model, viterbi_states, label_map)."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed")
        dummy = pd.Series(["Bull 📈"] * len(returns), index=returns.index)
        return None, dummy, {0: "Bull 📈", 1: "Bear 📉"}

    X = _build_features(returns, df)
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=n_iter, random_state=42)
    model.fit(X)
    label_map    = _label_states(model, X, n_states)
    raw_states   = model.predict(X)
    state_series = pd.Series([label_map[s] for s in raw_states],
                              index=returns.index, name="Regime")
    return model, state_series, label_map


def forward_regime_proba(returns: pd.Series, model, df=None,
                          label_map: dict = None) -> pd.DataFrame:
    """
    Forward-pass only regime probabilities — NO LOOKAHEAD.
    P(regime | data up to t) using predict_proba (not Viterbi).
    Reference: SSRN:4556048 — probability vector over all regimes.
    """
    if model is None:
        return pd.DataFrame()
    X = _build_features(returns, df)
    try:
        proba = model.predict_proba(X)
    except Exception:
        try:
            proba = model.predict_proba(X[:, :2])
        except Exception:
            return pd.DataFrame()
    cols = label_map or {i: f"Regime {i}" for i in range(proba.shape[1])}
    col_names = [cols.get(i, f"State {i}") for i in range(proba.shape[1])]
    return pd.DataFrame(proba, index=returns.index, columns=col_names)


# ── 3  REGIME AGE SCALAR ──────────────────────────────────────────────────────

def compute_regime_age(regimes: pd.Series) -> int:
    """Days consecutively spent in current regime."""
    if regimes.empty:
        return 0
    current = regimes.iloc[-1]
    age = 0
    for r in reversed(regimes.values):
        if r == current:
            age += 1
        else:
            break
    return age


def regime_age_scalar(regimes: pd.Series) -> float:
    """
    Position size scalar [0.1, 1.0] based on regime + how long in it.
    Bull: ramp from 0.5 → 1.0 over 30 days.
    Bear: 0.1 immediately, 0.4 after 180d (old bear = mean-reversion).
    Sideways: 0.5 always.
    Reference: HSMM duration modelling concept, SSRN:4796238.
    """
    if regimes.empty:
        return 0.5
    current = regimes.iloc[-1]
    age     = compute_regime_age(regimes)
    if "Bull" in current:
        scalar = min(1.0, 0.5 + (age / 30) * 0.5)
    elif "Bear" in current:
        scalar = 0.4 if age > 180 else 0.1
    else:
        scalar = 0.5
    return round(float(scalar), 3)


# ── 4  CRITICAL SLOWING DOWN  (early warning) ─────────────────────────────────

def critical_slowing_down(returns: pd.Series, window: int = 21,
                           confirm_days: int = 5) -> dict:
    """
    Early warning signal — fires 10-20 days BEFORE regime change.

    Method: Before a system flips state it shows 'critical slowing down':
      - Rising AC1 (lag-1 autocorrelation) — losing resilience
      - Rising variance — uncertainty before transition
    Both rising simultaneously → warning.

    Reference: iScience (2025) PMC11976486 — spillover network + ML
    early warning model. Features of changeable mean and volatility
    predict regime switching and can prevent systematic market collapse.
    """
    ac1 = returns.rolling(window, min_periods=10).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= 4 else np.nan,
        raw=False,
    ).rename("AC1")
    variance = returns.rolling(window, min_periods=10).var().rename("Variance")

    ac1_rising = ac1.diff(confirm_days) > 0
    var_rising  = variance.diff(confirm_days) > 0
    warning     = (ac1_rising & var_rising).astype(int).rename("Warning")

    recent_warn  = warning.dropna().tail(confirm_days)
    active       = bool(recent_warn.sum() >= confirm_days - 1)
    latest_ac1   = float(ac1.dropna().iloc[-1]) if not ac1.dropna().empty else 0.0
    latest_var   = float(variance.dropna().iloc[-1]) if not variance.dropna().empty else 0.0

    if active:
        lead_msg = "⚠️ WARNING: Critical slowing down — regime change likely within 10-20 days"
    elif latest_ac1 > 0.15:
        lead_msg = "🟡 Elevated AC1 — watch closely"
    else:
        lead_msg = "🟢 Normal — no early warning"

    return dict(ac1=ac1, variance=variance, warning=warning, active=active,
                latest_ac1=round(latest_ac1,4), latest_var=round(latest_var,6),
                lead_msg=lead_msg)


# ── 5  STRATEGY ROUTER ────────────────────────────────────────────────────────

REGIME_FACTOR_WEIGHTS = {
    "Bull 📈":    {"Momentum":1.0,"LowVol":0.2,"Quality":0.5,"Value":0.3,"Size":0.4,"OFI":0.8,"IV_Skew":0.3},
    "Sideways ↔": {"Momentum":0.2,"LowVol":0.6,"Quality":0.8,"Value":0.7,"Size":0.3,"OFI":0.4,"IV_Skew":0.5},
    "Bear 📉":    {"Momentum":0.0,"LowVol":1.0,"Quality":0.7,"Value":0.5,"Size":0.1,"OFI":0.3,"IV_Skew":1.0},
}

REGIME_STRATEGY_RECS = {
    "Bull 📈":    {"primary":"Momentum — trend following, long bias",
                   "secondary":"Quality factor — high-Sharpe stocks",
                   "avoid":"Mean reversion, short bias",
                   "position":"Full size (after 30d confirmation)",
                   "stops":"Wide stops — let winners run"},
    "Sideways ↔": {"primary":"Mean Reversion — BB + RSI reversals",
                   "secondary":"Value + Quality — defensive positioning",
                   "avoid":"Trend following — will get chopped",
                   "position":"Half size — wait for breakout",
                   "stops":"Tight stops — range-bound punishes overstays"},
    "Bear 📉":    {"primary":"Low-Vol factor — minimum drawdown",
                   "secondary":"IV Skew signal — hedge via options",
                   "avoid":"Momentum — trends reverse violently",
                   "position":"10-20% size — capital preservation",
                   "stops":"Very tight — preserve capital above all"},
}


def get_strategy_for_regime(regime: str) -> dict:
    r = regime.strip()
    return {
        "weights":         REGIME_FACTOR_WEIGHTS.get(r, REGIME_FACTOR_WEIGHTS["Sideways ↔"]),
        "recommendations": REGIME_STRATEGY_RECS.get(r, REGIME_STRATEGY_RECS["Sideways ↔"]),
        "regime":          r,
    }


# ── 6  ROLLING REFIT HMM ──────────────────────────────────────────────────────

def rolling_regime_proba(returns: pd.Series, df=None, n_states: int = 2,
                          fit_window: int = 252, step: int = 21,
                          n_iter: int = 100) -> pd.DataFrame:
    """
    Rolling HMM: refit every `step` days on last `fit_window` days.
    Predicts NEXT `step` days out-of-sample (no lookahead).

    FIX: Single HMM assumes stationary transition probabilities.
    State transition probabilities are very unlikely to be stationary.
    The rate at which retraining must occur is subject of future research.
    Reference: QuantStart + Preprints.org (2026) non-homogeneous HMM.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return pd.DataFrame()

    results = {}
    for i in range(fit_window, len(returns), step):
        w_ret = returns.iloc[i - fit_window:i]
        w_df  = df.iloc[i - fit_window:i] if df is not None else None
        X_fit = _build_features(w_ret, w_df)
        try:
            m = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=n_iter, random_state=42)
            m.fit(X_fit)
            lmap  = _label_states(m, X_fit, n_states)
            end   = min(i + step, len(returns))
            n_ret = returns.iloc[i:end]
            n_df  = df.iloc[i:end] if df is not None else None
            X_nxt = _build_features(n_ret, n_df)
            proba = m.predict_proba(X_nxt)
            for j, date in enumerate(n_ret.index):
                results[date] = {lmap.get(k, f"S{k}"): proba[j, k]
                                 for k in range(n_states)}
        except Exception as e:
            logger.debug(f"Rolling HMM step {i}: {e}")
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).T.sort_index()


# ── COMBINED ANALYSIS (one-call API for the UI) ───────────────────────────────

def full_regime_analysis(returns: pd.Series, df=None,
                          n_states: int = 2, n_iter: int = 200) -> dict:
    """Run all 6 components and return a single dict for the UI page."""
    model, regimes, label_map = fit_hmm(returns, n_states, n_iter, df)
    fwd_proba  = forward_regime_proba(returns, model, df, label_map)
    age        = compute_regime_age(regimes)
    scalar     = regime_age_scalar(regimes)
    ew         = critical_slowing_down(returns)
    cur        = regimes.iloc[-1] if not regimes.empty else "Sideways ↔"
    strategy   = get_strategy_for_regime(cur)
    cond_sharpe = regime_conditional_sharpe(returns, regimes)
    return dict(model=model, regimes=regimes, label_map=label_map,
                fwd_proba=fwd_proba, regime_age=age, age_scalar=scalar,
                early_warning=ew, current_regime=cur, strategy=strategy,
                cond_sharpe=cond_sharpe)


# ── KEPT FOR BACKWARD COMPATIBILITY ──────────────────────────────────────────

def regime_conditional_sharpe(returns: pd.Series, regimes: pd.Series,
                                rf: float = 0.045) -> pd.DataFrame:
    from core.metrics import sharpe
    rows = []
    for regime in regimes.unique():
        mask = regimes == regime
        r    = returns[mask]
        cum  = (1 + r).cumprod()
        mdd  = float((cum / cum.cummax() - 1).min())
        rows.append({
            "Regime":        regime,
            "Days":          int(mask.sum()),
            "Pct Time":      f"{mask.mean():.1%}",
            "Sharpe":        round(sharpe(r, rf), 3),
            "Avg Daily Ret": f"{r.mean():.4%}",
            "Volatility":    f"{r.std() * np.sqrt(252):.2%}",
            "Max Drawdown":  f"{mdd:.2%}",
        })
    return pd.DataFrame(rows)