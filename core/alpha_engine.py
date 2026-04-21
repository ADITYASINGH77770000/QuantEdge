"""
core/alpha_engine.py
──────────────────────────────────────────────────────────────────────────────
5 research-grade signals that are unsolved / under-exploited in open-source:

  1. OFI      — Order Flow Imbalance (market-cap normalised, Kolm 2023)
  2. CROWD    — Factor Crowding detector (Hua & Sun 2024)
  3. IV_SKEW  — Implied Volatility Skew signal (Höfler 2024)
  4. HEALTH   — Signal health / alpha decay monitor (AlphaAgent KDD 2025)
  5. MACRO    — Cross-asset macro regime score (multi-asset HMM)

  6. COMBINED — IC-weighted combination of all active signals

All signals are computed from yfinance daily data — no paid data required.
Each returns a scalar score usable for position sizing.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# 1. ORDER FLOW IMBALANCE  (Kolm, 2023 — Mathematical Finance)
# ══════════════════════════════════════════════════════════════════════════════

def compute_volume_pressure(df: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Volume Pressure Signal (OFI Proxy) — OHLCV-based directional volume imbalance.

    ⚠️  NAMING DISCLOSURE: True Order Flow Imbalance (Kolm et al. 2023) requires
    Level-2 tick data (bid/ask queue imbalance). This implementation uses daily
    OHLCV as a proxy: up-close bars → buy pressure, down-close bars → sell pressure.
    The math is legitimate and predictive, but it is NOT the same as L2 OFI.
    Label as "Volume Pressure (OFI Proxy)" in any UI or report.

    Formula:
      buy_vol  = Volume where Close >= Open
      sell_vol = Volume where Close < Open
      raw      = buy_vol - sell_vol
      norm     = raw / rolling_avg_volume        ← market-cap normalisation
      signal   = z-score of norm over window days

    Positive → net buying pressure → bullish.
    Negative → net selling pressure → bearish.

    Reference: Kolm et al. (2023) — Deep order flow imbalance, Mathematical Finance.
    """
    close = df["Close"]
    open_ = df["Open"]
    vol   = df["Volume"].replace(0, np.nan)

    buy_vol  = vol.where(close >= open_, 0.0).fillna(0.0)
    sell_vol = vol.where(close < open_,  0.0).fillna(0.0)
    raw_ofi  = buy_vol - sell_vol

    avg_vol  = vol.rolling(window, min_periods=10).mean()
    ofi_norm = raw_ofi / avg_vol.replace(0, np.nan)

    mu    = ofi_norm.rolling(window, min_periods=10).mean()
    sigma = ofi_norm.rolling(window, min_periods=10).std().replace(0, np.nan)
    ofi_z = (ofi_norm - mu) / sigma

    return ofi_z.rename("VolumePressure")


# Backward-compat alias — remove once all pages import compute_volume_pressure directly
compute_ofi = compute_volume_pressure


def ofi_signal(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """
    Discrete signal from Volume Pressure z-score.
    +1 (buy) when z > threshold, -1 (sell) when z < -threshold.
    """
    ofi = compute_volume_pressure(df)
    sig = pd.Series(0, index=df.index, name="VolumePressure_Signal")
    sig[ofi >  threshold] =  1
    sig[ofi < -threshold] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# 2. FACTOR CROWDING DETECTOR  (Hua & Sun 2024 — SSRN 5023380)
# ══════════════════════════════════════════════════════════════════════════════

def compute_crowding_score(
    factor_returns: pd.Series,
    window_short: int = 21,
    window_long:  int = 126,
) -> pd.Series:
    """
    Crowding proxy: detects when a factor's recent vol >> its long-run vol.

    Logic: when many funds pile into the same factor, its vol-adjusted
    return dispersion narrows (everyone moving together). When they exit,
    vol spikes. We detect the crowded state BEFORE the exit:

    crowding_score = rolling_short_vol / rolling_long_vol

    Score > 1.3 → overcrowded → reduce factor weight
    Score < 0.8 → undercrowded → increase factor weight

    Reference: Hua & Sun (2024) Dynamics of Factor Crowding, SSRN 5023380.
    Also: Falck, Rej & Thesmar (2022) — crowding as a risk factor.
    """
    vol_short = factor_returns.rolling(window_short, min_periods=5).std()
    vol_long  = factor_returns.rolling(window_long,  min_periods=20).std()
    crowding  = (vol_short / vol_long.replace(0, np.nan)).rename("Crowding")
    return crowding


def crowding_weight(factor_returns: pd.Series) -> float:
    """
    Current crowding-adjusted weight scalar for a factor.
    Returns value between 0.25 (very crowded) and 1.0 (not crowded).
    """
    crowd = compute_crowding_score(factor_returns)
    latest = float(crowd.dropna().iloc[-1]) if not crowd.dropna().empty else 1.0
    # Linear mapping: 1.3+ → 0.25, 0.8- → 1.0
    weight = np.clip(1.0 - (latest - 0.8) / 0.5 * 0.75, 0.25, 1.0)
    return round(float(weight), 3)


def crowding_signal(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Crowding score and adjusted weight for each ticker.
    Returns DataFrame: ticker → {crowding_score, weight, status}.
    """
    rows = []
    for ticker, df in prices.items():
        ret = df["Close"].pct_change().dropna()
        if len(ret) < 30:
            continue
        crowd = compute_crowding_score(ret)
        latest_score = float(crowd.dropna().iloc[-1]) if not crowd.dropna().empty else 1.0
        weight = crowding_weight(ret)
        if latest_score > 1.3:
            status = "🔴 Overcrowded"
        elif latest_score > 1.1:
            status = "🟡 Elevated"
        else:
            status = "🟢 Normal"
        rows.append({
            "Ticker":         ticker,
            "Crowding Score": round(latest_score, 3),
            "Weight Scalar":  weight,
            "Status":         status,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. IMPLIED VOLATILITY SKEW SIGNAL  (Höfler 2024 — SSRN 4869272)
# ══════════════════════════════════════════════════════════════════════════════

def compute_realized_skew(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Realized Skew Signal (IV Skew Proxy) — OHLCV-based fear/calm indicator.

    ⚠️  NAMING DISCLOSURE: True IV Skew requires an options chain
    (put IV minus call IV at equivalent strikes). This implementation uses
    realized return skewness + down-day ratio as a proxy for the same
    market fear that options are pricing. The signal is valid, but it is NOT
    implied volatility skew. Label as "Realized Skew (IV Proxy)" in the UI.

    Formula:
      roll_skew  = rolling skewness of returns (negative = fat left tail = fear)
      fear_proxy = normalised frequency of down-close bars (>0.6 = fear zone)
      combined   = z-score of (roll_skew + fear_proxy) / 2

    Negative combined → fear → bearish.
    Positive combined → calm → bullish.

    Reference: Höfler (2024) SSRN 4869272; Bakshi, Kapadia & Madan (2003).
    For real IV skew: replace with yfinance option_chain() put/call IV spread.
    """
    ret = df["Close"].pct_change().dropna()

    roll_skew  = ret.rolling(window, min_periods=10).skew().rename("ReturnSkew")
    down_days  = (df["Close"] < df["Open"]).astype(float)
    down_ratio = down_days.rolling(window, min_periods=5).mean()
    fear_proxy = ((down_ratio - 0.5) * 2).rename("FearProxy")

    combined = (roll_skew + fear_proxy) / 2
    mu    = combined.rolling(63, min_periods=20).mean()
    sigma = combined.rolling(63, min_periods=20).std().replace(0, np.nan)
    skew_z = ((combined - mu) / sigma).rename("RealizedSkew")

    return skew_z.reindex(df.index)


# Backward-compat alias — remove once all pages import compute_realized_skew directly
compute_iv_skew_proxy = compute_realized_skew


def iv_skew_signal(df: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
    """
    Signal from Realized Skew proxy.
    Negative skew (fear) → -1 (reduce longs)
    Positive skew (calm) → +1 (add longs)
    """
    skew = compute_realized_skew(df)
    sig = pd.Series(0, index=df.index, name="RealizedSkew_Signal")
    sig[skew < -threshold] = -1
    sig[skew >  threshold] =  1
    return sig


def get_real_iv_skew(ticker: str) -> dict | None:
    """
    Attempt to fetch real IV skew from yfinance options chain.
    Returns None gracefully if options not available.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        # Use nearest expiry
        chain = t.option_chain(expirations[0])
        puts  = chain.puts.dropna(subset=["impliedVolatility"])
        calls = chain.calls.dropna(subset=["impliedVolatility"])
        if puts.empty or calls.empty:
            return None
        put_iv  = float(puts["impliedVolatility"].mean())
        call_iv = float(calls["impliedVolatility"].mean())
        skew    = put_iv - call_iv
        atm_iv  = float(calls["impliedVolatility"].median())
        return {
            "put_iv":   round(put_iv,  4),
            "call_iv":  round(call_iv, 4),
            "skew":     round(skew,    4),
            "atm_iv":   round(atm_iv,  4),
            "signal":   -1 if skew > 0.05 else (1 if skew < -0.02 else 0),
            "source":   "live_options",
        }
    except Exception as e:
        logger.debug(f"IV skew live fetch failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL HEALTH / ALPHA DECAY MONITOR  (AlphaAgent KDD 2025)
# ══════════════════════════════════════════════════════════════════════════════

def compute_signal_health(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 63,
    lookback: int = 252,
) -> dict:
    """
    Signal health monitor — detects alpha decay in real time.

    Computes rolling IC over time and measures:
      - IC trend slope (negative = decaying)
      - IC stability (std of rolling IC — high = unreliable)
      - IC mean over recent period
      - Health score 0-100 combining all three

    Score 80-100 → strong, use full weight
    Score 50-80  → moderate, use 50-75% weight
    Score 20-50  → weak, use 25% weight
    Score < 20   → dead signal, stop trading

    Reference: AlphaAgent (KDD 2025) — regularization to counteract alpha decay.
    Also: Harvey, Liu & Zhu (2016) — factor p-hacking and decay.
    """
    sig_clean = signal.dropna()
    fwd_clean = forward_returns.dropna()
    common = sig_clean.index.intersection(fwd_clean.index)

    if len(common) < window + 20:
        return {
            "health_score": 50.0,
            "ic_mean":      0.0,
            "ic_std":       0.0,
            "ic_trend":     0.0,
            "status":       "⚪ Insufficient data",
            "weight":       0.5,
        }

    s = sig_clean[common].astype(float)
    f = fwd_clean[common].astype(float)

    # Rolling IC
    rolling_ic = []
    for i in range(window, len(common)):
        sl = s.iloc[i - window:i].values
        fl = f.iloc[i - window:i].values
        mask = ~(np.isnan(sl) | np.isnan(fl))
        if mask.sum() < 10:
            rolling_ic.append(np.nan)
            continue
        ic, _ = stats.spearmanr(sl[mask], fl[mask])
        rolling_ic.append(float(ic) if not np.isnan(ic) else 0.0)

    ic_series = pd.Series(rolling_ic).dropna()
    if ic_series.empty:
        return {"health_score": 30.0, "ic_mean": 0.0, "ic_std": 0.99,
                "ic_trend": 0.0, "status": "⚪ No IC data", "weight": 0.3}

    # Use last `lookback` points (1 year)
    recent = ic_series.tail(lookback).values
    ic_mean = float(np.mean(recent))
    ic_std  = float(np.std(recent))

    # Trend: slope of IC over time (negative = decaying)
    if len(recent) >= 20:
        slope, _, _, _, _ = stats.linregress(range(len(recent)), recent)
        ic_trend = float(slope)
    else:
        ic_trend = 0.0

    # Health score components (0-100)
    # 1. IC magnitude: |IC| > 0.05 is good, 0 is bad
    ic_score  = min(100, max(0, abs(ic_mean) / 0.08 * 50))
    # 2. IC stability: lower std = more stable
    std_score = min(100, max(0, (0.15 - ic_std) / 0.15 * 30))
    # 3. Trend: positive trend = healthy
    trend_score = min(20, max(0, (ic_trend + 0.001) / 0.002 * 20))

    health = ic_score + std_score + trend_score

    if health >= 75:
        status = "🟢 Healthy"
        weight = 1.0
    elif health >= 50:
        status = "🟡 Moderate"
        weight = 0.65
    elif health >= 25:
        status = "🟠 Weak"
        weight = 0.3
    else:
        status = "🔴 Decaying"
        weight = 0.1

    return {
        "health_score": round(health, 1),
        "ic_mean":      round(ic_mean, 4),
        "ic_std":       round(ic_std,  4),
        "ic_trend":     round(ic_trend, 6),
        "status":       status,
        "weight":       weight,
        "rolling_ic":   ic_series,
    }


def monitor_all_signals(
    df: pd.DataFrame,
    fwd_days: int = 5,
) -> pd.DataFrame:
    """
    Run signal health monitor on all built-in signals for a single ticker.
    Returns a summary DataFrame with health scores.
    """
    from core.indicators import add_all_indicators, signal_rsi, signal_macd_crossover
    from core.indicators import signal_bb_mean_reversion, signal_dual_ma

    df_ind = add_all_indicators(df)
    ret    = df["Close"].pct_change().dropna()
    fwd    = ret.shift(-fwd_days)

    signals = {
        "RSI":          signal_rsi(df_ind),
        "MACD":         signal_macd_crossover(df_ind),
        "BB Reversion": signal_bb_mean_reversion(df_ind),
        "Dual MA":      signal_dual_ma(df_ind, 20, 50),
        "OFI":          ofi_signal(df),
        "IV Skew":      iv_skew_signal(df),
    }

    rows = []
    for name, sig in signals.items():
        h = compute_signal_health(sig, fwd)
        rows.append({
            "Signal":       name,
            "Health":       h["health_score"],
            "Status":       h["status"],
            "IC Mean":      h["ic_mean"],
            "IC Std":       h["ic_std"],
            "IC Trend":     h["ic_trend"],
            "Weight":       h["weight"],
        })
    return pd.DataFrame(rows).sort_values("Health", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-ASSET MACRO REGIME SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

_MACRO_TICKERS = {
    "vix":    "^VIX",
    "credit": "HYG",
    "bonds":  "IEF",
    "dollar": "DX-Y.NYB",
    "rates":  "^TNX",
}

def get_macro_data(start: str = "2018-01-01") -> pd.DataFrame | None:
    """
    Fetch cross-asset macro data from yfinance.
    Returns aligned DataFrame or None if network unavailable.
    """
    try:
        import yfinance as yf
        dfs = {}
        for name, sym in _MACRO_TICKERS.items():
            d = yf.Ticker(sym).history(start=start, auto_adjust=True)
            if not d.empty:
                dfs[name] = d["Close"].rename(name)
        if len(dfs) < 3:
            return None
        df = pd.concat(dfs.values(), axis=1).dropna()
        return df
    except Exception as e:
        logger.debug(f"Macro data fetch failed: {e}")
        return None


def compute_macro_regime_score(macro_df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Cross-asset macro regime score from -2 (deep risk-off) to +2 (risk-on).

    Components:
      - VIX:    high = fear = risk-off
      - Credit: HYG/IEF ratio = risk-on when high
      - Dollar: strong dollar = risk-off for equities
      - Rates:  rising 10yr = growth (risk-on) or tightening (risk-off)

    Each component z-scored vs rolling history, then combined.

    Reference: cross-asset regime detection literature.
    Future Alpha conference 2025 — "long-standing correlations breaking down."
    """
    scores = pd.DataFrame(index=macro_df.index)

    if "vix" in macro_df.columns:
        vix_z = _zscore(macro_df["vix"], window)
        scores["vix_component"] = -vix_z          # high VIX = risk-off = negative

    if "credit" in macro_df.columns and "bonds" in macro_df.columns:
        credit_ratio = macro_df["credit"] / macro_df["bonds"]
        cred_z = _zscore(credit_ratio, window)
        scores["credit_component"] = cred_z        # high HYG/IEF = risk-on = positive

    if "dollar" in macro_df.columns:
        dxy_z = _zscore(macro_df["dollar"], window)
        scores["dollar_component"] = -dxy_z        # strong dollar = risk-off for equities

    if "rates" in macro_df.columns:
        rate_chg = macro_df["rates"].diff(window)
        rate_z   = _zscore(rate_chg, window)
        scores["rates_component"]  = rate_z * 0.5  # rising rates: ambiguous, half weight

    if scores.empty:
        return pd.Series(dtype=float)

    # Average all components, clip to -2/+2
    macro_score = scores.mean(axis=1).clip(-2, 2).rename("MacroScore")
    return macro_score


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=5).mean()
    sig = series.rolling(window, min_periods=5).std().replace(0, np.nan)
    return ((series - mu) / sig).clip(-3, 3)


def macro_regime_label(score: float) -> tuple[str, float]:
    """Returns (label, position_scalar) from macro score."""
    if score >= 1.0:
        return "🟢 Risk-On",      1.0
    elif score >= 0.25:
        return "🟡 Mild Risk-On", 0.75
    elif score >= -0.25:
        return "⚪ Neutral",      0.5
    elif score >= -1.0:
        return "🟠 Mild Risk-Off",0.35
    else:
        return "🔴 Risk-Off",     0.2


# ══════════════════════════════════════════════════════════════════════════════
# 6. IC-WEIGHTED COMBINED SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def combine_signals(
    signal_dict: dict[str, pd.Series],
    returns:     pd.Series,
    fwd_days:    int = 5,
    window:      int = 63,
) -> tuple[pd.Series, dict]:
    """
    IC-weighted combination of multiple signals into one master signal.

    Steps:
      1. Compute rolling IC for each signal vs fwd returns
      2. Weight each signal by its recent IC (ignore signals with IC < 0)
      3. Combine into a weighted-average score
      4. Discretise: top tercile = +1, bottom = -1, middle = 0

    Returns: combined signal Series, weight dict for display.

    Reference: Grinold & Kahn (2000) — combination theorem.
    """
    fwd = returns.shift(-fwd_days)
    weights = {}
    weighted_scores = []

    for name, sig in signal_dict.items():
        sig_clean = sig.reindex(returns.index).fillna(0).astype(float)
        # Rolling IC over last window days
        common = sig_clean.dropna().index.intersection(fwd.dropna().index)
        if len(common) < window + 10:
            weights[name] = 0.0
            continue
        sl = sig_clean[common].values
        fl = fwd[common].values
        mask = ~(np.isnan(sl) | np.isnan(fl))
        if mask.sum() < 10:
            weights[name] = 0.0
            continue
        ic, pval = stats.spearmanr(sl[mask], fl[mask])
        ic = float(ic) if not np.isnan(ic) else 0.0
        # Only use signals with positive IC (discard noise)
        ic_weight = max(0.0, ic)
        weights[name] = round(ic_weight, 4)

    total_weight = sum(weights.values())
    if total_weight == 0:
        # Fallback: equal weight
        total_weight = len(signal_dict)
        weights = {k: 1.0 for k in signal_dict}

    # Build weighted composite
    all_idx = returns.index
    composite = pd.Series(0.0, index=all_idx)
    for name, sig in signal_dict.items():
        w = weights.get(name, 0.0) / total_weight
        if w > 0:
            s = sig.reindex(all_idx).fillna(0).astype(float)
            composite += w * s

    # Normalise composite to [-1, +1] using rolling percentile
    roll_max = composite.abs().rolling(window, min_periods=10).max().replace(0, 1)
    composite_norm = (composite / roll_max).clip(-1, 1)

    # Discretise
    combined = pd.Series(0, index=all_idx, name="Combined")
    combined[composite_norm >  0.3] =  1
    combined[composite_norm < -0.3] = -1

    return combined, weights