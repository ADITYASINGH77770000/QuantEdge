"""
app/pages/08_portfolio.py — QuantEdge Portfolio Optimizer (Quant Grade)
═══════════════════════════════════════════════════════════════════════
All original portfolio logic preserved exactly.

BUG FIX — ValueError: 'covars' must be symmetric, positive-definite
  Root cause: _build_features() in regime_detector.py produces near-constant or
  duplicate columns when volume is synthetic/flat (e.g. demo data, low-volume stocks,
  or short series). hmmlearn's GaussianHMM("full") calls np.cov() on the raw feature
  matrix — if any column has std ≈ 0, the covariance matrix is rank-deficient
  (eigenvalue = 0) and therefore NOT positive-definite. hmmlearn correctly rejects it.

  Three-part fix applied inside _safe_fit_hmm():
    1. Drop columns with std < 1e-8 (constant/near-constant features)
    2. Standardise remaining columns (zero-mean, unit-variance) so no column
       dominates and numerical conditioning is improved
    3. Fallback chain: "full" → "diag" → rule-based regime
       "diag" covariance only needs per-state per-feature variance (always PD)

  The fix lives entirely inside _safe_fit_hmm() and does NOT touch regime_detector.py,
  so no other page is affected. detect_market_regime() now calls _safe_fit_hmm()
  instead of fit_hmm() directly.

AI LAYER — Gemini AI Portfolio Decoder (bottom of page, same 3-layer design as signals):
  Layer 1: Deterministic danger flags (always shown, no AI)
  Layer 2: Context builder + "Decode for Me" button
  Layer 3: Gemini output — structured 4-section explanation with mandatory disclaimer

  Uses GEMINI_API_KEY + GEMINI_MODEL from utils/config.py (already present in your .env).
  Falls back to deterministic explanation if key missing or API call fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings
from urllib import error as urlerror
from urllib import request as urlrequest

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st
from scipy.optimize import minimize

from core.data import align_returns, get_multi_ohlcv
from core.regime_detector import fit_hmm
from core.metrics import sharpe as calc_sharpe, max_drawdown, cagr
from app.data_engine import (
    render_data_engine_controls,
    render_multi_ticker_input,
    load_multi_ticker_data,
    get_global_start_date,
)
from utils.config import cfg

try:
    from utils.theme import qe_neon_divider, qe_faq_section
except ImportError:
    from utils.theme import qe_neon_divider

    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        qe_neon_divider()
        st.markdown(f"### {title}")
        for question, answer in faqs:
            st.markdown(
                f"""
                <div style="
                    background: rgba(14,22,42,0.82);
                    border: 1px solid rgba(11,224,255,0.18);
                    border-radius: 12px;
                    padding: 14px 16px;
                    margin: 10px 0;
                ">
                  <div style="font-weight:700;color:#e8f4fd;margin-bottom:6px;">Q. {question}</div>
                  <div style="color:var(--text-dim);line-height:1.55;">A. {answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX — SAFE HMM WRAPPER (positive-definite covariance guaranteed)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_hmm_features(returns: pd.Series, df=None, vol_window: int = 21) -> np.ndarray:
    """
    Builds the HMM feature matrix WITH positive-definiteness safeguards.

    Three-part fix:
      1. Drop near-constant columns (std < 1e-8) — they add zero rank to the
         covariance matrix and make it singular.
      2. Standardise every column to zero-mean, unit-variance — improves numerical
         conditioning so hmmlearn's Cholesky factorisation succeeds.
      3. Caller (_safe_fit_hmm) tries covariance_type="full" first, falls back to
         "diag" if full still fails (diag only needs per-feature variances, always PD).
    """
    vol   = returns.rolling(vol_window, min_periods=5).std().bfill()
    trend = returns.rolling(5, min_periods=2).mean().bfill()

    if df is not None and "High" in df.columns and "Volume" in df.columns:
        rng_ratio = (
            (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
        ).reindex(returns.index).bfill().ffill()
        vol_trend = (
            df["Volume"].pct_change()
            .rolling(5, min_periods=2).mean()
            .reindex(returns.index).bfill().ffill()
        )
        X = np.column_stack([
            returns.values, vol.values, trend.values,
            rng_ratio.values, vol_trend.values,
        ])
    else:
        X = np.column_stack([returns.values, vol.values])

    # Replace NaN / Inf introduced by any rolling operation
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── FIX 1: drop near-constant columns ────────────────────────────────────
    col_stds = X.std(axis=0)
    keep = col_stds >= 1e-8
    if keep.sum() < 2:
        # Always keep at least returns (col 0) and vol (col 1)
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[0] = True
        keep[min(1, X.shape[1] - 1)] = True
    X = X[:, keep]

    # ── FIX 2: standardise — zero-mean, unit-variance ────────────────────────
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds < 1e-12] = 1.0          # avoid divide-by-zero on residual constants
    X = (X - means) / stds

    return X


def _label_states_safe(model, X: np.ndarray, n_states: int) -> dict:
    """Label HMM states Bull/Sideways/Bear by mean return of state members."""
    try:
        raw   = model.predict(X)
        means = {s: X[raw == s, 0].mean() for s in range(n_states) if (raw == s).sum() > 0}
        ranked = sorted(means, key=means.get, reverse=True)
        if n_states == 2:
            return {ranked[0]: "Bull 📈", ranked[-1]: "Bear 📉"}
        return {ranked[0]: "Bull 📈", ranked[1]: "Sideways ↔", ranked[-1]: "Bear 📉"}
    except Exception:
        return {i: ["Bull 📈", "Sideways ↔", "Bear 📉"][min(i, 2)] for i in range(n_states)}


def _safe_fit_hmm(
    returns: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    df=None,
) -> tuple:
    """
    Positive-definite–safe HMM fit with three-tier fallback:
      Tier 1: GaussianHMM(covariance_type="full")  — best, richest model
      Tier 2: GaussianHMM(covariance_type="diag")  — always PD, less expressive
      Tier 3: Rule-based regime (_fallback_regime)  — no ML, always works

    Returns (model | None, state_series, label_map).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        dummy = pd.Series(["Bull 📈"] * len(returns), index=returns.index)
        return None, dummy, {0: "Bull 📈", 1: "Bear 📉"}

    X = _prepare_hmm_features(returns, df)

    for cov_type in ("full", "diag"):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                random_state=42,
            )
            model.fit(X)
            label_map  = _label_states_safe(model, X, n_states)
            raw_states = model.predict(X)
            state_series = pd.Series(
                [label_map.get(s, "Sideways ↔") for s in raw_states],
                index=returns.index,
                name="Regime",
            )
            return model, state_series, label_map
        except Exception:
            continue  # try next fallback

    # Tier 3: rule-based
    current, _ = _fallback_regime(returns)
    state_series = pd.Series([current] * len(returns), index=returns.index, name="Regime")
    return None, state_series, {0: current}


# ── 1. LEDOIT-WOLF SHRINKAGE COVARIANCE (original, unchanged) ─────────────────

def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf optimal shrinkage — replaces unstable raw .cov()"""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        return lw.covariance_ * TRADING_DAYS
    except Exception:
        return returns.cov().values * TRADING_DAYS


def covariance_health(returns: pd.DataFrame) -> dict:
    """T/N ratio signal: tells user whether weights are trustworthy."""
    T, N = returns.shape
    ratio = T / N
    if ratio >= 5:
        status, color = "Good", "green"
        msg = f"{T} days / {N} assets = {ratio:.1f}x — weights are trustworthy"
    elif ratio >= 2:
        status, color = "Warning", "orange"
        msg = f"{T} days / {N} assets = {ratio:.1f}x — weights may be noisy"
    else:
        status, color = "Danger", "red"
        msg = f"{T} days / {N} assets = {ratio:.1f}x — not enough data, weights unreliable"
    return {"status": status, "color": color, "msg": msg, "ratio": round(ratio, 2)}


# ── 2. ANALYTIC CONVEX-OPTIMISED FRONTIER (original, unchanged) ───────────────

def analytic_max_sharpe(mu, cov, rf=0.045, max_weight=0.40):
    n = len(mu)
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - rf) / vol if vol > 1e-10 else 0.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, max_weight)] * n
    rng = np.random.default_rng(42)
    best_w, best_sh = np.ones(n) / n, -np.inf
    for _ in range(10):
        res = minimize(neg_sharpe, rng.dirichlet(np.ones(n)), method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success and -res.fun > best_sh:
            best_sh = -res.fun
            best_w = res.x
    best_w = np.clip(best_w, 0, None)
    return best_w / best_w.sum()


def analytic_min_vol(mu, cov, max_weight=0.40):
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, max_weight)] * n
    res = minimize(lambda w: float(np.sqrt(w @ cov @ w)),
                   np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.clip(res.x if res.success else np.ones(n) / n, 0, None)
    return w / w.sum()


def risk_parity_weights_lw(cov):
    n = cov.shape[0]
    def objective(w):
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / sigma
        rc = w * mrc
        target = np.full(n, 1.0 / n)
        return float(np.sum((rc / rc.sum() - target) ** 2))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1.0)] * n
    res = minimize(objective, np.ones(n) / n, bounds=bounds,
                   constraints=constraints, method="SLSQP")
    w = np.clip(res.x if res.success else np.ones(n) / n, 0, None)
    return w / w.sum()


def build_analytic_frontier(mu, cov, rf=0.045, n_points=50):
    n = len(mu)
    min_ret = float(mu.min()) * 1.05
    max_ret = float(mu.max()) * 0.95
    frontier_vols, frontier_rets = [], []
    for target in np.linspace(min_ret, max_ret, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        res = minimize(lambda w: float(np.sqrt(w @ cov @ w)),
                       np.ones(n) / n, method="SLSQP",
                       bounds=[(0.0, 0.5)] * n, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            w = np.clip(res.x, 0, None); w /= w.sum()
            frontier_vols.append(float(np.sqrt(w @ cov @ w)))
            frontier_rets.append(float(w @ mu))
    return {"vols": np.array(frontier_vols), "rets": np.array(frontier_rets)}


# ── 3. REGIME DETECTION (fixed to use _safe_fit_hmm) ─────────────────────────

def _fallback_regime(eq_weighted: pd.Series) -> tuple:
    """
    Rule-based regime when HMM fails (too little data or numerical issues).
    Uses 63-day vs 252-day return + realised vol to classify.
    """
    if len(eq_weighted) < 63:
        return "Sideways ↔", {}
    ret_63  = float(eq_weighted.tail(63).sum())
    ret_252 = float(eq_weighted.tail(min(252, len(eq_weighted))).sum())
    vol_21  = float(eq_weighted.tail(21).std() * np.sqrt(252))
    vol_63  = float(eq_weighted.tail(63).std() * np.sqrt(252))
    if ret_63 > 0.03 and ret_252 > 0.05 and vol_21 < vol_63 * 1.3:
        regime = "Bull 📈"
    elif ret_63 < -0.03 or vol_21 > vol_63 * 1.5:
        regime = "Bear 📉"
    else:
        regime = "Sideways ↔"
    pct = {regime: 100.0}
    return regime, pct


def detect_market_regime(returns_df):
    eq_weighted = returns_df.mean(axis=1)
    state_series = None
    regime_pct = {}

    try:
        if len(eq_weighted) >= 120:
            # ← FIXED: use _safe_fit_hmm instead of fit_hmm
            _, state_series, _ = _safe_fit_hmm(eq_weighted, n_states=3)
            current_regime = state_series.iloc[-1]
            regime_pct = (
                state_series.value_counts() / len(state_series) * 100
            ).round(1).to_dict()
        else:
            raise ValueError("Too short for HMM")
    except Exception:
        try:
            current_regime, regime_pct = _fallback_regime(eq_weighted)
        except Exception:
            current_regime = "Sideways ↔"
            regime_pct = {}

    if "Bull" in str(current_regime):
        strategy, reason, color = "Max Sharpe", "Bull market — maximize risk-adjusted returns", "#1D9E75"
    elif "Bear" in str(current_regime):
        strategy, reason, color = "Min Variance", "Bear market — protect capital, minimize drawdown", "#E24B4A"
    else:
        strategy, reason, color = "Risk Parity", "Sideways market — spread risk equally", "#EF9F27"

    return {
        "current": current_regime, "strategy": strategy,
        "reason": reason, "color": color,
        "regime_pct": regime_pct, "series": state_series,
    }


# ── 4. NET-OF-COST SHARPE (original, unchanged) ───────────────────────────────

def net_of_cost_sharpe(weights, prev_weights, returns, cost_bps=10.0,
                        rebal_freq_days=21, rf=0.045):
    port_series = pd.Series(returns.values @ weights, index=returns.index)
    gross_sh = calc_sharpe(port_series, rf)
    turnover = float(np.sum(np.abs(weights - prev_weights)))
    annual_cost = turnover * (TRADING_DAYS / rebal_freq_days) * (cost_bps / 10000)
    net_series = port_series - annual_cost / TRADING_DAYS
    net_sh = calc_sharpe(net_series, rf)
    return {
        "gross_sharpe": round(gross_sh, 3),
        "net_sharpe": round(net_sh, 3),
        "annual_cost_pct": round(annual_cost * 100, 3),
        "turnover_pct": round(turnover * 100, 1),
        "sharpe_drag": round(gross_sh - net_sh, 3),
    }


# ── 5. CONCENTRATION SIGNAL (original, unchanged) ────────────────────────────

def concentration_signal(weights, tickers):
    hhi = float(np.sum(weights ** 2))
    n = len(weights)
    norm_hhi = (hhi - 1.0 / n) / (1 - 1.0 / n) * 100 if n > 1 else 100.0
    max_w = float(weights.max())
    top2 = float(np.sort(weights)[-2:].sum()) if n >= 2 else max_w
    top_ticker = tickers[int(np.argmax(weights))]
    if norm_hhi < 30:
        status, color = "Well Diversified", "green"
    elif norm_hhi < 60:
        status, color = "Moderate Concentration", "orange"
    else:
        status, color = "Highly Concentrated", "red"
    return {
        "hhi": round(hhi, 4), "norm_hhi": round(norm_hhi, 1),
        "status": status, "color": color,
        "max_weight": round(max_w * 100, 1),
        "top_ticker": top_ticker, "top2_pct": round(top2 * 100, 1),
    }


# ── HELPERS (original, unchanged) ─────────────────────────────────────────────

def portfolio_stats_full(weights, returns, mu, cov, rf=0.045):
    port_ret = float(weights @ mu)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    port_sh = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    ps = pd.Series(returns.values @ weights, index=returns.index)
    dd = max_drawdown(ps)
    port_cagr = cagr(ps)
    dside = ps[ps < 0].std()
    sortino = float(
        (ps.mean() * TRADING_DAYS - rf) / (dside * np.sqrt(TRADING_DAYS))
    ) if dside > 0 else 0.0
    return {
        "Annual Return": f"{port_ret:.2%}",
        "Volatility": f"{port_vol:.2%}",
        "Sharpe": f"{port_sh:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{dd:.2%}",
        "CAGR": f"{port_cagr:.2%}",
    }


def render_metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def signal_badge(label, value, color):
    bg = {"green": "#e8f5e9", "orange": "#fff3e0", "red": "#ffebee"}.get(color, "#f5f5f5")
    st.markdown(
        f'<div style="display:inline-block;padding:6px 14px;border-radius:8px;'
        f'background:{bg};border-left:4px solid {color};margin:4px 0">'
        f'<b style="color:{color}">{label}:</b> {value}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC DANGER FLAGS
# Runs before Gemini — always shown regardless of API availability
# ══════════════════════════════════════════════════════════════════════════════

def _compute_portfolio_danger_flags(
    tickers: list[str],
    health: dict,
    regime_info: dict,
    ms_stats: dict,
    mv_stats: dict,
    rp_stats: dict,
    ms_conc: dict,
    mv_conc: dict,
    ms_cost: dict,
    mv_cost: dict,
    rp_cost: dict,
    corr: pd.DataFrame,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the portfolio output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.
    """
    flags = []

    # ── Data quality ─────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Portfolio analysis for {', '.join(tickers)} is running on SYNTHETIC "
                "demo data, not real market prices. All weights, Sharpe ratios, and "
                "drawdown figures are illustrative only. Do not allocate real capital "
                "based on this output."
            ),
        })

    # ── Covariance health ─────────────────────────────────────────────────────
    if health["status"] == "Danger":
        flags.append({
            "severity": "DANGER",
            "code": "COV_DANGER",
            "message": (
                f"T/N ratio is {health['ratio']:.1f}x — critically low. "
                f"You have too few observations ({health['msg'].split(' days')[0].strip()} days) "
                f"relative to {len(tickers)} assets. "
                "The covariance matrix is estimated with very high sampling error, "
                "meaning the optimised weights are essentially noise. "
                "Need at least 5x as many observations as assets (rule of thumb: 5 * N * 252 trading days)."
            ),
        })
    elif health["status"] == "Warning":
        flags.append({
            "severity": "WARNING",
            "code": "COV_WARNING",
            "message": (
                f"T/N ratio is {health['ratio']:.1f}x — below the reliable threshold of 5x. "
                "Ledoit-Wolf shrinkage is applied, which helps, but weights may still "
                "be noisier than they appear. Consider extending the lookback period."
            ),
        })

    # ── Regime consistency ────────────────────────────────────────────────────
    regime = str(regime_info["current"])
    strategy = regime_info["strategy"]
    if "Bear" in regime:
        flags.append({
            "severity": "WARNING",
            "code": "BEAR_REGIME",
            "message": (
                f"Current regime is Bear 📉. The auto-selected strategy is Min Variance — "
                "focus is on capital preservation, not return maximisation. "
                "Max Sharpe weights optimised for this regime may produce significantly "
                "worse out-of-sample performance than in backtests. "
                "Preferred strategy: {strategy}."
            ),
        })

    # ── Sharpe below investable threshold ─────────────────────────────────────
    for label, stats, cost in [
        ("Max Sharpe", ms_stats, ms_cost),
        ("Min Variance", mv_stats, mv_cost),
        ("Risk Parity", rp_stats, rp_cost),
    ]:
        net_sh = cost["net_sharpe"]
        if net_sh < 0:
            flags.append({
                "severity": "DANGER",
                "code": f"NEGATIVE_NET_SHARPE_{label.upper().replace(' ', '_')}",
                "message": (
                    f"{label} net-of-cost Sharpe is {net_sh:.2f} — negative. "
                    "After transaction costs this strategy destroys value. "
                    "The strategy is not investable at current cost assumptions and rebalance frequency."
                ),
            })
        elif net_sh < 0.5:
            flags.append({
                "severity": "WARNING",
                "code": f"LOW_NET_SHARPE_{label.upper().replace(' ', '_')}",
                "message": (
                    f"{label} net-of-cost Sharpe is {net_sh:.2f} — below the conventional "
                    "minimum investable threshold of 0.5. At this level, the portfolio does "
                    "not adequately compensate for the risk taken after accounting for costs."
                ),
            })

    # ── Concentration risk ────────────────────────────────────────────────────
    if ms_conc["status"] == "Highly Concentrated":
        flags.append({
            "severity": "WARNING",
            "code": "HIGH_CONCENTRATION",
            "message": (
                f"Max Sharpe portfolio is Highly Concentrated "
                f"(HHI {ms_conc['norm_hhi']:.0f}/100). "
                f"The largest single position is {ms_conc['top_ticker']} at "
                f"{ms_conc['max_weight']:.1f}%, and the top-2 positions account "
                f"for {ms_conc['top2_pct']:.1f}% of the portfolio. "
                "This is not diversification — it is a concentrated bet. "
                "Consider reducing max_weight per asset in the sidebar."
            ),
        })

    # ── High average pairwise correlation ─────────────────────────────────────
    if corr is not None and len(corr) > 1:
        n = len(corr)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_corr = float(corr.values[mask].mean())
        if avg_corr > 0.7:
            flags.append({
                "severity": "WARNING",
                "code": "HIGH_CORRELATION",
                "message": (
                    f"Average pairwise correlation across the portfolio is {avg_corr:.2f} — "
                    "very high. When assets move together this closely, diversification is "
                    "largely illusory. The efficient frontier shown is wider than the real "
                    "frontier — out-of-sample drawdowns are likely to be correlated. "
                    "Consider adding assets from uncorrelated sectors or asset classes."
                ),
            })
        elif avg_corr > 0.5:
            flags.append({
                "severity": "INFO",
                "code": "MODERATE_CORRELATION",
                "message": (
                    f"Average pairwise correlation is {avg_corr:.2f} — moderate. "
                    "Some diversification benefit exists but it is limited during stress events "
                    "when correlations typically spike toward 1.0."
                ),
            })

    # ── Turnover cost warning ─────────────────────────────────────────────────
    for label, cost in [("Max Sharpe", ms_cost), ("Min Variance", mv_cost)]:
        if cost["annual_cost_pct"] > 1.0:
            flags.append({
                "severity": "WARNING",
                "code": f"HIGH_COST_{label.upper().replace(' ', '_')}",
                "message": (
                    f"{label} annual transaction cost is {cost['annual_cost_pct']:.2f}% — "
                    "above 1%. The strategy is turning over positions aggressively relative "
                    "to the rebalance frequency selected. "
                    f"Sharpe drag: {cost['sharpe_drag']:.3f}. "
                    "Increase rebalance frequency to Quarterly to reduce cost drag."
                ),
            })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — PORTFOLIO CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_portfolio_context(
    tickers: list[str],
    health: dict,
    regime_info: dict,
    ms_stats: dict,
    mv_stats: dict,
    rp_stats: dict,
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    rp_weights: np.ndarray,
    ms_conc: dict,
    mv_conc: dict,
    ms_cost: dict,
    mv_cost: dict,
    rp_cost: dict,
    corr: pd.DataFrame,
    cost_bps: float,
    rebal_freq: str,
    max_weight: float,
    danger_flags: list[dict],
    data_source: str = "unknown",
) -> dict:
    """Build the full structured context dict that gets sent to Gemini."""

    # Avg pairwise correlation
    avg_corr = None
    if corr is not None and len(corr) > 1:
        n = len(corr)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_corr = round(float(corr.values[mask].mean()), 4)

    def _w_table(names, weights):
        return [
            {"ticker": t, "weight_pct": round(float(w) * 100, 2)}
            for t, w in zip(names, weights)
        ]

    regime_dist = {
        str(k): float(v)
        for k, v in regime_info.get("regime_pct", {}).items()
    }

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "tickers": tickers,
        "n_assets": len(tickers),
        "data_source": data_source,
        "settings": {
            "cost_bps": cost_bps,
            "rebal_frequency": rebal_freq,
            "max_weight_per_asset_pct": round(max_weight * 100, 0),
        },

        # ── Regime ────────────────────────────────────────────────────────────
        "regime": {
            "current": str(regime_info["current"]),
            "auto_strategy": regime_info["strategy"],
            "reason": regime_info["reason"],
            "distribution_pct": regime_dist,
        },

        # ── Covariance health ─────────────────────────────────────────────────
        "covariance_health": {
            "status": health["status"],
            "t_over_n_ratio": health["ratio"],
            "message": health["msg"],
        },

        # ── Correlation ───────────────────────────────────────────────────────
        "average_pairwise_correlation": avg_corr,

        # ── Strategy outputs ──────────────────────────────────────────────────
        "strategies": {
            "max_sharpe": {
                "stats": ms_stats,
                "weights": _w_table(tickers, ms_weights),
                "concentration": {
                    "status": ms_conc["status"],
                    "norm_hhi": ms_conc["norm_hhi"],
                    "largest_position": f"{ms_conc['top_ticker']} {ms_conc['max_weight']}%",
                    "top2_pct": ms_conc["top2_pct"],
                },
                "net_of_cost": ms_cost,
            },
            "min_variance": {
                "stats": mv_stats,
                "weights": _w_table(tickers, mv_weights),
                "concentration": {
                    "status": mv_conc["status"],
                    "norm_hhi": mv_conc["norm_hhi"],
                    "largest_position": f"{mv_conc['top_ticker']} {mv_conc['max_weight']}%",
                },
                "net_of_cost": mv_cost,
            },
            "risk_parity": {
                "stats": rp_stats,
                "weights": _w_table(tickers, rp_weights),
                "net_of_cost": rp_cost,
            },
        },

        # ── Pre-computed danger flags ─────────────────────────────────────────
        "danger_flags": danger_flags,
        "danger_flag_count": len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in danger_flags if f["severity"] == "WARNING"]),

        # ── Reference thresholds ──────────────────────────────────────────────
        "reference_thresholds": {
            "t_over_n_ratio_good": 5.0,
            "t_over_n_ratio_warning": 2.0,
            "investable_min_sharpe": 0.5,
            "excellent_sharpe": 1.0,
            "high_concentration_hhi": 60,
            "max_annual_cost_pct": 1.0,
            "high_correlation_threshold": 0.7,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_portfolio_explanation(context: dict) -> str:
    regime    = context["regime"]["current"]
    strategy  = context["regime"]["auto_strategy"]
    ms        = context["strategies"]["max_sharpe"]
    mv        = context["strategies"]["min_variance"]
    rp        = context["strategies"]["risk_parity"]
    flags     = context.get("danger_flags", [])
    avg_corr  = context.get("average_pairwise_correlation")
    health    = context["covariance_health"]

    corr_line = (
        f"The average pairwise correlation across the portfolio is **{avg_corr:.2f}**."
        if avg_corr is not None else ""
    )

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    return (
        f"### What the output says\n"
        f"The portfolio optimiser analysed **{context['n_assets']} assets**: "
        f"{', '.join(context['tickers'])}. "
        f"Current market regime is **{regime}**, which means the recommended strategy is "
        f"**{strategy}**. {context['regime']['reason']}.\n\n"
        f"### What each number means\n"
        f"- **Covariance health**: {health['status']} (T/N = {health['t_over_n_ratio']:.1f}x). "
        f"{health['message']}.\n"
        f"- **Max Sharpe portfolio**: targets the best risk-adjusted return. "
        f"Net Sharpe after costs = {ms['net_of_cost']['net_sharpe']}, "
        f"annual cost = {ms['net_of_cost']['annual_cost_pct']}%. "
        f"Concentration: {ms['concentration']['status']}.\n"
        f"- **Min Variance portfolio**: targets the lowest possible volatility. "
        f"Net Sharpe = {mv['net_of_cost']['net_sharpe']}, "
        f"annual cost = {mv['net_of_cost']['annual_cost_pct']}%.\n"
        f"- **Risk Parity portfolio**: every asset contributes equally to portfolio risk. "
        f"Net Sharpe = {rp['net_of_cost']['net_sharpe']}, "
        f"annual cost = {rp['net_of_cost']['annual_cost_pct']}%.\n"
        f"- {corr_line}\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"The {strategy} strategy is most appropriate for the current {regime} environment. "
        f"Review the flags above before allocating capital.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI EXPLAINER
# Uses Google Gemini API via urllib (no new dependency)
# System prompt is threshold-aware, structured, danger-first, with disclaimer
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM_PROMPT = """You are a senior portfolio analyst embedded inside a professional quantitative portfolio management dashboard.

Your sole job: explain the portfolio optimisation output to a NON-TECHNICAL user — a family office client, allocator, or portfolio manager who understands investing but not the mathematics.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key number in one plain English sentence. Do not skip any metric.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should allocate" — explain what the analysis says, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No LaTeX. No formulas.

THRESHOLD KNOWLEDGE (built in — use these to interpret numbers):
- T/N ratio < 2: DANGER — weights are unreliable noise
- T/N ratio 2-5: WARNING — weights are noisy
- T/N ratio > 5: Good — weights are trustworthy
- Net Sharpe < 0: strategy destroys value after costs
- Net Sharpe 0-0.5: below investable minimum
- Net Sharpe 0.5-1.0: acceptable
- Net Sharpe > 1.0: excellent
- Normalised HHI > 60: portfolio is dangerously concentrated
- Annual transaction cost > 1%: rebalance frequency is too high given cost assumption
- Average pairwise correlation > 0.7: very little real diversification
- Average pairwise correlation 0.5-0.7: moderate diversification
- Average pairwise correlation < 0.5: genuine diversification
- Max Sharpe → best for Bull regime
- Min Variance → best for Bear regime
- Risk Parity → best for Sideways regime

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: the regime, auto-selected strategy, overall portfolio quality)

### What each number means
(Bullet per key metric: T/N ratio, each strategy's net Sharpe, concentration, avg correlation, annual cost)

### Red flags
(If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2-3 sentences max. What a smart non-quant should take away from this output.)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


def _call_gemini_explainer(context: dict) -> str:
    """
    Calls Google Gemini API with the portfolio context.
    Falls back to deterministic explanation on any error.
    """
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_portfolio_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current portfolio optimisation output from the dashboard. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    # Gemini REST endpoint: generateContent
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": _GEMINI_SYSTEM_PROMPT}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 900,
            "temperature": 0.2,   # low temperature: factual, consistent
        },
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        # Gemini response structure: candidates[0].content.parts[0].text
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        text  = "".join(p.get("text", "") for p in parts).strip()
        return text or _fallback_portfolio_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_portfolio_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Portfolio | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("Portfolio Optimizer — Quant Grade")
qe_neon_divider()

render_data_engine_controls("portfolio")

c1, c2, c3, c4 = st.columns(4)
tickers = render_multi_ticker_input("Tickers", key="portfolio_universe",
                                    default=cfg.DEFAULT_TICKERS, container=c1)
cost_bps   = c2.number_input("Transaction Cost (bps)", min_value=0.0,
                              max_value=100.0, value=10.0, step=1.0)
rebal_freq = c3.selectbox("Rebalance Frequency",
                           ["Daily (1d)", "Weekly (5d)", "Monthly (21d)", "Quarterly (63d)"],
                           index=2)
max_weight = c4.slider("Max Weight per Asset", 0.10, 1.0, 0.40, 0.05)

rebal_days = {"Daily (1d)": 1, "Weekly (5d)": 5,
              "Monthly (21d)": 21, "Quarterly (63d)": 63}[rebal_freq]
start = pd.to_datetime(get_global_start_date())

if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None

run_clicked = st.button("Run Portfolio Analysis", type="primary")
if run_clicked:
    if len(tickers) < 2:
        st.warning("Select at least 2 tickers.")
        st.session_state.portfolio_result = None
    else:
        with st.spinner("Loading data and optimizing portfolio..."):
            prices = load_multi_ticker_data(tickers, start=str(start))
            ret_df = align_returns(prices)

            if ret_df.empty or len(ret_df) < 30:
                st.error("Need at least 30 days of aligned returns. Try an earlier start date.")
                st.session_state.portfolio_result = None
            else:
                ret_df = ret_df.dropna(axis=1, thresh=int(len(ret_df) * 0.8)).dropna()
                tickers = list(ret_df.columns)
                equal_weights = np.ones(len(tickers)) / len(tickers)
                health = covariance_health(ret_df)
                regime_info = detect_market_regime(ret_df)

                eq_ret = ret_df.mean(axis=1).tail(252)
                if len(eq_ret) >= 120:
                    # ← FIXED: use _safe_fit_hmm
                    _, reg_s, _ = _safe_fit_hmm(eq_ret, n_states=3)
                    reg_map = {
                        r: (1 if "Bull" in str(r) else -1 if "Bear" in str(r) else 0)
                        for r in reg_s.unique()
                    }
                    regime_timeline = reg_s.map(reg_map)
                else:
                    roll = eq_ret.rolling(21).sum().fillna(0)
                    regime_timeline = np.sign(roll).rename("Regime")

                cov = ledoit_wolf_cov(ret_df)
                mu = ret_df.mean().values * TRADING_DAYS
                ms_weights = analytic_max_sharpe(mu, cov, cfg.RISK_FREE_RATE, max_weight)
                mv_weights = analytic_min_vol(mu, cov, max_weight)
                rp_weights = risk_parity_weights_lw(cov)
                frontier   = build_analytic_frontier(mu, cov, cfg.RISK_FREE_RATE)

                st.session_state.portfolio_result = {
                    "tickers": tickers,
                    "ret_df": ret_df,
                    "equal_weights": equal_weights,
                    "health": health,
                    "regime_info": regime_info,
                    "regime_timeline": regime_timeline,
                    "cov": cov,
                    "mu": mu,
                    "ms_weights": ms_weights,
                    "mv_weights": mv_weights,
                    "rp_weights": rp_weights,
                    "frontier": frontier,
                }
                st.session_state.portfolio_ai_summary = ""

portfolio_result = st.session_state.portfolio_result
if portfolio_result is None:
    st.info("Configure the inputs above, then press Run Portfolio Analysis.")
    qe_faq_section("FAQs", [
        ("How do I choose weights here?", "Use the optimizer output first. Max Sharpe, Min Variance, and Risk Parity give three different starting points depending on your goal."),
        ("Why does regime matter in portfolio construction?", "Bull, bear, and sideways markets reward different portfolio shapes, so the regime-adaptive tab helps the weights change with conditions."),
        ("What does concentration tell me?", "It shows whether one or two holdings dominate the portfolio. High concentration means more hidden risk if those names move against you."),
        ("When should I rebalance?", "Rebalance when the optimizer or regime view changes enough to justify the transaction costs, not on every small daily move."),
    ])
    st.stop()

tickers        = portfolio_result["tickers"]
ret_df         = portfolio_result["ret_df"]
equal_weights  = portfolio_result["equal_weights"]
health         = portfolio_result["health"]
regime_info    = portfolio_result["regime_info"]
regime_timeline = portfolio_result["regime_timeline"]
cov            = portfolio_result["cov"]
mu             = portfolio_result["mu"]
ms_weights     = portfolio_result["ms_weights"]
mv_weights     = portfolio_result["mv_weights"]
rp_weights     = portfolio_result["rp_weights"]
frontier       = portfolio_result["frontier"]

data_source = str(ret_df.attrs.get("data_source", "unknown"))

# ── Pre-compute all stats (needed for both tabs and AI layer) ─────────────────
ms_stats  = portfolio_stats_full(ms_weights, ret_df, mu, cov, cfg.RISK_FREE_RATE)
mv_stats  = portfolio_stats_full(mv_weights, ret_df, mu, cov, cfg.RISK_FREE_RATE)
rp_stats  = portfolio_stats_full(rp_weights, ret_df, mu, cov, cfg.RISK_FREE_RATE)
ms_cost   = net_of_cost_sharpe(ms_weights, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
mv_cost   = net_of_cost_sharpe(mv_weights, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
rp_cost   = net_of_cost_sharpe(rp_weights, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
ms_conc   = concentration_signal(ms_weights, tickers)
mv_conc   = concentration_signal(mv_weights, tickers)
corr      = ret_df.corr()

# ── Signals panel ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Live Signals")
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("**Covariance Health**")
    signal_badge("Status", f"{health['status']} (ratio {health['ratio']:.1f}x)", health["color"])
    st.caption(health["msg"])

with s2:
    st.markdown("**Market Regime**")
    signal_badge("Regime", regime_info["current"], regime_info["color"])
    st.caption(f"Recommended: **{regime_info['strategy']}** — {regime_info['reason']}")

with s3:
    st.markdown("**Regime Timeline (1yr)**")
    try:
        reg_num = regime_timeline
        fig_r = go.Figure(go.Scatter(
            x=reg_num.index, y=reg_num.values,
            fill="tozeroy", line=dict(width=0),
            fillcolor="rgba(29,158,117,0.3)",
        ))
        fig_r.update_layout(
            height=90, margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False, template="plotly_dark",
            yaxis=dict(showticklabels=False),
            xaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        st.caption("Chart unavailable")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Efficient Frontier", "Regime-Adaptive", "Risk Parity", "Correlation"
])

# ── TAB 1: Efficient Frontier ─────────────────────────────────────────────────
with tab1:
    st.subheader("Analytic Efficient Frontier")
    st.caption("Built by solving convex optimization at each target return — not Monte Carlo random sampling")

    if len(frontier["vols"]) > 0:
        sharpes_f = (frontier["rets"] - cfg.RISK_FREE_RATE) / (frontier["vols"] + 1e-10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontier["vols"] * 100, y=frontier["rets"] * 100,
            mode="lines+markers",
            marker=dict(size=5, color=sharpes_f, colorscale="RdYlGn",
                        colorbar=dict(title="Sharpe", thickness=12)),
            line=dict(width=2, color="rgba(255,255,255,0.2)"),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))
        for label, w, sym, clr in [
            ("Max Sharpe ★", ms_weights, "star", "#FFD700"),
            ("Min Variance ◆", mv_weights, "diamond", "#00BFFF"),
            ("Equal Weight", equal_weights, "circle", "#FF6B6B"),
        ]:
            r = float(w @ mu); v = float(np.sqrt(w @ cov @ w))
            fig.add_trace(go.Scatter(
                x=[v * 100], y=[r * 100], mode="markers+text",
                marker=dict(size=14, symbol=sym, color=clr,
                            line=dict(width=2, color="white")),
                text=[label], textposition="top center",
                textfont=dict(size=11, color=clr), name=label,
            ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    for col, w, label, stats, cost, conc in [
        (col_a, ms_weights, "Max Sharpe", ms_stats, ms_cost, ms_conc),
        (col_b, mv_weights, "Min Variance", mv_stats, mv_cost, mv_conc),
    ]:
        with col:
            st.markdown(f"**{label} Portfolio**")
            render_metric_row(dict(list(stats.items())[:3]))
            render_metric_row(dict(list(stats.items())[3:]))

            st.markdown("**Transaction Cost Impact**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Gross Sharpe", cost["gross_sharpe"])
            cc2.metric("Net Sharpe", cost["net_sharpe"],
                       delta=f"-{cost['sharpe_drag']}", delta_color="inverse")
            cc3.metric("Annual Cost", f"{cost['annual_cost_pct']}%")

            signal_badge("Concentration",
                         f"{conc['status']} (HHI {conc['norm_hhi']:.0f}/100)",
                         conc["color"])
            st.caption(f"Largest: {conc['top_ticker']} {conc['max_weight']}% | Top-2: {conc['top2_pct']}%")

            wt_df = pd.DataFrame({"Ticker": tickers, "Weight": w})
            fig_pie = px.pie(wt_df, names="Ticker", values="Weight",
                             title=f"{label} Weights", template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)


# ── TAB 2: REGIME-ADAPTIVE ────────────────────────────────────────────────────
with tab2:
    st.subheader("Regime-Adaptive Portfolio")
    st.caption("Strategy is automatically selected based on current HMM market regime")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown(
            f'<div style="background:#1e1e2e;border-radius:12px;padding:20px;'
            f'border:2px solid {regime_info["color"]}">'
            f'<h4 style="color:{regime_info["color"]};margin:0">Current Regime</h4>'
            f'<h2 style="color:white;margin:8px 0">{regime_info["current"]}</h2>'
            f'<hr style="border-color:{regime_info["color"]};opacity:0.3">'
            f'<p style="color:#aaa;margin:4px 0">Auto-Selected Strategy:</p>'
            f'<h3 style="color:{regime_info["color"]};margin:4px 0">{regime_info["strategy"]}</h3>'
            f'<p style="color:#888;font-size:13px">{regime_info["reason"]}</p></div>',
            unsafe_allow_html=True,
        )
        if regime_info["regime_pct"]:
            st.markdown("**Regime Distribution**")
            for reg, pct in regime_info["regime_pct"].items():
                clr = "#1D9E75" if "Bull" in str(reg) else "#E24B4A" if "Bear" in str(reg) else "#EF9F27"
                st.markdown(
                    f"<span style='color:{clr}'>{reg}</span>"
                    f"<span style='float:right;color:#aaa'>{pct}%</span>"
                    f"<div style='background:{clr};height:4px;width:{min(pct,100)}%;"
                    f"border-radius:2px;opacity:0.6;margin-bottom:6px'></div>",
                    unsafe_allow_html=True,
                )

    with rc2:
        strat  = regime_info["strategy"]
        reg_w  = ms_weights if strat == "Max Sharpe" else mv_weights if strat == "Min Variance" else rp_weights
        reg_stats = ms_stats if strat == "Max Sharpe" else mv_stats if strat == "Min Variance" else rp_stats
        reg_cost  = ms_cost  if strat == "Max Sharpe" else mv_cost  if strat == "Min Variance" else rp_cost

        st.markdown(f"**{strat} — Regime Weights**")
        render_metric_row(dict(list(reg_stats.items())[:3]))
        render_metric_row(dict(list(reg_stats.items())[3:]))

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Gross Sharpe", reg_cost["gross_sharpe"])
        cc2.metric("Net Sharpe",   reg_cost["net_sharpe"],
                   delta=f"-{reg_cost['sharpe_drag']}", delta_color="inverse")
        cc3.metric("Cost/yr",      f"{reg_cost['annual_cost_pct']}%")

        conc_reg = concentration_signal(reg_w, tickers)
        signal_badge("Concentration", conc_reg["status"], conc_reg["color"])

        wt_bar = pd.DataFrame({
            "Ticker": tickers, "Weight (%)": np.round(reg_w * 100, 2),
        }).sort_values("Weight (%)")
        fig_bar = go.Figure(go.Bar(
            x=wt_bar["Weight (%)"], y=wt_bar["Ticker"], orientation="h",
            marker_color=regime_info["color"],
            text=[f"{w:.1f}%" for w in wt_bar["Weight (%)"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            template="plotly_dark", height=280,
            title=f"Regime Weights ({strat})",
            xaxis_title="Weight (%)", margin=dict(l=0, r=50, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("**Strategy Comparison Table**")
    rows = []
    for sname, sw, sstats, scost in [
        ("Max Sharpe",   ms_weights, ms_stats, ms_cost),
        ("Min Variance", mv_weights, mv_stats, mv_cost),
        ("Risk Parity",  rp_weights, rp_stats, rp_cost),
        ("Equal Weight", equal_weights,
         portfolio_stats_full(equal_weights, ret_df, mu, cov, cfg.RISK_FREE_RATE),
         net_of_cost_sharpe(equal_weights, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)),
    ]:
        conc = concentration_signal(sw, tickers)
        rows.append({
            "Strategy": ("⭐ " if sname == strat else "") + sname,
            "Return": sstats["Annual Return"], "Vol": sstats["Volatility"],
            "Gross Sharpe": sstats["Sharpe"], "Net Sharpe": str(scost["net_sharpe"]),
            "Cost/yr": f"{scost['annual_cost_pct']}%",
            "Concentration": conc["status"], "Max Wt": f"{conc['max_weight']}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 3: RISK PARITY ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Equal Risk Contribution — Risk Parity")
    st.caption("Each asset contributes exactly 1/N of total portfolio volatility")

    render_metric_row(dict(list(rp_stats.items())[:3]))
    render_metric_row(dict(list(rp_stats.items())[3:]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Sharpe", rp_cost["gross_sharpe"])
    c2.metric("Net Sharpe",   rp_cost["net_sharpe"],
              delta=f"-{rp_cost['sharpe_drag']}", delta_color="inverse")
    c3.metric("Annual Cost",  f"{rp_cost['annual_cost_pct']}%")
    c4.metric("Turnover",     f"{rp_cost['turnover_pct']}%")

    conc_rp = concentration_signal(rp_weights, tickers)
    signal_badge("Concentration",
                 f"{conc_rp['status']} (HHI {conc_rp['norm_hhi']:.0f}/100)",
                 conc_rp["color"])

    col_rp1, col_rp2 = st.columns(2)
    with col_rp1:
        wt_rp_df = pd.DataFrame({"Ticker": tickers, "Weight": rp_weights})
        fig_rp = px.bar(wt_rp_df, x="Ticker", y="Weight",
                         template="plotly_dark", title="Risk Parity Weights",
                         color="Ticker", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_rp.update_layout(showlegend=False)
        st.plotly_chart(fig_rp, use_container_width=True)

    with col_rp2:
        sigma  = np.sqrt(rp_weights @ cov @ rp_weights)
        mrc    = cov @ rp_weights / sigma
        rc     = rp_weights * mrc
        rc_pct = rc / rc.sum() * 100
        rc_df  = pd.DataFrame({
            "Ticker": tickers,
            "Weight (%)": np.round(rp_weights * 100, 2),
            "Risk Contribution (%)": np.round(rc_pct, 2),
        })
        fig_rc = px.bar(rc_df, x="Ticker", y="Risk Contribution (%)",
                         template="plotly_dark", title="Actual Risk Contributions",
                         color="Ticker", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rc.add_hline(y=100 / len(tickers), line_dash="dash",
                          line_color="white", annotation_text="Target (equal)")
        fig_rc.update_layout(showlegend=False)
        st.plotly_chart(fig_rc, use_container_width=True)

    st.dataframe(
        rc_df.style.format({"Weight (%)": "{:.2f}", "Risk Contribution (%)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )


# ── TAB 4: CORRELATION ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Correlation Analysis")

    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          template="plotly_dark", title="Correlation Matrix",
                          zmin=-1, zmax=1)
    st.plotly_chart(fig_heat, use_container_width=True)

    if len(tickers) >= 2:
        t1, t2 = tickers[0], tickers[1]
        rc_roll = ret_df[[t1, t2]].rolling(63).corr().unstack()[t1][t2].dropna()
        fig_rc2 = go.Figure(go.Scatter(
            x=rc_roll.index, y=rc_roll.values,
            line=dict(color="cyan", width=1.5), fill="tozeroy",
            fillcolor="rgba(0,191,255,0.1)", name=f"{t1}-{t2} 63d Corr",
        ))
        fig_rc2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_rc2.update_layout(
            template="plotly_dark",
            title=f"Rolling 63-day Correlation: {t1} vs {t2}",
            yaxis=dict(range=[-1, 1]), height=300,
        )
        st.plotly_chart(fig_rc2, use_container_width=True)

    n = len(tickers)
    if n > 1:
        mask     = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_corr = float(corr.values[mask].mean())
        if avg_corr < 0.3:
            cs, cc = "Well Diversified", "green"
        elif avg_corr < 0.6:
            cs, cc = "Moderate Correlation", "orange"
        else:
            cs, cc = "Highly Correlated — low diversification benefit", "red"
        signal_badge("Diversification Signal",
                     f"{cs} (avg pairwise ρ = {avg_corr:.2f})", cc)


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — Gemini-powered
# Same 3-layer architecture as 06_signals.py:
#   Layer 1: Deterministic danger badges (always shown, no AI)
#   Layer 2: "Decode for Me" button → Gemini explanation
#   Layer 3: Structured AI output with disclaimer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Portfolio Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the quantitative portfolio output above into plain English. "
    "It reads the actual numbers from this analysis — not generic descriptions. "
    "It does not change the weights. It does not give financial advice."
)

# ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
danger_flags = _compute_portfolio_danger_flags(
    tickers=tickers,
    health=health,
    regime_info=regime_info,
    ms_stats=ms_stats,
    mv_stats=mv_stats,
    rp_stats=rp_stats,
    ms_conc=ms_conc,
    mv_conc=mv_conc,
    ms_cost=ms_cost,
    mv_cost=mv_cost,
    rp_cost=rp_cost,
    corr=corr,
    data_source=data_source,
)

if danger_flags:
    n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
    n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
    n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

    badge_html = ""
    if n_danger:
        badge_html += (
            f'<span style="background:#dc3232;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
            f'⛔ {n_danger} DANGER</span>'
        )
    if n_warning:
        badge_html += (
            f'<span style="background:#e67e00;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
            f'⚠️ {n_warning} WARNING</span>'
        )
    if n_info:
        badge_html += (
            f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;">'
            f'ℹ️ {n_info} INFO</span>'
        )
    st.markdown(f'<div style="margin:10px 0 6px;">{badge_html}</div>', unsafe_allow_html=True)

    for flag in danger_flags:
        color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
        bg_map    = {
            "DANGER": "rgba(220,50,50,0.08)",
            "WARNING": "rgba(230,126,0,0.08)",
            "INFO": "rgba(26,111,160,0.08)",
        }
        st.markdown(
            f"""<div style="
                background:{bg_map[flag['severity']]};
                border-left:3px solid {color_map[flag['severity']]};
                border-radius:0 6px 6px 0;
                padding:10px 14px;
                margin:6px 0;
                font-size:13px;
                line-height:1.55;
            ">
              <span style="font-weight:700;color:{color_map[flag['severity']]};">
                {flag['severity']} · {flag['code']}
              </span><br>
              {flag['message']}
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ Pre-flight checks passed — no critical flags detected for this portfolio.")

st.markdown("")

# ── LAYER 2: Build context + button ──────────────────────────────────────────
portfolio_context = _build_portfolio_context(
    tickers=tickers,
    health=health,
    regime_info=regime_info,
    ms_stats=ms_stats,
    mv_stats=mv_stats,
    rp_stats=rp_stats,
    ms_weights=ms_weights,
    mv_weights=mv_weights,
    rp_weights=rp_weights,
    ms_conc=ms_conc,
    mv_conc=mv_conc,
    ms_cost=ms_cost,
    mv_cost=mv_cost,
    rp_cost=rp_cost,
    corr=corr,
    cost_bps=cost_bps,
    rebal_freq=rebal_freq,
    max_weight=max_weight,
    danger_flags=danger_flags,
    data_source=data_source,
)

# Reset AI summary when context changes (new run)
context_key = json.dumps(
    {k: v for k, v in portfolio_context.items() if k != "danger_flags"},
    sort_keys=True, default=str,
)
if st.session_state.get("portfolio_ai_context_key") != context_key:
    st.session_state.portfolio_ai_context_key = context_key
    st.session_state.portfolio_ai_summary = ""

col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    st.markdown("**What Gemini sees:**")
    n_assets = len(tickers)
    n = len(corr)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    avg_corr_val = round(float(corr.values[mask].mean()), 3) if n > 1 else None

    preview_rows = [
        {"Field": "Assets",           "Value": ", ".join(tickers)},
        {"Field": "Regime",           "Value": str(regime_info["current"])},
        {"Field": "Auto-strategy",    "Value": regime_info["strategy"]},
        {"Field": "Data source",      "Value": data_source},
        {"Field": "Cov health",       "Value": f"{health['status']} ({health['ratio']:.1f}x)"},
        {"Field": "MS net Sharpe",    "Value": str(ms_cost["net_sharpe"])},
        {"Field": "MV net Sharpe",    "Value": str(mv_cost["net_sharpe"])},
        {"Field": "RP net Sharpe",    "Value": str(rp_cost["net_sharpe"])},
        {"Field": "MS concentration", "Value": ms_conc["status"]},
        {"Field": "Avg correlation",  "Value": str(avg_corr_val)},
        {"Field": "Danger flags",     "Value": str(len([f for f in danger_flags if f["severity"] == "DANGER"]))},
        {"Field": "Warning flags",    "Value": str(len([f for f in danger_flags if f["severity"] == "WARNING"]))},
    ]
    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    gemini_key_set = bool(getattr(cfg, "GEMINI_API_KEY", ""))
    if not gemini_key_set:
        st.info(
            "💡 Add `GEMINI_API_KEY` to your `.env` file for Gemini AI explanations. "
            "Without it, a deterministic quant-safe explanation is shown instead."
        )

    decode_clicked = st.button(
        "🤖 Decode for Me",
        type="primary",
        key="portfolio_ai_explain",
        use_container_width=True,
        help="Translates the portfolio output above into plain English using Gemini.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="portfolio_ai_clear",
        use_container_width=True,
    )

with col_ctx:
    st.markdown("**How this works:**")
    st.markdown("""
<div style="
    background: rgba(14,22,42,0.82);
    border: 1px solid rgba(11,224,255,0.18);
    border-radius: 10px;
    padding: 16px 18px;
    font-size:13px;
    line-height:1.65;
">
  <div style="font-weight:700;color:#e8f4fd;margin-bottom:10px;">What happens when you click Decode:</div>
  <ol style="margin:0;padding-left:18px;color:#a8c4d8;">
    <li style="margin-bottom:6px;">
      The <strong>pre-flight checks above run first</strong> — danger flags are always
      deterministic. They appear regardless of whether you click Decode.
    </li>
    <li style="margin-bottom:6px;">
      The actual numbers from this analysis (regime, all strategy Sharpe ratios,
      net-of-cost performance, concentration, correlation, covariance health,
      data source, and all flags) are sent to Gemini.
    </li>
    <li style="margin-bottom:6px;">
      Gemini explains each number in plain English, flags anything dangerous,
      and writes a plain-English conclusion — using the actual values, not generic descriptions.
    </li>
    <li style="margin-bottom:6px;">
      Output: <strong>4 sections</strong> — what the output says · what each number means ·
      red flags · plain-English conclusion.
    </li>
    <li>
      A <strong>mandatory disclaimer</strong> is appended — this is not financial advice.
    </li>
  </ol>
</div>
""", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.portfolio_ai_summary = ""

if decode_clicked:
    with st.spinner("Gemini is reading the portfolio output and writing your plain-English explanation..."):
        st.session_state.portfolio_ai_summary = _call_gemini_explainer(portfolio_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("portfolio_ai_summary"):
    st.markdown("")
    st.markdown(
        """<div style="
            background: rgba(14,22,42,0.82);
            border: 1px solid rgba(11,224,255,0.28);
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 8px;
        ">""",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.portfolio_ai_summary)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("")
    st.markdown(
        """<div style="
            border:1px dashed rgba(11,224,255,0.18);
            border-radius:10px;
            padding:20px;
            text-align:center;
            color:rgba(200,220,240,0.4);
            font-size:14px;
        ">
          Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
          of the portfolio output above.
        </div>""",
        unsafe_allow_html=True,
    )

# ── FAQs — shown after all output so they don't interrupt the analysis flow ───
st.markdown("")
qe_faq_section("FAQs", [
    ("How do I choose weights here?", "Use the optimizer output first. Max Sharpe, Min Variance, and Risk Parity give three different starting points depending on your goal."),
    ("Why does regime matter in portfolio construction?", "Bull, bear, and sideways markets reward different portfolio shapes, so the regime-adaptive tab helps the weights change with conditions."),
    ("What does concentration tell me?", "It shows whether one or two holdings dominate the portfolio. High concentration means more hidden risk if those names move against you."),
    ("When should I rebalance?", "Rebalance when the optimizer or regime view changes enough to justify the transaction costs, not on every small daily move."),
])