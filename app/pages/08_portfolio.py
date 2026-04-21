
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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

TRADING_DAYS = 252


# ── 1. LEDOIT-WOLF SHRINKAGE COVARIANCE ──────────────────────────────────────

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
    return {"status": status, "color": color, "msg": msg, "ratio": ratio}


# ── 2. ANALYTIC CONVEX-OPTIMISED FRONTIER ────────────────────────────────────

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


# ── 3. REGIME DETECTION ───────────────────────────────────────────────────────

def _fallback_regime(eq_weighted: pd.Series) -> tuple:
    """
    Rule-based regime when HMM fails (too little data or numerical issues).
    Uses 63-day vs 252-day return + realised vol to classify.
    """
    if len(eq_weighted) < 63:
        return "Sideways ↔", {}
    ret_63 = float(eq_weighted.tail(63).sum())
    ret_252 = float(eq_weighted.tail(min(252, len(eq_weighted))).sum())
    vol_21 = float(eq_weighted.tail(21).std() * np.sqrt(252))
    vol_63 = float(eq_weighted.tail(63).std() * np.sqrt(252))
    # Bull: positive momentum on both windows, not spiking vol
    if ret_63 > 0.03 and ret_252 > 0.05 and vol_21 < vol_63 * 1.3:
        regime = "Bull 📈"
    elif ret_63 < -0.03 or vol_21 > vol_63 * 1.5:
        regime = "Bear 📉"
    else:
        regime = "Sideways ↔"
    # Approximate pct from recent history
    pct = {regime: 100.0}
    return regime, pct


def detect_market_regime(returns_df):
    eq_weighted = returns_df.mean(axis=1)
    state_series = None
    regime_pct = {}

    # Try HMM first; fall back to rule-based on any failure
    try:
        if len(eq_weighted) >= 120:
            _, state_series, _ = fit_hmm(eq_weighted, n_states=3)
            current_regime = state_series.iloc[-1]
            regime_pct = (state_series.value_counts() / len(state_series) * 100).round(1).to_dict()
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

    return {"current": current_regime, "strategy": strategy,
            "reason": reason, "color": color,
            "regime_pct": regime_pct, "series": state_series}


# ── 4. NET-OF-COST SHARPE ─────────────────────────────────────────────────────

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


# ── 5. CONCENTRATION SIGNAL ───────────────────────────────────────────────────

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
    return {"hhi": round(hhi, 4), "norm_hhi": round(norm_hhi, 1),
            "status": status, "color": color,
            "max_weight": round(max_w * 100, 1),
            "top_ticker": top_ticker, "top2_pct": round(top2 * 100, 1)}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def portfolio_stats_full(weights, returns, mu, cov, rf=0.045):
    port_ret = float(weights @ mu)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    port_sh = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    ps = pd.Series(returns.values @ weights, index=returns.index)
    dd = max_drawdown(ps)
    port_cagr = cagr(ps)
    dside = ps[ps < 0].std()
    sortino = float((ps.mean() * TRADING_DAYS - rf) / (dside * np.sqrt(TRADING_DAYS))) if dside > 0 else 0.0
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


# ── PAGE ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio | QuantEdge", layout="wide")
st.title("Portfolio Optimizer — Quant Grade")
st.caption("Ledoit-Wolf Covariance · Analytic Frontier · Regime-Conditional Strategy · Net-of-Cost Sharpe")

render_data_engine_controls("portfolio")

c1, c2, c3, c4 = st.columns(4)
tickers = render_multi_ticker_input("Tickers", key="portfolio_universe",
                                    default=cfg.DEFAULT_TICKERS, container=c1)
cost_bps = c2.number_input("Transaction Cost (bps)", min_value=0.0,
                            max_value=100.0, value=10.0, step=1.0)
rebal_freq = c3.selectbox("Rebalance Frequency",
                           ["Daily (1d)", "Weekly (5d)", "Monthly (21d)", "Quarterly (63d)"],
                           index=2)
max_weight = c4.slider("Max Weight per Asset", 0.10, 1.0, 0.40, 0.05)

rebal_days = {"Daily (1d)": 1, "Weekly (5d)": 5,
              "Monthly (21d)": 21, "Quarterly (63d)": 63}[rebal_freq]
start = pd.to_datetime(get_global_start_date())

if len(tickers) < 2:
    st.warning("Select at least 2 tickers.")
    st.stop()

with st.spinner("Loading data..."):
    prices = load_multi_ticker_data(tickers, start=str(start))
    ret_df = align_returns(prices)

if ret_df.empty or len(ret_df) < 30:
    st.error("Need at least 30 days of aligned returns. Try an earlier start date.")
    st.stop()

ret_df = ret_df.dropna(axis=1, thresh=int(len(ret_df) * 0.8)).dropna()
tickers = list(ret_df.columns)
equal_weights = np.ones(len(tickers)) / len(tickers)

# ── Signals panel ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Live Signals")
s1, s2, s3 = st.columns(3)

with s1:
    health = covariance_health(ret_df)
    st.markdown("**Covariance Health**")
    signal_badge("Status", f"{health['status']} (ratio {health['ratio']:.1f}x)", health["color"])
    st.caption(health["msg"])

with s2:
    with st.spinner("Detecting regime..."):
        regime_info = detect_market_regime(ret_df)
    st.markdown("**Market Regime**")
    signal_badge("Regime", regime_info["current"], regime_info["color"])
    st.caption(f"Recommended: **{regime_info['strategy']}** — {regime_info['reason']}")

with s3:
    st.markdown("**Regime Timeline (1yr)**")
    try:
        eq_ret = ret_df.mean(axis=1).tail(252)
        if len(eq_ret) >= 120:
            _, reg_s, _ = fit_hmm(eq_ret, n_states=3)
            reg_map = {r: (1 if "Bull" in str(r) else -1 if "Bear" in str(r) else 0)
                       for r in reg_s.unique()}
            reg_num = reg_s.map(reg_map)
        else:
            # Fallback: rolling return sign as proxy
            roll = eq_ret.rolling(21).sum().fillna(0)
            reg_num = np.sign(roll).rename("Regime")
        fig_r = go.Figure(go.Scatter(x=reg_num.index, y=reg_num.values,
                                     fill="tozeroy", line=dict(width=0),
                                     fillcolor="rgba(29,158,117,0.3)"))
        fig_r.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0),
                             showlegend=False, template="plotly_dark",
                             yaxis=dict(showticklabels=False),
                             xaxis=dict(showticklabels=False))
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        st.caption("Chart unavailable")

st.markdown("---")

# ── Run optimisation ──────────────────────────────────────────────────────────
with st.spinner("Running Ledoit-Wolf + Analytic Optimization..."):
    cov = ledoit_wolf_cov(ret_df)
    mu = ret_df.mean().values * TRADING_DAYS
    ms_weights = analytic_max_sharpe(mu, cov, cfg.RISK_FREE_RATE, max_weight)
    mv_weights = analytic_min_vol(mu, cov, max_weight)
    rp_weights = risk_parity_weights_lw(cov)
    frontier = build_analytic_frontier(mu, cov, cfg.RISK_FREE_RATE)

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
        fig.update_layout(template="plotly_dark",
                          xaxis_title="Annual Volatility (%)",
                          yaxis_title="Annual Return (%)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          height=480)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    for col, w, label in [(col_a, ms_weights, "Max Sharpe"), (col_b, mv_weights, "Min Variance")]:
        with col:
            st.markdown(f"**{label} Portfolio**")
            stats = portfolio_stats_full(w, ret_df, mu, cov, cfg.RISK_FREE_RATE)
            render_metric_row(dict(list(stats.items())[:3]))
            render_metric_row(dict(list(stats.items())[3:]))

            cost = net_of_cost_sharpe(w, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
            st.markdown("**Transaction Cost Impact**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Gross Sharpe", cost["gross_sharpe"])
            cc2.metric("Net Sharpe", cost["net_sharpe"],
                       delta=f"-{cost['sharpe_drag']}", delta_color="inverse")
            cc3.metric("Annual Cost", f"{cost['annual_cost_pct']}%")

            conc = concentration_signal(w, tickers)
            signal_badge("Concentration", f"{conc['status']} (HHI {conc['norm_hhi']:.0f}/100)", conc["color"])
            st.caption(f"Largest: {conc['top_ticker']} {conc['max_weight']}% | Top-2: {conc['top2_pct']}%")

            wt_df = pd.DataFrame({"Ticker": tickers, "Weight": w})
            fig_pie = px.pie(wt_df, names="Ticker", values="Weight", title=f"{label} Weights",
                             template="plotly_dark",
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
                    f"<div style='background:{clr};height:4px;width:{min(pct,100)}%;border-radius:2px;opacity:0.6;margin-bottom:6px'></div>",
                    unsafe_allow_html=True,
                )

    with rc2:
        strat = regime_info["strategy"]
        reg_w = ms_weights if strat == "Max Sharpe" else mv_weights if strat == "Min Variance" else rp_weights

        st.markdown(f"**{strat} — Regime Weights**")
        stats_reg = portfolio_stats_full(reg_w, ret_df, mu, cov, cfg.RISK_FREE_RATE)
        render_metric_row(dict(list(stats_reg.items())[:3]))
        render_metric_row(dict(list(stats_reg.items())[3:]))

        cost_reg = net_of_cost_sharpe(reg_w, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Gross Sharpe", cost_reg["gross_sharpe"])
        cc2.metric("Net Sharpe", cost_reg["net_sharpe"],
                   delta=f"-{cost_reg['sharpe_drag']}", delta_color="inverse")
        cc3.metric("Cost/yr", f"{cost_reg['annual_cost_pct']}%")

        conc_reg = concentration_signal(reg_w, tickers)
        signal_badge("Concentration", conc_reg["status"], conc_reg["color"])

        wt_bar = pd.DataFrame({"Ticker": tickers,
                                "Weight (%)": np.round(reg_w * 100, 2)}).sort_values("Weight (%)")
        fig_bar = go.Figure(go.Bar(
            x=wt_bar["Weight (%)"], y=wt_bar["Ticker"], orientation="h",
            marker_color=regime_info["color"],
            text=[f"{w:.1f}%" for w in wt_bar["Weight (%)"]],
            textposition="outside",
        ))
        fig_bar.update_layout(template="plotly_dark", height=280,
                               title=f"Regime Weights ({strat})",
                               xaxis_title="Weight (%)",
                               margin=dict(l=0, r=50, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # All-strategy comparison table
    st.markdown("---")
    st.markdown("**Strategy Comparison Table**")
    rows = []
    for sname, sw in [("Max Sharpe", ms_weights), ("Min Variance", mv_weights),
                       ("Risk Parity", rp_weights), ("Equal Weight", equal_weights)]:
        s = portfolio_stats_full(sw, ret_df, mu, cov, cfg.RISK_FREE_RATE)
        c = net_of_cost_sharpe(sw, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
        conc = concentration_signal(sw, tickers)
        rows.append({
            "Strategy": ("⭐ " if sname == strat else "") + sname,
            "Return": s["Annual Return"], "Vol": s["Volatility"],
            "Gross Sharpe": s["Sharpe"], "Net Sharpe": str(c["net_sharpe"]),
            "Cost/yr": f"{c['annual_cost_pct']}%",
            "Concentration": conc["status"], "Max Wt": f"{conc['max_weight']}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 3: RISK PARITY ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Equal Risk Contribution — Risk Parity")
    st.caption("Each asset contributes exactly 1/N of total portfolio volatility")

    stats_rp = portfolio_stats_full(rp_weights, ret_df, mu, cov, cfg.RISK_FREE_RATE)
    render_metric_row(dict(list(stats_rp.items())[:3]))
    render_metric_row(dict(list(stats_rp.items())[3:]))

    cost_rp = net_of_cost_sharpe(rp_weights, equal_weights, ret_df, cost_bps, rebal_days, cfg.RISK_FREE_RATE)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Sharpe", cost_rp["gross_sharpe"])
    c2.metric("Net Sharpe", cost_rp["net_sharpe"],
              delta=f"-{cost_rp['sharpe_drag']}", delta_color="inverse")
    c3.metric("Annual Cost", f"{cost_rp['annual_cost_pct']}%")
    c4.metric("Turnover", f"{cost_rp['turnover_pct']}%")

    conc_rp = concentration_signal(rp_weights, tickers)
    signal_badge("Concentration", f"{conc_rp['status']} (HHI {conc_rp['norm_hhi']:.0f}/100)", conc_rp["color"])

    col_rp1, col_rp2 = st.columns(2)
    with col_rp1:
        wt_rp_df = pd.DataFrame({"Ticker": tickers, "Weight": rp_weights})
        fig_rp = px.bar(wt_rp_df, x="Ticker", y="Weight",
                         template="plotly_dark", title="Risk Parity Weights",
                         color="Ticker", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_rp.update_layout(showlegend=False)
        st.plotly_chart(fig_rp, use_container_width=True)

    with col_rp2:
        sigma = np.sqrt(rp_weights @ cov @ rp_weights)
        mrc = cov @ rp_weights / sigma
        rc = rp_weights * mrc
        rc_pct = rc / rc.sum() * 100
        rc_df = pd.DataFrame({"Ticker": tickers,
                               "Weight (%)": np.round(rp_weights * 100, 2),
                               "Risk Contribution (%)": np.round(rc_pct, 2)})
        fig_rc = px.bar(rc_df, x="Ticker", y="Risk Contribution (%)",
                         template="plotly_dark", title="Actual Risk Contributions",
                         color="Ticker", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rc.add_hline(y=100 / len(tickers), line_dash="dash",
                          line_color="white", annotation_text="Target (equal)")
        fig_rc.update_layout(showlegend=False)
        st.plotly_chart(fig_rc, use_container_width=True)

    st.dataframe(rc_df.style.format({"Weight (%)": "{:.2f}", "Risk Contribution (%)": "{:.2f}"}),
                 use_container_width=True, hide_index=True)


# ── TAB 4: CORRELATION ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Correlation Analysis")

    corr = ret_df.corr()
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
        fig_rc2.update_layout(template="plotly_dark",
                               title=f"Rolling 63-day Correlation: {t1} vs {t2}",
                               yaxis=dict(range=[-1, 1]), height=300)
        st.plotly_chart(fig_rc2, use_container_width=True)

    n = len(tickers)
    if n > 1:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_corr = float(corr.values[mask].mean())
        if avg_corr < 0.3:
            cs, cc = "Well Diversified", "green"
        elif avg_corr < 0.6:
            cs, cc = "Moderate Correlation", "orange"
        else:
            cs, cc = "Highly Correlated — low diversification benefit", "red"
        signal_badge("Diversification Signal",
                     f"{cs} (avg pairwise ρ = {avg_corr:.2f})", cc)