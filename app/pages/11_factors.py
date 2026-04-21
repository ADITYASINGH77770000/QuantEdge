import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from core.data import returns
import core.factor_engine as fe

# Bind commonly-used symbols from the factor engine module. Using module
# attribute access (instead of a large `from ... import ...`) reduces the
# chance of ImportError during partial module initialization and avoids
# fragile circular-import failures.
momentum_factor = fe.momentum_factor
low_vol_factor = fe.low_vol_factor
size_factor = fe.size_factor
quality_factor = fe.quality_factor
value_factor = fe.value_factor
FACTOR_FNS = getattr(fe, "FACTOR_FNS", {})

# Backwards-compatible helpers
build_factor_matrix = fe.build_factor_matrix
quintile_returns = fe.quintile_returns
factor_decay = fe.factor_decay

# Fix 2
compute_timeseries_ic = fe.compute_timeseries_ic
factor_summary_stats = fe.factor_summary_stats

# Fix 3
cost_adjusted_quintile_bt = fe.cost_adjusted_quintile_bt

# Fix 4
regime_factor_ic = fe.regime_factor_ic

# Fix 5
ic_weighted_composite = fe.ic_weighted_composite

# Fix 6
factor_attribution = fe.factor_attribution

# Fix 7
detect_factor_crowding = fe.detect_factor_crowding

# Fix 8
cross_sectional_decay = fe.cross_sectional_decay
from app.data_engine import (
    render_data_engine_controls,
    render_multi_ticker_input,
    load_multi_ticker_data,
    get_global_start_date,
)
from utils.charts import metric_card_row
from utils.config import cfg

st.set_page_config(page_title="Factors | QuantEdge", layout="wide")
st.title("🧪 Factor Research Lab")
st.caption(
    "8-fix upgrade — Honest Proxies · Time-Series IC · Cost-Adjusted Quintiles · "
    "Regime Conditioning · IC-Weighted Composite · Attribution · Crowding · Correct Decay"
)

# ── Controls ──────────────────────────────────────────────────────────────────
render_data_engine_controls("factors")
c1, c2, c3 = st.columns(3)
tickers  = render_multi_ticker_input("Universe", key="factors_universe",
               default=cfg.DEFAULT_TICKERS, container=c1)
fwd_days = c2.slider("Forward Return Window (days)", 1, 63, 21)
factor_choice = c3.selectbox("Primary Factor (for deep analysis)",
                               list(FACTOR_FNS.keys()), index=0)

start = pd.to_datetime(get_global_start_date())

if len(tickers) < 2:
    st.warning("Select at least 2 tickers to compute cross-sectional factors.")
    st.stop()

with st.spinner("Loading price data..."):
    prices = load_multi_ticker_data(tickers, start=str(start))

# ── Extra controls ────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Settings"):
    ec = st.columns(3)
    cost_bps   = ec[0].number_input("Round-trip cost (bps)", 10, 200, 40)
    rebal_freq = ec[1].number_input("Rebalance frequency (days)", 5, 63, 21)
    n_quintiles = ec[2].number_input("Quintiles", 3, 10, 5)

run_btn = st.button("▶  Run Full Factor Analysis", type="primary")

if not run_btn:
    st.info("Configure settings above and click **Run Full Factor Analysis**.")
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0 — Factor Score Matrix (all 5 factors, current snapshot)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📋 Factor Score Matrix  (rank-normalised, current snapshot)")
st.caption(
    "Each score is the percentile rank of that ticker on that factor (0 = worst, 1 = best). "
    "Scores are cross-sectional — they compare tickers to each other, not to history."
)

with st.spinner("Computing factor scores..."):
    factor_df = build_factor_matrix(prices)

st.dataframe(
    factor_df.style.background_gradient(cmap="RdYlGn", axis=1),
    use_container_width=True,
)
st.caption(
    "⚠️ Value and Quality use OHLCV-based proxies (price trend deviation, "
    "return consistency). True P/B and ROE require fundamental data."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — FIX 5: IC-Weighted Composite + Factor Weights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("⚖️ IC-Weighted Composite Factor Score  [Fix 5]")
st.caption(
    "Grinold & Kahn (1999): weight each factor by its recent IC. "
    "Stronger predictors get more weight — adapts to changing market conditions."
)

with st.spinner("Computing IC-weighted composite..."):
    # Run time-series IC for all factors (needed for weighting)
    ts_ic_all = {}
    for fname in FACTOR_FNS:
        ts_ic_all[fname] = compute_timeseries_ic(
            prices, factor_name=fname, fwd_days=fwd_days,
            rebalance_freq=int(rebal_freq))

    composite_scores, weights = ic_weighted_composite(prices, ts_ic_all)

# Weights chart
wt_df = pd.DataFrame([{"Factor": k, "IC Weight": round(v, 4)}
                        for k, v in weights.items()])
wt_df["Direction"] = wt_df["IC Weight"].apply(lambda x: "Positive" if x >= 0 else "Negative")

fig_wt = go.Figure(go.Bar(
    x=wt_df["Factor"], y=wt_df["IC Weight"],
    marker_color=["lime" if v >= 0 else "tomato" for v in wt_df["IC Weight"]],
    text=[f"{v:.4f}" for v in wt_df["IC Weight"]], textposition="outside",
))
fig_wt.update_layout(template="plotly_dark", height=300,
    title="Dynamic IC Weights  (updated each rebalance)",
    yaxis_title="Weight (proportional to recent IC)")
st.plotly_chart(fig_wt, use_container_width=True)

# Composite score table
comp_df = composite_scores.to_frame("Composite Score").round(4)
comp_df["Rank"] = comp_df["Composite Score"].rank(ascending=False).astype(int)
comp_df = comp_df.sort_values("Composite Score", ascending=False)
st.dataframe(comp_df.style.background_gradient(subset=["Composite Score"],
             cmap="RdYlGn"), use_container_width=True)
st.success(
    f"Top pick: **{comp_df.index[0]}** (composite={comp_df.iloc[0,0]:.3f}) · "
    f"Bottom: **{comp_df.index[-1]}** (composite={comp_df.iloc[-1,0]:.3f})"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — FIX 2: Time-Series IC (all factors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📈 Time-Series IC Analysis  [Fix 2]")
st.caption(
    "IC computed at every rebalance date across full history (not a single snapshot). "
    "Mean IC > 0.05 = meaningful. ICIR > 0.5 = reliable."
)

# Summary table
ic_summary_rows = []
for fname, ts_df in ts_ic_all.items():
    row = factor_summary_stats(ts_df)
    row["Factor"] = fname
    ic_summary_rows.append(row)

ic_sum_df = pd.DataFrame(ic_summary_rows)[["Factor","Mean IC","ICIR","IC > 0 %","Obs","Signal"]]

def _ic_colour(val):
    if isinstance(val, str):
        if "Strong" in val: return "color:lime"
        if "Moderate" in val: return "color:gold"
        if "Weak" in val: return "color:tomato"
    return ""

st.dataframe(ic_sum_df.style.map(_ic_colour, subset=["Signal"]),
             use_container_width=True)

# Rolling IC chart for selected factor
ts_sel = ts_ic_all.get(factor_choice, pd.DataFrame())
if not ts_sel.empty and "IC" in ts_sel.columns:
    fig_ic = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[f"{factor_choice} — IC per Rebalance",
                         "Rolling 12-period Mean IC & ICIR"],
        row_heights=[0.5, 0.5], vertical_spacing=0.08)

    fig_ic.add_trace(go.Bar(x=ts_sel["Date"], y=ts_sel["IC"],
        marker_color=["lime" if v >= 0 else "tomato" for v in ts_sel["IC"]],
        name="IC"), row=1, col=1)
    fig_ic.add_hline(y=0.05,  line_dash="dot", line_color="lime",
                      annotation_text="IC=0.05", row=1, col=1)
    fig_ic.add_hline(y=-0.05, line_dash="dot", line_color="tomato", row=1, col=1)
    fig_ic.add_hline(y=0, line_color="gray", row=1, col=1)

    if "RollingMeanIC" in ts_sel.columns:
        fig_ic.add_trace(go.Scatter(x=ts_sel["Date"], y=ts_sel["RollingMeanIC"],
            line=dict(color="cyan", width=2), name="Rolling Mean IC"), row=2, col=1)
    if "RollingICIR" in ts_sel.columns:
        fig_ic.add_trace(go.Scatter(x=ts_sel["Date"], y=ts_sel["RollingICIR"],
            line=dict(color="gold", width=1.5, dash="dash"), name="Rolling ICIR"),
            row=2, col=1)
    fig_ic.add_hline(y=0.5, line_dash="dot", line_color="lime",
                      annotation_text="ICIR=0.5", row=2, col=1)
    fig_ic.update_layout(template="plotly_dark", height=450,
        title=f"{factor_choice} Factor — Full History IC", showlegend=True)
    st.plotly_chart(fig_ic, use_container_width=True)
else:
    st.warning("Insufficient history to compute time-series IC. Use a longer date range.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — FIX 3: Cost-Adjusted Quintile Backtest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("💰 Cost-Adjusted Quintile Backtest  [Fix 3]")
st.caption(
    "Novy-Marx & Velikov (2016): factor strategies often look good gross of costs "
    "but lose money net. Deducts round-trip costs on turnover at each rebalance."
)

with st.spinner("Running cost-adjusted quintile backtest..."):
    qbt = cost_adjusted_quintile_bt(
        prices, factor_name=factor_choice,
        fwd_days=int(fwd_days),
        round_trip_cost_bps=float(cost_bps),
        n_quintiles=int(n_quintiles),
        rebalance_freq=int(rebal_freq),
    )

if "error" in qbt:
    st.warning(qbt["error"])
else:
    qc = st.columns(4)
    qc[0].metric("Gross L/S CAGR",
                  f"{qbt['ls_gross_cagr']:.2%}" if not np.isnan(qbt.get('ls_gross_cagr', np.nan)) else "N/A")
    qc[1].metric("Net L/S CAGR",
                  f"{qbt['ls_net_cagr']:.2%}" if not np.isnan(qbt.get('ls_net_cagr', np.nan)) else "N/A",
                  delta="After costs",
                  delta_color="normal" if not np.isnan(qbt.get('ls_net_cagr', np.nan)) and qbt['ls_net_cagr'] > 0 else "inverse")
    qc[2].metric("Avg Turnover",  f"{qbt['avg_turnover']:.1%}")
    qc[3].metric("Cost per rebal", f"{cost_bps} bps")

    tbl = qbt.get("table", pd.DataFrame())
    if not tbl.empty:
        st.dataframe(tbl, use_container_width=True)

    gross = qbt.get("ls_gross_cagr", np.nan)
    net   = qbt.get("ls_net_cagr",   np.nan)
    if not np.isnan(gross) and not np.isnan(net):
        if net > 0:
            st.success(
                f"Strategy is profitable NET of costs. "
                f"Gross CAGR = {gross:.2%}, Net CAGR = {net:.2%}. "
                f"Costs consume {(gross-net):.2%} of return."
            )
        else:
            st.error(
                f"Strategy is UNPROFITABLE net of {cost_bps}bps round-trip costs. "
                f"Gross CAGR = {gross:.2%}, Net CAGR = {net:.2%}. "
                f"Reduce rebalancing frequency or cut costs."
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — FIX 4: Regime-Conditioned Factor IC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🌦️ Regime-Conditioned Factor Performance  [Fix 4]")
st.caption(
    "Daniel & Moskowitz (2016): Momentum crashes in bear markets. "
    "Each factor's IC is broken down by Bull / Sideways / Bear regime."
)

with st.spinner("Computing regime-conditioned IC..."):
    reg_df = regime_factor_ic(prices, fwd_days=int(fwd_days))

if reg_df.empty:
    st.warning("Not enough data for regime analysis. Use a longer date range (3+ years).")
else:
    def _reg_colour(val):
        if isinstance(val, str):
            if "Strong" in val: return "color:lime"
            if "Moderate" in val: return "color:gold"
            if "Weak" in val: return "color:tomato"
        return ""

    st.dataframe(
        reg_df.style.map(_reg_colour, subset=["Signal"])
                    .background_gradient(subset=["Mean IC"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # Heatmap: Factor × Regime IC
    if not reg_df.empty:
        pivot = reg_df.pivot_table(
            index="Factor", columns="Regime", values="Mean IC", aggfunc="mean"
        )
        fig_rh = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}", textfont=dict(size=12),
        ))
        fig_rh.update_layout(template="plotly_dark", height=300,
            title="Factor IC by Regime  (green = factor works, red = factor hurts)")
        st.plotly_chart(fig_rh, use_container_width=True)

    best_regime = reg_df.sort_values("Mean IC", ascending=False).iloc[0]
    worst_regime = reg_df.sort_values("Mean IC").iloc[0]
    st.info(
        f"Best combo: **{best_regime['Factor']}** in **{best_regime['Regime']}** "
        f"(IC={best_regime['Mean IC']:.4f}) · "
        f"Worst: **{worst_regime['Factor']}** in **{worst_regime['Regime']}** "
        f"(IC={worst_regime['Mean IC']:.4f})"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — FIX 6: Factor Attribution (Carhart)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🔬 Factor Attribution (Carhart 1997)  [Fix 6]")
st.caption(
    "Decomposes returns of the composite factor strategy into alpha + factor betas. "
    "High alpha (t-stat > 2) = genuine skill. Low alpha = just riding factor premia."
)

with st.spinner("Running factor attribution..."):
    # Use equal-weight portfolio of tickers as the 'strategy'
    all_rets_list = []
    for t in tickers:
        r = prices[t]["Close"].pct_change().dropna().rename(t)
        all_rets_list.append(r)
    if all_rets_list:
        strat_ret = pd.concat(all_rets_list, axis=1).dropna().mean(axis=1)
        attr = factor_attribution(strat_ret, prices, rf=cfg.RISK_FREE_RATE)
    else:
        attr = {"error": "No return data available."}

if "error" in attr:
    st.warning(attr["error"])
else:
    ac = st.columns(3)
    ac[0].metric("Annualised Alpha",  attr["alpha_pct"])
    ac[1].metric("Alpha t-stat",      str(attr["alpha_tstat"]),
                  delta=("Significant ✅" if abs(attr["alpha_tstat"]) >= 2 else "Not significant ❌"),
                  delta_color="normal" if abs(attr["alpha_tstat"]) >= 2 else "inverse")
    ac[2].metric("R² (factor explained)", f"{attr['r_squared']:.2%}")

    def _sig_colour(val):
        if "Yes" in str(val): return "color:lime"
        if "No"  in str(val): return "color:gray"
        return ""

    st.dataframe(
        attr["table"].style.map(_sig_colour, subset=["Significant"]),
        use_container_width=True,
    )

    if abs(attr["alpha_tstat"]) >= 2.0:
        alpha_val = float(attr["alpha_pct"].replace("%","")) / 100
        st.success(
            f"Statistically significant alpha detected: {attr['alpha_pct']} per year "
            f"(t={attr['alpha_tstat']}). The strategy generates return beyond factor exposure."
        )
    else:
        st.warning(
            f"No significant alpha (t={attr['alpha_tstat']}). "
            "The strategy returns are explained by factor exposure alone. "
            "A cheap factor ETF would give similar returns at lower cost."
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — FIX 7: Factor Crowding Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🚨 Factor Crowding Detection  [Fix 7]")
st.caption(
    "Khandani & Lo (2007): crowded factors cause correlated liquidation — "
    "all funds lose simultaneously. Detected via collapse in cross-sectional score dispersion."
)

with st.spinner("Detecting factor crowding..."):
    crowd = detect_factor_crowding(prices, factor_name=factor_choice)

if "error" in crowd:
    st.warning(crowd["error"])
else:
    cc = st.columns(4)
    cc[0].metric("Crowding Level",      crowd["crowding_level"])
    cc[1].metric("Dispersion Percentile",
                  f"{crowd['current_pctile']:.1%}",
                  help="Low percentile = crowded (everyone holds same stocks)")
    cc[2].metric("Current Dispersion",  str(crowd["current_dispersion"]))
    cc[3].metric("Avg Score AutoCorr",  str(crowd["avg_autocorr"]),
                  help="High autocorrelation = rankings are persistent = crowded")

    if crowd["is_crowded"]:
        st.error(
            f"Factor crowding detected — level: {crowd['crowding_level']}. "
            f"Score dispersion at {crowd['current_pctile']:.0%} of historical range. "
            "Reduce position size or switch factors. "
            "High crowding → correlated losses when liquidation occurs."
        )
    else:
        st.success(
            f"No significant crowding — dispersion at {crowd['current_pctile']:.0%} "
            "of historical range. Factor appears uncrowded."
        )

    cdf = crowd.get("crowding_series")
    if cdf is not None and not cdf.empty:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Dispersion"],
            fill="tozeroy", fillcolor="rgba(100,180,255,0.15)",
            line=dict(color="royalblue", width=1.5), name="Score Dispersion"))
        lo_pct = float(cdf["Dispersion"].quantile(0.25))
        fig_c.add_hline(y=lo_pct, line_dash="dot", line_color="orange",
                         annotation_text="25th pct (crowding zone)")
        fig_c.update_layout(template="plotly_dark", height=300,
            title=f"{factor_choice} Factor Score Dispersion over Time  "
                   "(drops = crowding, spikes = dispersion / opportunity)")
        st.plotly_chart(fig_c, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — FIX 8: Correct Cross-Sectional Decay Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📉 Cross-Sectional Factor Decay Curve  [Fix 8]")
st.caption(
    "IC computed cross-sectionally at multiple sample dates for each horizon — "
    "not a single ticker's time series. Grinold & Kahn (1999) methodology. "
    "Tells you how fast the signal loses predictive power."
)

with st.spinner("Computing cross-sectional decay curve..."):
    decay_df = cross_sectional_decay(prices, factor_name=factor_choice,
                                      horizons=[1, 5, 10, 21, 63, 126])

if not decay_df.empty and decay_df["IC"].notna().any():
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=decay_df["Horizon (days)"], y=decay_df["IC"],
        mode="lines+markers", line=dict(color="cyan", width=2),
        error_y=dict(type="data", array=decay_df["IC Std"].fillna(0).tolist(),
                      visible=True, color="rgba(100,200,255,0.4)"),
        name="Mean IC",
    ))
    fig_d.add_hline(y=0,    line_dash="dash",  line_color="gray")
    fig_d.add_hline(y=0.05, line_dash="dot",   line_color="lime",
                     annotation_text="IC=0.05 threshold")
    fig_d.update_layout(template="plotly_dark", height=340,
        title=f"{factor_choice} — Cross-Sectional IC Decay "
               "(error bars = std across sample dates)",
        xaxis_title="Forward Horizon (days)",
        yaxis_title="Mean Cross-Sectional IC",
    )
    st.plotly_chart(fig_d, use_container_width=True)
    st.dataframe(decay_df, use_container_width=True)

    # Interpret optimal holding period
    pos_ic = decay_df[decay_df["IC"] > 0.02]
    if not pos_ic.empty:
        optimal = int(pos_ic["Horizon (days)"].max())
        st.info(
            f"Signal remains meaningful (IC > 0.02) up to **{optimal}-day** horizon. "
            f"Optimal rebalance frequency for {factor_choice}: every {optimal} days."
        )
    else:
        st.warning(
            "Signal IC is below 0.02 at all horizons. "
            f"{factor_choice} may not have predictive power in this universe."
        )
else:
    st.warning("Not enough data for decay curve. Use a longer date range (3+ years).")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — Summary & Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📋 Factor Research Summary")

best_factor = max(ic_summary_rows, key=lambda r: abs(r.get("Mean IC", 0)
                    if isinstance(r.get("Mean IC", 0), float) else 0),
                  default={}).get("Factor", "N/A")

recs = []

# IC quality
for row in ic_summary_rows:
    if isinstance(row.get("Mean IC"), float) and abs(row["Mean IC"]) < 0.02:
        recs.append(f"⚠️ **{row['Factor']}** has very low IC ({row['Mean IC']:.4f}). "
                     "Consider removing from composite — it adds noise, not signal.")

# Cost
if "ls_net_cagr" in qbt and not np.isnan(qbt.get("ls_net_cagr", np.nan)):
    if qbt["ls_net_cagr"] <= 0:
        recs.append(f"🔴 **{factor_choice}** quintile strategy loses money net of "
                     f"{cost_bps}bps costs. Reduce rebalancing frequency to "
                     f"at least {int(rebal_freq * 2)} days.")

# Crowding
if not crowd.get("is_crowded", True) is False:
    if crowd.get("is_crowded"):
        recs.append(f"🔴 **{factor_choice}** is crowded (dispersion at "
                     f"{crowd.get('current_pctile',0):.0%} percentile). "
                     "Reduce allocation or switch factors.")

# Alpha
if "alpha_tstat" in attr and not isinstance(attr.get("alpha_tstat"), str):
    if abs(attr["alpha_tstat"]) < 2.0:
        recs.append("⚠️ No significant alpha detected. Portfolio returns are "
                     "fully explained by factor premia — consider a cheaper "
                     "passive factor ETF instead.")

# Best factor recommendation
recs.append(f"✅ Best factor by IC: **{best_factor}**. "
             f"Use as primary signal with IC-weighted composite (Fix 5).")

for rec in recs:
    st.markdown(rec)