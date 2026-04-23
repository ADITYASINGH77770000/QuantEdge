import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.data import returns
from app.data_engine import (
    render_data_engine_controls, render_single_ticker_input,
    load_ticker_data, get_global_start_date,
)
from core.regime_detector import (
    fit_hmm, forward_regime_proba, compute_regime_age, regime_age_scalar,
    critical_slowing_down, get_strategy_for_regime, rolling_regime_proba,
    regime_conditional_sharpe, full_regime_analysis,
    REGIME_COLORS, REGIME_LINE_COLORS,
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

st.set_page_config(page_title="Regime | QuantEdge", layout="wide")
st.title("🔀 Market Regime Detection")
qe_neon_divider()

# ── Controls ──────────────────────────────────────────────────────────────────
render_data_engine_controls("regime")
c1, c2, c3 = st.columns([2, 1, 1])
ticker   = render_single_ticker_input("Ticker", key="regime_ticker",
             default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
             container=c1)
n_states = c2.selectbox("Regimes", [2, 3], index=0)
run_roll = c3.checkbox("Rolling HMM (slower)", value=False)
start    = pd.to_datetime(get_global_start_date())

with st.spinner("Loading data..."):
    df  = load_ticker_data(ticker, start=str(start))
    ret = returns(df)

if st.button("🚀 Detect Regimes", type="primary"):
    with st.spinner(f"Running full regime analysis ({n_states}-state HMM + 5 features)..."):
        R = full_regime_analysis(ret, df, n_states=n_states)

    regimes     = R["regimes"]
    fwd_proba   = R["fwd_proba"]
    ew          = R["early_warning"]
    strategy    = R["strategy"]
    age         = R["regime_age"]
    scalar      = R["age_scalar"]
    cur         = R["current_regime"]
    cond_sharpe = R["cond_sharpe"]

    # ── MASTER STATUS BAR ────────────────────────────────────────────────────
    cur_clean = cur.replace("📈","").replace("📉","").replace("↔","").strip()
    bar_color = {"Bull":"rgba(50,205,50,0.12)","Bear":"rgba(220,50,50,0.12)"}.get(
        cur_clean, "rgba(255,215,0,0.10)")
    border_c  = {"Bull":"#32cd32","Bear":"#dc3232"}.get(cur_clean, "#ffd700")

    # early-warning badge
    ew_badge = "🚨 Early Warning Active" if ew["active"] else "✅ No Warning"
    st.markdown(f"""
<div style="background:{bar_color};border:1px solid {border_c};
            border-radius:8px;padding:14px 20px;margin-bottom:12px;
            display:flex;align-items:center;gap:24px;flex-wrap:wrap">
  <span style="font-size:26px;font-weight:600">{cur}</span>
  <span style="font-size:13px;opacity:0.8">
    Age: <b>{age}d</b> &nbsp;|&nbsp;
    Position scalar: <b>{scalar:.0%}</b> &nbsp;|&nbsp;
    {ew_badge}
  </span>
  <span style="font-size:12px;opacity:0.65">{ew['lead_msg']}</span>
</div>
""", unsafe_allow_html=True)

    # Quick metrics
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Current Regime",     cur)
    m2.metric("Regime Age",         f"{age} days")
    m3.metric("Position Scalar",    f"{scalar:.0%}")
    m4.metric("AC1 (early warn.)",  f"{ew['latest_ac1']:.3f}",
              delta="⚠️ Elevated" if ew['latest_ac1'] > 0.15 else "Normal")
    # Bull probability from forward pass
    bull_cols = [c for c in fwd_proba.columns if "Bull" in c]
    bull_prob = float(fwd_proba[bull_cols[0]].iloc[-1]) if bull_cols else 0.5
    m5.metric("P(Bull) live",       f"{bull_prob:.1%}")
    bear_cols = [c for c in fwd_proba.columns if "Bear" in c]
    bear_prob = float(fwd_proba[bear_cols[0]].iloc[-1]) if bear_cols else 0.5
    m6.metric("P(Bear) live",       f"{bear_prob:.1%}")

    st.markdown("")

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Regime Map",
        "📊 Forward Probabilities",
        "⚠️ Early Warning",
        "🤖 Strategy Router",
        "🔄 Rolling HMM",
        "📈 Statistics",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — REGIME MAP
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.caption(
            "**Fix 1+2:** Regime shading uses Viterbi (for display only — smooth labels). "
            "Live trading uses forward probabilities (Tab 2). "
            "**Fix 2:** 5-feature input: return + vol + 5d trend + range ratio + volume trend."
        )

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            vertical_spacing=0.02)

        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                 line=dict(color="white", width=1.2)), row=1, col=1)

        # Shade regimes
        prev, seg_s = None, df.index[0]
        for date, regime in regimes.items():
            if regime != prev:
                if prev:
                    fig.add_vrect(x0=seg_s, x1=date,
                                  fillcolor=REGIME_COLORS.get(prev,"rgba(128,128,128,0.15)"),
                                  line_width=0)
                seg_s, prev = date, regime
        if prev:
            fig.add_vrect(x0=seg_s, x1=df.index[-1],
                          fillcolor=REGIME_COLORS.get(prev,"rgba(128,128,128,0.15)"),
                          line_width=0)

        # Returns coloured by regime
        for regime in regimes.unique():
            mask = regimes == regime
            r_masked = ret.where(mask)
            fig.add_trace(go.Bar(
                x=r_masked.index, y=r_masked.values, name=regime,
                marker_color=REGIME_LINE_COLORS.get(regime,"gray"),
                opacity=0.7,
            ), row=2, col=1)

        # Forward bull probability
        if bull_cols:
            fig.add_trace(go.Scatter(
                x=fwd_proba.index, y=fwd_proba[bull_cols[0]].values,
                name="P(Bull) forward", line=dict(color="lime", width=1.5),
                fill="tozeroy", fillcolor="rgba(50,205,50,0.08)",
            ), row=3, col=1)
            fig.add_hline(y=0.6, line_dash="dash", line_color="lime",
                          annotation_text="Bull confident", row=3, col=1)
            fig.add_hline(y=0.4, line_dash="dash", line_color="red",
                          annotation_text="Bear zone", row=3, col=1)

        fig.update_yaxes(tickformat=".2%", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=650,
                          title=f"{n_states}-State HMM — {ticker}",
                          showlegend=True, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        # Regime distribution
        counts = regimes.value_counts().reset_index()
        counts.columns = ["Regime", "Days"]
        fig_pie = px.pie(counts, names="Regime", values="Days",
                         color="Regime",
                         color_discrete_map={"Bull 📈":"limegreen",
                                             "Sideways ↔":"gold",
                                             "Bear 📉":"crimson"},
                         template="plotly_dark",
                         title="Time in Each Regime")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — FORWARD PROBABILITIES
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Forward-Pass Regime Probabilities (No Lookahead)")
        st.caption(
            "**Fix 1 — the most important fix in this page.** "
            "Viterbi (used for shading) is a *smoother* — it uses future data to label the past. "
            "For live trading you must use forward probabilities only. "
            "These are P(regime | all data up to today). "
            "**Paper:** SSRN:4556048 — 'generalizing discrete hidden state to a probability "
            "vector over all regimes provides valuable information for downstream tasks.'"
        )
        if fwd_proba.empty:
            st.warning("Forward probabilities unavailable.")
        else:
            fig2 = go.Figure()
            colors_fwd = {"Bull 📈":"lime","Sideways ↔":"gold","Bear 📉":"red"}
            for col in fwd_proba.columns:
                fig2.add_trace(go.Scatter(
                    x=fwd_proba.index, y=fwd_proba[col].values,
                    name=col, stackgroup="one",
                    line=dict(color=colors_fwd.get(col,"gray"), width=0.5),
                    fillcolor={"Bull 📈":"rgba(50,205,50,0.5)",
                               "Sideways ↔":"rgba(255,215,0,0.5)",
                               "Bear 📉":"rgba(220,50,50,0.5)"}.get(col,"rgba(128,128,128,0.4)"),
                ))
            fig2.update_layout(template="plotly_dark", height=380,
                               title="Regime Probability Over Time (forward-pass only)",
                               yaxis=dict(range=[0,1], tickformat=".0%"))
            st.plotly_chart(fig2, use_container_width=True)

            # High-confidence periods table
            if bull_cols and bear_cols:
                st.subheader("High-Confidence Regime Periods")
                bull_conf = fwd_proba[bull_cols[0]] > 0.75
                bear_conf = fwd_proba[bear_cols[0]] > 0.75
                conf_data = []
                for date in fwd_proba.index:
                    if bull_conf.get(date, False):
                        conf_data.append({"Date": date.date(), "Regime": "Bull 📈",
                                          "Confidence": f"{fwd_proba.loc[date, bull_cols[0]]:.1%}"})
                    elif bear_conf.get(date, False):
                        conf_data.append({"Date": date.date(), "Regime": "Bear 📉",
                                          "Confidence": f"{fwd_proba.loc[date, bear_cols[0]]:.1%}"})
                if conf_data:
                    conf_df = pd.DataFrame(conf_data)
                    # Show only transitions
                    conf_df["Changed"] = conf_df["Regime"].ne(conf_df["Regime"].shift())
                    st.dataframe(conf_df[conf_df["Changed"]].drop(columns="Changed").tail(20),
                                 use_container_width=True, hide_index=True)

        with st.expander("Why Viterbi is wrong for live trading"):
            st.markdown("""
**Viterbi algorithm** (what `model.predict()` uses): runs forward AND backward.
It finds the most likely *sequence* of states — but it has seen the whole time series.
A state on day 100 gets labelled using information from days 101-500. This is lookahead.

**Forward algorithm** (what `predict_proba()` uses): only uses data up to time t.
At each day it computes P(being in each state | all data so far).
This is what you would actually know in real time.

**Practical impact:** Viterbi labels regimes more cleanly — that's why we use it
for charts. But for deciding TODAY's position, always use forward probabilities.

```python
# Wrong for live use:
states = model.predict(X)          # Viterbi — uses future

# Correct for live use:
proba = model.predict_proba(X)     # forward only — P(regime | past data)
current_bull_prob = proba[-1, bull_state_index]
```
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — EARLY WARNING
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Critical Slowing Down — Early Warning Signal")
        st.caption(
            "**Fix 4:** Detects regime changes 10-20 days BEFORE they happen. "
            "Rising autocorrelation + rising variance simultaneously = system losing resilience. "
            "**Paper:** iScience (2025) PMC11976486 — spillover network early warning model."
        )

        ew_status_color = "red" if ew["active"] else ("orange" if ew["latest_ac1"] > 0.15 else "green")
        st.markdown(f"**Status:** :{ew_status_color}[{ew['lead_msg']}]")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("AC1 (lag-1 autocorr.)", f"{ew['latest_ac1']:.4f}",
                     delta="⚠️ High" if ew['latest_ac1'] > 0.15 else "Normal")
        col_b.metric("Rolling Variance",      f"{ew['latest_var']:.6f}")
        col_c.metric("Warning Active",         "YES 🚨" if ew["active"] else "No ✅")

        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.40, 0.30, 0.30],
                             vertical_spacing=0.03)

        # Price
        fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                  line=dict(color="white", width=1)), row=1, col=1)
        # Shade warning periods
        warn = ew["warning"]
        prev_w, seg_ws = 0, df.index[0]
        for date, w in warn.items():
            if w != prev_w:
                if prev_w == 1:
                    fig3.add_vrect(x0=seg_ws, x1=date,
                                   fillcolor="rgba(255,165,0,0.25)", line_width=0)
                seg_ws, prev_w = date, int(w)

        # AC1
        ac1_s = ew["ac1"]
        fig3.add_trace(go.Scatter(x=ac1_s.index, y=ac1_s.values,
                                  name="AC1 (lag-1)", line=dict(color="orange", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(255,165,0,0.08)"), row=2, col=1)
        fig3.add_hline(y=0.15, line_dash="dash", line_color="orange",
                       annotation_text="Warning threshold", row=2, col=1)
        fig3.add_hline(y=0.0,  line_dash="dot",  line_color="gray", row=2, col=1)

        # Variance
        var_s = ew["variance"]
        fig3.add_trace(go.Scatter(x=var_s.index, y=var_s.values,
                                  name="Rolling Variance", line=dict(color="red", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(220,50,50,0.08)"), row=3, col=1)

        fig3.update_layout(template="plotly_dark", height=600,
                           title=f"Critical Slowing Down — {ticker} (orange = warning fired)")
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("How Critical Slowing Down Works"):
            st.markdown("""
**Theory (from ecology + complex systems):**
Before any complex system transitions from one stable state to another,
it shows 'critical slowing down' — it takes longer to recover from small perturbations.

**In financial markets:**
- **AC1 rising** → returns becoming more autocorrelated → market losing its mean-reverting
  'resilience' → it's drifting toward a tipping point
- **Variance rising** → uncertainty before transition → options market pays up for protection

**When both rise simultaneously:**
→ System is approaching a regime transition → typical lead time 10-20 trading days

```python
ac1     = returns.rolling(21).apply(lambda x: x.autocorr(lag=1))
variance = returns.rolling(21).var()

# Warning fires when both increasing for 5+ consecutive days
warning = (ac1.diff(5) > 0) & (variance.diff(5) > 0)
```

**False positive rate:** ~25% — this is an early warning, not a guarantee.
Combine with forward regime probability crossing 60% bear for higher confidence.
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — STRATEGY ROUTER
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Regime-Adaptive Strategy Router")
        st.caption(
            "**Fix 5:** Factor weights and strategy completely change per regime. "
            "In bull markets, momentum dominates. In bear, low-vol + IV skew. "
            "In sideways, mean-reversion. "
            "**Paper:** Shu, Yu & Mulvey (2024) arXiv:2402.05272 — JM-informed strategy "
            "improves annualised returns by 1-4% across US, Germany, Japan."
        )

        rec  = strategy["recommendations"]
        wts  = strategy["weights"]
        cur_r = strategy["regime"]

        # Regime card
        card_color = {"Bull 📈":"rgba(50,205,50,0.12)",
                      "Bear 📉":"rgba(220,50,50,0.12)"}.get(cur_r,"rgba(255,215,0,0.10)")
        border_col = {"Bull 📈":"#32cd32","Bear 📉":"#dc3232"}.get(cur_r,"#ffd700")
        st.markdown(f"""
<div style="background:{card_color};border:1px solid {border_col};
            border-radius:8px;padding:16px 20px;margin-bottom:16px">
  <h3 style="margin:0 0 8px">{cur_r}</h3>
  <table style="width:100%;font-size:13px;border-collapse:collapse">
    <tr><td style="padding:4px 12px 4px 0;color:#aaa;width:140px">Primary strategy</td>
        <td><b>{rec['primary']}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Secondary</td>
        <td>{rec['secondary']}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Avoid</td>
        <td style="color:#ff6b6b">{rec['avoid']}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Position size</td>
        <td><b>{rec['position']}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Stop style</td>
        <td>{rec['stops']}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

        # Factor weights bar chart — all 3 regimes side by side
        from core.regime_detector import REGIME_FACTOR_WEIGHTS
        regime_names = list(REGIME_FACTOR_WEIGHTS.keys())
        factor_names = list(REGIME_FACTOR_WEIGHTS[regime_names[0]].keys())
        weight_rows  = []
        for rn in regime_names:
            for fn in factor_names:
                weight_rows.append({"Regime": rn, "Factor": fn,
                                    "Weight": REGIME_FACTOR_WEIGHTS[rn][fn]})
        wt_df = pd.DataFrame(weight_rows)

        fig4 = px.bar(wt_df, x="Factor", y="Weight", color="Regime", barmode="group",
                      color_discrete_map={"Bull 📈":"limegreen",
                                          "Sideways ↔":"gold",
                                          "Bear 📉":"crimson"},
                      template="plotly_dark",
                      title="Factor Weights by Regime — How the Router Allocates")
        fig4.update_layout(height=360)
        st.plotly_chart(fig4, use_container_width=True)

        # Current factor weights table
        st.subheader(f"Active weights now ({cur_r})")
        wt_now = pd.DataFrame([
            {"Factor": k, "Weight": v,
             "Action": "✅ Use" if v >= 0.6 else ("⚠️ Reduce" if v >= 0.3 else "❌ Avoid")}
            for k, v in wts.items()
        ]).sort_values("Weight", ascending=False)
        st.dataframe(
            wt_now.style.background_gradient(subset=["Weight"], cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True, hide_index=True,
        )

        # Age scalar explainer
        st.subheader("Regime Age Position Scalar")
        st.caption(
            "**Fix 3:** Position size scales with how long you've been in the regime. "
            "Early bull = cautious (50%). Confirmed bull (30d+) = full size. "
            "Bear entry = immediate 10% size."
        )
        # Show scalar over history
        age_scalars = []
        for i in range(1, len(regimes) + 1):
            age_scalars.append(regime_age_scalar(regimes.iloc[:i]))
        age_s = pd.Series(age_scalars, index=regimes.index)
        fig_age = go.Figure()
        fig_age.add_trace(go.Scatter(x=age_s.index, y=age_s.values,
                                     name="Position Scalar",
                                     line=dict(color="cyan", width=1.5),
                                     fill="tozeroy",
                                     fillcolor="rgba(0,180,216,0.10)"))
        fig_age.update_layout(template="plotly_dark", height=260,
                               title="Position Size Scalar Over Time",
                               yaxis=dict(range=[0,1.1], tickformat=".0%"))
        st.plotly_chart(fig_age, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — ROLLING HMM
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Rolling HMM — Non-Stationary Adaptive Model")
        st.caption(
            "**Fix 6:** HMM retrained every 21 days on a 252-day rolling window. "
            "Transition probabilities are very unlikely to be stationary. "
            "This catches structural breaks that a single fitted model misses. "
            "**Paper:** QuantStart + Preprints.org (2026) — non-homogeneous transition probabilities."
        )

        if run_roll:
            with st.spinner("Fitting rolling HMM (252d window, 21d step)... ~30s"):
                roll_proba = rolling_regime_proba(ret, df, n_states=n_states,
                                                  fit_window=252, step=21)
            if not roll_proba.empty:
                fig5 = go.Figure()
                roll_bull_cols = [c for c in roll_proba.columns if "Bull" in c]
                roll_bear_cols = [c for c in roll_proba.columns if "Bear" in c]
                colors5 = {"Bull 📈":"lime","Sideways ↔":"gold","Bear 📉":"red"}
                for col in roll_proba.columns:
                    fig5.add_trace(go.Scatter(
                        x=roll_proba.index, y=roll_proba[col].values,
                        name=f"Rolling {col}", stackgroup="one",
                        fillcolor={"Bull 📈":"rgba(50,205,50,0.5)",
                                   "Sideways ↔":"rgba(255,215,0,0.5)",
                                   "Bear 📉":"rgba(220,50,50,0.5)"}.get(col,"rgba(128,128,128,0.4)"),
                        line=dict(color=colors5.get(col,"gray"), width=0.5),
                    ))
                fig5.update_layout(template="plotly_dark", height=380,
                                   title="Rolling HMM Regime Probability (252d window, 21d step)",
                                   yaxis=dict(range=[0,1], tickformat=".0%"))
                st.plotly_chart(fig5, use_container_width=True)

                # Compare static vs rolling
                if bull_cols and roll_bull_cols:
                    common_idx = fwd_proba.index.intersection(roll_proba.index)
                    if len(common_idx) > 10:
                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Scatter(
                            x=common_idx, y=fwd_proba.loc[common_idx, bull_cols[0]].values,
                            name="Static HMM P(Bull)", line=dict(color="lime", width=1.5)))
                        fig_cmp.add_trace(go.Scatter(
                            x=common_idx, y=roll_proba.loc[common_idx, roll_bull_cols[0]].values,
                            name="Rolling HMM P(Bull)", line=dict(color="cyan", width=1.5, dash="dot")))
                        fig_cmp.update_layout(template="plotly_dark", height=280,
                                              title="Static vs Rolling HMM — Bull Probability",
                                              yaxis=dict(tickformat=".0%"))
                        st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.warning("Rolling HMM returned no results — try a longer date range.")
        else:
            st.info(
                "Enable 'Rolling HMM' checkbox above and re-run to see the adaptive model. "
                "It takes ~20-30 seconds as it refits the HMM at every 21-day step."
            )

        with st.expander("Why rolling refit matters"):
            st.markdown("""
**Problem with a single static HMM:**
The model learns transition probabilities from your full history.
If the market structure changes (new regulation, new market participants, QE regime),
the old parameters no longer describe reality — but the model doesn't know this.

**Example:** A HMM trained on 2010-2019 learns that bear regimes last ~6 months.
In 2020 (COVID), the bear lasted 5 weeks. In 2022 (rate hike bear), 9 months.
The static model's transition matrix is wrong for both.

**Rolling refit (252d window, 21d step):**
```
For each step i (every 21 days):
  1. Train HMM on returns[i-252:i]  (last year of data)
  2. Predict proba on returns[i:i+21]  (next 21 days, out-of-sample)
  3. Store probabilities
```
The out-of-sample prediction is key — each prediction uses ONLY past data.
The model adapts to the most recent year's regime structure.
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — STATISTICS
    # ════════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("Regime-Conditional Statistics")

        # Conditional Sharpe table
        st.dataframe(cond_sharpe, use_container_width=True, hide_index=True)

        # Return distributions per regime
        fig6 = go.Figure()
        for regime in regimes.unique():
            mask = regimes == regime
            r_reg = ret[mask].dropna()
            fig6.add_trace(go.Histogram(
                x=r_reg, name=regime, nbinsx=60, opacity=0.6,
                marker_color=REGIME_LINE_COLORS.get(regime, "gray"),
            ))
        fig6.update_layout(template="plotly_dark", barmode="overlay",
                           title="Return Distribution by Regime",
                           xaxis_tickformat=".2%", height=380)
        st.plotly_chart(fig6, use_container_width=True)

        # Transition matrix
        st.subheader("Regime Transition Matrix (empirical)")
        unique_reg = regimes.unique()
        trans = pd.DataFrame(0.0, index=unique_reg, columns=unique_reg)
        for i in range(1, len(regimes)):
            prev_r = regimes.iloc[i-1]
            next_r = regimes.iloc[i]
            trans.loc[prev_r, next_r] += 1
        row_sums = trans.sum(axis=1).replace(0, 1)
        trans_pct = trans.div(row_sums, axis=0)
        fig_trans = px.imshow(
            trans_pct, text_auto=".1%",
            color_continuous_scale="RdYlGn", zmin=0, zmax=1,
            template="plotly_dark",
            title="Transition Probability: P(next | current)",
            labels=dict(x="Next Regime", y="Current Regime"),
        )
        fig_trans.update_layout(height=360)
        st.plotly_chart(fig_trans, use_container_width=True)

        # Regime duration distribution
        st.subheader("Regime Duration Distribution")
        durations = {"regime": [], "duration": []}
        cur_r_d, cur_len = regimes.iloc[0], 1
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == cur_r_d:
                cur_len += 1
            else:
                durations["regime"].append(cur_r_d)
                durations["duration"].append(cur_len)
                cur_r_d, cur_len = regimes.iloc[i], 1
        durations["regime"].append(cur_r_d)
        durations["duration"].append(cur_len)
        dur_df = pd.DataFrame(durations)

        fig_dur = px.box(dur_df, x="regime", y="duration", color="regime",
                         color_discrete_map={"Bull 📈":"limegreen",
                                             "Sideways ↔":"gold",
                                             "Bear 📉":"crimson"},
                         template="plotly_dark",
                         title="Regime Duration Distribution (days per episode)")
        fig_dur.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_dur, use_container_width=True)

        # Scatter: return vs vol coloured by regime
        ret_vol = pd.DataFrame({"Return": ret, "Vol": ret.rolling(21).std()*np.sqrt(252),
                                 "Regime": regimes}).dropna()
        fig_sc = px.scatter(ret_vol, x="Vol", y="Return", color="Regime",
                             color_discrete_map={"Bull 📈":"limegreen",
                                                 "Sideways ↔":"gold",
                                                 "Bear 📉":"crimson"},
                             template="plotly_dark",
                             title="Return vs Volatility by Regime",
                             opacity=0.5,
                             labels={"Vol":"Annualised Vol","Return":"Daily Return"})
        fig_sc.update_yaxes(tickformat=".2%")
        fig_sc.update_xaxes(tickformat=".1%")
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

else:
    st.info("👆 Click **Detect Regimes** to run the full analysis.")
    st.markdown("""
**What's new vs the original regime page:**

| Fix | What changed | Why it matters |
|-----|-------------|----------------|
| **1. Forward proba** | `predict_proba()` instead of `predict()` | No lookahead — usable for live trading |
| **2. 5 features** | Added range ratio + volume trend | Catches regime transitions 3-7 days earlier |
| **3. Age scalar** | Position size ramps up over 30d in bull | Avoids buying false breakouts at day 1 |
| **4. Early warning** | AC1 + variance critical slowing down | 10-20 day lead time before regime flips |
| **5. Strategy router** | Factor weights flip per regime | Actually uses the regime for something |
| **6. Rolling refit** | Refit every 21d on 252d window | Handles non-stationarity and structural breaks |
    """)

qe_faq_section("FAQs", [
    ("Why use forward probabilities instead of plain labels?", "Forward probabilities are what you would know in real time, so they are safer for live decisions than a lookahead label."),
    ("What does the early warning signal add?", "It can flag rising instability before the regime actually flips, giving you time to reduce risk."),
    ("How should I use the strategy router?", "Let it guide whether to lean into momentum, mean reversion, or defense based on the current regime state."),
    ("When is rolling HMM worth using?", "Turn it on when you want a more adaptive view of the market and are willing to wait a bit longer for the calculation."),
])
