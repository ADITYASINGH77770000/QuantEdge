import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.data import get_ohlcv, get_multi_ohlcv, returns
from app.data_engine import (
    render_data_engine_controls, render_single_ticker_input,
    load_ticker_data, get_global_start_date, parse_ticker_list,
)
from core.indicators import (
    add_all_indicators, signal_rsi, signal_macd_crossover,
    signal_bb_mean_reversion, signal_dual_ma,
)
from core.metrics import information_coefficient
from core.alpha_engine import (
    compute_ofi, ofi_signal,
    compute_crowding_score, crowding_weight, crowding_signal,
    compute_iv_skew_proxy, iv_skew_signal, get_real_iv_skew,
    compute_signal_health, monitor_all_signals,
    get_macro_data, compute_macro_regime_score, macro_regime_label,
    combine_signals,
)
from utils.config import cfg

st.set_page_config(page_title="Signals | QuantEdge", layout="wide")
st.title("📡 Unified Alpha Signal Dashboard")
st.caption(
    "Volume Pressure · Crowding · Realized Skew · Signal Health · Macro Regime · IC-Weighted Combined Signal"
)

# ── Controls ──────────────────────────────────────────────────────────────────
render_data_engine_controls("signals")
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
ticker  = render_single_ticker_input("Ticker", key="sig_ticker",
            default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
            container=c1)
fwd_days = c2.slider("IC Forward Window (days)", 1, 21, 5)
ofi_thresh  = c3.slider("OFI threshold", 0.3, 2.0, 0.8, 0.1)
skew_thresh = c4.slider("Skew threshold", 0.3, 2.0, 0.7, 0.1)
start = pd.to_datetime(get_global_start_date())

with st.spinner("Loading data & computing all signals..."):
    df     = load_ticker_data(ticker, start=str(start))
    df_ind = add_all_indicators(df)
    ret    = returns(df)
    fwd    = ret.shift(-fwd_days)

# ── Compute all signals ───────────────────────────────────────────────────────
ofi_z    = compute_ofi(df)
ofi_sig  = ofi_signal(df, threshold=ofi_thresh)
skew_z   = compute_iv_skew_proxy(df)
skew_sig = iv_skew_signal(df, threshold=skew_thresh)
crowd    = compute_crowding_score(ret)
crowd_w  = crowding_weight(ret)

# Classic signals
rsi_sig  = signal_rsi(df_ind)
macd_sig = signal_macd_crossover(df_ind)
bb_sig   = signal_bb_mean_reversion(df_ind)
dma_sig  = signal_dual_ma(df_ind, 20, 50)

all_signals = {
    "RSI":          rsi_sig,
    "MACD":         macd_sig,
    "BB Reversion": bb_sig,
    "Dual MA":      dma_sig,
    "OFI":          ofi_sig,
    "Realized Skew":      skew_sig,
}

# IC-weighted combined
combined_sig, ic_weights = combine_signals(all_signals, ret, fwd_days=fwd_days)

# ── MASTER SIGNAL HEADER ──────────────────────────────────────────────────────
latest_combined = int(combined_sig.dropna().iloc[-1]) if not combined_sig.dropna().empty else 0
latest_ofi      = float(ofi_z.dropna().iloc[-1])      if not ofi_z.dropna().empty      else 0.0
latest_skew     = float(skew_z.dropna().iloc[-1])     if not skew_z.dropna().empty     else 0.0
latest_crowd    = float(crowd.dropna().iloc[-1])      if not crowd.dropna().empty      else 1.0

signal_color = {"🟢 BUY": "green", "🔴 SELL": "red", "⚪ HOLD": "gray"}
master_label = "🟢 BUY" if latest_combined == 1 else ("🔴 SELL" if latest_combined == -1 else "⚪ HOLD")

st.markdown(f"""
<div style="background:{'rgba(0,200,80,0.12)' if latest_combined==1 else ('rgba(220,50,50,0.12)' if latest_combined==-1 else 'rgba(120,120,120,0.08)')};
            border:1px solid {'#00c850' if latest_combined==1 else ('#dc3232' if latest_combined==-1 else '#666')};
            border-radius:8px;padding:14px 20px;margin-bottom:12px;display:flex;align-items:center;gap:20px">
  <span style="font-size:28px;font-weight:600">{master_label}</span>
  <span style="font-size:13px;opacity:0.7">IC-weighted combination of {len(all_signals)} signals · fwd window = {fwd_days}d · crowding weight = {crowd_w:.2f}x</span>
</div>
""", unsafe_allow_html=True)

# Quick metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Volume Pressure z-score",     f"{latest_ofi:.2f}", delta="Buy pressure" if latest_ofi > 0.5 else ("Sell pressure" if latest_ofi < -0.5 else "Neutral"))
m2.metric("Realized Skew z",       f"{latest_skew:.2f}", delta="Fear" if latest_skew < -0.5 else ("Calm" if latest_skew > 0.5 else "Neutral"))
m3.metric("Crowding",        f"{latest_crowd:.2f}x", delta="⚠️ Crowded" if latest_crowd > 1.3 else "Normal")
m4.metric("Crowd Weight",    f"{crowd_w:.0%}")
m5.metric("Active Signals",  sum(1 for v in ic_weights.values() if v > 0))
m6.metric("Best IC Weight",  f"{max(ic_weights.values(), default=0):.4f}")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Combined Signal",
    "🌊 Volume Pressure (OFI Proxy)",
    "😰 Realized Skew (IV Proxy)",
    "🏭 Crowding",
    "❤️ Signal Health",
    "🌍 Macro Regime",
])

# ── TAB 1: Combined Signal ────────────────────────────────────────────────────
with tab1:
    st.subheader("IC-Weighted Combined Signal")
    st.caption(
        "Each signal is weighted by its Spearman IC against forward returns. "
        "Signals with negative IC are automatically excluded. "
        "Final signal is the IC-weighted average, discretised."
    )

    # IC weights bar chart
    wt_df = pd.DataFrame([
        {"Signal": k, "IC Weight": v, "Active": "Yes" if v > 0 else "No"}
        for k, v in ic_weights.items()
    ]).sort_values("IC Weight", ascending=False)

    fig_wt = px.bar(wt_df, x="Signal", y="IC Weight", color="Active",
                    color_discrete_map={"Yes": "#00c850", "No": "#888"},
                    template="plotly_dark",
                    title=f"IC Weights — {fwd_days}d Forward Return")
    fig_wt.update_layout(height=280, showlegend=True)
    st.plotly_chart(fig_wt, use_container_width=True)

    # Combined signal on price chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="white", width=1.2)), row=1, col=1)

    # align combined signal to the price frame index to avoid length mismatch
    comb_aligned = combined_sig.reindex(df.index, fill_value=0)
    buys  = df.index[comb_aligned ==  1]
    sells = df.index[comb_aligned == -1]
    if len(buys):
        fig.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Close"], mode="markers",
                                 name="Combined Buy",
                                 marker=dict(symbol="triangle-up", size=9, color="lime")), row=1, col=1)
    if len(sells):
        fig.add_trace(go.Scatter(x=sells, y=df.loc[sells, "Close"], mode="markers",
                                 name="Combined Sell",
                                 marker=dict(symbol="triangle-down", size=9, color="red")), row=1, col=1)

    # Signal numeric plot
    fig.add_trace(go.Scatter(x=combined_sig.index, y=combined_sig.values,
                             name="Signal", line=dict(color="cyan", width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520,
                      title=f"IC-Weighted Combined Signal — {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative IC scatter
    ic_vals = {}
    for name, sig in all_signals.items():
        common = sig.dropna().index.intersection(fwd.dropna().index)
        if len(common) > 20:
            ic_vals[name] = round(information_coefficient(
                sig[common].astype(float), fwd[common]), 4)

    if ic_vals:
        st.subheader(f"IC vs {fwd_days}d Forward Returns")
        ic_cmp = pd.DataFrame([{"Signal": k, "IC": v,
                                 "Used": "✅" if v > 0 else "❌"}
                                for k, v in ic_vals.items()])
        st.dataframe(ic_cmp.sort_values("IC", ascending=False),
                     use_container_width=True, hide_index=True)

# ── TAB 2: OFI ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Volume Pressure (OFI Proxy)")
    st.caption(
        "**What it is:** OFI measures net buying vs selling pressure, normalised by volume. "
        "Positive = institutions accumulating. Negative = distribution. "
        "**Paper:** Kolm et al. (2023) — Deep order flow imbalance, Mathematical Finance. "
        "Market-cap normalisation from arxiv:2512.18648 (2025) shows 30%+ Sharpe improvement."
    )

    ofi_ic = information_coefficient(
        ofi_sig.reindex(fwd.dropna().index).fillna(0).astype(float),
        fwd.dropna()) if len(fwd.dropna()) > 20 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Volume Pressure z (latest)", f"{latest_ofi:.3f}")
    c2.metric("Volume Pressure Signal", "BUY" if latest_ofi > ofi_thresh else ("SELL" if latest_ofi < -ofi_thresh else "HOLD"))
    c3.metric(f"Volume Pressure IC ({fwd_days}d)", f"{ofi_ic:.4f}", delta="Signal" if abs(ofi_ic) > 0.05 else "Noise")

    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.03)
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=ofi_z.index, y=ofi_z.values, name="Volume Pressure z-score",
                              line=dict(color="cyan", width=1.5),
                              fill="tozeroy", fillcolor="rgba(0,180,216,0.12)"), row=2, col=1)
    fig2.add_hline(y= ofi_thresh, line_dash="dash", line_color="lime", row=2, col=1)
    fig2.add_hline(y=-ofi_thresh, line_dash="dash", line_color="red",  row=2, col=1)

    # Buy volume vs sell volume proxy
    close_, open_ = df["Close"], df["Open"]
    vol = df["Volume"]
    buy_v  = vol.where(close_ >= open_, 0)
    sell_v = vol.where(close_ <  open_, 0)
    fig2.add_trace(go.Bar(x=df.index, y=buy_v,  name="Buy Vol",  marker_color="rgba(0,200,80,0.6)"),  row=3, col=1)
    fig2.add_trace(go.Bar(x=df.index, y=-sell_v, name="Sell Vol", marker_color="rgba(220,50,50,0.6)"), row=3, col=1)
    fig2.update_layout(template="plotly_dark", height=560,
                       title=f"Volume Pressure Signal — {ticker}", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("⚠️ What This Signal Actually Is"):
        st.markdown("""
**⚠️ Naming Disclosure:** True Order Flow Imbalance (Kolm 2023) requires Level-2 tick data
(bid/ask queue imbalance at each price level). This signal uses daily OHLCV as a proxy.
It is predictive, but it is **not** the same as L2 OFI. The UI labels it "Volume Pressure (OFI Proxy)".

**Formula:**
```
buy_vol  = volume on up-close bars (Close >= Open)
sell_vol = volume on down-close bars (Close < Open)
raw      = buy_vol - sell_vol
norm     = raw / rolling_avg_volume   ← market-cap normalisation (Kolm 2023)
signal   = z-score(norm, window=63)
```

Signal: `z > threshold → BUY`, `z < -threshold → SELL`
        """)

# ── TAB 3: Realized Skew ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Realized Skew Signal (IV Skew Proxy)")
    st.caption(
        "⚠️ **Proxy disclosure:** True IV skew needs options chain data (put IV − call IV). "
        "This signal uses realized return skewness + down-day frequency as a proxy for the same fear "
        "the options market is pricing. The signal is valid but is **not** implied volatility. "
        "**References:** Höfler (2024) SSRN 4869272; Bakshi, Kapadia & Madan (2003)."
    )

    # Try live IV
    live_iv = get_real_iv_skew(ticker)
    if live_iv:
        st.success(f"✅ Live options data available for {ticker}")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Put IV (avg)",  f"{live_iv['put_iv']:.1%}")
        lc2.metric("Call IV (avg)", f"{live_iv['call_iv']:.1%}")
        lc3.metric("Realized Skew",       f"{live_iv['skew']:.4f}",
                   delta="Fear" if live_iv['skew'] > 0.05 else "Normal")
        lc4.metric("ATM IV",        f"{live_iv['atm_iv']:.1%}")
    else:
        st.info(
            "Live options data unavailable in current environment. "
            "Using OHLC-based skew proxy (return skewness + down-day ratio). "
            "On your local machine with internet, this will show real put/call IV."
        )

    skew_ic = information_coefficient(
        skew_sig.reindex(fwd.dropna().index).fillna(0).astype(float),
        fwd.dropna()) if len(fwd.dropna()) > 20 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Skew z (latest)",    f"{latest_skew:.3f}")
    c2.metric("Signal",             "SELL (fear)" if latest_skew < -skew_thresh else ("BUY (calm)" if latest_skew > skew_thresh else "HOLD"))
    c3.metric(f"Skew IC ({fwd_days}d)", f"{skew_ic:.4f}")

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.55, 0.45], vertical_spacing=0.04)
    fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    if len(buys):
        fig3.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Close"], mode="markers",
                                  name="Calm (Buy)",
                                  marker=dict(symbol="triangle-up", size=8, color="lime")), row=1, col=1)
    skew_sells = df.index[skew_sig == -1]
    if len(skew_sells):
        fig3.add_trace(go.Scatter(x=skew_sells, y=df.loc[skew_sells, "Close"], mode="markers",
                                  name="Fear (Sell)",
                                  marker=dict(symbol="triangle-down", size=8, color="orange")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=skew_z.index, y=skew_z.values, name="Skew z-score",
                              line=dict(color="orange", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,165,0,0.10)"), row=2, col=1)
    fig3.add_hline(y= skew_thresh, line_dash="dash", line_color="lime",   row=2, col=1)
    fig3.add_hline(y=-skew_thresh, line_dash="dash", line_color="orange", row=2, col=1)
    fig3.update_layout(template="plotly_dark", height=500,
                       title=f"Realized Skew Signal (IV Proxy) — {ticker}")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("How IV Skew Works"):
        st.markdown("""
**Real signal (with options data):**
```
skew = avg(put_IV) - avg(call_IV)
high positive skew → market buying puts → fear → SELL
negative skew      → market buying calls → calm → BUY
```
**Proxy signal (OHLC-based):**
```
return_skew = rolling 21-day skewness of returns
fear_proxy  = fraction of down-days (close < open)
combined    = (return_skew + fear_proxy) / 2  →  z-scored
```
Höfler (2024) shows IV surface signals survive transaction costs and are strongest
when financial intermediaries are constrained (e.g. after a market dislocation).
        """)

# ── TAB 4: Crowding ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Factor Crowding Detector")
    st.caption(
        "**What it is:** Detects when a factor/stock has too many funds in the same trade. "
        "Crowded trades unwind violently — like August 2007 and July 2025. "
        "**Paper:** Hua & Sun (2024) — Dynamics of Factor Crowding, SSRN 5023380. "
        "Falck, Rej & Thesmar (2022) — crowding as a systematic risk factor."
    )

    crowd_status = "🔴 Overcrowded" if latest_crowd > 1.3 else ("🟡 Elevated" if latest_crowd > 1.1 else "🟢 Normal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crowding Score", f"{latest_crowd:.3f}", delta="Danger zone" if latest_crowd > 1.3 else "OK")
    c2.metric("Status", crowd_status)
    c3.metric("Position Weight Scalar", f"{crowd_w:.0%}")
    c4.metric("Suggested Action", "Reduce 75%" if latest_crowd > 1.3 else ("Reduce 35%" if latest_crowd > 1.1 else "Full size"))

    # Crowding time series
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5], vertical_spacing=0.04)
    fig4.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=crowd.index, y=crowd.values, name="Crowding Score",
                              line=dict(color="yellow", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"), row=2, col=1)
    fig4.add_hline(y=1.3, line_dash="dash", line_color="red",
                   annotation_text="Overcrowded threshold", row=2, col=1)
    fig4.add_hline(y=1.1, line_dash="dot", line_color="orange", row=2, col=1)
    fig4.add_hline(y=0.8, line_dash="dot", line_color="lime",
                   annotation_text="Undercrowded", row=2, col=1)
    fig4.update_layout(template="plotly_dark", height=500,
                       title=f"Crowding Score — {ticker}")
    st.plotly_chart(fig4, use_container_width=True)

    # Multi-ticker crowding
    st.subheader("Multi-Ticker Crowding Comparison")
    port_raw = st.text_input("Tickers", value=", ".join(cfg.DEFAULT_TICKERS[:4]),
                              key="crowd_tickers")
    port_tickers = parse_ticker_list(port_raw)
    if len(port_tickers) >= 2:
        with st.spinner("Loading multi-ticker data..."):
            from core.data import get_multi_ohlcv
            multi = get_multi_ohlcv(port_tickers, start=str(start.date()))
        crowd_df = crowding_signal(multi)
        if not crowd_df.empty:
            fig_crowd = px.bar(crowd_df, x="Ticker", y="Crowding Score",
                               color="Status",
                               color_discrete_map={"🔴 Overcrowded": "red",
                                                   "🟡 Elevated": "orange",
                                                   "🟢 Normal": "green"},
                               template="plotly_dark",
                               title="Crowding Score by Ticker")
            fig_crowd.add_hline(y=1.3, line_dash="dash", line_color="red")
            fig_crowd.update_layout(height=320)
            st.plotly_chart(fig_crowd, use_container_width=True)
            st.dataframe(crowd_df, use_container_width=True, hide_index=True)

    with st.expander("How Crowding Detection Works"):
        st.markdown("""
**Crowding proxy formula:**
```
crowding_score = short_window_vol(21d) / long_window_vol(126d)

score > 1.3 → overcrowded → reduce position by 75%
score 1.1-1.3 → elevated → reduce by 35%
score < 1.0 → normal → full position

position_scalar = clip(1.0 - (score - 0.8) / 0.5 * 0.75, 0.25, 1.0)
```
**Why this works:** When many funds pile into the same factor, their correlated
trading compresses volatility temporarily. When they exit — simultaneously —
vol spikes. The crowding score detects the compression BEFORE the exit.

Hua & Sun (2024) show crowding vulnerability varies by factor — factors with
low barriers to entry are most susceptible. Momentum and size are most at risk.
        """)

# ── TAB 5: Signal Health ──────────────────────────────────────────────────────
with tab5:
    st.subheader("Signal Health & Alpha Decay Monitor")
    st.caption(
        "**What it is:** Real-time detector of whether each signal is still working. "
        "Measures IC trend over time — negative slope = signal decaying. "
        "**Paper:** AlphaAgent (KDD 2025) — regularization to counteract alpha decay. "
        "Harvey, Liu & Zhu (2016) — most discovered factors are false positives."
    )

    with st.spinner("Computing signal health for all signals..."):
        health_df = monitor_all_signals(df, fwd_days=fwd_days)

    # Health gauge bars
    fig_h = px.bar(health_df, x="Signal", y="Health", color="Health",
                   color_continuous_scale=["red", "orange", "yellow", "green"],
                   range_color=[0, 100],
                   template="plotly_dark",
                   title=f"Signal Health Scores — {ticker} ({fwd_days}d forward)")
    fig_h.add_hline(y=75, line_dash="dash", line_color="lime",
                    annotation_text="Healthy threshold (75)")
    fig_h.add_hline(y=25, line_dash="dash", line_color="red",
                    annotation_text="Kill threshold (25)")
    fig_h.update_layout(height=320)
    st.plotly_chart(fig_h, use_container_width=True)

    # Detailed health table
    st.dataframe(
        health_df.style
            .background_gradient(subset=["Health"], cmap="RdYlGn", vmin=0, vmax=100)
            .format({"Health": "{:.1f}", "IC Mean": "{:.4f}",
                     "IC Std": "{:.4f}", "IC Trend": "{:.6f}", "Weight": "{:.0%}"}),
        use_container_width=True, hide_index=True
    )

    # Rolling IC for best signal
    best_sig_name = health_df.iloc[0]["Signal"]
    best_sig      = all_signals.get(best_sig_name, rsi_sig)
    h_detail = compute_signal_health(best_sig, fwd, window=63)
    if "rolling_ic" in h_detail and not h_detail["rolling_ic"].empty:
        ic_ts = h_detail["rolling_ic"]
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(y=ic_ts.values, name=f"{best_sig_name} Rolling IC",
                                    line=dict(color="cyan", width=1.5),
                                    fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"))
        fig_ic.add_hline(y=0.05, line_dash="dash", line_color="lime",
                         annotation_text="IC = 0.05 (meaningful)")
        fig_ic.add_hline(y=0,    line_dash="dash", line_color="gray")
        # Trend line
        x = np.arange(len(ic_ts))
        slope, intercept = np.polyfit(x, ic_ts.fillna(0).values, 1)
        trend = slope * x + intercept
        fig_ic.add_trace(go.Scatter(y=trend, name="IC Trend",
                                    line=dict(color="orange", width=2, dash="dot")))
        fig_ic.update_layout(template="plotly_dark", height=320,
                              title=f"Rolling 63d IC — {best_sig_name} on {ticker}",
                              xaxis_title="Time (rolling windows)")
        st.plotly_chart(fig_ic, use_container_width=True)

    with st.expander("Health Score Formula"):
        st.markdown("""
```
ic_score    = min(100, |IC_mean| / 0.08 * 50)   # magnitude
std_score   = min(30,  (0.15 - IC_std) / 0.15 * 30)  # stability
trend_score = min(20,  (IC_trend + 0.001) / 0.002 * 20)  # momentum

health = ic_score + std_score + trend_score  (max 100)

≥75 → 🟢 Healthy → use full weight
≥50 → 🟡 Moderate → use 65% weight
≥25 → 🟠 Weak → use 30% weight
 <25 → 🔴 Decaying → stop trading this signal
```
        """)

# ── TAB 6: Macro Regime ───────────────────────────────────────────────────────
with tab6:
    st.subheader("Cross-Asset Macro Regime Signal")
    st.caption(
        "**What it is:** Combines VIX, credit spreads, dollar index, and yield curve "
        "into one macro regime score. Tells you whether to be aggressive or defensive. "
        "**Why unsolved:** Correlations between assets are non-stationary — "
        "what worked in 2018 breaks in 2022. (Future Alpha 2025)"
    )

    with st.spinner("Fetching macro data (VIX, HYG, IEF, DXY, rates)..."):
        macro_df = get_macro_data(start=str(start.date()))

    if macro_df is not None and not macro_df.empty:
        macro_score = compute_macro_regime_score(macro_df)
        latest_macro = float(macro_score.dropna().iloc[-1]) if not macro_score.dropna().empty else 0.0
        regime_label, pos_scalar = macro_regime_label(latest_macro)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Macro Score",      f"{latest_macro:.2f}")
        mc2.metric("Regime",           regime_label)
        mc3.metric("Position Scalar",  f"{pos_scalar:.0%}")
        mc4.metric("VIX (latest)",
                   f"{macro_df['vix'].iloc[-1]:.1f}" if "vix" in macro_df.columns else "N/A")

        # Macro score time series
        fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.5, 0.5], vertical_spacing=0.04)
        fig_m.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Price",
                                   line=dict(color="white", width=1)), row=1, col=1)
        common_m = macro_score.index.intersection(df.index)
        if len(common_m):
            fig_m.add_trace(go.Scatter(x=common_m,
                                       y=macro_score.reindex(common_m).values,
                                       name="Macro Score",
                                       line=dict(color="magenta", width=2),
                                       fill="tozeroy",
                                       fillcolor="rgba(200,0,200,0.08)"), row=2, col=1)
        fig_m.add_hline(y= 1.0, line_dash="dash", line_color="lime",
                        annotation_text="Risk-On",  row=2, col=1)
        fig_m.add_hline(y=-1.0, line_dash="dash", line_color="red",
                        annotation_text="Risk-Off", row=2, col=1)
        fig_m.add_hline(y=0,    line_dash="dot",  line_color="gray", row=2, col=1)
        fig_m.update_layout(template="plotly_dark", height=520,
                             title="Cross-Asset Macro Regime Score")
        st.plotly_chart(fig_m, use_container_width=True)

        # Component breakdown
        st.subheader("Component Breakdown (latest values)")
        comp_rows = []
        if "vix"    in macro_df.columns: comp_rows.append({"Asset": "VIX",          "Latest": round(macro_df["vix"].iloc[-1],   2), "Effect": "Risk-Off when high"})
        if "credit" in macro_df.columns: comp_rows.append({"Asset": "HYG (Credit)", "Latest": round(macro_df["credit"].iloc[-1],2), "Effect": "Risk-On when high"})
        if "bonds"  in macro_df.columns: comp_rows.append({"Asset": "IEF (Bonds)",  "Latest": round(macro_df["bonds"].iloc[-1], 2), "Effect": "Safety when high"})
        if "dollar" in macro_df.columns: comp_rows.append({"Asset": "DXY (Dollar)", "Latest": round(macro_df["dollar"].iloc[-1],2), "Effect": "Risk-Off for equities when high"})
        if "rates"  in macro_df.columns: comp_rows.append({"Asset": "10yr Yield",   "Latest": round(macro_df["rates"].iloc[-1], 2), "Effect": "Ambiguous (growth vs tightening)"})
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    else:
        st.warning(
            "Macro data (VIX, HYG, IEF, DXY) unavailable in current environment "
            "(network restricted). On your local machine this will fetch live data "
            "from yfinance and show the full cross-asset regime score."
        )
        st.markdown("""
**What this shows when live:**
- **VIX** — fear gauge. High VIX = risk-off
- **HYG/IEF ratio** — credit spread proxy. Low ratio = stress = risk-off
- **DXY** — dollar strength. Strong dollar = risk-off for equities
- **10yr yield** — rate environment. Rising rates = ambiguous signal

**Score interpretation:**
```
> +1.0  → Risk-On  → use full position size
+0.25 to +1.0 → Mild risk-on → 75% size
-0.25 to +0.25 → Neutral → 50% size
-1.0 to -0.25 → Mild risk-off → 35% size
< -1.0  → Risk-Off → 20% size (or flat)
```
        """)

    with st.expander("How Cross-Asset Regime Works"):
        st.markdown("""
**Why single-asset regime detection is not enough:**
Your existing HMM looks at equity returns only. But the real signal comes before
equity moves — from credit spreads widening, dollar strengthening, or VIX spiking.

**Formula:**
```python
vix_component    = -zscore(VIX)          # high VIX = risk-off = negative
credit_component = zscore(HYG/IEF)       # high ratio = risk-on = positive
dollar_component = -zscore(DXY)          # strong $ = risk-off for equities
rates_component  = zscore(Δ10yr) * 0.5  # ambiguous → half weight

macro_score = mean(components).clip(-2, 2)
```
**What's unsolved:** These correlations are non-stationary. The dollar was risk-off
in 2015-2018 but partially risk-on in 2022. HYG/equity correlation broke down in 2023.
A regime-adaptive version that detects correlation structure changes is the frontier.
        """)