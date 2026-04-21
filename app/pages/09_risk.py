import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import core.metrics as metrics
from core.data import get_ohlcv, get_multi_ohlcv, align_returns, returns
from app.data_engine import (
    render_data_engine_controls,
    render_single_ticker_input,
    load_ticker_data,
    get_global_start_date,
    get_data_engine_settings,
    parse_ticker_list,
)
from utils.charts import metric_card_row
from utils.config import cfg

var_historical = metrics.var_historical
cvar_historical = metrics.cvar_historical
var_parametric = metrics.var_parametric
var_t_dist = getattr(metrics, "var_t_dist", metrics.var_parametric)
var_garch = getattr(metrics, "var_garch", metrics.var_historical)
summary_table = metrics.summary_table
annualised_vol = metrics.annualised_vol
max_drawdown = metrics.max_drawdown


def _fallback_portfolio_var(
    returns_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    returns_df = returns_df.dropna()
    if returns_df.empty:
        return 0.0
    n_assets = returns_df.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    weights = np.array(weights, dtype=float)
    portfolio_returns = pd.Series(returns_df.values @ weights, index=returns_df.index)
    if method == "garch":
        return var_garch(portfolio_returns, confidence)
    if method == "parametric":
        return var_parametric(portfolio_returns, confidence)
    return var_historical(portfolio_returns, confidence)


def _fallback_kupiec_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> dict:
    aligned_returns, aligned_var = returns.align(var_series, join="inner")
    mask = aligned_returns.notna() & aligned_var.notna()
    aligned_returns = aligned_returns[mask]
    aligned_var = aligned_var[mask]
    if len(aligned_returns) < 30:
        return {
            "violations": 0,
            "expected_rate": 1 - confidence,
            "actual_rate": float("nan"),
            "p_value": float("nan"),
            "result": "Insufficient data",
        }
    violations = int((aligned_returns < aligned_var).sum())
    total = len(aligned_returns)
    expected_rate = 1 - confidence
    actual_rate = violations / total
    return {
        "violations": violations,
        "expected_rate": expected_rate,
        "actual_rate": round(actual_rate, 4),
        "p_value": float("nan"),
        "result": "Fallback model check",
    }


portfolio_var = getattr(metrics, "portfolio_var", _fallback_portfolio_var)
kupiec_test = getattr(metrics, "kupiec_test", _fallback_kupiec_test)

st.set_page_config(page_title="Risk | QuantEdge", layout="wide")
st.title("⚠️ Risk Analytics")
st.caption("VaR · CVaR · Fat-Tail · GARCH · Portfolio Risk · Stress Testing · Kupiec Backtest")

render_data_engine_controls("risk")
render_cols = st.columns([2, 1, 1, 1])
ticker = render_single_ticker_input(
    "Primary Ticker", key="risk_ticker",
    default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
    container=render_cols[0],
)
conf = render_cols[1].selectbox("Confidence Level", [0.95, 0.99], index=0)
rf_rate = render_cols[2].number_input(
    "Risk-Free Rate (%/yr)", value=cfg.RISK_FREE_RATE * 100,
    min_value=0.0, max_value=20.0, step=0.25, format="%.2f"
) / 100

start = pd.to_datetime(get_global_start_date())

with st.spinner("Loading data..."):
    df = load_ticker_data(ticker, start=str(start))
    ret = returns(df)

var_h  = var_historical(ret, conf)
cvar_h = cvar_historical(ret, conf)
var_td = var_t_dist(ret, conf)
var_g  = var_garch(ret, conf)
vol_a  = annualised_vol(ret)

st.markdown(metric_card_row({
    f"VaR {conf:.0%} (Hist)":    f"{var_h:.2%}",
    f"CVaR {conf:.0%} (Hist)":   f"{cvar_h:.2%}",
    f"VaR {conf:.0%} (t-dist)":  f"{var_td:.2%}",
    f"VaR {conf:.0%} (GARCH)":   f"{var_g:.2%}",
    "Ann. Volatility":            f"{vol_a:.2%}",
}), unsafe_allow_html=True)

st.caption(
    f"📌 RF Rate in use: **{rf_rate:.2%}/yr**  |  "
    f"ℹ️ GARCH VaR reflects today's conditional vol — spikes during stressed markets."
)
st.markdown("")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 VaR Comparison",
    "🌊 GARCH Rolling Risk",
    "🏦 Portfolio Risk",
    "💥 Stress Tests",
    "🔬 Kupiec Backtest",
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ret, nbinsx=80, name="Daily Returns",
                               marker_color="steelblue", opacity=0.75))
    for val, color, dash, label in [
        (var_h,  "red",    "dash", f"Hist VaR {conf:.0%}"),
        (cvar_h, "orange", "dot",  f"CVaR {conf:.0%}"),
        (var_td, "yellow", "dash", f"t-dist VaR {conf:.0%}"),
        (var_g,  "lime",   "dash", f"GARCH VaR {conf:.0%}"),
    ]:
        fig.add_vline(x=val, line_dash=dash, line_color=color,
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(template="plotly_dark",
                      title="Return Distribution — VaR Method Comparison",
                      xaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig, use_container_width=True)

    kurt = float(ret.kurt())
    method_df = pd.DataFrame([
        {"Method": "Historical",          "VaR": f"{var_h:.2%}",
         "Note": "From past returns directly. No distribution assumption."},
        {"Method": "Parametric (Gaussian)","VaR": f"{var_parametric(ret, conf):.2%}",
         "Note": f"Assumes normal dist. Kurtosis={kurt:.1f} — {'⚠️ fat tails, prefer t-dist' if kurt > 1 else '✅ approx normal'}."},
        {"Method": "Student-t (fat-tail)", "VaR": f"{var_td:.2%}",
         "Note": "Fits degrees-of-freedom from data. Better for equity fat tails."},
        {"Method": "GARCH(1,1)",           "VaR": f"{var_g:.2%}",
         "Note": "Accounts for vol clustering. Most responsive to current regime."},
    ])
    st.subheader("Method Comparison")
    st.dataframe(method_df, use_container_width=True, hide_index=True)

    tail = ret[ret <= var_h]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tail Events", len(tail))
    c2.metric("Worst Day", f"{ret.min():.2%}")
    c3.metric("Kurtosis", f"{kurt:.2f}")
    c4.metric("Max Drawdown", f"{max_drawdown(ret):.2%}")


# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("GARCH Conditional Volatility vs Rolling Volatility")
    st.caption("GARCH reacts faster during stress. Rolling vol lags reality.")
    window = st.slider("Rolling Window (days)", 21, 252, 63, key="garch_window")
    roll_var = ret.rolling(window).quantile(1 - conf)

    garch_var_series = pd.Series(dtype=float, index=ret.index)
    garch_vol_ann    = pd.Series(dtype=float, index=ret.index)
    garch_fit_ok = False
    try:
        from arch import arch_model
        from scipy import stats as sp_stats
        scaled = ret * 100
        gm = arch_model(scaled, vol="Garch", p=1, q=1, dist="t", rescale=False)
        gfit = gm.fit(disp="off", show_warning=False)
        cond_vol = gfit.conditional_volatility / 100
        nu = float(gfit.params.get("nu", 8))
        z = float(sp_stats.t.ppf(1 - conf, nu))
        garch_var_series = pd.Series(ret.mean() + z * cond_vol.values, index=ret.index)
        garch_vol_ann    = pd.Series(cond_vol.values * np.sqrt(252), index=ret.index)
        garch_fit_ok = True
    except Exception:
        st.warning("GARCH fit failed — showing rolling VaR only.")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ret.index, y=ret.values, name="Daily Return",
                              line=dict(color="lightblue", width=0.8), opacity=0.4))
    fig2.add_trace(go.Scatter(x=roll_var.index, y=roll_var.values,
                              name=f"Rolling VaR {conf:.0%} ({window}d)",
                              line=dict(color="red", width=2)))
    if garch_fit_ok:
        fig2.add_trace(go.Scatter(x=garch_var_series.index, y=garch_var_series.values,
                                  name=f"GARCH VaR {conf:.0%}",
                                  line=dict(color="lime", width=2)))
    fig2.update_layout(template="plotly_dark", title=f"Rolling vs GARCH VaR — {ticker}",
                       yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig2, use_container_width=True)

    if garch_fit_ok:
        roll_std = ret.rolling(window).std() * np.sqrt(252)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_std.index, y=roll_std.values,
                                     name=f"Rolling Ann. Vol ({window}d)",
                                     line=dict(color="orange", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=garch_vol_ann.index, y=garch_vol_ann.values,
                                     name="GARCH Ann. Conditional Vol",
                                     line=dict(color="lime", width=2)))
        fig_vol.update_layout(template="plotly_dark",
                              title="Annualised Vol: Rolling vs GARCH",
                              yaxis_tickformat=".1%", height=350)
        st.plotly_chart(fig_vol, use_container_width=True)


# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Portfolio-Level Risk")
    st.caption("Single-stock VaR misses correlations. Portfolio VaR accounts for how assets move together.")

    port_input = st.text_input("Tickers (comma-separated)",
                                value=", ".join(cfg.DEFAULT_TICKERS[:4]),
                                key="port_tickers")
    port_tickers = parse_ticker_list(port_input)
    c_w1, c_w2 = st.columns(2)
    equal_weight = c_w1.checkbox("Equal weight", value=True)
    port_method  = c_w2.selectbox("VaR Method", ["historical", "parametric", "garch"])

    if len(port_tickers) < 2:
        st.warning("Enter at least 2 tickers.")
    else:
        with st.spinner("Loading portfolio data..."):
            multi_data = get_multi_ohlcv(port_tickers, start=str(start.date()))
            ret_df = align_returns(multi_data).dropna()

        if ret_df.empty or ret_df.shape[1] < 2:
            st.error("Could not load data for these tickers.")
        else:
            valid_tickers = list(ret_df.columns)
            n = len(valid_tickers)
            if equal_weight:
                weights = np.ones(n) / n
            else:
                wcols = st.columns(n)
                raw_w = [wcols[i].number_input(valid_tickers[i], 0.0, 1.0,
                          value=round(1/n, 2), step=0.05, key=f"w_{valid_tickers[i]}")
                         for i in range(n)]
                total = sum(raw_w)
                weights = np.array(raw_w) / total if total > 0 else np.ones(n) / n

            port_ret_s = pd.Series(ret_df.values @ weights, index=ret_df.index)
            pvar  = portfolio_var(ret_df, weights, conf, port_method)
            pcvar = float(port_ret_s[port_ret_s <= pvar].mean()) if (port_ret_s <= pvar).sum() > 0 else 0.0
            pvol  = float(port_ret_s.std() * np.sqrt(252))
            pmdd  = max_drawdown(port_ret_s)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Portfolio VaR {conf:.0%}", f"{pvar:.2%}")
            m2.metric(f"Portfolio CVaR {conf:.0%}", f"{pcvar:.2%}")
            m3.metric("Portfolio Ann. Vol", f"{pvol:.2%}")
            m4.metric("Portfolio Max DD", f"{pmdd:.2%}")

            corr = ret_df.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Return Correlation Matrix",
                                 template="plotly_dark")
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

            indiv_vars = {t: var_historical(ret_df[t], conf) for t in valid_tickers}
            var_cmp = pd.DataFrame({
                "Asset": list(indiv_vars.keys()) + ["📦 Portfolio"],
                "VaR":   list(indiv_vars.values()) + [pvar],
            })
            fig_bar = px.bar(var_cmp, x="Asset", y="VaR", color="VaR",
                             color_continuous_scale="Reds_r", template="plotly_dark",
                             title=f"Individual vs Portfolio VaR ({conf:.0%})")
            fig_bar.update_yaxes(tickformat=".2%")
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

            cum_port = (1 + port_ret_s).cumprod()
            fig_cum = go.Figure()
            for t in valid_tickers:
                fig_cum.add_trace(go.Scatter(x=ret_df.index, y=(1 + ret_df[t]).cumprod().values,
                                             name=t, line=dict(width=1), opacity=0.6))
            fig_cum.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values,
                                         name="Portfolio", line=dict(color="cyan", width=2.5)))
            fig_cum.update_layout(template="plotly_dark",
                                  title="Cumulative Returns — Portfolio vs Components",
                                  height=380)
            st.plotly_chart(fig_cum, use_container_width=True)


# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    SCENARIOS = {
        "2008 Financial Crisis":  ("2008-09-01", "2009-03-01"),
        "COVID-19 Crash":         ("2020-02-01", "2020-04-01"),
        "2022 Rate Shock":        ("2022-01-01", "2022-10-01"),
        "2020 Tech Rally":        ("2020-04-01", "2021-01-01"),
        "2018 Q4 Selloff":        ("2018-10-01", "2018-12-31"),
    }
    rows = []
    for name, (s, e) in SCENARIOS.items():
        scenario_mask = (ret.index >= s) & (ret.index <= e)
        r = ret.loc[scenario_mask]
        if len(r) < 10:
            continue
        cum_r = (1 + r).cumprod()
        dd_s  = (cum_r - cum_r.cummax()) / cum_r.cummax()   # fixed: cumprod-based
        pnl   = (1 + r).prod() - 1
        rows.append({
            "Scenario":      name,
            "Period":        f"{s} → {e}",
            "Total Return":  f"{pnl:.2%}",
            "Max Drawdown":  f"{dd_s.min():.2%}",
            "Hist VaR 95%":  f"{var_historical(r, 0.95):.2%}",
            "t-dist VaR 95%":f"{var_t_dist(r, 0.95):.2%}",
            "Worst Day":     f"{r.min():.2%}",
            "Volatility":    f"{r.std() * np.sqrt(252):.2%}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Extend start date to cover stress test periods.")

    cum = (1 + ret).cumprod()
    fig3 = go.Figure(go.Scatter(x=cum.index, y=cum.values,
                                line=dict(color="cyan"), name="Cumulative Return"))
    for name, (s, e) in SCENARIOS.items():
        fig3.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.08,
                       annotation_text=name.split(" ")[0])
    fig3.update_layout(template="plotly_dark",
                       title=f"Cumulative Return with Stress Periods — {ticker}", height=420)
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Kupiec Proportion of Failures (POF) Test")
    st.caption(
        "How accurate is your VaR model? Counts actual losses > VaR vs expected. "
        "p > 0.05 = statistically valid. p < 0.05 = underestimating risk."
    )
    kup_window = st.slider("Rolling VaR Window (days)", 21, 126, 63, key="kup_window")
    roll_var_bt = ret.rolling(kup_window).quantile(1 - conf).shift(1)

    kup = kupiec_test(ret, roll_var_bt, conf)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", len(ret.dropna()))
    k2.metric("Violations", kup["violations"])
    k3.metric("Expected Rate", f"{kup['expected_rate']:.1%}")
    k4.metric("Actual Rate", f"{kup['actual_rate']:.1%}" if not np.isnan(kup["actual_rate"]) else "N/A")

    p_val = kup["p_value"]
    if not np.isnan(p_val):
        color = "green" if p_val > 0.05 else "red"
        st.markdown(f"**p-value:** `{p_val:.4f}`  |  **Result:** :{color}[{kup['result']}]")
    else:
        st.info(kup["result"])

    violations_mask = ret < roll_var_bt
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ret.index, y=ret.values, name="Daily Return",
                              line=dict(color="lightblue", width=0.8), opacity=0.5))
    fig4.add_trace(go.Scatter(x=roll_var_bt.index, y=roll_var_bt.values,
                              name=f"VaR {conf:.0%} (1d lag)", line=dict(color="red", width=1.5)))
    viol_ret = ret[violations_mask]
    fig4.add_trace(go.Scatter(x=viol_ret.index, y=viol_ret.values,
                              mode="markers", name="VaR Violation",
                              marker=dict(color="orange", size=5, symbol="x")))
    fig4.update_layout(template="plotly_dark",
                       title=f"VaR Violations — {ticker} ({kup['violations']} breaches)",
                       yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "💡 Too few violations = overcautious (wastes capital). "
        "Too many = underestimating risk. Target: violations ≈ (1 - confidence) × total days."
    )
