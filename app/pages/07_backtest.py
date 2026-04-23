import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import core.backtest_engine as backtest_engine


@dataclass
class _FallbackCostModel:
    market_name: str
    brokerage: float
    stt_buy: float
    stt_sell: float
    exchange_fee: float
    sebi_fee: float
    stamp_duty_buy: float
    gst_rate: float
    slippage_bps: float

    @property
    def slippage_frac(self) -> float:
        return self.slippage_bps / 10_000

    def total_round_trip_cost(self) -> float:
        gst = (self.brokerage + self.exchange_fee) * self.gst_rate
        buy = (
            self.brokerage + self.stt_buy + self.exchange_fee +
            self.sebi_fee + self.stamp_duty_buy + gst + self.slippage_frac
        )
        sell = (
            self.brokerage + self.stt_sell + self.exchange_fee +
            self.sebi_fee + gst + self.slippage_frac
        )
        return round(buy + sell, 6)

    def breakdown(self) -> dict:
        rt = self.total_round_trip_cost()
        gst_total = (self.brokerage + self.exchange_fee) * 2 * self.gst_rate
        return {
            "Market": self.market_name,
            "Brokerage (2 legs)": f"{self.brokerage * 2:.4%}",
            "STT (buy + sell)": f"{self.stt_buy + self.stt_sell:.4%}",
            "Exchange fee (2 legs)": f"{self.exchange_fee * 2:.4%}",
            "SEBI fee (2 legs)": f"{self.sebi_fee * 2:.4%}",
            "Stamp duty (buy)": f"{self.stamp_duty_buy:.4%}",
            "GST (18% on fees)": f"{gst_total:.4%}",
            "Slippage (2 legs)": f"{self.slippage_frac * 2:.4%}",
            "TOTAL round trip": f"{rt:.4%}",
            "vs Old model (0.21%)": f"{rt / 0.0021:.1f}x more realistic",
        }


def _default_cost_profiles() -> dict[str, _FallbackCostModel]:
    return {
        "India - Delivery": _FallbackCostModel(
            market_name="India - Delivery",
            brokerage=0.0003,
            stt_buy=0.0,
            stt_sell=0.001,
            exchange_fee=0.0000335,
            sebi_fee=0.000001,
            stamp_duty_buy=0.00015,
            gst_rate=0.18,
            slippage_bps=10.0,
        ),
        "India - Intraday": _FallbackCostModel(
            market_name="India - Intraday",
            brokerage=0.0003,
            stt_buy=0.00025,
            stt_sell=0.00025,
            exchange_fee=0.0000335,
            sebi_fee=0.000001,
            stamp_duty_buy=0.000003,
            gst_rate=0.18,
            slippage_bps=5.0,
        ),
        "US - Retail": _FallbackCostModel(
            market_name="US - Retail",
            brokerage=0.0005,
            stt_buy=0.0,
            stt_sell=0.0000229,
            exchange_fee=0.0,
            sebi_fee=0.0,
            stamp_duty_buy=0.0,
            gst_rate=0.0,
            slippage_bps=4.0,
        ),
        "Custom (Manual)": _FallbackCostModel(
            market_name="Custom (Manual)",
            brokerage=0.001,
            stt_buy=0.0,
            stt_sell=0.0,
            exchange_fee=0.0,
            sebi_fee=0.0,
            stamp_duty_buy=0.0,
            gst_rate=0.0,
            slippage_bps=5.0,
        ),
    }


BacktestConfig = backtest_engine.BacktestConfig
WalkForwardConfig = getattr(backtest_engine, "WalkForwardConfig", None)
if WalkForwardConfig is None:
    @dataclass
    class WalkForwardConfig:
        # Compatibility shim for stale engine imports during Streamlit reloads.
        train_months: int = 36
        test_months: int = 6
        min_train_periods: int = 60

COST_PROFILES = getattr(backtest_engine, "COST_PROFILES", _default_cost_profiles())
run_backtest = backtest_engine.run_backtest
run_walk_forward = backtest_engine.run_walk_forward
run_monte_carlo = backtest_engine.run_monte_carlo
run_regime_backtest = backtest_engine.run_regime_backtest
regime_strategy_matrix = backtest_engine.regime_strategy_matrix
momentum_strategy = backtest_engine.momentum_strategy
mean_reversion_strategy = backtest_engine.mean_reversion_strategy
rsi_strategy = backtest_engine.rsi_strategy
macd_crossover_strategy = backtest_engine.macd_crossover_strategy
dual_ma_strategy = backtest_engine.dual_ma_strategy
BULL = getattr(backtest_engine, "BULL", "Bull")
SIDEWAYS = getattr(backtest_engine, "SIDEWAYS", "Sideways")
BEAR = getattr(backtest_engine, "BEAR", "Bear")
from core.data import returns
from app.data_engine import (
    render_data_engine_controls, render_single_ticker_input,
    load_ticker_data, get_global_start_date,
)
from utils.charts import equity_curve, drawdown_chart, metric_card_row
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

st.set_page_config(page_title="Backtest | QuantEdge", layout="wide")
st.title("⚡ Strategy Backtester")
qe_neon_divider()

# ── Data controls ─────────────────────────────────────────────────────────────
render_data_engine_controls("backtest")
c1, c2, c3 = st.columns(3)
ticker   = render_single_ticker_input("Ticker", key="backtest_ticker",
               default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
               container=c1)
strategy = c2.selectbox("Strategy", [
    "Momentum", "Mean Reversion", "RSI", "MACD Crossover",
    "Dual MA", "Regime-Aware (Auto-Switch)"])
start    = pd.to_datetime(get_global_start_date())

# ── Cost model ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💸 Market Cost Model")
st.info(
    "The old model assumed 0.21% round-trip. Indian markets realistically cost "
    "0.4–0.6% after STT, stamp duty, SEBI charges and GST. Choose your market."
)
cc = st.columns([2, 1, 1, 1])
market_name  = cc[0].selectbox("Market / Cost Profile", list(COST_PROFILES.keys()))
cost_model   = COST_PROFILES[market_name]
rt           = cost_model.total_round_trip_cost()
cc[1].metric("Round-Trip Cost",      f"{rt:.3%}")
cc[2].metric("vs Old Model (0.21%)", f"{rt/0.0021:.1f}x")
cc[3].metric("Slippage",             f"{cost_model.slippage_bps:.0f} bps")
with st.expander("Full Cost Breakdown"):
    for k, v in cost_model.breakdown().items():
        st.write(f"**{k}:** {v}")

# ── Capital & strategy params ─────────────────────────────────────────────────
st.markdown("---")
capital = st.columns(3)[0].number_input("Initial Capital", 10_000, 10_000_000, 100_000, 10_000)

with st.expander("⚙️ Strategy Parameters"):
    sp = st.columns(2)
    p1 = sp[0].slider("Fast window / RSI period / Lookback", 5, 60, 14)
    p2 = sp[1].slider("Slow window / z-thresh×10", 10, 200, 50)
    if strategy == "Regime-Aware (Auto-Switch)":
        ra = st.columns(3)
        bull_s = ra[0].selectbox("Bull regime strategy",
                                  ["Dual MA", "Momentum", "MACD Crossover"])
        side_s = ra[1].selectbox("Sideways regime strategy",
                                  ["RSI", "Mean Reversion"])
        bear_s = ra[2].selectbox("Bear regime strategy",
                                  ["Cash", "Inverse", "Mean Reversion"])
    else:
        bull_s = "Dual MA"; side_s = "RSI"; bear_s = "Cash"

# ── Feature toggles ───────────────────────────────────────────────────────────
st.markdown("---")
ft = st.columns(3)
do_wf  = ft[0].toggle("Walk-Forward Test",       value=True,
    help="Train on rolling windows, test on unseen data — prevents overfitting.")
do_mc  = ft[1].toggle("Monte Carlo Simulation",   value=True,
    help="1000 paths with random entry delays — shows luck vs skill.")
do_mat = ft[2].toggle("Regime Performance Matrix", value=False,
    help="Which strategy wins in each regime? Helps configure auto-switch.")

wr = st.columns(3)
train_mo = wr[0].number_input("Train window (months)", 12, 60, 36, disabled=not do_wf)
test_mo  = wr[1].number_input("Test window (months)",   3, 18,  6, disabled=not do_wf)
n_sims   = wr[2].number_input("MC simulations",       200, 2000, 500, disabled=not do_mc)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_ticker_data(ticker, start=str(start))

# ── RUN ───────────────────────────────────────────────────────────────────────
if st.button("Run Backtest", type="primary"):

    bcfg = BacktestConfig(
        initial_capital=capital,
        commission_pct =cost_model.brokerage,
        slippage_bps   =cost_model.slippage_bps,
        risk_free_rate =cfg.RISK_FREE_RATE,
        cost_model     =cost_model,
    )

    with st.spinner("Running backtest..."):
        if strategy == "Momentum":
            signal = momentum_strategy(df, lookback=p1)
        elif strategy == "Mean Reversion":
            signal = mean_reversion_strategy(df, window=p1, z_thresh=p2/10)
        elif strategy == "RSI":
            signal = rsi_strategy(df, oversold=p1, overbought=p2)
        elif strategy == "MACD Crossover":
            signal = macd_crossover_strategy(df)
        elif strategy == "Dual MA":
            signal = dual_ma_strategy(df, fast=p1, slow=p2)
        else:
            signal = None

        if strategy == "Regime-Aware (Auto-Switch)":
            reg_res = run_regime_backtest(
                df, bcfg,
                bull_strategy=bull_s, sideways_strategy=side_s, bear_strategy=bear_s,
                p1=p1, p2=p2,
            )
            result = reg_res.base_result
        else:
            result = run_backtest(df["Close"], signal, bcfg)
            reg_res = None

    bh_ret = returns(df)

    # ── Section 1: Base Results ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Backtest Results")
    display_met = {k: result.metrics[k] for k in
                   ["CAGR","Sharpe","Sortino","Max Drawdown","Win Rate","Num Trades"]}
    st.markdown(metric_card_row(display_met), unsafe_allow_html=True)
    st.markdown("")
    st.plotly_chart(equity_curve(result.daily_returns, bh_ret,
                                  f"{strategy} — Equity Curve"), use_container_width=True)
    st.plotly_chart(drawdown_chart(result.daily_returns), use_container_width=True)

    # Rolling Sharpe
    fig_rs = go.Figure(go.Scatter(x=result.rolling_sharpe.index,
        y=result.rolling_sharpe.values, line=dict(color="gold"), name="Rolling Sharpe (63d)"))
    fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rs.add_hline(y=1, line_dash="dot",  line_color="lime")
    fig_rs.update_layout(template="plotly_dark", title="Rolling 63-Day Sharpe", height=280)
    st.plotly_chart(fig_rs, use_container_width=True)

    # ── Section 2: Regime-Aware Results ──────────────────────────────────────
    if reg_res is not None:
        st.markdown("---")
        st.subheader("🔄 Regime-Aware Breakdown")
        rc = st.columns(3)
        rc[0].metric("Bull Days",     result.metrics.get("Bull Days", "—"))
        rc[1].metric("Sideways Days", result.metrics.get("Sideways Days", "—"))
        rc[2].metric("Bear Days",     result.metrics.get("Bear Days", "—"))

        fig_r = go.Figure()
        for lbl, col in [(BULL,"lime"), (SIDEWAYS,"gold"), (BEAR,"tomato")]:
            mask = reg_res.regime_series == lbl
            eq   = reg_res.equity_curve.copy().astype(float)
            eq[~mask] = None
            fig_r.add_trace(go.Scatter(x=eq.index, y=eq.values, name=lbl,
                line=dict(color=col, width=1.5), connectgaps=False))
        fig_r.update_layout(template="plotly_dark", height=350,
            title="Equity Curve Coloured by Market Regime",
            legend=dict(orientation="h"))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Section 3: Walk-Forward ───────────────────────────────────────────────
    if do_wf and strategy != "Regime-Aware (Auto-Switch)":
        st.markdown("---")
        st.subheader("🔬 Walk-Forward Test  —  Overfit Detection")
        st.caption(
            "Every test fold uses data the strategy was **never** trained on. "
            "The Efficiency Ratio (OOS Sharpe ÷ IS Sharpe) tells you if performance "
            "holds up on unseen data.  **< 0.5 = overfit warning.**"
        )

        def make_sig_fn(strat, _p1, _p2):
            def _fn(price_s: pd.Series) -> pd.Series:
                _df = pd.DataFrame({"Close": price_s})
                if strat == "Momentum":       return momentum_strategy(_df, lookback=_p1)
                if strat == "Mean Reversion": return mean_reversion_strategy(_df, window=_p1, z_thresh=_p2/10)
                if strat == "RSI":            return rsi_strategy(_df, oversold=_p1, overbought=_p2)
                if strat == "MACD Crossover": return macd_crossover_strategy(_df)
                return dual_ma_strategy(_df, fast=_p1, slow=_p2)
            return _fn

        with st.spinner("Running walk-forward folds..."):
            try:
                wf = run_walk_forward(
                    df["Close"], make_sig_fn(strategy, p1, p2), bcfg,
                    WalkForwardConfig(train_months=int(train_mo), test_months=int(test_mo)),
                )

                if wf.overfit_warning:
                    st.error(
                        f"⚠️  OVERFIT WARNING — Efficiency Ratio: {wf.efficiency_ratio:.2f}  "
                        f"Strategy looks great in-sample but fails on unseen data. "
                        f"Do NOT deploy this live."
                    )
                else:
                    st.success(
                        f"✅  Strategy passes walk-forward — Efficiency Ratio: "
                        f"{wf.efficiency_ratio:.2f}   OOS performance is consistent."
                    )

                wc = st.columns(4)
                wc[0].metric("OOS CAGR",        wf.oos_metrics.get("CAGR","—"))
                wc[1].metric("OOS Sharpe",       wf.oos_metrics.get("Sharpe","—"))
                wc[2].metric("OOS Max Drawdown", wf.oos_metrics.get("Max Drawdown","—"))
                wc[3].metric("Folds Completed",  str(wf.n_folds))

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(x=result.equity_curve.index,
                    y=result.equity_curve.values, name="In-Sample (backtest)",
                    line=dict(color="royalblue", dash="dash")))
                fig_wf.add_trace(go.Scatter(x=wf.oos_equity.index,
                    y=wf.oos_equity.values, name="Out-of-Sample (walk-forward)",
                    line=dict(color="lime", width=2)))
                fig_wf.update_layout(template="plotly_dark", height=350,
                    title="In-Sample vs Out-of-Sample Equity",
                    legend=dict(orientation="h"))
                st.plotly_chart(fig_wf, use_container_width=True)

                def _status_colour(val):
                    return {"Strong":"color:lime","Acceptable":"color:gold",
                            "Weak":"color:orange","Failing":"color:tomato"}.get(val,"")
                st.markdown("**Per-Fold Results**")
                st.dataframe(wf.fold_metrics.style.map(_status_colour, subset=["Status"]),
                             use_container_width=True)
            except ValueError as e:
                st.warning(f"Walk-forward error: {e}")

    # ── Section 4: Monte Carlo ────────────────────────────────────────────────
    if do_mc:
        st.markdown("---")
        st.subheader("🎲 Monte Carlo Simulation  —  Luck vs Skill")
        st.caption(
            f"{int(n_sims)} simulations with random entry delays and execution noise. "
            "**Tight fan = robust strategy. Wide fan = lucky strategy.**"
        )

        with st.spinner(f"Running {int(n_sims)} Monte Carlo paths..."):
            mc = run_monte_carlo(
                daily_returns=result.daily_returns, bh_returns=bh_ret,
                initial_capital=capital, n_simulations=int(n_sims),
            )

        mc_c = st.columns(4)
        mc_c[0].metric("Probability of Profit",  f"{mc.prob_profit:.1%}")
        mc_c[1].metric("Beats Buy & Hold",        f"{mc.prob_beat_bh:.1%}")
        mc_c[2].metric("Risk of Ruin (>50% DD)",  f"{mc.risk_of_ruin:.1%}")
        mc_c[3].metric("Sharpe 90% CI",           f"{mc.sharpe_ci_low:.2f} – {mc.sharpe_ci_high:.2f}")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=list(mc.pct_5.index) + list(mc.pct_95.index[::-1]),
            y=list(mc.pct_5.values) + list(mc.pct_95.values[::-1]),
            fill="toself", fillcolor="rgba(255,255,255,0.05)",
            line=dict(color="rgba(0,0,0,0)"), name="5–95th pct"))
        fig_mc.add_trace(go.Scatter(
            x=list(mc.pct_25.index) + list(mc.pct_75.index[::-1]),
            y=list(mc.pct_25.values) + list(mc.pct_75.values[::-1]),
            fill="toself", fillcolor="rgba(100,180,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="25–75th pct"))
        fig_mc.add_trace(go.Scatter(x=mc.pct_50.index, y=mc.pct_50.values,
            name="Median", line=dict(color="royalblue", width=2)))
        fig_mc.add_trace(go.Scatter(x=mc.pct_5.index,  y=mc.pct_5.values,
            name="Worst 5%",  line=dict(color="tomato", width=1, dash="dot")))
        fig_mc.add_trace(go.Scatter(x=mc.pct_95.index, y=mc.pct_95.values,
            name="Best 5%",   line=dict(color="lime",   width=1, dash="dot")))
        fig_mc.add_hline(y=capital, line_dash="dash", line_color="gray",
                          annotation_text="Starting capital")
        fig_mc.update_layout(template="plotly_dark", height=420,
            title=f"Monte Carlo Fan Chart  ({int(n_sims)} simulations)",
            legend=dict(orientation="h"))
        st.plotly_chart(fig_mc, use_container_width=True)

        fig_d = go.Figure(go.Histogram(x=mc.final_values, nbinsx=60,
                                        marker_color="royalblue", opacity=0.8))
        fig_d.add_vline(x=capital, line_dash="dash", line_color="white",
                         annotation_text="Starting capital")
        fig_d.add_vline(x=float(np.median(mc.final_values)),
                         line_dash="dot", line_color="lime",
                         annotation_text="Median outcome")
        fig_d.update_layout(template="plotly_dark", height=300,
            title="Distribution of Final Portfolio Values")
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Section 5: Regime Matrix ──────────────────────────────────────────────
    if do_mat:
        st.markdown("---")
        st.subheader("🗺️  Strategy × Regime Performance Matrix")
        st.caption("Discover which strategy performs best in each market condition. "
                   "Use this to configure the Regime-Aware strategy above.")
        with st.spinner("Analysing all strategies across all regimes..."):
            mat = regime_strategy_matrix(df, bcfg)
        if not mat.empty:
            st.dataframe(mat.style.map(
                lambda v: ("color:lime"   if isinstance(v,(int,float)) and v > 0.5 else
                           "color:gold"   if isinstance(v,(int,float)) and 0 < v <= 0.5 else
                           "color:tomato" if isinstance(v,(int,float)) and v <= 0 else ""),
                subset=["Sharpe"]), use_container_width=True)
        else:
            st.info("Not enough regime data. Try a longer date range.")

    # ── Section 6: Trade log + full metrics ───────────────────────────────────
    st.markdown("---")
    st.subheader("Trade Log")
    if len(result.trade_log):
        st.dataframe(result.trade_log.style.map(
            lambda v: ("color:lime"   if isinstance(v,(int,float)) and v > 0 else
                       "color:tomato" if isinstance(v,(int,float)) and v < 0 else ""),
            subset=["PnL %","PnL"]), use_container_width=True)
    else:
        st.info("No completed trades found.")

    with st.expander("Full Metrics Table"):
        st.dataframe(pd.DataFrame.from_dict(result.metrics, orient="index",
                                             columns=["Value"]), use_container_width=True)

qe_faq_section("FAQs", [
    ("What should I run first on the backtester?", "Start with the default strategy and review the equity curve, drawdown, and trade log before trying more advanced settings."),
    ("Why do walk-forward and Monte Carlo matter?", "They check whether the strategy still works on unseen data and whether the result is robust or just lucky."),
    ("How should I read the regime-aware mode?", "It switches strategy by market regime, so it is useful when one strategy performs well only in certain conditions."),
    ("What is the main thing to watch?", "Focus on net performance after realistic costs, because a strategy that wins gross but fails after fees is not tradable."),
])
