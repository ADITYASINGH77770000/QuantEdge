
"""
app/main.py  —  QuantEdge Command Center
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import cfg
from utils.theme import (
    apply_quantedge_theme,
    apply_plotly_theme,
    qe_neon_divider,
    qe_section_header,
    qe_metric_cards,
    qe_regime_box,
    COLORS,
)
from core.data import get_ohlcv, get_multi_ohlcv
from core.metrics import sharpe, sortino, max_drawdown, var_historical, cagr as cagr_fn
from core.indicators import rsi, add_all_indicators

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantEdge | Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_quantedge_theme()

# ── Hide Streamlit's auto-generated nav (handled natively in .streamlit/config.toml) ──

# ── Sidebar — branded nav with emoji icons ────────────────────────────────────
with st.sidebar:
    st.divider()
    st.caption(f"RFR: **{cfg.RISK_FREE_RATE:.1%}**  ·  Cache: **{cfg.CACHE_TTL}s**")

# ── Constants ─────────────────────────────────────────────────────────────────
TICKERS = cfg.DEFAULT_TICKERS

now_utc = datetime.now(timezone.utc)
now_ist = now_utc.astimezone(
    __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("Asia/Kolkata")
)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_watchlist(tickers: tuple) -> dict[str, pd.DataFrame]:
    return get_multi_ohlcv(list(tickers))


@st.cache_data(ttl=300, show_spinner=False)
def compute_portfolio_metrics(tickers: tuple) -> dict:
    data  = get_multi_ohlcv(list(tickers))
    rets  = pd.DataFrame({t: data[t]["Close"].pct_change().dropna() for t in tickers}).dropna()
    port  = rets.mean(axis=1)
    mkt   = rets.mean(axis=1)
    cov   = np.cov(port, mkt)
    beta  = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

    roll_sharpe = port.rolling(30).apply(
        lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0, raw=True
    ).dropna().tail(90)

    equity = (1 + port).cumprod()

    return {
        "sharpe_1y":   round(sharpe(port.tail(252)), 2),
        "sortino_1y":  round(sortino(port.tail(252)), 2),
        "max_dd":      round(max_drawdown(port) * 100, 2),
        "var_95_1d":   round(var_historical(port, 0.95) * 100, 2),
        "cvar_95_1d":  round(var_historical(port, 0.99) * 100, 2),
        "beta":        round(float(beta), 2),
        "cagr":        round(cagr_fn(port) * 100, 2),
        "win_rate":    round(float((port > 0).mean() * 100), 1),
        "port_rets":   port,
        "equity":      equity,
        "roll_sharpe": roll_sharpe,
    }


@st.cache_data(ttl=300, show_spinner=False)
def detect_regime(ticker: str) -> dict:
    df      = get_ohlcv(ticker)
    close   = df["Close"]
    sma50   = close.rolling(50).mean()
    sma200  = close.rolling(200).mean()
    vol20   = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_avg = vol20.rolling(60).mean()

    s50, s200       = float(sma50.iloc[-1]), float(sma200.iloc[-1])
    v_now, v_avg    = float(vol20.iloc[-1]), float(vol_avg.iloc[-1])

    if s50 > s200 and v_now < v_avg * 1.2:
        regime, conf = "BULL",     min(0.75 + (s50/s200-1)*1.0, 0.95)
    elif s50 < s200 and v_now > v_avg:
        regime, conf = "BEAR",     min(0.70 + (1-s50/s200)*1.0, 0.95)
    elif v_now > v_avg * 1.4:
        regime, conf = "VOLATILE", 0.65
    else:
        regime, conf = "SIDEWAYS", 0.60

    cross   = sma50 > sma200 if s50 > s200 else sma50 < sma200
    days_in = int((cross[::-1].cumprod()).sum())
    bear_prob = min(max((v_now/v_avg-0.8)*0.5, 0.0), 1.0) if s50 < s200 \
                else max(0.1-(s50/s200-1)*0.5, 0.0)

    return {"regime": regime, "conf": round(conf,2), "days_in": days_in,
            "bear_prob": round(bear_prob,2), "vol_ratio": round(v_now/v_avg,2)}


@st.cache_data(ttl=300, show_spinner=False)
def scan_signals(tickers: tuple) -> list[dict]:
    signals = []
    for ticker in tickers:
        try:
            df        = get_ohlcv(ticker)
            df        = add_all_indicators(df)
            rsi_val   = float(df["RSI"].iloc[-1])
            macd_val  = float(df["MACD"].iloc[-1])
            sig_val   = float(df["Signal"].iloc[-1])
            macd_prev = float(df["MACD"].iloc[-2])
            sig_prev  = float(df["Signal"].iloc[-2])
            price     = float(df["Close"].iloc[-1])
            chg_1d    = float(df["Close"].pct_change().iloc[-1] * 100)

            if rsi_val < 30:
                signals.append({"ticker": ticker, "signal": "RSI Oversold",
                                 "detail": f"RSI {rsi_val:.1f}", "type": "BUY",
                                 "strength": "HIGH", "price": price, "chg": chg_1d})
            elif rsi_val > 70:
                signals.append({"ticker": ticker, "signal": "RSI Overbought",
                                 "detail": f"RSI {rsi_val:.1f}", "type": "SELL",
                                 "strength": "HIGH", "price": price, "chg": chg_1d})

            if macd_val > sig_val and macd_prev <= sig_prev:
                signals.append({"ticker": ticker, "signal": "MACD Bull Cross",
                                 "detail": f"MACD {macd_val:.3f}", "type": "BUY",
                                 "strength": "MEDIUM", "price": price, "chg": chg_1d})
            elif macd_val < sig_val and macd_prev >= sig_prev:
                signals.append({"ticker": ticker, "signal": "MACD Bear Cross",
                                 "detail": f"MACD {macd_val:.3f}", "type": "SELL",
                                 "strength": "MEDIUM", "price": price, "chg": chg_1d})
        except Exception:
            continue
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: rgba(10,16,30,0.7);
    border: 1px solid rgba(80,110,160,0.22);
    border-radius: 16px;
    padding: 32px 36px 26px 36px;
    margin-bottom: 24px;
">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
    <div>
      <div style="
          font-size:2.2rem;font-weight:800;letter-spacing:-1px;
          background:linear-gradient(135deg,#e8f4fd 0%,#0be0ff 55%,#a55efd 100%);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;line-height:1.1;margin-bottom:10px;">
        QuantEdge
      </div>
      <div style="
          font-size:0.82rem;color:#7a9abb;
          font-family:'JetBrains Mono',monospace;
          letter-spacing:0.5px;line-height:1.7;max-width:640px;">
        A <strong style="color:#0be0ff;">multi-agent quantitative research platform</strong>
        combining ML-driven signal generation, regime-aware portfolio construction,
        and institutional-grade risk analytics — all on live market data.
        Built for systematic traders, quant researchers, and portfolio engineers.
      </div>
      <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;">
        <span style="background:rgba(0,245,160,0.1);border:1px solid rgba(0,245,160,0.3);
                     color:#00f5a0;font-size:0.7rem;padding:3px 10px;border-radius:20px;
                     font-family:monospace;">🤖 ML Signals</span>
        <span style="background:rgba(11,224,255,0.1);border:1px solid rgba(11,224,255,0.3);
                     color:#0be0ff;font-size:0.7rem;padding:3px 10px;border-radius:20px;
                     font-family:monospace;">📡 Live Data</span>
        <span style="background:rgba(165,94,253,0.1);border:1px solid rgba(165,94,253,0.3);
                     color:#a55efd;font-size:0.7rem;padding:3px 10px;border-radius:20px;
                     font-family:monospace;">🌐 HMM Regime</span>
        <span style="background:rgba(255,215,0,0.1);border:1px solid rgba(255,215,0,0.3);
                     color:#ffd700;font-size:0.7rem;padding:3px 10px;border-radius:20px;
                     font-family:monospace;">⚠️ Risk Engine</span>
        <span style="background:rgba(255,71,87,0.1);border:1px solid rgba(255,71,87,0.3);
                     color:#ff4757;font-size:0.7rem;padding:3px 10px;border-radius:20px;
                     font-family:monospace;">🔔 Smart Alerts</span>
      </div>
    </div>
    <div style="text-align:right;min-width:160px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#546e8a;">
        {now_ist.strftime('%d %b %Y')}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                  color:#0be0ff;font-weight:600;margin-top:2px;">
        {now_ist.strftime('%H:%M:%S IST')}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                  color:#546e8a;margin-top:6px;">
        {'🟡 DEMO MODE' if cfg.DEMO_MODE else '🟢 LIVE'}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#546e8a;margin-top:2px;">
        Watchlist: {', '.join(TICKERS)}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Portfolio KPIs ────────────────────────────────────────────────────────────
st.subheader("Portfolio Risk / Return  —  Equal Weight")

with st.spinner("Computing portfolio metrics…"):
    try:
        pm     = compute_portfolio_metrics(tuple(TICKERS))
        kpi_ok = True
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        kpi_ok = False

if kpi_ok:
    qe_metric_cards({
        "Sharpe (1Y)":   f"{pm['sharpe_1y']}",
        "Sortino (1Y)":  f"{pm['sortino_1y']}",
        "CAGR":          f"{pm['cagr']}%",
        "Max Drawdown":  f"{pm['max_dd']}%",
        "VaR 95% (1D)":  f"{pm['var_95_1d']}%",
        "CVaR 99% (1D)": f"{pm['cvar_95_1d']}%",
        "Beta":          f"{pm['beta']}",
        "Win Rate":      f"{pm['win_rate']}%",
    })

qe_neon_divider()

# ── Equity curve + Rolling Sharpe ─────────────────────────────────────────────
if kpi_ok:
    st.subheader("Equity Curve  &  Rolling 30D Sharpe")
    ec_col, rs_col = st.columns([3, 2])

    with ec_col:
        equity = pm["equity"]
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="Portfolio NAV",
            line=dict(color=COLORS["accent_blue"], width=2),
            fill="tozeroy", fillcolor="rgba(11,224,255,0.06)",
        ))
        roll_max = equity.cummax()
        dd_floor = (equity / roll_max) * equity.iloc[0]
        fig_eq.add_trace(go.Scatter(
            x=dd_floor.index, y=dd_floor.values,
            mode="lines", name="Drawdown floor",
            line=dict(color=COLORS["signal_sell"], width=1, dash="dot"),
            opacity=0.4,
        ))
        apply_plotly_theme(fig_eq, "Cumulative NAV (Equal-Weight)", height=280)
        st.plotly_chart(fig_eq, use_container_width=True)

    with rs_col:
        rs = pm["roll_sharpe"]
        colors_rs = [COLORS["signal_buy"] if v >= 1.0 else
                     COLORS["neutral"]    if v >= 0.0 else
                     COLORS["signal_sell"] for v in rs.values]
        fig_rs = go.Figure()
        fig_rs.add_hrect(y0=1.0, y1=max(rs.max()*1.1, 1.5),
                         fillcolor="rgba(0,245,160,0.04)", line_width=0)
        fig_rs.add_hrect(y0=min(rs.min()*1.1, -0.5), y1=0,
                         fillcolor="rgba(255,71,87,0.04)", line_width=0)
        fig_rs.add_trace(go.Bar(x=rs.index, y=rs.values,
                                marker_color=colors_rs, name="30D Sharpe"))
        fig_rs.add_hline(y=1.0, line_dash="dash",
                         line_color=COLORS["signal_buy"], opacity=0.5)
        fig_rs.add_hline(y=0, line_dash="dash",
                         line_color=COLORS["signal_sell"], opacity=0.4)
        apply_plotly_theme(fig_rs, "Rolling 30D Sharpe", height=280)
        st.plotly_chart(fig_rs, use_container_width=True)

qe_neon_divider()

# ── Regime + Return Distribution ─────────────────────────────────────────────
st.subheader("Regime Detection  &  Return Distribution")
reg_col, dist_col = st.columns([1, 2])

with reg_col:
    with st.spinner("Detecting regime…"):
        try:
            reg = detect_regime(TICKERS[0])
        except Exception:
            reg = {"regime": "UNKNOWN", "conf": 0.0, "days_in": 0,
                   "bear_prob": 0.0, "vol_ratio": 1.0}

    REGIME_ICONS = {"BULL": "📈", "BEAR": "📉", "VOLATILE": "⚡",
                    "SIDEWAYS": "↔", "UNKNOWN": "❓"}
    qe_regime_box(
        regime=f"{REGIME_ICONS.get(reg['regime'],'❓')} {reg['regime']}",
        recommendation=(
            f"Confidence {reg['conf']:.0%}  ·  "
            f"Active {reg['days_in']}d  ·  "
            f"Vol ratio {reg['vol_ratio']:.2f}×"
        ),
    )

    bear_p = reg["bear_prob"]
    gauge_color = (COLORS["signal_sell"] if bear_p > 0.5 else
                   COLORS["neutral"]     if bear_p > 0.3 else
                   COLORS["signal_buy"])
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bear_p * 100,
        number={"suffix": "%", "font": {"size": 22, "color": gauge_color}},
        title={"text": "Bear Probability", "font": {"size": 12, "color": "#7a9abb"}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#546e8a"},
            "bar":   {"color": gauge_color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30],   "color": "rgba(0,245,160,0.08)"},
                {"range": [30, 60],  "color": "rgba(255,215,0,0.08)"},
                {"range": [60, 100], "color": "rgba(255,71,87,0.10)"},
            ],
            "threshold": {"line": {"color": "#ff4757", "width": 2}, "value": 70},
        },
    ))
    apply_plotly_theme(fig_gauge, height=200)
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if reg["regime"] == "BEAR":
        st.error("⚠️ Bear regime — review risk limits")
    elif reg["regime"] == "VOLATILE":
        st.warning("⚡ High volatility — reduce sizing")
    elif reg["regime"] == "BULL":
        st.success("✅ Bull regime — signal models active")
    else:
        st.info("↔️ Sideways — mean-reversion favoured")

with dist_col:
    if kpi_ok:
        port_rets = pm["port_rets"].tail(252)
        var95     = float(np.percentile(port_rets, 5))
        cvar95    = float(port_rets[port_rets <= var95].mean())

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=port_rets * 100, nbinsx=60,
            name="Daily Returns",
            marker_color=COLORS["accent_blue"], opacity=0.65,
        ))
        tail = port_rets[port_rets <= var95] * 100
        fig_dist.add_trace(go.Histogram(
            x=tail, nbinsx=20,
            name="Tail (VaR 95%)",
            marker_color=COLORS["signal_sell"], opacity=0.85,
        ))
        fig_dist.add_vline(x=var95*100, line_dash="dash",
                           line_color=COLORS["signal_sell"],
                           annotation_text=f"VaR 95%: {var95*100:.2f}%",
                           annotation_font_color=COLORS["signal_sell"])
        fig_dist.add_vline(x=cvar95*100, line_dash="dot",
                           line_color=COLORS["neutral"],
                           annotation_text=f"CVaR: {cvar95*100:.2f}%",
                           annotation_font_color=COLORS["neutral"],
                           annotation_position="top left")
        fig_dist.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_dist.update_layout(barmode="overlay", showlegend=True,
                               legend=dict(font=dict(size=10)))
        apply_plotly_theme(fig_dist, "Return Distribution  (1Y Daily, Equal-Weight)", height=340)
        st.plotly_chart(fig_dist, use_container_width=True)

qe_neon_divider()

# ── Signals + Correlation heatmap ─────────────────────────────────────────────
st.subheader("Active Signals  &  Correlation Matrix")
sig_col, corr_col = st.columns([1, 1])

with sig_col:
    with st.spinner("Scanning signals…"):
        try:
            signals = scan_signals(tuple(TICKERS))
        except Exception:
            signals = []

    if not signals:
        st.info("No active signals at this time.")
    else:
        for sig in signals:
            is_buy  = sig["type"] == "BUY"
            is_high = sig["strength"] == "HIGH"
            bg      = "rgba(0,245,160,0.07)"  if is_buy else "rgba(255,71,87,0.07)"
            border  = COLORS["signal_buy"]    if is_buy else COLORS["signal_sell"]
            icon    = "🟢" if is_buy else "🔴"
            chg_str = f"{sig['chg']:+.2f}%" if "chg" in sig else ""
            badge   = (
                f'<span style="background:rgba(255,71,87,0.15);color:#ff4757;'
                f'font-size:0.68rem;padding:2px 8px;border-radius:10px;'
                f'font-family:monospace;">🔥 HIGH</span>'
                if is_high else
                f'<span style="background:rgba(11,224,255,0.1);color:#0be0ff;'
                f'font-size:0.68rem;padding:2px 8px;border-radius:10px;'
                f'font-family:monospace;">📶 MED</span>'
            )
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:10px;
                        padding:10px 14px;margin:6px 0;display:flex;align-items:center;gap:12px;">
              <span style="font-size:1.1rem;">{icon}</span>
              <div style="flex:1;">
                <div style="font-weight:700;font-size:0.9rem;color:#e8f4fd;">
                  {sig['ticker']}
                  <span style="font-weight:400;color:#7a9abb;font-size:0.8rem;margin-left:6px;">
                    {sig['signal']}
                  </span>
                </div>
                <div style="font-size:0.78rem;color:#546e8a;font-family:monospace;margin-top:2px;">
                  {sig['detail']}  ·  {chg_str}
                </div>
              </div>
              {badge}
            </div>
            """, unsafe_allow_html=True)

    high_cnt = sum(1 for s in signals if s["strength"] == "HIGH")
    st.caption(f"**{len(signals)}** signals  ·  **{high_cnt}** HIGH  ·  Covers: {', '.join(TICKERS)}")

with corr_col:
    try:
        wl      = load_watchlist(tuple(TICKERS))
        rets_df = pd.DataFrame({t: wl[t]["Close"].pct_change().dropna()
                                for t in TICKERS}).dropna()
        corr    = rets_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, COLORS["signal_sell"]],
                [0.5, "#03050d"],
                [1.0, COLORS["signal_buy"]],
            ],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        ))
        apply_plotly_theme(fig_corr, "Return Correlation Matrix", height=320)
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Correlation unavailable: {e}")

qe_neon_divider()

# ── Market Snapshot table ─────────────────────────────────────────────────────
st.subheader("Market Snapshot  —  Watchlist")

with st.spinner("Loading market data…"):
    try:
        wl   = load_watchlist(tuple(TICKERS))
        rows = []
        for ticker, df in wl.items():
            close   = df["Close"]
            rets    = close.pct_change().dropna()
            rsi_val = float(rsi(close).iloc[-1])
            rows.append({
                "Ticker":        ticker,
                "Last Price":    round(float(close.iloc[-1]), 2),
                "1D %":          round(float(close.pct_change().iloc[-1] * 100), 2),
                "5D %":          round(float(close.pct_change(5).iloc[-1] * 100), 2),
                "1M %":          round(float(close.pct_change(21).iloc[-1] * 100), 2),
                "RSI (14)":      round(rsi_val, 1),
                "Ann Vol (20D)": round(float(rets.tail(20).std() * np.sqrt(252) * 100), 1),
                "Sharpe (6M)":   round(sharpe(rets.tail(126)), 2),
                "VaR 95%":       round(float(np.percentile(rets.tail(63), 5) * 100), 2),
            })

        snap_df = pd.DataFrame(rows).set_index("Ticker")

        def _pct_color(v):
            c = "#00f5a0" if v > 0 else "#ff4757" if v < 0 else "#7a9abb"
            return f"color:{c};font-weight:500"

        def _rsi_color(v):
            if v < 30: return "color:#00f5a0;font-weight:700"
            if v > 70: return "color:#ff4757;font-weight:700"
            return "color:#e8f4fd"

        def _sharpe_color(v):
            if v >= 1.5: return "color:#00f5a0;font-weight:600"
            if v < 0:    return "color:#ff4757;font-weight:600"
            return "color:#ffd700"

        styled = (
            snap_df.style
            .map(_pct_color,    subset=["1D %", "5D %", "1M %"])
            .map(_rsi_color,    subset=["RSI (14)"])
            .map(_sharpe_color, subset=["Sharpe (6M)"])
            .format({
                "Last Price":    "${:.2f}",
                "1D %":          "{:+.2f}%",
                "5D %":          "{:+.2f}%",
                "1M %":          "{:+.2f}%",
                "RSI (14)":      "{:.1f}",
                "Ann Vol (20D)": "{:.1f}%",
                "Sharpe (6M)":   "{:.2f}",
                "VaR 95%":       "{:.2f}%",
            })
        )
        st.dataframe(styled, use_container_width=True, height=220)

    except Exception as e:
        st.error(f"Could not load watchlist: {e}")

qe_neon_divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:14px 0 4px 0;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#546e8a;">
    QuantEdge  ·  {'Demo' if cfg.DEMO_MODE else 'Live'} Mode
    ·  Data via yfinance  ·  RFR {cfg.RISK_FREE_RATE:.1%}
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#546e8a;">
    Last updated {now_ist.strftime('%H:%M:%S IST')}
  </div>
</div>
""", unsafe_allow_html=True)