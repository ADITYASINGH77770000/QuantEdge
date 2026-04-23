# """
# app/main.py
# ──────────────────────────────────────────────────────────────────────────────
# QuantEdge — main entry point and navigation router.
# Run with: streamlit run app/main.py
# """

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# import streamlit as st
# from utils.config import cfg

# st.set_page_config(
#     page_title="QuantEdge |  Quant Research",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── Sidebar branding ──────────────────────────────────────────────────────────
# st.sidebar.title("📊 QuantEdge")
# st.sidebar.caption("Institutional Quant Research Platform")
# if cfg.DEMO_MODE:
#     st.sidebar.info("🎭 **Demo Mode** — synthetic data, no API keys needed.")
# st.sidebar.divider()

# # ── Home page ─────────────────────────────────────────────────────────────────
# st.title("QuantEdge — Institutional Quant Research Platform")
# st.markdown("""
# Welcome to **QuantEdge**, upgraded from Finnovix into a hedge-fund-grade research platform.

# | Phase | Modules | Status |
# |-------|---------|--------|
# | Foundation | Dashboard, Graphs, Auditing, Prediction | ✅ Refactored |
# | Quant Core | Signals, Backtest, Portfolio, Risk | ✅ New |
# | Research Grade | Factors, Regime | ✅ Extreme |
# | Polish | Tests, Docker, PDF export, Demo mode | ✅ Complete |

# **Navigate** using the sidebar pages ←
# """)

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Modules", "11")
# col2.metric("Build Phases", "4")
# col3.metric("Lines of Code", "3,000+")
# col4.metric("Mode", "Demo" if cfg.DEMO_MODE else "Live")

# st.divider()
# st.subheader("Quick Start")
# st.code("""
# # 1. Install dependencies
# pip install -r requirements.txt

# # 2. Configure secrets
# cp .env.example .env
# # Edit .env with your API keys

# # 3. Run the app
# streamlit run app/main.py
# """, language="bash")

"""
app/main.py
──────────────────────────────────────────────────────────────────────────────
QuantEdge — Command center home screen.
Shows live system health, portfolio KPIs, regime, signals, and market snapshot.
Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np

from utils.config import cfg
from core.data import get_ohlcv, get_multi_ohlcv
from core.metrics import sharpe, sortino, max_drawdown, var_historical
from core.indicators import rsi, macd, add_all_indicators

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantEdge | Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📡 QuantEdge")
st.sidebar.caption("Quant Research Platform")
if cfg.DEMO_MODE:
    st.sidebar.info("🎭 **Demo Mode** — synthetic data active.")
st.sidebar.divider()
st.sidebar.markdown("**Navigation**")
st.sidebar.page_link("pages/01_dashboard.py",  label="📊 Dashboard")
st.sidebar.page_link("pages/06_signals.py",    label="⚡ Signals")
st.sidebar.page_link("pages/07_backtest.py",   label="🔁 Backtest")
st.sidebar.page_link("pages/08_portfolio.py",  label="🏦 Portfolio")
st.sidebar.page_link("pages/09_risk.py",       label="⚠️ Risk")
st.sidebar.page_link("pages/05_alerts.py",     label="🔔 Alerts")
st.sidebar.page_link("pages/12_regime.py",     label="🌐 Regime")


# ── Data helpers ──────────────────────────────────────────────────────────────
TICKERS = cfg.DEFAULT_TICKERS  # e.g. ["GOOG", "NVDA", "META", "AMZN"]

@st.cache_data(ttl=300, show_spinner=False)
def load_watchlist(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return get_multi_ohlcv(tickers)

@st.cache_data(ttl=300, show_spinner=False)
def compute_portfolio_metrics(tickers: list[str]) -> dict:
    """
    Equal-weight portfolio of all tickers.
    Returns key risk/return metrics computed from real pipeline.
    """
    data = get_multi_ohlcv(tickers)
    rets_dict = {t: data[t]["Close"].pct_change().dropna() for t in tickers}

    # Align and equal-weight
    rets_df = pd.DataFrame(rets_dict).dropna()
    port_rets = rets_df.mean(axis=1)  # equal weight

    # Beta vs SPY-like market proxy (use average of all tickers as market)
    mkt = rets_df.mean(axis=1)
    cov = np.cov(port_rets, mkt)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

    return {
        "sharpe_1y":   round(sharpe(port_rets.tail(252)),  2),
        "sortino_1y":  round(sortino(port_rets.tail(252)), 2),
        "max_dd":      round(max_drawdown(port_rets) * 100, 2),
        "var_95_1d":   round(var_historical(port_rets, confidence=0.95) * 100, 2),
        "beta":        round(float(beta), 2),
        "n_days":      len(port_rets),
        "cagr":        round(((1 + port_rets).prod() ** (252 / len(port_rets)) - 1) * 100, 2),
    }

@st.cache_data(ttl=300, show_spinner=False)
def detect_regime_simple(ticker: str) -> dict:
    """
    Fast rule-based regime: uses 50/200 SMA crossover + VIX-proxy from vol.
    Returns regime string + confidence + days_in_regime.
    """
    df = get_ohlcv(ticker)
    close = df["Close"]
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    vol_20   = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_avg  = vol_20.rolling(60).mean()

    latest_sma50  = float(sma50.iloc[-1])
    latest_sma200 = float(sma200.iloc[-1])
    latest_vol    = float(vol_20.iloc[-1])
    avg_vol       = float(vol_avg.iloc[-1])

    # Bull: 50 > 200 and volatility normal
    if latest_sma50 > latest_sma200 and latest_vol < avg_vol * 1.2:
        regime, conf = "BULL", 0.75 + 0.1 * (latest_sma50 / latest_sma200 - 1) * 10
    # Bear: 50 < 200 and elevated vol
    elif latest_sma50 < latest_sma200 and latest_vol > avg_vol:
        regime, conf = "BEAR", 0.70 + 0.1 * (1 - latest_sma50 / latest_sma200) * 10
    # High vol sideways
    elif latest_vol > avg_vol * 1.4:
        regime, conf = "VOLATILE", 0.65
    else:
        regime, conf = "SIDEWAYS", 0.60

    # Days in current regime (count backward)
    if latest_sma50 > latest_sma200:
        cross_series = sma50 > sma200
    else:
        cross_series = sma50 < sma200
    days_in = int((cross_series[::-1].cumprod()).sum())

    return {
        "regime":   regime,
        "conf":     round(min(conf, 0.95), 2),
        "days_in":  days_in,
    }

@st.cache_data(ttl=300, show_spinner=False)
def scan_signals(tickers: list[str]) -> list[dict]:
    """Scan all tickers for RSI + MACD signals. Returns list of active signals."""
    signals = []
    for ticker in tickers:
        df = get_ohlcv(ticker)
        df = add_all_indicators(df)
        rsi_val  = float(df["RSI"].iloc[-1])
        macd_val = float(df["MACD"].iloc[-1])
        sig_val  = float(df["Signal"].iloc[-1])
        macd_prev = float(df["MACD"].iloc[-2])
        sig_prev  = float(df["Signal"].iloc[-2])
        price     = float(df["Close"].iloc[-1])
        chg_1d    = float(df["Close"].pct_change().iloc[-1] * 100)

        if rsi_val < 30:
            signals.append({"ticker": ticker, "signal": "RSI Oversold",
                             "detail": f"RSI {rsi_val:.1f}", "type": "BUY", "strength": "HIGH"})
        elif rsi_val > 70:
            signals.append({"ticker": ticker, "signal": "RSI Overbought",
                             "detail": f"RSI {rsi_val:.1f}", "type": "SELL", "strength": "HIGH"})

        if macd_val > sig_val and macd_prev <= sig_prev:
            signals.append({"ticker": ticker, "signal": "MACD Bull Cross",
                             "detail": f"MACD {macd_val:.3f}", "type": "BUY", "strength": "MEDIUM"})
        elif macd_val < sig_val and macd_prev >= sig_prev:
            signals.append({"ticker": ticker, "signal": "MACD Bear Cross",
                             "detail": f"MACD {macd_val:.3f}", "type": "SELL", "strength": "MEDIUM"})

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Zone 1: System health bar ─────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_ist = now_utc.astimezone(
    __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("Asia/Kolkata")
)

h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("🕐 Time (IST)",    now_ist.strftime("%H:%M:%S"))
h2.metric("📡 Data Feed",     "DEMO" if cfg.DEMO_MODE else "LIVE")
h3.metric("⚡ Cache TTL",     f"{cfg.CACHE_TTL}s")
h4.metric("🔢 Tickers",       str(len(TICKERS)))
h5.metric("💾 Mode",          "Demo" if cfg.DEMO_MODE else "Live")
h6.metric("📅 As of",         now_ist.strftime("%d %b %Y"))

st.divider()


# ── Zone 2: Live Portfolio KPIs ───────────────────────────────────────────────
st.subheader("Portfolio KPIs — Equal-Weight Across Watchlist")

with st.spinner("Computing portfolio metrics…"):
    try:
        pm = compute_portfolio_metrics(TICKERS)
        ok = True
    except Exception as e:
        st.warning(f"Could not compute live metrics: {e}")
        ok = False

if ok:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Sharpe (1Y)",
        pm["sharpe_1y"],
        delta="Good" if pm["sharpe_1y"] > 1.0 else "Below target",
        delta_color="normal" if pm["sharpe_1y"] > 1.0 else "inverse",
    )
    k2.metric(
        "Sortino (1Y)",
        pm["sortino_1y"],
        delta="Good" if pm["sortino_1y"] > 1.2 else "Below target",
        delta_color="normal" if pm["sortino_1y"] > 1.2 else "inverse",
    )
    k3.metric(
        "Max Drawdown",
        f"{pm['max_dd']}%",
        delta="Within limit" if abs(pm["max_dd"]) < 15 else "Breached",
        delta_color="normal" if abs(pm["max_dd"]) < 15 else "inverse",
    )
    k4.metric(
        "VaR 95% (1D)",
        f"{pm['var_95_1d']}%",
        delta="Within limit" if abs(pm["var_95_1d"]) < 2.0 else "Elevated",
        delta_color="normal" if abs(pm["var_95_1d"]) < 2.0 else "inverse",
    )
    k5.metric("Portfolio Beta", pm["beta"])
    k6.metric("CAGR",          f"{pm['cagr']}%")

st.divider()


# ── Zone 3 + 4: Regime & Signals side-by-side ────────────────────────────────
col_regime, col_signals = st.columns([1, 2])

with col_regime:
    st.subheader("Market Regime")
    with st.spinner("Detecting regime…"):
        try:
            reg = detect_regime_simple(TICKERS[0])
        except Exception:
            reg = {"regime": "UNKNOWN", "conf": 0.0, "days_in": 0}

    REGIME_COLORS = {
        "BULL":     "normal",
        "BEAR":     "inverse",
        "VOLATILE": "off",
        "SIDEWAYS": "off",
        "UNKNOWN":  "off",
    }
    REGIME_ICONS = {
        "BULL": "🟢", "BEAR": "🔴", "VOLATILE": "🟡", "SIDEWAYS": "⚪", "UNKNOWN": "❓"
    }
    icon = REGIME_ICONS.get(reg["regime"], "❓")
    st.metric(
        f"{icon} Regime ({TICKERS[0]})",
        reg["regime"],
        delta=f"Confidence {reg['conf']:.0%}",
        delta_color=REGIME_COLORS.get(reg["regime"], "off"),
    )
    st.caption(f"Active for **{reg['days_in']}** trading days")

    if reg["regime"] == "BEAR":
        st.error("⚠️ Bear regime detected — review risk limits.")
    elif reg["regime"] == "VOLATILE":
        st.warning("⚡ High volatility regime — reduce position sizing.")
    elif reg["regime"] == "BULL":
        st.success("✅ Bull regime — signal models active.")
    else:
        st.info("↔️ Sideways regime — mean-reversion strategies favoured.")

with col_signals:
    st.subheader("Active Signals — Watchlist Scan")
    with st.spinner("Scanning signals…"):
        try:
            signals = scan_signals(TICKERS)
        except Exception:
            signals = []

    if not signals:
        st.info("No active signals at this time.")
    else:
        for sig in signals:
            color = "🟢" if sig["type"] == "BUY" else "🔴"
            strength_badge = "🔥 HIGH" if sig["strength"] == "HIGH" else "📶 MED"
            if sig["strength"] == "HIGH":
                st.warning(
                    f"{color} **{sig['ticker']}** — {sig['signal']} | "
                    f"{sig['detail']} | {strength_badge}"
                )
            else:
                st.info(
                    f"{color} **{sig['ticker']}** — {sig['signal']} | "
                    f"{sig['detail']} | {strength_badge}"
                )

    high_count = sum(1 for s in signals if s["strength"] == "HIGH")
    st.caption(
        f"**{len(signals)}** signals active | "
        f"**{high_count}** HIGH strength | "
        f"Scan covers: {', '.join(TICKERS)}"
    )

st.divider()


# ── Zone 5: Market Snapshot ───────────────────────────────────────────────────
st.subheader("Market Snapshot — Watchlist")

with st.spinner("Loading market data…"):
    try:
        watchlist = load_watchlist(TICKERS)
        rows = []
        for ticker, df in watchlist.items():
            close   = df["Close"]
            rets    = close.pct_change().dropna()
            rsi_val = float(rsi(close).iloc[-1])
            rows.append({
                "Ticker":       ticker,
                "Last Price":   round(float(close.iloc[-1]), 2),
                "1D %":         round(float(close.pct_change().iloc[-1] * 100), 2),
                "5D %":         round(float(close.pct_change(5).iloc[-1] * 100), 2),
                "1M %":         round(float(close.pct_change(21).iloc[-1] * 100), 2),
                "RSI (14)":     round(rsi_val, 1),
                "Vol (20D ann)":round(float(rets.tail(20).std() * np.sqrt(252) * 100), 1),
                "Sharpe (6M)":  round(sharpe(rets.tail(126)), 2),
            })

        snapshot_df = pd.DataFrame(rows).set_index("Ticker")

        def color_pct(val):
            color = "#1a9641" if val > 0 else "#d7191c" if val < 0 else "#888"
            return f"color: {color}; font-weight: 500"

        def color_rsi(val):
            if val < 30:
                return "color: #1a9641; font-weight: 600"
            elif val > 70:
                return "color: #d7191c; font-weight: 600"
            return ""

        styled = (
            snapshot_df.style
            .applymap(color_pct, subset=["1D %", "5D %", "1M %"])
            .applymap(color_rsi, subset=["RSI (14)"])
            .format({
                "Last Price":    "${:.2f}",
                "1D %":          "{:+.2f}%",
                "5D %":          "{:+.2f}%",
                "1M %":          "{:+.2f}%",
                "RSI (14)":      "{:.1f}",
                "Vol (20D ann)": "{:.1f}%",
                "Sharpe (6M)":   "{:.2f}",
            })
        )
        st.dataframe(styled, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load watchlist data: {e}")

st.divider()


# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    f"QuantEdge · {'Demo' if cfg.DEMO_MODE else 'Live'} Mode · "
    f"Data via yfinance · Risk-free rate {cfg.RISK_FREE_RATE:.1%} · "
    f"Last computed {now_ist.strftime('%H:%M:%S IST')}"
)