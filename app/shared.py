"""
app/ui_pages/_shared.py
──────────────────────────────────────────────────────────────────────────────
Shared utilities imported by every QuantEdge page:
  - DARK_CSS       : full dark theme injected once per page
  - apply_theme()  : call at top of every page
  - _start_str()   : global start date from session state
  - _top_bar()     : live metrics strip (Price, Volume, Vol, RSI, Signal)
  - _header()      : page title + subtitle
  - _sb_sec()      : sidebar section label
  - _ticker_sb()   : sidebar ticker text input
  - _tickers_sb()  : sidebar multi-ticker text input
  - render_data_engine_sidebar() : Static/Live mode controls in sidebar
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.data_engine import load_ticker_data, normalize_ticker, parse_ticker_list
from core.data import returns
from core.indicators import rsi as rsi_series
from utils.config import cfg

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section.main, .main .block-container {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stHeader"] { background: #0e1117 !important; }
[data-testid="stSidebar"] > div:first-child {
    background-color: #161b27 !important;
    border-right: 1px solid #252d3e;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c8ccd8 !important; }

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: #1a2035 !important;
    color: #e0e0e0 !important;
    border-color: #2a3350 !important;
}
[data-testid="stButton"] > button {
    background: #1565c0 !important; color: #fff !important;
    border: none !important; border-radius: 7px !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] > button:hover { background: #1976d2 !important; }
[data-testid="stButton"] > button[kind="secondary"] {
    background: #1e2840 !important; color: #c8ccd8 !important;
}
[data-testid="metric-container"] {
    background: #161b27 !important;
    border: 1px solid #252d3e !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] > div { color: #7a8099 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] > div { color: #e8ecf4 !important; font-weight: 600 !important; }
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #252d3e !important; }
[data-testid="stTabs"] [role="tab"] {
    color: #7a8099 !important; font-size: 13px !important; padding: 8px 16px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] {
    border: 1px solid #252d3e !important;
    border-radius: 8px !important;
    background: #161b27 !important;
}
[data-testid="stDataFrame"] { background: #161b27 !important; }
hr { border-color: #252d3e !important; }

/* ── Top metrics bar ──────────────────────────────────────────── */
.qe-top-bar {
    display: flex; flex-wrap: wrap; gap: 10px;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid #252d3e;
    margin-bottom: 20px;
}
.qe-chip {
    background: #161b27; border: 1px solid #252d3e; border-radius: 8px;
    padding: 8px 16px; display: flex; flex-direction: column; min-width: 120px;
}
.qe-chip-label {
    color: #5a6180; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.qe-chip-value { color: #e0e0e0; font-size: 15px; font-weight: 600; }
.qe-pos  { color: #2ecc71 !important; font-weight: 700 !important; }
.qe-neg  { color: #e74c3c !important; font-weight: 700 !important; }
.qe-warn { color: #f0c040 !important; font-weight: 700 !important; }
.qe-buy  { color: #2ecc71 !important; font-weight: 700 !important; }
.qe-sell { color: #e74c3c !important; font-weight: 700 !important; }
.qe-hold { color: #f0c040 !important; font-weight: 700 !important; }

/* ── Page elements ────────────────────────────────────────────── */
.qe-page-title { font-size: 22px; font-weight: 700; color: #e8ecf4; margin-bottom: 2px; }
.qe-page-sub   { color: #5a6180; font-size: 13px; margin-bottom: 14px; }
.qe-sb-section {
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #3d4560 !important; padding: 10px 0 4px 0;
}
.qe-logo {
    font-size: 18px; font-weight: 700; color: #4fc3f7 !important;
    display: flex; align-items: center; gap: 8px; padding-bottom: 14px;
}
.qe-demo-pill {
    background: #163556; color: #4fc3f7 !important; font-size: 10px;
    padding: 2px 7px; border-radius: 999px; font-weight: 600;
}
</style>
"""


def apply_theme():
    """Inject the dark CSS. Call once at the top of every page."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    # Logo in sidebar
    demo_pill = '<span class="qe-demo-pill">DEMO</span>' if cfg.DEMO_MODE else ""
    st.sidebar.markdown(
        f'<div class="qe-logo">📊 QuantEdge {demo_pill}</div>',
        unsafe_allow_html=True,
    )


def render_data_engine_sidebar():
    """Data Engine section in sidebar (Static/Live mode, start date, interval)."""
    _sb_sec("Data Engine")
    data_mode = st.sidebar.radio("Mode", ["Static", "Live"], key="qe_data_mode", horizontal=True)
    if data_mode == "Static":
        st.sidebar.date_input(
            "Start Date",
            value=pd.to_datetime(cfg.DEFAULT_START).date(),
            key="qe_static_start_date",
        )
    else:
        st.sidebar.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], key="qe_live_interval")
        st.sidebar.selectbox("Lookback", ["1d","2d","5d"], key="qe_live_lookback")
        st.sidebar.slider("Refresh (s)", 1, 300, 60, key="qe_refresh_seconds")
    st.sidebar.markdown("---")


def _start_str() -> str:
    """Return current global start date as a string."""
    mode = st.session_state.get("qe_data_mode", "Static")
    if mode == "Static":
        return str(st.session_state.get("qe_static_start_date", cfg.DEFAULT_START))
    return str((pd.Timestamp.today() - pd.Timedelta(days=5)).date())


def _ticker_sb(key: str, default: str | None = None) -> str:
    d   = default or (cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG")
    raw = st.sidebar.text_input("Ticker", value=d, key=key)
    return normalize_ticker(raw) or d


def _tickers_sb(key: str, default: list | None = None) -> list:
    d   = default or cfg.DEFAULT_TICKERS
    raw = st.sidebar.text_input("Tickers (comma-separated)", value=",".join(d), key=key)
    r   = parse_ticker_list(raw)
    return r if r else d


def _header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="qe-page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="qe-page-sub">{subtitle}</div>', unsafe_allow_html=True)


def _sb_sec(label: str):
    st.sidebar.markdown(f'<div class="qe-sb-section">{label}</div>', unsafe_allow_html=True)


def _top_bar(ticker: str) -> pd.DataFrame:
    """Render live metrics strip. Returns the loaded DataFrame."""
    try:
        # Always load daily data for the top bar so Price/Volume are
        # consistent between Static and Live modes (intraday bars have
        # different Volume scales than daily bars).
        from core.data import get_ohlcv
        df_daily = get_ohlcv(ticker, _start_str())
        df       = load_ticker_data(ticker, start=_start_str())   # used as return value

        src = df_daily if not df_daily.empty else df
        if src.empty:
            st.warning(f"No data for {ticker}")
            return df

        last  = float(src["Close"].iloc[-1])
        prev  = float(src["Close"].iloc[-2]) if len(src) > 1 else last
        chg   = (last - prev) / prev * 100

        # Volume: use 20-day average daily volume to avoid intraday distortion
        vol_col = src["Volume"].replace(0, float("nan")).dropna()
        avg_vol = int(vol_col.tail(20).mean()) if len(vol_col) >= 5 else 0
        vol_s   = f"{avg_vol/1e6:.1f}M" if avg_vol >= 1_000_000 else f"{avg_vol/1e3:.0f}K"

        av    = returns(src).std() * (252**0.5) * 100
        rsi_v = float(rsi_series(src["Close"]).iloc[-1])

        chg_cls = "qe-pos" if chg >= 0 else "qe-neg"
        arrow   = "▲" if chg >= 0 else "▼"

        if   rsi_v < 35: sig_lbl, sig_cls = "BUY ▲",  "qe-buy"
        elif rsi_v > 65: sig_lbl, sig_cls = "SELL ▼", "qe-sell"
        else:            sig_lbl, sig_cls = "HOLD —",  "qe-hold"

        st.markdown(f"""
        <div class="qe-top-bar">
            <div class="qe-chip">
                <div class="qe-chip-label">Price</div>
                <span class="qe-chip-value">
                    ${last:,.2f}&nbsp;<span class="{chg_cls}">{arrow}{abs(chg):.2f}%</span>
                </span>
            </div>
            <div class="qe-chip">
                <div class="qe-chip-label">Avg Vol (20d)</div>
                <span class="qe-chip-value">{vol_s}</span>
            </div>
            <div class="qe-chip">
                <div class="qe-chip-label">Volatility (Ann.)</div>
                <span class="qe-chip-value qe-warn">{av:.1f}%</span>
            </div>
            <div class="qe-chip">
                <div class="qe-chip-label">RSI (14)</div>
                <span class="qe-chip-value">{rsi_v:.1f}</span>
            </div>
            <div class="qe-chip">
                <div class="qe-chip-label">Signal</div>
                <span class="{sig_cls}" style="font-size:15px;font-weight:700">{sig_lbl}</span>
            </div>
            <div class="qe-chip">
                <div class="qe-chip-label">Ticker</div>
                <span class="qe-chip-value" style="color:#4fc3f7">{ticker}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return df
    except Exception as exc:
        st.caption(f"⚠️ Top bar unavailable: {exc}")
        return pd.DataFrame()