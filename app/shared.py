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
DARK_CSS = ""

def apply_theme():
    """Inject the dark CSS via the unified utils.theme engine."""
    from utils.theme import apply_quantedge_theme
    apply_quantedge_theme()

def render_data_engine_sidebar():
    """Branded nav + Data Engine section in sidebar — called by every page."""

    st.sidebar.divider()

    # ── Data Engine controls ───────────────────────────────────────────────────
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
    st.markdown(
        '<hr style="border:none;height:1px;background:#252d3e;margin:10px 0 18px 0;">',
        unsafe_allow_html=True,
    )


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
