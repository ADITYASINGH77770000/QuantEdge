"""QuantEdge dashboard — image-matched layout with all bug fixes applied."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data_engine import load_ticker_data
from app.ui_pages._shared import (
    _header,
    _sb_sec,
    _start_str,
    _ticker_sb,
    _top_bar,
    apply_theme,
    render_data_engine_sidebar,
)
from core.data import returns
from core.metrics import summary_table
from utils.config import cfg
try:
    from utils.theme import qe_faq_section
except ImportError:
    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        st.markdown("---")
        st.markdown(f"### {title}")
        for question, answer in faqs:
            with st.expander(question):
                st.write(answer)


st.set_page_config(page_title="Dashboard | QuantEdge", page_icon="📈", layout="wide")
apply_theme()

st.markdown(
    """
<style>
/* ── Sidebar nav styling ── */
section[data-testid="stSidebar"] {
    background: #0f1623 !important;
    border-right: 1px solid #1e2a3e !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}

/* ── Boxed panel cards ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, #131c2e 0%, #0f1520 100%);
    border: 1px solid #1e2d47 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.5rem 0.65rem 0.65rem 0.65rem;
}

/* ── Top metric bar ── */
.qe-topbar {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.qe-topbar-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #131c2e;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 10px 18px;
    min-width: 160px;
    flex: 1;
}
.qe-topbar-label {
    color: #5a6a87;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.qe-topbar-val {
    color: #e2e8f4;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.qe-topbar-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 5px;
}
.badge-green  { background: rgba(74,222,128,.15); color: #4ade80; }
.badge-red    { background: rgba(248,113,113,.15); color: #f87171; }
.badge-yellow { background: rgba(250,204,21,.13);  color: #facc15; }
.badge-blue   { background: rgba(125,211,252,.13); color: #7dd3fc; }

/* ── Section heading ── */
.qe-section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0 14px 0;
}
.qe-section-head h3 {
    margin: 0;
    color: #c8d3e8;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.qe-section-dot {
    width: 8px; height: 8px;
    border-radius: 999px;
    background: #2d4060;
    box-shadow: 0 0 0 4px rgba(45,64,96,.2);
}
.qe-section-line {
    height: 1px; flex: 1;
    background: linear-gradient(to right, #1e2d47, transparent);
}

/* ── Panel header inside each card ── */
.qe-panel-title {
    color: #d4ddf0;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 2px;
    letter-spacing: -0.01em;
}
.qe-panel-sub {
    color: #4a5878;
    font-size: 11px;
    margin-bottom: 8px;
}
.qe-panel-rule {
    height: 1px;
    background: linear-gradient(to right, #1e2d47, transparent);
    margin: 0 0 10px 0;
}

/* ── Metric rows (Key Metrics / Risk & Return) ── */
.qe-stat-table { display: grid; gap: 7px; }
.qe-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border: 1px solid #1a2840;
    border-radius: 9px;
    background: linear-gradient(180deg, #111928 0%, #0d1421 100%);
}
.qe-stat-label {
    color: #c8d3e8;
    font-size: 13px;
    font-weight: 600;
}
.qe-stat-sub {
    color: #3d5070;
    font-size: 10px;
    margin-top: 2px;
}
.qe-stat-value {
    font-size: 15px;
    font-weight: 700;
    text-align: right;
    white-space: nowrap;
}
.qe-metric-pos     { color: #4ade80 !important; }
.qe-metric-neg     { color: #f87171 !important; }
.qe-metric-neutral { color: #facc15 !important; }
.qe-metric-na      { color: #4a5878 !important; }

/* ── Plotly chart radius ── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
render_data_engine_sidebar()
_sb_sec("Controls")
ticker = _ticker_sb("dash_ticker")
ma_periods = st.sidebar.multiselect(
    "Moving Averages",
    [10, 20, 50, 100, 200],
    default=[20, 50],
    help="Overlay selected moving averages on the price chart.",
)
st.sidebar.markdown("---")
_sb_sec("Chart Options")
show_vol_panel     = st.sidebar.checkbox("Volume panel",     value=True, key="dash_vol")
show_vol21_panel   = st.sidebar.checkbox("Volatility panel", value=True, key="dash_v21")
show_rsi_macd      = st.sidebar.checkbox("RSI & MACD panel", value=True, key="dash_rsi")
candle_style = st.sidebar.radio(
    "Price style",
    ["Candlestick", "Line"],
    horizontal=True,
    key="dash_style",
)

# ── Page header ────────────────────────────────────────────────────────────────
_header("📈 Dashboard", "Price · Volume · Volatility · RSI · MACD · Metrics")

# ── Data load ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_ticker_data(ticker, start=_start_str())

if df.empty:
    st.warning("No data available — try a different ticker or start date.")
    st.stop()

ret = returns(df)
met = summary_table(ret, cfg.RISK_FREE_RATE)

# ── DEBUG: uncomment once to confirm exact met keys, then remove ───────────────
# st.write("summary_table keys →", list(met.keys()))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_get(key: str) -> str:
    """Return formatted metric value; guard None / NaN."""
    val = met.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, float) and pd.isna(val):
        return "N/A"
    return str(val)


def _color_class(key: str, value: str) -> str:
    if value == "N/A":
        return "qe-metric-na"
    try:
        parsed = float(
            value.replace("%", "")
                 .replace("x", "")
                 .replace("$", "")
                 .replace(",", "")
                 .strip()
        )
        if key in {"Max Drawdown", "CVaR 95%", "VaR 95%"}:
            return "qe-metric-neg"
        if parsed > 0:
            return "qe-metric-pos"
        if parsed < 0:
            return "qe-metric-neg"
    except Exception:
        pass
    return "qe-metric-neutral" if "Volatility" in key else ""


def _metric_row(label: str, sub: str) -> str:
    value       = _safe_get(label)
    value_class = _color_class(label, value)
    return f"""
    <div class="qe-stat-row">
        <div>
            <div class="qe-stat-label">{label}</div>
            <div class="qe-stat-sub">{sub}</div>
        </div>
        <div class="qe-stat-value {value_class}">{value}</div>
    </div>"""


# ── Top metric bar ─────────────────────────────────────────────────────────────
last_close  = df["Close"].iloc[-1]
prev_close  = df["Close"].iloc[-2] if len(df) > 1 else last_close
pct_chg     = (last_close - prev_close) / prev_close * 100
vol_today   = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
ann_vol_pct = _safe_get("Ann. Volatility")
sharpe_val  = _safe_get("Sharpe")

chg_sign    = "+" if pct_chg >= 0 else ""
chg_cls     = "badge-green" if pct_chg >= 0 else "badge-red"
vol_fmt     = f"{vol_today/1e6:.2f}M" if vol_today >= 1e6 else f"{vol_today/1e3:.0f}K"

st.markdown(
    f"""
<div class="qe-topbar">
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Price</div>
            <div class="qe-topbar-val">${last_close:,.2f}
                <span class="qe-topbar-badge {chg_cls}">{chg_sign}{pct_chg:.2f}%</span>
            </div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Volume</div>
            <div class="qe-topbar-val">{vol_fmt}</div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Volatility</div>
            <div class="qe-topbar-val">
                <span class="qe-topbar-badge badge-yellow">{ann_vol_pct}</span>
            </div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Sharpe</div>
            <div class="qe-topbar-val">
                <span class="qe-topbar-badge badge-blue">{sharpe_val}</span>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Section: Market Overview ───────────────────────────────────────────────────
st.markdown(
    """
<div class="qe-section-head">
    <h3>Market Overview</h3>
    <div class="qe-section-dot"></div>
    <div class="qe-section-line"></div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Figure builders (unchanged logic) ─────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1421",
    margin=dict(l=8, r=8, t=8, b=8),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left",   x=0,
        bgcolor="rgba(13,20,33,0.85)",
        bordercolor="#1e2d47",
        borderwidth=1,
        font=dict(size=10),
    ),
)
GRID = dict(showgrid=True, gridcolor="#172035", gridwidth=0.6)


def _build_price_figure() -> go.Figure:
    has_volume  = show_vol_panel and "Volume" in df.columns
    rows        = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    if candle_style == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"],
                low=df["Low"],   close=df["Close"],
                name="Price",
                increasing_line_color="#4ade80",
                decreasing_line_color="#f87171",
                increasing_fillcolor="rgba(74,222,128,0.85)",
                decreasing_fillcolor="rgba(248,113,113,0.85)",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"],
                name="Close",
                line=dict(color="#7dd3fc", width=2),
                fill="tozeroy",
                fillcolor="rgba(125,211,252,0.07)",
            ),
            row=1, col=1,
        )

    ma_colors = {10: "#facc15", 20: "#a78bfa", 50: "#fb923c", 100: "#34d399", 200: "#f472b6"}
    for period in ma_periods:
        ma = df["Close"].rolling(period).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index, y=ma.values,
                name=f"MA {period}",
                line=dict(color=ma_colors.get(period, "#ffffff"), width=1.4, dash="dot"),
            ),
            row=1, col=1,
        )

    if has_volume:
        bar_colors = [
            "rgba(74,222,128,0.65)" if c >= o else "rgba(248,113,113,0.65)"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=2, col=1,
        )
        avg_vol = df["Volume"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=avg_vol.index, y=avg_vol.values,
                name="Vol MA(20)",
                line=dict(color="#facc15", width=1.2, dash="dot"),
            ),
            row=2, col=1,
        )
        fig.update_yaxes(
            title_text="Volume", row=2, col=1,
            tickfont=dict(size=9), title_font=dict(size=9),
        )

    fig.update_layout(**PLOT_THEME, height=400, xaxis_rangeslider_visible=False)
    fig.update_xaxes(**GRID, rangeslider_visible=False)
    fig.update_yaxes(**GRID)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickfont=dict(size=9), title_font=dict(size=9))
    return fig


def _build_rsi_macd_figure() -> go.Figure:
    """RSI + MACD + Signal panel — mirrors image right-hand chart."""
    close = df["Close"]

    # RSI
    delta    = close.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0, float("nan"))
    rsi      = 100 - (100 / (1 + rs))

    # MACD
    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    signal   = macd.ewm(span=9, adjust=False).mean()
    hist     = macd - signal

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.38, 0.24],
        subplot_titles=["RSI (14)", "MACD", ""],
    )

    # RSI panel
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.07)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(74,222,128,0.07)",  line_width=0, row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.4)", line_width=1, row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(74,222,128,0.4)",  line_width=1, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=rsi.index, y=rsi.values, name="RSI",
                   line=dict(color="#a78bfa", width=2)),
        row=1, col=1,
    )

    # MACD line + signal
    fig.add_trace(
        go.Scatter(x=macd.index, y=macd.values, name="MACD",
                   line=dict(color="#7dd3fc", width=1.8)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=signal.index, y=signal.values, name="Signal",
                   line=dict(color="#fb923c", width=1.5)),
        row=2, col=1,
    )

    # Histogram
    hist_colors = [
        "rgba(74,222,128,0.65)" if v >= 0 else "rgba(248,113,113,0.65)"
        for v in hist
    ]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist.values, name="Histogram",
               marker_color=hist_colors, showlegend=False),
        row=3, col=1,
    )

    fig.update_layout(**PLOT_THEME, height=400)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    fig.update_yaxes(title_text="RSI",  row=1, col=1, tickfont=dict(size=9), range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Hist", row=3, col=1, tickfont=dict(size=9))
    return fig


def _build_volatility_figure() -> go.Figure:
    vol21    = ret.rolling(21).std() * (252 ** 0.5) * 100
    vol_slow = vol21.rolling(21).mean()
    vmax     = max(float(vol21.max()) if not vol21.dropna().empty else 0.0, 45.0)

    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=20,        fillcolor="rgba(74,222,128,0.04)",  line_width=0)
    fig.add_hrect(y0=20, y1=40,        fillcolor="rgba(250,204,21,0.05)",  line_width=0)
    fig.add_hrect(y0=40, y1=vmax*1.05, fillcolor="rgba(248,113,113,0.06)", line_width=0)
    fig.add_trace(go.Scatter(
        x=vol21.index, y=vol21.values,
        name="21d Vol",
        line=dict(color="#a78bfa", width=2.2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=vol_slow.index, y=vol_slow.values,
        name="Trend",
        line=dict(color="#7dd3fc", width=1.4, dash="dot"),
    ))
    fig.update_layout(**PLOT_THEME, height=280)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID, ticksuffix="%", title_text="Volatility")
    return fig


# Sentiment panel removed


# ── Row 1: Price Chart  |  RSI & MACD ─────────────────────────────────────────
col_left, col_right = st.columns([1.05, 0.95], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Price Chart</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Candles · Moving averages · Volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.plotly_chart(_build_price_figure(), use_container_width=True, config={"displayModeBar": False})

with col_right:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">RSI &amp; MACD Indicators</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Momentum · Trend divergence · Histogram</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        if show_rsi_macd:
            st.plotly_chart(_build_rsi_macd_figure(), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Enable `RSI & MACD panel` in the sidebar to show this card.")


# ── Row 2: Key Metrics  |  News Sentiment ─────────────────────────────────────
col_bl, col_br = st.columns([0.95, 1.05], gap="medium")

with col_bl:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Key Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Core performance numbers</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qe-stat-table">'
            + _metric_row("Sharpe",       "Risk-adjusted return")
            + _metric_row("Sortino",      "Downside-risk adjusted return")
            + _metric_row("Max Drawdown", "Largest peak-to-trough decline")
            + _metric_row("Win Rate",     "Share of positive periods")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Volatility sparkline below metrics (matches image Risk Level row) ──
        if show_vol21_panel:
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(_build_volatility_figure(), use_container_width=True, config={"displayModeBar": False})

with col_br:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Risk &amp; Return Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Return · Drawdown · VaR · CVaR</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qe-stat-table">'
            + _metric_row("CAGR",          "Compound annual growth rate")
            + _metric_row("Ann. Return",   "Arithmetic annual return")
            + _metric_row("Ann. Volatility","Annualised standard deviation")
            + _metric_row("Calmar",        "CAGR divided by max drawdown")
            + _metric_row("VaR 95%",       "One-day 95% value at risk")
            + _metric_row("CVaR 95%",      "Expected shortfall beyond VaR")
            + "</div>",
            unsafe_allow_html=True,
        )

        # Sentiment panel removed


# ── Row 3: Recent OHLCV ────────────────────────────────────────────────────────
st.markdown(
    """
<div class="qe-section-head" style="margin-top:18px;">
    <h3>Recent OHLCV Data</h3>
    <div class="qe-section-dot"></div>
    <div class="qe-section-line"></div>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="qe-panel-sub">Latest 30 sessions — colour-coded by session direction</div>', unsafe_allow_html=True)
    st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)

    tail = df.tail(30).copy()
    if "Volume" in tail.columns:
        tail["Volume"] = tail["Volume"].apply(
            lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if not pd.isna(v) else "-")
        )

    def _style_row(row):
        color = "color: #4ade80" if row.get("Close", 0) >= row.get("Open", 0) else "color: #f87171"
        return [color] * len(row)

    num_cols  = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in tail.columns]
    formatted = tail.copy()
    # BUG FIX: capture loop variable with default arg to avoid closure issues
    for col in num_cols:
        formatted[col] = formatted[col].map(
            lambda v, _col=col: f"${v:,.2f}" if not pd.isna(v) else "-"
        )

    st.dataframe(
        formatted.iloc[::-1].style.apply(_style_row, axis=1),
        use_container_width=True,
        height=280,
    )

qe_faq_section("FAQs", [
    ("What should I look at first on the dashboard?", "Start with the top metrics row and the latest OHLCV table. They give a quick read on trend, volatility, and recent market behavior."),
    ("How does this dashboard help me trade?", "It condenses the current state of the symbol into one screen so you can compare trend, risk, and momentum before moving to deeper analysis pages."),
    ("Why is the recent data table important?", "It shows the freshest sessions and helps you spot gaps, large candles, or unusual volume before trusting a signal."),
    ("When should I switch to another page?", "Use the dashboard as the starting point, then move to signals, risk, or backtest once you want a more specific answer."),
])
