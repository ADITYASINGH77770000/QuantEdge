"""
app/pages/graphs_research.py — QuantEdge Graph Research Lab
════════════════════════════════════════════════════════════
All original graph logic preserved exactly.

AI LAYER — Gemini AI Graph Decoder (bottom of page, same 3-layer design
as 08_portfolio.py and 07_factors.py):

  KEY DESIGN DECISION (graph-specific):
  Because the user selects ONE graph at a time from 7 distinct research
  views, the AI layer is GRAPH-AWARE. Each of the 7 feature views has:
    • Its own deterministic danger flags (tuned to that view's metrics)
    • Its own context builder (packages only the relevant data)
    • Its own Gemini system prompt (threshold knowledge specific to that view)
    • Its own fallback prose explanation

  A single dispatcher (_dispatch_*) routes to the correct handler based on
  feature_key. This means Gemini always receives a tight, relevant context
  and never receives data from a different graph type.

  Layer 1: Deterministic danger flags — always shown, no AI, graph-specific
  Layer 2: Context builder + "Decode for Me" button — graph-specific context
  Layer 3: Gemini 4-section output — graph-specific system prompt

  Architecture:
    _flags_<feature>()          ← 7 deterministic flag functions
    _context_<feature>()        ← 7 context builders
    _fallback_<feature>()       ← 7 deterministic fallback explanations
    _PROMPT_<FEATURE>           ← 7 Gemini system prompts
    _call_gemini_graph()        ← single urllib dispatcher (routes by feature)
    render_ai_decoder()         ← unified UI renderer (Layer 1 + 2 + 3)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings
from urllib import error as urlerror
from urllib import request as urlrequest

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st

from app.data_engine import (
    data_engine_status,
    get_global_start_date,
    load_ticker_data,
    render_data_engine_controls,
    render_multi_ticker_input,
    render_single_ticker_input,
)
from core.graph_features import (
    DEFAULT_GRAPH_BENCHMARK,
    GRAPH_FEATURES,
    build_graph_feature_payload,
)

try:
    from utils.theme import qe_faq_section
except ImportError:
    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        st.markdown("---")
        st.markdown(f"### {title}")
        for question, answer in faqs:
            with st.expander(question):
                st.write(answer)

try:
    from utils.config import cfg
except ImportError:
    cfg = type("cfg", (), {"GEMINI_API_KEY": "", "GEMINI_MODEL": "gemini-1.5-flash"})()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — shared across all graph renderers (original code, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _records_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()

def _format_metric(label: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, str):
        return value
    lower = label.lower()
    if "price" in lower or "close" in lower:
        return f"${float(value):,.2f}"
    if "days" in lower or "count" in lower:
        return f"{int(round(float(value))):,}"
    if "beta" in lower:
        return f"{float(value):.2f}"
    if any(t in lower for t in ["return","rate","gap","move","vol","inside","pos",
                                  "compression","dist","body","loc","avg"]):
        return f"{float(value):.2%}"
    if abs(float(value)) >= 1000:
        return f"{float(value):,.0f}"
    return f"{float(value):.3f}"

def _show_metric_cards(payload: dict, cards: dict) -> None:
    base_cards = {
        "Rows": payload.get("rows"),
        "Last Close": payload.get("last_close"),
        "Avg Volume": payload.get("avg_volume"),
    }
    merged = {**base_cards, **cards}
    cols = st.columns(min(6, max(1, len(merged))))
    for idx, (label, value) in enumerate(merged.items()):
        cols[idx % len(cols)].metric(label, _format_metric(label, value))

def _line_trace(x, y, name: str, color: str, *, dash: str | None = None) -> go.Scatter:
    return go.Scatter(x=x, y=y, mode="lines", name=name,
                       line=dict(color=color, width=2, dash=dash))

def _dark_layout(title: str, **kwargs) -> dict:
    layout = dict(template="plotly_dark", title=title, height=340)
    layout.update(kwargs)
    return layout


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH RENDERERS — original logic, completely unchanged
# ══════════════════════════════════════════════════════════════════════════════

def _render_relative_strength(payload: dict, feature: dict) -> None:
    relative = _records_frame(feature.get("relative_series", []))
    rolling  = _records_frame(feature.get("rolling_stats", []))
    if relative.empty:
        st.warning("Not enough aligned history to compute relative strength.")
        return
    benchmark = feature.get("benchmark", payload.get("benchmark", DEFAULT_GRAPH_BENCHMARK))
    fig = go.Figure()
    fig.add_trace(_line_trace(relative["Date"], relative[payload["ticker"]], payload["ticker"], "#00f5ff"))
    fig.add_trace(_line_trace(relative["Date"], relative[benchmark], benchmark, "#ffd700"))
    fig.update_layout(_dark_layout(f"{payload['ticker']} vs {benchmark}", yaxis_title="Normalised = 100"))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(_line_trace(relative["Date"], relative["RelativeRatio"], "Relative Ratio", "#00ff88"))
    if not rolling.empty:
        fig2.add_trace(_line_trace(rolling["Date"], rolling["Outperformance21d"], "21d Outperformance", "#ff8c00"))
        fig2.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Beta63d"], mode="lines",
            name="63d Beta", yaxis="y2", line=dict(color="#b44fff", width=2)))
    fig2.update_layout(_dark_layout("Relative Ratio - 21d Outperformance - 63d Beta",
        yaxis=dict(title="Relative / Outperformance"),
        yaxis2=dict(title="Beta", overlaying="y", side="right")))
    st.plotly_chart(fig2, use_container_width=True)

def _render_volume_profile(payload: dict, feature: dict) -> None:
    profile       = _records_frame(feature.get("profile", []))
    price_context = _records_frame(feature.get("price_context", []))
    if profile.empty or price_context.empty:
        st.warning("Not enough history to compute a volume profile.")
        return
    st.caption(feature.get("note", ""))
    cards = feature.get("cards", {})
    poc             = cards.get("POC Price")
    value_area_low  = cards.get("Value Area Low")
    value_area_high = cards.get("Value Area High")
    fig = go.Figure(go.Bar(x=profile["Volume"], y=profile["PriceMid"], orientation="h",
        marker_color="rgba(0,245,255,0.7)", name="Volume",
        hovertemplate="Price %{y:.2f}<br>Volume %{x:,.0f}<extra></extra>"))
    fig.update_layout(_dark_layout("Volume by Price", xaxis_title="Volume",
                                    yaxis_title="Typical Price", height=380))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(_line_trace(price_context["Date"], price_context["Close"],
                                f"{payload['ticker']} Close", "white"))
    if poc is not None:
        fig2.add_trace(_line_trace(price_context["Date"], [poc]*len(price_context), "POC", "#ffd700", dash="dash"))
    if value_area_low is not None:
        fig2.add_trace(_line_trace(price_context["Date"], [value_area_low]*len(price_context),
                                    "Value Area Low", "#00ff88", dash="dot"))
    if value_area_high is not None:
        fig2.add_trace(_line_trace(price_context["Date"], [value_area_high]*len(price_context),
                                    "Value Area High", "#ff8c00", dash="dot"))
    fig2.update_layout(_dark_layout("Close vs Value Area", xaxis_title="Date", yaxis_title="Price"))
    st.plotly_chart(fig2, use_container_width=True)

def _render_gap_session(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    if frame.empty:
        st.warning("Not enough history to compute gap/session decomposition.")
        return
    fig = go.Figure()
    fig.add_bar(x=frame["Date"], y=frame["GapReturn"], name="Gap Return",
                 marker_color="rgba(0,245,255,0.65)")
    fig.add_bar(x=frame["Date"], y=frame["SessionReturn"], name="Intraday Return",
                 marker_color="rgba(255,215,0,0.65)")
    fig.update_layout(_dark_layout("Gap vs Intraday Return", barmode="group", yaxis_tickformat=".1%"))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=frame["GapReturn"], y=frame["SessionReturn"], mode="markers",
        text=frame["Date"],
        marker=dict(size=8, color=frame["Filled"].map(lambda v: "#ff8c00" if v else "#00ff88"),
                     symbol=frame["SameDirection"].map(lambda v: "circle" if v else "diamond"),
                     opacity=0.75),
        hovertemplate="%{text}<br>Gap %{x:.2%}<br>Session %{y:.2%}<extra></extra>",
        name="Gap Days"))
    fig2.update_layout(_dark_layout("Gap Continuation Map",
        xaxis_title="Gap Return", yaxis_title="Intraday Return"))
    fig2.update_xaxes(tickformat=".1%")
    fig2.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)

def _render_seasonality(feature: dict) -> None:
    weekday = _records_frame(feature.get("weekday", []))
    monthly = _records_frame(feature.get("monthly_heatmap", []))
    if weekday.empty or monthly.empty:
        st.warning("Not enough history to compute seasonality.")
        return
    fig = go.Figure()
    fig.add_bar(x=weekday["Weekday"], y=weekday["AvgReturn"], name="Avg Return",
                 marker_color=["#00ff88" if v >= 0 else "#ff3366" for v in weekday["AvgReturn"]])
    fig.add_trace(go.Scatter(x=weekday["Weekday"], y=weekday["HitRate"], mode="lines+markers",
        name="Hit Rate", yaxis="y2", line=dict(color="#ffd700", width=2)))
    fig.update_layout(_dark_layout("Weekday Return & Hit Rate",
        yaxis=dict(title="Avg Return", tickformat=".1%"),
        yaxis2=dict(title="Hit Rate", overlaying="y", side="right", tickformat=".0%")))
    st.plotly_chart(fig, use_container_width=True)
    pivot = monthly.pivot(index="Year", columns="Month", values="Return").reindex(columns=month_order)
    fig2 = go.Figure(data=[go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
        y=pivot.index.tolist(), colorscale="RdYlGn", zmid=0,
        hovertemplate="Year %{y}<br>%{x}: %{z:.2%}<extra></extra>")])
    fig2.update_layout(_dark_layout("Monthly Return Heatmap", xaxis_title="Month",
                                     yaxis_title="Year", height=360))
    st.plotly_chart(fig2, use_container_width=True)

def _render_volume_shock(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    if frame.empty:
        st.warning("Not enough history to compute volume shock analytics.")
        return
    fig = go.Figure()
    fig.add_trace(_line_trace(frame["Date"], frame["VolumeZ"], "Volume Z-Score", "#00f5ff"))
    fig.add_hline(y=2, line_dash="dash", line_color="#ff3366", annotation_text="Shock Threshold")
    fig.update_layout(_dark_layout("Volume Z-Score Over Time", yaxis_title="Z-Score"))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=frame["VolumeZ"], y=frame["Next5dReturn"], mode="markers",
        text=frame["Date"],
        marker=dict(
            size=frame["Next5dVol"].fillna(0).map(lambda v: max(6, min(16, v * 250 + 4))),
            color=frame["ShockFlag"].map(lambda v: "#ffd700" if v else "#00ff88"),
            opacity=0.72),
        hovertemplate="%{text}<br>Vol Z %{x:.2f}<br>Next 5d %{y:.2%}<extra></extra>",
        name="Forward 5d Return"))
    fig2.update_layout(_dark_layout("Shock vs Forward Outcome",
        xaxis_title="Volume Z-Score", yaxis_title="Next 5d Return"))
    fig2.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)

def _render_breakout_context(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    if frame.empty:
        st.warning("Not enough history to compute breakout context.")
        return
    fig = go.Figure()
    fig.add_trace(_line_trace(frame["Date"], frame["RangePos252d"], "252d Range Position", "#00ff88"))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["Dist63dHigh"], mode="lines",
        name="Dist to 63d High", yaxis="y2", line=dict(color="#ffd700", width=2)))
    fig.update_layout(_dark_layout("Range Position & Breakout Distance",
        yaxis=dict(title="Range Position", tickformat=".0%"),
        yaxis2=dict(title="Distance to 63d High", overlaying="y", side="right", tickformat=".1%")))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(_line_trace(frame["Date"], frame["Compression20d"], "20d Compression", "#b44fff"))
    fig2.add_trace(go.Scatter(x=frame["Date"], y=frame["Dist20dHigh"], mode="lines",
        name="Dist to 20d High", yaxis="y2", line=dict(color="#00f5ff", width=2)))
    fig2.update_layout(_dark_layout("20d Compression & Short-Term Breakout Distance",
        yaxis=dict(title="Compression", tickformat=".1%"),
        yaxis2=dict(title="Distance to 20d High", overlaying="y", side="right", tickformat=".1%")))
    st.plotly_chart(fig2, use_container_width=True)

def _render_candle_structure(feature: dict) -> None:
    frame         = _records_frame(feature.get("series", []))
    close_buckets = _records_frame(feature.get("close_location_buckets", []))
    wick_buckets  = _records_frame(feature.get("wick_buckets", []))
    if frame.empty:
        st.warning("Not enough history to compute candle structure analytics.")
        return
    recent = frame.tail(180)
    fig = go.Figure()
    fig.add_trace(_line_trace(recent["Date"], recent["BodyPct"],      "Body % of Range", "#00f5ff"))
    fig.add_trace(_line_trace(recent["Date"], recent["UpperWickPct"], "Upper Wick %",    "#ff3366"))
    fig.add_trace(_line_trace(recent["Date"], recent["LowerWickPct"], "Lower Wick %",    "#00ff88"))
    fig.add_trace(_line_trace(recent["Date"], recent["CloseLocation"],"Close Location",  "#ffd700"))
    fig.update_layout(_dark_layout("Daily Candle Anatomy", yaxis_tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    if not close_buckets.empty:
        fig2.add_bar(x=close_buckets["Bucket"], y=close_buckets["AvgNext1d"],
                      name="Close Location Bucket", marker_color="rgba(0,245,255,0.7)")
    if not wick_buckets.empty:
        fig2.add_bar(x=wick_buckets["Bucket"], y=wick_buckets["AvgNext1d"],
                      name="Wick Imbalance Bucket", marker_color="rgba(255,215,0,0.7)")
    fig2.update_layout(_dark_layout("Next-Day Return by Structure Bucket",
                                     barmode="group", yaxis_tickformat=".2%"))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GRAPH-SPECIFIC SYSTEM PROMPTS
# One prompt per feature. Each is threshold-aware for its own metrics.
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_RELATIVE_STRENGTH = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Relative Strength graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- Relative ratio > 1: ticker outperforming benchmark (good for long thesis)
- Relative ratio < 1: ticker underperforming (momentum headwind)
- Relative ratio trending up for 21+ days: sustained outperformance
- 21d outperformance > 0.02 (2%): meaningful edge over benchmark
- 21d outperformance < -0.02: meaningful lag — reassess conviction
- Beta > 1.5: high amplification — moves sharply with/against market
- Beta < 0.5: defensive — low correlation to market
- Beta negative: inverse relationship to benchmark — unusual, flag it
- Normalised price diverging > 20% from benchmark: structural leadership or breakdown

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_VOLUME_PROFILE = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Volume Profile graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- POC (Point of Control): price level with the most traded volume — acts as a magnet or support/resistance
- Value Area: the band where 70% of volume traded — price inside = fair value, outside = extension
- Price above Value Area High: extended above fair value — watch for mean reversion or continuation
- Price below Value Area Low: distressed selling or breakdown below fair value
- Price near POC: highest liquidity zone — price often consolidates or reverses here
- Wide value area (>15% price range): high disagreement on fair value — volatile market
- Narrow value area (<5% price range): consensus on price — low volatility regime, often precedes breakout
- Low volume nodes between price and POC: fast-move zones — price can travel through quickly

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_GAP_SESSION = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Gap & Session Decomposition graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- Gap return: overnight move from prior close to today's open — reflects after-hours/pre-market conviction
- Session return: intraday move from open to close — reflects real-time price discovery
- Gap fill rate > 70%: gaps tend to reverse during the session — fading gaps has been historically profitable
- Gap fill rate < 40%: gaps tend to hold — momentum continuation after gaps is the dominant pattern
- Same direction rate > 60%: gap and session returns usually move the same way — trend-following structure
- Same direction rate < 40%: gap and session fight each other — mean-reversion intraday structure
- Large gap (>1.5%) with gap fill: strong signal that opening volatility overshot fair value
- Large gap (>1.5%) without fill: institutional conviction at the open — continuation likely
- Avg gap return and avg session return both positive: persistent bullish structure

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_SEASONALITY = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Seasonality graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- Hit rate > 60% on a weekday: the stock goes up on that day more than 60% of the time — statistically meaningful
- Hit rate < 40%: the stock goes down on that day more than 60% of the time — statistically meaningful
- Avg return > 0.2% on a weekday: economically meaningful daily edge
- Monthly heatmap dark green: strong consistent positive returns in that month across years
- Monthly heatmap dark red: strong consistent negative returns in that month across years
- One or two dominant months (e.g., Jan effect, year-end): seasonal concentration risk
- Inconsistent monthly pattern (mixed green/red): no reliable seasonal signal
- Strong Monday or Friday effect: typically linked to short-term trader behaviour
- Avoid treating seasonality as a standalone signal — use to confirm other views

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_VOLUME_SHOCK = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Volume Shock graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- Volume Z-score > 2: unusual volume — more than 2 standard deviations above average — called a "shock"
- Volume Z-score > 3: extreme volume event — typically marks capitulation, breakout, or news-driven move
- Volume Z-score < -1: unusually quiet — holiday, pre-event silence, or liquidity vacuum
- Shock with positive next-5d return: demand shock — institutional accumulation signal
- Shock with negative next-5d return: supply shock — distribution or forced selling signal
- Shock with near-zero next-5d return: noise event — volume without follow-through
- High next-5d volatility after shock: shock increased uncertainty, not just direction
- Frequency of shocks > 10% of days: stock is prone to volatility events — not low-risk
- Recent shock (within 5 days): current elevated state — monitor for follow-through

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_BREAKOUT_CONTEXT = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Breakout Context graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- 252d range position > 80%: stock is near its 1-year high — in breakout territory
- 252d range position < 20%: stock is near its 1-year low — in breakdown territory
- 252d range position 40-60%: mid-range — no directional edge from range position alone
- Distance to 63d high < 2%: very close to 3-month high — potential breakout imminent
- Distance to 63d high > 15%: far from recent high — needs significant recovery before breakout
- 20d compression < 5%: price coiling tightly — energy building for a directional move
- 20d compression > 15%: wide daily range — volatile regime, breakout signals less reliable
- Compression + near high: classic breakout setup — institutional traders watch this pattern
- Compression + near low: potential breakdown or base-building — direction unclear
- Distance to 20d high declining over time: price tightening toward resistance — bullish coil

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

_PROMPT_CANDLE_STRUCTURE = """You are a senior quantitative analyst inside a professional market research dashboard.
Explain the Candle Structure graph output to a NON-TECHNICAL user (allocator, PM, trader).

RULES:
1. Use ONLY numbers from the JSON context. Never invent figures.
2. If danger flags exist, address them FIRST and prominently.
3. Explain every metric in one plain English sentence.
4. No jargon, no LaTeX, no formulas. Short paragraphs only.

THRESHOLD KNOWLEDGE:
- Body % > 70%: large decisive candles — strong directional conviction in recent trading
- Body % < 30%: small bodies with large wicks — indecision, high rejection of extremes
- Upper wick % > 40%: sellers consistently pushing price back from highs — supply overhead
- Lower wick % > 40%: buyers consistently pushing price back from lows — demand below
- Close location > 0.7 (top 30%): closing near day highs — bullish intraday structure
- Close location < 0.3 (bottom 30%): closing near day lows — bearish intraday structure
- Close location near 0.5: balanced — neither bulls nor bears dominating close
- High close location buckets with positive next-day return: closing high predicts follow-through
- High close location buckets with negative next-day return: closing high is a sell-the-close pattern
- Wick imbalance (upper >> lower): distribution pattern — price repeatedly rejected at highs
- Wick imbalance (lower >> upper): accumulation pattern — price repeatedly supported at lows

OUTPUT FORMAT — exactly 4 sections:
### What the output says
### What each number means
### Red flags
### Plain English conclusion

End with: ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""

# Prompt dispatch map
_PROMPT_MAP = {
    "relative_strength": _PROMPT_RELATIVE_STRENGTH,
    "volume_profile":    _PROMPT_VOLUME_PROFILE,
    "gap_session":       _PROMPT_GAP_SESSION,
    "seasonality":       _PROMPT_SEASONALITY,
    "volume_shock":      _PROMPT_VOLUME_SHOCK,
    "breakout_context":  _PROMPT_BREAKOUT_CONTEXT,
    "candle_structure":  _PROMPT_CANDLE_STRUCTURE,
}


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GRAPH-SPECIFIC DANGER FLAGS
# Each function checks only the metrics relevant to that graph type.
# ══════════════════════════════════════════════════════════════════════════════

def _flags_relative_strength(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    rolling = _records_frame(feature.get("rolling_stats", []))
    relative = _records_frame(feature.get("relative_series", []))
    cards = feature.get("cards", {})

    if relative.empty:
        flags.append({"severity": "DANGER", "code": "NO_RELATIVE_DATA",
            "message": f"Insufficient aligned history to compute relative strength for {ticker}. "
                       "Results are unreliable — extend the date range."})
        return flags

    # Beta check
    beta = cards.get("Beta 63d")
    if beta is not None:
        try:
            b = float(beta)
            if b > 1.8:
                flags.append({"severity": "WARNING", "code": "HIGH_BETA",
                    "message": f"{ticker} 63d Beta = {b:.2f} — very high amplification. "
                               "The stock moves ~{b:.1f}x the benchmark. "
                               "Drawdowns will be proportionally larger than the index in a selloff."})
            elif b < 0:
                flags.append({"severity": "WARNING", "code": "NEGATIVE_BETA",
                    "message": f"{ticker} 63d Beta = {b:.2f} — negative. "
                               "The stock moves inversely to the benchmark. "
                               "Verify this is structural (e.g. inverse ETF) and not a data artefact."})
        except (TypeError, ValueError):
            pass

    # Sustained underperformance
    outperf = cards.get("21d Outperformance")
    if outperf is not None:
        try:
            op = float(str(outperf).replace("%", "")) / 100 if "%" in str(outperf) else float(outperf)
            if op < -0.05:
                flags.append({"severity": "WARNING", "code": "SUSTAINED_UNDERPERFORMANCE",
                    "message": f"{ticker} 21d outperformance is {op:.2%} — lagging the benchmark "
                               "by more than 5% over the past 21 trading days. "
                               "This is a significant momentum headwind. Reassess the thesis."})
        except (TypeError, ValueError):
            pass

    # Relative ratio direction
    if "RelativeRatio" in relative.columns and len(relative) >= 21:
        last_ratio = relative["RelativeRatio"].iloc[-1]
        ratio_21d_ago = relative["RelativeRatio"].iloc[-21]
        if last_ratio < ratio_21d_ago * 0.95:
            flags.append({"severity": "INFO", "code": "RELATIVE_RATIO_DECLINING",
                "message": f"{ticker} relative ratio has declined over the past 21 days "
                           f"({ratio_21d_ago:.3f} → {last_ratio:.3f}). "
                           "Relative momentum is shifting toward the benchmark."})
    return flags


def _flags_volume_profile(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    cards = feature.get("cards", {})
    last_close = payload.get("last_close")

    poc = cards.get("POC Price")
    val = cards.get("Value Area Low")
    vah = cards.get("Value Area High")

    if poc is None or val is None or vah is None:
        flags.append({"severity": "INFO", "code": "INCOMPLETE_PROFILE",
            "message": "Volume profile cards are incomplete. "
                       "POC or Value Area boundaries could not be computed — "
                       "likely insufficient trading history. Results may be approximate."})
        return flags

    try:
        poc_f = float(poc); val_f = float(val); vah_f = float(vah)
        close_f = float(last_close) if last_close else None

        va_width_pct = (vah_f - val_f) / poc_f if poc_f > 0 else 0

        if va_width_pct > 0.20:
            flags.append({"severity": "WARNING", "code": "WIDE_VALUE_AREA",
                "message": f"Value Area width is {va_width_pct:.1%} of POC price — very wide. "
                           "This signals high disagreement on fair value. "
                           "Wide value areas are associated with volatile, trending markets "
                           "where mean-reversion strategies underperform."})
        elif va_width_pct < 0.04:
            flags.append({"severity": "INFO", "code": "NARROW_VALUE_AREA",
                "message": f"Value Area width is only {va_width_pct:.1%} of POC price — very tight. "
                           "Price consensus is high. Narrow value areas often precede "
                           "directional breakouts when new information arrives."})

        if close_f is not None:
            if close_f > vah_f:
                flags.append({"severity": "INFO", "code": "PRICE_ABOVE_VALUE_AREA",
                    "message": f"Current close (${close_f:.2f}) is above the Value Area High (${vah_f:.2f}). "
                               "Price is trading above where 70% of volume was done. "
                               "This can signal bullish continuation or overextension depending on volume."})
            elif close_f < val_f:
                flags.append({"severity": "WARNING", "code": "PRICE_BELOW_VALUE_AREA",
                    "message": f"Current close (${close_f:.2f}) is below the Value Area Low (${val_f:.2f}). "
                               "Price has broken below the fair-value consensus zone. "
                               "This is bearish unless buying volume is expanding at these levels."})
    except (TypeError, ValueError):
        pass

    return flags


def _flags_gap_session(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    cards = feature.get("cards", {})

    fill_rate = cards.get("Gap Fill Rate") or cards.get("Fill Rate")
    same_dir  = cards.get("Same Direction Rate") or cards.get("Continuation Rate")
    avg_gap   = cards.get("Avg Gap Return") or cards.get("Avg Gap")

    try:
        if fill_rate is not None:
            fr = float(str(fill_rate).replace("%",""))/100 if "%" in str(fill_rate) else float(fill_rate)
            if fr > 0.75:
                flags.append({"severity": "INFO", "code": "HIGH_GAP_FILL_RATE",
                    "message": f"{ticker} fills {fr:.0%} of gaps intraday. "
                               "Fading the open (buying gap-down opens, selling gap-up opens) "
                               "has historically been a profitable intraday structure for this stock."})
            elif fr < 0.35:
                flags.append({"severity": "INFO", "code": "LOW_GAP_FILL_RATE",
                    "message": f"{ticker} fills only {fr:.0%} of gaps. "
                               "The opening gap direction tends to hold — momentum continuation "
                               "strategies at the open have been more effective than mean-reversion."})
    except (TypeError, ValueError):
        pass

    try:
        if same_dir is not None:
            sd = float(str(same_dir).replace("%",""))/100 if "%" in str(same_dir) else float(same_dir)
            if sd < 0.35:
                flags.append({"severity": "WARNING", "code": "GAP_SESSION_CONFLICT",
                    "message": f"{ticker} gap and session returns move in the same direction only {sd:.0%} of the time. "
                               "The stock frequently gaps up but closes down (or vice versa). "
                               "Opening prints are not reliable indicators of day direction."})
    except (TypeError, ValueError):
        pass

    return flags


def _flags_seasonality(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    weekday = _records_frame(feature.get("weekday", []))
    monthly = _records_frame(feature.get("monthly_heatmap", []))

    if weekday.empty:
        flags.append({"severity": "INFO", "code": "INSUFFICIENT_SEASONALITY_DATA",
            "message": "Not enough history for reliable weekday seasonality. "
                       "At least 2 years of data is recommended for statistical significance."})
        return flags

    # Check for dominant weekday effect
    if "HitRate" in weekday.columns and "AvgReturn" in weekday.columns:
        max_hit    = weekday["HitRate"].max()
        min_hit    = weekday["HitRate"].min()
        best_day   = weekday.loc[weekday["HitRate"].idxmax(), "Weekday"]
        worst_day  = weekday.loc[weekday["HitRate"].idxmin(), "Weekday"]
        dispersion = max_hit - min_hit

        if dispersion > 0.30:
            flags.append({"severity": "INFO", "code": "STRONG_WEEKDAY_PATTERN",
                "message": f"Large weekday hit-rate spread: {best_day} ({max_hit:.0%}) vs "
                           f"{worst_day} ({min_hit:.0%}). "
                           "This is a strong day-of-week pattern. "
                           "Verify it is not an artefact of earnings or macro release timing."})

    # Check for dominant monthly effect
    if not monthly.empty and "Return" in monthly.columns and "Month" in monthly.columns:
        monthly_avg = monthly.groupby("Month")["Return"].mean()
        if not monthly_avg.empty:
            best_month  = monthly_avg.idxmax()
            worst_month = monthly_avg.idxmin()
            best_ret    = monthly_avg.max()
            worst_ret   = monthly_avg.min()
            if best_ret > 0.04:
                flags.append({"severity": "INFO", "code": "STRONG_MONTHLY_EFFECT",
                    "message": f"{ticker} has a strong positive seasonal: {best_month} averages "
                               f"{best_ret:.1%}. This may reflect recurring events (earnings, "
                               "dividends, index rebalancing). Verify the sample size before trading it."})
            if worst_ret < -0.04:
                flags.append({"severity": "WARNING", "code": "NEGATIVE_SEASONAL",
                    "message": f"{ticker} has historically weak performance in {worst_month} "
                               f"(avg {worst_ret:.1%}). "
                               "Consider reducing exposure during this window if other signals confirm weakness."})

    return flags


def _flags_volume_shock(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    frame = _records_frame(feature.get("series", []))
    cards = feature.get("cards", {})

    if frame.empty:
        flags.append({"severity": "DANGER", "code": "NO_SHOCK_DATA",
            "message": "No volume shock data available. Extend the date range."})
        return flags

    # Recent shock check
    if "ShockFlag" in frame.columns and "Date" in frame.columns:
        recent_shocks = frame.tail(5)
        n_recent = int(recent_shocks["ShockFlag"].sum()) if "ShockFlag" in recent_shocks.columns else 0
        if n_recent >= 2:
            flags.append({"severity": "WARNING", "code": "RECENT_VOLUME_SHOCK",
                "message": f"{ticker} has experienced {n_recent} volume shocks in the last 5 trading days. "
                           "This is an elevated activity period. "
                           "Wait for volume normalisation before assuming the move is complete."})
        elif n_recent == 1:
            flags.append({"severity": "INFO", "code": "SINGLE_RECENT_SHOCK",
                "message": f"{ticker} had a volume shock in the last 5 trading days. "
                           "Monitor the 5-day forward return to classify it as demand or supply."})

    # Shock frequency
    if "ShockFlag" in frame.columns and len(frame) > 20:
        shock_freq = frame["ShockFlag"].mean()
        if shock_freq > 0.12:
            flags.append({"severity": "WARNING", "code": "HIGH_SHOCK_FREQUENCY",
                "message": f"Volume shocks occur on {shock_freq:.0%} of days — very frequently. "
                           f"This stock is prone to sudden volume events. "
                           "Position sizing should account for this elevated volatility regime."})

    # Avg next-5d return after shocks
    if "ShockFlag" in frame.columns and "Next5dReturn" in frame.columns:
        shock_rows = frame[frame["ShockFlag"] == True]
        if len(shock_rows) >= 5:
            avg_post_shock = shock_rows["Next5dReturn"].mean()
            if avg_post_shock < -0.02:
                flags.append({"severity": "WARNING", "code": "NEGATIVE_POST_SHOCK_DRIFT",
                    "message": f"On average, {ticker} returns {avg_post_shock:.2%} in the 5 days after a "
                               "volume shock — negative. Volume spikes have historically preceded "
                               "short-term price weakness. This is a supply-shock-dominant pattern."})
            elif avg_post_shock > 0.02:
                flags.append({"severity": "INFO", "code": "POSITIVE_POST_SHOCK_DRIFT",
                    "message": f"On average, {ticker} returns {avg_post_shock:.2%} in the 5 days after a "
                               "volume shock — positive. Volume spikes have historically been "
                               "demand-driven, with follow-through price strength."})
    return flags


def _flags_breakout_context(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    frame = _records_frame(feature.get("series", []))
    cards = feature.get("cards", {})

    if frame.empty:
        flags.append({"severity": "DANGER", "code": "NO_BREAKOUT_DATA",
            "message": "No breakout context data available. Extend the date range."})
        return flags

    # Current range position
    range_pos = cards.get("252d Range Position") or cards.get("Range Pos 252d")
    compression = cards.get("20d Compression") or cards.get("Compression 20d")
    dist_63h = cards.get("Dist to 63d High") or cards.get("Dist 63d High")

    try:
        if range_pos is not None:
            rp = float(str(range_pos).replace("%",""))/100 if "%" in str(range_pos) else float(range_pos)
            if rp > 0.90:
                flags.append({"severity": "INFO", "code": "AT_1Y_HIGH",
                    "message": f"{ticker} is at {rp:.0%} of its 1-year range — near annual highs. "
                               "Breakout conditions are strong. "
                               "Risk is a failed breakout if volume does not confirm."})
            elif rp < 0.10:
                flags.append({"severity": "WARNING", "code": "AT_1Y_LOW",
                    "message": f"{ticker} is at only {rp:.0%} of its 1-year range — near annual lows. "
                               "No breakout conditions present. "
                               "The stock is in a breakdown or base-building phase."})
    except (TypeError, ValueError):
        pass

    try:
        if compression is not None:
            comp = float(str(compression).replace("%",""))/100 if "%" in str(compression) else float(compression)
            if comp < 0.04:
                flags.append({"severity": "INFO", "code": "TIGHT_COMPRESSION",
                    "message": f"{ticker} 20d compression is {comp:.1%} — very tight price coil. "
                               "This is a classic pre-breakout setup. "
                               "The direction of the eventual break is not determined by compression alone — "
                               "use range position and volume to confirm direction."})
            elif comp > 0.20:
                flags.append({"severity": "WARNING", "code": "HIGH_COMPRESSION_VOLATILITY",
                    "message": f"{ticker} 20d compression is {comp:.1%} — very wide daily ranges. "
                               "Breakout signals are less reliable in high-volatility regimes. "
                               "Wait for compression before acting on breakout setups."})
    except (TypeError, ValueError):
        pass

    try:
        if dist_63h is not None:
            d = float(str(dist_63h).replace("%",""))/100 if "%" in str(dist_63h) else float(dist_63h)
            if d < 0.01:
                flags.append({"severity": "INFO", "code": "NEAR_63D_HIGH",
                    "message": f"{ticker} is within {d:.1%} of its 63-day high — potential breakout imminent. "
                               "Watch for a volume-confirmed close above this level."})
    except (TypeError, ValueError):
        pass

    return flags


def _flags_candle_structure(ticker: str, feature: dict, payload: dict) -> list[dict]:
    flags = []
    frame         = _records_frame(feature.get("series", []))
    close_buckets = _records_frame(feature.get("close_location_buckets", []))
    cards         = feature.get("cards", {})

    if frame.empty:
        flags.append({"severity": "DANGER", "code": "NO_CANDLE_DATA",
            "message": "No candle structure data available. Extend the date range."})
        return flags

    # Recent close location trend
    if "CloseLocation" in frame.columns and len(frame) >= 10:
        recent_cl = frame["CloseLocation"].tail(10).mean()
        if recent_cl < 0.3:
            flags.append({"severity": "WARNING", "code": "BEARISH_CLOSE_STRUCTURE",
                "message": f"{ticker} has been closing in the bottom 30% of its daily range "
                           f"on average over the past 10 days (avg close location = {recent_cl:.2f}). "
                           "Bearish intraday structure — sellers dominating the close."})
        elif recent_cl > 0.7:
            flags.append({"severity": "INFO", "code": "BULLISH_CLOSE_STRUCTURE",
                "message": f"{ticker} has been closing in the top 30% of its daily range "
                           f"on average over the past 10 days (avg close location = {recent_cl:.2f}). "
                           "Bullish intraday structure — buyers dominating the close."})

    # Upper wick dominance (supply)
    if "UpperWickPct" in frame.columns and "LowerWickPct" in frame.columns:
        recent = frame.tail(20)
        avg_upper = recent["UpperWickPct"].mean()
        avg_lower = recent["LowerWickPct"].mean()
        if avg_upper > avg_lower * 1.8:
            flags.append({"severity": "WARNING", "code": "UPPER_WICK_DOMINANCE",
                "message": f"Over the past 20 days, {ticker}'s upper wicks ({avg_upper:.1%}) are "
                           f"significantly larger than lower wicks ({avg_lower:.1%}). "
                           "Supply is overhanging — sellers are consistently pushing price back from highs. "
                           "This is a distribution pattern."})
        elif avg_lower > avg_upper * 1.8:
            flags.append({"severity": "INFO", "code": "LOWER_WICK_DOMINANCE",
                "message": f"Over the past 20 days, {ticker}'s lower wicks ({avg_lower:.1%}) are "
                           f"significantly larger than upper wicks ({avg_upper:.1%}). "
                           "Demand is absorbing selling — buyers consistently support price at lows. "
                           "This is an accumulation pattern."})

    # Body size check
    if "BodyPct" in frame.columns:
        avg_body = frame.tail(20)["BodyPct"].mean()
        if avg_body < 0.25:
            flags.append({"severity": "INFO", "code": "SMALL_BODY_INDECISION",
                "message": f"{ticker} average body size over the past 20 days is {avg_body:.0%} of range — small. "
                           "Indecisive candles dominate. "
                           "Neither bulls nor bears are in control at the close. "
                           "Wait for larger-bodied candles before committing to a direction."})

    return flags


# Flag dispatch map
_FLAG_DISPATCH = {
    "relative_strength": _flags_relative_strength,
    "volume_profile":    _flags_volume_profile,
    "gap_session":       _flags_gap_session,
    "seasonality":       _flags_seasonality,
    "volume_shock":      _flags_volume_shock,
    "breakout_context":  _flags_breakout_context,
    "candle_structure":  _flags_candle_structure,
}


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GRAPH-SPECIFIC CONTEXT BUILDERS
# Each builds only the relevant data for its graph type.
# ══════════════════════════════════════════════════════════════════════════════

def _build_context(
    feature_key: str,
    ticker: str,
    benchmark: str,
    payload: dict,
    feature: dict,
    danger_flags: list[dict],
    data_rows: int,
    last_close,
) -> dict:
    """Master context builder — routes to the correct feature context."""
    base = {
        "feature_type":    feature_key,
        "ticker":          ticker,
        "benchmark":       benchmark,
        "data_rows":       data_rows,
        "last_close":      float(last_close) if last_close else None,
        "danger_flags":    danger_flags,
        "danger_flag_count":  len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in danger_flags if f["severity"] == "WARNING"]),
        "reference_thresholds": _THRESHOLDS.get(feature_key, {}),
    }

    # Attach cards (always useful regardless of feature type)
    base["metric_cards"] = feature.get("cards", {})

    # Feature-specific data snippets (keep small — Gemini has 900 token budget)
    if feature_key == "relative_strength":
        relative = _records_frame(feature.get("relative_series", []))
        rolling  = _records_frame(feature.get("rolling_stats", []))
        base["last_relative_ratio"] = round(float(relative["RelativeRatio"].iloc[-1]), 4) \
            if not relative.empty and "RelativeRatio" in relative.columns else None
        base["relative_ratio_21d_ago"] = round(float(relative["RelativeRatio"].iloc[-21]), 4) \
            if not relative.empty and len(relative) >= 21 and "RelativeRatio" in relative.columns else None
        base["last_21d_outperformance"] = round(float(rolling["Outperformance21d"].iloc[-1]), 4) \
            if not rolling.empty and "Outperformance21d" in rolling.columns else None
        base["last_beta_63d"] = round(float(rolling["Beta63d"].iloc[-1]), 4) \
            if not rolling.empty and "Beta63d" in rolling.columns else None

    elif feature_key == "volume_profile":
        base["poc_price"]       = feature.get("cards", {}).get("POC Price")
        base["value_area_low"]  = feature.get("cards", {}).get("Value Area Low")
        base["value_area_high"] = feature.get("cards", {}).get("Value Area High")
        base["value_area_volume_pct"] = feature.get("cards", {}).get("Value Area Vol %")

    elif feature_key == "gap_session":
        frame = _records_frame(feature.get("series", []))
        if not frame.empty:
            base["avg_gap_return"]     = round(float(frame["GapReturn"].mean()), 4)
            base["avg_session_return"] = round(float(frame["SessionReturn"].mean()), 4)
            base["gap_fill_rate"]      = round(float(frame["Filled"].mean()), 4) \
                if "Filled" in frame.columns else None
            base["same_direction_rate"] = round(float(frame["SameDirection"].mean()), 4) \
                if "SameDirection" in frame.columns else None
            base["n_gap_days"]         = len(frame)

    elif feature_key == "seasonality":
        weekday = _records_frame(feature.get("weekday", []))
        monthly = _records_frame(feature.get("monthly_heatmap", []))
        if not weekday.empty and "HitRate" in weekday.columns:
            best_idx  = weekday["HitRate"].idxmax()
            worst_idx = weekday["HitRate"].idxmin()
            base["best_weekday"]  = {"day": weekday.loc[best_idx, "Weekday"],
                                      "hit_rate": round(float(weekday.loc[best_idx, "HitRate"]), 4),
                                      "avg_return": round(float(weekday.loc[best_idx, "AvgReturn"]), 4)}
            base["worst_weekday"] = {"day": weekday.loc[worst_idx, "Weekday"],
                                      "hit_rate": round(float(weekday.loc[worst_idx, "HitRate"]), 4),
                                      "avg_return": round(float(weekday.loc[worst_idx, "AvgReturn"]), 4)}
        if not monthly.empty and "Return" in monthly.columns and "Month" in monthly.columns:
            monthly_avg = monthly.groupby("Month")["Return"].mean().round(4)
            base["monthly_avg_returns"] = monthly_avg.to_dict()

    elif feature_key == "volume_shock":
        frame = _records_frame(feature.get("series", []))
        if not frame.empty:
            base["shock_frequency"]      = round(float(frame["ShockFlag"].mean()), 4) \
                if "ShockFlag" in frame.columns else None
            base["max_volume_z"]         = round(float(frame["VolumeZ"].max()), 2) \
                if "VolumeZ" in frame.columns else None
            base["recent_5d_shock_count"] = int(frame.tail(5)["ShockFlag"].sum()) \
                if "ShockFlag" in frame.columns else 0
            shock_rows = frame[frame["ShockFlag"] == True] if "ShockFlag" in frame.columns else pd.DataFrame()
            base["avg_post_shock_5d_return"] = round(float(shock_rows["Next5dReturn"].mean()), 4) \
                if len(shock_rows) >= 3 and "Next5dReturn" in shock_rows.columns else None

    elif feature_key == "breakout_context":
        frame = _records_frame(feature.get("series", []))
        if not frame.empty:
            last = frame.iloc[-1]
            base["current_range_pos_252d"] = round(float(last.get("RangePos252d", 0)), 4) \
                if "RangePos252d" in last else None
            base["current_dist_63d_high"]  = round(float(last.get("Dist63dHigh", 0)), 4) \
                if "Dist63dHigh" in last else None
            base["current_compression_20d"] = round(float(last.get("Compression20d", 0)), 4) \
                if "Compression20d" in last else None
            base["current_dist_20d_high"]  = round(float(last.get("Dist20dHigh", 0)), 4) \
                if "Dist20dHigh" in last else None

    elif feature_key == "candle_structure":
        frame = _records_frame(feature.get("series", []))
        if not frame.empty:
            recent = frame.tail(20)
            base["avg_body_pct_20d"]        = round(float(recent["BodyPct"].mean()), 4) \
                if "BodyPct" in recent.columns else None
            base["avg_upper_wick_20d"]      = round(float(recent["UpperWickPct"].mean()), 4) \
                if "UpperWickPct" in recent.columns else None
            base["avg_lower_wick_20d"]      = round(float(recent["LowerWickPct"].mean()), 4) \
                if "LowerWickPct" in recent.columns else None
            base["avg_close_location_10d"]  = round(float(frame.tail(10)["CloseLocation"].mean()), 4) \
                if "CloseLocation" in frame.columns else None
        close_b = _records_frame(feature.get("close_location_buckets", []))
        if not close_b.empty and "Bucket" in close_b.columns and "AvgNext1d" in close_b.columns:
            base["close_location_bucket_returns"] = {
                str(row["Bucket"]): round(float(row["AvgNext1d"]), 4)
                for _, row in close_b.iterrows()
            }
    return base


# Reference thresholds per feature (embedded in context for Gemini)
_THRESHOLDS = {
    "relative_strength": {
        "outperformance_strong": 0.02, "outperformance_weak": -0.02,
        "beta_high": 1.5, "beta_defensive": 0.5,
    },
    "volume_profile": {
        "wide_value_area_pct": 0.15, "narrow_value_area_pct": 0.04,
    },
    "gap_session": {
        "high_fill_rate": 0.70, "low_fill_rate": 0.40,
        "same_direction_strong": 0.60, "same_direction_weak": 0.35,
    },
    "seasonality": {
        "strong_hit_rate": 0.60, "weak_hit_rate": 0.40,
        "strong_monthly_return": 0.04, "weak_monthly_return": -0.04,
    },
    "volume_shock": {
        "shock_z_threshold": 2.0, "extreme_shock_z": 3.0,
        "high_shock_frequency": 0.10, "meaningful_post_shock": 0.02,
    },
    "breakout_context": {
        "high_range_pos": 0.80, "low_range_pos": 0.20,
        "tight_compression": 0.05, "wide_compression": 0.15,
        "near_high_dist": 0.02,
    },
    "candle_structure": {
        "strong_body": 0.70, "weak_body": 0.30,
        "bullish_close_location": 0.70, "bearish_close_location": 0.30,
        "dominant_wick_ratio": 1.8,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK EXPLANATIONS (one per graph type)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_explanation(feature_key: str, context: dict) -> str:
    ticker  = context.get("ticker", "N/A")
    flags   = context.get("danger_flags", [])
    cards   = context.get("metric_cards", {})
    thres   = context.get("reference_thresholds", {})

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}" for f in flags
        )

    cards_text = "\n".join(f"- **{k}**: {v}" for k, v in cards.items()) if cards else "No cards available."

    intros = {
        "relative_strength": (
            f"**{ticker}** was compared against the benchmark. "
            f"Key metrics: relative ratio = {context.get('last_relative_ratio', 'N/A')}, "
            f"21d outperformance = {context.get('last_21d_outperformance', 'N/A')}, "
            f"63d beta = {context.get('last_beta_63d', 'N/A')}."
        ),
        "volume_profile": (
            f"**{ticker}** volume profile: POC = {context.get('poc_price', 'N/A')}, "
            f"Value Area Low = {context.get('value_area_low', 'N/A')}, "
            f"Value Area High = {context.get('value_area_high', 'N/A')}. "
            f"Current close = {context.get('last_close', 'N/A')}."
        ),
        "gap_session": (
            f"**{ticker}** gap analysis: avg gap return = {context.get('avg_gap_return', 'N/A'):.2%}, "
            f"avg session return = {context.get('avg_session_return', 'N/A'):.2%}, "
            f"gap fill rate = {context.get('gap_fill_rate', 'N/A'):.0%}, "
            f"same direction rate = {context.get('same_direction_rate', 'N/A'):.0%}."
        ) if isinstance(context.get('avg_gap_return'), float) else f"**{ticker}** gap/session data.",
        "seasonality": (
            f"**{ticker}** seasonality: best weekday = {context.get('best_weekday', {}).get('day', 'N/A')}, "
            f"worst weekday = {context.get('worst_weekday', {}).get('day', 'N/A')}."
        ),
        "volume_shock": (
            f"**{ticker}** volume shock analysis: shock frequency = {context.get('shock_frequency', 'N/A'):.0%}, "
            f"max Z-score = {context.get('max_volume_z', 'N/A')}, "
            f"avg post-shock 5d return = {context.get('avg_post_shock_5d_return', 'N/A')}."
        ) if isinstance(context.get('shock_frequency'), float) else f"**{ticker}** volume shock data.",
        "breakout_context": (
            f"**{ticker}** breakout context: 252d range pos = {context.get('current_range_pos_252d', 'N/A'):.0%}, "
            f"dist to 63d high = {context.get('current_dist_63d_high', 'N/A'):.1%}, "
            f"20d compression = {context.get('current_compression_20d', 'N/A'):.1%}."
        ) if isinstance(context.get('current_range_pos_252d'), float) else f"**{ticker}** breakout context data.",
        "candle_structure": (
            f"**{ticker}** candle structure: avg body = {context.get('avg_body_pct_20d', 'N/A'):.0%}, "
            f"upper wick = {context.get('avg_upper_wick_20d', 'N/A'):.0%}, "
            f"lower wick = {context.get('avg_lower_wick_20d', 'N/A'):.0%}, "
            f"close location (10d) = {context.get('avg_close_location_10d', 'N/A')}."
        ) if isinstance(context.get('avg_body_pct_20d'), float) else f"**{ticker}** candle structure data.",
    }

    intro = intros.get(feature_key, f"**{ticker}** {feature_key} analysis.")

    return (
        f"### What the output says\n{intro}\n\n"
        f"### What each number means\n{cards_text}\n\n"
        f"### Red flags\n"
        f"{flag_text if flags else 'No critical flags detected.'}\n\n"
        f"### Plain English conclusion\n"
        f"Review the metrics and flags above before using this view for trading decisions.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI CALLER (single function, routes prompt by feature_key)
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_graph(context: dict) -> str:
    """
    Calls Google Gemini API with the graph research context.
    Routes to the correct system prompt based on context['feature_type'].
    Falls back to deterministic explanation on any error.
    """
    feature_key  = context.get("feature_type", "")
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    system_prompt = _PROMPT_MAP.get(feature_key, "")
    if not gemini_key or not system_prompt:
        return _fallback_explanation(feature_key, context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        f"Here is the current {feature_key.replace('_', ' ').title()} graph output "
        f"for ticker {context.get('ticker', 'N/A')}. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"maxOutputTokens": 900, "temperature": 0.2},
    }

    req = urlrequest.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        text  = "".join(p.get("text", "") for p in parts).strip()
        return text or _fallback_explanation(feature_key, context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_explanation(feature_key, context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — UNIFIED UI RENDERER
# Called once per ticker after the graph is drawn.
# Layers: 1 (flags) → 2 (context preview + button) → 3 (AI output)
# ══════════════════════════════════════════════════════════════════════════════

def render_ai_decoder(
    feature_key: str,
    ticker: str,
    benchmark: str,
    payload: dict,
    feature: dict,
) -> None:
    """
    Renders the full 3-layer AI decoder below each graph view.
    Mirrors the portfolio and factor page decoder UI exactly.
    """
    st.markdown("---")
    st.markdown(f"""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Graph Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    {feature_key.replace('_',' ').title()} · {ticker} · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
    st.caption(
        "Translates the graph output above into plain English. "
        "Reads the actual computed numbers — not generic descriptions. "
        "Does not give financial advice."
    )

    # ── LAYER 1: Deterministic danger flags ───────────────────────────────────
    flag_fn = _FLAG_DISPATCH.get(feature_key)
    danger_flags = flag_fn(ticker, feature, payload) if flag_fn else []

    if danger_flags:
        n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
        n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
        n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

        badge_html = ""
        if n_danger:
            badge_html += (f'<span style="background:#dc3232;color:#fff;border-radius:4px;'
                           f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
                           f'⛔ {n_danger} DANGER</span>')
        if n_warning:
            badge_html += (f'<span style="background:#e67e00;color:#fff;border-radius:4px;'
                           f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
                           f'⚠️ {n_warning} WARNING</span>')
        if n_info:
            badge_html += (f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;'
                           f'padding:2px 8px;font-size:12px;font-weight:600;">'
                           f'ℹ️ {n_info} INFO</span>')
        st.markdown(f'<div style="margin:10px 0 6px;">{badge_html}</div>',
                     unsafe_allow_html=True)

        for flag in danger_flags:
            color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
            bg_map    = {"DANGER": "rgba(220,50,50,0.08)",
                          "WARNING": "rgba(230,126,0,0.08)",
                          "INFO":    "rgba(26,111,160,0.08)"}
            st.markdown(
                f"""<div style="background:{bg_map[flag['severity']]};
                    border-left:3px solid {color_map[flag['severity']]};
                    border-radius:0 6px 6px 0;padding:10px 14px;margin:6px 0;
                    font-size:13px;line-height:1.55;">
                  <span style="font-weight:700;color:{color_map[flag['severity']]};">
                    {flag['severity']} · {flag['code']}</span><br>{flag['message']}
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ Pre-flight checks passed — no critical flags for this graph view.")

    st.markdown("")

    # ── LAYER 2: Context builder + button ─────────────────────────────────────
    graph_context = _build_context(
        feature_key=feature_key,
        ticker=ticker,
        benchmark=benchmark,
        payload=payload,
        feature=feature,
        danger_flags=danger_flags,
        data_rows=payload.get("rows", 0),
        last_close=payload.get("last_close"),
    )

    # Session-state key is scoped per ticker + feature so multiple tickers
    # and switching between graph views all work independently
    state_key     = f"graph_ai_{ticker}_{feature_key}"
    ctx_key_store = f"graph_ctx_{ticker}_{feature_key}"

    context_fingerprint = json.dumps(
        {k: v for k, v in graph_context.items() if k != "danger_flags"},
        sort_keys=True, default=str,
    )
    if st.session_state.get(ctx_key_store) != context_fingerprint:
        st.session_state[ctx_key_store] = context_fingerprint
        st.session_state[state_key]     = ""

    col_btn, col_ctx = st.columns([1, 2])

    with col_btn:
        st.markdown("**What Gemini sees:**")
        preview_rows = [
            {"Field": "Ticker",        "Value": ticker},
            {"Field": "Feature",       "Value": feature_key.replace("_", " ").title()},
            {"Field": "Benchmark",     "Value": benchmark},
            {"Field": "Data rows",     "Value": str(payload.get("rows", "N/A"))},
            {"Field": "Last close",    "Value": str(payload.get("last_close", "N/A"))},
            {"Field": "Danger flags",  "Value": str(sum(1 for f in danger_flags if f["severity"] == "DANGER"))},
            {"Field": "Warning flags", "Value": str(sum(1 for f in danger_flags if f["severity"] == "WARNING"))},
        ]
        # Add the most important metric cards to the preview
        for k, v in list(feature.get("cards", {}).items())[:5]:
            preview_rows.append({"Field": k, "Value": str(v)})

        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

        gemini_key_set = bool(getattr(cfg, "GEMINI_API_KEY", ""))
        if not gemini_key_set:
            st.info("💡 Add `GEMINI_API_KEY` to your `.env` for AI explanations. "
                    "Deterministic fallback shown without it.")

        decode_btn = st.button(
            "🤖 Decode for Me",
            type="primary",
            key=f"decode_{ticker}_{feature_key}",
            use_container_width=True,
            help=f"Explains the {feature_key.replace('_',' ').title()} output for {ticker} in plain English.",
        )
        clear_btn = st.button(
            "Clear explanation",
            key=f"clear_{ticker}_{feature_key}",
            use_container_width=True,
        )

    with col_ctx:
        st.markdown("**How this works:**")
        st.markdown(f"""
<div style="background:rgba(14,22,42,0.82);border:1px solid rgba(11,224,255,0.18);
    border-radius:10px;padding:16px 18px;font-size:13px;line-height:1.65;">
  <div style="font-weight:700;color:#e8f4fd;margin-bottom:10px;">What happens when you click Decode:</div>
  <ol style="margin:0;padding-left:18px;color:#a8c4d8;">
    <li style="margin-bottom:6px;">
      Pre-flight checks run first — danger flags above are always deterministic,
      regardless of whether you click Decode.
    </li>
    <li style="margin-bottom:6px;">
      The actual computed numbers for the
      <strong>{feature_key.replace('_',' ').title()}</strong> view
      ({ticker} vs {benchmark}) are sent to Gemini — not generic descriptions.
    </li>
    <li style="margin-bottom:6px;">
      Gemini uses a <strong>graph-specific system prompt</strong> with the
      correct thresholds for this view (different from portfolio or factor prompts).
    </li>
    <li style="margin-bottom:6px;">
      Output: <strong>4 sections</strong> — what the output says · what each number means ·
      red flags · plain-English conclusion.
    </li>
    <li>A <strong>mandatory disclaimer</strong> is always appended.</li>
  </ol>
</div>""", unsafe_allow_html=True)

    if clear_btn:
        st.session_state[state_key] = ""

    if decode_btn:
        with st.spinner(f"Gemini is reading the {feature_key.replace('_',' ').title()} "
                         f"output for {ticker} and writing your explanation..."):
            st.session_state[state_key] = _call_gemini_graph(graph_context)

    # ── LAYER 3: AI output ────────────────────────────────────────────────────
    if st.session_state.get(state_key):
        st.markdown("")
        st.markdown(
            """<div style="background:rgba(14,22,42,0.82);border:1px solid rgba(11,224,255,0.28);
                border-radius:12px;padding:20px 24px;margin-top:8px;">""",
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state[state_key])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("")
        st.markdown(
            f"""<div style="border:1px dashed rgba(11,224,255,0.18);border-radius:10px;
                padding:20px;text-align:center;color:rgba(200,220,240,0.4);font-size:14px;">
              Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
              of the {feature_key.replace('_',' ').title()} view for {ticker}.
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP & ORIGINAL PAGE LOGIC (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Graphs | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("Graph Research Lab")
st.caption("Seven non-overlapping quant views built from the shared OHLCV engine")
render_data_engine_controls("graphs")
global_start = get_global_start_date()
st.session_state.setdefault("graphs_generated", False)

feature_options = [item["label"] for item in GRAPH_FEATURES]
feature_lookup  = {item["label"]: item["key"] for item in GRAPH_FEATURES}

render_cols = st.columns([2.2, 1.4, 1.1])
tickers = render_multi_ticker_input(
    "Tickers (comma-separated)",
    key="graphs_tickers",
    default=["GOOG"],
    container=render_cols[0],
)
selected_feature_label = render_cols[1].selectbox("Research View", feature_options)
benchmark = render_single_ticker_input(
    "Benchmark", key="graphs_benchmark",
    default=DEFAULT_GRAPH_BENCHMARK, container=render_cols[2],
)

if st.button("Generate Graph View", type="primary"):
    st.session_state["graphs_generated"] = True

if st.session_state["graphs_generated"]:
    benchmark_df = load_ticker_data(benchmark)
    feature_key  = feature_lookup[selected_feature_label]

    for ticker in tickers:
        raw_df  = load_ticker_data(ticker)
        payload = build_graph_feature_payload(
            raw_df, ticker=ticker,
            benchmark_df=benchmark_df,
            benchmark_ticker=benchmark,
        )
        feature = payload["features"][feature_key]

        st.caption(f"{ticker} - {data_engine_status(raw_df)}")
        st.caption(f"Global Static Start Date: {global_start}")
        st.subheader(f"{selected_feature_label} - {ticker}")
        _show_metric_cards(payload, feature.get("cards", {}))

        # ── Original graph renderers (unchanged) ──────────────────────────────
        if feature_key == "relative_strength":
            _render_relative_strength(payload, feature)
        elif feature_key == "volume_profile":
            _render_volume_profile(payload, feature)
        elif feature_key == "gap_session":
            _render_gap_session(feature)
        elif feature_key == "seasonality":
            _render_seasonality(feature)
        elif feature_key == "volume_shock":
            _render_volume_shock(feature)
        elif feature_key == "breakout_context":
            _render_breakout_context(feature)
        elif feature_key == "candle_structure":
            _render_candle_structure(feature)

        # ── AI DECODER — renders immediately below each ticker's graph ────────
        render_ai_decoder(
            feature_key=feature_key,
            ticker=ticker,
            benchmark=benchmark,
            payload=payload,
            feature=feature,
        )

qe_faq_section("FAQs", [
    ("What is the graphs page for?",
     "It shows several research-style views that are built from the same market data engine, "
     "so you can compare structure without changing screens."),
    ("How should I choose a view?",
     "Pick the view that matches your question: relative strength, volume, "
     "seasonality, gaps, breakouts, or candle structure."),
    ("Do I need to generate graphs every time?",
     "Yes, because the page only computes the selected view after you click Generate Graph View. "
     "That keeps the app responsive."),
    ("What is the benchmark doing?",
     "The benchmark gives you a reference line so the ticker can be judged against "
     "something more stable than its own history."),
    ("How does the AI decoder know which graph I'm looking at?",
     "The decoder is graph-aware — it uses a different system prompt, different danger flags, "
     "and different context for each of the 7 graph views. "
     "Switching the Research View dropdown changes what Gemini knows and checks."),
    ("Can I decode multiple tickers at once?",
     "Yes — if you enter multiple tickers, each gets its own graph and its own "
     "Decode button with independent session state."),
])