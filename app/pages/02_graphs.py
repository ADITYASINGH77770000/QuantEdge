"""Graph research lab backed by the shared market data engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_engine import (
    data_engine_status,
    get_global_start_date,
    load_ticker_data,
    render_data_engine_controls,
    render_multi_ticker_input,
    render_single_ticker_input,
)
from core.graph_features import DEFAULT_GRAPH_BENCHMARK, GRAPH_FEATURES, build_graph_feature_payload
try:
    from utils.theme import qe_faq_section
except ImportError:
    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        st.markdown("---")
        st.markdown(f"### {title}")
        for question, answer in faqs:
            with st.expander(question):
                st.write(answer)


st.set_page_config(page_title="Graphs | QuantEdge", layout="wide")
st.title("Graph Research Lab")
st.caption("Seven non-overlapping quant views built from the shared OHLCV engine")
render_data_engine_controls("graphs")
global_start = get_global_start_date()
st.session_state.setdefault("graphs_generated", False)

feature_options = [item["label"] for item in GRAPH_FEATURES]
feature_lookup = {item["label"]: item["key"] for item in GRAPH_FEATURES}
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
    if any(
        token in lower
        for token in [
            "return",
            "rate",
            "gap",
            "move",
            "vol",
            "inside",
            "pos",
            "compression",
            "dist",
            "body",
            "loc",
            "avg",
        ]
    ):
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
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2, dash=dash))


def _dark_layout(title: str, **kwargs) -> dict:
    layout = dict(template="plotly_dark", title=title, height=340)
    layout.update(kwargs)
    return layout


def _render_relative_strength(payload: dict, feature: dict) -> None:
    relative = _records_frame(feature.get("relative_series", []))
    rolling = _records_frame(feature.get("rolling_stats", []))
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
        fig2.add_trace(
            go.Scatter(
                x=rolling["Date"],
                y=rolling["Beta63d"],
                mode="lines",
                name="63d Beta",
                yaxis="y2",
                line=dict(color="#b44fff", width=2),
            )
        )
    fig2.update_layout(
        _dark_layout(
            "Relative Ratio - 21d Outperformance - 63d Beta",
            yaxis=dict(title="Relative / Outperformance"),
            yaxis2=dict(title="Beta", overlaying="y", side="right"),
        )
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_volume_profile(payload: dict, feature: dict) -> None:
    profile = _records_frame(feature.get("profile", []))
    price_context = _records_frame(feature.get("price_context", []))
    if profile.empty or price_context.empty:
        st.warning("Not enough history to compute a volume profile.")
        return

    st.caption(feature.get("note", ""))
    cards = feature.get("cards", {})
    poc = cards.get("POC Price")
    value_area_low = cards.get("Value Area Low")
    value_area_high = cards.get("Value Area High")

    fig = go.Figure(
        go.Bar(
            x=profile["Volume"],
            y=profile["PriceMid"],
            orientation="h",
            marker_color="rgba(0,245,255,0.7)",
            name="Volume",
            hovertemplate="Price %{y:.2f}<br>Volume %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(_dark_layout("Volume by Price", xaxis_title="Volume", yaxis_title="Typical Price", height=380))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(_line_trace(price_context["Date"], price_context["Close"], f"{payload['ticker']} Close", "white"))
    if poc is not None:
        fig2.add_trace(_line_trace(price_context["Date"], [poc] * len(price_context), "POC", "#ffd700", dash="dash"))
    if value_area_low is not None:
        fig2.add_trace(
            _line_trace(price_context["Date"], [value_area_low] * len(price_context), "Value Area Low", "#00ff88", dash="dot")
        )
    if value_area_high is not None:
        fig2.add_trace(
            _line_trace(price_context["Date"], [value_area_high] * len(price_context), "Value Area High", "#ff8c00", dash="dot")
        )
    fig2.update_layout(_dark_layout("Close vs Value Area", xaxis_title="Date", yaxis_title="Price"))
    st.plotly_chart(fig2, use_container_width=True)


def _render_gap_session(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    if frame.empty:
        st.warning("Not enough history to compute gap/session decomposition.")
        return

    fig = go.Figure()
    fig.add_bar(x=frame["Date"], y=frame["GapReturn"], name="Gap Return", marker_color="rgba(0,245,255,0.65)")
    fig.add_bar(x=frame["Date"], y=frame["SessionReturn"], name="Intraday Return", marker_color="rgba(255,215,0,0.65)")
    fig.update_layout(_dark_layout("Gap vs Intraday Return", barmode="group", yaxis_tickformat=".1%"))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=frame["GapReturn"],
            y=frame["SessionReturn"],
            mode="markers",
            text=frame["Date"],
            marker=dict(
                size=8,
                color=frame["Filled"].map(lambda value: "#ff8c00" if value else "#00ff88"),
                symbol=frame["SameDirection"].map(lambda value: "circle" if value else "diamond"),
                opacity=0.75,
            ),
            hovertemplate="%{text}<br>Gap %{x:.2%}<br>Session %{y:.2%}<extra></extra>",
            name="Gap Days",
        )
    )
    fig2.update_layout(_dark_layout("Gap Continuation Map", xaxis_title="Gap Return", yaxis_title="Intraday Return"))
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
    fig.add_bar(
        x=weekday["Weekday"],
        y=weekday["AvgReturn"],
        name="Avg Return",
        marker_color=["#00ff88" if value >= 0 else "#ff3366" for value in weekday["AvgReturn"]],
    )
    fig.add_trace(
        go.Scatter(
            x=weekday["Weekday"],
            y=weekday["HitRate"],
            mode="lines+markers",
            name="Hit Rate",
            yaxis="y2",
            line=dict(color="#ffd700", width=2),
        )
    )
    fig.update_layout(
        _dark_layout(
            "Weekday Return & Hit Rate",
            yaxis=dict(title="Avg Return", tickformat=".1%"),
            yaxis2=dict(title="Hit Rate", overlaying="y", side="right", tickformat=".0%"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    pivot = monthly.pivot(index="Year", columns="Month", values="Return").reindex(columns=month_order)
    fig2 = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                hovertemplate="Year %{y}<br>%{x}: %{z:.2%}<extra></extra>",
            )
        ]
    )
    fig2.update_layout(_dark_layout("Monthly Return Heatmap", xaxis_title="Month", yaxis_title="Year", height=360))
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
    fig2.add_trace(
        go.Scatter(
            x=frame["VolumeZ"],
            y=frame["Next5dReturn"],
            mode="markers",
            text=frame["Date"],
            marker=dict(
                size=frame["Next5dVol"].fillna(0).map(lambda value: max(6, min(16, value * 250 + 4))),
                color=frame["ShockFlag"].map(lambda value: "#ffd700" if value else "#00ff88"),
                opacity=0.72,
            ),
            hovertemplate="%{text}<br>Vol Z %{x:.2f}<br>Next 5d %{y:.2%}<extra></extra>",
            name="Forward 5d Return",
        )
    )
    fig2.update_layout(_dark_layout("Shock vs Forward Outcome", xaxis_title="Volume Z-Score", yaxis_title="Next 5d Return"))
    fig2.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)


def _render_breakout_context(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    if frame.empty:
        st.warning("Not enough history to compute breakout context.")
        return

    fig = go.Figure()
    fig.add_trace(_line_trace(frame["Date"], frame["RangePos252d"], "252d Range Position", "#00ff88"))
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Dist63dHigh"],
            mode="lines",
            name="Dist to 63d High",
            yaxis="y2",
            line=dict(color="#ffd700", width=2),
        )
    )
    fig.update_layout(
        _dark_layout(
            "Range Position & Breakout Distance",
            yaxis=dict(title="Range Position", tickformat=".0%"),
            yaxis2=dict(title="Distance to 63d High", overlaying="y", side="right", tickformat=".1%"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(_line_trace(frame["Date"], frame["Compression20d"], "20d Compression", "#b44fff"))
    fig2.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Dist20dHigh"],
            mode="lines",
            name="Dist to 20d High",
            yaxis="y2",
            line=dict(color="#00f5ff", width=2),
        )
    )
    fig2.update_layout(
        _dark_layout(
            "20d Compression & Short-Term Breakout Distance",
            yaxis=dict(title="Compression", tickformat=".1%"),
            yaxis2=dict(title="Distance to 20d High", overlaying="y", side="right", tickformat=".1%"),
        )
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_candle_structure(feature: dict) -> None:
    frame = _records_frame(feature.get("series", []))
    close_buckets = _records_frame(feature.get("close_location_buckets", []))
    wick_buckets = _records_frame(feature.get("wick_buckets", []))
    if frame.empty:
        st.warning("Not enough history to compute candle structure analytics.")
        return

    recent = frame.tail(180)
    fig = go.Figure()
    fig.add_trace(_line_trace(recent["Date"], recent["BodyPct"], "Body % of Range", "#00f5ff"))
    fig.add_trace(_line_trace(recent["Date"], recent["UpperWickPct"], "Upper Wick %", "#ff3366"))
    fig.add_trace(_line_trace(recent["Date"], recent["LowerWickPct"], "Lower Wick %", "#00ff88"))
    fig.add_trace(_line_trace(recent["Date"], recent["CloseLocation"], "Close Location", "#ffd700"))
    fig.update_layout(_dark_layout("Daily Candle Anatomy", yaxis_tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    if not close_buckets.empty:
        fig2.add_bar(
            x=close_buckets["Bucket"],
            y=close_buckets["AvgNext1d"],
            name="Close Location Bucket",
            marker_color="rgba(0,245,255,0.7)",
        )
    if not wick_buckets.empty:
        fig2.add_bar(
            x=wick_buckets["Bucket"],
            y=wick_buckets["AvgNext1d"],
            name="Wick Imbalance Bucket",
            marker_color="rgba(255,215,0,0.7)",
        )
    fig2.update_layout(_dark_layout("Next-Day Return by Structure Bucket", barmode="group", yaxis_tickformat=".2%"))
    st.plotly_chart(fig2, use_container_width=True)


render_cols = st.columns([2.2, 1.4, 1.1])
tickers = render_multi_ticker_input(
    "Tickers (comma-separated)",
    key="graphs_tickers",
    default=["GOOG"],
    container=render_cols[0],
)
selected_feature_label = render_cols[1].selectbox("Research View", feature_options)
benchmark = render_single_ticker_input("Benchmark", key="graphs_benchmark", default=DEFAULT_GRAPH_BENCHMARK, container=render_cols[2])

if st.button("Generate Graph View", type="primary"):
    st.session_state["graphs_generated"] = True

if st.session_state["graphs_generated"]:
    benchmark_df = load_ticker_data(benchmark)
    feature_key = feature_lookup[selected_feature_label]

    for ticker in tickers:
        raw_df = load_ticker_data(ticker)
        payload = build_graph_feature_payload(raw_df, ticker=ticker, benchmark_df=benchmark_df, benchmark_ticker=benchmark)
        feature = payload["features"][feature_key]

        st.caption(f"{ticker} - {data_engine_status(raw_df)}")
        st.caption(f"Global Static Start Date: {global_start}")
        st.subheader(f"{selected_feature_label} - {ticker}")
        _show_metric_cards(payload, feature.get("cards", {}))

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

qe_faq_section("FAQs", [
    ("What is the graphs page for?", "It shows several research-style views that are built from the same market data engine, so you can compare structure without changing screens."),
    ("How should I choose a view?", "Pick the view that matches your question: relative strength, volume, seasonality, gaps, breakouts, or candle structure."),
    ("Do I need to generate graphs every time?", "Yes, because the page only computes the selected view after you click Generate Graph View. That keeps the app responsive."),
    ("What is the benchmark doing?", "The benchmark gives you a reference line so the ticker can be judged against something more stable than its own history."),
])
