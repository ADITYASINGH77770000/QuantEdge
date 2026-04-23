"""Multi-model prediction studio backed by the modular forecasting pipeline."""

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
    render_single_ticker_input,
)
from core.prediction import rerun_prediction_inference, run_multi_model_prediction
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


st.set_page_config(page_title="Prediction | QuantEdge", layout="wide")
st.title("Prediction Studio")
qe_neon_divider()
render_data_engine_controls("prediction")
global_start = get_global_start_date()

_STATE_KEY = "prediction_result"
_HAS_RUN_KEY = "prediction_generated"
st.session_state.setdefault(_HAS_RUN_KEY, False)


def _currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def _percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def _metric_value(frame: pd.DataFrame, model_name: str, column: str):
    if frame.empty or model_name not in frame.index or column not in frame.columns:
        return None
    value = frame.loc[model_name, column]
    return None if pd.isna(value) else value


def _render_summary(payload: dict) -> None:
    history = payload["history"]
    forecast_frame = payload["forecast_frame"]
    last_close = float(history["Close"].iloc[-1]) if not history.empty else None
    final_prediction = payload.get("final_prediction")
    confidence = payload.get("confidence_score")
    delta = None if last_close in {None, 0} or final_prediction is None else (final_prediction / last_close) - 1.0

    cols = st.columns(4)
    cols[0].metric("Last Close", _currency(last_close))
    cols[1].metric("Ensemble Forecast", _currency(final_prediction))
    cols[2].metric("Forecast Delta", _percent(delta))
    cols[3].metric("Confidence", _percent(confidence))

    fig = go.Figure()
    history_tail = history.tail(90)
    fig.add_trace(
        go.Scatter(
            x=history_tail.index,
            y=history_tail["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#ffffff", width=2),
        )
    )
    for model_name in ["LSTM", "XGBoost", "Transformer"]:
        if model_name in forecast_frame.columns:
            color = {"LSTM": "#00f5ff", "XGBoost": "#ffd700", "Transformer": "#ff6b6b"}[model_name]
            fig.add_trace(
                go.Scatter(
                    x=forecast_frame.index,
                    y=forecast_frame[model_name],
                    mode="lines",
                    name=model_name,
                    line=dict(color=color, width=2, dash="dot"),
                )
            )

    ensemble_column = "Weighted Ensemble" if payload.get("ensemble_method") == "weighted" else "Simple Average"
    if ensemble_column in forecast_frame.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_frame.index,
                y=forecast_frame[ensemble_column],
                mode="lines",
                name=ensemble_column,
                line=dict(color="#00ff88", width=3),
            )
        )

    fig.update_layout(template="plotly_dark", title="Forecast Stack", xaxis_title="Date", yaxis_title="Price", height=420)
    st.plotly_chart(fig, use_container_width=True)

    if payload["warnings"]:
        for warning in payload["warnings"]:
            st.warning(warning)


def _render_model_tab(result: dict, model_name: str) -> None:
    metrics = result["metrics"]
    model_metrics = result["model_metrics"].get(model_name, {})
    forecast = result["forecasts"].get(model_name)

    if forecast is None or forecast.empty:
        warning = model_metrics.get("warning") or f"{model_name} is unavailable for the current run."
        st.info(warning)
        return

    cols = st.columns(4)
    cols[0].metric("Status", str(model_metrics.get("status", "N/A")).upper())
    cols[1].metric("Backend", str(model_metrics.get("backend", "N/A")))
    cols[2].metric("MAE", "N/A" if _metric_value(metrics, model_name, "mae") is None else f"{float(_metric_value(metrics, model_name, 'mae')):.4f}")
    cols[3].metric("RMSE", "N/A" if _metric_value(metrics, model_name, "rmse") is None else f"{float(_metric_value(metrics, model_name, 'rmse')):.4f}")

    history_tail = result["history"].tail(90)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_tail.index,
            y=history_tail["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#ffffff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["Predicted Close"],
            mode="lines+markers",
            name=model_name,
            line=dict(color="#00f5ff", width=3),
        )
    )
    fig.update_layout(template="plotly_dark", title=f"{model_name} Forecast", xaxis_title="Date", yaxis_title="Price", height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast.style.format("{:.2f}"), use_container_width=True)
    st.download_button(
        f"Download {model_name} Forecast",
        data=forecast.to_csv().encode(),
        file_name=f"{result['ticker'].lower()}_{model_name.lower()}_forecast.csv",
        mime="text/csv",
        key=f"download_{model_name.lower()}",
    )


controls = st.columns([1.2, 0.8, 0.8, 0.8, 0.8])
ticker = render_single_ticker_input("Ticker", key="prediction_ticker", default="GOOG", container=controls[0])
steps = controls[1].number_input("Forecast Days", min_value=1, max_value=30, value=10, step=1)
look_back = controls[2].slider("Look-back", min_value=20, max_value=120, value=60, step=5)
epochs = controls[3].slider("Epochs", min_value=5, max_value=50, value=10, step=1)
ensemble_method = controls[4].selectbox("Ensemble", ["weighted", "simple"])
include_transformer = st.checkbox("Include Transformer", value=True)

action_cols = st.columns([1, 1, 4])
train_clicked = action_cols[0].button("Train Models", type="primary")
refresh_clicked = action_cols[1].button("Refresh Forecast")

if train_clicked:
    raw_df = load_ticker_data(ticker)
    with st.spinner("Training prediction stack..."):
        result = run_multi_model_prediction(
            raw_df,
            ticker=ticker,
            steps=int(steps),
            look_back=int(look_back),
            epochs=int(epochs),
            include_transformer=include_transformer,
            ensemble_method=ensemble_method,
        )
    st.session_state[_STATE_KEY] = result
    st.session_state[_HAS_RUN_KEY] = True

if refresh_clicked:
    previous = st.session_state.get(_STATE_KEY)
    if previous is None:
        st.warning("Train the models once before refreshing inference.")
    else:
        raw_df = load_ticker_data(ticker)
        with st.spinner("Refreshing forecasts using the trained models..."):
            result = rerun_prediction_inference(
                raw_df,
                ticker=ticker,
                trained_models=previous["trained_models"],
                model_metrics=previous["model_metrics"],
                steps=int(steps),
                ensemble_method=ensemble_method,
                warnings=previous.get("warnings"),
            )
        st.session_state[_STATE_KEY] = result
        st.session_state[_HAS_RUN_KEY] = True

if st.session_state[_HAS_RUN_KEY]:
    result = st.session_state.get(_STATE_KEY)
    if result is not None:
        st.caption(f"{ticker} - {data_engine_status(result['history'])}")
        st.caption(f"Global Static Start Date: {global_start}")
        _render_summary(result)

        weights = pd.Series(result.get("ensemble_weights", {}), name="Weight")
        info_cols = st.columns([1.4, 1])
        info_cols[0].dataframe(result["metrics"], use_container_width=True)
        if not weights.empty:
            info_cols[1].dataframe(weights.to_frame().style.format("{:.2%}"), use_container_width=True)
        else:
            info_cols[1].info("No ensemble weights were generated for this run.")

        overview_tab, lstm_tab, xgb_tab, transformer_tab = st.tabs(["Overview", "LSTM", "XGBoost", "Transformer"])

        with overview_tab:
            forecast_frame = result["forecast_frame"].copy()
            st.dataframe(forecast_frame.style.format("{:.2f}"), use_container_width=True)
            st.download_button(
                "Download Ensemble Forecast",
                data=forecast_frame.to_csv().encode(),
                file_name=f"{ticker.lower()}_prediction_ensemble.csv",
                mime="text/csv",
                key="download_ensemble",
            )

        with lstm_tab:
            _render_model_tab(result, "LSTM")

        with xgb_tab:
            _render_model_tab(result, "XGBoost")

        with transformer_tab:
            _render_model_tab(result, "Transformer")

qe_faq_section("FAQs", [
    ("What should I train first on this page?", "Start with the default ticker and run Train Models once. That builds the LSTM, XGBoost, and Transformer forecasts together so you can compare them side by side."),
    ("Why is one model unavailable sometimes?", "Some backends may be missing in your environment or may fail on short histories. The ensemble still works with the remaining models, and the page shows which backend was used."),
    ("When should I trust the ensemble forecast?", "Use it as a directional guide, not a guarantee. The forecast is most useful when model confidence is high and all three model lines are moving in the same direction."),
    ("What is the best workflow here?", "Train once, review the ensemble chart, then use Refresh Forecast after changing the date range or inputs so you keep the trained models but update the prediction view."),
])
