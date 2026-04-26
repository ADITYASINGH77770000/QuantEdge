"""Multi-model prediction studio backed by the modular forecasting pipeline.

AI LAYER — Gemini AI Prediction Decoder (bottom of page, same 3-layer design as portfolio & dashboard):
  Layer 1: Deterministic danger flags (always shown, no AI)
            Checks: low confidence score, model unavailability, high MAE/RMSE,
                    ensemble disagreement (models diverge), negative forecast delta,
                    data source = demo, too few history days.
  Layer 2: Context builder + "Decode for Me" button
            Packages ensemble forecast, model metrics (MAE/RMSE/status/backend),
            confidence score, forecast delta, ensemble weights, warnings,
            and data source into a structured JSON context for Gemini.
  Layer 3: Gemini output — structured 4-section explanation with mandatory disclaimer
            Falls back to deterministic explanation if key missing or API call fails.

  Uses GEMINI_API_KEY + GEMINI_MODEL from utils/config.py (already present in .env).
  Architecture mirrors 08_portfolio.py and 01_dashboard.py exactly — same flag
  severity system, same context-key cache-busting, same fallback chain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings as pywarnings
from urllib import error as urlerror
from urllib import request as urlrequest

pywarnings.filterwarnings("ignore")

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
    render_single_ticker_input,
)
from core.prediction import rerun_prediction_inference, run_multi_model_prediction
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


st.set_page_config(page_title="Prediction | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("Prediction Studio")
qe_neon_divider()
render_data_engine_controls("prediction")
global_start = get_global_start_date()

_STATE_KEY   = "prediction_result"
_HAS_RUN_KEY = "prediction_generated"
st.session_state.setdefault(_HAS_RUN_KEY, False)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
    history        = payload["history"]
    forecast_frame = payload["forecast_frame"]
    last_close     = float(history["Close"].iloc[-1]) if not history.empty else None
    final_pred     = payload.get("final_prediction")
    confidence     = payload.get("confidence_score")
    delta = None if last_close in {None, 0} or final_pred is None else (final_pred / last_close) - 1.0

    cols = st.columns(4)
    cols[0].metric("Last Close",        _currency(last_close))
    cols[1].metric("Ensemble Forecast", _currency(final_pred))
    cols[2].metric("Forecast Delta",    _percent(delta))
    cols[3].metric("Confidence",        _percent(confidence))

    fig = go.Figure()
    history_tail = history.tail(90)
    fig.add_trace(go.Scatter(
        x=history_tail.index, y=history_tail["Close"],
        mode="lines", name="Historical Close",
        line=dict(color="#ffffff", width=2),
    ))
    for model_name in ["LSTM", "XGBoost", "Transformer"]:
        if model_name in forecast_frame.columns:
            color = {"LSTM": "#00f5ff", "XGBoost": "#ffd700", "Transformer": "#ff6b6b"}[model_name]
            fig.add_trace(go.Scatter(
                x=forecast_frame.index, y=forecast_frame[model_name],
                mode="lines", name=model_name,
                line=dict(color=color, width=2, dash="dot"),
            ))
    ensemble_col = "Weighted Ensemble" if payload.get("ensemble_method") == "weighted" else "Simple Average"
    if ensemble_col in forecast_frame.columns:
        fig.add_trace(go.Scatter(
            x=forecast_frame.index, y=forecast_frame[ensemble_col],
            mode="lines", name=ensemble_col,
            line=dict(color="#00ff88", width=3),
        ))
    fig.update_layout(
        template="plotly_dark", title="Forecast Stack",
        xaxis_title="Date", yaxis_title="Price", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    if payload["warnings"]:
        for warning in payload["warnings"]:
            st.warning(warning)


def _render_model_tab(result: dict, model_name: str) -> None:
    metrics       = result["metrics"]
    model_metrics = result["model_metrics"].get(model_name, {})
    forecast      = result["forecasts"].get(model_name)

    if forecast is None or forecast.empty:
        warning = model_metrics.get("warning") or f"{model_name} is unavailable for the current run."
        st.info(warning)
        return

    cols = st.columns(4)
    cols[0].metric("Status",  str(model_metrics.get("status",  "N/A")).upper())
    cols[1].metric("Backend", str(model_metrics.get("backend", "N/A")))
    mae_val  = _metric_value(metrics, model_name, "mae")
    rmse_val = _metric_value(metrics, model_name, "rmse")
    cols[2].metric("MAE",  "N/A" if mae_val  is None else f"{float(mae_val):.4f}")
    cols[3].metric("RMSE", "N/A" if rmse_val is None else f"{float(rmse_val):.4f}")

    history_tail = result["history"].tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_tail.index, y=history_tail["Close"],
        mode="lines", name="Historical Close",
        line=dict(color="#ffffff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["Predicted Close"],
        mode="lines+markers", name=model_name,
        line=dict(color="#00f5ff", width=3),
    ))
    fig.update_layout(
        template="plotly_dark", title=f"{model_name} Forecast",
        xaxis_title="Date", yaxis_title="Price", height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast.style.format("{:.2f}"), use_container_width=True)
    st.download_button(
        f"Download {model_name} Forecast",
        data=forecast.to_csv().encode(),
        file_name=f"{result['ticker'].lower()}_{model_name.lower()}_forecast.csv",
        mime="text/csv",
        key=f"download_{model_name.lower()}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER 1 — DETERMINISTIC DANGER FLAGS
# Always runs after a prediction is available. No AI required.
# Mirrors _compute_dashboard_danger_flags() / _compute_portfolio_danger_flags()
# ══════════════════════════════════════════════════════════════════════════════

def _compute_prediction_danger_flags(
    result: dict,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the prediction output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.

    Thresholds:
      Confidence < 0.5:           DANGER  — models disagree significantly
      Confidence 0.5–0.65:        WARNING — low model agreement
      Any model unavailable:      WARNING — ensemble running on fewer models
      Forecast delta < -10%:      WARNING — bearish ensemble forecast
      Forecast delta > +30%:      WARNING — potentially overfit / unrealistic upside
      MAE > 5% of last close:     WARNING — high absolute error per model
      RMSE > 8% of last close:    WARNING — high root-mean-square error
      history < 252 trading days: WARNING — less than 1yr of data; model may underfit
      Model warnings present:     INFO    — pass through pipeline warnings
      Demo data:                  INFO
    """
    flags = []

    ticker         = result.get("ticker", "UNKNOWN")
    history        = result.get("history", pd.DataFrame())
    model_metrics  = result.get("model_metrics", {})
    metrics_frame  = result.get("metrics", pd.DataFrame())
    confidence     = result.get("confidence_score")
    final_pred     = result.get("final_prediction")
    pipeline_warns = result.get("warnings", [])
    last_close     = float(history["Close"].iloc[-1]) if not history.empty else None

    # ── Data quality ──────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Prediction for {ticker} is running on SYNTHETIC demo data, not real "
                "market prices. All forecast figures — ensemble price, MAE, RMSE — are "
                "illustrative only. Do not make trading decisions based on this output."
            ),
        })

    # ── History length ─────────────────────────────────────────────────────────
    n_days = len(history)
    if n_days < 252:
        flags.append({
            "severity": "WARNING",
            "code": "SHORT_HISTORY",
            "message": (
                f"Only {n_days} trading days of history available — less than one full year. "
                "Time-series models (especially LSTM) typically need 252+ days to capture "
                "meaningful seasonal and trend patterns. The forecast may be unreliable."
            ),
        })

    # ── Confidence score ───────────────────────────────────────────────────────
    if confidence is not None:
        if confidence < 0.50:
            flags.append({
                "severity": "DANGER",
                "code": "LOW_CONFIDENCE",
                "message": (
                    f"Ensemble confidence score is {confidence:.2%} — critically low. "
                    "The three models are pointing in significantly different directions. "
                    "The ensemble average may be meaningless noise. Do not rely on this "
                    "forecast for directional decision-making."
                ),
            })
        elif confidence < 0.65:
            flags.append({
                "severity": "WARNING",
                "code": "MODERATE_CONFIDENCE",
                "message": (
                    f"Ensemble confidence score is {confidence:.2%} — below the reliable "
                    "threshold of 65%. Models have meaningful disagreement. Treat the "
                    "forecast as directional indication only, not a price target."
                ),
            })

    # ── Model availability ────────────────────────────────────────────────────
    unavailable = [
        name for name, mm in model_metrics.items()
        if mm.get("status", "").lower() not in ("ok", "success", "trained", "fitted")
    ]
    if unavailable:
        flags.append({
            "severity": "WARNING",
            "code": "MODELS_UNAVAILABLE",
            "message": (
                f"The following model(s) did not produce a valid forecast: "
                f"{', '.join(unavailable)}. "
                "The ensemble is running on fewer models than configured, which reduces "
                "robustness and may bias the ensemble toward the available model(s)."
            ),
        })

    # ── Forecast delta extremes ────────────────────────────────────────────────
    if last_close and final_pred:
        delta = (final_pred / last_close) - 1.0
        if delta < -0.10:
            flags.append({
                "severity": "WARNING",
                "code": "BEARISH_FORECAST",
                "message": (
                    f"The ensemble forecasts a decline of {delta:.2%} from the last close "
                    f"(${last_close:,.2f} → ${final_pred:,.2f}). "
                    "This is a meaningfully bearish signal. Verify against the individual "
                    "model tabs — if only one model is driving the decline, discount accordingly."
                ),
            })
        elif delta > 0.30:
            flags.append({
                "severity": "WARNING",
                "code": "AGGRESSIVE_UPSIDE",
                "message": (
                    f"The ensemble forecasts a gain of {delta:.2%} — unusually large upside. "
                    "This level of predicted appreciation over the forecast horizon may indicate "
                    "an overfit model or a data leakage issue. Review model MAE/RMSE carefully."
                ),
            })

    # ── MAE / RMSE per model ──────────────────────────────────────────────────
    if last_close and not metrics_frame.empty:
        for model_name in metrics_frame.index:
            mae_val  = _metric_value(metrics_frame, model_name, "mae")
            rmse_val = _metric_value(metrics_frame, model_name, "rmse")
            if mae_val is not None and float(mae_val) > last_close * 0.05:
                flags.append({
                    "severity": "WARNING",
                    "code": f"HIGH_MAE_{model_name.upper()}",
                    "message": (
                        f"{model_name} MAE is {float(mae_val):.4f} — exceeds 5% of the last "
                        f"close price (${last_close:,.2f}). The model's average error is large "
                        "relative to the price level. Treat its forecast with caution."
                    ),
                })
            if rmse_val is not None and float(rmse_val) > last_close * 0.08:
                flags.append({
                    "severity": "WARNING",
                    "code": f"HIGH_RMSE_{model_name.upper()}",
                    "message": (
                        f"{model_name} RMSE is {float(rmse_val):.4f} — exceeds 8% of the "
                        f"last close (${last_close:,.2f}). High RMSE means the model has "
                        "significant large-error events in its validation period."
                    ),
                })

    # ── Pipeline warnings passthrough ─────────────────────────────────────────
    for warn in pipeline_warns:
        if warn:
            flags.append({
                "severity": "INFO",
                "code": "PIPELINE_WARNING",
                "message": f"Pipeline: {warn}",
            })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER 2 — PREDICTION CONTEXT BUILDER
# Mirrors _build_portfolio_context() / _build_dashboard_context() exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _build_prediction_context(
    result: dict,
    danger_flags: list[dict],
    steps: int,
    look_back: int,
    epochs: int,
    ensemble_method: str,
    data_source: str = "unknown",
) -> dict:
    ticker         = result.get("ticker", "UNKNOWN")
    history        = result.get("history", pd.DataFrame())
    model_metrics  = result.get("model_metrics", {})
    metrics_frame  = result.get("metrics", pd.DataFrame())
    confidence     = result.get("confidence_score")
    final_pred     = result.get("final_prediction")
    ensemble_wts   = result.get("ensemble_weights", {})
    last_close     = float(history["Close"].iloc[-1]) if not history.empty else None
    delta          = (final_pred / last_close - 1.0) if (last_close and final_pred) else None

    # Per-model metrics
    models_out = {}
    for name, mm in model_metrics.items():
        mae_val  = _metric_value(metrics_frame, name, "mae")
        rmse_val = _metric_value(metrics_frame, name, "rmse")
        models_out[name] = {
            "status":  mm.get("status", "N/A"),
            "backend": mm.get("backend", "N/A"),
            "mae":     round(float(mae_val), 6)  if mae_val  is not None else None,
            "rmse":    round(float(rmse_val), 6) if rmse_val is not None else None,
            "warning": mm.get("warning"),
        }

    # Ensemble weights
    wts_out = {k: round(float(v), 4) for k, v in ensemble_wts.items()} if ensemble_wts else {}

    # Next-step per-model predictions
    next_preds = {
        k: round(float(v), 4)
        for k, v in result.get("next_step_predictions", {}).items()
    }

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticker":      ticker,
        "data_source": data_source,
        "n_history_days": len(history),

        # ── Run configuration ─────────────────────────────────────────────────
        "run_config": {
            "forecast_days":   steps,
            "look_back_days":  look_back,
            "epochs":          epochs,
            "ensemble_method": ensemble_method,
        },

        # ── Price ─────────────────────────────────────────────────────────────
        "price": {
            "last_close":         round(last_close, 2) if last_close else None,
            "ensemble_forecast":  round(final_pred, 2) if final_pred else None,
            "forecast_delta_pct": round(delta * 100, 4) if delta is not None else None,
        },

        # ── Ensemble ──────────────────────────────────────────────────────────
        "ensemble": {
            "confidence_score":       round(float(confidence), 4) if confidence else None,
            "weights":                wts_out,
            "next_step_predictions":  next_preds,
        },

        # ── Per-model metrics ─────────────────────────────────────────────────
        "models": models_out,

        # ── Pre-computed flags ────────────────────────────────────────────────
        "danger_flags":        danger_flags,
        "danger_flag_count":   len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count":  len([f for f in danger_flags if f["severity"] == "WARNING"]),

        # ── Reference thresholds ──────────────────────────────────────────────
        "reference_thresholds": {
            "confidence_danger":   0.50,
            "confidence_warning":  0.65,
            "confidence_good":     0.80,
            "forecast_delta_bearish_pct": -10.0,
            "forecast_delta_aggressive_pct": 30.0,
            "mae_pct_of_price_warning": 5.0,
            "rmse_pct_of_price_warning": 8.0,
            "min_history_days": 252,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK
# Mirrors _fallback_dashboard_explanation() exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_prediction_explanation(context: dict) -> str:
    price    = context["price"]
    ensemble = context["ensemble"]
    models   = context["models"]
    flags    = context.get("danger_flags", [])
    cfg_run  = context["run_config"]

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    model_lines = "\n".join(
        f"- **{name}**: status={m['status']}, backend={m['backend']}, "
        f"MAE={m['mae']}, RMSE={m['rmse']}"
        for name, m in models.items()
    )

    return (
        f"### What the output says\n"
        f"Prediction for **{context['ticker']}** using {len(models)} model(s) "
        f"({', '.join(models.keys())}). "
        f"Forecast horizon: **{cfg_run['forecast_days']} days**. "
        f"Ensemble method: **{cfg_run['ensemble_method']}**. "
        f"Data source: **{context['data_source']}**. "
        f"History: **{context['n_history_days']} trading days**.\n\n"
        f"### What each number means\n"
        f"- **Last Close**: ${price['last_close']:,.2f} — most recent observed price.\n"
        f"- **Ensemble Forecast**: ${price['ensemble_forecast']:,.2f} — "
        f"the combined model prediction for the next {cfg_run['forecast_days']} day(s).\n"
        f"- **Forecast Delta**: {price['forecast_delta_pct']:+.2f}% — "
        f"how much the ensemble expects the price to move.\n"
        f"- **Confidence Score**: {ensemble['confidence_score']:.2%} — "
        f"model agreement level "
        f"({'good' if ensemble['confidence_score'] and ensemble['confidence_score'] >= 0.65 else 'low'}).\n"
        f"- **Per-model results**:\n{model_lines}\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"Review confidence and MAE/RMSE flags before acting on this forecast. "
        f"Cross-reference with the individual model tabs for disagreement patterns.\n\n"
        f"⚠️ *This explanation is generated from prediction outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI EXPLAINER
# Direct mirror of _call_gemini_dashboard_explainer() from 01_dashboard.py.
# Uses urllib only (no new dependency). Same fallback chain.
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_PREDICTION_SYSTEM_PROMPT = """You are a senior quantitative analyst embedded inside a professional multi-model stock prediction studio.

Your sole job: explain the prediction output to a NON-TECHNICAL user — a retail investor, family office client, or portfolio manager who understands investing but not machine learning or statistics.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key output in one plain English sentence: confidence score, ensemble forecast, forecast delta, per-model MAE/RMSE, model status, ensemble weights.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should sell" — explain what the models say, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Explain what LSTM, XGBoost, and Transformer each do in one sentence. Do not assume the user knows ML.
8. Write in short paragraphs. No jargon. No LaTeX. No formulas.

THRESHOLD KNOWLEDGE (use these to interpret numbers):
- Confidence < 50%: models are strongly disagreeing — the ensemble output is unreliable
- Confidence 50-65%: moderate disagreement — treat as direction, not price target
- Confidence > 65%: acceptable agreement
- Confidence > 80%: strong agreement — highest reliability
- Forecast delta > +30%: suspiciously large upside — check for overfitting
- Forecast delta < -10%: bearish signal — verify which model is driving it
- MAE > 5% of last close: the model's average error is large relative to price
- RMSE > 8% of last close: significant large-error events in validation period
- History < 252 days: not enough data for robust time-series modelling
- LSTM: recurrent neural network that learns sequential price patterns
- XGBoost: gradient-boosted decision trees that learn from engineered features
- Transformer: attention-based model that captures long-range price dependencies

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: ticker, forecast horizon, data source, ensemble method, overall quality assessment)

### What each number means
(Bullet per key output: ensemble forecast price, forecast delta, confidence score, each model's MAE/RMSE and status, ensemble weights if available)

### Red flags
(If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2-3 sentences max. What a smart non-quant should take away from this prediction output.)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from prediction outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


def _call_gemini_prediction_explainer(context: dict) -> str:
    """
    Calls Google Gemini API with the prediction context.
    Falls back to deterministic explanation on any error.
    Mirrors _call_gemini_dashboard_explainer() exactly.
    """
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_prediction_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current multi-model prediction studio output. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_PREDICTION_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"maxOutputTokens": 900, "temperature": 0.2},
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        text  = "".join(p.get("text", "") for p in parts).strip()
        return text or _fallback_prediction_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_prediction_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONTROLS (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

controls = st.columns([1.2, 0.8, 0.8, 0.8, 0.8])
ticker              = render_single_ticker_input("Ticker", key="prediction_ticker", default="GOOG", container=controls[0])
steps               = controls[1].number_input("Forecast Days", min_value=1, max_value=30, value=10, step=1)
look_back           = controls[2].slider("Look-back", min_value=20, max_value=120, value=60, step=5)
epochs              = controls[3].slider("Epochs", min_value=5, max_value=50, value=10, step=1)
ensemble_method     = controls[4].selectbox("Ensemble", ["weighted", "simple"])
include_transformer = st.checkbox("Include Transformer", value=True)

action_cols    = st.columns([1, 1, 4])
train_clicked  = action_cols[0].button("Train Models", type="primary")
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
    st.session_state[_STATE_KEY]   = result
    st.session_state[_HAS_RUN_KEY] = True
    # Reset AI summary on new training run
    st.session_state["prediction_ai_summary"]     = ""
    st.session_state["prediction_ai_context_key"] = ""

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
        st.session_state[_STATE_KEY]   = result
        st.session_state[_HAS_RUN_KEY] = True
        st.session_state["prediction_ai_summary"]     = ""
        st.session_state["prediction_ai_context_key"] = ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RESULT DISPLAY (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state[_HAS_RUN_KEY]:
    result = st.session_state.get(_STATE_KEY)
    if result is not None:
        st.caption(f"{ticker} - {data_engine_status(result['history'])}")
        st.caption(f"Global Static Start Date: {global_start}")
        _render_summary(result)

        weights   = pd.Series(result.get("ensemble_weights", {}), name="Weight")
        info_cols = st.columns([1.4, 1])
        info_cols[0].dataframe(result["metrics"], use_container_width=True)
        if not weights.empty:
            info_cols[1].dataframe(weights.to_frame().style.format("{:.2%}"), use_container_width=True)
        else:
            info_cols[1].info("No ensemble weights were generated for this run.")

        overview_tab, lstm_tab, xgb_tab, transformer_tab = st.tabs(
            ["Overview", "LSTM", "XGBoost", "Transformer"]
        )
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


        # ══════════════════════════════════════════════════════════════════════
        # AI DECODER SECTION — Gemini-powered
        # Same 3-layer architecture as 08_portfolio.py and 01_dashboard.py:
        #   Layer 1: Deterministic danger badges (always shown, no AI)
        #   Layer 2: "Decode for Me" button → Gemini explanation
        #   Layer 3: Structured AI output with disclaimer
        # ══════════════════════════════════════════════════════════════════════

        qe_neon_divider()

        st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Prediction Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
        st.caption(
            "This section translates the multi-model prediction output above into plain English. "
            "It reads the actual forecast numbers and model metrics — not generic descriptions. "
            "It does not retrain the models. It does not give financial advice."
        )

        # ── LAYER 1: Deterministic danger flags ───────────────────────────────
        data_source  = str(result.get("history", pd.DataFrame()).attrs.get("data_source", "unknown"))
        danger_flags = _compute_prediction_danger_flags(result, data_source=data_source)

        if danger_flags:
            n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
            n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
            n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

            badge_html = ""
            if n_danger:
                badge_html += (
                    f'<span style="background:#dc3232;color:#fff;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
                    f'⛔ {n_danger} DANGER</span>'
                )
            if n_warning:
                badge_html += (
                    f'<span style="background:#e67e00;color:#fff;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
                    f'⚠️ {n_warning} WARNING</span>'
                )
            if n_info:
                badge_html += (
                    f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;'
                    f'padding:2px 8px;font-size:12px;font-weight:600;">'
                    f'ℹ️ {n_info} INFO</span>'
                )
            st.markdown(f'<div style="margin:10px 0 6px;">{badge_html}</div>', unsafe_allow_html=True)

            color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
            bg_map    = {
                "DANGER":  "rgba(220,50,50,0.08)",
                "WARNING": "rgba(230,126,0,0.08)",
                "INFO":    "rgba(26,111,160,0.08)",
            }
            for flag in danger_flags:
                st.markdown(
                    f"""<div style="
                        background:{bg_map[flag['severity']]};
                        border-left:3px solid {color_map[flag['severity']]};
                        border-radius:0 6px 6px 0;
                        padding:10px 14px; margin:6px 0;
                        font-size:13px; line-height:1.55;
                    ">
                      <span style="font-weight:700;color:{color_map[flag['severity']]};">
                        {flag['severity']} · {flag['code']}
                      </span><br>
                      {flag['message']}
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ Pre-flight checks passed — no critical flags detected for this prediction run.")

        st.markdown("")

        # ── LAYER 2: Build context + button ──────────────────────────────────
        prediction_context = _build_prediction_context(
            result=result,
            danger_flags=danger_flags,
            steps=int(steps),
            look_back=int(look_back),
            epochs=int(epochs),
            ensemble_method=ensemble_method,
            data_source=data_source,
        )

        # Cache-bust on new training run (mirrors portfolio + dashboard)
        context_key = json.dumps(
            {k: v for k, v in prediction_context.items() if k != "danger_flags"},
            sort_keys=True, default=str,
        )
        if st.session_state.get("prediction_ai_context_key") != context_key:
            st.session_state.prediction_ai_context_key = context_key
            st.session_state.prediction_ai_summary = ""

        col_btn, col_ctx = st.columns([1, 2])

        with col_btn:
            st.markdown("**What Gemini sees:**")

            # Build preview rows
            history_local = result.get("history", pd.DataFrame())
            last_c        = float(history_local["Close"].iloc[-1]) if not history_local.empty else None
            final_p       = result.get("final_prediction")
            conf_v        = result.get("confidence_score")
            delta_v       = (final_p / last_c - 1.0) if (last_c and final_p) else None
            mf            = result.get("metrics", pd.DataFrame())

            preview_rows = [
                {"Field": "Ticker",           "Value": ticker},
                {"Field": "Data source",      "Value": data_source},
                {"Field": "History days",     "Value": str(len(history_local))},
                {"Field": "Forecast days",    "Value": str(steps)},
                {"Field": "Look-back days",   "Value": str(look_back)},
                {"Field": "Epochs",           "Value": str(epochs)},
                {"Field": "Ensemble method",  "Value": ensemble_method},
                {"Field": "Last close",       "Value": f"${last_c:,.2f}" if last_c else "N/A"},
                {"Field": "Ensemble forecast","Value": f"${final_p:,.2f}" if final_p else "N/A"},
                {"Field": "Forecast delta",   "Value": f"{delta_v:+.2%}" if delta_v is not None else "N/A"},
                {"Field": "Confidence",       "Value": f"{conf_v:.2%}" if conf_v else "N/A"},
            ]
            for name in ["LSTM", "XGBoost", "Transformer"]:
                mae_v  = _metric_value(mf, name, "mae")
                rmse_v = _metric_value(mf, name, "rmse")
                preview_rows.append({
                    "Field": f"{name} MAE",
                    "Value": f"{float(mae_v):.4f}" if mae_v is not None else "N/A",
                })
                preview_rows.append({
                    "Field": f"{name} RMSE",
                    "Value": f"{float(rmse_v):.4f}" if rmse_v is not None else "N/A",
                })
            preview_rows += [
                {"Field": "Danger flags",  "Value": str(sum(1 for f in danger_flags if f["severity"] == "DANGER"))},
                {"Field": "Warning flags", "Value": str(sum(1 for f in danger_flags if f["severity"] == "WARNING"))},
            ]
            st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

            gemini_key_set = bool(getattr(cfg, "GEMINI_API_KEY", ""))
            if not gemini_key_set:
                st.info(
                    "💡 Add `GEMINI_API_KEY` to your `.env` file for Gemini AI explanations. "
                    "Without it, a deterministic quant-safe explanation is shown instead."
                )

            decode_clicked = st.button(
                "🤖 Decode for Me",
                type="primary",
                key="prediction_ai_explain",
                use_container_width=True,
                help="Translates the prediction output above into plain English using Gemini.",
            )
            clear_clicked = st.button(
                "Clear explanation",
                key="prediction_ai_clear",
                use_container_width=True,
            )

        with col_ctx:
            st.markdown("**How this works:**")
            st.markdown("""
<div style="
    background: rgba(14,22,42,0.82);
    border: 1px solid rgba(11,224,255,0.18);
    border-radius: 10px;
    padding: 16px 18px;
    font-size:13px;
    line-height:1.65;
">
  <div style="font-weight:700;color:#e8f4fd;margin-bottom:10px;">What happens when you click Decode:</div>
  <ol style="margin:0;padding-left:18px;color:#a8c4d8;">
    <li style="margin-bottom:6px;">
      The <strong>pre-flight checks above run first</strong> — danger flags are always
      deterministic. They appear regardless of whether you click Decode.
    </li>
    <li style="margin-bottom:6px;">
      The actual numbers from this prediction run (ensemble forecast, forecast delta,
      confidence score, per-model MAE/RMSE/status/backend, ensemble weights,
      data source, history length, and all flags) are sent to Gemini.
    </li>
    <li style="margin-bottom:6px;">
      Gemini explains what LSTM, XGBoost, and Transformer each found, interprets
      the confidence score, flags any dangerous outputs, and writes a plain-English
      conclusion — using the actual values, not generic descriptions.
    </li>
    <li style="margin-bottom:6px;">
      Output: <strong>4 sections</strong> — what the output says · what each number means ·
      red flags · plain-English conclusion.
    </li>
    <li>
      A <strong>mandatory disclaimer</strong> is appended — this is not financial advice.
    </li>
  </ol>
</div>
""", unsafe_allow_html=True)

        if clear_clicked:
            st.session_state.prediction_ai_summary = ""

        if decode_clicked:
            with st.spinner("Gemini is reading the prediction output and writing your plain-English explanation..."):
                st.session_state.prediction_ai_summary = _call_gemini_prediction_explainer(prediction_context)

        # ── LAYER 3: AI output ────────────────────────────────────────────────
        if st.session_state.get("prediction_ai_summary"):
            st.markdown("")
            st.markdown(
                """<div style="
                    background: rgba(14,22,42,0.82);
                    border: 1px solid rgba(11,224,255,0.28);
                    border-radius: 12px;
                    padding: 20px 24px;
                    margin-top: 8px;
                ">""",
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.prediction_ai_summary)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("")
            st.markdown(
                """<div style="
                    border:1px dashed rgba(11,224,255,0.18);
                    border-radius:10px; padding:20px;
                    text-align:center;
                    color:rgba(200,220,240,0.4);
                    font-size:14px;
                ">
                  Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
                  of the prediction output above.
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# FAQs (extended with AI decoder entries)
# ══════════════════════════════════════════════════════════════════════════════

qe_faq_section("FAQs", [
    ("What should I train first on this page?", "Start with the default ticker and run Train Models once. That builds the LSTM, XGBoost, and Transformer forecasts together so you can compare them side by side."),
    ("Why is one model unavailable sometimes?", "Some backends may be missing in your environment or may fail on short histories. The ensemble still works with the remaining models, and the page shows which backend was used."),
    ("When should I trust the ensemble forecast?", "Use it as a directional guide, not a guarantee. The forecast is most useful when model confidence is high and all three model lines are moving in the same direction."),
    ("What is the best workflow here?", "Train once, review the ensemble chart, then use Refresh Forecast after changing the date range or inputs so you keep the trained models but update the prediction view."),
    ("What does the AI Decoder do?", "It reads the actual computed numbers — confidence score, MAE, RMSE, forecast delta, ensemble weights — and explains what each one means in plain English. It does not retrain models or give financial advice."),
    ("When do the danger flags appear?", "Flags fire deterministically after every training run — no AI needed. They check: confidence below thresholds, model unavailability, extreme forecast delta, high MAE/RMSE relative to price, insufficient history, and pipeline warnings."),
    ("What is confidence score?", "It measures how much the three models agree on direction. Above 65% means reasonable agreement. Below 50% means the models are pointing in different directions — the ensemble average may be unreliable."),
])