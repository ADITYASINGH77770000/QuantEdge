"""End-to-end orchestration for multi-model stock prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from .evaluation import agreement_confidence, metrics_frame, regression_metrics, simple_average, weighted_average, weights_from_errors
from .models import ModelBundle, predict_next_close, train_lstm_regressor, train_transformer_regressor, train_xgboost_regressor
from .preprocessing import MODEL_FEATURE_COLUMNS, build_feature_frame, clean_ohlcv, infer_next_ohlcv_row, prepare_sequence_dataset, prepare_tabular_dataset


def _record_failure(metrics: dict[str, dict[str, Any]], model_name: str, exc: Exception) -> None:
    metrics[model_name] = {
        "backend": None,
        "status": "unavailable",
        "mse": None,
        "mae": None,
        "rmse": None,
        "warning": str(exc),
    }


def _record_success(
    metrics: dict[str, dict[str, Any]],
    bundle: ModelBundle,
    y_true,
    y_pred,
) -> None:
    scores = regression_metrics(y_true, y_pred)
    metrics[bundle.name] = {
        "backend": bundle.backend,
        "status": "ok",
        **scores,
        "warning": bundle.warning,
    }


def _forecast_model(bundle: ModelBundle, history: pd.DataFrame, steps: int) -> pd.DataFrame:
    synthetic_history = clean_ohlcv(history)
    predictions: list[float] = []
    future_dates: list[pd.Timestamp] = []

    for _ in range(steps):
        feature_frame = build_feature_frame(synthetic_history)
        next_close = predict_next_close(bundle, feature_frame)
        next_row = infer_next_ohlcv_row(synthetic_history, next_close)
        predictions.append(float(next_close))
        future_dates.append(next_row.index[0])
        synthetic_history = pd.concat([synthetic_history, next_row])

    return pd.DataFrame({"Predicted Close": predictions}, index=pd.DatetimeIndex(future_dates, name="Date"))


def _build_prediction_payload(
    *,
    ticker: str,
    clean_history: pd.DataFrame,
    trained_models: dict[str, ModelBundle],
    model_metrics: dict[str, dict[str, Any]],
    steps: int,
    ensemble_method: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Create the shared response payload for train-time and inference-only runs."""
    if not trained_models:
        raise ValueError("No trained models are available for inference.")

    feature_frame = build_feature_frame(clean_history)
    forecasts = {
        model_name: _forecast_model(bundle, clean_history, steps)
        for model_name, bundle in trained_models.items()
    }
    prediction_frame = pd.DataFrame(
        {model_name: forecast["Predicted Close"] for model_name, forecast in forecasts.items()}
    )
    prediction_frame.index.name = "Date"

    ensemble_weights = weights_from_errors(model_metrics)
    prediction_frame["Simple Average"] = simple_average(prediction_frame)
    prediction_frame["Weighted Ensemble"] = weighted_average(
        prediction_frame[[column for column in trained_models if column in prediction_frame.columns]],
        ensemble_weights,
    )
    prediction_frame["Confidence"] = agreement_confidence(
        prediction_frame[[column for column in trained_models if column in prediction_frame.columns]]
    )

    ensemble_column = "Weighted Ensemble" if ensemble_method == "weighted" else "Simple Average"
    next_step_predictions = {column: float(prediction_frame[column].iloc[0]) for column in trained_models}
    warning_list = warnings or []

    return {
        "ticker": ticker,
        "history": clean_history,
        "feature_frame": feature_frame,
        "feature_columns": sorted(
            {column for bundle in trained_models.values() for column in bundle.feature_columns}
        ),
        "forecasts": forecasts,
        "forecast_frame": prediction_frame,
        "metrics": metrics_frame(model_metrics),
        "model_metrics": model_metrics,
        "trained_models": trained_models,
        "ensemble_weights": ensemble_weights,
        "ensemble_method": ensemble_method,
        "next_step_predictions": next_step_predictions,
        "final_prediction": float(prediction_frame[ensemble_column].iloc[0]),
        "confidence_score": float(prediction_frame["Confidence"].iloc[0]),
        "warnings": warning_list + [
            row["warning"]
            for row in model_metrics.values()
            if row.get("warning") and row["warning"] not in warning_list
        ],
    }


def run_multi_model_prediction(
    df: pd.DataFrame,
    ticker: str,
    steps: int = 5,
    look_back: int = 60,
    epochs: int = 10,
    include_transformer: bool = True,
    validation_fraction: float = 0.2,
    ensemble_method: str = "weighted",
) -> dict[str, Any]:
    """Train available models, score them, and produce ensemble forecasts."""
    clean_history = clean_ohlcv(df)
    feature_frame = build_feature_frame(clean_history)
    feature_columns = [column for column in MODEL_FEATURE_COLUMNS if column in feature_frame.columns]

    flat_X, flat_y, _ = prepare_tabular_dataset(feature_frame, feature_columns=feature_columns)
    seq_X, seq_y, _ = prepare_sequence_dataset(
        feature_frame,
        feature_columns=feature_columns,
        look_back=look_back,
    )

    trained_models: dict[str, ModelBundle] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    try:
        xgb_bundle, xgb_true, xgb_pred = train_xgboost_regressor(
            flat_X,
            flat_y,
            feature_columns=feature_columns,
            val_fraction=validation_fraction,
        )
        trained_models[xgb_bundle.name] = xgb_bundle
        _record_success(model_metrics, xgb_bundle, xgb_true, xgb_pred)
        if xgb_bundle.warning:
            warnings.append(xgb_bundle.warning)
    except Exception as exc:
        logger.warning(f"XGBoost training failed: {exc}")
        _record_failure(model_metrics, "XGBoost", exc)

    try:
        lstm_bundle, lstm_true, lstm_pred = train_lstm_regressor(
            seq_X,
            seq_y,
            feature_columns=feature_columns,
            look_back=look_back,
            epochs=epochs,
            val_fraction=validation_fraction,
        )
        trained_models[lstm_bundle.name] = lstm_bundle
        _record_success(model_metrics, lstm_bundle, lstm_true, lstm_pred)
    except Exception as exc:
        logger.warning(f"LSTM training failed: {exc}")
        _record_failure(model_metrics, "LSTM", exc)

    if include_transformer:
        try:
            transformer_bundle, transformer_true, transformer_pred = train_transformer_regressor(
                seq_X,
                seq_y,
                feature_columns=feature_columns,
                look_back=look_back,
                epochs=epochs,
                val_fraction=validation_fraction,
            )
            trained_models[transformer_bundle.name] = transformer_bundle
            _record_success(model_metrics, transformer_bundle, transformer_true, transformer_pred)
        except Exception as exc:
            logger.warning(f"Transformer training failed: {exc}")
            _record_failure(model_metrics, "Transformer", exc)

    if not trained_models:
        raise ValueError("No prediction models were available. Check model dependencies and input history.")

    return _build_prediction_payload(
        ticker=ticker,
        clean_history=clean_history,
        trained_models=trained_models,
        model_metrics=model_metrics,
        steps=steps,
        ensemble_method=ensemble_method,
        warnings=warnings,
    )


def rerun_prediction_inference(
    df: pd.DataFrame,
    *,
    ticker: str,
    trained_models: dict[str, ModelBundle],
    model_metrics: dict[str, dict[str, Any]],
    steps: int = 5,
    ensemble_method: str = "weighted",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Reuse previously trained models to forecast against fresh market data."""
    clean_history = clean_ohlcv(df)
    return _build_prediction_payload(
        ticker=ticker,
        clean_history=clean_history,
        trained_models=trained_models,
        model_metrics=model_metrics,
        steps=steps,
        ensemble_method=ensemble_method,
        warnings=warnings,
    )
