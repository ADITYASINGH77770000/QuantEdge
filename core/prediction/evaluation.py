"""Evaluation and ensembling helpers for the prediction backend."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def weights_from_errors(metrics: dict[str, dict[str, object]], error_key: str = "mae") -> dict[str, float]:
    """Assign larger weights to models with lower validation error."""
    raw_weights: dict[str, float] = {}
    for model_name, row in metrics.items():
        error = row.get(error_key)
        if error is None or not np.isfinite(error):
            continue
        raw_weights[model_name] = 1.0 / max(float(error), 1e-8)

    if not raw_weights:
        return {}

    total = sum(raw_weights.values())
    return {model_name: weight / total for model_name, weight in raw_weights.items()}


def simple_average(prediction_frame: pd.DataFrame) -> pd.Series:
    """Simple mean ensemble across available model columns."""
    if prediction_frame.empty:
        return pd.Series(dtype=float)
    return prediction_frame.mean(axis=1)


def weighted_average(
    prediction_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Weighted ensemble using inverse-error weights."""
    if prediction_frame.empty:
        return pd.Series(dtype=float)

    active_columns = [column for column in prediction_frame.columns if column in weights]
    if not active_columns:
        return simple_average(prediction_frame)

    weight_array = np.array([weights[column] for column in active_columns], dtype=float)
    weight_array = weight_array / weight_array.sum()
    weighted_values = prediction_frame[active_columns].to_numpy(dtype=float) @ weight_array
    return pd.Series(weighted_values, index=prediction_frame.index, name="Weighted Ensemble")


def agreement_confidence(prediction_frame: pd.DataFrame) -> pd.Series:
    """Confidence rises when models agree and falls when they diverge."""
    if prediction_frame.empty:
        return pd.Series(dtype=float)

    if prediction_frame.shape[1] == 1:
        return pd.Series(1.0, index=prediction_frame.index, name="Confidence")

    magnitude = prediction_frame.abs().mean(axis=1).clip(lower=1e-6)
    dispersion = prediction_frame.std(axis=1, ddof=0) / magnitude
    confidence = (1.0 / (1.0 + dispersion)).clip(lower=0.0, upper=1.0)
    confidence.name = "Confidence"
    return confidence


def metrics_frame(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert model metrics into a display-friendly table."""
    table = pd.DataFrame.from_dict(metrics, orient="index")
    if table.empty:
        return table

    ordered_columns = [column for column in ["backend", "status", "mse", "mae", "rmse", "warning"] if column in table.columns]
    table = table[ordered_columns]
    table.index.name = "Model"
    return table.sort_index()
