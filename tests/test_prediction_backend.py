"""Unit tests for the modular prediction backend."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.prediction.evaluation import agreement_confidence, regression_metrics, weighted_average, weights_from_errors
from core.prediction.preprocessing import MODEL_FEATURE_COLUMNS, build_feature_frame, infer_next_ohlcv_row, prepare_sequence_dataset, prepare_tabular_dataset


def _sample_ohlcv(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2021-01-01", periods=rows, freq="B")
    close = 120 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, rows)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, rows)),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, rows))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, rows))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 4_000_000, rows).astype(float),
        },
        index=index,
    )


def test_build_feature_frame_adds_required_columns():
    feature_frame = build_feature_frame(_sample_ohlcv())
    for column in MODEL_FEATURE_COLUMNS:
        assert column in feature_frame.columns


def test_prepare_datasets_produce_samples():
    feature_frame = build_feature_frame(_sample_ohlcv())
    X_flat, y_flat, _ = prepare_tabular_dataset(feature_frame, MODEL_FEATURE_COLUMNS)
    X_seq, y_seq, _ = prepare_sequence_dataset(feature_frame, MODEL_FEATURE_COLUMNS, look_back=30)

    assert len(X_flat) == len(y_flat)
    assert len(X_seq) == len(y_seq)
    assert X_seq.shape[1] == 30
    assert X_seq.shape[2] == len(MODEL_FEATURE_COLUMNS)


def test_weighted_average_prefers_lower_error_model():
    predictions = pd.DataFrame(
        {
            "LSTM": [101.0, 102.0],
            "XGBoost": [100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    metrics = {
        "LSTM": {"mae": 2.0},
        "XGBoost": {"mae": 1.0},
    }

    weights = weights_from_errors(metrics)
    ensemble = weighted_average(predictions, weights)

    assert weights["XGBoost"] > weights["LSTM"]
    assert ensemble.iloc[0] < predictions["LSTM"].iloc[0]


def test_agreement_confidence_drops_when_models_diverge():
    aligned = pd.DataFrame({"A": [100.0], "B": [100.5], "C": [99.8]})
    dispersed = pd.DataFrame({"A": [100.0], "B": [120.0], "C": [80.0]})

    assert agreement_confidence(aligned).iloc[0] > agreement_confidence(dispersed).iloc[0]


def test_regression_metrics_are_non_negative():
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_infer_next_ohlcv_row_appends_business_day():
    history = _sample_ohlcv()
    next_row = infer_next_ohlcv_row(history, predicted_close=125.0)

    assert len(next_row) == 1
    assert next_row.index[0] > history.index[-1]
    assert next_row["High"].iloc[0] >= max(next_row["Open"].iloc[0], next_row["Close"].iloc[0])
    assert next_row["Low"].iloc[0] <= min(next_row["Open"].iloc[0], next_row["Close"].iloc[0])
