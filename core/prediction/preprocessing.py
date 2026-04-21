"""Data preparation helpers for the prediction backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import rsi as _rsi_canonical   # canonical RSI — single source of truth


BASE_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MODEL_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ma_5",
    "ma_10",
    "ma_20",
    "return_1",
    "return_5",
    "volatility_5",
    "volatility_10",
    "rsi_14",
    "price_range",
    "close_open_change",
    "volume_change",
    "volume_ratio_10",
]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize OHLCV data for model consumption."""
    if df.empty:
        raise ValueError("No OHLCV data available for prediction.")

    missing = [col for col in BASE_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    clean = df.copy()
    clean = clean.sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last"), BASE_OHLCV_COLUMNS]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.interpolate(method="linear", limit_direction="both")
    clean = clean.ffill().bfill().dropna()

    if len(clean) < 80:
        raise ValueError("Need at least 80 rows of history to train the prediction models.")

    return clean


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI — delegates to core.indicators.rsi (canonical single source of truth)."""
    return _rsi_canonical(close, period=period).fillna(50.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators for tabular and sequence models."""
    features = clean_ohlcv(df)
    close = features["Close"]
    volume = features["Volume"]

    features["ma_5"] = close.rolling(5).mean()
    features["ma_10"] = close.rolling(10).mean()
    features["ma_20"] = close.rolling(20).mean()
    features["return_1"] = close.pct_change()
    features["return_5"] = close.pct_change(5)
    features["volatility_5"] = features["return_1"].rolling(5).std()
    features["volatility_10"] = features["return_1"].rolling(10).std()
    features["rsi_14"] = calculate_rsi(close, 14)
    features["price_range"] = (features["High"] - features["Low"]) / close.replace(0.0, np.nan)
    features["close_open_change"] = (features["Close"] - features["Open"]) / features["Open"].replace(0.0, np.nan)
    features["volume_change"] = volume.pct_change()
    features["volume_ratio_10"] = volume / volume.rolling(10).mean().replace(0.0, np.nan)

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < 60:
        raise ValueError("Not enough clean rows remain after feature engineering.")
    return features


def prepare_tabular_dataset(
    feature_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Prepare flat features for tree-based regression models."""
    feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
    dataset = feature_frame[feature_columns].copy()
    dataset["target"] = feature_frame["Close"].shift(-horizon)
    dataset = dataset.dropna()
    return (
        dataset[feature_columns].to_numpy(dtype=np.float32),
        dataset["target"].to_numpy(dtype=np.float32),
        dataset.index,
    )


def prepare_sequence_dataset(
    feature_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    look_back: int = 60,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Prepare rolling windows for LSTM and Transformer models."""
    feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
    values = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    targets = feature_frame["Close"].to_numpy(dtype=np.float32)

    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    target_index: list[pd.Timestamp] = []

    last_start = len(feature_frame) - horizon + 1
    for end in range(look_back, last_start):
        target_pos = end + horizon - 1
        X_seq.append(values[end - look_back:end])
        y_seq.append(float(targets[target_pos]))
        target_index.append(feature_frame.index[target_pos])

    if not X_seq:
        raise ValueError("Not enough history for sequence models. Reduce look-back or load more data.")

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        pd.Index(target_index),
    )


def infer_next_ohlcv_row(history: pd.DataFrame, predicted_close: float) -> pd.DataFrame:
    """Construct a synthetic next OHLCV row for recursive forecasting."""
    clean_history = clean_ohlcv(history)
    last_row = clean_history.iloc[-1]
    returns = clean_history["Close"].pct_change().dropna().tail(20)
    daily_vol = float(returns.std()) if not returns.empty else 0.01
    daily_vol = float(np.clip(daily_vol, 0.0025, 0.08))

    next_open = float(last_row["Close"])
    next_close = float(predicted_close)
    intraday_buffer = max(0.002, daily_vol * 0.5)
    next_high = max(next_open, next_close) * (1.0 + intraday_buffer)
    next_low = min(next_open, next_close) * (1.0 - intraday_buffer)

    avg_volume = clean_history["Volume"].tail(10).mean()
    next_volume = float(avg_volume if pd.notna(avg_volume) else last_row["Volume"])
    next_date = pd.date_range(clean_history.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")[0]

    next_row = pd.DataFrame(
        {
            "Open": [next_open],
            "High": [next_high],
            "Low": [next_low],
            "Close": [next_close],
            "Volume": [next_volume],
        },
        index=pd.DatetimeIndex([next_date], name=clean_history.index.name or "Date"),
    )
    return next_row
