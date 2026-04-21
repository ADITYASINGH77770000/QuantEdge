"""
core/models.py
──────────────────────────────────────────────────────────────────────────────
Forecasting models: LSTM (deep learning), ARIMA (returns), GARCH (volatility).
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from loguru import logger


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm(input_shape: tuple, output_shape: int):
    """Two-layer LSTM model for multi-feature prediction."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(64),
        Dropout(0.1),
        Dense(output_shape),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def prepare_lstm_data(df: pd.DataFrame, features: list,
                      look_back: int = 60):
    """Scale and window data for LSTM."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y   = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back: i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler


def lstm_forecast(df: pd.DataFrame, features: list,
                  steps: int = 5, look_back: int = 60,
                  epochs: int = 10, batch_size: int = 32,
                  progress_cb=None) -> pd.DataFrame:
    """
    Train LSTM and predict `steps` days ahead.

    Returns DataFrame with Date index and 'Predicted <feature>' columns.
    """
    X, y, scaler = prepare_lstm_data(df, features, look_back)
    model = build_lstm((X.shape[1], X.shape[2]), len(features))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_seq   = X[-1].copy()
    preds      = []
    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1, *last_seq.shape), verbose=0)
        preds.append(pred[0])
        last_seq = np.vstack([last_seq[1:], pred])

    preds_inv = scaler.inverse_transform(np.array(preds))
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1),
                                 periods=steps, freq="B")
    result = pd.DataFrame({"Date": future_dates})
    for i, feat in enumerate(features):
        result[f"Predicted {feat}"] = preds_inv[:, i]
    return result.set_index("Date")


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def arima_forecast(returns: pd.Series,
                   steps: int = 5) -> pd.DataFrame:
    """
    Auto-ARIMA on daily returns with confidence intervals.
    Returns DataFrame: forecast, lower_ci, upper_ci.
    """
    try:
        from pmdarima import auto_arima
        model = auto_arima(returns.dropna(), seasonal=False,
                           suppress_warnings=True, error_action="ignore",
                           stepwise=True)
        fc, ci = model.predict(n_periods=steps, return_conf_int=True)
        last   = returns.dropna().index[-1]
        dates  = pd.date_range(last + timedelta(days=1), periods=steps, freq="B")
        return pd.DataFrame({"forecast": fc,
                             "lower_ci": ci[:, 0],
                             "upper_ci": ci[:, 1]}, index=dates)
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        return pd.DataFrame()


# ── GARCH ─────────────────────────────────────────────────────────────────────

def garch_forecast(returns: pd.Series,
                   steps: int = 5) -> pd.DataFrame:
    """
    GARCH(1,1) volatility forecast.
    Returns DataFrame: vol_forecast (annualised %).
    """
    try:
        from arch import arch_model
        scaled = returns.dropna() * 100   # arch works better in % scale
        am     = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal")
        res    = am.fit(disp="off")
        fc     = res.forecast(horizon=steps)
        var    = fc.variance.iloc[-1].values
        vol    = np.sqrt(var) / 100 * np.sqrt(252)   # annualised
        last   = returns.dropna().index[-1]
        dates  = pd.date_range(last + timedelta(days=1), periods=steps, freq="B")
        return pd.DataFrame({"vol_forecast": vol}, index=dates)
    except Exception as e:
        logger.warning(f"GARCH failed: {e}")
        return pd.DataFrame()
