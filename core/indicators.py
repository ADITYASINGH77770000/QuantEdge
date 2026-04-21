"""
core/indicators.py
──────────────────────────────────────────────────────────────────────────────
Technical indicators — vectorised, pandas-native, no loops.
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


def macd(close: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = (ema_fast - ema_slow).rename("MACD")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("Signal")
    hist        = (macd_line - signal_line).rename("Histogram")
    return pd.concat([macd_line, signal_line, hist], axis=1)


def bollinger_bands(close: pd.Series,
                    period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle, upper, lower, %B, bandwidth."""
    mid   = close.rolling(period).mean().rename("BB_Mid")
    sigma = close.rolling(period).std()
    upper = (mid + std_dev * sigma).rename("BB_Upper")
    lower = (mid - std_dev * sigma).rename("BB_Lower")
    pct_b = ((close - lower) / (upper - lower)).rename("BB_PctB")
    bw    = ((upper - lower) / mid).rename("BB_BandWidth")
    return pd.concat([mid, upper, lower, pct_b, bw], axis=1)


def atr(high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean().rename("ATR")


def momentum(close: pd.Series, period: int = 12) -> pd.Series:
    """12-month (or n-month) price momentum: return over last `period` months."""
    return close.pct_change(period * 21).rename(f"Momentum_{period}m")


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append RSI, MACD, Bollinger Bands, and ATR to an OHLCV DataFrame.
    Returns a copy with extra columns.
    """
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    _macd = macd(df["Close"])
    df = pd.concat([df, _macd], axis=1)
    _bb = bollinger_bands(df["Close"])
    df = pd.concat([df, _bb], axis=1)
    df["ATR"] = atr(df["High"], df["Low"], df["Close"])
    df["Momentum_1m"] = momentum(df["Close"], 1)
    df["Momentum_3m"] = momentum(df["Close"], 3)
    df["Momentum_12m"] = momentum(df["Close"], 12)
    return df


def signal_rsi(df: pd.DataFrame,
               oversold: int = 30, overbought: int = 70) -> pd.Series:
    """RSI-based signal: +1 buy, -1 sell, 0 hold."""
    sig = pd.Series(0, index=df.index)
    sig[df["RSI"] < oversold]   =  1
    sig[df["RSI"] > overbought] = -1
    return sig.rename("RSI_Signal")


def signal_macd_crossover(df: pd.DataFrame) -> pd.Series:
    """MACD crossover signal: +1 when MACD crosses above Signal."""
    cross_up   = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1))
    cross_down = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig.rename("MACD_Signal")


def signal_bb_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band mean-reversion: buy at lower band, sell at upper."""
    sig = pd.Series(0, index=df.index)
    sig[df["Close"] < df["BB_Lower"]] =  1
    sig[df["Close"] > df["BB_Upper"]] = -1
    return sig.rename("BB_Signal")


def signal_dual_ma(df: pd.DataFrame,
                   fast: int = 20, slow: int = 50) -> pd.Series:
    """Dual moving-average crossover momentum signal."""
    ma_fast = df["Close"].rolling(fast).mean()
    ma_slow = df["Close"].rolling(slow).mean()
    cross_up   = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig.rename(f"DualMA_{fast}_{slow}")
