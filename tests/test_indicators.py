"""tests/test_indicators.py — Unit tests for core/indicators.py"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.indicators import (rsi, macd, bollinger_bands, atr,
                              momentum, add_all_indicators,
                              signal_rsi, signal_macd_crossover,
                              signal_bb_mean_reversion, signal_dual_ma)


@pytest.fixture
def sample_ohlcv():
    n   = 300
    rng = np.random.default_rng(99)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame({
        "Open":   close * (1 + rng.normal(0, 0.002, n)),
        "High":   close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def close(sample_ohlcv):
    return sample_ohlcv["Close"]


# ── RSI ───────────────────────────────────────────────────────────────────────
def test_rsi_range(close):
    r = rsi(close)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"


def test_rsi_length(close):
    r = rsi(close, period=14)
    assert len(r) == len(close), "RSI must have same length as input"


def test_rsi_no_all_nan(close):
    r = rsi(close)
    assert r.notna().sum() > 0


# ── MACD ──────────────────────────────────────────────────────────────────────
def test_macd_columns(close):
    m = macd(close)
    assert set(m.columns) == {"MACD", "Signal", "Histogram"}


def test_macd_histogram_equals_diff(close):
    m = macd(close).dropna()
    diff = (m["MACD"] - m["Signal"]).round(8)
    hist = m["Histogram"].round(8)
    pd.testing.assert_series_equal(diff, hist, check_names=False)


def test_macd_length(close):
    m = macd(close)
    assert len(m) == len(close)


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def test_bb_upper_gt_mid(close):
    bb = bollinger_bands(close).dropna()
    assert (bb["BB_Upper"] >= bb["BB_Mid"]).all()


def test_bb_lower_lt_mid(close):
    bb = bollinger_bands(close).dropna()
    assert (bb["BB_Lower"] <= bb["BB_Mid"]).all()


def test_bb_pct_b_range(close):
    bb = bollinger_bands(close).dropna()
    # %B can go outside [0,1] during breakouts — just check it's finite
    assert np.isfinite(bb["BB_PctB"]).all()


def test_bb_columns(close):
    bb = bollinger_bands(close)
    assert {"BB_Mid", "BB_Upper", "BB_Lower", "BB_PctB", "BB_BandWidth"}.issubset(bb.columns)


# ── ATR ───────────────────────────────────────────────────────────────────────
def test_atr_positive(sample_ohlcv):
    a = atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
    assert (a.dropna() > 0).all(), "ATR must be positive"


def test_atr_length(sample_ohlcv):
    a = atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
    assert len(a) == len(sample_ohlcv)


# ── Momentum ──────────────────────────────────────────────────────────────────
def test_momentum_not_all_nan(close):
    m = momentum(close, period=1)
    assert m.notna().sum() > 0


# ── add_all_indicators ────────────────────────────────────────────────────────
def test_add_all_indicators_columns(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    expected = ["RSI", "MACD", "Signal", "Histogram",
                "BB_Mid", "BB_Upper", "BB_Lower", "ATR"]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_add_all_indicators_no_shape_change(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    assert len(df) == len(sample_ohlcv)


# ── Signals ───────────────────────────────────────────────────────────────────
def test_signal_rsi_values(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    sig = signal_rsi(df)
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_signal_macd_values(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    sig = signal_macd_crossover(df)
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_signal_bb_values(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    sig = signal_bb_mean_reversion(df)
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_signal_dual_ma_values(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    sig = signal_dual_ma(df, fast=10, slow=30)
    assert set(sig.unique()).issubset({-1, 0, 1})
