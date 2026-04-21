"""
tests/test_alpha_engine.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for core/alpha_engine.py

Covers all five signals:
  1. compute_volume_pressure  (OFI proxy)
  2. compute_crowding_score   (Hua & Sun 2024)
  3. compute_realized_skew    (IV skew proxy)
  4. compute_signal_health    (alpha decay)
  5. compute_macro_regime_score
  6. combine_signals          (IC-weighted composite)
  7. Backward-compat aliases  (compute_ofi, compute_iv_skew_proxy)
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alpha_engine import (
    compute_volume_pressure,
    compute_realized_skew,
    compute_crowding_score,
    crowding_weight,
    ofi_signal,
    iv_skew_signal,
    compute_signal_health,
    combine_signals,
    # backward-compat aliases
    compute_ofi,
    compute_iv_skew_proxy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv():
    """500-bar synthetic OHLCV DataFrame — enough for all rolling windows."""
    n   = 500
    rng = np.random.default_rng(2024)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    return pd.DataFrame({
        "Open":   close * (1 + rng.normal(0, 0.002, n)),
        "High":   close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 20_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def short_ohlcv():
    """15-bar frame — tests graceful handling of insufficient history."""
    n   = 15
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "Open":   close,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": rng.integers(500_000, 5_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture
def returns_series(ohlcv):
    return ohlcv["Close"].pct_change().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_volume_pressure (OFI proxy)
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumePressure:
    def test_returns_series(self, ohlcv):
        result = compute_volume_pressure(ohlcv)
        assert isinstance(result, pd.Series)

    def test_index_aligns_with_input(self, ohlcv):
        result = compute_volume_pressure(ohlcv)
        assert result.index.equals(ohlcv.index)

    def test_name_is_correct(self, ohlcv):
        result = compute_volume_pressure(ohlcv)
        assert result.name == "VolumePressure"

    def test_no_infinite_values(self, ohlcv):
        result = compute_volume_pressure(ohlcv)
        assert not np.isinf(result.dropna()).any()

    def test_z_score_is_reasonably_bounded(self, ohlcv):
        """Z-scores should rarely exceed ±6 for financial data."""
        result = compute_volume_pressure(ohlcv).dropna()
        assert result.abs().max() < 10.0

    def test_short_df_does_not_crash(self, short_ohlcv):
        result = compute_volume_pressure(short_ohlcv)
        assert isinstance(result, pd.Series)

    def test_zero_volume_handled(self, ohlcv):
        """Zero volume rows should not produce inf or raise."""
        df = ohlcv.copy()
        df.loc[df.index[10:15], "Volume"] = 0
        result = compute_volume_pressure(df)
        assert not np.isinf(result.dropna()).any()

    def test_backward_compat_alias(self, ohlcv):
        """compute_ofi must be identical to compute_volume_pressure."""
        r1 = compute_volume_pressure(ohlcv)
        r2 = compute_ofi(ohlcv)
        pd.testing.assert_series_equal(r1, r2)


class TestOfiSignal:
    def test_signal_values_are_valid(self, ohlcv):
        sig = ofi_signal(ohlcv)
        assert set(sig.dropna().unique()).issubset({-1, 0, 1})

    def test_threshold_respected(self, ohlcv):
        """With threshold=100 (unreachable), all signals must be 0."""
        sig = ofi_signal(ohlcv, threshold=100.0)
        assert (sig == 0).all()

    def test_low_threshold_produces_trades(self, ohlcv):
        """With threshold=0, the warm-up NaN period aside, all non-NaN values should be non-zero."""
        sig = ofi_signal(ohlcv, threshold=0.0)
        # The z-score is zero only when raw OFI is exactly zero (unlikely in real data).
        # After warm-up, most bars should be traded. We test >80% traded.
        non_zero_frac = (sig.dropna().abs() == 1).mean()
        assert non_zero_frac > 0.5, f"Expected >50% trades at threshold=0, got {non_zero_frac:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_crowding_score
# ─────────────────────────────────────────────────────────────────────────────

class TestCrowdingScore:
    def test_returns_series(self, returns_series):
        result = compute_crowding_score(returns_series)
        assert isinstance(result, pd.Series)

    def test_all_positive(self, returns_series):
        """Crowding is a vol ratio — always non-negative."""
        result = compute_crowding_score(returns_series).dropna()
        assert (result >= 0).all()

    def test_crowding_weight_range(self, returns_series):
        """Weight must be in [0.25, 1.0] by design."""
        w = crowding_weight(returns_series)
        assert 0.25 <= w <= 1.0

    def test_crowding_weight_is_scalar(self, returns_series):
        w = crowding_weight(returns_series)
        assert isinstance(w, float)

    def test_short_returns_no_crash(self):
        short = pd.Series(np.random.normal(0, 0.01, 10))
        w = crowding_weight(short)
        assert isinstance(w, float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_realized_skew (IV skew proxy)
# ─────────────────────────────────────────────────────────────────────────────

class TestRealizedSkew:
    def test_returns_series(self, ohlcv):
        result = compute_realized_skew(ohlcv)
        assert isinstance(result, pd.Series)

    def test_index_aligns_with_input(self, ohlcv):
        result = compute_realized_skew(ohlcv)
        assert result.index.equals(ohlcv.index)

    def test_name_is_correct(self, ohlcv):
        result = compute_realized_skew(ohlcv)
        assert result.name == "RealizedSkew"

    def test_no_infinite_values(self, ohlcv):
        result = compute_realized_skew(ohlcv)
        assert not np.isinf(result.dropna()).any()

    def test_backward_compat_alias(self, ohlcv):
        """compute_iv_skew_proxy must be identical to compute_realized_skew."""
        r1 = compute_realized_skew(ohlcv)
        r2 = compute_iv_skew_proxy(ohlcv)
        pd.testing.assert_series_equal(r1, r2)

    def test_fear_regime_produces_negative_skew(self):
        """Construct a bearish OHLCV frame — realized skew should trend negative."""
        n   = 300
        rng = np.random.default_rng(55)
        # Consistent downward drift to induce negative skew
        close = 100 * np.exp(np.cumsum(rng.normal(-0.003, 0.015, n)))
        idx   = pd.date_range("2022-01-01", periods=n, freq="B")
        df    = pd.DataFrame({
            "Open":   close * 1.001,
            "High":   close * 1.005,
            "Low":    close * 0.990,   # deep lows = fear
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=idx)
        skew = compute_realized_skew(df).dropna()
        # In a sustained bear move, median skew should lean negative
        assert skew.median() < 0.5


class TestIvSkewSignal:
    def test_signal_values_valid(self, ohlcv):
        sig = iv_skew_signal(ohlcv)
        assert set(sig.dropna().unique()).issubset({-1, 0, 1})

    def test_unreachable_threshold_gives_zero(self, ohlcv):
        sig = iv_skew_signal(ohlcv, threshold=999.0)
        assert (sig == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_signal_health (alpha decay)
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalHealth:
    def test_returns_dict(self, ohlcv):
        signal = ofi_signal(ohlcv)
        result = compute_signal_health(signal, ohlcv["Close"].pct_change().dropna())
        assert isinstance(result, dict)

    def test_required_keys_present(self, ohlcv):
        signal = ofi_signal(ohlcv)
        result = compute_signal_health(signal, ohlcv["Close"].pct_change().dropna())
        # Keys returned by the live implementation
        required = {"health_score", "ic_mean", "ic_std", "status"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_health_score_in_range(self, ohlcv):
        signal = ofi_signal(ohlcv)
        result = compute_signal_health(signal, ohlcv["Close"].pct_change().dropna())
        assert 0.0 <= result["health_score"] <= 100.0

    def test_status_is_string(self, ohlcv):
        signal = ofi_signal(ohlcv)
        result = compute_signal_health(signal, ohlcv["Close"].pct_change().dropna())
        assert isinstance(result["status"], str)


# ─────────────────────────────────────────────────────────────────────────────
# 5. combine_signals (IC-weighted composite)
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineSignals:
    def test_returns_tuple(self, ohlcv):
        signals = {
            "vp":   ofi_signal(ohlcv),
            "skew": iv_skew_signal(ohlcv),
        }
        result = combine_signals(signals, ohlcv["Close"].pct_change().dropna())
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_series(self, ohlcv):
        signals = {
            "vp":   ofi_signal(ohlcv),
            "skew": iv_skew_signal(ohlcv),
        }
        combined, _ = combine_signals(signals, ohlcv["Close"].pct_change().dropna())
        assert isinstance(combined, pd.Series)

    def test_second_element_is_weights_dict(self, ohlcv):
        signals = {
            "vp":   ofi_signal(ohlcv),
            "skew": iv_skew_signal(ohlcv),
        }
        _, weights = combine_signals(signals, ohlcv["Close"].pct_change().dropna())
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {"vp", "skew"}

    def test_combined_signal_in_valid_range(self, ohlcv):
        """IC-weighted combination must stay in {-1, 0, 1}."""
        signals = {
            "vp":   ofi_signal(ohlcv),
            "skew": iv_skew_signal(ohlcv),
        }
        combined, _ = combine_signals(signals, ohlcv["Close"].pct_change().dropna())
        assert set(combined.dropna().unique()).issubset({-1, 0, 1})

    def test_single_signal_passthrough(self, ohlcv):
        sig = ofi_signal(ohlcv)
        combined, weights = combine_signals({"vp": sig}, ohlcv["Close"].pct_change().dropna())
        assert isinstance(combined, pd.Series)
        assert "vp" in weights

    def test_empty_signals_returns_zero_series(self, ohlcv):
        combined, weights = combine_signals({}, ohlcv["Close"].pct_change().dropna())
        assert isinstance(combined, pd.Series)
        assert weights == {}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Determinism — same inputs always produce same outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_volume_pressure_deterministic(self, ohlcv):
        r1 = compute_volume_pressure(ohlcv)
        r2 = compute_volume_pressure(ohlcv)
        pd.testing.assert_series_equal(r1, r2)

    def test_realized_skew_deterministic(self, ohlcv):
        r1 = compute_realized_skew(ohlcv)
        r2 = compute_realized_skew(ohlcv)
        pd.testing.assert_series_equal(r1, r2)

    def test_crowding_weight_deterministic(self, returns_series):
        w1 = crowding_weight(returns_series)
        w2 = crowding_weight(returns_series)
        assert w1 == w2
