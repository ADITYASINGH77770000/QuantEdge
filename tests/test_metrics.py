"""tests/test_metrics.py - Unit tests for core/metrics.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import (
    annualised_vol,
    cagr,
    calmar,
    cvar_historical,
    icir,
    information_coefficient,
    max_drawdown,
    sharpe,
    sortino,
    summary_table,
    var_historical,
    var_parametric,
    win_rate,
)


@pytest.fixture
def flat_returns():
    return pd.Series([0.01] * 252)


@pytest.fixture
def random_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0004, 0.012, 500))


@pytest.fixture
def zero_returns():
    return pd.Series([0.0] * 252)


def test_sharpe_positive_returns(flat_returns):
    assert sharpe(flat_returns, rf=0.0) > 0


def test_sharpe_zero_std(zero_returns):
    # All returns are 0. With rf=0 excess is 0/0 — undefined.
    # The implementation clips this to -999. Either 0 or -999 is
    # acceptable; we just verify the result is finite and bounded.
    result = sharpe(zero_returns)
    assert -999.0 <= result <= 999.0


def test_sharpe_caps_constant_positive_returns(flat_returns):
    assert sharpe(flat_returns) == pytest.approx(999.0)


def test_sharpe_reasonably_valued(random_returns):
    value = sharpe(random_returns)
    assert -10 < value < 10


def test_sortino_gte_sharpe_for_positive_returns(flat_returns):
    assert sortino(flat_returns) >= sharpe(flat_returns)


def test_sortino_zero_std(zero_returns):
    assert sortino(zero_returns) == 0.0


def test_sortino_caps_constant_positive_returns(flat_returns):
    assert sortino(flat_returns) == pytest.approx(999.0)


def test_max_drawdown_is_negative(random_returns):
    assert max_drawdown(random_returns) <= 0


def test_max_drawdown_flat():
    assert max_drawdown(pd.Series([0.0] * 100)) == 0.0


def test_max_drawdown_bounds(random_returns):
    value = max_drawdown(random_returns)
    assert -1.0 <= value <= 0.0


def test_cagr_positive_drift(flat_returns):
    assert cagr(flat_returns) > 0


def test_cagr_zero():
    assert cagr(pd.Series([0.0] * 252)) == pytest.approx(0.0, abs=1e-6)


def test_calmar_caps_constant_positive_returns(flat_returns):
    assert calmar(flat_returns) == pytest.approx(999.0)


def test_var_is_negative(random_returns):
    assert var_historical(random_returns, 0.95) < 0


def test_cvar_lte_var(random_returns):
    var = var_historical(random_returns, 0.95)
    cvar = cvar_historical(random_returns, 0.95)
    assert cvar <= var


def test_var_99_lte_var_95(random_returns):
    v95 = var_historical(random_returns, 0.95)
    v99 = var_historical(random_returns, 0.99)
    assert v99 <= v95


def test_var_parametric_negative(random_returns):
    assert var_parametric(random_returns, 0.95) < 0


def test_win_rate_bounds(random_returns):
    value = win_rate(random_returns)
    assert 0.0 <= value <= 1.0


def test_win_rate_all_positive(flat_returns):
    assert win_rate(flat_returns) == pytest.approx(1.0)


def test_ic_perfect_correlation():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert information_coefficient(series, series) == pytest.approx(1.0, abs=0.01)


def test_ic_negative_correlation():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    neg = pd.Series([5, 4, 3, 2, 1], dtype=float)
    assert information_coefficient(series, neg) == pytest.approx(-1.0, abs=0.01)


def test_ic_bounds(random_returns):
    scores = random_returns.reset_index(drop=True)
    fwd = scores.shift(1).dropna()
    shifted = scores.iloc[1:].reset_index(drop=True)
    value = information_coefficient(shifted, fwd.reset_index(drop=True))
    assert -1.0 <= value <= 1.0


def test_icir_consistent_positive_ic_caps():
    assert icir(pd.Series([0.05] * 20)) == pytest.approx(999.0)


def test_icir_zero_mean_constant_series():
    assert icir(pd.Series([0.0] * 10)) == 0.0


def test_summary_table_keys(random_returns):
    table = summary_table(random_returns)
    # Key names match the live summary_table() in core/metrics.py
    required = [
        "CAGR", "Sharpe", "Sortino", "Calmar", "Max Drawdown", "Win Rate",
        "VaR 95% (Hist)", "CVaR 95% (Hist)",
    ]
    for key in required:
        assert key in table, f"Missing key '{key}' in summary_table output: {list(table.keys())}"


def test_summary_table_values_are_strings(random_returns):
    table = summary_table(random_returns)
    assert all(isinstance(value, str) for value in table.values())


def test_summary_table_empty_returns_is_na():
    table = summary_table(pd.Series(dtype=float))
    assert all(value == "N/A" for value in table.values())


def test_empty_metric_functions_return_zero():
    empty = pd.Series(dtype=float)
    assert max_drawdown(empty) == 0.0
    assert calmar(empty) == 0.0
    assert cagr(empty) == 0.0
    assert var_historical(empty) == 0.0
    assert cvar_historical(empty) == 0.0
    assert var_parametric(empty) == 0.0
    assert win_rate(empty) == 0.0
    assert annualised_vol(empty) == 0.0
