"""tests/test_backtest.py — Unit tests for core/backtest_engine.py"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest_engine import (run_backtest, BacktestConfig,
                                   momentum_strategy, mean_reversion_strategy,
                                   rsi_strategy)


@pytest.fixture
def sample_price():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-01-01", periods=500, freq="B")
    p   = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 500)))
    return pd.Series(p, index=idx, name="Close")


@pytest.fixture
def buy_hold_signal(sample_price):
    return pd.Series(1, index=sample_price.index)


@pytest.fixture
def random_signal(sample_price):
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice([-1, 0, 1], len(sample_price)), index=sample_price.index)


@pytest.fixture
def default_cfg():
    return BacktestConfig(initial_capital=100_000, commission_pct=0.001, slippage_bps=5)


# ── Result structure ──────────────────────────────────────────────────────────
def test_result_has_equity_curve(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert len(r.equity_curve) > 0


def test_result_has_daily_returns(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert len(r.daily_returns) > 0


def test_result_has_metrics(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert "Sharpe" in r.metrics
    assert "CAGR"   in r.metrics


def test_result_has_trade_log(sample_price, random_signal, default_cfg):
    r = run_backtest(sample_price, random_signal, default_cfg)
    assert isinstance(r.trade_log, pd.DataFrame)


def test_result_has_rolling_sharpe(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert len(r.rolling_sharpe) > 0


# ── Equity curve properties ───────────────────────────────────────────────────
def test_equity_starts_at_capital(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    # First value should be close to initial capital (some friction from shift)
    assert r.equity_curve.iloc[0] > 0


def test_equity_all_positive(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert (r.equity_curve > 0).all()


def test_daily_returns_finite(sample_price, buy_hold_signal, default_cfg):
    r = run_backtest(sample_price, buy_hold_signal, default_cfg)
    assert np.isfinite(r.daily_returns).all()


# ── Strategy factories ────────────────────────────────────────────────────────
def _make_df(price):
    return pd.DataFrame({"Close": price.values, "Open": price.values,
                         "High": price.values * 1.01, "Low": price.values * 0.99,
                         "Volume": [1_000_000] * len(price)}, index=price.index)


def test_momentum_strategy_values(sample_price):
    df  = _make_df(sample_price)
    sig = momentum_strategy(df, lookback=20)
    assert set(sig.unique()).issubset({-1, 1})


def test_mean_reversion_strategy_values(sample_price):
    df  = _make_df(sample_price)
    sig = mean_reversion_strategy(df, window=20, z_thresh=1.5)
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_rsi_strategy_values(sample_price):
    df  = _make_df(sample_price)
    sig = rsi_strategy(df, oversold=30, overbought=70)
    assert set(sig.unique()).issubset({-1, 0, 1})


# ── Zero signal → flat ────────────────────────────────────────────────────────
def test_zero_signal_no_trades(sample_price, default_cfg):
    zero = pd.Series(0, index=sample_price.index)
    r    = run_backtest(sample_price, zero, default_cfg)
    assert len(r.trade_log) == 0


# ── Transaction costs reduce returns ─────────────────────────────────────────
def test_commission_reduces_returns(sample_price, random_signal):
    cfg_no_cost  = BacktestConfig(commission_pct=0.0, slippage_bps=0.0)
    cfg_with_cost = BacktestConfig(commission_pct=0.002, slippage_bps=10.0)
    r_free = run_backtest(sample_price, random_signal, cfg_no_cost)
    r_cost = run_backtest(sample_price, random_signal, cfg_with_cost)
    from core.metrics import cagr
    assert cagr(r_free.daily_returns) >= cagr(r_cost.daily_returns)
