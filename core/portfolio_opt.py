"""
core/portfolio_opt.py
──────────────────────────────────────────────────────────────────────────────
Portfolio construction: mean-variance, risk parity, CVaR optimisation.

Functions
---------
monte_carlo_frontier      — random-sampling frontier (fast visualisation overlay)
analytical_frontier       — scipy-optimised precise efficient frontier
efficient_frontier        — combined: analytical curve + MC cloud + key portfolios
risk_parity_weights       — Equal Risk Contribution (Maillard et al. 2010)
portfolio_stats           — return / vol / Sharpe for any weight vector
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

TRADING_DAYS = 252


def _annualised_moments(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu  = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values  * TRADING_DAYS
    return mu, cov


def _port_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def _port_ret(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _port_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    v = _port_vol(w, cov)
    return (_port_ret(w, mu) - rf) / v if v > 1e-12 else 0.0


def _base_constraints(n: int) -> list[dict]:
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _bounds(n: int) -> list[tuple]:
    return [(0.0, 1.0)] * n


# ── 1. Monte Carlo frontier ───────────────────────────────────────────────────

def monte_carlo_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 1_000,
    rf: float = 0.045,
) -> dict:
    """
    Efficient frontier via Monte Carlo random portfolio sampling.
    Returns approximate max-Sharpe and min-vol portfolios plus the full cloud.
    Use analytical_frontier() for precise optimised portfolios.
    """
    mu, cov = _annualised_moments(returns)
    n       = len(mu)
    rng     = np.random.default_rng(42)

    all_weights, all_rets, all_vols, all_sharpes = [], [], [], []

    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        r = _port_ret(w, mu)
        v = _port_vol(w, cov)
        s = (r - rf) / v if v > 1e-12 else 0.0
        all_weights.append(w)
        all_rets.append(r)
        all_vols.append(v)
        all_sharpes.append(s)

    all_weights = np.array(all_weights)
    all_rets    = np.array(all_rets)
    all_vols    = np.array(all_vols)
    all_sharpes = np.array(all_sharpes)

    max_sh_idx  = all_sharpes.argmax()
    min_vol_idx = all_vols.argmin()

    return {
        "weights":  all_weights,
        "returns":  all_rets,
        "vols":     all_vols,
        "sharpes":  all_sharpes,
        "max_sharpe": {
            "weights": all_weights[max_sh_idx],
            "ret":     all_rets[max_sh_idx],
            "vol":     all_vols[max_sh_idx],
            "sharpe":  all_sharpes[max_sh_idx],
        },
        "min_vol": {
            "weights": all_weights[min_vol_idx],
            "ret":     all_rets[min_vol_idx],
            "vol":     all_vols[min_vol_idx],
            "sharpe":  all_sharpes[min_vol_idx],
        },
        "tickers": list(returns.columns),
    }


# ── 2. Analytical efficient frontier ─────────────────────────────────────────

def _solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    n  = len(mu)
    res = minimize(
        lambda w: -_port_sharpe(w, mu, cov, rf),
        np.ones(n) / n,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    w = res.x
    return w / w.sum()


def _solve_min_vol(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n  = len(mu)
    res = minimize(
        lambda w: _port_vol(w, cov),
        np.ones(n) / n,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    w = res.x
    return w / w.sum()


def _solve_target_return(
    mu: np.ndarray, cov: np.ndarray, target: float
) -> np.ndarray | None:
    n = len(mu)
    constraints = _base_constraints(n) + [
        {"type": "eq", "fun": lambda w, t=target: _port_ret(w, mu) - t}
    ]
    res = minimize(
        lambda w: _port_vol(w, cov),
        np.ones(n) / n,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1_000},
    )
    if not res.success:
        return None
    w = res.x
    return w / w.sum()


def analytical_frontier(
    returns: pd.DataFrame,
    rf: float = 0.045,
    n_points: int = 60,
) -> dict:
    """
    Precise mean-variance efficient frontier via scipy optimisation.

    For each target return along the feasible range, finds the minimum-variance
    portfolio using SLSQP. Max-Sharpe and min-vol portfolios are solved directly
    (not read from random samples) so they are exact to numerical tolerance.

    This replaces the MC-only approach where the 'best' portfolio was whichever
    random sample happened to land nearest the true optimum.

    Parameters
    ----------
    returns   : daily returns DataFrame (tickers as columns)
    rf        : risk-free rate (annualised)
    n_points  : resolution of the frontier curve

    Returns
    -------
    dict with keys: frontier_vols, frontier_rets, frontier_sharpes,
                    max_sharpe, min_vol, tickers, method="analytical"
    """
    mu, cov = _annualised_moments(returns)
    tickers = list(returns.columns)

    ms_w  = _solve_max_sharpe(mu, cov, rf)
    mv_w  = _solve_min_vol(mu, cov)

    def _pack(w: np.ndarray) -> dict:
        return {
            "weights": w,
            "ret":     _port_ret(w, mu),
            "vol":     _port_vol(w, cov),
            "sharpe":  _port_sharpe(w, mu, cov, rf),
        }

    mv_ret   = _port_ret(mv_w, mu)
    ret_hi   = float(mu.max()) * 0.98
    targets  = np.linspace(mv_ret, ret_hi, n_points)

    f_vols, f_rets, f_sharpes = [], [], []
    for tr in targets:
        w = _solve_target_return(mu, cov, tr)
        if w is None:
            continue
        f_vols.append(_port_vol(w, cov))
        f_rets.append(tr)
        f_sharpes.append(_port_sharpe(w, mu, cov, rf))

    return {
        "frontier_vols":    np.array(f_vols),
        "frontier_rets":    np.array(f_rets),
        "frontier_sharpes": np.array(f_sharpes),
        "max_sharpe":       _pack(ms_w),
        "min_vol":          _pack(mv_w),
        "tickers":          tickers,
        "method":           "analytical",
    }


def efficient_frontier(
    returns: pd.DataFrame,
    rf: float = 0.045,
    n_mc: int = 1_000,
    n_frontier_points: int = 60,
) -> dict:
    """
    Combined frontier: precise analytical curve + MC feasibility cloud.
    The analytical frontier provides the key portfolios; MC provides the
    scatter background that shows the full feasible region.
    """
    anal = analytical_frontier(returns, rf=rf, n_points=n_frontier_points)
    mc   = monte_carlo_frontier(returns, n_portfolios=n_mc, rf=rf)
    return {
        **anal,
        "mc_vols":    mc["vols"],
        "mc_rets":    mc["returns"],
        "mc_sharpes": mc["sharpes"],
    }


# ── 3. Risk Parity ────────────────────────────────────────────────────────────

def risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Equal Risk Contribution portfolio (Maillard, Roncalli & Teïletche 2010).
    Each asset contributes equally to total portfolio volatility.
    """
    cov = returns.cov().values
    n   = len(returns.columns)

    def objective(w: np.ndarray) -> float:
        sigma = np.sqrt(w @ cov @ w)
        mrc   = cov @ w / sigma
        rc    = w * mrc
        total = rc.sum() + 1e-12
        target = np.full(n, 1.0 / n)
        return float(np.sum((rc / total - target) ** 2))

    res = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.01, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    w = res.x
    return w / w.sum()


# ── 4. Portfolio statistics ───────────────────────────────────────────────────

def portfolio_stats(
    weights: np.ndarray,
    returns: pd.DataFrame,
    rf: float = 0.045,
) -> dict:
    """Annualised return, volatility, and Sharpe for a given weight vector."""
    mu, cov = _annualised_moments(returns)
    return {
        "return":     _port_ret(weights, mu),
        "volatility": _port_vol(weights, cov),
        "sharpe":     _port_sharpe(weights, mu, cov, rf),
    }
