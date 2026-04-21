"""
core/metrics.py
Quant metrics library: Sharpe, Sortino, Calmar, VaR, CVaR, CAGR, IC, ICIR.
Enhanced with: fat-tail VaR (Student-t), GARCH(1,1) VaR, portfolio VaR,
Kupiec backtest, and dynamic risk-free rate support.
Imported by backtest, portfolio, risk, and factor pages.
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def _clean_series(values: pd.Series) -> pd.Series:
    """Return a numeric Series with NaNs removed."""
    if isinstance(values, pd.Series):
        return values.dropna()
    return pd.Series(values, dtype=float).dropna()


def _clip_ratio(value: float) -> float:
    """Keep display-facing ratios finite and bounded."""
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, -999.0, 999.0))


def sharpe(returns: pd.Series, rf: float | None = None) -> float:
    """Annualised Sharpe ratio. Uses cfg.RISK_FREE_RATE if rf not provided."""
    if rf is None:
        try:
            from utils.config import cfg
            rf = cfg.RISK_FREE_RATE
        except Exception:
            rf = 0.045
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    std = excess.std()
    if std == 0:
        ratio = float("inf") if excess.mean() > 0 else (float("-inf") if excess.mean() < 0 else 0.0)
        return _clip_ratio(ratio)
    return _clip_ratio(float(excess.mean() / std * np.sqrt(TRADING_DAYS)))


def sortino(returns: pd.Series, rf: float | None = None) -> float:
    """Annualised Sortino ratio (downside deviation only). Uses cfg.RISK_FREE_RATE if rf not provided."""
    if rf is None:
        try:
            from utils.config import cfg
            rf = cfg.RISK_FREE_RATE
        except Exception:
            rf = 0.045
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        ratio = float("inf") if excess.mean() > 0 else 0.0
        return _clip_ratio(ratio)
    return _clip_ratio(float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS)))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def calmar(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        growth = cagr(returns)
        ratio = float("inf") if growth > 0 else (float("-inf") if growth < 0 else 0.0)
        return _clip_ratio(ratio)
    return _clip_ratio(float(cagr(returns) / mdd))


def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    n_years = len(returns) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    total = (1 + returns).prod()
    return float(total ** (1 / n_years) - 1)


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk at given confidence level (negative number)."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall (mean of tail losses)."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    cutoff = var_historical(returns, confidence)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    mu, sigma = returns.mean(), returns.std()
    return float(stats.norm.ppf(1 - confidence, mu, sigma))


def var_t_dist(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Fat-tail VaR using Student-t distribution.
    Fits degrees-of-freedom from data — captures tail events much better
    than Gaussian, especially for equities with kurtosis > 3.
    """
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    df_t, loc, scale = stats.t.fit(returns)
    return float(stats.t.ppf(1 - confidence, df_t, loc, scale))


def var_garch(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """
    GARCH(1,1) one-step-ahead VaR.
    Accounts for volatility clustering — when vol is high, next-day VaR is
    higher. Much more responsive than rolling-window methods.
    Returns daily VaR scaled to given horizon.
    """
    returns = _clean_series(returns)
    if len(returns) < 100:
        return var_historical(returns, confidence)
    try:
        from arch import arch_model
        scaled = returns * 100  # arch works better on percentage returns
        model = arch_model(scaled, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=horizon, reindex=False)
        cond_vol = float(np.sqrt(forecast.variance.values[-1, 0])) / 100
        nu = float(res.params.get("nu", 8))
        z = float(stats.t.ppf(1 - confidence, nu))
        mu = float(returns.mean())
        var = mu + z * cond_vol
        return float(var * np.sqrt(horizon))
    except Exception:
        return var_historical(returns, confidence)


def portfolio_var(
    returns_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Portfolio-level VaR using covariance structure.
    method: 'historical' | 'parametric' | 'garch'
    weights: array of portfolio weights (equal-weighted if None).
    Accounts for cross-asset correlations — essential for real risk mgmt.
    """
    returns_df = returns_df.dropna()
    if returns_df.empty:
        return 0.0
    n = returns_df.shape[1]
    if weights is None:
        weights = np.ones(n) / n
    weights = np.array(weights)
    port_ret = returns_df.values @ weights

    if method == "garch":
        return var_garch(pd.Series(port_ret), confidence)
    if method == "parametric":
        mu = port_ret.mean()
        cov = returns_df.cov().values
        port_vol = float(np.sqrt(weights @ cov @ weights))
        return float(stats.norm.ppf(1 - confidence, mu, port_vol))
    return float(np.percentile(port_ret, (1 - confidence) * 100))


def kupiec_test(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> dict:
    """
    Kupiec Proportion of Failures (POF) backtest.
    Tests if your VaR model is actually accurate:
    - counts how often real losses exceeded VaR
    - compares to expected rate (1 - confidence)
    - p_value > 0.05 means the model is statistically valid
    """
    aligned = returns.align(var_series, join="inner")
    ret_a, var_a = aligned[0].dropna(), aligned[1].dropna()
    common = ret_a.index.intersection(var_a.index)
    ret_a, var_a = ret_a[common], var_a[common]
    if len(ret_a) < 30:
        return {"violations": 0, "expected_rate": 1 - confidence,
                "actual_rate": float("nan"), "p_value": float("nan"), "result": "Insufficient data"}
    violations = int((ret_a < var_a).sum())
    T = len(ret_a)
    p_expected = 1 - confidence
    p_actual = violations / T
    if violations == 0 or violations == T:
        return {"violations": violations, "expected_rate": p_expected,
                "actual_rate": p_actual, "p_value": float("nan"), "result": "Edge case — check data"}
    lr = -2 * (
        np.log(p_expected**violations * (1 - p_expected)**(T - violations))
        - np.log(p_actual**violations * (1 - p_actual)**(T - violations))
    )
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    result = "✅ Valid model" if p_value > 0.05 else "❌ Model underestimates risk"
    return {
        "violations": violations,
        "expected_rate": p_expected,
        "actual_rate": round(p_actual, 4),
        "p_value": round(p_value, 4),
        "result": result,
    }


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive return days."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def annualised_vol(returns: pd.Series) -> float:
    """Annualised volatility."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def annualised_return(returns: pd.Series) -> float:
    """Annualised arithmetic mean return."""
    returns = _clean_series(returns)
    if returns.empty:
        return 0.0
    return float(returns.mean() * TRADING_DAYS)


def information_coefficient(factor_scores: pd.Series,
                             fwd_returns: pd.Series) -> float:
    """
    Spearman rank correlation between factor scores and forward returns.
    IC > 0.05 is considered meaningful signal.
    """
    common = factor_scores.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < 5:
        return 0.0
    ic, _ = stats.spearmanr(factor_scores[common], fwd_returns[common])
    return 0.0 if pd.isna(ic) else float(ic)


def rolling_ic(factor_scores: pd.Series, fwd_returns: pd.Series,
               window: int = 63) -> pd.Series:
    """Rolling Information Coefficient over a given window."""
    ics = []
    idx = []
    common = factor_scores.dropna().index.intersection(fwd_returns.dropna().index)
    common = sorted(common)
    for i in range(window, len(common)):
        w = common[i - window: i]
        ic = information_coefficient(factor_scores[w], fwd_returns[w])
        ics.append(ic)
        idx.append(common[i])
    return pd.Series(ics, index=idx, name="IC")


def icir(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Reliability > 0.5."""
    ic_series = _clean_series(ic_series)
    if ic_series.empty:
        return 0.0
    std = ic_series.std()
    if std < 1e-10:
        mean_ic = ic_series.mean()
        ratio = float("inf") if mean_ic > 0 else (float("-inf") if mean_ic < 0 else 0.0)
        return _clip_ratio(ratio)
    return _clip_ratio(float(ic_series.mean() / std))


def summary_table(returns: pd.Series, rf: float | None = None) -> dict:
    """Full metric summary dict for display in Streamlit."""
    if rf is None:
        try:
            from utils.config import cfg
            rf = cfg.RISK_FREE_RATE
        except Exception:
            rf = 0.045
    returns = _clean_series(returns)
    metric_keys = [
        "CAGR", "Ann. Return", "Ann. Volatility", "Sharpe", "Sortino", "Calmar",
        "Max Drawdown", "Win Rate",
        "VaR 95% (Hist)", "CVaR 95% (Hist)", "VaR 95% (t-dist)", "VaR 95% (GARCH)",
        "VaR 99%", "CVaR 99%",
    ]
    if len(returns) < 2:
        return {key: "N/A" for key in metric_keys}
    return {
        "CAGR":                 f"{cagr(returns):.2%}",
        "Ann. Return":          f"{annualised_return(returns):.2%}",
        "Ann. Volatility":      f"{annualised_vol(returns):.2%}",
        "Sharpe":               f"{sharpe(returns, rf):.2f}",
        "Sortino":              f"{sortino(returns, rf):.2f}",
        "Calmar":               f"{calmar(returns):.2f}",
        "Max Drawdown":         f"{max_drawdown(returns):.2%}",
        "Win Rate":             f"{win_rate(returns):.2%}",
        "VaR 95% (Hist)":       f"{var_historical(returns, 0.95):.2%}",
        "CVaR 95% (Hist)":      f"{cvar_historical(returns, 0.95):.2%}",
        "VaR 95% (t-dist)":     f"{var_t_dist(returns, 0.95):.2%}",
        "VaR 95% (GARCH)":      f"{var_garch(returns, 0.95):.2%}",
        "VaR 99%":              f"{var_historical(returns, 0.99):.2%}",
        "CVaR 99%":             f"{cvar_historical(returns, 0.99):.2%}",
    }