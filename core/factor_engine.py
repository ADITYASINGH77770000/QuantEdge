"""
core/factor_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Professional 8-fix Factor Engine — all fixes in one file.

FIX 1  Honest factor proxies        — momentum corrected; value/quality use
                                       the best OHLCV-computable substitutes
                                       with clear disclosure of limitations.
FIX 2  Time-series IC (proper)      — IC computed at every rebalance date
                                       across full history, not one snapshot.
FIX 3  Cost-adjusted quintile BT    — net-of-cost spread shown alongside gross.
FIX 4  Regime-conditioned factors   — factors scored separately inside
                                       Bull / Sideways / Bear regimes.
FIX 5  IC-weighted composite score  — dynamic weighting by rolling IC strength
                                       (Grinold & Kahn 1999).
FIX 6  Factor attribution / Carhart — strategy return decomposed into alpha +
                                       beta to each factor (Carhart 1997).
FIX 7  Crowding detection           — cross-sectional score dispersion collapse
                                       signals crowded factor (Khandani & Lo 2007).
FIX 8  Cross-sectional decay curve  — IC at each horizon computed properly
                                       across all tickers simultaneously.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from core import metrics as m
from core.backtest_engine import detect_regime   # single source of truth for regime classification


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _close(prices: dict, ticker: str) -> pd.Series:
    return prices[ticker]["Close"].dropna()


def _ret(prices: dict, ticker: str) -> pd.Series:
    return _close(prices, ticker).pct_change().dropna()


def _cs_ic(scores: pd.Series, fwd: pd.Series) -> float:
    """Cross-sectional Spearman IC between factor scores and forward returns."""
    common = scores.dropna().index.intersection(fwd.dropna().index)
    if len(common) < 3:
        return np.nan
    ic, _ = stats.spearmanr(scores[common], fwd[common])
    return float(ic) if not np.isnan(ic) else np.nan


def _rank_norm(s: pd.Series) -> pd.Series:
    """Rank-normalise to [0, 1] cross-sectionally."""
    return s.rank(pct=True)


def _winsorise(s: pd.Series, pct: float = 0.05) -> pd.Series:
    """Clip extreme values at pct/2 and (1-pct/2) percentiles."""
    lo, hi = s.quantile(pct / 2), s.quantile(1 - pct / 2)
    return s.clip(lo, hi)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 1 — HONEST FACTOR PROXIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def momentum_factor(prices: dict,
                    lookback_months: int = 12,
                    skip_months: int = 1) -> pd.Series:
    """
    FIX 1a — Jegadeesh-Titman (1993) 12-1 momentum.

    Correct formula: return from t-12m to t-1m (skipping last month avoids
    short-term reversal contamination). Original code used t-12m to t-0,
    which includes the reversal period and dilutes the signal.

    Also winsorises raw returns to avoid extreme events dominating the score.
    """
    lb = lookback_months * 21
    sk = skip_months * 21
    scores = {}
    for ticker, df in prices.items():
        close = df["Close"].dropna()
        if len(close) < lb + 5:
            scores[ticker] = np.nan
            continue
        # t-12m to t-1m  (correct Jegadeesh-Titman formulation)
        scores[ticker] = float(close.iloc[-sk] / close.iloc[-lb] - 1)
    s = pd.Series(scores, name="Momentum")
    return _rank_norm(_winsorise(s.dropna())).reindex(s.index)


def low_vol_factor(prices: dict, window: int = 63) -> pd.Series:
    """
    FIX 1b — Baker, Bradley, Wurgler (2011) low-volatility anomaly.

    Volatility measured as realised vol over trailing window. Negated so that
    lower-vol stocks score higher (anomaly: low-vol outperforms on risk-adj basis).
    Also adds a beta component: low market-beta stocks score higher.
    """
    scores = {}
    tickers = list(prices.keys())
    # Build market return proxy (equal-weight of all tickers)
    all_rets = []
    for t in tickers:
        r = prices[t]["Close"].pct_change().dropna()
        if len(r) >= window:
            all_rets.append(r.iloc[-window:].rename(t))
    if all_rets:
        mkt = pd.concat(all_rets, axis=1).mean(axis=1)
    else:
        mkt = None

    for ticker, df in prices.items():
        r = df["Close"].pct_change().dropna()
        if len(r) < window:
            scores[ticker] = np.nan
            continue
        rv = r.iloc[-window:].std() * np.sqrt(252)  # realised vol
        # Beta to market proxy (lower beta = better)
        if mkt is not None and len(mkt) >= 10:
            common = r.iloc[-window:].index.intersection(mkt.index)
            if len(common) >= 10:
                cov  = r.loc[common].cov(mkt.loc[common])
                var  = mkt.loc[common].var()
                beta = cov / var if var > 1e-10 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        # Combined score: 70% realised vol + 30% beta (both negated)
        scores[ticker] = -(0.7 * rv + 0.3 * beta)

    s = pd.Series(scores, name="LowVol")
    return _rank_norm(_winsorise(s.dropna())).reindex(s.index)


def size_factor(prices: dict) -> pd.Series:
    """
    FIX 1c — Banz (1981) size anomaly.

    Proxy: log(price × avg_volume_21d). Smaller = higher score.
    Limitation disclosed: true size needs shares outstanding × price.
    Dollar-volume proxy is the best OHLCV approximation.
    """
    scores = {}
    for ticker, df in prices.items():
        if df["Close"].empty or df["Volume"].empty:
            scores[ticker] = np.nan
            continue
        dv = float(df["Close"].iloc[-1]) * float(df["Volume"].tail(21).mean())
        scores[ticker] = -np.log1p(dv)  # smaller firm = higher score
    s = pd.Series(scores, name="Size")
    return _rank_norm(_winsorise(s.dropna())).reindex(s.index)


def quality_factor(prices: dict, window: int = 252) -> pd.Series:
    """
    FIX 1d — Quality proxy using three OHLCV-computable metrics.

    True quality (ROE, gross profit margin) requires fundamental data.
    Best OHLCV proxies (each independently predictive in research):
      1. Return consistency  : % of months with positive returns (stability)
      2. Drawdown resilience : 1 - |max_drawdown| (company doesn't crash badly)
      3. Trend strength      : R² of log-price on linear time trend (smooth grower)

    Combined as equal-weight composite — all three measure different aspects
    of business quality that price data can reveal.
    Limitation: cannot capture balance sheet quality without fundamentals.
    """
    scores = {}
    for ticker, df in prices.items():
        close = df["Close"].dropna()
        if len(close) < window // 2:
            scores[ticker] = np.nan
            continue
        r = close.pct_change().dropna().iloc[-window:]

        # 1. Return consistency (% positive months using 21-day rolling windows)
        monthly = close.iloc[-window:].pct_change(21).dropna()
        consistency = float((monthly > 0).mean()) if len(monthly) > 0 else 0.5

        # 2. Drawdown resilience
        cum  = (1 + r).cumprod()
        mdd  = float((cum / cum.cummax() - 1).min())  # negative number
        resilience = 1.0 + mdd  # 0 = destroyed 100%, 1 = no drawdown

        # 3. Trend R² (log-price linearity)
        lp = np.log(close.iloc[-window:].values + 1e-10)
        x  = np.arange(len(lp))
        if len(lp) > 5:
            slope, intercept, r_val, *_ = stats.linregress(x, lp)
            r_sq = r_val ** 2
        else:
            r_sq = 0.0

        scores[ticker] = (consistency + resilience + r_sq) / 3.0

    s = pd.Series(scores, name="Quality")
    return _rank_norm(_winsorise(s.dropna())).reindex(s.index)


def value_factor(prices: dict, window: int = 252) -> pd.Series:
    """
    FIX 1e — Value proxy: price mean-reversion potential.

    True value (P/B, P/E, EV/EBITDA) requires fundamental data.
    Best OHLCV proxy: price deviation from its long-run trend.
    A stock far below its historical trend is relatively 'cheap'.
    This is related to the Residual Momentum factor (Blitz et al. 2011)
    and avoids confusing 'anti-momentum' with 'value'.

    Two components:
      1. Price-to-52wk-high ratio (inverted): further below peak = cheaper
      2. Z-score below long-run moving average: statistically cheap vs history
    Limitation: cannot capture P/B or earnings yield without fundamentals.
    """
    scores = {}
    for ticker, df in prices.items():
        close = df["Close"].dropna()
        if len(close) < window:
            scores[ticker] = np.nan
            continue

        # 1. Price-to-52wk-high (inverted — further below high = more 'value')
        high_52 = close.iloc[-window:].max()
        p2h     = close.iloc[-1] / high_52   # 0-1; lower = cheaper
        ptoh_score = 1.0 - p2h               # higher score = cheaper

        # 2. Z-score vs long-run mean (negative z = below average = cheap)
        mu    = close.iloc[-window:].mean()
        sigma = close.iloc[-window:].std()
        z     = (close.iloc[-1] - mu) / (sigma + 1e-10)
        z_score = -z   # negative z (cheap) → high score

        scores[ticker] = (ptoh_score + z_score) / 2.0

    s = pd.Series(scores, name="Value")
    return _rank_norm(_winsorise(s.dropna())).reindex(s.index)


# Convenience map used throughout
FACTOR_FNS: dict[str, callable] = {
    "Momentum": momentum_factor,
    "LowVol":   low_vol_factor,
    "Size":     size_factor,
    "Quality":  quality_factor,
    "Value":    value_factor,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 2 — TIME-SERIES IC (proper cross-sectional, full history)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_timeseries_ic(prices: dict,
                           factor_name: str = "Momentum",
                           fwd_days: int = 5,
                           rebalance_freq: int = 21,
                           min_stocks: int = 3) -> pd.DataFrame:
    """
    FIX 2 — Compute IC at every rebalance date across full history.

    Original code: IC from 4 tickers at one point = statistically meaningless.
    Correct method (Grinold & Kahn 1999): for each rebalance date t,
      1. Compute factor score for every ticker using data up to t
      2. Compute forward return for every ticker from t to t+fwd_days
      3. Compute cross-sectional Spearman rank correlation
      4. Repeat for every t in history

    Returns DataFrame with columns:
      Date, IC, RollingMeanIC, RollingICIR

    Mean IC > 0.05  = meaningful signal
    ICIR    > 0.5   = reliable signal
    """
    ffn = FACTOR_FNS.get(factor_name, momentum_factor)

    # Build aligned close price panel
    tickers = list(prices.keys())
    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()

    if close_panel.shape[1] < min_stocks:
        return pd.DataFrame(columns=["Date", "IC", "RollingMeanIC", "RollingICIR"])

    # Forward return panel
    fwd_panel = close_panel.pct_change(fwd_days).shift(-fwd_days)

    # Rebalance dates (every rebalance_freq trading days, leaving room for lookback)
    min_history = 252 + fwd_days
    all_dates   = close_panel.index
    if len(all_dates) < min_history:
        return pd.DataFrame(columns=["Date", "IC", "RollingMeanIC", "RollingICIR"])

    rebal_dates = all_dates[min_history::rebalance_freq]
    records = []

    for date in rebal_dates:
        # Slice prices up to this date
        prices_slice = {
            t: prices[t].loc[:date] for t in tickers
            if date in prices[t].index or date > prices[t].index[0]
        }
        prices_slice = {t: df for t, df in prices_slice.items()
                        if len(df) >= 252}
        if len(prices_slice) < min_stocks:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = ffn(prices_slice)
            except Exception:
                continue

        # Forward returns from this date
        fwd = fwd_panel.loc[date] if date in fwd_panel.index else pd.Series(dtype=float)
        fwd = fwd.dropna()

        ic = _cs_ic(scores, fwd)
        if not np.isnan(ic):
            records.append({"Date": date, "IC": ic})

    if not records:
        return pd.DataFrame(columns=["Date", "IC", "RollingMeanIC", "RollingICIR"])

    df = pd.DataFrame(records).set_index("Date").sort_index()
    df["RollingMeanIC"] = df["IC"].rolling(12).mean()
    df["RollingICIR"]   = (df["IC"].rolling(12).mean() /
                            df["IC"].rolling(12).std().replace(0, np.nan))
    return df.reset_index()


def factor_summary_stats(ts_ic_df: pd.DataFrame) -> dict:
    """Aggregate IC series into key statistics."""
    if ts_ic_df.empty or "IC" not in ts_ic_df.columns:
        return {"Mean IC": "N/A", "ICIR": "N/A", "IC > 0": "N/A",
                "Obs": 0, "Signal": "Insufficient data"}
    ic     = ts_ic_df["IC"].dropna()
    mean   = ic.mean()
    ir     = ic.mean() / ic.std() if ic.std() > 1e-10 else 0.0
    pct_pos = (ic > 0).mean()

    if abs(mean) >= 0.05 and abs(ir) >= 0.5:
        signal = "Strong ✅"
    elif abs(mean) >= 0.03 or abs(ir) >= 0.3:
        signal = "Moderate ⚠️"
    else:
        signal = "Weak ❌"

    return {
        "Mean IC":   round(float(mean), 4),
        "ICIR":      round(float(ir),   3),
        "IC > 0 %":  f"{pct_pos:.1%}",
        "Obs":       int(len(ic)),
        "Signal":    signal,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 3 — COST-ADJUSTED QUINTILE BACKTEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quintile_returns(factor_scores: pd.Series,
                     fwd_returns:   pd.Series,
                     n: int = 5) -> pd.DataFrame:
    """Original quintile function — kept for UI compatibility."""
    df = pd.DataFrame({"score": factor_scores,
                        "fwd_ret": fwd_returns}).dropna()
    if len(df) < n:
        return pd.DataFrame()
    df["quintile"] = pd.qcut(df["score"], n,
                              labels=[f"Q{i+1}" for i in range(n)])
    return df.groupby("quintile")["fwd_ret"].mean().to_frame("avg_fwd_ret")


def cost_adjusted_quintile_bt(prices: dict,
                               factor_name: str = "Momentum",
                               fwd_days: int = 21,
                               round_trip_cost_bps: float = 40.0,
                               n_quintiles: int = 5,
                               rebalance_freq: int = 21) -> dict:
    """
    FIX 3 — Full time-series quintile backtest net of transaction costs.

    Novy-Marx & Velikov (2016): factor strategies that look good gross of
    costs often lose money net of costs due to high turnover.

    Methodology:
      At each rebalance date:
        1. Score all tickers on the chosen factor
        2. Assign to quintiles (Q1 = best, Q5 = worst)
        3. Compute forward return for each quintile
        4. Deduct round-trip cost on turnover (stocks that changed quintile)

    Returns per-quintile:
      - Gross CAGR, Net CAGR, Gross Sharpe, Net Sharpe
      - Long-short (Q1 - Q5) gross and net spread
      - Turnover rate (% of portfolio replaced each rebalance)
    """
    ffn     = FACTOR_FNS.get(factor_name, momentum_factor)
    tickers = list(prices.keys())
    cost    = round_trip_cost_bps / 10_000

    # Aligned close panel
    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()

    min_hist = 252 + fwd_days
    if close_panel.shape[1] < n_quintiles or len(close_panel) < min_hist:
        return {"error": "Insufficient data for quintile backtest."}

    all_dates   = close_panel.index
    rebal_dates = all_dates[min_hist::rebalance_freq]

    # Store per-quintile gross and net returns at each rebalance
    quintile_gross = {f"Q{i+1}": [] for i in range(n_quintiles)}
    quintile_net   = {f"Q{i+1}": [] for i in range(n_quintiles)}
    turnover_log   = []
    prev_assignments: dict[str, str] = {}

    for date in rebal_dates:
        prices_slice = {t: prices[t].loc[:date] for t in tickers
                        if len(prices[t].loc[:date]) >= 252}
        if len(prices_slice) < n_quintiles:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = ffn(prices_slice).dropna()
            except Exception:
                continue

        if len(scores) < n_quintiles:
            continue

        # Assign quintiles
        try:
            labels  = [f"Q{i+1}" for i in range(n_quintiles)]
            q_assign = pd.qcut(scores, n_quintiles, labels=labels)
        except Exception:
            continue

        # Forward returns (from date to date + fwd_days)
        fwd_idx = all_dates[all_dates > date]
        if len(fwd_idx) < fwd_days:
            continue
        fwd_date = fwd_idx[fwd_days - 1]

        fwd_rets = {}
        for t in scores.index:
            if t in close_panel.columns:
                p_now  = close_panel.loc[date, t]   if date     in close_panel.index else np.nan
                p_fwd  = close_panel.loc[fwd_date, t] if fwd_date in close_panel.index else np.nan
                if pd.notna(p_now) and pd.notna(p_fwd) and p_now > 0:
                    fwd_rets[t] = (p_fwd / p_now) - 1

        # Turnover: stocks that changed quintile since last rebalance
        turnover_count = sum(
            1 for t in q_assign.index
            if prev_assignments.get(t) != str(q_assign[t])
        )
        turnover_rate = turnover_count / max(len(q_assign), 1)
        turnover_log.append(turnover_rate)
        prev_assignments = {t: str(q_assign[t]) for t in q_assign.index}

        for q_label in labels:
            q_tickers = q_assign[q_assign == q_label].index.tolist()
            if not q_tickers:
                continue
            q_rets = [fwd_rets[t] for t in q_tickers if t in fwd_rets]
            if not q_rets:
                continue
            gross = float(np.mean(q_rets))
            # Cost: deduct round-trip on portion that turned over
            net   = gross - cost * turnover_rate
            quintile_gross[q_label].append(gross)
            quintile_net[q_label].append(net)

    # Aggregate
    rows = []
    for q_label in [f"Q{i+1}" for i in range(n_quintiles)]:
        g = np.array(quintile_gross[q_label])
        n_ = np.array(quintile_net[q_label])
        if len(g) < 2:
            continue
        periods_per_year = 252 / rebalance_freq
        gross_cagr = float((1 + g.mean()) ** periods_per_year - 1)
        net_cagr   = float((1 + n_.mean()) ** periods_per_year - 1)
        gross_sh   = float(g.mean() / (g.std() + 1e-10) * np.sqrt(periods_per_year))
        net_sh     = float(n_.mean() / (n_.std() + 1e-10) * np.sqrt(periods_per_year))
        rows.append({
            "Quintile":     q_label,
            "Gross CAGR":   f"{gross_cagr:.2%}",
            "Net CAGR":     f"{net_cagr:.2%}",
            "Gross Sharpe": round(gross_sh, 2),
            "Net Sharpe":   round(net_sh,   2),
            "Obs":          len(g),
        })

    result_df = pd.DataFrame(rows)
    avg_turnover = float(np.mean(turnover_log)) if turnover_log else 0.0

    # Long-short spread
    q1g = np.array(quintile_gross.get("Q1", []))
    q5g = np.array(quintile_gross.get("Q5", []))
    q1n = np.array(quintile_net.get("Q1", []))
    q5n = np.array(quintile_net.get("Q5", []))
    min_len = min(len(q1g), len(q5g))

    if min_len > 1:
        ls_gross = float((q1g[:min_len] - q5g[:min_len]).mean())
        ls_net   = float((q1n[:min_len] - q5n[:min_len]).mean())
    else:
        ls_gross = ls_net = np.nan

    ppa = 252 / rebalance_freq
    return {
        "table":               result_df,
        "avg_turnover":        avg_turnover,
        "ls_gross_per_period": ls_gross,
        "ls_net_per_period":   ls_net,
        "ls_gross_cagr":       float((1 + ls_gross) ** ppa - 1) if not np.isnan(ls_gross) else np.nan,
        "ls_net_cagr":         float((1 + ls_net)   ** ppa - 1) if not np.isnan(ls_net)   else np.nan,
        "cost_bps":            round_trip_cost_bps,
        "rebalance_freq_days": rebalance_freq,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 4 — REGIME-CONDITIONED FACTOR PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BULL     = "Bull 📈"
SIDEWAYS = "Sideways ↔"
BEAR     = "Bear 📉"


def _detect_regime_series(prices: dict, window: int = 63) -> pd.Series:
    """
    Regime classifier for a multi-ticker universe.

    Builds an equal-weight market proxy from all tickers, then delegates
    to core.backtest_engine.detect_regime — the single source of truth for
    regime classification logic across the entire codebase.
    """
    tickers = list(prices.keys())
    rets = pd.DataFrame(
        {t: prices[t]["Close"].pct_change().dropna() for t in tickers}
    ).dropna(how="all")
    mkt_price = (1 + rets.mean(axis=1)).cumprod()   # equal-weight index
    return detect_regime(mkt_price, window=window)


def regime_factor_ic(prices: dict,
                      fwd_days: int = 21,
                      regime_window: int = 63) -> pd.DataFrame:
    """
    FIX 4 — IC of each factor broken down by market regime.

    Daniel & Moskowitz (2016): Momentum crashes specifically in volatile
    bear markets and rebounds. Running momentum blindly through all regimes
    is harmful. This function shows which factor works in which regime
    so the composite can be tuned accordingly.

    Returns DataFrame: Factor × Regime grid of mean IC and observation count.
    """
    regime_series = _detect_regime_series(prices, window=regime_window)
    tickers       = list(prices.keys())

    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()
    fwd_panel = close_panel.pct_change(fwd_days).shift(-fwd_days)

    min_hist   = 252 + fwd_days
    all_dates  = close_panel.index
    rebal_freq = fwd_days  # rebalance every fwd_days

    if len(all_dates) < min_hist:
        return pd.DataFrame()

    rebal_dates = all_dates[min_hist::rebal_freq]

    # For each factor × regime, collect ICs
    ic_store: dict[tuple, list] = {}
    for fname, ffn in FACTOR_FNS.items():
        for reg in [BULL, SIDEWAYS, BEAR]:
            ic_store[(fname, reg)] = []

    for date in rebal_dates:
        reg = regime_series.get(date, SIDEWAYS)
        prices_slice = {t: prices[t].loc[:date] for t in tickers
                        if len(prices[t].loc[:date]) >= 252}
        if len(prices_slice) < 3:
            continue
        fwd = fwd_panel.loc[date] if date in fwd_panel.index else pd.Series(dtype=float)
        fwd = fwd.dropna()
        if len(fwd) < 3:
            continue

        for fname, ffn in FACTOR_FNS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    scores = ffn(prices_slice).dropna()
                except Exception:
                    continue
            ic = _cs_ic(scores, fwd)
            if not np.isnan(ic):
                ic_store[(fname, reg)].append(ic)

    rows = []
    for (fname, reg), ics in ic_store.items():
        if not ics:
            continue
        rows.append({
            "Factor":   fname,
            "Regime":   reg,
            "Mean IC":  round(float(np.mean(ics)),  4),
            "ICIR":     round(float(np.mean(ics) / (np.std(ics) + 1e-10)), 3),
            "Obs":      len(ics),
            "Signal":   ("Strong ✅" if abs(np.mean(ics)) >= 0.05 else
                          "Moderate ⚠️" if abs(np.mean(ics)) >= 0.02 else "Weak ❌"),
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
            .sort_values(["Regime", "Mean IC"], ascending=[True, False])
            .reset_index(drop=True))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 5 — IC-WEIGHTED COMPOSITE FACTOR SCORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_factor_matrix(prices: dict) -> pd.DataFrame:
    """Original factor matrix — kept for UI compatibility."""
    factors = pd.DataFrame({
        name: ffn(prices) for name, ffn in FACTOR_FNS.items()
    })
    return factors.rank(pct=True)


def ic_weighted_composite(prices: dict,
                            ts_ic_results: dict[str, pd.DataFrame],
                            ic_lookback: int = 12) -> pd.Series:
    """
    FIX 5 — IC-weighted composite factor score.

    Grinold & Kahn (1999) "Active Portfolio Management":
      Composite = Σ(IC_i × Score_i) / Σ|IC_i|

    Rationale: weight each factor by its recent predictive power.
    A factor with IC=0.08 gets 8x more weight than one with IC=0.01.
    This adapts to changing market conditions — when momentum stops
    working, it automatically downweights momentum.

    ic_lookback: number of recent IC observations to use for weighting.
    """
    # Compute individual factor scores
    factor_scores = {}
    for name, ffn in FACTOR_FNS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                factor_scores[name] = ffn(prices)
            except Exception:
                factor_scores[name] = pd.Series(dtype=float)

    # Get recent mean IC for each factor
    ic_weights = {}
    for name, ts_df in ts_ic_results.items():
        if ts_df.empty or "IC" not in ts_df.columns:
            ic_weights[name] = 0.0
        else:
            recent_ic = ts_df["IC"].dropna().iloc[-ic_lookback:]
            ic_weights[name] = float(recent_ic.mean()) if len(recent_ic) > 0 else 0.0

    total_abs_ic = sum(abs(v) for v in ic_weights.values())
    if total_abs_ic < 1e-8:
        # Fallback: equal weight if no IC available
        weights = {k: 1.0 / len(ic_weights) for k in ic_weights}
    else:
        weights = {k: v / total_abs_ic for k, v in ic_weights.items()}

    # Build composite
    tickers = list(prices.keys())
    composite = pd.Series(0.0, index=pd.Index(tickers))
    for name, weight in weights.items():
        scores = factor_scores.get(name, pd.Series(dtype=float))
        scores = _rank_norm(scores.dropna()).reindex(composite.index).fillna(0.5)
        composite += weight * scores

    return composite.rename("Composite"), weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 6 — FACTOR ATTRIBUTION (Carhart 1997)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def factor_attribution(strategy_returns: pd.Series,
                        prices: dict,
                        rf: float = 0.045) -> dict:
    """
    FIX 6 — Carhart (1997) four-factor attribution + quality/value extension.

    Decomposes strategy returns into:
      Alpha   — return unexplained by any factor (true skill)
      Market  — simple market exposure (beta × market return)
      Mom     — momentum factor return
      LowVol  — low-volatility factor return
      Size    — size factor return
      Quality — quality factor return
      Value   — value factor return

    High alpha = genuine stock-picking / timing skill.
    Low alpha + high factor betas = strategy just rides known factor premia
    (which any cheap ETF provides).

    Method: OLS regression of excess strategy returns on excess factor returns.
    t-stat on alpha > 2.0 indicates statistically significant skill.
    """
    tickers = list(prices.keys())
    if not tickers or strategy_returns.empty:
        return {"error": "Need strategy returns and price data."}

    # Build factor return series (long Q1, short Q5 for each factor)
    factor_rets: dict[str, pd.Series] = {}
    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()

    daily_rets = close_panel.pct_change().dropna()
    if daily_rets.empty:
        return {"error": "Cannot compute factor returns."}

    # Market factor = equal-weight portfolio
    factor_rets["Market"] = daily_rets.mean(axis=1)

    # Per-factor: compute score quarterly, use as portfolio weight
    for fname, ffn in FACTOR_FNS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                score = ffn(prices).dropna()
            except Exception:
                continue
        if len(score) < 2:
            continue
        # Rank-normalise → use as weight (top = long, bottom = short)
        score_rn = _rank_norm(score)
        weight   = (score_rn - 0.5) * 2   # centre at 0, range [-1, 1]
        weight   = weight / (weight.abs().sum() + 1e-10)
        fret     = daily_rets[weight.index.intersection(daily_rets.columns)].dot(
                    weight.reindex(daily_rets.columns).fillna(0))
        factor_rets[fname] = fret

    # Align everything
    common_idx = strategy_returns.index
    for k in factor_rets:
        common_idx = common_idx.intersection(factor_rets[k].index)
    if len(common_idx) < 30:
        return {"error": "Insufficient overlapping dates for attribution."}

    rf_daily = rf / 252
    y  = strategy_returns.loc[common_idx] - rf_daily
    X  = pd.DataFrame({k: v.loc[common_idx] - rf_daily
                        for k, v in factor_rets.items()}).dropna(axis=1)
    X_aug = np.column_stack([np.ones(len(X)), X.values])

    try:
        beta, resid, *_ = np.linalg.lstsq(X_aug, y.values, rcond=None)
    except Exception:
        return {"error": "OLS regression failed."}

    # t-stats
    n, k  = len(y), X_aug.shape[1]
    if n <= k:
        return {"error": "Too few observations for OLS."}
    sigma2 = float(np.sum(resid if len(resid) else (y.values - X_aug @ beta)**2) / (n - k))
    XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
    se    = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    tstat = beta / (se + 1e-12)

    labels = ["Alpha"] + X.columns.tolist()
    rows   = []
    for i, lbl in enumerate(labels):
        b = float(beta[i])
        t = float(tstat[i])
        if lbl == "Alpha":
            ann = b * 252
            rows.append({
                "Factor":     "Alpha (annualised)",
                "Coefficient": f"{ann:.4%}",
                "t-stat":      round(t, 2),
                "Significant": "Yes ✅" if abs(t) >= 2.0 else "No ❌",
                "Interpretation": ("Genuine skill" if ann > 0 and abs(t) >= 2.0
                                    else "No significant alpha"),
            })
        else:
            rows.append({
                "Factor":     lbl,
                "Coefficient": round(b, 4),
                "t-stat":      round(t, 2),
                "Significant": "Yes ✅" if abs(t) >= 2.0 else "No ❌",
                "Interpretation": (f"Exposed to {lbl} factor" if abs(t) >= 2.0
                                    else "Not significant"),
            })

    y_hat = X_aug @ beta
    ss_res = np.sum((y.values - y_hat) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "table":      pd.DataFrame(rows),
        "r_squared":  round(r2, 4),
        "n_obs":      n,
        "alpha_pct":  f"{float(beta[0]) * 252:.4%}",
        "alpha_tstat":round(float(tstat[0]), 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 7 — FACTOR CROWDING DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_factor_crowding(prices: dict,
                            factor_name: str = "Momentum",
                            window: int = 63,
                            lookback: int = 252) -> dict:
    """
    FIX 7 — Factor crowding detection via score dispersion collapse.

    Khandani & Lo (2007) post-mortem of the 2007 Quant Quake:
    When many funds run the same factor strategy, they hold the same stocks.
    Forced liquidation by one fund cascades to all.

    Detection method: track the cross-sectional standard deviation of factor
    scores over time. When dispersion COLLAPSES (std drops sharply), it means
    all stocks are scoring similarly — the factor is crowded. The bottom quintile
    of historical dispersion is the danger zone.

    Also tracks: factor score autocorrelation (high = persistent crowding).

    Returns:
      crowding_series  : pd.Series — rolling dispersion over time
      current_pctile   : float    — current dispersion vs history (low = crowded)
      is_crowded        : bool
      crowding_level    : str     — 'Low' | 'Moderate' | 'High' | 'Extreme'
    """
    ffn     = FACTOR_FNS.get(factor_name, momentum_factor)
    tickers = list(prices.keys())

    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()

    if len(close_panel) < lookback or close_panel.shape[1] < 3:
        return {
            "error": "Insufficient data for crowding detection.",
            "is_crowded": False,
            "crowding_level": "Unknown",
        }

    all_dates   = close_panel.index
    rebal_dates = all_dates[252::21]   # monthly rebalance

    disp_records = []
    prev_scores  = None

    for date in rebal_dates:
        prices_slice = {t: prices[t].loc[:date] for t in tickers
                        if len(prices[t].loc[:date]) >= 252}
        if len(prices_slice) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = ffn(prices_slice).dropna()
            except Exception:
                continue
        if len(scores) < 3:
            continue

        dispersion = float(scores.std())   # cross-sectional std of factor scores
        # Score autocorrelation (persistence of rankings)
        if prev_scores is not None:
            common = scores.index.intersection(prev_scores.index)
            if len(common) >= 3:
                ac, _ = stats.spearmanr(scores[common], prev_scores[common])
            else:
                ac = np.nan
        else:
            ac = np.nan

        disp_records.append({"Date": date,
                              "Dispersion": dispersion,
                              "ScoreAutoCorr": ac})
        prev_scores = scores

    if not disp_records:
        return {"error": "No rebalance dates produced valid scores.",
                "is_crowded": False, "crowding_level": "Unknown"}

    disp_df = pd.DataFrame(disp_records).set_index("Date")
    current_disp = float(disp_df["Dispersion"].iloc[-1])
    hist_disp    = disp_df["Dispersion"].dropna()
    current_pctile = float(stats.percentileofscore(hist_disp, current_disp)) / 100.0

    # Low dispersion percentile = crowded (everyone scored similarly)
    if current_pctile <= 0.10:
        crowding_level, is_crowded = "Extreme 🔴", True
    elif current_pctile <= 0.25:
        crowding_level, is_crowded = "High 🟠", True
    elif current_pctile <= 0.40:
        crowding_level, is_crowded = "Moderate 🟡", False
    else:
        crowding_level, is_crowded = "Low ✅", False

    return {
        "crowding_series":   disp_df.reset_index(),
        "current_dispersion":round(current_disp, 4),
        "current_pctile":    round(current_pctile, 3),
        "is_crowded":        is_crowded,
        "crowding_level":    crowding_level,
        "avg_autocorr":      round(float(disp_df["ScoreAutoCorr"].dropna().mean()), 3),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 8 — CROSS-SECTIONAL DECAY CURVE (correct methodology)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def factor_decay(prices: dict, factor_fn=None,
                 horizons: list = None) -> pd.DataFrame:
    """Original decay — kept for UI compatibility (used in old page)."""
    if horizons is None:
        horizons = [1, 5, 10, 21, 63, 126]
    if factor_fn is None:
        factor_fn = momentum_factor
    results = []
    scores  = factor_fn(prices).dropna()
    ticker  = list(prices.keys())[0]
    close   = prices[ticker]["Close"]
    for h in horizons:
        fwd = close.pct_change(h).shift(-h).dropna()
        ic  = m.information_coefficient(
            pd.Series([scores.get(t, np.nan) for t in fwd.index]), fwd)
        results.append({"Horizon (days)": h, "IC": round(ic, 4)})
    return pd.DataFrame(results)


def cross_sectional_decay(prices: dict,
                           factor_name: str = "Momentum",
                           horizons: list | None = None,
                           n_samples: int = 20) -> pd.DataFrame:
    """
    FIX 8 — Correct cross-sectional factor decay curve.

    Original code: time-series IC for one ticker at each horizon.
    Correct method: at each sample date, score ALL tickers → compute
    cross-sectional IC against their forward returns at horizon h.
    Repeat for n_samples dates → average IC per horizon.

    This is the decay curve described in Grinold & Kahn (1999):
    it tells you how quickly the cross-sectional signal loses its
    predictive power as you look further into the future.

    Interpretation:
      IC drops to 0 by day 5   → trade very frequently (signal decays fast)
      IC stays >0 until day 63 → monthly rebalancing is fine
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 63, 126]

    ffn     = FACTOR_FNS.get(factor_name, momentum_factor)
    tickers = list(prices.keys())

    close_panel = pd.DataFrame(
        {t: prices[t]["Close"].dropna() for t in tickers}
    ).dropna(how="all").sort_index()

    min_hist = 252 + max(horizons) + 5
    if len(close_panel) < min_hist or close_panel.shape[1] < 3:
        return pd.DataFrame({"Horizon (days)": horizons,
                              "IC": [np.nan] * len(horizons),
                              "IC Std": [np.nan] * len(horizons)})

    # Sample n_samples evenly spaced dates
    usable = close_panel.index[252:-max(horizons)-5]
    step   = max(1, len(usable) // n_samples)
    sample_dates = usable[::step][:n_samples]

    horizon_ics: dict[int, list] = {h: [] for h in horizons}

    for date in sample_dates:
        prices_slice = {t: prices[t].loc[:date] for t in tickers
                        if len(prices[t].loc[:date]) >= 252}
        if len(prices_slice) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = ffn(prices_slice).dropna()
            except Exception:
                continue

        if len(scores) < 3:
            continue

        for h in horizons:
            fwd_panel = close_panel.pct_change(h).shift(-h)
            if date not in fwd_panel.index:
                continue
            fwd = fwd_panel.loc[date].dropna()
            ic  = _cs_ic(scores, fwd)
            if not np.isnan(ic):
                horizon_ics[h].append(ic)

    rows = []
    for h in horizons:
        ics = horizon_ics[h]
        rows.append({
            "Horizon (days)": h,
            "IC":             round(float(np.mean(ics)),  4) if ics else np.nan,
            "IC Std":         round(float(np.std(ics)),   4) if ics else np.nan,
            "IC > 0 %":       f"{(np.array(ics) > 0).mean():.0%}" if ics else "N/A",
            "Obs":            len(ics),
        })
    return pd.DataFrame(rows)