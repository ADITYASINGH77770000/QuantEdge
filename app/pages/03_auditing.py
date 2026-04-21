
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as sp_stats

from core.data import get_ohlcv
from app.data_engine import (
    render_data_engine_controls,
    render_single_ticker_input,
    load_ticker_data,
    get_global_start_date,
)
from utils.charts import metric_card_row
from utils.config import cfg

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Auditing | QuantEdge", layout="wide")
st.title("🔍 Professional Data Auditing Engine")
st.caption(
    "8 quant-grade checks — runs all at once. "
    "Catches the problems that actually destroy strategies in production."
)

render_data_engine_controls("auditing")
col_a, col_b = st.columns(2)
ticker = render_single_ticker_input(
    "Ticker", key="auditing_ticker",
    default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
    container=col_a,
)
start = pd.to_datetime(get_global_start_date())

with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Full Audit", type="primary", use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Severity badge HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _badge(level: str) -> str:
    colours = {
        "PASS":    ("#d4edda", "#155724"),
        "WARN":    ("#fff3cd", "#856404"),
        "FAIL":    ("#f8d7da", "#721c24"),
        "INFO":    ("#d1ecf1", "#0c5460"),
    }
    bg, fg = colours.get(level, colours["INFO"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:4px;font-size:12px;font-weight:600">{level}</span>')


def _section(title: str, badge: str):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:28px 0 8px">'
        f'<span style="font-size:17px;font-weight:600">{title}</span>'
        f'{_badge(badge)}</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 1 — CORPORATE ACTION MISALIGNMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_corporate_actions(df: pd.DataFrame) -> dict:
    """
    Detects split and dividend contamination by comparing raw Close vs Adj Close.

    A sudden large divergence between the two columns = unadjusted corporate event.
    Overnight gaps > 15% in raw price with near-zero Adj Change = split/dividend.
    Strategy danger: generates massive false BUY/SELL signals on event days.
    """
    results = {"events": [], "severity": "PASS", "n_events": 0}

    if "Adj Close" not in df.columns or "Close" not in df.columns:
        results["severity"] = "INFO"
        results["note"] = "Adj Close column not available — cannot run corporate action audit."
        return results

    raw   = df["Close"].dropna()
    adj   = df["Adj Close"].dropna()
    idx   = raw.index.intersection(adj.index)
    raw, adj = raw.loc[idx], adj.loc[idx]

    # Metric 1: Ratio divergence — sudden jump in Close/Adj ratio
    ratio       = raw / adj
    ratio_chg   = ratio.pct_change().abs().dropna()

    # Metric 2: Overnight gap in raw price vs overnight gap in adj price
    raw_gap = raw.pct_change().abs().dropna()
    adj_gap = adj.pct_change().abs().dropna()

    # Flag days where raw gap > 12% AND adj gap < 1% → likely unadjusted split
    split_mask   = (raw_gap > 0.12) & (adj_gap < 0.01)
    # Flag days where ratio changed > 1% → dividend or partial adjustment
    div_mask     = ratio_chg > 0.01

    events = []
    for date in raw_gap[split_mask].index:
        events.append({
            "Date":       str(date.date()),
            "Type":       "Likely Split / Merger",
            "Raw Gap":    f"{raw_gap.loc[date]:.2%}",
            "Adj Gap":    f"{adj_gap.loc[date]:.2%}",
            "Risk":       "HIGH — false signal likely",
        })
    for date in ratio_chg[div_mask & ~split_mask].index:
        events.append({
            "Date":       str(date.date()),
            "Type":       "Dividend / Partial Adj.",
            "Raw Gap":    f"{raw_gap.loc[date]:.2%}" if date in raw_gap.index else "—",
            "Adj Gap":    f"{adj_gap.loc[date]:.2%}" if date in adj_gap.index else "—",
            "Risk":       "MEDIUM — check adjustment",
        })

    events_df = pd.DataFrame(events).drop_duplicates("Date") if events else pd.DataFrame()
    results["events"]   = events_df
    results["n_events"] = len(events_df)
    results["ratio_series"] = ratio

    if len(events_df) == 0:
        results["severity"] = "PASS"
    elif any("HIGH" in str(r) for r in events_df.get("Risk", [])):
        results["severity"] = "FAIL"
    else:
        results["severity"] = "WARN"

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 2 — STALE PRICE / ILLIQUIDITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_stale_prices(df: pd.DataFrame) -> dict:
    """
    Detects illiquid / stale pricing: consecutive days with identical close price.

    Getmansky, Lo, Asness (2004): stale prices create artificial serial correlation,
    making strategies appear to have alpha when they're trading noise from carried
    forward prices. Also measures lag-1 return autocorrelation as a staleness proxy.
    """
    close = df["Close"].dropna()
    ret   = close.pct_change().dropna()

    # Consecutive identical prices
    is_same    = (close == close.shift(1))
    streak_len = is_same.groupby((~is_same).cumsum()).cumsum()
    max_streak = int(streak_len.max())
    stale_days = int(is_same.sum())
    zero_ret_pct = float((ret == 0).mean())

    # Lag-1 autocorrelation — high positive value = stale price contamination
    lag1_ac = float(ret.autocorr(lag=1)) if len(ret) > 10 else 0.0

    # Streaks table — find streaks > 2
    streaks = []
    in_streak = False
    start_date = None
    count = 0
    for date, same in is_same.items():
        if same and not in_streak:
            in_streak  = True
            start_date = date
            count      = 2
        elif same and in_streak:
            count += 1
        elif not same and in_streak:
            if count >= 3:
                streaks.append({"Start": str(start_date.date()),
                                "Days":  count,
                                "Price": round(close.loc[start_date], 2)})
            in_streak = False

    severity = "PASS"
    if max_streak >= 5 or zero_ret_pct > 0.15:
        severity = "FAIL"
    elif max_streak >= 3 or zero_ret_pct > 0.05:
        severity = "WARN"

    return {
        "severity":      severity,
        "stale_days":    stale_days,
        "max_streak":    max_streak,
        "zero_ret_pct":  zero_ret_pct,
        "lag1_ac":       lag1_ac,
        "streaks_df":    pd.DataFrame(streaks) if streaks else pd.DataFrame(),
        "ret":           ret,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 3 — FAT TAILS & NORMALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_fat_tails(df: pd.DataFrame) -> dict:
    """
    Tests whether return distribution is Gaussian.

    Sharpe/VaR/Sortino all assume normality. Fat tails (kurtosis > 3) mean
    extreme losses occur far more often than the model predicts.

    Jarque-Bera test: H0 = normal distribution.
    p < 0.05 → reject normality → risk metrics are underestimated.
    """
    ret = df["Close"].pct_change().dropna()

    skew     = float(ret.skew())
    kurt     = float(ret.kurt())          # Excess kurtosis (normal = 0)
    jb_stat, jb_pval = sp_stats.jarque_bera(ret.values)
    is_normal = jb_pval >= 0.05

    # Tail ratio: actual 1st/99th percentile vs Gaussian prediction
    p01_actual   = float(np.percentile(ret, 1))
    p99_actual   = float(np.percentile(ret, 99))
    mu, sigma    = ret.mean(), ret.std()
    p01_gaussian = float(sp_stats.norm.ppf(0.01, mu, sigma))
    p99_gaussian = float(sp_stats.norm.ppf(0.99, mu, sigma))

    left_tail_ratio  = p01_actual  / p01_gaussian  if p01_gaussian  != 0 else 1.0
    right_tail_ratio = p99_actual  / p99_gaussian  if p99_gaussian  != 0 else 1.0

    # VaR underestimation
    var_95_gaussian   = float(sp_stats.norm.ppf(0.05, mu, sigma))
    var_95_historical = float(np.percentile(ret, 5))
    var_underestimate = abs(var_95_historical - var_95_gaussian) / abs(var_95_gaussian) if var_95_gaussian != 0 else 0.0

    severity = "PASS"
    if not is_normal and abs(kurt) > 3:
        severity = "FAIL"
    elif not is_normal:
        severity = "WARN"

    return {
        "severity":         severity,
        "skew":             skew,
        "excess_kurtosis":  kurt,
        "jb_stat":          jb_stat,
        "jb_pval":          jb_pval,
        "is_normal":        is_normal,
        "left_tail_ratio":  left_tail_ratio,
        "right_tail_ratio": right_tail_ratio,
        "var_underestimate":var_underestimate,
        "ret":              ret,
        "var_95_gaussian":  var_95_gaussian,
        "var_95_historical":var_95_historical,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 4 — VOLATILITY REGIME SHIFT (CUSUM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_volatility_regime(df: pd.DataFrame) -> dict:
    """
    CUSUM test for structural breaks in return variance.

    Diebold & Inoue (2001): structural breaks in volatility invalidate historical
    risk metrics computed across the break. If a break is detected, Sharpe/VaR
    calculated over the full period are meaningless — they average two different
    regimes that no longer exist as blended form.

    Also computes rolling 63-day vol to visualise regime transitions.
    """
    ret       = df["Close"].pct_change().dropna()
    sq_ret    = ret ** 2    # Proxy for variance
    n         = len(sq_ret)

    # CUSUM: cumulative sum of (sq_ret - mean) standardised by std
    mean_var  = sq_ret.mean()
    std_var   = sq_ret.std()
    cusum     = ((sq_ret - mean_var) / (std_var + 1e-10)).cumsum()
    cusum_max = float(cusum.abs().max())

    # Critical value: ~1.36 * sqrt(n) at 5% significance (Ploberger & Kramer 1992)
    critical_val = 1.36 * np.sqrt(n)
    break_detected = cusum_max > critical_val

    # Locate approximate break point
    break_idx  = int(cusum.abs().idxmax().value) if break_detected else None
    break_date = cusum.abs().idxmax() if break_detected else None

    # Rolling volatility
    roll_vol   = ret.rolling(63).std() * np.sqrt(252)

    # Vol before vs after break
    if break_detected and break_date is not None:
        vol_before = float(ret[:break_date].std() * np.sqrt(252))
        vol_after  = float(ret[break_date:].std()  * np.sqrt(252))
        vol_ratio  = vol_after / vol_before if vol_before > 0 else 1.0
    else:
        vol_before = vol_after = vol_ratio = None

    severity = "FAIL" if break_detected else "PASS"

    return {
        "severity":        severity,
        "break_detected":  break_detected,
        "break_date":      break_date,
        "cusum_max":       cusum_max,
        "critical_val":    critical_val,
        "cusum_series":    cusum,
        "roll_vol":        roll_vol,
        "vol_before":      vol_before,
        "vol_after":       vol_after,
        "vol_ratio":       vol_ratio,
        "ret":             ret,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 5 — BID-ASK BOUNCE (Roll 1984)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_bid_ask_bounce(df: pd.DataFrame) -> dict:
    """
    Roll (1984) effective spread estimator from price return serial covariance.

    When closing price alternates between bid and ask, lag-1 return autocorrelation
    becomes spuriously negative. Roll showed: Spread ≈ 2 * sqrt(-Cov(r_t, r_{t-1}))

    High estimated spread = your strategy is fighting microstructure noise.
    Strong negative lag-1 autocorr = strong bid-ask bounce = mean reversion
    strategies will see false signals.
    """
    ret      = df["Close"].pct_change().dropna()
    cov_lag1 = float(ret.cov(ret.shift(1).dropna().reindex(ret.index).fillna(0)))
    lag1_ac  = float(ret.autocorr(lag=1))
    lag2_ac  = float(ret.autocorr(lag=2))
    lag5_ac  = float(ret.autocorr(lag=5))

    # Roll spread estimate (only valid when covariance is negative)
    roll_spread = 2 * np.sqrt(-cov_lag1) if cov_lag1 < 0 else 0.0

    # As fraction of average price
    avg_price   = float(df["Close"].mean())
    spread_bps  = (roll_spread / avg_price * 10_000) if avg_price > 0 else 0.0

    # Compare with actual High-Low spread as sanity check
    hl_spread_bps = float(
        ((df["High"] - df["Low"]) / df["Close"]).mean() * 10_000
    )

    severity = "PASS"
    if abs(lag1_ac) > 0.15 or spread_bps > 30:
        severity = "FAIL"
    elif abs(lag1_ac) > 0.05 or spread_bps > 10:
        severity = "WARN"

    return {
        "severity":      severity,
        "lag1_ac":       lag1_ac,
        "lag2_ac":       lag2_ac,
        "lag5_ac":       lag5_ac,
        "cov_lag1":      cov_lag1,
        "roll_spread":   roll_spread,
        "spread_bps":    spread_bps,
        "hl_spread_bps": hl_spread_bps,
        "ret":           ret,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 6 — MULTICOLLINEARITY OF SIGNALS (VIF)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_multicollinearity(df: pd.DataFrame) -> dict:
    """
    Variance Inflation Factor (VIF) across RSI, MACD, BB%B, Momentum, DualMA signals.

    VIF > 5  = serious multicollinearity (signals are near-redundant)
    VIF > 10 = extreme multicollinearity (signals carry no independent info)

    When you use correlated signals, you think you have 5 independent views
    on the market. You actually have 1 view, amplified 5 times. This inflates
    confidence in signals and underestimates real strategy risk.
    """
    close = df["Close"].dropna()
    factors = {}

    # RSI (14-day)
    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss     = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rs       = gain / loss.replace(0, np.nan)
    factors["RSI_14"] = (100 - 100 / (1 + rs))

    # MACD line (12-26)
    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    factors["MACD_line"] = ema12 - ema26

    # Bollinger %B (20-day)
    mid      = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    upper    = mid + 2 * std20
    lower    = mid - 2 * std20
    factors["BB_PctB"] = (close - lower) / (upper - lower + 1e-10)

    # Momentum 20-day
    factors["Momentum_20"] = close.pct_change(20)

    # Dual MA spread (14 vs 50)
    ma14     = close.rolling(14).mean()
    ma50     = close.rolling(50).mean()
    factors["DualMA_Spread"] = (ma14 - ma50) / (ma50 + 1e-10)

    factor_df = pd.DataFrame(factors).dropna()

    # Correlation matrix
    corr_matrix = factor_df.corr()

    # VIF: for each factor j, regress on all others → VIF = 1/(1-R²)
    vif_records = []
    cols = factor_df.columns.tolist()
    for i, col in enumerate(cols):
        others = [c for c in cols if c != col]
        X = factor_df[others].values
        y = factor_df[col].values
        # OLS R²
        X_aug   = np.column_stack([np.ones(len(X)), X])
        try:
            beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            y_hat    = X_aug @ beta
            ss_res   = np.sum((y - y_hat) ** 2)
            ss_tot   = np.sum((y - y.mean()) ** 2)
            r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif      = 1 / (1 - r2) if r2 < 1 else 999.0
        except Exception:
            vif = 0.0; r2 = 0.0
        vif_records.append({
            "Signal":    col,
            "VIF":       round(vif, 2),
            "R²":        round(r2, 4),
            "Verdict":   ("OK" if vif < 5 else "HIGH" if vif < 10 else "EXTREME"),
        })

    vif_df     = pd.DataFrame(vif_records)
    max_vif    = float(vif_df["VIF"].max())
    n_high     = int((vif_df["VIF"] >= 5).sum())

    severity   = "PASS"
    if max_vif >= 10:
        severity = "FAIL"
    elif max_vif >= 5:
        severity = "WARN"

    return {
        "severity":     severity,
        "vif_df":       vif_df,
        "corr_matrix":  corr_matrix,
        "max_vif":      max_vif,
        "n_high":       n_high,
        "factor_df":    factor_df,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 7 — TIMESTAMP & CALENDAR INTEGRITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_timestamp_integrity(df: pd.DataFrame, ticker: str) -> dict:
    """
    Full OHLCV and calendar integrity check:
    - Missing trading dates (vs business day calendar)
    - Duplicate timestamps
    - Future-dated rows (impossible)
    - High < Low rows (data corruption)
    - Close outside [Low, High] (OHLCV logic violation)
    - Volume = 0 on non-holiday trading days
    - Weekend / holiday rows
    """
    issues  = []
    today   = pd.Timestamp.today().normalize()

    # 1. Duplicates
    n_dups = int(df.index.duplicated().sum())
    if n_dups:
        issues.append({"Check": "Duplicate timestamps",
                        "Count": n_dups, "Severity": "FAIL"})

    # 2. Future dates
    future = df[df.index > today]
    if len(future):
        issues.append({"Check": "Future-dated rows",
                        "Count": len(future), "Severity": "FAIL"})

    # 3. Weekend rows
    weekends = df[df.index.dayofweek >= 5]
    if len(weekends):
        issues.append({"Check": "Weekend rows",
                        "Count": len(weekends), "Severity": "WARN"})

    # 4. High < Low
    bad_hl = df[df["High"] < df["Low"]]
    if len(bad_hl):
        issues.append({"Check": "High < Low (impossible)",
                        "Count": len(bad_hl), "Severity": "FAIL"})

    # 5. Close outside [Low, High]
    bad_close = df[(df["Close"] < df["Low"]) | (df["Close"] > df["High"])]
    if len(bad_close):
        issues.append({"Check": "Close outside [Low, High]",
                        "Count": len(bad_close), "Severity": "FAIL"})

    # 6. Zero volume on weekdays
    if "Volume" in df.columns:
        zero_vol = df[(df["Volume"] == 0) & (df.index.dayofweek < 5)]
        if len(zero_vol):
            issues.append({"Check": "Zero volume (weekday)",
                            "Count": len(zero_vol), "Severity": "WARN"})

    # 7. Missing business dates
    biz_dates    = pd.date_range(df.index.min(), df.index.max(), freq="B")
    missing_dates = biz_dates.difference(df.index)
    # Allow up to 15 missing days (holidays vary by exchange)
    if len(missing_dates) > 15:
        issues.append({"Check": f"Missing business dates (>{len(missing_dates)} gaps)",
                        "Count": len(missing_dates), "Severity": "WARN"})

    # 8. NaN values
    n_nan = int(df.isnull().sum().sum())
    if n_nan:
        issues.append({"Check": "NaN values",
                        "Count": n_nan, "Severity": "WARN"})

    n_issues = len(issues)
    severity = "PASS"
    if any(i["Severity"] == "FAIL" for i in issues):
        severity = "FAIL"
    elif n_issues > 0:
        severity = "WARN"

    return {
        "severity":      severity,
        "issues_df":     pd.DataFrame(issues) if issues else pd.DataFrame(),
        "n_issues":      n_issues,
        "total_rows":    len(df),
        "date_range":    f"{df.index.min().date()} → {df.index.max().date()}",
        "missing_dates": missing_dates,
        "n_missing":     len(missing_dates),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 8 — LOOKAHEAD & DATA FRESHNESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_lookahead_freshness(df: pd.DataFrame) -> dict:
    """
    Data freshness + Adj Close drift audit.

    Freshness: how old is the most recent data row? Stale data = strategy
    running on yesterday's signals thinking it's today.

    Adj Close drift: yfinance retroactively restates historical Adj Close
    when you re-download. If your backtest was run in June and you re-run
    in December, the historical data may have changed — same dates,
    different prices. This is silent data mutation.

    We detect this by measuring the cumulative drift between Close and
    Adj Close over time — sudden jumps = retroactive restatement events.
    """
    today    = pd.Timestamp.today().normalize()
    last_row = df.index.max()
    days_old = int((today - last_row).days)

    # Strip weekends from staleness calc (data can't update on weekends)
    biz_days_old = np.busday_count(last_row.date(), today.date())

    freshness_ok = biz_days_old <= 2

    # Adj Close drift analysis
    drift_events = []
    if "Adj Close" in df.columns and "Close" in df.columns:
        ratio      = (df["Adj Close"] / df["Close"]).dropna()
        ratio_chg  = ratio.pct_change().abs().dropna()
        # Sudden ratio jumps > 0.5% = retroactive restatement
        restatements = ratio_chg[ratio_chg > 0.005]
        for date, chg in restatements.items():
            drift_events.append({
                "Date":  str(date.date()),
                "Drift": f"{chg:.4%}",
                "Note":  "Retroactive Adj Close restatement",
            })

    # Price range sanity (extreme price levels)
    close      = df["Close"].dropna()
    price_low  = float(close.min())
    price_high = float(close.max())
    price_ratio = price_high / price_low if price_low > 0 else 1.0

    # Return series completeness
    ret          = close.pct_change().dropna()
    n_inf        = int(np.isinf(ret).sum())
    n_nan_ret    = int(ret.isna().sum())

    severity = "PASS"
    if not freshness_ok or n_inf > 0:
        severity = "FAIL"
    elif len(drift_events) > 3 or n_nan_ret > 0:
        severity = "WARN"

    return {
        "severity":       severity,
        "days_old":       days_old,
        "biz_days_old":   biz_days_old,
        "freshness_ok":   freshness_ok,
        "last_date":      str(last_row.date()),
        "drift_events":   pd.DataFrame(drift_events) if drift_events else pd.DataFrame(),
        "n_restatements": len(drift_events),
        "price_low":      price_low,
        "price_high":     price_high,
        "price_ratio":    price_ratio,
        "n_inf_returns":  n_inf,
        "n_nan_returns":  n_nan_ret,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PAGE RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if run_btn:
    with st.spinner(f"Loading data for {ticker}..."):
        df = load_ticker_data(ticker, start=str(start))

    if df is None or df.empty:
        st.error("No data returned. Check the ticker or date range.")
        st.stop()

    with st.spinner("Running all 8 audit checks..."):
        r_corp  = check_corporate_actions(df)
        r_stale = check_stale_prices(df)
        r_fat   = check_fat_tails(df)
        r_vol   = check_volatility_regime(df)
        r_bid   = check_bid_ask_bounce(df)
        r_vif   = check_multicollinearity(df)
        r_ts    = check_timestamp_integrity(df, ticker)
        r_fresh = check_lookahead_freshness(df)

    # ── Summary scorecard ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Audit Scorecard — {ticker}")

    checks = [
        ("Corporate Actions",         r_corp["severity"]),
        ("Stale Prices",              r_stale["severity"]),
        ("Fat Tails / Normality",     r_fat["severity"]),
        ("Volatility Regime Shift",   r_vol["severity"]),
        ("Bid-Ask Bounce",            r_bid["severity"]),
        ("Signal Multicollinearity",  r_vif["severity"]),
        ("Timestamp Integrity",       r_ts["severity"]),
        ("Data Freshness",            r_fresh["severity"]),
    ]
    n_pass = sum(1 for _, s in checks if s == "PASS")
    n_warn = sum(1 for _, s in checks if s == "WARN")
    n_fail = sum(1 for _, s in checks if s == "FAIL")

    sc = st.columns(4)
    sc[0].metric("Checks Run",  len(checks))
    sc[1].metric("✅ Pass",      n_pass)
    sc[2].metric("⚠️ Warnings",  n_warn)
    sc[3].metric("🔴 Failures",  n_fail)

    # Badge row
    badge_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:12px 0">'
    sev_icon   = {"PASS": "✅", "WARN": "⚠️", "FAIL": "🔴", "INFO": "ℹ️"}
    for name, sev in checks:
        badge_html += (
            f'<div style="border:1px solid #444;border-radius:6px;'
            f'padding:6px 14px;font-size:13px">'
            f'{sev_icon.get(sev,"ℹ️")} {name} — {_badge(sev)}</div>'
        )
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)

    if n_fail == 0 and n_warn == 0:
        st.success("All checks passed. Data appears clean and reliable.")
    elif n_fail > 0:
        st.error(f"{n_fail} critical issue(s) detected. Review FAIL sections before running any strategy.")
    else:
        st.warning(f"{n_warn} warning(s) detected. Review before deploying live.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 1 — CORPORATE ACTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("1 — Corporate Action Misalignment", r_corp["severity"])
    st.caption(
        "Compares raw Close vs Adjusted Close to detect splits, dividends, and mergers "
        "that create artificial price jumps — the #1 cause of false signals in backtests."
    )

    if r_corp.get("note"):
        st.info(r_corp["note"])
    else:
        cc = st.columns(3)
        cc[0].metric("Events Detected", r_corp["n_events"])
        cc[1].metric("Severity", r_corp["severity"])
        cc[2].metric("Action Required",
                      "Review flagged dates" if r_corp["n_events"] else "None")

        if r_corp["n_events"] > 0:
            st.dataframe(r_corp["events"], use_container_width=True)

        if "ratio_series" in r_corp:
            ratio = r_corp["ratio_series"]
            fig = go.Figure(go.Scatter(x=ratio.index, y=ratio.values,
                line=dict(color="orange", width=1.5), name="Close / Adj Close ratio"))
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                           annotation_text="No divergence")
            fig.update_layout(template="plotly_dark", height=260,
                title="Raw Close ÷ Adjusted Close  (jumps = corporate events)",
                yaxis_title="Ratio")
            st.plotly_chart(fig, use_container_width=True)

        if r_corp["severity"] == "PASS":
            st.success("No corporate action contamination detected.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 2 — STALE PRICES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("2 — Stale Price / Illiquidity Detection", r_stale["severity"])
    st.caption(
        "Detects consecutive identical closing prices — a sign that a stock wasn't "
        "trading and prices were carried forward. Creates fake smooth equity curves "
        "and misleadingly high Sharpe ratios. (Getmansky, Lo, Asness 2004)"
    )

    sc2 = st.columns(4)
    sc2[0].metric("Stale Days Total",    r_stale["stale_days"])
    sc2[1].metric("Max Price Streak",    f"{r_stale['max_streak']} days")
    sc2[2].metric("Zero-Return Days",    f"{r_stale['zero_ret_pct']:.1%}")
    sc2[3].metric("Lag-1 Autocorr",      f"{r_stale['lag1_ac']:.4f}",
                   help="High positive = stale pricing. > 0.10 is suspicious.")

    if not r_stale["streaks_df"].empty:
        st.warning("Stale price streaks detected (3+ consecutive identical prices):")
        st.dataframe(r_stale["streaks_df"], use_container_width=True)

    # Return distribution with zero-return annotation
    ret      = r_stale["ret"]
    zero_pct = r_stale["zero_ret_pct"]
    fig_s = go.Figure(go.Histogram(x=ret.values, nbinsx=80,
        marker_color="steelblue", opacity=0.75, name="Daily Returns"))
    fig_s.add_vline(x=0, line_dash="dash", line_color="red",
                     annotation_text=f"{zero_pct:.1%} zero returns")
    fig_s.update_layout(template="plotly_dark", height=260,
        title="Return Distribution — Zero Returns Highlighted")
    st.plotly_chart(fig_s, use_container_width=True)

    if r_stale["severity"] == "PASS":
        st.success("No significant stale pricing detected.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 3 — FAT TAILS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("3 — Fat Tails & Normality Test", r_fat["severity"])
    st.caption(
        "Jarque-Bera test + tail ratio analysis. Sharpe/VaR assume normal returns. "
        "Fat tails (excess kurtosis > 3) mean extreme losses occur 10-100x more often "
        "than your risk model predicts. (Mandelbrot 1963, Fama 1965)"
    )

    ft1, ft2, ft3, ft4 = st.columns(4)
    ft1.metric("Excess Kurtosis",    f"{r_fat['excess_kurtosis']:.3f}",
                help="Normal = 0. Values > 3 = fat tails.")
    ft2.metric("Skewness",           f"{r_fat['skew']:.3f}",
                help="Negative = crash-prone (left tail heavier).")
    ft3.metric("Jarque-Bera p-val",  f"{r_fat['jb_pval']:.4f}",
                help="< 0.05 = reject normality.")
    ft4.metric("VaR Underestimate",  f"{r_fat['var_underestimate']:.1%}",
                help="How much Gaussian VaR underestimates true tail risk.")

    if r_fat["is_normal"]:
        st.success("Returns appear Gaussian (p ≥ 0.05). Risk metrics are reliable.")
    else:
        st.error(
            f"Returns are NOT normal (JB p={r_fat['jb_pval']:.4f}). "
            f"Excess kurtosis = {r_fat['excess_kurtosis']:.2f}. "
            f"VaR is underestimated by ~{r_fat['var_underestimate']:.1%}. "
            f"Use historical VaR, not parametric."
        )

    ret     = r_fat["ret"]
    mu, sig = ret.mean(), ret.std()
    x_range = np.linspace(ret.min(), ret.max(), 200)
    gauss   = sp_stats.norm.pdf(x_range, mu, sig)

    fig_ft = make_subplots(rows=1, cols=2,
        subplot_titles=["Return Distribution vs Gaussian", "Q-Q Plot"])

    # Histogram + Gaussian overlay
    hist_vals, bin_edges = np.histogram(ret.values, bins=80, density=True)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig_ft.add_trace(go.Bar(x=bin_mids, y=hist_vals, name="Actual",
        marker_color="steelblue", opacity=0.7), row=1, col=1)
    fig_ft.add_trace(go.Scatter(x=x_range, y=gauss, name="Gaussian",
        line=dict(color="red", width=2, dash="dash")), row=1, col=1)

    # Q-Q plot
    osm, osr = sp_stats.probplot(ret.values, dist="norm")
    fig_ft.add_trace(go.Scatter(x=osm[0], y=osm[1], mode="markers",
        marker=dict(color="steelblue", size=3), name="Quantiles"), row=1, col=2)
    mn, mx = float(osm[0].min()), float(osm[0].max())
    slope, intercept = osr[0], osr[1]
    fig_ft.add_trace(go.Scatter(x=[mn, mx],
        y=[slope*mn+intercept, slope*mx+intercept],
        line=dict(color="red", width=2), name="Normal line"), row=1, col=2)

    fig_ft.update_layout(template="plotly_dark", height=340,
        showlegend=False, title_text=None)
    st.plotly_chart(fig_ft, use_container_width=True)

    # VaR comparison table
    var_tbl = pd.DataFrame([{
        "Metric":     "VaR 95%",
        "Gaussian":   f"{r_fat['var_95_gaussian']:.4%}",
        "Historical": f"{r_fat['var_95_historical']:.4%}",
        "Difference": f"{abs(r_fat['var_95_historical'] - r_fat['var_95_gaussian']):.4%}",
        "Safer to use": "Historical",
    }])
    st.dataframe(var_tbl, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 4 — VOLATILITY REGIME SHIFT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("4 — Volatility Regime Shift (CUSUM)", r_vol["severity"])
    st.caption(
        "CUSUM test detects structural breaks in return variance. If found, "
        "Sharpe/VaR calculated over the full period blend two different regimes — "
        "making both estimates meaningless. (Diebold & Inoue 2001)"
    )

    vc = st.columns(4)
    vc[0].metric("Break Detected",  "YES 🔴" if r_vol["break_detected"] else "NO ✅")
    vc[1].metric("Break Date",      str(r_vol["break_date"].date()) if r_vol["break_date"] else "N/A")
    vc[2].metric("Vol Before Break", f"{r_vol['vol_before']:.2%}" if r_vol["vol_before"] else "—")
    vc[3].metric("Vol After Break",  f"{r_vol['vol_after']:.2%}"  if r_vol["vol_after"]  else "—")

    if r_vol["break_detected"]:
        ratio = r_vol["vol_ratio"]
        st.error(
            f"Structural break detected at {r_vol['break_date'].date()}. "
            f"Volatility changed {ratio:.1f}x across the break. "
            f"Metrics computed over the full period are unreliable. "
            f"Split your backtest at this date."
        )
    else:
        st.success("No significant volatility regime shift detected.")

    # CUSUM chart + Rolling vol chart
    fig_cv = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["CUSUM of Squared Returns", "Rolling 63-Day Annualised Volatility"],
        row_heights=[0.5, 0.5], vertical_spacing=0.08)

    cusum = r_vol["cusum_series"]
    fig_cv.add_trace(go.Scatter(x=cusum.index, y=cusum.values,
        line=dict(color="gold", width=1.5), name="CUSUM"), row=1, col=1)
    fig_cv.add_hline(y=r_vol["critical_val"],  line_dash="dot", line_color="red",
                      annotation_text="Upper critical (5%)", row=1, col=1)
    fig_cv.add_hline(y=-r_vol["critical_val"], line_dash="dot", line_color="red",
                      row=1, col=1)
    if r_vol["break_date"] is not None:
        fig_cv.add_vline(x=r_vol["break_date"], line_dash="dash",
                          line_color="orange", row=1, col=1)

    rv = r_vol["roll_vol"]
    fig_cv.add_trace(go.Scatter(x=rv.index, y=rv.values,
        line=dict(color="cyan", width=1.5), name="Rolling Vol"), row=2, col=1)
    if r_vol["break_date"] is not None:
        fig_cv.add_vline(x=r_vol["break_date"], line_dash="dash",
                          line_color="orange", row=2, col=1)

    fig_cv.update_layout(template="plotly_dark", height=420, showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 5 — BID-ASK BOUNCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("5 — Bid-Ask Bounce (Roll 1984)", r_bid["severity"])
    st.caption(
        "Estimates effective bid-ask spread from return serial covariance. "
        "Strong negative lag-1 autocorrelation = price bouncing between bid and ask, "
        "creating false mean-reversion signals. (Roll 1984)"
    )

    bc = st.columns(4)
    bc[0].metric("Lag-1 Autocorr",       f"{r_bid['lag1_ac']:.4f}",
                  help="< -0.05 = bid-ask bounce present.")
    bc[1].metric("Roll Spread Est.",      f"{r_bid['spread_bps']:.1f} bps")
    bc[2].metric("H-L Spread (actual)",   f"{r_bid['hl_spread_bps']:.1f} bps")
    bc[3].metric("Lag-2 / Lag-5 AC",
                  f"{r_bid['lag2_ac']:.3f} / {r_bid['lag5_ac']:.3f}")

    if r_bid["cov_lag1"] < 0:
        st.warning(
            f"Negative return covariance detected (Roll spread ≈ {r_bid['spread_bps']:.1f} bps). "
            f"Mean-reversion strategies may be exploiting microstructure noise, not real alpha."
        )
    else:
        st.success(
            f"No significant bid-ask bounce. "
            f"Lag-1 autocorr = {r_bid['lag1_ac']:.4f} (close to zero is healthy)."
        )

    # Autocorrelation bar chart (lag 1–10)
    lags = range(1, 11)
    ac_vals = [float(r_bid["ret"].autocorr(lag=k)) for k in lags]
    fig_ac = go.Figure(go.Bar(x=list(lags), y=ac_vals,
        marker_color=["tomato" if v < -0.05 else "steelblue" for v in ac_vals],
        name="Autocorrelation"))
    fig_ac.add_hline(y=0, line_color="white", line_dash="solid")
    fig_ac.add_hline(y=1.96/np.sqrt(len(r_bid["ret"])),
                      line_dash="dot", line_color="lime",
                      annotation_text="95% CI upper")
    fig_ac.add_hline(y=-1.96/np.sqrt(len(r_bid["ret"])),
                      line_dash="dot", line_color="lime",
                      annotation_text="95% CI lower")
    fig_ac.update_layout(template="plotly_dark", height=280,
        title="Return Autocorrelation by Lag (bars outside CI = significant)",
        xaxis_title="Lag (days)", yaxis_title="Autocorrelation")
    st.plotly_chart(fig_ac, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 6 — MULTICOLLINEARITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("6 — Signal Multicollinearity (VIF)", r_vif["severity"])
    st.caption(
        "Variance Inflation Factor across RSI, MACD, Bollinger %B, Momentum, Dual MA. "
        "VIF > 5 = signals are redundant. Combining them does NOT diversify — "
        "it amplifies one signal multiple times with false confidence."
    )

    vc2 = st.columns(3)
    vc2[0].metric("Max VIF",               f"{r_vif['max_vif']:.1f}")
    vc2[1].metric("High VIF Signals (≥5)", str(r_vif["n_high"]))
    vc2[2].metric("Verdict",
                   "Redundant signals" if r_vif["max_vif"] >= 5 else "Signals are independent")

    # VIF table with colour
    def _vif_colour(row):
        vif = row["VIF"]
        if vif >= 10: return ["", "background-color:#f8d7da", "", ""]
        if vif >= 5:  return ["", "background-color:#fff3cd", "", ""]
        return ["", "background-color:#d4edda", "", ""]

    st.dataframe(
        r_vif["vif_df"].style.apply(_vif_colour, axis=1),
        use_container_width=True,
    )

    # Correlation heatmap
    corr = r_vif["corr_matrix"]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig_corr.update_layout(template="plotly_dark", height=320,
        title="Signal Correlation Matrix  (deep red = highly correlated = redundant)")
    st.plotly_chart(fig_corr, use_container_width=True)

    if r_vif["max_vif"] >= 5:
        st.warning(
            "Signals are highly correlated. Consider using only 1–2 non-overlapping "
            "indicators (e.g. RSI for momentum + ATR for volatility) rather than "
            "5 price-based signals that all say the same thing."
        )
    else:
        st.success("Signals show acceptable independence. Combining them adds information.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 7 — TIMESTAMP INTEGRITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("7 — Timestamp & Calendar Integrity", r_ts["severity"])
    st.caption(
        "Full OHLCV logic validation: duplicates, future rows, weekend rows, "
        "High < Low corruption, Close outside [Low, High], zero volume, "
        "NaN values, and missing business dates."
    )

    tc = st.columns(4)
    tc[0].metric("Total Rows",        r_ts["total_rows"])
    tc[1].metric("Date Range",        r_ts["date_range"])
    tc[2].metric("Missing Biz Dates", r_ts["n_missing"])
    tc[3].metric("Issues Found",      r_ts["n_issues"])

    if not r_ts["issues_df"].empty:
        def _issue_colour(row):
            return (["background-color:#f8d7da"]*3 if row["Severity"] == "FAIL"
                    else ["background-color:#fff3cd"]*3)
        st.dataframe(
            r_ts["issues_df"].style.apply(_issue_colour, axis=1),
            use_container_width=True,
        )
    else:
        st.success(
            "All timestamp and OHLCV integrity checks passed. "
            f"Data covers {r_ts['date_range']} with {r_ts['total_rows']} clean rows."
        )

    if r_ts["n_missing"] > 0:
        with st.expander(f"View {r_ts['n_missing']} missing business dates"):
            st.dataframe(
                pd.DataFrame({"Missing Date": r_ts["missing_dates"]}),
                use_container_width=True,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RENDER CHECK 8 — DATA FRESHNESS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    _section("8 — Lookahead & Data Freshness", r_fresh["severity"])
    st.caption(
        "Checks how old the data is (stale data = running on yesterday's signals "
        "thinking it's today). Also detects retroactive Adj Close restatements — "
        "the silent data mutation that makes two backtest runs on the same dates give "
        "different results."
    )

    fc = st.columns(4)
    fc[0].metric("Last Data Row",        r_fresh["last_date"])
    fc[1].metric("Business Days Old",    r_fresh["biz_days_old"],
                  delta=("Fresh ✅" if r_fresh["freshness_ok"] else "Stale ⚠️"))
    fc[2].metric("Adj Close Restatements", r_fresh["n_restatements"])
    fc[3].metric("Infinite Returns",      r_fresh["n_inf_returns"])

    if not r_fresh["freshness_ok"]:
        st.error(
            f"Data is {r_fresh['biz_days_old']} business days old. "
            "Strategy may be running on stale signals. Re-download before live deployment."
        )
    else:
        st.success(f"Data is fresh ({r_fresh['biz_days_old']} business days old).")

    if not r_fresh["drift_events"].empty:
        st.warning(
            f"{r_fresh['n_restatements']} retroactive Adj Close restatement(s) detected. "
            "Your backtest results may differ if data is re-downloaded in future."
        )
        st.dataframe(r_fresh["drift_events"], use_container_width=True)

    if r_fresh["n_inf_returns"] > 0:
        st.error(
            f"{r_fresh['n_inf_returns']} infinite return values detected. "
            "This usually means a price of 0 exists in the series — corrupted data."
        )

    # Price range sanity
    st.info(
        f"Price range: ₹/$ {r_fresh['price_low']:.2f} → {r_fresh['price_high']:.2f}  "
        f"({r_fresh['price_ratio']:.1f}x range over the period)"
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINAL RECOMMENDATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.subheader("📋 Audit Summary & Recommendations")

    recs = []
    if r_corp["severity"] != "PASS":
        recs.append("🔴 **Corporate Actions**: Use Adj Close for all strategy calculations. "
                     "Exclude flagged dates from backtest or verify adjustments manually.")
    if r_stale["severity"] != "PASS":
        recs.append(f"🔴 **Stale Prices**: Stock shows {r_stale['max_streak']}-day streak of identical prices. "
                     "Avoid this ticker in high-frequency or mean-reversion strategies.")
    if r_fat["severity"] != "PASS":
        recs.append(f"⚠️ **Fat Tails**: Use historical VaR (not Gaussian). "
                     f"Increase risk buffer by ~{r_fat['var_underestimate']:.0%}. "
                     f"Skewness = {r_fat['skew']:.2f} — {'left tail heavy (crash risk)' if r_fat['skew'] < 0 else 'right tail heavy'}.")
    if r_vol["break_detected"]:
        recs.append(f"🔴 **Regime Shift**: Split backtest at {r_vol['break_date'].date()}. "
                     "Run separate analyses before and after the break. "
                     "Metrics computed across the break are unreliable.")
    if r_bid["severity"] != "PASS":
        recs.append(f"⚠️ **Bid-Ask Bounce**: Estimated spread = {r_bid['spread_bps']:.1f} bps. "
                     "Mean-reversion strategies may be trading microstructure noise. "
                     "Add minimum holding period of 3-5 days.")
    if r_vif["severity"] != "PASS":
        recs.append(f"⚠️ **Multicollinearity**: Max VIF = {r_vif['max_vif']:.1f}. "
                     "Reduce to 1-2 uncorrelated signals. "
                     "Using all 5 current signals provides no diversification benefit.")
    if r_ts["severity"] != "PASS":
        recs.append(f"🔴 **Data Integrity**: {r_ts['n_issues']} OHLCV issue(s) found. "
                     "Fix data before running any strategy.")
    if not r_fresh["freshness_ok"]:
        recs.append(f"⚠️ **Stale Data**: Re-download before live trading.")

    if not recs:
        st.success(
            "Data passed all 8 professional audit checks. "
            "This dataset is suitable for strategy development and backtesting."
        )
    else:
        for rec in recs:
            st.markdown(rec)