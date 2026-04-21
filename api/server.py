"""
api/server.py
──────────────────────────────────────────────────────────────────────────────
Thin FastAPI wrapper around existing QuantEdge core modules.
NO backend logic is duplicated — every endpoint calls core/ directly.

Run:  uvicorn api.server:app --reload --port 8000
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Core imports (UNTOUCHED backend) ─────────────────────────────────────────
from core.data import get_ohlcv, returns, get_multi_ohlcv, align_returns
from core.graph_features import DEFAULT_GRAPH_BENCHMARK, build_graph_feature_payload
from core.metrics import (
    summary_table, var_historical, cvar_historical,
    var_parametric, annualised_vol, sharpe, information_coefficient, icir,
)
from core.models import lstm_forecast, arima_forecast, garch_forecast
from core.indicators import add_all_indicators, rsi, macd, bollinger_bands
from core.indicators import (
    signal_rsi, signal_macd_crossover,
    signal_bb_mean_reversion, signal_dual_ma,
)
from core.backtest_engine import (
    run_backtest, BacktestConfig,
    momentum_strategy, mean_reversion_strategy, rsi_strategy,
)
from core.portfolio_opt import monte_carlo_frontier, risk_parity_weights, portfolio_stats
from core.regime_detector import fit_hmm, regime_conditional_sharpe
from core.factor_engine import (
    build_factor_matrix, momentum_factor, low_vol_factor,
    size_factor, quality_factor, value_factor, quintile_returns, factor_decay,
)
# RL environment removed
from utils.config import cfg
from utils.notifications import build_alert_body, send_email

# ── Remove Streamlit cache decorators for API usage ──────────────────────────
# We patch out st.cache_data since we're not in streamlit context
import streamlit as st
st.cache_data = lambda *a, **kw: (lambda f: f)  # no-op decorator

app = FastAPI(title="QuantEdge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALERT_INSIGHTS = {
    "GOOG": {
        "Open": "Opening price reflects initial market sentiment.",
        "Close": "Closing price is the day's final market consensus.",
        "High": "New highs signal bullish momentum.",
        "Low": "New lows suggest selling pressure.",
    },
    "NVDA": {
        "Open": "Higher open can reflect strong pre-market demand.",
        "Close": "Closing strength is useful for trend confirmation.",
        "High": "Breaking highs often attracts additional momentum buyers.",
        "Low": "Sharp lows can reflect stop-loss cascades.",
    },
    "META": {
        "Open": "The open often reacts quickly to platform and ad-market sentiment.",
        "Close": "Closing price helps confirm the market's end-of-day view.",
        "High": "Fresh highs can indicate strong support for growth expectations.",
        "Low": "Lower lows can signal pressure on risk appetite.",
    },
    "AMZN": {
        "Open": "The open often reflects retail and cloud expectations.",
        "Close": "The close is useful for daily trend confirmation.",
        "High": "Higher highs can show confidence in execution.",
        "Low": "Lower lows may signal concern around margin or demand.",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serialisable records."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index()
    # Convert numpy types
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].round(6)
    return df.to_dict(orient="records")


def series_to_records(s: pd.Series, name: str = "value") -> list[dict]:
    """Convert Series to JSON-serialisable records."""
    df = s.rename(name).to_frame()
    return df_to_records(df)


def _metric_to_float(value):
    """Normalise string-formatted metric values into numeric floats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return 0.0
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _lowercase_ohlcv(df: pd.DataFrame) -> list[dict]:
    """Return lowercase OHLCV records expected by the React client."""
    frame = df.copy()
    frame.index = frame.index.strftime("%Y-%m-%d")
    frame = frame.reset_index().rename(columns={
        frame.index.name or "index": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    cols = ["date", "open", "high", "low", "close", "volume"]
    return frame[cols].to_dict(orient="records")


def _strategy_signal(strategy: str, df: pd.DataFrame, df_ind: pd.DataFrame,
                     fast_window: int = 20, slow_window: int = 50) -> pd.Series:
    """Map frontend strategy names to existing backend signal generators."""
    if strategy == "Momentum":
        return momentum_strategy(df, lookback=fast_window)
    if strategy == "Mean Reversion":
        return mean_reversion_strategy(df, window=fast_window, z_thresh=slow_window / 10)
    if strategy == "RSI":
        return rsi_strategy(df, oversold=30, overbought=70)
    if strategy == "MACD Crossover":
        return signal_macd_crossover(df_ind)
    return signal_dual_ma(df_ind, fast=fast_window, slow=slow_window)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/config")
def get_config():
    return {
        "tickers": cfg.DEFAULT_TICKERS,
        "demo_mode": cfg.DEMO_MODE,
        "risk_free_rate": cfg.RISK_FREE_RATE,
        "initial_capital": cfg.INITIAL_CAPITAL,
        "default_start": cfg.DEFAULT_START,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/ohlcv/{ticker}")
def api_ohlcv(ticker: str, start: str = "2020-01-01"):
    try:
        df = get_ohlcv(ticker, start)
        return {"data": df_to_records(df), "rows": len(df)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/returns/{ticker}")
def api_returns(ticker: str, start: str = "2020-01-01"):
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        return {"data": series_to_records(ret, "return"), "rows": len(ret)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics/{ticker}")
def api_metrics(ticker: str, start: str = "2020-01-01"):
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        met = summary_table(ret, cfg.RISK_FREE_RATE)
        return met
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  RISK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/risk/{ticker}")
def api_risk(ticker: str, start: str = "2015-01-01", confidence: float = 0.95):
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        var_h = var_historical(ret, confidence)
        cvar_h = cvar_historical(ret, confidence)
        var_p = var_parametric(ret, confidence)
        vol_a = annualised_vol(ret)

        # Return distribution data
        hist_data = ret.dropna().tolist()

        # Rolling VaR
        roll_var = ret.rolling(63).quantile(1 - confidence).dropna()

        # Stress tests
        scenarios = {
            "2008 Financial Crisis": ("2008-09-01", "2009-03-01"),
            "COVID-19 Crash": ("2020-02-01", "2020-04-01"),
            "2022 Rate Shock": ("2022-01-01", "2022-10-01"),
            "2020 Tech Rally": ("2020-04-01", "2021-01-01"),
            "2018 Q4 Selloff": ("2018-10-01", "2018-12-31"),
        }
        stress_rows = []
        for name, (s, e) in scenarios.items():
            mask = (df.index >= s) & (df.index <= e)
            if mask.sum() < 10:
                continue
            r = ret[mask]
            pnl = float((1 + r).prod() - 1)
            mv = var_historical(r, 0.95)
            stress_rows.append({
                "scenario": name, "period": f"{s} → {e}",
                "total_return": round(pnl, 4),
                "max_drawdown": round(float(r.cumsum().min()), 4),
                "var_95": round(mv, 4),
                "worst_day": round(float(r.min()), 4),
            })

        # Cumulative return
        cum = ((1 + ret).cumprod()).dropna()

        return {
            "var_historical": round(var_h, 6),
            "cvar_historical": round(cvar_h, 6),
            "var_parametric": round(var_p, 6),
            "annualised_vol": round(vol_a, 6),
            "return_distribution": hist_data[:500],
            "rolling_var": series_to_records(roll_var, "rolling_var"),
            "returns_ts": series_to_records(ret, "return"),
            "stress_tests": stress_rows,
            "cumulative_return": series_to_records(cum, "cumulative"),
            "tail_events": int((ret <= var_h).sum()),
            "worst_day": round(float(ret.min()), 6),
            "best_day": round(float(ret.max()), 6),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/indicators/{ticker}")
def api_indicators(ticker: str, start: str = "2020-01-01"):
    try:
        df = get_ohlcv(ticker, start)
        df_ind = add_all_indicators(df)
        return {"data": df_to_records(df_ind), "rows": len(df_ind)}
    except Exception as e:
        raise HTTPException(500, str(e))


# RL trainer removed

        if req.strategy == "Momentum":
            signal = momentum_strategy(df, lookback=req.fast_window)
        elif req.strategy == "Mean Reversion":
            signal = mean_reversion_strategy(df, window=req.fast_window, z_thresh=req.slow_window / 10)
        elif req.strategy == "RSI":
            signal = rsi_strategy(df, oversold=req.fast_window, overbought=req.slow_window)
        elif req.strategy == "MACD Crossover":
            signal = signal_macd_crossover(df_ind)
        else:
            signal = signal_dual_ma(df_ind, fast=req.fast_window, slow=req.slow_window)

        bcfg = BacktestConfig(
            initial_capital=req.capital,
            commission_pct=req.commission,
            slippage_bps=req.slippage_bps,
            risk_free_rate=cfg.RISK_FREE_RATE,
        )
        result = run_backtest(df["Close"], signal, bcfg)

        ret = returns(df)
        bh_cum = ((1 + ret).cumprod()).dropna()

        return {
            "metrics": result.metrics,
            "equity_curve": series_to_records(result.equity_curve, "equity"),
            "daily_returns": series_to_records(result.daily_returns, "return"),
            "trade_log": result.trade_log.to_dict(orient="records") if len(result.trade_log) else [],
            "rolling_sharpe": series_to_records(result.rolling_sharpe, "sharpe"),
            "buy_hold_cumulative": series_to_records(bh_cum, "cumulative"),
            "strategy_cumulative": series_to_records(
                (1 + result.daily_returns).cumprod().dropna(), "cumulative"
            ),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    ticker: str = "GOOG"
    steps: int = 5
    start: str = "2018-01-01"
    features: list[str] = ["Close"]
    epochs: int = 10
    look_back: int = 60


@app.post("/api/prediction/arima")
def api_arima(req: PredictionRequest):
    try:
        df = get_ohlcv(req.ticker, req.start)
        ret = returns(df)
        result = arima_forecast(ret, req.steps)
        if result.empty:
            return {"data": [], "historical": series_to_records(ret.tail(60), "return")}
        return {
            "data": df_to_records(result),
            "historical": series_to_records(ret.tail(60), "return"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/prediction/garch")
def api_garch(req: PredictionRequest):
    try:
        df = get_ohlcv(req.ticker, req.start)
        ret = returns(df)
        result = garch_forecast(ret, req.steps)
        hist_vol = (ret.rolling(21).std().dropna() * (252 ** 0.5)).tail(60)
        if result.empty:
            return {"data": [], "historical_vol": series_to_records(hist_vol, "vol")}
        return {
            "data": df_to_records(result),
            "historical_vol": series_to_records(hist_vol, "vol"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/prediction/lstm")
def api_lstm(req: PredictionRequest):
    try:
        df = get_ohlcv(req.ticker, req.start)
        result = lstm_forecast(df, req.features, req.steps, req.look_back, req.epochs)
        hist = df["Close"].tail(60)
        return {
            "data": df_to_records(result),
            "historical": series_to_records(hist, "close"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioRequest(BaseModel):
    tickers: list[str] = ["GOOG", "NVDA", "META", "AMZN"]
    n_portfolios: int = 1000
    start: str = "2018-01-01"


@app.post("/api/portfolio/frontier")
def api_frontier(req: PortfolioRequest):
    try:
        prices = get_multi_ohlcv(req.tickers, req.start)
        ret_df = align_returns(prices)
        if ret_df.empty or len(ret_df) < 30:
            raise HTTPException(400, "Not enough data for portfolio optimisation")

        result = monte_carlo_frontier(ret_df, req.n_portfolios, cfg.RISK_FREE_RATE)
        ms = result["max_sharpe"]
        mv = result["min_vol"]

        return {
            "frontier": {
                "returns": result["returns"].tolist(),
                "vols": result["vols"].tolist(),
                "sharpes": result["sharpes"].tolist(),
            },
            "max_sharpe": {
                "weights": ms["weights"].tolist(),
                "ret": round(ms["ret"], 4),
                "vol": round(ms["vol"], 4),
                "sharpe": round(ms["sharpe"], 4),
            },
            "min_vol": {
                "weights": mv["weights"].tolist(),
                "ret": round(mv["ret"], 4),
                "vol": round(mv["vol"], 4),
                "sharpe": round(mv["sharpe"], 4),
            },
            "tickers": req.tickers,
            "correlation": ret_df.corr().round(4).to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/portfolio/risk-parity")
def api_risk_parity(req: PortfolioRequest):
    try:
        prices = get_multi_ohlcv(req.tickers, req.start)
        ret_df = align_returns(prices)
        rp_w = risk_parity_weights(ret_df)
        stats = portfolio_stats(rp_w, ret_df, cfg.RISK_FREE_RATE)
        return {
            "weights": rp_w.tolist(),
            "tickers": req.tickers,
            "stats": {k: round(v, 4) for k, v in stats.items()},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# Sentiment API removed


# ══════════════════════════════════════════════════════════════════════════════
#  FACTORS
# ══════════════════════════════════════════════════════════════════════════════

class FactorRequest(BaseModel):
    tickers: list[str] = ["GOOG", "NVDA", "META", "AMZN"]
    fwd_days: int = 5
    start: str = "2015-01-01"


@app.post("/api/factors")
def api_factors(req: FactorRequest):
    try:
        prices = get_multi_ohlcv(req.tickers, req.start)
        factor_df = build_factor_matrix(prices)

        # IC per factor
        fwd_map = {}
        for t in req.tickers:
            ret = returns(prices[t])
            fwd_map[t] = ret.shift(-req.fwd_days).dropna()

        factor_names = ["Momentum", "LowVol", "Size", "Quality", "Value"]
        factor_fns = [momentum_factor, low_vol_factor, size_factor, quality_factor, value_factor]

        ic_rows = []
        for fname, ffn in zip(factor_names, factor_fns):
            scores = ffn(prices).dropna()
            fwd_latest = pd.Series({
                t: fwd_map[t].iloc[-1] if len(fwd_map[t]) > 0 else np.nan
                for t in req.tickers
            })
            ic = information_coefficient(scores, fwd_latest)
            ic_rows.append({"factor": fname, "ic": round(ic, 4)})

        # Decay curve
        decay = factor_decay(prices, momentum_factor, horizons=[1, 5, 10, 21, 63, 126])

        return {
            "factor_matrix": df_to_records(factor_df),
            "ic_scores": ic_rows,
            "decay_curve": df_to_records(decay),
            "tickers": req.tickers,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  REGIME
# ══════════════════════════════════════════════════════════════════════════════

class RegimeRequest(BaseModel):
    n_states: int = 2


@app.post("/api/regime/{ticker}")
def api_regime(ticker: str, req: RegimeRequest, start: str = "2015-01-01"):
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        model, regimes, state_map = fit_hmm(ret, n_states=req.n_states)

        cond_df = regime_conditional_sharpe(ret, regimes, cfg.RISK_FREE_RATE)
        counts = regimes.value_counts().to_dict()
        current = regimes.iloc[-1]

        recs = {
            "Bull 📈": "Momentum strategy — trend-following, long bias, wider stops.",
            "Sideways ↔": "Mean-reversion — Bollinger Band reversals, tighter range trading.",
            "Bear 📉": "Defensive / short bias — reduce position sizes, VIX hedges.",
        }

        # Price + regime data
        price_data = df[["Close"]].copy()
        price_data["regime"] = regimes.reindex(price_data.index).ffill()
        price_data["return"] = ret.reindex(price_data.index)

        return {
            "price_data": df_to_records(price_data),
            "regime_counts": {k: int(v) for k, v in counts.items()},
            "conditional_sharpe": df_to_records(cond_df),
            "current_regime": current,
            "recommendation": recs.get(current, "Hold / analyse"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# Microstructure endpoints removed


# RL trainer removed


# ══════════════════════════════════════════════════════════════════════════════
#  AUDITING
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/audit/{ticker}")
def api_audit(ticker: str, start: str = "2015-01-01"):
    try:
        from statsmodels.tsa.stattools import acf, adfuller, pacf

        df = get_ohlcv(ticker, start)
        ret = df["Close"].pct_change().dropna()
        thresh = 2 * ret.std()
        anomalies = ret[ret.abs() > thresh]
        all_dates = pd.date_range(df.index.min(), df.index.max(), freq="B")
        missing_dates = [dt.strftime("%Y-%m-%d") for dt in all_dates.difference(df.index)]

        close = df["Close"].dropna()
        adf_result = adfuller(close)
        max_lags = min(40, max(len(close) // 2 - 1, 1))
        acf_values = acf(close, nlags=max_lags)
        pacf_values = pacf(close, nlags=max_lags)

        return {
            "total_rows": len(df),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "anomalies": len(anomalies),
            "integrity_issues": int((df["High"] < df["Low"]).sum()),
            "missing_dates": missing_dates,
            "correlation": df[["Open", "High", "Low", "Close", "Volume"]].corr().round(4).to_dict(),
            "statistics": df.describe().round(4).to_dict(),
            "anomaly_data": series_to_records(anomalies.head(20), "return"),
            "adf": {
                "statistic": round(float(adf_result[0]), 6),
                "p_value": round(float(adf_result[1]), 6),
                "critical_values": {k: round(float(v), 6) for k, v in adf_result[4].items()},
                "is_stationary": bool(adf_result[1] < 0.05),
            },
            "acf": [{"lag": idx, "value": round(float(value), 6)} for idx, value in enumerate(acf_values)],
            "pacf": [{"lag": idx, "value": round(float(value), 6)} for idx, value in enumerate(pacf_values)],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/alerts/{ticker}")
def api_alerts(ticker: str,
               open_thresh: float = 150, close_thresh: float = 155,
               high_thresh: float = 160, low_thresh: float = 140,
               send_notifications: bool = False):
    try:
        df = get_ohlcv(ticker)
        latest = df.iloc[-1]
        date = str(df.index[-1].date())

        alerts = []
        emails_sent = 0
        thresholds = {"Open": open_thresh, "Close": close_thresh,
                      "High": high_thresh, "Low": low_thresh}
        for metric, threshold in thresholds.items():
            price = float(latest[metric])
            triggered = price > threshold
            insight = ALERT_INSIGHTS.get(ticker, {}).get(metric, "")
            email_sent = False
            if triggered and send_notifications:
                body = build_alert_body(ticker, metric, price, threshold, insight)
                email_sent = send_email(f"QuantEdge Alert: {ticker} {metric}", body)
                emails_sent += int(email_sent)
            alerts.append({
                "metric": metric,
                "price": round(price, 2),
                "threshold": threshold,
                "triggered": triggered,
                "insight": insight,
                "email_sent": email_sent,
            })

        return {
            "ticker": ticker,
            "date": date,
            "latest": {k: round(float(latest[k]), 2) for k in ["Open", "Close", "High", "Low", "Volume"]},
            "alerts": alerts,
            "notifications_requested": send_notifications,
            "emails_sent": emails_sent,
            "email_configured": bool(cfg.GMAIL_SENDER and cfg.GMAIL_PASSWORD and cfg.GMAIL_RECEIVER),
            "receiver_configured": bool(cfg.GMAIL_RECEIVER),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHS (multi-chart types)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/graphs/{ticker}")
def api_graphs(ticker: str, start: str = "2020-01-01", benchmark: str = DEFAULT_GRAPH_BENCHMARK):
    try:
        df = get_ohlcv(ticker, start)
        benchmark_df = get_ohlcv(benchmark, start)
        return build_graph_feature_payload(
            df,
            ticker=ticker,
            benchmark_df=benchmark_df,
            benchmark_ticker=benchmark,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# -----------------------------------------------------------------------------
# Frontend compatibility routes
# -----------------------------------------------------------------------------

@app.get("/ohlcv")
def compat_ohlcv(ticker: str = Query("GOOG"), start: str = "2020-01-01", end: str | None = None):
    del end
    try:
        df = get_ohlcv(ticker, start)
        return _lowercase_ohlcv(df)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/metrics")
def compat_metrics(ticker: str = Query("GOOG"), start: str = "2020-01-01", end: str | None = None):
    del end
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        met = summary_table(ret, cfg.RISK_FREE_RATE)
        return {
            "sharpe": _metric_to_float(met.get("Sharpe")),
            "sortino": _metric_to_float(met.get("Sortino")),
            "cagr": _metric_to_float(met.get("CAGR")),
            "max_drawdown": _metric_to_float(met.get("Max Drawdown")),
            "win_rate": _metric_to_float(met.get("Win Rate")),
            "var_95": _metric_to_float(met.get("VaR 95%")),
            "cvar_95": _metric_to_float(met.get("CVaR 95%")),
            "ann_vol": _metric_to_float(met.get("Ann. Volatility")),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/signals")
def compat_signals(ticker: str = Query("GOOG"), start: str = "2020-01-01", end: str | None = None):
    del end
    try:
        df = get_ohlcv(ticker, start)
        df_ind = add_all_indicators(df)
        rsi_sig = signal_rsi(df_ind).fillna(0)
        macd_sig = signal_macd_crossover(df_ind).fillna(0)
        bb_sig = signal_bb_mean_reversion(df_ind).fillna(0)
        combined = (rsi_sig + macd_sig + bb_sig).clip(-1, 1)

        rows = []
        for idx, row in df_ind.iterrows():
            rows.append({
                "date": idx.strftime("%Y-%m-%d"),
                "rsi": round(float(row.get("RSI", 0.0)), 6),
                "macd": round(float(row.get("MACD", 0.0)), 6),
                "signal": int(combined.get(idx, 0)),
                "histogram": round(float(row.get("MACD_Hist", 0.0)), 6),
                "rsi_signal": int(rsi_sig.get(idx, 0)),
                "macd_signal": int(macd_sig.get(idx, 0)),
                "bb_signal": int(bb_sig.get(idx, 0)),
            })
        return rows
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/backtest")
def compat_backtest(ticker: str = Query("GOOG"), start: str = "2018-01-01",
                    end: str | None = None, strategy: str = "Momentum"):
    del end
    try:
        df = get_ohlcv(ticker, start)
        df_ind = add_all_indicators(df)
        signal = _strategy_signal(strategy, df, df_ind)
        result = run_backtest(
            df["Close"],
            signal,
            BacktestConfig(
                initial_capital=cfg.INITIAL_CAPITAL,
                commission_pct=0.001,
                slippage_bps=5.0,
                risk_free_rate=cfg.RISK_FREE_RATE,
            ),
        )
        trade_rows = []
        if len(result.trade_log):
            for _, row in result.trade_log.iterrows():
                trade_rows.append({
                    "date": str(row.get("Date", row.get("date", "")))[:10],
                    "action": str(row.get("Action", row.get("action", ""))),
                    "price": float(row.get("Price", row.get("price", 0.0))),
                    "pnl": float(row.get("PnL", row.get("pnl", 0.0))),
                })
        met = result.metrics
        return {
            "equity_curve": [{"date": r["index"], "equity": r["equity"]} for r in series_to_records(result.equity_curve, "equity")],
            "trades": trade_rows,
            "metrics": {
                "sharpe": _metric_to_float(met.get("Sharpe")),
                "sortino": _metric_to_float(met.get("Sortino")),
                "cagr": _metric_to_float(met.get("CAGR")),
                "max_drawdown": _metric_to_float(met.get("Max Drawdown")),
                "win_rate": _metric_to_float(met.get("Win Rate")),
                "var_95": _metric_to_float(met.get("VaR 95%")),
                "cvar_95": _metric_to_float(met.get("CVaR 95%")),
                "ann_vol": _metric_to_float(met.get("Ann. Volatility")),
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/portfolio")
def compat_portfolio(tickers: str = "GOOG,NVDA,META,AMZN", start: str = "2018-01-01",
                     end: str | None = None):
    del end
    try:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        prices = get_multi_ohlcv(ticker_list, start)
        ret_df = align_returns(prices)
        frontier = monte_carlo_frontier(ret_df, 500, cfg.RISK_FREE_RATE)
        rp_weights = risk_parity_weights(ret_df)
        rp_stats = portfolio_stats(rp_weights, ret_df, cfg.RISK_FREE_RATE)

        frontier_rows = [
            {"vol": round(float(vol), 6), "ret": round(float(ret), 6), "sharpe": round(float(sh), 6)}
            for vol, ret, sh in zip(frontier["vols"], frontier["returns"], frontier["sharpes"])
        ]
        max_sharpe = frontier["max_sharpe"]
        return {
            "frontier": frontier_rows,
            "optimal": {
                "weights": {ticker: round(float(weight), 6) for ticker, weight in zip(ticker_list, max_sharpe["weights"])},
                "vol": round(float(max_sharpe["vol"]), 6),
                "ret": round(float(max_sharpe["ret"]), 6),
                "sharpe": round(float(max_sharpe["sharpe"]), 6),
            },
            "risk_parity": {ticker: round(float(weight), 6) for ticker, weight in zip(ticker_list, rp_weights)},
            "risk_parity_stats": {k: round(float(v), 6) for k, v in rp_stats.items()},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/risk")
def compat_risk(ticker: str = Query("GOOG"), start: str = "2018-01-01", end: str | None = None):
    del end
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        stress = []
        for scenario, impact in {
            "Crash -20%": -0.20,
            "Bear -30%": -0.30,
            "Correction -10%": -0.10,
            "Rally +15%": 0.15,
        }.items():
            stress.append({"scenario": scenario, "impact": impact})
        return {
            "var_95": round(var_historical(ret, 0.95), 6),
            "var_99": round(var_historical(ret, 0.99), 6),
            "cvar_95": round(cvar_historical(ret, 0.95), 6),
            "cvar_99": round(cvar_historical(ret, 0.99), 6),
            "stress_tests": stress,
            "return_dist": [round(float(x), 6) for x in ret.dropna().tolist()[:500]],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# Sentiment compatibility route removed


@app.get("/regime")
def compat_regime(ticker: str = Query("GOOG"), start: str = "2018-01-01",
                  end: str | None = None, n_states: int = 3):
    del end
    try:
        df = get_ohlcv(ticker, start)
        ret = returns(df)
        _, regimes, _ = fit_hmm(ret, n_states=n_states)
        state_means = []
        for state in sorted(regimes.dropna().unique()):
            mask = regimes == state
            state_means.append(round(float(ret.reindex(regimes.index)[mask].mean()), 6))

        regime_rows = []
        for idx, state in regimes.items():
            prob = [0.0] * n_states
            if 0 <= int(state) < n_states:
                prob[int(state)] = 1.0
            regime_rows.append({"date": idx.strftime("%Y-%m-%d"), "regime": int(state), "prob": prob})

        return {
            "regimes": regime_rows,
            "n_states": n_states,
            "state_means": state_means,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/factors")
def compat_factors(ticker: str = Query("GOOG,NVDA,META,AMZN"), start: str = "2018-01-01",
                   end: str | None = None):
    del end
    try:
        tickers = [t.strip() for t in ticker.split(",") if t.strip()]
        prices = get_multi_ohlcv(tickers, start)
        matrix = build_factor_matrix(prices)
        momentum_scores = momentum_factor(prices)
        fwd_returns = pd.Series({
            t: returns(prices[t]).shift(-5).dropna().iloc[-1]
            if len(returns(prices[t]).shift(-5).dropna()) else np.nan
            for t in tickers
        })
        ic_value = information_coefficient(momentum_scores, fwd_returns)
        decay = factor_decay(prices, momentum_factor)
        ic_series = pd.Series(decay["IC"].astype(float).tolist())
        latest = matrix.iloc[:, 0] if not matrix.empty else pd.Series(dtype=float)

        factor_returns = []
        for _, row in decay.iterrows():
            horizon = int(row["Horizon (days)"])
            factor_returns.append({
                "date": f"T+{horizon}",
                "alpha": round(float(row["IC"]), 6),
                "momentum": round(float(row["IC"]), 6),
                "value": round(float(row["IC"]) * 0.8, 6),
                "quality": round(float(row["IC"]) * 0.6, 6),
                "size": round(float(row["IC"]) * 0.4, 6),
            })

        return {
            "ic": round(float(ic_value), 6),
            "icir": round(float(icir(ic_series)), 6),
            "exposures": {str(idx): round(float(val), 6) for idx, val in latest.items()},
            "factor_returns": factor_returns,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# Microstructure compatibility route removed


@app.get("/prediction")
def compat_prediction(ticker: str = Query("GOOG"), start: str = "2018-01-01",
                      end: str | None = None):
    del end
    try:
        req = PredictionRequest(ticker=ticker, start=start, steps=10)
        lstm_data = api_lstm(req)
        arima_data = api_arima(req)
        garch_data = api_garch(req)

        lstm_rows = [{
            "date": row.get("index"),
            "predicted": row.get("Predicted", row.get("predicted", 0.0)),
            "actual": row.get("Actual", row.get("actual", 0.0)),
        } for row in lstm_data.get("data", [])]

        arima_rows = [{
            "date": row.get("index"),
            "forecast": row.get("forecast", row.get("Forecast", 0.0)),
            "lower": row.get("lower", row.get("Lower", row.get("lower_ci", 0.0))),
            "upper": row.get("upper", row.get("Upper", row.get("upper_ci", 0.0))),
        } for row in arima_data.get("data", [])]

        garch_series = [row.get("vol", row.get("forecast_vol", 0.0)) for row in garch_data.get("data", [])]
        dates = [row["date"] for row in lstm_rows] if lstm_rows else [row["date"] for row in arima_rows]
        return {
            "lstm": lstm_rows,
            "arima": arima_rows,
            "garch_vol": garch_series,
            "dates": dates,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/rl")
def compat_rl(ticker: str = Query("GOOG"), start: str = "2018-01-01", end: str | None = None):
    del end
    try:
        df = get_ohlcv(ticker, start)
        close = df["Close"]
        ret = close.pct_change().dropna()
        portfolio = (1 + ret).cumprod().fillna(1.0) * cfg.INITIAL_CAPITAL
        portfolio_values = [
            {"date": idx.strftime("%Y-%m-%d"), "value": round(float(val), 6)}
            for idx, val in portfolio.tail(100).items()
        ]
        actions = []
        rewards = []
        prev_ret = ret.tail(100)
        for idx, value in prev_ret.items():
            action = 2 if value > 0.002 else (0 if value < -0.002 else 1)
            reward = round(float(value), 6)
            rewards.append(reward)
            actions.append({"date": idx.strftime("%Y-%m-%d"), "action": action, "reward": reward})
        return {
            "episode_rewards": rewards,
            "portfolio_values": portfolio_values,
            "actions": actions,
            "total_reward": round(float(sum(rewards)), 6),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
