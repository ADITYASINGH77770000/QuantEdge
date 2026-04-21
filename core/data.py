"""
Central data module for QuantEdge.

The app consumes one standard OHLCV structure from here in either:
- static mode: cached historical daily data
- live mode: cached historical daily data plus incremental intraday refreshes
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf
from loguru import logger

from utils.config import cfg


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
_CACHE_DIR = cfg.CACHE_DIR


def _set_data_source(df: pd.DataFrame, source: str, note: str | None = None) -> pd.DataFrame:
    frame = df.copy()
    frame.attrs["data_source"] = source
    if note:
        frame.attrs["data_source_note"] = note
    return frame


def _cache_path(ticker: str, start: str, end: str, interval: str = "1d") -> Path:
    key = hashlib.md5(f"{ticker}{start}{end}{interval}".encode()).hexdigest()[:10]
    return _CACHE_DIR / f"{ticker}_{interval}_{key}.parquet"


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _business_or_intraday_freq(interval: str) -> str:
    if interval.endswith("m"):
        return interval
    if interval.endswith("h"):
        return interval
    return "B"


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and enforce the QuantEdge OHLCV schema."""
    if df is None or df.empty:
        empty = pd.DataFrame(columns=OHLCV_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty

    source_meta = dict(getattr(df, "attrs", {}))
    frame = df.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels >= 2:
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = [" ".join(str(part) for part in col if part) for col in frame.columns]

    frame.columns = [str(col).strip() for col in frame.columns]
    if "Adj Close" not in frame.columns and "Close" in frame.columns:
        frame["Adj Close"] = frame["Close"]

    present = [column for column in OHLCV_COLUMNS if column in frame.columns]
    frame = frame[present].copy()

    for column in OHLCV_COLUMNS:
        if column not in frame.columns:
            frame[column] = frame["Close"] if column == "Adj Close" and "Close" in frame.columns else pd.NA

    frame = frame[OHLCV_COLUMNS]
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.attrs.update(source_meta)
    return frame


def _merge_ohlcv(existing: pd.DataFrame | None, incoming: pd.DataFrame | None) -> pd.DataFrame:
    """Append fresh rows, drop duplicates, and preserve chronological order."""
    existing_frame = _normalise_ohlcv(existing if existing is not None else pd.DataFrame())
    incoming_frame = _normalise_ohlcv(incoming if incoming is not None else pd.DataFrame())

    if existing_frame.empty:
        return incoming_frame
    if incoming_frame.empty:
        return existing_frame

    merged = pd.concat([existing_frame, incoming_frame])
    merged = merged.sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged = merged[OHLCV_COLUMNS]

    existing_source = str(existing_frame.attrs.get("data_source", "")).lower()
    incoming_source = str(incoming_frame.attrs.get("data_source", "")).lower()
    if existing_source and incoming_source and existing_source != incoming_source:
        return _set_data_source(merged, "mixed")
    if incoming_source:
        return _set_data_source(merged, incoming_source)
    if existing_source:
        return _set_data_source(merged, existing_source)
    return merged


def _aggregate_intraday_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convert recent intraday bars into replacement daily OHLCV rows."""
    intraday = _normalise_ohlcv(df)
    if intraday.empty:
        return intraday

    grouped = intraday.groupby(intraday.index.normalize()).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "Adj Close": "last",
        }
    )
    grouped.index = pd.DatetimeIndex(grouped.index, name="Date")
    grouped = _normalise_ohlcv(grouped)
    source = intraday.attrs.get("data_source")
    return _set_data_source(grouped, source) if source else grouped


def _demo_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demo mode."""
    import numpy as np

    freq = _business_or_intraday_freq(interval)
    dates = pd.date_range(start=start, end=end, freq=freq)
    if not len(dates):
        dates = pd.date_range(end=end, periods=1, freq=freq)

    n = len(dates)
    seed = abs(hash(f"{ticker}:{interval}")) % (2**32)
    rng = np.random.default_rng(seed)
    step_sigma = 0.015 if interval == "1d" else 0.0015
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, step_sigma, n)))

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.0025, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 50_000_000, n).astype(float),
            "Adj Close": close,
        },
        index=dates,
    )
    df.index.name = "Date"
    return _set_data_source(_normalise_ohlcv(df), "demo")


def _download_yfinance(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Thin yfinance wrapper so tests can patch a single fetch point."""
    kwargs = {
        "progress": False,
        "auto_adjust": False,
        "threads": False,
        "interval": interval,
        "group_by": "column",
    }
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    return _set_data_source(_normalise_ohlcv(yf.download(ticker, **kwargs)), "real")


@st.cache_data(ttl=cfg.CACHE_TTL, show_spinner=False)
def _get_static_ohlcv_cached(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    if cfg.DEMO_MODE:
        logger.debug(f"DEMO_MODE: generating synthetic data for {ticker} ({interval})")
        return _set_data_source(_demo_ohlcv(ticker, start, end, interval=interval), "demo", "demo_mode")

    cache_file = _cache_path(ticker, start, end, interval=interval)
    if cache_file.exists():
        logger.debug(f"Cache hit: {cache_file.name}")
        return _set_data_source(_normalise_ohlcv(pd.read_parquet(cache_file)), "real", "cache")

    try:
        df = _download_yfinance(ticker, start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df[OHLCV_COLUMNS].to_parquet(cache_file)
        logger.info(f"Downloaded & cached {ticker} ({len(df)} rows, interval={interval})")
        return _set_data_source(df, "real", "yfinance")
    except Exception as exc:
        logger.warning(f"yfinance failed for {ticker}: {exc} - using demo data")
        return _set_data_source(_demo_ohlcv(ticker, start, end, interval=interval), "demo", "fallback")


def get_ohlcv(
    ticker: str,
    start: str = cfg.DEFAULT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Historical daily OHLCV used as the canonical base dataset."""
    return _get_static_ohlcv_cached(ticker, start, end or _today(), interval="1d")


def get_live_data(
    ticker: str,
    time_interval: str = "1m",
    existing_df: pd.DataFrame | None = None,
    *,
    start: str = cfg.DEFAULT_START,
    end: str | None = None,
    lookback_period: str = "1d",
) -> pd.DataFrame:
    """
    Incrementally update a canonical daily OHLCV dataset from recent intraday bars.

    The returned DataFrame keeps the same daily OHLCV schema the rest of the app
    already expects, while refreshing the most recent session from intraday data.
    """
    end = end or _today()
    base = _normalise_ohlcv(existing_df) if existing_df is not None else get_ohlcv(ticker, start, end)

    if cfg.DEMO_MODE:
        demo_intraday = _demo_ohlcv(ticker, start=end, end=end, interval=time_interval)
        return _set_data_source(_merge_ohlcv(base, _aggregate_intraday_to_daily(demo_intraday)), "demo", "demo_mode")

    try:
        latest = _download_yfinance(ticker, period=lookback_period, interval=time_interval)
        latest_daily = _aggregate_intraday_to_daily(latest)
        if latest_daily.empty:
            return base
        return _set_data_source(_merge_ohlcv(base, latest_daily), "real", "live_refresh")
    except Exception as exc:
        logger.warning(f"Live refresh failed for {ticker}: {exc}")
        return base


def get_multi_ohlcv(
    tickers: list[str],
    start: str = cfg.DEFAULT_START,
    end: str | None = None,
    *,
    live_mode: bool = False,
    time_interval: str = "1m",
    lookback_period: str = "1d",
    existing_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers using the same centralized data path."""
    end = end or _today()
    if not live_mode:
        return {ticker: get_ohlcv(ticker, start, end) for ticker in tickers}

    existing_data = existing_data or {}
    return {
        ticker: get_live_data(
            ticker,
            time_interval=time_interval,
            existing_df=existing_data.get(ticker),
            start=start,
            end=end,
            lookback_period=lookback_period,
        )
        for ticker in tickers
    }


def returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Percentage returns for a price series."""
    return df[col].pct_change().dropna()


def rolling_returns(df: pd.DataFrame, window: int = 21, col: str = "Close") -> pd.Series:
    """Rolling n-period return."""
    return df[col].pct_change(window).dropna()


def align_returns(data: dict[str, pd.DataFrame], col: str = "Close") -> pd.DataFrame:
    """Align multiple tickers into a single returns DataFrame."""
    prices = pd.DataFrame({ticker: frame[col] for ticker, frame in data.items()})
    return prices.pct_change().dropna()


__all__ = [
    "OHLCV_COLUMNS",
    "_aggregate_intraday_to_daily",
    "_merge_ohlcv",
    "_normalise_ohlcv",
    "align_returns",
    "get_live_data",
    "get_multi_ohlcv",
    "get_ohlcv",
    "returns",
    "rolling_returns",
]
