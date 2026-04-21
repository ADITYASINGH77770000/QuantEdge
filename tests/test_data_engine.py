"""Tests for the shared static/live data engine."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data as data_module
from app.data_engine import normalize_ticker, parse_ticker_list


def _daily_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000.0, 1200.0],
            "Adj Close": [101.0, 102.0],
        },
        index=index,
    )


def test_merge_ohlcv_deduplicates_and_sorts():
    existing = _daily_frame()
    incoming = pd.DataFrame(
        {
            "Open": [105.0, 101.5],
            "High": [107.0, 104.0],
            "Low": [104.0, 100.5],
            "Close": [106.0, 103.5],
            "Volume": [1400.0, 1300.0],
            "Adj Close": [106.0, 103.5],
        },
        index=pd.to_datetime(["2024-01-04", "2024-01-03"]),
    )

    merged = data_module._merge_ohlcv(existing, incoming)

    assert list(merged.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert float(merged.loc["2024-01-03", "Close"]) == 103.5


def test_aggregate_intraday_to_daily_rolls_up_ohlcv_correctly():
    intraday = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 103.0, 104.0],
            "Low": [99.5, 100.5, 101.5],
            "Close": [100.5, 102.5, 103.5],
            "Volume": [100.0, 200.0, 300.0],
            "Adj Close": [100.5, 102.5, 103.5],
        },
        index=pd.to_datetime(["2024-01-03 09:30", "2024-01-03 10:30", "2024-01-03 11:30"]),
    )

    daily = data_module._aggregate_intraday_to_daily(intraday)

    assert len(daily) == 1
    row = daily.iloc[0]
    assert float(row["Open"]) == 100.0
    assert float(row["High"]) == 104.0
    assert float(row["Low"]) == 99.5
    assert float(row["Close"]) == 103.5
    assert float(row["Volume"]) == 600.0


def test_get_live_data_updates_existing_frame_with_latest_intraday(monkeypatch):
    existing = _daily_frame()
    intraday = pd.DataFrame(
        {
            "Open": [102.1, 102.4],
            "High": [103.0, 104.2],
            "Low": [101.8, 102.0],
            "Close": [102.8, 103.9],
            "Volume": [300.0, 500.0],
            "Adj Close": [102.8, 103.9],
        },
        index=pd.to_datetime(["2024-01-03 09:30", "2024-01-03 15:59"]),
    )

    monkeypatch.setattr(data_module.cfg, "DEMO_MODE", False)
    monkeypatch.setattr(data_module, "_download_yfinance", lambda *args, **kwargs: intraday)

    updated = data_module.get_live_data(
        "GOOG",
        time_interval="1m",
        existing_df=existing,
        start="2024-01-02",
        lookback_period="1d",
    )

    assert list(updated.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    latest = updated.loc["2024-01-03"]
    assert float(latest["Open"]) == 102.1
    assert float(latest["High"]) == 104.2
    assert float(latest["Low"]) == 101.8
    assert float(latest["Close"]) == 103.9
    assert float(latest["Volume"]) == 800.0


def test_ticker_parsing_accepts_custom_symbols():
    assert normalize_ticker("  tsla ") == "TSLA"
    assert normalize_ticker("^nsei") == "^NSEI"
    assert parse_ticker_list("tsla, aapl,msft,TSLA") == ["TSLA", "AAPL", "MSFT"]
