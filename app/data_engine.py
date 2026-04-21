"""Shared Streamlit data engine controls, ticker inputs, and session-backed loaders."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.data import get_live_data, get_multi_ohlcv, get_ohlcv
from utils.config import cfg


_DATA_STORE_KEY = "_qe_data_store"
_ENGINE_STATE_KEY = "_qe_data_engine_state"
_TICKER_UNIVERSE_KEY = "_qe_ticker_universe"
_LIVE_INTERVAL_OPTIONS = ["1m", "2m", "5m", "15m", "30m", "60m"]
_LIVE_LOOKBACK_OPTIONS = ["1d", "2d", "5d"]


def _engine_defaults() -> dict[str, object]:
    return {
        "mode": "Live Mode" if cfg.LIVE_MODE_DEFAULT else "Static Mode",
        "live_interval": cfg.LIVE_INTERVAL if cfg.LIVE_INTERVAL in _LIVE_INTERVAL_OPTIONS else "1m",
        "lookback_period": cfg.LIVE_LOOKBACK_PERIOD if cfg.LIVE_LOOKBACK_PERIOD in _LIVE_LOOKBACK_OPTIONS else "1d",
        "refresh_seconds": max(cfg.LIVE_REFRESH_SECONDS, 1),
        "static_start_date": pd.to_datetime(cfg.DEFAULT_START).date(),
    }


def get_data_engine_settings() -> dict[str, object]:
    state = st.session_state.setdefault(_ENGINE_STATE_KEY, _engine_defaults())
    return {
        "mode": state["mode"],
        "live_mode": state["mode"] == "Live Mode",
        "live_interval": state["live_interval"],
        "lookback_period": state["lookback_period"],
        "refresh_seconds": int(state["refresh_seconds"]),
        "static_start_date": state["static_start_date"],
    }


def normalize_ticker(raw: str) -> str:
    text = (raw or "").strip().upper()
    return "".join(ch for ch in text if ch.isalnum() or ch in {".", "-", "_", "^", "="})


def parse_ticker_list(raw: str) -> list[str]:
    values = [normalize_ticker(chunk) for chunk in (raw or "").split(",")]
    seen: set[str] = set()
    parsed: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            parsed.append(value)
    return parsed


def _ticker_universe() -> list[str]:
    universe = st.session_state.setdefault(_TICKER_UNIVERSE_KEY, list(cfg.DEFAULT_TICKERS))
    seen: set[str] = set()
    unique: list[str] = []
    for item in universe:
        ticker = normalize_ticker(str(item))
        if ticker and ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)
    st.session_state[_TICKER_UNIVERSE_KEY] = unique
    return unique


def _remember_tickers(tickers: list[str]) -> None:
    if not tickers:
        return
    universe = _ticker_universe()
    for ticker in tickers:
        if ticker not in universe:
            universe.append(ticker)
    st.session_state[_TICKER_UNIVERSE_KEY] = universe


def render_single_ticker_input(
    label: str,
    *,
    key: str,
    default: str | None = None,
    container=None,
) -> str:
    target = container or st
    fallback = normalize_ticker(default or (cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG")) or "GOOG"
    raw = target.text_input(label, value=fallback, key=key, help="Type any ticker symbol, for example TSLA or AAPL.")
    ticker = normalize_ticker(raw) or fallback
    _remember_tickers([ticker])
    return ticker


def render_multi_ticker_input(
    label: str,
    *,
    key: str,
    default: list[str] | None = None,
    container=None,
) -> list[str]:
    target = container or st
    fallback_list = [normalize_ticker(ticker) for ticker in (default or cfg.DEFAULT_TICKERS)]
    fallback_list = [ticker for ticker in fallback_list if ticker]
    fallback = ",".join(fallback_list) if fallback_list else "GOOG"
    raw = target.text_input(
        label,
        value=fallback,
        key=key,
        help="Enter multiple tickers separated by commas, for example AAPL,MSFT,TSLA.",
    )
    tickers = parse_ticker_list(raw)
    if not tickers:
        tickers = fallback_list[:]
    _remember_tickers(tickers)
    return tickers


def _sync_engine_settings() -> None:
    st.session_state[_ENGINE_STATE_KEY] = {
        "mode": st.session_state["qe_data_mode"],
        "live_interval": st.session_state["qe_live_interval"],
        "lookback_period": st.session_state["qe_live_lookback"],
        "refresh_seconds": st.session_state["qe_refresh_seconds"],
        "static_start_date": st.session_state["qe_static_start_date"],
    }


def get_global_start_date() -> str:
    """Single global start date used by all pages in Static Mode."""
    settings = get_data_engine_settings()
    return str(settings["static_start_date"])


def _enable_timed_refresh(page_key: str, refresh_seconds: int) -> None:
    refresh_ms = int(refresh_seconds * 1000)
    autorefresh = getattr(st, "autorefresh", None)
    if callable(autorefresh):
        autorefresh(interval=refresh_ms, key=f"qe_autorefresh_{page_key}")
        return

    st.html(
        f"""
        <script>
        const key = "qe-refresh-{page_key}";
        const prior = window.sessionStorage.getItem(key);
        if (prior) {{
          clearTimeout(Number(prior));
        }}
        const handle = window.setTimeout(() => window.location.reload(), {refresh_ms});
        window.sessionStorage.setItem(key, String(handle));
        </script>
        """
    )


def render_data_engine_controls(page_key: str, *, auto_refresh: bool = True) -> dict[str, object]:
    """Render one shared set of data controls in the sidebar."""
    defaults = get_data_engine_settings()
    source_badge = "DEMO DATA" if cfg.DEMO_MODE else "LIVE REAL DATA"

    st.sidebar.subheader("Data Engine")
    st.sidebar.caption(f"Source Badge: {source_badge}")
    st.session_state.setdefault("qe_static_start_date", defaults["static_start_date"])
    st.sidebar.radio(
        "Update Mode",
        ["Static Mode", "Live Mode"],
        index=0 if defaults["mode"] == "Static Mode" else 1,
        key="qe_data_mode",
        on_change=_sync_engine_settings,
    )
    if st.session_state["qe_data_mode"] == "Static Mode":
        st.sidebar.date_input(
            "Static Start Date",
            value=st.session_state["qe_static_start_date"],
            key="qe_static_start_date",
            on_change=_sync_engine_settings,
        )
    st.sidebar.selectbox(
        "Live Interval",
        _LIVE_INTERVAL_OPTIONS,
        index=_LIVE_INTERVAL_OPTIONS.index(str(defaults["live_interval"])),
        key="qe_live_interval",
        disabled=st.session_state["qe_data_mode"] != "Live Mode",
        on_change=_sync_engine_settings,
    )
    st.sidebar.selectbox(
        "Live Fetch Window",
        _LIVE_LOOKBACK_OPTIONS,
        index=_LIVE_LOOKBACK_OPTIONS.index(str(defaults["lookback_period"])),
        key="qe_live_lookback",
        disabled=st.session_state["qe_data_mode"] != "Live Mode",
        on_change=_sync_engine_settings,
    )
    st.sidebar.slider(
        "Refresh Seconds",
        min_value=1,
        max_value=300,
        value=int(defaults["refresh_seconds"]),
        step=1,
        key="qe_refresh_seconds",
        disabled=st.session_state["qe_data_mode"] != "Live Mode",
        on_change=_sync_engine_settings,
    )

    settings = get_data_engine_settings()
    if settings["live_mode"]:
        st.sidebar.caption(
            f"Refreshing every {settings['refresh_seconds']}s with {settings['live_interval']} intraday updates."
        )
        if auto_refresh:
            _enable_timed_refresh(page_key, int(settings["refresh_seconds"]))
    else:
        st.sidebar.caption("Using cached historical data starting from the global Static Start Date.")

    return settings


def _single_store_key(ticker: str, start: str, end: str | None, settings: dict[str, object]) -> str:
    return "|".join(
        [
            "single",
            ticker,
            str(start),
            str(end or ""),
            str(settings["mode"]),
            str(settings["live_interval"]),
            str(settings["lookback_period"]),
        ]
    )


def _multi_store_key(tickers: list[str], start: str, end: str | None, settings: dict[str, object]) -> str:
    return "|".join(
        [
            "multi",
            ",".join(sorted(tickers)),
            str(start),
            str(end or ""),
            str(settings["mode"]),
            str(settings["live_interval"]),
            str(settings["lookback_period"]),
        ]
    )


def load_ticker_data(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load one ticker through the shared static/live engine."""
    start = start or get_global_start_date()
    settings = get_data_engine_settings()
    store = st.session_state.setdefault(_DATA_STORE_KEY, {})
    key = _single_store_key(ticker, start, end, settings)

    if settings["live_mode"]:
        existing = store.get(key)
        data = get_live_data(
            ticker,
            time_interval=str(settings["live_interval"]),
            existing_df=existing if isinstance(existing, pd.DataFrame) else None,
            start=start,
            end=end,
            lookback_period=str(settings["lookback_period"]),
        )
    else:
        data = get_ohlcv(ticker, start, end)

    store[key] = data
    return data


def load_multi_ticker_data(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a ticker universe through the shared static/live engine."""
    start = start or get_global_start_date()
    settings = get_data_engine_settings()
    store = st.session_state.setdefault(_DATA_STORE_KEY, {})
    key = _multi_store_key(tickers, start, end, settings)

    if settings["live_mode"]:
        existing = store.get(key)
        data = get_multi_ohlcv(
            tickers,
            start,
            end,
            live_mode=True,
            time_interval=str(settings["live_interval"]),
            lookback_period=str(settings["lookback_period"]),
            existing_data=existing if isinstance(existing, dict) else None,
        )
    else:
        data = get_multi_ohlcv(tickers, start, end)

    store[key] = data
    return data


def data_engine_status(df: pd.DataFrame) -> str:
    settings = get_data_engine_settings()
    if df.empty:
        return "No data available."

    source = str(df.attrs.get("data_source", "demo" if cfg.DEMO_MODE else "real")).lower()
    if source == "real":
        badge = "LIVE REAL DATA"
    elif source == "demo":
        badge = "DEMO DATA"
    elif source == "mixed":
        badge = "MIXED DATA (REAL + DEMO)"
    else:
        badge = "UNKNOWN DATA SOURCE"

    last_ts = df.index[-1]
    last_text = last_ts.strftime("%Y-%m-%d %H:%M")
    if settings["live_mode"]:
        return (
            f"{badge} | Live Mode | last updated bar {last_text} | interval {settings['live_interval']} | "
            f"refreshed {datetime.now().strftime('%H:%M:%S')}"
        )
    return f"{badge} | Static Mode | dataset through {last_text}"
