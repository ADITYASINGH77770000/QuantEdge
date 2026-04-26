"""Backwards-compatible wrapper for shared UI helpers."""

from app import shared as _shared

DARK_CSS = _shared.DARK_CSS
apply_theme = _shared.apply_theme
_start_str = _shared._start_str
_ticker_sb = _shared._ticker_sb
_tickers_sb = _shared._tickers_sb
_header = _shared._header
_sb_sec = _shared._sb_sec
_top_bar = _shared._top_bar

__all__ = [
    "DARK_CSS",
    "apply_theme",
    "_start_str",
    "_ticker_sb",
    "_tickers_sb",
    "_header",
    "_sb_sec",
    "_top_bar",
]
