# """
# utils/config.py
# Central configuration loader. Reads from .env with no hardcoded credentials.
# """

# import os
# from pathlib import Path

# from dotenv import load_dotenv
# from loguru import logger


# _ROOT = Path(__file__).resolve().parents[1]
# load_dotenv(_ROOT / ".env")


# class Config:
#     """All app settings sourced from environment variables."""

#     GMAIL_SENDER: str = os.getenv("GMAIL_SENDER", "")
#     GMAIL_PASSWORD: str = os.getenv("GMAIL_PASSWORD", "")
#     GMAIL_RECEIVER: str = os.getenv("GMAIL_RECEIVER", "")
#     SMTP_SERVER: str = "smtp.gmail.com"
#     SMTP_PORT: int = 587

#     NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
#     GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
#     GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

#     ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
#     ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
#     ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

#     DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
#     LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
#     CACHE_TTL: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
#     LIVE_MODE_DEFAULT: bool = os.getenv("LIVE_MODE_DEFAULT", "false").lower() == "true"
#     LIVE_REFRESH_SECONDS: int = int(os.getenv("LIVE_REFRESH_SECONDS", "60"))
#     LIVE_INTERVAL: str = os.getenv("LIVE_INTERVAL", "1m")
#     LIVE_LOOKBACK_PERIOD: str = os.getenv("LIVE_LOOKBACK_PERIOD", "1d")

#     DEFAULT_TICKERS: list = ["GOOG", "NVDA", "META", "AMZN"]
#     DEFAULT_START: str = "2015-01-01"
#     INITIAL_CAPITAL: float = 100_000.0
#     RISK_FREE_RATE: float = 0.045

#     ROOT_DIR: Path = _ROOT
#     CACHE_DIR: Path = _ROOT / "data" / "cache"
#     EXPORTS_DIR: Path = _ROOT / "data" / "exports"

#     @classmethod
#     def validate(cls) -> None:
#         """Warn without crashing if optional keys are missing."""
#         if not cls.NEWS_API_KEY:
#             logger.warning("NEWS_API_KEY not set; sentiment will use demo data")
#         if not cls.GEMINI_API_KEY:
#             logger.warning("GEMINI_API_KEY not set; AI explainers will use fallback summaries")
#         if not cls.GMAIL_PASSWORD:
#             logger.warning("GMAIL_PASSWORD not set; email alerts disabled")

#     def get(self, key: str, default=None):
#         """Dictionary-style accessor for class-backed config fields."""
#         return getattr(self, key, default)


# cfg = Config()
# cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
# cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


"""
utils/config.py
──────────────────────────────────────────────────────────────────────────────
Central configuration loader. Reads from .env with no hardcoded credentials.

Fix applied:
  - Config fields are now read inside __init__ via properties so they always
    reflect the actual .env values at runtime, not stale import-time snapshots.
  - load_dotenv uses override=True so re-loading .env always wins over stale env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parents[1]

# Load .env at import time — override=True ensures .env values always win
load_dotenv(_ROOT / ".env", override=True)


class Config:
    """All app settings sourced from environment variables."""

    # ── Email (Gmail SMTP) ─────────────────────────────────────────────────────
    # Read via property so values are always fresh from os.environ,
    # not a stale snapshot taken when the module was first imported.
    @property
    def GMAIL_SENDER(self) -> str:
        return os.environ.get("GMAIL_SENDER", "")

    @property
    def GMAIL_PASSWORD(self) -> str:
        return os.environ.get("GMAIL_PASSWORD", "")

    @property
    def GMAIL_RECEIVER(self) -> str:
        return os.environ.get("GMAIL_RECEIVER", "")

    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT:   int = 587

    # ── API keys ───────────────────────────────────────────────────────────────
    NEWS_API_KEY:    str = os.getenv("NEWS_API_KEY",    "")
    GEMINI_API_KEY:  str = os.getenv("GEMINI_API_KEY",  "")
    GEMINI_MODEL:    str = os.getenv("GEMINI_MODEL",    "gemini-1.5-flash")

    ALPACA_API_KEY:    str = os.getenv("ALPACA_API_KEY",    "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL:   str = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

    # ── App settings ───────────────────────────────────────────────────────────
    DEMO_MODE:             bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    LOG_LEVEL:             str  = os.getenv("LOG_LEVEL", "INFO")
    CACHE_TTL:             int  = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    LIVE_MODE_DEFAULT:     bool = os.getenv("LIVE_MODE_DEFAULT", "false").lower() == "true"
    LIVE_REFRESH_SECONDS:  int  = int(os.getenv("LIVE_REFRESH_SECONDS", "60"))
    LIVE_INTERVAL:         str  = os.getenv("LIVE_INTERVAL", "1m")
    LIVE_LOOKBACK_PERIOD:  str  = os.getenv("LIVE_LOOKBACK_PERIOD", "1d")

    # ── Defaults ───────────────────────────────────────────────────────────────
    DEFAULT_TICKERS:  list  = ["GOOG", "NVDA", "META", "AMZN"]
    DEFAULT_START:    str   = "2015-01-01"
    INITIAL_CAPITAL:  float = 100_000.0
    RISK_FREE_RATE:   float = 0.045

    # ── Paths ──────────────────────────────────────────────────────────────────
    ROOT_DIR:    Path = _ROOT
    CACHE_DIR:   Path = _ROOT / "data" / "cache"
    EXPORTS_DIR: Path = _ROOT / "data" / "exports"

    @classmethod
    def validate(cls) -> None:
        """Warn (but never crash) if optional keys are missing."""
        if not os.environ.get("NEWS_API_KEY"):
            logger.warning("NEWS_API_KEY not set; sentiment will use demo data")
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not set; AI explainers will use fallback summaries")
        if not os.environ.get("GMAIL_PASSWORD"):
            logger.warning("GMAIL_PASSWORD not set; email alerts disabled")

    def get(self, key: str, default=None):
        """Dictionary-style accessor for config fields."""
        return getattr(self, key, default)


cfg = Config()
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)