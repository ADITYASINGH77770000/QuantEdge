# # core/alerts.py

# class AlertEngine:
#     def __init__(self, config=None):
#         self.config = config or {
#             "signal_threshold": 0.8,
#             "drawdown_limit": 0.1,
#             "var_limit": 0.05,
#             "sharpe_min": 1.0,
#             "confidence_threshold": 0.85,
#             "volume_spike_multiplier": 2.5
#         }

#     def generate_alerts(self, data):
#         alerts = []

#         # 1. SIGNAL ALERT
#         if data.get("signal_score", 0) > self.config["signal_threshold"]:
#             alerts.append({
#                 "type": "Signal",
#                 "message": f"Strong signal detected (Score: {data['signal_score']:.2f})",
#                 "level": "HIGH"
#             })

#         # 2. REGIME ALERT
#         if data.get("current_regime") != data.get("prev_regime"):
#             alerts.append({
#                 "type": "Regime",
#                 "message": f"Market regime changed: {data['prev_regime']} → {data['current_regime']}",
#                 "level": "MEDIUM"
#             })

#         # 3. RISK ALERT
#         if data.get("drawdown", 0) > self.config["drawdown_limit"]:
#             alerts.append({
#                 "type": "Risk",
#                 "message": f"Drawdown exceeded: {data['drawdown']:.2%}",
#                 "level": "CRITICAL"
#             })

#         if data.get("var", 0) > self.config["var_limit"]:
#             alerts.append({
#                 "type": "Risk",
#                 "message": f"VaR exceeded: {data['var']:.2%}",
#                 "level": "CRITICAL"
#             })

#         # 4. PERFORMANCE ALERT
#         if data.get("sharpe", 0) < self.config["sharpe_min"]:
#             alerts.append({
#                 "type": "Performance",
#                 "message": f"Sharpe dropped: {data['sharpe']:.2f}",
#                 "level": "MEDIUM"
#             })

#         # 5. PREDICTION ALERT
#         if data.get("confidence", 0) > self.config["confidence_threshold"]:
#             alerts.append({
#                 "type": "Prediction",
#                 "message": f"High confidence prediction ({data['confidence']:.2f})",
#                 "level": "HIGH"
#             })

#         # 6. TRADE ALERT
#         if data.get("action"):
#             alerts.append({
#                 "type": "Trade",
#                 "message": f"{data['action']} {data['ticker']} | Size: {data['position_size']:.2%}",
#                 "level": "INFO"
#             })

#         # 7. PORTFOLIO ALERT
#         old_w = data.get("old_weights", {})
#         new_w = data.get("new_weights", {})
#         for asset in new_w:
#             if abs(new_w[asset] - old_w.get(asset, 0)) > 0.05:
#                 alerts.append({
#                     "type": "Portfolio",
#                     "message": f"{asset}: {old_w.get(asset,0):.2%} → {new_w[asset]:.2%}",
#                     "level": "MEDIUM"
#                 })

#         # 8. ANOMALY ALERT
#         if data.get("volume", 0) > data.get("avg_volume", 1) * self.config["volume_spike_multiplier"]:
#             alerts.append({
#                 "type": "Anomaly",
#                 "message": f"Volume spike detected",
#                 "level": "HIGH"
#             })

#         return alerts

"""
core/alerts.py
──────────────────────────────────────────────────────────────────────────────
Production-grade AlertEngine for QuantEdge.

Upgrades over the original:
  1. COOLDOWN REGISTRY      — per-type TTL prevents alert storms
  2. DYNAMIC THRESHOLDS     — drawdown/VaR use rolling-quantile, not static
  3. REGIME PROBABILITY     — uses P(Bear) crossing threshold, not label flip
  4. RELATIVE DRIFT CHECK   — portfolio drift scaled to position size
  5. ROBUST VOLUME SPIKE    — 20-day median not mean (outlier-resistant)
  6. PRIORITY QUEUE         — CRITICAL suppresses redundant MEDIUM in same batch
  7. EMAIL HOOK             — CRITICAL/HIGH alerts route to notifications.py
  8. ALERT LOG              — persistent JSON log in data/exports/alert_log.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────

LEVELS     = ["CRITICAL", "HIGH", "MEDIUM", "INFO"]
LEVEL_RANK = {l: i for i, l in enumerate(LEVELS)}

_DEFAULT_COOLDOWNS: dict[str, int] = {
    "Risk":        14_400,
    "VaR":         14_400,
    "Regime":       7_200,
    "Signal":       3_600,
    "Prediction":   3_600,
    "Anomaly":      1_800,
    "Performance":  7_200,
    "Trade":            0,
    "Portfolio":    3_600,
    "EarlyWarning": 7_200,
}

_EMAIL_LEVELS = {"CRITICAL", "HIGH"}

_DEFAULT_CONFIG: dict[str, Any] = {
    "signal_threshold":         0.80,
    "bear_prob_threshold":      0.70,
    "regime_min_age":              3,
    "drawdown_multiplier":       2.0,
    "drawdown_fallback":         0.10,
    # Backwards-compatibility keys used by older callers
    "drawdown_limit":            0.10,
    "var_multiplier":            1.5,
    "var_fallback":              0.05,
    "var_limit":                 0.05,
    "sharpe_min":                1.0,
    "confidence_threshold":      0.85,
    "volume_spike_multiplier":   2.5,
    "drift_relative_threshold":  0.25,
    "drift_absolute_min":        0.03,
}


class AlertEngine:
    """
    Unified alert engine.  Call generate_alerts(data) each tick.

    data dict keys (all optional — missing keys are skipped):
      signal_score      float   combined alpha signal score  [-1, 1]
      bear_prob         float   P(Bear) from HMM forward pass [0, 1]
      prev_bear_prob    float   P(Bear) at previous tick
      current_regime    str     e.g. "Bull 📈"
      prev_regime       str
      regime_age        int     bars in current regime
      early_warning     bool    critical-slowing-down flag
      drawdown          float   current drawdown (negative, e.g. -0.12)
      rolling_dd_p90    float   rolling 90th-pct drawdown magnitude (positive)
      var               float   current 1-day VaR (negative)
      rolling_var_p90   float   rolling 90th-pct VaR magnitude (positive)
      sharpe            float   rolling Sharpe ratio
      confidence        float   ML prediction confidence [0, 1]
      pred_direction    str     "UP" / "DOWN"
      action            str     "BUY" / "SELL" / None
      ticker            str
      position_size     float
      old_weights       dict    {ticker: weight}
      new_weights       dict    {ticker: weight}
      volume            float   today's volume
      avg_volume        float   20-day median volume
    """

    def __init__(self, config: dict | None = None, log_path: Path | None = None):
        self.config: dict[str, Any] = _DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self._normalize_config()

        self._last_fired: dict[str, float] = {}
        self._cooldowns = _DEFAULT_COOLDOWNS.copy()

        self._log_path: Path = log_path or (
            Path(__file__).resolve().parents[1] / "data" / "exports" / "alert_log.json"
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_alerts(self, data: dict) -> list[dict]:
        self._normalize_config()
        raw: list[dict] = []
        raw += self._check_signal(data)
        raw += self._check_regime(data)
        raw += self._check_early_warning(data)
        raw += self._check_drawdown(data)
        raw += self._check_var(data)
        raw += self._check_sharpe(data)
        raw += self._check_prediction(data)
        raw += self._check_trade(data)
        raw += self._check_portfolio_drift(data)
        raw += self._check_volume_spike(data)

        filtered = [a for a in raw if self._can_fire(a["type"])]
        for a in filtered:
            self._last_fired[a["type"]] = time.time()

        alerts = self._deduplicate(filtered)
        alerts.sort(key=lambda a: LEVEL_RANK.get(a["level"], 99))

        self._log_alerts(alerts)
        self._maybe_email(alerts)
        return alerts

    def get_alert_log(self, n: int = 50) -> list[dict]:
        if not self._log_path.exists():
            return []
        try:
            with open(self._log_path) as f:
                return json.load(f)[-n:]
        except Exception:
            return []

    def set_cooldown(self, alert_type: str, seconds: int) -> None:
        self._cooldowns[alert_type] = seconds

    def time_until_next(self, alert_type: str) -> int:
        ttl = self._cooldowns.get(alert_type, 3600)
        return max(0, int(ttl - (time.time() - self._last_fired.get(alert_type, 0.0))))

    def cooldown_status(self) -> dict[str, int]:
        return {t: self.time_until_next(t) for t in _DEFAULT_COOLDOWNS}

    def _normalize_config(self) -> None:
        # Keep legacy/static and newer/dynamic config keys in sync so partially
        # specified overrides do not break older alert paths.
        drawdown_limit = self.config.get("drawdown_limit")
        drawdown_fallback = self.config.get("drawdown_fallback")
        if drawdown_limit is None and drawdown_fallback is None:
            drawdown_limit = drawdown_fallback = _DEFAULT_CONFIG["drawdown_limit"]
        elif drawdown_limit is None:
            drawdown_limit = drawdown_fallback
        elif drawdown_fallback is None:
            drawdown_fallback = drawdown_limit

        var_limit = self.config.get("var_limit")
        var_fallback = self.config.get("var_fallback")
        if var_limit is None and var_fallback is None:
            var_limit = var_fallback = _DEFAULT_CONFIG["var_limit"]
        elif var_limit is None:
            var_limit = var_fallback
        elif var_fallback is None:
            var_fallback = var_limit

        self.config["drawdown_limit"] = drawdown_limit
        self.config["drawdown_fallback"] = drawdown_fallback
        self.config["var_limit"] = var_limit
        self.config["var_fallback"] = var_fallback

        for key, value in _DEFAULT_CONFIG.items():
            self.config.setdefault(key, value)

    # ── Cooldown / dedup ────────────────────────────────────────────────────────

    def _can_fire(self, alert_type: str) -> bool:
        ttl = self._cooldowns.get(alert_type, 3600)
        return (time.time() - self._last_fired.get(alert_type, 0.0)) >= ttl

    @staticmethod
    def _deduplicate(alerts: list[dict]) -> list[dict]:
        best: dict[str, dict] = {}
        for a in alerts:
            t = a["type"]
            if t not in best or LEVEL_RANK[a["level"]] < LEVEL_RANK[best[t]["level"]]:
                best[t] = a
        return list(best.values())

    # ── Checks ─────────────────────────────────────────────────────────────────

    def _check_signal(self, d: dict) -> list[dict]:
        score = d.get("signal_score", 0.0)
        if score > self.config["signal_threshold"]:
            return [_alert("Signal", f"Strong bullish alpha signal (score={score:.3f})", "HIGH")]
        if score < -self.config["signal_threshold"]:
            return [_alert("Signal", f"Strong bearish signal (score={score:.3f}) — consider reducing exposure", "HIGH")]
        return []

    def _check_regime(self, d: dict) -> list[dict]:
        bear_prob      = d.get("bear_prob")
        prev_bear_prob = d.get("prev_bear_prob")
        regime_age     = d.get("regime_age", 99)
        alerts = []

        if bear_prob is not None and prev_bear_prob is not None:
            thresh = self.config["bear_prob_threshold"]
            if bear_prob >= thresh and prev_bear_prob < thresh:
                alerts.append(_alert(
                    "Regime",
                    f"Bear regime probability crossed {thresh:.0%} (P={bear_prob:.2%}) — reduce position size",
                    "HIGH",
                ))
            elif bear_prob < (1 - thresh) and prev_bear_prob >= (1 - thresh):
                alerts.append(_alert(
                    "Regime",
                    f"Bear probability falling ({bear_prob:.2%}) — possible regime recovery",
                    "MEDIUM",
                ))
        elif d.get("current_regime") and d.get("prev_regime"):
            if (d["current_regime"] != d["prev_regime"]
                    and regime_age >= self.config["regime_min_age"]):
                alerts.append(_alert(
                    "Regime",
                    f"Regime change: {d['prev_regime']} → {d['current_regime']} (age={regime_age}d)",
                    "MEDIUM",
                ))
        return alerts

    def _check_early_warning(self, d: dict) -> list[dict]:
        if d.get("early_warning", False):
            return [_alert(
                "EarlyWarning",
                "Critical slowing down detected — regime change possible within 10-20 days",
                "HIGH",
            )]
        return []

    def _check_drawdown(self, d: dict) -> list[dict]:
        dd = d.get("drawdown")
        if dd is None:
            return []
        dd_abs = abs(dd)
        p90    = d.get("rolling_dd_p90", self.config["drawdown_fallback"])
        thresh = self.config["drawdown_multiplier"] * max(p90, self.config["drawdown_fallback"])
        if dd_abs > thresh:
            return [_alert(
                "Risk",
                f"Drawdown {dd:.2%} exceeds dynamic threshold ({thresh:.2%} = "
                f"{self.config['drawdown_multiplier']}× P90={p90:.2%})",
                "CRITICAL",
            )]
        return []

    def _check_var(self, d: dict) -> list[dict]:
        var = d.get("var")
        if var is None:
            return []
        var_abs = abs(var)
        p90    = d.get("rolling_var_p90", self.config["var_fallback"])
        thresh = self.config["var_multiplier"] * max(p90, self.config["var_fallback"])
        if var_abs > thresh:
            return [_alert(
                "VaR",
                f"VaR {var:.2%} exceeds dynamic threshold ({thresh:.2%} = "
                f"{self.config['var_multiplier']}× P90={p90:.2%})",
                "CRITICAL",
            )]
        return []

    def _check_sharpe(self, d: dict) -> list[dict]:
        sharpe = d.get("sharpe")
        if sharpe is None:
            return []
        if sharpe < self.config["sharpe_min"]:
            return [_alert(
                "Performance",
                f"Sharpe ratio deteriorated to {sharpe:.2f} (min={self.config['sharpe_min']:.1f})",
                "MEDIUM",
            )]
        return []

    def _check_prediction(self, d: dict) -> list[dict]:
        conf = d.get("confidence", 0.0)
        if conf > self.config["confidence_threshold"]:
            direction = d.get("pred_direction", "")
            dir_str   = f" ({direction})" if direction else ""
            return [_alert(
                "Prediction",
                f"High-confidence ML prediction{dir_str} (confidence={conf:.2%})",
                "HIGH",
            )]
        return []

    def _check_trade(self, d: dict) -> list[dict]:
        if not d.get("action"):
            return []
        return [_alert(
            "Trade",
            f"{d['action']} {d.get('ticker','N/A')} | Size: {d.get('position_size',0):.2%}",
            "INFO",
        )]

    def _check_portfolio_drift(self, d: dict) -> list[dict]:
        old_w = d.get("old_weights", {})
        new_w = d.get("new_weights", {})
        if not new_w:
            return []
        rel_thresh = self.config["drift_relative_threshold"]
        abs_min    = self.config["drift_absolute_min"]
        alerts = []
        for asset in new_w:
            nw = new_w[asset]
            ow = old_w.get(asset, 0.0)
            abs_change = abs(nw - ow)
            rel_change = abs_change / max(abs(ow), 1e-6) if ow != 0 else 1.0
            if abs_change >= abs_min and rel_change >= rel_thresh:
                alerts.append(_alert(
                    "Portfolio",
                    f"{asset}: {ow:.2%} → {nw:.2%} (Δ={abs_change:.2%}, {rel_change:.0%} relative)",
                    "MEDIUM",
                ))
        return alerts

    def _check_volume_spike(self, d: dict) -> list[dict]:
        volume     = d.get("volume", 0.0)
        avg_volume = d.get("avg_volume")
        if avg_volume is None or avg_volume <= 0:
            return []
        mult = self.config["volume_spike_multiplier"]
        if volume > avg_volume * mult:
            ratio = volume / avg_volume
            return [_alert(
                "Anomaly",
                f"Volume spike: {volume:,.0f} = {ratio:.1f}× median ({avg_volume:,.0f})",
                "HIGH",
            )]
        return []

    # ── Side effects ───────────────────────────────────────────────────────────

    def _log_alerts(self, alerts: list[dict]) -> None:
        if not alerts:
            return
        try:
            existing: list[dict] = []
            if self._log_path.exists():
                with open(self._log_path) as f:
                    existing = json.load(f)
            existing.extend(alerts)
            with open(self._log_path, "w") as f:
                json.dump(existing[-500:], f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Alert log write failed: {e}")

    def _maybe_email(self, alerts: list[dict]) -> None:
        try:
            from utils.notifications import send_email, build_alert_body
            from utils.config import cfg
            if not cfg.GMAIL_PASSWORD:
                return
            for a in alerts:
                if a["level"] in _EMAIL_LEVELS:
                    send_email(
                        subject=f"QuantEdge {a['level']}: {a['type']} Alert",
                        body=build_alert_body(
                            ticker=a.get("ticker", "N/A"),
                            metric=a["type"],
                            price=0.0,
                            threshold=0.0,
                            insight=a["message"],
                        ),
                    )
        except Exception as e:
            logger.debug(f"Email alert skipped: {e}")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _alert(alert_type: str, message: str, level: str) -> dict:
    return {
        "type":      alert_type,
        "message":   message,
        "level":     level,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ── Data builder ───────────────────────────────────────────────────────────────

def build_alert_data(
    df: pd.DataFrame,
    ticker: str = "N/A",
    bear_prob: float | None = None,
    prev_bear_prob: float | None = None,
    current_regime: str | None = None,
    prev_regime: str | None = None,
    regime_age: int = 0,
    early_warning: bool = False,
    action: str | None = None,
    position_size: float = 0.0,
    old_weights: dict | None = None,
    new_weights: dict | None = None,
    confidence: float = 0.0,
    pred_direction: str = "",
) -> dict:
    """
    Derive all alert inputs from a live OHLCV DataFrame + regime/prediction extras.
    Returns a dict ready for AlertEngine.generate_alerts().
    """
    from core.metrics import var_historical, sharpe as sharpe_fn
    from core.data import returns as returns_fn

    ret = returns_fn(df)

    # Drawdown
    cum      = (1 + ret).cumprod()
    roll_max = cum.cummax()
    dd_series   = (cum - roll_max) / roll_max
    current_dd  = float(dd_series.iloc[-1]) if not dd_series.empty else 0.0
    rolling_dd_p90 = float(
        dd_series.abs().rolling(63, min_periods=21).quantile(0.90).iloc[-1]
    ) if len(dd_series) >= 21 else 0.10

    # VaR
    current_var = var_historical(ret, 0.95) if len(ret) >= 30 else -0.05
    rolling_var_p90 = float(
        ret.rolling(63, min_periods=21).apply(
            lambda x: abs(np.percentile(x, 5)), raw=True
        ).quantile(0.90)
    ) if len(ret) >= 21 else 0.05

    # Sharpe (63-day rolling)
    rolling_sharpe = sharpe_fn(ret.tail(63)) if len(ret) >= 63 else sharpe_fn(ret)

    # Volume — 20-day median
    recent_vol     = df["Volume"].replace(0, np.nan).dropna()
    current_volume = float(recent_vol.iloc[-1]) if not recent_vol.empty else 0.0
    median_volume  = float(recent_vol.tail(20).median()) if len(recent_vol) >= 5 else 0.0

    # Signal score from RSI + MACD
    try:
        from core.indicators import add_all_indicators, signal_rsi, signal_macd_crossover
        df_ind    = add_all_indicators(df)
        rsi_sig   = signal_rsi(df_ind).iloc[-1]
        macd_sig  = signal_macd_crossover(df_ind).iloc[-1]
        signal_score = float(np.clip((rsi_sig + macd_sig) / 2, -1, 1))
    except Exception:
        signal_score = 0.0

    return {
        "ticker":          ticker,
        "volume":          current_volume,
        "avg_volume":      median_volume,
        "drawdown":        current_dd,
        "rolling_dd_p90":  rolling_dd_p90,
        "var":             current_var,
        "rolling_var_p90": rolling_var_p90,
        "sharpe":          rolling_sharpe,
        "bear_prob":       bear_prob,
        "prev_bear_prob":  prev_bear_prob,
        "current_regime":  current_regime,
        "prev_regime":     prev_regime,
        "regime_age":      regime_age,
        "early_warning":   early_warning,
        "signal_score":    signal_score,
        "confidence":      confidence,
        "pred_direction":  pred_direction,
        "action":          action,
        "position_size":   position_size,
        "old_weights":     old_weights or {},
        "new_weights":     new_weights or {},
    }
