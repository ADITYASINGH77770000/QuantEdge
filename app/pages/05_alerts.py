# """app/pages/05_alerts.py — Price alert system (credentials secured via .env)."""
# import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# import streamlit as st
# import pandas as pd

# from core.data import get_ohlcv
# from app.data_engine import (
#     render_data_engine_controls,
#     render_single_ticker_input,
#     load_ticker_data,
#     get_global_start_date,
# )
# from utils.notifications import send_email, build_alert_body
# from utils.config import cfg

# st.set_page_config(page_title="Alerts | QuantEdge", layout="wide")
# st.title("🔔 Price Alert System")
# st.caption("Credentials secured via .env — no hardcoded passwords.")

# EDUCATIONAL_INSIGHTS = {
#     "GOOG": {"Open":"Opening price reflects initial market sentiment.",
#              "Close":"Closing price is the day's final market consensus.",
#              "High":"New highs signal bullish momentum.",
#              "Low":"New lows suggest selling pressure."},
#     "NVDA": {"Open":"Higher open = strong pre-market demand.",
#              "Close":"Watch close vs prior close for trend.",
#              "High":"Breaking highs often attracts more buyers.",
#              "Low":"Drop may trigger stop-losses."},
#     "META": {"Open":"Influenced by global social-media sentiment.",
#              "Close":"Key indicator of day's investor behaviour.",
#              "High":"New high = strong market support.",
#              "Low":"Drop may signal growth concerns."},
#     "AMZN": {"Open":"Reflects earnings expectations.",
#              "Close":"Vital for trend analysis.",
#              "High":"Indicates confidence in business model.",
#              "Low":"May signal retail sector headwinds."},
# }

# # ── Threshold editor ──────────────────────────────────────────────────────────
# st.subheader("Configure Thresholds")
# render_data_engine_controls("alerts")
# ticker = render_single_ticker_input("Ticker", key="alerts_ticker", default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"))
# start = pd.to_datetime(get_global_start_date())

# default_thresholds = {
#     "GOOG": {"Open": 150.0, "Close": 155.0, "High": 160.0, "Low": 140.0},
#     "NVDA": {"Open": 800.0, "Close": 820.0, "High": 850.0, "Low": 780.0},
#     "META": {"Open": 490.0, "Close": 500.0, "High": 510.0, "Low": 470.0},
#     "AMZN": {"Open": 185.0, "Close": 190.0, "High": 195.0, "Low": 175.0},
# }

# cols = st.columns(4)
# thresholds = {}
# for i, metric in enumerate(["Open","Close","High","Low"]):
#     thresholds[metric] = cols[i].number_input(
#         f"{metric} Threshold ($)",
#         value=default_thresholds[ticker][metric])

# # ── Check button ──────────────────────────────────────────────────────────────
# if st.button("Check & Send Alerts", type="primary"):
#     with st.spinner("Fetching latest data..."):
#         df = load_ticker_data(ticker)

#     latest = df.iloc[-1]
#     st.subheader(f"Latest Data — {ticker} ({df.index[-1].date()})")
#     st.dataframe(latest.rename("Value").to_frame().T, use_container_width=True)

#     alerts_sent = 0
#     for metric, threshold in thresholds.items():
#         price = float(latest[metric])
#         if price > threshold:
#             st.warning(f"⚠️ **{ticker} {metric}** = ${price:,.2f} > threshold ${threshold:,.2f}")
#             insight = EDUCATIONAL_INSIGHTS.get(ticker, {}).get(metric, "")
#             body    = build_alert_body(ticker, metric, price, threshold, insight)
#             ok      = send_email(f"QuantEdge Alert: {ticker} {metric}", body)
#             if ok:
#                 st.success(f"📧 Email sent for {metric}")
#                 alerts_sent += 1
#             else:
#                 st.info("📧 Email not sent (credentials not configured in .env)")
#         else:
#             st.info(f"✅ {ticker} {metric} = ${price:,.2f} — below threshold ${threshold:,.2f}")

#     if alerts_sent == 0 and not cfg.GMAIL_PASSWORD:
#         st.info("💡 To enable email alerts, add GMAIL_SENDER / GMAIL_PASSWORD to your .env file.")

# # ── Credentials status ────────────────────────────────────────────────────────
# with st.expander("📋 Credential Status"):
#     st.write(f"GMAIL_SENDER configured: **{'✅' if cfg.GMAIL_SENDER else '❌'}**")
#     st.write(f"GMAIL_PASSWORD configured: **{'✅' if cfg.GMAIL_PASSWORD else '❌'}**")
#     st.caption("Set these in your .env file — never hardcode credentials.")

"""
app/pages/05_alerts.py
──────────────────────────────────────────────────────────────────────────────
QuantEdge Alert Dashboard — fully wired to live data pipeline.

Uses:
  - core/alerts.py     AlertEngine + build_alert_data
  - core/data.py       get_ohlcv / returns
  - core/regime_detector.py  forward_regime_proba / full_regime_analysis / critical_slowing_down
  - core/metrics.py    sharpe / max_drawdown / var_historical
  - utils/notifications.py  send_email (if .env configured)
"""

import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from app.data_engine import (
    render_data_engine_controls,
    render_single_ticker_input,
    load_ticker_data,
    get_global_start_date,
)
from core import alerts as _alerts_module

# Prefer the canonical symbols but tolerate older/misaligned installs where the
# helper may be absent by providing a conservative fallback implementation.
AlertEngine = _alerts_module.AlertEngine
try:
    build_alert_data = _alerts_module.build_alert_data
except Exception:
    def build_alert_data(df, ticker: str = "N/A", **kwargs):
        """Minimal fallback for building alert input dict when core.alerts
        doesn't expose the full helper (keeps app usable during partial upgrades).
        """
        from core.metrics import var_historical, sharpe as _sharpe
        from core.data import returns as _returns
        import numpy as _np

        ret = _returns(df)

        # Drawdown (best-effort)
        try:
            cum = (1 + ret).cumprod()
            roll_max = cum.cummax()
            dd_series = (cum - roll_max) / roll_max
            current_dd = float(dd_series.iloc[-1]) if not dd_series.empty else 0.0
        except Exception:
            current_dd = 0.0

        current_var = var_historical(ret, 0.95) if len(ret) >= 30 else -0.05
        rolling_dd_p90 = 0.10
        rolling_var_p90 = 0.05
        rolling_sharpe = _sharpe(ret.tail(63)) if len(ret) >= 63 else _sharpe(ret)

        recent_vol = df.get("Volume")
        if recent_vol is None:
            current_volume = 0.0
            median_volume = 0.0
        else:
            recent_vol = recent_vol.replace(0, _np.nan).dropna()
            current_volume = float(recent_vol.iloc[-1]) if not recent_vol.empty else 0.0
            median_volume = float(recent_vol.tail(20).median()) if len(recent_vol) >= 5 else 0.0

        return {
            "ticker": ticker,
            "volume": current_volume,
            "avg_volume": median_volume,
            "drawdown": current_dd,
            "rolling_dd_p90": rolling_dd_p90,
            "var": current_var,
            "rolling_var_p90": rolling_var_p90,
            "sharpe": rolling_sharpe,
            "bear_prob": kwargs.get("bear_prob"),
            "prev_bear_prob": kwargs.get("prev_bear_prob"),
            "current_regime": kwargs.get("current_regime"),
            "prev_regime": kwargs.get("prev_regime"),
            "regime_age": kwargs.get("regime_age", 0),
            "early_warning": kwargs.get("early_warning", False),
            "signal_score": 0.0,
            "confidence": kwargs.get("confidence", 0.0),
            "pred_direction": kwargs.get("pred_direction", ""),
            "action": kwargs.get("action"),
            "position_size": kwargs.get("position_size", 0.0),
            "old_weights": kwargs.get("old_weights", {}) or {},
            "new_weights": kwargs.get("new_weights", {}) or {},
        }
from core.data import returns as returns_fn
from utils.config import cfg
from utils.notifications import send_email, build_alert_body
try:
    from utils.theme import qe_neon_divider, qe_faq_section
except ImportError:
    from utils.theme import qe_neon_divider

    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        qe_neon_divider()
        st.markdown(f"### {title}")
        for question, answer in faqs:
            st.markdown(
                f"""
                <div style="
                    background: rgba(14,22,42,0.82);
                    border: 1px solid rgba(11,224,255,0.18);
                    border-radius: 12px;
                    padding: 14px 16px;
                    margin: 10px 0;
                ">
                  <div style="font-weight:700;color:#e8f4fd;margin-bottom:6px;">Q. {question}</div>
                  <div style="color:var(--text-dim);line-height:1.55;">A. {answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.set_page_config(page_title="Alerts | QuantEdge", layout="wide")

# ── Minimal inline dark styling ───────────────────────────────────────────────
st.markdown("""
<style>
.alert-card {
    border-radius: 10px; padding: 12px 18px; margin: 6px 0;
    border-left: 4px solid; font-size: 14px;
}
.level-CRITICAL { background:#1a0a0a; border-color:#e74c3c; color:#f5b7b1; }
.level-HIGH     { background:#1a130a; border-color:#e67e22; color:#f0c080; }
.level-MEDIUM   { background:#0a1220; border-color:#3498db; color:#a9cce3; }
.level-INFO     { background:#0a1a0a; border-color:#27ae60; color:#a9dfbf; }
.cooldown-tag   { font-size:11px; color:#666; float:right; }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Alert Dashboard")
qe_neon_divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
render_data_engine_controls("alerts")
ticker = render_single_ticker_input(
    "Ticker", key="alerts_ticker",
    default=cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG",
)
start = pd.to_datetime(get_global_start_date())


def _build_alert_engine_config(
    signal_threshold: float,
    bear_prob_threshold: float,
    drawdown_multiplier: float,
    var_multiplier: float,
    sharpe_minimum: float,
    volume_spike_multiplier: float,
) -> dict:
    # Include both current and legacy keys so the page remains compatible with
    # older AlertEngine implementations that expect static limit names.
    return {
        "signal_threshold": signal_threshold,
        "bear_prob_threshold": bear_prob_threshold,
        "drawdown_multiplier": drawdown_multiplier,
        "drawdown_fallback": 0.10,
        "drawdown_limit": 0.10,
        "var_multiplier": var_multiplier,
        "var_fallback": 0.05,
        "var_limit": 0.05,
        "sharpe_min": sharpe_minimum,
        "confidence_threshold": 0.85,
        "volume_spike_multiplier": volume_spike_multiplier,
        "drift_relative_threshold": 0.25,
        "drift_absolute_min": 0.03,
    }


def _normalize_alert_item(item: object) -> dict:
    """Backwards-compatible alert shape normalizer."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    if isinstance(item, dict):
        normalized = dict(item)
    else:
        normalized = {"message": str(item)}

    normalized["type"] = str(normalized.get("type") or "Alert")
    normalized["message"] = str(normalized.get("message") or "No message provided")
    normalized["level"] = str(normalized.get("level") or "INFO").upper()
    normalized["timestamp"] = str(normalized.get("timestamp") or timestamp)
    return normalized

# Alert engine config overrides
with st.sidebar.expander("⚙️ Threshold Config", expanded=False):
    sig_thresh   = st.slider("Signal threshold",      0.50, 1.00, 0.80, 0.05)
    bear_thresh  = st.slider("Bear prob threshold",   0.50, 0.95, 0.70, 0.05)
    dd_mult      = st.slider("Drawdown multiplier",   1.0,  4.0,  2.0, 0.5)
    var_mult     = st.slider("VaR multiplier",        1.0,  3.0,  1.5, 0.5)
    sharpe_min   = st.slider("Min Sharpe",           -1.0,  3.0,  1.0, 0.25)
    vol_mult     = st.slider("Volume spike ×",        1.5,  5.0,  2.5, 0.5)

with st.sidebar.expander("🔕 Cooldown Settings (seconds)", expanded=False):
    cd_risk      = st.number_input("Risk / VaR",       value=14400, step=3600)
    cd_regime    = st.number_input("Regime / EW",      value=7200,  step=1800)
    cd_signal    = st.number_input("Signal / Pred",    value=3600,  step=900)
    cd_anomaly   = st.number_input("Anomaly",          value=1800,  step=600)

# ── Build AlertEngine ─────────────────────────────────────────────────────────
engine = AlertEngine(config=_build_alert_engine_config(
    signal_threshold=sig_thresh,
    bear_prob_threshold=bear_thresh,
    drawdown_multiplier=dd_mult,
    var_multiplier=var_mult,
    sharpe_minimum=sharpe_min,
    volume_spike_multiplier=vol_mult,
))
if hasattr(engine, "config") and isinstance(engine.config, dict):
    engine.config.setdefault("drawdown_limit", engine.config.get("drawdown_fallback", 0.10))
    engine.config.setdefault("var_limit", engine.config.get("var_fallback", 0.05))
if hasattr(engine, "set_cooldown") and callable(getattr(engine, "set_cooldown")):
    engine.set_cooldown("Risk",        int(cd_risk))
    engine.set_cooldown("VaR",         int(cd_risk))
    engine.set_cooldown("Regime",      int(cd_regime))
    engine.set_cooldown("EarlyWarning",int(cd_regime))
    engine.set_cooldown("Signal",      int(cd_signal))
    engine.set_cooldown("Prediction",  int(cd_signal))
    engine.set_cooldown("Anomaly",     int(cd_anomaly))
else:
    # Best-effort: if the implementation exposes an internal `_cooldowns`
    # mapping use it; otherwise attach one so later code can query cooldowns.
    try:
        engine._cooldowns
    except Exception:
        engine._cooldowns = {}
    engine._cooldowns.update({
        "Risk": int(cd_risk),
        "VaR": int(cd_risk),
        "Regime": int(cd_regime),
        "EarlyWarning": int(cd_regime),
        "Signal": int(cd_signal),
        "Prediction": int(cd_signal),
        "Anomaly": int(cd_anomaly),
    })

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} data…"):
    try:
        df = load_ticker_data(ticker, start=str(start))
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

if df is None or df.empty:
    st.warning(f"No data available for {ticker}")
    st.stop()

ret = returns_fn(df)

# ── Regime analysis ───────────────────────────────────────────────────────────
bear_prob = prev_bear_prob = None
current_regime = prev_regime = None
regime_age = 0
early_warning_flag = False

try:
    from core.regime_detector import (
        fit_hmm, forward_regime_proba, compute_regime_age,
        critical_slowing_down,
    )
    model, regimes, label_map = fit_hmm(ret, n_states=2, df=df)
    fwd = forward_regime_proba(ret, model, df, label_map)

    # Resolve bear column name
    bear_col = next((c for c in fwd.columns if "Bear" in c), None)
    if bear_col and len(fwd) >= 2:
        bear_prob      = float(fwd[bear_col].iloc[-1])
        prev_bear_prob = float(fwd[bear_col].iloc[-2])

    if not regimes.empty:
        current_regime = str(regimes.iloc[-1])
        prev_regime    = str(regimes.iloc[-2]) if len(regimes) >= 2 else current_regime
        regime_age     = compute_regime_age(regimes)

    ew = critical_slowing_down(ret)
    early_warning_flag = bool(ew.get("active", False))

except Exception as e:
    logger.debug(f"Regime detection skipped: {e}")

# ── Build alert data dict from real pipeline ──────────────────────────────────
alert_data = build_alert_data(
    df=df,
    ticker=ticker,
    bear_prob=bear_prob,
    prev_bear_prob=prev_bear_prob,
    current_regime=current_regime,
    prev_regime=prev_regime,
    regime_age=regime_age,
    early_warning=early_warning_flag,
)

# ── Generate alerts ────────────────────────────────────────────────────────────
raw_alerts = engine.generate_alerts(alert_data)
if raw_alerts is None:
    raw_alerts = []
elif not isinstance(raw_alerts, list):
    raw_alerts = [raw_alerts]
alerts = [_normalize_alert_item(a) for a in raw_alerts]

# ── Layout: 3 columns ────────────────────────────────────────────────────────
col_live, col_metrics, col_log = st.columns([2, 1.2, 1.4])

# ── Column 1: Live alerts ─────────────────────────────────────────────────────
with col_live:
    st.subheader("📢 Active Alerts")

    if alerts:
        for a in alerts:
            lvl = a["level"]
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "INFO": "🟢"}.get(lvl, "⚪")
            st.markdown(
                f'<div class="alert-card level-{lvl}">'
                f'{icon} <strong>[{lvl}] {a["type"]}</strong><br>'
                f'{a["message"]}'
                f'<span class="cooldown-tag">{a["timestamp"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No alerts triggered — all checks within thresholds")

    # Manual email test
    st.divider()
    if st.button("📧 Test Email Alert", help="Sends a test email if credentials are in .env"):
        ok = send_email(
            "QuantEdge Test Alert",
            build_alert_body(ticker, "Test", 0, 0, "Manual test from dashboard"),
        )
        if ok:
            st.success("Email sent successfully")
        else:
            st.info("Email not sent — check GMAIL_SENDER, GMAIL_PASSWORD, and GMAIL_RECEIVER in .env")

    # Cooldown status
    with st.expander("🔕 Cooldown status"):
        if hasattr(engine, "cooldown_status") and callable(getattr(engine, "cooldown_status")):
            cd = engine.cooldown_status()
        else:
            cd = {}
            cooldowns = getattr(engine, "_cooldowns", {}) or {}
            last_fired = getattr(engine, "_last_fired", {}) or {}
            now_ts = time.time()
            if isinstance(cooldowns, dict):
                for alert_type, ttl in cooldowns.items():
                    try:
                        ttl_secs = int(ttl)
                    except Exception:
                        ttl_secs = 0
                    last_ts = float(last_fired.get(alert_type, 0.0))
                    cd[alert_type] = max(0, int(ttl_secs - (now_ts - last_ts)))

        if cd:
            for alert_type, secs in cd.items():
                bar = "✅ Ready" if secs == 0 else f"⏳ {secs//60}m {secs%60}s"
                st.text(f"{str(alert_type):15s}  {bar}")
        else:
            st.caption("Cooldown status unavailable for this AlertEngine version.")

# ── Column 2: Live metrics panel ──────────────────────────────────────────────
with col_metrics:
    st.subheader("📊 Current Risk Metrics")

    from core.metrics import sharpe as sharpe_fn, var_historical, max_drawdown as mdd_fn

    current_dd    = alert_data["drawdown"]
    p90_dd        = alert_data["rolling_dd_p90"]
    current_var   = alert_data["var"]
    p90_var       = alert_data["rolling_var_p90"]
    current_sharpe= alert_data["sharpe"]
    signal_score  = alert_data["signal_score"]

    def _color(val, good_above=None, bad_below=None):
        if good_above is not None and val >= good_above:
            return "🟢"
        if bad_below is not None and val <= bad_below:
            return "🔴"
        return "🟡"

    metrics_data = [
        ("Drawdown",       f"{current_dd:.2%}",    _color(current_dd, good_above=-0.03, bad_below=-p90_dd*dd_mult)),
        ("DD P90 thresh",  f"{-p90_dd*dd_mult:.2%}","ℹ️"),
        ("VaR (95%)",      f"{current_var:.2%}",   _color(current_var, good_above=-0.02, bad_below=-p90_var*var_mult)),
        ("VaR P90 thresh", f"{-p90_var*var_mult:.2%}","ℹ️"),
        ("Sharpe (63d)",   f"{current_sharpe:.2f}", _color(current_sharpe, good_above=1.5, bad_below=sharpe_min)),
        ("Signal Score",   f"{signal_score:+.3f}",  _color(signal_score, good_above=0.5, bad_below=-0.5)),
        ("Regime",         current_regime or "N/A", "📍"),
        ("Bear P",         f"{bear_prob:.2%}" if bear_prob is not None else "N/A",
                           _color(-(bear_prob or 0), good_above=-0.3, bad_below=-bear_thresh)),
        ("Regime age",     f"{regime_age}d",         "⏱"),
        ("Early warning",  "⚠️ YES" if early_warning_flag else "✅ No", ""),
    ]

    for label, value, icon in metrics_data:
        c1, c2 = st.columns([2, 1])
        c1.caption(label)
        c2.markdown(f"**{value}** {icon}")

    # Volume
    vol_now = alert_data["volume"]
    vol_med = alert_data["avg_volume"]
    if vol_med > 0:
        ratio = vol_now / vol_med
        st.caption("Volume vs 20d median")
        st.progress(min(ratio / (vol_mult * 1.5), 1.0), text=f"{ratio:.2f}×")

# ── Column 3: Alert history log ───────────────────────────────────────────────
with col_log:
    st.subheader("🗂 Alert History")
    log = engine.get_alert_log(n=30)
    if log:
        if not isinstance(log, list):
            log = [log]
        normalized_log = [_normalize_alert_item(entry) for entry in log]
        log_df = pd.DataFrame(normalized_log[::-1])   # newest first
        log_df = log_df[["timestamp", "level", "type", "message"]]
        log_df.columns = ["Time", "Level", "Type", "Message"]

        def _row_style(row):
            colors = {
                "CRITICAL": "background-color:#1a0a0a; color:#f5b7b1",
                "HIGH":     "background-color:#1a130a; color:#f0c080",
                "MEDIUM":   "background-color:#0a1220; color:#a9cce3",
                "INFO":     "background-color:#0a1a0a; color:#a9dfbf",
            }
            return [colors.get(row["Level"], "")] * len(row)

        st.dataframe(
            log_df.style.apply(_row_style, axis=1),
            use_container_width=True,
            height=420,
        )
        if st.button("🗑 Clear log"):
            try:
                from utils.config import cfg as _cfg
                log_path = (
                    Path(__file__).resolve().parents[2]
                    / "data" / "exports" / "alert_log.json"
                )
                log_path.write_text("[]")
                st.success("Log cleared")
                st.rerun()
            except Exception as ex:
                st.error(f"Could not clear log: {ex}")
    else:
        st.info("No alert history yet — alerts are logged here when triggered.")

# ── Bottom: credential status ─────────────────────────────────────────────────
with st.expander("📋 Credential & Config Status"):
    st.write(f"GMAIL_SENDER configured: **{'✅' if cfg.GMAIL_SENDER else '❌'}**")
    st.write(f"GMAIL_PASSWORD configured: **{'✅' if cfg.GMAIL_PASSWORD else '❌'}**")
    st.write(f"GMAIL_RECEIVER configured: **{'✅' if cfg.GMAIL_RECEIVER else '❌'}**")
    st.caption("Set these in your .env file — never hardcode credentials.")
    st.write(f"DEMO_MODE: **{cfg.DEMO_MODE}**")
    st.write(f"Ticker: **{ticker}** | Data rows: **{len(df)}** | Returns: **{len(ret)}**")

qe_faq_section("FAQs", [
    ("Why did my alert not send email?", "Check that GMAIL_SENDER, GMAIL_PASSWORD, and GMAIL_RECEIVER are set in .env. For Gmail, the password usually needs to be an app password."),
    ("What does the cooldown do?", "Cooldown prevents the same alert from firing repeatedly in a short period. That keeps the dashboard useful instead of noisy."),
    ("What is the dashboard watching?", "It watches price thresholds, risk metrics, regime changes, and signal conditions from the live data pipeline."),
    ("How do I test email alerts safely?", "Use the Test Email Alert button first. If that works, the SMTP setup is correct and real alerts should also send."),
])
