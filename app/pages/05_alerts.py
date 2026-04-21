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

# app/pages/alerts.py

import streamlit as st
from core.alerts import AlertEngine

st.set_page_config(page_title="Alerts", layout="wide")

st.title("🚨 Quant Alerts System")

# Initialize engine
engine = AlertEngine()

# --- SAMPLE DATA (Replace with your real pipeline outputs) ---
data = {
    "signal_score": st.slider("Signal Score", 0.0, 1.0, 0.9),
    "current_regime": st.selectbox("Current Regime", ["bull", "bear", "sideways"]),
    "prev_regime": st.selectbox("Previous Regime", ["bull", "bear", "sideways"]),
    "drawdown": st.slider("Drawdown", 0.0, 0.2, 0.12),
    "var": st.slider("VaR", 0.0, 0.1, 0.06),
    "sharpe": st.slider("Sharpe Ratio", -1.0, 3.0, 0.8),
    "confidence": st.slider("Prediction Confidence", 0.0, 1.0, 0.92),
    "action": st.selectbox("Trade Action", ["BUY", "SELL", None]),
    "ticker": "RELIANCE",
    "position_size": 0.08,
    "old_weights": {"RELIANCE": 0.05},
    "new_weights": {"RELIANCE": 0.12},
    "volume": st.number_input("Volume", value=3000000),
    "avg_volume": st.number_input("Avg Volume", value=1000000),
}

# Generate alerts
alerts = engine.generate_alerts(data)

# --- DISPLAY ALERTS ---
st.subheader("📢 Active Alerts")

if alerts:
    for alert in alerts:
        if alert["level"] == "CRITICAL":
            st.error(f"[{alert['type']}] {alert['message']}")
        elif alert["level"] == "HIGH":
            st.warning(f"[{alert['type']}] {alert['message']}")
        elif alert["level"] == "MEDIUM":
            st.info(f"[{alert['type']}] {alert['message']}")
        else:
            st.success(f"[{alert['type']}] {alert['message']}")
else:
    st.success("No alerts triggered ✅")