# """QuantEdge dashboard — image-matched layout with all bug fixes applied.

# AI LAYER — Gemini AI Dashboard Decoder (bottom of page, same 3-layer design as portfolio):
#   Layer 1: Deterministic danger flags (always shown, no AI)
#             Checks: Sharpe < 0.5, Max Drawdown < -30%, Ann. Volatility > 40%,
#                     Win Rate < 45%, CVaR breach, Calmar < 0.5, data source = demo
#   Layer 2: Context builder + "Decode for Me" button
#             Packages all dashboard metrics, price action, volatility regime,
#             RSI reading, MACD signal, and data source into a structured JSON context.
#   Layer 3: Gemini output — structured 4-section explanation with mandatory disclaimer
#             Falls back to deterministic explanation if key missing or API call fails.

#   Uses GEMINI_API_KEY + GEMINI_MODEL from utils/config.py (already present in .env).
#   Architecture mirrors 08_portfolio.py exactly — same flag severity system,
#   same context-key cache-busting, same fallback chain.
# """

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# import json
# import warnings
# from urllib import error as urlerror
# from urllib import request as urlrequest

# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# try:
#     import streamlit as st
# except Exception:
#     from utils._stubs import st as st
# from plotly.subplots import make_subplots

# from app.data_engine import load_ticker_data
# from app.ui_pages._shared import (
#     _header,
#     _sb_sec,
#     _start_str,
#     _ticker_sb,
#     apply_theme,
# )
# from app.data_engine import render_data_engine_controls
# from core.data import returns
# from core.metrics import summary_table
# from utils.config import cfg

# try:
#     from utils.theme import qe_faq_section
# except ImportError:
#     def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
#         st.markdown("---")
#         st.markdown(f"### {title}")
#         for question, answer in faqs:
#             with st.expander(question):
#                 st.write(answer)


# st.set_page_config(page_title="Dashboard | QuantEdge", page_icon="📈", layout="wide")
# apply_theme()

# st.markdown(
#     """
# <style>
# section[data-testid="stSidebar"] {
#     background: #0f1623 !important;
#     border-right: 1px solid #1e2a3e !important;
# }
# section[data-testid="stSidebar"] .block-container {
#     padding-top: 1rem !important;
# }
# div[data-testid="stVerticalBlockBorderWrapper"] {
#     background: linear-gradient(160deg, #131c2e 0%, #0f1520 100%);
#     border: 1px solid #1e2d47 !important;
#     border-radius: 14px !important;
#     box-shadow: 0 4px 24px rgba(0,0,0,0.35);
# }
# div[data-testid="stVerticalBlockBorderWrapper"] > div {
#     padding: 0.5rem 0.65rem 0.65rem 0.65rem;
# }
# .qe-topbar { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
# .qe-topbar-pill {
#     display: flex; align-items: center; gap: 10px;
#     background: #131c2e; border: 1px solid #1e2d47; border-radius: 10px;
#     padding: 10px 18px; min-width: 160px; flex: 1;
# }
# .qe-topbar-label { color: #5a6a87; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
# .qe-topbar-val { color: #e2e8f4; font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
# .qe-topbar-badge { font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 5px; }
# .badge-green  { background: rgba(74,222,128,.15); color: #4ade80; }
# .badge-red    { background: rgba(248,113,113,.15); color: #f87171; }
# .badge-yellow { background: rgba(250,204,21,.13);  color: #facc15; }
# .badge-blue   { background: rgba(125,211,252,.13); color: #7dd3fc; }
# .qe-section-head { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
# .qe-section-head h3 { margin: 0; color: #c8d3e8; font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }
# .qe-section-dot { width: 8px; height: 8px; border-radius: 999px; background: #2d4060; box-shadow: 0 0 0 4px rgba(45,64,96,.2); }
# .qe-section-line { height: 1px; flex: 1; background: linear-gradient(to right, #1e2d47, transparent); }
# .qe-panel-title { color: #d4ddf0; font-size: 14px; font-weight: 700; margin-bottom: 2px; letter-spacing: -0.01em; }
# .qe-panel-sub { color: #4a5878; font-size: 11px; margin-bottom: 8px; }
# .qe-panel-rule { height: 1px; background: linear-gradient(to right, #1e2d47, transparent); margin: 0 0 10px 0; }
# .qe-stat-table { display: grid; gap: 7px; }
# .qe-stat-row {
#     display: flex; justify-content: space-between; align-items: center;
#     padding: 10px 14px; border: 1px solid #1a2840; border-radius: 9px;
#     background: linear-gradient(180deg, #111928 0%, #0d1421 100%);
# }
# .qe-stat-label { color: #c8d3e8; font-size: 13px; font-weight: 600; }
# .qe-stat-sub { color: #3d5070; font-size: 10px; margin-top: 2px; }
# .qe-stat-value { font-size: 15px; font-weight: 700; text-align: right; white-space: nowrap; }
# .qe-metric-pos     { color: #4ade80 !important; }
# .qe-metric-neg     { color: #f87171 !important; }
# .qe-metric-neutral { color: #facc15 !important; }
# .qe-metric-na      { color: #4a5878 !important; }
# [data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }
# [data-testid="stDataFrame"]   { border-radius: 10px; overflow: hidden; }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# # ══════════════════════════════════════════════════════════════════════════════
# # AI LAYER 1 — DETERMINISTIC DANGER FLAGS
# # Always runs before Gemini. Mirrors _compute_portfolio_danger_flags() exactly.
# # ══════════════════════════════════════════════════════════════════════════════

# def _compute_dashboard_danger_flags(
#     ticker: str,
#     met: dict,
#     last_close: float,
#     pct_chg: float,
#     rsi_last: float,
#     macd_last: float,
#     signal_last: float,
#     vol21_last: float,
#     data_source: str = "unknown",
# ) -> list[dict]:
#     """
#     Deterministic pre-flight checks for the dashboard output.
#     Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.

#     Thresholds:
#       Sharpe < 0:        DANGER  — negative risk-adjusted return
#       Sharpe 0–0.5:      WARNING — below investable minimum
#       Max Drawdown < -40%: DANGER
#       Max Drawdown < -25%: WARNING
#       Ann. Volatility > 50%: DANGER
#       Ann. Volatility > 35%: WARNING
#       Win Rate < 40%:    WARNING
#       Calmar < 0.5:      WARNING
#       CVaR 95% < -4%:    WARNING — high tail risk
#       RSI > 75:          WARNING — overbought
#       RSI < 25:          WARNING — oversold
#       MACD < Signal:     INFO    — bearish momentum
#       Vol21 > 40%:       WARNING — elevated vol regime
#     """
#     flags = []

#     def _parse_pct(key: str):
#         val = met.get(key, "N/A")
#         if val == "N/A":
#             return None
#         try:
#             return float(str(val).replace("%", "").strip()) / 100
#         except ValueError:
#             return None

#     def _parse_float(key: str):
#         val = met.get(key, "N/A")
#         if val == "N/A":
#             return None
#         try:
#             return float(str(val).replace("x", "").strip())
#         except ValueError:
#             return None

#     # ── Data quality ──────────────────────────────────────────────────────────
#     if data_source in ("demo", ""):
#         flags.append({
#             "severity": "INFO",
#             "code": "DEMO_DATA",
#             "message": (
#                 f"Dashboard analysis for {ticker} is running on SYNTHETIC demo data, "
#                 "not real market prices. All metrics — Sharpe, VaR, drawdown — are "
#                 "illustrative only. Do not make trading decisions based on this output."
#             ),
#         })

#     # ── Sharpe ────────────────────────────────────────────────────────────────
#     sharpe_val = _parse_float("Sharpe")
#     if sharpe_val is not None:
#         if sharpe_val < 0:
#             flags.append({
#                 "severity": "DANGER",
#                 "code": "NEGATIVE_SHARPE",
#                 "message": (
#                     f"Sharpe ratio is {sharpe_val:.2f} — negative. The asset is producing "
#                     "negative risk-adjusted returns over the selected period. Returns do not "
#                     "compensate for the volatility risk taken."
#                 ),
#             })
#         elif sharpe_val < 0.5:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "LOW_SHARPE",
#                 "message": (
#                     f"Sharpe ratio is {sharpe_val:.2f} — below the conventional minimum "
#                     "investable threshold of 0.5. Risk-adjusted performance is weak. "
#                     "Consider comparing against a benchmark before allocating."
#                 ),
#             })

#     # ── Max Drawdown ──────────────────────────────────────────────────────────
#     mdd = _parse_pct("Max Drawdown")
#     if mdd is not None:
#         if mdd < -0.40:
#             flags.append({
#                 "severity": "DANGER",
#                 "code": "SEVERE_DRAWDOWN",
#                 "message": (
#                     f"Maximum drawdown is {mdd:.2%} — catastrophic. The asset has lost more "
#                     "than 40% from peak at some point in the selected window. This level of "
#                     "loss is psychologically and financially very difficult to recover from."
#                 ),
#             })
#         elif mdd < -0.25:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "HIGH_DRAWDOWN",
#                 "message": (
#                     f"Maximum drawdown is {mdd:.2%}. A drawdown exceeding 25% indicates "
#                     "meaningful tail risk. Ensure position sizing accounts for this historical "
#                     "peak-to-trough decline."
#                 ),
#             })

#     # ── Annual Volatility ─────────────────────────────────────────────────────
#     ann_vol = _parse_pct("Ann. Volatility")
#     if ann_vol is not None:
#         if ann_vol > 0.50:
#             flags.append({
#                 "severity": "DANGER",
#                 "code": "EXTREME_VOLATILITY",
#                 "message": (
#                     f"Annualised volatility is {ann_vol:.2%} — extremely high. "
#                     "At this level, a single bad year could produce a 50%+ loss even "
#                     "without a structural blow-up. VaR and CVaR figures require special attention."
#                 ),
#             })
#         elif ann_vol > 0.35:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "HIGH_VOLATILITY",
#                 "message": (
#                     f"Annualised volatility is {ann_vol:.2%} — elevated, well above typical "
#                     "equity market volatility (~15-20%). Risk metrics should be interpreted carefully."
#                 ),
#             })

#     # ── Win Rate ──────────────────────────────────────────────────────────────
#     wr = _parse_pct("Win Rate")
#     if wr is not None and wr < 0.40:
#         flags.append({
#             "severity": "WARNING",
#             "code": "LOW_WIN_RATE",
#             "message": (
#                 f"Win rate is {wr:.2%} — fewer than 40% of trading days produced a "
#                 "positive return. Verify against Calmar and Sortino before assuming profitability."
#             ),
#         })

#     # ── Calmar ────────────────────────────────────────────────────────────────
#     calmar_val = _parse_float("Calmar")
#     if calmar_val is not None and calmar_val < 0.5:
#         flags.append({
#             "severity": "WARNING",
#             "code": "LOW_CALMAR",
#             "message": (
#                 f"Calmar ratio is {calmar_val:.2f} — below 0.5. "
#                 "The compound growth rate is low relative to the maximum drawdown. "
#                 "Return earned per unit of drawdown risk is weak."
#             ),
#         })

#     # ── CVaR breach ───────────────────────────────────────────────────────────
#     cvar_95 = _parse_pct("CVaR 95% (Hist)")
#     if cvar_95 is not None and cvar_95 < -0.04:
#         flags.append({
#             "severity": "WARNING",
#             "code": "HIGH_CVAR",
#             "message": (
#                 f"CVaR 95% (Expected Shortfall) is {cvar_95:.2%} per day. "
#                 "On the worst 5% of trading days, the average loss exceeds 4%. "
#                 "This is a meaningful tail-risk exposure that should inform position sizing."
#             ),
#         })

#     # ── RSI extremes ──────────────────────────────────────────────────────────
#     if rsi_last is not None and not np.isnan(rsi_last):
#         if rsi_last > 75:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "RSI_OVERBOUGHT",
#                 "message": (
#                     f"RSI(14) is {rsi_last:.1f} — deep overbought territory (>75). "
#                     "Short-term mean-reversion risk is elevated. "
#                     "Price may be running ahead of fundamentals."
#                 ),
#             })
#         elif rsi_last < 25:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "RSI_OVERSOLD",
#                 "message": (
#                     f"RSI(14) is {rsi_last:.1f} — deep oversold territory (<25). "
#                     "Capitulation risk is elevated. Watch for reversal signals before adding exposure."
#                 ),
#             })

#     # ── MACD bearish cross ────────────────────────────────────────────────────
#     if (macd_last is not None and signal_last is not None
#             and not np.isnan(macd_last) and not np.isnan(signal_last)):
#         if macd_last < signal_last:
#             flags.append({
#                 "severity": "INFO",
#                 "code": "MACD_BEARISH",
#                 "message": (
#                     f"MACD ({macd_last:.4f}) is below Signal ({signal_last:.4f}) — "
#                     "momentum is currently bearish. The short-term EMA is trailing the longer-term one."
#                 ),
#             })

#     # ── Elevated short-term volatility ────────────────────────────────────────
#     if vol21_last is not None and not np.isnan(vol21_last):
#         if vol21_last > 0.40:
#             flags.append({
#                 "severity": "WARNING",
#                 "code": "ELEVATED_REALISED_VOL",
#                 "message": (
#                     f"21-day realised volatility (annualised) is {vol21_last:.2%} — "
#                     "above the 40% danger threshold. The stock is in a high-vol regime right now."
#                 ),
#             })

#     return flags


# # ══════════════════════════════════════════════════════════════════════════════
# # AI LAYER 2 — DASHBOARD CONTEXT BUILDER
# # Packages all live dashboard data into a structured JSON dict for Gemini.
# # Mirrors _build_portfolio_context() exactly.
# # ══════════════════════════════════════════════════════════════════════════════

# def _build_dashboard_context(
#     ticker: str,
#     met: dict,
#     last_close: float,
#     pct_chg: float,
#     vol_today: float,
#     rsi_last: float,
#     macd_last: float,
#     signal_last: float,
#     vol21_last: float,
#     n_days: int,
#     danger_flags: list[dict],
#     data_source: str = "unknown",
# ) -> dict:
#     # RSI regime label
#     if rsi_last is not None and not np.isnan(rsi_last):
#         rsi_regime = "Overbought" if rsi_last > 70 else ("Oversold" if rsi_last < 30 else "Neutral")
#     else:
#         rsi_regime = "N/A"

#     # MACD signal label
#     if (macd_last is not None and signal_last is not None
#             and not np.isnan(macd_last) and not np.isnan(signal_last)):
#         macd_label = "Bullish (MACD above Signal)" if macd_last >= signal_last else "Bearish (MACD below Signal)"
#     else:
#         macd_label = "N/A"

#     # Vol regime label
#     if vol21_last is not None and not np.isnan(vol21_last):
#         if vol21_last < 0.20:
#             vol_regime = "Low (<20%)"
#         elif vol21_last < 0.35:
#             vol_regime = "Normal (20-35%)"
#         elif vol21_last < 0.50:
#             vol_regime = "Elevated (35-50%)"
#         else:
#             vol_regime = "Extreme (>50%)"
#     else:
#         vol_regime = "N/A"

#     return {
#         "ticker": ticker,
#         "data_source": data_source,
#         "n_trading_days_analysed": n_days,
#         "price_action": {
#             "last_close": round(last_close, 2),
#             "day_change_pct": round(pct_chg, 4),
#             "volume_today": int(vol_today) if vol_today else 0,
#         },
#         "performance": {
#             "cagr":           met.get("CAGR", "N/A"),
#             "ann_return":     met.get("Ann. Return", "N/A"),
#             "ann_volatility": met.get("Ann. Volatility", "N/A"),
#             "sharpe":         met.get("Sharpe", "N/A"),
#             "sortino":        met.get("Sortino", "N/A"),
#             "calmar":         met.get("Calmar", "N/A"),
#             "max_drawdown":   met.get("Max Drawdown", "N/A"),
#             "win_rate":       met.get("Win Rate", "N/A"),
#         },
#         "risk": {
#             "var_95_hist":  met.get("VaR 95% (Hist)", "N/A"),
#             "cvar_95_hist": met.get("CVaR 95% (Hist)", "N/A"),
#             "var_95_tdist": met.get("VaR 95% (t-dist)", "N/A"),
#             "var_95_garch": met.get("VaR 95% (GARCH)", "N/A"),
#             "var_99":       met.get("VaR 99%", "N/A"),
#             "cvar_99":      met.get("CVaR 99%", "N/A"),
#         },
#         "technicals": {
#             "rsi_14":                round(float(rsi_last), 2) if rsi_last and not np.isnan(rsi_last) else None,
#             "rsi_regime":            rsi_regime,
#             "macd":                  round(float(macd_last), 6) if macd_last and not np.isnan(macd_last) else None,
#             "macd_signal":           round(float(signal_last), 6) if signal_last and not np.isnan(signal_last) else None,
#             "macd_signal_label":     macd_label,
#             "vol21_annualised_pct":  round(float(vol21_last) * 100, 2) if vol21_last and not np.isnan(vol21_last) else None,
#             "vol_regime":            vol_regime,
#         },
#         "danger_flags":        danger_flags,
#         "danger_flag_count":   len([f for f in danger_flags if f["severity"] == "DANGER"]),
#         "warning_flag_count":  len([f for f in danger_flags if f["severity"] == "WARNING"]),
#         "reference_thresholds": {
#             "sharpe_excellent":     1.0,
#             "sharpe_acceptable":    0.5,
#             "sharpe_danger":        0.0,
#             "max_drawdown_danger":  -0.40,
#             "max_drawdown_warning": -0.25,
#             "ann_vol_danger":       0.50,
#             "ann_vol_warning":      0.35,
#             "win_rate_warning":     0.40,
#             "calmar_warning":       0.50,
#             "cvar_95_warning":      -0.04,
#             "rsi_overbought":       70,
#             "rsi_oversold":         30,
#             "vol21_elevated":       0.40,
#         },
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# # AI LAYER — DETERMINISTIC FALLBACK
# # Shown when Gemini key is missing or API call fails.
# # Mirrors _fallback_portfolio_explanation() exactly.
# # ══════════════════════════════════════════════════════════════════════════════

# def _fallback_dashboard_explanation(context: dict) -> str:
#     perf   = context["performance"]
#     risk   = context["risk"]
#     tech   = context["technicals"]
#     flags  = context.get("danger_flags", [])
#     ticker = context["ticker"]
#     price  = context["price_action"]

#     flag_text = ""
#     if flags:
#         flag_text = "\n\n**Flags detected:**\n" + "\n".join(
#             f"- **{f['severity']}** ({f['code']}): {f['message']}"
#             for f in flags
#         )

#     return (
#         f"### What the output says\n"
#         f"Dashboard analysis for **{ticker}** over "
#         f"**{context['n_trading_days_analysed']} trading days**. "
#         f"Last close: **${price['last_close']:,.2f}** "
#         f"({'▲' if price['day_change_pct'] >= 0 else '▼'}"
#         f"{abs(price['day_change_pct']):.2f}% today). "
#         f"Data source: **{context['data_source']}**.\n\n"
#         f"### What each number means\n"
#         f"- **CAGR**: {perf['cagr']} — compound annual growth rate over the full window.\n"
#         f"- **Sharpe**: {perf['sharpe']} — risk-adjusted return "
#         f"(threshold: 0.5 acceptable, 1.0 excellent).\n"
#         f"- **Sortino**: {perf['sortino']} — return per unit of downside risk only.\n"
#         f"- **Calmar**: {perf['calmar']} — CAGR divided by maximum drawdown.\n"
#         f"- **Max Drawdown**: {perf['max_drawdown']} — largest peak-to-trough loss.\n"
#         f"- **Win Rate**: {perf['win_rate']} — fraction of days with a positive return.\n"
#         f"- **VaR 95% (Hist)**: {risk['var_95_hist']} — worst expected daily loss 1-in-20 days.\n"
#         f"- **CVaR 95%**: {risk['cvar_95_hist']} — average loss on the worst 5% of days.\n"
#         f"- **RSI(14)**: {tech['rsi_14']} — momentum reading: **{tech['rsi_regime']}**.\n"
#         f"- **MACD**: {tech['macd_signal_label']}.\n"
#         f"- **21d Vol (ann.)**: {tech['vol21_annualised_pct']}% — regime: **{tech['vol_regime']}**.\n"
#         f"{flag_text}\n\n"
#         f"### Plain English conclusion\n"
#         f"Review all flags above before making any allocation decisions for {ticker}. "
#         f"Cross-reference with the Signals and Backtest pages for deeper validation.\n\n"
#         f"⚠️ *This explanation is generated from dashboard outputs only. "
#         f"It is not financial advice. Always verify with your own judgment.*"
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# # AI LAYER — GEMINI EXPLAINER
# # Direct mirror of _call_gemini_explainer() from 08_portfolio.py.
# # Uses urllib only (no new dependency). Same fallback chain.
# # ══════════════════════════════════════════════════════════════════════════════

# _GEMINI_DASHBOARD_SYSTEM_PROMPT = """You are a senior quantitative analyst embedded inside a professional equity dashboard.

# Your sole job: explain the dashboard output for a SINGLE STOCK to a NON-TECHNICAL user — a retail investor, family office client, or portfolio manager who understands investing but not the mathematics.

# RULES (follow all, no exceptions):
# 1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
# 2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
# 3. Explain every key metric in one plain English sentence. Do not skip Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, VaR, CVaR, RSI, MACD, or volatility regime.
# 4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
# 5. Never say "you should buy" or "you should sell" — explain what the analysis says, not what to do.
# 6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
# 7. Write in short paragraphs. No jargon. No LaTeX. No formulas.

# THRESHOLD KNOWLEDGE (use these to interpret numbers):
# - Sharpe < 0: returns destroy value on a risk-adjusted basis
# - Sharpe 0-0.5: below investable minimum
# - Sharpe 0.5-1.0: acceptable
# - Sharpe > 1.0: excellent
# - Max Drawdown < -40%: catastrophic loss risk
# - Max Drawdown -25% to -40%: significant drawdown
# - Ann. Volatility > 50%: extreme — treat like a speculative asset
# - Ann. Volatility 35-50%: elevated above typical equity levels
# - Win Rate < 40%: fewer than 2-in-5 days positive
# - Calmar < 0.5: poor return per unit of drawdown risk
# - CVaR 95% worse than -4%/day: meaningful tail risk
# - RSI > 70: overbought — momentum stretched
# - RSI < 30: oversold — potential capitulation
# - MACD below Signal: bearish momentum
# - 21d Vol > 40%: elevated current volatility regime

# OUTPUT FORMAT — exactly 4 sections with these markdown headings:
# ### What the output says
# (One paragraph: ticker, period, data source, overall quality assessment)

# ### What each number means
# (Bullet per key metric: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, CAGR, VaR 95%, CVaR 95%, RSI regime, MACD signal, volatility regime)

# ### Red flags
# (If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

# ### Plain English conclusion
# (2-3 sentences max. What a smart non-quant should take away from this dashboard.)

# End your response with this exact line — no modifications:
# ⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


# def _call_gemini_dashboard_explainer(context: dict) -> str:
#     """
#     Calls Google Gemini API with the dashboard context.
#     Falls back to deterministic explanation on any error.
#     Direct mirror of _call_gemini_explainer() from 08_portfolio.py.
#     """
#     gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
#     gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

#     if not gemini_key:
#         return _fallback_dashboard_explanation(context)

#     safe_context = json.loads(json.dumps(context, default=str))
#     user_text = (
#         "Here is the current single-stock dashboard output. "
#         "Please explain it for a non-technical user:\n\n"
#         + json.dumps(safe_context, indent=2)
#     )

#     url = (
#         f"https://generativelanguage.googleapis.com/v1beta/models/"
#         f"{gemini_model}:generateContent?key={gemini_key}"
#     )
#     payload = {
#         "system_instruction": {"parts": [{"text": _GEMINI_DASHBOARD_SYSTEM_PROMPT}]},
#         "contents": [{"role": "user", "parts": [{"text": user_text}]}],
#         "generationConfig": {"maxOutputTokens": 900, "temperature": 0.2},
#     }

#     req = urlrequest.Request(
#         url,
#         data=json.dumps(payload).encode("utf-8"),
#         headers={"Content-Type": "application/json"},
#         method="POST",
#     )
#     try:
#         with urlrequest.urlopen(req, timeout=30) as response:
#             data = json.loads(response.read().decode("utf-8"))
#         candidates = data.get("candidates", [])
#         if not candidates:
#             raise ValueError("No candidates in Gemini response")
#         parts = candidates[0].get("content", {}).get("parts", [])
#         text  = "".join(p.get("text", "") for p in parts).strip()
#         return text or _fallback_dashboard_explanation(context)
#     except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
#         return (
#             _fallback_dashboard_explanation(context)
#             + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
#             "Add GEMINI_API_KEY to .env for AI explanations.*"
#         )


# # ── Sidebar ────────────────────────────────────────────────────────────────────
# render_data_engine_controls("dashboard")
# _sb_sec("Controls")
# ticker = _ticker_sb("dash_ticker")
# ma_periods = st.sidebar.multiselect(
#     "Moving Averages",
#     [10, 20, 50, 100, 200],
#     default=[20, 50],
#     help="Overlay selected moving averages on the price chart.",
# )
# st.sidebar.markdown("---")
# _sb_sec("Chart Options")
# show_vol_panel   = st.sidebar.checkbox("Volume panel",     value=True, key="dash_vol")
# show_vol21_panel = st.sidebar.checkbox("Volatility panel", value=True, key="dash_v21")
# show_rsi_macd    = st.sidebar.checkbox("RSI & MACD panel", value=True, key="dash_rsi")
# candle_style = st.sidebar.radio(
#     "Price style", ["Candlestick", "Line"], horizontal=True, key="dash_style",
# )

# # ── Page header ────────────────────────────────────────────────────────────────
# _header("📈 Dashboard", "Price · Volume · Volatility · RSI · MACD · Metrics")

# # ── Data load ──────────────────────────────────────────────────────────────────
# with st.spinner("Loading data..."):
#     df = load_ticker_data(ticker, start=_start_str())

# if df.empty:
#     st.warning("No data available — try a different ticker or start date.")
#     st.stop()

# ret = returns(df)
# met = summary_table(ret, cfg.RISK_FREE_RATE)


# # ── Helpers ────────────────────────────────────────────────────────────────────
# def _safe_get(key: str) -> str:
#     val = met.get(key)
#     if val is None:
#         return "N/A"
#     if isinstance(val, float) and pd.isna(val):
#         return "N/A"
#     return str(val)


# def _color_class(key: str, value: str) -> str:
#     if value == "N/A":
#         return "qe-metric-na"
#     try:
#         parsed = float(
#             value.replace("%", "").replace("x", "").replace("$", "").replace(",", "").strip()
#         )
#         if key in {"Max Drawdown", "CVaR 95%", "VaR 95%"}:
#             return "qe-metric-neg"
#         if parsed > 0:
#             return "qe-metric-pos"
#         if parsed < 0:
#             return "qe-metric-neg"
#     except Exception:
#         pass
#     return "qe-metric-neutral" if "Volatility" in key else ""


# def _metric_row(label: str, sub: str) -> str:
#     value       = _safe_get(label)
#     value_class = _color_class(label, value)
#     return f"""
#     <div class="qe-stat-row">
#         <div>
#             <div class="qe-stat-label">{label}</div>
#             <div class="qe-stat-sub">{sub}</div>
#         </div>
#         <div class="qe-stat-value {value_class}">{value}</div>
#     </div>"""


# # ── Top metric bar ─────────────────────────────────────────────────────────────
# last_close  = df["Close"].iloc[-1]
# prev_close  = df["Close"].iloc[-2] if len(df) > 1 else last_close
# pct_chg     = (last_close - prev_close) / prev_close * 100
# vol_today   = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
# ann_vol_pct = _safe_get("Ann. Volatility")
# sharpe_val  = _safe_get("Sharpe")

# chg_sign = "+" if pct_chg >= 0 else ""
# chg_cls  = "badge-green" if pct_chg >= 0 else "badge-red"
# vol_fmt  = f"{vol_today/1e6:.2f}M" if vol_today >= 1e6 else f"{vol_today/1e3:.0f}K"

# st.markdown(
#     f"""
# <div class="qe-topbar">
#     <div class="qe-topbar-pill">
#         <div>
#             <div class="qe-topbar-label">Price</div>
#             <div class="qe-topbar-val">${last_close:,.2f}
#                 <span class="qe-topbar-badge {chg_cls}">{chg_sign}{pct_chg:.2f}%</span>
#             </div>
#         </div>
#     </div>
#     <div class="qe-topbar-pill">
#         <div>
#             <div class="qe-topbar-label">Volume</div>
#             <div class="qe-topbar-val">{vol_fmt}</div>
#         </div>
#     </div>
#     <div class="qe-topbar-pill">
#         <div>
#             <div class="qe-topbar-label">Volatility</div>
#             <div class="qe-topbar-val">
#                 <span class="qe-topbar-badge badge-yellow">{ann_vol_pct}</span>
#             </div>
#         </div>
#     </div>
#     <div class="qe-topbar-pill">
#         <div>
#             <div class="qe-topbar-label">Sharpe</div>
#             <div class="qe-topbar-val">
#                 <span class="qe-topbar-badge badge-blue">{sharpe_val}</span>
#             </div>
#         </div>
#     </div>
# </div>
# """,
#     unsafe_allow_html=True,
# )


# # ── Section: Market Overview ───────────────────────────────────────────────────
# st.markdown("""
# <div class="qe-section-head">
#     <h3>Market Overview</h3>
#     <div class="qe-section-dot"></div>
#     <div class="qe-section-line"></div>
# </div>
# """, unsafe_allow_html=True)

# # ── Plot theme ─────────────────────────────────────────────────────────────────
# PLOT_THEME = dict(
#     template="plotly_dark",
#     paper_bgcolor="rgba(0,0,0,0)",
#     plot_bgcolor="#0d1421",
#     margin=dict(l=8, r=8, t=8, b=8),
#     hovermode="x unified",
#     legend=dict(
#         orientation="h", yanchor="bottom", y=1.02,
#         xanchor="left", x=0,
#         bgcolor="rgba(13,20,33,0.85)", bordercolor="#1e2d47",
#         borderwidth=1, font=dict(size=10),
#     ),
# )
# GRID = dict(showgrid=True, gridcolor="#172035", gridwidth=0.6)


# def _build_price_figure() -> go.Figure:
#     has_volume  = show_vol_panel and "Volume" in df.columns
#     rows        = 2 if has_volume else 1
#     row_heights = [0.75, 0.25] if has_volume else [1.0]
#     fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
#                         vertical_spacing=0.04, row_heights=row_heights)
#     if candle_style == "Candlestick":
#         fig.add_trace(go.Candlestick(
#             x=df.index, open=df["Open"], high=df["High"],
#             low=df["Low"], close=df["Close"], name="Price",
#             increasing_line_color="#4ade80", decreasing_line_color="#f87171",
#             increasing_fillcolor="rgba(74,222,128,0.85)",
#             decreasing_fillcolor="rgba(248,113,113,0.85)",
#         ), row=1, col=1)
#     else:
#         fig.add_trace(go.Scatter(
#             x=df.index, y=df["Close"], name="Close",
#             line=dict(color="#7dd3fc", width=2),
#             fill="tozeroy", fillcolor="rgba(125,211,252,0.07)",
#         ), row=1, col=1)
#     ma_colors = {10: "#facc15", 20: "#a78bfa", 50: "#fb923c", 100: "#34d399", 200: "#f472b6"}
#     for period in ma_periods:
#         ma = df["Close"].rolling(period).mean()
#         fig.add_trace(go.Scatter(
#             x=ma.index, y=ma.values, name=f"MA {period}",
#             line=dict(color=ma_colors.get(period, "#ffffff"), width=1.4, dash="dot"),
#         ), row=1, col=1)
#     if has_volume:
#         bar_colors = [
#             "rgba(74,222,128,0.65)" if c >= o else "rgba(248,113,113,0.65)"
#             for c, o in zip(df["Close"], df["Open"])
#         ]
#         fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
#                              marker_color=bar_colors, showlegend=False), row=2, col=1)
#         avg_vol = df["Volume"].rolling(20).mean()
#         fig.add_trace(go.Scatter(
#             x=avg_vol.index, y=avg_vol.values, name="Vol MA(20)",
#             line=dict(color="#facc15", width=1.2, dash="dot"),
#         ), row=2, col=1)
#         fig.update_yaxes(title_text="Volume", row=2, col=1,
#                          tickfont=dict(size=9), title_font=dict(size=9))
#     fig.update_layout(**PLOT_THEME, height=400, xaxis_rangeslider_visible=False)
#     fig.update_xaxes(**GRID, rangeslider_visible=False)
#     fig.update_yaxes(**GRID)
#     fig.update_yaxes(title_text="Price ($)", row=1, col=1,
#                      tickfont=dict(size=9), title_font=dict(size=9))
#     return fig


# def _build_rsi_macd_figure() -> go.Figure:
#     close  = df["Close"]
#     delta  = close.diff()
#     gain   = delta.clip(lower=0).rolling(14).mean()
#     loss   = (-delta.clip(upper=0)).rolling(14).mean()
#     rs     = gain / loss.replace(0, float("nan"))
#     rsi    = 100 - (100 / (1 + rs))
#     ema12  = close.ewm(span=12, adjust=False).mean()
#     ema26  = close.ewm(span=26, adjust=False).mean()
#     macd   = ema12 - ema26
#     signal = macd.ewm(span=9, adjust=False).mean()
#     hist   = macd - signal
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
#                         row_heights=[0.38, 0.38, 0.24],
#                         subplot_titles=["RSI (14)", "MACD", ""])
#     fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.07)", line_width=0, row=1, col=1)
#     fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(74,222,128,0.07)",  line_width=0, row=1, col=1)
#     fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.4)", line_width=1, row=1, col=1)
#     fig.add_hline(y=30, line_dash="dot", line_color="rgba(74,222,128,0.4)",  line_width=1, row=1, col=1)
#     fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI",
#                              line=dict(color="#a78bfa", width=2)), row=1, col=1)
#     fig.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD",
#                              line=dict(color="#7dd3fc", width=1.8)), row=2, col=1)
#     fig.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Signal",
#                              line=dict(color="#fb923c", width=1.5)), row=2, col=1)
#     hist_colors = ["rgba(74,222,128,0.65)" if v >= 0 else "rgba(248,113,113,0.65)" for v in hist]
#     fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histogram",
#                          marker_color=hist_colors, showlegend=False), row=3, col=1)
#     fig.update_layout(**PLOT_THEME, height=400)
#     fig.update_xaxes(**GRID)
#     fig.update_yaxes(**GRID)
#     fig.update_yaxes(title_text="RSI",  row=1, col=1, tickfont=dict(size=9), range=[0, 100])
#     fig.update_yaxes(title_text="MACD", row=2, col=1, tickfont=dict(size=9))
#     fig.update_yaxes(title_text="Hist", row=3, col=1, tickfont=dict(size=9))
#     return fig


# def _build_volatility_figure() -> go.Figure:
#     vol21    = ret.rolling(21).std() * (252 ** 0.5) * 100
#     vol_slow = vol21.rolling(21).mean()
#     vmax     = max(float(vol21.max()) if not vol21.dropna().empty else 0.0, 45.0)
#     fig = go.Figure()
#     fig.add_hrect(y0=0,  y1=20,        fillcolor="rgba(74,222,128,0.04)",  line_width=0)
#     fig.add_hrect(y0=20, y1=40,        fillcolor="rgba(250,204,21,0.05)",  line_width=0)
#     fig.add_hrect(y0=40, y1=vmax*1.05, fillcolor="rgba(248,113,113,0.06)", line_width=0)
#     fig.add_trace(go.Scatter(x=vol21.index, y=vol21.values, name="21d Vol",
#                              line=dict(color="#a78bfa", width=2.2),
#                              fill="tozeroy", fillcolor="rgba(167,139,250,0.12)"))
#     fig.add_trace(go.Scatter(x=vol_slow.index, y=vol_slow.values, name="Trend",
#                              line=dict(color="#7dd3fc", width=1.4, dash="dot")))
#     fig.update_layout(**PLOT_THEME, height=280)
#     fig.update_xaxes(**GRID)
#     fig.update_yaxes(**GRID, ticksuffix="%", title_text="Volatility")
#     return fig


# # ── Row 1: Price Chart  |  RSI & MACD ─────────────────────────────────────────
# col_left, col_right = st.columns([1.05, 0.95], gap="medium")
# with col_left:
#     with st.container(border=True):
#         st.markdown('<div class="qe-panel-title">Price Chart</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-sub">Candles · Moving averages · Volume</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
#         st.plotly_chart(_build_price_figure(), use_container_width=True, config={"displayModeBar": False})

# with col_right:
#     with st.container(border=True):
#         st.markdown('<div class="qe-panel-title">RSI &amp; MACD Indicators</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-sub">Momentum · Trend divergence · Histogram</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
#         if show_rsi_macd:
#             st.plotly_chart(_build_rsi_macd_figure(), use_container_width=True, config={"displayModeBar": False})
#         else:
#             st.info("Enable `RSI & MACD panel` in the sidebar to show this card.")


# # ── Row 2: Key Metrics  |  Risk & Return ──────────────────────────────────────
# col_bl, col_br = st.columns([0.95, 1.05], gap="medium")
# with col_bl:
#     with st.container(border=True):
#         st.markdown('<div class="qe-panel-title">Key Metrics</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-sub">Core performance numbers</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
#         st.markdown(
#             '<div class="qe-stat-table">'
#             + _metric_row("Sharpe",       "Risk-adjusted return")
#             + _metric_row("Sortino",      "Downside-risk adjusted return")
#             + _metric_row("Max Drawdown", "Largest peak-to-trough decline")
#             + _metric_row("Win Rate",     "Share of positive periods")
#             + "</div>",
#             unsafe_allow_html=True,
#         )
#         if show_vol21_panel:
#             st.markdown("<br>", unsafe_allow_html=True)
#             st.plotly_chart(_build_volatility_figure(), use_container_width=True, config={"displayModeBar": False})

# with col_br:
#     with st.container(border=True):
#         st.markdown('<div class="qe-panel-title">Risk &amp; Return Snapshot</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-sub">Return · Drawdown · VaR · CVaR</div>', unsafe_allow_html=True)
#         st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
#         st.markdown(
#             '<div class="qe-stat-table">'
#             + _metric_row("CAGR",           "Compound annual growth rate")
#             + _metric_row("Ann. Return",    "Arithmetic annual return")
#             + _metric_row("Ann. Volatility","Annualised standard deviation")
#             + _metric_row("Calmar",         "CAGR divided by max drawdown")
#             + _metric_row("VaR 95%",        "One-day 95% value at risk")
#             + _metric_row("CVaR 95%",       "Expected shortfall beyond VaR")
#             + "</div>",
#             unsafe_allow_html=True,
#         )


# # ── Row 3: Recent OHLCV ────────────────────────────────────────────────────────
# st.markdown("""
# <div class="qe-section-head" style="margin-top:18px;">
#     <h3>Recent OHLCV Data</h3>
#     <div class="qe-section-dot"></div>
#     <div class="qe-section-line"></div>
# </div>
# """, unsafe_allow_html=True)

# with st.container(border=True):
#     st.markdown('<div class="qe-panel-sub">Latest 30 sessions — colour-coded by session direction</div>', unsafe_allow_html=True)
#     st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
#     tail = df.tail(30).copy()
#     if "Volume" in tail.columns:
#         tail["Volume"] = tail["Volume"].apply(
#             lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if not pd.isna(v) else "-")
#         )
#     def _style_row(row):
#         color = "color: #4ade80" if row.get("Close", 0) >= row.get("Open", 0) else "color: #f87171"
#         return [color] * len(row)
#     num_cols  = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in tail.columns]
#     formatted = tail.copy()
#     for col in num_cols:
#         formatted[col] = formatted[col].map(
#             lambda v, _col=col: f"${v:,.2f}" if not pd.isna(v) else "-"
#         )
#     st.dataframe(
#         formatted.iloc[::-1].style.apply(_style_row, axis=1),
#         use_container_width=True, height=280,
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# # PRE-COMPUTE LIVE INDICATOR VALUES (shared by flag engine + context builder)
# # Computed once here — RSI, MACD, vol21.
# # ══════════════════════════════════════════════════════════════════════════════

# _close  = df["Close"]
# _delta  = _close.diff()
# _gain   = _delta.clip(lower=0).rolling(14).mean()
# _loss   = (-_delta.clip(upper=0)).rolling(14).mean()
# _rs     = _gain / _loss.replace(0, float("nan"))
# _rsi_s  = 100 - (100 / (1 + _rs))
# _ema12  = _close.ewm(span=12, adjust=False).mean()
# _ema26  = _close.ewm(span=26, adjust=False).mean()
# _macd_s = _ema12 - _ema26
# _sig_s  = _macd_s.ewm(span=9, adjust=False).mean()
# _vol21  = ret.rolling(21).std() * (252 ** 0.5)

# rsi_last    = float(_rsi_s.iloc[-1])   if len(_rsi_s.dropna()) > 0   else float("nan")
# macd_last   = float(_macd_s.iloc[-1])  if len(_macd_s.dropna()) > 0  else float("nan")
# signal_last = float(_sig_s.iloc[-1])   if len(_sig_s.dropna()) > 0   else float("nan")
# vol21_last  = float(_vol21.iloc[-1])   if len(_vol21.dropna()) > 0   else float("nan")
# data_source = str(df.attrs.get("data_source", "unknown"))


# # ══════════════════════════════════════════════════════════════════════════════
# # AI DECODER SECTION — Gemini-powered
# # Same 3-layer architecture as 08_portfolio.py:
# #   Layer 1: Deterministic danger badges (always shown, no AI)
# #   Layer 2: "Decode for Me" button → Gemini explanation
# #   Layer 3: Structured AI output with disclaimer
# # ══════════════════════════════════════════════════════════════════════════════

# st.markdown("---")

# st.markdown("""
# <div style="margin: 8px 0 4px;">
#   <span style="font-size:20px;font-weight:600;">🤖 AI Dashboard Decoder</span>
#   <span style="font-size:12px;opacity:0.55;margin-left:12px;">
#     Plain-English explanation for non-technical users · Powered by Gemini
#   </span>
# </div>
# """, unsafe_allow_html=True)
# st.caption(
#     "This section translates the quantitative dashboard output above into plain English. "
#     "It reads the actual numbers from this analysis — not generic descriptions. "
#     "It does not change the metrics. It does not give financial advice."
# )

# # ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
# danger_flags = _compute_dashboard_danger_flags(
#     ticker=ticker, met=met,
#     last_close=last_close, pct_chg=pct_chg,
#     rsi_last=rsi_last, macd_last=macd_last,
#     signal_last=signal_last, vol21_last=vol21_last,
#     data_source=data_source,
# )

# if danger_flags:
#     n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
#     n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
#     n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

#     badge_html = ""
#     if n_danger:
#         badge_html += (
#             f'<span style="background:#dc3232;color:#fff;border-radius:4px;'
#             f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
#             f'⛔ {n_danger} DANGER</span>'
#         )
#     if n_warning:
#         badge_html += (
#             f'<span style="background:#e67e00;color:#fff;border-radius:4px;'
#             f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
#             f'⚠️ {n_warning} WARNING</span>'
#         )
#     if n_info:
#         badge_html += (
#             f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;'
#             f'padding:2px 8px;font-size:12px;font-weight:600;">'
#             f'ℹ️ {n_info} INFO</span>'
#         )
#     st.markdown(f'<div style="margin:10px 0 6px;">{badge_html}</div>', unsafe_allow_html=True)

#     color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
#     bg_map    = {
#         "DANGER":  "rgba(220,50,50,0.08)",
#         "WARNING": "rgba(230,126,0,0.08)",
#         "INFO":    "rgba(26,111,160,0.08)",
#     }
#     for flag in danger_flags:
#         st.markdown(
#             f"""<div style="
#                 background:{bg_map[flag['severity']]};
#                 border-left:3px solid {color_map[flag['severity']]};
#                 border-radius:0 6px 6px 0;
#                 padding:10px 14px; margin:6px 0;
#                 font-size:13px; line-height:1.55;
#             ">
#               <span style="font-weight:700;color:{color_map[flag['severity']]};">
#                 {flag['severity']} · {flag['code']}
#               </span><br>
#               {flag['message']}
#             </div>""",
#             unsafe_allow_html=True,
#         )
# else:
#     st.success("✅ Pre-flight checks passed — no critical flags detected for this ticker.")

# st.markdown("")

# # ── LAYER 2: Build context + button ──────────────────────────────────────────
# dashboard_context = _build_dashboard_context(
#     ticker=ticker, met=met,
#     last_close=last_close, pct_chg=pct_chg,
#     vol_today=vol_today, rsi_last=rsi_last,
#     macd_last=macd_last, signal_last=signal_last,
#     vol21_last=vol21_last, n_days=len(df),
#     danger_flags=danger_flags, data_source=data_source,
# )

# # Cache-bust: reset AI summary on ticker/data change (mirrors portfolio)
# context_key = json.dumps(
#     {k: v for k, v in dashboard_context.items() if k != "danger_flags"},
#     sort_keys=True, default=str,
# )
# if st.session_state.get("dashboard_ai_context_key") != context_key:
#     st.session_state.dashboard_ai_context_key = context_key
#     st.session_state.dashboard_ai_summary = ""

# col_btn, col_ctx = st.columns([1, 2])

# with col_btn:
#     st.markdown("**What Gemini sees:**")
#     preview_rows = [
#         {"Field": "Ticker",          "Value": ticker},
#         {"Field": "Data source",     "Value": data_source},
#         {"Field": "Trading days",    "Value": str(len(df))},
#         {"Field": "Last close",      "Value": f"${last_close:,.2f}"},
#         {"Field": "Day change",      "Value": f"{pct_chg:+.2f}%"},
#         {"Field": "Sharpe",          "Value": met.get("Sharpe", "N/A")},
#         {"Field": "Sortino",         "Value": met.get("Sortino", "N/A")},
#         {"Field": "Calmar",          "Value": met.get("Calmar", "N/A")},
#         {"Field": "Max Drawdown",    "Value": met.get("Max Drawdown", "N/A")},
#         {"Field": "Ann. Volatility", "Value": met.get("Ann. Volatility", "N/A")},
#         {"Field": "Win Rate",        "Value": met.get("Win Rate", "N/A")},
#         {"Field": "VaR 95% (Hist)",  "Value": met.get("VaR 95% (Hist)", "N/A")},
#         {"Field": "CVaR 95% (Hist)", "Value": met.get("CVaR 95% (Hist)", "N/A")},
#         {"Field": "RSI(14)",         "Value": f"{rsi_last:.1f}" if not np.isnan(rsi_last) else "N/A"},
#         {"Field": "MACD signal",     "Value": "Bullish" if macd_last >= signal_last else "Bearish"},
#         {"Field": "Vol21 (ann.)",    "Value": f"{vol21_last:.2%}" if not np.isnan(vol21_last) else "N/A"},
#         {"Field": "Danger flags",    "Value": str(sum(1 for f in danger_flags if f["severity"] == "DANGER"))},
#         {"Field": "Warning flags",   "Value": str(sum(1 for f in danger_flags if f["severity"] == "WARNING"))},
#     ]
#     st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

#     gemini_key_set = bool(getattr(cfg, "GEMINI_API_KEY", ""))
#     if not gemini_key_set:
#         st.info(
#             "💡 Add `GEMINI_API_KEY` to your `.env` file for Gemini AI explanations. "
#             "Without it, a deterministic quant-safe explanation is shown instead."
#         )

#     decode_clicked = st.button(
#         "🤖 Decode for Me",
#         type="primary",
#         key="dashboard_ai_explain",
#         use_container_width=True,
#         help="Translates the dashboard output above into plain English using Gemini.",
#     )
#     clear_clicked = st.button(
#         "Clear explanation",
#         key="dashboard_ai_clear",
#         use_container_width=True,
#     )

# with col_ctx:
#     st.markdown("**How this works:**")
#     st.markdown("""
# <div style="
#     background: rgba(14,22,42,0.82);
#     border: 1px solid rgba(11,224,255,0.18);
#     border-radius: 10px;
#     padding: 16px 18px;
#     font-size:13px;
#     line-height:1.65;
# ">
#   <div style="font-weight:700;color:#e8f4fd;margin-bottom:10px;">What happens when you click Decode:</div>
#   <ol style="margin:0;padding-left:18px;color:#a8c4d8;">
#     <li style="margin-bottom:6px;">
#       The <strong>pre-flight checks above run first</strong> — danger flags are always
#       deterministic. They appear regardless of whether you click Decode.
#     </li>
#     <li style="margin-bottom:6px;">
#       The actual numbers from this analysis (Sharpe, Sortino, Calmar, Max Drawdown,
#       Win Rate, VaR, CVaR, CAGR, RSI reading, MACD signal, 21-day volatility regime,
#       data source, and all flags) are sent to Gemini.
#     </li>
#     <li style="margin-bottom:6px;">
#       Gemini explains each number in plain English, flags anything dangerous,
#       and writes a plain-English conclusion — using the actual values, not generic descriptions.
#     </li>
#     <li style="margin-bottom:6px;">
#       Output: <strong>4 sections</strong> — what the output says · what each number means ·
#       red flags · plain-English conclusion.
#     </li>
#     <li>
#       A <strong>mandatory disclaimer</strong> is appended — this is not financial advice.
#     </li>
#   </ol>
# </div>
# """, unsafe_allow_html=True)

# if clear_clicked:
#     st.session_state.dashboard_ai_summary = ""

# if decode_clicked:
#     with st.spinner("Gemini is reading the dashboard output and writing your plain-English explanation..."):
#         st.session_state.dashboard_ai_summary = _call_gemini_dashboard_explainer(dashboard_context)

# # ── LAYER 3: AI output ────────────────────────────────────────────────────────
# if st.session_state.get("dashboard_ai_summary"):
#     st.markdown("")
#     st.markdown(
#         """<div style="
#             background: rgba(14,22,42,0.82);
#             border: 1px solid rgba(11,224,255,0.28);
#             border-radius: 12px;
#             padding: 20px 24px;
#             margin-top: 8px;
#         ">""",
#         unsafe_allow_html=True,
#     )
#     st.markdown(st.session_state.dashboard_ai_summary)
#     st.markdown("</div>", unsafe_allow_html=True)
# else:
#     st.markdown("")
#     st.markdown(
#         """<div style="
#             border:1px dashed rgba(11,224,255,0.18);
#             border-radius:10px; padding:20px;
#             text-align:center;
#             color:rgba(200,220,240,0.4);
#             font-size:14px;
#         ">
#           Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
#           of the dashboard output above.
#         </div>""",
#         unsafe_allow_html=True,
#     )

# # ── FAQs ──────────────────────────────────────────────────────────────────────
# st.markdown("")
# qe_faq_section("FAQs", [
#     ("What should I look at first on the dashboard?", "Start with the top metrics row and the latest OHLCV table. They give a quick read on trend, volatility, and recent market behavior."),
#     ("How does this dashboard help me trade?", "It condenses the current state of the symbol into one screen so you can compare trend, risk, and momentum before moving to deeper analysis pages."),
#     ("Why is the recent data table important?", "It shows the freshest sessions and helps you spot gaps, large candles, or unusual volume before trusting a signal."),
#     ("When should I switch to another page?", "Use the dashboard as the starting point, then move to signals, risk, or backtest once you want a more specific answer."),
#     ("What does the AI Decoder do?", "It reads the actual computed numbers from this dashboard and explains what each one means in plain English — no jargon, no formulas. It does not give financial advice."),
#     ("When do the danger flags appear?", "Flags are computed deterministically from the metrics every time the page loads — no AI needed. They fire on hard thresholds: negative Sharpe, severe drawdown, RSI extremes, elevated volatility, and bearish MACD crossovers."),
# ])

"""QuantEdge dashboard — image-matched layout with all bug fixes applied.

AI LAYER — Gemini AI Dashboard Decoder (bottom of page, same 3-layer design as portfolio):
  Layer 1: Deterministic danger flags (always shown, no AI)
            Checks: Sharpe < 0.5, Max Drawdown < -30%, Ann. Volatility > 40%,
                    Win Rate < 45%, CVaR breach, Calmar < 0.5, data source = demo
  Layer 2: Context builder + "Decode for Me" button
            Packages all dashboard metrics, price action, volatility regime,
            RSI reading, MACD signal, and data source into a structured JSON context.
  Layer 3: Gemini output — structured 4-section explanation with mandatory disclaimer
            Falls back to deterministic explanation if key missing or API call fails.

  Uses GEMINI_API_KEY + GEMINI_MODEL from utils/config.py (already present in .env).
  Architecture mirrors 08_portfolio.py exactly — same flag severity system,
  same context-key cache-busting, same fallback chain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings
from urllib import error as urlerror
from urllib import request as urlrequest

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st
from plotly.subplots import make_subplots

from app.data_engine import load_ticker_data
from app.ui_pages._shared import (
    _header,
    _sb_sec,
    _start_str,
    _ticker_sb,
    apply_theme,
)
from app.data_engine import render_data_engine_controls
from core.data import returns
from core.metrics import summary_table
from utils.config import cfg

try:
    from utils.theme import qe_faq_section
except ImportError:
    def qe_faq_section(title: str, faqs: list[tuple[str, str]]) -> None:
        st.markdown("---")
        st.markdown(f"### {title}")
        for question, answer in faqs:
            with st.expander(question):
                st.write(answer)


st.set_page_config(page_title="Dashboard | QuantEdge", page_icon="📈", layout="wide")
apply_theme()

st.markdown(
    """
<style>
section[data-testid="stSidebar"] {
    background: #0f1623 !important;
    border-right: 1px solid #1e2a3e !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, #131c2e 0%, #0f1520 100%);
    border: 1px solid #1e2d47 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.5rem 0.65rem 0.65rem 0.65rem;
}
.qe-topbar { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.qe-topbar-pill {
    display: flex; align-items: center; gap: 10px;
    background: #131c2e; border: 1px solid #1e2d47; border-radius: 10px;
    padding: 10px 18px; min-width: 160px; flex: 1;
}
.qe-topbar-label { color: #5a6a87; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.qe-topbar-val { color: #e2e8f4; font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
.qe-topbar-badge { font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 5px; }
.badge-green  { background: rgba(74,222,128,.15); color: #4ade80; }
.badge-red    { background: rgba(248,113,113,.15); color: #f87171; }
.badge-yellow { background: rgba(250,204,21,.13);  color: #facc15; }
.badge-blue   { background: rgba(125,211,252,.13); color: #7dd3fc; }
.qe-section-head { display: flex; align-items: center; gap: 12px; margin: 6px 0 14px 0; }
.qe-section-head h3 { margin: 0; color: #c8d3e8; font-size: 18px; font-weight: 700; letter-spacing: -0.01em; }
.qe-section-dot { width: 8px; height: 8px; border-radius: 999px; background: #2d4060; box-shadow: 0 0 0 4px rgba(45,64,96,.2); }
.qe-section-line { height: 1px; flex: 1; background: linear-gradient(to right, #1e2d47, transparent); }
.qe-panel-title { color: #d4ddf0; font-size: 14px; font-weight: 700; margin-bottom: 2px; letter-spacing: -0.01em; }
.qe-panel-sub { color: #4a5878; font-size: 11px; margin-bottom: 8px; }
.qe-panel-rule { height: 1px; background: linear-gradient(to right, #1e2d47, transparent); margin: 0 0 10px 0; }
.qe-stat-table { display: grid; gap: 7px; }
.qe-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border: 1px solid #1a2840; border-radius: 9px;
    background: linear-gradient(180deg, #111928 0%, #0d1421 100%);
}
.qe-stat-label { color: #c8d3e8; font-size: 13px; font-weight: 600; }
.qe-stat-sub { color: #3d5070; font-size: 10px; margin-top: 2px; }
.qe-stat-value { font-size: 15px; font-weight: 700; text-align: right; white-space: nowrap; }
.qe-metric-pos     { color: #4ade80 !important; }
.qe-metric-neg     { color: #f87171 !important; }
.qe-metric-neutral { color: #facc15 !important; }
.qe-metric-na      { color: #4a5878 !important; }
[data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"]   { border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER 1 — DETERMINISTIC DANGER FLAGS
# Always runs before Gemini. Mirrors _compute_portfolio_danger_flags() exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_dashboard_danger_flags(
    ticker: str,
    met: dict,
    last_close: float,
    pct_chg: float,
    rsi_last: float,
    macd_last: float,
    signal_last: float,
    vol21_last: float,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the dashboard output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.

    Thresholds:
      Sharpe < 0:        DANGER  — negative risk-adjusted return
      Sharpe 0–0.5:      WARNING — below investable minimum
      Max Drawdown < -40%: DANGER
      Max Drawdown < -25%: WARNING
      Ann. Volatility > 50%: DANGER
      Ann. Volatility > 35%: WARNING
      Win Rate < 40%:    WARNING
      Calmar < 0.5:      WARNING
      CVaR 95% < -4%:    WARNING — high tail risk
      RSI > 75:          WARNING — overbought
      RSI < 25:          WARNING — oversold
      MACD < Signal:     INFO    — bearish momentum
      Vol21 > 40%:       WARNING — elevated vol regime
    """
    flags = []

    def _parse_pct(key: str):
        val = met.get(key, "N/A")
        if val == "N/A":
            return None
        try:
            return float(str(val).replace("%", "").strip()) / 100
        except ValueError:
            return None

    def _parse_float(key: str):
        val = met.get(key, "N/A")
        if val == "N/A":
            return None
        try:
            return float(str(val).replace("x", "").strip())
        except ValueError:
            return None

    # ── Data quality ──────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Dashboard analysis for {ticker} is running on SYNTHETIC demo data, "
                "not real market prices. All metrics — Sharpe, VaR, drawdown — are "
                "illustrative only. Do not make trading decisions based on this output."
            ),
        })

    # ── Sharpe ────────────────────────────────────────────────────────────────
    sharpe_val = _parse_float("Sharpe")
    if sharpe_val is not None:
        if sharpe_val < 0:
            flags.append({
                "severity": "DANGER",
                "code": "NEGATIVE_SHARPE",
                "message": (
                    f"Sharpe ratio is {sharpe_val:.2f} — negative. The asset is producing "
                    "negative risk-adjusted returns over the selected period. Returns do not "
                    "compensate for the volatility risk taken."
                ),
            })
        elif sharpe_val < 0.5:
            flags.append({
                "severity": "WARNING",
                "code": "LOW_SHARPE",
                "message": (
                    f"Sharpe ratio is {sharpe_val:.2f} — below the conventional minimum "
                    "investable threshold of 0.5. Risk-adjusted performance is weak. "
                    "Consider comparing against a benchmark before allocating."
                ),
            })

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    mdd = _parse_pct("Max Drawdown")
    if mdd is not None:
        if mdd < -0.40:
            flags.append({
                "severity": "DANGER",
                "code": "SEVERE_DRAWDOWN",
                "message": (
                    f"Maximum drawdown is {mdd:.2%} — catastrophic. The asset has lost more "
                    "than 40% from peak at some point in the selected window. This level of "
                    "loss is psychologically and financially very difficult to recover from."
                ),
            })
        elif mdd < -0.25:
            flags.append({
                "severity": "WARNING",
                "code": "HIGH_DRAWDOWN",
                "message": (
                    f"Maximum drawdown is {mdd:.2%}. A drawdown exceeding 25% indicates "
                    "meaningful tail risk. Ensure position sizing accounts for this historical "
                    "peak-to-trough decline."
                ),
            })

    # ── Annual Volatility ─────────────────────────────────────────────────────
    ann_vol = _parse_pct("Ann. Volatility")
    if ann_vol is not None:
        if ann_vol > 0.50:
            flags.append({
                "severity": "DANGER",
                "code": "EXTREME_VOLATILITY",
                "message": (
                    f"Annualised volatility is {ann_vol:.2%} — extremely high. "
                    "At this level, a single bad year could produce a 50%+ loss even "
                    "without a structural blow-up. VaR and CVaR figures require special attention."
                ),
            })
        elif ann_vol > 0.35:
            flags.append({
                "severity": "WARNING",
                "code": "HIGH_VOLATILITY",
                "message": (
                    f"Annualised volatility is {ann_vol:.2%} — elevated, well above typical "
                    "equity market volatility (~15-20%). Risk metrics should be interpreted carefully."
                ),
            })

    # ── Win Rate ──────────────────────────────────────────────────────────────
    wr = _parse_pct("Win Rate")
    if wr is not None and wr < 0.40:
        flags.append({
            "severity": "WARNING",
            "code": "LOW_WIN_RATE",
            "message": (
                f"Win rate is {wr:.2%} — fewer than 40% of trading days produced a "
                "positive return. Verify against Calmar and Sortino before assuming profitability."
            ),
        })

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar_val = _parse_float("Calmar")
    if calmar_val is not None and calmar_val < 0.5:
        flags.append({
            "severity": "WARNING",
            "code": "LOW_CALMAR",
            "message": (
                f"Calmar ratio is {calmar_val:.2f} — below 0.5. "
                "The compound growth rate is low relative to the maximum drawdown. "
                "Return earned per unit of drawdown risk is weak."
            ),
        })

    # ── CVaR breach ───────────────────────────────────────────────────────────
    cvar_95 = _parse_pct("CVaR 95% (Hist)")
    if cvar_95 is not None and cvar_95 < -0.04:
        flags.append({
            "severity": "WARNING",
            "code": "HIGH_CVAR",
            "message": (
                f"CVaR 95% (Expected Shortfall) is {cvar_95:.2%} per day. "
                "On the worst 5% of trading days, the average loss exceeds 4%. "
                "This is a meaningful tail-risk exposure that should inform position sizing."
            ),
        })

    # ── RSI extremes ──────────────────────────────────────────────────────────
    if rsi_last is not None and not np.isnan(rsi_last):
        if rsi_last > 75:
            flags.append({
                "severity": "WARNING",
                "code": "RSI_OVERBOUGHT",
                "message": (
                    f"RSI(14) is {rsi_last:.1f} — deep overbought territory (>75). "
                    "Short-term mean-reversion risk is elevated. "
                    "Price may be running ahead of fundamentals."
                ),
            })
        elif rsi_last < 25:
            flags.append({
                "severity": "WARNING",
                "code": "RSI_OVERSOLD",
                "message": (
                    f"RSI(14) is {rsi_last:.1f} — deep oversold territory (<25). "
                    "Capitulation risk is elevated. Watch for reversal signals before adding exposure."
                ),
            })

    # ── MACD bearish cross ────────────────────────────────────────────────────
    if (macd_last is not None and signal_last is not None
            and not np.isnan(macd_last) and not np.isnan(signal_last)):
        if macd_last < signal_last:
            flags.append({
                "severity": "INFO",
                "code": "MACD_BEARISH",
                "message": (
                    f"MACD ({macd_last:.4f}) is below Signal ({signal_last:.4f}) — "
                    "momentum is currently bearish. The short-term EMA is trailing the longer-term one."
                ),
            })

    # ── Elevated short-term volatility ────────────────────────────────────────
    if vol21_last is not None and not np.isnan(vol21_last):
        if vol21_last > 0.40:
            flags.append({
                "severity": "WARNING",
                "code": "ELEVATED_REALISED_VOL",
                "message": (
                    f"21-day realised volatility (annualised) is {vol21_last:.2%} — "
                    "above the 40% danger threshold. The stock is in a high-vol regime right now."
                ),
            })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER 2 — DASHBOARD CONTEXT BUILDER
# Packages all live dashboard data into a structured JSON dict for Gemini.
# Mirrors _build_portfolio_context() exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _build_dashboard_context(
    ticker: str,
    met: dict,
    last_close: float,
    pct_chg: float,
    vol_today: float,
    rsi_last: float,
    macd_last: float,
    signal_last: float,
    vol21_last: float,
    n_days: int,
    danger_flags: list[dict],
    data_source: str = "unknown",
) -> dict:
    # RSI regime label
    if rsi_last is not None and not np.isnan(rsi_last):
        rsi_regime = "Overbought" if rsi_last > 70 else ("Oversold" if rsi_last < 30 else "Neutral")
    else:
        rsi_regime = "N/A"

    # MACD signal label
    if (macd_last is not None and signal_last is not None
            and not np.isnan(macd_last) and not np.isnan(signal_last)):
        macd_label = "Bullish (MACD above Signal)" if macd_last >= signal_last else "Bearish (MACD below Signal)"
    else:
        macd_label = "N/A"

    # Vol regime label
    if vol21_last is not None and not np.isnan(vol21_last):
        if vol21_last < 0.20:
            vol_regime = "Low (<20%)"
        elif vol21_last < 0.35:
            vol_regime = "Normal (20-35%)"
        elif vol21_last < 0.50:
            vol_regime = "Elevated (35-50%)"
        else:
            vol_regime = "Extreme (>50%)"
    else:
        vol_regime = "N/A"

    return {
        "ticker": ticker,
        "data_source": data_source,
        "n_trading_days_analysed": n_days,
        "price_action": {
            "last_close": round(last_close, 2),
            "day_change_pct": round(pct_chg, 4),
            "volume_today": int(vol_today) if vol_today else 0,
        },
        "performance": {
            "cagr":           met.get("CAGR", "N/A"),
            "ann_return":     met.get("Ann. Return", "N/A"),
            "ann_volatility": met.get("Ann. Volatility", "N/A"),
            "sharpe":         met.get("Sharpe", "N/A"),
            "sortino":        met.get("Sortino", "N/A"),
            "calmar":         met.get("Calmar", "N/A"),
            "max_drawdown":   met.get("Max Drawdown", "N/A"),
            "win_rate":       met.get("Win Rate", "N/A"),
        },
        "risk": {
            "var_95_hist":  met.get("VaR 95% (Hist)", "N/A"),
            "cvar_95_hist": met.get("CVaR 95% (Hist)", "N/A"),
            "var_95_tdist": met.get("VaR 95% (t-dist)", "N/A"),
            "var_95_garch": met.get("VaR 95% (GARCH)", "N/A"),
            "var_99":       met.get("VaR 99%", "N/A"),
            "cvar_99":      met.get("CVaR 99%", "N/A"),
        },
        "technicals": {
            "rsi_14":                round(float(rsi_last), 2) if rsi_last and not np.isnan(rsi_last) else None,
            "rsi_regime":            rsi_regime,
            "macd":                  round(float(macd_last), 6) if macd_last and not np.isnan(macd_last) else None,
            "macd_signal":           round(float(signal_last), 6) if signal_last and not np.isnan(signal_last) else None,
            "macd_signal_label":     macd_label,
            "vol21_annualised_pct":  round(float(vol21_last) * 100, 2) if vol21_last and not np.isnan(vol21_last) else None,
            "vol_regime":            vol_regime,
        },
        "danger_flags":        danger_flags,
        "danger_flag_count":   len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count":  len([f for f in danger_flags if f["severity"] == "WARNING"]),
        "reference_thresholds": {
            "sharpe_excellent":     1.0,
            "sharpe_acceptable":    0.5,
            "sharpe_danger":        0.0,
            "max_drawdown_danger":  -0.40,
            "max_drawdown_warning": -0.25,
            "ann_vol_danger":       0.50,
            "ann_vol_warning":      0.35,
            "win_rate_warning":     0.40,
            "calmar_warning":       0.50,
            "cvar_95_warning":      -0.04,
            "rsi_overbought":       70,
            "rsi_oversold":         30,
            "vol21_elevated":       0.40,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK
# Shown when Gemini key is missing or API call fails.
# Mirrors _fallback_portfolio_explanation() exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_dashboard_explanation(context: dict) -> str:
    perf   = context["performance"]
    risk   = context["risk"]
    tech   = context["technicals"]
    flags  = context.get("danger_flags", [])
    ticker = context["ticker"]
    price  = context["price_action"]

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    return (
        f"### What the output says\n"
        f"Dashboard analysis for **{ticker}** over "
        f"**{context['n_trading_days_analysed']} trading days**. "
        f"Last close: **${price['last_close']:,.2f}** "
        f"({'▲' if price['day_change_pct'] >= 0 else '▼'}"
        f"{abs(price['day_change_pct']):.2f}% today). "
        f"Data source: **{context['data_source']}**.\n\n"
        f"### What each number means\n"
        f"- **CAGR**: {perf['cagr']} — compound annual growth rate over the full window.\n"
        f"- **Sharpe**: {perf['sharpe']} — risk-adjusted return "
        f"(threshold: 0.5 acceptable, 1.0 excellent).\n"
        f"- **Sortino**: {perf['sortino']} — return per unit of downside risk only.\n"
        f"- **Calmar**: {perf['calmar']} — CAGR divided by maximum drawdown.\n"
        f"- **Max Drawdown**: {perf['max_drawdown']} — largest peak-to-trough loss.\n"
        f"- **Win Rate**: {perf['win_rate']} — fraction of days with a positive return.\n"
        f"- **VaR 95% (Hist)**: {risk['var_95_hist']} — worst expected daily loss 1-in-20 days.\n"
        f"- **CVaR 95%**: {risk['cvar_95_hist']} — average loss on the worst 5% of days.\n"
        f"- **RSI(14)**: {tech['rsi_14']} — momentum reading: **{tech['rsi_regime']}**.\n"
        f"- **MACD**: {tech['macd_signal_label']}.\n"
        f"- **21d Vol (ann.)**: {tech['vol21_annualised_pct']}% — regime: **{tech['vol_regime']}**.\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"Review all flags above before making any allocation decisions for {ticker}. "
        f"Cross-reference with the Signals and Backtest pages for deeper validation.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI EXPLAINER
# Direct mirror of _call_gemini_explainer() from 08_portfolio.py.
# Uses urllib only (no new dependency). Same fallback chain.
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_DASHBOARD_SYSTEM_PROMPT = """You are a senior quantitative analyst embedded inside a professional equity dashboard.

Your sole job: explain the dashboard output for a SINGLE STOCK to a NON-TECHNICAL user — a retail investor, family office client, or portfolio manager who understands investing but not the mathematics.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key metric in one plain English sentence. Do not skip Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, VaR, CVaR, RSI, MACD, or volatility regime.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should sell" — explain what the analysis says, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No LaTeX. No formulas.

THRESHOLD KNOWLEDGE (use these to interpret numbers):
- Sharpe < 0: returns destroy value on a risk-adjusted basis
- Sharpe 0-0.5: below investable minimum
- Sharpe 0.5-1.0: acceptable
- Sharpe > 1.0: excellent
- Max Drawdown < -40%: catastrophic loss risk
- Max Drawdown -25% to -40%: significant drawdown
- Ann. Volatility > 50%: extreme — treat like a speculative asset
- Ann. Volatility 35-50%: elevated above typical equity levels
- Win Rate < 40%: fewer than 2-in-5 days positive
- Calmar < 0.5: poor return per unit of drawdown risk
- CVaR 95% worse than -4%/day: meaningful tail risk
- RSI > 70: overbought — momentum stretched
- RSI < 30: oversold — potential capitulation
- MACD below Signal: bearish momentum
- 21d Vol > 40%: elevated current volatility regime

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: ticker, period, data source, overall quality assessment)

### What each number means
(Bullet per key metric: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, CAGR, VaR 95%, CVaR 95%, RSI regime, MACD signal, volatility regime)

### Red flags
(If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2-3 sentences max. What a smart non-quant should take away from this dashboard.)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


def _call_gemini_dashboard_explainer(context: dict) -> str:
    """
    Calls Google Gemini API with the dashboard context.
    Falls back to deterministic explanation on any error.
    Direct mirror of _call_gemini_explainer() from 08_portfolio.py.
    """
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_dashboard_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current single-stock dashboard output. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_DASHBOARD_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"maxOutputTokens": 900, "temperature": 0.2},
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")
        parts = candidates[0].get("content", {}).get("parts", [])
        text  = "".join(p.get("text", "") for p in parts).strip()
        return text or _fallback_dashboard_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_dashboard_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
render_data_engine_controls("dashboard")
_sb_sec("Controls")
ticker = _ticker_sb("dash_ticker")
ma_periods = st.sidebar.multiselect(
    "Moving Averages",
    [10, 20, 50, 100, 200],
    default=[20, 50],
    help="Overlay selected moving averages on the price chart.",
)
st.sidebar.markdown("---")
_sb_sec("Chart Options")
show_vol_panel   = st.sidebar.checkbox("Volume panel",     value=True, key="dash_vol")
show_vol21_panel = st.sidebar.checkbox("Volatility panel", value=True, key="dash_v21")
show_rsi_macd    = st.sidebar.checkbox("RSI & MACD panel", value=True, key="dash_rsi")
candle_style = st.sidebar.radio(
    "Price style", ["Candlestick", "Line"], horizontal=True, key="dash_style",
)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;
                background:linear-gradient(135deg,#e8f4fd 0%,#0be0ff 45%,#a55efd 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:4px;">
        📈 Dashboard
    </div>
    <div style="color:rgba(180,200,230,0.55);font-size:0.88rem;margin-bottom:12px;">
        Price · Volume · Volatility · RSI · MACD · Metrics
    </div>
    <div style="height:3px;width:100%;border-radius:999px;
                background:linear-gradient(90deg,
                    rgba(0,245,160,0) 0%,rgba(0,245,160,0.95) 15%,
                    rgba(11,224,255,0.95) 50%,rgba(165,94,253,0.95) 85%,
                    rgba(165,94,253,0) 100%);
                box-shadow:0 0 14px rgba(11,224,255,0.6),0 0 28px rgba(165,94,253,0.3);
                margin-bottom:20px;">
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Data load ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_ticker_data(ticker, start=_start_str())

if df.empty:
    st.warning("No data available — try a different ticker or start date.")
    st.stop()

ret = returns(df)
met = summary_table(ret, cfg.RISK_FREE_RATE)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_get(key: str) -> str:
    val = met.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, float) and pd.isna(val):
        return "N/A"
    return str(val)


def _color_class(key: str, value: str) -> str:
    if value == "N/A":
        return "qe-metric-na"
    try:
        parsed = float(
            value.replace("%", "").replace("x", "").replace("$", "").replace(",", "").strip()
        )
        if key in {"Max Drawdown", "CVaR 95%", "VaR 95%"}:
            return "qe-metric-neg"
        if parsed > 0:
            return "qe-metric-pos"
        if parsed < 0:
            return "qe-metric-neg"
    except Exception:
        pass
    return "qe-metric-neutral" if "Volatility" in key else ""


def _metric_row(label: str, sub: str) -> str:
    value       = _safe_get(label)
    value_class = _color_class(label, value)
    return f"""
    <div class="qe-stat-row">
        <div>
            <div class="qe-stat-label">{label}</div>
            <div class="qe-stat-sub">{sub}</div>
        </div>
        <div class="qe-stat-value {value_class}">{value}</div>
    </div>"""


# ── Top metric bar ─────────────────────────────────────────────────────────────
last_close  = df["Close"].iloc[-1]
prev_close  = df["Close"].iloc[-2] if len(df) > 1 else last_close
pct_chg     = (last_close - prev_close) / prev_close * 100
vol_today   = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
ann_vol_pct = _safe_get("Ann. Volatility")
sharpe_val  = _safe_get("Sharpe")

chg_sign = "+" if pct_chg >= 0 else ""
chg_cls  = "badge-green" if pct_chg >= 0 else "badge-red"
vol_fmt  = f"{vol_today/1e6:.2f}M" if vol_today >= 1e6 else f"{vol_today/1e3:.0f}K"

st.markdown(
    f"""
<div class="qe-topbar">
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Price</div>
            <div class="qe-topbar-val">${last_close:,.2f}
                <span class="qe-topbar-badge {chg_cls}">{chg_sign}{pct_chg:.2f}%</span>
            </div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Volume</div>
            <div class="qe-topbar-val">{vol_fmt}</div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Volatility</div>
            <div class="qe-topbar-val">
                <span class="qe-topbar-badge badge-yellow">{ann_vol_pct}</span>
            </div>
        </div>
    </div>
    <div class="qe-topbar-pill">
        <div>
            <div class="qe-topbar-label">Sharpe</div>
            <div class="qe-topbar-val">
                <span class="qe-topbar-badge badge-blue">{sharpe_val}</span>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Section: Market Overview ───────────────────────────────────────────────────
st.markdown("""
<div class="qe-section-head">
    <h3>Market Overview</h3>
    <div class="qe-section-dot"></div>
    <div class="qe-section-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1421",
    margin=dict(l=8, r=8, t=8, b=8),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(13,20,33,0.85)", bordercolor="#1e2d47",
        borderwidth=1, font=dict(size=10),
    ),
)
GRID = dict(showgrid=True, gridcolor="#172035", gridwidth=0.6)


def _build_price_figure() -> go.Figure:
    has_volume  = show_vol_panel and "Volume" in df.columns
    rows        = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=row_heights)
    if candle_style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#4ade80", decreasing_line_color="#f87171",
            increasing_fillcolor="rgba(74,222,128,0.85)",
            decreasing_fillcolor="rgba(248,113,113,0.85)",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(color="#7dd3fc", width=2),
            fill="tozeroy", fillcolor="rgba(125,211,252,0.07)",
        ), row=1, col=1)
    ma_colors = {10: "#facc15", 20: "#a78bfa", 50: "#fb923c", 100: "#34d399", 200: "#f472b6"}
    for period in ma_periods:
        ma = df["Close"].rolling(period).mean()
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma.values, name=f"MA {period}",
            line=dict(color=ma_colors.get(period, "#ffffff"), width=1.4, dash="dot"),
        ), row=1, col=1)
    if has_volume:
        bar_colors = [
            "rgba(74,222,128,0.65)" if c >= o else "rgba(248,113,113,0.65)"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=bar_colors, showlegend=False), row=2, col=1)
        avg_vol = df["Volume"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=avg_vol.index, y=avg_vol.values, name="Vol MA(20)",
            line=dict(color="#facc15", width=1.2, dash="dot"),
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         tickfont=dict(size=9), title_font=dict(size=9))
    fig.update_layout(**PLOT_THEME, height=400, xaxis_rangeslider_visible=False)
    fig.update_xaxes(**GRID, rangeslider_visible=False)
    fig.update_yaxes(**GRID)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                     tickfont=dict(size=9), title_font=dict(size=9))
    return fig


def _build_rsi_macd_figure() -> go.Figure:
    close  = df["Close"]
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, float("nan"))
    rsi    = 100 - (100 / (1 + rs))
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.38, 0.38, 0.24],
                        subplot_titles=["RSI (14)", "MACD", ""])
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.07)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(74,222,128,0.07)",  line_width=0, row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.4)", line_width=1, row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(74,222,128,0.4)",  line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI",
                             line=dict(color="#a78bfa", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD",
                             line=dict(color="#7dd3fc", width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Signal",
                             line=dict(color="#fb923c", width=1.5)), row=2, col=1)
    hist_colors = ["rgba(74,222,128,0.65)" if v >= 0 else "rgba(248,113,113,0.65)" for v in hist]
    fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histogram",
                         marker_color=hist_colors, showlegend=False), row=3, col=1)
    fig.update_layout(**PLOT_THEME, height=400)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    fig.update_yaxes(title_text="RSI",  row=1, col=1, tickfont=dict(size=9), range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Hist", row=3, col=1, tickfont=dict(size=9))
    return fig


def _build_volatility_figure() -> go.Figure:
    vol21    = ret.rolling(21).std() * (252 ** 0.5) * 100
    vol_slow = vol21.rolling(21).mean()
    vmax     = max(float(vol21.max()) if not vol21.dropna().empty else 0.0, 45.0)
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=20,        fillcolor="rgba(74,222,128,0.04)",  line_width=0)
    fig.add_hrect(y0=20, y1=40,        fillcolor="rgba(250,204,21,0.05)",  line_width=0)
    fig.add_hrect(y0=40, y1=vmax*1.05, fillcolor="rgba(248,113,113,0.06)", line_width=0)
    fig.add_trace(go.Scatter(x=vol21.index, y=vol21.values, name="21d Vol",
                             line=dict(color="#a78bfa", width=2.2),
                             fill="tozeroy", fillcolor="rgba(167,139,250,0.12)"))
    fig.add_trace(go.Scatter(x=vol_slow.index, y=vol_slow.values, name="Trend",
                             line=dict(color="#7dd3fc", width=1.4, dash="dot")))
    fig.update_layout(**PLOT_THEME, height=280)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID, ticksuffix="%", title_text="Volatility")
    return fig


# ── Row 1: Price Chart  |  RSI & MACD ─────────────────────────────────────────
col_left, col_right = st.columns([1.05, 0.95], gap="medium")
with col_left:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Price Chart</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Candles · Moving averages · Volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.plotly_chart(_build_price_figure(), use_container_width=True, config={"displayModeBar": False})

with col_right:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">RSI &amp; MACD Indicators</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Momentum · Trend divergence · Histogram</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        if show_rsi_macd:
            st.plotly_chart(_build_rsi_macd_figure(), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Enable `RSI & MACD panel` in the sidebar to show this card.")


# ── Row 2: Key Metrics  |  Risk & Return ──────────────────────────────────────
col_bl, col_br = st.columns([0.95, 1.05], gap="medium")
with col_bl:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Key Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Core performance numbers</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qe-stat-table">'
            + _metric_row("Sharpe",       "Risk-adjusted return")
            + _metric_row("Sortino",      "Downside-risk adjusted return")
            + _metric_row("Max Drawdown", "Largest peak-to-trough decline")
            + _metric_row("Win Rate",     "Share of positive periods")
            + "</div>",
            unsafe_allow_html=True,
        )
        if show_vol21_panel:
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(_build_volatility_figure(), use_container_width=True, config={"displayModeBar": False})

with col_br:
    with st.container(border=True):
        st.markdown('<div class="qe-panel-title">Risk &amp; Return Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-sub">Return · Drawdown · VaR · CVaR</div>', unsafe_allow_html=True)
        st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="qe-stat-table">'
            + _metric_row("CAGR",           "Compound annual growth rate")
            + _metric_row("Ann. Return",    "Arithmetic annual return")
            + _metric_row("Ann. Volatility","Annualised standard deviation")
            + _metric_row("Calmar",         "CAGR divided by max drawdown")
            + _metric_row("VaR 95%",        "One-day 95% value at risk")
            + _metric_row("CVaR 95%",       "Expected shortfall beyond VaR")
            + "</div>",
            unsafe_allow_html=True,
        )


# ── Row 3: Recent OHLCV ────────────────────────────────────────────────────────
st.markdown("""
<div class="qe-section-head" style="margin-top:18px;">
    <h3>Recent OHLCV Data</h3>
    <div class="qe-section-dot"></div>
    <div class="qe-section-line"></div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="qe-panel-sub">Latest 30 sessions — colour-coded by session direction</div>', unsafe_allow_html=True)
    st.markdown('<div class="qe-panel-rule"></div>', unsafe_allow_html=True)
    tail = df.tail(30).copy()
    if "Volume" in tail.columns:
        tail["Volume"] = tail["Volume"].apply(
            lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if not pd.isna(v) else "-")
        )
    def _style_row(row):
        color = "color: #4ade80" if row.get("Close", 0) >= row.get("Open", 0) else "color: #f87171"
        return [color] * len(row)
    num_cols  = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in tail.columns]
    formatted = tail.copy()
    for col in num_cols:
        formatted[col] = formatted[col].map(
            lambda v, _col=col: f"${v:,.2f}" if not pd.isna(v) else "-"
        )
    st.dataframe(
        formatted.iloc[::-1].style.apply(_style_row, axis=1),
        use_container_width=True, height=280,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE LIVE INDICATOR VALUES (shared by flag engine + context builder)
# Computed once here — RSI, MACD, vol21.
# ══════════════════════════════════════════════════════════════════════════════

_close  = df["Close"]
_delta  = _close.diff()
_gain   = _delta.clip(lower=0).rolling(14).mean()
_loss   = (-_delta.clip(upper=0)).rolling(14).mean()
_rs     = _gain / _loss.replace(0, float("nan"))
_rsi_s  = 100 - (100 / (1 + _rs))
_ema12  = _close.ewm(span=12, adjust=False).mean()
_ema26  = _close.ewm(span=26, adjust=False).mean()
_macd_s = _ema12 - _ema26
_sig_s  = _macd_s.ewm(span=9, adjust=False).mean()
_vol21  = ret.rolling(21).std() * (252 ** 0.5)

rsi_last    = float(_rsi_s.iloc[-1])   if len(_rsi_s.dropna()) > 0   else float("nan")
macd_last   = float(_macd_s.iloc[-1])  if len(_macd_s.dropna()) > 0  else float("nan")
signal_last = float(_sig_s.iloc[-1])   if len(_sig_s.dropna()) > 0   else float("nan")
vol21_last  = float(_vol21.iloc[-1])   if len(_vol21.dropna()) > 0   else float("nan")
data_source = str(df.attrs.get("data_source", "unknown"))


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — Gemini-powered
# Same 3-layer architecture as 08_portfolio.py:
#   Layer 1: Deterministic danger badges (always shown, no AI)
#   Layer 2: "Decode for Me" button → Gemini explanation
#   Layer 3: Structured AI output with disclaimer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Dashboard Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the quantitative dashboard output above into plain English. "
    "It reads the actual numbers from this analysis — not generic descriptions. "
    "It does not change the metrics. It does not give financial advice."
)

# ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
danger_flags = _compute_dashboard_danger_flags(
    ticker=ticker, met=met,
    last_close=last_close, pct_chg=pct_chg,
    rsi_last=rsi_last, macd_last=macd_last,
    signal_last=signal_last, vol21_last=vol21_last,
    data_source=data_source,
)

if danger_flags:
    n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
    n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
    n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

    badge_html = ""
    if n_danger:
        badge_html += (
            f'<span style="background:#dc3232;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
            f'⛔ {n_danger} DANGER</span>'
        )
    if n_warning:
        badge_html += (
            f'<span style="background:#e67e00;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">'
            f'⚠️ {n_warning} WARNING</span>'
        )
    if n_info:
        badge_html += (
            f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;font-weight:600;">'
            f'ℹ️ {n_info} INFO</span>'
        )
    st.markdown(f'<div style="margin:10px 0 6px;">{badge_html}</div>', unsafe_allow_html=True)

    color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
    bg_map    = {
        "DANGER":  "rgba(220,50,50,0.08)",
        "WARNING": "rgba(230,126,0,0.08)",
        "INFO":    "rgba(26,111,160,0.08)",
    }
    for flag in danger_flags:
        st.markdown(
            f"""<div style="
                background:{bg_map[flag['severity']]};
                border-left:3px solid {color_map[flag['severity']]};
                border-radius:0 6px 6px 0;
                padding:10px 14px; margin:6px 0;
                font-size:13px; line-height:1.55;
            ">
              <span style="font-weight:700;color:{color_map[flag['severity']]};">
                {flag['severity']} · {flag['code']}
              </span><br>
              {flag['message']}
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ Pre-flight checks passed — no critical flags detected for this ticker.")

st.markdown("")

# ── LAYER 2: Build context + button ──────────────────────────────────────────
dashboard_context = _build_dashboard_context(
    ticker=ticker, met=met,
    last_close=last_close, pct_chg=pct_chg,
    vol_today=vol_today, rsi_last=rsi_last,
    macd_last=macd_last, signal_last=signal_last,
    vol21_last=vol21_last, n_days=len(df),
    danger_flags=danger_flags, data_source=data_source,
)

# Cache-bust: reset AI summary on ticker/data change (mirrors portfolio)
context_key = json.dumps(
    {k: v for k, v in dashboard_context.items() if k != "danger_flags"},
    sort_keys=True, default=str,
)
if st.session_state.get("dashboard_ai_context_key") != context_key:
    st.session_state.dashboard_ai_context_key = context_key
    st.session_state.dashboard_ai_summary = ""

col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    st.markdown("**What Gemini sees:**")
    preview_rows = [
        {"Field": "Ticker",          "Value": ticker},
        {"Field": "Data source",     "Value": data_source},
        {"Field": "Trading days",    "Value": str(len(df))},
        {"Field": "Last close",      "Value": f"${last_close:,.2f}"},
        {"Field": "Day change",      "Value": f"{pct_chg:+.2f}%"},
        {"Field": "Sharpe",          "Value": met.get("Sharpe", "N/A")},
        {"Field": "Sortino",         "Value": met.get("Sortino", "N/A")},
        {"Field": "Calmar",          "Value": met.get("Calmar", "N/A")},
        {"Field": "Max Drawdown",    "Value": met.get("Max Drawdown", "N/A")},
        {"Field": "Ann. Volatility", "Value": met.get("Ann. Volatility", "N/A")},
        {"Field": "Win Rate",        "Value": met.get("Win Rate", "N/A")},
        {"Field": "VaR 95% (Hist)",  "Value": met.get("VaR 95% (Hist)", "N/A")},
        {"Field": "CVaR 95% (Hist)", "Value": met.get("CVaR 95% (Hist)", "N/A")},
        {"Field": "RSI(14)",         "Value": f"{rsi_last:.1f}" if not np.isnan(rsi_last) else "N/A"},
        {"Field": "MACD signal",     "Value": "Bullish" if macd_last >= signal_last else "Bearish"},
        {"Field": "Vol21 (ann.)",    "Value": f"{vol21_last:.2%}" if not np.isnan(vol21_last) else "N/A"},
        {"Field": "Danger flags",    "Value": str(sum(1 for f in danger_flags if f["severity"] == "DANGER"))},
        {"Field": "Warning flags",   "Value": str(sum(1 for f in danger_flags if f["severity"] == "WARNING"))},
    ]
    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    gemini_key_set = bool(getattr(cfg, "GEMINI_API_KEY", ""))
    if not gemini_key_set:
        st.info(
            "💡 Add `GEMINI_API_KEY` to your `.env` file for Gemini AI explanations. "
            "Without it, a deterministic quant-safe explanation is shown instead."
        )

    decode_clicked = st.button(
        "🤖 Decode for Me",
        type="primary",
        key="dashboard_ai_explain",
        use_container_width=True,
        help="Translates the dashboard output above into plain English using Gemini.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="dashboard_ai_clear",
        use_container_width=True,
    )

with col_ctx:
    st.markdown("**How this works:**")
    st.markdown("""
<div style="
    background: rgba(14,22,42,0.82);
    border: 1px solid rgba(11,224,255,0.18);
    border-radius: 10px;
    padding: 16px 18px;
    font-size:13px;
    line-height:1.65;
">
  <div style="font-weight:700;color:#e8f4fd;margin-bottom:10px;">What happens when you click Decode:</div>
  <ol style="margin:0;padding-left:18px;color:#a8c4d8;">
    <li style="margin-bottom:6px;">
      The <strong>pre-flight checks above run first</strong> — danger flags are always
      deterministic. They appear regardless of whether you click Decode.
    </li>
    <li style="margin-bottom:6px;">
      The actual numbers from this analysis (Sharpe, Sortino, Calmar, Max Drawdown,
      Win Rate, VaR, CVaR, CAGR, RSI reading, MACD signal, 21-day volatility regime,
      data source, and all flags) are sent to Gemini.
    </li>
    <li style="margin-bottom:6px;">
      Gemini explains each number in plain English, flags anything dangerous,
      and writes a plain-English conclusion — using the actual values, not generic descriptions.
    </li>
    <li style="margin-bottom:6px;">
      Output: <strong>4 sections</strong> — what the output says · what each number means ·
      red flags · plain-English conclusion.
    </li>
    <li>
      A <strong>mandatory disclaimer</strong> is appended — this is not financial advice.
    </li>
  </ol>
</div>
""", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.dashboard_ai_summary = ""

if decode_clicked:
    with st.spinner("Gemini is reading the dashboard output and writing your plain-English explanation..."):
        st.session_state.dashboard_ai_summary = _call_gemini_dashboard_explainer(dashboard_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("dashboard_ai_summary"):
    st.markdown("")
    st.markdown(
        """<div style="
            background: rgba(14,22,42,0.82);
            border: 1px solid rgba(11,224,255,0.28);
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 8px;
        ">""",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.dashboard_ai_summary)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("")
    st.markdown(
        """<div style="
            border:1px dashed rgba(11,224,255,0.18);
            border-radius:10px; padding:20px;
            text-align:center;
            color:rgba(200,220,240,0.4);
            font-size:14px;
        ">
          Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
          of the dashboard output above.
        </div>""",
        unsafe_allow_html=True,
    )

# ── FAQs ──────────────────────────────────────────────────────────────────────
st.markdown("")
qe_faq_section("FAQs", [
    ("What should I look at first on the dashboard?", "Start with the top metrics row and the latest OHLCV table. They give a quick read on trend, volatility, and recent market behavior."),
    ("How does this dashboard help me trade?", "It condenses the current state of the symbol into one screen so you can compare trend, risk, and momentum before moving to deeper analysis pages."),
    ("Why is the recent data table important?", "It shows the freshest sessions and helps you spot gaps, large candles, or unusual volume before trusting a signal."),
    ("When should I switch to another page?", "Use the dashboard as the starting point, then move to signals, risk, or backtest once you want a more specific answer."),
    ("What does the AI Decoder do?", "It reads the actual computed numbers from this dashboard and explains what each one means in plain English — no jargon, no formulas. It does not give financial advice."),
    ("When do the danger flags appear?", "Flags are computed deterministically from the metrics every time the page loads — no AI needed. They fire on hard thresholds: negative Sharpe, severe drawdown, RSI extremes, elevated volatility, and bearish MACD crossovers."),
])