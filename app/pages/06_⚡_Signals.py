
"""
app/pages/06_signals.py — QuantEdge Unified Alpha Signal Dashboard
═══════════════════════════════════════════════════════════════════
All original signal logic preserved exactly.

AI LAYER CHANGES (quant-engineer designed):
  - Added an AI explanation layer (structured output, same dashboard logic)
  - Added _compute_danger_flags()  — deterministic pre-flight checks that run BEFORE AI
    so dangerous readings are always flagged, even if the AI call fails
  - Enhanced _build_signal_context() — now includes data_source, danger_flags,
    health scores, and individual signal ICs so the AI gets full context
  - New system prompt: threshold-aware, structured 4-section output, mandatory disclaimer
  - UI: shows deterministic danger badges first, then AI explanation below
  - Fallback: if API key missing or call fails, deterministic explanation is shown

"""

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings; warnings.filterwarnings("ignore")
import json
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.data import get_ohlcv, get_multi_ohlcv, returns
from app.data_engine import (
    render_data_engine_controls, render_single_ticker_input,
    load_ticker_data, get_global_start_date, parse_ticker_list,
)
from core.indicators import (
    add_all_indicators, signal_rsi, signal_macd_crossover,
    signal_bb_mean_reversion, signal_dual_ma,
)
from core.metrics import information_coefficient
from core.alpha_engine import (
    compute_ofi, ofi_signal,
    compute_crowding_score, crowding_weight, crowding_signal,
    compute_iv_skew_proxy, iv_skew_signal, get_real_iv_skew,
    compute_signal_health, monitor_all_signals,
    get_macro_data, compute_macro_regime_score, macro_regime_label,
    combine_signals,
)
from utils.config import cfg

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


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _latest_non_na(series: pd.Series, default: float = 0.0) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return default
    return float(cleaned.iloc[-1])


def _signal_word(value: float) -> str:
    if value > 0:
        return "BUY"
    if value < 0:
        return "SELL"
    return "HOLD"


def _signal_summary_rows(all_signals: dict[str, pd.Series]) -> list[dict]:
    rows: list[dict] = []
    for name, sig in all_signals.items():
        latest = _latest_non_na(sig, 0.0)
        rows.append({"name": name, "value": latest, "label": _signal_word(latest)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# NEW: DETERMINISTIC DANGER FLAGS
# Runs before AI — these are always shown regardless of API availability.
# Each flag has: severity ("DANGER" | "WARNING" | "INFO"), code, message
# ─────────────────────────────────────────────────────────────────────────────

def _compute_danger_flags(
    ticker: str,
    data_source: str,
    latest_combined: int,
    latest_ofi: float,
    latest_skew: float,
    latest_crowd: float,
    crowd_w: float,
    ic_weights: dict[str, float],
    all_signals: dict[str, pd.Series],
    health_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Deterministic pre-flight checks. No AI involved.
    Returns a list of flag dicts with keys: severity, code, message
    """
    flags = []

    # ── Data quality ─────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Analysis for {ticker} is running on SYNTHETIC demo data, "
                "not real market prices. All signals, ICs, and metrics are "
                "illustrative only — do not use for live trading decisions."
            ),
        })
    elif data_source == "mixed":
        flags.append({
            "severity": "WARNING",
            "code": "MIXED_DATA",
            "message": (
                "Data source is marked 'mixed' — historical cache and live data "
                "were blended. Verify that no corporate actions (splits/dividends) "
                "occurred in the overlap period before trusting return calculations."
            ),
        })

    # ── Crowding risk ─────────────────────────────────────────────────────────
    if latest_crowd > 1.3:
        flags.append({
            "severity": "DANGER",
            "code": "CROWDING_DANGER",
            "message": (
                f"Crowding score is {latest_crowd:.2f}x — well above the 1.30x danger threshold. "
                "This means many funds may be in the same trade. Crowded positions can unwind "
                "violently and simultaneously (cf. August 2007 quant crisis). "
                f"Position scalar is already reduced to {crowd_w:.0%}. "
                "Treat any BUY signal here with extreme caution."
            ),
        })
    elif latest_crowd > 1.1:
        flags.append({
            "severity": "WARNING",
            "code": "CROWDING_ELEVATED",
            "message": (
                f"Crowding is elevated at {latest_crowd:.2f}x (above the 1.10x warning level). "
                f"Position size has been scaled down to {crowd_w:.0%} automatically. "
                "Monitor for further increase."
            ),
        })

    # ── Signal conflict: combined vs volume pressure ───────────────────────────
    ofi_direction = 1 if latest_ofi > 0.5 else (-1 if latest_ofi < -0.5 else 0)
    if ofi_direction != 0 and latest_combined != 0 and ofi_direction != latest_combined:
        flags.append({
            "severity": "WARNING",
            "code": "SIGNAL_CONFLICT",
            "message": (
                f"Volume pressure (OFI z={latest_ofi:.2f}) disagrees with the combined signal "
                f"({_signal_word(latest_combined)}). When price-based signals and flow signals "
                "conflict, reliability drops significantly. Consider waiting for alignment."
            ),
        })

    # ── Low IC across all signals ─────────────────────────────────────────────
    max_ic = max(ic_weights.values(), default=0.0)
    active_count = sum(1 for v in ic_weights.values() if v > 0)
    if max_ic < 0.02:
        flags.append({
            "severity": "WARNING",
            "code": "LOW_IC",
            "message": (
                f"The best IC weight across all signals is only {max_ic:.4f} — "
                "very close to noise level (threshold: 0.02). "
                "In this regime, the signal stack has minimal predictive power. "
                "IC > 0.05 is the conventional minimum for a tradeable signal."
            ),
        })
    elif active_count == 0:
        flags.append({
            "severity": "DANGER",
            "code": "NO_ACTIVE_SIGNALS",
            "message": (
                "Zero signals have positive IC weight. The combined signal is "
                "effectively uninformative. Do not trade based on this output."
            ),
        })

    # ── Signal health: decaying signals ──────────────────────────────────────
    if health_df is not None and not health_df.empty and "Health" in health_df.columns:
        decaying = health_df[health_df["Health"] < 25]["Signal"].tolist()
        if decaying:
            flags.append({
                "severity": "WARNING",
                "code": "SIGNAL_DECAY",
                "message": (
                    f"Signal(s) {', '.join(decaying)} have health scores below 25 — "
                    "the kill threshold. These signals are in alpha decay and should not "
                    "be included in live trading until health recovers above 50."
                ),
            })

    # ── Skew fear spike ───────────────────────────────────────────────────────
    if latest_skew < -1.5:
        flags.append({
            "severity": "WARNING",
            "code": "SKEW_FEAR",
            "message": (
                f"Realized skew z-score is {latest_skew:.2f} — strongly negative, "
                "indicating elevated fear or downside stress in recent returns. "
                "This is consistent with a market under stress. "
                "BUY signals during high fear periods carry higher tail risk."
            ),
        })

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED: SIGNAL CONTEXT BUILDER
# Now includes data_source, danger_flags, health_scores, individual ICs
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal_context(
    ticker: str,
    fwd_days: int,
    latest_combined: int,
    latest_ofi: float,
    latest_skew: float,
    latest_crowd: float,
    crowd_w: float,
    ic_weights: dict[str, float],
    all_signals: dict[str, pd.Series],
    data_source: str = "unknown",
    danger_flags: list[dict] | None = None,
    health_df: pd.DataFrame | None = None,
    fwd: pd.Series | None = None,
) -> dict:
    signal_rows = _signal_summary_rows(all_signals)
    active_weights = [
        {"signal": name, "weight": round(weight, 4)}
        for name, weight in sorted(ic_weights.items(), key=lambda item: item[1], reverse=True)
        if weight > 0
    ]
    top_signals = [
        {
            "signal": row["name"],
            "latest_value": row["value"],
            "direction": row["label"],
            "ic_weight": round(ic_weights.get(row["name"], 0.0), 4),
        }
        for row in sorted(signal_rows, key=lambda item: abs(item["value"]), reverse=True)
    ]

    # Individual ICs for each signal
    individual_ics = {}
    if fwd is not None:
        for name, sig in all_signals.items():
            common = sig.dropna().index.intersection(fwd.dropna().index)
            if len(common) > 20:
                individual_ics[name] = round(
                    information_coefficient(sig[common].astype(float), fwd[common]), 4
                )

    # Health scores
    health_scores = {}
    if health_df is not None and not health_df.empty:
        for _, row in health_df.iterrows():
            health_scores[row["Signal"]] = {
                "health_score": round(float(row.get("Health", 0)), 1),
                "status": str(row.get("Status", "Unknown")),
            }

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "ticker": ticker,
        "forward_window_days": fwd_days,
        "data_source": data_source,  # "demo" | "real" | "mixed" | "unknown"

        # ── Master signal ─────────────────────────────────────────────────
        "combined_signal_value": latest_combined,
        "combined_signal_direction": _signal_word(latest_combined),

        # ── Flow & fear ───────────────────────────────────────────────────
        "volume_pressure_z": round(latest_ofi, 3),
        "volume_pressure_interpretation": (
            "Strong buying pressure" if latest_ofi > 1.5 else
            "Mild buying pressure" if latest_ofi > 0.5 else
            "Mild selling pressure" if latest_ofi < -0.5 else
            "Strong selling pressure" if latest_ofi < -1.5 else
            "Neutral"
        ),
        "realized_skew_z": round(latest_skew, 3),
        "realized_skew_interpretation": (
            "Strong fear / downside stress" if latest_skew < -1.5 else
            "Moderate fear" if latest_skew < -0.5 else
            "Calm / low fear" if latest_skew > 0.5 else
            "Neutral"
        ),

        # ── Crowding ──────────────────────────────────────────────────────
        "crowding_score": round(latest_crowd, 3),
        "crowding_status": (
            "DANGER — overcrowded (>1.30x)" if latest_crowd > 1.3 else
            "WARNING — elevated (>1.10x)" if latest_crowd > 1.1 else
            "Normal (<1.10x)"
        ),
        "position_size_scalar": round(crowd_w, 3),

        # ── IC quality ────────────────────────────────────────────────────
        "active_signal_count": sum(1 for v in ic_weights.values() if v > 0),
        "max_ic_weight": round(max(ic_weights.values(), default=0.0), 4),
        "individual_signal_ics": individual_ics,
        "ic_weighted_active_signals": active_weights,

        # ── All signals ───────────────────────────────────────────────────
        "signal_snapshot": top_signals,
        "signal_health_scores": health_scores,

        # ── Pre-computed danger flags ─────────────────────────────────────
        "danger_flags": danger_flags or [],
        "danger_flag_count": len([f for f in (danger_flags or []) if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in (danger_flags or []) if f["severity"] == "WARNING"]),

        # ── Thresholds (so AI knows the reference points) ─────────────────
        "reference_thresholds": {
            "ic_meaningful": 0.05,
            "ic_noise_floor": 0.02,
            "crowding_danger": 1.30,
            "crowding_warning": 1.10,
            "volume_pressure_signal": 0.80,
            "signal_health_healthy": 75,
            "signal_health_kill": 25,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK (no AI needed, always works)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_signal_explanation(context: dict) -> str:
    direction = context["combined_signal_direction"]
    ofi_interp = context["volume_pressure_interpretation"]
    skew_interp = context["realized_skew_interpretation"]
    crowd_status = context["crowding_status"]
    flags = context.get("danger_flags", [])

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}" for f in flags
        )

    top_signals = [s["signal"] for s in context["signal_snapshot"][:3]]
    active = ", ".join(top_signals) if top_signals else "the current signal stack"

    return (
        f"### What the output says\n"
        f"{context['ticker']} is showing a **{direction}** signal on a "
        f"{context['forward_window_days']}-day forward horizon.\n\n"
        f"### What each number means\n"
        f"- **Volume Pressure z = {context['volume_pressure_z']:.2f}**: {ofi_interp}. "
        f"This measures whether more volume is flowing into up-bars vs down-bars.\n"
        f"- **Realized Skew z = {context['realized_skew_z']:.2f}**: {skew_interp}. "
        f"Negative values mean the market has been making asymmetric downside moves.\n"
        f"- **Crowding Score = {context['crowding_score']:.2f}x**: {crowd_status}. "
        f"Your position is sized at {context['position_size_scalar']:.0%} of normal due to crowding.\n"
        f"- **Active signals contributing**: {context['active_signal_count']} of "
        f"{len(context['signal_snapshot'])} (those with positive IC against forward returns).\n"
        f"- **Strongest predictors**: {active}.\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"The quantitative signal stack is pointing {direction} for {context['ticker']}. "
        f"However, always review the flags above before acting.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE AI EXPLAINER
# Uses the Gemini API directly via urllib (no new dependency)
# System prompt is threshold-aware, danger-first, structured, with disclaimer
# ─────────────────────────────────────────────────────────────────────────────

_AI_SYSTEM_PROMPT = """You are a senior quant analyst embedded inside a professional algorithmic trading dashboard.

Your sole job: explain the signal output to a NON-TECHNICAL user — a portfolio manager, family office client, or founder who understands markets but not the mathematics.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key number in one plain English sentence. Do not skip any metric.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should sell" — explain what the signals say, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No LaTeX. No formulas.

THRESHOLD KNOWLEDGE (built in):
- IC weight < 0.02: signal is noise, not worth trusting
- IC weight > 0.05: signal has meaningful predictive power
- Crowding > 1.30x: danger zone — positions at risk of violent unwind
- Crowding > 1.10x: elevated — reduce size
- Volume pressure z > 0.80: buying pressure; < -0.80: selling pressure
- Realized skew z < -0.50: fear / downside stress in recent price action
- Signal health < 25: signal is decaying — kill it
- Signal health > 75: signal is healthy and reliable
- Combined signal: +1 = BUY, -1 = SELL, 0 = HOLD

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: the master signal, what it means, how confident the stack is)

### What each number means
(Bullet per key metric: volume pressure, skew, crowding, position scalar, active signal count, max IC weight)

### Red flags
(If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2-3 sentences max. What a smart non-quant should take away from this output.)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


def _call_gemini_explainer(context: dict) -> str:
    """
    Calls the configured Gemini API with the signal context.
    Falls back to deterministic explanation on any error.
    """
    gemini_api_key = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash")

    if not gemini_api_key:
        return _fallback_signal_explanation(context)

    system_prompt = _AI_SYSTEM_PROMPT
    user_prompt = (
        "Here is the current signal output from the dashboard. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(context, indent=2, default=str)
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 800,
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_api_key}"
    )
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
            return _fallback_signal_explanation(context)
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        return text or _fallback_signal_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_signal_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Signals | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("📡 Unified Alpha Signal Dashboard")
st.caption(
    "Volume Pressure · Crowding · Realized Skew · Signal Health · Macro Regime · IC-Weighted Combined Signal"
)
qe_neon_divider()

# ── Controls ──────────────────────────────────────────────────────────────────
render_data_engine_controls("signals")
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
ticker  = render_single_ticker_input("Ticker", key="sig_ticker",
            default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
            container=c1)
fwd_days = c2.slider("IC Forward Window (days)", 1, 21, 5)
ofi_thresh  = c3.slider("OFI threshold", 0.3, 2.0, 0.8, 0.1)
skew_thresh = c4.slider("Skew threshold", 0.3, 2.0, 0.7, 0.1)
start = pd.to_datetime(get_global_start_date())

if "signal_result" not in st.session_state:
    st.session_state.signal_result = None

run_clicked = st.button("Run Signal Analysis", type="primary")
if run_clicked:
    with st.spinner("Loading data & computing all signals..."):
        df      = load_ticker_data(ticker, start=str(start))
        df_ind  = add_all_indicators(df)
        ret     = returns(df)
        fwd     = ret.shift(-fwd_days)

        ofi_z    = compute_ofi(df)
        ofi_sig  = ofi_signal(df, threshold=ofi_thresh)
        skew_z   = compute_iv_skew_proxy(df)
        skew_sig = iv_skew_signal(df, threshold=skew_thresh)
        crowd    = compute_crowding_score(ret)
        crowd_w  = crowding_weight(ret)

        rsi_sig  = signal_rsi(df_ind)
        macd_sig = signal_macd_crossover(df_ind)
        bb_sig   = signal_bb_mean_reversion(df_ind)
        dma_sig  = signal_dual_ma(df_ind, 20, 50)

        all_signals = {
            "RSI":           rsi_sig,
            "MACD":          macd_sig,
            "BB Reversion":  bb_sig,
            "Dual MA":       dma_sig,
            "OFI":           ofi_sig,
            "Realized Skew": skew_sig,
        }

        combined_sig, ic_weights = combine_signals(all_signals, ret, fwd_days=fwd_days)

        latest_combined = int(combined_sig.dropna().iloc[-1]) if not combined_sig.dropna().empty else 0
        latest_ofi      = float(ofi_z.dropna().iloc[-1]) if not ofi_z.dropna().empty else 0.0
        latest_skew     = float(skew_z.dropna().iloc[-1]) if not skew_z.dropna().empty else 0.0
        latest_crowd    = float(crowd.dropna().iloc[-1]) if not crowd.dropna().empty else 1.0
        data_source     = str(df.attrs.get("data_source", "unknown"))

        st.session_state.signal_result = {
            "df": df, "df_ind": df_ind, "ret": ret, "fwd": fwd,
            "ofi_z": ofi_z, "ofi_sig": ofi_sig,
            "skew_z": skew_z, "skew_sig": skew_sig,
            "crowd": crowd, "crowd_w": crowd_w,
            "rsi_sig": rsi_sig, "macd_sig": macd_sig, "bb_sig": bb_sig, "dma_sig": dma_sig,
            "all_signals": all_signals,
            "combined_sig": combined_sig,
            "ic_weights": ic_weights,
            "latest_combined": latest_combined,
            "latest_ofi": latest_ofi,
            "latest_skew": latest_skew,
            "latest_crowd": latest_crowd,
            "data_source": data_source,
        }
        # Clear any previous AI summary when re-running analysis
        st.session_state.signal_ai_summary = ""

signal_result = st.session_state.signal_result
if signal_result is None:
    st.info("Configure the inputs above, then press Run Signal Analysis.")
    qe_faq_section("FAQs", [
        ("How do I use the signal dashboard?", "Pick a ticker, set the thresholds, and click Run Signal Analysis. That computes the full signal stack and stores it for the current session."),
        ("What does the combined signal represent?", "It is an IC-weighted blend of the available signals, so stronger historical predictors get more influence than weak ones."),
        ("Why should I check crowding and health?", "A signal can look good on paper but still be crowded or decaying. Those tabs help you avoid using a fragile setup."),
        ("What should I do after the signal changes?", "Treat the new signal as a prompt to review price context, regime context, and macro confirmation before acting."),
    ])
    st.stop()

df          = signal_result["df"]
df_ind      = signal_result["df_ind"]
ret         = signal_result["ret"]
fwd         = signal_result["fwd"]
ofi_z       = signal_result["ofi_z"]
ofi_sig     = signal_result["ofi_sig"]
skew_z      = signal_result["skew_z"]
skew_sig    = signal_result["skew_sig"]
crowd       = signal_result["crowd"]
crowd_w     = signal_result["crowd_w"]
rsi_sig     = signal_result["rsi_sig"]
macd_sig    = signal_result["macd_sig"]
bb_sig      = signal_result["bb_sig"]
dma_sig     = signal_result["dma_sig"]
all_signals = signal_result["all_signals"]
combined_sig    = signal_result["combined_sig"]
ic_weights      = signal_result["ic_weights"]
latest_combined = signal_result["latest_combined"]
latest_ofi      = signal_result["latest_ofi"]
latest_skew     = signal_result["latest_skew"]
latest_crowd    = signal_result["latest_crowd"]
data_source     = signal_result.get("data_source", "unknown")

master_label = "🟢 BUY" if latest_combined == 1 else ("🔴 SELL" if latest_combined == -1 else "⚪ HOLD")

st.markdown(f"""
<div style="background:{'rgba(0,200,80,0.12)' if latest_combined==1 else ('rgba(220,50,50,0.12)' if latest_combined==-1 else 'rgba(120,120,120,0.08)')};
            border:1px solid {'#00c850' if latest_combined==1 else ('#dc3232' if latest_combined==-1 else '#666')};
            border-radius:8px;padding:14px 20px;margin-bottom:12px;display:flex;align-items:center;gap:20px">
  <span style="font-size:28px;font-weight:600">{master_label}</span>
  <span style="font-size:13px;opacity:0.7">IC-weighted combination of {len(all_signals)} signals · fwd window = {fwd_days}d · crowding weight = {crowd_w:.2f}x</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Volume Pressure z-score", f"{latest_ofi:.2f}", delta="Buy pressure" if latest_ofi > 0.5 else ("Sell pressure" if latest_ofi < -0.5 else "Neutral"))
m2.metric("Realized Skew z",  f"{latest_skew:.2f}", delta="Fear" if latest_skew < -0.5 else ("Calm" if latest_skew > 0.5 else "Neutral"))
m3.metric("Crowding",         f"{latest_crowd:.2f}x", delta="⚠️ Crowded" if latest_crowd > 1.3 else "Normal")
m4.metric("Crowd Weight",     f"{crowd_w:.0%}")
m5.metric("Active Signals",   sum(1 for v in ic_weights.values() if v > 0))
m6.metric("Best IC Weight",   f"{max(ic_weights.values(), default=0):.4f}")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# TABS — all original, unchanged
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Combined Signal",
    "🌊 Volume Pressure (OFI Proxy)",
    "😰 Realized Skew (IV Proxy)",
    "🏭 Crowding",
    "❤️ Signal Health",
    "🌍 Macro Regime",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("IC-Weighted Combined Signal")
    st.caption(
        "Each signal is weighted by its Spearman IC against forward returns. "
        "Signals with negative IC are automatically excluded. "
        "Final signal is the IC-weighted average, discretised."
    )
    wt_df = pd.DataFrame([
        {"Signal": k, "IC Weight": v, "Active": "Yes" if v > 0 else "No"}
        for k, v in ic_weights.items()
    ]).sort_values("IC Weight", ascending=False)
    fig_wt = px.bar(wt_df, x="Signal", y="IC Weight", color="Active",
                    color_discrete_map={"Yes": "#00c850", "No": "#888"},
                    template="plotly_dark",
                    title=f"IC Weights — {fwd_days}d Forward Return")
    fig_wt.update_layout(height=280, showlegend=True)
    st.plotly_chart(fig_wt, use_container_width=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="white", width=1.2)), row=1, col=1)
    comb_aligned = combined_sig.reindex(df.index, fill_value=0)
    buys  = df.index[comb_aligned ==  1]
    sells = df.index[comb_aligned == -1]
    if len(buys):
        fig.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Close"], mode="markers",
                                 name="Combined Buy",
                                 marker=dict(symbol="triangle-up", size=9, color="lime")), row=1, col=1)
    if len(sells):
        fig.add_trace(go.Scatter(x=sells, y=df.loc[sells, "Close"], mode="markers",
                                 name="Combined Sell",
                                 marker=dict(symbol="triangle-down", size=9, color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=combined_sig.index, y=combined_sig.values,
                             name="Signal", line=dict(color="cyan", width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520,
                      title=f"IC-Weighted Combined Signal — {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    ic_vals = {}
    for name, sig in all_signals.items():
        common = sig.dropna().index.intersection(fwd.dropna().index)
        if len(common) > 20:
            ic_vals[name] = round(information_coefficient(
                sig[common].astype(float), fwd[common]), 4)
    if ic_vals:
        st.subheader(f"IC vs {fwd_days}d Forward Returns")
        ic_cmp = pd.DataFrame([{"Signal": k, "IC": v, "Used": "✅" if v > 0 else "❌"}
                                for k, v in ic_vals.items()])
        st.dataframe(ic_cmp.sort_values("IC", ascending=False),
                     use_container_width=True, hide_index=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Volume Pressure (OFI Proxy)")
    st.caption(
        "**What it is:** OFI measures net buying vs selling pressure, normalised by volume. "
        "Positive = institutions accumulating. Negative = distribution. "
        "**Paper:** Kolm et al. (2023) — Deep order flow imbalance, Mathematical Finance."
    )
    ofi_ic = information_coefficient(
        ofi_sig.reindex(fwd.dropna().index).fillna(0).astype(float),
        fwd.dropna()) if len(fwd.dropna()) > 20 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume Pressure z (latest)", f"{latest_ofi:.3f}")
    c2.metric("Volume Pressure Signal", "BUY" if latest_ofi > ofi_thresh else ("SELL" if latest_ofi < -ofi_thresh else "HOLD"))
    c3.metric(f"Volume Pressure IC ({fwd_days}d)", f"{ofi_ic:.4f}", delta="Signal" if abs(ofi_ic) > 0.05 else "Noise")
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.03)
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=ofi_z.index, y=ofi_z.values, name="Volume Pressure z-score",
                              line=dict(color="cyan", width=1.5),
                              fill="tozeroy", fillcolor="rgba(0,180,216,0.12)"), row=2, col=1)
    fig2.add_hline(y= ofi_thresh, line_dash="dash", line_color="lime", row=2, col=1)
    fig2.add_hline(y=-ofi_thresh, line_dash="dash", line_color="red",  row=2, col=1)
    close_, open_ = df["Close"], df["Open"]
    vol = df["Volume"]
    buy_v  = vol.where(close_ >= open_, 0)
    sell_v = vol.where(close_ <  open_, 0)
    fig2.add_trace(go.Bar(x=df.index, y=buy_v,   name="Buy Vol",  marker_color="rgba(0,200,80,0.6)"),  row=3, col=1)
    fig2.add_trace(go.Bar(x=df.index, y=-sell_v, name="Sell Vol", marker_color="rgba(220,50,50,0.6)"), row=3, col=1)
    fig2.update_layout(template="plotly_dark", height=560,
                       title=f"Volume Pressure Signal — {ticker}", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("⚠️ What This Signal Actually Is"):
        st.markdown("""
**⚠️ Naming Disclosure:** True OFI requires Level-2 tick data. This uses daily OHLCV as a proxy.

**Formula:**
```
buy_vol  = volume on up-close bars (Close >= Open)
sell_vol = volume on down-close bars (Close < Open)
raw      = buy_vol - sell_vol
norm     = raw / rolling_avg_volume
signal   = z-score(norm, window=63)
```
""")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Realized Skew Signal (IV Skew Proxy)")
    st.caption(
        "⚠️ **Proxy disclosure:** True IV skew needs options chain data. "
        "This uses realized return skewness + down-day frequency as a proxy. "
        "**References:** Höfler (2024) SSRN 4869272; Bakshi, Kapadia & Madan (2003)."
    )
    live_iv = get_real_iv_skew(ticker)
    if live_iv:
        st.success(f"✅ Live options data available for {ticker}")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Put IV (avg)",  f"{live_iv['put_iv']:.1%}")
        lc2.metric("Call IV (avg)", f"{live_iv['call_iv']:.1%}")
        lc3.metric("Realized Skew", f"{live_iv['skew']:.4f}",
                   delta="Fear" if live_iv['skew'] > 0.05 else "Normal")
        lc4.metric("ATM IV",        f"{live_iv['atm_iv']:.1%}")
    else:
        st.info("Live options data unavailable. Using OHLC-based skew proxy.")
    skew_ic = information_coefficient(
        skew_sig.reindex(fwd.dropna().index).fillna(0).astype(float),
        fwd.dropna()) if len(fwd.dropna()) > 20 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Skew z (latest)",    f"{latest_skew:.3f}")
    c2.metric("Signal",             "SELL (fear)" if latest_skew < -skew_thresh else ("BUY (calm)" if latest_skew > skew_thresh else "HOLD"))
    c3.metric(f"Skew IC ({fwd_days}d)", f"{skew_ic:.4f}")
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.55, 0.45], vertical_spacing=0.04)
    fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    if len(buys):
        fig3.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Close"], mode="markers",
                                  name="Calm (Buy)",
                                  marker=dict(symbol="triangle-up", size=8, color="lime")), row=1, col=1)
    skew_sells = df.index[skew_sig == -1]
    if len(skew_sells):
        fig3.add_trace(go.Scatter(x=skew_sells, y=df.loc[skew_sells, "Close"], mode="markers",
                                  name="Fear (Sell)",
                                  marker=dict(symbol="triangle-down", size=8, color="orange")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=skew_z.index, y=skew_z.values, name="Skew z-score",
                              line=dict(color="orange", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,165,0,0.10)"), row=2, col=1)
    fig3.add_hline(y= skew_thresh, line_dash="dash", line_color="lime",   row=2, col=1)
    fig3.add_hline(y=-skew_thresh, line_dash="dash", line_color="orange", row=2, col=1)
    fig3.update_layout(template="plotly_dark", height=500,
                       title=f"Realized Skew Signal (IV Proxy) — {ticker}")
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Factor Crowding Detector")
    st.caption(
        "**Paper:** Hua & Sun (2024) — Dynamics of Factor Crowding, SSRN 5023380. "
        "Crowded trades unwind violently — like August 2007 and July 2025."
    )
    crowd_status = "🔴 Overcrowded" if latest_crowd > 1.3 else ("🟡 Elevated" if latest_crowd > 1.1 else "🟢 Normal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crowding Score", f"{latest_crowd:.3f}", delta="Danger zone" if latest_crowd > 1.3 else "OK")
    c2.metric("Status", crowd_status)
    c3.metric("Position Weight Scalar", f"{crowd_w:.0%}")
    c4.metric("Suggested Action", "Reduce 75%" if latest_crowd > 1.3 else ("Reduce 35%" if latest_crowd > 1.1 else "Full size"))
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5], vertical_spacing=0.04)
    fig4.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=crowd.index, y=crowd.values, name="Crowding Score",
                              line=dict(color="yellow", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"), row=2, col=1)
    fig4.add_hline(y=1.3, line_dash="dash", line_color="red",
                   annotation_text="Overcrowded threshold", row=2, col=1)
    fig4.add_hline(y=1.1, line_dash="dot",  line_color="orange", row=2, col=1)
    fig4.add_hline(y=0.8, line_dash="dot",  line_color="lime",
                   annotation_text="Undercrowded", row=2, col=1)
    fig4.update_layout(template="plotly_dark", height=500,
                       title=f"Crowding Score — {ticker}")
    st.plotly_chart(fig4, use_container_width=True)
    st.subheader("Multi-Ticker Crowding Comparison")
    port_raw = st.text_input("Tickers", value=", ".join(cfg.DEFAULT_TICKERS[:4]), key="crowd_tickers")
    port_tickers = parse_ticker_list(port_raw)
    if len(port_tickers) >= 2:
        with st.spinner("Loading multi-ticker data..."):
            multi = get_multi_ohlcv(port_tickers, start=str(start.date()))
        crowd_df = crowding_signal(multi)
        if not crowd_df.empty:
            fig_crowd = px.bar(crowd_df, x="Ticker", y="Crowding Score", color="Status",
                               color_discrete_map={"🔴 Overcrowded": "red", "🟡 Elevated": "orange", "🟢 Normal": "green"},
                               template="plotly_dark", title="Crowding Score by Ticker")
            fig_crowd.add_hline(y=1.3, line_dash="dash", line_color="red")
            fig_crowd.update_layout(height=320)
            st.plotly_chart(fig_crowd, use_container_width=True)
            st.dataframe(crowd_df, use_container_width=True, hide_index=True)

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Signal Health & Alpha Decay Monitor")
    st.caption(
        "**Paper:** AlphaAgent (KDD 2025) — regularization to counteract alpha decay. "
        "Harvey, Liu & Zhu (2016) — most discovered factors are false positives."
    )
    with st.spinner("Computing signal health..."):
        health_df = monitor_all_signals(df, fwd_days=fwd_days)
    fig_h = px.bar(health_df, x="Signal", y="Health", color="Health",
                   color_continuous_scale=["red", "orange", "yellow", "green"],
                   range_color=[0, 100], template="plotly_dark",
                   title=f"Signal Health Scores — {ticker} ({fwd_days}d forward)")
    fig_h.add_hline(y=75, line_dash="dash", line_color="lime",  annotation_text="Healthy (75)")
    fig_h.add_hline(y=25, line_dash="dash", line_color="red",   annotation_text="Kill threshold (25)")
    fig_h.update_layout(height=320)
    st.plotly_chart(fig_h, use_container_width=True)
    st.dataframe(
        health_df.style
            .background_gradient(subset=["Health"], cmap="RdYlGn", vmin=0, vmax=100)
            .format({"Health": "{:.1f}", "IC Mean": "{:.4f}",
                     "IC Std": "{:.4f}", "IC Trend": "{:.6f}", "Weight": "{:.0%}"}),
        use_container_width=True, hide_index=True
    )
    best_sig_name = health_df.iloc[0]["Signal"]
    best_sig      = all_signals.get(best_sig_name, rsi_sig)
    h_detail      = compute_signal_health(best_sig, fwd, window=63)
    if "rolling_ic" in h_detail and not h_detail["rolling_ic"].empty:
        ic_ts = h_detail["rolling_ic"]
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(y=ic_ts.values, name=f"{best_sig_name} Rolling IC",
                                    line=dict(color="cyan", width=1.5),
                                    fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"))
        fig_ic.add_hline(y=0.05, line_dash="dash", line_color="lime",
                         annotation_text="IC = 0.05 (meaningful)")
        fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
        x = np.arange(len(ic_ts))
        slope, intercept = np.polyfit(x, ic_ts.fillna(0).values, 1)
        trend = slope * x + intercept
        fig_ic.add_trace(go.Scatter(y=trend, name="IC Trend",
                                    line=dict(color="orange", width=2, dash="dot")))
        fig_ic.update_layout(template="plotly_dark", height=320,
                              title=f"Rolling 63d IC — {best_sig_name} on {ticker}")
        st.plotly_chart(fig_ic, use_container_width=True)

# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Cross-Asset Macro Regime Signal")
    st.caption(
        "Combines VIX, credit spreads, dollar index, and yield curve "
        "into one macro regime score."
    )
    with st.spinner("Fetching macro data (VIX, HYG, IEF, DXY, rates)..."):
        macro_df = get_macro_data(start=str(start.date()))
    if macro_df is not None and not macro_df.empty:
        macro_score = compute_macro_regime_score(macro_df)
        latest_macro = float(macro_score.dropna().iloc[-1]) if not macro_score.dropna().empty else 0.0
        regime_label_str, pos_scalar = macro_regime_label(latest_macro)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Macro Score",     f"{latest_macro:.2f}")
        mc2.metric("Regime",          regime_label_str)
        mc3.metric("Position Scalar", f"{pos_scalar:.0%}")
        mc4.metric("VIX (latest)",    f"{macro_df['vix'].iloc[-1]:.1f}" if "vix" in macro_df.columns else "N/A")
        fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.5, 0.5], vertical_spacing=0.04)
        fig_m.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{ticker} Price",
                                   line=dict(color="white", width=1)), row=1, col=1)
        common_m = macro_score.index.intersection(df.index)
        if len(common_m):
            fig_m.add_trace(go.Scatter(x=common_m,
                                       y=macro_score.reindex(common_m).values,
                                       name="Macro Score",
                                       line=dict(color="magenta", width=2),
                                       fill="tozeroy", fillcolor="rgba(200,0,200,0.08)"), row=2, col=1)
        fig_m.add_hline(y= 1.0, line_dash="dash", line_color="lime",
                        annotation_text="Risk-On",  row=2, col=1)
        fig_m.add_hline(y=-1.0, line_dash="dash", line_color="red",
                        annotation_text="Risk-Off", row=2, col=1)
        fig_m.add_hline(y=0,    line_dash="dot",  line_color="gray", row=2, col=1)
        fig_m.update_layout(template="plotly_dark", height=520,
                             title="Cross-Asset Macro Regime Score")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.warning(
            "Macro data unavailable in current environment. "
            "On your local machine with internet access, this shows live VIX, HYG, IEF, DXY data."
        )


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — NEW, quant-engineer designed
# Positioned at the bottom, after all signal output is complete.
# Three-layer design:
#   Layer 1: Deterministic danger badges (always shown, no AI)
#   Layer 2: "Decode This" button → AI explanation
#   Layer 3: Structured AI output with disclaimer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# Header
st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🧠 AI Signal Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by AI
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the quantitative output above into plain English. "
    "It reads the actual numbers from this analysis — it does not give generic descriptions. "
    "It does not change the signal. It does not give financial advice."
)

# ── Compute health_df for danger flags (may already exist from tab5) ──────────
try:
    _health_df_for_flags = health_df  # reuse if tab5 computed it
except NameError:
    _health_df_for_flags = None

# ── LAYER 1: Deterministic danger flags (no AI, always runs) ─────────────────
danger_flags = _compute_danger_flags(
    ticker=ticker,
    data_source=data_source,
    latest_combined=latest_combined,
    latest_ofi=latest_ofi,
    latest_skew=latest_skew,
    latest_crowd=latest_crowd,
    crowd_w=crowd_w,
    ic_weights=ic_weights,
    all_signals=all_signals,
    health_df=_health_df_for_flags,
)

if danger_flags:
    n_danger  = sum(1 for f in danger_flags if f["severity"] == "DANGER")
    n_warning = sum(1 for f in danger_flags if f["severity"] == "WARNING")
    n_info    = sum(1 for f in danger_flags if f["severity"] == "INFO")

    badge_html = ""
    if n_danger:
        badge_html += f'<span style="background:#dc3232;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">⛔ {n_danger} DANGER</span>'
    if n_warning:
        badge_html += f'<span style="background:#e67e00;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;margin-right:6px;">⚠️ {n_warning} WARNING</span>'
    if n_info:
        badge_html += f'<span style="background:#1a6fa0;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;">ℹ️ {n_info} INFO</span>'

    st.markdown(
        f'<div style="margin:10px 0 6px;">{badge_html}</div>',
        unsafe_allow_html=True,
    )

    for flag in danger_flags:
        color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
        bg_map    = {"DANGER": "rgba(220,50,50,0.08)", "WARNING": "rgba(230,126,0,0.08)", "INFO": "rgba(26,111,160,0.08)"}
        st.markdown(
            f"""<div style="
                background:{bg_map[flag['severity']]};
                border-left:3px solid {color_map[flag['severity']]};
                border-radius:0 6px 6px 0;
                padding:10px 14px;
                margin:6px 0;
                font-size:13px;
                line-height:1.55;
            ">
              <span style="font-weight:700;color:{color_map[flag['severity']]};">{flag['severity']} · {flag['code']}</span><br>
              {flag['message']}
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ Pre-flight checks passed — no critical flags detected for this signal stack.")

st.markdown("")

# ── LAYER 2: Build full context + button ─────────────────────────────────────
signal_context = _build_signal_context(
    ticker=ticker,
    fwd_days=fwd_days,
    latest_combined=latest_combined,
    latest_ofi=latest_ofi,
    latest_skew=latest_skew,
    latest_crowd=latest_crowd,
    crowd_w=crowd_w,
    ic_weights=ic_weights,
    all_signals=all_signals,
    data_source=data_source,
    danger_flags=danger_flags,
    health_df=_health_df_for_flags,
    fwd=fwd,
)

# Detect if context changed since last explanation
signal_context_key = json.dumps(
    {k: v for k, v in signal_context.items() if k != "danger_flags"},
    sort_keys=True, default=str
)
if st.session_state.get("signal_ai_context_key") != signal_context_key:
    st.session_state.signal_ai_context_key = signal_context_key
    st.session_state.signal_ai_summary = ""

# Context preview + button row
col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    # Show what the AI will receive — key facts only, not the full JSON
    st.markdown("**What the AI sees:**")
    preview_rows = [
        {"Field": "Ticker", "Value": ticker},
        {"Field": "Signal", "Value": _signal_word(latest_combined)},
        {"Field": "Data source", "Value": data_source},
        {"Field": "Fwd window", "Value": f"{fwd_days}d"},
        {"Field": "Crowding", "Value": f"{latest_crowd:.2f}x"},
        {"Field": "Vol pressure z", "Value": f"{latest_ofi:.3f}"},
        {"Field": "Skew z", "Value": f"{latest_skew:.3f}"},
        {"Field": "Active signals", "Value": str(sum(1 for v in ic_weights.values() if v > 0))},
        {"Field": "Max IC weight", "Value": f"{max(ic_weights.values(), default=0):.4f}"},
        {"Field": "Danger flags", "Value": str(len([f for f in danger_flags if f['severity'] == 'DANGER']))},
    ]
    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    decode_clicked = st.button(
        "🧠 Decode for Me",
        type="primary",
        key="signal_ai_explain",
        use_container_width=True,
        help="Translates the signal output above into plain English using the actual numbers from this analysis.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="signal_ai_clear",
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
      The <strong>pre-flight checks above run first</strong> — danger flags are always deterministic.
      They show regardless of whether you click Decode.
    </li>
    <li style="margin-bottom:6px;">
      The actual numbers from this analysis (signal direction, IC weights, crowding score,
      volume pressure, skew, health scores, data source, and all flags) are sent to the AI.
    </li>
    <li style="margin-bottom:6px;">
      The AI explains each number in plain English, flags anything dangerous,
      and writes a plain-English conclusion.
    </li>
    <li style="margin-bottom:6px;">
      The output is structured into 4 sections: what the output says · what each number means ·
      red flags · plain-English conclusion.
    </li>
    <li>
      A <strong>mandatory disclaimer</strong> is appended — this is not financial advice.
    </li>
  </ol>
</div>
""", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.signal_ai_summary = ""

if decode_clicked:
    with st.spinner("The AI is reading the signal stack and writing your plain-English explanation..."):
        st.session_state.signal_ai_summary = _call_gemini_explainer(signal_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("signal_ai_summary"):
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
    st.markdown(st.session_state.signal_ai_summary)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("")
    st.markdown(
        """<div style="
            border:1px dashed rgba(11,224,255,0.18);
            border-radius:10px;
            padding:20px;
            text-align:center;
            color:rgba(200,220,240,0.4);
            font-size:14px;
        ">
          Click <strong>🧠 Decode for Me</strong> to get a plain-English explanation of the signal output above.
        </div>""",
        unsafe_allow_html=True,
    )
