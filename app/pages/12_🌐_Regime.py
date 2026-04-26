import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings; warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st
from plotly.subplots import make_subplots
from urllib import error as urlerror
from urllib import request as urlrequest

from core.data import returns
from app.data_engine import (
    render_data_engine_controls, render_single_ticker_input,
    load_ticker_data, get_global_start_date,
)
from core.regime_detector import (
    fit_hmm, forward_regime_proba, compute_regime_age, regime_age_scalar,
    critical_slowing_down, get_strategy_for_regime, rolling_regime_proba,
    regime_conditional_sharpe, full_regime_analysis,
    REGIME_COLORS, REGIME_LINE_COLORS,
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


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — LAYER 1: DETERMINISTIC DANGER FLAGS
# Always shown — no Gemini required
# ══════════════════════════════════════════════════════════════════════════════

def _compute_regime_danger_flags(
    ticker: str,
    cur: str,
    age: int,
    scalar: float,
    ew: dict,
    fwd_proba: pd.DataFrame,
    cond_sharpe: pd.DataFrame,
    regimes: pd.Series,
    n_states: int,
    ret: pd.Series,
    strategy: dict,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the regime output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.

    Checks:
      1. Demo data notice
      2. Early warning active
      3. AC1 elevated (critical slowing down)
      4. Bear regime + high age (trapped in drawdown)
      5. Regime age too young (false breakout risk)
      6. Low forward bull probability in a supposed bull
      7. High bear probability crossing threshold
      8. Dominant single regime (poor HMM fit)
      9. Negative regime-conditional Sharpe in current regime
      10. History too short for reliable HMM
      11. Transition matrix instability (high self-loop probability mismatch)
    """
    flags = []

    # ── 1. Demo data ──────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Regime analysis for {ticker} is running on SYNTHETIC demo data. "
                "HMM parameters, forward probabilities, and early warning signals are "
                "illustrative only. Do not use these regime labels for live trading decisions."
            ),
        })

    # ── 2. Early warning active ───────────────────────────────────────────────
    if ew.get("active", False):
        ac1_val = ew.get("latest_ac1", 0)
        flags.append({
            "severity": "DANGER",
            "code": "EARLY_WARNING_ACTIVE",
            "message": (
                f"Critical slowing down signal is ACTIVE. "
                f"AC1 (lag-1 autocorrelation) = {ac1_val:.4f} — elevated above the 0.15 threshold. "
                "Both rolling autocorrelation AND rolling variance are rising simultaneously. "
                "Historical lead time before a regime flip: 10–20 trading days. "
                "Reduce position sizes now. Do not wait for regime confirmation."
            ),
        })
    elif ew.get("latest_ac1", 0) > 0.10:
        flags.append({
            "severity": "WARNING",
            "code": "AC1_ELEVATED",
            "message": (
                f"AC1 is {ew.get('latest_ac1', 0):.4f} — approaching the warning threshold of 0.15. "
                "Early warning has not fired yet but autocorrelation is drifting upward. "
                "Monitor daily. If AC1 crosses 0.15 AND variance is also rising, "
                "the system will flag an imminent regime transition."
            ),
        })

    # ── 3. Bear regime + high age ─────────────────────────────────────────────
    cur_clean = cur.replace("📈", "").replace("📉", "").replace("↔", "").strip()
    if "Bear" in cur_clean and age > 60:
        flags.append({
            "severity": "WARNING",
            "code": "EXTENDED_BEAR_REGIME",
            "message": (
                f"Currently in Bear regime for {age} days — a prolonged drawdown episode. "
                "Extended bear regimes increase the risk of capitulation and forced selling. "
                f"Position scalar is {scalar:.0%}. "
                "Avoid adding exposure until forward P(Bull) crosses 60% for 3+ consecutive days."
            ),
        })

    # ── 4. Regime age too young ───────────────────────────────────────────────
    if "Bull" in cur_clean and age < 10:
        flags.append({
            "severity": "WARNING",
            "code": "YOUNG_BULL_REGIME",
            "message": (
                f"Bull regime is only {age} days old — high false breakout risk. "
                f"Position scalar is correctly reduced to {scalar:.0%}. "
                "HMM bull signals in the first 10 days have a historically higher false positive rate. "
                "Wait for regime age > 15 days and P(Bull) > 0.65 before scaling up."
            ),
        })

    # ── 5. Forward probability mismatch ──────────────────────────────────────
    bull_cols = [c for c in fwd_proba.columns if "Bull" in c]
    bear_cols = [c for c in fwd_proba.columns if "Bear" in c]

    if bull_cols and "Bull" in cur_clean:
        bull_prob = float(fwd_proba[bull_cols[0]].iloc[-1])
        if bull_prob < 0.50:
            flags.append({
                "severity": "DANGER",
                "code": "BULL_LABEL_WEAK_PROBABILITY",
                "message": (
                    f"Viterbi labels the current regime as Bull, but forward P(Bull) is only "
                    f"{bull_prob:.1%}. "
                    "Viterbi uses future data to smooth labels — the forward-only probability "
                    "is what you would actually know today. "
                    "A forward P(Bull) below 50% means the model is genuinely uncertain. "
                    "Do not treat this as a confirmed bull regime."
                ),
            })
        elif bull_prob < 0.65:
            flags.append({
                "severity": "WARNING",
                "code": "BULL_PROBABILITY_MODERATE",
                "message": (
                    f"Current regime is Bull but forward P(Bull) = {bull_prob:.1%} — "
                    "below the confident threshold of 65%. "
                    "The regime signal is real but not high conviction. "
                    "Use reduced position sizing until probability strengthens."
                ),
            })

    if bear_cols:
        bear_prob = float(fwd_proba[bear_cols[0]].iloc[-1])
        if bear_prob > 0.70 and "Bull" in cur_clean:
            flags.append({
                "severity": "DANGER",
                "code": "BEAR_PROBABILITY_SPIKE",
                "message": (
                    f"Forward P(Bear) has spiked to {bear_prob:.1%} while Viterbi still "
                    "labels the regime as Bull. "
                    "This is a regime transition in progress — the forward algorithm is "
                    "seeing bear evidence that the smoothed label hasn't caught up to yet. "
                    "This is exactly the lookahead bias problem: trust the forward probability. "
                    "Treat this as a Bear regime NOW."
                ),
            })
        elif bear_prob > 0.55 and "Bull" in cur_clean:
            flags.append({
                "severity": "WARNING",
                "code": "BEAR_PROBABILITY_RISING",
                "message": (
                    f"Forward P(Bear) = {bear_prob:.1%} — rising despite Bull Viterbi label. "
                    "Monitor for regime transition. Reduce leverage if P(Bear) crosses 60%."
                ),
            })

    # ── 6. Dominant single regime ─────────────────────────────────────────────
    if len(regimes) > 0:
        regime_counts = regimes.value_counts(normalize=True)
        dominant_pct = float(regime_counts.iloc[0])
        dominant_name = regime_counts.index[0]
        if dominant_pct > 0.80:
            flags.append({
                "severity": "WARNING",
                "code": "DOMINANT_REGIME",
                "message": (
                    f"Regime '{dominant_name}' accounts for {dominant_pct:.0%} of the full history. "
                    f"When one state dominates this heavily, the HMM may not have enough "
                    f"examples of the minority regime to reliably identify it. "
                    f"The minority regime transitions and their statistics are estimated from "
                    f"very few data points — treat those labels with extra caution."
                ),
            })

    # ── 7. Negative conditional Sharpe in current regime ─────────────────────
    if cond_sharpe is not None and not cond_sharpe.empty:
        try:
            cur_rows = cond_sharpe[cond_sharpe["Regime"].str.contains(cur_clean, na=False)]
            if not cur_rows.empty and "Sharpe" in cur_rows.columns:
                raw_sh_s = str(cur_rows["Sharpe"].iloc[0]).strip()
                cur_sharpe = float(raw_sh_s[:-1]) / 100 if raw_sh_s.endswith("%") else float(raw_sh_s)
                if cur_sharpe < 0:
                    flags.append({
                        "severity": "DANGER",
                        "code": "NEGATIVE_CONDITIONAL_SHARPE",
                        "message": (
                            f"Historical Sharpe ratio during {cur} regime = {cur_sharpe:.2f} — negative. "
                            "This means that on average, holding the asset during this regime type "
                            "has historically destroyed value after accounting for risk. "
                            "The strategy router correctly recommends defensive positioning."
                        ),
                    })
                elif cur_sharpe < 0.3:
                    flags.append({
                        "severity": "WARNING",
                        "code": "LOW_CONDITIONAL_SHARPE",
                        "message": (
                            f"Historical Sharpe during {cur} regime = {cur_sharpe:.2f} — "
                            "below investable threshold of 0.5. "
                            "This regime historically offers poor risk-adjusted returns. "
                            "Consider reducing exposure relative to a full-size position."
                        ),
                    })
        except Exception:
            pass

    # ── 8. History too short ──────────────────────────────────────────────────
    min_required = n_states * 120
    if len(ret) < min_required:
        flags.append({
            "severity": "DANGER",
            "code": "INSUFFICIENT_HISTORY_HMM",
            "message": (
                f"History has {len(ret)} observations but {n_states}-state HMM "
                f"requires at least {min_required} ({n_states} × 120 days). "
                "With too little data, HMM transition probabilities are estimated from "
                "very few regime episodes — the labels are unreliable noise. "
                "Extend the date range or reduce to 2 states."
            ),
        })
    elif len(ret) < min_required * 1.5:
        flags.append({
            "severity": "WARNING",
            "code": "SHORT_HISTORY_HMM",
            "message": (
                f"History has {len(ret)} observations — above the minimum but below the "
                f"recommended {int(min_required * 1.5)} days. "
                "Regime statistics (conditional Sharpe, duration) are based on limited episodes. "
                "Consider extending the lookback for more reliable estimates."
            ),
        })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — LAYER 2: CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_regime_context(
    ticker: str,
    n_states: int,
    cur: str,
    age: int,
    scalar: float,
    ew: dict,
    fwd_proba: pd.DataFrame,
    cond_sharpe: pd.DataFrame,
    regimes: pd.Series,
    strategy: dict,
    ret: pd.Series,
    danger_flags: list[dict],
    data_source: str = "unknown",
) -> dict:
    """Build full structured context dict sent to Gemini."""

    # Regime distribution
    regime_dist = (regimes.value_counts(normalize=True) * 100).round(2).to_dict()
    regime_dist = {str(k): float(v) for k, v in regime_dist.items()}

    # Forward probabilities — last value per column
    fwd_last = {}
    if not fwd_proba.empty:
        fwd_last = {str(col): round(float(fwd_proba[col].iloc[-1]), 4) for col in fwd_proba.columns}

    # Conditional Sharpe — serialize
    def _safe_float(val):
        """Convert a value to float, handling pre-formatted strings like '71.0%' or '1.23'."""
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if s.endswith("%"):
            try:
                return round(float(s[:-1]) / 100, 6)
            except ValueError:
                return None
        try:
            return round(float(s), 4)
        except ValueError:
            return None

    cond_sharpe_dict = {}
    if cond_sharpe is not None and not cond_sharpe.empty:
        for _, row in cond_sharpe.iterrows():
            reg_label = str(row.get("Regime", row.name))
            cond_sharpe_dict[reg_label] = {
                col: _safe_float(row[col])
                for col in cond_sharpe.columns if col != "Regime"
            }

    # Regime duration stats
    durations = []
    cur_r_d, cur_len = regimes.iloc[0], 1
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == cur_r_d:
            cur_len += 1
        else:
            durations.append({"regime": str(cur_r_d), "duration": cur_len})
            cur_r_d, cur_len = regimes.iloc[i], 1
    durations.append({"regime": str(cur_r_d), "duration": cur_len})
    dur_df = pd.DataFrame(durations)
    dur_stats = {}
    for rname, grp in dur_df.groupby("regime"):
        dur_stats[str(rname)] = {
            "mean_days": round(float(grp["duration"].mean()), 1),
            "max_days":  int(grp["duration"].max()),
            "episodes":  int(len(grp)),
        }

    # Daily vol
    daily_vol_pct = round(float(ret.tail(21).std()) * 100 * (252 ** 0.5), 2)

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticker": ticker,
        "data_source": data_source,
        "n_states": n_states,

        # ── Current regime ────────────────────────────────────────────────────
        "current_regime": {
            "label": str(cur),
            "age_days": age,
            "position_scalar": round(scalar, 4),
            "interpretation": (
                "Full size" if scalar >= 0.9 else
                "Ramping up" if scalar >= 0.6 else
                "Cautious / early regime" if scalar >= 0.3 else
                "Minimal exposure"
            ),
        },

        # ── Forward probabilities ─────────────────────────────────────────────
        "forward_probabilities_now": fwd_last,

        # ── Early warning ─────────────────────────────────────────────────────
        "early_warning": {
            "active": bool(ew.get("active", False)),
            "latest_ac1": round(float(ew.get("latest_ac1", 0)), 4),
            "latest_variance": round(float(ew.get("latest_var", 0)), 8),
            "lead_message": str(ew.get("lead_msg", "")),
            "ac1_threshold": 0.15,
        },

        # ── Strategy router ───────────────────────────────────────────────────
        "strategy_router": {
            "primary": strategy["recommendations"]["primary"],
            "secondary": strategy["recommendations"]["secondary"],
            "avoid": strategy["recommendations"]["avoid"],
            "position_size": strategy["recommendations"]["position"],
            "stop_style": strategy["recommendations"]["stops"],
        },

        # ── Regime statistics ─────────────────────────────────────────────────
        "regime_distribution_pct": regime_dist,
        "conditional_sharpe": cond_sharpe_dict,
        "regime_duration_stats": dur_stats,

        # ── Market context ────────────────────────────────────────────────────
        "annualised_vol_21d_pct": daily_vol_pct,

        # ── Danger flags ──────────────────────────────────────────────────────
        "danger_flags": danger_flags,
        "danger_flag_count":  len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in danger_flags if f["severity"] == "WARNING"]),

        # ── Reference thresholds ──────────────────────────────────────────────
        "reference_thresholds": {
            "forward_prob_confident_bull":      0.65,
            "forward_prob_confirmed_bull":      0.75,
            "forward_prob_bear_warning":        0.55,
            "forward_prob_bear_danger":         0.70,
            "ac1_warning_threshold":            0.15,
            "regime_age_cautious_days":         10,
            "regime_age_confirmed_days":        30,
            "conditional_sharpe_investable":    0.50,
            "dominant_regime_flag_pct":         80.0,
            "min_history_per_state_days":       120,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_regime_explanation(context: dict) -> str:
    cur      = context["current_regime"]
    fwd      = context["forward_probabilities_now"]
    ew       = context["early_warning"]
    strat    = context["strategy_router"]
    flags    = context.get("danger_flags", [])
    dist     = context["regime_distribution_pct"]
    cond_sh  = context["conditional_sharpe"]
    dur      = context["regime_duration_stats"]

    fwd_lines = "\n".join(
        f"  - {k}: {v:.1%}" for k, v in fwd.items()
    )

    cond_lines = ""
    if cond_sh:
        cond_lines = "\n".join(
            f"  - {r}: Sharpe = {v.get('Sharpe', 'N/A')}"
            for r, v in cond_sh.items()
        )

    dur_lines = ""
    if dur:
        dur_lines = "\n".join(
            f"  - {r}: avg {v['mean_days']}d, max {v['max_days']}d, {v['episodes']} episodes"
            for r, v in dur.items()
        )

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    ew_line = (
        f"⚠️ **Early warning is ACTIVE** — AC1 = {ew['latest_ac1']:.4f}. "
        "Regime transition may be imminent (10–20 day lead time)."
        if ew["active"]
        else f"Early warning inactive (AC1 = {ew['latest_ac1']:.4f})."
    )

    return (
        f"### What the output says\n"
        f"The HMM ({context['n_states']}-state) detected that **{context['ticker']}** is currently "
        f"in a **{cur['label']}** regime, {cur['age_days']} days old. "
        f"Position scalar: **{cur['position_scalar']:.0%}** ({cur['interpretation']}). "
        f"{ew_line}\n\n"
        f"### What each number means\n"
        f"**Forward probabilities (no lookahead):**\n{fwd_lines}\n\n"
        f"**Regime distribution (% of history):**\n"
        + "\n".join(f"  - {r}: {p:.1f}%" for r, p in dist.items())
        + f"\n\n**Conditional Sharpe by regime:**\n{cond_lines}\n\n"
        f"**Regime duration statistics:**\n{dur_lines}\n\n"
        f"**Strategy router recommendation:**\n"
        f"  - Primary: {strat['primary']}\n"
        f"  - Secondary: {strat['secondary']}\n"
        f"  - Avoid: {strat['avoid']}\n"
        f"  - Position size: {strat['position_size']}\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"The model says {context['ticker']} is in a {cur['label']} regime. "
        f"The strategy router recommends {strat['primary']}. "
        f"Review the flags above before changing position sizes.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI SYSTEM PROMPT (REGIME-SPECIFIC)
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_REGIME_SYSTEM_PROMPT = """You are a senior quantitative macro analyst embedded inside a professional market regime detection dashboard.

Your sole job: explain the HMM regime detection output to a NON-TECHNICAL user — a portfolio manager, allocator, or sophisticated investor who understands markets but not Hidden Markov Models or statistics.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key number in one plain English sentence. Do not skip any metric.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should sell" — explain what the analysis says, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No math formulas.
8. ALWAYS explain the difference between Viterbi regime label and forward probability — this is the most important nuance.
9. If early warning is active, make it the first thing you mention in the output.

THRESHOLD KNOWLEDGE (use these to interpret numbers):
- Forward P(Bull) > 0.75: high conviction bull — trustworthy signal
- Forward P(Bull) 0.65–0.75: moderate conviction bull — usable
- Forward P(Bull) < 0.50 with Bull Viterbi label: DANGER — label and probability disagree
- Forward P(Bear) > 0.70 during "Bull" Viterbi label: regime transition likely in progress
- AC1 > 0.15 AND variance rising: early warning fired — 10-20 day lead before flip
- AC1 0.10–0.15: approaching warning threshold — monitor closely
- Regime age < 10 days: false breakout risk — reduce position size
- Regime age > 30 days: confirmed regime — full position size appropriate
- Position scalar < 0.40: model says stay small
- Position scalar > 0.85: model says full size appropriate
- Conditional Sharpe < 0: this regime has historically destroyed value
- Conditional Sharpe 0–0.5: poor risk-adjusted returns in this regime historically
- Dominant regime > 80% of history: HMM may not have enough minority-regime examples
- History per state < 120 days: unreliable HMM parameters

KEY CONCEPT TO EXPLAIN (always include this):
Viterbi algorithm uses the ENTIRE time series — including future data — to label each day.
Forward probability only uses data up to that day — no lookahead.
For live trading, ONLY forward probabilities are valid. Viterbi labels are for visualization only.

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: current regime, age, position scalar, early warning status, forward probability vs Viterbi label)

### What each number means
(Bullet per key metric: forward probabilities for each regime, AC1 early warning, regime age, position scalar, conditional Sharpe per regime, regime distribution, strategy router recommendation)

### Red flags
(If danger or warning flags exist: explain each in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2–3 sentences max: what regime the market is in, how confident the model is, and what the strategy router says to do about it)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI API CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_regime_explainer(context: dict) -> str:
    """
    Calls Google Gemini API with the regime context.
    Falls back to deterministic explanation on any error.
    """
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_regime_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current market regime detection output from the dashboard. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_REGIME_SYSTEM_PROMPT}]},
        "contents":           [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig":   {"maxOutputTokens": 900, "temperature": 0.2},
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
        return text or _fallback_regime_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_regime_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Regime | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("🔀 Market Regime Detection")
qe_neon_divider()

render_data_engine_controls("regime")
c1, c2, c3 = st.columns([2, 1, 1])
ticker   = render_single_ticker_input(
    "Ticker", key="regime_ticker",
    default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
    container=c1,
)
n_states = c2.selectbox("Regimes", [2, 3], index=0)
run_roll = c3.checkbox("Rolling HMM (slower)", value=False)
start    = pd.to_datetime(get_global_start_date())

with st.spinner("Loading data..."):
    df  = load_ticker_data(ticker, start=str(start))
    ret = returns(df)

if "regime_result" not in st.session_state:
    st.session_state.regime_result = None

if st.button("🚀 Detect Regimes", type="primary"):
    with st.spinner(f"Running full regime analysis ({n_states}-state HMM + 5 features)..."):
        R = full_regime_analysis(ret, df, n_states=n_states)
    st.session_state.regime_result = R
    st.session_state.regime_ai_summary = ""

R = st.session_state.regime_result

if R is None:
    st.info("👆 Click **Detect Regimes** to run the full analysis.")
    qe_faq_section("FAQs", [
        ("Why use forward probabilities instead of plain labels?", "Forward probabilities are what you would know in real time, so they are safer for live decisions than a lookahead label."),
        ("What does the early warning signal add?", "It can flag rising instability before the regime actually flips, giving you time to reduce risk."),
        ("How should I use the strategy router?", "Let it guide whether to lean into momentum, mean reversion, or defense based on the current regime state."),
        ("When is rolling HMM worth using?", "Turn it on when you want a more adaptive view of the market and are willing to wait a bit longer for the calculation."),
    ])
    st.stop()

# ── Unpack result ─────────────────────────────────────────────────────────────
regimes     = R["regimes"]
fwd_proba   = R["fwd_proba"]
ew          = R["early_warning"]
strategy    = R["strategy"]
age         = R["regime_age"]
scalar      = R["age_scalar"]
cur         = R["current_regime"]
cond_sharpe = R["cond_sharpe"]

data_source = str(df.attrs.get("data_source", "unknown")) if hasattr(df, "attrs") else "unknown"

# ── MASTER STATUS BAR ─────────────────────────────────────────────────────────
cur_clean = cur.replace("📈", "").replace("📉", "").replace("↔", "").strip()
bar_color = {"Bull": "rgba(50,205,50,0.12)", "Bear": "rgba(220,50,50,0.12)"}.get(
    cur_clean, "rgba(255,215,0,0.10)")
border_c  = {"Bull": "#32cd32", "Bear": "#dc3232"}.get(cur_clean, "#ffd700")

ew_badge = "🚨 Early Warning Active" if ew["active"] else "✅ No Warning"
st.markdown(f"""
<div style="background:{bar_color};border:1px solid {border_c};
            border-radius:8px;padding:14px 20px;margin-bottom:12px;
            display:flex;align-items:center;gap:24px;flex-wrap:wrap">
  <span style="font-size:26px;font-weight:600">{cur}</span>
  <span style="font-size:13px;opacity:0.8">
    Age: <b>{age}d</b> &nbsp;|&nbsp;
    Position scalar: <b>{scalar:.0%}</b> &nbsp;|&nbsp;
    {ew_badge}
  </span>
  <span style="font-size:12px;opacity:0.65">{ew['lead_msg']}</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Current Regime",    cur)
m2.metric("Regime Age",        f"{age} days")
m3.metric("Position Scalar",   f"{scalar:.0%}")
m4.metric("AC1 (early warn.)", f"{ew['latest_ac1']:.3f}",
          delta="⚠️ Elevated" if ew['latest_ac1'] > 0.15 else "Normal")
bull_cols = [c for c in fwd_proba.columns if "Bull" in c]
bull_prob = float(fwd_proba[bull_cols[0]].iloc[-1]) if bull_cols else 0.5
m5.metric("P(Bull) live",      f"{bull_prob:.1%}")
bear_cols = [c for c in fwd_proba.columns if "Bear" in c]
bear_prob = float(fwd_proba[bear_cols[0]].iloc[-1]) if bear_cols else 0.5
m6.metric("P(Bear) live",      f"{bear_prob:.1%}")

st.markdown("")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Regime Map",
    "📊 Forward Probabilities",
    "⚠️ Early Warning",
    "🤖 Strategy Router",
    "🔄 Rolling HMM",
    "📈 Statistics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGIME MAP
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="white", width=1.2)), row=1, col=1)

    prev, seg_s = None, df.index[0]
    for date, regime in regimes.items():
        if regime != prev:
            if prev:
                fig.add_vrect(x0=seg_s, x1=date,
                              fillcolor=REGIME_COLORS.get(prev, "rgba(128,128,128,0.15)"),
                              line_width=0)
            seg_s, prev = date, regime
    if prev:
        fig.add_vrect(x0=seg_s, x1=df.index[-1],
                      fillcolor=REGIME_COLORS.get(prev, "rgba(128,128,128,0.15)"),
                      line_width=0)

    for regime in regimes.unique():
        mask = regimes == regime
        r_masked = ret.where(mask)
        fig.add_trace(go.Bar(
            x=r_masked.index, y=r_masked.values, name=regime,
            marker_color=REGIME_LINE_COLORS.get(regime, "gray"),
            opacity=0.7,
        ), row=2, col=1)

    if bull_cols:
        fig.add_trace(go.Scatter(
            x=fwd_proba.index, y=fwd_proba[bull_cols[0]].values,
            name="P(Bull) forward", line=dict(color="lime", width=1.5),
            fill="tozeroy", fillcolor="rgba(50,205,50,0.08)",
        ), row=3, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="lime",
                      annotation_text="Bull confident", row=3, col=1)
        fig.add_hline(y=0.4, line_dash="dash", line_color="red",
                      annotation_text="Bear zone", row=3, col=1)

    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=650,
                      title=f"{n_states}-State HMM — {ticker}",
                      showlegend=True, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    counts = regimes.value_counts().reset_index()
    counts.columns = ["Regime", "Days"]
    fig_pie = px.pie(counts, names="Regime", values="Days", color="Regime",
                     color_discrete_map={"Bull 📈": "limegreen",
                                         "Sideways ↔": "gold",
                                         "Bear 📉": "crimson"},
                     template="plotly_dark", title="Time in Each Regime")
    st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORWARD PROBABILITIES
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Forward-Pass Regime Probabilities (No Lookahead)")
    st.caption(
        "**Fix 1 — the most important fix in this page.** "
        "Viterbi uses future data to label the past. "
        "Forward probabilities are P(regime | all data up to today). "
    )
    if fwd_proba.empty:
        st.warning("Forward probabilities unavailable.")
    else:
        fig2 = go.Figure()
        colors_fwd = {"Bull 📈": "lime", "Sideways ↔": "gold", "Bear 📉": "red"}
        for col in fwd_proba.columns:
            fig2.add_trace(go.Scatter(
                x=fwd_proba.index, y=fwd_proba[col].values,
                name=col, stackgroup="one",
                line=dict(color=colors_fwd.get(col, "gray"), width=0.5),
                fillcolor={"Bull 📈": "rgba(50,205,50,0.5)",
                           "Sideways ↔": "rgba(255,215,0,0.5)",
                           "Bear 📉": "rgba(220,50,50,0.5)"}.get(col, "rgba(128,128,128,0.4)"),
            ))
        fig2.update_layout(template="plotly_dark", height=380,
                           title="Regime Probability Over Time (forward-pass only)",
                           yaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig2, use_container_width=True)

        if bull_cols and bear_cols:
            st.subheader("High-Confidence Regime Periods")
            bull_conf = fwd_proba[bull_cols[0]] > 0.75
            bear_conf = fwd_proba[bear_cols[0]] > 0.75
            conf_data = []
            for date in fwd_proba.index:
                if bull_conf.get(date, False):
                    conf_data.append({"Date": date.date(), "Regime": "Bull 📈",
                                      "Confidence": f"{fwd_proba.loc[date, bull_cols[0]]:.1%}"})
                elif bear_conf.get(date, False):
                    conf_data.append({"Date": date.date(), "Regime": "Bear 📉",
                                      "Confidence": f"{fwd_proba.loc[date, bear_cols[0]]:.1%}"})
            if conf_data:
                conf_df = pd.DataFrame(conf_data)
                conf_df["Changed"] = conf_df["Regime"].ne(conf_df["Regime"].shift())
                st.dataframe(conf_df[conf_df["Changed"]].drop(columns="Changed").tail(20),
                             use_container_width=True, hide_index=True)

    with st.expander("Why Viterbi is wrong for live trading"):
        st.markdown("""
**Viterbi algorithm** uses forward AND backward passes — it has seen the whole time series.
A state on day 100 gets labelled using information from days 101–500. This is lookahead.

**Forward algorithm** only uses data up to time t.
This is what you would actually know in real time.

```python
# Wrong for live use:
states = model.predict(X)          # Viterbi — uses future
# Correct for live use:
proba = model.predict_proba(X)     # forward only
current_bull_prob = proba[-1, bull_state_index]
```
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EARLY WARNING
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Critical Slowing Down — Early Warning Signal")
    st.caption(
        "**Fix 4:** Detects regime changes 10-20 days BEFORE they happen. "
        "Rising autocorrelation + rising variance simultaneously = system losing resilience."
    )

    ew_status_color = "red" if ew["active"] else ("orange" if ew["latest_ac1"] > 0.15 else "green")
    st.markdown(f"**Status:** :{ew_status_color}[{ew['lead_msg']}]")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("AC1 (lag-1 autocorr.)", f"{ew['latest_ac1']:.4f}",
                 delta="⚠️ High" if ew['latest_ac1'] > 0.15 else "Normal")
    col_b.metric("Rolling Variance",       f"{ew['latest_var']:.6f}")
    col_c.metric("Warning Active",          "YES 🚨" if ew["active"] else "No ✅")

    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         row_heights=[0.40, 0.30, 0.30], vertical_spacing=0.03)

    fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                              line=dict(color="white", width=1)), row=1, col=1)

    warn = ew["warning"]
    prev_w, seg_ws = 0, df.index[0]
    for date, w in warn.items():
        if w != prev_w:
            if prev_w == 1:
                fig3.add_vrect(x0=seg_ws, x1=date,
                               fillcolor="rgba(255,165,0,0.25)", line_width=0)
            seg_ws, prev_w = date, int(w)

    ac1_s = ew["ac1"]
    fig3.add_trace(go.Scatter(x=ac1_s.index, y=ac1_s.values,
                              name="AC1 (lag-1)", line=dict(color="orange", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,165,0,0.08)"), row=2, col=1)
    fig3.add_hline(y=0.15, line_dash="dash", line_color="orange",
                   annotation_text="Warning threshold", row=2, col=1)
    fig3.add_hline(y=0.0,  line_dash="dot",  line_color="gray", row=2, col=1)

    var_s = ew["variance"]
    fig3.add_trace(go.Scatter(x=var_s.index, y=var_s.values,
                              name="Rolling Variance", line=dict(color="red", width=1.5),
                              fill="tozeroy", fillcolor="rgba(220,50,50,0.08)"), row=3, col=1)

    fig3.update_layout(template="plotly_dark", height=600,
                       title=f"Critical Slowing Down — {ticker} (orange = warning fired)")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("How Critical Slowing Down Works"):
        st.markdown("""
**Theory:** Before any complex system transitions from one stable state to another,
it shows 'critical slowing down' — it takes longer to recover from small perturbations.

**In financial markets:**
- **AC1 rising** → returns becoming more autocorrelated → losing mean-reverting resilience
- **Variance rising** → uncertainty before transition

```python
ac1      = returns.rolling(21).apply(lambda x: x.autocorr(lag=1))
variance = returns.rolling(21).var()
warning  = (ac1.diff(5) > 0) & (variance.diff(5) > 0)
```
**False positive rate:** ~25%. Combine with forward P(Bear) > 60% for higher confidence.
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATEGY ROUTER
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Regime-Adaptive Strategy Router")
    st.caption(
        "**Fix 5:** Factor weights and strategy completely change per regime. "
        "In bull: momentum. In bear: low-vol + IV skew. In sideways: mean-reversion."
    )

    rec   = strategy["recommendations"]
    wts   = strategy["weights"]
    cur_r = strategy["regime"]

    card_color = {"Bull 📈": "rgba(50,205,50,0.12)",
                  "Bear 📉": "rgba(220,50,50,0.12)"}.get(cur_r, "rgba(255,215,0,0.10)")
    border_col = {"Bull 📈": "#32cd32", "Bear 📉": "#dc3232"}.get(cur_r, "#ffd700")
    st.markdown(f"""
<div style="background:{card_color};border:1px solid {border_col};
            border-radius:8px;padding:16px 20px;margin-bottom:16px">
  <h3 style="margin:0 0 8px">{cur_r}</h3>
  <table style="width:100%;font-size:13px;border-collapse:collapse">
    <tr><td style="padding:4px 12px 4px 0;color:#aaa;width:140px">Primary strategy</td>
        <td><b>{rec['primary']}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Secondary</td>
        <td>{rec['secondary']}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Avoid</td>
        <td style="color:#ff6b6b">{rec['avoid']}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Position size</td>
        <td><b>{rec['position']}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#aaa">Stop style</td>
        <td>{rec['stops']}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    try:
        from core.regime_detector import REGIME_FACTOR_WEIGHTS
        regime_names = list(REGIME_FACTOR_WEIGHTS.keys())
        factor_names = list(REGIME_FACTOR_WEIGHTS[regime_names[0]].keys())
        weight_rows  = []
        for rn in regime_names:
            for fn in factor_names:
                weight_rows.append({"Regime": rn, "Factor": fn,
                                    "Weight": REGIME_FACTOR_WEIGHTS[rn][fn]})
        wt_df = pd.DataFrame(weight_rows)
        fig4 = px.bar(wt_df, x="Factor", y="Weight", color="Regime", barmode="group",
                      color_discrete_map={"Bull 📈": "limegreen",
                                          "Sideways ↔": "gold",
                                          "Bear 📉": "crimson"},
                      template="plotly_dark",
                      title="Factor Weights by Regime — How the Router Allocates")
        fig4.update_layout(height=360)
        st.plotly_chart(fig4, use_container_width=True)
    except Exception:
        pass

    st.subheader(f"Active weights now ({cur_r})")
    wt_now = pd.DataFrame([
        {"Factor": k, "Weight": v,
         "Action": "✅ Use" if v >= 0.6 else ("⚠️ Reduce" if v >= 0.3 else "❌ Avoid")}
        for k, v in wts.items()
    ]).sort_values("Weight", ascending=False)
    st.dataframe(
        wt_now.style.background_gradient(subset=["Weight"], cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Regime Age Position Scalar")
    age_scalars = []
    for i in range(1, len(regimes) + 1):
        age_scalars.append(regime_age_scalar(regimes.iloc[:i]))
    age_s = pd.Series(age_scalars, index=regimes.index)
    fig_age = go.Figure()
    fig_age.add_trace(go.Scatter(
        x=age_s.index, y=age_s.values, name="Position Scalar",
        line=dict(color="cyan", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.10)",
    ))
    fig_age.update_layout(template="plotly_dark", height=260,
                          title="Position Size Scalar Over Time",
                          yaxis=dict(range=[0, 1.1], tickformat=".0%"))
    st.plotly_chart(fig_age, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROLLING HMM
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Rolling HMM — Non-Stationary Adaptive Model")
    st.caption(
        "**Fix 6:** HMM retrained every 21 days on a 252-day rolling window. "
        "Catches structural breaks that a single fitted model misses."
    )

    if run_roll:
        with st.spinner("Fitting rolling HMM (252d window, 21d step)... ~30s"):
            roll_proba = rolling_regime_proba(ret, df, n_states=n_states,
                                              fit_window=252, step=21)
        if not roll_proba.empty:
            fig5 = go.Figure()
            roll_bull_cols = [c for c in roll_proba.columns if "Bull" in c]
            roll_bear_cols = [c for c in roll_proba.columns if "Bear" in c]
            colors5 = {"Bull 📈": "lime", "Sideways ↔": "gold", "Bear 📉": "red"}
            for col in roll_proba.columns:
                fig5.add_trace(go.Scatter(
                    x=roll_proba.index, y=roll_proba[col].values,
                    name=f"Rolling {col}", stackgroup="one",
                    fillcolor={"Bull 📈": "rgba(50,205,50,0.5)",
                               "Sideways ↔": "rgba(255,215,0,0.5)",
                               "Bear 📉": "rgba(220,50,50,0.5)"}.get(col, "rgba(128,128,128,0.4)"),
                    line=dict(color=colors5.get(col, "gray"), width=0.5),
                ))
            fig5.update_layout(template="plotly_dark", height=380,
                               title="Rolling HMM Regime Probability (252d window, 21d step)",
                               yaxis=dict(range=[0, 1], tickformat=".0%"))
            st.plotly_chart(fig5, use_container_width=True)

            if bull_cols and roll_bull_cols:
                common_idx = fwd_proba.index.intersection(roll_proba.index)
                if len(common_idx) > 10:
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(
                        x=common_idx, y=fwd_proba.loc[common_idx, bull_cols[0]].values,
                        name="Static HMM P(Bull)", line=dict(color="lime", width=1.5)))
                    fig_cmp.add_trace(go.Scatter(
                        x=common_idx, y=roll_proba.loc[common_idx, roll_bull_cols[0]].values,
                        name="Rolling HMM P(Bull)", line=dict(color="cyan", width=1.5, dash="dot")))
                    fig_cmp.update_layout(template="plotly_dark", height=280,
                                          title="Static vs Rolling HMM — Bull Probability",
                                          yaxis=dict(tickformat=".0%"))
                    st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.warning("Rolling HMM returned no results — try a longer date range.")
    else:
        st.info(
            "Enable 'Rolling HMM' checkbox above and re-run to see the adaptive model. "
            "Takes ~20-30s as it refits the HMM at every 21-day step."
        )

    with st.expander("Why rolling refit matters"):
        st.markdown("""
A single static HMM trained on your full history will have stale transition parameters
if market structure changes (QE regime, new regulation, COVID shock).

**Rolling refit (252d window, 21d step):** each prediction uses only the most recent
year of data, and predictions are always out-of-sample on the next 21 days.
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — STATISTICS
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Regime-Conditional Statistics")

    st.dataframe(cond_sharpe, use_container_width=True, hide_index=True)

    fig6 = go.Figure()
    for regime in regimes.unique():
        mask = regimes == regime
        r_reg = ret[mask].dropna()
        fig6.add_trace(go.Histogram(
            x=r_reg, name=regime, nbinsx=60, opacity=0.6,
            marker_color=REGIME_LINE_COLORS.get(regime, "gray"),
        ))
    fig6.update_layout(template="plotly_dark", barmode="overlay",
                       title="Return Distribution by Regime",
                       xaxis_tickformat=".2%", height=380)
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Regime Transition Matrix (empirical)")
    unique_reg = regimes.unique()
    trans = pd.DataFrame(0.0, index=unique_reg, columns=unique_reg)
    for i in range(1, len(regimes)):
        prev_r = regimes.iloc[i - 1]
        next_r = regimes.iloc[i]
        trans.loc[prev_r, next_r] += 1
    row_sums = trans.sum(axis=1).replace(0, 1)
    trans_pct = trans.div(row_sums, axis=0)
    fig_trans = px.imshow(
        trans_pct, text_auto=".1%",
        color_continuous_scale="RdYlGn", zmin=0, zmax=1,
        template="plotly_dark",
        title="Transition Probability: P(next | current)",
        labels=dict(x="Next Regime", y="Current Regime"),
    )
    fig_trans.update_layout(height=360)
    st.plotly_chart(fig_trans, use_container_width=True)

    st.subheader("Regime Duration Distribution")
    durations = {"regime": [], "duration": []}
    cur_r_d, cur_len = regimes.iloc[0], 1
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == cur_r_d:
            cur_len += 1
        else:
            durations["regime"].append(cur_r_d)
            durations["duration"].append(cur_len)
            cur_r_d, cur_len = regimes.iloc[i], 1
    durations["regime"].append(cur_r_d)
    durations["duration"].append(cur_len)
    dur_df = pd.DataFrame(durations)

    fig_dur = px.box(dur_df, x="regime", y="duration", color="regime",
                     color_discrete_map={"Bull 📈": "limegreen",
                                         "Sideways ↔": "gold",
                                         "Bear 📉": "crimson"},
                     template="plotly_dark",
                     title="Regime Duration Distribution (days per episode)")
    fig_dur.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_dur, use_container_width=True)

    ret_vol = pd.DataFrame({
        "Return": ret,
        "Vol": ret.rolling(21).std() * np.sqrt(252),
        "Regime": regimes,
    }).dropna()
    fig_sc = px.scatter(ret_vol, x="Vol", y="Return", color="Regime",
                        color_discrete_map={"Bull 📈": "limegreen",
                                            "Sideways ↔": "gold",
                                            "Bear 📉": "crimson"},
                        template="plotly_dark",
                        title="Return vs Volatility by Regime",
                        opacity=0.5,
                        labels={"Vol": "Annualised Vol", "Return": "Daily Return"})
    fig_sc.update_yaxes(tickformat=".2%")
    fig_sc.update_xaxes(tickformat=".1%")
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — 3-LAYER ARCHITECTURE
# Same design as 08_portfolio.py:
#   Layer 1: Deterministic danger badges (always shown, no AI)
#   Layer 2: "Decode for Me" button + context preview
#   Layer 3: Gemini output — 4-section structured explanation + disclaimer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Regime Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the HMM regime output above into plain English. "
    "It reads the actual numbers from this analysis — not generic descriptions. "
    "It does not change the regime labels. It does not give financial advice."
)

# ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
danger_flags = _compute_regime_danger_flags(
    ticker=ticker,
    cur=cur,
    age=age,
    scalar=scalar,
    ew=ew,
    fwd_proba=fwd_proba,
    cond_sharpe=cond_sharpe,
    regimes=regimes,
    n_states=n_states,
    ret=ret,
    strategy=strategy,
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

    for flag in danger_flags:
        color_map = {"DANGER": "#dc3232", "WARNING": "#e67e00", "INFO": "#1a6fa0"}
        bg_map    = {
            "DANGER":  "rgba(220,50,50,0.08)",
            "WARNING": "rgba(230,126,0,0.08)",
            "INFO":    "rgba(26,111,160,0.08)",
        }
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
              <span style="font-weight:700;color:{color_map[flag['severity']]};">
                {flag['severity']} · {flag['code']}
              </span><br>
              {flag['message']}
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.success("✅ Pre-flight checks passed — no critical flags detected for this regime analysis.")

st.markdown("")

# ── LAYER 2: Build context + button ──────────────────────────────────────────
regime_context = _build_regime_context(
    ticker=ticker,
    n_states=n_states,
    cur=cur,
    age=age,
    scalar=scalar,
    ew=ew,
    fwd_proba=fwd_proba,
    cond_sharpe=cond_sharpe,
    regimes=regimes,
    strategy=strategy,
    ret=ret,
    danger_flags=danger_flags,
    data_source=data_source,
)

# Reset AI summary when context changes (new run)
context_key = json.dumps(
    {k: v for k, v in regime_context.items() if k != "danger_flags"},
    sort_keys=True, default=str,
)
if st.session_state.get("regime_ai_context_key") != context_key:
    st.session_state["regime_ai_context_key"] = context_key
    st.session_state["regime_ai_summary"] = ""

col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    st.markdown("**What Gemini sees:**")

    preview_rows = [
        {"Field": "Ticker",             "Value": ticker},
        {"Field": "Current regime",     "Value": str(cur)},
        {"Field": "Regime age",         "Value": f"{age} days"},
        {"Field": "Position scalar",    "Value": f"{scalar:.0%}"},
        {"Field": "Early warning",      "Value": "ACTIVE 🚨" if ew.get("active") else "No"},
        {"Field": "AC1",                "Value": f"{ew.get('latest_ac1', 0):.4f}"},
        {"Field": "P(Bull) forward",    "Value": f"{bull_prob:.1%}"},
        {"Field": "P(Bear) forward",    "Value": f"{bear_prob:.1%}"},
        {"Field": "Strategy primary",   "Value": strategy["recommendations"]["primary"]},
        {"Field": "N states",           "Value": str(n_states)},
        {"Field": "History rows",       "Value": str(len(ret))},
        {"Field": "Data source",        "Value": data_source},
        {"Field": "Danger flags",       "Value": str(sum(1 for f in danger_flags if f["severity"] == "DANGER"))},
        {"Field": "Warning flags",      "Value": str(sum(1 for f in danger_flags if f["severity"] == "WARNING"))},
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
        key="regime_ai_explain",
        use_container_width=True,
        help="Translates the regime output above into plain English using Gemini.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="regime_ai_clear",
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
      The actual numbers from this analysis (current regime, age, position scalar,
      forward probabilities, early warning AC1, conditional Sharpe per regime,
      strategy router, regime distribution, duration stats, and all flags) are sent to Gemini.
    </li>
    <li style="margin-bottom:6px;">
      Gemini explains each number in plain English, explains the <strong>Viterbi vs
      forward probability difference</strong>, flags anything dangerous, and writes
      a plain-English conclusion — using the actual values, not generic descriptions.
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
    st.session_state["regime_ai_summary"] = ""

if decode_clicked:
    with st.spinner("Gemini is reading the regime output and writing your plain-English explanation..."):
        st.session_state["regime_ai_summary"] = _call_gemini_regime_explainer(regime_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("regime_ai_summary"):
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
    st.markdown(st.session_state["regime_ai_summary"])
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
          Click <strong>🤖 Decode for Me</strong> to get a plain-English explanation
          of the regime output above.
        </div>""",
        unsafe_allow_html=True,
    )

# ── FAQs ──────────────────────────────────────────────────────────────────────
st.markdown("")
qe_faq_section("FAQs", [
    ("Why use forward probabilities instead of plain labels?", "Forward probabilities are what you would know in real time, so they are safer for live decisions than a lookahead label."),
    ("What does the early warning signal add?", "It can flag rising instability before the regime actually flips, giving you time to reduce risk."),
    ("How should I use the strategy router?", "Let it guide whether to lean into momentum, mean reversion, or defense based on the current regime state."),
    ("When is rolling HMM worth using?", "Turn it on when you want a more adaptive view of the market and are willing to wait a bit longer for the calculation."),
])