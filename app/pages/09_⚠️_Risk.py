import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st
from urllib import error as urlerror
from urllib import request as urlrequest

import core.metrics as metrics
from core.data import get_ohlcv, get_multi_ohlcv, align_returns, returns
from app.data_engine import (
    render_data_engine_controls,
    render_single_ticker_input,
    load_ticker_data,
    get_global_start_date,
    get_data_engine_settings,
    parse_ticker_list,
)
from utils.charts import metric_card_row
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

# ── Metric aliases (original, unchanged) ─────────────────────────────────────
var_historical  = metrics.var_historical
cvar_historical = metrics.cvar_historical
var_parametric  = metrics.var_parametric
var_t_dist      = getattr(metrics, "var_t_dist",  metrics.var_parametric)
var_garch       = getattr(metrics, "var_garch",   metrics.var_historical)
summary_table   = metrics.summary_table
annualised_vol  = metrics.annualised_vol
max_drawdown    = metrics.max_drawdown


def _fallback_portfolio_var(
    returns_df: pd.DataFrame,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    returns_df = returns_df.dropna()
    if returns_df.empty:
        return 0.0
    n_assets = returns_df.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    weights = np.array(weights, dtype=float)
    portfolio_returns = pd.Series(returns_df.values @ weights, index=returns_df.index)
    if method == "garch":
        return var_garch(portfolio_returns, confidence)
    if method == "parametric":
        return var_parametric(portfolio_returns, confidence)
    return var_historical(portfolio_returns, confidence)


def _fallback_kupiec_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> dict:
    aligned_returns, aligned_var = returns.align(var_series, join="inner")
    mask = aligned_returns.notna() & aligned_var.notna()
    aligned_returns = aligned_returns[mask]
    aligned_var     = aligned_var[mask]
    if len(aligned_returns) < 30:
        return {
            "violations": 0,
            "expected_rate": 1 - confidence,
            "actual_rate": float("nan"),
            "p_value": float("nan"),
            "result": "Insufficient data",
        }
    violations   = int((aligned_returns < aligned_var).sum())
    total        = len(aligned_returns)
    expected_rate = 1 - confidence
    actual_rate  = violations / total
    return {
        "violations": violations,
        "expected_rate": expected_rate,
        "actual_rate": round(actual_rate, 4),
        "p_value": float("nan"),
        "result": "Fallback model check",
    }


portfolio_var = getattr(metrics, "portfolio_var", _fallback_portfolio_var)
kupiec_test   = getattr(metrics, "kupiec_test",   _fallback_kupiec_test)


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — LAYER 1: DETERMINISTIC DANGER FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_risk_danger_flags(
    ticker: str,
    ret: pd.Series,
    var_h: float,
    cvar_h: float,
    var_td: float,
    var_g: float,
    vol_a: float,
    conf: float,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the risk output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.

    Checks:
      1.  Demo data notice
      2.  GARCH VaR spike vs historical (volatility clustering stress)
      3.  CVaR/VaR ratio (tail severity — fat tail risk)
      4.  t-dist vs Gaussian gap (fat tail significance)
      5.  Annual volatility classification
      6.  Worst daily return vs VaR (black swan check)
      7.  Max drawdown severity
      8.  Kurtosis — fat tail warning
      9.  Skewness — left-tail asymmetry
      10. History length (reliability of VaR estimates)
      11. VaR as percentage of price (interpretability aid)
    """
    flags = []

    # ── 1. Demo data ──────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Risk analysis for {ticker} is running on SYNTHETIC demo data. "
                "All VaR, CVaR, volatility and drawdown figures are illustrative only. "
                "Do not use these numbers for real position sizing or capital allocation."
            ),
        })

    if ret is None or len(ret) < 10:
        return flags

    ret_clean = ret.dropna()

    # ── 2. GARCH VaR spike vs historical ─────────────────────────────────────
    if var_h != 0 and abs(var_g) > 0:
        garch_ratio = abs(var_g) / abs(var_h)
        if garch_ratio > 1.5:
            flags.append({
                "severity": "DANGER",
                "code": "GARCH_VAR_SPIKE",
                "message": (
                    f"GARCH VaR ({var_g:.2%}) is {garch_ratio:.1f}× the historical VaR "
                    f"({var_h:.2%}). "
                    "GARCH is reacting to a spike in recent conditional volatility — the market "
                    "is in a stressed regime right now. "
                    "Historical VaR is underestimating current risk by a large margin. "
                    "Use GARCH VaR for position sizing until volatility normalises."
                ),
            })
        elif garch_ratio > 1.2:
            flags.append({
                "severity": "WARNING",
                "code": "GARCH_VAR_ELEVATED",
                "message": (
                    f"GARCH VaR ({var_g:.2%}) is {garch_ratio:.1f}× historical VaR ({var_h:.2%}). "
                    "Conditional volatility is elevated above the long-run average. "
                    "Consider using GARCH VaR rather than historical for current sizing."
                ),
            })

    # ── 3. CVaR/VaR ratio — tail severity ────────────────────────────────────
    if var_h != 0:
        cvar_ratio = abs(cvar_h) / abs(var_h)
        if cvar_ratio > 1.5:
            flags.append({
                "severity": "DANGER",
                "code": "SEVERE_TAIL_RISK",
                "message": (
                    f"CVaR ({cvar_h:.2%}) is {cvar_ratio:.1f}× VaR ({var_h:.2%}). "
                    "When VaR is breached, the AVERAGE loss is {cvar_ratio:.1f}× the VaR threshold. "
                    "This indicates a heavy left tail — losses during bad days are far worse "
                    "than VaR implies. Never use VaR alone for risk limits on this asset."
                ),
            })
        elif cvar_ratio > 1.25:
            flags.append({
                "severity": "WARNING",
                "code": "ELEVATED_TAIL_RISK",
                "message": (
                    f"CVaR ({cvar_h:.2%}) is {cvar_ratio:.1f}× VaR ({var_h:.2%}). "
                    "Tail losses are moderately severe beyond the VaR threshold. "
                    "Consider CVaR as your primary risk limit rather than VaR."
                ),
            })

    # ── 4. t-dist vs Gaussian gap ─────────────────────────────────────────────
    var_param = var_parametric(ret_clean, conf)
    if var_param != 0:
        tdist_gap = abs(var_td - var_param) / abs(var_param)
        if tdist_gap > 0.20:
            flags.append({
                "severity": "WARNING",
                "code": "FAT_TAIL_SIGNIFICANT",
                "message": (
                    f"Student-t VaR ({var_td:.2%}) is {tdist_gap:.0%} larger than Gaussian VaR "
                    f"({var_param:.2%}). "
                    "The fat-tail correction is material — the normal distribution is significantly "
                    "underestimating tail risk for this asset. "
                    "Use t-distribution or historical VaR as your primary measure."
                ),
            })

    # ── 5. Annual volatility classification ───────────────────────────────────
    if vol_a > 0.60:
        flags.append({
            "severity": "DANGER",
            "code": "EXTREME_VOLATILITY",
            "message": (
                f"Annualised volatility is {vol_a:.1%} — extremely high. "
                "This is above 60%, which places this asset in the speculative/crypto-like "
                "volatility tier. A 1-standard-deviation annual move exceeds 60% of price. "
                "VaR estimates at this volatility level have very wide confidence intervals."
            ),
        })
    elif vol_a > 0.35:
        flags.append({
            "severity": "WARNING",
            "code": "HIGH_VOLATILITY",
            "message": (
                f"Annualised volatility is {vol_a:.1%} — high relative to broad equity benchmarks "
                "(typically 15–25%). This asset is more than 1.5× as volatile as the S&P 500. "
                "Position sizes should be scaled down proportionally."
            ),
        })

    # ── 6. Worst day vs VaR ───────────────────────────────────────────────────
    worst_day = float(ret_clean.min())
    if var_h != 0:
        worst_ratio = abs(worst_day) / abs(var_h)
        if worst_ratio > 3.0:
            flags.append({
                "severity": "DANGER",
                "code": "BLACK_SWAN_OBSERVED",
                "message": (
                    f"Worst single-day return in history is {worst_day:.2%} — "
                    f"{worst_ratio:.1f}× the {conf:.0%} VaR ({var_h:.2%}). "
                    "At least one extreme tail event (black swan) is present in this dataset. "
                    "VaR models trained on this history may be underweighting that tail. "
                    "Consider stress-testing against this scenario explicitly."
                ),
            })
        elif worst_ratio > 2.0:
            flags.append({
                "severity": "WARNING",
                "code": "LARGE_TAIL_EVENT_OBSERVED",
                "message": (
                    f"Worst day ({worst_day:.2%}) is {worst_ratio:.1f}× VaR ({var_h:.2%}). "
                    "This asset has experienced losses well beyond its VaR estimate historically. "
                    "CVaR is more appropriate than VaR for tail risk limits."
                ),
            })

    # ── 7. Max drawdown severity ──────────────────────────────────────────────
    mdd = max_drawdown(ret_clean)
    if mdd < -0.50:
        flags.append({
            "severity": "DANGER",
            "code": "SEVERE_DRAWDOWN",
            "message": (
                f"Maximum drawdown is {mdd:.1%} — a more-than-50% peak-to-trough loss "
                "is present in this history. "
                "Recovery from a 50% drawdown requires a 100% gain just to break even. "
                "Daily VaR figures do not capture this multi-month risk."
            ),
        })
    elif mdd < -0.30:
        flags.append({
            "severity": "WARNING",
            "code": "HIGH_DRAWDOWN",
            "message": (
                f"Maximum drawdown is {mdd:.1%}. "
                "A 30%+ peak-to-trough loss has occurred historically. "
                "This is above typical institutional drawdown limits (15–20%). "
                "Consider drawdown-based position limits, not just daily VaR limits."
            ),
        })

    # ── 8. Kurtosis — fat tails ───────────────────────────────────────────────
    kurt = float(ret_clean.kurt())
    if kurt > 5.0:
        flags.append({
            "severity": "WARNING",
            "code": "EXCESS_KURTOSIS",
            "message": (
                f"Excess kurtosis is {kurt:.2f} — significantly above 0 (normal distribution). "
                "High kurtosis means extreme returns occur more often than a normal distribution "
                "predicts. Gaussian VaR systematically underestimates risk for this asset. "
                "t-distribution VaR is more appropriate."
            ),
        })

    # ── 9. Skewness — left-tail asymmetry ────────────────────────────────────
    skew = float(ret_clean.skew())
    if skew < -1.0:
        flags.append({
            "severity": "WARNING",
            "code": "NEGATIVE_SKEW",
            "message": (
                f"Return skewness is {skew:.2f} — significantly negative. "
                "Negative skew means large losses occur more frequently than large gains. "
                "This is a left-tail asymmetry: the asset has a tendency to fall sharply "
                "and recover slowly. VaR will understate downside frequency."
            ),
        })

    # ── 10. History length ────────────────────────────────────────────────────
    n_obs = len(ret_clean)
    expected_violations = int(n_obs * (1 - conf))
    if n_obs < 252:
        flags.append({
            "severity": "DANGER",
            "code": "INSUFFICIENT_HISTORY",
            "message": (
                f"Only {n_obs} daily observations — less than 1 year. "
                f"At {conf:.0%} confidence, you expect {expected_violations} VaR violations "
                f"in {n_obs} days. With so few observations, VaR estimates have very high "
                "sampling error. Extend the date range to at least 2–3 years."
            ),
        })
    elif n_obs < 504:
        flags.append({
            "severity": "WARNING",
            "code": "SHORT_HISTORY",
            "message": (
                f"{n_obs} observations ({n_obs//21:.0f} months). "
                "Less than 2 years of data means VaR estimates may not capture a full "
                "market cycle. Historical VaR is particularly sensitive to this — "
                "it can only estimate risks that actually occurred in the sample."
            ),
        })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — LAYER 2: CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_risk_context(
    ticker: str,
    ret: pd.Series,
    var_h: float,
    cvar_h: float,
    var_td: float,
    var_g: float,
    vol_a: float,
    conf: float,
    rf_rate: float,
    danger_flags: list[dict],
    data_source: str = "unknown",
) -> dict:
    """Build the full structured context dict sent to Gemini."""

    ret_clean = ret.dropna()
    n_obs     = len(ret_clean)
    worst_day = float(ret_clean.min())
    best_day  = float(ret_clean.max())
    kurt      = float(ret_clean.kurt())
    skew      = float(ret_clean.skew())
    mdd       = max_drawdown(ret_clean)
    var_norm  = var_parametric(ret_clean, conf)

    # CVaR/VaR ratio
    cvar_ratio = round(abs(float(cvar_h)) / abs(float(var_h)), 4) if var_h != 0 else None

    # GARCH vs historical ratio
    garch_ratio = round(abs(float(var_g)) / abs(float(var_h)), 4) if var_h != 0 else None

    # Fat tail gap
    tdist_gap_pct = round(
        abs(float(var_td) - float(var_norm)) / abs(float(var_norm)) * 100, 2
    ) if var_norm != 0 else None

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticker": ticker,
        "data_source": data_source,
        "settings": {
            "confidence_level": conf,
            "risk_free_rate_pct": round(rf_rate * 100, 2),
            "observations": n_obs,
            "approx_years": round(n_obs / 252, 1),
        },

        # ── VaR estimates ─────────────────────────────────────────────────────
        "var_estimates": {
            "historical":       round(float(var_h),  6),
            "parametric_gaussian": round(float(var_norm), 6),
            "student_t":        round(float(var_td), 6),
            "garch":            round(float(var_g),  6),
            "cvar_historical":  round(float(cvar_h), 6),
        },

        # ── Derived risk ratios ───────────────────────────────────────────────
        "risk_ratios": {
            "cvar_over_var":        cvar_ratio,
            "garch_over_historical": garch_ratio,
            "tdist_gap_pct":        tdist_gap_pct,
        },

        # ── Return distribution stats ─────────────────────────────────────────
        "return_distribution": {
            "annualised_vol_pct":  round(float(vol_a) * 100, 2),
            "worst_single_day":    round(worst_day, 6),
            "best_single_day":     round(best_day,  6),
            "kurtosis":            round(kurt, 4),
            "skewness":            round(skew, 4),
            "max_drawdown":        round(float(mdd), 6),
            "tail_events_count":   int((ret_clean <= var_h).sum()),
        },

        # ── Danger flags ──────────────────────────────────────────────────────
        "danger_flags": danger_flags,
        "danger_flag_count":  len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in danger_flags if f["severity"] == "WARNING"]),

        # ── Reference thresholds ──────────────────────────────────────────────
        "reference_thresholds": {
            "cvar_var_ratio_danger":        1.5,
            "cvar_var_ratio_warning":       1.25,
            "garch_hist_ratio_danger":      1.5,
            "garch_hist_ratio_warning":     1.2,
            "vol_extreme_pct":              60.0,
            "vol_high_pct":                 35.0,
            "worst_day_var_ratio_danger":   3.0,
            "worst_day_var_ratio_warning":  2.0,
            "max_drawdown_danger":          -0.50,
            "max_drawdown_warning":         -0.30,
            "kurtosis_fat_tail":            5.0,
            "skew_left_tail":               -1.0,
            "min_obs_reliable":             504,
            "min_obs_danger":               252,
            "tdist_gap_material_pct":       20.0,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_risk_explanation(context: dict) -> str:
    var_e  = context["var_estimates"]
    dist   = context["return_distribution"]
    ratios = context["risk_ratios"]
    flags  = context.get("danger_flags", [])
    cfg_   = context["settings"]

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    return (
        f"### What the output says\n"
        f"Risk analysis for **{context['ticker']}** over "
        f"**{cfg_['approx_years']} years** ({cfg_['observations']} observations) "
        f"at **{cfg_['confidence_level']:.0%} confidence**. "
        f"Historical VaR = {var_e['historical']:.2%}, "
        f"CVaR = {var_e['cvar_historical']:.2%}, "
        f"GARCH VaR = {var_e['garch']:.2%}, "
        f"annualised vol = {dist['annualised_vol_pct']:.1f}%.\n\n"
        f"### What each number means\n"
        f"- **Historical VaR {cfg_['confidence_level']:.0%}**: {var_e['historical']:.2%} — "
        "on the worst 5% of days, you lose at least this much.\n"
        f"- **CVaR**: {var_e['cvar_historical']:.2%} — when VaR is breached, "
        f"the average loss is {var_e['cvar_historical']:.2%} "
        f"(CVaR/VaR ratio: {ratios['cvar_over_var']}).\n"
        f"- **GARCH VaR**: {var_e['garch']:.2%} — reflects today's conditional vol. "
        f"GARCH/Historical ratio: {ratios['garch_over_historical']}.\n"
        f"- **Annualised Vol**: {dist['annualised_vol_pct']:.1f}%\n"
        f"- **Max Drawdown**: {dist['max_drawdown']:.2%}\n"
        f"- **Kurtosis**: {dist['kurtosis']:.2f} | **Skewness**: {dist['skewness']:.2f}\n"
        f"- **Worst day**: {dist['worst_single_day']:.2%}\n"
        f"{flag_text}\n\n"
        f"### Plain English conclusion\n"
        f"On a typical bad day (worst 5%), {context['ticker']} loses around "
        f"{var_e['historical']:.2%}. In extreme tail events, losses average "
        f"{var_e['cvar_historical']:.2%}. Review flags above before sizing positions.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI SYSTEM PROMPT (RISK-SPECIFIC)
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_RISK_SYSTEM_PROMPT = """You are a senior risk manager embedded inside a professional quantitative risk analytics dashboard.

Your sole job: explain the VaR/CVaR risk output to a NON-TECHNICAL user — a portfolio manager, allocator, or investor who understands markets but not statistical risk models.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key number in one plain English sentence. Do not skip any metric.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should sell" — explain what the analysis says, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No formulas.
8. Always explain VaR vs CVaR vs GARCH — these three numbers tell a different story.

THRESHOLD KNOWLEDGE (use these to interpret numbers):
- Historical VaR: the minimum loss on the worst (1-conf)% of days in the sample
- CVaR: the average loss WHEN VaR is breached — always worse than VaR
- GARCH VaR: reacts to recent vol clustering — if GARCH >> Historical, current risk is elevated
- CVaR/VaR ratio > 1.5: DANGER — tail losses are severely worse than VaR implies
- GARCH/Historical ratio > 1.5: DANGER — market is in a stressed volatility regime right now
- t-dist gap > 20%: fat tails are material — Gaussian VaR is significantly underestimating risk
- Annualised vol > 60%: extreme volatility tier (crypto-like)
- Annualised vol 35–60%: high volatility — needs scaled-down positions
- Annualised vol < 20%: low volatility — typical for large-cap equities/bonds
- Max drawdown < -50%: DANGER — over half of capital has been lost from peak historically
- Max drawdown -30% to -50%: WARNING — above typical institutional limits
- Kurtosis > 5: fat tails — normal distribution severely underestimates tail risk
- Skewness < -1: negative skew — losses are asymmetrically large and frequent
- Observations < 252: DANGER — less than 1 year, VaR is unreliable
- Observations 252–504: WARNING — less than 2 years, needs caution

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: ticker, confidence level, all four VaR numbers, vol, data length)

### What each number means
(Bullet per key metric: historical VaR, CVaR, GARCH VaR, t-dist VaR, CVaR/VaR ratio, GARCH/historical ratio, annualised vol, max drawdown, kurtosis, skewness, worst day, observation count)

### Red flags
(If danger or warning flags exist: explain each in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2–3 sentences max: how risky is this asset right now, which VaR number to trust most given the flags, and what the tail risk picture looks like)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI API CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_risk_explainer(context: dict) -> str:
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_risk_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current risk analytics output from the dashboard. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _GEMINI_RISK_SYSTEM_PROMPT}]},
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
        return text or _fallback_risk_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_risk_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Risk | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("⚠️ Risk Analytics")
qe_neon_divider()

render_data_engine_controls("risk")
render_cols = st.columns([2, 1, 1, 1])
ticker = render_single_ticker_input(
    "Primary Ticker", key="risk_ticker",
    default=(cfg.DEFAULT_TICKERS[0] if cfg.DEFAULT_TICKERS else "GOOG"),
    container=render_cols[0],
)
conf    = render_cols[1].selectbox("Confidence Level", [0.95, 0.99], index=0)
rf_rate = render_cols[2].number_input(
    "Risk-Free Rate (%/yr)", value=cfg.RISK_FREE_RATE * 100,
    min_value=0.0, max_value=20.0, step=0.25, format="%.2f"
) / 100

start = pd.to_datetime(get_global_start_date())

if "risk_result" not in st.session_state:
    st.session_state.risk_result = None

run_clicked = st.button("Run Risk Analysis", type="primary")
if run_clicked:
    with st.spinner("Loading data and calculating risk metrics..."):
        df  = load_ticker_data(ticker, start=str(start))
        ret = returns(df)
        st.session_state.risk_result = {
            "df":    df,
            "ret":   ret,
            "var_h":  var_historical(ret, conf),
            "cvar_h": cvar_historical(ret, conf),
            "var_td": var_t_dist(ret, conf),
            "var_g":  var_garch(ret, conf),
            "vol_a":  annualised_vol(ret),
        }
        # clear stale AI summary on new run
        st.session_state["risk_ai_summary"]     = ""
        st.session_state["risk_ai_context_key"] = ""

risk_result = st.session_state.risk_result
if risk_result is None:
    st.info("Configure the inputs above, then press Run Risk Analysis.")
    st.stop()

df     = risk_result["df"]
ret    = risk_result["ret"]
var_h  = risk_result["var_h"]
cvar_h = risk_result["cvar_h"]
var_td = risk_result["var_td"]
var_g  = risk_result["var_g"]
vol_a  = risk_result["vol_a"]

data_source = str(df.attrs.get("data_source", "unknown")) if hasattr(df, "attrs") else "unknown"

st.markdown(metric_card_row({
    f"VaR {conf:.0%} (Hist)":   f"{var_h:.2%}",
    f"CVaR {conf:.0%} (Hist)":  f"{cvar_h:.2%}",
    f"VaR {conf:.0%} (t-dist)": f"{var_td:.2%}",
    f"VaR {conf:.0%} (GARCH)":  f"{var_g:.2%}",
    "Ann. Volatility":           f"{vol_a:.2%}",
}), unsafe_allow_html=True)

st.caption(
    f"📌 RF Rate in use: **{rf_rate:.2%}/yr**  |  "
    f"ℹ️ GARCH VaR reflects today's conditional vol — spikes during stressed markets."
)
st.markdown("")

# ── TABS (original logic unchanged) ──────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 VaR Comparison",
    "🌊 GARCH Rolling Risk",
    "🏦 Portfolio Risk",
    "💥 Stress Tests",
    "🔬 Kupiec Backtest",
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ret, nbinsx=80, name="Daily Returns",
                               marker_color="steelblue", opacity=0.75))
    for val, color, dash, label in [
        (var_h,  "red",    "dash", f"Hist VaR {conf:.0%}"),
        (cvar_h, "orange", "dot",  f"CVaR {conf:.0%}"),
        (var_td, "yellow", "dash", f"t-dist VaR {conf:.0%}"),
        (var_g,  "lime",   "dash", f"GARCH VaR {conf:.0%}"),
    ]:
        fig.add_vline(x=val, line_dash=dash, line_color=color,
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(template="plotly_dark",
                      title="Return Distribution — VaR Method Comparison",
                      xaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig, use_container_width=True)

    kurt = float(ret.kurt())
    method_df = pd.DataFrame([
        {"Method": "Historical",           "VaR": f"{var_h:.2%}",
         "Note": "From past returns directly. No distribution assumption."},
        {"Method": "Parametric (Gaussian)", "VaR": f"{var_parametric(ret, conf):.2%}",
         "Note": f"Assumes normal dist. Kurtosis={kurt:.1f} — {'⚠️ fat tails, prefer t-dist' if kurt > 1 else '✅ approx normal'}."},
        {"Method": "Student-t (fat-tail)",  "VaR": f"{var_td:.2%}",
         "Note": "Fits degrees-of-freedom from data. Better for equity fat tails."},
        {"Method": "GARCH(1,1)",            "VaR": f"{var_g:.2%}",
         "Note": "Accounts for vol clustering. Most responsive to current regime."},
    ])
    st.subheader("Method Comparison")
    st.dataframe(method_df, use_container_width=True, hide_index=True)

    tail = ret[ret <= var_h]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tail Events",  len(tail))
    c2.metric("Worst Day",    f"{ret.min():.2%}")
    c3.metric("Kurtosis",     f"{kurt:.2f}")
    c4.metric("Max Drawdown", f"{max_drawdown(ret):.2%}")


# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("GARCH Conditional Volatility vs Rolling Volatility")
    st.caption("GARCH reacts faster during stress. Rolling vol lags reality.")
    window   = st.slider("Rolling Window (days)", 21, 252, 63, key="garch_window")
    roll_var = ret.rolling(window).quantile(1 - conf)

    garch_var_series = pd.Series(dtype=float, index=ret.index)
    garch_vol_ann    = pd.Series(dtype=float, index=ret.index)
    garch_fit_ok     = False
    try:
        from arch import arch_model
        from scipy import stats as sp_stats
        scaled = ret * 100
        gm     = arch_model(scaled, vol="Garch", p=1, q=1, dist="t", rescale=False)
        gfit   = gm.fit(disp="off", show_warning=False)
        cond_vol = gfit.conditional_volatility / 100
        nu = float(gfit.params.get("nu", 8))
        z  = float(sp_stats.t.ppf(1 - conf, nu))
        garch_var_series = pd.Series(ret.mean() + z * cond_vol.values, index=ret.index)
        garch_vol_ann    = pd.Series(cond_vol.values * np.sqrt(252),    index=ret.index)
        garch_fit_ok = True
    except Exception:
        st.warning("GARCH fit failed — showing rolling VaR only.")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ret.index, y=ret.values, name="Daily Return",
                              line=dict(color="lightblue", width=0.8), opacity=0.4))
    fig2.add_trace(go.Scatter(x=roll_var.index, y=roll_var.values,
                              name=f"Rolling VaR {conf:.0%} ({window}d)",
                              line=dict(color="red", width=2)))
    if garch_fit_ok:
        fig2.add_trace(go.Scatter(x=garch_var_series.index, y=garch_var_series.values,
                                  name=f"GARCH VaR {conf:.0%}",
                                  line=dict(color="lime", width=2)))
    fig2.update_layout(template="plotly_dark", title=f"Rolling vs GARCH VaR — {ticker}",
                       yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig2, use_container_width=True)

    if garch_fit_ok:
        roll_std = ret.rolling(window).std() * np.sqrt(252)
        fig_vol  = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_std.index, y=roll_std.values,
                                     name=f"Rolling Ann. Vol ({window}d)",
                                     line=dict(color="orange", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=garch_vol_ann.index, y=garch_vol_ann.values,
                                     name="GARCH Ann. Conditional Vol",
                                     line=dict(color="lime", width=2)))
        fig_vol.update_layout(template="plotly_dark",
                               title="Annualised Vol: Rolling vs GARCH",
                               yaxis_tickformat=".1%", height=350)
        st.plotly_chart(fig_vol, use_container_width=True)


# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Portfolio-Level Risk")
    st.caption("Single-stock VaR misses correlations. Portfolio VaR accounts for how assets move together.")

    port_input   = st.text_input("Tickers (comma-separated)",
                                  value=", ".join(cfg.DEFAULT_TICKERS[:4]),
                                  key="port_tickers")
    port_tickers = parse_ticker_list(port_input)
    c_w1, c_w2  = st.columns(2)
    equal_weight = c_w1.checkbox("Equal weight", value=True)
    port_method  = c_w2.selectbox("VaR Method", ["historical", "parametric", "garch"])

    if len(port_tickers) < 2:
        st.warning("Enter at least 2 tickers.")
    else:
        with st.spinner("Loading portfolio data..."):
            multi_data = get_multi_ohlcv(port_tickers, start=str(start.date()))
            ret_df     = align_returns(multi_data).dropna()

        if ret_df.empty or ret_df.shape[1] < 2:
            st.error("Could not load data for these tickers.")
        else:
            valid_tickers = list(ret_df.columns)
            n = len(valid_tickers)
            if equal_weight:
                weights = np.ones(n) / n
            else:
                wcols = st.columns(n)
                raw_w = [wcols[i].number_input(
                             valid_tickers[i], 0.0, 1.0,
                             value=round(1 / n, 2), step=0.05,
                             key=f"w_{valid_tickers[i]}")
                         for i in range(n)]
                total   = sum(raw_w)
                weights = np.array(raw_w) / total if total > 0 else np.ones(n) / n

            port_ret_s = pd.Series(ret_df.values @ weights, index=ret_df.index)
            pvar   = portfolio_var(ret_df, weights, conf, port_method)
            pcvar  = float(port_ret_s[port_ret_s <= pvar].mean()) if (port_ret_s <= pvar).sum() > 0 else 0.0
            pvol   = float(port_ret_s.std() * np.sqrt(252))
            pmdd   = max_drawdown(port_ret_s)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Portfolio VaR {conf:.0%}",  f"{pvar:.2%}")
            m2.metric(f"Portfolio CVaR {conf:.0%}", f"{pcvar:.2%}")
            m3.metric("Portfolio Ann. Vol",          f"{pvol:.2%}")
            m4.metric("Portfolio Max DD",            f"{pmdd:.2%}")

            corr     = ret_df.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Return Correlation Matrix",
                                  template="plotly_dark")
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

            indiv_vars = {t: var_historical(ret_df[t], conf) for t in valid_tickers}
            var_cmp    = pd.DataFrame({
                "Asset": list(indiv_vars.keys()) + ["📦 Portfolio"],
                "VaR":   list(indiv_vars.values()) + [pvar],
            })
            fig_bar = px.bar(var_cmp, x="Asset", y="VaR", color="VaR",
                              color_continuous_scale="Reds_r", template="plotly_dark",
                              title=f"Individual vs Portfolio VaR ({conf:.0%})")
            fig_bar.update_yaxes(tickformat=".2%")
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

            cum_port = (1 + port_ret_s).cumprod()
            fig_cum  = go.Figure()
            for t in valid_tickers:
                fig_cum.add_trace(go.Scatter(
                    x=ret_df.index, y=(1 + ret_df[t]).cumprod().values,
                    name=t, line=dict(width=1), opacity=0.6))
            fig_cum.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port.values,
                name="Portfolio", line=dict(color="cyan", width=2.5)))
            fig_cum.update_layout(template="plotly_dark",
                                   title="Cumulative Returns — Portfolio vs Components",
                                   height=380)
            st.plotly_chart(fig_cum, use_container_width=True)


# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    SCENARIOS = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-01"),
        "COVID-19 Crash":        ("2020-02-01", "2020-04-01"),
        "2022 Rate Shock":       ("2022-01-01", "2022-10-01"),
        "2020 Tech Rally":       ("2020-04-01", "2021-01-01"),
        "2018 Q4 Selloff":       ("2018-10-01", "2018-12-31"),
    }
    rows = []
    for name, (s, e) in SCENARIOS.items():
        scenario_mask = (ret.index >= s) & (ret.index <= e)
        r = ret.loc[scenario_mask]
        if len(r) < 10:
            continue
        cum_r = (1 + r).cumprod()
        dd_s  = (cum_r - cum_r.cummax()) / cum_r.cummax()
        pnl   = (1 + r).prod() - 1
        rows.append({
            "Scenario":       name,
            "Period":         f"{s} → {e}",
            "Total Return":   f"{pnl:.2%}",
            "Max Drawdown":   f"{dd_s.min():.2%}",
            "Hist VaR 95%":   f"{var_historical(r, 0.95):.2%}",
            "t-dist VaR 95%": f"{var_t_dist(r, 0.95):.2%}",
            "Worst Day":      f"{r.min():.2%}",
            "Volatility":     f"{r.std() * np.sqrt(252):.2%}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Extend start date to cover stress test periods.")

    cum   = (1 + ret).cumprod()
    fig3  = go.Figure(go.Scatter(x=cum.index, y=cum.values,
                                  line=dict(color="cyan"), name="Cumulative Return"))
    for name, (s, e) in SCENARIOS.items():
        fig3.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.08,
                       annotation_text=name.split(" ")[0])
    fig3.update_layout(template="plotly_dark",
                       title=f"Cumulative Return with Stress Periods — {ticker}", height=420)
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Kupiec Proportion of Failures (POF) Test")
    st.caption(
        "How accurate is your VaR model? Counts actual losses > VaR vs expected. "
        "p > 0.05 = statistically valid. p < 0.05 = underestimating risk."
    )
    kup_window  = st.slider("Rolling VaR Window (days)", 21, 126, 63, key="kup_window")
    roll_var_bt = ret.rolling(kup_window).quantile(1 - conf).shift(1)

    kup = kupiec_test(ret, roll_var_bt, conf)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", len(ret.dropna()))
    k2.metric("Violations",   kup["violations"])
    k3.metric("Expected Rate", f"{kup['expected_rate']:.1%}")
    k4.metric("Actual Rate",
              f"{kup['actual_rate']:.1%}" if not np.isnan(kup["actual_rate"]) else "N/A")

    p_val = kup["p_value"]
    if not np.isnan(p_val):
        color = "green" if p_val > 0.05 else "red"
        st.markdown(f"**p-value:** `{p_val:.4f}`  |  **Result:** :{color}[{kup['result']}]")
    else:
        st.info(kup["result"])

    violations_mask = ret < roll_var_bt
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ret.index, y=ret.values, name="Daily Return",
                              line=dict(color="lightblue", width=0.8), opacity=0.5))
    fig4.add_trace(go.Scatter(x=roll_var_bt.index, y=roll_var_bt.values,
                              name=f"VaR {conf:.0%} (1d lag)",
                              line=dict(color="red", width=1.5)))
    viol_ret = ret[violations_mask]
    fig4.add_trace(go.Scatter(x=viol_ret.index, y=viol_ret.values,
                              mode="markers", name="VaR Violation",
                              marker=dict(color="orange", size=5, symbol="x")))
    fig4.update_layout(template="plotly_dark",
                       title=f"VaR Violations — {ticker} ({kup['violations']} breaches)",
                       yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "💡 Too few violations = overcautious (wastes capital). "
        "Too many = underestimating risk. "
        "Target: violations ≈ (1 - confidence) × total days."
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — 3-LAYER ARCHITECTURE
# Placed ABOVE FAQs so output appears directly after the analysis tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Risk Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the VaR/CVaR risk output above into plain English. "
    "It reads the actual numbers from this analysis — not generic descriptions. "
    "It does not change the risk metrics. It does not give financial advice."
)

# ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
danger_flags = _compute_risk_danger_flags(
    ticker=ticker,
    ret=ret,
    var_h=var_h,
    cvar_h=cvar_h,
    var_td=var_td,
    var_g=var_g,
    vol_a=vol_a,
    conf=conf,
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
    st.success("✅ Pre-flight checks passed — no critical risk flags detected.")

st.markdown("")

# ── LAYER 2: Build context + button ──────────────────────────────────────────
risk_context = _build_risk_context(
    ticker=ticker,
    ret=ret,
    var_h=var_h,
    cvar_h=cvar_h,
    var_td=var_td,
    var_g=var_g,
    vol_a=vol_a,
    conf=conf,
    rf_rate=rf_rate,
    danger_flags=danger_flags,
    data_source=data_source,
)

# Reset AI summary when context changes
context_key = json.dumps(
    {k: v for k, v in risk_context.items() if k != "danger_flags"},
    sort_keys=True, default=str,
)
if st.session_state.get("risk_ai_context_key") != context_key:
    st.session_state["risk_ai_context_key"] = context_key
    st.session_state["risk_ai_summary"]     = ""

col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    st.markdown("**What Gemini sees:**")

    var_e  = risk_context["var_estimates"]
    dist   = risk_context["return_distribution"]
    ratios = risk_context["risk_ratios"]

    preview_rows = [
        {"Field": "Ticker",              "Value": ticker},
        {"Field": "Confidence",          "Value": f"{conf:.0%}"},
        {"Field": "Observations",        "Value": str(dist.get("annualised_vol_pct", ""))},
        {"Field": "Hist VaR",            "Value": f"{var_e['historical']:.4%}"},
        {"Field": "CVaR",                "Value": f"{var_e['cvar_historical']:.4%}"},
        {"Field": "t-dist VaR",          "Value": f"{var_e['student_t']:.4%}"},
        {"Field": "GARCH VaR",           "Value": f"{var_e['garch']:.4%}"},
        {"Field": "Ann. Vol",            "Value": f"{dist['annualised_vol_pct']:.2f}%"},
        {"Field": "CVaR/VaR ratio",      "Value": str(ratios["cvar_over_var"])},
        {"Field": "GARCH/Hist ratio",    "Value": str(ratios["garch_over_historical"])},
        {"Field": "Max Drawdown",        "Value": f"{dist['max_drawdown']:.2%}"},
        {"Field": "Kurtosis",            "Value": str(dist["kurtosis"])},
        {"Field": "Skewness",            "Value": str(dist["skewness"])},
        {"Field": "Worst Day",           "Value": f"{dist['worst_single_day']:.2%}"},
        {"Field": "Data source",         "Value": data_source},
        {"Field": "Danger flags",        "Value": str(risk_context["danger_flag_count"])},
        {"Field": "Warning flags",       "Value": str(risk_context["warning_flag_count"])},
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
        key="risk_ai_explain",
        use_container_width=True,
        help="Translates the risk output above into plain English using Gemini.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="risk_ai_clear",
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
      The actual numbers from this run (all four VaR estimates, CVaR, GARCH/historical ratio,
      CVaR/VaR ratio, annualised vol, kurtosis, skewness, max drawdown, worst day,
      observation count, and all flags) are sent to Gemini.
    </li>
    <li style="margin-bottom:6px;">
      Gemini explains each number in plain English — specifically the difference between
      <strong>VaR, CVaR, and GARCH VaR</strong> — flags anything dangerous, and writes
      a plain-English conclusion.
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
    st.session_state["risk_ai_summary"] = ""

if decode_clicked:
    with st.spinner("Gemini is reading the risk output and writing your plain-English explanation..."):
        st.session_state["risk_ai_summary"] = _call_gemini_risk_explainer(risk_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("risk_ai_summary"):
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
    st.markdown(st.session_state["risk_ai_summary"])
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
          of the risk output above.
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FAQs — always at the very bottom, after all output and AI decoder
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("")
qe_faq_section("FAQs", [
    ("Which risk number should I trust most?", "Use VaR, CVaR, and GARCH together. VaR is the baseline, CVaR shows tail severity, and GARCH reacts fastest to stress."),
    ("Why compare portfolio risk to single-stock risk?", "A portfolio can be much safer than any one asset because correlations matter. That comparison shows the benefit of diversification."),
    ("What does the Kupiec test tell me?", "It checks whether your VaR model is underestimating or overestimating losses by counting actual violations against the expected rate."),
    ("What should I do if risk is too high?", "Reduce size, lower concentration, or switch to a more defensive allocation until drawdown and volatility stabilize."),
])