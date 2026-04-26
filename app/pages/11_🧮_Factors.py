"""
app/pages/07_factors.py — QuantEdge Factor Research Lab
════════════════════════════════════════════════════════
All original factor-research logic preserved exactly.

AI LAYER — Gemini AI Factor Decoder (bottom of page, identical 3-layer design
as 08_portfolio.py):
  Layer 1: Deterministic danger flags (always shown, no AI)
            Checks IC quality, cost-adjusted profitability, crowding, alpha,
            data sufficiency, composite weight anomalies, decay cliff.
  Layer 2: Context builder + "Decode for Me" button
            Packages the full factor analysis output into a structured JSON
            context dict that mirrors the portfolio page's _build_portfolio_context().
  Layer 3: Gemini output — structured 4-section explanation with mandatory
            disclaimer (identical format: What the output says / What each
            number means / Red flags / Plain English conclusion).

  Uses GEMINI_API_KEY + GEMINI_MODEL from utils/config.py (same as portfolio page).
  Falls back to deterministic explanation if key missing or API call fails.

  Architecture mirrors 08_portfolio.py exactly:
    _compute_factor_danger_flags()   ← deterministic pre-flight checks
    _build_factor_context()          ← structured JSON context for Gemini
    _fallback_factor_explanation()   ← deterministic fallback prose
    _GEMINI_FACTOR_SYSTEM_PROMPT     ← threshold-aware system prompt
    _call_gemini_factor_explainer()  ← urllib-based Gemini REST call
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
import plotly.express as px
from plotly.subplots import make_subplots
try:
    import streamlit as st
except Exception:
    from utils._stubs import st as st

from core.data import returns
import core.factor_engine as fe

momentum_factor   = fe.momentum_factor
low_vol_factor    = fe.low_vol_factor
size_factor       = fe.size_factor
quality_factor    = fe.quality_factor
value_factor      = fe.value_factor
FACTOR_FNS        = getattr(fe, "FACTOR_FNS", {})

build_factor_matrix      = fe.build_factor_matrix
quintile_returns         = fe.quintile_returns
factor_decay             = fe.factor_decay
compute_timeseries_ic    = fe.compute_timeseries_ic
factor_summary_stats     = fe.factor_summary_stats
cost_adjusted_quintile_bt = fe.cost_adjusted_quintile_bt
regime_factor_ic         = fe.regime_factor_ic
ic_weighted_composite    = fe.ic_weighted_composite
factor_attribution       = fe.factor_attribution
detect_factor_crowding   = fe.detect_factor_crowding
cross_sectional_decay    = fe.cross_sectional_decay

from app.data_engine import (
    render_data_engine_controls,
    render_multi_ticker_input,
    load_multi_ticker_data,
    get_global_start_date,
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


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC DANGER FLAGS
# Mirrors _compute_portfolio_danger_flags() in 08_portfolio.py exactly.
# Runs before Gemini — always shown regardless of API availability.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_factor_danger_flags(
    tickers: list[str],
    factor_choice: str,
    ic_summary_rows: list[dict],
    qbt: dict,
    crowd: dict,
    attr: dict,
    composite_scores: pd.Series,
    weights: dict,
    decay_df: pd.DataFrame,
    fwd_days: int,
    cost_bps: float,
    rebal_freq: int,
    data_source: str = "unknown",
) -> list[dict]:
    """
    Deterministic pre-flight checks for the factor research output.
    Returns flags with: severity ("DANGER" | "WARNING" | "INFO"), code, message.
    """
    flags = []

    # ── Data quality ──────────────────────────────────────────────────────────
    if data_source in ("demo", ""):
        flags.append({
            "severity": "INFO",
            "code": "DEMO_DATA",
            "message": (
                f"Factor analysis for {', '.join(tickers)} is running on SYNTHETIC "
                "demo data, not real market prices. All IC values, backtest CAGRs, "
                "and crowding metrics are illustrative only. Do not make allocation "
                "decisions based on this output."
            ),
        })

    # ── IC quality — factor by factor ─────────────────────────────────────────
    for row in ic_summary_rows:
        fname = row.get("Factor", "Unknown")
        ic    = row.get("Mean IC")
        icir  = row.get("ICIR")

        if isinstance(ic, float):
            if abs(ic) < 0.02:
                flags.append({
                    "severity": "DANGER",
                    "code": f"LOW_IC_{fname.upper()}",
                    "message": (
                        f"{fname} Mean IC is {ic:.4f} — effectively zero. "
                        "A factor with |IC| < 0.02 has no meaningful predictive power "
                        "in this universe. Including it in the composite adds noise, "
                        "not signal. Remove it or reduce its weight."
                    ),
                })
            elif abs(ic) < 0.05:
                flags.append({
                    "severity": "WARNING",
                    "code": f"WEAK_IC_{fname.upper()}",
                    "message": (
                        f"{fname} Mean IC is {ic:.4f} — below the 0.05 threshold "
                        "considered meaningful in academic literature (Grinold & Kahn 1999). "
                        "The signal exists but is weak. Verify it holds out-of-sample."
                    ),
                })

        if isinstance(icir, float) and abs(icir) < 0.3 and isinstance(ic, float) and abs(ic) >= 0.02:
            flags.append({
                "severity": "WARNING",
                "code": f"LOW_ICIR_{fname.upper()}",
                "message": (
                    f"{fname} ICIR is {icir:.4f} — IC is highly volatile relative "
                    "to its mean. An ICIR below 0.3 means the signal is unreliable "
                    "period-to-period even if the mean looks positive. "
                    "Use with caution or blend with more stable factors."
                ),
            })

    # ── Cost-adjusted backtest ────────────────────────────────────────────────
    if "error" not in qbt:
        net_cagr   = qbt.get("ls_net_cagr",   float("nan"))
        gross_cagr = qbt.get("ls_gross_cagr", float("nan"))
        turnover   = qbt.get("avg_turnover",  float("nan"))

        if not np.isnan(net_cagr):
            if net_cagr < 0:
                flags.append({
                    "severity": "DANGER",
                    "code": "NEGATIVE_NET_CAGR",
                    "message": (
                        f"{factor_choice} long/short strategy net CAGR is "
                        f"{net_cagr:.2%} — negative after {cost_bps}bps round-trip "
                        "costs. The factor destroys value at this rebalance frequency. "
                        f"Gross CAGR = {gross_cagr:.2%}. "
                        f"Costs alone consume {(gross_cagr - net_cagr):.2%} per year. "
                        f"Minimum viable rebalance period is approximately "
                        f"{int(rebal_freq * max(1, abs(gross_cagr / (gross_cagr - net_cagr + 1e-9))))} days."
                    ),
                })
            elif net_cagr < 0.03:
                flags.append({
                    "severity": "WARNING",
                    "code": "LOW_NET_CAGR",
                    "message": (
                        f"{factor_choice} net CAGR is {net_cagr:.2%} — positive but "
                        "marginal. After costs the strategy barely clears zero. "
                        "Transaction cost drag is consuming most of the gross alpha. "
                        f"Consider rebalancing less frequently (current: every {rebal_freq} days)."
                    ),
                })

        if not np.isnan(turnover) and turnover > 0.8:
            flags.append({
                "severity": "WARNING",
                "code": "HIGH_TURNOVER",
                "message": (
                    f"Average turnover is {turnover:.1%} per rebalance period — very high. "
                    f"At {cost_bps}bps round-trip cost, annual transaction drag is "
                    f"approximately {turnover * (252 / rebal_freq) * cost_bps / 100:.1f}bps. "
                    "High turnover erodes returns and signals the factor ranking is unstable."
                ),
            })

    # ── Factor crowding ───────────────────────────────────────────────────────
    if "error" not in crowd:
        if crowd.get("is_crowded"):
            pctile = crowd.get("current_pctile", 0)
            flags.append({
                "severity": "DANGER",
                "code": "FACTOR_CROWDED",
                "message": (
                    f"{factor_choice} factor is crowded — score dispersion at "
                    f"{pctile:.0%} of its historical range. "
                    "Crowding (Khandani & Lo 2007) means many funds hold the same "
                    "positions. When they exit simultaneously, the factor suffers "
                    "correlated drawdowns. Reduce allocation size until crowding clears."
                ),
            })

    # ── Alpha significance ────────────────────────────────────────────────────
    if "error" not in attr:
        tstat    = attr.get("alpha_tstat")
        r_sq     = attr.get("r_squared", 0)
        alpha_pc = attr.get("alpha_pct", "N/A")

        if isinstance(tstat, (int, float)):
            if abs(tstat) < 2.0:
                flags.append({
                    "severity": "WARNING",
                    "code": "NO_SIGNIFICANT_ALPHA",
                    "message": (
                        f"Factor attribution shows no statistically significant alpha "
                        f"(t-stat = {tstat:.2f}, need |t| ≥ 2.0). "
                        f"R² = {r_sq:.2%} — returns are {r_sq:.0%} explained by "
                        "standard factor premia (Carhart 1997). "
                        "The strategy provides no return beyond what a cheap factor "
                        "ETF would deliver. Consider switching to a passive implementation."
                    ),
                })
            if isinstance(r_sq, float) and r_sq > 0.90:
                flags.append({
                    "severity": "INFO",
                    "code": "HIGH_FACTOR_EXPLAINED",
                    "message": (
                        f"R² = {r_sq:.2%} — over 90% of strategy returns are explained "
                        "by standard factor exposures. The portfolio is essentially a "
                        "factor-tilted index. Alpha (if any) is a small residual."
                    ),
                })

    # ── Composite weight anomalies ────────────────────────────────────────────
    if weights:
        neg_factors = [f for f, w in weights.items() if w < -0.1]
        zero_factors = [f for f, w in weights.items() if abs(w) < 0.01]

        if neg_factors:
            flags.append({
                "severity": "WARNING",
                "code": "NEGATIVE_IC_WEIGHTS",
                "message": (
                    f"Factors {', '.join(neg_factors)} have negative IC weights "
                    "in the composite. This means these factors are currently "
                    "inversely predictive — high-scoring stocks underperform. "
                    "The composite is partially shorting signal direction. "
                    "Verify this is intentional and regime-consistent."
                ),
            })

        if zero_factors:
            flags.append({
                "severity": "INFO",
                "code": "ZERO_WEIGHT_FACTORS",
                "message": (
                    f"Factors {', '.join(zero_factors)} have near-zero IC weight "
                    "and contribute almost nothing to the composite. "
                    "Consider removing them to reduce model complexity."
                ),
            })

    # ── Decay cliff detection ─────────────────────────────────────────────────
    if not decay_df.empty and "IC" in decay_df.columns:
        ic_vals = decay_df["IC"].dropna().values
        horizons = decay_df["Horizon (days)"].values if "Horizon (days)" in decay_df.columns else []

        if len(ic_vals) >= 2:
            short_ic = ic_vals[0]
            long_ic  = ic_vals[-1]
            if short_ic > 0.05 and long_ic < 0.01:
                flags.append({
                    "severity": "INFO",
                    "code": "FAST_DECAY",
                    "message": (
                        f"{factor_choice} IC decays rapidly — strong at short horizons "
                        f"(IC={short_ic:.3f}) but collapses at longer horizons "
                        f"(IC={long_ic:.3f}). "
                        f"Optimal rebalance period is short — holding positions too long "
                        "will dilute alpha significantly."
                    ),
                })
            elif short_ic < 0.02 and len(ic_vals) > 2 and max(ic_vals) > 0.05:
                # IC peaks in the middle — unusual, flag it
                peak_horizon = horizons[np.argmax(ic_vals)] if len(horizons) == len(ic_vals) else "unknown"
                flags.append({
                    "severity": "INFO",
                    "code": "DELAYED_IC_PEAK",
                    "message": (
                        f"{factor_choice} IC peaks at {peak_horizon}-day horizon, not at "
                        "the shortest window. This is a slow-moving signal — "
                        "short holding periods are suboptimal. Rebalance at or near "
                        f"the {peak_horizon}-day mark."
                    ),
                })

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — FACTOR CONTEXT BUILDER
# Mirrors _build_portfolio_context() structure in 08_portfolio.py exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _build_factor_context(
    tickers: list[str],
    factor_choice: str,
    fwd_days: int,
    cost_bps: float,
    rebal_freq: int,
    n_quintiles: int,
    ic_summary_rows: list[dict],
    composite_scores: pd.Series,
    weights: dict,
    qbt: dict,
    crowd: dict,
    attr: dict,
    decay_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    danger_flags: list[dict],
    data_source: str = "unknown",
) -> dict:
    """Build the full structured context dict that gets sent to Gemini."""

    # ── Decay summary ─────────────────────────────────────────────────────────
    decay_summary = []
    if not decay_df.empty and "IC" in decay_df.columns:
        for _, row in decay_df.iterrows():
            decay_summary.append({
                "horizon_days": int(row.get("Horizon (days)", 0)),
                "mean_ic": round(float(row["IC"]), 4) if pd.notna(row["IC"]) else None,
                "ic_std": round(float(row["IC Std"]), 4) if "IC Std" in row and pd.notna(row["IC Std"]) else None,
            })

    # ── Regime IC summary ─────────────────────────────────────────────────────
    regime_ic_summary = []
    if not reg_df.empty:
        for _, row in reg_df.iterrows():
            regime_ic_summary.append({
                "factor": str(row.get("Factor", "")),
                "regime": str(row.get("Regime", "")),
                "mean_ic": round(float(row.get("Mean IC", 0)), 4) if pd.notna(row.get("Mean IC")) else None,
                "signal": str(row.get("Signal", "")),
            })

    # ── IC summary ────────────────────────────────────────────────────────────
    ic_summary_clean = []
    for row in ic_summary_rows:
        ic_summary_clean.append({
            "factor":    row.get("Factor", ""),
            "mean_ic":   round(float(row["Mean IC"]), 4) if isinstance(row.get("Mean IC"), float) else None,
            "icir":      round(float(row["ICIR"]), 4) if isinstance(row.get("ICIR"), float) else None,
            "ic_pos_pct": row.get("IC > 0 %"),
            "obs":       row.get("Obs"),
            "signal":    row.get("Signal", ""),
        })

    # ── Composite scores (top 3 / bottom 3) ───────────────────────────────────
    sorted_comp = composite_scores.sort_values(ascending=False) if composite_scores is not None and len(composite_scores) > 0 else pd.Series(dtype=float)
    top3    = [{"ticker": t, "score": round(float(v), 4)} for t, v in sorted_comp.head(3).items()]
    bottom3 = [{"ticker": t, "score": round(float(v), 4)} for t, v in sorted_comp.tail(3).items()]

    # ── Backtest ──────────────────────────────────────────────────────────────
    backtest_summary = {}
    if "error" not in qbt:
        backtest_summary = {
            "ls_gross_cagr": round(float(qbt.get("ls_gross_cagr", float("nan"))), 4) if not np.isnan(qbt.get("ls_gross_cagr", float("nan"))) else None,
            "ls_net_cagr":   round(float(qbt.get("ls_net_cagr",   float("nan"))), 4) if not np.isnan(qbt.get("ls_net_cagr",   float("nan"))) else None,
            "avg_turnover":  round(float(qbt.get("avg_turnover",  float("nan"))), 4) if not np.isnan(qbt.get("avg_turnover",  float("nan"))) else None,
            "cost_bps":      cost_bps,
        }

    # ── Crowding ──────────────────────────────────────────────────────────────
    crowding_summary = {}
    if "error" not in crowd:
        crowding_summary = {
            "is_crowded":         crowd.get("is_crowded", False),
            "crowding_level":     crowd.get("crowding_level", ""),
            "current_pctile":     round(float(crowd.get("current_pctile", 0)), 4),
            "avg_autocorr":       crowd.get("avg_autocorr"),
            "current_dispersion": crowd.get("current_dispersion"),
        }

    # ── Attribution ───────────────────────────────────────────────────────────
    attribution_summary = {}
    if "error" not in attr:
        attribution_summary = {
            "alpha_pct":   attr.get("alpha_pct", "N/A"),
            "alpha_tstat": attr.get("alpha_tstat"),
            "r_squared":   round(float(attr.get("r_squared", 0)), 4),
            "significant_alpha": abs(attr.get("alpha_tstat", 0)) >= 2.0 if isinstance(attr.get("alpha_tstat"), (int, float)) else False,
        }

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "tickers":      tickers,
        "n_assets":     len(tickers),
        "data_source":  data_source,
        "primary_factor": factor_choice,
        "settings": {
            "fwd_days":    fwd_days,
            "cost_bps":    cost_bps,
            "rebal_freq":  rebal_freq,
            "n_quintiles": n_quintiles,
        },

        # ── IC summary (all factors) ───────────────────────────────────────────
        "ic_summary": ic_summary_clean,

        # ── IC-weighted composite ─────────────────────────────────────────────
        "composite": {
            "weights":    {k: round(float(v), 4) for k, v in weights.items()},
            "top_picks":  top3,
            "bottom_picks": bottom3,
        },

        # ── Cost-adjusted backtest ────────────────────────────────────────────
        "backtest": backtest_summary,

        # ── Regime IC ────────────────────────────────────────────────────────
        "regime_ic": regime_ic_summary,

        # ── Factor attribution ────────────────────────────────────────────────
        "attribution": attribution_summary,

        # ── Crowding ─────────────────────────────────────────────────────────
        "crowding": crowding_summary,

        # ── Decay curve ──────────────────────────────────────────────────────
        "decay_curve": decay_summary,

        # ── Pre-computed danger flags ─────────────────────────────────────────
        "danger_flags":       danger_flags,
        "danger_flag_count":  len([f for f in danger_flags if f["severity"] == "DANGER"]),
        "warning_flag_count": len([f for f in danger_flags if f["severity"] == "WARNING"]),

        # ── Reference thresholds ──────────────────────────────────────────────
        "reference_thresholds": {
            "ic_strong":           0.05,
            "ic_meaningful":       0.02,
            "icir_reliable":       0.5,
            "icir_minimum":        0.3,
            "net_cagr_investable": 0.03,
            "alpha_tstat_significant": 2.0,
            "crowding_pctile_danger": 0.25,
            "high_turnover":       0.80,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — DETERMINISTIC FALLBACK
# Mirrors _fallback_portfolio_explanation() in 08_portfolio.py exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_factor_explanation(context: dict) -> str:
    ic_rows   = context.get("ic_summary", [])
    bt        = context.get("backtest", {})
    crowd     = context.get("crowding", {})
    attr      = context.get("attribution", {})
    composite = context.get("composite", {})
    flags     = context.get("danger_flags", [])
    factor    = context.get("primary_factor", "N/A")
    tickers   = context.get("tickers", [])

    # Best factor by IC
    valid_ic = [r for r in ic_rows if isinstance(r.get("mean_ic"), float)]
    best_factor = max(valid_ic, key=lambda r: abs(r["mean_ic"]), default={}).get("factor", "N/A")

    ic_lines = "\n".join(
        f"- **{r['factor']}**: Mean IC = {r.get('mean_ic', 'N/A')}, "
        f"ICIR = {r.get('icir', 'N/A')}, Signal = {r.get('signal', 'N/A')}"
        for r in ic_rows
    )

    bt_line = (
        f"Gross L/S CAGR = {bt.get('ls_gross_cagr', 'N/A')}, "
        f"Net CAGR = {bt.get('ls_net_cagr', 'N/A')}, "
        f"Avg Turnover = {bt.get('avg_turnover', 'N/A')}"
        if bt else "Backtest not available."
    )

    crowd_line = (
        f"Crowding level: {crowd.get('crowding_level', 'N/A')}, "
        f"Dispersion percentile: {crowd.get('current_pctile', 'N/A'):.0%}"
        if crowd else "Crowding data not available."
    )

    attr_line = (
        f"Alpha: {attr.get('alpha_pct', 'N/A')}, "
        f"t-stat = {attr.get('alpha_tstat', 'N/A')}, "
        f"R² = {attr.get('r_squared', 'N/A'):.2%}"
        if attr else "Attribution not available."
    ) if isinstance(attr.get("r_squared"), float) else "Attribution not available."

    top_picks = ", ".join(
        f"{p['ticker']} ({p['score']:.3f})" for p in composite.get("top_picks", [])
    )

    flag_text = ""
    if flags:
        flag_text = "\n\n**Flags detected:**\n" + "\n".join(
            f"- **{f['severity']}** ({f['code']}): {f['message']}"
            for f in flags
        )

    return (
        f"### What the output says\n"
        f"The factor research lab analysed **{len(tickers)} assets**: "
        f"{', '.join(tickers)}. Primary factor under deep analysis: **{factor}**. "
        f"Best factor by absolute IC: **{best_factor}**. "
        f"Top composite picks: {top_picks or 'N/A'}.\n\n"
        f"### What each number means\n"
        f"{ic_lines}\n\n"
        f"- **Cost-adjusted backtest ({factor})**: {bt_line}\n"
        f"- **Factor attribution**: {attr_line}\n"
        f"- **Crowding**: {crowd_line}\n\n"
        f"### Red flags\n"
        f"{flag_text if flags else 'No critical flags detected.'}\n\n"
        f"### Plain English conclusion\n"
        f"Use **{best_factor}** as the primary signal. "
        f"Review the flags above before trading the factor strategy.\n\n"
        f"⚠️ *This explanation is generated from dashboard outputs only. "
        f"It is not financial advice. Always verify with your own judgment.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI SYSTEM PROMPT
# Threshold-aware, structured, danger-first, with mandatory disclaimer.
# Mirrors _GEMINI_SYSTEM_PROMPT in 08_portfolio.py exactly.
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_FACTOR_SYSTEM_PROMPT = """You are a senior quantitative researcher embedded inside a professional factor research dashboard.

Your sole job: explain the factor analysis output to a NON-TECHNICAL user — a portfolio manager, allocator, or analyst who understands investing but not the mathematics of IC, ICIR, or factor attribution.

RULES (follow all, no exceptions):
1. Use ONLY the numbers and labels in the provided JSON context. Never invent figures.
2. If danger_flag_count > 0 or warning_flag_count > 0, address them FIRST and prominently.
3. Explain every key number in one plain English sentence. Do not skip any metric.
4. Use the reference_thresholds in the context to judge whether each number is good, borderline, or dangerous.
5. Never say "you should buy" or "you should trade" — explain what the analysis says, not what to do.
6. If data_source is "demo", state clearly that these are synthetic numbers, not real prices.
7. Write in short paragraphs. No jargon. No LaTeX. No formulas.

THRESHOLD KNOWLEDGE (built in — use these to interpret numbers):
- Mean IC > 0.05: strong signal (Grinold & Kahn 1999)
- Mean IC 0.02–0.05: weak but meaningful signal
- Mean IC < 0.02: no predictive power — factor should be removed from composite
- ICIR > 0.5: reliable factor (signal is consistent)
- ICIR 0.3–0.5: moderate reliability
- ICIR < 0.3: highly variable — signal exists but is erratic
- Net L/S CAGR < 0: strategy destroys value after costs — not investable
- Net L/S CAGR 0–3%: marginal — costs eat most of the alpha
- Net L/S CAGR > 3%: acceptable net return
- Alpha t-stat ≥ 2.0: statistically significant — genuine skill beyond factor premia
- Alpha t-stat < 2.0: no significant alpha — returns are just factor tilts
- Crowding dispersion pctile < 25%: crowded — liquidation risk is high
- Turnover > 80% per rebalance: very high — transaction costs will be severe
- IC decays fast (strong at 1d, near-zero at 21d): signal is short-lived — rebalance frequently
- Negative IC weights in composite: some factors are inversely predictive in current regime

OUTPUT FORMAT — exactly 4 sections with these markdown headings:
### What the output says
(One paragraph: the primary factor, best factor by IC, composite top picks, overall quality of the factor universe)

### What each number means
(Bullet per key metric: each factor's Mean IC and signal rating, ICIR, net-of-cost CAGR, alpha t-stat, crowding level, decay curve summary)

### Red flags
(If danger or warning flags exist: explain each one in plain English. If none: write "No critical flags detected.")

### Plain English conclusion
(2–3 sentences max. What a smart non-quant should take away from this factor research output.)

End your response with this exact line — no modifications:
⚠️ This explanation is generated by AI from dashboard outputs only. It is not financial advice. Always verify with your own judgment and a qualified professional."""


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER — GEMINI EXPLAINER
# Mirrors _call_gemini_explainer() in 08_portfolio.py exactly.
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini_factor_explainer(context: dict) -> str:
    """
    Calls Google Gemini API with the factor research context.
    Falls back to deterministic explanation on any error.
    """
    gemini_key   = getattr(cfg, "GEMINI_API_KEY", "") or ""
    gemini_model = getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    if not gemini_key:
        return _fallback_factor_explanation(context)

    safe_context = json.loads(json.dumps(context, default=str))
    user_text = (
        "Here is the current factor research output from the dashboard. "
        "Please explain it for a non-technical user:\n\n"
        + json.dumps(safe_context, indent=2)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{gemini_model}:generateContent?key={gemini_key}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": _GEMINI_FACTOR_SYSTEM_PROMPT}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 900,
            "temperature": 0.2,
        },
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
        return text or _fallback_factor_explanation(context)
    except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
        return (
            _fallback_factor_explanation(context)
            + f"\n\n*Note: Gemini API unavailable ({exc.__class__.__name__}). "
            "Add GEMINI_API_KEY to .env for AI explanations.*"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Factors | QuantEdge", layout="wide")
from app.shared import apply_theme
apply_theme()
st.title("🧪 Factor Research Lab")
qe_neon_divider()

# ── Controls ──────────────────────────────────────────────────────────────────
render_data_engine_controls("factors")
c1, c2, c3 = st.columns(3)
tickers      = render_multi_ticker_input("Universe", key="factors_universe",
                   default=cfg.DEFAULT_TICKERS, container=c1)
fwd_days     = c2.slider("Forward Return Window (days)", 1, 63, 21)
factor_choice = c3.selectbox("Primary Factor (for deep analysis)",
                               list(FACTOR_FNS.keys()), index=0)

start = pd.to_datetime(get_global_start_date())

if len(tickers) < 2:
    st.warning("Select at least 2 tickers to compute cross-sectional factors.")
    qe_faq_section("FAQs", [
        ("What is the most important first output here?", "Start with the factor score matrix and IC summary. They tell you which signals are actually predictive in your universe."),
        ("Why does the composite factor change over time?", "Because the page reweights factors by recent IC, so stronger predictors get more influence when markets shift."),
        ("What is the best use of the backtest section?", "Use it to see whether a factor still works after transaction costs and whether the result survives a more realistic test."),
        ("How do I know if a factor is crowded?", "Check the crowding section. A crowded factor can look strong until everyone exits at the same time."),
    ])
    st.stop()

with st.spinner("Loading price data..."):
    prices = load_multi_ticker_data(tickers, start=str(start))

# ── Extra controls ────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Settings"):
    ec = st.columns(3)
    cost_bps    = ec[0].number_input("Round-trip cost (bps)", 10, 200, 40)
    rebal_freq  = ec[1].number_input("Rebalance frequency (days)", 5, 63, 21)
    n_quintiles = ec[2].number_input("Quintiles", 3, 10, 5)

# ── Session-state initialisation ──────────────────────────────────────────────
# KEY FIX: all heavy computation results live in st.session_state.factor_result.
# The Decode button only sets factor_ai_summary — it does NOT re-run any
# computation, so clicking it never wipes the displayed output.
if "factor_result" not in st.session_state:
    st.session_state.factor_result = None
if "factor_ai_summary" not in st.session_state:
    st.session_state.factor_ai_summary = ""
if "factor_ai_context_key" not in st.session_state:
    st.session_state.factor_ai_context_key = ""

run_btn = st.button("▶  Run Full Factor Analysis", type="primary")

if run_btn:
    with st.spinner("Running full factor analysis — this may take a moment..."):
        # ── All heavy computation happens here, once, on Run click ────────────
        factor_df = build_factor_matrix(prices)

        ts_ic_all = {}
        for fname in FACTOR_FNS:
            ts_ic_all[fname] = compute_timeseries_ic(
                prices, factor_name=fname, fwd_days=fwd_days,
                rebalance_freq=int(rebal_freq))

        composite_scores, weights = ic_weighted_composite(prices, ts_ic_all)

        ic_summary_rows = []
        for fname, ts_df in ts_ic_all.items():
            row = factor_summary_stats(ts_df)
            row["Factor"] = fname
            ic_summary_rows.append(row)

        qbt = cost_adjusted_quintile_bt(
            prices, factor_name=factor_choice,
            fwd_days=int(fwd_days),
            round_trip_cost_bps=float(cost_bps),
            n_quintiles=int(n_quintiles),
            rebalance_freq=int(rebal_freq),
        )

        reg_df = regime_factor_ic(prices, fwd_days=int(fwd_days))

        all_rets_list = []
        for t in tickers:
            r = prices[t]["Close"].pct_change().dropna().rename(t)
            all_rets_list.append(r)
        if all_rets_list:
            strat_ret = pd.concat(all_rets_list, axis=1).dropna().mean(axis=1)
            attr = factor_attribution(strat_ret, prices, rf=cfg.RISK_FREE_RATE)
        else:
            attr = {"error": "No return data available."}

        crowd = detect_factor_crowding(prices, factor_name=factor_choice)

        decay_df = cross_sectional_decay(
            prices, factor_name=factor_choice,
            horizons=[1, 5, 10, 21, 63, 126])

        data_source = str(prices.attrs.get("data_source", "unknown")) if hasattr(prices, "attrs") else "unknown"

        # ── Store everything in session_state ─────────────────────────────────
        st.session_state.factor_result = {
            "tickers":          tickers,
            "factor_choice":    factor_choice,
            "fwd_days":         fwd_days,
            "cost_bps":         float(cost_bps),
            "rebal_freq":       int(rebal_freq),
            "n_quintiles":      int(n_quintiles),
            "prices":           prices,
            "factor_df":        factor_df,
            "ts_ic_all":        ts_ic_all,
            "composite_scores": composite_scores,
            "weights":          weights,
            "ic_summary_rows":  ic_summary_rows,
            "qbt":              qbt,
            "reg_df":           reg_df,
            "attr":             attr,
            "crowd":            crowd,
            "decay_df":         decay_df,
            "data_source":      data_source,
        }
        # Reset AI summary whenever a fresh run completes
        st.session_state.factor_ai_summary = ""
        st.session_state.factor_ai_context_key = ""

# ── Guard: nothing to show until Run has been clicked at least once ───────────
if st.session_state.factor_result is None:
    st.info("Configure settings above and click **▶ Run Full Factor Analysis**.")
    qe_faq_section("FAQs", [
        ("What is the most important first output here?", "Start with the factor score matrix and IC summary. They tell you which signals are actually predictive in your universe."),
        ("Why does the composite factor change over time?", "Because the page reweights factors by recent IC, so stronger predictors get more influence when markets shift."),
        ("What is the best use of the backtest section?", "Use it to see whether a factor still works after transaction costs and whether the result survives a more realistic test."),
        ("How do I know if a factor is crowded?", "Check the crowding section. A crowded factor can look strong until everyone exits at the same time."),
    ])
    st.stop()

# ── Unpack session state — all rendering reads from here ──────────────────────
_r              = st.session_state.factor_result
tickers         = _r["tickers"]
factor_choice   = _r["factor_choice"]
fwd_days        = _r["fwd_days"]
cost_bps        = _r["cost_bps"]
rebal_freq      = _r["rebal_freq"]
n_quintiles     = _r["n_quintiles"]
prices          = _r["prices"]
factor_df       = _r["factor_df"]
ts_ic_all       = _r["ts_ic_all"]
composite_scores = _r["composite_scores"]
weights         = _r["weights"]
ic_summary_rows = _r["ic_summary_rows"]
qbt             = _r["qbt"]
reg_df          = _r["reg_df"]
attr            = _r["attr"]
crowd           = _r["crowd"]
decay_df        = _r["decay_df"]
data_source     = _r["data_source"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0 — Factor Score Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📋 Factor Score Matrix  (rank-normalised, current snapshot)")
st.caption(
    "Each score is the percentile rank of that ticker on that factor (0 = worst, 1 = best). "
    "Scores are cross-sectional — they compare tickers to each other, not to history."
)

st.dataframe(
    factor_df.style.background_gradient(cmap="RdYlGn", axis=1),
    use_container_width=True,
)
st.caption(
    "⚠️ Value and Quality use OHLCV-based proxies (price trend deviation, "
    "return consistency). True P/B and ROE require fundamental data."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — IC-Weighted Composite Factor Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("⚖️ IC-Weighted Composite Factor Score  [Fix 5]")
st.caption(
    "Grinold & Kahn (1999): weight each factor by its recent IC. "
    "Stronger predictors get more weight — adapts to changing market conditions."
)

wt_df = pd.DataFrame([{"Factor": k, "IC Weight": round(v, 4)}
                        for k, v in weights.items()])
wt_df["Direction"] = wt_df["IC Weight"].apply(lambda x: "Positive" if x >= 0 else "Negative")

fig_wt = go.Figure(go.Bar(
    x=wt_df["Factor"], y=wt_df["IC Weight"],
    marker_color=["lime" if v >= 0 else "tomato" for v in wt_df["IC Weight"]],
    text=[f"{v:.4f}" for v in wt_df["IC Weight"]], textposition="outside",
))
fig_wt.update_layout(template="plotly_dark", height=300,
    title="Dynamic IC Weights  (updated each rebalance)",
    yaxis_title="Weight (proportional to recent IC)")
st.plotly_chart(fig_wt, use_container_width=True)

comp_df = composite_scores.to_frame("Composite Score").round(4)
comp_df["Rank"] = comp_df["Composite Score"].rank(ascending=False).astype(int)
comp_df = comp_df.sort_values("Composite Score", ascending=False)
st.dataframe(comp_df.style.background_gradient(subset=["Composite Score"],
             cmap="RdYlGn"), use_container_width=True)
st.success(
    f"Top pick: **{comp_df.index[0]}** (composite={comp_df.iloc[0,0]:.3f}) · "
    f"Bottom: **{comp_df.index[-1]}** (composite={comp_df.iloc[-1,0]:.3f})"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Time-Series IC Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📈 Time-Series IC Analysis  [Fix 2]")
st.caption(
    "IC computed at every rebalance date across full history (not a single snapshot). "
    "Mean IC > 0.05 = meaningful. ICIR > 0.5 = reliable."
)

ic_sum_df = pd.DataFrame(ic_summary_rows)[["Factor", "Mean IC", "ICIR", "IC > 0 %", "Obs", "Signal"]]

def _ic_colour(val):
    if isinstance(val, str):
        if "Strong"   in val: return "color:lime"
        if "Moderate" in val: return "color:gold"
        if "Weak"     in val: return "color:tomato"
    return ""

st.dataframe(ic_sum_df.style.map(_ic_colour, subset=["Signal"]),
             use_container_width=True)

ts_sel = ts_ic_all.get(factor_choice, pd.DataFrame())
if not ts_sel.empty and "IC" in ts_sel.columns:
    fig_ic = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[f"{factor_choice} — IC per Rebalance",
                         "Rolling 12-period Mean IC & ICIR"],
        row_heights=[0.5, 0.5], vertical_spacing=0.08)

    fig_ic.add_trace(go.Bar(x=ts_sel["Date"], y=ts_sel["IC"],
        marker_color=["lime" if v >= 0 else "tomato" for v in ts_sel["IC"]],
        name="IC"), row=1, col=1)
    fig_ic.add_hline(y=0.05,  line_dash="dot", line_color="lime",
                      annotation_text="IC=0.05", row=1, col=1)
    fig_ic.add_hline(y=-0.05, line_dash="dot", line_color="tomato", row=1, col=1)
    fig_ic.add_hline(y=0, line_color="gray", row=1, col=1)

    if "RollingMeanIC" in ts_sel.columns:
        fig_ic.add_trace(go.Scatter(x=ts_sel["Date"], y=ts_sel["RollingMeanIC"],
            line=dict(color="cyan", width=2), name="Rolling Mean IC"), row=2, col=1)
    if "RollingICIR" in ts_sel.columns:
        fig_ic.add_trace(go.Scatter(x=ts_sel["Date"], y=ts_sel["RollingICIR"],
            line=dict(color="gold", width=1.5, dash="dash"), name="Rolling ICIR"),
            row=2, col=1)
    fig_ic.add_hline(y=0.5, line_dash="dot", line_color="lime",
                      annotation_text="ICIR=0.5", row=2, col=1)
    fig_ic.update_layout(template="plotly_dark", height=450,
        title=f"{factor_choice} Factor — Full History IC", showlegend=True)
    st.plotly_chart(fig_ic, use_container_width=True)
else:
    st.warning("Insufficient history to compute time-series IC. Use a longer date range.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Cost-Adjusted Quintile Backtest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("💰 Cost-Adjusted Quintile Backtest  [Fix 3]")
st.caption(
    "Novy-Marx & Velikov (2016): factor strategies often look good gross of costs "
    "but lose money net. Deducts round-trip costs on turnover at each rebalance."
)

if "error" in qbt:
    st.warning(qbt["error"])
else:
    qc = st.columns(4)
    qc[0].metric("Gross L/S CAGR",
                  f"{qbt['ls_gross_cagr']:.2%}" if not np.isnan(qbt.get('ls_gross_cagr', np.nan)) else "N/A")
    qc[1].metric("Net L/S CAGR",
                  f"{qbt['ls_net_cagr']:.2%}" if not np.isnan(qbt.get('ls_net_cagr', np.nan)) else "N/A",
                  delta="After costs",
                  delta_color="normal" if not np.isnan(qbt.get('ls_net_cagr', np.nan)) and qbt['ls_net_cagr'] > 0 else "inverse")
    qc[2].metric("Avg Turnover",  f"{qbt['avg_turnover']:.1%}")
    qc[3].metric("Cost per rebal", f"{cost_bps} bps")

    tbl = qbt.get("table", pd.DataFrame())
    if not tbl.empty:
        st.dataframe(tbl, use_container_width=True)

    gross = qbt.get("ls_gross_cagr", np.nan)
    net   = qbt.get("ls_net_cagr",   np.nan)
    if not np.isnan(gross) and not np.isnan(net):
        if net > 0:
            st.success(
                f"Strategy is profitable NET of costs. "
                f"Gross CAGR = {gross:.2%}, Net CAGR = {net:.2%}. "
                f"Costs consume {(gross-net):.2%} of return."
            )
        else:
            st.error(
                f"Strategy is UNPROFITABLE net of {cost_bps}bps round-trip costs. "
                f"Gross CAGR = {gross:.2%}, Net CAGR = {net:.2%}. "
                f"Reduce rebalancing frequency or cut costs."
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Regime-Conditioned Factor IC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🌦️ Regime-Conditioned Factor Performance  [Fix 4]")
st.caption(
    "Daniel & Moskowitz (2016): Momentum crashes in bear markets. "
    "Each factor's IC is broken down by Bull / Sideways / Bear regime."
)

if reg_df.empty:
    st.warning("Not enough data for regime analysis. Use a longer date range (3+ years).")
else:
    def _reg_colour(val):
        if isinstance(val, str):
            if "Strong"   in val: return "color:lime"
            if "Moderate" in val: return "color:gold"
            if "Weak"     in val: return "color:tomato"
        return ""

    st.dataframe(
        reg_df.style.map(_reg_colour, subset=["Signal"])
                    .background_gradient(subset=["Mean IC"], cmap="RdYlGn"),
        use_container_width=True,
    )

    pivot = reg_df.pivot_table(
        index="Factor", columns="Regime", values="Mean IC", aggfunc="mean"
    )
    fig_rh = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig_rh.update_layout(template="plotly_dark", height=300,
        title="Factor IC by Regime  (green = factor works, red = factor hurts)")
    st.plotly_chart(fig_rh, use_container_width=True)

    best_regime  = reg_df.sort_values("Mean IC", ascending=False).iloc[0]
    worst_regime = reg_df.sort_values("Mean IC").iloc[0]
    st.info(
        f"Best combo: **{best_regime['Factor']}** in **{best_regime['Regime']}** "
        f"(IC={best_regime['Mean IC']:.4f}) · "
        f"Worst: **{worst_regime['Factor']}** in **{worst_regime['Regime']}** "
        f"(IC={worst_regime['Mean IC']:.4f})"
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Factor Attribution (Carhart)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🔬 Factor Attribution (Carhart 1997)  [Fix 6]")
st.caption(
    "Decomposes returns of the composite factor strategy into alpha + factor betas. "
    "High alpha (t-stat > 2) = genuine skill. Low alpha = just riding factor premia."
)

if "error" in attr:
    st.warning(attr["error"])
else:
    ac = st.columns(3)
    ac[0].metric("Annualised Alpha",  attr["alpha_pct"])
    ac[1].metric("Alpha t-stat",      str(attr["alpha_tstat"]),
                  delta=("Significant ✅" if abs(attr["alpha_tstat"]) >= 2 else "Not significant ❌"),
                  delta_color="normal" if abs(attr["alpha_tstat"]) >= 2 else "inverse")
    ac[2].metric("R² (factor explained)", f"{attr['r_squared']:.2%}")

    def _sig_colour(val):
        if "Yes" in str(val): return "color:lime"
        if "No"  in str(val): return "color:gray"
        return ""

    st.dataframe(
        attr["table"].style.map(_sig_colour, subset=["Significant"]),
        use_container_width=True,
    )

    if abs(attr["alpha_tstat"]) >= 2.0:
        st.success(
            f"Statistically significant alpha detected: {attr['alpha_pct']} per year "
            f"(t={attr['alpha_tstat']}). The strategy generates return beyond factor exposure."
        )
    else:
        st.warning(
            f"No significant alpha (t={attr['alpha_tstat']}). "
            "The strategy returns are explained by factor exposure alone. "
            "A cheap factor ETF would give similar returns at lower cost."
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — Factor Crowding Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("🚨 Factor Crowding Detection  [Fix 7]")
st.caption(
    "Khandani & Lo (2007): crowded factors cause correlated liquidation — "
    "all funds lose simultaneously. Detected via collapse in cross-sectional score dispersion."
)

if "error" in crowd:
    st.warning(crowd["error"])
else:
    cc = st.columns(4)
    cc[0].metric("Crowding Level",      crowd["crowding_level"])
    cc[1].metric("Dispersion Percentile",
                  f"{crowd['current_pctile']:.1%}",
                  help="Low percentile = crowded (everyone holds same stocks)")
    cc[2].metric("Current Dispersion",  str(crowd["current_dispersion"]))
    cc[3].metric("Avg Score AutoCorr",  str(crowd["avg_autocorr"]),
                  help="High autocorrelation = rankings are persistent = crowded")

    if crowd["is_crowded"]:
        st.error(
            f"Factor crowding detected — level: {crowd['crowding_level']}. "
            f"Score dispersion at {crowd['current_pctile']:.0%} of historical range. "
            "Reduce position size or switch factors. "
            "High crowding → correlated losses when liquidation occurs."
        )
    else:
        st.success(
            f"No significant crowding — dispersion at {crowd['current_pctile']:.0%} "
            "of historical range. Factor appears uncrowded."
        )

    cdf = crowd.get("crowding_series")
    if cdf is not None and not cdf.empty:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Dispersion"],
            fill="tozeroy", fillcolor="rgba(100,180,255,0.15)",
            line=dict(color="royalblue", width=1.5), name="Score Dispersion"))
        lo_pct = float(cdf["Dispersion"].quantile(0.25))
        fig_c.add_hline(y=lo_pct, line_dash="dot", line_color="orange",
                         annotation_text="25th pct (crowding zone)")
        fig_c.update_layout(template="plotly_dark", height=300,
            title=f"{factor_choice} Factor Score Dispersion over Time  "
                   "(drops = crowding, spikes = dispersion / opportunity)")
        st.plotly_chart(fig_c, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — Cross-Sectional Factor Decay Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📉 Cross-Sectional Factor Decay Curve  [Fix 8]")
st.caption(
    "IC computed cross-sectionally at multiple sample dates for each horizon — "
    "not a single ticker's time series. Grinold & Kahn (1999) methodology. "
    "Tells you how fast the signal loses predictive power."
)

if not decay_df.empty and decay_df["IC"].notna().any():
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=decay_df["Horizon (days)"], y=decay_df["IC"],
        mode="lines+markers", line=dict(color="cyan", width=2),
        error_y=dict(type="data", array=decay_df["IC Std"].fillna(0).tolist(),
                      visible=True, color="rgba(100,200,255,0.4)"),
        name="Mean IC",
    ))
    fig_d.add_hline(y=0,    line_dash="dash",  line_color="gray")
    fig_d.add_hline(y=0.05, line_dash="dot",   line_color="lime",
                     annotation_text="IC=0.05 threshold")
    fig_d.update_layout(template="plotly_dark", height=340,
        title=f"{factor_choice} — Cross-Sectional IC Decay "
               "(error bars = std across sample dates)",
        xaxis_title="Forward Horizon (days)",
        yaxis_title="Mean Cross-Sectional IC",
    )
    st.plotly_chart(fig_d, use_container_width=True)
    st.dataframe(decay_df, use_container_width=True)

    pos_ic = decay_df[decay_df["IC"] > 0.02]
    if not pos_ic.empty:
        optimal = int(pos_ic["Horizon (days)"].max())
        st.info(
            f"Signal remains meaningful (IC > 0.02) up to **{optimal}-day** horizon. "
            f"Optimal rebalance frequency for {factor_choice}: every {optimal} days."
        )
    else:
        st.warning(
            "Signal IC is below 0.02 at all horizons. "
            f"{factor_choice} may not have predictive power in this universe."
        )
else:
    st.warning("Not enough data for decay curve. Use a longer date range (3+ years).")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — Summary & Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.subheader("📋 Factor Research Summary")

best_factor = max(
    ic_summary_rows,
    key=lambda r: abs(r.get("Mean IC", 0)) if isinstance(r.get("Mean IC", 0), float) else 0,
    default={},
).get("Factor", "N/A")

recs = []

for row in ic_summary_rows:
    if isinstance(row.get("Mean IC"), float) and abs(row["Mean IC"]) < 0.02:
        recs.append(f"⚠️ **{row['Factor']}** has very low IC ({row['Mean IC']:.4f}). "
                     "Consider removing from composite — it adds noise, not signal.")

if "ls_net_cagr" in qbt and not np.isnan(qbt.get("ls_net_cagr", np.nan)):
    if qbt["ls_net_cagr"] <= 0:
        recs.append(f"🔴 **{factor_choice}** quintile strategy loses money net of "
                     f"{cost_bps}bps costs. Reduce rebalancing frequency to "
                     f"at least {int(rebal_freq * 2)} days.")

if crowd.get("is_crowded"):
    recs.append(f"🔴 **{factor_choice}** is crowded (dispersion at "
                 f"{crowd.get('current_pctile', 0):.0%} percentile). "
                 "Reduce allocation or switch factors.")

if "alpha_tstat" in attr and not isinstance(attr.get("alpha_tstat"), str):
    if abs(attr["alpha_tstat"]) < 2.0:
        recs.append("⚠️ No significant alpha detected. Portfolio returns are "
                     "fully explained by factor premia — consider a cheaper "
                     "passive factor ETF instead.")

recs.append(f"✅ Best factor by IC: **{best_factor}**. "
             f"Use as primary signal with IC-weighted composite (Fix 5).")

for rec in recs:
    st.markdown(rec)


# ══════════════════════════════════════════════════════════════════════════════
# AI DECODER SECTION — Gemini-powered
# Identical 3-layer architecture to 08_portfolio.py:
#   Layer 1: Deterministic danger badges (always shown, no AI)
#   Layer 2: "Decode for Me" button → Gemini explanation
#   Layer 3: Structured AI output with disclaimer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown("""
<div style="margin: 8px 0 4px;">
  <span style="font-size:20px;font-weight:600;">🤖 AI Factor Decoder</span>
  <span style="font-size:12px;opacity:0.55;margin-left:12px;">
    Plain-English explanation for non-technical users · Powered by Gemini
  </span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "This section translates the quantitative factor research output above into plain English. "
    "It reads the actual numbers from this analysis — not generic descriptions. "
    "It does not change the factor scores. It does not give financial advice."
)

# ── LAYER 1: Deterministic danger flags ───────────────────────────────────────
data_source = str(prices.attrs.get("data_source", "unknown")) if hasattr(prices, "attrs") else "unknown"

danger_flags = _compute_factor_danger_flags(
    tickers=tickers,
    factor_choice=factor_choice,
    ic_summary_rows=ic_summary_rows,
    qbt=qbt,
    crowd=crowd,
    attr=attr,
    composite_scores=composite_scores,
    weights=weights,
    decay_df=decay_df,
    fwd_days=fwd_days,
    cost_bps=float(cost_bps),
    rebal_freq=int(rebal_freq),
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
    st.success("✅ Pre-flight checks passed — no critical flags detected for this factor analysis.")

st.markdown("")

# ── LAYER 2: Build context + button ──────────────────────────────────────────
factor_context = _build_factor_context(
    tickers=tickers,
    factor_choice=factor_choice,
    fwd_days=fwd_days,
    cost_bps=float(cost_bps),
    rebal_freq=int(rebal_freq),
    n_quintiles=int(n_quintiles),
    ic_summary_rows=ic_summary_rows,
    composite_scores=composite_scores,
    weights=weights,
    qbt=qbt,
    crowd=crowd,
    attr=attr,
    decay_df=decay_df,
    reg_df=reg_df if not reg_df.empty else pd.DataFrame(),
    danger_flags=danger_flags,
    data_source=data_source,
)

# Reset AI summary when context changes (new run)
context_key = json.dumps(
    {k: v for k, v in factor_context.items() if k != "danger_flags"},
    sort_keys=True, default=str,
)
if st.session_state.get("factor_ai_context_key") != context_key:
    st.session_state.factor_ai_context_key = context_key
    st.session_state.factor_ai_summary = ""

col_btn, col_ctx = st.columns([1, 2])

with col_btn:
    st.markdown("**What Gemini sees:**")

    # Preview table — mirrors portfolio page exactly
    valid_ic = [r for r in ic_summary_rows if isinstance(r.get("Mean IC"), float)]
    best_factor_preview = max(valid_ic, key=lambda r: abs(r["mean_ic"]) if isinstance(r.get("mean_ic"), float) else abs(r.get("Mean IC", 0)), default={}).get("Factor", "N/A") if valid_ic else "N/A"

    preview_rows = [
        {"Field": "Assets",               "Value": ", ".join(tickers)},
        {"Field": "Primary factor",       "Value": factor_choice},
        {"Field": "Data source",          "Value": data_source},
        {"Field": "Fwd return window",    "Value": f"{fwd_days}d"},
        {"Field": "Rebal frequency",      "Value": f"{rebal_freq}d"},
        {"Field": "Cost (bps)",           "Value": str(cost_bps)},
        {"Field": "Best factor (IC)",     "Value": best_factor},
        {"Field": "Net L/S CAGR",         "Value": f"{qbt.get('ls_net_cagr', float('nan')):.2%}" if "ls_net_cagr" in qbt and not np.isnan(qbt.get("ls_net_cagr", float("nan"))) else "N/A"},
        {"Field": "Crowding level",       "Value": crowd.get("crowding_level", "N/A") if "error" not in crowd else "N/A"},
        {"Field": "Alpha t-stat",         "Value": str(attr.get("alpha_tstat", "N/A")) if "error" not in attr else "N/A"},
        {"Field": "Danger flags",         "Value": str(len([f for f in danger_flags if f["severity"] == "DANGER"]))},
        {"Field": "Warning flags",        "Value": str(len([f for f in danger_flags if f["severity"] == "WARNING"]))},
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
        key="factor_ai_explain",
        use_container_width=True,
        help="Translates the factor research output above into plain English using Gemini.",
    )
    clear_clicked = st.button(
        "Clear explanation",
        key="factor_ai_clear",
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
      The actual numbers from this analysis (IC for every factor, ICIR, composite
      weights, cost-adjusted CAGR, alpha t-stat, crowding level, decay curve,
      regime-conditioned IC, data source, and all flags) are sent to Gemini.
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
    st.session_state.factor_ai_summary = ""

if decode_clicked:
    with st.spinner("Gemini is reading the factor research output and writing your plain-English explanation..."):
        st.session_state.factor_ai_summary = _call_gemini_factor_explainer(factor_context)

# ── LAYER 3: AI output ────────────────────────────────────────────────────────
if st.session_state.get("factor_ai_summary"):
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
    st.markdown(st.session_state.factor_ai_summary)
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
          of the factor research output above.
        </div>""",
        unsafe_allow_html=True,
    )

# ── FAQs ──────────────────────────────────────────────────────────────────────
st.markdown("")
qe_faq_section("FAQs", [
    ("What is the most important first output here?", "Start with the factor score matrix and IC summary. They tell you which signals are actually predictive in your universe."),
    ("Why does the composite factor change over time?", "Because the page reweights factors by recent IC, so stronger predictors get more influence when markets shift."),
    ("What is the best use of the backtest section?", "Use it to see whether a factor still works after transaction costs and whether the result survives a more realistic test."),
    ("How do I know if a factor is crowded?", "Check the crowding section. A crowded factor can look strong until everyone exits at the same time."),
])