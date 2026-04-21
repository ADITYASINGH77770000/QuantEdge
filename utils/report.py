"""
utils/report.py
──────────────────────────────────────────────────────────────────────────────
One-click PDF report export using reportlab.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd

from utils.config import cfg


def generate_pdf_report(ticker: str,
                         metrics: dict,
                         trade_log: pd.DataFrame = None,
                         factor_scores: pd.DataFrame = None) -> Path:
    """
    Generate a formatted PDF analysis report.

    Parameters
    ----------
    ticker       : str
    metrics      : dict of metric_name → value_string
    trade_log    : optional DataFrame of trades
    factor_scores: optional DataFrame of factor data

    Returns
    -------
    Path to the generated PDF file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.units import cm
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    out_path = cfg.EXPORTS_DIR / f"{ticker}_report_{datetime.today().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc      = SimpleDocTemplate(str(out_path), pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    styles   = getSampleStyleSheet()
    story    = []

    # ── Title ─────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 textColor=colors.HexColor("#1a1a2e"),
                                 fontSize=22, spaceAfter=6)
    story.append(Paragraph("QuantEdge Analysis Report", title_style))
    story.append(Paragraph(f"Ticker: <b>{ticker}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                           styles["Normal"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4f4fd4")))
    story.append(Spacer(1, 0.4*cm))

    # ── Metrics table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Performance Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    met_data = [["Metric", "Value"]] + [[k, v] for k, v in metrics.items()]
    met_table = Table(met_data, colWidths=[10*cm, 6*cm])
    met_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#4f4fd4")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f0f8"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#ccccee")),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(met_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Trade Log ─────────────────────────────────────────────────────────────
    if trade_log is not None and len(trade_log) > 0:
        story.append(Paragraph("Trade Log (last 20 trades)", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        sub = trade_log.tail(20)
        tl_data = [list(sub.columns)] + sub.values.tolist()
        tl_table = Table(tl_data, repeatRows=1)
        tl_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2d2d6d")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f8ff"), colors.white]),
        ]))
        story.append(tl_table)
        story.append(Spacer(1, 0.4*cm))

    # ── Factor Scores ─────────────────────────────────────────────────────────
    if factor_scores is not None and len(factor_scores) > 0:
        story.append(Paragraph("Factor Score Matrix", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        fs_data = [["Ticker"] + list(factor_scores.columns)]
        for idx, row in factor_scores.iterrows():
            fs_data.append([str(idx)] + [f"{v:.3f}" for v in row])
        fs_table = Table(fs_data)
        fs_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2d2d6d")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.grey),
        ]))
        story.append(fs_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph("<i>Generated by QuantEdge — Aditya Singh | MCA (Generative AI)</i>",
                           styles["Italic"]))

    doc.build(story)
    return out_path
