"""
utils/notifications.py
──────────────────────────────────────────────────────────────────────────────
Email notification system with credentials sourced from .env (never hardcoded).
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from loguru import logger
from utils.config import cfg


def send_email(subject: str, body: str) -> bool:
    """
    Send a plain-text email alert via Gmail SMTP.

    Parameters
    ----------
    subject : str  — email subject line
    body    : str  — email body text

    Returns
    -------
    bool — True on success, False on failure
    """
    receiver = cfg.GMAIL_RECEIVER or cfg.GMAIL_SENDER

    if not cfg.GMAIL_PASSWORD or not cfg.GMAIL_SENDER:
        logger.warning("Email credentials not configured — skipping notification")
        return False
    if not receiver:
        logger.warning("Email receiver not configured — skipping notification")
        return False

    msg = MIMEMultipart()
    msg["From"]    = cfg.GMAIL_SENDER
    msg["To"]      = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(cfg.SMTP_SERVER, cfg.SMTP_PORT) as server:
            server.starttls()
            server.login(cfg.GMAIL_SENDER, cfg.GMAIL_PASSWORD)
            server.sendmail(cfg.GMAIL_SENDER, receiver, msg.as_string())
        logger.info(f"Alert email sent: {subject}")
        return True
    except Exception as exc:
        logger.error(f"Failed to send email: {exc}")
        return False


def build_alert_body(ticker: str, metric: str, price: float,
                     threshold: float, insight: str) -> str:
    """Compose a formatted alert email body."""
    return (
        f"QuantEdge Price Alert\n"
        f"{'─' * 40}\n"
        f"Ticker  : {ticker}\n"
        f"Metric  : {metric}\n"
        f"Current : ${price:,.2f}\n"
        f"Threshold: ${threshold:,.2f}\n\n"
        f"Educational Insight\n"
        f"{'─' * 40}\n"
        f"{insight}\n"
    )
