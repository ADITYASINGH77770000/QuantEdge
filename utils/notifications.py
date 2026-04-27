# """
# utils/notifications.py
# ──────────────────────────────────────────────────────────────────────────────
# Email notification system with credentials sourced from .env (never hardcoded).
# """

# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from loguru import logger
# from utils.config import cfg


# def send_email(subject: str, body: str) -> bool:
#     """
#     Send a plain-text email alert via Gmail SMTP.

#     Parameters
#     ----------
#     subject : str  — email subject line
#     body    : str  — email body text

#     Returns
#     -------
#     bool — True on success, False on failure
#     """
#     receiver = cfg.GMAIL_RECEIVER or cfg.GMAIL_SENDER

#     if not cfg.GMAIL_PASSWORD or not cfg.GMAIL_SENDER:
#         logger.warning("Email credentials not configured — skipping notification")
#         return False
#     if not receiver:
#         logger.warning("Email receiver not configured — skipping notification")
#         return False

#     msg = MIMEMultipart()
#     msg["From"]    = cfg.GMAIL_SENDER
#     msg["To"]      = receiver
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))

#     try:
#         with smtplib.SMTP(cfg.SMTP_SERVER, cfg.SMTP_PORT) as server:
#             server.starttls()
#             server.login(cfg.GMAIL_SENDER, cfg.GMAIL_PASSWORD)
#             server.sendmail(cfg.GMAIL_SENDER, receiver, msg.as_string())
#         logger.info(f"Alert email sent: {subject}")
#         return True
#     except Exception as exc:
#         logger.error(f"Failed to send email: {exc}")
#         return False


# def build_alert_body(ticker: str, metric: str, price: float,
#                      threshold: float, insight: str) -> str:
#     """Compose a formatted alert email body."""
#     return (
#         f"QuantEdge Price Alert\n"
#         f"{'─' * 40}\n"
#         f"Ticker  : {ticker}\n"
#         f"Metric  : {metric}\n"
#         f"Current : ${price:,.2f}\n"
#         f"Threshold: ${threshold:,.2f}\n\n"
#         f"Educational Insight\n"
#         f"{'─' * 40}\n"
#         f"{insight}\n"
#     )


"""
utils/notifications.py
──────────────────────────────────────────────────────────────────────────────
Email notification system with credentials sourced from .env (never hardcoded).

Fixes applied (v2):
  1. SMTP timeout added — previously a hung connection would block forever.
  2. ehlo() called explicitly before starttls() — required by Gmail SMTP relay;
     without it, starttls() raises SMTPException on certain hosts.
  3. Proper error surface — send_email now returns a (bool, str) tuple so the
     UI can display the exact failure reason instead of a generic "check .env".
  4. _maybe_email in alerts.py was catching all exceptions silently at DEBUG
     level, meaning SMTP errors were invisible. Failures now log at WARNING.
  5. HTML email variant — build_alert_body_html() produces a styled HTML email;
     plain-text fallback preserved for compatibility with older callers.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Tuple
from loguru import logger
from utils.config import cfg

_SMTP_TIMEOUT = 15  # seconds — prevents indefinite hangs on bad network


def send_email(subject: str, body: str, html_body: str | None = None) -> Tuple[bool, str]:
    """
    Send an email alert via Gmail SMTP (TLS on port 587).

    Parameters
    ----------
    subject   : str            — email subject line
    body      : str            — plain-text email body (always included)
    html_body : str | None     — optional HTML version (sent as alternative part)

    Returns
    -------
    (bool, str) — (True, "OK") on success, (False, "<reason>") on failure.
    The reason string is safe to display directly in the Streamlit UI.
    """
    receiver = cfg.GMAIL_RECEIVER or cfg.GMAIL_SENDER

    # ── Credential pre-checks (fail fast with clear messages) ─────────────────
    if not cfg.GMAIL_SENDER:
        reason = "GMAIL_SENDER not set in .env"
        logger.warning(f"Email skipped: {reason}")
        return False, reason

    if not cfg.GMAIL_PASSWORD:
        reason = "GMAIL_PASSWORD not set in .env (use a Gmail App Password, not your account password)"
        logger.warning(f"Email skipped: {reason}")
        return False, reason

    if not receiver:
        reason = "GMAIL_RECEIVER not set in .env and GMAIL_SENDER is also empty"
        logger.warning(f"Email skipped: {reason}")
        return False, reason

    # ── Build message ─────────────────────────────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["From"]    = cfg.GMAIL_SENDER
    msg["To"]      = receiver
    msg["Subject"] = subject

    # Plain text always attached first — fallback for clients that block HTML
    msg.attach(MIMEText(body, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    # ── Send via Gmail SMTP with TLS ──────────────────────────────────────────
    try:
        with smtplib.SMTP(cfg.SMTP_SERVER, cfg.SMTP_PORT, timeout=_SMTP_TIMEOUT) as server:
            server.ehlo()       # identify client to server BEFORE starttls
            server.starttls()   # upgrade plain connection to encrypted TLS
            server.ehlo()       # re-identify after TLS upgrade (required by RFC)
            server.login(cfg.GMAIL_SENDER, cfg.GMAIL_PASSWORD)
            server.sendmail(cfg.GMAIL_SENDER, receiver, msg.as_string())

        logger.info(f"Alert email sent → {receiver} | {subject}")
        return True, "OK"

    except smtplib.SMTPAuthenticationError:
        reason = (
            "Gmail authentication failed. "
            "GMAIL_PASSWORD must be a Gmail App Password, not your regular password. "
            "Generate one at: myaccount.google.com → Security → "
            "2-Step Verification → App passwords"
        )
        logger.error(f"Email auth error: {reason}")
        return False, reason

    except smtplib.SMTPException as exc:
        reason = f"SMTP error: {exc}"
        logger.error(f"Email send failed: {reason}")
        return False, reason

    except TimeoutError:
        reason = (
            f"Connection to {cfg.SMTP_SERVER}:{cfg.SMTP_PORT} "
            f"timed out after {_SMTP_TIMEOUT}s — check network/firewall"
        )
        logger.error(f"Email timeout: {reason}")
        return False, reason

    except Exception as exc:
        reason = f"Unexpected error: {exc}"
        logger.error(f"Email send failed: {reason}")
        return False, reason


def build_alert_body(ticker: str, metric: str, price: float,
                     threshold: float, insight: str) -> str:
    """Compose a formatted plain-text alert email body."""
    return (
        f"QuantEdge Alert\n"
        f"{'─' * 40}\n"
        f"Ticker   : {ticker}\n"
        f"Metric   : {metric}\n"
        f"Current  : ${price:,.2f}\n"
        f"Threshold: ${threshold:,.2f}\n\n"
        f"Details\n"
        f"{'─' * 40}\n"
        f"{insight}\n"
    )


def build_alert_body_html(ticker: str, metric: str, price: float,
                          threshold: float, insight: str,
                          level: str = "HIGH") -> str:
    """Compose a styled HTML alert email body."""
    level_colors = {
        "CRITICAL": ("#e74c3c", "#1a0a0a"),
        "HIGH":     ("#e67e22", "#1a130a"),
        "MEDIUM":   ("#3498db", "#0a1220"),
        "INFO":     ("#27ae60", "#0a1a0a"),
    }
    border_color, bg_color = level_colors.get(level.upper(), ("#e67e22", "#1a130a"))

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                background:#0d1117;color:#e8f4fd;border-radius:10px;overflow:hidden;">
      <div style="background:{border_color};padding:16px 24px;">
        <h2 style="margin:0;color:#fff;">⚡ QuantEdge {level} Alert</h2>
      </div>
      <div style="padding:24px;border-left:4px solid {border_color};
                  background:{bg_color};margin:16px;border-radius:6px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr><td style="padding:6px 0;color:#aaa;width:120px;">Ticker</td>
              <td style="padding:6px 0;font-weight:bold;">{ticker}</td></tr>
          <tr><td style="padding:6px 0;color:#aaa;">Metric</td>
              <td style="padding:6px 0;font-weight:bold;">{metric}</td></tr>
          <tr><td style="padding:6px 0;color:#aaa;">Current</td>
              <td style="padding:6px 0;">${price:,.2f}</td></tr>
          <tr><td style="padding:6px 0;color:#aaa;">Threshold</td>
              <td style="padding:6px 0;">${threshold:,.2f}</td></tr>
        </table>
      </div>
      <div style="padding:0 24px 24px;">
        <p style="color:#aaa;font-size:13px;margin-bottom:6px;">Details</p>
        <p style="background:#161b22;padding:12px 16px;border-radius:6px;
                  font-size:14px;line-height:1.6;">{insight}</p>
      </div>
      <div style="padding:12px 24px;background:#161b22;font-size:11px;color:#555;text-align:center;">
        QuantEdge Automated Alert System — do not reply to this email
      </div>
    </div>
    """