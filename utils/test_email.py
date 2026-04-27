"""
test_email.py
─────────────────────────────────────────────────────────────────
Run this from the project root to verify your email setup:

    python test_email.py

It will tell you exactly what's wrong if the email fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from utils.config import cfg
from utils.notifications import send_email, build_alert_body, build_alert_body_html

print("=" * 55)
print("QuantEdge Email Diagnostics")
print("=" * 55)
print(f"GMAIL_SENDER    : {'✅ ' + cfg.GMAIL_SENDER if cfg.GMAIL_SENDER else '❌ NOT SET'}")
print(f"GMAIL_PASSWORD  : {'✅ SET (hidden)' if cfg.GMAIL_PASSWORD else '❌ NOT SET'}")
print(f"GMAIL_RECEIVER  : {'✅ ' + cfg.GMAIL_RECEIVER if cfg.GMAIL_RECEIVER else '⚠️  not set (will use SENDER)'}")
print(f"SMTP            : {cfg.SMTP_SERVER}:{cfg.SMTP_PORT}")
print()

if not cfg.GMAIL_SENDER or not cfg.GMAIL_PASSWORD:
    print("❌ Cannot test — fill in GMAIL_SENDER and GMAIL_PASSWORD in your .env file")
    print("   See .env.example for instructions on generating a Gmail App Password")
    sys.exit(1)

print("Sending test email...")
plain = build_alert_body("TEST", "Diagnostics", 0, 0, "This is a test email from QuantEdge diagnostics script.")
html  = build_alert_body_html("TEST", "Diagnostics", 0, 0, "This is a test email from QuantEdge diagnostics script.", level="INFO")

ok, reason = send_email(
    subject="QuantEdge — Email Test",
    body=plain,
    html_body=html,
)

print()
if ok:
    print(f"✅ Email sent successfully to {cfg.GMAIL_RECEIVER or cfg.GMAIL_SENDER}")
    print("   Check your inbox — alerts are now working.")
else:
    print(f"❌ Email failed: {reason}")
    print()
    if "authentication" in reason.lower():
        print("   FIX: Your GMAIL_PASSWORD is wrong or not an App Password.")
        print("   Generate one at: myaccount.google.com → Security → App passwords")
    elif "timeout" in reason.lower():
        print("   FIX: Port 587 is blocked. Try from a different network or check firewall.")
    elif "not set" in reason.lower():
        print("   FIX: Fill in the missing field in your .env file.")