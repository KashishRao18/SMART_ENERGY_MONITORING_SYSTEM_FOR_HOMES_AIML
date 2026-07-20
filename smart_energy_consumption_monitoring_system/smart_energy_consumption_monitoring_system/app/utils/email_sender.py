import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

logger = logging.getLogger(__name__)


def _get_email_config():
    """
    Read email credentials from environment.
    Returns (sender_email, sender_password, smtp_server, smtp_port, fallback_to_console)
    """
    sender_email = os.environ.get("EMAIL_SENDER_EMAIL")
    sender_password = os.environ.get("EMAIL_SENDER_PASSWORD")
    smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", 587))
    fallback_to_console = os.environ.get("EMAIL_FALLBACK_CONSOLE", "true").lower() == "true"
    return sender_email, sender_password, smtp_server, smtp_port, fallback_to_console


def _format_sender(sender_email: str) -> str:
    return formataddr(("Smart Energy Ai", sender_email))


def send_otp_email(recipient_email, username, otp):
    """Sends an email with a one-time password (OTP) to the user."""
    sender_email, sender_password, smtp_server, smtp_port, fallback_to_console = _get_email_config()

    if not sender_email or not sender_password:
        if fallback_to_console:
            logger.info(
                "[DEV-FALLBACK] OTP email not sent (missing creds). "
                "To: %s | User: %s | OTP: %s",
                recipient_email,
                username,
                otp,
            )
            return True
        logger.warning("Email sender credentials not set. Skipping OTP email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your One-Time Password (OTP) for Energy Household"
    msg["From"] = _format_sender(sender_email)
    msg["To"] = recipient_email

    text = f"""Hi {username},
Your One-Time Password (OTP) for Energy Household is: {otp}
This OTP is valid for 10 minutes.
If you did not request this, please ignore this email.
Regards,
The Energy Household Team
"""
    html = f"""\
    <html>
      <body>
        <p>Hi {username},</p>
        <p>Your One-Time Password (OTP) for Energy Household is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 10 minutes.</p>
        <p>If you did not request this, please ignore this email.</p>
        <p>Regards,</p>
        <p>The Energy Household Team</p>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("OTP email sent to %s", recipient_email)
        return True
    except Exception as e:
        logger.exception("Error sending OTP email to %s", recipient_email)
        return False


def send_password_reset_email(recipient_email, username, otp):
    """Sends a password reset OTP email to the user."""
    sender_email, sender_password, smtp_server, smtp_port, fallback_to_console = _get_email_config()

    if not sender_email or not sender_password:
        if fallback_to_console:
            logger.info(
                "[DEV-FALLBACK] Password reset email not sent (missing creds). "
                "To: %s | User: %s | OTP: %s",
                recipient_email,
                username,
                otp,
            )
            return True
        logger.warning("Email sender credentials not set. Skipping password reset email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Password Reset OTP for Energy Household"
    msg["From"] = _format_sender(sender_email)
    msg["To"] = recipient_email

    text = f"""Hi {username},
You requested to reset your password for Energy Household.
Your OTP for password reset is: {otp}
This OTP is valid for 10 minutes.
If you did not request this, please ignore this email.
Regards,
The Energy Household Team
"""
    html = f"""\
    <html>
      <body>
        <p>Hi {username},</p>
        <p>You requested to reset your password for Energy Household.</p>
        <p>Your OTP for password reset is: <strong>{otp}</strong></p>
        <p>This OTP is valid for 10 minutes.</p>
        <p>If you did not request this, please ignore this email.</p>
        <p>Regards,</p>
        <p>The Energy Household Team</p>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("Password reset email sent to %s", recipient_email)
        return True
    except Exception as e:
        logger.exception("Error sending password reset email to %s", recipient_email)
        return False
