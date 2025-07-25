
        # auto_trader/notification.py

import smtplib
import requests
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils.logger import setup_logger

logger = setup_logger(__name__)


class NotificationManager:
    def __init__(self, user_config: dict):
        """
        Initialize the Notification Manager with user-specific settings.

        :param user_config: A dictionary containing user preferences such as email, webhook_url, etc.
        """
        self.user_id = user_config.get("user_id")
        self.email = user_config.get("email")
        self.webhook_url = user_config.get("webhook_url")
        self.enable_email = user_config.get("enable_email", False)
        self.enable_webhook = user_config.get("enable_webhook", False)
        self.smtp_config = user_config.get("smtp", {})

    def send_email(self, subject: str, message: str) -> bool:
        if not self.enable_email or not self.email:
            logger.warning(f"[{self.user_id}] Email notification skipped: Email disabled or missing.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_config.get("from_email")
            msg["To"] = self.email
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            with smtplib.SMTP(self.smtp_config.get("host"), self.smtp_config.get("port")) as server:
                server.starttls()
                server.login(self.smtp_config.get("from_email"), self.smtp_config.get("password"))
                server.send_message(msg)

            logger.info(f"[{self.user_id}] Email notification sent to {self.email}")
            return True
        except Exception as e:
            logger.error(f"[{self.user_id}] Failed to send email: {e}")
            return False

    def send_webhook(self, message: str) -> bool:
        if not self.enable_webhook or not self.webhook_url:
            logger.warning(f"[{self.user_id}] Webhook notification skipped: Webhook disabled or missing.")
            return False

        try:
            response = requests.post(self.webhook_url, json={"text": message})
            response.raise_for_status()
            logger.info(f"[{self.user_id}] Webhook notification sent")
            return True
        except Exception as e:
            logger.error(f"[{self.user_id}] Failed to send webhook: {e}")
            return False

    def notify(self, subject: str, message: str) -> None:
        """
        Unified notification dispatcher to send messages through all enabled channels.
        """
        if self.enable_email:
            self.send_email(subject, message)

        if self.enable_webhook:
            self.send_webhook(message)