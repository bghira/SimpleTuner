"""SMTP email notification channel."""

from __future__ import annotations

import logging
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from ..models import ChannelConfig, DeliveryResult, NotificationEvent
from ..protocols import ChannelType, DeliveryStatus
from .base import BaseNotificationChannel

logger = logging.getLogger(__name__)


class EmailChannel(BaseNotificationChannel):
    """Send notifications via SMTP email.

    Features:
    - Provider presets (Gmail, Outlook, custom)
    - TLS/STARTTLS support
    - HTML templates with text fallback
    - Approval action buttons (Reply-To pattern)
    - Async SMTP via aiosmtplib
    """

    channel_type = ChannelType.EMAIL

    async def send(
        self,
        event: NotificationEvent,
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> DeliveryResult:
        """Send email notification.

        Args:
            event: Notification event
            recipient: Email address
            template_vars: Template variables

        Returns:
            DeliveryResult
        """
        start_time = time.monotonic()

        try:
            # Build email message
            msg = self._build_message(event, recipient, template_vars)

            # Send via SMTP
            message_id = await self._send_smtp(msg)

            latency = (time.monotonic() - start_time) * 1000

            return DeliveryResult(
                success=True,
                channel_id=self._config.id,
                channel_type=self.channel_type,
                recipient=recipient,
                event_type=event.event_type,
                delivery_status=DeliveryStatus.SENT,
                latency_ms=latency,
                provider_message_id=message_id,
            )

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            logger.error("Email send failed to %s: %s", recipient, exc)

            return DeliveryResult(
                success=False,
                channel_id=self._config.id,
                channel_type=self.channel_type,
                recipient=recipient,
                event_type=event.event_type,
                delivery_status=DeliveryStatus.FAILED,
                error_message=str(exc),
                latency_ms=latency,
            )

    def _build_message(
        self,
        event: NotificationEvent,
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> MIMEMultipart:
        """Build email message.

        Args:
            event: Notification event
            recipient: Recipient email
            template_vars: Template variables

        Returns:
            MIMEMultipart message
        """
        msg = MIMEMultipart("alternative")

        # Headers
        from_name = self._config.smtp_from_name or "SimpleTuner"
        from_addr = self._config.smtp_from_address or self._config.smtp_username
        msg["From"] = f"{from_name} <{from_addr}>"
        msg["To"] = recipient
        msg["Subject"] = self._build_subject(event, template_vars)

        # Add response token to subject for reply tracking
        if event.response_token:
            msg["Subject"] = f"{msg['Subject']} [REF:{event.response_token[:8]}]"
            # Also add a Reply-To that includes the token
            msg["Reply-To"] = from_addr

        # Build content
        text_content = self._render_text(event, template_vars)
        html_content = self._render_html(event, template_vars)

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def _build_subject(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> str:
        """Build email subject line.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            Subject line
        """
        prefix = "[SimpleTuner]"
        title = self._format_title(event)

        if event.job_id:
            return f"{prefix} {title} - {event.job_id[:8]}"
        return f"{prefix} {title}"

    def _render_text(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> str:
        """Render plain text email content.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            Plain text content
        """
        lines = [
            self._format_title(event),
            "=" * 40,
            "",
            self._format_message(event, template_vars),
            "",
        ]

        if event.job_id:
            lines.append(f"Job ID: {event.job_id}")

        if template_vars.get("estimated_cost"):
            lines.append(f"Estimated Cost: ${template_vars['estimated_cost']:.2f}")

        if template_vars.get("config_name"):
            lines.append(f"Configuration: {template_vars['config_name']}")

        # Add approval instructions for approval events
        if event.event_type.value == "approval.required":
            lines.extend(
                [
                    "",
                    "To APPROVE, reply with: ok, yes, approved",
                    "To REJECT, reply with: no, reject, denied",
                    "",
                    "Or visit the SimpleTuner admin panel.",
                ]
            )

        lines.extend(
            [
                "",
                "-" * 40,
                "SimpleTuner Cloud Training",
            ]
        )

        return "\n".join(lines)

    def _render_html(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> str:
        """Render HTML email content.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            HTML content
        """
        severity_colors = {
            "info": "#3b82f6",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "critical": "#dc2626",
        }
        color = severity_colors.get(event.severity, "#6b7280")

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; ",
            "       line-height: 1.5; color: #1f2937; margin: 0; padding: 20px; }",
            ".container { max-width: 600px; margin: 0 auto; background: #ffffff; ",
            "             border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            f".header {{ background: {color}; color: white; padding: 20px; }}",
            ".header h1 { margin: 0; font-size: 18px; }",
            ".content { padding: 20px; }",
            ".content p { margin: 0 0 12px; }",
            ".meta { background: #f9fafb; padding: 12px; border-radius: 6px; margin: 16px 0; }",
            ".meta-item { display: flex; margin-bottom: 8px; }",
            ".meta-label { color: #6b7280; width: 120px; }",
            ".meta-value { color: #1f2937; font-weight: 500; }",
            ".actions { margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb; }",
            ".action-text { color: #6b7280; font-size: 14px; }",
            ".footer { background: #f9fafb; padding: 16px 20px; text-align: center; ",
            "          color: #6b7280; font-size: 12px; }",
            "</style>",
            "</head>",
            "<body>",
            '<div class="container">',
            f'<div class="header"><h1>{self._format_title(event)}</h1></div>',
            '<div class="content">',
            f"<p>{self._format_message(event, template_vars)}</p>",
        ]

        # Meta info
        if event.job_id or template_vars.get("estimated_cost") or template_vars.get("config_name"):
            html_parts.append('<div class="meta">')
            if event.job_id:
                html_parts.append(
                    f'<div class="meta-item"><span class="meta-label">Job ID:</span>'
                    f'<span class="meta-value">{event.job_id}</span></div>'
                )
            if template_vars.get("estimated_cost"):
                html_parts.append(
                    f'<div class="meta-item"><span class="meta-label">Est. Cost:</span>'
                    f'<span class="meta-value">${template_vars["estimated_cost"]:.2f}</span></div>'
                )
            if template_vars.get("config_name"):
                html_parts.append(
                    f'<div class="meta-item"><span class="meta-label">Config:</span>'
                    f'<span class="meta-value">{template_vars["config_name"]}</span></div>'
                )
            html_parts.append("</div>")

        # Approval instructions
        if event.event_type.value == "approval.required":
            html_parts.extend(
                [
                    '<div class="actions">',
                    '<p class="action-text">',
                    "<strong>To respond:</strong><br>",
                    "Reply with <code>ok</code>, <code>yes</code>, or <code>approved</code> to approve<br>",
                    "Reply with <code>no</code>, <code>reject</code>, or <code>denied</code> to reject",
                    "</p>",
                    "</div>",
                ]
            )

        html_parts.extend(
            [
                "</div>",
                '<div class="footer">SimpleTuner Cloud Training</div>',
                "</div>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_parts)

    async def _send_smtp(self, msg: MIMEMultipart) -> Optional[str]:
        """Send message via SMTP.

        Args:
            msg: Email message

        Returns:
            Message ID if available
        """
        try:
            import aiosmtplib
        except ImportError:
            raise ImportError("aiosmtplib is required for email notifications. " "Install with: pip install aiosmtplib")

        # Get password from secrets manager
        password = await self._get_smtp_password()

        # Connect and send
        async with aiosmtplib.SMTP(
            hostname=self._config.smtp_host,
            port=self._config.smtp_port,
            use_tls=False,  # We'll use STARTTLS
            start_tls=self._config.smtp_use_tls,
        ) as smtp:
            if self._config.smtp_username and password:
                await smtp.login(self._config.smtp_username, password)

            response = await smtp.send_message(msg)

            # Extract message ID from response if available
            return response[1] if response else None

    async def _get_smtp_password(self) -> Optional[str]:
        """Get SMTP password from secrets manager.

        Returns:
            Password or None
        """
        if not self._config.smtp_password_key:
            return None

        try:
            from ...secrets import get_secrets_manager

            secrets = get_secrets_manager()
            return secrets.get(self._config.smtp_password_key)
        except Exception as exc:
            logger.debug("Could not get SMTP password: %s", exc)
            return None

    async def validate_config(self, config: ChannelConfig) -> tuple[bool, Optional[str]]:
        """Validate email channel configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config.smtp_host:
            return False, "SMTP host is required"

        if not config.smtp_port or config.smtp_port < 1 or config.smtp_port > 65535:
            return False, "Invalid SMTP port"

        if not config.smtp_from_address:
            return False, "From address is required"

        # Validate email format
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, config.smtp_from_address):
            return False, "Invalid from email address format"

        return True, None

    async def test_connection(self, config: ChannelConfig) -> tuple[bool, Optional[str], Optional[float]]:
        """Test SMTP connection.

        Args:
            config: Configuration to test

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        try:
            import aiosmtplib
        except ImportError:
            return False, "aiosmtplib not installed", None

        start_time = time.monotonic()

        try:
            # Try to connect
            async with aiosmtplib.SMTP(
                hostname=config.smtp_host,
                port=config.smtp_port,
                use_tls=False,
                start_tls=config.smtp_use_tls,
                timeout=10.0,
            ) as smtp:
                # Try login if credentials provided
                if config.smtp_username and config.smtp_password_key:
                    password = await self._get_smtp_password()
                    if password:
                        await smtp.login(config.smtp_username, password)

            latency = (time.monotonic() - start_time) * 1000
            return True, None, latency

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            return False, str(exc), latency
