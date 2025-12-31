"""Data models for the notification system.

This module defines the core data structures used throughout the notification
system, including events, channel configurations, preferences, and results.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .protocols import ChannelType, DeliveryStatus, NotificationEventType, Severity


@dataclass
class NotificationEvent:
    """A notification event to be delivered.

    This is the core payload that gets routed through channels.
    """

    event_type: NotificationEventType
    title: str = ""
    message: str = ""
    severity: str = Severity.INFO.value
    job_id: Optional[str] = None
    user_id: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # For approval response tracking
    approval_request_id: Optional[int] = None
    response_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "approval_request_id": self.approval_request_id,
        }

    @classmethod
    def from_context(
        cls,
        event_type: NotificationEventType,
        context: Dict[str, Any],
    ) -> "NotificationEvent":
        """Create an event from a context dictionary."""
        return cls(
            event_type=event_type,
            title=context.get("title", ""),
            message=context.get("message", ""),
            severity=context.get("severity", Severity.INFO.value),
            job_id=context.get("job_id"),
            user_id=context.get("user_id"),
            data=context.get("data", {}),
            approval_request_id=context.get("approval_request_id"),
        )

    def with_response_token(self) -> "NotificationEvent":
        """Return a copy with a generated response token."""
        token = secrets.token_urlsafe(32)
        return NotificationEvent(
            event_type=self.event_type,
            title=self.title,
            message=self.message,
            severity=self.severity,
            job_id=self.job_id,
            user_id=self.user_id,
            data=self.data,
            created_at=self.created_at,
            approval_request_id=self.approval_request_id,
            response_token=token,
        )


@dataclass
class ChannelConfig:
    """Configuration for a notification channel.

    Stores all settings needed to send notifications via a specific channel.
    """

    id: int = 0
    channel_type: ChannelType = ChannelType.EMAIL
    name: str = ""
    is_enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    # SMTP settings (for email channel)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password_key: Optional[str] = None  # Reference to secrets manager
    smtp_use_tls: bool = True
    smtp_from_address: Optional[str] = None
    smtp_from_name: Optional[str] = None

    # Webhook settings (for webhook/slack channels)
    webhook_url: Optional[str] = None
    webhook_secret_key: Optional[str] = None  # Reference to secrets manager

    # IMAP settings (for email response handling)
    imap_enabled: bool = False
    imap_host: Optional[str] = None
    imap_port: int = 993
    imap_username: Optional[str] = None
    imap_password_key: Optional[str] = None  # Reference to secrets manager
    imap_use_ssl: bool = True
    imap_folder: str = "INBOX"

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_secrets: If True, include secret key references
        """
        result = {
            "id": self.id,
            "channel_type": self.channel_type.value,
            "name": self.name,
            "is_enabled": self.is_enabled,
            "config": self.config,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_username": self.smtp_username,
            "smtp_use_tls": self.smtp_use_tls,
            "smtp_from_address": self.smtp_from_address,
            "smtp_from_name": self.smtp_from_name,
            "webhook_url": self.webhook_url,
            "imap_enabled": self.imap_enabled,
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
            "imap_username": self.imap_username,
            "imap_use_ssl": self.imap_use_ssl,
            "imap_folder": self.imap_folder,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_secrets:
            result["smtp_password_key"] = self.smtp_password_key
            result["webhook_secret_key"] = self.webhook_secret_key
            result["imap_password_key"] = self.imap_password_key
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelConfig":
        """Create from dictionary."""
        channel_type = data.get("channel_type", "email")
        if isinstance(channel_type, str):
            channel_type = ChannelType(channel_type)

        return cls(
            id=data.get("id", 0),
            channel_type=channel_type,
            name=data.get("name", ""),
            is_enabled=data.get("is_enabled", True),
            config=data.get("config", {}),
            smtp_host=data.get("smtp_host"),
            smtp_port=data.get("smtp_port", 587),
            smtp_username=data.get("smtp_username"),
            smtp_password_key=data.get("smtp_password_key"),
            smtp_use_tls=data.get("smtp_use_tls", True),
            smtp_from_address=data.get("smtp_from_address"),
            smtp_from_name=data.get("smtp_from_name"),
            webhook_url=data.get("webhook_url"),
            webhook_secret_key=data.get("webhook_secret_key"),
            imap_enabled=data.get("imap_enabled", False),
            imap_host=data.get("imap_host"),
            imap_port=data.get("imap_port", 993),
            imap_username=data.get("imap_username"),
            imap_password_key=data.get("imap_password_key"),
            imap_use_ssl=data.get("imap_use_ssl", True),
            imap_folder=data.get("imap_folder", "INBOX"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class NotificationPreference:
    """User/admin preferences for notification events.

    Controls which events get routed to which channels.
    """

    id: int = 0
    user_id: Optional[int] = None  # None = applies to all admins
    event_type: NotificationEventType = NotificationEventType.APPROVAL_REQUIRED
    channel_id: int = 0
    is_enabled: bool = True
    recipients: List[str] = field(default_factory=list)
    min_severity: str = Severity.INFO.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "channel_id": self.channel_id,
            "is_enabled": self.is_enabled,
            "recipients": self.recipients,
            "min_severity": self.min_severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationPreference":
        """Create from dictionary."""
        event_type = data.get("event_type", "approval.required")
        if isinstance(event_type, str):
            event_type = NotificationEventType(event_type)

        recipients = data.get("recipients", [])
        if isinstance(recipients, str):
            import json

            recipients = json.loads(recipients)

        return cls(
            id=data.get("id", 0),
            user_id=data.get("user_id"),
            event_type=event_type,
            channel_id=data.get("channel_id", 0),
            is_enabled=data.get("is_enabled", True),
            recipients=recipients,
            min_severity=data.get("min_severity", Severity.INFO.value),
        )


@dataclass
class DeliveryResult:
    """Result of a notification delivery attempt."""

    success: bool
    channel_id: int
    channel_type: ChannelType
    recipient: str
    event_type: NotificationEventType
    delivery_status: DeliveryStatus
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    provider_message_id: Optional[str] = None
    sent_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type.value,
            "recipient": self.recipient,
            "event_type": self.event_type.value,
            "delivery_status": self.delivery_status.value,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "provider_message_id": self.provider_message_id,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
        }


@dataclass
class ResponseAction:
    """Parsed action from an inbound response (e.g., email reply)."""

    action: str  # "approve", "reject", "unknown"
    approval_request_id: Optional[int] = None
    sender_email: str = ""
    raw_body: str = ""
    confidence: float = 1.0
    suggested_reply: Optional[str] = None  # Message for "unknown" responses

    def is_approval(self) -> bool:
        """Check if this is an approval action."""
        return self.action == "approve"

    def is_rejection(self) -> bool:
        """Check if this is a rejection action."""
        return self.action == "reject"

    def is_unknown(self) -> bool:
        """Check if the response was not understood."""
        return self.action == "unknown"


@dataclass
class PendingResponse:
    """Tracks a pending response for email reply correlation."""

    id: int = 0
    response_token: str = ""
    approval_request_id: int = 0
    channel_id: int = 0
    authorized_senders: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "response_token": self.response_token,
            "approval_request_id": self.approval_request_id,
            "channel_id": self.channel_id,
            "authorized_senders": self.authorized_senders,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
        }


@dataclass
class NotificationLog:
    """Record of a notification delivery attempt."""

    id: int = 0
    channel_id: int = 0
    event_type: str = ""
    job_id: Optional[str] = None
    user_id: Optional[int] = None
    recipient: str = ""
    delivery_status: str = DeliveryStatus.PENDING.value
    error_message: Optional[str] = None
    sent_at: Optional[str] = None
    delivered_at: Optional[str] = None
    provider_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "event_type": self.event_type,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "recipient": self.recipient,
            "delivery_status": self.delivery_status,
            "error_message": self.error_message,
            "sent_at": self.sent_at,
            "delivered_at": self.delivered_at,
            "provider_message_id": self.provider_message_id,
        }


# Email provider presets for easy configuration
EMAIL_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "gmail": {
        "name": "Gmail",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "instructions": (
            "Use an App Password if 2FA is enabled. "
            "Enable IMAP in Gmail settings (Settings > See all settings > "
            "Forwarding and POP/IMAP > Enable IMAP)."
        ),
        "docs_url": "https://support.google.com/accounts/answer/185833",
    },
    "outlook": {
        "name": "Outlook / Microsoft 365",
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "instructions": (
            "Use your Microsoft account password or an app password. " "IMAP must be enabled in Outlook settings."
        ),
        "docs_url": "https://support.microsoft.com/en-us/office/pop-imap-and-smtp-settings-8361e398-8af4-4e97-b147-6c6c4ac95353",
    },
    "yahoo": {
        "name": "Yahoo Mail",
        "smtp_host": "smtp.mail.yahoo.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "imap_host": "imap.mail.yahoo.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "instructions": (
            "Generate an App Password in Yahoo Account Security settings. "
            "Enable 'Allow apps that use less secure sign-in'."
        ),
        "docs_url": "https://help.yahoo.com/kb/generate-manage-third-party-app-passwords-sln15241.html",
    },
    "fastmail": {
        "name": "Fastmail",
        "smtp_host": "smtp.fastmail.com",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "imap_host": "imap.fastmail.com",
        "imap_port": 993,
        "imap_use_ssl": True,
        "instructions": "Use an App Password from Fastmail settings.",
        "docs_url": "https://www.fastmail.help/hc/en-us/articles/360058752854",
    },
    "protonmail": {
        "name": "ProtonMail Bridge",
        "smtp_host": "127.0.0.1",
        "smtp_port": 1025,
        "smtp_use_tls": False,
        "imap_host": "127.0.0.1",
        "imap_port": 1143,
        "imap_use_ssl": False,
        "instructions": (
            "Requires ProtonMail Bridge running locally. " "Use the Bridge password, not your account password."
        ),
        "docs_url": "https://proton.me/support/protonmail-bridge-install",
    },
    "custom": {
        "name": "Custom SMTP Server",
        "smtp_host": "",
        "smtp_port": 587,
        "smtp_use_tls": True,
        "imap_host": "",
        "imap_port": 993,
        "imap_use_ssl": True,
        "instructions": "Configure your own SMTP and IMAP server details.",
        "docs_url": None,
    },
}


def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """Get email provider preset configuration.

    Args:
        preset_name: Name of the preset (gmail, outlook, custom, etc.)

    Returns:
        Preset configuration dict, or None if not found
    """
    return EMAIL_PROVIDER_PRESETS.get(preset_name.lower())


def list_presets() -> List[Dict[str, Any]]:
    """List all available email provider presets.

    Returns:
        List of preset info dicts with id and name
    """
    return [
        {"id": key, "name": preset["name"], "has_docs": preset.get("docs_url") is not None}
        for key, preset in EMAIL_PROVIDER_PRESETS.items()
    ]
