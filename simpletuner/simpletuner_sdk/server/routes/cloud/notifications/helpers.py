"""Helper functions for notification endpoints."""

from __future__ import annotations

from ....services.cloud.notification import ChannelConfig, NotificationPreference
from .models import ChannelResponse, PreferenceResponse


def channel_to_response(channel: ChannelConfig) -> ChannelResponse:
    """Convert ChannelConfig to response model."""
    return ChannelResponse(
        id=channel.id,
        channel_type=channel.channel_type.value,
        name=channel.name,
        is_enabled=channel.is_enabled,
        smtp_host=channel.smtp_host,
        smtp_port=channel.smtp_port,
        smtp_username=channel.smtp_username,
        smtp_use_tls=channel.smtp_use_tls,
        smtp_from_address=channel.smtp_from_address,
        smtp_from_name=channel.smtp_from_name,
        webhook_url=channel.webhook_url,
        imap_enabled=channel.imap_enabled,
        imap_host=channel.imap_host,
        imap_port=channel.imap_port,
        imap_username=channel.imap_username,
        imap_use_ssl=channel.imap_use_ssl,
        imap_folder=channel.imap_folder,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
    )


def preference_to_response(pref: NotificationPreference) -> PreferenceResponse:
    """Convert NotificationPreference to response model."""
    return PreferenceResponse(
        id=pref.id,
        user_id=pref.user_id,
        event_type=pref.event_type.value,
        channel_id=pref.channel_id,
        is_enabled=pref.is_enabled,
        recipients=pref.recipients,
        min_severity=pref.min_severity,
    )
