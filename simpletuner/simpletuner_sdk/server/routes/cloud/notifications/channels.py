"""Notification channel management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....services.cloud.container import get_notification_service
from ....services.cloud.notification import ChannelType, NotificationService, get_preset_config, list_presets
from .helpers import channel_to_response
from .models import (
    ChannelCreateRequest,
    ChannelResponse,
    ChannelsListResponse,
    ChannelUpdateRequest,
    PresetDetailResponse,
    PresetsListResponse,
    TestConnectionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Channel Management ---


@router.get("/channels", response_model=ChannelsListResponse)
async def list_channels(
    enabled_only: bool = Query(False, description="Only return enabled channels"),
    svc: NotificationService = Depends(get_notification_service),
) -> ChannelsListResponse:
    """List all configured notification channels."""
    channels = await svc.get_channels(enabled_only=enabled_only)
    return ChannelsListResponse(
        channels=[channel_to_response(c) for c in channels],
        total=len(channels),
    )


@router.post("/channels", response_model=ChannelResponse, status_code=status.HTTP_201_CREATED)
async def create_channel(
    request: ChannelCreateRequest,
    svc: NotificationService = Depends(get_notification_service),
) -> ChannelResponse:
    """Create a new notification channel."""
    try:
        channel_type = ChannelType(request.channel_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid channel type: {request.channel_type}. "
            f"Valid types: {', '.join(t.value for t in ChannelType)}",
        )

    # Store secrets if provided
    config = request.model_dump(exclude={"smtp_password", "webhook_secret", "imap_password"})

    if request.smtp_password:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = f"notification_smtp_{request.name.lower().replace(' ', '_')}"
        secrets.set(key, request.smtp_password)
        config["smtp_password_key"] = key

    if request.webhook_secret:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = f"notification_webhook_{request.name.lower().replace(' ', '_')}"
        secrets.set(key, request.webhook_secret)
        config["webhook_secret_key"] = key

    if request.imap_password:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = f"notification_imap_{request.name.lower().replace(' ', '_')}"
        secrets.set(key, request.imap_password)
        config["imap_password_key"] = key

    channel_id = await svc.configure_channel(
        channel_type=channel_type,
        name=request.name,
        config=config,
    )

    channel = await svc.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create channel",
        )

    return channel_to_response(channel)


@router.get("/channels/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: int,
    svc: NotificationService = Depends(get_notification_service),
) -> ChannelResponse:
    """Get a specific notification channel."""
    channel = await svc.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Channel {channel_id} not found",
        )
    return channel_to_response(channel)


@router.patch("/channels/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: int,
    request: ChannelUpdateRequest,
    svc: NotificationService = Depends(get_notification_service),
) -> ChannelResponse:
    """Update a notification channel."""
    channel = await svc.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Channel {channel_id} not found",
        )

    updates = request.model_dump(exclude_unset=True)

    # Handle password updates
    if "smtp_password" in updates and updates["smtp_password"]:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = channel.smtp_password_key or f"notification_smtp_{channel_id}"
        secrets.set(key, updates.pop("smtp_password"))
        updates["smtp_password_key"] = key

    if "webhook_secret" in updates and updates["webhook_secret"]:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = channel.webhook_secret_key or f"notification_webhook_{channel_id}"
        secrets.set(key, updates.pop("webhook_secret"))
        updates["webhook_secret_key"] = key

    if "imap_password" in updates and updates["imap_password"]:
        from ....services.cloud.secrets import get_secrets_manager

        secrets = get_secrets_manager()
        key = channel.imap_password_key or f"notification_imap_{channel_id}"
        secrets.set(key, updates.pop("imap_password"))
        updates["imap_password_key"] = key

    success = await svc.update_channel(channel_id, updates)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update channel",
        )

    channel = await svc.get_channel(channel_id)
    return channel_to_response(channel)


@router.delete("/channels/{channel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_channel(
    channel_id: int,
    svc: NotificationService = Depends(get_notification_service),
) -> None:
    """Delete a notification channel."""
    channel = await svc.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Channel {channel_id} not found",
        )

    success = await svc.delete_channel(channel_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete channel",
        )


@router.post("/channels/{channel_id}/test", response_model=TestConnectionResponse)
async def test_channel(
    channel_id: int,
    svc: NotificationService = Depends(get_notification_service),
) -> TestConnectionResponse:
    """Test a channel's connectivity."""
    channel = await svc.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Channel {channel_id} not found",
        )

    success, error, latency = await svc.test_channel(channel_id)
    return TestConnectionResponse(
        success=success,
        error=error,
        latency_ms=latency,
    )


# --- Provider Presets ---


@router.get("/presets", response_model=PresetsListResponse)
async def list_email_presets() -> PresetsListResponse:
    """List available email provider presets."""
    return PresetsListResponse(presets=list_presets())


@router.get("/presets/{preset_id}", response_model=PresetDetailResponse)
async def get_email_preset(preset_id: str) -> PresetDetailResponse:
    """Get details for an email provider preset."""
    preset = get_preset_config(preset_id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{preset_id}' not found",
        )

    return PresetDetailResponse(
        id=preset_id,
        name=preset["name"],
        smtp_host=preset["smtp_host"],
        smtp_port=preset["smtp_port"],
        smtp_use_tls=preset["smtp_use_tls"],
        imap_host=preset.get("imap_host"),
        imap_port=preset.get("imap_port"),
        imap_use_ssl=preset.get("imap_use_ssl"),
        instructions=preset.get("instructions"),
        docs_url=preset.get("docs_url"),
    )
