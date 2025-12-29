"""Notification preferences, events, history, and status endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....services.cloud.container import get_notification_service
from ....services.cloud.notification import NotificationEventType, NotificationPreference, NotificationService
from .helpers import preference_to_response
from .models import (
    DeliveryHistoryResponse,
    DeliveryLogEntry,
    EventTypesResponse,
    NotificationStatusResponse,
    PreferenceRequest,
    PreferenceResponse,
    PreferencesListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Preferences ---


@router.get("/preferences", response_model=PreferencesListResponse)
async def list_preferences(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    svc: NotificationService = Depends(get_notification_service),
) -> PreferencesListResponse:
    """List notification preferences."""
    preferences = await svc.get_preferences(user_id=user_id)
    return PreferencesListResponse(
        preferences=[preference_to_response(p) for p in preferences],
    )


@router.post("/preferences", response_model=PreferenceResponse, status_code=status.HTTP_201_CREATED)
async def create_preference(
    request: PreferenceRequest,
    svc: NotificationService = Depends(get_notification_service),
) -> PreferenceResponse:
    """Create a notification preference."""
    try:
        event_type = NotificationEventType(request.event_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event type: {request.event_type}",
        )

    # Verify channel exists
    channel = await svc.get_channel(request.channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Channel {request.channel_id} not found",
        )

    preference = NotificationPreference(
        event_type=event_type,
        channel_id=request.channel_id,
        is_enabled=request.is_enabled,
        recipients=request.recipients,
        min_severity=request.min_severity,
    )

    pref_id = await svc.set_preference(preference)
    preference.id = pref_id

    return preference_to_response(preference)


@router.patch("/preferences/{preference_id}", response_model=PreferenceResponse)
async def update_preference(
    preference_id: int,
    request: PreferenceRequest,
    svc: NotificationService = Depends(get_notification_service),
) -> PreferenceResponse:
    """Update a notification preference."""
    try:
        event_type = NotificationEventType(request.event_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event type: {request.event_type}",
        )

    preference = NotificationPreference(
        id=preference_id,
        event_type=event_type,
        channel_id=request.channel_id,
        is_enabled=request.is_enabled,
        recipients=request.recipients,
        min_severity=request.min_severity,
    )

    await svc.set_preference(preference)
    return preference_to_response(preference)


@router.delete("/preferences/{preference_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preference(
    preference_id: int,
    svc: NotificationService = Depends(get_notification_service),
) -> None:
    """Delete a notification preference."""
    success = await svc.delete_preference(preference_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preference {preference_id} not found",
        )


# --- Event Types ---


@router.get("/events", response_model=EventTypesResponse)
async def list_event_types() -> EventTypesResponse:
    """List all available notification event types."""
    events = []

    # Group by category
    categories = {
        "Approval": NotificationEventType.approval_events(),
        "Job Lifecycle": NotificationEventType.job_events(),
        "Quota & Billing": NotificationEventType.quota_events(),
        "System Health": NotificationEventType.system_events(),
        "Authentication": NotificationEventType.auth_events(),
        "Administration": NotificationEventType.admin_events(),
    }

    for category, event_types in categories.items():
        for event_type in event_types:
            events.append(
                {
                    "id": event_type.value,
                    "name": event_type.value.replace(".", " ").replace("_", " ").title(),
                    "category": category,
                }
            )

    return EventTypesResponse(event_types=events)


# --- Delivery History ---


@router.get("/history", response_model=DeliveryHistoryResponse)
async def get_delivery_history(
    limit: int = Query(50, ge=1, le=500),
    job_id: Optional[str] = Query(None),
    channel_id: Optional[int] = Query(None),
    svc: NotificationService = Depends(get_notification_service),
) -> DeliveryHistoryResponse:
    """Get notification delivery history."""
    logs = await svc.get_delivery_history(
        limit=limit,
        job_id=job_id,
        channel_id=channel_id,
    )
    return DeliveryHistoryResponse(
        entries=[DeliveryLogEntry(**log) for log in logs],
        total=len(logs),
    )


# --- Status ---


@router.get("/status", response_model=NotificationStatusResponse)
async def get_notification_status(
    svc: NotificationService = Depends(get_notification_service),
) -> NotificationStatusResponse:
    """Get notification system status."""
    status_data = await svc.get_status()
    return NotificationStatusResponse(**status_data)


@router.post("/skip", status_code=status.HTTP_204_NO_CONTENT)
async def skip_notifications() -> None:
    """Mark notifications as configured but skipped.

    This allows the user to dismiss the setup wizard without configuring
    any notification channels.
    """
    from ....services.cloud.container import get_job_store

    store = get_job_store()
    # Mark as "configured" so the hero CTA doesn't show again
    await store.update_provider_config(
        "notifications",
        {"configured": True, "skipped": True},
    )
