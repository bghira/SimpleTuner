"""Notification configuration and management API endpoints.

This package is the **API layer** for the notification system:

    routes/cloud/notifications/
    ├── __init__.py      # Router composition
    ├── models.py        # Pydantic request/response models (DTOs)
    ├── channels.py      # Channel CRUD endpoints
    ├── preferences.py   # Preferences, history, events endpoints
    └── helpers.py       # Conversion between domain and API models

The **services layer** (services/cloud/notification/) provides:
    - Domain models (dataclasses)
    - SQLite persistence (notification_store.py)
    - Business logic (notification_service.py)

Endpoints:
    - ``/channels`` - CRUD for notification channels (email, webhook, Slack)
    - ``/preferences`` - User preferences per event type
    - ``/events`` - List available notification event types
    - ``/history`` - Delivery history and logs
    - ``/presets`` - Email provider presets (Gmail, Outlook, etc.)
"""

from __future__ import annotations

from fastapi import APIRouter

from .channels import router as channels_router
from .preferences import router as preferences_router

# Create the main notifications router
router = APIRouter(prefix="/notifications", tags=["notifications"])

# Include sub-routers
router.include_router(channels_router)
router.include_router(preferences_router)

# Re-export for backwards compatibility
__all__ = ["router"]
