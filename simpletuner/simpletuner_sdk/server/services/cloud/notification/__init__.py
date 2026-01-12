"""Notification system for SimpleTuner cloud administration.

This package provides a pluggable notification system supporting multiple
channels (email, webhook, Slack) with mobile approval workflows via IMAP.

Architecture
------------
This is the **services layer** (business logic + persistence):

    services/cloud/notification/
    ├── __init__.py           # Package exports and singleton accessors
    ├── models.py             # Domain models (dataclasses): NotificationEvent,
    │                         # ChannelConfig, NotificationPreference, etc.
    ├── protocols.py          # Interfaces and enums: ChannelType, DeliveryStatus
    ├── notification_store.py # SQLite persistence for channels, preferences, logs
    ├── notification_service.py # Service orchestration
    ├── notification_router.py  # Event routing logic
    ├── channels/             # Channel implementations (email, webhook, slack)
    └── response_handlers/    # IMAP handler for email reply processing

The **API layer** (routes/cloud/notifications/) provides:
    - Pydantic request/response models (DTOs)
    - FastAPI routers for REST endpoints

The domain models here (dataclasses) are internal. The API models (Pydantic)
handle validation and serialization at the HTTP boundary.

Usage
-----
From FastAPI routes (via dependency injection)::

    from fastapi import Depends
    from .notification import get_notification_service

    @router.post("/notify")
    async def send_notification(
        svc: NotificationService = Depends(get_notification_service),
    ):
        await svc.notify(NotificationEventType.JOB_COMPLETED, {...})

From anywhere else (module-level singleton)::

    from .notification import get_notifier

    notifier = get_notifier()
    await notifier.notify(NotificationEventType.APPROVAL_REQUIRED, {...})
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from .models import (
    EMAIL_PROVIDER_PRESETS,
    ChannelConfig,
    DeliveryResult,
    NotificationEvent,
    NotificationLog,
    NotificationPreference,
    PendingResponse,
    ResponseAction,
    get_preset_config,
    list_presets,
)
from .protocols import (
    ChannelType,
    DeliveryStatus,
    NotificationChannelProtocol,
    NotificationEventType,
    NotificationServiceProtocol,
    ResponseHandlerProtocol,
    Severity,
)

if TYPE_CHECKING:
    from .notification_service import NotificationService
else:
    # Allow runtime import for container.py
    from .notification_service import NotificationService

__all__ = [
    # Protocols
    "NotificationChannelProtocol",
    "NotificationServiceProtocol",
    "ResponseHandlerProtocol",
    # Enums
    "NotificationEventType",
    "ChannelType",
    "DeliveryStatus",
    "Severity",
    # Models
    "NotificationEvent",
    "ChannelConfig",
    "NotificationPreference",
    "DeliveryResult",
    "ResponseAction",
    "PendingResponse",
    "NotificationLog",
    # Presets
    "EMAIL_PROVIDER_PRESETS",
    "get_preset_config",
    "list_presets",
    # Service access
    "get_notifier",
    "get_notification_service",
    "NotificationService",
]

# Module-level singleton for non-FastAPI access
_notifier: Optional["NotificationService"] = None
_notifier_lock = threading.Lock()


def get_notifier() -> "NotificationService":
    """Get notification service for non-FastAPI code paths.

    This provides a module-level singleton that can be used anywhere
    in the codebase without requiring FastAPI dependency injection.

    Returns:
        NotificationService instance

    Example:
        from simpletuner.simpletuner_sdk.server.services.cloud.notification import (
            get_notifier,
            NotificationEventType,
        )

        notifier = get_notifier()
        await notifier.notify(
            NotificationEventType.JOB_COMPLETED,
            {"job_id": "123", "message": "Training complete"},
        )
    """
    global _notifier
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                from .notification_service import NotificationService
                from .notification_store import NotificationStore

                store = NotificationStore()
                _notifier = NotificationService(store)
    return _notifier


def get_notification_service() -> "NotificationService":
    """FastAPI dependency that provides NotificationService.

    This is registered in the ServiceContainer and used via Depends().

    Returns:
        NotificationService instance
    """
    # For now, delegate to get_notifier() for simplicity
    # Once container registration is complete, this will use container.get()
    return get_notifier()


def reset_notifier() -> None:
    """Reset the notification service singleton (for testing)."""
    global _notifier
    with _notifier_lock:
        if _notifier is not None:
            # Stop any running handlers
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_notifier.shutdown())
                else:
                    loop.run_until_complete(_notifier.shutdown())
            except Exception:
                pass
            _notifier = None
