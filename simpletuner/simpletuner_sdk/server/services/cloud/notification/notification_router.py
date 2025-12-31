"""Route notification events to appropriate channels based on preferences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .models import ChannelConfig, NotificationEvent, NotificationPreference
from .protocols import NotificationEventType, Severity

if TYPE_CHECKING:
    from .notification_store import NotificationStore

logger = logging.getLogger(__name__)


class NotificationRouter:
    """Determine which channels receive which events.

    Routes events to configured channels based on:
    - Event type preferences
    - Severity thresholds
    - Channel availability
    - Recipient expansion
    """

    def __init__(self, store: "NotificationStore"):
        """Initialize the router.

        Args:
            store: NotificationStore for preference lookups
        """
        self._store = store

    async def get_routes(
        self,
        event: NotificationEvent,
    ) -> List[Tuple[ChannelConfig, List[str]]]:
        """Get channels and recipients for an event.

        Args:
            event: The notification event to route

        Returns:
            List of (channel_config, recipients) tuples
        """
        # Get preferences for this event type
        preferences = await self._store.get_preferences_for_event(event.event_type)

        if not preferences:
            logger.debug("No preferences found for event type: %s", event.event_type.value)
            return []

        routes: List[Tuple[ChannelConfig, List[str]]] = []

        for pref in preferences:
            # Check if preference is enabled
            if not pref.is_enabled:
                continue

            # Check severity threshold
            if not Severity.meets_threshold(event.severity, pref.min_severity):
                logger.debug(
                    "Event severity %s below threshold %s for preference %d",
                    event.severity,
                    pref.min_severity,
                    pref.id,
                )
                continue

            # Get channel configuration
            channel = await self._store.get_channel(pref.channel_id)
            if not channel or not channel.is_enabled:
                logger.debug(
                    "Channel %d not available for preference %d",
                    pref.channel_id,
                    pref.id,
                )
                continue

            # Resolve recipients
            recipients = await self._resolve_recipients(pref, event)
            if not recipients:
                logger.debug("No recipients resolved for preference %d", pref.id)
                continue

            routes.append((channel, recipients))
            logger.debug(
                "Routed event %s to channel %s with %d recipients",
                event.event_type.value,
                channel.name,
                len(recipients),
            )

        return routes

    async def _resolve_recipients(
        self,
        preference: NotificationPreference,
        event: NotificationEvent,
    ) -> List[str]:
        """Resolve recipients for a preference.

        Priority:
        1. Explicit recipients in preference
        2. Event context (e.g., user_id -> user email)
        3. Default admin recipients

        Args:
            preference: Notification preference
            event: Notification event

        Returns:
            List of recipient addresses
        """
        # Use explicit recipients if configured
        if preference.recipients:
            return preference.recipients

        # Try to resolve from event context
        recipients = []

        # If event has user_id, try to get user's email
        if event.user_id is not None:
            try:
                from ..auth import UserStore

                user_store = UserStore()
                user = await user_store.get_user_by_id(event.user_id)
                if user and user.email:
                    recipients.append(user.email)
            except Exception as exc:
                logger.debug("Could not resolve user email: %s", exc)

        # If event has approvers in data, use those
        if "approvers" in event.data:
            approvers = event.data["approvers"]
            if isinstance(approvers, list):
                recipients.extend(approvers)

        return recipients

    async def get_default_routes_for_event(self, event_type: NotificationEventType) -> List[ChannelConfig]:
        """Get default channels for an event type (for quick access).

        Args:
            event_type: Event type

        Returns:
            List of configured channels for this event
        """
        preferences = await self._store.get_preferences_for_event(event_type)

        channels = []
        seen_ids = set()

        for pref in preferences:
            if pref.channel_id in seen_ids:
                continue
            channel = await self._store.get_channel(pref.channel_id)
            if channel and channel.is_enabled:
                channels.append(channel)
                seen_ids.add(pref.channel_id)

        return channels

    async def has_any_preferences(self) -> bool:
        """Check if any notification preferences are configured.

        Returns:
            True if at least one preference exists
        """
        # Check each event type category
        for event_type in NotificationEventType:
            prefs = await self._store.get_preferences_for_event(event_type)
            if prefs:
                return True
        return False

    async def get_event_coverage(self) -> Dict[str, bool]:
        """Get which event types have preferences configured.

        Returns:
            Dict mapping event type values to whether they have preferences
        """
        coverage = {}
        for event_type in NotificationEventType:
            prefs = await self._store.get_preferences_for_event(event_type)
            coverage[event_type.value] = len(prefs) > 0
        return coverage
