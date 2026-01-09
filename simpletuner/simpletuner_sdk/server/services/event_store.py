"""
Event store service for managing training events and callbacks.
Provides shared storage for events in unified mode.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger("EventStore")


class EventStore:
    """Thread-safe event storage with circular buffer."""

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.lock = threading.Lock()
        self.event_counter = 0

    def add_event(self, event: Dict[str, Any]) -> int:
        """
        Add an event to the store.

        Args:
            event: Event data to store

        Returns:
            Index of the added event
        """
        with self.lock:
            # Add timestamp if not present
            if "timestamp" not in event:
                event["timestamp"] = time.time()

            # Add index
            event["_index"] = self.event_counter
            self.event_counter += 1

            self.events.append(event)
            return event["_index"]

    def get_events_since(self, since_index: int) -> List[Dict[str, Any]]:
        """
        Get all events since a given index.

        Args:
            since_index: Index to get events after

        Returns:
            List of events after the given index
        """
        # Snapshot events while holding lock, filter without lock to reduce contention
        with self.lock:
            snapshot = list(self.events)

        # Filter without holding lock
        return [event for event in snapshot if event.get("_index", -1) > since_index]

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all stored events."""
        with self.lock:
            return list(self.events)

    def clear(self):
        """Clear all stored events."""
        with self.lock:
            self.events.clear()
            # Don't reset counter to preserve ordering

    def get_latest_event(self) -> Optional[Dict[str, Any]]:
        """Get the most recent event."""
        with self.lock:
            if self.events:
                return self.events[-1]
            return None

    def get_event_count(self) -> int:
        """Get the number of stored events."""
        with self.lock:
            return len(self.events)


# Global default store for standalone mode
_default_store: Optional[EventStore] = None
_store_lock = threading.Lock()


def get_default_store() -> EventStore:
    """Get or create the default event store."""
    import os

    global _default_store
    with _store_lock:
        if _default_store is None:
            max_events = int(os.environ.get("SIMPLETUNER_EVENT_BUFFER_SIZE", "1000"))
            _default_store = EventStore(max_events=max_events)
        return _default_store


class UnifiedEventDispatcher:
    """
    Event dispatcher for unified mode that can bypass HTTP for local events.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.webhook_url = None
        self.is_local = False

    def configure(self, webhook_url: str):
        """Configure the dispatcher with webhook URL."""
        self.webhook_url = webhook_url

        # Check if webhook is local
        if webhook_url:
            local_patterns = ["localhost:", "127.0.0.1:", "0.0.0.0:"]
            self.is_local = any(pattern in webhook_url for pattern in local_patterns)

    def dispatch_event(self, event: Dict[str, Any]) -> bool:
        """
        Dispatch an event, using direct store if local, HTTP otherwise.

        Args:
            event: Event to dispatch

        Returns:
            Success status
        """
        try:
            if self.is_local:
                # Direct dispatch to event store
                self.event_store.add_event(event)
                logger.debug(f"Direct dispatch of {event.get('message_type', 'unknown')} event")
                return True
            else:
                # Use HTTP webhook
                return self._send_webhook(event)
        except Exception as e:
            logger.error(f"Failed to dispatch event: {e}")
            return False

    def _send_webhook(self, event: Dict[str, Any]) -> bool:
        """Send event via HTTP webhook."""
        if not self.webhook_url:
            return False

        try:
            import requests

            response = requests.post(self.webhook_url, json=event, timeout=5)
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False
