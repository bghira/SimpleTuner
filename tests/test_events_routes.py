"""
Unit tests for events API routes.

Tests cover:
- POST /api/events/callback
- Client disconnect handling
"""

from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import ClientDisconnect, Request

from simpletuner.simpletuner_sdk.server.services.callback_service import CallbackService
from simpletuner.simpletuner_sdk.server.services.event_store import EventStore
from simpletuner.simpletuner_sdk.server.services.sse_manager import SSEManager


class CallbackHandlerTestCase(unittest.TestCase):
    """Test suite for callback handler behavior."""

    def test_callback_client_disconnect_handled_gracefully(self) -> None:
        """Test that ClientDisconnect is caught and doesn't raise.

        This tests the exception handling added to prevent stack traces
        when a training process is killed mid-callback (e.g., during cancellation).
        """
        from simpletuner.simpletuner_sdk.server.routes.events import handle_callback

        # Create a mock request that raises ClientDisconnect on json()
        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=ClientDisconnect())

        # Run the async handler - should not raise
        result = asyncio.run(handle_callback(mock_request))

        # Should return gracefully with appropriate message
        self.assertEqual(result["message"], "Client disconnected")

    def test_callback_success(self) -> None:
        """Test that callback handler processes valid requests."""
        from simpletuner.simpletuner_sdk.server.routes.events import handle_callback

        # Create a mock request with valid JSON
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"type": "test", "status": "ok"})

        # Mock the callback service
        mock_service = MagicMock()
        mock_service.handle_incoming.return_value = None

        with unittest.mock.patch(
            "simpletuner.simpletuner_sdk.server.routes.events._get_callback_service",
            return_value=mock_service,
        ):
            result = asyncio.run(handle_callback(mock_request))

        self.assertEqual(result["message"], "Callback received successfully")
        mock_service.handle_incoming.assert_called_once_with({"type": "test", "status": "ok"})


class ValidationReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_validation_replays_on_new_connections_and_reconnects(self):
        from simpletuner.simpletuner_sdk.server.routes.events import events_stream

        callbacks = CallbackService(EventStore())
        before = callbacks.handle_incoming({"type": "notification", "message": "Before validation"})
        callbacks.handle_incoming({"type": "validation.image", "message": "Sampling", "images": ["preview"]})
        result = callbacks.handle_incoming({"type": "training.validation", "message": "Finished", "images": ["final"]})
        marker = callbacks.handle_incoming({"type": "notification", "message": "After validation"})
        for headers in ([], [(b"last-event-id", before.id.encode())]):
            with self.subTest(headers=headers):
                manager = SSEManager()
                request = Request(
                    {
                        "type": "http",
                        "headers": headers,
                        "app": SimpleNamespace(state=SimpleNamespace(callback_service=callbacks)),
                    }
                )
                with (
                    patch("simpletuner.simpletuner_sdk.server.routes.events.get_sse_manager", return_value=manager),
                    patch(
                        "simpletuner.simpletuner_sdk.server.routes.events.get_auth_middleware",
                        return_value=AsyncMock(return_value=SimpleNamespace(is_authenticated=False)),
                    ),
                ):
                    response = await events_stream(request, _user=None)
                    images = []
                    async with aclosing(response.body_iterator) as stream:
                        async with asyncio.timeout(2):
                            async for item in stream:
                                if item.get("event") == "callback:validation":
                                    images.append((item["id"], json.loads(item["data"])))
                                if item.get("id") == marker.id:
                                    break
                    self.assertEqual([identifier for identifier, _ in images], [result.id])
                    self.assertEqual(images[0][1]["images"], ["final"])
                    self.assertTrue(images[0][1]["is_replay"])


if __name__ == "__main__":
    unittest.main()
