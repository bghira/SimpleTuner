"""
Unit tests for events API routes.

Tests cover:
- POST /api/events/callback
- Client disconnect handling
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from starlette.requests import ClientDisconnect


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


if __name__ == "__main__":
    unittest.main()
