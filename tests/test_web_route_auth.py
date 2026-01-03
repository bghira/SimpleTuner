"""Tests for web route authentication protection.

Verifies that trainer tab routes and related endpoints require authentication.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User


class TestWebRouteAuthProtection(unittest.TestCase):
    """Test that web routes require authentication."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock authenticated user
        self.mock_user = MagicMock(spec=User)
        self.mock_user.id = 1
        self.mock_user.username = "testuser"
        self.mock_user.email = "test@example.com"
        self.mock_user.is_admin = False
        self.mock_user.has_permission = MagicMock(return_value=True)

    def _create_app_with_auth_override(self, authenticated: bool = False):
        """Create app with auth dependency overridden."""
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user

        app = create_app(mode=ServerMode.TRAINER)

        if authenticated:
            # Override to return mock user
            async def mock_get_current_user():
                return self.mock_user

            app.dependency_overrides[get_current_user] = mock_get_current_user
        else:
            # Override to raise 401
            async def mock_get_current_user_unauthorized():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            app.dependency_overrides[get_current_user] = mock_get_current_user_unauthorized

        return app

    def test_tabs_endpoint_requires_auth(self):
        """Test that /web/trainer/tabs/{tab_name} requires authentication."""
        app = self._create_app_with_auth_override(authenticated=False)

        with TestClient(app, raise_server_exceptions=False) as client:
            # Test various tab endpoints
            tabs = ["basic", "model", "training", "advanced", "datasets", "environments", "validation", "publishing"]

            for tab in tabs:
                response = client.get(f"/web/trainer/tabs/{tab}")
                self.assertEqual(
                    response.status_code, 401, f"Tab '{tab}' should require auth but got {response.status_code}"
                )

    def test_config_selector_requires_auth(self):
        """Test that /web/trainer/config-selector requires authentication."""
        app = self._create_app_with_auth_override(authenticated=False)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/web/trainer/config-selector")
            self.assertEqual(response.status_code, 401)

    def test_tab_list_requires_auth(self):
        """Test that /web/trainer/tab-list requires authentication."""
        app = self._create_app_with_auth_override(authenticated=False)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/web/trainer/tab-list")
            self.assertEqual(response.status_code, 401)

    def test_search_requires_auth(self):
        """Test that /web/trainer/search requires authentication."""
        app = self._create_app_with_auth_override(authenticated=False)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/web/trainer/search?q=test")
            self.assertEqual(response.status_code, 401)

    def test_main_trainer_page_accessible_without_auth(self):
        """Test that /web/trainer (main page) is accessible without auth.

        The main page shows the login form when not authenticated.
        """
        app = self._create_app_with_auth_override(authenticated=False)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/web/trainer")
            # Main page should be accessible (shows login form)
            self.assertIn(response.status_code, [200, 302])


class TestWebSocketAuthProtection(unittest.TestCase):
    """Test WebSocket authentication functions."""

    def test_get_current_user_ws_rejects_unauthenticated(self):
        """Test that get_current_user_ws raises exception for unauthenticated WebSocket."""
        import asyncio

        from fastapi import WebSocketException

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import AuthContext, get_current_user_ws

        # Create mock WebSocket
        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "127.0.0.1"

        async def run_test():
            # Mock get_auth_context_ws to return unauthenticated context
            unauthenticated_context = AuthContext(
                is_authenticated=False,
                client_ip="127.0.0.1",
            )

            with patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware.get_auth_context_ws",
                new=AsyncMock(return_value=unauthenticated_context),
            ):
                with self.assertRaises(WebSocketException) as ctx:
                    await get_current_user_ws(mock_websocket)

                # WebSocket 1008 = Policy Violation (used for auth failures)
                self.assertEqual(ctx.exception.code, 1008)

        asyncio.run(run_test())

    def test_get_current_user_ws_accepts_authenticated_session(self):
        """Test that get_current_user_ws returns user for valid session."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import AuthContext, get_current_user_ws
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

        # Create mock authenticated user
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.username = "testuser"

        # Create mock WebSocket
        mock_websocket = MagicMock()

        async def run_test():
            # Mock get_auth_context_ws to return authenticated context
            authenticated_context = AuthContext(
                user=mock_user,
                session_id="valid-session-token",
                is_authenticated=True,
                client_ip="127.0.0.1",
            )

            with patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware.get_auth_context_ws",
                new=AsyncMock(return_value=authenticated_context),
            ):
                result = await get_current_user_ws(mock_websocket)
                self.assertEqual(result.id, mock_user.id)
                self.assertEqual(result.username, mock_user.username)

        asyncio.run(run_test())

    def test_get_current_user_ws_accepts_api_key(self):
        """Test that get_current_user_ws returns user for valid API key."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import AuthContext, get_current_user_ws
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import APIKey, User

        # Create mock authenticated user and API key
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.username = "apiuser"

        mock_api_key = MagicMock(spec=APIKey)
        mock_api_key.key_prefix = "st_test"
        mock_api_key.scoped_permissions = None

        # Create mock WebSocket
        mock_websocket = MagicMock()

        async def run_test():
            # Mock get_auth_context_ws to return authenticated context via API key
            authenticated_context = AuthContext(
                user=mock_user,
                api_key=mock_api_key,
                is_authenticated=True,
                client_ip="127.0.0.1",
            )

            with patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware.get_auth_context_ws",
                new=AsyncMock(return_value=authenticated_context),
            ):
                result = await get_current_user_ws(mock_websocket)
                self.assertEqual(result.id, mock_user.id)
                self.assertEqual(result.username, mock_user.username)

        asyncio.run(run_test())


class TestAuthMiddlewareFunctions(unittest.TestCase):
    """Test auth middleware helper functions."""

    def test_get_auth_context_ws_returns_unauthenticated_context(self):
        """Test get_auth_context_ws returns unauthenticated context when no credentials."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import (
            get_auth_context_ws,
            get_auth_middleware,
        )

        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "127.0.0.1"
        mock_websocket.url = MagicMock()
        mock_websocket.url.path = "/ws/test"
        mock_websocket.headers = MagicMock()
        mock_websocket.headers.get = MagicMock(return_value=None)
        mock_websocket.cookies = MagicMock()
        mock_websocket.cookies.get = MagicMock(return_value=None)

        async def run_test():
            middleware = get_auth_middleware()
            middleware._single_user_checked = True
            middleware._single_user_mode = False
            middleware._local_admin = None

            # Create a mock store
            mock_store = MagicMock()
            mock_store.authenticate_api_key = AsyncMock(return_value=None)
            mock_store.get_session_user = AsyncMock(return_value=None)

            # Patch _store directly (the private attribute)
            middleware._store = mock_store

            context = await get_auth_context_ws(mock_websocket)

            self.assertFalse(context.is_authenticated)
            self.assertIsNone(context.user)
            self.assertEqual(context.client_ip, "127.0.0.1")

        asyncio.run(run_test())

    def test_get_auth_context_ws_returns_authenticated_context(self):
        """Test get_auth_context_ws returns authenticated context with valid session."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import (
            get_auth_context_ws,
            get_auth_middleware,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

        mock_user = MagicMock(spec=User)
        mock_user.id = 42
        mock_user.username = "wsuser"

        mock_websocket = MagicMock()
        mock_websocket.client = MagicMock()
        mock_websocket.client.host = "192.168.1.100"
        mock_websocket.url = MagicMock()
        mock_websocket.url.path = "/ws/test"
        mock_websocket.headers = MagicMock()
        mock_websocket.headers.get = MagicMock(return_value=None)
        mock_websocket.cookies = MagicMock()
        mock_websocket.cookies.get = MagicMock(return_value="session-123")

        async def run_test():
            middleware = get_auth_middleware()
            middleware._single_user_checked = True
            middleware._single_user_mode = False

            # Create a mock store
            mock_store = MagicMock()
            mock_store.authenticate_api_key = AsyncMock(return_value=None)
            mock_store.get_session_user = AsyncMock(return_value=mock_user)

            # Patch _store directly (the private attribute)
            middleware._store = mock_store

            context = await get_auth_context_ws(mock_websocket)

            self.assertTrue(context.is_authenticated)
            self.assertEqual(context.user.id, 42)
            self.assertEqual(context.user.username, "wsuser")
            self.assertEqual(context.session_id, "session-123")
            self.assertEqual(context.client_ip, "192.168.1.100")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
