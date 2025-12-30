"""Tests for authentication routes.

Tests cover:
- First-admin creation endpoint race condition
- Login and logout functionality
- Session management
- API key operations
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore


class TestFirstAdminRaceCondition(unittest.IsolatedAsyncioTestCase):
    """Test cases for first-admin endpoint TOCTOU race condition fix."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.db_path = Path(self.temp_dir) / "test_users.db"

        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.base import BaseAuthStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        UserStore.reset_instance()
        BaseAuthStore._instances.clear()
        BaseSQLiteStore._instances.clear()

        # Patch the _get_store function in auth routes to use our temp db
        self._store_patcher = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.auth._get_store",
            lambda: UserStore(self.db_path),
        )
        self._store_patcher.start()

        # Initialize UserStore with temp path to set up the singleton
        UserStore(self.db_path)

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        self._store_patcher.stop()

        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.base import BaseAuthStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        UserStore.reset_instance()
        BaseAuthStore._instances.clear()
        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_concurrent_first_admin_creation_prevented(self) -> None:
        """Test that concurrent first-admin requests only create one user.

        This test verifies the fix for the TOCTOU race condition where
        two simultaneous requests could both pass the has_any_users() check
        before either creates a user, resulting in two admin users.
        """
        from simpletuner.simpletuner_sdk.server.routes.cloud.auth import FirstRunSetupRequest, create_first_admin
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore

        # Mock request and response objects
        mock_request = MagicMock()
        mock_request.url.scheme = "https"
        mock_request.headers.get = MagicMock(return_value="test-agent")
        mock_request.cookies.get = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.set_cookie = MagicMock()

        # Track the number of successful creations and conflicts
        successful_creations = []
        conflicts = []

        async def attempt_create_admin(attempt_id: int):
            """Attempt to create the first admin user."""
            try:
                # Each attempt uses slightly different data
                data = FirstRunSetupRequest(
                    email=f"admin{attempt_id}@example.com",
                    username=f"admin{attempt_id}",
                    password="test-password-123",
                    display_name=f"Admin {attempt_id}",
                )

                # Attempt to create admin
                result = await create_first_admin(mock_request, mock_response, data)
                successful_creations.append((attempt_id, result))
            except Exception as exc:
                if "409" in str(exc) or "Setup already completed" in str(exc):
                    conflicts.append((attempt_id, str(exc)))
                else:
                    raise

        # Launch multiple concurrent requests
        num_attempts = 10
        await asyncio.gather(*[attempt_create_admin(i) for i in range(num_attempts)])

        # Verify only ONE admin was created successfully
        self.assertEqual(
            len(successful_creations),
            1,
            f"Expected exactly 1 successful creation, got {len(successful_creations)}",
        )

        # Verify the rest received 409 conflict errors
        self.assertEqual(
            len(conflicts),
            num_attempts - 1,
            f"Expected {num_attempts - 1} conflicts, got {len(conflicts)}",
        )

        # Verify that exactly one user exists in the database
        store = UserStore()
        user_count = await store.get_user_count()
        self.assertEqual(
            user_count,
            1,
            f"Expected exactly 1 user in database, got {user_count}",
        )

    async def test_first_admin_after_users_exist(self) -> None:
        """Test that first-admin endpoint rejects when users already exist."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.auth import FirstRunSetupRequest, create_first_admin
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore

        store = UserStore()

        # Create an existing user
        await store.create_user(
            email="existing@example.com",
            username="existing",
            password="password123",
            is_admin=True,
            level_names=["admin"],
        )

        # Mock request and response
        mock_request = MagicMock()
        mock_request.url.scheme = "https"
        mock_request.headers.get = MagicMock(return_value="test-agent")

        mock_response = MagicMock()

        # Attempt to create first admin
        data = FirstRunSetupRequest(
            email="newadmin@example.com",
            username="newadmin",
            password="test-password",
        )

        # Should raise 409 conflict
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await create_first_admin(mock_request, mock_response, data)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("Setup already completed", ctx.exception.detail)

    async def test_first_admin_success(self) -> None:
        """Test successful first admin creation when no users exist."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.auth import FirstRunSetupRequest, create_first_admin
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore

        # Mock request and response
        mock_request = MagicMock()
        mock_request.url.scheme = "https"
        mock_request.headers.get = MagicMock(return_value="test-agent")
        mock_request.cookies.get = MagicMock(return_value=None)

        mock_response = MagicMock()
        mock_response.set_cookie = MagicMock()

        # Create first admin
        data = FirstRunSetupRequest(
            email="admin@example.com",
            username="admin",
            password="secure-password-123",
            display_name="System Administrator",
        )

        result = await create_first_admin(mock_request, mock_response, data)

        # Verify response
        self.assertTrue(result.success)
        self.assertIsNotNone(result.user)
        self.assertEqual(result.user["username"], "admin")
        self.assertEqual(result.user["email"], "admin@example.com")
        self.assertTrue(result.user["is_admin"])

        # Verify session cookie was set
        mock_response.set_cookie.assert_called_once()

        # Verify user in database
        store = UserStore()
        user_count = await store.get_user_count()
        self.assertEqual(user_count, 1)

        user = await store.get_user(result.user["id"])
        self.assertEqual(user.username, "admin")
        self.assertTrue(user.is_admin)


if __name__ == "__main__":
    unittest.main()
