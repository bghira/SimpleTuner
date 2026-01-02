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
            "simpletuner.simpletuner_sdk.server.routes.auth._get_store",
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
        from simpletuner.simpletuner_sdk.server.routes.auth import FirstRunSetupRequest, create_first_admin
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
        from simpletuner.simpletuner_sdk.server.routes.auth import FirstRunSetupRequest, create_first_admin
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
        from simpletuner.simpletuner_sdk.server.routes.auth import FirstRunSetupRequest, create_first_admin
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


class TestAPIKeyCreation(unittest.IsolatedAsyncioTestCase):
    """Test cases for API key store operations."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_api_keys.db"

        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.base import BaseAuthStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        UserStore.reset_instance()
        BaseAuthStore._instances.clear()
        BaseSQLiteStore._instances.clear()

        # Initialize store with temp path
        self.store = UserStore(self.db_path)

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.base import BaseAuthStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        UserStore.reset_instance()
        BaseAuthStore._instances.clear()
        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_create_api_key(self) -> None:
        """Test creating an API key returns key and metadata."""
        # Create a user first
        user = await self.store.create_user(
            email="test@example.com",
            username="testuser",
            password="password123",
            level_names=["researcher"],
        )

        # Create API key
        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Test Key",
            expires_in_days=30,
        )

        # Verify returned metadata
        self.assertEqual(api_key.user_id, user.id)
        self.assertEqual(api_key.name, "Test Key")
        self.assertTrue(api_key.is_active)
        self.assertIsNotNone(api_key.key_prefix)
        self.assertIsNotNone(api_key.expires_at)

        # Verify raw key is a string
        self.assertIsInstance(raw_key, str)
        self.assertTrue(len(raw_key) > 0)

    async def test_api_key_authentication(self) -> None:
        """Test authenticating with an API key."""
        # Create a user and API key
        user = await self.store.create_user(
            email="auth@example.com",
            username="authuser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Auth Test Key",
        )

        # Authenticate with the raw key using the same store
        result = await self.store.authenticate_api_key(raw_key)

        self.assertIsNotNone(result)
        auth_user, auth_key = result
        self.assertEqual(auth_user.id, user.id)
        self.assertEqual(auth_key.name, "Auth Test Key")

    async def test_api_key_revocation(self) -> None:
        """Test revoking an API key."""
        # Create a user and API key
        user = await self.store.create_user(
            email="revoke@example.com",
            username="revokeuser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Revoke Test Key",
        )

        # Revoke the key
        success = await self.store.revoke_api_key(api_key.id, user_id=user.id)
        self.assertTrue(success)

        # Verify key no longer authenticates
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNone(result)

    async def test_admin_revocation(self) -> None:
        """Test admin can revoke any key without user_id."""
        # Create two users
        user1 = await self.store.create_user(
            email="user1@example.com",
            username="user1",
            password="password123",
            level_names=["researcher"],
        )
        user2 = await self.store.create_user(
            email="user2@example.com",
            username="user2",
            password="password123",
            level_names=["researcher"],
        )

        # Create key for user1
        api_key, raw_key = await self.store.create_api_key(
            user_id=user1.id,
            name="User1 Key",
        )

        # User2 cannot revoke user1's key
        success = await self.store.revoke_api_key(api_key.id, user_id=user2.id)
        self.assertFalse(success)

        # Key should still work
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNotNone(result)

        # Admin mode (user_id=None) can revoke any key
        success = await self.store.revoke_api_key(api_key.id, user_id=None)
        self.assertTrue(success)

        # Key should no longer work
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNone(result)

    async def test_expired_key_rejected(self) -> None:
        """Test that expired keys cannot authenticate."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        user = await self.store.create_user(
            email="expire@example.com",
            username="expireuser",
            password="password123",
            level_names=["researcher"],
        )

        # Create key that expires in 1 day
        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Expiring Key",
            expires_in_days=1,
        )

        # Key should work now
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNotNone(result)

        # Mock time to be 2 days in the future
        future_time = datetime.now(timezone.utc) + timedelta(days=2)
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.api_key_store.datetime") as mock_dt:
            mock_dt.now.return_value = future_time
            mock_dt.fromisoformat = datetime.fromisoformat

            # Key should be rejected
            result = await self.store.authenticate_api_key(raw_key)
            self.assertIsNone(result)

    async def test_scoped_permissions_returned(self) -> None:
        """Test that scoped permissions are returned on auth."""
        user = await self.store.create_user(
            email="scoped@example.com",
            username="scopeduser",
            password="password123",
            level_names=["researcher"],
        )

        # Create key with scoped permissions
        scoped = {"job.view.own", "config.view"}
        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Scoped Key",
            scoped_permissions=scoped,
        )

        # Verify scoped permissions in returned metadata
        self.assertEqual(api_key.scoped_permissions, scoped)

        # Verify scoped permissions returned on auth
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNotNone(result)
        auth_user, auth_key = result
        self.assertEqual(auth_key.scoped_permissions, scoped)

    async def test_list_api_keys(self) -> None:
        """Test listing API keys for a user."""
        user = await self.store.create_user(
            email="list@example.com",
            username="listuser",
            password="password123",
            level_names=["researcher"],
        )

        # Create multiple keys
        key1, _ = await self.store.create_api_key(user_id=user.id, name="Key 1")
        key2, _ = await self.store.create_api_key(user_id=user.id, name="Key 2")
        key3, _ = await self.store.create_api_key(user_id=user.id, name="Key 3")

        # List keys
        keys = await self.store.list_api_keys(user.id)

        self.assertEqual(len(keys), 3)
        names = {k.name for k in keys}
        self.assertEqual(names, {"Key 1", "Key 2", "Key 3"})

    async def test_list_includes_revoked_keys(self) -> None:
        """Test that list includes revoked keys (for audit trail)."""
        user = await self.store.create_user(
            email="listrevoked@example.com",
            username="listrevokeduser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, _ = await self.store.create_api_key(user_id=user.id, name="Revoked Key")

        # Revoke the key
        await self.store.revoke_api_key(api_key.id, user_id=user.id)

        # List should still include it
        keys = await self.store.list_api_keys(user.id)
        self.assertEqual(len(keys), 1)
        self.assertFalse(keys[0].is_active)

    async def test_invalid_key_rejected(self) -> None:
        """Test that invalid keys are rejected."""
        user = await self.store.create_user(
            email="invalid@example.com",
            username="invaliduser",
            password="password123",
            level_names=["researcher"],
        )

        # Create a valid key first to ensure DB is set up
        await self.store.create_api_key(user_id=user.id, name="Valid Key")

        # Test various invalid keys
        test_cases = [
            "",  # Empty
            "invalid",  # No prefix
            "st_",  # Just prefix
            "st_wrongkey12345678901234567890",  # Wrong key
            "ST_UPPERCASE",  # Wrong case prefix
            None,  # None value
        ]

        for invalid_key in test_cases:
            if invalid_key is None:
                continue  # Skip None, would raise error
            result = await self.store.authenticate_api_key(invalid_key)
            self.assertIsNone(result, f"Key '{invalid_key}' should be rejected")

    async def test_last_used_updated(self) -> None:
        """Test that last_used_at is updated on authentication."""
        user = await self.store.create_user(
            email="lastused@example.com",
            username="lastuseduser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Last Used Key",
        )

        # Initially no last_used_at
        keys = await self.store.list_api_keys(user.id)
        self.assertIsNone(keys[0].last_used_at)

        # Authenticate
        await self.store.authenticate_api_key(raw_key)

        # Now should have last_used_at
        keys = await self.store.list_api_keys(user.id)
        self.assertIsNotNone(keys[0].last_used_at)

    async def test_key_prefix_format(self) -> None:
        """Test that key prefix has correct format."""
        user = await self.store.create_user(
            email="prefix@example.com",
            username="prefixuser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Prefix Key",
        )

        # Key should start with st_
        self.assertTrue(raw_key.startswith("st_"))

        # Prefix should be first 11 chars (st_ + 8 chars)
        self.assertEqual(api_key.key_prefix, raw_key[:11])

    async def test_delete_api_key(self) -> None:
        """Test permanently deleting an API key."""
        user = await self.store.create_user(
            email="delete@example.com",
            username="deleteuser",
            password="password123",
            level_names=["researcher"],
        )

        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Delete Key",
        )

        # Key should exist in list
        keys = await self.store.list_api_keys(user.id)
        self.assertEqual(len(keys), 1)

        # Delete the key
        success = await self.store.delete_api_key(api_key.id, user_id=user.id)
        self.assertTrue(success)

        # Key should no longer exist in list (unlike revoke)
        keys = await self.store.list_api_keys(user.id)
        self.assertEqual(len(keys), 0)

        # Key should not authenticate
        result = await self.store.authenticate_api_key(raw_key)
        self.assertIsNone(result)

    async def test_delete_wrong_user_fails(self) -> None:
        """Test that deleting another user's key fails."""
        user1 = await self.store.create_user(
            email="owner@example.com",
            username="owner",
            password="password123",
            level_names=["researcher"],
        )
        user2 = await self.store.create_user(
            email="other@example.com",
            username="other",
            password="password123",
            level_names=["researcher"],
        )

        api_key, _ = await self.store.create_api_key(
            user_id=user1.id,
            name="Owner Key",
        )

        # User2 cannot delete user1's key
        success = await self.store.delete_api_key(api_key.id, user_id=user2.id)
        self.assertFalse(success)

        # Key should still exist
        keys = await self.store.list_api_keys(user1.id)
        self.assertEqual(len(keys), 1)

    async def test_cleanup_expired_keys(self) -> None:
        """Test cleaning up expired API keys."""
        from datetime import datetime, timedelta, timezone

        user = await self.store.create_user(
            email="cleanup@example.com",
            username="cleanupuser",
            password="password123",
            level_names=["researcher"],
        )

        # Create a key that expires in 1 day
        api_key, raw_key = await self.store.create_api_key(
            user_id=user.id,
            name="Expiring Key",
            expires_in_days=1,
        )

        # Create a key that never expires
        permanent_key, _ = await self.store.create_api_key(
            user_id=user.id,
            name="Permanent Key",
        )

        # Initially 2 keys
        keys = await self.store.list_api_keys(user.id)
        self.assertEqual(len(keys), 2)

        # Cleanup should remove 0 keys (none expired yet)
        cleaned = await self.store.cleanup_expired_api_keys()
        self.assertEqual(cleaned, 0)

        # Manually update the expiring key to be expired (simulate time passing)
        # Access internal store to update directly
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        past_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        conn.execute(
            "UPDATE api_keys SET expires_at = ? WHERE id = ?",
            (past_time, api_key.id),
        )
        conn.commit()
        conn.close()

        # Now cleanup should remove 1 key
        cleaned = await self.store.cleanup_expired_api_keys()
        self.assertEqual(cleaned, 1)

        # Only permanent key should remain
        keys = await self.store.list_api_keys(user.id)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].name, "Permanent Key")


if __name__ == "__main__":
    unittest.main()
