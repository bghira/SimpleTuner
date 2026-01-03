"""Tests for route authentication protection.

Verifies that routes properly enforce authentication when multi-user mode
is enabled (i.e., when users exist in the database).

Tests cover:
- Unauthenticated requests return 401 when auth is required
- Permission-based routes (like backup) enforce specific permissions
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List, Tuple

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode
from tests.unittest_support import APITestCase


class TestRouteAuthProtection(APITestCase, unittest.TestCase):
    """Test that routes require authentication when multi-user mode is enabled."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

        # Create a user to enable multi-user mode (disable auto-auth)
        self._setup_multi_user_mode()

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def _setup_multi_user_mode(self) -> None:
        """Create a user to disable single-user auto-authentication."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore

        async def create_test_user():
            store = UserStore(self.tmp_path / "test_auth.db")
            await store.create_user(
                email="testuser@example.com",
                username="testuser",
                password="testpassword123",
                is_admin=False,
                level_names=["researcher"],
            )

        asyncio.run(create_test_user())

    def _get_protected_routes(self) -> List[Tuple[str, str]]:
        """Return list of (method, path) tuples for protected routes to test."""
        return [
            # configs.py routes
            ("GET", "/api/configs/"),
            ("GET", "/api/configs/test-config"),
            ("POST", "/api/configs/test-config/validate"),
            # models.py routes
            ("GET", "/api/models/requirements"),
            # datasets.py routes
            ("GET", "/api/datasets/plan"),
            ("GET", "/api/datasets/blueprints"),
            ("GET", "/api/datasets/browse"),
            # training.py routes
            ("GET", "/api/training/status"),
            # prompt_libraries.py routes
            ("GET", "/api/prompt-libraries/"),
            # webui_state.py routes
            ("GET", "/api/webui/state"),
            # git.py routes
            ("GET", "/api/git/status"),
            # caption_filters.py routes
            ("GET", "/api/caption-filters/"),
            # validation.py routes
            ("POST", "/api/validate/config"),
            # fields.py routes
            ("GET", "/api/fields/metadata"),
            ("GET", "/api/fields/tabs"),
            # lycoris.py routes
            ("GET", "/api/lycoris/metadata"),
            # publishing.py routes
            ("GET", "/api/publishing/token/validate"),
            ("GET", "/api/publishing/namespaces"),
            # system.py routes
            ("GET", "/api/system/status"),
            # hardware.py routes
            ("GET", "/api/hardware/gpus"),
            ("GET", "/api/hardware/memory"),
            # metrics.py routes
            ("GET", "/api/metrics"),
            ("GET", "/api/metrics/health"),
            ("GET", "/api/metrics/prometheus"),
        ]

    def test_unauthenticated_requests_return_401(self) -> None:
        """Verify unauthenticated requests to protected routes return 401."""
        protected_routes = self._get_protected_routes()
        failed_routes = []

        for method, path in protected_routes:
            if method == "GET":
                response = self.client.get(path)
            elif method == "POST":
                response = self.client.post(path, json={})
            elif method == "PUT":
                response = self.client.put(path, json={})
            elif method == "PATCH":
                response = self.client.patch(path, json={})
            elif method == "DELETE":
                response = self.client.delete(path)
            else:
                continue

            if response.status_code != 401:
                failed_routes.append(f"{method} {path}: expected 401, got {response.status_code}")

        if failed_routes:
            self.fail(f"Routes not properly protected:\n" + "\n".join(failed_routes))


class TestBackupPermissionProtection(APITestCase, unittest.TestCase):
    """Test that backup routes enforce admin.backup permission."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def _create_user_without_backup_permission(self) -> str:
        """Create a researcher user and return their session token."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.auth import UserStore

        async def setup():
            store = UserStore(self.tmp_path / "test_auth.db")

            # Create researcher level user (doesn't have admin.backup permission)
            user = await store.create_user(
                email="researcher@example.com",
                username="researcher",
                password="testpassword123",
                is_admin=False,
                level_names=["researcher"],
            )

            # create_session returns the token string directly
            session_token = await store.create_session(user.id, user_agent="test")
            return session_token

        return asyncio.run(setup())

    def test_backup_routes_require_permission(self) -> None:
        """Verify backup routes require admin.backup permission."""
        # Create user without backup permission
        session_token = self._create_user_without_backup_permission()
        cookies = {"simpletuner_session": session_token}

        backup_routes = [
            ("GET", "/api/backup"),
            ("POST", "/api/backup"),
            ("GET", "/api/backup/test-backup-id"),
            ("POST", "/api/backup/test-backup-id/restore"),
            ("DELETE", "/api/backup/test-backup-id"),
        ]

        for method, path in backup_routes:
            if method == "GET":
                response = self.client.get(path, cookies=cookies)
            elif method == "POST":
                response = self.client.post(path, json={}, cookies=cookies)
            elif method == "DELETE":
                response = self.client.delete(path, cookies=cookies)
            else:
                continue

            # Should get 403 (Forbidden) not 401 (Unauthorized)
            # User is authenticated but lacks permission
            self.assertEqual(
                response.status_code,
                403,
                f"{method} {path}: expected 403 (permission denied), " f"got {response.status_code}",
            )


if __name__ == "__main__":
    unittest.main()
