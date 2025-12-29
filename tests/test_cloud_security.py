"""Unit tests for cloud security features.

Tests for webhook verification, rate limiting, IP allowlists,
security configuration, and path traversal protection.
"""

import base64
import hashlib
import hmac
import os
import tempfile
import time
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore


class TestWebhookSignatureVerification(unittest.TestCase):
    """Test cases for webhook signature verification."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a test secret (base64 encoded)
        self.secret_raw = os.urandom(32)
        self.secret_b64 = base64.b64encode(self.secret_raw).decode()

    def test_valid_signature(self) -> None:
        """Test verification of a valid signature."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.webhooks import _verify_webhook_signature

        webhook_id = "msg_test123"
        webhook_timestamp = str(int(time.time()))
        body = b'{"event": "test"}'

        # Create valid signature - format is id.timestamp.body (note trailing dot)
        signed_content = f"{webhook_id}.{webhook_timestamp}.".encode() + body
        signature = hmac.new(self.secret_raw, signed_content, hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()
        webhook_signature = f"v1,{signature_b64}"

        result = _verify_webhook_signature(body, webhook_id, webhook_timestamp, webhook_signature, self.secret_b64)
        self.assertTrue(result)

    def test_invalid_signature(self) -> None:
        """Test rejection of an invalid signature."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.webhooks import _verify_webhook_signature

        webhook_id = "msg_test456"
        webhook_timestamp = str(int(time.time()))
        body = b'{"event": "test"}'

        # Create invalid signature
        webhook_signature = "v1,invalid_signature_here"

        result = _verify_webhook_signature(body, webhook_id, webhook_timestamp, webhook_signature, self.secret_b64)
        self.assertFalse(result)

    def test_expired_timestamp(self) -> None:
        """Test rejection of expired timestamp (replay attack protection)."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.webhooks import _verify_webhook_signature

        webhook_id = "msg_test789"
        # Timestamp from 10 minutes ago
        webhook_timestamp = str(int(time.time()) - 600)
        body = b'{"event": "test"}'

        # Create valid signature for the old timestamp
        signed_content = f"{webhook_id}.{webhook_timestamp}.".encode() + body
        signature = hmac.new(self.secret_raw, signed_content, hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()
        webhook_signature = f"v1,{signature_b64}"

        result = _verify_webhook_signature(body, webhook_id, webhook_timestamp, webhook_signature, self.secret_b64)
        self.assertFalse(result)

    def test_whsec_prefix(self) -> None:
        """Test that whsec_ prefix is handled correctly."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.webhooks import _verify_webhook_signature

        webhook_id = "msg_whsec_test"
        webhook_timestamp = str(int(time.time()))
        body = b'{"event": "test"}'

        # Create valid signature
        signed_content = f"{webhook_id}.{webhook_timestamp}.".encode() + body
        signature = hmac.new(self.secret_raw, signed_content, hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()
        webhook_signature = f"v1,{signature_b64}"

        # Secret with whsec_ prefix
        secret_with_prefix = f"whsec_{self.secret_b64}"

        result = _verify_webhook_signature(body, webhook_id, webhook_timestamp, webhook_signature, secret_with_prefix)
        self.assertTrue(result)


class TestRateLimiter(unittest.TestCase):
    """Test cases for rate limiter."""

    def test_allows_under_limit(self) -> None:
        """Test that requests under limit are allowed."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # First 5 should be allowed
        for i in range(5):
            self.assertTrue(limiter.is_allowed("test-key"), f"Request {i+1} should be allowed")

    def test_blocks_over_limit(self) -> None:
        """Test that requests over limit are blocked."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # First 3 should be allowed
        for _ in range(3):
            self.assertTrue(limiter.is_allowed("test-key"))

        # 4th should be blocked
        self.assertFalse(limiter.is_allowed("test-key"))

    def test_separate_keys(self) -> None:
        """Test that different keys have separate limits."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Exhaust key1
        self.assertTrue(limiter.is_allowed("key1"))
        self.assertTrue(limiter.is_allowed("key1"))
        self.assertFalse(limiter.is_allowed("key1"))

        # key2 should still work
        self.assertTrue(limiter.is_allowed("key2"))


class TestIPAllowlist(unittest.TestCase):
    """Test cases for IP allowlist checking."""

    def test_single_ip_match(self) -> None:
        """Test matching a single IP."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import check_ip_allowlist

        self.assertTrue(check_ip_allowlist("192.168.1.100", ["192.168.1.100"]))
        self.assertFalse(check_ip_allowlist("192.168.1.101", ["192.168.1.100"]))

    def test_cidr_range(self) -> None:
        """Test matching CIDR ranges."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import check_ip_allowlist

        allowed = ["10.0.0.0/8", "192.168.0.0/16"]

        self.assertTrue(check_ip_allowlist("10.0.0.1", allowed))
        self.assertTrue(check_ip_allowlist("10.255.255.255", allowed))
        self.assertTrue(check_ip_allowlist("192.168.1.1", allowed))
        self.assertFalse(check_ip_allowlist("172.16.0.1", allowed))

    def test_invalid_ip(self) -> None:
        """Test handling of invalid IP addresses."""
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import check_ip_allowlist

        self.assertFalse(check_ip_allowlist("invalid-ip", ["192.168.1.0/24"]))
        self.assertFalse(check_ip_allowlist("", ["192.168.1.0/24"]))


class TestSecurityConfigStorage(unittest.IsolatedAsyncioTestCase):
    """Test cases for security configuration storage."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        await self.store.close()
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore singletons
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_webhook_require_signature_default(self) -> None:
        """Test that webhook signature is required by default."""
        config = await self.store.get_provider_config("replicate")
        # Default should be True (not in config means use default)
        require_sig = config.get("webhook_require_signature", True)
        self.assertTrue(require_sig)

    async def test_webhook_allowed_ips_storage(self) -> None:
        """Test storing and retrieving IP allowlist."""
        allowed_ips = ["192.168.1.0/24", "10.0.0.1"]
        await self.store.save_provider_config("replicate", {"webhook_allowed_ips": allowed_ips})

        config = await self.store.get_provider_config("replicate")
        self.assertEqual(config.get("webhook_allowed_ips"), allowed_ips)

    async def test_ssl_settings_storage(self) -> None:
        """Test storing SSL configuration."""
        await self.store.save_provider_config(
            "replicate",
            {
                "ssl_verify": False,
                "ssl_ca_bundle": "/path/to/ca-bundle.crt",
            },
        )

        config = await self.store.get_provider_config("replicate")
        self.assertFalse(config.get("ssl_verify"))
        self.assertEqual(config.get("ssl_ca_bundle"), "/path/to/ca-bundle.crt")


class TestPathTraversalProtection(unittest.TestCase):
    """Test cases for path traversal protection in cloud_upload_service."""

    def test_forbidden_system_paths(self) -> None:
        """Test that forbidden system paths are rejected."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        # These should all be rejected
        forbidden = [
            Path("/etc/passwd"),
            Path("/etc"),
            Path("/var/log"),
            Path("/usr/bin"),
            Path("/root"),
            Path("/proc/self"),
        ]

        for path in forbidden:
            self.assertFalse(_is_safe_path(path), f"Path {path} should be forbidden")

    def test_sensitive_patterns_rejected(self) -> None:
        """Test that paths with sensitive patterns are rejected."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        # Create a temp directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # These patterns should be rejected
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            env_dir = Path(temp_dir) / ".env"
            env_dir.mkdir()

            ssh_dir = Path(temp_dir) / ".ssh"
            ssh_dir.mkdir()

            self.assertFalse(_is_safe_path(git_dir))
            self.assertFalse(_is_safe_path(env_dir))
            self.assertFalse(_is_safe_path(ssh_dir))

    def test_safe_user_directory(self) -> None:
        """Test that safe user directories are allowed."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        # A typical user data directory should be allowed
        # Use current working directory as base (not /tmp which is in FORBIDDEN_PATHS)
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            data_dir = Path(temp_dir) / "my_dataset"
            data_dir.mkdir()

            self.assertTrue(_is_safe_path(data_dir))

    def test_symlink_attack_prevention(self) -> None:
        """Test that symlinks pointing outside base are rejected."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "dataset"
            base_dir.mkdir()

            # Create a real file
            real_file = base_dir / "real.txt"
            real_file.write_text("test")

            # Create a symlink to /etc/passwd (simulated)
            outside_file = Path(temp_dir) / "outside.txt"
            outside_file.write_text("outside content")

            symlink_file = base_dir / "symlink.txt"
            try:
                symlink_file.symlink_to(outside_file)

                # The symlink should be rejected
                self.assertFalse(_is_safe_file(symlink_file, base_dir))

                # The real file should be allowed
                self.assertTrue(_is_safe_file(real_file, base_dir))
            except OSError:
                # Skip if symlinks aren't supported
                pass

    def test_hidden_files_skipped(self) -> None:
        """Test that hidden files are skipped."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "dataset"
            base_dir.mkdir()

            hidden_file = base_dir / ".hidden"
            hidden_file.write_text("secret")

            normal_file = base_dir / "normal.txt"
            normal_file.write_text("public")

            self.assertFalse(_is_safe_file(hidden_file, base_dir))
            self.assertTrue(_is_safe_file(normal_file, base_dir))

    def test_sensitive_extensions_skipped(self) -> None:
        """Test that files with sensitive extensions are skipped."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "dataset"
            base_dir.mkdir()

            # Sensitive extensions
            for ext in [".env", ".pem", ".key"]:
                sensitive = base_dir / f"secret{ext}"
                sensitive.write_text("secret")
                self.assertFalse(
                    _is_safe_file(sensitive, base_dir),
                    f"File with {ext} extension should be rejected",
                )

            # Normal extensions should be allowed
            for ext in [".txt", ".png", ".jpg"]:
                normal = base_dir / f"normal{ext}"
                normal.write_text("public")
                self.assertTrue(
                    _is_safe_file(normal, base_dir),
                    f"File with {ext} extension should be allowed",
                )


if __name__ == "__main__":
    unittest.main()
