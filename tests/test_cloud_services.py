"""Unit tests for cloud services."""

import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobInfo, CloudJobStatus, JobType, UnifiedJob


class TestJobStore(unittest.IsolatedAsyncioTestCase):
    """Test cases for AsyncJobStore."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore class-level singletons (JobRepository, UploadProgressStore, etc.)
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create job store with async initialization
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

    async def test_add_and_get_job(self) -> None:
        """Test adding and retrieving a job."""
        job = UnifiedJob(
            job_id="test-job-123",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_name="test-config",
        )

        await self.store.add_job(job)

        retrieved = await self.store.get_job("test-job-123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.job_id, "test-job-123")
        self.assertEqual(retrieved.provider, "replicate")
        self.assertEqual(retrieved.config_name, "test-config")

    async def test_update_job(self) -> None:
        """Test updating a job."""
        job = UnifiedJob(
            job_id="test-job-456",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        await self.store.add_job(job)
        await self.store.update_job("test-job-456", {"status": CloudJobStatus.RUNNING.value})

        retrieved = await self.store.get_job("test-job-456")
        self.assertEqual(retrieved.status, CloudJobStatus.RUNNING.value)

    async def test_delete_job(self) -> None:
        """Test deleting a job."""
        job = UnifiedJob(
            job_id="test-job-789",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        await self.store.add_job(job)
        success = await self.store.delete_job("test-job-789")

        self.assertTrue(success)
        self.assertIsNone(await self.store.get_job("test-job-789"))

    async def test_list_jobs(self) -> None:
        """Test listing jobs."""
        for i in range(3):
            job = UnifiedJob(
                job_id=f"test-job-{i}",
                job_type=JobType.CLOUD,
                provider="replicate",
                status=CloudJobStatus.COMPLETED.value,
                config_name=f"test-config-{i}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            await self.store.add_job(job)

        jobs = await self.store.list_jobs()
        self.assertEqual(len(jobs), 3)

    async def test_cleanup_old_jobs(self) -> None:
        """Test cleanup of old jobs."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        new_date = datetime.now(timezone.utc).isoformat()

        # Add old job
        old_job = UnifiedJob(
            job_id="old-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="old-config",
            created_at=old_date,
        )
        await self.store.add_job(old_job)

        # Add new job
        new_job = UnifiedJob(
            job_id="new-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="new-config",
            created_at=new_date,
        )
        await self.store.add_job(new_job)

        # Cleanup with 90 day retention
        removed = await self.store.cleanup_old_jobs(retention_days=90)

        self.assertEqual(removed, 1)
        self.assertIsNone(await self.store.get_job("old-job"))
        self.assertIsNotNone(await self.store.get_job("new-job"))

    async def test_provider_config(self) -> None:
        """Test provider configuration storage."""
        config = {
            "webhook_url": "https://example.com/webhook",
            "cost_limit_enabled": True,
            "cost_limit_amount": 50.0,
        }

        await self.store.save_provider_config("replicate", config)
        retrieved = await self.store.get_provider_config("replicate")

        self.assertEqual(retrieved["webhook_url"], "https://example.com/webhook")
        self.assertTrue(retrieved["cost_limit_enabled"])
        self.assertEqual(retrieved["cost_limit_amount"], 50.0)

    async def test_audit_logging(self) -> None:
        """Test audit logging functionality."""
        await self.store.log_audit_event(
            action="job.submitted",
            job_id="test-audit-job",
            provider="replicate",
            config_name="test-config",
            details={"data_uploaded": True},
        )

        audit_log = await self.store.get_audit_log(limit=10)
        self.assertEqual(len(audit_log), 1)
        self.assertEqual(audit_log[0]["action"], "job.submitted")
        self.assertEqual(audit_log[0]["job_id"], "test-audit-job")
        self.assertEqual(audit_log[0]["details"]["data_uploaded"], True)

    async def test_audit_log_cleanup(self) -> None:
        """Test cleanup of old audit log entries."""
        import sqlite3

        # Add old entry directly to SQLite
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn = sqlite3.connect(str(self.store.get_database_path()))
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, details) VALUES (?, ?, ?)",
            (old_date, "old.event", "{}"),
        )
        conn.commit()
        conn.close()

        # Add new entry
        await self.store.log_audit_event(action="new.event")

        # Cleanup with 90 day retention
        removed = await self.store.cleanup_audit_log(max_age_days=90)

        self.assertEqual(removed, 1)
        audit_log = await self.store.get_audit_log()
        self.assertEqual(len(audit_log), 1)
        self.assertEqual(audit_log[0]["action"], "new.event")

    async def test_upload_progress_update_and_get(self) -> None:
        """Test updating and getting upload progress."""
        upload_id = "test-upload-123"

        # Update progress (sync method)
        self.store.update_upload_progress(
            upload_id=upload_id,
            stage="uploading",
            current=50,
            total=100,
            message="Uploading...",
        )

        # Get progress (sync method)
        progress = self.store.get_upload_progress(upload_id)

        self.assertIsNotNone(progress)
        self.assertEqual(progress["stage"], "uploading")
        self.assertEqual(progress["current"], 50)
        self.assertEqual(progress["total"], 100)
        self.assertEqual(progress["percent"], 50.0)
        self.assertEqual(progress["message"], "Uploading...")
        self.assertFalse(progress["done"])
        self.assertIsNone(progress["error"])

    async def test_upload_progress_cleanup(self) -> None:
        """Test cleanup of upload progress."""
        upload_id = "test-upload-456"

        # Create progress (sync method)
        self.store.update_upload_progress(
            upload_id=upload_id,
            stage="complete",
            current=100,
            total=100,
            done=True,
        )

        # Verify it exists (sync method)
        self.assertIsNotNone(self.store.get_upload_progress(upload_id))

        # Cleanup (sync method)
        result = self.store.cleanup_upload_progress(upload_id)
        self.assertTrue(result)

        # Verify it's gone
        self.assertIsNone(self.store.get_upload_progress(upload_id))

    async def test_get_job_by_upload_token(self) -> None:
        """Test looking up jobs by their upload token."""
        import secrets

        # Create a job with an upload token
        upload_token = secrets.token_urlsafe(32)
        job = UnifiedJob(
            job_id="token-test-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.RUNNING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            upload_token=upload_token,
        )
        await self.store.add_job(job)

        # Should find the job by token
        found = await self.store.get_job_by_upload_token(upload_token)
        self.assertIsNotNone(found)
        self.assertEqual(found.job_id, "token-test-job")

        # Should not find job with wrong token
        wrong_token = secrets.token_urlsafe(32)
        not_found = await self.store.get_job_by_upload_token(wrong_token)
        self.assertIsNone(not_found)

        # Should not find job with empty token
        empty_result = await self.store.get_job_by_upload_token("")
        self.assertIsNone(empty_result)

        # Should not find job with None token
        none_result = await self.store.get_job_by_upload_token(None)
        self.assertIsNone(none_result)

    async def test_get_job_by_upload_token_inactive_status(self) -> None:
        """Test that completed/failed jobs are not returned by token lookup."""
        import secrets

        upload_token = secrets.token_urlsafe(32)

        # Create a job that's already completed
        job = UnifiedJob(
            job_id="completed-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,  # Not active
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            upload_token=upload_token,
        )
        await self.store.add_job(job)

        # Should NOT find the job since it's not in an active state
        found = await self.store.get_job_by_upload_token(upload_token)
        self.assertIsNone(found)

        # Same for failed jobs
        failed_token = secrets.token_urlsafe(32)
        failed_job = UnifiedJob(
            job_id="failed-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.FAILED.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            upload_token=failed_token,
        )
        await self.store.add_job(failed_job)
        self.assertIsNone(await self.store.get_job_by_upload_token(failed_token))

    def test_upload_token_serialization(self) -> None:
        """Test that upload tokens are properly serialized and deserialized."""
        import secrets

        upload_token = secrets.token_urlsafe(32)
        job = UnifiedJob(
            job_id="serialization-test",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.RUNNING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            upload_token=upload_token,
        )

        # Test to_dict includes upload_token
        job_dict = job.to_dict()
        self.assertEqual(job_dict["upload_token"], upload_token)

        # Test from_dict restores upload_token
        restored = UnifiedJob.from_dict(job_dict)
        self.assertEqual(restored.upload_token, upload_token)


class TestHardwarePricingConfig(unittest.IsolatedAsyncioTestCase):
    """Test cases for configurable hardware pricing."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        # Clear hardware cache
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import clear_hardware_info_cache

        clear_hardware_info_cache()

        # Clear provider config cache (global cache shared across stores)
        from simpletuner.simpletuner_sdk.server.services.cloud.cache import get_provider_config_cache

        get_provider_config_cache().clear()

        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create job store with async initialization
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

        # Clear hardware cache
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import clear_hardware_info_cache

        clear_hardware_info_cache()

    async def test_default_hardware_info(self) -> None:
        """Test that default hardware info is used when not configured."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            DEFAULT_HARDWARE_INFO,
            clear_hardware_info_cache,
            get_hardware_info_async,
        )

        # Ensure cache is cleared before testing defaults
        clear_hardware_info_cache()

        hardware = await get_hardware_info_async(self.store)
        self.assertEqual(hardware, DEFAULT_HARDWARE_INFO)
        self.assertIn("gpu-l40s", hardware)
        self.assertIn("gpu-a100-large", hardware)

    async def test_configured_hardware_info(self) -> None:
        """Test that configured hardware info overrides defaults."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            clear_hardware_info_cache,
            get_hardware_info_async,
        )

        # Configure custom hardware info
        custom_hardware = {
            "gpu-a100-large": {"cost_per_second": 0.0014, "name": "A100 (80GB)"},
            "gpu-h100": {"name": "H100 (80GB)", "cost_per_second": 0.0014},
            "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
        }
        await self.store.save_provider_config("replicate", {"hardware_info": custom_hardware})

        # Clear cache to pick up new config
        clear_hardware_info_cache()

        hardware = await get_hardware_info_async(self.store)
        self.assertEqual(hardware, custom_hardware)
        self.assertIn("gpu-h100", hardware)

    async def test_default_hardware_cost_per_hour(self) -> None:
        """Test getting default hardware cost per hour."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            clear_hardware_info_cache,
            get_default_hardware_cost_per_hour,
        )

        # Ensure cache is cleared before testing defaults
        clear_hardware_info_cache()

        cost = await get_default_hardware_cost_per_hour(self.store)
        # L40S at $0.000975/sec = $3.51/hr
        self.assertAlmostEqual(cost, 3.51, places=2)

    async def test_configured_hardware_cost_per_hour(self) -> None:
        """Test that configured hardware cost is used."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            clear_hardware_info_cache,
            get_default_hardware_cost_per_hour,
        )

        # Configure custom L40S price
        custom_hardware = {
            "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.001},  # $3.60/hr
        }
        await self.store.save_provider_config("replicate", {"hardware_info": custom_hardware})
        clear_hardware_info_cache()

        cost = await get_default_hardware_cost_per_hour(self.store)
        self.assertAlmostEqual(cost, 3.60, places=2)  # 0.001 * 3600 = 3.60


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
    """Test cases for rate limiter.

    Note: The RateLimiter class in rate_limiting.py is deprecated.
    Rate limiting is now handled by RateLimitMiddleware in security_middleware.py.
    These tests are kept to verify the deprecated module still works.
    """

    def test_allows_under_limit(self) -> None:
        """Test that requests under limit are allowed."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.rate_limiting import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # First 5 should be allowed
        for i in range(5):
            self.assertTrue(limiter.is_allowed("test-key"), f"Request {i+1} should be allowed")

    def test_blocks_over_limit(self) -> None:
        """Test that requests over limit are blocked."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.rate_limiting import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # First 3 should be allowed
        for _ in range(3):
            self.assertTrue(limiter.is_allowed("test-key"))

        # 4th should be blocked
        self.assertFalse(limiter.is_allowed("test-key"))

    def test_separate_keys(self) -> None:
        """Test that different keys have separate limits."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.rate_limiting import RateLimiter

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


class TestCloudJobAccessControl(unittest.TestCase):
    """Tests for CloudJob access control methods."""

    def _create_job(self, user_id: int = 1) -> "UnifiedJob":
        """Create a test job with a specific user_id."""
        return UnifiedJob(
            job_id="test-job-access",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.RUNNING.value,
            config_name="test",
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
        )

    def test_can_access_owner(self) -> None:
        """Test owner can access their own job."""
        job = self._create_job(user_id=1)
        self.assertTrue(job.can_access(user_id=1))

    def test_can_access_different_user(self) -> None:
        """Test different user cannot access job."""
        job = self._create_job(user_id=1)
        self.assertFalse(job.can_access(user_id=2))

    def test_can_access_with_view_all_permission(self) -> None:
        """Test user with view_all permission can access any job."""
        job = self._create_job(user_id=1)
        self.assertTrue(job.can_access(user_id=2, has_view_all=True))

    def test_can_access_unauthenticated_blocked_by_default(self) -> None:
        """Test unauthenticated access is blocked when not in single-user mode."""
        job = self._create_job(user_id=1)
        # Without is_single_user_mode=True, unauthenticated access should be denied
        self.assertFalse(job.can_access(user_id=None))
        self.assertFalse(job.can_access(user_id=None, is_single_user_mode=False))

    def test_can_access_unauthenticated_allowed_in_single_user_mode(self) -> None:
        """Test unauthenticated access is allowed in single-user mode."""
        job = self._create_job(user_id=1)
        self.assertTrue(job.can_access(user_id=None, is_single_user_mode=True))

    def test_can_cancel_by_owner(self) -> None:
        """Test owner with permission can cancel their job."""
        job = self._create_job(user_id=1)
        self.assertTrue(job.can_cancel_by(user_id=1, has_cancel_all=False, has_cancel_own=True))

    def test_can_cancel_by_other_user(self) -> None:
        """Test non-owner without cancel_all cannot cancel."""
        job = self._create_job(user_id=1)
        self.assertFalse(job.can_cancel_by(user_id=2, has_cancel_all=False, has_cancel_own=True))

    def test_can_cancel_by_admin(self) -> None:
        """Test user with cancel_all can cancel any job."""
        job = self._create_job(user_id=1)
        self.assertTrue(job.can_cancel_by(user_id=2, has_cancel_all=True, has_cancel_own=False))


if __name__ == "__main__":
    unittest.main()
