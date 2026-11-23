"""Unit tests for cloud services."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
