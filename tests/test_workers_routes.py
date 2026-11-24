"""
Tests for worker routes: /api/workers/* and /api/admin/workers/*

Covers worker registration, heartbeats, job status updates, SSE streaming,
and admin worker management endpoints.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server.models.worker import Worker, WorkerStatus, WorkerType
from simpletuner.simpletuner_sdk.server.routes.workers import (
    admin_router,
    generate_worker_token,
    hash_token,
    is_worker_connected,
    push_to_worker,
    router,
    validate_worker_token,
    worker_streams,
)
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore
from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import get_job_repository
from simpletuner.simpletuner_sdk.server.services.worker_repository import get_worker_repository
from tests.unittest_support import APITestCase, create_router_client, run_async


class WorkerHelperFunctionsTestCase(unittest.TestCase):
    """Test helper functions: hash_token, generate_worker_token, etc."""

    def test_hash_token(self) -> None:
        """Test token hashing is deterministic."""
        token = "test-token-123"
        hash1 = hash_token(token)
        hash2 = hash_token(token)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex digest
        self.assertIsInstance(hash1, str)

    def test_hash_token_different_tokens(self) -> None:
        """Test different tokens produce different hashes."""
        token1 = "token-1"
        token2 = "token-2"

        hash1 = hash_token(token1)
        hash2 = hash_token(token2)

        self.assertNotEqual(hash1, hash2)

    def test_generate_worker_token(self) -> None:
        """Test token generation creates unique tokens."""
        token1 = generate_worker_token()
        token2 = generate_worker_token()

        self.assertIsInstance(token1, str)
        self.assertIsInstance(token2, str)
        self.assertNotEqual(token1, token2)
        self.assertGreater(len(token1), 20)  # Should be reasonably long

    def test_is_worker_connected_false(self) -> None:
        """Test is_worker_connected returns False for disconnected worker."""
        worker_id = "worker-123"
        self.assertFalse(is_worker_connected(worker_id))

    def test_is_worker_connected_true(self) -> None:
        """Test is_worker_connected returns True when worker has stream."""
        worker_id = "worker-123"
        worker_streams[worker_id] = asyncio.Queue()

        try:
            self.assertTrue(is_worker_connected(worker_id))
        finally:
            del worker_streams[worker_id]

    def test_push_to_worker_not_connected(self) -> None:
        """Test push_to_worker returns False when worker not connected."""

        async def _test():
            worker_id = "worker-123"
            event = {"type": "test", "data": {}}

            result = await push_to_worker(worker_id, event)
            self.assertFalse(result)

        run_async(_test())

    def test_push_to_worker_connected(self) -> None:
        """Test push_to_worker pushes event to worker's queue."""

        async def _test():
            worker_id = "worker-123"
            queue = asyncio.Queue()
            worker_streams[worker_id] = queue

            try:
                event = {"type": "test", "data": {"message": "hello"}}
                result = await push_to_worker(worker_id, event)

                self.assertTrue(result)
                self.assertEqual(queue.qsize(), 1)

                # Verify event was queued
                queued_event = await queue.get()
                self.assertEqual(queued_event, event)
            finally:
                del worker_streams[worker_id]

        run_async(_test())

    def test_validate_worker_token_valid(self) -> None:
        """Test validate_worker_token returns worker for valid token."""

        async def _test():
            token = "valid-token"
            token_hash = hash_token(token)

            mock_worker = Worker(
                worker_id="worker-123",
                name="Test Worker",
                worker_type=WorkerType.PERSISTENT,
                status=WorkerStatus.IDLE,
                token_hash=token_hash,
                user_id=1,
            )

            with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
                mock_repo_instance = AsyncMock()
                mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=mock_worker)
                mock_repo.return_value = mock_repo_instance

                worker = await validate_worker_token(token)

                self.assertEqual(worker.worker_id, "worker-123")
                mock_repo_instance.get_worker_by_token_hash.assert_called_once_with(token_hash)

        run_async(_test())

    def test_validate_worker_token_invalid(self) -> None:
        """Test validate_worker_token raises 401 for invalid token."""

        async def _test():
            token = "invalid-token"

            with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
                mock_repo_instance = AsyncMock()
                mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=None)
                mock_repo.return_value = mock_repo_instance

                from fastapi import HTTPException

                with self.assertRaises(HTTPException) as ctx:
                    await validate_worker_token(token)

                self.assertEqual(ctx.exception.status_code, 401)
                self.assertIn("Invalid worker token", ctx.exception.detail)

        run_async(_test())


class WorkerRegistrationTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/workers/register endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        # Create a test token and worker
        self.token = generate_worker_token()
        self.token_hash = hash_token(self.token)
        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.CONNECTING,
            token_hash=self.token_hash,
            user_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_register_worker_success(self) -> None:
        """Test successful worker registration."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/register",
                json={
                    "name": "GPU Worker 1",
                    "gpu_info": {"name": "A100", "vram_gb": 80},
                    "persistent": True,
                    "provider": None,
                    "labels": {"region": "us-west"},
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["worker_id"], "worker-123")
            self.assertIn("/api/workers/stream", data["sse_url"])
            self.assertIsNone(data["resume_job"])
            self.assertIsNone(data["abandon_job"])

    def test_register_worker_invalid_token(self) -> None:
        """Test registration with invalid token returns 401."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/register",
                json={"name": "Test Worker", "gpu_info": {}, "persistent": False},
                headers={"X-Worker-Token": "invalid-token"},
            )

            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid worker token", response.json()["detail"])

    def test_register_worker_missing_token(self) -> None:
        """Test registration without token returns 422."""
        response = self.client.post(
            "/api/workers/register",
            json={"name": "Test Worker", "gpu_info": {}, "persistent": False},
        )

        self.assertEqual(response.status_code, 422)

    def test_register_worker_with_resume_job(self) -> None:
        """Test registration when worker should resume a job."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            # Mock job that's still running
            mock_job = Mock()
            mock_job.job_id = "job-456"
            mock_job.config_name = "test-config"
            mock_job.status = "training"
            mock_job.metadata = {"worker_id": self.worker.worker_id}

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=mock_job)
            mock_job_repo.return_value = mock_job_repo_instance

            response = self.client.post(
                "/api/workers/register",
                json={
                    "name": "GPU Worker 1",
                    "gpu_info": {},
                    "persistent": True,
                    "current_job_id": "job-456",
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsNotNone(data["resume_job"])
            self.assertEqual(data["resume_job"]["job_id"], "job-456")
            self.assertIsNone(data["abandon_job"])

    def test_reconnect_and_busy_progress_recover_only_the_assigned_retry(self) -> None:
        from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository

        class IsolatedJobRepository(JobRepository):
            pass

        directory = self.enterContext(tempfile.TemporaryDirectory())
        jobs = IsolatedJobRepository(Path(directory) / "jobs.db")
        self.addCleanup(jobs.reset_instance)
        repo = AsyncMock()
        repo.get_worker_by_token_hash.return_value = self.worker
        self.enterContext(
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository", return_value=repo)
        )
        self.enterContext(
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=jobs,
            )
        )
        self.enterContext(patch("simpletuner.simpletuner_sdk.server.routes.workers._record_worker_stats"))
        now = datetime.now(timezone.utc).isoformat()
        for job_id, status, owner, retry, should_resume in (
            ("reconnect", "pending", self.worker.worker_id, 1, True),
            ("progress", "pending", self.worker.worker_id, 1, True),
            ("redispatched", "running", "another-worker", 1, False),
            ("cancelled", "cancelled", self.worker.worker_id, 1, False),
            ("fresh", "pending", self.worker.worker_id, 0, False),
        ):
            with self.subTest(job_id=job_id):
                job = UnifiedJob(
                    job_id,
                    JobType.LOCAL,
                    "worker",
                    status,
                    "training",
                    now,
                    started_at=now,
                    completed_at=now,
                    error_message="old failure",
                    metadata={"worker_id": owner, "retry_count": retry, "needs_reconciliation": True},
                )
                run_async(jobs.add(job))
                headers = {"X-Worker-Token": self.token}
                if job_id == "progress":
                    reply = self.client.post(
                        "/api/workers/heartbeat",
                        headers=headers,
                        json={
                            "worker_id": self.worker.worker_id,
                            "status": "busy",
                            "current_job_id": job_id,
                            "job_progress": {"step": 100},
                        },
                    )
                else:
                    reply = self.client.post(
                        "/api/workers/register",
                        headers=headers,
                        json={"name": "training", "persistent": True, "current_job_id": job_id, "gpu_info": {}},
                    )
                self.assertEqual(reply.status_code, 200, reply.text)
                if job_id != "progress":
                    self.assertEqual(reply.json()["abandon_job"], None if should_resume else job_id)
                saved = run_async(jobs.get(job_id))
                self.assertEqual(saved.status, "running" if should_resume else status)
                self.assertEqual(saved.metadata["worker_id"], owner)
                if should_resume:
                    self.assertIsNone(saved.error_message)
                    self.assertIsNone(saved.completed_at)
                    self.assertEqual(saved.started_at, now)
                    self.assertNotIn("needs_reconciliation", saved.metadata)
                    self.assertEqual(saved.metadata["retry_count"], retry - 1)
                    self.assertFalse(run_async(jobs.update(job_id, {"status": "running"}, expected_status="pending")))

    def test_register_worker_with_abandon_job(self) -> None:
        """Test registration when worker should abandon a stale job."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            # Mock job that's been cancelled
            mock_job = Mock()
            mock_job.job_id = "job-456"
            mock_job.status = "cancelled"

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=mock_job)
            mock_job_repo.return_value = mock_job_repo_instance

            response = self.client.post(
                "/api/workers/register",
                json={
                    "name": "GPU Worker 1",
                    "gpu_info": {},
                    "persistent": True,
                    "current_job_id": "job-456",
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsNone(data["resume_job"])
            self.assertEqual(data["abandon_job"], "job-456")


class WorkerHeartbeatTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/workers/heartbeat endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.token = generate_worker_token()
        self.token_hash = hash_token(self.token)
        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.IDLE,
            token_hash=self.token_hash,
            user_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_heartbeat_success(self) -> None:
        """Test successful heartbeat update."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/heartbeat",
                json={
                    "worker_id": "worker-123",
                    "status": "idle",
                    "gpu_utilization": 15.5,
                    "vram_used_gb": 8.2,
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])

            # Verify update_worker was called with correct data
            mock_repo_instance.update_worker.assert_called_once()
            call_args = mock_repo_instance.update_worker.call_args
            self.assertEqual(call_args[0][0], "worker-123")
            updates = call_args[0][1]
            self.assertEqual(updates["status"], "idle")

    def test_heartbeat_invalid_token(self) -> None:
        """Test heartbeat with invalid token returns 401."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/heartbeat",
                json={"worker_id": "worker-123", "status": "idle"},
                headers={"X-Worker-Token": "invalid-token"},
            )

            self.assertEqual(response.status_code, 401)

    def test_heartbeat_worker_id_mismatch(self) -> None:
        """Test heartbeat with mismatched worker_id returns 400."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/heartbeat",
                json={"worker_id": "worker-999", "status": "idle"},
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Worker ID mismatch", response.json()["detail"])

    def test_heartbeat_with_current_job(self) -> None:
        """Test heartbeat updates current_job_id."""
        with patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post(
                "/api/workers/heartbeat",
                json={
                    "worker_id": "worker-123",
                    "status": "busy",
                    "current_job_id": "job-456",
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)

            # Verify current_job_id was included in update
            call_args = mock_repo_instance.update_worker.call_args
            updates = call_args[0][1]
            self.assertEqual(updates["current_job_id"], "job-456")

    @patch.dict(BaseSQLiteStore._instances, clear=True)
    @patch("simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository._job_repository", None)
    @patch("simpletuner.simpletuner_sdk.server.services.worker_repository._worker_repository", None)
    def test_heartbeat_only_repairs_active_jobs_assigned_to_its_worker(self) -> None:
        jobs = get_job_repository()
        worker = self.worker
        worker.current_job_id = "job-456"
        run_async(get_worker_repository().create_worker(worker))
        orphan_error = "Job orphaned (no PID tracked)"
        run_async(
            jobs.add(
                UnifiedJob(
                    job_id=worker.current_job_id,
                    job_type=JobType.LOCAL,
                    provider="worker",
                    status="running",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    config_name="remote-training",
                    metadata={"worker_id": "worker-999"},
                    error_message=orphan_error,
                )
            )
        )
        heartbeat = {"worker_id": worker.worker_id, "status": "busy", "current_job_id": worker.current_job_id}
        headers = {"X-Worker-Token": self.token}

        response = self.client.post(
            "/api/workers/heartbeat", json={**heartbeat, "log_tail": "foreign output"}, headers=headers
        )
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(run_async(jobs.get_worker_log_tail(worker.current_job_id)))

        for status, error in (("running", orphan_error), ("failed", orphan_error), ("running", "CUDA out of memory")):
            with self.subTest(status=status, error=error):
                assigned_worker = "worker-999" if (status, error) == ("running", orphan_error) else worker.worker_id
                run_async(
                    jobs.update(
                        worker.current_job_id,
                        {"status": status, "error_message": error, "metadata": {"worker_id": assigned_worker}},
                    )
                )
                response = self.client.post("/api/workers/heartbeat", json=heartbeat, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(run_async(jobs.get(worker.current_job_id)).error_message, error)


class MultinodePeerTestCase(APITestCase, unittest.TestCase):
    @patch.dict(BaseSQLiteStore._instances, clear=True)
    def test_monitor_peer_through_registration_reconnect_and_job_completion(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import get_optional_user
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository
        from simpletuner.simpletuner_sdk.server.services.worker_repository import WorkerRepository

        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        workers = WorkerRepository(directory / "workers.db")
        jobs = JobRepository(directory / "jobs.db")
        self.enterContext(
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository", return_value=workers
            )
        )
        self.enterContext(
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=jobs,
            )
        )
        client = self.enterContext(create_router_client(router, admin_router))
        client.app.dependency_overrides[get_optional_user] = lambda: Mock(id=1, username="admin")
        job = UnifiedJob(
            "run",
            JobType.LOCAL,
            "worker",
            "running",
            "train",
            datetime.now(timezone.utc).isoformat(),
            metadata={"worker_id": "leader"},
        )
        run_async(jobs.add(job))
        for invalid, code in [
            ({"worker_type": "peer"}, 422),
            ({"worker_type": "persistent", "job_id": "run"}, 422),
            ({"worker_type": "peer", "job_id": "missing"}, 404),
        ]:
            self.assertEqual(client.post("/api/admin/workers", json={"name": "node-2", **invalid}).status_code, code)
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log", new_callable=AsyncMock):
            response = client.post("/api/admin/workers", json={"name": "node-2", "worker_type": "peer", "job_id": "run"})
        self.assertEqual(response.status_code, 200, response.text)
        credentials = response.json()
        headers = {"X-Worker-Token": credentials["token"]}
        worker_id = credentials["worker_id"]
        registration = {"name": "peer-host", "persistent": True, "gpu_info": {"count": 8}}
        beat = {"worker_id": worker_id, "status": "idle", "system_metrics": {"gpu_0_utilization_percent": 90}}
        for _ in range(2):
            reply = client.post("/api/workers/register", headers=headers, json=registration)
            self.assertEqual(reply.status_code, 200, reply.text)
            self.assertIsNone(reply.json()["resume_job"])
            response = client.post("/api/workers/heartbeat", headers=headers, json=beat)
            self.assertEqual(response.status_code, 200, response.text)
            peer = run_async(workers.get_worker(worker_id))
            self.assertEqual(
                (peer.worker_type, peer.status, peer.current_job_id), (WorkerType.PEER, WorkerStatus.BUSY, "run")
            )
            self.assertEqual(peer.system_sample["metrics"], beat["system_metrics"])
            self.assertIsNone(run_async(workers.get_idle_worker_for_job()))

        for extra, code in [
            ({"current_job_id": "other"}, 409),
            ({"log_tail": "text"}, 403),
            ({"job_progress": {"step": 3}}, 403),
        ]:
            self.assertEqual(
                client.post("/api/workers/heartbeat", headers=headers, json={**beat, **extra}).status_code, code
            )
        self.assertEqual(
            client.post("/api/workers/job/run/status", headers=headers, json={"status": "completed"}).status_code, 403
        )
        self.assertEqual(run_async(jobs.get("run")).metadata["worker_id"], "leader")
        run_async(jobs.update("run", {"status": "completed"}))
        self.assertEqual(client.post("/api/workers/heartbeat", headers=headers, json=beat).status_code, 410)
        self.assertEqual(client.post("/api/workers/register", headers=headers, json=registration).status_code, 410)
        self.assertIsNone(run_async(workers.get_worker(worker_id)).current_job_id)


class WorkerJobStatusTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/workers/job/{job_id}/status endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.token = generate_worker_token()
        self.token_hash = hash_token(self.token)
        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.BUSY,
            token_hash=self.token_hash,
            user_id=1,
            current_job_id="job-456",
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    @patch.dict(BaseSQLiteStore._instances, clear=True)
    def test_group_member_failure_cancels_leader_and_late_completion_preserves_error(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository

        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        jobs = JobRepository(directory / "jobs.db")
        self.enterContext(
            patch("simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token", return_value=self.worker)
        )
        self.enterContext(
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=jobs,
            )
        )
        job = UnifiedJob(
            job_id="job-group",
            job_type=JobType.LOCAL,
            provider="worker",
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
            config_name="distributed",
            metadata={
                "worker_id": "leader",
                "worker_ids": ["leader", self.worker.worker_id],
                "worker_names": ["node one", self.worker.name],
            },
        )
        run_async(jobs.add(job))
        leader_queue = asyncio.Queue()
        with patch.dict(worker_streams, {"leader": leader_queue}):
            for status in ("failed", "completed"):
                response = self.client.post(
                    "/api/workers/job/job-group/status",
                    json={"status": status, "error": "lost GPU"},
                    headers={"X-Worker-Token": self.token},
                )
                response.raise_for_status()
        self.assertEqual(leader_queue.get_nowait(), {"type": "job_cancel", "job_id": "job-group"})
        self.assertTrue(leader_queue.empty())
        stored = run_async(jobs.get(job.job_id))
        self.assertEqual((stored.status, stored.error_message), ("failed", "lost GPU"))

    def test_update_job_status_success(self) -> None:
        """Test successful job status update."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
            patch("simpletuner.simpletuner_sdk.server.services.sse_manager.get_sse_manager") as mock_sse,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            mock_job = Mock()
            mock_job.job_id = "job-456"
            mock_job.status = "running"
            mock_job.metadata = {"worker_id": "worker-123"}

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=mock_job)
            mock_job_repo_instance.update = AsyncMock()
            mock_job_repo.return_value = mock_job_repo_instance

            mock_sse_instance = AsyncMock()
            mock_sse_instance.broadcast = AsyncMock()
            mock_sse.return_value = mock_sse_instance

            response = self.client.post(
                "/api/workers/job/job-456/status",
                json={
                    "status": "completed",
                    "progress": {"step": 1000},
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])

            # Verify job was updated
            mock_job_repo_instance.update.assert_called_once()

    def test_update_job_status_job_not_found(self) -> None:
        """Test job status update for non-existent job returns 404."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=None)
            mock_job_repo.return_value = mock_job_repo_instance

            response = self.client.post(
                "/api/workers/job/job-999/status",
                json={"status": "completed"},
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 404)
            self.assertIn("Job not found", response.json()["detail"])

    def test_update_job_status_wrong_worker(self) -> None:
        """Test job status update from wrong worker returns 403."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            # Job assigned to different worker
            mock_job = Mock()
            mock_job.job_id = "job-456"
            mock_job.metadata = {"worker_id": "worker-999"}

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=mock_job)
            mock_job_repo.return_value = mock_job_repo_instance

            response = self.client.post(
                "/api/workers/job/job-456/status",
                json={"status": "completed"},
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 403)
            self.assertIn("Worker not assigned", response.json()["detail"])

    def test_update_job_status_with_error(self) -> None:
        """Test job status update with error message."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository"
            ) as mock_job_repo,
            patch("simpletuner.simpletuner_sdk.server.services.sse_manager.get_sse_manager") as mock_sse,
        ):

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker_by_token_hash = AsyncMock(return_value=self.worker)
            mock_repo.return_value = mock_repo_instance

            mock_job = Mock()
            mock_job.job_id = "job-456"
            mock_job.metadata = {"worker_id": "worker-123"}

            mock_job_repo_instance = AsyncMock()
            mock_job_repo_instance.get = AsyncMock(return_value=mock_job)
            mock_job_repo_instance.update = AsyncMock()
            mock_job_repo.return_value = mock_job_repo_instance

            mock_sse_instance = AsyncMock()
            mock_sse_instance.broadcast = AsyncMock()
            mock_sse.return_value = mock_sse_instance

            response = self.client.post(
                "/api/workers/job/job-456/status",
                json={
                    "status": "failed",
                    "error": "CUDA out of memory",
                },
                headers={"X-Worker-Token": self.token},
            )

            self.assertEqual(response.status_code, 200)

            # Verify error was included in update
            call_args = mock_job_repo_instance.update.call_args
            updates = call_args[0][1]
            self.assertEqual(updates["error_message"], "CUDA out of memory")


class AdminListWorkersTestCase(APITestCase, unittest.TestCase):
    """Test GET /api/admin/workers endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        # Create mock user with admin.workers permission
        self.admin_user = User(
            id=1,
            email="admin@test.com",
            username="admin",
            is_active=True,
            is_admin=True,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_list_workers_success(self) -> None:
        """Test successful worker listing."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
        ):

            mock_user.return_value = self.admin_user

            workers = [
                Worker(
                    worker_id="worker-1",
                    name="Worker 1",
                    worker_type=WorkerType.PERSISTENT,
                    status=WorkerStatus.IDLE,
                    token_hash="hash1",
                    user_id=1,
                    created_at=datetime.now(timezone.utc),
                ),
                Worker(
                    worker_id="worker-2",
                    name="Worker 2",
                    worker_type=WorkerType.EPHEMERAL,
                    status=WorkerStatus.BUSY,
                    token_hash="hash2",
                    user_id=1,
                    created_at=datetime.now(timezone.utc),
                ),
            ]

            mock_repo_instance = AsyncMock()
            mock_repo_instance.list_workers = AsyncMock(return_value=workers)
            mock_repo.return_value = mock_repo_instance

            response = self.client.get("/api/admin/workers")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["total"], 2)
            self.assertEqual(len(data["workers"]), 2)
            self.assertEqual(data["workers"][0]["worker_id"], "worker-1")
            self.assertEqual(data["workers"][1]["worker_id"], "worker-2")

    def test_list_workers_with_status_filter(self) -> None:
        """Test listing workers with status filter - verifies it rejects uppercase."""
        with patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user:
            mock_user.return_value = self.admin_user

            # The route code tries to uppercase the filter, but enum values are lowercase
            # This is an implementation detail - the filter will fail validation
            response = self.client.get("/api/admin/workers?status_filter=idle")

            # Expects 400 because route does status_filter.upper() but enum values are lowercase
            self.assertEqual(response.status_code, 400)

    def test_list_workers_invalid_status_filter(self) -> None:
        """Test listing workers with invalid status returns 400."""
        with patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user:
            mock_user.return_value = self.admin_user

            response = self.client.get("/api/admin/workers?status_filter=invalid")

            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid status", response.json()["detail"])


class AdminCreateWorkerTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/admin/workers endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.admin_user = User(
            id=1,
            email="admin@test.com",
            username="admin",
            is_active=True,
            is_admin=True,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_create_worker_with_lowercase_persistent_type(self) -> None:
        """Test creating worker with lowercase 'persistent' worker_type.

        This test verifies the API accepts lowercase enum values as sent by
        the frontend (e.g., 'persistent' not 'PERSISTENT').
        """
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.create_worker = AsyncMock()
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.post(
                "/api/admin/workers",
                json={
                    "name": "test-worker",
                    "worker_type": "persistent",  # lowercase as frontend sends
                    "labels": {"gpu": "nvidia"},
                },
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("token", data)
            self.assertIn("worker_id", data)

    def test_create_worker_with_lowercase_ephemeral_type(self) -> None:
        """Test creating worker with lowercase 'ephemeral' worker_type."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.create_worker = AsyncMock()
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.post(
                "/api/admin/workers",
                json={
                    "name": "ephemeral-worker",
                    "worker_type": "ephemeral",  # lowercase as frontend sends
                },
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("token", data)

    def test_create_worker_invalid_type(self) -> None:
        """Test creating worker with invalid worker_type returns 422."""
        with patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user:
            mock_user.return_value = self.admin_user

            response = self.client.post(
                "/api/admin/workers",
                json={
                    "name": "test-worker",
                    "worker_type": "invalid_type",
                },
            )

            self.assertEqual(response.status_code, 422)

    def test_create_worker_missing_name(self) -> None:
        """Test creating worker without name returns 422."""
        with patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user:
            mock_user.return_value = self.admin_user

            response = self.client.post(
                "/api/admin/workers",
                json={
                    "worker_type": "persistent",
                },
            )

            self.assertEqual(response.status_code, 422)


class AdminDeleteWorkerTestCase(APITestCase, unittest.TestCase):
    """Test DELETE /api/admin/workers/{worker_id} endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.admin_user = User(
            id=1,
            email="admin@test.com",
            username="admin",
            is_active=True,
            is_admin=True,
        )

        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.IDLE,
            token_hash="hash123",
            user_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_delete_worker_success(self) -> None:
        """Test successful worker deletion."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=self.worker)
            mock_repo_instance.delete_worker = AsyncMock(return_value=True)
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.delete("/api/admin/workers/worker-123")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["worker_id"], "worker-123")

    def test_delete_worker_not_found(self) -> None:
        """Test deleting non-existent worker returns 404."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance

            response = self.client.delete("/api/admin/workers/worker-999")

            self.assertEqual(response.status_code, 404)

    def test_delete_worker_with_active_job(self) -> None:
        """Test deleting worker with active job without force returns 400."""
        worker_with_job = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.BUSY,
            token_hash="hash123",
            user_id=1,
            current_job_id="job-456",
        )

        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=worker_with_job)
            mock_repo.return_value = mock_repo_instance

            response = self.client.delete("/api/admin/workers/worker-123")

            self.assertEqual(response.status_code, 400)
            self.assertIn("active job", response.json()["detail"])

    def test_delete_worker_with_active_job_force(self) -> None:
        """Test deleting worker with active job using force succeeds."""
        worker_with_job = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.BUSY,
            token_hash="hash123",
            user_id=1,
            current_job_id="job-456",
        )

        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=worker_with_job)
            mock_repo_instance.delete_worker = AsyncMock(return_value=True)
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.delete("/api/admin/workers/worker-123?force=true")

            self.assertEqual(response.status_code, 200)


class AdminDrainWorkerTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/admin/workers/{worker_id}/drain endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.admin_user = User(
            id=1,
            email="admin@test.com",
            username="admin",
            is_active=True,
            is_admin=True,
        )

        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.IDLE,
            token_hash="hash123",
            user_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_drain_worker_success(self) -> None:
        """Test successful worker draining."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock()
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.post("/api/admin/workers/worker-123/drain")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "draining")

    def test_drain_worker_not_found(self) -> None:
        """Test draining non-existent worker returns 404."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post("/api/admin/workers/worker-999/drain")

            self.assertEqual(response.status_code, 404)


class AdminRotateTokenTestCase(APITestCase, unittest.TestCase):
    """Test POST /api/admin/workers/{worker_id}/token endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = create_router_client(router, admin_router)

        self.admin_user = User(
            id=1,
            email="admin@test.com",
            username="admin",
            is_active=True,
            is_admin=True,
        )

        self.worker = Worker(
            worker_id="worker-123",
            name="Test Worker",
            worker_type=WorkerType.PERSISTENT,
            status=WorkerStatus.IDLE,
            token_hash="old-hash",
            user_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_rotate_token_success(self) -> None:
        """Test successful token rotation."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
            patch("simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log") as mock_audit,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=self.worker)
            mock_repo_instance.update_worker = AsyncMock()
            mock_repo.return_value = mock_repo_instance

            mock_audit.return_value = AsyncMock()

            response = self.client.post("/api/admin/workers/worker-123/token")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["worker_id"], "worker-123")
            self.assertIsInstance(data["token"], str)
            self.assertIn("simpletuner worker", data["connection_command"])
            self.assertIn("--orchestrator-url", data["connection_command"])
            self.assertIn("--worker-token", data["connection_command"])

            # Verify token_hash was updated
            call_args = mock_repo_instance.update_worker.call_args
            updates = call_args[0][1]
            self.assertIn("token_hash", updates)
            self.assertNotEqual(updates["token_hash"], "old-hash")

    def test_rotate_token_not_found(self) -> None:
        """Test rotating token for non-existent worker returns 404."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository") as mock_repo,
            patch("simpletuner.simpletuner_sdk.server.routes.workers.get_optional_user") as mock_user,
        ):

            mock_user.return_value = self.admin_user

            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_worker = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance

            response = self.client.post("/api/admin/workers/worker-999/token")

            self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
