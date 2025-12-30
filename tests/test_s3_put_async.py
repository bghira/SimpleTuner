"""Async tests for S3 PUT routes.

These tests use async test patterns to properly handle AsyncJobStore
and avoid the async context issues that occur with FastAPI TestClient.

Tests cover:
- PUT object with valid upload token
- PUT object requires authentication
- PUT object with invalid token
- PUT object path traversal protection
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient

from simpletuner.simpletuner_sdk.server import ServerMode, create_app
from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.base import UnifiedJob
from tests.unittest_support import AsyncAPITestCase


class TestS3PutRoutesAsync(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    """Async tests for S3 PUT routes."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        await super().asyncSetUp()

        # Create temp upload directory
        self._upload_dir = tempfile.mkdtemp()
        self._upload_dir_patcher = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.storage.get_local_upload_dir",
            return_value=Path(self._upload_dir),
        )
        self._upload_dir_patcher.start()

        # Reset AsyncJobStore singleton
        AsyncJobStore._instance = None

        # Create app
        self.app = create_app(mode=ServerMode.UNIFIED)

        # Initialize AsyncJobStore with test database
        db_path = self.tmp_path / "test_jobs.db"
        self.job_store = await AsyncJobStore.get_instance(str(db_path))

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        self._upload_dir_patcher.stop()

        # Clean up job store
        if AsyncJobStore._instance:
            await AsyncJobStore._instance.close()
            AsyncJobStore._instance = None

        # Clean up temp directory
        import shutil

        shutil.rmtree(self._upload_dir, ignore_errors=True)

        await super().asyncTearDown()

    async def _create_test_job_with_token(self, job_id: str, upload_token: str) -> UnifiedJob:
        """Create a test job with upload token."""
        from datetime import datetime, timezone

        from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType

        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="test-provider",
            status="running",
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            user_id=1,
            upload_token=upload_token,
        )
        await self.job_store.add_job(job)
        return job

    async def test_put_object_requires_authentication(self):
        """PUT object should require authentication via upload token."""
        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/test-bucket/test-file.txt",
                content=b"test content",
            )

            self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
            data = response.json()
            self.assertIn("Authentication required", data["detail"])
            self.assertIn("X-Upload-Token", data["detail"])

    async def test_put_object_invalid_token(self):
        """PUT object should reject invalid upload tokens."""
        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/test-bucket/test-file.txt",
                content=b"test content",
                headers={"X-Upload-Token": "invalid-token-12345"},
            )

            self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
            data = response.json()
            self.assertIn("Invalid upload token", data["detail"])

    async def test_put_object_with_valid_token(self):
        """PUT object should succeed with valid upload token."""
        # Create a job with upload token
        upload_token = "valid-token-abc123"
        await self._create_test_job_with_token("job-123", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/test-bucket/test-file.txt",
                content=b"test content",
                headers={"X-Upload-Token": upload_token},
            )

            self.assertEqual(response.status_code, status.HTTP_200_OK)
            data = response.json()

            # Check response structure
            self.assertIn("ETag", data)
            self.assertEqual(data["Key"], "test-file.txt")
            self.assertEqual(data["Bucket"], "test-bucket")

            # Verify file was written
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "test-bucket" / "test-file.txt"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_bytes(), b"test content")

    async def test_put_object_alternative_header(self):
        """PUT object should accept X-SimpleTuner-Secret header."""
        upload_token = "secret-token-xyz789"
        await self._create_test_job_with_token("job-456", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/bucket-2/file.bin",
                content=b"binary data",
                headers={"X-SimpleTuner-Secret": upload_token},
            )

            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Verify file was written
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "bucket-2" / "file.bin"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_bytes(), b"binary data")

    async def test_put_object_path_traversal_in_bucket(self):
        """PUT object should block path traversal attempts in bucket name."""
        upload_token = "token-traversal-test"
        await self._create_test_job_with_token("job-traversal", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            # Attempt path traversal with ".." in bucket name
            # Note: The .. check happens in the handler, not in routing
            response = await client.put(
                "/api/cloud/storage/..%2Fetc/passwd",  # URL-encoded path traversal
                content=b"malicious",
                headers={"X-Upload-Token": upload_token},
            )

            # Should reject with 400 or 404 (depending on URL normalization)
            self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    async def test_put_object_path_traversal_in_key(self):
        """PUT object should block path traversal attempts in key."""
        upload_token = "token-traversal-key"
        await self._create_test_job_with_token("job-traversal-key", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            # Attempt path traversal with ".." in key
            # The ".." check in the handler should catch this
            response = await client.put(
                "/api/cloud/storage/bucket/..%2F..%2Fetc%2Fpasswd",  # URL-encoded path traversal
                content=b"malicious",
                headers={"X-Upload-Token": upload_token},
            )

            # Should reject with 400 or 404 (depending on URL normalization and routing)
            self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    async def test_put_object_creates_parent_directories(self):
        """PUT object should create parent directories as needed."""
        upload_token = "token-nested-dirs"
        await self._create_test_job_with_token("job-nested", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/bucket/subdir1/subdir2/file.txt",
                content=b"nested file",
                headers={"X-Upload-Token": upload_token},
            )

            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Verify file was created in nested directories
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "bucket" / "subdir1" / "subdir2" / "file.txt"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_bytes(), b"nested file")

    async def test_put_object_overwrites_existing_file(self):
        """PUT object should overwrite existing files."""
        upload_token = "token-overwrite"
        await self._create_test_job_with_token("job-overwrite", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First upload
            response = await client.put(
                "/api/cloud/storage/bucket/overwrite.txt",
                content=b"original content",
                headers={"X-Upload-Token": upload_token},
            )
            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Second upload (overwrite)
            response = await client.put(
                "/api/cloud/storage/bucket/overwrite.txt",
                content=b"new content",
                headers={"X-Upload-Token": upload_token},
            )
            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Verify file contains new content
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "bucket" / "overwrite.txt"
            self.assertEqual(file_path.read_bytes(), b"new content")

    async def test_put_object_handles_large_files(self):
        """PUT object should handle large file uploads."""
        upload_token = "token-large-file"
        await self._create_test_job_with_token("job-large", upload_token)

        # Create 1MB of data
        large_content = b"x" * (1024 * 1024)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/bucket/large.bin",
                content=large_content,
                headers={"X-Upload-Token": upload_token},
            )

            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Verify file size
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "bucket" / "large.bin"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.stat().st_size, 1024 * 1024)

    async def test_put_object_empty_file(self):
        """PUT object should handle empty files."""
        upload_token = "token-empty"
        await self._create_test_job_with_token("job-empty", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/cloud/storage/bucket/empty.txt",
                content=b"",
                headers={"X-Upload-Token": upload_token},
            )

            self.assertEqual(response.status_code, status.HTTP_200_OK)

            # Verify empty file was created
            upload_dir = Path(self._upload_dir)
            file_path = upload_dir / "bucket" / "empty.txt"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.stat().st_size, 0)

    async def test_put_object_etag_generation(self):
        """PUT object should generate consistent ETag for same content."""
        upload_token = "token-etag"
        await self._create_test_job_with_token("job-etag", upload_token)

        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Upload same content twice
            response1 = await client.put(
                "/api/cloud/storage/bucket/file1.txt",
                content=b"test content",
                headers={"X-Upload-Token": upload_token},
            )

            response2 = await client.put(
                "/api/cloud/storage/bucket/file2.txt",
                content=b"test content",
                headers={"X-Upload-Token": upload_token},
            )

            # ETags should be the same for identical content
            etag1 = response1.json()["ETag"]
            etag2 = response2.json()["ETag"]
            self.assertEqual(etag1, etag2)

            # Different content should have different ETag
            response3 = await client.put(
                "/api/cloud/storage/bucket/file3.txt",
                content=b"different content",
                headers={"X-Upload-Token": upload_token},
            )

            etag3 = response3.json()["ETag"]
            self.assertNotEqual(etag1, etag3)


if __name__ == "__main__":
    unittest.main()
