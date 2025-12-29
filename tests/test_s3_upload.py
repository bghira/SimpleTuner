"""Tests for S3-compatible local upload endpoints.

Tests the S3 upload API:
- Authentication via X-Upload-Token header
- Path traversal prevention
- File upload and retrieval
- Bucket listing
- Object listing
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class MockRequest:
    """Mock FastAPI Request object."""

    def __init__(
        self,
        headers: dict = None,
        body: bytes = b"",
        client_host: str = "127.0.0.1",
    ):
        self.headers = headers or {}
        self._body = body
        self.client = MagicMock()
        self.client.host = client_host

    async def body(self) -> bytes:
        return self._body


class MockJob:
    """Mock job record."""

    def __init__(self, job_id: str = "test-job-123"):
        self.job_id = job_id


class TestS3PutObjectAuthentication(unittest.TestCase):
    """Test S3 PUT endpoint authentication."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.patcher_dir = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_local_upload_dir",
            return_value=Path(self.temp_dir),
        )
        self.patcher_dir.start()

    def tearDown(self):
        """Clean up."""
        self.patcher_dir.stop()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_put_without_token_returns_401(self):
        """Test PUT without authentication token returns 401."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(headers={}, body=b"test content")

        # Mock job store to return None (no valid job for token)
        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=None)

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(s3_put_object("test-bucket", "test.txt", request))
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("Authentication required", ctx.exception.detail)

    def test_put_with_invalid_token_returns_401(self):
        """Test PUT with invalid token returns 401."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(
            headers={"X-Upload-Token": "invalid-token"},
            body=b"test content",
        )

        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=None)

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(s3_put_object("test-bucket", "test.txt", request))
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("Invalid upload token", ctx.exception.detail)

    def test_put_with_valid_token_succeeds(self):
        """Test PUT with valid token succeeds."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(
            headers={"X-Upload-Token": "valid-token-123"},
            body=b"test content here",
        )

        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=MockJob("job-456"))

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            result = asyncio.get_event_loop().run_until_complete(s3_put_object("test-bucket", "test.txt", request))

        self.assertIn("ETag", result)
        self.assertEqual(result["Key"], "test.txt")
        self.assertEqual(result["Bucket"], "test-bucket")

        # Verify file was created
        file_path = Path(self.temp_dir) / "test-bucket" / "test.txt"
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_bytes(), b"test content here")

    def test_x_simpletuner_secret_header_also_works(self):
        """Test X-SimpleTuner-Secret header is accepted for auth."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(
            headers={"X-SimpleTuner-Secret": "valid-token-123"},
            body=b"secret content",
        )

        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=MockJob("job-789"))

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            result = asyncio.get_event_loop().run_until_complete(s3_put_object("bucket2", "secret.bin", request))

        self.assertIn("ETag", result)


class TestS3PathTraversalPrevention(unittest.TestCase):
    """Test path traversal attack prevention."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.patcher_dir = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_local_upload_dir",
            return_value=Path(self.temp_dir),
        )
        self.patcher_dir.start()

    def tearDown(self):
        """Clean up."""
        self.patcher_dir.stop()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_put_with_dotdot_in_bucket_blocked(self):
        """Test PUT with .. in bucket name is blocked."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(
            headers={"X-Upload-Token": "valid"},
            body=b"malicious",
        )

        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=MockJob())

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(s3_put_object("../escape", "file.txt", request))
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Invalid", ctx.exception.detail)

    def test_put_with_dotdot_in_key_blocked(self):
        """Test PUT with .. in key is blocked."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_put_object

        request = MockRequest(
            headers={"X-Upload-Token": "valid"},
            body=b"malicious",
        )

        mock_store = MagicMock()
        mock_store.get_job_by_upload_token = AsyncMock(return_value=MockJob())

        with patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_job_store",
            return_value=mock_store,
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(s3_put_object("bucket", "../../../etc/passwd", request))
            self.assertEqual(ctx.exception.status_code, 400)

    def test_get_with_dotdot_in_bucket_blocked(self):
        """Test GET with .. in bucket is blocked."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_get_object

        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(s3_get_object("../escape", "file.txt"))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_get_with_dotdot_in_key_blocked(self):
        """Test GET with .. in key is blocked."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_get_object

        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(s3_get_object("bucket", "../../etc/passwd"))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_list_objects_with_path_traversal_blocked(self):
        """Test list objects with path traversal is blocked."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_list_objects

        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(s3_list_objects("../escape"))
        self.assertEqual(ctx.exception.status_code, 400)


class TestS3GetObject(unittest.TestCase):
    """Test S3 GET object endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.patcher_dir = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_local_upload_dir",
            return_value=Path(self.temp_dir),
        )
        self.patcher_dir.start()

    def tearDown(self):
        """Clean up."""
        self.patcher_dir.stop()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_existing_object_returns_content(self):
        """Test GET existing object returns its content."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_get_object

        # Create a file
        bucket_dir = Path(self.temp_dir) / "mybucket"
        bucket_dir.mkdir()
        (bucket_dir / "myfile.txt").write_bytes(b"hello world")

        result = asyncio.get_event_loop().run_until_complete(s3_get_object("mybucket", "myfile.txt"))

        self.assertEqual(result, b"hello world")

    def test_get_nonexistent_object_returns_404(self):
        """Test GET nonexistent object returns 404."""
        import asyncio

        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_get_object

        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(s3_get_object("nobucket", "nofile.txt"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_nested_object_path(self):
        """Test GET object in nested path."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_get_object

        # Create nested structure
        nested_dir = Path(self.temp_dir) / "bucket" / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        (nested_dir / "deep.txt").write_bytes(b"deep content")

        result = asyncio.get_event_loop().run_until_complete(s3_get_object("bucket", "subdir/nested/deep.txt"))

        self.assertEqual(result, b"deep content")


class TestS3ListBuckets(unittest.TestCase):
    """Test S3 list buckets endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.patcher_dir = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_local_upload_dir",
            return_value=Path(self.temp_dir),
        )
        self.patcher_dir.start()

    def tearDown(self):
        """Clean up."""
        self.patcher_dir.stop()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_empty_returns_empty_list(self):
        """Test list buckets with no buckets returns empty."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_list_buckets

        result = asyncio.get_event_loop().run_until_complete(s3_list_buckets())

        self.assertEqual(result["Buckets"], [])
        self.assertEqual(result["total_size"], 0)

    def test_list_buckets_returns_bucket_info(self):
        """Test list buckets returns bucket metadata."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_list_buckets

        # Create buckets with files
        bucket1 = Path(self.temp_dir) / "bucket1"
        bucket1.mkdir()
        (bucket1 / "file1.txt").write_bytes(b"abc")  # 3 bytes
        (bucket1 / "file2.txt").write_bytes(b"defgh")  # 5 bytes

        bucket2 = Path(self.temp_dir) / "bucket2"
        bucket2.mkdir()
        (bucket2 / "big.bin").write_bytes(b"x" * 100)  # 100 bytes

        result = asyncio.get_event_loop().run_until_complete(s3_list_buckets())

        self.assertEqual(len(result["Buckets"]), 2)
        self.assertEqual(result["total_size"], 108)

        # Check bucket names
        bucket_names = [b["Name"] for b in result["Buckets"]]
        self.assertIn("bucket1", bucket_names)
        self.assertIn("bucket2", bucket_names)


class TestS3ListObjects(unittest.TestCase):
    """Test S3 list objects endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.patcher_dir = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.s3.get_local_upload_dir",
            return_value=Path(self.temp_dir),
        )
        self.patcher_dir.start()

    def tearDown(self):
        """Clean up."""
        self.patcher_dir.stop()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_objects_in_empty_bucket(self):
        """Test list objects in empty bucket returns empty list."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_list_objects

        result = asyncio.get_event_loop().run_until_complete(s3_list_objects("nonexistent"))

        self.assertEqual(result["Contents"], [])
        self.assertEqual(result["Name"], "nonexistent")

    def test_list_objects_returns_object_metadata(self):
        """Test list objects returns object details."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.routes.cloud.s3 import s3_list_objects

        # Create bucket with files
        bucket = Path(self.temp_dir) / "mybucket"
        bucket.mkdir()
        (bucket / "file1.txt").write_bytes(b"content1")
        subdir = bucket / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_bytes(b"content2content2")

        result = asyncio.get_event_loop().run_until_complete(s3_list_objects("mybucket"))

        self.assertEqual(len(result["Contents"]), 2)

        # Check keys include relative paths
        keys = [obj["Key"] for obj in result["Contents"]]
        self.assertIn("file1.txt", keys)
        self.assertIn("subdir/file2.txt", keys)

        # Check sizes
        for obj in result["Contents"]:
            self.assertIn("Size", obj)
            self.assertIn("LastModified", obj)


if __name__ == "__main__":
    unittest.main()
