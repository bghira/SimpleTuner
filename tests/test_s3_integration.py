"""Tests for S3-compatible storage routes and cloud upload service.

Tests cover:
- S3-compatible PUT/GET/LIST operations
- Path traversal protection
- Token-based authentication
- Cloud upload service safety checks
- Archive creation and upload
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from tests.unittest_support import APITestCase


class TestS3StorageRoutes(APITestCase, unittest.TestCase):
    """Tests for S3-compatible storage API routes."""

    def setUp(self):
        super().setUp()
        self._upload_dir = tempfile.mkdtemp()
        # Patch the local upload dir to use temp directory
        self._upload_dir_patcher = patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.storage.get_local_upload_dir",
            return_value=Path(self._upload_dir),
        )
        self._upload_dir_patcher.start()
        # Mock UserStore.has_any_users to return False (single-user mode, no auth required)
        self._user_store_patcher = patch("simpletuner.simpletuner_sdk.server.routes.cloud.storage.UserStore")
        mock_user_store_cls = self._user_store_patcher.start()
        mock_user_store_cls.return_value.has_any_users = AsyncMock(return_value=False)

    def tearDown(self):
        self._user_store_patcher.stop()
        self._upload_dir_patcher.stop()
        # Clean up temp directory
        import shutil

        shutil.rmtree(self._upload_dir, ignore_errors=True)
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    # PUT route coverage lives in tests/test_s3_put_async.py to ensure AsyncJobStore runs in a real async context.

    def test_get_object_path_traversal_protection(self):
        """GET object should block path traversal attempts."""
        with self._get_client() as client:
            # Use URL-encoded path traversal (unencoded ../ gets normalized by URL parser)
            response = client.get("/api/cloud/storage/..%2Fetc/passwd")
            self.assertEqual(response.status_code, 400)

    def test_get_object_not_found(self):
        """GET object should return 404 for missing files."""
        with self._get_client() as client:
            response = client.get("/api/cloud/storage/test-bucket/missing-file.txt")
            self.assertEqual(response.status_code, 404)

    def test_list_buckets_empty(self):
        """LIST buckets should return empty list when no buckets exist."""
        with self._get_client() as client:
            response = client.get("/api/cloud/storage")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("Buckets", data)
            self.assertEqual(len(data["Buckets"]), 0)
            self.assertEqual(data["total_size"], 0)

    def test_list_buckets_with_content(self):
        """LIST buckets should return bucket info when content exists."""
        # Create a bucket with a file
        bucket_path = Path(self._upload_dir) / "test-bucket"
        bucket_path.mkdir()
        (bucket_path / "test-file.txt").write_bytes(b"test content")

        with self._get_client() as client:
            response = client.get("/api/cloud/storage")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data["Buckets"]), 1)
            self.assertEqual(data["Buckets"][0]["Name"], "test-bucket")
            self.assertEqual(data["Buckets"][0]["FileCount"], 1)
            self.assertGreater(data["Buckets"][0]["Size"], 0)

    def test_list_objects_in_bucket(self):
        """LIST objects should return files in bucket."""
        # Create a bucket with files
        bucket_path = Path(self._upload_dir) / "test-bucket"
        bucket_path.mkdir()
        (bucket_path / "file1.txt").write_bytes(b"content 1")
        (bucket_path / "file2.txt").write_bytes(b"content 2")

        with self._get_client() as client:
            response = client.get("/api/cloud/storage/test-bucket")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["Name"], "test-bucket")
            self.assertEqual(len(data["Contents"]), 2)
            keys = [obj["Key"] for obj in data["Contents"]]
            self.assertIn("file1.txt", keys)
            self.assertIn("file2.txt", keys)

    def test_list_objects_bucket_not_found(self):
        """LIST objects should return empty list for non-existent bucket."""
        with self._get_client() as client:
            response = client.get("/api/cloud/storage/nonexistent-bucket")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["Contents"], [])

    def test_list_bucket_path_traversal_protection(self):
        """LIST objects should block path traversal in bucket name."""
        with self._get_client() as client:
            response = client.get("/api/cloud/storage/..%2Fetc")
            self.assertEqual(response.status_code, 400)


class TestCloudUploadServiceSafety(unittest.TestCase):
    """Tests for CloudUploadService path safety functions."""

    def test_is_safe_path_forbidden_paths(self):
        """Should reject forbidden system paths."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        forbidden = ["/etc", "/var", "/usr", "/bin", "/root", "/proc", "/sys"]
        for path in forbidden:
            self.assertFalse(
                _is_safe_path(Path(path)),
                f"Path {path} should be forbidden",
            )

    def test_is_safe_path_sensitive_patterns(self):
        """Should reject paths containing sensitive patterns."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create paths with sensitive patterns
            sensitive = Path(tmpdir) / ".git"
            sensitive.mkdir()

            # Even if the path exists, it should be rejected
            self.assertFalse(_is_safe_path(sensitive))

    def test_is_safe_path_allowed_base(self):
        """Should enforce allowed base directory."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_path

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "allowed"
            base_dir.mkdir()
            child_path = base_dir / "subdir"
            child_path.mkdir()

            # Need to patch FORBIDDEN_PATHS to not include /tmp for this test
            # since tempfile.TemporaryDirectory() creates directories in /tmp
            with patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service.FORBIDDEN_PATHS",
                {"/etc", "/var", "/usr", "/bin", "/sbin", "/lib", "/root"},
            ):
                # Child of allowed base should pass
                self.assertTrue(_is_safe_path(child_path, allowed_base=base_dir))

                # Path outside allowed base should fail
                outside = Path(tmpdir) / "outside"
                outside.mkdir()
                self.assertFalse(_is_safe_path(outside, allowed_base=base_dir))

    def test_is_safe_file_hidden_files(self):
        """Should reject hidden files."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            hidden = base / ".hidden_file"
            hidden.touch()

            self.assertFalse(_is_safe_file(hidden, base))

    def test_is_safe_file_sensitive_extensions(self):
        """Should reject files with sensitive extensions."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            sensitive_exts = [".env", ".pem", ".key", ".crt", ".p12", ".pfx"]

            for ext in sensitive_exts:
                sensitive_file = base / f"secret{ext}"
                sensitive_file.touch()
                self.assertFalse(
                    _is_safe_file(sensitive_file, base),
                    f"Extension {ext} should be rejected",
                )

    def test_is_safe_file_symlink_protection(self):
        """Should reject symlinks pointing outside base directory."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import _is_safe_file

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            outside = Path(tmpdir) / "outside"
            outside.mkdir()
            target = outside / "target.txt"
            target.write_text("secret")

            symlink = base / "link.txt"
            try:
                symlink.symlink_to(target)
            except OSError:
                # Skip on systems where symlinks aren't supported
                return

            # Symlink to file outside base should be rejected
            self.assertFalse(_is_safe_file(symlink, base))


class TestCloudUploadServiceExtractPaths(unittest.TestCase):
    """Tests for CloudUploadService path extraction."""

    def test_extract_local_paths_basic(self):
        """Should extract local dataset paths from config."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        # Create test dir in home (not /tmp which is in FORBIDDEN_PATHS)
        test_base = Path.home() / ".cache" / "simpletuner_test"
        test_base.mkdir(parents=True, exist_ok=True)
        dataset_path = test_base / "dataset"
        dataset_path.mkdir(exist_ok=True)

        try:
            service = CloudUploadService()
            config = [
                {"type": "local", "instance_data_dir": str(dataset_path)},
            ]

            paths = service._extract_local_paths(config)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0], dataset_path.resolve())
        finally:
            import shutil

            shutil.rmtree(test_base, ignore_errors=True)

    def test_extract_local_paths_skips_non_local(self):
        """Should skip non-local dataset types."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        service = CloudUploadService()
        config = [
            {"type": "aws", "bucket": "my-bucket"},
            {"type": "huggingface", "dataset_name": "my-dataset"},
        ]

        paths = service._extract_local_paths(config)
        self.assertEqual(len(paths), 0)

    def test_extract_local_paths_skips_nonexistent(self):
        """Should skip paths that don't exist."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        service = CloudUploadService()
        config = [
            {"type": "local", "instance_data_dir": "/nonexistent/path"},
        ]

        paths = service._extract_local_paths(config)
        self.assertEqual(len(paths), 0)

    def test_extract_local_paths_rejects_forbidden(self):
        """Should reject forbidden system paths."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        service = CloudUploadService()
        config = [
            {"type": "local", "instance_data_dir": "/etc"},
            {"type": "local", "instance_data_dir": "/root"},
        ]

        paths = service._extract_local_paths(config)
        self.assertEqual(len(paths), 0)


class TestCloudUploadServiceUtilities(unittest.TestCase):
    """Tests for CloudUploadService utility methods."""

    def test_format_bytes(self):
        """Should format byte sizes correctly."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        self.assertEqual(CloudUploadService._format_bytes(0), "0.0 B")
        self.assertEqual(CloudUploadService._format_bytes(512), "512.0 B")
        self.assertEqual(CloudUploadService._format_bytes(1024), "1.0 KB")
        self.assertEqual(CloudUploadService._format_bytes(1536), "1.5 KB")
        self.assertEqual(CloudUploadService._format_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(CloudUploadService._format_bytes(1024 * 1024 * 1024), "1.0 GB")

    def test_has_local_data(self):
        """Should detect local data in config."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        # Create test dir in home (not /tmp which is in FORBIDDEN_PATHS)
        test_base = Path.home() / ".cache" / "simpletuner_test"
        test_base.mkdir(parents=True, exist_ok=True)
        dataset_path = test_base / "dataset"
        dataset_path.mkdir(exist_ok=True)

        try:
            service = CloudUploadService()

            # Config with local data
            config_with_local = [
                {"type": "local", "instance_data_dir": str(dataset_path)},
            ]
            self.assertTrue(service.has_local_data(config_with_local))

            # Config without local data
            config_without_local = [
                {"type": "aws", "bucket": "my-bucket"},
            ]
            self.assertFalse(service.has_local_data(config_without_local))
        finally:
            import shutil

            shutil.rmtree(test_base, ignore_errors=True)

    def test_estimate_upload_size(self):
        """Should estimate upload size correctly."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        # Create test dir in home (not /tmp which is in FORBIDDEN_PATHS)
        test_base = Path.home() / ".cache" / "simpletuner_test"
        test_base.mkdir(parents=True, exist_ok=True)
        dataset_path = test_base / "dataset"
        dataset_path.mkdir(exist_ok=True)

        try:
            # Create test files
            (dataset_path / "file1.txt").write_bytes(b"x" * 100)
            (dataset_path / "file2.txt").write_bytes(b"y" * 200)

            service = CloudUploadService()
            config = [
                {"type": "local", "instance_data_dir": str(dataset_path)},
            ]

            size = service.estimate_upload_size(config)
            self.assertEqual(size, 300)
        finally:
            import shutil

            shutil.rmtree(test_base, ignore_errors=True)


class TestReplicateUploadBackend(unittest.IsolatedAsyncioTestCase):
    """Tests for ReplicateUploadBackend."""

    async def test_upload_requires_token(self):
        """Upload should require REPLICATE_API_TOKEN."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import ReplicateUploadBackend

        with patch.object(ReplicateUploadBackend, "_token", None):
            backend = ReplicateUploadBackend()
            with self.assertRaises(ValueError) as ctx:
                await backend.upload_archive("/tmp/test.zip")
            self.assertIn("REPLICATE_API_TOKEN", str(ctx.exception))

    async def test_upload_handles_http_error(self):
        """Upload should handle HTTP errors gracefully."""
        import httpx

        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import ReplicateUploadBackend

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"test archive content")
            temp_path = f.name

        try:
            with patch.object(ReplicateUploadBackend, "_token", "test-token"):
                backend = ReplicateUploadBackend()

                # Mock HTTP client to return error
                mock_response = MagicMock()
                mock_response.status_code = 403
                mock_response.text = "Forbidden"
                mock_response.raise_for_status = MagicMock(
                    side_effect=httpx.HTTPStatusError(
                        "403 Forbidden",
                        request=MagicMock(),
                        response=mock_response,
                    )
                )

                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_client.is_closed = False

                with patch.object(backend, "_get_http_client", return_value=mock_client):
                    with self.assertRaises(ValueError) as ctx:
                        await backend.upload_archive(temp_path)
                    self.assertIn("Replicate upload error", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    async def test_upload_rejects_oversized_archive(self):
        """Upload should reject archives larger than the Replicate limit."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import ReplicateUploadBackend

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"0123456789")
            temp_path = f.name

        try:
            with patch.object(ReplicateUploadBackend, "_token", "test-token"):
                backend = ReplicateUploadBackend()
                with patch.object(ReplicateUploadBackend, "MAX_UPLOAD_BYTES", 1):
                    with self.assertRaises(ValueError) as ctx:
                        await backend.upload_archive(temp_path)
                    self.assertIn("upload limit", str(ctx.exception))
        finally:
            os.unlink(temp_path)


class TestCloudUploadServiceUploadLimit(unittest.TestCase):
    """Tests for CloudUploadService upload limits."""

    def test_estimated_size_blocks_oversized_archive(self):
        """Estimated dataset size should enforce the upload limit."""
        from simpletuner.simpletuner_sdk.server.services.cloud.cloud_upload_service import CloudUploadService

        with self.assertRaises(ValueError) as ctx:
            CloudUploadService._raise_if_exceeds_upload_limit(total_bytes=2, max_upload_bytes=1)

        self.assertIn("upload limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
