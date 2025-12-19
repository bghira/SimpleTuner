"""Tests for dataset upload, folder creation, and caption endpoints."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults, WebUIStateStore
from tests.unittest_support import APITestCase


def _png_bytes(color: str = "red", size: tuple[int, int] = (16, 16)) -> bytes:
    """Generate a tiny PNG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class DatasetUploadAndCaptionRoutesTestCase(APITestCase, unittest.TestCase):
    """Exercise new dataset upload, zip, and captioning endpoints."""

    def setUp(self) -> None:
        super().setUp()
        self._home_tmpdir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._home_tmpdir.name)
        self.datasets_dir = self.temp_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Patch HOME so the state store persists under the temp root
        self._home_patch = patch.dict(os.environ, {"HOME": str(self.temp_dir)}, clear=False)
        self._home_patch.start()

        # Ensure WebUI defaults point at the temp datasets dir
        self.state_store = WebUIStateStore()
        defaults = WebUIDefaults(
            datasets_dir=str(self.datasets_dir),
            allow_dataset_paths_outside_dir=False,
        )
        self.state_store.save_defaults(defaults)

        # Patch the route-level state store singleton
        self._store_patch = patch(
            "simpletuner.simpletuner_sdk.server.routes.datasets.WebUIStateStore",
            return_value=self.state_store,
        )
        self._store_patch.start()

        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        self._store_patch.stop()
        self._home_patch.stop()
        super().tearDown()
        self._home_tmpdir.cleanup()

    def test_create_folder_and_reject_invalid(self) -> None:
        """Creating a new folder succeeds; path traversal is rejected."""
        response = self.client.post(
            "/api/datasets/folders",
            data={"parent_path": str(self.datasets_dir), "folder_name": "new-dataset"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertTrue((self.datasets_dir / "new-dataset").is_dir())

        bad_response = self.client.post(
            "/api/datasets/folders",
            data={"parent_path": str(self.datasets_dir), "folder_name": "../hack"},
        )
        self.assertEqual(bad_response.status_code, 400)
        self.assertIn("Folder", bad_response.json()["detail"])

    def test_upload_files_accepts_images_and_rejects_unknown_types(self) -> None:
        """File uploads should save allowed files and flag unsupported types."""
        png_payload = _png_bytes()
        ok_response = self.client.post(
            "/api/datasets/upload",
            data={"target_path": str(self.datasets_dir)},
            files=[("files", ("sample.png", png_payload, "image/png"))],
        )
        self.assertEqual(ok_response.status_code, 200)
        ok_body = ok_response.json()
        self.assertTrue(ok_body["success"])
        self.assertEqual(ok_body["files_uploaded"], 1)
        self.assertTrue((self.datasets_dir / "sample.png").exists())

        bad_response = self.client.post(
            "/api/datasets/upload",
            data={"target_path": str(self.datasets_dir)},
            files=[("files", ("malicious.exe", b"nope", "application/octet-stream"))],
        )
        self.assertEqual(bad_response.status_code, 200)
        bad_body = bad_response.json()
        self.assertFalse(bad_body["success"])
        self.assertEqual(bad_body["files_uploaded"], 0)
        self.assertTrue(any("Unsupported file type" in err for err in bad_body["errors"]))

    def test_upload_zip_extracts_allowed_files_and_skips_unsafe_entries(self) -> None:
        """ZIP uploads should extract allowed files and skip traversal/unsupported ones."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("image.png", _png_bytes())
            zf.writestr("../escape.txt", "nope")  # traversal attempt
            zf.writestr("note.exe", "deny")  # unsupported extension
        zip_buffer.seek(0)

        response = self.client.post(
            "/api/datasets/upload/zip",
            data={"target_path": str(self.datasets_dir)},
            files=[("file", ("archive.zip", zip_buffer.read(), "application/zip"))],
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        # Only the image is accepted
        self.assertEqual(body["files_uploaded"], 1)
        self.assertEqual(body["files_skipped"], 2)
        self.assertTrue((self.datasets_dir / "image.png").exists())
        self.assertFalse((self.datasets_dir / "escape.txt").exists())
        self.assertFalse((self.datasets_dir / "note.exe").exists())

    def test_upload_zip_rejects_non_zip_payload(self) -> None:
        """Non-zip uploads to the zip endpoint return a 400."""
        response = self.client.post(
            "/api/datasets/upload/zip",
            data={"target_path": str(self.datasets_dir)},
            files=[("file", ("not-a-zip.txt", b"plain", "text/plain"))],
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("ZIP", response.json()["detail"])

    def test_upload_zip_preserves_subfolder_structure(self) -> None:
        """ZIP uploads should preserve the subfolder structure from the archive."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("subdir1/image1.png", _png_bytes("red"))
            zf.writestr("subdir1/nested/image2.png", _png_bytes("blue"))
            zf.writestr("subdir2/image3.png", _png_bytes("green"))
            zf.writestr("root_image.png", _png_bytes("yellow"))
        zip_buffer.seek(0)

        response = self.client.post(
            "/api/datasets/upload/zip",
            data={"target_path": str(self.datasets_dir)},
            files=[("file", ("archive.zip", zip_buffer.read(), "application/zip"))],
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["files_uploaded"], 4)
        self.assertEqual(body["files_skipped"], 0)

        # Verify subfolder structure was preserved
        self.assertTrue((self.datasets_dir / "subdir1" / "image1.png").exists())
        self.assertTrue((self.datasets_dir / "subdir1" / "nested" / "image2.png").exists())
        self.assertTrue((self.datasets_dir / "subdir2" / "image3.png").exists())
        self.assertTrue((self.datasets_dir / "root_image.png").exists())

    def test_caption_status_thumbnails_and_writes(self) -> None:
        """Caption endpoints should report coverage, generate thumbnails, and write captions."""
        img1 = self.datasets_dir / "one.png"
        img2 = self.datasets_dir / "two.png"
        img1.write_bytes(_png_bytes("blue"))
        img2.write_bytes(_png_bytes("green"))
        (self.datasets_dir / "one.txt").write_text("existing caption", encoding="utf-8")

        status_resp = self.client.get("/api/datasets/captions/status", params={"path": str(self.datasets_dir)})
        self.assertEqual(status_resp.status_code, 200)
        status_body = status_resp.json()
        self.assertEqual(status_body["total_images"], 2)
        self.assertEqual(status_body["with_caption"], 1)
        self.assertEqual(status_body["without_caption"], 1)
        self.assertGreater(status_body["coverage_ratio"], 0)

        thumb_resp = self.client.get(
            "/api/datasets/captions/thumbnails",
            params={"path": str(self.datasets_dir), "limit": 1, "offset": 0},
        )
        self.assertEqual(thumb_resp.status_code, 200)
        thumbs: List[Dict[str, str]] = thumb_resp.json()
        self.assertEqual(len(thumbs), 1)
        self.assertTrue(thumbs[0]["thumbnail"].startswith("data:image"))
        self.assertEqual(Path(thumbs[0]["path"]).name, "two.png")

        save_resp = self.client.post(
            "/api/datasets/captions",
            json={"captions": {str(img2): "new caption"}},
        )
        self.assertEqual(save_resp.status_code, 200)
        save_body = save_resp.json()
        self.assertTrue(save_body["success"])
        self.assertEqual(save_body["files_written"], 1)
        self.assertTrue(img2.with_suffix(".txt").exists())
        self.assertEqual(img2.with_suffix(".txt").read_text(encoding="utf-8"), "new caption")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
