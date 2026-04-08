"""Tests for DatasetViewerService cache file discovery and summary generation."""

import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import (
    DatasetViewerService,
    _data_backend_cache,
    generate_thumbnail,
    generate_thumbnail_from_pil,
)


class TestFindCacheFile(unittest.TestCase):
    """Verify _find_cache_file locates cache files across all backend types."""

    def setUp(self):
        self.service = DatasetViewerService()
        self.tmpdir = tempfile.mkdtemp()

    def _write_cache(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["a.jpg", "b.jpg"]},
                    "filtering_statistics": {"total_processed": 2, "skipped": {}},
                }
            )
        )

    def test_finds_cache_in_instance_data_dir(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertEqual(result, cache_file)

    def test_finds_cache_in_cache_dir(self):
        cache_dir = Path(self.tmpdir) / "cache_vae"
        cache_file = cache_dir / "aspect_ratio_bucket_indices_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "cache_dir": str(cache_dir)}
        result = self.service._find_cache_file(config, "indices")
        self.assertEqual(result, cache_file)

    def test_finds_cache_in_huggingface_metadata_dir(self):
        """HuggingFace backends store cache at cache/huggingface/{id}/huggingface_metadata/{id}/."""
        hf_dir = Path(self.tmpdir) / "cache" / "huggingface" / "hf-ds" / "huggingface_metadata" / "hf-ds"
        cache_file = hf_dir / "aspect_ratio_bucket_indices_hf-ds.json"
        self._write_cache(cache_file)

        config = {"id": "hf-ds"}
        # The service uses relative Path("cache"), so chdir to tmpdir
        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            result = self.service._find_cache_file(config, "indices")
            self.assertIsNotNone(result)
            self.assertEqual(result.resolve(), cache_file.resolve())
        finally:
            os.chdir(original_cwd)

    def test_finds_metadata_file_type(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_metadata_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "metadata")
        self.assertEqual(result, cache_file)

    def test_returns_none_when_no_cache_exists(self):
        config = {"id": "nonexistent-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertIsNone(result)

    def test_returns_none_for_empty_id(self):
        config = {"id": "", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertIsNone(result)

    def test_fallback_rglob_under_cache_dir(self):
        """Deeply nested cache files should be found via rglob fallback."""
        nested_dir = Path(self.tmpdir) / "cache" / "deep" / "nested" / "path"
        cache_file = nested_dir / "aspect_ratio_bucket_indices_deep-ds.json"
        self._write_cache(cache_file)

        config = {"id": "deep-ds"}
        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            result = self.service._find_cache_file(config, "indices")
            self.assertIsNotNone(result)
            self.assertEqual(result.resolve(), cache_file.resolve())
        finally:
            os.chdir(original_cwd)


class TestGetDatasetSummary(unittest.TestCase):
    """Verify summary generation from cache files."""

    def setUp(self):
        self.service = DatasetViewerService()
        self.tmpdir = tempfile.mkdtemp()

    def test_summary_with_no_cache(self):
        config = {"id": "no-cache", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertEqual(summary.dataset_id, "no-cache")
        self.assertFalse(summary.has_cache)
        self.assertEqual(summary.total_files, 0)

    def test_summary_with_valid_cache(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_test-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {
                        "1.0": ["a.jpg", "b.jpg", "c.jpg"],
                        "0.75": ["d.jpg", "e.jpg"],
                    },
                    "filtering_statistics": {
                        "total_processed": 10,
                        "skipped": {"too_small": 5},
                    },
                }
            )
        )

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertTrue(summary.has_cache)
        self.assertEqual(summary.total_files, 5)
        self.assertEqual(len(summary.buckets), 2)
        self.assertEqual(summary.filtering_statistics["total_processed"], 10)

    def test_summary_with_empty_buckets(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_empty-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {},
                    "filtering_statistics": {
                        "total_processed": 50,
                        "skipped": {"metadata_missing": 50},
                    },
                }
            )
        )

        config = {"id": "empty-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertTrue(summary.has_cache)
        self.assertEqual(summary.total_files, 0)
        self.assertEqual(len(summary.buckets), 0)

    def test_summary_with_corrupt_json(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_bad-ds.json"
        cache_file.write_text("not valid json{{{")

        config = {"id": "bad-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertFalse(summary.has_cache)


class TestThumbnailGeneration(unittest.TestCase):
    """Verify thumbnail generation from file paths and PIL images."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _create_test_image(self, size=(100, 100), mode="RGB") -> Path:
        img = Image.new(mode, size, (128, 64, 32))
        path = Path(self.tmpdir) / "test.jpg"
        img.save(path, format="JPEG")
        return path

    def test_generate_thumbnail_from_file(self):
        path = self._create_test_image()
        result = generate_thumbnail(path)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))

    def test_generate_thumbnail_from_pil_image(self):
        img = Image.new("RGB", (200, 200), (255, 0, 0))
        result = generate_thumbnail_from_pil(img)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))

    def test_generate_thumbnail_from_rgba(self):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = generate_thumbnail_from_pil(img)
        self.assertIsNotNone(result)

    def test_generate_thumbnail_nonexistent_file(self):
        result = generate_thumbnail(Path("/nonexistent/image.jpg"))
        self.assertIsNone(result)


class TestLocalThumbnails(unittest.TestCase):
    """Verify get_dataset_thumbnails for local backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService()

    def _setup_local_dataset(self):
        """Create a local dataset with an image and bucket cache."""
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        img.save(Path(self.tmpdir) / "photo.jpg", format="JPEG")

        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_local-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["photo.jpg"]},
                }
            )
        )
        return {"id": "local-ds", "type": "local", "instance_data_dir": self.tmpdir}

    def test_local_thumbnail_generated(self):
        config = self._setup_local_dataset()
        result = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].thumbnail)
        self.assertIsNone(result[0].error)

    def test_local_missing_file_returns_error(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_missing-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["gone.jpg"]},
                }
            )
        )
        config = {"id": "missing-ds", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].error, "File not found")


class TestHuggingfaceDisplayLabels(unittest.TestCase):
    """Verify viewer labels use dataset row metadata for Hugging Face backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

    def _write_hf_cache(
        self, dataset_id: str, indices: Dict[str, List[str]], metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        hf_dir = Path(self.tmpdir) / "cache" / "huggingface" / dataset_id / "huggingface_metadata" / dataset_id
        hf_dir.mkdir(parents=True, exist_ok=True)
        (hf_dir / f"aspect_ratio_bucket_indices_{dataset_id}.json").write_text(
            json.dumps({"config": {}, "aspect_ratio_bucket_indices": indices})
        )
        if metadata is not None:
            (hf_dir / f"aspect_ratio_bucket_metadata_{dataset_id}.json").write_text(json.dumps(metadata))

    def test_huggingface_thumbnail_uses_row_filename_label(self):
        dataset_id = "hf-ds"
        self._write_hf_cache(dataset_id, {"1.0": ["1.jpg"]})
        config = {"id": dataset_id, "type": "huggingface", "image_column": "image"}

        backend = MagicMock()
        backend._get_index_from_path.return_value = 1
        backend.get_dataset_item.return_value = {"file_name": "nested/path/real_photo.png"}

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with (
                patch.object(self.service, "_get_or_create_data_backend", return_value=backend),
                patch.object(self.service, "_read_remote_thumbnail", return_value=None),
            ):
                thumbnails = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        finally:
            os.chdir(original_cwd)

        self.assertEqual(len(thumbnails), 1)
        self.assertEqual(thumbnails[0].path, "1.jpg")
        self.assertEqual(thumbnails[0].filename, "real_photo.png")

    def test_huggingface_file_metadata_exposes_display_name(self):
        dataset_id = "hf-meta"
        self._write_hf_cache(
            dataset_id,
            {"1.0": ["7.jpg"]},
            metadata={
                "7.jpg": {
                    "original_size": [1024, 768],
                    "aspect_ratio": 1.3333,
                }
            },
        )
        config = {"id": dataset_id, "type": "huggingface", "image_column": "image"}

        backend = MagicMock()
        backend._get_index_from_path.return_value = 7
        backend.get_dataset_item.return_value = {"image": {"path": "/data/archive/folder/actual_name.webp"}}

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with patch.object(self.service, "_get_or_create_data_backend", return_value=backend):
                metadata = self.service.get_file_metadata(config, "7.jpg")
        finally:
            os.chdir(original_cwd)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.path, "7.jpg")
        self.assertEqual(metadata.display_name, "actual_name.webp")


class TestImagePreview(unittest.TestCase):
    """Verify get_image_preview crop simulation pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

    def _create_image(self, size=(800, 600)):
        img = Image.new("RGB", size, (128, 64, 32))
        path = Path(self.tmpdir) / "photo.jpg"
        img.save(path, format="JPEG")
        return path

    def test_preview_without_metadata(self):
        self._create_image()
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_image_preview(config, "photo.jpg")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["original"])
        self.assertIsNone(result["intermediary"])
        self.assertIsNone(result["cropped"])

    def test_preview_with_crop_metadata(self):
        self._create_image(size=(2000, 1500))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [1200, 900],
            "crop_coordinates": [100, 50],
            "target_size": [1024, 768],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["original"])
        self.assertIsNotNone(result["intermediary"])
        self.assertIsNotNone(result["cropped"])
        # All should be base64 data URLs
        for key in ("original", "intermediary", "cropped"):
            self.assertTrue(result[key].startswith("data:image/jpeg;base64,"))

    def test_preview_with_zero_crop_offset(self):
        """Center or corner crop with (0,0) offset should work."""
        self._create_image(size=(1024, 1024))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [1024, 1024],
            "crop_coordinates": [0, 0],
            "target_size": [1024, 1024],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])

    def test_preview_missing_file(self):
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_image_preview(config, "missing.jpg")
        self.assertIsNone(result)

    def test_preview_partial_metadata_skips_crop(self):
        """If only some crop fields are present, skip simulation."""
        self._create_image()
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {"intermediary_size": [400, 300]}  # missing crop_coordinates and target_size
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["original"])
        self.assertIsNone(result["intermediary"])
        self.assertIsNone(result["cropped"])

    def test_preview_clamps_crop_to_intermediary_bounds(self):
        """Crop coordinates that would exceed intermediary bounds get clamped.

        Reproduces the user-reported case: 810x1080 original, 512x682 intermediary,
        512x512 target at crop_coordinates (0, 143) in (left, top) order.
        The batch crop path in training_sample.py stores (crop_x, crop_y) = (left, top).
        """
        self._create_image(size=(810, 1080))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [512, 682],
            "crop_coordinates": [0, 143],  # (left=0, top=143) from batch crop path
            "target_size": [512, 512],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        # Decode the cropped image to verify dimensions
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # Should be 512x512 since (left=0, top=143) fits in 512w x 682h
        self.assertEqual(cropped_img.size[0], 512)
        self.assertEqual(cropped_img.size[1], 512)

    def test_preview_clamps_when_crop_truly_overflows(self):
        """When crop genuinely overflows, the result should be clamped smaller."""
        self._create_image(size=(400, 400))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        # Intermediary 200x200, target 200x200, but crop starts at (50, 50)
        # Clamped: right = 50 + min(200, 200-50) = 200, bottom = 50 + min(200, 200-50) = 200
        # Result: 150x150 (clamped)
        metadata = {
            "intermediary_size": [200, 200],
            "crop_coordinates": [50, 50],  # (left=50, top=50)
            "target_size": [200, 200],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # Clamped: 150x150
        self.assertEqual(cropped_img.size[0], 150)
        self.assertEqual(cropped_img.size[1], 150)

    def test_preview_crop_coordinate_order_asymmetric(self):
        """Verify (left, top) order: left=0 should produce full-width crop on portrait."""
        self._create_image(size=(512, 800))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        # Portrait intermediary 512w x 682h, target 512x512
        # crop_coordinates = (left=0, top=211) → full width, offset vertically
        metadata = {
            "intermediary_size": [512, 682],
            "crop_coordinates": [0, 211],  # (left=0, top=211)
            "target_size": [512, 512],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # left=0, width=512 (full width); top=211, height=min(512, 682-211)=471
        self.assertEqual(cropped_img.size[0], 512)
        self.assertEqual(cropped_img.size[1], 471)


class TestFileCaption(unittest.TestCase):
    """Verify caption loading in get_file_metadata."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.dataset_id = "cap-test"
        # Create image dir with image, caption sidecar, and metadata cache
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (800, 600)).save(self.img_dir / "photo.jpg")
        (self.img_dir / "photo.txt").write_text("a painting of a sunset")
        # Place metadata cache in instance_data_dir so _find_cache_file finds it
        meta_file = self.img_dir / f"aspect_ratio_bucket_metadata_{self.dataset_id}.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "target_size": [512, 512],
                        "aspect_ratio": 1.333,
                    }
                }
            )
        )

    def _config(self, **overrides):
        base = {
            "id": self.dataset_id,
            "type": "local",
            "instance_data_dir": str(self.img_dir),
            "caption_strategy": "textfile",
        }
        base.update(overrides)
        return base

    def test_textfile_caption_loaded(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "a painting of a sunset")

    def test_filename_caption(self):
        meta = self.service.get_file_metadata(self._config(caption_strategy="filename"), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "photo")

    def test_instanceprompt_caption(self):
        meta = self.service.get_file_metadata(
            self._config(caption_strategy="instanceprompt", instance_prompt="a dog"),
            "photo.jpg",
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "a dog")

    def test_missing_textfile_returns_none(self):
        """When caption file doesn't exist, caption should be None."""
        (self.img_dir / "photo.txt").unlink()
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertIsNone(meta.caption)


class TestUpdateCropCoordinates(unittest.TestCase):
    """Verify crop coordinate persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        meta_file = self.img_dir / "aspect_ratio_bucket_metadata_crop-ds.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "intermediary_size": [512, 384],
                        "target_size": [512, 384],
                        "crop_coordinates": [0, 0],
                        "aspect_ratio": 1.333,
                    }
                }
            )
        )

    def _config(self):
        return {"id": "crop-ds", "type": "local", "instance_data_dir": str(self.img_dir)}

    def test_update_saves_to_cache(self):
        ok = self.service.update_crop_coordinates(self._config(), "photo.jpg", [50, 30])
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertEqual(meta.crop_coordinates, [50, 30])

    def test_update_missing_file_returns_false(self):
        ok = self.service.update_crop_coordinates(self._config(), "missing.jpg", [10, 10])
        self.assertFalse(ok)

    def test_update_no_cache_returns_false(self):
        config = {"id": "no-cache", "type": "local", "instance_data_dir": self.tmpdir}
        ok = self.service.update_crop_coordinates(config, "photo.jpg", [10, 10])
        self.assertFalse(ok)


class TestUpdateBboxEntities(unittest.TestCase):
    """Verify bbox entity persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        meta_file = self.img_dir / "aspect_ratio_bucket_metadata_bbox-ds.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "target_size": [512, 384],
                        "aspect_ratio": 1.333,
                        "bbox_entities": [
                            {"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.8]},
                        ],
                    }
                }
            )
        )

    def _config(self):
        return {"id": "bbox-ds", "type": "local", "instance_data_dir": str(self.img_dir)}

    def test_get_file_metadata_returns_bbox_entities(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta.bbox_entities)
        self.assertEqual(len(meta.bbox_entities), 1)
        self.assertEqual(meta.bbox_entities[0]["label"], "cat")
        self.assertAlmostEqual(meta.bbox_entities[0]["bbox"][0], 0.1)

    def test_bbox_entities_not_in_extra(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertNotIn("bbox_entities", meta.extra)

    def test_update_saves_to_cache(self):
        new_entities = [
            {"label": "dog", "bbox": [0.2, 0.3, 0.6, 0.9]},
            {"label": "cat", "bbox": [0.0, 0.0, 0.5, 0.5]},
        ]
        ok = self.service.update_bbox_entities(self._config(), "photo.jpg", new_entities)
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertEqual(len(meta.bbox_entities), 2)
        self.assertEqual(meta.bbox_entities[0]["label"], "dog")

    def test_update_with_none_removes_entities(self):
        ok = self.service.update_bbox_entities(self._config(), "photo.jpg", None)
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNone(meta.bbox_entities)

    def test_update_missing_file_returns_false(self):
        ok = self.service.update_bbox_entities(self._config(), "missing.jpg", [])
        self.assertFalse(ok)

    def test_update_no_cache_returns_false(self):
        config = {"id": "no-cache", "type": "local", "instance_data_dir": self.tmpdir}
        ok = self.service.update_bbox_entities(config, "photo.jpg", [])
        self.assertFalse(ok)


class TestConditioningFileMatch(unittest.TestCase):
    """Verify conditioning file matching by stem."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

        # Source dataset
        self.src_dir = Path(self.tmpdir) / "source_images"
        self.src_dir.mkdir(parents=True)
        # Create a dummy source image
        img = Image.new("RGB", (64, 64), color="red")
        img.save(self.src_dir / "photo123.jpg")

        # Conditioning dataset (in plan)
        self.cond_dir = Path(self.tmpdir) / "cond_images"
        self.cond_dir.mkdir(parents=True)
        # Create a matching conditioning image
        cond_img = Image.new("RGB", (64, 64), color="green")
        cond_img.save(self.cond_dir / "photo123.png")

        # Bucket cache for conditioning dataset
        cond_cache = self.cond_dir / "aspect_ratio_bucket_indices_cond-ds.json"
        cond_cache.write_text(
            json.dumps(
                {
                    "aspect_ratio_bucket_indices": {"1.0": ["photo123.png"]},
                }
            )
        )

    def _src_config(self):
        return {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning_data": ["cond-ds"],
        }

    def _cond_config(self):
        return {
            "id": "cond-ds",
            "type": "local",
            "instance_data_dir": str(self.cond_dir),
            "conditioning_type": "controlnet",
        }

    def test_match_found_via_conditioning_data(self):
        all_datasets = [self._src_config(), self._cond_config()]
        result = self.service.get_conditioning_file_match(self._src_config(), all_datasets, "photo123.jpg")
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "cond-ds")
        self.assertEqual(result["conditioning_type"], "controlnet")
        self.assertIn("thumbnail", result)

    def test_no_match_for_missing_stem(self):
        all_datasets = [self._src_config(), self._cond_config()]
        result = self.service.get_conditioning_file_match(self._src_config(), all_datasets, "nonexistent.jpg")
        self.assertIsNone(result)

    def test_no_match_without_conditioning(self):
        config = {"id": "plain-ds", "type": "local", "instance_data_dir": str(self.src_dir)}
        result = self.service.get_conditioning_file_match(config, [config], "photo123.jpg")
        self.assertIsNone(result)

    def test_match_via_filesystem_discovery(self):
        """Auto-generated conditioning images found on disk with explicit id."""
        gen_dir = Path(self.tmpdir) / "conditioning_data" / "auto-gen-canny"
        gen_dir.mkdir(parents=True)
        gen_img = Image.new("RGB", (64, 64), color="blue")
        gen_img.save(gen_dir / "photo123.png")

        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning": [
                {"id": "auto-gen-canny", "type": "canny", "instance_data_dir": str(gen_dir)},
            ],
        }
        result = self.service.get_conditioning_file_match(src_config, [src_config], "photo123.jpg")
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "auto-gen-canny")
        self.assertEqual(result["conditioning_type"], "canny")

    def test_match_via_derived_id_and_path(self):
        """Generator config without explicit id uses {source_id}_conditioning_{type}."""
        # Simulate the directory structure the duplicator would create
        gen_dir = Path(self.tmpdir) / "conditioning_data" / "src-ds_conditioning_canny"
        gen_dir.mkdir(parents=True)
        gen_img = Image.new("RGB", (64, 64), color="blue")
        gen_img.save(gen_dir / "photo123.png")

        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            # cache_dir_vae lets _derive_conditioning_dir find the cache root
            "cache_dir_vae": str(Path(self.tmpdir) / "vae" / "model" / "src-ds"),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
        }
        result = self.service.get_conditioning_file_match(src_config, [src_config], "photo123.jpg")
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "src-ds_conditioning_canny")
        self.assertEqual(result["conditioning_type"], "controlnet")

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_viewer_service.GENERATOR_REGISTRY_AVAILABLE", True)
    def test_on_the_fly_generation(self):
        """When conditioning images don't exist, generate preview on-the-fly."""
        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
        }

        # Mock the generator to avoid needing trainingsample C extension
        mock_result = Image.new("RGB", (64, 64), color="white")
        mock_generator = MagicMock()
        mock_generator.transform_batch.return_value = [mock_result]

        mock_registry = {"canny": MagicMock(return_value=mock_generator)}
        with patch(
            "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service.GENERATOR_REGISTRY",
            mock_registry,
        ):
            result = self.service.get_conditioning_file_match(src_config, [src_config], "photo123.jpg")

        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "src-ds_conditioning_canny")
        self.assertEqual(result["conditioning_type"], "controlnet")
        self.assertTrue(result.get("generated_on_the_fly"))
        self.assertIn("thumbnail", result)


class TestRemoteThumbnails(unittest.TestCase):
    """Verify on-demand thumbnail fetching for remote backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        # Clear the module-level backend cache between tests
        _data_backend_cache.clear()

    def tearDown(self):
        _data_backend_cache.clear()

    def _setup_hf_cache(self, dataset_id="hf-ds"):
        """Create a bucket cache for a HuggingFace dataset."""
        cache_dir = Path(self.tmpdir) / "cache" / "huggingface" / dataset_id / "huggingface_metadata" / dataset_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"aspect_ratio_bucket_indices_{dataset_id}.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["0.jpg", "1.jpg"]},
                }
            )
        )
        return cache_dir

    def test_cached_thumbnail_served_from_disk(self):
        """Pre-cached thumbnails should be returned without creating a backend."""
        self._setup_hf_cache()
        # Pre-populate the thumbnail cache
        path_hash = hashlib.md5(b"0.jpg").hexdigest()
        thumb_dir = Path(self.tmpdir) / "cache" / "viewer_thumbnails" / "hf-ds"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        img.save(thumb_dir / f"{path_hash}.jpg", format="JPEG")

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)
            self.assertEqual(len(result), 1)
            self.assertIsNotNone(result[0].thumbnail)
            self.assertIsNone(result[0].error)
        finally:
            os.chdir(original_cwd)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_on_demand_fetch_generates_and_caches_thumbnail(self, mock_get_backend):
        """When no cached thumbnail, should fetch from backend, cache, and return."""
        self._setup_hf_cache()
        mock_backend = MagicMock()
        mock_backend.read_image.return_value = Image.new("RGB", (200, 200), (0, 255, 0))
        mock_get_backend.return_value = mock_backend

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNotNone(result[0].thumbnail)
            self.assertIsNone(result[0].error)
            mock_backend.read_image.assert_called_once_with("0.jpg")

            # Verify thumbnail was cached to disk
            path_hash = hashlib.md5(b"0.jpg").hexdigest()
            cached = Path(self.tmpdir) / "cache" / "viewer_thumbnails" / "hf-ds" / f"{path_hash}.jpg"
            self.assertTrue(cached.is_file())
        finally:
            os.chdir(original_cwd)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_backend_failure_returns_error(self, mock_get_backend):
        """If backend creation fails, thumbnail should report error."""
        self._setup_hf_cache()
        mock_get_backend.side_effect = RuntimeError("Cannot connect")

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].thumbnail)
            self.assertEqual(result[0].error, "Failed to load remote image")
        finally:
            os.chdir(original_cwd)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_read_image_returns_none(self, mock_get_backend):
        """If read_image returns None, thumbnail should report error."""
        self._setup_hf_cache()
        mock_backend = MagicMock()
        mock_backend.read_image.return_value = None
        mock_get_backend.return_value = mock_backend

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].thumbnail)
            self.assertEqual(result[0].error, "Failed to load remote image")
        finally:
            os.chdir(original_cwd)


class TestClearVaeCache(unittest.TestCase):
    """Verify VAE cache clearing in DatasetScanService."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_deletes_directory(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        vae_dir = Path(self.tmpdir) / "cache" / "vae" / "test-ds"
        vae_dir.mkdir(parents=True, exist_ok=True)
        (vae_dir / "latent_001.pt").write_bytes(b"fake")
        (vae_dir / "latent_002.pt").write_bytes(b"fake")

        service = DatasetScanService()
        config = {"id": "test-ds", "cache_dir_vae": str(vae_dir)}
        result = service.clear_vae_cache(config, {})
        self.assertIsNotNone(result)
        self.assertFalse(vae_dir.exists())

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_no_dir_returns_none(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        service = DatasetScanService()
        config = {"id": "test-ds", "cache_dir_vae": "/nonexistent/path"}
        result = service.clear_vae_cache(config, {})
        self.assertIsNone(result)

    def test_clear_vae_cache_no_config_returns_none(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        service = DatasetScanService()
        config = {"id": "test-ds"}
        result = service.clear_vae_cache(config, {})
        self.assertIsNone(result)

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_resolves_template_variables(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        vae_dir = Path(self.tmpdir) / "cache" / "vae" / "flux" / "my-ds"
        vae_dir.mkdir(parents=True, exist_ok=True)
        (vae_dir / "latent.pt").write_bytes(b"fake")

        service = DatasetScanService()
        config = {
            "id": "my-ds",
            "cache_dir_vae": "{output_dir}/cache/vae/{model_family}/{id}",
        }
        global_config = {"model_family": "flux"}
        result = service.clear_vae_cache(config, global_config)
        self.assertIsNotNone(result)
        self.assertFalse(vae_dir.exists())


if __name__ == "__main__":
    unittest.main()
