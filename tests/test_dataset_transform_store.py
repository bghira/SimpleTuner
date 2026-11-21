import json
import unittest
from contextlib import nullcontext
from hashlib import sha256
from io import IOBase
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image

from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend
from simpletuner.helpers.metadata.preparation import DatasetViewChangedError
from simpletuner.helpers.metadata.transform_store import GEOMETRY_FIELDS, DatasetTransformStore, transform_recipe_directory
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import DatasetViewerService


class FileBackedS3(LocalDataBackend):
    def __init__(self, accelerator, id):
        super().__init__(accelerator, id)
        self.type = "aws"

    def read_with_version(self, path):
        if not self.exists(path):
            return None, None
        content = self.read(path)
        return content, sha256(content).hexdigest()

    def write(self, path, content):
        return super().write(path, content.read() if isinstance(content, IOBase) else content)

    def write_if_version(self, path, content, *, expected_version):
        _, version = self.read_with_version(path)
        if expected_version != version:
            raise OSError("S3 precondition failed")
        self.write(path, content)
        return self.read_with_version(path)[1]


class DatasetTransformStoreTests(unittest.TestCase):
    def make_provider(self, directory, provider_class, *, storage_type="local"):
        image = directory / "sample.png"
        if not image.exists():
            Image.new("RGB", (400, 200)).save(image)
        accelerator = SimpleNamespace(is_main_process=True, process_index=0, num_processes=1, is_local_main_process=True)
        storage_class = FileBackedS3 if storage_type == "aws" else LocalDataBackend
        storage = storage_class(accelerator, "images")
        storage.source_files.publish([(str(directory), [], [str(image)])])
        config = {
            "dataset_type": "image",
            "resolution": 128,
            "resolution_type": "pixel",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
        }
        args = SimpleNamespace(
            aspect_bucket_alignment=8,
            aspect_bucket_rounding=2,
            aspect_bucket_worker_count=1,
            enable_multiprocessing=False,
            ignore_missing_files=False,
            model_family="sdxl",
        )
        settings = dict(
            id="images",
            instance_data_dir=str(directory),
            cache_file=str(directory / "aspect_ratio_bucket_indices_images"),
            metadata_file=str(directory / "aspect_ratio_bucket_metadata_images"),
            data_backend=storage,
            accelerator=accelerator,
            batch_size=1,
            resolution=128,
            resolution_type="pixel",
            dataset_config=config,
            runtime_args=args,
        )
        if provider_class is ParquetMetadataBackend:
            table = directory / "source.parquet"
            if not table.exists():
                pd.DataFrame({"filename": [image.stem], "width": [400], "height": [200]}).to_parquet(table)
            settings["parquet_config"] = {"path": str(table), "filename_column": "filename"}
        return provider_class(**settings), str(image)

    def test_discovery_and_parquet_publish_measurements_and_geometry_separately(self):
        for provider_class in (DiscoveryMetadataBackend, ParquetMetadataBackend):
            with (
                self.subTest(provider=provider_class.__name__),
                TemporaryDirectory() as directory,
                patch.object(StateTracker, "get_model", return_value=None),
                patch.object(StateTracker, "_save_to_disk"),
            ):
                provider, image = self.make_provider(Path(directory), provider_class)
                provider.refresh_buckets()
                expected = provider.get_metadata_by_filepath(image)
                self.assertIsNotNone(expected)
                self.assertEqual(set(expected).intersection(GEOMETRY_FIELDS), GEOMETRY_FIELDS)

                publication = json.loads(provider.cache_file.read_text())
                measured = json.loads(provider.metadata_file.read_text())
                self.assertEqual(measured[image]["original_size"], [400, 200])
                self.assertFalse(GEOMETRY_FIELDS.intersection(measured[image]))
                transforms = DatasetTransformStore(
                    transform_recipe_directory(provider.metadata_file, publication["transform_recipe"])
                )
                self.assertEqual(json.loads(transforms.measurements_path.read_text()), measured)
                self.assertEqual(set(transforms[image]), GEOMETRY_FIELDS)

                recovered, _ = self.make_provider(Path(directory), provider_class)
                recovered.load_image_metadata()
                expected = json.loads(json.dumps(expected))
                self.assertEqual(
                    list(recovered.get_metadata_batch([image, "absent.png", image])), [expected, None, expected]
                )

    def test_legacy_discovery_cache_migrates_the_existing_crop(self):
        with TemporaryDirectory() as directory, patch.object(StateTracker, "_save_to_disk"):
            provider, image = self.make_provider(Path(directory), DiscoveryMetadataBackend)
            record = {
                "original_size": [400, 200],
                "aspect_ratio": 1.0,
                "target_size": [128, 128],
                "intermediary_size": [256, 128],
                "crop_coordinates": [0, 37],
            }
            provider.metadata_file.write_text(json.dumps({image: record}))
            provider.cache_file.write_text(json.dumps({"aspect_ratio_bucket_indices": {"1.0": [image]}}))
            provider.reload_cache()

            with patch.object(provider, "_process_for_bucket", side_effect=AssertionError("Recomputed an existing crop")):
                provider.refresh_buckets()

            self.assertEqual(provider.get_metadata_by_filepath(image), record)
            self.assertEqual(json.loads(provider.metadata_file.read_text()), {image: {"original_size": [400, 200]}})
            reopened, _ = self.make_provider(Path(directory), DiscoveryMetadataBackend)
            reopened.load_image_metadata()
            self.assertEqual(reopened.get_metadata_by_filepath(image), record)

    def test_incomplete_legacy_cache_is_rebucketed(self):
        for provider_class in (DiscoveryMetadataBackend, ParquetMetadataBackend):
            with (
                self.subTest(provider=provider_class.__name__),
                TemporaryDirectory() as directory,
                patch.object(StateTracker, "get_model", return_value=None),
                patch.object(StateTracker, "_save_to_disk"),
            ):
                provider, image = self.make_provider(Path(directory), provider_class)
                provider.metadata_file.write_text(json.dumps({image: {"original_size": [400, 200]}}))
                provider.cache_file.write_text(json.dumps({"aspect_ratio_bucket_indices": {"1.0": [image]}}))
                provider.reload_cache()

                provider.refresh_buckets()

                index = json.loads(provider.cache_file.read_text())
                self.assertIn("transform_recipe", index)
                self.assertEqual(list(index["aspect_ratio_bucket_indices"].values()), [[image]])

    def test_viewer_writes_crops_to_the_recipe_used_by_training(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "_save_to_disk"),
        ):
            root = Path(directory)
            provider, image = self.make_provider(root, DiscoveryMetadataBackend)
            provider.refresh_buckets()
            config = {"id": "images", "type": "local", "instance_data_dir": directory}
            viewer = DatasetViewerService(project_root=root)
            before = viewer.get_file_metadata(config, image)
            self.assertEqual(before.target_size, [128, 128])
            edit = viewer.update_crop_coordinates(config, image, [0, 37], metadata_revision=before.metadata_revision)
            current = viewer.get_file_metadata(config, image)
            self.assertEqual(current.crop_coordinates, [0, 37])
            self.assertEqual(current.metadata_revision, edit.revision)
            self.assertNotEqual(current.metadata_revision, before.metadata_revision)
            with self.assertRaises(DatasetViewChangedError):
                viewer.update_crop_coordinates(config, image, [0, 30], metadata_revision=before.metadata_revision)

            provider.save_image_metadata()
            reopened, _ = self.make_provider(root, DiscoveryMetadataBackend)
            reopened.load_image_metadata()
            self.assertEqual(reopened.get_metadata_by_filepath(image)["crop_coordinates"], [0, 37])
            self.assertNotIn("crop_coordinates", json.loads(provider.metadata_file.read_text())[image])

    def test_s3_index_failure_keeps_the_old_measurements_with_the_old_recipe(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "_save_to_disk"),
        ):
            root = Path(directory)
            provider, image = self.make_provider(root, DiscoveryMetadataBackend, storage_type="aws")
            provider.refresh_buckets()
            initial_index = provider.cache_file.read_bytes()
            initial_recipe = json.loads(initial_index)["transform_recipe"]
            initial_metadata = provider.metadata_file.read_bytes()
            old_geometry = json.loads(json.dumps(provider.get_metadata_by_filepath(image)))

            Image.new("RGB", (800, 200)).save(image)
            write = provider.data_backend.write

            def fail_transform_upload(path, data):
                if path.suffix == ".sqlite3":
                    raise OSError("transform upload failed")
                return write(path, data)

            with patch.object(provider.data_backend, "write", side_effect=fail_transform_upload):
                with self.assertRaisesRegex(OSError, "transform upload failed"):
                    provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(provider.cache_file.read_bytes(), initial_index)

            with patch.object(provider.data_backend, "write_if_version", side_effect=OSError("index write failed")):
                with self.assertRaisesRegex(OSError, "index write failed"):
                    provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)

            self.assertEqual(provider.cache_file.read_bytes(), initial_index)
            self.assertEqual(provider.metadata_file.read_bytes(), initial_metadata)
            self.assertEqual(provider._transform_recipe, initial_recipe)
            self.assertEqual(json.loads(json.dumps(provider.get_metadata_by_filepath(image))), old_geometry)
            self.assertEqual(len(list(provider.metadata_file.with_suffix(".transforms").glob("*/metadata.json"))), 2)
            reader, _ = self.make_provider(root, DiscoveryMetadataBackend, storage_type="aws")
            reader.load_image_metadata()
            self.assertEqual(reader.get_metadata_by_filepath(image), old_geometry)

            provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            reader.reload_cache()
            self.assertNotEqual(reader._transform_recipe, initial_recipe)
            self.assertEqual(reader.get_metadata_by_filepath(image)["original_size"], [800, 200])

    def test_s3_stale_writer_cannot_replace_newer_geometry(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "_save_to_disk"),
        ):
            root = Path(directory)
            current, image = self.make_provider(root, DiscoveryMetadataBackend, storage_type="aws")
            current.refresh_buckets()
            stale, _ = self.make_provider(root, DiscoveryMetadataBackend, storage_type="aws")
            stale.load_image_metadata()
            Image.new("RGB", (640, 256)).save(image)
            current.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            published = current.cache_file.read_bytes()
            with self.assertRaisesRegex(OSError, "S3 precondition failed"):
                stale.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(current.cache_file.read_bytes(), published)
            stale.reload_cache()
            self.assertEqual(stale.get_metadata_by_filepath(image)["original_size"], [640, 256])

    def test_s3_bucket_preparation_has_one_writer_across_nodes(self):
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        with TemporaryDirectory() as directory:
            metadata, _ = self.make_provider(Path(directory), DiscoveryMetadataBackend, storage_type="aws")
            registry = SimpleNamespace(
                accelerator=SimpleNamespace(is_main_process=False, is_local_main_process=True),
                _dataset_startup_phase=lambda *args: nullcontext(),
                _is_multi_process=lambda: True,
            )
            data = {"id": "images", "metadata_backend": metadata, "config": {"resolution": 128}}
            with (
                patch.object(metadata, "refresh_buckets") as refresh,
                patch.object(metadata, "reload_cache") as reload,
                patch.object(StateTracker, "load_aspect_resolution_map"),
            ):
                FactoryRegistry._prepare_bucket_view(registry, data, discovers_buckets=True)
                refresh.assert_not_called()
                reload.assert_called_once()
                registry.accelerator.is_main_process = True
                FactoryRegistry._prepare_bucket_view(registry, data, discovers_buckets=True)
                refresh.assert_called_once()

    def test_missing_published_geometry_is_reported(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "_save_to_disk"),
        ):
            provider, image = self.make_provider(Path(directory), DiscoveryMetadataBackend)
            provider.refresh_buckets()
            recipe = json.loads(provider.cache_file.read_text())["transform_recipe"]
            transforms = DatasetTransformStore(transform_recipe_directory(provider.metadata_file, recipe))
            del transforms[image]
            reader, _ = self.make_provider(Path(directory), DiscoveryMetadataBackend)
            with self.assertRaisesRegex(ValueError, "Published bucket sample is missing geometry"):
                reader.load_image_metadata()
