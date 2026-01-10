#!/usr/bin/env python
"""
Edge cases and error condition tests for factory.py.

This test suite covers:
1. Edge cases and boundary conditions
2. Error handling scenarios
3. Configuration validation
4. DeepFloyd model special cases
5. Memory and performance edge cases
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpletuner.helpers.data_backend.dataset_types import DatasetType


class TestFactoryEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for factory.py."""

    def setUp(self):
        """Set up test fixtures."""
        self.accelerator = MagicMock()
        self.accelerator.is_main_process = True
        self.accelerator.is_local_main_process = True
        self.accelerator.device = "cuda"
        self.accelerator.main_process_first.return_value.__enter__ = MagicMock()
        self.accelerator.main_process_first.return_value.__exit__ = MagicMock()
        self.accelerator.wait_for_everyone = MagicMock()

        self.model = MagicMock()
        self.model.requires_conditioning_latents.return_value = False
        self.model.requires_conditioning_dataset.return_value = False
        self.model.get_vae.return_value = MagicMock()
        self.model.get_pipeline.return_value = MagicMock()
        self.model.AUTOENCODER_CLASS = "AutoencoderKL"

        self.text_encoders = [MagicMock(), MagicMock()]
        self.tokenizers = [MagicMock(), MagicMock()]

        self.args = MagicMock()
        self._setup_args()

        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def _setup_args(self):
        """Set up args with default values."""
        self.args.model_type = "sdxl"
        self.args.model_family = "sdxl"
        self.args.resolution = 1024
        self.args.resolution_type = "area"
        self.args.minimum_image_size = 0.5
        self.args.maximum_image_size = 2.0
        self.args.target_downsample_size = 1.25
        self.args.caption_dropout_probability = 0.1
        self.args.cache_dir_text = "/tmp/text_cache"
        self.args.cache_dir_vae = "/tmp/vae_cache"
        self.args.cache_dir = "/tmp/cache"
        self.args.compress_disk_cache = True
        self.args.delete_problematic_images = False
        self.args.delete_unwanted_images = False
        self.args.metadata_update_interval = 60
        self.args.train_batch_size = 1
        self.args.aws_max_pool_connections = 128
        self.args.vae_cache_scan_behaviour = "ignore"
        self.args.vae_cache_ondemand = False
        self.args.skip_file_discovery = ""
        self.args.gradient_accumulation_steps = 1
        self.args.caption_strategy = "filename"
        self.args.prepend_instance_prompt = False
        self.args.instance_prompt = None
        self.args.only_instance_prompt = False
        self.args.debug_aspect_buckets = False
        self.args.vae_batch_size = 4
        self.args.write_batch_size = 64
        self.args.read_batch_size = 64
        self.args.max_workers = 4
        self.args.image_processing_batch_size = 8
        self.args.max_train_steps = 0
        self.args.override_dataset_config = False
        self.args.cache_file_suffix = None
        self.args.eval_dataset_id = None
        self.args.controlnet = False

    def _create_temp_config(self, config_data: List[Dict[str, Any]]) -> str:
        """Create a temporary config file for testing."""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return config_path

    def test_config_file_not_found(self):
        """Test behavior when config file doesn't exist."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        self.args.data_backend_config = "/non/existent/file.json"

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        with self.assertRaises(FileNotFoundError):
            factory.load_configuration()

    def test_empty_config_file(self):
        """Test behavior with empty configuration."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = []
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        with self.assertRaises(ValueError) as context:
            factory.load_configuration()

        self.assertIn("at least one data backend", str(context.exception))

    def test_malformed_json_config(self):
        """Test behavior with malformed JSON."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config_path = os.path.join(self.temp_dir, "malformed.json")
        with open(config_path, "w") as f:
            f.write("{ invalid json content")

        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        with self.assertRaises(json.JSONDecodeError):
            factory.load_configuration()

    @patch("simpletuner.helpers.data_backend.factory.FactoryRegistry._validate_dataset_paths")
    def test_missing_required_fields(self, mock_validate_paths):
        """Test behavior when required fields are missing."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                # Missing 'id' field
                "type": "local",
                "instance_data_dir": "/tmp/images",
            }
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = {}
            mock_state_tracker.clear_data_backends.return_value = None

            with self.assertRaises(ValueError) as context:
                factory.configure_data_backends(loaded_config)

            self.assertIn("unique 'id' field", str(context.exception))

    @patch("simpletuner.helpers.data_backend.factory.FactoryRegistry._validate_dataset_paths")
    def test_duplicate_backend_ids(self, mock_validate_paths):
        """Test behavior with duplicate backend IDs."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {"id": "duplicate_id", "type": "local", "instance_data_dir": "/tmp/images1", "cache_dir_vae": "/tmp/vae1"},
            {
                "id": "duplicate_id",  # Same ID
                "type": "local",
                "instance_data_dir": "/tmp/images2",
                "cache_dir_vae": "/tmp/vae2",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = {"duplicate_id": {"id": "duplicate_id"}}
            mock_state_tracker.clear_data_backends.return_value = None

            with self.assertRaises(ValueError) as context:
                factory.configure_data_backends(loaded_config)

        self.assertIn("unique 'id' field", str(context.exception))

    def test_inline_conditioning_auto_generation_for_image_dataset(self):
        """Inline conditioning blocks on image datasets should spawn auto-generated conditioning datasets."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "primary",
                "type": "huggingface",
                "dataset_type": "image",
                "cache_dir_vae": "/tmp/vae/primary",
                "conditioning": [
                    {
                        "type": "canny",
                        "conditioning_type": "controlnet",
                        "params": {},
                    }
                ],
            }
        ]

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        processed = factory.process_conditioning_datasets(deepcopy(config))

        self.assertEqual(len(processed), 2)
        primary = processed[0]
        self.assertNotIn("conditioning", primary)
        self.assertEqual(primary.get("conditioning_data"), ["primary_conditioning_canny"])

        generated = next(entry for entry in processed if entry["id"] == "primary_conditioning_canny")
        self.assertEqual(generated.get("dataset_type"), "conditioning")
        self.assertTrue(generated.get("auto_generated"))
        self.assertEqual(generated.get("source_dataset_id"), "primary")
        conditioning_cfg = generated.get("conditioning_config") or {}
        self.assertEqual(conditioning_cfg.get("type"), "canny")
        self.assertEqual(conditioning_cfg.get("conditioning_type"), "controlnet")

    def test_huggingface_metadata_paths_without_instance_data_dir(self):
        """Huggingface metadata backend should allow missing instance_data_dir."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        backend = {
            "id": "hf_video",
            "type": "huggingface",
            "dataset_type": "video",
            "metadata_backend": "huggingface",
            "huggingface": {},
        }
        init_backend = {
            "id": backend["id"],
            "config": backend.copy(),
            "instance_data_dir": "",
            "data_backend": MagicMock(),
            "bucket_report": MagicMock(),
        }

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        with patch("simpletuner.helpers.metadata.backends.huggingface.HuggingfaceMetadataBackend") as mock_backend:
            mock_backend_instance = MagicMock()
            mock_backend.return_value = mock_backend_instance

            factory._configure_metadata_backend(backend, init_backend)

        self.assertIs(init_backend["metadata_backend"], mock_backend_instance)
        metadata_file = mock_backend.call_args.kwargs.get("metadata_file")
        cache_file = mock_backend.call_args.kwargs.get("cache_file")
        self.assertIsInstance(metadata_file, str)
        self.assertIsInstance(cache_file, str)

    def test_multiple_default_text_embed_backends(self):
        """Test error when multiple text embed backends are marked as default."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "text_embeds1",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache1",
            },
            {
                "id": "text_embeds2",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,  # Second default
                "cache_dir": "/tmp/text_cache2",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)

            with self.assertRaises(ValueError) as context:
                factory.configure_text_embed_backends(loaded_config)

            self.assertIn("Only one text embed backend can be marked as default", str(context.exception))

    @patch("simpletuner.helpers.data_backend.factory.FactoryRegistry._validate_dataset_paths")
    def test_deepfloyd_model_warnings(self, mock_validate_paths):
        """Test DeepFloyd model specific warnings and handling."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        self.args.model_type = "deepfloyd-if"

        config = [
            {
                "id": "deepfloyd_test",
                "type": "local",
                "instance_data_dir": "/tmp/images",
                "resolution_type": "area",
                "resolution": 0.3,  # > 0.25 megapixels
                "cache_dir_vae": "/tmp/vae",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.warning_log") as mock_warning:
            with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
                self._setup_comprehensive_mocks(mock_state_tracker)

                factory.configure_text_embed_backends(loaded_config)
                factory.configure_data_backends(loaded_config)

                # Check that warnings were logged (the exact calls depend on the implementation)
                # We mainly want to ensure no exceptions are raised for DeepFloyd

    @patch("simpletuner.helpers.data_backend.factory.FactoryRegistry._validate_dataset_paths")
    def test_pixel_area_resolution_conversion(self, mock_validate_paths):
        """Test pixel_area to area resolution type conversion."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "pixel_area_test",
                "type": "local",
                "instance_data_dir": "/tmp/images",
                "resolution_type": "pixel_area",
                "resolution": 1024,  # pixel edge length
                "maximum_image_size": 1536,
                "target_downsample_size": 1280,
                "minimum_image_size": 512,
                "cache_dir_vae": "/tmp/vae",
                "repeats": 100,  # High repeats to avoid validation failure with small/no dataset
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_comprehensive_mocks(mock_state_tracker)

            # Patch metadata backend to return non-zero length (avoid validation failure)
            with patch("simpletuner.helpers.metadata.backends.discovery.DiscoveryMetadataBackend") as mock_metadata:
                mock_metadata_instance = MagicMock()
                mock_metadata_instance.refresh_buckets.return_value = None
                mock_metadata_instance.split_buckets_between_processes.return_value = None
                mock_metadata_instance.save_cache.return_value = None
                mock_metadata_instance.__len__.return_value = 10  # Non-zero to pass validation
                mock_metadata_instance.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg"]}
                mock_metadata_instance.config = {}
                mock_metadata_instance.resolution_type = "area"
                mock_metadata_instance.resolution = 1024
                mock_metadata.return_value = mock_metadata_instance

                factory.configure_text_embed_backends(loaded_config)
                factory.configure_data_backends(loaded_config)

                # Check that resolution_type was converted
                backend_config = loaded_config[0]
                self.assertEqual(backend_config["resolution_type"], "area")
                # Check that resolution was converted from pixel_area to area
                expected_resolution = (1024 * 1024) / (1000**2)  # Convert to megapixels
                self.assertAlmostEqual(backend_config["resolution"], expected_resolution, places=6)

    def test_csv_backend_invalid_config(self):
        """Test CSV backend with invalid configuration."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "invalid_csv",
                "type": "csv",
                # Missing required csv_file
                "csv_caption_column": "caption",
                "csv_cache_dir": "/tmp/csv_cache",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)

            # Should raise an error during CSV backend configuration
            with self.assertRaises((ValueError, KeyError)):
                factory.configure_data_backends(loaded_config)

    def test_aws_backend_invalid_config(self):
        """Test AWS backend with missing credentials."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "invalid_aws",
                "type": "aws",
                "aws_bucket_name": "test-bucket",
                # Missing aws_access_key_id and aws_secret_access_key
                "cache_dir_vae": "/tmp/vae",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)

            # Should raise an error during AWS config validation
            with self.assertRaises((ValueError, KeyError)):
                factory.configure_data_backends(loaded_config)

    @patch("simpletuner.helpers.data_backend.factory.FactoryRegistry._validate_dataset_paths")
    def test_parquet_backend_missing_config(self, mock_validate_paths):
        """Test parquet metadata backend with missing parquet config."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "invalid_parquet",
                "type": "local",
                "instance_data_dir": "/tmp/images",
                "metadata_backend": "parquet",
                # Missing "parquet" configuration
                "cache_dir_vae": "/tmp/vae",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)

            with self.assertRaises(ValueError) as context:
                factory.configure_data_backends(loaded_config)

            self.assertIn("parquet", str(context.exception))

    def test_memory_tracking_functionality(self):
        """Test memory tracking and metrics functionality."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        # Test memory usage tracking
        initial_memory = factory._get_memory_usage()
        self.assertIsInstance(initial_memory, float)
        self.assertGreaterEqual(initial_memory, 0)

        # Test peak memory update
        factory._update_peak_memory()
        self.assertGreaterEqual(factory.metrics["memory_usage"]["peak"], 0)

        # Test performance logging
        factory._log_performance_metrics("test_stage", {"test_metric": 123})

        # Test metrics finalization
        factory._finalize_metrics()
        self.assertGreater(factory.metrics["initialization_time"], 0)

    def test_no_data_backends_error(self):
        """Test error when no data backends are found after configuration."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            }
            # No image/video backends
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)
            # Mock to return empty list for image/video backends
            mock_state_tracker.get_data_backends.return_value = {}

            factory.configure_text_embed_backends(loaded_config)

            with self.assertRaises(ValueError) as context:
                factory.configure_data_backends(loaded_config)

            self.assertIn("at least one data backend", str(context.exception))

    def _setup_state_tracker_mocks(self, mock_state_tracker):
        """Set up basic StateTracker mocks."""
        mock_state_tracker.get_args.return_value = self.args
        mock_state_tracker.get_accelerator.return_value = self.accelerator
        mock_state_tracker.get_webhook_handler.return_value = None
        mock_state_tracker.get_data_backends.return_value = {}
        mock_state_tracker.get_model_family.return_value = "sdxl"
        mock_state_tracker.get_data_backend_config.return_value = {}
        mock_state_tracker.clear_data_backends.return_value = None
        mock_state_tracker.register_data_backend.return_value = None
        mock_state_tracker.set_data_backend_config.return_value = None
        mock_state_tracker.set_default_text_embed_cache.return_value = None

        # Mock TextEmbeddingCache to prevent serializing MagicMock embeddings
        text_cache_patcher = patch("simpletuner.helpers.data_backend.factory.TextEmbeddingCache")
        mock_text_cache = text_cache_patcher.start()
        self.addCleanup(text_cache_patcher.stop)
        mock_cache_instance = MagicMock()
        mock_cache_instance.discover_all_files.return_value = None
        mock_cache_instance.compute_embeddings_for_prompts.return_value = None
        mock_cache_instance.set_webhook_handler.return_value = None
        mock_text_cache.return_value = mock_cache_instance

    def _setup_comprehensive_mocks(self, mock_state_tracker):
        """Set up comprehensive mocks for full configuration testing."""
        self._setup_state_tracker_mocks(mock_state_tracker)

        # Additional mocks for complete configuration
        mock_state_tracker.get_vae.return_value = MagicMock()
        mock_state_tracker.get_image_files.return_value = ["image1.jpg", "image2.jpg"]
        mock_state_tracker.set_image_files.return_value = ["image1.jpg", "image2.jpg"]
        mock_state_tracker.get_conditioning_mappings.return_value = []
        mock_state_tracker.delete_cache_files.return_value = None
        mock_state_tracker.load_aspect_resolution_map.return_value = None

        # Mock all the backend classes
        local_backend_patcher = patch("simpletuner.helpers.data_backend.factory.LocalDataBackend")
        mock_local = local_backend_patcher.start()
        self.addCleanup(local_backend_patcher.stop)
        mock_local.return_value.list_files.return_value = ["image1.jpg", "image2.jpg"]

        # Note: TextEmbeddingCache is already mocked in _setup_state_tracker_mocks

        # Use autospec to validate constructor parameters
        vae_cache_patcher = patch("simpletuner.helpers.data_backend.factory.VAECache", autospec=True)
        mock_vae_cache = vae_cache_patcher.start()
        self.addCleanup(vae_cache_patcher.stop)

        # Create a properly spec'd instance
        mock_vae_instance = MagicMock()
        mock_vae_instance.discover_all_files.return_value = None
        mock_vae_instance.discover_unprocessed_files.return_value = []
        mock_vae_instance.build_vae_cache_filename_map.return_value = None
        mock_vae_instance.set_webhook_handler.return_value = None
        mock_vae_cache.return_value = mock_vae_instance

        metadata_backend_patcher = patch("simpletuner.helpers.metadata.backends.discovery.DiscoveryMetadataBackend")
        mock_metadata = metadata_backend_patcher.start()
        self.addCleanup(metadata_backend_patcher.stop)
        mock_metadata_instance = MagicMock()
        mock_metadata_instance.refresh_buckets.return_value = None
        mock_metadata_instance.has_single_underfilled_bucket.return_value = False
        mock_metadata_instance.split_buckets_between_processes.return_value = None
        mock_metadata_instance.save_cache.return_value = None
        mock_metadata_instance.__len__.return_value = 10
        mock_metadata_instance.config = {}
        mock_metadata_instance.resolution_type = "area"
        mock_metadata_instance.resolution = 1024
        mock_metadata.return_value = mock_metadata_instance

        dataset_patcher = patch("simpletuner.helpers.multiaspect.dataset.MultiAspectDataset")
        dataset_patcher.start()
        self.addCleanup(dataset_patcher.stop)

        sampler_patcher = patch("simpletuner.helpers.multiaspect.sampler.MultiAspectSampler")
        mock_sampler = sampler_patcher.start()
        self.addCleanup(sampler_patcher.stop)
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.caption_strategy = "filename"
        mock_sampler.return_value = mock_sampler_instance

        prompt_handler_patcher = patch("simpletuner.helpers.prompts.PromptHandler")
        mock_prompt = prompt_handler_patcher.start()
        self.addCleanup(prompt_handler_patcher.stop)
        mock_prompt.get_all_captions.return_value = (["caption1"], [])

    def test_empty_dataset_validation_training(self):
        """Test that training datasets with no usable samples raise an error."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        # Create a minimal backend config
        backend_config = {
            "id": "test_empty_training",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": self.temp_dir,
            "resolution": 512,
            "resolution_type": "pixel",
        }

        with patch("simpletuner.helpers.training.state_tracker.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = {}
            mock_state_tracker.get_data_backend.return_value = None
            mock_state_tracker.set_data_backend_config.return_value = None
            mock_state_tracker.register_data_backend.return_value = None

            with patch("simpletuner.helpers.data_backend.factory.init_backend_config") as mock_init:
                # Mock metadata backend that returns 0 length (no usable samples)
                mock_metadata = MagicMock()
                mock_metadata.__len__.return_value = 0
                mock_metadata.aspect_ratio_bucket_indices = {"1.0": []}

                mock_init_backend = {
                    "id": "test_empty_training",
                    "config": backend_config.copy(),
                    "dataset_type": "image",
                    "instance_data_dir": self.temp_dir,
                    "metadata_backend": mock_metadata,
                    "bucket_report": MagicMock(),
                }
                mock_init_backend["bucket_report"].format_empty_dataset_message.return_value = "Test: No usable samples"
                mock_init.return_value = mock_init_backend

                factory = FactoryRegistry(
                    args=self.args,
                    accelerator=self.accelerator,
                    text_encoders=self.text_encoders,
                    tokenizers=self.tokenizers,
                    model=self.model,
                )

                # Should raise ValueError for empty training dataset
                with self.assertRaises(ValueError) as context:
                    factory._handle_config_versioning(backend_config, mock_init_backend)

                self.assertIn("Dataset produced no usable samples", str(context.exception))
                self.assertIn("batch_size", str(context.exception))
                self.assertIn("repeats", str(context.exception))

    def test_empty_dataset_validation_conditioning(self):
        """Test that conditioning datasets with no usable samples raise an error."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        # Create a minimal conditioning backend config
        backend_config = {
            "id": "test_empty_conditioning",
            "type": "local",
            "dataset_type": "conditioning",
            "conditioning_type": "reference_strict",
            "instance_data_dir": self.temp_dir,
            "resolution": 512,
            "resolution_type": "pixel",
        }

        with patch("simpletuner.helpers.training.state_tracker.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = {}
            mock_state_tracker.get_data_backend.return_value = None
            mock_state_tracker.set_data_backend_config.return_value = None
            mock_state_tracker.register_data_backend.return_value = None

            with patch("simpletuner.helpers.data_backend.factory.init_backend_config") as mock_init:
                # Mock metadata backend that returns 0 length (no usable samples)
                mock_metadata = MagicMock()
                mock_metadata.__len__.return_value = 0
                mock_metadata.aspect_ratio_bucket_indices = {}

                mock_init_backend = {
                    "id": "test_empty_conditioning",
                    "config": backend_config.copy(),
                    "dataset_type": "conditioning",
                    "instance_data_dir": self.temp_dir,
                    "metadata_backend": mock_metadata,
                    "bucket_report": MagicMock(),
                }
                mock_init_backend["bucket_report"].format_empty_dataset_message.return_value = "Test: No usable samples"
                mock_init.return_value = mock_init_backend

                factory = FactoryRegistry(
                    args=self.args,
                    accelerator=self.accelerator,
                    text_encoders=self.text_encoders,
                    tokenizers=self.tokenizers,
                    model=self.model,
                )

                # Should raise ValueError for empty conditioning dataset
                with self.assertRaises(ValueError) as context:
                    factory._handle_config_versioning(backend_config, mock_init_backend)

                self.assertIn("Dataset produced no usable samples", str(context.exception))
                self.assertIn("batch_size", str(context.exception))

    def test_early_validation_impossible_config(self):
        """Test that impossible configurations are caught early with specific guidance."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Create a mock backend with only the necessary attributes for split_buckets
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_impossible"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]}

        # Mock accelerator with 8 GPUs
        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 8
        mock_backend.accelerator = mock_accelerator

        # Mock StateTracker.get_args() to return args with allow_dataset_oversubscription=False
        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            # Mock StateTracker.get_data_backend_config to return empty config (no manual repeats)
            with patch.object(StateTracker, "get_data_backend_config", return_value={}):
                # Call the real split_buckets_between_processes method
                # This should raise ValueError with specific guidance
                # 4 images × (0+1) repeats = 4 samples
                # batch_size=4 × 8 GPUs × 1 grad_accum = 32 effective batch size
                # 4 samples < 32 effective batch size → should fail
                with self.assertRaises(ValueError) as context:
                    MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=1)

        error_msg = str(context.exception)
        # Check that error message contains key information
        self.assertIn("Dataset configuration will produce zero usable batches", error_msg)
        self.assertIn("Total samples: 4", error_msg)
        self.assertIn("Repeats: 0", error_msg)
        self.assertIn("Batch size: 4", error_msg)
        self.assertIn("Number of GPUs: 8", error_msg)
        self.assertIn("Effective batch size: 32", error_msg)
        self.assertIn("Minimum repeats required:", error_msg)
        # Should suggest repeats=7 (4 images × 8 = 32 samples needed, so need repeats=7)
        self.assertIn("Increase repeats to at least 7", error_msg)

    def test_early_validation_sufficient_repeats(self):
        """Test that configuration works when repeats are sufficient."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Create a mock backend with sufficient repeats
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_sufficient"
        mock_backend.batch_size = 1
        mock_backend.repeats = 1  # This should be sufficient
        mock_backend.bucket_report = None
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]}

        # Mock accelerator with 2 GPUs
        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 2

        # Mock split_between_processes to distribute images properly
        def split_side_effect(images, apply_padding=False):
            mock_context = MagicMock()
            # Each GPU gets half the images
            mock_context.__enter__ = MagicMock(return_value=images[: len(images) // 2])
            mock_context.__exit__ = MagicMock(return_value=False)
            return mock_context

        mock_accelerator.split_between_processes = MagicMock(side_effect=split_side_effect)
        mock_backend.accelerator = mock_accelerator

        # Mock StateTracker.get_args() to return args with allow_dataset_oversubscription=False
        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            # Mock StateTracker.get_data_backend_config to return empty config (no manual repeats)
            with patch.object(StateTracker, "get_data_backend_config", return_value={}):
                # 4 images with repeats=1
                # 4 images × (1+1) repeats = 8 samples
                # batch_size=1 × 2 GPUs × 1 grad_accum = 2 effective batch size
                # 8 samples >= 2 effective batch size → should succeed (not raise during validation)
                try:
                    MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=1)
                except ValueError as e:
                    if "Dataset configuration will produce zero usable batches" in str(e):
                        self.fail(f"Should not raise ValueError with sufficient repeats: {e}")

    def test_eval_dataset_ignores_gradient_accumulation(self):
        """Eval datasets should not fail bucket validation due to gradient accumulation."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_eval_ga"
        mock_backend.batch_size = 1
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.EVAL
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg"]}

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 1

        def split_side_effect(images, apply_padding=False):
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=images)
            mock_context.__exit__ = MagicMock(return_value=False)
            return mock_context

        mock_accelerator.split_between_processes = MagicMock(side_effect=split_side_effect)
        mock_backend.accelerator = mock_accelerator

        with (
            patch.object(StateTracker, "get_args") as mock_get_args,
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
        ):
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            try:
                MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=4)
            except ValueError as e:
                if "Dataset configuration will produce zero usable batches" in str(e):
                    self.fail(f"Eval dataset should not consider grad accumulation: {e}")

        mock_accelerator.split_between_processes.assert_called_once()

    def test_oversubscription_auto_adjustment(self):
        """Test that --allow_dataset_oversubscription automatically adjusts repeats."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Create a mock backend with insufficient images
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_auto_adjust"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]}

        # Mock accelerator with 8 GPUs
        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 8
        mock_backend.accelerator = mock_accelerator

        # Mock StateTracker to enable oversubscription and NO user-set repeats
        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = True
            mock_get_args.return_value = mock_args

            with patch.object(StateTracker, "get_data_backend_config") as mock_get_config:
                # Return config WITHOUT 'repeats' key (not user-set)
                mock_get_config.return_value = {"id": "test_auto_adjust"}

                # Mock the split_between_processes to avoid actual splitting
                def split_side_effect(images, apply_padding=False):
                    mock_context = MagicMock()
                    mock_context.__enter__ = MagicMock(return_value=images)
                    mock_context.__exit__ = MagicMock(return_value=False)
                    return mock_context

                mock_accelerator.split_between_processes = MagicMock(side_effect=split_side_effect)

                # Should NOT raise, should auto-adjust repeats
                try:
                    MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=1)
                    # Check that repeats was adjusted
                    self.assertEqual(mock_backend.repeats, 7, "Repeats should be auto-adjusted to 7")
                except ValueError as e:
                    self.fail(f"Should not raise with oversubscription enabled: {e}")

    def test_oversubscription_respects_manual_repeats(self):
        """Test that manual repeats settings are respected even with oversubscription enabled."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Create a mock backend with manually-set but insufficient repeats
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_manual_repeats"
        mock_backend.batch_size = 4
        mock_backend.repeats = 2  # User set this, but it's not enough (needs 7)
        mock_backend.bucket_report = None
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]}

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 8
        mock_backend.accelerator = mock_accelerator

        # Mock StateTracker with oversubscription enabled BUT user set repeats
        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = True
            mock_get_args.return_value = mock_args

            with patch.object(StateTracker, "get_data_backend_config") as mock_get_config:
                # Return config WITH 'repeats' key (user-set)
                mock_get_config.return_value = {"id": "test_manual_repeats", "repeats": 2}

                # Should raise because manual repeats are insufficient
                with self.assertRaises(ValueError) as context:
                    MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=1)

                error_msg = str(context.exception)
                self.assertIn("Dataset configuration will produce zero usable batches", error_msg)
                self.assertIn("manually set repeats=2", error_msg)
                self.assertIn("will not override manual repeats", error_msg)

    def test_oversubscription_disabled_raises_error(self):
        """Test that error is raised when oversubscription is disabled."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_disabled"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]}

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 8
        mock_backend.accelerator = mock_accelerator

        # Mock StateTracker with oversubscription DISABLED
        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            with patch.object(StateTracker, "get_data_backend_config") as mock_get_config:
                mock_get_config.return_value = {"id": "test_disabled"}

                # Should raise error with suggestion to enable flag
                with self.assertRaises(ValueError) as context:
                    MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=1)

                error_msg = str(context.exception)
                self.assertIn("Dataset configuration will produce zero usable batches", error_msg)
                self.assertIn("Enable --allow_dataset_oversubscription", error_msg)

    def test_image_embeds_backend_configuration(self):
        """Image embed configuration should not instantiate VAE cache directly."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config = [
            {
                "id": "image_embeds_test",
                "dataset_type": "image_embeds",
                "type": "local",
                "instance_data_dir": self.temp_dir,
                "cache_dir": "/tmp/image_embeds_cache",
            }
        ]
        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        loaded_config = factory.load_configuration()

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            self._setup_state_tracker_mocks(mock_state_tracker)
            mock_state_tracker.get_vae.return_value = MagicMock()

            with patch("simpletuner.helpers.data_backend.factory.LocalDataBackend") as mock_local:
                mock_local.return_value.list_files.return_value = []

                # Use autospec=True to validate constructor parameters
                with patch("simpletuner.helpers.data_backend.factory.VAECache", autospec=True) as mock_vae_cache:
                    mock_vae_instance = MagicMock()
                    mock_vae_instance.discover_all_files.return_value = None
                    mock_vae_cache.return_value = mock_vae_instance

                    # This should work without raising TypeError if parameters are correct
                    factory.configure_image_embed_backends(loaded_config)

                    # Image embed setup should not instantiate the cache; the owning dataset does.
                    mock_vae_cache.assert_not_called()

    def test_qwen_edit_model_invalid_conditioning_type(self):
        """Test that Qwen edit models reject invalid conditioning_type values."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        # Set up args for Qwen edit model
        self.args.model_family = "qwen_image"
        self.args.model_flavour = "edit-v1"

        # Mock the model's edit check methods
        self.model.is_edit_v1_model = MagicMock(return_value=True)
        self.model.is_edit_v2_model = MagicMock(return_value=False)
        self.model.requires_conditioning_dataset.return_value = True

        # Create config with invalid conditioning_type
        config = [
            {
                "id": "test-images",
                "type": "local",
                "instance_data_dir": self.temp_dir,
                "dataset_type": "image",
                "conditioning_data": ["test-conditioning"],
            },
            {
                "id": "test-conditioning",
                "type": "local",
                "instance_data_dir": self.temp_dir,
                "dataset_type": "conditioning",
                "conditioning_type": "controlnet",  # Invalid for Qwen edit!
            },
            {
                "id": "test-text-embeds",
                "type": "local",
                "dataset_type": "text_embeds",
                "default": True,
            },
        ]

        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        # Should raise ValueError about invalid conditioning_type
        with self.assertRaises(ValueError) as context:
            factory = FactoryRegistry(
                self.args,
                self.accelerator,
                self.text_encoders,
                self.tokenizers,
                self.model,
            )
            # Call the validation method directly
            factory._validate_edit_model_conditioning_type(config)

        error_msg = str(context.exception)
        self.assertIn("Invalid conditioning_type='controlnet'", error_msg)
        self.assertIn("reference_strict", error_msg)
        self.assertIn("reference_loose", error_msg)
        self.assertIn("dimension mismatches", error_msg)

    def test_qwen_edit_model_valid_conditioning_type(self):
        """Test that Qwen edit models accept valid conditioning_type values."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        # Set up args for Qwen edit model
        self.args.model_family = "qwen_image"
        self.args.model_flavour = "edit-v2"

        # Mock the model's edit check methods
        self.model.is_edit_v1_model = MagicMock(return_value=False)
        self.model.is_edit_v2_model = MagicMock(return_value=True)
        self.model.requires_conditioning_dataset.return_value = True

        # Test both valid conditioning_type values
        for conditioning_type in ["reference_strict", "reference_loose"]:
            config = [
                {
                    "id": "test-images",
                    "type": "local",
                    "instance_data_dir": self.temp_dir,
                    "dataset_type": "image",
                    "conditioning_data": ["test-conditioning"],
                },
                {
                    "id": "test-conditioning",
                    "type": "local",
                    "instance_data_dir": self.temp_dir,
                    "dataset_type": "conditioning",
                    "conditioning_type": conditioning_type,  # Valid!
                },
                {
                    "id": "test-text-embeds",
                    "type": "local",
                    "dataset_type": "text_embeds",
                    "default": True,
                },
            ]

            config_path = self._create_temp_config(config)
            self.args.data_backend_config = config_path

            # Should NOT raise with valid conditioning_type
            try:
                factory = FactoryRegistry(
                    self.args,
                    self.accelerator,
                    self.text_encoders,
                    self.tokenizers,
                    self.model,
                )
                # Just call the validation method directly since full setup is complex
                factory._validate_edit_model_conditioning_type(config)
            except ValueError as e:
                self.fail(f"Should not raise with conditioning_type='{conditioning_type}': {e}")

    def test_qwen_non_edit_model_allows_any_conditioning_type(self):
        """Test that non-edit Qwen models (e.g., v1.0) don't restrict conditioning_type."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        # Set up args for non-edit Qwen model
        self.args.model_family = "qwen_image"
        self.args.model_flavour = "v1.0"  # Not an edit model

        # Mock the model's edit check methods
        self.model.is_edit_v1_model = MagicMock(return_value=False)
        self.model.is_edit_v2_model = MagicMock(return_value=False)

        config = [
            {
                "id": "test-conditioning",
                "type": "local",
                "instance_data_dir": self.temp_dir,
                "dataset_type": "conditioning",
                "conditioning_type": "controlnet",  # Should be OK for non-edit models
            },
        ]

        # Should NOT raise for non-edit models
        try:
            factory = FactoryRegistry(
                self.args,
                self.accelerator,
                self.text_encoders,
                self.tokenizers,
                self.model,
            )
            factory._validate_edit_model_conditioning_type(config)
        except ValueError as e:
            self.fail(f"Non-edit Qwen model should allow any conditioning_type: {e}")


if __name__ == "__main__":
    unittest.main()
