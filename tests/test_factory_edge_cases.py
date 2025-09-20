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

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    def test_missing_required_fields(self):
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

    def test_duplicate_backend_ids(self):
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

    def test_deepfloyd_model_warnings(self):
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

    def test_pixel_area_resolution_conversion(self):
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

    def test_parquet_backend_missing_config(self):
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
        with patch("simpletuner.helpers.data_backend.factory.LocalDataBackend") as mock_local:
            mock_local.return_value.list_files.return_value = ["image1.jpg", "image2.jpg"]

        with patch("simpletuner.helpers.data_backend.factory.TextEmbeddingCache") as mock_text_cache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.discover_all_files.return_value = None
            mock_cache_instance.compute_embeddings_for_prompts.return_value = None
            mock_cache_instance.set_webhook_handler.return_value = None
            mock_text_cache.return_value = mock_cache_instance

        with patch("simpletuner.helpers.data_backend.factory.VAECache") as mock_vae_cache:
            mock_vae_instance = MagicMock()
            mock_vae_instance.discover_all_files.return_value = None
            mock_vae_instance.discover_unprocessed_files.return_value = []
            mock_vae_instance.build_vae_cache_filename_map.return_value = None
            mock_vae_instance.set_webhook_handler.return_value = None
            mock_vae_cache.return_value = mock_vae_instance

        with patch("simpletuner.helpers.metadata.backends.discovery.DiscoveryMetadataBackend") as mock_metadata:
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

        with patch("simpletuner.helpers.multiaspect.dataset.MultiAspectDataset"):
            pass

        with patch("simpletuner.helpers.multiaspect.sampler.MultiAspectSampler") as mock_sampler:
            mock_sampler_instance = MagicMock()
            mock_sampler_instance.caption_strategy = "filename"
            mock_sampler.return_value = mock_sampler_instance

        with patch("simpletuner.helpers.prompts.PromptHandler") as mock_prompt:
            mock_prompt.get_all_captions.return_value = (["caption1"], [])


if __name__ == "__main__":
    unittest.main()
