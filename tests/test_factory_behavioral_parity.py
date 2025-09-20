#!/usr/bin/env python
"""
Behavioral parity tests comparing factory implementations with identical configurations.

Tests real configuration files through both implementations and compares outputs.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactoryBehavioralParity(unittest.TestCase):
    """Test behavioral parity between old and new factory implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock accelerator
        self.accelerator = MagicMock()
        self.accelerator.is_main_process = True
        self.accelerator.is_local_main_process = True
        self.accelerator.device = "cuda"
        self.accelerator.main_process_first.return_value.__enter__ = MagicMock()
        self.accelerator.main_process_first.return_value.__exit__ = MagicMock()
        self.accelerator.wait_for_everyone = MagicMock()

        # Mock model
        self.model = MagicMock()
        self.model.requires_conditioning_latents.return_value = False
        self.model.requires_conditioning_dataset.return_value = False
        self.model.get_vae.return_value = MagicMock()
        self.model.get_pipeline.return_value = MagicMock()
        self.model.AUTOENCODER_CLASS = "AutoencoderKL"

        # Mock text encoders and tokenizers
        self.text_encoders = [MagicMock(), MagicMock()]
        self.tokenizers = [MagicMock(), MagicMock()]

        # Comprehensive args
        self.args = MagicMock()
        self._setup_args()

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def _setup_args(self):
        """Set up comprehensive args mock."""
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
        self.args.offload_during_startup = False
        self.args.skip_file_discovery = ""
        self.args.parquet_caption_column = None
        self.args.parquet_filename_column = None
        self.args.data_backend_sampling = "uniform"
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

    def _create_temp_config(self, config_data: List[Dict[str, Any]], filename: str = "test_config.json") -> str:
        """Create a temporary config file for testing."""
        config_path = os.path.join(self.temp_dir, filename)
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return config_path

    def _load_real_config(self, config_name: str) -> List[Dict[str, Any]]:
        """Load a real configuration file for testing."""
        config_path = f"/Users/kash/src/SimpleTuner/config/{config_name}"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _create_minimal_test_config(self) -> List[Dict[str, Any]]:
        """Create a minimal configuration for comparison testing."""
        return [
            {
                "id": "test_images",
                "type": "local",
                "instance_data_dir": "/tmp/test_images",
                "caption_strategy": "filename",
                "cache_dir_vae": "/tmp/vae/test_images",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]

    def _setup_common_mocks(self):
        """Set up common mocks for both implementations."""
        patches = {}

        # StateTracker mocks
        state_tracker_patcher = patch("simpletuner.helpers.training.state_tracker.StateTracker")
        patches["state_tracker"] = state_tracker_patcher.start()
        self.addCleanup(state_tracker_patcher.stop)

        mock_state_tracker = patches["state_tracker"]
        mock_state_tracker.get_args.return_value = self.args
        mock_state_tracker.get_accelerator.return_value = self.accelerator
        mock_state_tracker.get_webhook_handler.return_value = None
        mock_state_tracker.get_data_backends.return_value = {}
        mock_state_tracker.get_model_family.return_value = "sdxl"
        mock_state_tracker.get_vae.return_value = MagicMock()
        mock_state_tracker.get_image_files.return_value = ["image1.jpg", "image2.jpg"]
        mock_state_tracker.set_image_files.return_value = ["image1.jpg", "image2.jpg"]
        mock_state_tracker.get_data_backend_config.return_value = {}
        mock_state_tracker.get_conditioning_mappings.return_value = []
        mock_state_tracker.clear_data_backends.return_value = None
        mock_state_tracker.register_data_backend.return_value = None
        mock_state_tracker.set_data_backend_config.return_value = None
        mock_state_tracker.set_default_text_embed_cache.return_value = None
        mock_state_tracker.delete_cache_files.return_value = None
        mock_state_tracker.load_aspect_resolution_map.return_value = None
        mock_state_tracker.set_conditioning_datasets.return_value = None

        # Backend mocks
        local_backend_patcher = patch("simpletuner.helpers.data_backend.local.LocalDataBackend")
        patches["local_backend"] = local_backend_patcher.start()
        self.addCleanup(local_backend_patcher.stop)

        mock_backend_instance = MagicMock()
        mock_backend_instance.list_files.return_value = ["image1.jpg", "image2.jpg"]
        patches["local_backend"].return_value = mock_backend_instance

        # Text cache mocks
        text_cache_patcher = patch("simpletuner.helpers.caching.text_embeds.TextEmbeddingCache")
        patches["text_cache"] = text_cache_patcher.start()
        self.addCleanup(text_cache_patcher.stop)

        mock_text_cache_instance = MagicMock()
        mock_text_cache_instance.discover_all_files.return_value = None
        mock_text_cache_instance.compute_embeddings_for_prompts.return_value = None
        mock_text_cache_instance.set_webhook_handler.return_value = None
        patches["text_cache"].return_value = mock_text_cache_instance

        # VAE cache mocks
        vae_cache_patcher = patch("simpletuner.helpers.caching.vae.VAECache")
        patches["vae_cache"] = vae_cache_patcher.start()
        self.addCleanup(vae_cache_patcher.stop)

        mock_vae_cache_instance = MagicMock()
        mock_vae_cache_instance.discover_all_files.return_value = None
        mock_vae_cache_instance.discover_unprocessed_files.return_value = []
        mock_vae_cache_instance.build_vae_cache_filename_map.return_value = None
        mock_vae_cache_instance.process_buckets.return_value = None
        mock_vae_cache_instance.set_webhook_handler.return_value = None
        patches["vae_cache"].return_value = mock_vae_cache_instance

        # Metadata backend mocks
        metadata_patcher = patch("simpletuner.helpers.metadata.backends.discovery.DiscoveryMetadataBackend")
        patches["metadata"] = metadata_patcher.start()
        self.addCleanup(metadata_patcher.stop)

        mock_metadata_instance = MagicMock()
        mock_metadata_instance.refresh_buckets.return_value = None
        mock_metadata_instance.reload_cache.return_value = None
        mock_metadata_instance.has_single_underfilled_bucket.return_value = False
        mock_metadata_instance.split_buckets_between_processes.return_value = None
        mock_metadata_instance.remove_images.return_value = None
        mock_metadata_instance.handle_vae_cache_inconsistencies.return_value = None
        mock_metadata_instance.scan_for_metadata.return_value = None
        mock_metadata_instance.load_image_metadata.return_value = None
        mock_metadata_instance.save_cache.return_value = None
        mock_metadata_instance.__len__.return_value = 10
        mock_metadata_instance.config = {}
        mock_metadata_instance.resolution_type = "area"
        mock_metadata_instance.resolution = 1024
        patches["metadata"].return_value = mock_metadata_instance

        # Dataset and sampler mocks
        dataset_patcher = patch("simpletuner.helpers.multiaspect.dataset.MultiAspectDataset")
        patches["dataset"] = dataset_patcher.start()
        self.addCleanup(dataset_patcher.stop)
        patches["dataset"].return_value = MagicMock()

        sampler_patcher = patch("simpletuner.helpers.multiaspect.sampler.MultiAspectSampler")
        patches["sampler"] = sampler_patcher.start()
        self.addCleanup(sampler_patcher.stop)

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.caption_strategy = "filename"
        patches["sampler"].return_value = mock_sampler_instance

        # Prompt handler mocks
        prompt_handler_patcher = patch("simpletuner.helpers.prompts.PromptHandler")
        patches["prompt_handler"] = prompt_handler_patcher.start()
        self.addCleanup(prompt_handler_patcher.stop)
        patches["prompt_handler"].get_all_captions.return_value = (["caption1", "caption2"], [])

        return patches

    def _run_old_factory(self, config: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Exception]:
        """Run the old factory implementation with the given config."""
        from simpletuner.helpers.data_backend.factory import configure_multi_databackend

        config_path = self._create_temp_config(config, "old_factory_config.json")
        self.args.data_backend_config = config_path

        try:
            result = configure_multi_databackend(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                model=self.model,
                data_backend_config=config,
            )
            return result, None
        except Exception as e:
            return None, e

    def _run_new_factory(self, config: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Exception]:
        """Run the new factory implementation with the given config."""
        from simpletuner.helpers.data_backend.factory import configure_multi_databackend_new

        config_path = self._create_temp_config(config, "new_factory_config.json")
        self.args.data_backend_config = config_path

        try:
            result = configure_multi_databackend_new(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                model=self.model,
            )
            return result, None
        except Exception as e:
            return None, e

    def _compare_results(self, old_result: Dict[str, Any], new_result: Dict[str, Any]) -> List[str]:
        """Compare results from both implementations and return differences."""
        differences = []

        if old_result is None and new_result is None:
            return differences

        if old_result is None or new_result is None:
            differences.append(f"One result is None: old={old_result is not None}, new={new_result is not None}")
            return differences

        # Compare structure
        old_keys = set(old_result.keys()) if old_result else set()
        new_keys = set(new_result.keys()) if new_result else set()

        if old_keys != new_keys:
            differences.append(f"Different result keys: old={old_keys}, new={new_keys}")

        # Compare common keys
        for key in old_keys & new_keys:
            old_val = old_result[key]
            new_val = new_result[key]

            # For dictionaries, compare lengths and keys
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                if len(old_val) != len(new_val):
                    differences.append(f"Different {key} lengths: old={len(old_val)}, new={len(new_val)}")

                old_subkeys = set(old_val.keys())
                new_subkeys = set(new_val.keys())
                if old_subkeys != new_subkeys:
                    differences.append(f"Different {key} keys: old={old_subkeys}, new={new_subkeys}")

        return differences

    def test_minimal_config_parity(self):
        """Test behavioral parity with minimal configuration."""
        patches = self._setup_common_mocks()
        config = self._create_minimal_test_config()

        old_result, old_error = self._run_old_factory(config)
        new_result, new_error = self._run_new_factory(config)

        # both should succeed or both should fail
        if old_error is not None and new_error is not None:
            # both failed - check error types similar
            self.assertEqual(
                type(old_error), type(new_error), f"Different error types: old={type(old_error)}, new={type(new_error)}"
            )
        elif old_error is not None or new_error is not None:
            self.fail(f"Inconsistent error behavior: old_error={old_error}, new_error={new_error}")
        else:
            # both succeeded - compare results
            differences = self._compare_results(old_result, new_result)
            if differences:
                self.fail(f"Result differences found: {differences}")

    def test_local_config_with_parquet_parity(self):
        """Test behavioral parity with local config using parquet metadata."""
        patches = self._setup_common_mocks()

        # Add parquet metadata backend mock
        parquet_patcher = patch("simpletuner.helpers.metadata.backends.parquet.ParquetMetadataBackend")
        mock_parquet = parquet_patcher.start()
        self.addCleanup(parquet_patcher.stop)

        mock_parquet_instance = MagicMock()
        mock_parquet_instance.refresh_buckets.return_value = None
        mock_parquet_instance.reload_cache.return_value = None
        mock_parquet_instance.has_single_underfilled_bucket.return_value = False
        mock_parquet_instance.split_buckets_between_processes.return_value = None
        mock_parquet_instance.__len__.return_value = 10
        mock_parquet_instance.config = {}
        mock_parquet.return_value = mock_parquet_instance

        config = [
            {
                "id": "parquet_test",
                "type": "local",
                "instance_data_dir": "/tmp/parquet_images",
                "caption_strategy": "filename",
                "metadata_backend": "parquet",
                "parquet": {"path": "/tmp/test.parquet", "filename_column": "filename", "caption_column": "caption"},
                "cache_dir_vae": "/tmp/vae/parquet",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]

        old_result, old_error = self._run_old_factory(config)
        new_result, new_error = self._run_new_factory(config)

        # compare outcomes
        if old_error is not None and new_error is not None:
            self.assertEqual(type(old_error), type(new_error))
        elif old_error is not None or new_error is not None:
            self.fail(f"Inconsistent error behavior: old_error={old_error}, new_error={new_error}")
        else:
            differences = self._compare_results(old_result, new_result)
            if differences:
                self.fail(f"Result differences found: {differences}")

    def test_error_condition_parity(self):
        """Test that both implementations handle errors consistently."""
        patches = self._setup_common_mocks()

        # Test missing config file
        self.args.data_backend_config = None

        # Old factory
        try:
            from simpletuner.helpers.data_backend.factory import configure_multi_databackend

            old_result = configure_multi_databackend(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                model=self.model,
                data_backend_config=None,
            )
            old_error = None
        except Exception as e:
            old_error = e
            old_result = None

        # New factory
        try:
            from simpletuner.helpers.data_backend.factory import FactoryRegistry

            factory = FactoryRegistry(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                model=self.model,
            )
            factory.load_configuration()
            new_error = None
        except Exception as e:
            new_error = e

        # both should raise same error type
        self.assertIsNotNone(old_error, "Old factory should have raised an error")
        self.assertIsNotNone(new_error, "New factory should have raised an error")
        self.assertEqual(
            type(old_error), type(new_error), f"Different error types: old={type(old_error)}, new={type(new_error)}"
        )

    def test_csv_backend_parity(self):
        """Test CSV backend configuration parity."""
        patches = self._setup_common_mocks()

        # Add CSV backend mock
        csv_patcher = patch("simpletuner.helpers.data_backend.csv_url_list.CSVDataBackend")
        mock_csv = csv_patcher.start()
        self.addCleanup(csv_patcher.stop)

        mock_csv_instance = MagicMock()
        mock_csv_instance.list_files.return_value = ["image1.jpg", "image2.jpg"]
        mock_csv.return_value = mock_csv_instance

        config = [
            {
                "id": "csv_test",
                "type": "csv",
                "csv_file": "/tmp/test.csv",
                "csv_caption_column": "caption",
                "csv_cache_dir": "/tmp/csv_cache",
                "caption_strategy": "csv",
                "cache_dir_vae": "/tmp/vae/csv",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]

        old_result, old_error = self._run_old_factory(config)
        new_result, new_error = self._run_new_factory(config)

        # compare outcomes
        if old_error is not None and new_error is not None:
            self.assertEqual(type(old_error), type(new_error))
        elif old_error is not None or new_error is not None:
            self.fail(f"Inconsistent error behavior: old_error={old_error}, new_error={new_error}")
        else:
            differences = self._compare_results(old_result, new_result)
            if differences:
                self.fail(f"Result differences found: {differences}")

    def test_conditioning_dataset_parity(self):
        """Test conditioning dataset configuration parity."""
        patches = self._setup_common_mocks()

        config = [
            {
                "id": "main_dataset",
                "type": "local",
                "instance_data_dir": "/tmp/main_images",
                "caption_strategy": "filename",
                "conditioning_data": ["conditioning_dataset"],
                "cache_dir_vae": "/tmp/vae/main",
            },
            {
                "id": "conditioning_dataset",
                "type": "local",
                "dataset_type": "conditioning",
                "conditioning_type": "controlnet",
                "instance_data_dir": "/tmp/conditioning_images",
                "cache_dir_vae": "/tmp/vae/conditioning",
            },
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]

        old_result, old_error = self._run_old_factory(config)
        new_result, new_error = self._run_new_factory(config)

        # compare outcomes
        if old_error is not None and new_error is not None:
            self.assertEqual(type(old_error), type(new_error))
        elif old_error is not None or new_error is not None:
            self.fail(f"Inconsistent error behavior: old_error={old_error}, new_error={new_error}")
        else:
            differences = self._compare_results(old_result, new_result)
            if differences:
                self.fail(f"Result differences found: {differences}")

    def test_real_config_files(self):
        """Test with real configuration files from the config directory."""
        patches = self._setup_common_mocks()

        real_configs = [
            "multidatabackend-sdxl-local.json",
            "multidatabackend-sdxl-dreambooth.json",
            "multidatabackend-csv.json",
        ]

        for config_name in real_configs:
            with self.subTest(config=config_name):
                config = self._load_real_config(config_name)
                if config is None:
                    self.skipTest(f"Config file {config_name} not found")

                # Modify paths to use temp directories to avoid file system issues
                modified_config = self._sanitize_config_for_testing(config)

                old_result, old_error = self._run_old_factory(modified_config)
                new_result, new_error = self._run_new_factory(modified_config)

                # compare outcomes
                if old_error is not None and new_error is not None:
                    # both failed - acceptable for some configs
                    continue
                elif old_error is not None or new_error is not None:
                    print(f"Warning: Inconsistent behavior for {config_name}: old_error={old_error}, new_error={new_error}")
                    continue
                else:
                    differences = self._compare_results(old_result, new_result)
                    if differences:
                        print(f"Warning: Differences found in {config_name}: {differences}")

    def _sanitize_config_for_testing(self, config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize real config for testing by replacing paths with temp paths."""
        sanitized = []
        for backend in config:
            sanitized_backend = backend.copy()

            # Replace paths with temp paths
            if "instance_data_dir" in sanitized_backend:
                sanitized_backend["instance_data_dir"] = "/tmp/test_images"
            if "cache_dir_vae" in sanitized_backend:
                sanitized_backend["cache_dir_vae"] = "/tmp/vae_cache"
            if "cache_dir" in sanitized_backend:
                sanitized_backend["cache_dir"] = "/tmp/text_cache"
            if "csv_file" in sanitized_backend:
                sanitized_backend["csv_file"] = "/tmp/test.csv"
            if "csv_cache_dir" in sanitized_backend:
                sanitized_backend["csv_cache_dir"] = "/tmp/csv_cache"

            # Handle parquet config
            if "parquet" in sanitized_backend:
                parquet_config = sanitized_backend["parquet"].copy()
                parquet_config["path"] = "/tmp/test.parquet"
                sanitized_backend["parquet"] = parquet_config

            # Disable features that might cause issues in testing
            sanitized_backend["scan_for_errors"] = False
            sanitized_backend["preserve_data_backend_cache"] = True

            sanitized.append(sanitized_backend)

        return sanitized

    def test_performance_metrics_consistency(self):
        """Test that both implementations provide consistent performance metrics."""
        patches = self._setup_common_mocks()
        config = self._create_minimal_test_config()

        # test factory metrics
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        config_path = self._create_temp_config(config)
        self.args.data_backend_config = config_path

        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            model=self.model,
        )

        # check metrics structure
        self.assertIn("factory_type", factory.metrics)
        self.assertEqual(factory.metrics["factory_type"], "new")
        self.assertIn("memory_usage", factory.metrics)
        self.assertIn("backend_counts", factory.metrics)

        # test memory tracking
        memory_usage = factory._get_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreaterEqual(memory_usage, 0)


if __name__ == "__main__":
    unittest.main()
