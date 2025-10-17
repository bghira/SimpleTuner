#!/usr/bin/env python
"""Direct comparison test between factory.py and factory.py"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactoryComparison(unittest.TestCase):
    """Compare outputs of factory.py vs factory.py for identical configs"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock accelerator
        self.accelerator = MagicMock()
        self.accelerator.is_main_process = True
        self.accelerator.is_local_main_process = True
        self.accelerator.device = "cuda"

        # Mock model
        self.model = MagicMock()
        self.model.requires_conditioning_latents.return_value = False
        self.model.get_vae.return_value = MagicMock()
        self.model.get_pipeline.return_value = MagicMock()

        # Mock text encoders
        self.text_encoders = [MagicMock(), MagicMock()]

        # Basic args
        self.args = MagicMock()
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
        self.args.compress_disk_cache = True
        self.args.delete_problematic_images = False
        self.args.metadata_update_interval = 60
        self.args.train_batch_size = 1
        self.args.aws_max_pool_connections = 128
        self.args.vae_cache_scan_behaviour = "ignore"
        self.args.vae_cache_ondemand = False
        self.args.offload_during_startup = False
        self.args.skip_file_discovery = ""
        self.args.delete_unwanted_images = False
        self.args.parquet_caption_column = None
        self.args.parquet_filename_column = None
        self.args.data_backend_sampling = "uniform"
        self.args.gradient_accumulation_steps = 1

    @patch("simpletuner.helpers.data_backend.factory.StateTracker")
    @patch("simpletuner.helpers.data_backend.factory.LocalDataBackend")
    @patch("simpletuner.helpers.data_backend.factory.TextEmbeddingCache")
    def test_simple_config_old_factory(self, mock_text_cache, mock_local_backend, mock_state_tracker):
        """Test simple config with old factory"""
        from simpletuner.helpers.data_backend.factory import configure_multi_databackend

        # Setup mocks
        mock_state_tracker.get_args.return_value = self.args
        mock_state_tracker.get_accelerator.return_value = self.accelerator
        mock_state_tracker.get_webhook_handler.return_value = None
        mock_state_tracker.get_data_backends.return_value = {}
        mock_state_tracker.get_model_family.return_value = "sdxl"

        config = [
            {"id": "test_backend", "type": "local", "instance_data_dir": "/tmp/test_images", "caption_strategy": "filename"},
            {
                "id": "text_embeds",
                "dataset_type": "text_embeds",
                "type": "local",
                "default": True,
                "cache_dir": "/tmp/text_cache",
            },
        ]

        try:
            result = configure_multi_databackend(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=[MagicMock(), MagicMock()],
                model=self.model,
                data_backend_config=config,
            )
            self.assertIsNotNone(result)
            # Check StateTracker calls
            self.assertTrue(mock_state_tracker.register_data_backend.called)
            self.assertTrue(mock_state_tracker.set_data_backend_config.called)
        except Exception as e:
            self.fail(f"Old factory failed: {str(e)}")

    @patch("simpletuner.helpers.data_backend.factory.StateTracker")
    @patch("simpletuner.helpers.data_backend.factory.LocalDataBackend")
    @patch("simpletuner.helpers.data_backend.factory.TextEmbeddingCache")
    def test_simple_config_new_factory(self, mock_text_cache, mock_local_backend, mock_state_tracker):
        """Test simple config with new factory"""
        try:
            from simpletuner.helpers.data_backend.factory import FactoryRegistry

            # Setup mocks
            mock_state_tracker.get_args.return_value = self.args
            mock_state_tracker.get_accelerator.return_value = self.accelerator
            mock_state_tracker.get_webhook_handler.return_value = None
            mock_state_tracker.get_data_backends.return_value = {}
            mock_state_tracker.get_model_family.return_value = "sdxl"

            config = [
                {
                    "id": "test_backend",
                    "type": "local",
                    "instance_data_dir": "/tmp/test_images",
                    "caption_strategy": "filename",
                },
                {
                    "id": "text_embeds",
                    "dataset_type": "text_embeds",
                    "type": "local",
                    "default": True,
                    "cache_dir": "/tmp/text_cache",
                },
            ]

            factory = FactoryRegistry(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=self.text_encoders,
                tokenizers=[MagicMock(), MagicMock()],
                model=self.model,
            )

            result = factory.configure(data_backend_config=config)
            self.assertIsNotNone(result)
            # Check StateTracker calls
            self.assertTrue(mock_state_tracker.register_data_backend.called)
            self.assertTrue(mock_state_tracker.set_data_backend_config.called)
        except Exception as e:
            self.fail(f"New factory failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()
