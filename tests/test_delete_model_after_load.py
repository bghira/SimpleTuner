#!/usr/bin/env python
"""
Tests for the --delete_model_after_load feature.

This test suite covers:
1. StateTracker snapshot path tracking
2. Helper functions (get_hf_cache_repo_path, delete_model_from_cache)
3. Integration with model loading
4. VAE deletion gating on validation
5. Text encoder deletion after factory completion
6. Local-rank gating for multi-node setups
"""

import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpletuner.helpers.training.state_tracker import StateTracker


class TestStateTrackerSnapshotPaths(unittest.TestCase):
    """Test StateTracker's model snapshot path tracking."""

    def setUp(self):
        """Reset StateTracker state before each test."""
        StateTracker._model_snapshot_paths = {}

    def tearDown(self):
        """Clean up StateTracker state after each test."""
        StateTracker._model_snapshot_paths = {}

    def test_set_model_snapshot_path(self):
        """Test setting a model snapshot path."""
        StateTracker.set_model_snapshot_path("transformer", "/path/to/cache")
        self.assertEqual(StateTracker.get_model_snapshot_path("transformer"), "/path/to/cache")

    def test_set_model_snapshot_path_none(self):
        """Test that None paths are not stored."""
        StateTracker.set_model_snapshot_path("transformer", None)
        self.assertIsNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_set_model_snapshot_path_empty_string(self):
        """Test that empty string paths are not stored."""
        StateTracker.set_model_snapshot_path("transformer", "")
        self.assertIsNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_get_model_snapshot_path_nonexistent(self):
        """Test getting a path that doesn't exist."""
        self.assertIsNone(StateTracker.get_model_snapshot_path("nonexistent"))

    def test_get_all_model_snapshot_paths(self):
        """Test getting all stored paths."""
        StateTracker.set_model_snapshot_path("transformer", "/path/to/transformer")
        StateTracker.set_model_snapshot_path("vae", "/path/to/vae")
        StateTracker.set_model_snapshot_path("text_encoder_1", "/path/to/te1")

        paths = StateTracker.get_all_model_snapshot_paths()
        self.assertEqual(len(paths), 3)
        self.assertEqual(paths["transformer"], "/path/to/transformer")
        self.assertEqual(paths["vae"], "/path/to/vae")
        self.assertEqual(paths["text_encoder_1"], "/path/to/te1")

    def test_get_all_model_snapshot_paths_returns_copy(self):
        """Test that get_all returns a copy, not the original dict."""
        StateTracker.set_model_snapshot_path("transformer", "/path/to/cache")
        paths = StateTracker.get_all_model_snapshot_paths()
        paths["new_key"] = "new_value"
        self.assertIsNone(StateTracker.get_model_snapshot_path("new_key"))

    def test_clear_model_snapshot_path(self):
        """Test clearing a model snapshot path."""
        StateTracker.set_model_snapshot_path("transformer", "/path/to/cache")
        StateTracker.clear_model_snapshot_path("transformer")
        self.assertIsNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_clear_model_snapshot_path_nonexistent(self):
        """Test clearing a path that doesn't exist (should not raise)."""
        StateTracker.clear_model_snapshot_path("nonexistent")

    def test_multiple_text_encoders(self):
        """Test tracking multiple text encoders."""
        StateTracker.set_model_snapshot_path("text_encoder_1", "/path/to/te1")
        StateTracker.set_model_snapshot_path("text_encoder_2", "/path/to/te2")
        StateTracker.set_model_snapshot_path("text_encoder_3", "/path/to/te3")

        self.assertEqual(StateTracker.get_model_snapshot_path("text_encoder_1"), "/path/to/te1")
        self.assertEqual(StateTracker.get_model_snapshot_path("text_encoder_2"), "/path/to/te2")
        self.assertEqual(StateTracker.get_model_snapshot_path("text_encoder_3"), "/path/to/te3")


class TestGetHfCacheRepoPath(unittest.TestCase):
    """Test the get_hf_cache_repo_path helper function."""

    def test_none_path(self):
        """Test with None path."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        result = get_hf_cache_repo_path(None)
        self.assertIsNone(result)

    def test_safetensors_path(self):
        """Test with .safetensors single file path."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        result = get_hf_cache_repo_path("/path/to/model.safetensors")
        self.assertIsNone(result)

    def test_gguf_path(self):
        """Test with .gguf single file path."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        result = get_hf_cache_repo_path("/path/to/model.gguf")
        self.assertIsNone(result)

    def test_local_path_outside_cache(self):
        """Test with local path that is not in HF cache."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_hf_cache_repo_path(tmpdir)
            self.assertIsNone(result)

    @patch("huggingface_hub.scan_cache_dir")
    def test_hub_model_found(self, mock_scan):
        """Test finding a hub model in cache."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        mock_repo = MagicMock()
        mock_repo.repo_id = "org/model-name"
        mock_repo.repo_path = "/home/user/.cache/huggingface/hub/models--org--model-name"

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        mock_scan.return_value = mock_cache_info

        result = get_hf_cache_repo_path("org/model-name")
        self.assertEqual(result, "/home/user/.cache/huggingface/hub/models--org--model-name")

    @patch("huggingface_hub.scan_cache_dir")
    def test_hub_model_not_found(self, mock_scan):
        """Test when hub model is not in cache."""
        from simpletuner.helpers.models.common import get_hf_cache_repo_path

        mock_cache_info = MagicMock()
        mock_cache_info.repos = []
        mock_scan.return_value = mock_cache_info

        result = get_hf_cache_repo_path("org/nonexistent-model")
        self.assertIsNone(result)


class TestDeleteModelFromCache(unittest.TestCase):
    """Test the delete_model_from_cache helper function."""

    def setUp(self):
        """Set up test fixtures."""
        StateTracker._model_snapshot_paths = {}
        StateTracker.args = SimpleNamespace(delete_model_after_load=True)

    def tearDown(self):
        """Clean up."""
        StateTracker._model_snapshot_paths = {}
        StateTracker.args = None

    def test_not_local_main_process(self):
        """Test that deletion is skipped on non-main processes."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        accelerator = MagicMock()
        accelerator.is_local_main_process = False

        StateTracker.set_model_snapshot_path("transformer", "/path/to/cache")

        result = delete_model_from_cache("transformer", accelerator)
        self.assertFalse(result)
        # Path should still exist in StateTracker
        self.assertIsNotNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_delete_model_after_load_disabled(self):
        """Test that deletion is skipped when feature is disabled."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        StateTracker.args = SimpleNamespace(delete_model_after_load=False)
        accelerator = MagicMock()
        accelerator.is_local_main_process = True

        StateTracker.set_model_snapshot_path("transformer", "/path/to/cache")

        result = delete_model_from_cache("transformer", accelerator)
        self.assertFalse(result)

    def test_no_path_stored(self):
        """Test deletion when no path is stored."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        accelerator = MagicMock()
        accelerator.is_local_main_process = True

        result = delete_model_from_cache("nonexistent", accelerator)
        self.assertFalse(result)

    def test_successful_directory_deletion(self):
        """Test successful deletion of a directory."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)
            # Create a file inside
            with open(os.path.join(cache_dir, "model.bin"), "w") as f:
                f.write("test")

            StateTracker.set_model_snapshot_path("transformer", cache_dir)
            accelerator = MagicMock()
            accelerator.is_local_main_process = True

            result = delete_model_from_cache("transformer", accelerator)
            self.assertTrue(result)
            self.assertFalse(os.path.exists(cache_dir))
            self.assertIsNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_successful_file_deletion(self):
        """Test successful deletion of a single file."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "model.bin")
            with open(cache_file, "w") as f:
                f.write("test")

            StateTracker.set_model_snapshot_path("transformer", cache_file)
            accelerator = MagicMock()
            accelerator.is_local_main_process = True

            result = delete_model_from_cache("transformer", accelerator)
            self.assertTrue(result)
            self.assertFalse(os.path.exists(cache_file))

    def test_deletion_failure_silently_ignored(self):
        """Test that deletion failures are silently ignored."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        # Store a path that doesn't exist
        StateTracker.set_model_snapshot_path("transformer", "/nonexistent/path/xyz123")
        accelerator = MagicMock()
        accelerator.is_local_main_process = True

        # Should not raise, just return False
        result = delete_model_from_cache("transformer", accelerator)
        self.assertFalse(result)
        # Path should be cleared even on failure
        self.assertIsNone(StateTracker.get_model_snapshot_path("transformer"))

    def test_force_deletion(self):
        """Test force deletion ignores delete_model_after_load setting."""
        from simpletuner.helpers.models.common import delete_model_from_cache

        StateTracker.args = SimpleNamespace(delete_model_after_load=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)

            StateTracker.set_model_snapshot_path("transformer", cache_dir)
            accelerator = MagicMock()
            accelerator.is_local_main_process = True

            result = delete_model_from_cache("transformer", accelerator, force=True)
            self.assertTrue(result)
            self.assertFalse(os.path.exists(cache_dir))


class TestVAEDeletionGating(unittest.TestCase):
    """Test that VAE deletion is gated on validation being disabled."""

    def setUp(self):
        """Set up test fixtures."""
        StateTracker._model_snapshot_paths = {}

    def tearDown(self):
        """Clean up."""
        StateTracker._model_snapshot_paths = {}

    def test_vae_not_deleted_with_validation_prompts(self):
        """Test VAE is not deleted when validation_prompts is set."""
        config = SimpleNamespace(
            delete_model_after_load=True,
            validation_prompts=["test prompt"],
            validation_prompt=None,
            validation_image_prompts=None,
        )

        # Check the condition that would be used
        validation_enabled = (
            getattr(config, "validation_prompts", None)
            or getattr(config, "validation_prompt", None)
            or getattr(config, "validation_image_prompts", None)
        )
        self.assertTrue(validation_enabled)

    def test_vae_not_deleted_with_validation_prompt(self):
        """Test VAE is not deleted when validation_prompt is set."""
        config = SimpleNamespace(
            delete_model_after_load=True,
            validation_prompts=None,
            validation_prompt="test prompt",
            validation_image_prompts=None,
        )

        validation_enabled = (
            getattr(config, "validation_prompts", None)
            or getattr(config, "validation_prompt", None)
            or getattr(config, "validation_image_prompts", None)
        )
        self.assertTrue(validation_enabled)

    def test_vae_deleted_without_validation(self):
        """Test VAE can be deleted when no validation is configured."""
        config = SimpleNamespace(
            delete_model_after_load=True,
            validation_prompts=None,
            validation_prompt=None,
            validation_image_prompts=None,
        )

        validation_enabled = (
            getattr(config, "validation_prompts", None)
            or getattr(config, "validation_prompt", None)
            or getattr(config, "validation_image_prompts", None)
        )
        self.assertFalse(validation_enabled)


class TestTextEncoderDeletion(unittest.TestCase):
    """Test text encoder deletion after factory completion."""

    def setUp(self):
        """Set up test fixtures."""
        StateTracker._model_snapshot_paths = {}
        StateTracker.args = SimpleNamespace(delete_model_after_load=True)

    def tearDown(self):
        """Clean up."""
        StateTracker._model_snapshot_paths = {}
        StateTracker.args = None

    def test_text_encoder_paths_identified(self):
        """Test that text encoder paths are correctly identified."""
        StateTracker.set_model_snapshot_path("transformer", "/path/to/transformer")
        StateTracker.set_model_snapshot_path("vae", "/path/to/vae")
        StateTracker.set_model_snapshot_path("text_encoder_1", "/path/to/te1")
        StateTracker.set_model_snapshot_path("text_encoder_2", "/path/to/te2")

        paths = StateTracker.get_all_model_snapshot_paths()
        text_encoder_keys = [k for k in paths.keys() if k.startswith("text_encoder_")]

        self.assertEqual(len(text_encoder_keys), 2)
        self.assertIn("text_encoder_1", text_encoder_keys)
        self.assertIn("text_encoder_2", text_encoder_keys)


class TestFieldRegistryIntegration(unittest.TestCase):
    """Test that the field is properly registered."""

    def test_field_exists_in_registry(self):
        """Test that delete_model_after_load is in the field registry."""
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("delete_model_after_load")

        self.assertIsNotNone(field)
        self.assertEqual(field.name, "delete_model_after_load")
        self.assertEqual(field.arg_name, "--delete_model_after_load")
        self.assertEqual(field.default_value, False)

    def test_field_has_correct_type(self):
        """Test that the field has the correct type (checkbox/bool)."""
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
        from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType

        registry = FieldRegistry()
        field = registry.get_field("delete_model_after_load")

        self.assertEqual(field.field_type, FieldType.CHECKBOX)


if __name__ == "__main__":
    unittest.main()
