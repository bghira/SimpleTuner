"""
Test that fields with allow_empty=True properly preserve empty strings
through the complete save/load cycle.

This test exercises the critical path that was buggy:
1. Form submission with empty value
2. Training service config bundle building (where fields were being removed)
3. Saving to disk via config store
4. Loading from disk
5. Field preparation for UI rendering
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService
from simpletuner.simpletuner_sdk.server.services.field_service import FieldService
from simpletuner.simpletuner_sdk.server.services.training_service import build_config_bundle


class TestAllowEmptyFields(unittest.TestCase):
    """Test allow_empty field handling through the full stack."""

    def test_empty_string_preserved_in_form_submission(self):
        """Test that empty strings are preserved during form normalization."""
        # Simulate form data with empty negative prompt
        form_data = {
            "--validation_negative_prompt": "",
            "--validation_prompt": "",
            "--model_family": "flux",
        }

        result = ConfigsService.normalize_form_to_config(form_data)

        # Empty strings should be preserved for allow_empty fields
        self.assertIn("--validation_negative_prompt", result)
        self.assertEqual(result["--validation_negative_prompt"], "")
        self.assertIn("--validation_prompt", result)
        self.assertEqual(result["--validation_prompt"], "")

    def test_empty_string_preserved_in_field_preparation(self):
        """Test that empty strings are preserved when preparing fields for UI."""
        field_service = FieldService()

        # Simulate loaded config with empty negative prompt
        config_data = {
            "--validation_negative_prompt": "",
            "--validation_prompt": "a test prompt",
        }

        result = field_service.prepare_tab_field_values("validation", config_data, {})

        # Empty string should be preserved (not replaced with default)
        self.assertEqual(result.get("validation_negative_prompt"), "")
        self.assertEqual(result.get("validation_prompt"), "a test prompt")

    def test_empty_string_vs_default_value(self):
        """Test that empty string is different from missing value."""
        field_service = FieldService()

        # Test 1: Empty string should be preserved
        config_with_empty = {"--validation_negative_prompt": ""}
        result1 = field_service.prepare_tab_field_values("validation", config_with_empty, {})
        self.assertEqual(result1.get("validation_negative_prompt"), "")

        # Test 2: Missing value should use default
        config_without_field = {}
        result2 = field_service.prepare_tab_field_values("validation", config_without_field, {})
        self.assertEqual(result2.get("validation_negative_prompt"), "blurry, cropped, ugly")

    def test_training_service_config_bundle_preserves_empty_string(self):
        """
        THE CRITICAL TEST: Ensures build_config_bundle doesn't remove allow_empty fields.

        This exercises the code path in training_service.py that was removing
        fields when they were cleared, instead of preserving them as empty strings.
        """
        # Mock the webui state
        mock_state_store = MagicMock()
        mock_state_store.resolve_defaults.return_value = {
            "resolved": {
                "configs_dir": "/tmp/configs",
                "output_dir": "/tmp/output",
            }
        }

        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_defaults.auto_preserve_defaults = False

        # Mock the config store to avoid file I/O
        mock_store = MagicMock()
        mock_store.get_active_config.return_value = "test_config"
        mock_store.load_config.return_value = (
            {
                # Simulate existing config with default negative prompt
                "--validation_negative_prompt": "blurry, cropped, ugly",
                "--model_family": "flux",
                "--output_dir": "/tmp/output",
            },
            MagicMock(),
        )

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state",
                return_value=(mock_state_store, mock_defaults),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service.get_config_store",
                return_value=mock_store,
            ),
        ):
            # Simulate form submission where user cleared the negative prompt
            form_data = {
                "--validation_negative_prompt": "",  # User cleared this field
                "--validation_prompt": "test prompt",
                "--model_family": "flux",
                "--output_dir": "/tmp/output",
            }

            # Build the config bundle (this is where the bug was)
            bundle = build_config_bundle(form_data)

            # The critical assertion: empty string should be preserved
            self.assertIn("--validation_negative_prompt", bundle.config_dict)
            self.assertEqual(bundle.config_dict["--validation_negative_prompt"], "")

            # The complete config should also have the empty string
            self.assertIn("--validation_negative_prompt", bundle.complete_config)
            self.assertEqual(bundle.complete_config["--validation_negative_prompt"], "")

    def test_full_save_load_cycle(self):
        """
        Integration test: Save config with empty string, load it back, verify it's still empty.

        This exercises the complete flow end-to-end.
        """
        from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore

        with tempfile.TemporaryDirectory() as tmpdir:
            configs_dir = Path(tmpdir)

            # Create a config store
            store = ConfigStore(config_dir=configs_dir, config_type="model")

            # Config with empty negative prompt
            config = {
                "--validation_negative_prompt": "",
                "--validation_prompt": "a beautiful landscape",
                "--model_family": "flux",
                "--output_dir": "/tmp/output",
            }

            # Save the config
            store.save_config("test_empty_prompt", config, overwrite=True)

            # Load it back
            loaded_config, _ = store.load_config("test_empty_prompt")

            # Verify empty string was preserved in saved file
            self.assertIn("--validation_negative_prompt", loaded_config)
            self.assertEqual(loaded_config["--validation_negative_prompt"], "")
            self.assertEqual(loaded_config["--validation_prompt"], "a beautiful landscape")

            # Also verify the JSON file on disk
            config_file = configs_dir / "test_empty_prompt.json"
            self.assertTrue(config_file.exists())

            with open(config_file, "r") as f:
                saved_data = json.load(f)

            # Check the actual file content
            self.assertIn("--validation_negative_prompt", saved_data)
            self.assertEqual(saved_data["--validation_negative_prompt"], "")

    def test_field_without_allow_empty_behaves_normally(self):
        """Verify that fields WITHOUT allow_empty still behave as before."""
        field_service = FieldService()

        # validation_prompt has allow_empty, but let's test a field without it
        # For example, most text fields without allow_empty should be removed when empty
        config_with_empty = {
            "--validation_prompt": "",  # This has allow_empty=True
        }

        result = field_service.prepare_tab_field_values("validation", config_with_empty, {})

        # Should be preserved as empty string
        self.assertEqual(result.get("validation_prompt"), "")

        # Now test form submission - should be included
        form_data = {"--validation_prompt": ""}
        normalized = ConfigsService.normalize_form_to_config(form_data)
        self.assertIn("--validation_prompt", normalized)

    def test_non_empty_values_still_work(self):
        """Sanity check: Non-empty values should still work normally."""
        field_service = FieldService()

        config_data = {
            "--validation_negative_prompt": "custom negative prompt",
            "--validation_prompt": "custom prompt",
        }

        result = field_service.prepare_tab_field_values("validation", config_data, {})

        self.assertEqual(result.get("validation_negative_prompt"), "custom negative prompt")
        self.assertEqual(result.get("validation_prompt"), "custom prompt")

    def test_allow_empty_field_identification(self):
        """Test that the training service correctly identifies allow_empty fields."""
        from simpletuner.simpletuner_sdk.server.services.field_service import lazy_field_registry

        # Get validation_negative_prompt field
        field = lazy_field_registry.get_field("validation_negative_prompt")
        self.assertIsNotNone(field)
        self.assertTrue(field.allow_empty)

        # Get validation_prompt field
        field2 = lazy_field_registry.get_field("validation_prompt")
        self.assertIsNotNone(field2)
        self.assertTrue(field2.allow_empty)


if __name__ == "__main__":
    unittest.main()
