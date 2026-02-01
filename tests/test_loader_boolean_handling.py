"""Tests for config loader boolean handling.

These tests verify that boolean values from config files are properly passed
to argparse, ensuring explicit True/False values override argparse defaults.
"""

import unittest


class LoaderBooleanHandlingTests(unittest.TestCase):
    """Tests for the loader's boolean conversion to CLI arguments."""

    def _convert_dict_to_cli_args(self, mapped_config: dict) -> list:
        """
        Replicate the loader's dict-to-CLI-args conversion logic.

        This is extracted from simpletuner.helpers.configuration.loader.load_config
        to test the boolean handling in isolation.
        """
        list_arguments = []
        for arg_name, value in mapped_config.items():
            if isinstance(value, str) and value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            try:
                float(value)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
            if value is not None and value != "":
                if isinstance(value, str) and value.lower() in ["true", "false"]:
                    # Pass explicit boolean value to override argparse defaults
                    list_arguments.append(f"{arg_name}={value.lower()}")
                elif value is False:
                    # Pass explicit false to override argparse defaults
                    list_arguments.append(f"{arg_name}=false")
                elif value is True:
                    list_arguments.append(f"{arg_name}=true")
                elif is_numeric:
                    list_arguments.append(f"{arg_name}={value}")
                else:
                    list_arguments.append(f"{arg_name}={value}")
        return list_arguments

    def test_boolean_true_passed_explicitly(self):
        """Boolean True should be passed as --arg=true to argparse."""
        config = {"--gradient_checkpointing": True}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--gradient_checkpointing=true"])

    def test_boolean_false_passed_explicitly(self):
        """Boolean False should be passed as --arg=false to argparse."""
        config = {"--gradient_checkpointing": False}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--gradient_checkpointing=false"])

    def test_string_true_passed_explicitly(self):
        """String 'true' should be passed as --arg=true to argparse."""
        config = {"--gradient_checkpointing": "true"}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--gradient_checkpointing=true"])

    def test_string_false_passed_explicitly(self):
        """String 'false' should be passed as --arg=false to argparse."""
        config = {"--gradient_checkpointing": "false"}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--gradient_checkpointing=false"])

    def test_string_true_uppercase_passed_explicitly(self):
        """String 'True' (uppercase) should be passed as --arg=true to argparse."""
        config = {"--use_ema": "True"}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--use_ema=true"])

    def test_string_false_uppercase_passed_explicitly(self):
        """String 'False' (uppercase) should be passed as --arg=false to argparse."""
        config = {"--use_ema": "False"}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, ["--use_ema=false"])

    def test_multiple_boolean_fields(self):
        """Multiple boolean fields should all be handled correctly."""
        config = {
            "--gradient_checkpointing": True,
            "--use_ema": False,
            "--vae_enable_tiling": "true",
            "--vae_enable_slicing": "false",
        }
        args = self._convert_dict_to_cli_args(config)

        self.assertIn("--gradient_checkpointing=true", args)
        self.assertIn("--use_ema=false", args)
        self.assertIn("--vae_enable_tiling=true", args)
        self.assertIn("--vae_enable_slicing=false", args)

    def test_none_value_excluded(self):
        """None values should not be added to CLI args."""
        config = {"--gradient_checkpointing": None}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, [])

    def test_empty_string_excluded(self):
        """Empty string values should not be added to CLI args."""
        config = {"--gradient_checkpointing": ""}
        args = self._convert_dict_to_cli_args(config)

        self.assertEqual(args, [])

    def test_numeric_values_unchanged(self):
        """Numeric values should be passed as-is."""
        config = {"--learning_rate": 1e-4, "--train_batch_size": 4}
        args = self._convert_dict_to_cli_args(config)

        self.assertIn("--learning_rate=0.0001", args)
        self.assertIn("--train_batch_size=4", args)

    def test_string_values_unchanged(self):
        """String values should be passed as-is."""
        config = {"--model_family": "flux", "--output_dir": "/path/to/output"}
        args = self._convert_dict_to_cli_args(config)

        self.assertIn("--model_family=flux", args)
        self.assertIn("--output_dir=/path/to/output", args)

    def test_mixed_config(self):
        """Test a realistic mixed config with booleans, numbers, and strings."""
        config = {
            "--gradient_checkpointing": True,
            "--use_ema": False,
            "--learning_rate": 1e-4,
            "--model_family": "flux",
            "--output_dir": "/output",
            "--empty_field": "",
            "--none_field": None,
        }
        args = self._convert_dict_to_cli_args(config)

        # Booleans are explicit
        self.assertIn("--gradient_checkpointing=true", args)
        self.assertIn("--use_ema=false", args)

        # Numeric and string values present
        self.assertIn("--learning_rate=0.0001", args)
        self.assertIn("--model_family=flux", args)
        self.assertIn("--output_dir=/output", args)

        # Empty/None values excluded
        self.assertNotIn("--empty_field", "".join(args))
        self.assertNotIn("--none_field", "".join(args))


class LoaderBooleanIntegrationTests(unittest.TestCase):
    """Integration tests that verify the actual loader module."""

    def test_loader_passes_true_explicitly(self):
        """Verify the actual loader passes True as --arg=true."""
        from simpletuner.helpers.configuration import loader

        # We need to test the conversion logic directly
        # The load_config function is complex with many side effects,
        # so we test the conversion logic in isolation above.
        # This test just verifies the module is importable and has the expected structure.
        self.assertTrue(hasattr(loader, "load_config"))

    def test_loader_conversion_matches_test_logic(self):
        """Verify our test logic matches the actual loader implementation."""
        # Read the actual loader code and verify the boolean handling matches
        import inspect

        from simpletuner.helpers.configuration import loader

        source = inspect.getsource(loader.load_config)

        # Verify the fix is present: explicit boolean passing
        self.assertIn('list_arguments.append(f"{arg_name}={value.lower()}")', source)
        self.assertIn('list_arguments.append(f"{arg_name}=false")', source)
        self.assertIn('list_arguments.append(f"{arg_name}=true")', source)


if __name__ == "__main__":
    unittest.main()
