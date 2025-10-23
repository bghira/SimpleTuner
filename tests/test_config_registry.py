"""Tests for the configuration registry system."""

import unittest

from simpletuner.helpers.configuration.doc_generator import ConfigDocumentationGenerator
from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_choice_rule,
    make_default_rule,
    make_required_rule,
)
from simpletuner.helpers.configuration.validator import ConfigValidationEngine
from simpletuner.helpers.data_backend.config_rules import register_dataloader_rules

# Import models and rules to register them once before tests
from simpletuner.helpers.models.flux.model import Flux


class RegistryTestBase(unittest.TestCase):
    """Base class for tests that need a clean registry."""

    def setUp(self):
        """Store current registry state."""
        self._saved_rules = ConfigRegistry._rules.copy()
        self._saved_validators = ConfigRegistry._validators.copy()
        self._saved_documentation = ConfigRegistry._documentation.copy()
        ConfigRegistry.clear_registry()

    def tearDown(self):
        """Restore registry state."""
        ConfigRegistry._rules = self._saved_rules
        ConfigRegistry._validators = self._saved_validators
        ConfigRegistry._documentation = self._saved_documentation


class TestConfigRegistry(RegistryTestBase):
    """Test the configuration registry functionality."""

    def test_register_single_rule(self):
        """Test registering a single rule."""
        rule = make_default_rule(field_name="test_field", default_value=42, message="Test field with default value")
        ConfigRegistry.register_rule("test", rule)

        rules = ConfigRegistry.get_rules("test")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].field_name, "test_field")
        self.assertEqual(rules[0].value, 42)

    def test_register_multiple_rules(self):
        """Test registering multiple rules at once."""
        rules = [
            make_default_rule("field1", 10, "First field"),
            make_required_rule("field2", "Second field is required"),
            make_choice_rule("field3", ["a", "b", "c"], "Third field choices"),
        ]
        ConfigRegistry.register_rules("test", rules)

        registered_rules = ConfigRegistry.get_rules("test")
        self.assertEqual(len(registered_rules), 3)
        self.assertEqual(registered_rules[0].field_name, "field1")
        self.assertEqual(registered_rules[1].field_name, "field2")
        self.assertEqual(registered_rules[2].field_name, "field3")

    def test_register_validator(self):
        """Test registering a custom validator."""

        def test_validator(config):
            return [ValidationResult(passed=True, field="test", message="Test passed")]

        ConfigRegistry.register_validator("test", test_validator, "Test validator documentation", ["Example 1", "Example 2"])

        validators = ConfigRegistry.get_validators("test")
        self.assertEqual(len(validators), 1)
        self.assertEqual(validators[0].doc, "Test validator documentation")
        self.assertEqual(len(validators[0].examples), 2)


class TestConfigValidator(RegistryTestBase):
    """Test the configuration validation engine."""

    def setUp(self):
        """Set up test configuration rules."""
        super().setUp()

        # Register test rules
        rules = [
            make_required_rule("required_field", "This field is required"),
            make_default_rule("optional_field", "default_value", "Optional field with default"),
            make_choice_rule("choice_field", ["option1", "option2"], "Field with choices"),
            ConfigRule(field_name="min_field", rule_type=RuleType.MIN, value=10, message="Minimum value is 10"),
            ConfigRule(field_name="max_field", rule_type=RuleType.MAX, value=100, message="Maximum value is 100"),
        ]
        ConfigRegistry.register_rules("test", rules)

    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = {"required_field": "present", "choice_field": "option1", "min_field": 50, "max_field": 50}

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "test")

        errors = engine.get_errors(results)
        self.assertEqual(len(errors), 0)

    def test_validate_missing_required(self):
        """Test validation fails on missing required field."""
        config = {"choice_field": "option1"}

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "test")

        errors = engine.get_errors(results)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].field, "required_field")

    def test_validate_invalid_choice(self):
        """Test validation fails on invalid choice."""
        config = {"required_field": "present", "choice_field": "invalid_option"}

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "test")

        errors = engine.get_errors(results)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].field, "choice_field")

    def test_validate_min_max_constraints(self):
        """Test min/max constraint validation."""
        # Test value below minimum
        config = {"required_field": "present", "min_field": 5}
        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "test")
        errors = engine.get_errors(results)
        self.assertTrue(any(e.field == "min_field" for e in errors))

        # Test value above maximum
        config = {"required_field": "present", "max_field": 150}
        results = engine.validate_config(config, "test")
        errors = engine.get_errors(results)
        self.assertTrue(any(e.field == "max_field" for e in errors))

    def test_format_results(self):
        """Test formatting validation results."""
        config = {"choice_field": "invalid"}

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "test")
        formatted = engine.format_results(results)

        self.assertIn("Configuration Errors", formatted)
        self.assertIn("required_field", formatted)
        self.assertIn("choice_field", formatted)


class TestConfigDocGenerator(RegistryTestBase):
    """Test the documentation generator."""

    def setUp(self):
        """Set up test rules for documentation."""
        super().setUp()

        rules = [
            make_required_rule(
                "api_key",
                "API key for authentication",
                example="api_key: your-secret-key",
                suggestion="Generate an API key from the dashboard",
            ),
            make_default_rule("timeout", 30, "Request timeout in seconds", example="timeout: 60"),
            make_choice_rule("log_level", ["debug", "info", "warning", "error"], "Logging level", example="log_level: info"),
        ]
        ConfigRegistry.register_rules("api", rules)
        ConfigRegistry.register_documentation("api", "API client configuration options")

    def test_generate_docs(self):
        """Test generating documentation structure."""
        generator = ConfigDocumentationGenerator()
        docs = generator.generate_all_docs()

        self.assertIn("models", docs)
        self.assertIn("base", docs)
        self.assertIn("all_fields", docs)

        # Check specific category docs
        api_docs = generator._generate_category_docs("api")
        self.assertEqual(api_docs["description"], "API client configuration options")
        self.assertEqual(len(api_docs["required"]), 1)
        self.assertEqual(len(api_docs["defaults"]), 1)
        self.assertEqual(len(api_docs["choices"]), 1)

    def test_generate_markdown(self):
        """Test generating markdown documentation."""
        generator = ConfigDocumentationGenerator()
        markdown = generator.to_markdown()

        self.assertIn("# SimpleTuner Configuration Reference", markdown)
        self.assertIn("## Table of Contents", markdown)
        self.assertIn("Required Fields", markdown)
        self.assertIn("Default Values", markdown)
        self.assertIn("Valid Choices", markdown)

    def test_generate_example_config(self):
        """Test generating example configuration."""
        generator = ConfigDocumentationGenerator()
        example = generator.generate_example_config("api")

        self.assertIn("api_key", example)
        self.assertEqual(example["timeout"], 30)  # Default value
        # data_backend_config is only added for actual model families, not 'api'


class TestFluxIntegration(unittest.TestCase):
    """Test Flux model integration with registry."""

    # No need for setUpClass since we import at module level

    def test_flux_rules_registered(self):
        """Test that Flux rules are properly registered."""
        rules = ConfigRegistry.get_rules("flux")
        self.assertGreater(len(rules), 0)

        # Check specific rules
        field_names = [r.field_name for r in rules]
        self.assertIn("aspect_bucket_alignment", field_names)
        self.assertIn("tokenizer_max_length", field_names)
        self.assertIn("base_model_precision", field_names)

    def test_flux_validation(self):
        """Test Flux-specific validation."""
        config = {
            "model_family": "flux",
            "aspect_bucket_alignment": 32,  # Should be overridden to 64
            "tokenizer_max_length": 1024,  # Should warn
            "prediction_type": "epsilon",  # Should warn
            "unet_attention_slice": True,  # Should warn on MPS
        }

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "flux")

        warnings = engine.get_warnings(results)
        self.assertGreaterEqual(len(warnings), 2)  # At least tokenizer_max_length and prediction_type


class TestDataloaderIntegration(unittest.TestCase):
    """Test dataloader configuration rules."""

    # No need for setUpClass since we import at module level

    def test_dataloader_rules_registered(self):
        """Test that dataloader rules are registered."""
        rules = ConfigRegistry.get_rules("dataloader")
        self.assertGreater(len(rules), 0)

        field_names = [r.field_name for r in rules]
        self.assertIn("data_backend_config", field_names)
        self.assertIn("datasets", field_names)

    def test_dataloader_validation(self):
        """Test dataloader validation."""
        # Invalid config - missing text embeds
        config = {
            "data_backend_config": "config.json",
            "datasets": [{"id": "images", "type": "local", "dataset_type": "image"}],
        }

        engine = ConfigValidationEngine()
        results = engine.validate_config(config, "dataloader")

        errors = engine.get_errors(results)
        self.assertTrue(any("text_embed" in e.message for e in errors))

        # Valid config
        config["datasets"].append({"id": "text", "dataset_type": "text_embeds", "type": "local", "default": True})

        results = engine.validate_config(config)
        errors = engine.get_errors(results)
        # Should have fewer errors now
        self.assertFalse(any("text_embed dataset" in e.message for e in errors))


if __name__ == "__main__":
    unittest.main()
