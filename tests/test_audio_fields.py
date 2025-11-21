import unittest

from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType


class TestAudioFields(unittest.TestCase):
    """Test that audio configuration fields are properly registered."""

    def setUp(self):
        self.registry = FieldRegistry()

    def test_audio_fields_registered(self):
        """Test that all audio fields are present in the registry."""
        audio_fields = [
            "audio_max_duration_seconds",
            "audio_min_duration_seconds",
            "audio_channels",
            "audio_duration_interval",
            "audio_truncation_mode",
        ]

        for field_name in audio_fields:
            field = self.registry.get_field(field_name)
            self.assertIsNotNone(field, f"Field {field_name} should be registered")
            self.assertEqual(field.tab, "basic", f"Field {field_name} should be in basic tab")
            self.assertEqual(field.section, "dataset_config", f"Field {field_name} should be in dataset_config section")
            self.assertEqual(field.subsection, "audio", f"Field {field_name} should be in audio subsection")

    def test_audio_field_types(self):
        """Test field types for audio configurations."""
        self.assertEqual(self.registry.get_field("audio_max_duration_seconds").field_type, FieldType.NUMBER)
        self.assertEqual(self.registry.get_field("audio_truncation_mode").field_type, FieldType.SELECT)
        self.assertEqual(self.registry.get_field("audio_channels").field_type, FieldType.NUMBER)

    def test_audio_field_defaults(self):
        """Test default values for audio fields."""
        self.assertEqual(self.registry.get_field("audio_channels").default_value, 1)
        self.assertEqual(self.registry.get_field("audio_duration_interval").default_value, 3.0)
        self.assertEqual(self.registry.get_field("audio_truncation_mode").default_value, "beginning")


if __name__ == "__main__":
    unittest.main()
