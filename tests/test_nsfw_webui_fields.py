import unittest

from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
from simpletuner.simpletuner_sdk.server.services.field_registry.types import ImportanceLevel


class TestNsfwWebuiFields(unittest.TestCase):
    def setUp(self):
        self.registry = FieldRegistry()

    def test_nsfw_fields_are_basic_training_data_fields(self):
        nsfw_fields = [
            "enable_nsfw_check",
            "nsfw_check_models",
            "nsfw_check_min_votes",
            "nsfw_check_backend_types",
            "nsfw_check_sample_types",
            "delete_nsfw_images",
            "nsfw_check_video_frame_count",
            "nsfw_check_video_frame_selection",
            "nsfw_check_video_min_flagged_frames",
        ]

        for field_name in nsfw_fields:
            field = self.registry.get_field(field_name)
            self.assertIsNotNone(field, f"Field {field_name} should be registered")
            self.assertEqual(field.tab, "basic", f"Field {field_name} should be in the Basic tab")
            self.assertEqual(
                field.section,
                "training_data",
                f"Field {field_name} should be in the Training Data section",
            )

    def test_common_nsfw_fields_are_visible_by_default(self):
        common_fields = [
            "enable_nsfw_check",
            "nsfw_check_models",
            "nsfw_check_min_votes",
        ]

        for field_name in common_fields:
            field = self.registry.get_field(field_name)
            self.assertIsNotNone(field, f"Field {field_name} should be registered")
            self.assertIsNone(field.subsection, f"Field {field_name} should not be in an advanced subsection")
            self.assertEqual(field.importance, ImportanceLevel.IMPORTANT)

    def test_advanced_nsfw_fields_use_training_data_advanced_subsection(self):
        advanced_fields = [
            "nsfw_check_backend_types",
            "nsfw_check_sample_types",
            "delete_nsfw_images",
            "nsfw_check_video_frame_count",
            "nsfw_check_video_frame_selection",
            "nsfw_check_video_min_flagged_frames",
        ]

        for field_name in advanced_fields:
            field = self.registry.get_field(field_name)
            self.assertIsNotNone(field, f"Field {field_name} should be registered")
            self.assertEqual(field.subsection, "advanced")
            self.assertEqual(field.importance, ImportanceLevel.ADVANCED)


if __name__ == "__main__":
    unittest.main()
