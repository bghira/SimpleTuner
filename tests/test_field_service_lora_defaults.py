import unittest

from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.field_service import FieldService


class FieldServiceLoRADefaultsTests(unittest.TestCase):
    def setUp(self):
        lazy_field_registry.clear_cache()

    def test_lora_alpha_defaults_to_rank_when_unset(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 12}

        resolved = service.apply_field_transformations("lora_alpha", None, config)

        self.assertEqual(resolved, 12)

    def test_lora_alpha_preserves_explicit_values(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 16}

        resolved = service.apply_field_transformations("lora_alpha", 24, config)

        self.assertEqual(resolved, 24)

    def test_lora_alpha_field_dependency_only_on_model_type(self):
        field = lazy_field_registry.get_field("lora_alpha")
        self.assertIsNotNone(field, "Expected lora_alpha field to be registered")

        dependency_fields = {dependency.field for dependency in getattr(field, "dependencies", [])}

        self.assertEqual(dependency_fields, {"model_type"})


if __name__ == "__main__":
    unittest.main()
