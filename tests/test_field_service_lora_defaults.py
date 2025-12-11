import unittest

from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.field_service import FieldService


class FieldServiceLoRADefaultsTests(unittest.TestCase):
    def setUp(self):
        lazy_field_registry.clear_cache()

    def test_lora_alpha_stays_unset_without_user_value(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 16}

        resolved = service.apply_field_transformations("lora_alpha", None, config)

        self.assertIsNone(resolved)

    def test_lora_alpha_preserves_explicit_values(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 16}

        resolved = service.apply_field_transformations("lora_alpha", 24, config)

        self.assertEqual(resolved, 24)

    def test_prepare_tab_values_do_not_inject_alpha(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 12}

        values = service.prepare_tab_field_values("model", config, {})

        self.assertIsNone(values.get("lora_alpha"))
        self.assertIsNone(values.get("--lora_alpha"))
        hint = values.get("lora_alpha__hint", "")
        self.assertIn("LoRA rank", hint)
        self.assertIn("12", hint)

    def test_prepare_tab_values_keep_explicit_alpha(self):
        service = FieldService()
        config = {"model_type": "lora", "lora_rank": 16, "lora_alpha": 8}

        values = service.prepare_tab_field_values("model", config, {})

        self.assertEqual(values.get("lora_alpha"), 8)
        self.assertEqual(values.get("--lora_alpha"), 8)

    def test_lora_alpha_field_dependency_only_on_model_type(self):
        field = lazy_field_registry.get_field("lora_alpha")
        self.assertIsNotNone(field, "Expected lora_alpha field to be registered")

        dependency_fields = {dependency.field for dependency in getattr(field, "dependencies", [])}

        self.assertEqual(dependency_fields, {"model_type"})


if __name__ == "__main__":
    unittest.main()
