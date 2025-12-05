import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.training.validation import Validation
from simpletuner.simpletuner_sdk.server.services.prompt_library_service import PromptLibraryEntry


class ValidationAdapterStrengthTests(unittest.TestCase):
    def setUp(self):
        self.validator = Validation.__new__(Validation)
        self.validator.validation_prompt_metadata = {}

    def test_prepare_validation_work_items_captures_strength(self):
        entries = [
            PromptLibraryEntry(prompt="p1", adapter_strength=0.3),
            {"prompt": "p2", "adapter_strength": "0.7"},
            "plain",
        ]
        work_items = self.validator._prepare_validation_work_items(entries)
        strengths = [item.adapter_strength for item in work_items]
        self.assertEqual(strengths[:2], [0.3, 0.7])
        self.assertIsNone(strengths[2])

    def test_set_adapter_strength_updates_peft_layers(self):
        class DummyLoraLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.active_adapters = ["default", "assistant"]
                self.scales = {}

            def set_scale(self, adapter_name: str, strength: float):
                self.scales[adapter_name] = strength

        class DummyComponent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = DummyLoraLayer()
                self.add_module("layer", self.layer)

        class DummyModel:
            def __init__(self, component):
                self.component = component
                self.assistant_adapter_name = "assistant"

            def get_trained_component(self, unwrap_model: bool = False):
                return self.component

        component = DummyComponent()
        validator = Validation.__new__(Validation)
        validator.model = DummyModel(component)
        validator.accelerator = SimpleNamespace()
        validator.config = SimpleNamespace(model_type="lora", lora_type="standard", validation_adapter_strength=0.9)

        with patch("simpletuner.helpers.training.validation.LoraLayer", DummyLoraLayer):
            validator._set_adapter_strength(0.25)
            self.assertEqual(component.layer.scales["default"], 0.25)
            self.assertNotIn("assistant", component.layer.scales)
            validator._set_adapter_strength(None)
            self.assertAlmostEqual(component.layer.scales["default"], 0.9)


if __name__ == "__main__":
    unittest.main()
