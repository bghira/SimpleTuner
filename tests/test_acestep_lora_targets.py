import os
import sys
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from simpletuner.helpers.models.ace_step.model import ACEStep


class TestACEStepLoraTargets(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.accelerator = MagicMock()
        # Mock things required for ACEStep init
        self.config.tokenizer_max_length = 256
        self.config.flow_schedule_shift = 3.0
        # Avoid downloading model in init
        self.config.pretrained_model_name_or_path = "dummy_path"
        self.config.model_family = "ace_step"

    def test_default_target(self):
        self.config.acestep_lora_target = "attn_qkv+linear_qkv"
        self.config.controlnet = False

        # We can mock ACEStep.__init__ to avoid super().__init__ calls if needed,
        # but ACEStep.__init__ calls super().__init__ which might be heavy.
        # Let's try to instantiate it but mock out the heavy parts if possible.
        # Actually, let's just mock the get_lora_target_layers method context or
        # verify logic in isolation if possible. But we added the method to the class.

        # Better approach: Subclass ACEStep and mock out __init__ completely
        class MockACEStep(ACEStep):
            def __init__(self, config, accelerator):
                self.config = config
                self.accelerator = accelerator
                self.DEFAULT_LORA_TARGET = ACEStep.DEFAULT_LORA_TARGET
                self.DEFAULT_CONTROLNET_LORA_TARGET = ACEStep.DEFAULT_CONTROLNET_LORA_TARGET

        model = MockACEStep(self.config, self.accelerator)
        targets = model.get_lora_target_layers()
        expected = [
            "linear_q",
            "linear_k",
            "linear_v",
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ]
        self.assertEqual(set(targets), set(expected))

    def test_attn_qkv_target(self):
        self.config.acestep_lora_target = "attn_qkv"
        self.config.controlnet = False

        class MockACEStep(ACEStep):
            def __init__(self, config, accelerator):
                self.config = config
                self.accelerator = accelerator
                self.DEFAULT_LORA_TARGET = ACEStep.DEFAULT_LORA_TARGET
                self.DEFAULT_CONTROLNET_LORA_TARGET = ACEStep.DEFAULT_CONTROLNET_LORA_TARGET

        model = MockACEStep(self.config, self.accelerator)
        targets = model.get_lora_target_layers()
        expected = ["to_q", "to_k", "to_v", "to_out.0"]
        self.assertEqual(set(targets), set(expected))

    def test_speech_embedder_target(self):
        self.config.acestep_lora_target = "attn_qkv+linear_qkv+speech_embedder"
        self.config.controlnet = False

        class MockACEStep(ACEStep):
            def __init__(self, config, accelerator):
                self.config = config
                self.accelerator = accelerator
                self.DEFAULT_LORA_TARGET = ACEStep.DEFAULT_LORA_TARGET
                self.DEFAULT_CONTROLNET_LORA_TARGET = ACEStep.DEFAULT_CONTROLNET_LORA_TARGET

        model = MockACEStep(self.config, self.accelerator)
        targets = model.get_lora_target_layers()
        expected = [
            "speaker_embedder",
            "linear_q",
            "linear_k",
            "linear_v",
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ]
        self.assertEqual(set(targets), set(expected))


if __name__ == "__main__":
    unittest.main()
