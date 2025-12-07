import unittest

from simpletuner.helpers.assistant_lora import build_adapter_stack


class DummyConfig:
    def __init__(self, *, train_strength=1.0, infer_strength=None, disable=False):
        self.assistant_lora_strength = train_strength
        if infer_strength is not None:
            self.assistant_lora_inference_strength = infer_strength
        self.disable_assistant_lora = disable


class AssistantLoraTests(unittest.TestCase):
    def test_build_adapter_stack_includes_default_and_assistant(self):
        peft_config = {"default": object(), "assistant": object()}
        adapter_names, weight_arg, freeze_names = build_adapter_stack(
            peft_config=peft_config, assistant_adapter_name="assistant", assistant_weight=0.5
        )
        self.assertEqual(adapter_names, ["assistant", "default"])
        self.assertEqual(weight_arg, [0.5, 1.0])
        self.assertEqual(freeze_names, ["assistant"])


if __name__ == "__main__":
    unittest.main()
