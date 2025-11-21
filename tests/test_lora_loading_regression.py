import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from simpletuner.helpers.models.common import ModelFoundation, ModelTypes


class TestModel(ModelFoundation):
    # Set the Enum value as it would be in a real model class
    MODEL_TYPE = ModelTypes.UNET

    def __init__(self):
        # We purposefully do NOT call super().__init__ to avoid full model initialization overhead.
        # We only set up what add_lora_adapter needs.
        self.config = MagicMock()
        self.config.lora_rank = 4
        self.config.lora_alpha = 4
        self.config.lora_dropout = 0.0
        self.config.lora_initialisation_style = "default"
        self.config.use_dora = False
        self.config.controlnet = False
        self.config.peft_lora_mode = None
        # This is the key trigger for the code path we want to test
        self.config.init_lora = "/path/to/dummy_lora.safetensors"

        self.model = MagicMock()
        # Mock add_adapter on the model so it doesn't fail
        self.model.add_adapter = MagicMock()

        self.accelerator = MagicMock()
        self.controlnet = None

    # Abstract methods we don't care about for this test
    def model_predict(self, *args, **kwargs):
        pass

    def _encode_prompts(self, *args, **kwargs):
        pass

    def convert_text_embed_for_pipeline(self, *args, **kwargs):
        pass

    def convert_negative_text_embed_for_pipeline(self, *args, **kwargs):
        pass

    def get_lora_target_layers(self):
        return ["to_k", "to_v"]


class TestLoraLoadingRegression(unittest.TestCase):

    @patch("simpletuner.helpers.models.common.load_lora_weights")
    def test_init_lora_passes_string_key(self, mock_load_lora_weights):
        """
        Regression test to ensure that when init_lora is set, the key passed to load_lora_weights
        is a string (e.g. 'unet') and not an Enum (e.g. ModelTypes.UNET).
        Passing an Enum causes a TypeError when concatenated with strings later.
        """
        # Setup
        model_instance = TestModel()

        # Mock return value
        mock_load_lora_weights.return_value = (set(), set())

        # Execute
        # This will call load_lora_weights internally because self.config.init_lora is set
        model_instance.add_lora_adapter()

        # Verify
        self.assertTrue(mock_load_lora_weights.called, "load_lora_weights should have been called")

        # Get the arguments passed to load_lora_weights
        # Signature is load_lora_weights(dictionary, filename, ...)
        call_args = mock_load_lora_weights.call_args
        dictionary_arg = call_args[0][0]

        # Check the dictionary keys
        self.assertIsInstance(dictionary_arg, dict)
        keys = list(dictionary_arg.keys())
        self.assertEqual(len(keys), 1)

        # CRITICAL CHECK: The key must be a string
        key = keys[0]
        self.assertIsInstance(key, str, f"Key passed to load_lora_weights must be str, got {type(key)}")
        self.assertEqual(key, "unet")


if __name__ == "__main__":
    unittest.main()
