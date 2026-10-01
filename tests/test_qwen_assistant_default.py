import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.models.common import ImageModelFoundation
from simpletuner.helpers.models.qwen_image.model import QwenImage


class QwenAssistantDefaultTests(unittest.TestCase):
    def model(self, **kwargs):
        config = dict(model_flavour="v2.1", model_type="lora", aspect_bucket_alignment=32, prediction_type="flow_matching")
        config.update(kwargs)
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(**config)
        model.assistant_adapter_name = "assistant"
        model.model = Mock()
        model.unwrap_model = Mock(return_value=model.model)
        model._validate_xm_support = Mock()
        return model

    def check(self, model):
        with patch.object(ImageModelFoundation, "check_user_config"):
            model.check_user_config()

    def test_default_is_recorded_and_loaded(self):
        model = self.model()
        self.check(model)
        self.assertEqual(model.config.assistant_lora_path, "SimpleTuner/Qwen-Image-2.1-training-assistant-v2")
        self.assertEqual(model.config.assistant_lora_weight_name, "pytorch_lora_weights.safetensors")
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter", return_value=True) as load:
            model._maybe_load_assistant_lora()
        self.assertEqual(load.call_args.kwargs["lora_path"], model.config.assistant_lora_path)
        self.assertTrue(model.assistant_lora_loaded)

    def test_explicit_adapter_and_filename_are_preserved(self):
        model = self.model(assistant_lora_path="custom/adapter", assistant_lora_weight_name="custom.safetensors")
        self.check(model)
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter", return_value=True) as load:
            model._maybe_load_assistant_lora()
        self.assertEqual(load.call_args.kwargs["lora_path"], "custom/adapter")
        self.assertEqual(load.call_args.kwargs["weight_name"], "custom.safetensors")

    def test_disabled_full_training_and_older_flavours_do_not_load_default(self):
        for config in (
            {"disable_assistant_lora": True},
            {"model_type": "full"},
            {"model_flavour": "v2.0"},
            {"model_flavour": "v1.0"},
            {"model_flavour": "edit-v3"},
        ):
            with self.subTest(config=config):
                model = self.model(**config)
                self.check(model)
                with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter") as load:
                    model._maybe_load_assistant_lora()
                load.assert_not_called()
                self.assertFalse(getattr(model.config, "assistant_lora_path", None))

    def test_older_supported_flavour_can_still_load_explicit_adapter(self):
        model = self.model(model_flavour="v2.0", assistant_lora_path="custom/older")
        self.check(model)
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter", return_value=True) as load:
            model._maybe_load_assistant_lora()
        self.assertEqual(load.call_args.kwargs["lora_path"], "custom/older")
