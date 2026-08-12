import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.models.common import ImageModelFoundation


class QwenTextModel:
    pass


class QwenTokenizer:
    pass


class ClipTextModel:
    pass


class ClipTokenizer:
    pass


class DummyQwenFoundation(ImageModelFoundation):
    NAME = "Dummy Qwen"

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        raise NotImplementedError

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        raise NotImplementedError

    def convert_text_embed_for_pipeline(self, text_embedding):
        raise NotImplementedError

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        raise NotImplementedError


class QwenTextEncoderOverrideTests(unittest.TestCase):
    def _model(self, text_encoder_configuration):
        model = object.__new__(DummyQwenFoundation)
        model.config = SimpleNamespace(
            model_family="qwen_image",
            pretrained_model_name_or_path="base/model",
            qwen_text_encoder_model_name_or_path="custom/qwen",
        )
        model.TEXT_ENCODER_CONFIGURATION = text_encoder_configuration
        return model

    def test_single_qwen_encoder_uses_qwen_override_and_clears_component_subfolders(self):
        qwen_config = {
            "name": "Qwen2.5-VL",
            "tokenizer": QwenTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": QwenTextModel,
            "subfolder": "text_encoder",
        }
        clip_config = {
            "name": "CLIP-L/14",
            "tokenizer": ClipTokenizer,
            "tokenizer_subfolder": "tokenizer_2",
            "model": ClipTextModel,
            "subfolder": "text_encoder_2",
        }
        model = self._model(
            {
                "text_encoder": qwen_config,
                "text_encoder_2": clip_config,
            }
        )

        self.assertEqual(model._resolve_text_encoder_path(qwen_config), "custom/qwen")
        self.assertIsNone(model._resolve_text_encoder_subfolder(qwen_config, "subfolder", "text_encoder"))
        self.assertIsNone(model._resolve_text_encoder_subfolder(qwen_config, "tokenizer_subfolder", "tokenizer"))

        self.assertEqual(model._resolve_text_encoder_path(clip_config), "base/model")
        self.assertEqual(
            model._resolve_text_encoder_subfolder(clip_config, "subfolder", "text_encoder"),
            "text_encoder_2",
        )

    def test_multiple_qwen_encoders_ignore_override_and_warn(self):
        first_qwen_config = {
            "name": "Qwen3-A",
            "tokenizer": QwenTokenizer,
            "model": QwenTextModel,
            "subfolder": "text_encoder",
        }
        second_qwen_config = {
            "name": "Qwen3-B",
            "tokenizer": QwenTokenizer,
            "model": QwenTextModel,
            "subfolder": "text_encoder_2",
        }
        model = self._model(
            {
                "text_encoder": first_qwen_config,
                "text_encoder_2": second_qwen_config,
            }
        )

        with patch("simpletuner.helpers.models.common.logger.warning") as warning:
            self.assertEqual(model._resolve_text_encoder_path(first_qwen_config), "base/model")

        warning.assert_called_once()
        self.assertIn("Ignoring qwen_text_encoder_model_name_or_path", warning.call_args.args[0])
        self.assertEqual(warning.call_args.args[2], 2)
        self.assertEqual(
            model._resolve_text_encoder_subfolder(first_qwen_config, "subfolder", "text_encoder"),
            "text_encoder",
        )

    def test_webui_field_and_cli_parser_include_qwen_override(self):
        from simpletuner.helpers.configuration.cmd_args import get_argument_parser
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("qwen_text_encoder_model_name_or_path")

        self.assertIsNotNone(field)
        self.assertEqual(field.arg_name, "--qwen_text_encoder_model_name_or_path")
        self.assertEqual(field.documentation, "OPTIONS.md#--qwen_text_encoder_model_name_or_path")

        parser = get_argument_parser()
        args = parser.parse_args(
            [
                "--model_family",
                "krea2",
                "--output_dir",
                "/tmp/simpletuner-test",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--data_backend_config",
                "/tmp/backend.json",
                "--qwen_text_encoder_model_name_or_path",
                "custom/qwen",
            ]
        )

        self.assertEqual(args.qwen_text_encoder_model_name_or_path, "custom/qwen")

    def test_env_mapping_includes_qwen_override(self):
        from simpletuner.helpers.configuration.env_file import env_to_args_map

        self.assertEqual(
            env_to_args_map["QWEN_TEXT_ENCODER_MODEL_NAME_OR_PATH"],
            "--qwen_text_encoder_model_name_or_path",
        )


if __name__ == "__main__":
    unittest.main()
