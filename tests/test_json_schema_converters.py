import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from pydantic import ValidationError
from transformers.feature_extraction_utils import BatchFeature

from simpletuner.helpers.configuration.cmd_args import get_argument_parser
from simpletuner.helpers.models.ideogram.model import Ideogram4
from simpletuner.helpers.models.ideogram.prompting import Ideogram4JSONSchemaConverter
from simpletuner.helpers.models.json_schema import JSONSchemaConverter
from simpletuner.helpers.models.moss_image.model import MOSSImage
from simpletuner.helpers.models.moss_image.prompting import MossImageJSONSchemaConverter
from simpletuner.helpers.models.prompt_upsampling import PromptUpsamplingRequest, TransformersJSONPromptUpsampler
from simpletuner.helpers.training.validation import Validation, _precache_validation_prompt_embeddings


def native_layout():
    return {
        "summary": {"caption": "A photograph of 🟫 beneath a blue sign."},
        "items": [
            {"label": "character", "caption": "🟫 in the center", "bbox": [-192, -128, 192, 128]},
            {"label": "text", "caption": "Blue sign above 🟫", "text": "HELLO", "bbox": [-240, -180, -200, 180]},
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def model(**overrides):
    foundation = MOSSImage.__new__(MOSSImage)
    foundation.config = SimpleNamespace(
        model_flavour="sft",
        moss_image_auto_json=True,
        moss_image_prompt_upsample=False,
        moss_image_uncond_skip_layers=[22],
        resolution=512,
        **overrides,
    )
    foundation._encode_prompts = Mock(side_effect=lambda prompts, is_negative_prompt=False: prompts)
    foundation._format_text_embedding = lambda value: value
    return foundation


class JSONSchemaConverterTests(unittest.TestCase):
    def test_converter_contract_and_model_factories(self):
        with self.assertRaises(TypeError):
            JSONSchemaConverter()
        self.assertIsInstance(model().get_json_schema_converter(), MossImageJSONSchemaConverter)
        ideogram = Ideogram4.__new__(Ideogram4)
        ideogram.config = SimpleNamespace(ideogram_auto_json=True)
        self.assertIsInstance(ideogram.get_json_schema_converter(), Ideogram4JSONSchemaConverter)
        ideogram.config.ideogram_auto_json = False
        self.assertIsNone(ideogram.get_json_schema_converter())
        foundation = model()
        foundation.config.moss_image_auto_json = False
        self.assertIsNone(foundation.get_json_schema_converter())
        self.assertEqual(foundation.encode_text_batch(["a caption"]), ["a caption"])

    def test_dropout_and_negative_entry_points_preserve_each_models_conditioning(self):
        ideogram = Ideogram4.__new__(Ideogram4)
        ideogram.config = SimpleNamespace(ideogram_auto_json=True)
        ideogram._encode_prompts = Mock(side_effect=lambda prompts, is_negative_prompt=False: prompts)
        ideogram._format_text_embedding = lambda value: value
        empty = ideogram.encode_dropout_caption()
        negative = ideogram.encode_validation_negative_prompt("not blurry")
        self.assertIn("compositional_deconstruction", json.loads(empty[0]))
        self.assertIn("not blurry", negative[0])
        foundation = model()
        self.assertEqual(foundation.encode_dropout_caption(), [""])
        self.assertEqual(foundation.encode_validation_negative_prompt("bad anatomy"), ["bad anatomy"])

    def test_native_schema_and_existing_json_remain_model_specific(self):
        converter = MossImageJSONSchemaConverter()
        self.assertEqual(converter.json_schema()["properties"]["bbox_format"]["const"], "yxyx")
        text = converter.convert(json.dumps(native_layout()), height=512, width=512)
        self.assertEqual(json.loads(text)["items"], native_layout()["items"])
        self.assertNotIn("high_level_description", json.loads(text))
        wrapped = json.loads(converter.convert("A photograph of 🟫."))
        self.assertEqual(wrapped["summary"]["caption"], "A photograph of 🟫.")
        self.assertEqual(wrapped["items"], [])
        self.assertEqual(converter.convert(""), "")

    def test_invalid_layouts_fail_instead_of_inventing_geometry(self):
        converter = MossImageJSONSchemaConverter()
        for change in (
            {"bbox_format": "xyxy"},
            {"bbox_coord_space": "normalized"},
            {"items": [{"label": "text", "caption": "sign", "bbox": [0, 0, 100, 100]}]},
            {"items": [{"label": "object", "caption": "subject", "bbox": [100, 0, 0, 100]}]},
            {"items": [{"label": "object", "caption": "subject", "bbox": [0, 0, 300, 100]}]},
        ):
            with self.subTest(change=change), self.assertRaises((ValueError, ValidationError)):
                converter.convert(json.dumps(native_layout() | change), height=512, width=512)
        with self.assertRaises(json.JSONDecodeError):
            converter.convert("{invalid json")

    def test_foundation_converts_encoding_and_pipeline_prompts_with_dimensions(self):
        foundation = model()
        layout = json.dumps(native_layout())
        result = foundation.encode_text_batch([layout, "", "🟫"], prompt_contexts=[{"height": 512, "width": 512}, {}, {}])
        self.assertEqual(json.loads(result[0])["items"], native_layout()["items"])
        self.assertEqual(result[1], "")
        self.assertEqual(json.loads(result[2])["summary"]["caption"], "🟫")
        self.assertEqual(foundation.encode_text_batch(["bad anatomy", ""], is_negative_prompt=True), ["bad anatomy", ""])
        self.assertIsNone(foundation._current_prompt_contexts)
        kwargs = foundation.update_pipeline_call_kwargs({"prompt": layout, "height": 512, "width": 512})
        self.assertEqual(json.loads(kwargs["prompt"])["items"], native_layout()["items"])
        with self.assertRaisesRegex(ValueError, "height"):
            foundation.encode_text_batch([layout], prompt_contexts=[{"height": 256, "width": 512}])

    def test_cache_namespaces_separate_schema_and_caption_model(self):
        foundation = model()
        schema_namespace = foundation.text_embed_cache_namespace()
        self.assertIn("moss-image-layout-v1", schema_namespace)
        foundation.config.moss_image_auto_json = False
        self.assertIsNone(foundation.text_embed_cache_namespace())
        foundation.config.moss_image_prompt_upsample = True
        foundation.config.moss_image_prompt_upsampler_model_id = "caption/model-a"
        first = foundation.text_embed_cache_namespace()
        foundation.config.moss_image_prompt_upsampler_model_id = "caption/model-b"
        self.assertNotEqual(first, foundation.text_embed_cache_namespace())

    def test_moss_caption_metadata_uses_real_target_images_and_bucket_size(self):
        foundation = model()
        foundation.config.moss_image_prompt_upsample = True
        backend = {
            "id": "images",
            "metadata_backend": SimpleNamespace(get_metadata_by_filepath=Mock(return_value={"target_size": (384, 672)})),
        }
        result = foundation.text_embed_cache_metadata_for_filepath(init_backend=backend, image_path="photo.png")
        self.assertEqual(result, {"width": 384, "height": 672, "image_paths": ["photo.png"], "data_backend_ids": ["images"]})
        lookup = foundation.text_embed_cache_metadata_for_sample(
            example={"target_size": (384, 672), "image_path": "photo.png", "data_backend_id": "images"}
        )
        self.assertEqual(result, lookup)
        self.assertEqual(foundation.text_embed_cache_context_key(result), foundation.text_embed_cache_context_key(lookup))

    def test_validation_precache_and_lookup_use_each_output_resolution(self):
        foundation = model()
        foundation.config.moss_image_prompt_upsample = True
        foundation.config.moss_image_prompt_upsampler_model_id = "caption/model"
        args = SimpleNamespace(model_family="moss_image", model_flavour="base", validation_resolution="512x768,1024x1024")
        cache = Mock()
        cache.compute_embeddings_for_prompts.return_value = {
            "prompt_embeds": torch.zeros(1, 2, 4),
            "attention_masks": torch.ones(1, 2, dtype=torch.bool),
        }
        _precache_validation_prompt_embeddings(
            args, cache, foundation, [{"prompt": "🟫", "key": "subject"}], is_validation=True, load_from_cache=False
        )
        records = cache.compute_embeddings_for_prompts.call_args.args[0]
        self.assertEqual(
            [record["metadata"] for record in records], [{"width": 512, "height": 768}, {"width": 1024, "height": 1024}]
        )
        self.assertNotEqual(
            foundation.text_embed_cache_context_key(records[0]["metadata"]),
            foundation.text_embed_cache_context_key(records[1]["metadata"]),
        )
        validation = Validation.__new__(Validation)
        validation.model = foundation
        validation.embed_cache = cache
        validation.inference_device = torch.device("cpu")
        with patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args):
            validation._gather_prompt_embeds("🟫", "subject", resolution=(512, 768))
        lookup = cache.compute_embeddings_for_prompts.call_args.args[0][0]
        self.assertEqual(lookup, records[0])

    def test_moss_options_parse_without_enabling_json_by_default(self):
        parser = get_argument_parser()
        required = [
            "--model_family",
            "moss_image",
            "--model_type",
            "lora",
            "--optimizer",
            "adamw_bf16",
            "--output_dir",
            "output",
            "--data_backend_config",
            "config/dataloader.json",
        ]
        defaults = parser.parse_args(required)
        self.assertFalse(defaults.moss_image_auto_json)
        self.assertFalse(defaults.moss_image_prompt_upsample)
        parsed = parser.parse_args(
            required
            + [
                "--moss_image_auto_json",
                "true",
                "--moss_image_prompt_upsample",
                "true",
                "--moss_image_prompt_upsampler_model_id",
                "caption/model",
            ]
        )
        self.assertTrue(parsed.moss_image_auto_json)
        self.assertTrue(parsed.moss_image_prompt_upsample)
        self.assertEqual(parsed.moss_image_prompt_upsampler_model_id, "caption/model")


class TransformersJSONUpsamplerTests(unittest.TestCase):
    def test_generation_uses_model_schema_images_and_removes_thinking(self):
        processor = Mock()
        processor.apply_chat_template.return_value = BatchFeature({"input_ids": torch.tensor([[1, 2]])})
        processor.tokenizer.get_vocab.return_value = {"</think>": 9}
        processor.decode.return_value = json.dumps(native_layout())
        language_model = Mock()
        language_model.eval.return_value = language_model
        language_model.requires_grad_.return_value = language_model
        language_model.generate.return_value = torch.tensor([[1, 2, 7, 9, 10, 11]])
        encoder = Mock()
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
            patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=language_model),
        ):
            upsampler = TransformersJSONPromptUpsampler(
                model_id="caption/model",
                converter=MossImageJSONSchemaConverter(),
                device=torch.device("cpu"),
                dtype=torch.float32,
                text_encoders=[encoder],
            )
        image = Image.new("RGB", (512, 512))
        result = upsampler.upsample(
            [PromptUpsamplingRequest("🟫 in front of a sign", height=512, width=512, reference_images=(image,))]
        )
        self.assertEqual(json.loads(result[0])["items"], native_layout()["items"])
        content = processor.apply_chat_template.call_args.args[0][0]["content"]
        self.assertIs(content[0]["image"], image)
        self.assertIn("abs_centralized", content[1]["text"])
        self.assertIn("512 pixels wide by 512 pixels high", content[1]["text"])
        self.assertEqual(processor.decode.call_args.args[0].tolist(), [10, 11])
        encoder.to.assert_called_with("cpu")
        language_model.to.assert_called_with("cpu")
        processor.decode.return_value = "not JSON"
        with self.assertRaises(ValueError):
            upsampler.upsample([PromptUpsamplingRequest("caption", height=512, width=512)])


if __name__ == "__main__":
    unittest.main()
