import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.configuration.cmd_args import get_argument_parser
from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.models.flux3_image.model import Flux3Image
from simpletuner.helpers.models.flux3_image.prompt_upsampler import Flux3ImagePromptUpsampler, GeneratedTokenPresencePenalty
from simpletuner.helpers.models.ideogram.prompt_enhancer import generate_captions
from simpletuner.helpers.models.prompt_upsampling import (
    PromptUpsamplingRequest,
    reference_images_from_context,
    upsample_prompt_batch,
)
from simpletuner.helpers.training.validation import _validation_reference_prompt_metadata
from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry


class RecordingUpsampler:
    def upsample(self, requests):
        self.requests = requests
        return [f"expanded {request.prompt}" for request in requests]


def foundation(enabled=True):
    model = Flux3Image.__new__(Flux3Image)
    model.config = SimpleNamespace(flux3_prompt_upsample=enabled, resolution=512, weight_dtype=torch.float32)
    model.accelerator = SimpleNamespace(device=torch.device("cpu"))
    model.text_encoders = [torch.nn.Linear(1, 1)]
    model._prompt_upsampler = RecordingUpsampler()
    model._encode_prompts = Mock(side_effect=lambda prompts, is_negative_prompt: prompts)
    model._format_text_embedding = lambda encoded: encoded
    return model


class PromptUpsamplingTests(unittest.TestCase):
    def test_shared_encoding_rewrites_positive_prompts_and_preserves_empty_and_negative(self):
        model = foundation()
        reference = Image.new("RGB", (32, 16), "red")
        output = model.encode_text_batch(
            ["edit", "", "caption"],
            prompt_contexts=[{"reference_images": [reference], "height": 256, "width": 512}, {}, {}],
        )
        self.assertEqual(output, ["expanded edit", "", "expanded caption"])
        requests = model._prompt_upsampler.requests
        self.assertEqual(requests[0].reference_images[0].size, (32, 16))
        self.assertEqual((requests[0].height, requests[0].width), (256, 512))
        self.assertFalse(requests[1].reference_images)
        self.assertIsNone(model._current_prompt_contexts)
        self.assertEqual(model.encode_text_batch(["negative", ""], is_negative_prompt=True), ["negative", ""])
        self.assertEqual(model.encode_text_batch(["", " "]), ["", " "])
        self.assertIs(requests, model._prompt_upsampler.requests)

    def test_disabled_path_does_not_load_an_upsampler_or_change_caption_cache_keys(self):
        model = foundation(False)
        model.create_prompt_upsampler = Mock(side_effect=AssertionError("unexpected upsampler load"))
        model._prompt_upsampler = None
        self.assertEqual(model.encode_text_batch(["caption"]), ["caption"])
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.CAPTION)
        self.assertIsNone(model.text_embed_cache_namespace())
        self.assertFalse(model.requires_text_embed_image_context())
        self.assertIsNone(model.text_embed_cache_context_key({}))

    def test_edit_reference_paths_use_the_selected_backends_and_keep_order(self):
        red = Image.new("RGB", (32, 16), "red")
        blue = Image.new("RGB", (16, 48), "blue")
        backends = {
            "first": {"data_backend": SimpleNamespace(read_image=Mock(return_value=red))},
            "second": {"data_backend": SimpleNamespace(read_image=Mock(return_value=blue))},
        }
        context = {"image_paths": ["a.png", "b.png"], "data_backend_ids": ["first", "second"]}
        with patch("simpletuner.helpers.models.prompt_upsampling.StateTracker.get_data_backend", side_effect=backends.get):
            images = reference_images_from_context(context)
        self.assertEqual([image.size for image in images], [(32, 16), (16, 48)])
        backends["second"]["data_backend"].read_image.assert_called_once_with("b.png")
        self.assertFalse(reference_images_from_context({"image_path": "target.png", "data_backend_id": "target"}))

    def test_ragged_validation_images_preserve_colors_and_sizes(self):
        images = [Image.new("RGB", (32, 16), "red"), Image.new("RGB", (16, 48), "blue")]
        context = _validation_reference_prompt_metadata(images)
        restored = reference_images_from_context(context)
        self.assertEqual([image.size for image in restored], [(32, 16), (16, 48)])
        self.assertEqual(restored[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(restored[1].getpixel((0, 0)), (0, 0, 255))

    def test_cache_separates_plain_prompts_and_different_reference_selections(self):
        model = foundation(False)
        cache = TextEmbeddingCache.__new__(TextEmbeddingCache)
        cache.batch_write_thread = Thread()
        cache.model = model
        cache.model_type = "flux3_image"
        cache.key_type = TextEmbedCacheKey.CAPTION
        cache._requires_path_based_keys = False
        plain = cache.create_hash("caption")
        self.assertEqual(plain, hashlib.md5(b"caption").hexdigest() + "-flux3_image")
        model.config.flux3_prompt_upsample = True
        self.assertNotEqual(plain, cache.create_hash("caption"))
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)
        self.assertTrue(model.requires_text_embed_image_context())
        self.assertFalse(model.requires_text_embed_conditioning_pixels())
        first = cache.create_hash(
            "dataset:sample.png", prompt="edit", metadata={"image_paths": ["a.png"], "data_backend_ids": ["a"]}
        )
        second = cache.create_hash(
            "dataset:sample.png", prompt="edit", metadata={"image_paths": ["b.png"], "data_backend_ids": ["b"]}
        )
        self.assertNotEqual(first, second)
        dropout = cache.create_hash("__caption_dropout__", prompt="")
        self.assertEqual(
            dropout,
            cache.create_hash(
                "__caption_dropout__", prompt="", metadata={"image_paths": ["a.png"], "data_backend_ids": ["a"]}
            ),
        )
        cache.model = None
        with patch("simpletuner.helpers.caching.text_embeds.StateTracker.get_model", return_value=model):
            after_unload = cache.create_hash(
                "dataset:sample.png", prompt="edit", metadata={"image_paths": ["a.png"], "data_backend_ids": ["a"]}
            )
        self.assertEqual(first, after_unload)

    def test_invalid_context_or_backend_output_raises(self):
        with self.assertRaisesRegex(ValueError, "one entry"):
            upsample_prompt_batch(RecordingUpsampler(), ["x"], height=512, width=512, prompt_contexts=[])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            reference_images_from_context({"image_paths": ["a.png"], "data_backend_ids": []})
        for output in ([], [""], [None]):
            with self.subTest(output=output), self.assertRaises(ValueError):
                upsample_prompt_batch(Mock(upsample=Mock(return_value=output)), ["x"], height=512, width=512)

    def test_validation_cache_tracks_original_references_across_output_resolutions(self):
        model = foundation()
        cache = TextEmbeddingCache.__new__(TextEmbeddingCache)
        cache.batch_write_thread = Thread()
        cache.model = model
        cache.model_type = "flux3_image"
        cache.key_type = TextEmbedCacheKey.DATASET_AND_FILENAME
        cache._requires_path_based_keys = True
        original = [Image.new("RGB", (64, 32), "red"), Image.new("RGB", (32, 64), "blue")]
        prepared = [image.resize((512, 256)) for image in original]
        raw_context = model.validation_prompt_image_context(original, prepared)
        self.assertIs(raw_context, original)
        precache = _validation_reference_prompt_metadata(original, image_path="target.png")
        lookup = _validation_reference_prompt_metadata(raw_context)
        first = cache.hash_prompt({"key": "sample", "prompt": "edit", "metadata": precache})
        self.assertEqual(first, cache.hash_prompt({"key": "sample", "prompt": "edit", "metadata": lookup}))
        changed = _validation_reference_prompt_metadata([Image.new("RGB", (64, 32), "green"), original[1]])
        self.assertNotEqual(first, cache.hash_prompt({"key": "sample", "prompt": "edit", "metadata": changed}))
        model.config.flux3_prompt_upsample = False
        self.assertIs(model.validation_prompt_image_context(original, prepared), prepared)

    def test_option_is_opt_in_and_registered_for_cli_and_webui(self):
        field = FieldRegistry().get_field("flux3_prompt_upsample")
        self.assertFalse(field.default_value)
        self.assertEqual(field.model_specific, ["flux3_image"])
        args, _ = get_argument_parser().parse_known_args(
            [
                "--model_family",
                "flux3_image",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--output_dir",
                "output",
                "--data_backend_config",
                "config/dataloader.json",
                "--flux3_prompt_upsample",
                "true",
            ]
        )
        self.assertTrue(args.flux3_prompt_upsample)

    def test_presence_penalty_counts_only_new_unique_tokens(self):
        processor = GeneratedTokenPresencePenalty(2)
        scores = torch.zeros(1, 8)
        output = processor(torch.tensor([[1, 2, 3, 3, 4]]), scores)
        self.assertEqual(output.tolist(), [[0, 0, 0, -1.5, -1.5, 0, 0, 0]])
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))

    def test_flux_backend_uses_official_modes_and_releases_accelerator_on_error(self):
        module = "simpletuner.helpers.models.flux3_image.prompt_upsampler"
        processor = Mock()
        processor.apply_chat_template.side_effect = lambda *args, **kwargs: BatchFeature(
            {"input_ids": torch.tensor([[1, 2]])}
        )
        processor.decode.return_value = "expanded scene"
        language_model = torch.nn.Linear(1, 1)
        language_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4]]))
        encoder = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system_prompts.json"
            path.write_text(json.dumps({"t2i": "text mode", "i2i": "edit mode"}))
            with (
                patch(module + ".AutoProcessor.from_pretrained", return_value=processor),
                patch(module + ".Qwen3_5ForConditionalGeneration.from_pretrained", return_value=language_model),
                patch(module + ".hf_hub_download", return_value=str(path)),
            ):
                backend = Flux3ImagePromptUpsampler(torch.device("cpu"), torch.float32, [encoder])
        requests = [
            PromptUpsamplingRequest("scene", 512, 512),
            PromptUpsamplingRequest("edit", 512, 512, (Image.new("RGB", (16, 16)),)),
        ]
        self.assertEqual(backend.upsample(requests), ["expanded scene", "expanded scene"])
        text_messages = processor.apply_chat_template.call_args_list[0].args[0]
        edit_messages = processor.apply_chat_template.call_args_list[1].args[0]
        self.assertEqual(text_messages[0]["content"], "text mode")
        self.assertEqual(edit_messages[0]["content"], "edit mode")
        self.assertEqual(edit_messages[1]["content"][0]["type"], "image")
        self.assertFalse(processor.apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(len(language_model.generate.call_args_list[0].kwargs["logits_processor"]), 1)
        self.assertEqual(language_model.generate.call_args_list[1].kwargs["logits_processor"], [])
        self.assertTrue(all(not parameter.requires_grad for parameter in language_model.parameters()))
        language_model.generate.side_effect = RuntimeError("generation failed")
        with patch.object(language_model, "to", wraps=language_model.to) as move:
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                backend.upsample(requests[:1])
            self.assertEqual(move.call_args.args, ("cpu",))

    def test_ideogram_generator_passes_reference_pixels_and_edit_instructions(self):
        processor = Mock()
        processor.apply_chat_template.return_value = BatchFeature(
            {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(1, 3)}
        )
        tokenizer = Mock()
        tokenizer.decode.return_value = '{"high_level_description":"edited image"}'
        generator = Mock(device=torch.device("cpu"))
        generator.generate.return_value = torch.tensor([[1, 2, 3]])
        reference = Image.new("RGB", (16, 32), "red")
        output = generate_captions(
            generator, tokenizer, None, "change color", 512, 768, reference_images=(reference,), processor=processor
        )
        self.assertIn("edited image", output[0])
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertIn("requested final image", messages[0]["content"])
        self.assertIs(messages[1]["content"][0]["image"], reference)
        self.assertIn("pixel_values", generator.generate.call_args.kwargs)
