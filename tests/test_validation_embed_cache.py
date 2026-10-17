"""Validation precomputation must use the same cache keys as preview generation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.training.validation import _validation_negative_prompt_record, prepare_validation_prompt_list
from tests.test_text_embeds import _DummyThread, _make_cache


class TestValidationEmbedCacheKeys(unittest.TestCase):
    def test_precomputed_negative_prompts_can_be_loaded_without_an_encoder(self):
        for key_type in (TextEmbedCacheKey.CAPTION, TextEmbedCacheKey.DATASET_AND_FILENAME):
            for negative_prompt in ("", "blurry image"):
                with self.subTest(key_type=key_type, negative_prompt=negative_prompt):
                    args = SimpleNamespace(
                        model_family="flux3_image",
                        model_flavour="klein-base",
                        validation_disable_unconditional=False,
                        validation_prompt="a photograph of a toy",
                        validation_prompt_library=False,
                        user_prompt_library=None,
                        validation_negative_prompt=negative_prompt,
                        controlnet=False,
                        control=False,
                        validation_using_datasets=False,
                    )
                    cache = _make_cache(key_type)
                    cache.model_type = "flux3_image"
                    model = cache.model
                    model.uses_text_embeddings_cache = lambda: True
                    model.requires_conditioning_validation_inputs = lambda: False
                    model.validation_negative_prompt_requires_prompt_context = lambda: False
                    model.should_precompute_validation_negative_prompt = lambda: True
                    model.uses_validation_negative_prompt = lambda: True
                    model.log_model_devices = lambda: None
                    model.unpack_text_embeddings_from_cache = lambda embeddings: embeddings
                    model.VALIDATION_REQUIRES_EMPTY_NEGATIVE_PROMPT = True
                    cache._encode_text_batch = Mock(return_value={"prompt_embeds": torch.ones(1, 2, 3)})
                    stored = {}
                    cache.save_to_cache = lambda filename, embeddings: stored.__setitem__(filename, embeddings)

                    def load(filename):
                        if filename not in stored:
                            raise FileNotFoundError(filename)
                        return stored[filename]

                    cache.data_backend.torch_load = load
                    with (
                        patch("simpletuner.helpers.caching.text_embeds.Thread", return_value=_DummyThread()),
                        patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
                        patch(
                            "simpletuner.helpers.training.validation.StateTracker.get_validation_sample_images",
                            return_value=None,
                        ),
                        patch("simpletuner.helpers.caching.text_embeds.StateTracker.get_text_cache_files", return_value={}),
                    ):
                        prepare_validation_prompt_list(args, cache, model)
                        cache._encode_text_batch.side_effect = AssertionError("The encoder was unloaded")
                        negative_input = (
                            _validation_negative_prompt_record(args, model, negative_prompt, "validation")
                            if cache._requires_path_based_keys
                            else negative_prompt
                        )
                        result = cache.compute_embeddings_for_prompts(
                            [negative_input], is_validation=True, load_from_cache=True, is_negative_prompt=True
                        )
                    torch.testing.assert_close(result["prompt_embeds"], torch.ones(1, 2, 3))
