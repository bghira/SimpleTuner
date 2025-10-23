import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.training.collate import collate_fn
from simpletuner.helpers.training.state_tracker import StateTracker


class _StubModel:
    def __init__(self, requires_conditioning: bool = False):
        self._requires_conditioning = requires_conditioning

    def requires_conditioning_image_embeds(self):
        return self._requires_conditioning

    def collate_prompt_embeds(self, encoder_outputs):
        return {}


class CollateFunctionTests(unittest.TestCase):
    def setUp(self):
        self.base_args = SimpleNamespace(
            caption_dropout_probability=0.0,
            controlnet=False,
            model_family="flux",
            vae_cache_ondemand=False,
            data_aesthetic_score=0.0,
            conditioning_multidataset_sampling=None,
        )
        self.base_batch = [
            {
                "training_samples": [
                    {
                        "image_path": "sample.png",
                        "instance_prompt_text": "caption",
                        "luminance": 0.5,
                        "original_size": (64, 64),
                        "image_data": MagicMock(),
                        "crop_coordinates": [0, 0, 32, 32],
                        "data_backend_id": "backend-1",
                        "aspect_ratio": 1.0,
                    }
                ],
                "conditioning_samples": [],
            }
        ]

    def _patch_state_tracker(self, *, model, data_backend, text_outputs):
        backend_patcher = patch.object(StateTracker, "get_data_backend", return_value=data_backend)
        patchers = [
            patch.object(StateTracker, "get_args", return_value=self.base_args),
            patch.object(StateTracker, "get_model", return_value=model),
            patch.object(StateTracker, "get_model_family", return_value="flux"),
            patch.object(StateTracker, "get_accelerator", return_value=SimpleNamespace(device="cpu")),
            patch.object(StateTracker, "get_weight_dtype", return_value=torch.float32),
            backend_patcher,
            patch("simpletuner.helpers.training.collate.compute_latents", return_value=[torch.zeros(1, 4, 4, 4)]),
            patch(
                "simpletuner.helpers.training.collate.check_latent_shapes",
                side_effect=lambda latents, *_: latents,
            ),
            patch(
                "simpletuner.helpers.training.collate.compute_prompt_embeddings",
                return_value=text_outputs,
            ),
            patch("simpletuner.helpers.training.collate.gather_conditional_sdxl_size_features"),
            patch("simpletuner.helpers.training.collate.gather_conditional_pixart_size_features"),
            patch("torch.backends.mps.is_available", return_value=False),
        ]
        return patchers, backend_patcher

    def test_collate_fn_basic_path_uses_backend_for_text_cache(self):
        text_outputs = {
            "prompt_embeds": torch.zeros(1, 1),
            "pooled_prompt_embeds": torch.zeros(1, 1),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
        }

        model = _StubModel(requires_conditioning=False)
        patchers, backend_patcher = self._patch_state_tracker(  # pylint: disable=unused-variable
            model=model, data_backend=backend_dict, text_outputs=text_outputs
        )
        with ExitStack() as stack:
            active_mocks = [stack.enter_context(patcher) for patcher in patchers]
            result = collate_fn(self.base_batch)

        self.assertIn("latent_batch", result)
        self.assertTrue(torch.equal(result["prompt_embeds"], text_outputs["prompt_embeds"]))
        self.assertTrue(torch.equal(result["add_text_embeds"], text_outputs["pooled_prompt_embeds"]))
        backend_mock = active_mocks[5]
        backend_mock.assert_called()

    def test_collate_fn_stacks_conditioning_image_embeds(self):
        conditioning_tensor = torch.ones(2, 4)
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=True),
            "conditioning_image_embed_cache": MagicMock(retrieve_from_cache=MagicMock(return_value=conditioning_tensor)),
        }
        model = _StubModel(requires_conditioning=True)

        patchers, _ = self._patch_state_tracker(model=model, data_backend=backend_dict, text_outputs=text_outputs)
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            result = collate_fn(self.base_batch)

        cache = backend_dict["conditioning_image_embed_cache"]
        cache.retrieve_from_cache.assert_called_once_with("sample.png")
        self.assertIsNotNone(result["conditioning_image_embeds"])


if __name__ == "__main__":
    unittest.main()
