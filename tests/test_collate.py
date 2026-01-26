import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.training.collate import collate_fn, describe_missing_conditioning_pairs
from simpletuner.helpers.training.state_tracker import StateTracker


class _StubModel:
    def __init__(
        self,
        requires_conditioning: bool = False,
        use_reference_embeds: bool = False,
        requires_conditioning_latents: bool = False,
        requires_conditioning_dataset: bool = False,
        requires_text_embed_image_context: bool = False,
    ):
        self._requires_conditioning = requires_conditioning
        self._use_reference_embeds = use_reference_embeds
        self._requires_conditioning_latents = requires_conditioning_latents
        self._requires_conditioning_dataset = requires_conditioning_dataset
        self._requires_text_embed_context = requires_text_embed_image_context

    def requires_conditioning_image_embeds(self):
        return self._requires_conditioning

    def requires_conditioning_latents(self):
        return self._requires_conditioning_latents

    def requires_conditioning_dataset(self):
        return self._requires_conditioning_dataset

    def collate_prompt_embeds(self, encoder_outputs):
        return {}

    def conditioning_image_embeds_use_reference_dataset(self):
        return self._use_reference_embeds

    def get_transforms(self, dataset_type: str | None = None):
        return None

    def requires_text_embed_image_context(self):
        return self._requires_text_embed_context


class _StubConditioningSample:
    def __init__(self, training_path: str, backend_id: str, caption=None, paired_training_path: str = None):
        self._training_path = training_path
        self._source_dataset_id = backend_id
        self._image_path = training_path
        self.caption = caption
        self.data_backend_id = backend_id
        self._paired_training_path = paired_training_path or training_path

    def training_sample_path(self, training_dataset_id: str):
        return self._paired_training_path

    def get_conditioning_type(self):
        return None

    def image_path(self, basename_only=False):
        return self._training_path


def _make_stub_data_backend():
    return SimpleNamespace(
        read_image=lambda *_: Image.new("RGB", (8, 8), color=0),
        get_abs_path=lambda path: path,
    )


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
                        "target_size": (32, 32),
                        "intermediary_size": (64, 64),
                    }
                ],
                "conditioning_samples": [],
            }
        ]

    def _patch_state_tracker(self, *, model, data_backend, text_outputs, backend_lookup=None):
        backend_lookup = backend_lookup or {}

        def _get_backend(backend_id):
            if backend_id in backend_lookup:
                return backend_lookup[backend_id]
            return data_backend

        backend_patcher = patch.object(StateTracker, "get_data_backend", side_effect=_get_backend)
        config_patcher = patch.object(
            StateTracker,
            "get_data_backend_config",
            return_value={"resolution": 1024, "resolution_type": "pixel", "instance_data_dir": "/train"},
        )
        patchers = [
            patch.object(StateTracker, "get_args", return_value=self.base_args),
            patch.object(StateTracker, "get_model", return_value=model),
            patch.object(StateTracker, "get_model_family", return_value="flux"),
            patch.object(StateTracker, "get_accelerator", return_value=SimpleNamespace(device="cpu")),
            patch.object(StateTracker, "get_weight_dtype", return_value=torch.float32),
            backend_patcher,
            config_patcher,
            patch("simpletuner.helpers.training.collate.compute_latents", return_value=[torch.zeros(1, 4, 4, 4)]),
            patch(
                "simpletuner.helpers.training.collate.check_latent_shapes",
                side_effect=lambda latents, *_, **__: latents,
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
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }

        model = _StubModel(requires_conditioning=False)
        patchers, backend_patcher = self._patch_state_tracker(  # pylint: disable=unused-variable
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
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
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }
        model = _StubModel(requires_conditioning=True)

        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            result = collate_fn(self.base_batch)

        cache = backend_dict["conditioning_image_embed_cache"]
        cache.retrieve_from_cache.assert_called_once_with("sample.png", caption=None)
        self.assertIsNotNone(result["conditioning_image_embeds"])

    def test_collate_fn_uses_reference_conditioning_embeds(self):
        conditioning_tensor = torch.ones(2, 4)
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=True),
            "conditioning_image_embed_cache": MagicMock(),
            "conditioning_data": [{"id": "cond-1"}],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        cond_backend = {
            "conditioning_image_embed_cache": MagicMock(retrieve_from_cache=MagicMock(return_value=conditioning_tensor)),
            "data_backend": SimpleNamespace(read_image=lambda *_: Image.new("RGB", (8, 8), color=0)),
            "config": {"instance_data_dir": "/cond"},
        }
        model = _StubModel(requires_conditioning=True, use_reference_embeds=True)
        conditioning_sample = _StubConditioningSample(
            "/cond/sample.png",
            "cond-1",
            caption="reference caption",
            paired_training_path="sample.png",
        )

        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            result = collate_fn(batch)

        cond_cache = cond_backend["conditioning_image_embed_cache"]
        cond_cache.retrieve_from_cache.assert_called_once_with("/cond/sample.png", caption="reference caption")
        self.assertEqual(result["conditioning_captions"], ["reference caption"])

    def test_conditioning_pixels_use_training_sample_path_mapping(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        cond_backend = {
            "id": "cond-1",
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/cond"},
            "conditioning_image_embed_cache": SimpleNamespace(),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "conditioning_data": [cond_backend],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(
            requires_conditioning=False,
            requires_conditioning_latents=True,
            requires_conditioning_dataset=True,
            requires_text_embed_image_context=True,
        )

        conditioning_sample = _StubConditioningSample(
            "/cond/ref.png",
            "cond-1",
            paired_training_path="/train/custom.png",
        )
        batch = copy.deepcopy(self.base_batch)
        batch[0]["training_samples"][0]["image_path"] = "/train/custom.png"
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )

        fake_pixels = torch.stack([torch.zeros(3, 8, 8)], dim=0)

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            with patch(
                "simpletuner.helpers.training.collate.conditioning_pixels",
                return_value=fake_pixels,
            ) as pixels_mock:
                result = collate_fn(batch)

        pixels_mock.assert_called_once()
        self.assertEqual(pixels_mock.call_args[0][1], ["/train/custom.png"])
        self.assertEqual(pixels_mock.call_args[0][2][0]["image_path"], "/train/custom.png")
        self.assertIsNotNone(result["conditioning_pixel_values"])
        self.assertIsNotNone(result["conditioning_latents"])

    def test_collate_fn_handles_slash_prefixed_training_pairs(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        cond_backend = {
            "id": "cond-1",
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/cond"},
            "conditioning_image_embed_cache": SimpleNamespace(),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "conditioning_data": [cond_backend],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(
            requires_conditioning=False,
            requires_conditioning_latents=True,
            requires_conditioning_dataset=True,
            requires_text_embed_image_context=True,
        )

        conditioning_sample = _StubConditioningSample("/cond/ref.png", "cond-1", paired_training_path="/sample.png")
        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )

        fake_pixels = torch.stack([torch.zeros(3, 8, 8)], dim=0)

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            with patch(
                "simpletuner.helpers.training.collate.conditioning_pixels",
                return_value=fake_pixels,
            ):
                result = collate_fn(batch)

        self.assertIsNotNone(result["conditioning_pixel_values"])
        self.assertIsNotNone(result["conditioning_latents"])

    def test_describe_missing_conditioning_pairs_reports_backend(self):
        examples = [{"image_path": "sample.png", "data_backend_id": "backend-1"}]
        conditioning_backends = [{"id": "cond-1"}, {"id": "cond-2"}]
        conditioning_examples = [_StubConditioningSample("/train/sample.png", "cond-1")]

        messages = describe_missing_conditioning_pairs(
            examples=examples,
            conditioning_examples=conditioning_examples,
            conditioning_backends=conditioning_backends,
            training_backend_id="backend-1",
            training_root="/train",
        )

        self.assertTrue(messages)
        self.assertIn("cond-2 missing 1 pair(s)", messages[0])
        self.assertIn("sample.png", messages[0])

    def test_collate_fn_includes_missing_conditioning_detail(self):
        text_outputs = {
            "prompt_embeds": torch.zeros(1, 1),
            "pooled_prompt_embeds": torch.zeros(1, 1),
        }
        backend_dict = {
            "config": {"instance_data_dir": "/train"},
            "conditioning_data": [{"id": "cond-1"}, {"id": "cond-2"}],
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(requires_conditioning=False)
        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )

        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [_StubConditioningSample("/train/sample.png", "cond-1")]

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            with self.assertRaisesRegex(ValueError, "cond-2 missing 1 pair\\(s\\)"):
                collate_fn(batch)


if __name__ == "__main__":
    unittest.main()
