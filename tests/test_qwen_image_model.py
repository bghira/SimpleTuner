import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class _DummyTransformer:
    def __init__(self, output=None, forward=None):
        self.config = SimpleNamespace(patch_size=2)
        self.output = output
        self.forward = forward
        self.call_args = None

    def __call__(self, **kwargs):
        self.call_args = SimpleNamespace(kwargs=kwargs)
        if self.forward is not None:
            return self.forward(**kwargs)
        return self.output


class QwenImageModelTests(unittest.TestCase):
    def setUp(self):
        self.model = QwenImage.__new__(QwenImage)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
        self.model.config = SimpleNamespace(weight_dtype=torch.float32, twinflow_enabled=False, controlnet=False)
        self.model.vae_scale_factor = 8
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model._force_packed_transformer_output = lambda model: contextlib.nullcontext()
        self.model._is_edit_v1_flavour = lambda: False
        self.model._is_edit_v2_flavour = lambda: False
        self.model._is_edit_v2_plus_flavour = lambda: False
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=False,
            candidate_count=1,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )
        self.model.model = _DummyTransformer()

        class _Pipeline:
            @staticmethod
            def _pack_latents(latents, batch_size, num_channels, latent_height, latent_width):
                return torch.randn(batch_size, (latent_height // 2) * (latent_width // 2), num_channels * 4)

            @staticmethod
            def _unpack_latents(latents, pixel_height, pixel_width, vae_scale_factor):
                batch_size, _, channels = latents.shape
                latent_h = pixel_height // vae_scale_factor
                latent_w = pixel_width // vae_scale_factor
                out_channels = channels // 4
                return torch.randn(batch_size, out_channels, latent_h, latent_w)

        self.model.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: _Pipeline}

    def _enable_xm(self, *, training_target="noise", selection_scope="sample", block_size=0, candidate_count=2):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )

    def test_validate_xm_support_rejects_unsupported_qwen_settings(self):
        self._enable_xm(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

        self._enable_xm(selection_scope="block", block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

        self._enable_xm(block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self.model._validate_xm_support()

    def test_xm_noise_candidates_expand_qwen_conditioning_candidate_major(self):
        self._enable_xm(candidate_count=3)
        latents = torch.arange(2 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 2, 2)
        candidate_noise = torch.full((6, 1, 2, 2), 4.0)
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75]),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2]),
            "image_ids": torch.tensor([[10, 11], [20, 21]]),
            "metadata": ["a", "b"],
            "control_latent_list": [[torch.ones(1, 2, 2)], [torch.full((1, 2, 2), 2.0)]],
            "conditioning_latents": [torch.full((2, 1, 2, 2), 3.0)],
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.tensor([0.1, 0.2, 0.1, 0.2, 0.1, 0.2])))
        self.assertEqual(batch["metadata"], ["a", "b", "a", "b", "a", "b"])
        self.assertEqual(len(batch["control_latent_list"]), 6)
        self.assertEqual(tuple(batch["conditioning_latents"][0].shape), (6, 1, 2, 2))
        sigma_grid = batch["sigmas"].view(6, 1, 1, 1)
        expected_noisy = (1.0 - sigma_grid) * batch["latents"] + sigma_grid * candidate_noise
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.allclose(batch["flow_target"], candidate_noise - batch["latents"]))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_xm_loss_selects_winners_before_nextlat_auxiliary_loss(self):
        self._enable_xm()
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": ["a", "b", "a", "b"],
            "control_latent_list": [["a0"], ["b0"], ["a1"], ["b1"]],
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "crepa_hidden_states": hidden.clone(),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], ["a", "b"])
        self.assertEqual(prepared_batch["control_latent_list"], [["a1"], ["b0"]])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertIn("xm_winner_indices", prepared_batch)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

        class FakeNextLat:
            enabled = True

            def __init__(self):
                self.hidden_shape = None
                self.prediction_shape = None

            def compute_loss(self, hidden_states_buffer, output):
                self.hidden_shape = tuple(hidden_states_buffer["layer_0"].shape)
                self.prediction_shape = tuple(output["model_prediction"].shape)
                return torch.tensor(0.5), {"nextlat_loss": 0.5}

        nextlat = FakeNextLat()
        self.model.nextlat_regularizer = nextlat
        self.model.crepa_regularizer = None
        self.model.internal_guidance_regularizer = None
        self.model._twinflow_active = MagicMock(return_value=False)
        aux_loss, aux_logs = self.model.auxiliary_loss(
            model_output=model_output,
            prepared_batch=prepared_batch,
            loss=loss,
            apply_layersync=False,
            clear_hidden_state_buffer=False,
        )

        self.assertEqual(nextlat.hidden_shape, (2, 3, 2))
        self.assertEqual(nextlat.prediction_shape, (2, 1, 1, 1))
        self.assertAlmostEqual(aux_loss.item(), 0.5)
        self.assertEqual(aux_logs["nextlat_loss"], 0.5)

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_3"] = torch.full((1, 4, 8), 3.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = _DummyTransformer(forward=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([500.0]),
            "prompt_embeds": torch.randn(1, 2, 16),
            "crepa_capture_block_index": 7,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.5], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        self.model.model = _DummyTransformer(output=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 250.0, 750.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 2, 16),
        }

        self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.25, 0.75]], dtype=torch.float32),
            )
        )

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.config.crepa_self_flow_mask_ratio = 0.5
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.75], dtype=torch.float32), torch.tensor([750.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4),
            "input_noise": torch.ones(1, 16, 4, 4),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = self.model._prepare_crepa_self_flow_batch(batch, state={"global_step": 0})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 4]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))
        self.assertEqual(updated["crepa_teacher_noisy_latents"].shape, torch.Size([1, 16, 4, 4]))


if __name__ == "__main__":
    unittest.main()
