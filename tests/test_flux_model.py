import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from diffusers.models.attention_processor import Attention

from simpletuner.helpers.models.flux.attention import FluxFusedFlashAttnProcessor3, FluxFusedSDPAProcessor
from simpletuner.helpers.models.flux.model import Flux
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig
from simpletuner.helpers.training.grounding.types import GroundingBatch


class FluxModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Flux.__new__(Flux)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weight_dtype=torch.float32,
            flux_guidance_mode="constant",
            flux_guidance_value=1.0,
            flux_attention_masked_training=False,
            twinflow_enabled=False,
            tread_config=None,
            model_flavour="kontext",
            crepa_self_flow_mask_ratio=0.0,
            fuse_qkv_projections=False,
        )
        self.model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.model.get_trained_component = MagicMock(
            return_value=SimpleNamespace(config=SimpleNamespace(guidance_embeds=False))
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([750.0]), torch.tensor([750.0])))

    def _enable_xm(self, candidate_count=2):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )

    def test_validate_xm_support_rejects_unsupported_flux_settings(self):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="route",
            selection_scope="sample",
            block_size=0,
        )

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="block",
            block_size=2,
        )

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

    def test_model_predict_expands_xm_candidates_and_repeats_conditioning(self):
        self._enable_xm(candidate_count=2)
        self.model.config.flux_guidance_mode = "random-range"
        self.model.config.flux_guidance_min = 1.0
        self.model.config.flux_guidance_max = 2.0
        self.model.get_trained_component = MagicMock(
            return_value=SimpleNamespace(config=SimpleNamespace(guidance_embeds=True))
        )
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.arange(4 * 5 * 2, dtype=torch.float32).view(4, 5, 2)
            return (torch.zeros(4, 5, 64),)

        self.model.model = MagicMock(side_effect=_forward)
        latents = torch.arange(2 * 16 * 4 * 4, dtype=torch.float32).view(2, 16, 4, 4)
        candidate_noise = torch.full((4, 16, 4, 4), 3.0)
        grounding_batch = GroundingBatch(
            boxes=torch.arange(2 * 1 * 4, dtype=torch.float32).view(2, 1, 4),
            validity_mask=torch.ones(2, 1),
            spatial_masks=torch.ones(2, 1, 2, 2),
            text_embeds=torch.ones(2, 1, 6),
            image_embeds=None,
            text_masks=torch.ones(2, 1),
            image_masks=torch.ones(2, 1),
            max_entities=1,
        )
        prepared_batch = {
            "noisy_latents": torch.zeros_like(latents),
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([100.0, 200.0]),
            "prompt_embeds": torch.randn(2, 3, 16),
            "add_text_embeds": torch.randn(2, 8),
            "conditioning_packed_latents": torch.randn(2, 1, 64),
            "conditioning_ids": torch.zeros(2, 1, 3),
            "grounding_batch": grounding_batch,
        }

        with (
            patch("torch.randn_like", return_value=candidate_noise),
            patch("random.uniform", side_effect=[1.25, 1.75]),
        ):
            result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(tuple(prepared_batch["latents"].shape), (4, 16, 4, 4))
        self.assertTrue(torch.equal(prepared_batch["latents"], latents.repeat(2, 1, 1, 1)))
        self.assertTrue(torch.equal(prepared_batch["noise"], candidate_noise))
        expected_noisy = 0.75 * prepared_batch["latents"] + 0.25 * candidate_noise
        self.assertTrue(torch.allclose(prepared_batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.equal(prepared_batch["flow_target"], candidate_noise - prepared_batch["latents"]))
        self.assertTrue(torch.equal(transformer_kwargs["guidance"], torch.tensor([1.25, 1.75, 1.25, 1.75])))
        self.assertEqual(tuple(transformer_kwargs["hidden_states"].shape), (4, 5, 64))
        self.assertEqual(tuple(transformer_kwargs["grounding_kwargs"]["boxes"].shape), (4, 1, 4))
        self.assertEqual(tuple(result["hidden_states_buffer"]["layer_2"].shape), (4, 5, 2))

    def test_xm_loss_selects_winners_before_nextlat_auxiliary_loss(self):
        self._enable_xm(candidate_count=2)
        self.model.config.loss_type = "l2"
        self.model.config.huber_schedule = "constant"
        self.model.config.huber_c = 0.1
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "conditioning_packed_latents": torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertEqual(tuple(prepared_batch["conditioning_packed_latents"].shape), (2, 2, 3))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
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
        aux_loss, aux_logs = self.model.auxiliary_loss(model_output, prepared_batch, loss)

        self.assertEqual(tuple(nextlat.hidden_shape), (2, 3, 2))
        self.assertEqual(tuple(nextlat.prediction_shape), (2, 1, 1, 1))
        self.assertAlmostEqual(aux_loss.item(), 0.5)
        self.assertEqual(aux_logs["nextlat_loss"], 0.5)

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_9"] = torch.full((1, 4, 8), 9.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([250.0]),
            "prompt_embeds": torch.randn(1, 3, 16),
            "add_text_embeds": torch.randn(1, 8),
            "crepa_capture_block_index": 9,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 16, 4, 4))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 9.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.25], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 500.0, 700.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 3, 16),
            "add_text_embeds": torch.randn(1, 8),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 16, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.5, 0.7]], dtype=torch.float32),
            )
        )

    def test_model_predict_appends_clean_conditioning_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.arange(6 * 8, dtype=torch.float32).view(1, 6, 8)
            return (torch.randn(1, 6, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 500.0, 700.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 3, 16),
            "add_text_embeds": torch.randn(1, 8),
            "conditioning_packed_latents": torch.randn(1, 2, 64),
            "conditioning_ids": torch.zeros(1, 2, 3),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 16, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.5, 0.7, 0.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertEqual(result["crepa_hidden_states"].shape, (1, 4, 8))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        batch = {
            "latents": torch.randn(1, 16, 4, 4),
            "input_noise": torch.randn(1, 16, 4, 4),
            "sigmas": torch.tensor([250.0]),
            "timesteps": torch.tensor([250.0]),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["crepa_self_flow_mask"].shape, (1, 2, 2))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))

    def test_unfuse_qkv_projections_uses_explicit_unfuse_method(self):
        attn = Attention(query_dim=8, heads=1, dim_head=8)
        attn.unfuse_projections = MagicMock()
        self.model.model = torch.nn.Sequential(attn)
        self.model.controlnet = None
        self.model.config.fuse_qkv_projections = True
        self.model._qkv_projections_fused = True

        self.model.unfuse_qkv_projections()

        attn.unfuse_projections.assert_called_once_with()
        self.assertFalse(self.model._qkv_projections_fused)

    def test_fused_qkv_processor_uses_sdpa_for_default_attention(self):
        self.model.config.attention_mechanism = "diffusers"

        processor = self.model._get_fused_qkv_attention_processor()

        self.assertIsInstance(processor, FluxFusedSDPAProcessor)

    def test_fused_qkv_processor_uses_packed_flash_for_flash_attention(self):
        self.model.config.attention_mechanism = "flash-attn-varlen-hub"
        backend = SimpleNamespace(capabilities=SimpleNamespace(fixed_qkvpacked=True))

        with patch("simpletuner.helpers.models.flux.attention.get_packed_attention_backend", return_value=backend):
            processor = self.model._get_fused_qkv_attention_processor()

        self.assertIsInstance(processor, FluxFusedFlashAttnProcessor3)


if __name__ == "__main__":
    unittest.main()
