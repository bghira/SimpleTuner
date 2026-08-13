import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.minimaxmusic.condition_encoder import MiniMaxMusic3ConditionEncoder
from simpletuner.helpers.models.minimaxmusic.model import MiniMaxMusic
from simpletuner.helpers.models.minimaxmusic.transformer import MiniMaxMusic3Transformer1DModel
from simpletuner.helpers.models.registry import ModelRegistry


def _tiny_transformer(enable_time_sign_embed: bool = False) -> MiniMaxMusic3Transformer1DModel:
    return MiniMaxMusic3Transformer1DModel(
        in_channels=4,
        condition_dim=8,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=6,
        ff_inner_dim=16,
        rotary_dim=4,
        fourier_embedding_dim=8,
        enable_time_sign_embed=enable_time_sign_embed,
    )


class MiniMaxMusicModelTests(unittest.TestCase):
    def _build_model(self, transformer=None):
        model = MiniMaxMusic.__new__(MiniMaxMusic)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            flow_schedule_shift=1.0,
            input_perturbation=0.0,
            input_perturbation_steps=None,
            logit_mean=0.0,
            logit_std=1.0,
            twinflow_enabled=False,
            distillation_method=None,
            distillation_config={},
            lora_type="standard",
            controlnet=False,
            model_family="minimaxmusic",
            model_flavour="music3",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
        )
        model.model = transformer or _tiny_transformer(enable_time_sign_embed=True)
        model.condition_encoder = None
        model.crepa_regularizer = None
        model.layersync_regularizer = None
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.get_trained_component = lambda: model.model
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model._twinflow_active = lambda: False
        return model

    def test_registry_exposes_minimaxmusic_family(self):
        registered = ModelRegistry.get("minimaxmusic")

        self.assertIsNotNone(registered)
        self.assertEqual(registered.NAME, "MiniMax Music 3")
        self.assertIn("music3", registered.get_flavour_choices())

    def test_condition_encoder_resamples_frame_hidden_states(self):
        encoder = MiniMaxMusic3ConditionEncoder(
            condition_hidden_dim=8,
            num_condition_layers=2,
            out_dim=8,
            input_sampling_rate=24000,
            input_hop_length=960,
            output_sampling_rate=44100,
            output_hop_length=512,
        )

        output = encoder(torch.randn(1, 5, 16))

        self.assertEqual(output.shape, (1, 17, 8))

    def test_transformer_supports_tiny_forward_and_hidden_capture(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        hidden_states_buffer = {}

        output = transformer(
            hidden_states=torch.randn(1, 4, 6),
            timestep=torch.tensor([0.5]),
            encoder_hidden_states=torch.randn(1, 6, 8),
            timestep_sign=torch.tensor([-1.0]),
            r_timestep=torch.tensor([0.8]),
            skip_layers=[99],
            hidden_states_buffer=hidden_states_buffer,
            output_hidden_states=True,
            hidden_state_layer=1,
            return_dict=True,
        )

        self.assertEqual(output.sample.shape, (1, 4, 6))
        self.assertEqual(output.hidden_states.shape, (1, 6, 12))
        self.assertEqual(sorted(hidden_states_buffer), ["layer_0", "layer_1"])

    def test_transformer_accepts_tokenwise_timesteps_for_self_flow(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="t-r")

        output = transformer(
            hidden_states=torch.randn(1, 4, 5),
            timestep=torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.2]]),
            encoder_hidden_states=torch.randn(1, 5, 8),
            timestep_sign=torch.tensor([1.0]),
            r_timestep=torch.tensor([[0.1, 0.7, 0.1, 0.7, 0.1]]),
            return_dict=False,
        )

        self.assertEqual(output[0].shape, (1, 4, 5))

    def test_transformer_requires_initialized_timestep_conditioning(self):
        transformer = _tiny_transformer()
        kwargs = {
            "hidden_states": torch.randn(1, 4, 4),
            "timestep": torch.tensor([0.5]),
            "encoder_hidden_states": torch.randn(1, 4, 8),
        }

        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            transformer(**kwargs, timestep_sign=torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "FlowMap"):
            transformer(**kwargs, r_timestep=torch.tensor([0.25]))

    def test_prepare_batch_uses_precomputed_latents_and_data_ward_flow_target(self):
        model = self._build_model()
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.25, 0.75]), torch.tensor([0.75, 0.25])))
        batch = {
            "latent_batch": torch.randn(2, 6, 4),
            "encoder_hidden_states": torch.randn(2, 6, 8),
        }

        prepared = model.prepare_batch(batch, state={"global_step": 0})

        self.assertEqual(prepared["latents"].shape, (2, 4, 6))
        self.assertEqual(prepared["encoder_hidden_states"].shape, (2, 6, 8))
        self.assertEqual(prepared["sigmas"].shape, (2, 1, 1))
        torch.testing.assert_close(prepared["timesteps"], torch.tensor([0.75, 0.25]))
        torch.testing.assert_close(model.get_prediction_target(prepared), prepared["latents"] - prepared["noise"])

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        model = self._build_model()
        model.config.crepa_self_flow_mask_ratio = 0.5
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.7]), torch.tensor([0.3])))
        batch = {
            "latents": torch.zeros(1, 4, 5, dtype=torch.float32),
            "input_noise": torch.ones(1, 4, 5, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([0.8], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[0.2, 0.7, 0.1, 0.9, 0.4]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["sigmas"].shape, (1, 1, 5))
        self.assertEqual(result["timesteps"].shape, (1, 5))
        torch.testing.assert_close(result["timesteps"], torch.tensor([[0.3, 0.8, 0.3, 0.8, 0.3]]))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))
        torch.testing.assert_close(result["crepa_teacher_timesteps"], torch.tensor([0.8]))

    def test_model_predict_forwards_feature_kwargs_and_captures_hidden_states(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        model = self._build_model(transformer)
        model.config.twinflow_enabled = True
        model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=1, wants_hidden_states=lambda: True)
        hidden_states_buffer = {}
        model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(2, 4, 6),
                "encoder_hidden_states": torch.randn(2, 6, 8),
                "timesteps": torch.tensor([250.0, 750.0]),
                "sigmas": torch.tensor([0.25, 0.75]),
                "twinflow_time_sign": torch.tensor([-1.0, 1.0]),
                MiniMaxMusic.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.8, 0.6]),
            }
        )

        self.assertEqual(result["model_prediction"].shape, (2, 4, 6))
        self.assertEqual(result["crepa_hidden_states"].shape, (2, 6, 12))
        self.assertEqual(sorted(hidden_states_buffer), ["layer_0", "layer_1"])

    def test_timestep_helpers_convert_noise_sigmas_to_minimax_data_time(self):
        model = self._build_model()

        converted = model.flow_matching_timesteps_from_sigmas(torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(converted, torch.tensor([0.75, 0.25]))

        scalar_timesteps = model._timesteps_for_transformer(
            {"timesteps": torch.tensor([250.0]), "sigmas": torch.tensor([0.25])},
            torch.zeros(1, 4, 6),
        )
        torch.testing.assert_close(scalar_timesteps, torch.tensor([0.75]))

        tokenwise_timesteps = model._timesteps_for_transformer(
            {
                "timesteps": torch.tensor([[250.0, 750.0, 250.0]], dtype=torch.float32),
                "sigmas": torch.tensor([[[0.25, 0.75, 0.25]]], dtype=torch.float32),
            },
            torch.zeros(1, 4, 3),
        )
        torch.testing.assert_close(tokenwise_timesteps, torch.tensor([[0.75, 0.25, 0.75]]))


if __name__ == "__main__":
    unittest.main()
