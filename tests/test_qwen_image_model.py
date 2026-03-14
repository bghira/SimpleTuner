import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import QwenImage


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
        self.model.model = MagicMock()
        self.model.model.config = SimpleNamespace(patch_size=2)

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

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_3"] = torch.full((1, 4, 8), 3.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

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
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

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
