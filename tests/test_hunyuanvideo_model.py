import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.hunyuanvideo.model import HunyuanVideo


class HunyuanVideoModelTests(unittest.TestCase):
    def test_model_supports_crepa_self_flow(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_builds_tokenwise_student_and_teacher_views(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=SimpleNamespace(patch_size=1, patch_size_t=1))
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.8], dtype=torch.float32), torch.tensor([800.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 2, 2, 2, 2, dtype=torch.float32),
            "input_noise": torch.ones(1, 2, 2, 2, 2, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor(
            [[[[0.2, 0.7], [0.9, 0.1]], [[0.4, 0.6], [0.8, 0.3]]]],
            dtype=torch.float32,
        )

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 8))
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 2, 2))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertEqual(result["crepa_teacher_timesteps"].item(), 200.0)
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            vision_num_semantic_tokens=4,
            vision_states_dim=6,
            text_embed_2_dim=4,
        )
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model._is_i2v_like_flavour = lambda: False
        model._prepare_cond_latents = lambda conditioning_latents, latents, task_type: (
            torch.zeros_like(latents),
            torch.zeros(
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=latents.device,
                dtype=latents.dtype,
            ),
        )

        captured = torch.randn(1, 2, 4, 8)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = captured
            return (torch.randn(1, 2, 2, 2, 2),)

        model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(image_embed_dim=6, text_embed_2_dim=4))

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "encoder_attention_mask": torch.ones(1, 3),
            "timesteps": tokenwise_timesteps,
            "crepa_capture_block_index": 7,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_states"].shape, (1, 5, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
