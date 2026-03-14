import types
import unittest

import torch

from simpletuner.helpers.models.z_image_omni.model import ZImageOmni
from simpletuner.helpers.models.z_image_omni.transformer import ZImageOmniTransformer2DModel


class TestZImageOmniModel(unittest.TestCase):
    def test_model_supports_crepa_self_flow(self):
        model = object.__new__(ZImageOmni)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        model = object.__new__(ZImageOmni)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=1.0)
        model.model = types.SimpleNamespace(config=types.SimpleNamespace(all_patch_size=(2,)))
        model.unwrap_model = lambda wrapped=None, model_obj=None, model=None: (
            wrapped if wrapped is not None else (model if model is not None else model_obj)
        )
        model.sample_flow_sigmas = lambda batch, state: (
            torch.tensor([0.9], dtype=torch.float32),
            torch.tensor([900.0], dtype=torch.float32),
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 16, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.1], dtype=torch.float32),
            "timesteps": torch.tensor([100.0], dtype=torch.float32),
        }

        result = model._prepare_crepa_self_flow_batch(batch, state={})
        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertTrue(torch.equal(result["crepa_teacher_timesteps"], torch.tensor([100.0], dtype=torch.float32)))

    def test_model_predict_accepts_tensor_siglip_embeds(self):
        model = object.__new__(ZImageOmni)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(weight_dtype=torch.float32)
        model._new_hidden_state_buffer = lambda: {}
        model.unwrap_model = lambda wrapped=None, model_obj=None, model=None: (
            wrapped if wrapped is not None else (model if model is not None else model_obj)
        )

        def fake_transformer(latent_list, *_args, **_kwargs):
            return ([torch.zeros_like(sample) for sample in latent_list],)

        model.model = fake_transformer

        batch_size = 2
        channels, height, width = 4, 2, 2
        latents = torch.randn(batch_size, channels, height, width)
        encoder_hidden_states = torch.randn(batch_size, 3, 6)
        attention_mask = torch.ones(batch_size, 3, dtype=torch.long)
        timesteps = torch.tensor([100.0, 200.0])
        siglip_embeds = torch.randn(batch_size, 1, 1, 8)

        result = model.model_predict(
            {
                "noisy_latents": latents,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
                "timesteps": timesteps,
                "siglip_embeds": siglip_embeds,
            }
        )

        self.assertIn("model_prediction", result)
        self.assertEqual(result["model_prediction"].shape, (batch_size, channels, height, width))

    def test_model_predict_accepts_tokenwise_timesteps_and_capture_override(self):
        model = object.__new__(ZImageOmni)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(weight_dtype=torch.float32, twinflow_enabled=False)
        hidden_states_buffer = {"layer_7": torch.randn(1, 4, 8)}
        model._new_hidden_state_buffer = lambda: hidden_states_buffer
        model.unwrap_model = lambda wrapped=None, model_obj=None, model=None: (
            wrapped if wrapped is not None else (model if model is not None else model_obj)
        )
        model.crepa_regularizer = types.SimpleNamespace(enabled=True, block_index=3)

        def fake_transformer(latent_list, t, *_args, **_kwargs):
            fake_transformer.last_t = t
            return ([torch.zeros_like(sample) for sample in latent_list],)

        model.model = fake_transformer

        latents = torch.randn(1, 4, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 6)
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        timesteps = torch.tensor([[100.0, 900.0, 250.0, 750.0]])

        result = model.model_predict(
            {
                "noisy_latents": latents,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
                "timesteps": timesteps,
                "crepa_capture_block_index": 7,
            }
        )

        self.assertIs(result["crepa_hidden_states"], hidden_states_buffer["layer_7"])
        self.assertTrue(torch.allclose(fake_transformer.last_t, torch.tensor([[0.9, 0.1, 0.75, 0.25]], dtype=torch.float32)))


class TestZImageOmniTransformer(unittest.TestCase):
    def _build_small_transformer(self):
        return ZImageOmniTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=2,
            dim=12,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=3,
            n_kv_heads=3,
            norm_eps=1e-5,
            qk_norm=False,
            cap_feat_dim=4,
            siglip_feat_dim=3,
            rope_theta=10.0,
            axes_dims=[1, 1, 2],
            axes_lens=[4, 4, 4],
        )

    def test_forward_handles_mixed_siglip_presence(self):
        transformer = self._build_small_transformer().eval()

        batch_size = 2
        x = [torch.zeros(2, 2, 2, 2), torch.zeros(2, 2, 2, 2)]
        cap_feats = [[torch.zeros(2, 4)] for _ in range(batch_size)]
        cond_latents = [[] for _ in range(batch_size)]
        siglip_feats = [[torch.zeros(2, 2, 3)], [None]]
        timesteps = torch.tensor([0.5, 0.5])

        output = transformer.forward(
            x=x,
            t=timesteps,
            cap_feats=cap_feats,
            cond_latents=cond_latents,
            siglip_feats=siglip_feats,
        )

        self.assertEqual(len(output.sample), batch_size)
        for sample in output.sample:
            self.assertEqual(sample.shape, (2, 2, 2, 2))

    def test_forward_handles_all_none_siglip(self):
        transformer = self._build_small_transformer().eval()

        batch_size = 2
        x = [torch.zeros(2, 2, 2, 2), torch.zeros(2, 2, 2, 2)]
        cap_feats = [[torch.zeros(2, 4)] for _ in range(batch_size)]
        cond_latents = [[] for _ in range(batch_size)]
        siglip_feats = [None, None]
        timesteps = torch.tensor([0.25, 0.75])

        output = transformer.forward(
            x=x,
            t=timesteps,
            cap_feats=cap_feats,
            cond_latents=cond_latents,
            siglip_feats=siglip_feats,
        )

        self.assertEqual(len(output.sample), batch_size)
        for sample in output.sample:
            self.assertEqual(sample.shape, (2, 2, 2, 2))

    def test_forward_accepts_tokenwise_timesteps(self):
        transformer = self._build_small_transformer().eval()

        batch_size = 2
        x = [torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)]
        cap_feats = [[torch.zeros(2, 4)] for _ in range(batch_size)]
        cond_latents = [[] for _ in range(batch_size)]
        siglip_feats = [None, None]
        timesteps = torch.tensor([[0.5, 0.25, 0.75, 0.1], [0.5, 0.25, 0.75, 0.1]])

        output = transformer.forward(
            x=x,
            t=timesteps,
            cap_feats=cap_feats,
            cond_latents=cond_latents,
            siglip_feats=siglip_feats,
        )

        self.assertEqual(len(output.sample), batch_size)
        for sample in output.sample:
            self.assertEqual(sample.shape, (2, 1, 4, 4))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        transformer = self._build_small_transformer().eval()

        x = [torch.zeros(2, 1, 4, 4)]
        cap_feats = [[torch.zeros(2, 4)]]
        cond_latents = [[]]
        siglip_feats = [None]
        timesteps = torch.tensor([[0.5, 0.25, 0.75]])

        with self.assertRaisesRegex(ValueError, "expected tokenwise timesteps with sequence length 4"):
            transformer.forward(
                x=x,
                t=timesteps,
                cap_feats=cap_feats,
                cond_latents=cond_latents,
                siglip_feats=siglip_feats,
            )


if __name__ == "__main__":
    unittest.main()
