import types
import unittest

import torch

from simpletuner.helpers.models.z_image_omni.model import ZImageOmni
from simpletuner.helpers.models.z_image_omni.transformer import ZImageOmniTransformer2DModel


class TestZImageOmniModel(unittest.TestCase):
    def test_model_predict_accepts_tensor_siglip_embeds(self):
        model = object.__new__(ZImageOmni)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(weight_dtype=torch.float32)

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


if __name__ == "__main__":
    unittest.main()
