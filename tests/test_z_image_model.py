import types
import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.z_image.model import ZImage


class DummyAccelerator:
    device = torch.device("cpu")


class DummyTransformer:
    def __init__(self):
        self.received = None

    def __call__(self, latent_list, timestep, prompt_list):
        self.received = (latent_list, timestep, prompt_list)
        outputs = [torch.zeros_like(lat) for lat in latent_list]
        return outputs, {}


class ZImageModelTests(unittest.TestCase):
    def _build_model(self):
        model = ZImage.__new__(ZImage)
        model.accelerator = DummyAccelerator()
        model.config = types.SimpleNamespace(
            weight_dtype=torch.float32,
            model_family="z_image",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=1.0,
        )
        model.model = DummyTransformer()
        return model

    def test_convert_text_embed_for_pipeline_masks_tokens(self):
        model = self._build_model()
        prompt_embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)

        converted = model.convert_text_embed_for_pipeline(
            {"prompt_embeds": prompt_embeds, "attention_mask": attention_mask}
        )["prompt_embeds"]

        self.assertEqual(len(converted), 2)
        self.assertTrue(torch.equal(converted[0], prompt_embeds[0][:2]))
        self.assertTrue(torch.equal(converted[1], prompt_embeds[1][[0, 2, 3]]))

    def test_model_predict_shapes_and_timesteps(self):
        model = self._build_model()
        latents = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([100.0, 500.0])
        prompt_embeds = torch.randn(2, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

        output = model.model_predict(
            {
                "noisy_latents": latents,
                "timesteps": timesteps,
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
            }
        )

        noise_pred = output["model_prediction"]
        self.assertEqual(noise_pred.shape, latents.shape)

        latent_list, received_t, prompt_list = model.model.received
        self.assertEqual(len(latent_list), latents.shape[0])
        self.assertEqual(len(prompt_list[0]), attention_mask[0].sum().item())
        self.assertEqual(len(prompt_list[1]), attention_mask[1].sum().item())
        # Timesteps are normalized and flipped as in the pipeline: (1000 - t) / 1000
        self.assertTrue(torch.allclose(received_t, torch.tensor([0.9, 0.5])))


if __name__ == "__main__":
    unittest.main()
