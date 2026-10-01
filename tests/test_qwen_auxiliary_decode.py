import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.models.common import ImageModelFoundation
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.training.crepa import CrepaRegularizer


class QwenAuxiliaryDecodeTests(unittest.TestCase):
    def test_21_vae_decode_supplies_rgb_to_feature_encoder(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.1")
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.vae = AutoencoderKLQwenImage21(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1, 1],
            num_res_blocks=1,
            latents_mean=[0.1] * 4,
            latents_std=[0.5] * 4,
            in_channels=4,
            out_channels=4,
            scale_factor_spatial=16,
        ).eval()
        latents = torch.randn(1, 4, 2, 2)
        with torch.no_grad():
            raw = model.vae.decode((latents * 0.5 + 0.1).unsqueeze(2)).sample
        regularizer = CrepaRegularizer.__new__(CrepaRegularizer)
        regularizer.model_foundation = model
        regularizer.use_tae = False
        pixels = regularizer._decode_latents_unified(latents, model.vae)
        self.assertEqual(pixels.shape, (1, 1, 3, 32, 32))
        torch.testing.assert_close(pixels, ((raw[:, :3].clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4))
        self.assertFalse(pixels.requires_grad)
        self.assertTrue(torch.isfinite(pixels).all())
        self.assertEqual(raw.shape[1], 4)

    def test_older_flavour_decode_is_unchanged(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.0")
        rgb = torch.rand(2, 1, 3, 8, 8)
        with patch.object(ImageModelFoundation, "decode_latents_to_pixels", return_value=rgb) as decode:
            latents = torch.zeros(1)
            self.assertIs(model.decode_latents_to_pixels(latents, use_tae=True), rgb)
            decode.assert_called_once_with(latents, use_tae=True)
