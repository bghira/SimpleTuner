import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.crepa import CrepaRegularizer


class _DummyVAE(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        # Parameter establishes the VAE's dtype/device.
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=None)

    def decode(self, latents):
        # Ensure inputs arrive in the same dtype as the VAE parameters.
        assert latents.dtype == self._dummy.dtype
        # Return a minimal video tensor: (B, C, T, H, W)
        sample = torch.ones(latents.shape[0], 3, latents.shape[2], 2, 2, device=latents.device, dtype=latents.dtype)
        return SimpleNamespace(sample=sample)


class _DummyEncoder(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.register_parameter("_w", torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False))

    def forward(self, x):
        # Ensure incoming dtype matches encoder weights.
        assert x.dtype == self._w.dtype
        # Return patch tokens shaped like (BT, tokens, dim)
        bt = x.shape[0]
        tokens = torch.ones(bt, 2, 4, device=x.device, dtype=x.dtype)
        return {"x_norm_patchtokens": tokens}


class CrepaDecodeTests(unittest.TestCase):
    def test_decode_preserves_vae_dtype(self):
        config = SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=1,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_model=None,
            crepa_encoder_image_size=8,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
            crepa_cumulative_neighbors=False,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=8)

        vae = _DummyVAE(dtype=torch.bfloat16)
        latents = torch.randn(1, 4, 2, 2, 2, dtype=torch.float32)

        decoded = reg._decode_latents(latents, vae)
        self.assertEqual(decoded.dtype, vae._dummy.dtype)
        self.assertEqual(decoded.shape, (1, 2, 3, 2, 2))

    def test_encoder_inputs_cast_to_encoder_dtype(self):
        config = SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=1,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_model=None,
            crepa_encoder_image_size=8,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
            crepa_cumulative_neighbors=False,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=8)
        reg.encoder = _DummyEncoder(dtype=torch.float32)
        reg.encoder_dim = 4
        reg.projector = torch.nn.Identity()

        video = torch.randn(1, 2, 3, 4, 4, dtype=torch.bfloat16)
        tokens = reg._encode_frames(video)
        self.assertEqual(tokens.dtype, torch.float32)
        self.assertEqual(tokens.shape, (1, 2, 2, 4))


if __name__ == "__main__":
    unittest.main()
