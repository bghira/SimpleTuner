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


if __name__ == "__main__":
    unittest.main()
