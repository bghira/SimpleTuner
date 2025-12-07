import unittest

import torch

from simpletuner.helpers.models.hunyuanvideo import autoencoder as ae


class TestHunyuanVideoVaePatchConv(unittest.TestCase):
    def test_patch_causal_conv3d_splitting_matches_full_conv(self):
        """PatchCausalConv3d split path should produce the same result as a full forward."""
        original_limit = ae.MEMORY_LIMIT
        try:
            ae.MEMORY_LIMIT = 8  # force split logic on small tensors
            conv_split = ae.PatchCausalConv3d(1, 1, kernel_size=1, bias=False)
            conv_full = torch.nn.Conv3d(1, 1, kernel_size=1, bias=False)
            conv_full.weight.data.copy_(conv_split.weight.data)

            sample = torch.arange(1 * 1 * 4 * 2 * 2, dtype=torch.float32).view(1, 1, 4, 2, 2)
            out_split = conv_split(sample)
            out_full = conv_full(sample)

            self.assertTrue(torch.allclose(out_split, out_full))
        finally:
            ae.MEMORY_LIMIT = original_limit

    def test_causal_conv_uses_patch_impl_when_enabled(self):
        conv = ae.CausalConv3d(1, 1, kernel_size=3, enable_patch_conv=True)
        self.assertIsInstance(conv.conv, ae.PatchCausalConv3d)

    def test_autoencoder_enables_patch_conv_and_runs(self):
        vae = ae.AutoencoderKLConv3D(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8, 16),
            layers_per_block=1,
            ffactor_spatial=2,
            ffactor_temporal=1,
            sample_size=8,
            sample_tsize=2,
            enable_patch_conv=True,
        )

        self.assertIsInstance(vae.encoder.mid.attn_1.q, ae.PatchConv3d)
        self.assertIsInstance(vae.encoder.down[0].block[0].conv1.conv, ae.PatchCausalConv3d)

        with torch.no_grad():
            sample = torch.randn(1, 3, 2, 8, 8)
            posterior = vae.encode(sample).latent_dist
            decoded = vae.decode(posterior.mode()).sample

        self.assertEqual(decoded.shape, (1, 3, 2, 8, 8))

    def test_encoder_temporal_roll_matches_base_outputs(self):
        """Temporal rolling path should be numerically identical when weights match."""
        base = ae.Encoder(
            in_channels=3,
            z_channels=4,
            block_out_channels=(8, 16),
            num_res_blocks=1,
            ffactor_spatial=2,
            ffactor_temporal=2,
            downsample_match_channel=True,
            enable_patch_conv=False,
            temporal_roll=False,
        )
        rolled = ae.Encoder(
            in_channels=3,
            z_channels=4,
            block_out_channels=(8, 16),
            num_res_blocks=1,
            ffactor_spatial=2,
            ffactor_temporal=2,
            downsample_match_channel=True,
            enable_patch_conv=False,
            temporal_roll=True,
        )
        rolled.load_state_dict(base.state_dict(), strict=False)

        sample = torch.randn(1, 3, 5, 8, 8)
        with torch.no_grad():
            out_base = base(sample)
            out_roll = rolled(sample)

        self.assertEqual(out_base.shape, out_roll.shape)
        self.assertTrue(torch.allclose(out_base, out_roll, atol=0, rtol=0))

    def test_decoder_temporal_roll_matches_base_outputs(self):
        """Decoder streaming toggle should not change outputs when parameters align."""
        base = ae.Decoder(
            z_channels=4,
            out_channels=3,
            block_out_channels=(16, 8),
            num_res_blocks=1,
            ffactor_spatial=2,
            ffactor_temporal=2,
            upsample_match_channel=True,
            enable_patch_conv=False,
            temporal_roll=False,
        )
        rolled = ae.Decoder(
            z_channels=4,
            out_channels=3,
            block_out_channels=(16, 8),
            num_res_blocks=1,
            ffactor_spatial=2,
            ffactor_temporal=2,
            upsample_match_channel=True,
            enable_patch_conv=False,
            temporal_roll=True,
        )
        rolled.load_state_dict(base.state_dict(), strict=False)

        latent = torch.randn(1, 4, 3, 4, 4)
        with torch.no_grad():
            out_base = base(latent)
            out_roll = rolled(latent)

        self.assertEqual(out_base.shape, out_roll.shape)
        self.assertTrue(torch.allclose(out_base, out_roll, atol=0, rtol=0))


if __name__ == "__main__":
    unittest.main()
