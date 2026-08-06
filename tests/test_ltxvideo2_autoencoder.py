import unittest

import torch

from simpletuner.helpers.models.ltxvideo2.autoencoder import LTX2VideoDecoder3d, LTX2VideoUpBlock3d, LTX2VideoUpsampler3d


class TestLTX2VideoAutoencoder(unittest.TestCase):
    def test_up_block_uses_ltx2_upsampler_for_temporal_scaling(self):
        up_block = LTX2VideoUpBlock3d(
            in_channels=8,
            out_channels=8,
            num_layers=1,
            spatio_temporal_scale=True,
            upsample_type="temporal",
        )

        self.assertIsNotNone(up_block.upsamplers)
        self.assertIsInstance(up_block.upsamplers[0], LTX2VideoUpsampler3d)
        self.assertEqual(up_block.upsamplers[0].stride, (2, 1, 1))

    def test_up_block_rejects_unknown_upsample_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported upsample_type"):
            LTX2VideoUpBlock3d(
                in_channels=8,
                out_channels=8,
                num_layers=1,
                spatio_temporal_scale=True,
                upsample_type="invalid",
            )

    def test_up_block_conv_in_projects_to_upsampler_width(self):
        up_block = LTX2VideoUpBlock3d(
            in_channels=48,
            out_channels=32,
            num_layers=1,
            spatio_temporal_scale=True,
            upscale_factor=2,
        )

        self.assertIsNotNone(up_block.conv_in)

        sample = torch.randn(1, 48, 4, 8, 8)
        with torch.no_grad():
            output = up_block(sample, causal=False)

        self.assertEqual(output.shape, (1, 32, 7, 16, 16))

    def test_up_block_skips_conv_in_when_input_matches_upsampler_width(self):
        up_block = LTX2VideoUpBlock3d(
            in_channels=64,
            out_channels=32,
            num_layers=1,
            spatio_temporal_scale=True,
            upscale_factor=2,
        )

        self.assertIsNone(up_block.conv_in)

        sample = torch.randn(1, 64, 4, 8, 8)
        with torch.no_grad():
            output = up_block(sample, causal=False)

        self.assertEqual(output.shape, (1, 32, 7, 16, 16))

    def test_decoder_accepts_non_nominal_constant_width_blocks(self):
        decoder = LTX2VideoDecoder3d(
            in_channels=4,
            out_channels=3,
            block_out_channels=(16, 16, 16),
            layers_per_block=(1, 1, 1, 1),
            patch_size=1,
            patch_size_t=1,
            inject_noise=(False, False, False, False),
        )

        latents = torch.randn(1, 4, 3, 4, 4)
        with torch.no_grad():
            output = decoder(latents, causal=False)

        self.assertEqual(output.shape, (1, 3, 17, 32, 32))


if __name__ == "__main__":
    unittest.main()
