import unittest

import torch

from simpletuner.helpers.models.hunyuanvideo_vae import AutoencoderKLHunyuanVideoOptimized, HunyuanVideoCausalConv3d


class TestKandinsky5VaePatchConv(unittest.TestCase):
    def test_temporal_chunking_toggle(self):
        """Test that temporal chunking can be enabled/disabled on the VAE."""
        vae = AutoencoderKLHunyuanVideoOptimized(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            down_block_types=(
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            up_block_types=(
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
        )

        # Initially disabled
        self.assertFalse(HunyuanVideoCausalConv3d._temporal_chunking_enabled)

        # Enable via VAE method
        vae.enable_temporal_chunking()
        self.assertTrue(HunyuanVideoCausalConv3d._temporal_chunking_enabled)

        # Disable via VAE method
        vae.disable_temporal_chunking()
        self.assertFalse(HunyuanVideoCausalConv3d._temporal_chunking_enabled)

    def test_conv_output_matches_with_chunking(self):
        """Test that temporal chunking produces same output as non-chunked."""
        conv = HunyuanVideoCausalConv3d(4, 4, kernel_size=3, bias=False)
        sample = torch.randn(1, 4, 8, 4, 4)

        # Get output without chunking
        HunyuanVideoCausalConv3d._temporal_chunking_enabled = False
        with torch.no_grad():
            out_normal = conv(sample.clone())

        # Get output with chunking enabled (small memory limit to force chunking)
        HunyuanVideoCausalConv3d._temporal_chunking_enabled = True
        with torch.no_grad():
            out_chunked = conv(sample.clone())

        # Reset
        HunyuanVideoCausalConv3d._temporal_chunking_enabled = False

        # Outputs should match (or be very close due to floating point)
        self.assertTrue(
            torch.allclose(out_normal, out_chunked, atol=1e-5),
            f"Chunked output differs from normal. Max diff: {(out_normal - out_chunked).abs().max()}",
        )


if __name__ == "__main__":
    unittest.main()
