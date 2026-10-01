import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21


class QwenImage21DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_vae(self, patch_size=None):
        factor = patch_size or 1
        return AutoencoderKLQwenImage21(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1, 1],
            num_res_blocks=1,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
            in_channels=4 * factor**2,
            out_channels=4 * factor**2,
            patch_size=patch_size,
            scale_factor_spatial=16 * factor,
        ).eval()

    def test_image_decode_matches_temporally_cached_reference(self):
        torch.manual_seed(42)
        for patch_size in (None, 2):
            vae = self.make_vae(patch_size)
            original_forward = vae.decoder.forward

            def cached_forward(x, **kwargs):
                kwargs["feat_cache"] = [None] * vae._cached_conv_counts["decoder"]
                kwargs["feat_idx"] = [0]
                return original_forward(x, **kwargs)

            for batch_size, slicing in ((1, False), (2, False), (2, True)):
                vae.use_slicing = slicing
                z = torch.randn(batch_size, 4, 1, 5, 7)
                for tiled in (False, True):
                    with self.subTest(patch_size=patch_size, batch_size=batch_size, slicing=slicing, tiled=tiled):
                        ratio = vae.spatial_compression_ratio
                        vae.enable_tiling(
                            tile_sample_min_height=3 * ratio,
                            tile_sample_min_width=3 * ratio,
                            tile_sample_stride_height=2 * ratio,
                            tile_sample_stride_width=2 * ratio,
                        )
                        vae.use_tiling = tiled
                        with torch.no_grad(), patch.object(vae.decoder, "forward", side_effect=cached_forward):
                            expected = vae.decode(z).sample
                        with torch.no_grad(), patch.object(vae.decoder, "forward", wraps=original_forward) as decoder:
                            actual = vae.decode(z, return_dict=False)[0]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(actual.shape, (batch_size, 4, 1, 5 * ratio, 7 * ratio))
                        for call in decoder.call_args_list:
                            self.assertIsNone(call.kwargs.get("feat_cache"))

    def test_image_decode_rejects_multiple_frames(self):
        vae = self.make_vae()
        z = torch.zeros(1, 4, 2, 5, 7)
        for decode in (vae.decode, vae.tiled_decode):
            with self.subTest(decode=decode.__name__):
                with self.assertRaisesRegex(ValueError, "exactly one frame"):
                    decode(z)
