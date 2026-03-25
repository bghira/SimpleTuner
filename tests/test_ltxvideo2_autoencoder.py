import unittest

from simpletuner.helpers.models.ltxvideo2.autoencoder import LTX2VideoUpBlock3d, LTX2VideoUpsampler3d


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


if __name__ == "__main__":
    unittest.main()
