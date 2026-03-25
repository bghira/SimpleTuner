import unittest

from simpletuner.helpers.models.ltxvideo2.vocoder import AntiAliasAct1d, LTX2Vocoder, SnakeBeta


class TestLTX2Vocoder(unittest.TestCase):
    def test_antialias_act_snake_uses_non_beta_variant(self):
        act = AntiAliasAct1d("snake", channels=4)

        self.assertIsInstance(act.act, SnakeBeta)
        self.assertFalse(act.act.use_beta)

    def test_vocoder_snake_output_activation_respects_antialias_flag(self):
        plain_vocoder = LTX2Vocoder(
            in_channels=4,
            hidden_channels=8,
            out_channels=2,
            upsample_kernel_sizes=[4],
            upsample_factors=[2],
            resnet_kernel_sizes=[3],
            resnet_dilations=[[1]],
            act_fn="snake",
            antialias=False,
            final_act_fn=None,
        )
        antialias_vocoder = LTX2Vocoder(
            in_channels=4,
            hidden_channels=8,
            out_channels=2,
            upsample_kernel_sizes=[4],
            upsample_factors=[2],
            resnet_kernel_sizes=[3],
            resnet_dilations=[[1]],
            act_fn="snakebeta",
            antialias=True,
            final_act_fn=None,
        )

        self.assertIsInstance(plain_vocoder.act_out, SnakeBeta)
        self.assertFalse(plain_vocoder.act_out.use_beta)
        self.assertIsInstance(antialias_vocoder.act_out, AntiAliasAct1d)
        self.assertIsInstance(antialias_vocoder.act_out.act, SnakeBeta)
        self.assertTrue(antialias_vocoder.act_out.act.use_beta)


if __name__ == "__main__":
    unittest.main()
