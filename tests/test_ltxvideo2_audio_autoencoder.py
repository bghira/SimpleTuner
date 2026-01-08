import unittest

import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from simpletuner.helpers.models.ltxvideo2.audio_autoencoder import AutoencoderKLLTX2Audio


class TestLTXVideo2AudioAutoencoder(unittest.TestCase):
    def test_encode_expands_latent_channels(self):
        model = AutoencoderKLLTX2Audio(
            base_channels=8,
            output_channels=1,
            ch_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=(),
            in_channels=1,
            resolution=4,
            latent_channels=4,
            double_z=False,
            norm_type="pixel",
            causality_axis="height",
            mel_bins=4,
        )
        spectrogram = torch.randn(1, 1, 4, 4)

        output = model.encode(spectrogram)

        self.assertIsInstance(output, AutoencoderKLOutput)
        self.assertEqual(output.latent_dist.mean.shape[1], 4)
        self.assertEqual(tuple(model.latents_mean.shape), (4,))
        self.assertEqual(tuple(model.latents_std.shape), (4,))


if __name__ == "__main__":
    unittest.main()
