import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.ace_step.music_dcae.music_dcae_pipeline import MusicDCAE


class TestMusicDCAE(unittest.TestCase):
    def setUp(self):
        # Mock the AutoencoderDC and ADaMoSHiFiGANV1 so we don't load real weights
        self.patcher_ae = patch("simpletuner.helpers.models.ace_step.music_dcae.music_dcae_pipeline.AutoencoderDC")
        self.patcher_voc = patch("simpletuner.helpers.models.ace_step.music_dcae.music_dcae_pipeline.ADaMoSHiFiGANV1")
        self.MockAE = self.patcher_ae.start()
        self.MockVoc = self.patcher_voc.start()

        # Setup mock instances
        self.mock_ae_instance = MagicMock()
        self.MockAE.from_pretrained.return_value = self.mock_ae_instance

        self.mock_voc_instance = MagicMock()
        self.MockVoc.from_pretrained.return_value = self.mock_voc_instance

        self.model = MusicDCAE(dcae_checkpoint_path="dummy", vocoder_checkpoint_path="dummy")

    def tearDown(self):
        self.patcher_ae.stop()
        self.patcher_voc.stop()

    def test_encode_basic(self):
        # Mock encoder output
        # Input: N x 2 x T (resampled) -> Mel (N x C x H x W) -> Encoder -> Latent
        # Let's just verify shapes flow through

        # Mock vocoder.mel_transform
        # Input audio: [1, 2, 48000] (1 sec)
        # Mel transform usually outputs [1, 1, H, W]? No, VAE expects specific input
        # forward_mel calls self.vocoder.mel_transform(audios[i])

        def mock_mel_transform(audio):
            # audio: [2, T]
            # return [1, 80, T_mel] ?
            # forward_mel stacks them.
            return torch.randn(1, 80, 128)  # Dummy mel

        self.mock_voc_instance.mel_transform = MagicMock(side_effect=mock_mel_transform)

        # Mock dcae.encoder
        # Input: [N, 1, 80, 128] (normalized)
        # Output: [N, 8, H_latent, W_latent]
        self.mock_ae_instance.encoder.return_value = torch.randn(1, 8, 16, 16)

        audios = torch.randn(1, 2, 48000)

        latents, lengths = self.model.encode(audios, sr=48000)

        self.assertIsNotNone(latents)
        self.assertEqual(latents.shape[1], 8)  # Latent channels
        self.assertIsNotNone(lengths)

    def test_decode_basic(self):
        # Mock decoder output
        # Input: Latent
        # Output: Mel

        self.mock_ae_instance.decoder.return_value = torch.randn(1, 2, 80, 128)

        # Mock vocoder.decode
        # Input: Mel [1, 80, 128]
        # Output: Wav [1, 1, T]
        self.mock_voc_instance.decode.return_value = torch.randn(1, 1, 44100)

        latents = torch.randn(1, 8, 16, 16)

        sr, wavs = self.model.decode(latents, sr=44100)

        self.assertEqual(sr, 44100)
        self.assertEqual(len(wavs), 1)
        self.assertEqual(wavs[0].shape[0], 2)  # 2 channels


if __name__ == "__main__":
    unittest.main()
