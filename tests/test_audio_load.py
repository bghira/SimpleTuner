import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.audio.load import load_audio


class AudioLoadTests(unittest.TestCase):
    def test_filesystem_path_uses_ffmpeg_when_torchaudio_fails(self):
        expected = torch.ones(2, 32)
        with (
            patch("simpletuner.helpers.audio.load.torchaudio.load", side_effect=OSError("torchcodec unavailable")),
            patch(
                "simpletuner.helpers.audio.load._load_with_ffmpeg",
                return_value=(expected, 16_000),
            ) as ffmpeg_load,
        ):
            waveform, sample_rate = load_audio("/tmp/example.flac")

        ffmpeg_load.assert_called_once_with("/tmp/example.flac")
        self.assertIs(waveform, expected)
        self.assertEqual(sample_rate, 16_000)


if __name__ == "__main__":
    unittest.main()
