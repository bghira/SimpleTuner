import math
import os
import struct
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.local import LocalDataBackend


def _write_test_wav(path: str, sample_rate: int = 8000, num_samples: int = 1600) -> None:
    """Write a simple sine wave to the provided path."""
    tone_frequency = 440.0
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = [int(32767 * math.sin(2 * math.pi * tone_frequency * (i / sample_rate))) for i in range(num_samples)]
        frame_bytes = b"".join(struct.pack("<h", frame) for frame in frames)
        wav_file.writeframes(frame_bytes)


class TestLocalDataBackendAudio(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.backend = LocalDataBackend(accelerator=None, id="test_local")
        # Set dataset type to audio to ensure correct file extensions
        self.backend.dataset_type = DatasetType.AUDIO

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_audio_single(self):
        wav_path = os.path.join(self.temp_dir.name, "test.wav")
        _write_test_wav(wav_path, sample_rate=16000, num_samples=16000)

        # LocalDataBackend.read_image should handle audio files if extension is in audio_file_extensions
        # audio_file_extensions is imported in local.py.
        # We assume .wav is in it.

        result = self.backend.read_image(wav_path)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        waveform, sr = result
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(sr, 16000)
        self.assertEqual(waveform.shape[1], 16000)  # (Channels, Samples)

    def test_read_audio_batch(self):
        paths = []
        for i in range(3):
            p = os.path.join(self.temp_dir.name, f"test_{i}.wav")
            _write_test_wav(p, sample_rate=16000, num_samples=16000)
            paths.append(p)

        keys, results = self.backend.read_image_batch(paths)

        self.assertEqual(len(keys), 3)
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIsInstance(res, tuple)
            self.assertEqual(res[1], 16000)

    def test_list_files_audio(self):
        # Create some files
        _write_test_wav(os.path.join(self.temp_dir.name, "a.wav"))
        _write_test_wav(os.path.join(self.temp_dir.name, "b.mp3"))  # Assuming mp3 is supported extension
        _write_test_wav(os.path.join(self.temp_dir.name, "c.txt"))  # Should be ignored

        # default extensions for audio
        files = self.backend.list_files(file_extensions=None, instance_data_dir=self.temp_dir.name)

        # Flatten results: [(root, [], [files...])]
        all_files = []
        for _, _, fs in files:
            all_files.extend(fs)

        self.assertTrue(any("a.wav" in f for f in all_files))
        self.assertFalse(any("c.txt" in f for f in all_files))
        # mp3 should be there if in defaults
        # self.assertTrue(any("b.mp3" in f for f in all_files))


if __name__ == "__main__":
    unittest.main()
