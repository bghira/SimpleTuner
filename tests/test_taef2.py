"""Published TAEF2 checkpoint key conversion and completeness."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from simpletuner.helpers.models.tae.taef2 import TAEF2


class TAEF2CheckpointTests(unittest.TestCase):
    def published_state(self, model):
        state = {}
        for key, value in model.state_dict().items():
            _, _, index, *suffix = key.split(".")
            state[".".join(("decoder", "layers", str(int(index) - 1), *suffix))] = value
        state["encoder.layers.0.weight"] = torch.ones(1)
        return state

    def test_published_checkpoint_restores_every_decoder_tensor(self):
        torch.manual_seed(91)
        expected = TAEF2()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taef2.safetensors"
            save_file(self.published_state(expected), path)
            actual = TAEF2(decoder_path=str(path))
        self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)

    def test_incomplete_checkpoint_raises_instead_of_using_random_weights(self):
        state = self.published_state(TAEF2())
        del state["decoder.layers.0.weight"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taef2.safetensors"
            save_file(state, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                TAEF2(decoder_path=str(path))
