import math
import types
import unittest

import torch
import torch.nn as nn

from simpletuner.helpers.training.crepa import CrepaRegularizer


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")


class _DummyVAE(nn.Module):
    pass


class CrepaRegularizerTests(unittest.TestCase):
    def setUp(self):
        self.accelerator = _DummyAccelerator()

    def _make_regularizer(self, adjacent_distance: int) -> CrepaRegularizer:
        config = types.SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=adjacent_distance,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
        )
        regularizer = CrepaRegularizer(config, self.accelerator, hidden_size=4)

        def fake_load_encoder():
            regularizer.encoder = nn.Identity()
            regularizer.encoder_dim = 4

        regularizer._load_encoder = fake_load_encoder
        model = nn.Module()
        regularizer.attach_to_model(model)
        regularizer.projector = nn.Identity()
        model.crepa_projector = regularizer.projector
        regularizer._decode_latents = lambda latents, vae: torch.zeros(1, 3, 3, 2, 2)
        regularizer._encode_frames = lambda video: torch.ones(1, 3, 2, 4)
        return regularizer

    def test_alignment_with_neighbors_matches_expected_weighting(self):
        regularizer = self._make_regularizer(adjacent_distance=1)
        hidden_states = torch.ones(1, 3, 2, 4)
        latents = torch.zeros(1, 4, 3, 2, 2)

        loss, logs = regularizer.compute_loss(hidden_states, latents, _DummyVAE())

        self.assertIsNotNone(loss)
        self.assertIsNotNone(logs)
        weight = math.exp(-1.0)
        expected = -((3.0 + 4.0 * weight) / 3.0) * 0.5
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertIn("crepa_loss", logs)
        self.assertIn("crepa_similarity", logs)

    def test_alignment_without_neighbors_respects_distance_zero(self):
        regularizer = self._make_regularizer(adjacent_distance=0)
        hidden_states = torch.ones(1, 3, 2, 4)
        latents = torch.zeros(1, 4, 3, 2, 2)

        loss, logs = regularizer.compute_loss(hidden_states, latents, _DummyVAE())

        self.assertIsNotNone(loss)
        self.assertIsNotNone(logs)
        expected = -1.0 * 0.5
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertIn("crepa_loss", logs)
        self.assertIn("crepa_similarity", logs)


if __name__ == "__main__":
    unittest.main()
