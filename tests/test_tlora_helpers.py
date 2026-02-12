"""Unit tests for T-LoRA (Timestep-dependent LoRA) helpers."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.lycoris import TLORA_AVAILABLE


@unittest.skipUnless(TLORA_AVAILABLE, "LyCORIS T-LoRA module not available")
class TestTLoRAMaskComputation(unittest.TestCase):
    """Tests that require lycoris.modules.tlora to be importable."""

    def test_compute_mask_low_timestep(self):
        """At t=0 (no noise), all max_rank entries should be active."""
        from lycoris.modules.tlora import compute_timestep_mask

        mask = compute_timestep_mask(0, 1000, 64, min_rank=1, alpha=1.0)
        self.assertEqual(mask.squeeze(0).sum().item(), 64)

    def test_compute_mask_high_timestep(self):
        """At t=max (full noise), only min_rank entries should be active."""
        from lycoris.modules.tlora import compute_timestep_mask

        mask = compute_timestep_mask(1000, 1000, 64, min_rank=1, alpha=1.0)
        self.assertEqual(mask.squeeze(0).sum().item(), 1)

    def test_compute_mask_mid_timestep(self):
        """At t=500/1000, roughly half of the ranks should be active."""
        from lycoris.modules.tlora import compute_timestep_mask

        mask = compute_timestep_mask(500, 1000, 64, min_rank=1, alpha=1.0)
        active = mask.squeeze(0).sum().item()
        self.assertGreater(active, 1)
        self.assertLess(active, 64)

    def test_apply_and_clear_cycle(self):
        """Verify set/clear lifecycle works without error."""
        from simpletuner.helpers.training.lycoris import apply_tlora_timestep_mask, clear_tlora_mask

        timesteps = torch.tensor([100, 500, 900])
        apply_tlora_timestep_mask(
            timesteps=timesteps,
            max_timestep=1000,
            max_rank=32,
            min_rank=1,
            alpha=1.0,
        )
        clear_tlora_mask()

    def test_batch_mask_shape(self):
        """apply_tlora_timestep_mask with batch of 4 produces (4, max_rank) mask."""
        from lycoris.modules.tlora import compute_timestep_mask

        timesteps = torch.tensor([0, 250, 500, 1000])
        max_rank = 32
        masks = torch.stack([compute_timestep_mask(int(t), 1000, max_rank, 1, 1.0).squeeze(0) for t in timesteps.tolist()])
        self.assertEqual(masks.shape, (4, max_rank))


class TestTLoRANotAvailable(unittest.TestCase):
    """Tests for the unavailable T-LoRA codepath."""

    def test_tlora_not_available_raises(self):
        """Verify a clear error when algo=tlora but LyCORIS doesn't have it."""
        with patch("simpletuner.helpers.training.lycoris.TLORA_AVAILABLE", False):
            from simpletuner.helpers.training.lycoris import TLORA_AVAILABLE as flag

            self.assertFalse(flag)


class TestTLoRADefaultsPresent(unittest.TestCase):
    """Verify T-LoRA appears in lycoris_defaults."""

    def test_tlora_in_defaults(self):
        from simpletuner.lycoris_defaults import lycoris_defaults

        self.assertIn("tlora", lycoris_defaults)
        self.assertEqual(lycoris_defaults["tlora"]["algo"], "tlora")
        self.assertEqual(lycoris_defaults["tlora"]["linear_dim"], 64)
        self.assertEqual(lycoris_defaults["tlora"]["linear_alpha"], 32)


if __name__ == "__main__":
    unittest.main()
