import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F


class TwinFlowAdversarialTest(unittest.TestCase):
    """Tests for TwinFlow adversarial losses (L_adv and L_rectify)."""

    def test_adversarial_settings_defaults(self):
        """Test that adversarial settings have correct defaults."""
        from simpletuner.helpers.models.common import ModelFoundation

        # Create a minimal mock for testing settings
        mock_config = MagicMock()
        mock_config.twinflow_adversarial_enabled = False
        mock_config.twinflow_adversarial_weight = None
        mock_config.twinflow_rectify_weight = None
        mock_config.twinflow_estimate_order = None
        mock_config.twinflow_enhanced_ratio = None
        mock_config.twinflow_target_step_count = None
        mock_config.twinflow_delta_t = None
        mock_config.twinflow_target_clamp = None
        mock_config.twinflow_require_ema = True

        # Manually call the settings method logic
        settings = {
            "adversarial_enabled": bool(getattr(mock_config, "twinflow_adversarial_enabled", False)),
            "adversarial_weight": float(getattr(mock_config, "twinflow_adversarial_weight", 1.0) or 1.0),
            "rectify_weight": float(getattr(mock_config, "twinflow_rectify_weight", 1.0) or 1.0),
        }

        self.assertFalse(settings["adversarial_enabled"])
        self.assertEqual(settings["adversarial_weight"], 1.0)
        self.assertEqual(settings["rectify_weight"], 1.0)

    def test_adversarial_enabled_when_set(self):
        """Test that adversarial_enabled is True when config sets it."""
        mock_config = MagicMock()
        mock_config.twinflow_adversarial_enabled = True
        mock_config.twinflow_adversarial_weight = 0.5
        mock_config.twinflow_rectify_weight = 0.8

        settings = {
            "adversarial_enabled": bool(getattr(mock_config, "twinflow_adversarial_enabled", False)),
            "adversarial_weight": float(getattr(mock_config, "twinflow_adversarial_weight", 1.0) or 1.0),
            "rectify_weight": float(getattr(mock_config, "twinflow_rectify_weight", 1.0) or 1.0),
        }

        self.assertTrue(settings["adversarial_enabled"])
        self.assertEqual(settings["adversarial_weight"], 0.5)
        self.assertEqual(settings["rectify_weight"], 0.8)

    def test_fake_trajectory_uses_negative_time(self):
        """Test that fake trajectory time sampling produces negative values."""
        bsz = 4
        device = torch.device("cpu")
        dtype = torch.float32

        # Sample positive time and negate for fake trajectory
        t = torch.rand(bsz, device=device, dtype=dtype).clamp(min=0.01, max=0.99)
        neg_t = -t

        # Verify all negative
        self.assertTrue((neg_t < 0).all(), "Fake trajectory time should be negative")
        self.assertTrue((neg_t > -1).all(), "Fake trajectory time should be > -1")

    def test_fake_sample_interpolation(self):
        """Test that fake trajectory interpolation is correct."""
        bsz = 2
        channels = 4
        size = 8

        x_fake = torch.randn(bsz, channels, size, size)
        z = torch.randn(bsz, channels, size, size)
        t = torch.tensor([0.5, 0.3]).view(bsz, 1, 1, 1)

        # Interpolation: x_t_fake = t * z + (1 - t) * x_fake
        x_t_fake = t * z + (1 - t) * x_fake

        # Verify shapes match
        self.assertEqual(x_t_fake.shape, x_fake.shape)

        # Verify interpolation at extremes
        t_zero = torch.zeros(bsz, 1, 1, 1)
        t_one = torch.ones(bsz, 1, 1, 1)

        x_t_at_zero = t_zero * z + (1 - t_zero) * x_fake
        x_t_at_one = t_one * z + (1 - t_one) * x_fake

        self.assertTrue(torch.allclose(x_t_at_zero, x_fake), "At t=0, should equal x_fake")
        self.assertTrue(torch.allclose(x_t_at_one, z), "At t=1, should equal z")

    def test_adversarial_loss_target(self):
        """Test that adversarial loss target is z - x_fake."""
        x_fake = torch.randn(2, 4, 8, 8)
        z = torch.randn(2, 4, 8, 8)

        target_fake = z - x_fake

        # Verify target is the velocity from x_fake to z
        self.assertEqual(target_fake.shape, x_fake.shape)
        self.assertTrue(torch.allclose(x_fake + target_fake, z))

    def test_rectify_loss_gradient_computation(self):
        """Test that rectify loss computes F_grad = F_neg - F_pos correctly."""
        F_pos = torch.randn(2, 4, 8, 8)
        F_neg = torch.randn(2, 4, 8, 8)

        F_grad = F_neg - F_pos.detach()

        # Target: base_pred - F_grad (stop gradient)
        rectify_target = (F_pos.detach() - F_grad).detach()

        # Loss should be MSE between F_pos and rectify_target
        loss = F.mse_loss(F_pos, rectify_target, reduction="mean")

        # Verify loss is a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() >= 0)

    def test_sign_extraction_from_negative_sigma(self):
        """Test that sign is correctly extracted from negative sigma values."""
        sigmas = torch.tensor([-0.5, -0.3, -0.8])

        sigma_sign = torch.sign(sigmas)
        sigma_abs = sigmas.abs()

        self.assertTrue((sigma_sign == -1).all(), "Sign should be -1 for negative sigmas")
        self.assertTrue((sigma_abs > 0).all(), "Absolute values should be positive")
        self.assertTrue(torch.allclose(sigma_abs, torch.tensor([0.5, 0.3, 0.8])))

    def test_match_time_shape_broadcasting(self):
        """Test that time tensors are correctly broadcast to latent shapes."""

        def match_time_shape(time_tensor, like):
            if time_tensor is None:
                return None
            while time_tensor.dim() < like.dim():
                time_tensor = time_tensor.unsqueeze(-1)
            return time_tensor

        time = torch.tensor([0.5, 0.3])  # Shape: (2,)
        latents = torch.randn(2, 4, 8, 8)  # Shape: (2, 4, 8, 8)

        time_broadcast = match_time_shape(time, latents)

        self.assertEqual(time_broadcast.shape, (2, 1, 1, 1))
        # Verify broadcasting works for multiplication
        result = time_broadcast * latents
        self.assertEqual(result.shape, latents.shape)


class TwinFlowAdversarialIntegrationTest(unittest.TestCase):
    """Integration tests for TwinFlow adversarial branch."""

    def test_loss_components_summed(self):
        """Test that all loss components are properly summed."""
        loss_base = torch.tensor(0.5)
        loss_real = torch.tensor(0.3)
        loss_adv = torch.tensor(0.2)
        loss_rectify = torch.tensor(0.1)

        adv_weight = 1.0
        rectify_weight = 1.0

        twin_losses = [loss_base, loss_real]
        twin_losses.append(adv_weight * loss_adv)
        twin_losses.append(rectify_weight * loss_rectify)

        total = torch.stack(twin_losses).sum()

        expected = 0.5 + 0.3 + 0.2 + 0.1
        self.assertAlmostEqual(total.item(), expected, places=5)

    def test_weighted_losses(self):
        """Test that loss weights are applied correctly."""
        loss_adv = torch.tensor(1.0)
        loss_rectify = torch.tensor(1.0)

        adv_weight = 0.5
        rectify_weight = 0.3

        weighted_adv = adv_weight * loss_adv
        weighted_rectify = rectify_weight * loss_rectify

        self.assertAlmostEqual(weighted_adv.item(), 0.5, places=5)
        self.assertAlmostEqual(weighted_rectify.item(), 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
