import math
import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.crepa import CrepaMode, CrepaRegularizer, CrepaScheduler


class _DummyVAE(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        # Parameter establishes the VAE's dtype/device.
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=None)

    def decode(self, latents):
        # Ensure inputs arrive in the same dtype as the VAE parameters.
        assert latents.dtype == self._dummy.dtype
        # Return a minimal video tensor: (B, C, T, H, W)
        sample = torch.ones(latents.shape[0], 3, latents.shape[2], 2, 2, device=latents.device, dtype=latents.dtype)
        return SimpleNamespace(sample=sample)


class _DummyEncoder(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.register_parameter("_w", torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False))

    def forward(self, x):
        # Ensure incoming dtype matches encoder weights.
        assert x.dtype == self._w.dtype
        # Return patch tokens shaped like (BT, tokens, dim)
        bt = x.shape[0]
        tokens = torch.ones(bt, 2, 4, device=x.device, dtype=x.dtype)
        return {"x_norm_patchtokens": tokens}


class CrepaDecodeTests(unittest.TestCase):
    def test_decode_preserves_vae_dtype(self):
        config = SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=1,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_model=None,
            crepa_encoder_image_size=8,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
            crepa_cumulative_neighbors=False,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=8)

        vae = _DummyVAE(dtype=torch.bfloat16)
        latents = torch.randn(1, 4, 2, 2, 2, dtype=torch.float32)

        decoded = reg._decode_latents_legacy(latents, vae)
        self.assertEqual(decoded.dtype, vae._dummy.dtype)
        self.assertEqual(decoded.shape, (1, 2, 3, 2, 2))

    def test_encoder_inputs_cast_to_encoder_dtype(self):
        config = SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=1,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_model=None,
            crepa_encoder_image_size=8,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
            crepa_cumulative_neighbors=False,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=8)
        reg.encoder = _DummyEncoder(dtype=torch.float32)
        reg.encoder_dim = 4
        reg.projector = torch.nn.Identity()

        video = torch.randn(1, 2, 3, 4, 4, dtype=torch.bfloat16)
        tokens = reg._encode_frames(video)
        self.assertEqual(tokens.dtype, torch.float32)
        self.assertEqual(tokens.shape, (1, 2, 2, 4))

    def test_hidden_projection_casts_to_projector_dtype(self):
        config = SimpleNamespace(
            crepa_enabled=True,
            crepa_block_index=0,
            crepa_adjacent_distance=1,
            crepa_adjacent_tau=1.0,
            crepa_lambda=0.5,
            crepa_model=None,
            crepa_encoder_image_size=8,
            crepa_normalize_by_frames=True,
            crepa_spatial_align=True,
            crepa_cumulative_neighbors=False,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=4)
        reg.encoder = _DummyEncoder(dtype=torch.float32)
        reg.encoder_dim = 4
        reg.projector = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        reg.projector = reg.projector.to(dtype=torch.float32)

        hidden = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
        projected = reg._project_hidden_states(hidden)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(projected.shape, (1, 2, 3, 4))


class CrepaSchedulerTests(unittest.TestCase):
    def _make_config(self, **kwargs):
        defaults = {
            "crepa_scheduler": "constant",
            "crepa_lambda": 0.5,
            "crepa_warmup_steps": 0,
            "crepa_decay_steps": 0,
            "crepa_lambda_end": 0.0,
            "crepa_cutoff_step": 0,
            "crepa_similarity_threshold": None,
            "crepa_similarity_ema_decay": 0.99,
            "crepa_threshold_mode": "permanent",
            "crepa_power": 1.0,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_constant_scheduler_returns_base_weight(self):
        config = self._make_config(crepa_scheduler="constant", crepa_lambda=0.5)
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(500), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(1000), 0.5)

    def test_warmup_ramps_from_zero(self):
        config = self._make_config(
            crepa_scheduler="constant",
            crepa_lambda=1.0,
            crepa_warmup_steps=100,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 0.0)
        self.assertAlmostEqual(scheduler.get_weight(50), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(100), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(200), 1.0)

    def test_linear_decay(self):
        config = self._make_config(
            crepa_scheduler="linear",
            crepa_lambda=1.0,
            crepa_lambda_end=0.0,
            crepa_warmup_steps=0,
            crepa_decay_steps=100,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(50), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0)

    def test_cosine_decay(self):
        config = self._make_config(
            crepa_scheduler="cosine",
            crepa_lambda=1.0,
            crepa_lambda_end=0.0,
            crepa_warmup_steps=0,
            crepa_decay_steps=100,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 1.0)
        expected_midpoint = 0.5  # Cosine decay at 50% should give 0.5
        self.assertAlmostEqual(scheduler.get_weight(50), expected_midpoint, places=2)
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0, places=5)

    def test_polynomial_decay_with_power(self):
        config = self._make_config(
            crepa_scheduler="polynomial",
            crepa_lambda=1.0,
            crepa_lambda_end=0.0,
            crepa_warmup_steps=0,
            crepa_decay_steps=100,
            crepa_power=2.0,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 1.0)
        # At 50%: (1 - 0.5)^2 = 0.25
        expected = (1.0 - 0.0) * ((1 - 0.5) ** 2.0) + 0.0
        self.assertAlmostEqual(scheduler.get_weight(50), expected, places=5)
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0, places=5)

    def test_step_cutoff(self):
        config = self._make_config(
            crepa_scheduler="constant",
            crepa_lambda=1.0,
            crepa_cutoff_step=50,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        self.assertAlmostEqual(scheduler.get_weight(0), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(49), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(50), 0.0)
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0)

    def test_similarity_threshold_permanent_cutoff(self):
        config = self._make_config(
            crepa_scheduler="constant",
            crepa_lambda=1.0,
            crepa_similarity_threshold=0.9,
            crepa_similarity_ema_decay=0.0,  # No smoothing for predictable tests
            crepa_threshold_mode="permanent",
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        # Below threshold - should return weight
        self.assertAlmostEqual(scheduler.get_weight(0, similarity=0.5), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(1, similarity=0.8), 1.0)

        # At threshold - should trigger cutoff
        self.assertAlmostEqual(scheduler.get_weight(2, similarity=0.95), 0.0)

        # Permanent: even if similarity drops, stays cut off
        self.assertAlmostEqual(scheduler.get_weight(3, similarity=0.5), 0.0)
        self.assertTrue(scheduler.is_cutoff())

    def test_similarity_threshold_recoverable_cutoff(self):
        config = self._make_config(
            crepa_scheduler="constant",
            crepa_lambda=1.0,
            crepa_similarity_threshold=0.9,
            crepa_similarity_ema_decay=0.0,  # No smoothing for predictable tests
            crepa_threshold_mode="recoverable",
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        # Below threshold
        self.assertAlmostEqual(scheduler.get_weight(0, similarity=0.5), 1.0)

        # At threshold - should return 0
        self.assertAlmostEqual(scheduler.get_weight(1, similarity=0.95), 0.0)

        # Recoverable: if similarity drops, CREPA re-enables
        self.assertAlmostEqual(scheduler.get_weight(2, similarity=0.5), 1.0)

    def test_similarity_ema_tracking(self):
        config = self._make_config(
            crepa_scheduler="constant",
            crepa_lambda=1.0,
            crepa_similarity_threshold=0.95,
            crepa_similarity_ema_decay=0.5,  # Fast decay for testing
            crepa_threshold_mode="permanent",
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        # First call initializes EMA
        scheduler.get_weight(0, similarity=0.5)
        self.assertAlmostEqual(scheduler.get_similarity_ema(), 0.5)

        # Second call updates EMA: 0.5 * 0.5 + 0.5 * 0.9 = 0.7
        scheduler.get_weight(1, similarity=0.9)
        self.assertAlmostEqual(scheduler.get_similarity_ema(), 0.7)

    def test_warmup_then_decay(self):
        config = self._make_config(
            crepa_scheduler="linear",
            crepa_lambda=1.0,
            crepa_lambda_end=0.0,
            crepa_warmup_steps=100,
            crepa_decay_steps=200,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        # Warmup phase
        self.assertAlmostEqual(scheduler.get_weight(0), 0.0)
        self.assertAlmostEqual(scheduler.get_weight(50), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(100), 1.0)

        # Decay phase: starts at step 100, ends at step 200
        # At step 150: (150-100)/(200-100) = 0.5 progress
        self.assertAlmostEqual(scheduler.get_weight(150), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(200), 0.0)

    def test_combined_warmup_decay_cutoff(self):
        config = self._make_config(
            crepa_scheduler="linear",
            crepa_lambda=1.0,
            crepa_lambda_end=0.2,
            crepa_warmup_steps=50,
            crepa_decay_steps=150,
            crepa_cutoff_step=100,
        )
        scheduler = CrepaScheduler(config, max_train_steps=1000)

        # Warmup
        self.assertAlmostEqual(scheduler.get_weight(25), 0.5)

        # After warmup, before cutoff
        weight_at_75 = scheduler.get_weight(75)
        self.assertGreater(weight_at_75, 0.2)
        self.assertLess(weight_at_75, 1.0)

        # After cutoff
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0)

    def test_decay_steps_zero_uses_max_train_steps(self):
        config = self._make_config(
            crepa_scheduler="linear",
            crepa_lambda=1.0,
            crepa_lambda_end=0.0,
            crepa_warmup_steps=0,
            crepa_decay_steps=0,  # Should use max_train_steps
        )
        scheduler = CrepaScheduler(config, max_train_steps=100)

        self.assertAlmostEqual(scheduler.get_weight(0), 1.0)
        self.assertAlmostEqual(scheduler.get_weight(50), 0.5)
        self.assertAlmostEqual(scheduler.get_weight(100), 0.0)


class CrepaModeTests(unittest.TestCase):
    """Tests for CrepaMode-based shape interpretation."""

    def _make_config(self, **kwargs):
        defaults = {
            "crepa_enabled": True,
            "crepa_block_index": 0,
            "crepa_adjacent_distance": 1,
            "crepa_adjacent_tau": 1.0,
            "crepa_lambda": 0.5,
            "crepa_model": None,
            "crepa_encoder_image_size": 8,
            "crepa_normalize_by_frames": True,
            "crepa_spatial_align": True,
            "crepa_cumulative_neighbors": False,
            "crepa_use_backbone_features": False,
            "crepa_use_tae": False,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_default_mode_is_video_for_backward_compat(self):
        """Without model_foundation, mode should default to VIDEO."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        reg = CrepaRegularizer(config, accelerator, hidden_size=8, model_foundation=None)

        self.assertEqual(reg.mode, CrepaMode.VIDEO)

    def test_mode_from_model_foundation_image(self):
        """Model foundation with crepa_mode=IMAGE should set IMAGE mode."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        model_foundation = SimpleNamespace(crepa_mode=CrepaMode.IMAGE)
        reg = CrepaRegularizer(config, accelerator, hidden_size=8, model_foundation=model_foundation)

        self.assertEqual(reg.mode, CrepaMode.IMAGE)

    def test_mode_from_model_foundation_video(self):
        """Model foundation with crepa_mode=VIDEO should set VIDEO mode."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        model_foundation = SimpleNamespace(crepa_mode=CrepaMode.VIDEO)
        reg = CrepaRegularizer(config, accelerator, hidden_size=8, model_foundation=model_foundation)

        self.assertEqual(reg.mode, CrepaMode.VIDEO)

    def _make_projector(self, hidden_size=4):
        """Create a minimal projector with parameters for dtype detection."""
        return torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, hidden_size))

    def test_video_mode_hidden_state_reshape_3d(self):
        """VIDEO mode: (B, T, D) -> (B, T, 1, D)."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        model_foundation = SimpleNamespace(crepa_mode=CrepaMode.VIDEO)
        reg = CrepaRegularizer(config, accelerator, hidden_size=4, model_foundation=model_foundation)
        reg.projector = self._make_projector(4)

        hidden = torch.randn(2, 5, 4)  # (B=2, T=5, D=4)
        projected = reg._project_hidden_states(hidden)

        # Should become (B=2, T=5, P=1, D=4)
        self.assertEqual(projected.shape, (2, 5, 1, 4))

    def test_image_mode_hidden_state_reshape_3d(self):
        """IMAGE mode: (B, S, D) -> (B, 1, S, D)."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        model_foundation = SimpleNamespace(crepa_mode=CrepaMode.IMAGE)
        reg = CrepaRegularizer(config, accelerator, hidden_size=4, model_foundation=model_foundation)
        reg.projector = self._make_projector(4)

        hidden = torch.randn(2, 64, 4)  # (B=2, S=64 spatial tokens, D=4)
        projected = reg._project_hidden_states(hidden)

        # Should become (B=2, T=1, P=64, D=4)
        self.assertEqual(projected.shape, (2, 1, 64, 4))

    def test_4d_input_unchanged_regardless_of_mode(self):
        """4D inputs should pass through unchanged in both modes."""
        config = self._make_config()
        accelerator = SimpleNamespace(device=torch.device("cpu"))

        for mode in [CrepaMode.IMAGE, CrepaMode.VIDEO]:
            model_foundation = SimpleNamespace(crepa_mode=mode)
            reg = CrepaRegularizer(config, accelerator, hidden_size=4, model_foundation=model_foundation)
            reg.projector = self._make_projector(4)

            hidden = torch.randn(2, 5, 16, 4)  # Already (B, T, P, D)
            projected = reg._project_hidden_states(hidden)

            self.assertEqual(projected.shape, (2, 5, 16, 4))


if __name__ == "__main__":
    unittest.main()
