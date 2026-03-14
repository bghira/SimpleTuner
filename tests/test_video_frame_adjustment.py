import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from simpletuner.helpers.image_manipulation.load import adjust_video_frames_for_model
from simpletuner.helpers.models.common import VideoModelFoundation
from simpletuner.helpers.models.longcat_video.model import LongCatVideo
from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2
from simpletuner.helpers.models.ltxvideo.model import LTXVideo
from simpletuner.helpers.models.sanavideo.model import SanaVideo
from simpletuner.helpers.models.wan.model import Wan
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer


class TestVideoFrameAdjustmentBase(unittest.TestCase):
    """Test base VideoModelFoundation frame adjustment"""

    def test_adjust_video_frames_default_no_adjustment(self):
        """Default implementation should return input unchanged"""
        self.assertEqual(VideoModelFoundation.adjust_video_frames(50), 50)
        self.assertEqual(VideoModelFoundation.adjust_video_frames(119), 119)
        self.assertEqual(VideoModelFoundation.adjust_video_frames(1), 1)

    def test_adjust_video_frames_returns_int(self):
        """Result should always be an integer"""
        result = VideoModelFoundation.adjust_video_frames(50)
        self.assertIsInstance(result, int)


class TestLTXVideo2FrameAdjustment(unittest.TestCase):
    """Test LTXVideo2 frame constraint (frames % 8 == 1)"""

    def test_valid_frame_count_unchanged(self):
        """Frame counts satisfying frames % 8 == 1 should be unchanged"""
        valid_counts = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
        for count in valid_counts:
            with self.subTest(frames=count):
                result = LTXVideo2.adjust_video_frames(count)
                self.assertEqual(result, count)
                self.assertEqual(result % 8, 1)

    def test_invalid_frame_count_rounds_down(self):
        """Invalid frame counts should round down to nearest valid value"""
        test_cases = [
            (119, 113),  # 119 -> 113
            (50, 49),  # 50 -> 49
            (100, 97),  # 100 -> 97
            (2, 1),  # 2 -> 1
            (8, 1),  # 8 -> 1
        ]
        for input_frames, expected_frames in test_cases:
            with self.subTest(input_frames=input_frames):
                result = LTXVideo2.adjust_video_frames(input_frames)
                self.assertEqual(result, expected_frames)
                self.assertEqual(result % 8, 1)

    def test_single_frame_valid(self):
        """Single frame (1) is valid for LTX-2"""
        result = LTXVideo2.adjust_video_frames(1)
        self.assertEqual(result, 1)

    def test_never_zero_frames(self):
        """Never return 0 frames, minimum is 1"""
        result = LTXVideo2.adjust_video_frames(0)
        self.assertGreaterEqual(result, 1)

    def test_adjustment_formula(self):
        """Verify the adjustment formula: ((frames - 1) // 8) * 8 + 1"""
        for frames in range(1, 150):
            adjusted = LTXVideo2.adjust_video_frames(frames)
            # Verify the formula is correct
            expected = ((frames - 1) // 8) * 8 + 1
            self.assertEqual(adjusted, expected)
            # Verify constraint is satisfied
            self.assertEqual(adjusted % 8, 1)
            # Verify adjustment only rounds down
            self.assertLessEqual(adjusted, frames)

    def test_crepa_self_flow_patch_size_supports_split_temporal_and_spatial_config(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.model = MagicMock(config=MagicMock(patch_size=2, patch_size_t=4))
        model.unwrap_model = lambda wrapped=None, model_obj=None: wrapped if wrapped is not None else model_obj

        self.assertEqual(model._crepa_self_flow_patch_size(), (4, 2, 2))

    def test_model_predict_preserves_tokenwise_timesteps_for_crepa_capture_override(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.config = MagicMock(weight_dtype=torch.float32, framerate=24, twinflow_enabled=False, tread_config=None)
        model._load_connectors = MagicMock()
        model.connectors = MagicMock(
            return_value=(
                torch.randn(1, 8, 16),
                torch.randn(1, 6, 16),
                torch.ones(1, 8),
            )
        )
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        model.model = MagicMock(config=MagicMock(patch_size=1, patch_size_t=1))
        model.model.return_value = (
            torch.randn(1, 8, 128),
            torch.randn(1, 6, 8),
            torch.randn(1, 2, 4, 16),
        )

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2, 2),
            "audio_noisy_latents": torch.randn(1, 8, 6, 8),
            "audio_latents": torch.randn(1, 8, 6, 8),
            "encoder_hidden_states": torch.randn(1, 12, 16),
            "encoder_attention_mask": torch.ones(1, 12),
            "timesteps": tokenwise_timesteps,
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_latents", return_value=torch.randn(1, 8, 128)):
            with patch(
                "simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_audio_latents",
                return_value=torch.randn(1, 6, 8),
            ):
                with patch(
                    "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_latents",
                    return_value=torch.randn(1, 128, 2, 2, 2),
                ):
                    with patch(
                        "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_audio_latents",
                        return_value=torch.randn(1, 8, 6, 8),
                    ):
                        result = model.model_predict(prepared_batch)

        self.assertIsNotNone(result["crepa_hidden_states"])
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertTrue(torch.equal(transformer_kwargs["audio_timestep"], torch.tensor([100.0])))
        self.assertEqual(transformer_kwargs["hidden_state_layer"], 7)
        self.assertTrue(transformer_kwargs["output_hidden_states"])

    def test_model_predict_prefers_explicit_audio_timesteps(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.config = MagicMock(weight_dtype=torch.float32, framerate=24, twinflow_enabled=False, tread_config=None)
        model._load_connectors = MagicMock()
        model.connectors = MagicMock(
            return_value=(
                torch.randn(1, 8, 16),
                torch.randn(1, 6, 16),
                torch.ones(1, 8),
            )
        )
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        model.model = MagicMock(config=MagicMock(patch_size=1, patch_size_t=1))
        model.model.return_value = (
            torch.randn(1, 8, 128),
            torch.randn(1, 6, 8),
            torch.randn(1, 2, 4, 16),
        )

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2, 2),
            "audio_noisy_latents": torch.randn(1, 8, 6, 8),
            "audio_latents": torch.randn(1, 8, 6, 8),
            "encoder_hidden_states": torch.randn(1, 12, 16),
            "encoder_attention_mask": torch.ones(1, 12),
            "timesteps": torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]]),
            "audio_timesteps": torch.tensor([500.0]),
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_latents", return_value=torch.randn(1, 8, 128)):
            with patch(
                "simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_audio_latents",
                return_value=torch.randn(1, 6, 8),
            ):
                with patch(
                    "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_latents",
                    return_value=torch.randn(1, 128, 2, 2, 2),
                ):
                    with patch(
                        "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_audio_latents",
                        return_value=torch.randn(1, 8, 6, 8),
                    ):
                        model.model_predict(prepared_batch)

        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["audio_timestep"], torch.tensor([500.0])))

    def test_prepare_crepa_self_flow_batch_sets_audio_teacher_metadata(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=MagicMock(patch_size=2, patch_size_t=1))
        model.unwrap_model = lambda wrapped=None, model_obj=None: wrapped if wrapped is not None else model_obj
        alt_sigmas = torch.tensor([0.8], dtype=torch.float32)
        alt_timesteps = torch.tensor([800.0], dtype=torch.float32)
        model.sample_flow_sigmas = MagicMock(return_value=(alt_sigmas, alt_timesteps))

        batch = {
            "latents": torch.zeros(1, 1, 2, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 1, 2, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor(
            [[[[0.2, 0.7], [0.9, 0.1]], [[0.4, 0.6], [0.8, 0.3]]]],
            dtype=torch.float32,
        )

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 8))
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 4, 4))
        self.assertEqual(result["audio_timesteps"].shape, (1,))
        self.assertEqual(result["audio_sigmas"].shape, (1,))
        self.assertEqual(result["crepa_teacher_audio_timesteps"].shape, (1,))
        self.assertEqual(result["crepa_teacher_audio_sigmas"].shape, (1,))
        self.assertEqual(result["audio_timesteps"].item(), 200.0)
        self.assertAlmostEqual(result["audio_sigmas"].item(), 0.2)
        self.assertEqual(result["crepa_teacher_audio_timesteps"].item(), 200.0)
        self.assertAlmostEqual(result["crepa_teacher_audio_sigmas"].item(), 0.2)
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_prepare_crepa_self_flow_batch_builds_tokenwise_audio_schedule(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=MagicMock(patch_size=2, patch_size_t=1))
        model.unwrap_model = lambda wrapped=None, model_obj=None: wrapped if wrapped is not None else model_obj
        alt_sigmas = torch.tensor([0.8], dtype=torch.float32)
        alt_timesteps = torch.tensor([800.0], dtype=torch.float32)
        model.sample_flow_sigmas = MagicMock(return_value=(alt_sigmas, alt_timesteps))

        batch = {
            "latents": torch.zeros(1, 1, 2, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 1, 2, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
            "audio_latent_batch": torch.zeros(1, 8, 3, 4, dtype=torch.float32),
        }
        fake_video_mask_rand = torch.tensor(
            [[[[0.2, 0.7], [0.9, 0.1]], [[0.4, 0.6], [0.8, 0.3]]]],
            dtype=torch.float32,
        )
        fake_audio_mask_rand = torch.tensor([[0.2, 0.8, 0.3]], dtype=torch.float32)

        with patch("torch.rand", side_effect=[fake_video_mask_rand, fake_audio_mask_rand]):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["audio_timesteps"].shape, (1, 3))
        self.assertEqual(result["audio_sigmas"].shape, (1, 1, 3, 1))
        self.assertEqual(set(result["audio_timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertTrue(torch.equal(result["crepa_self_flow_audio_mask"], fake_audio_mask_rand < 0.5))

    def test_prepare_batch_conditions_prefers_explicit_audio_sigmas_and_builds_teacher_audio_view(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(
            weight_dtype=torch.float32,
            input_perturbation=0.0,
            input_perturbation_steps=None,
            framerate=24,
        )
        model._warned_missing_audio = False
        model._warned_missing_video = False
        model._calculate_expected_audio_latent_length = MagicMock(return_value=3)

        batch = {
            "latents": torch.zeros(1, 128, 2, 2, 2, dtype=torch.float32),
            "sigmas": torch.full((1, 1, 2, 2, 2), 0.9, dtype=torch.float32),
            "timesteps": torch.tensor([[100.0, 900.0]], dtype=torch.float32),
            "audio_latent_batch": torch.zeros(1, 8, 3, 4, dtype=torch.float32),
            "audio_sigmas": torch.tensor([0.2], dtype=torch.float32),
            "audio_timesteps": torch.tensor([200.0], dtype=torch.float32),
            "crepa_teacher_audio_sigmas": torch.tensor([0.1], dtype=torch.float32),
            "crepa_teacher_audio_timesteps": torch.tensor([100.0], dtype=torch.float32),
        }

        with patch("torch.randn_like", return_value=torch.ones(1, 8, 3, 4, dtype=torch.float32)):
            result = model.prepare_batch_conditions(batch=batch, state={"global_step": 0})

        self.assertTrue(torch.allclose(result["audio_sigmas"], torch.full((1, 1, 1, 1), 0.2)))
        self.assertTrue(torch.allclose(result["audio_noisy_latents"], torch.full((1, 8, 3, 4), 0.2)))
        self.assertTrue(torch.allclose(result["crepa_teacher_audio_sigmas"], torch.full((1, 1, 1, 1), 0.1)))
        self.assertTrue(torch.allclose(result["crepa_teacher_audio_noisy_latents"], torch.full((1, 8, 3, 4), 0.1)))
        self.assertTrue(torch.equal(result["audio_timesteps"], torch.tensor([200.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(result["crepa_teacher_audio_timesteps"], torch.tensor([100.0], dtype=torch.float32)))

    def test_model_predict_preserves_tokenwise_audio_timesteps(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.config = MagicMock(weight_dtype=torch.float32, framerate=24, twinflow_enabled=False, tread_config=None)
        model._load_connectors = MagicMock()
        model.connectors = MagicMock(
            return_value=(
                torch.randn(1, 8, 16),
                torch.randn(1, 6, 16),
                torch.ones(1, 8),
            )
        )
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        model.model = MagicMock(config=MagicMock(patch_size=1, patch_size_t=1))
        model.model.return_value = (
            torch.randn(1, 8, 128),
            torch.randn(1, 6, 8),
            torch.randn(1, 2, 4, 16),
        )

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2, 2),
            "audio_noisy_latents": torch.randn(1, 8, 6, 8),
            "audio_latents": torch.randn(1, 8, 6, 8),
            "encoder_hidden_states": torch.randn(1, 12, 16),
            "encoder_attention_mask": torch.ones(1, 12),
            "timesteps": torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]]),
            "audio_timesteps": torch.tensor([[200.0, 800.0, 200.0, 800.0, 200.0, 800.0]]),
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_latents", return_value=torch.randn(1, 8, 128)):
            with patch(
                "simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_audio_latents",
                return_value=torch.randn(1, 6, 8),
            ):
                with patch(
                    "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_latents",
                    return_value=torch.randn(1, 128, 2, 2, 2),
                ):
                    with patch(
                        "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_audio_latents",
                        return_value=torch.randn(1, 8, 6, 8),
                    ):
                        model.model_predict(prepared_batch)

        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["audio_timestep"],
                torch.tensor([[200.0, 800.0, 200.0, 800.0, 200.0, 800.0]]),
            )
        )

    def test_model_predict_preserves_tokenwise_target_timesteps_with_reference_tokens(self):
        model = LTXVideo2.__new__(LTXVideo2)
        model.config = MagicMock(weight_dtype=torch.float32, framerate=24, twinflow_enabled=False, tread_config=None)
        model._load_connectors = MagicMock()
        model.connectors = MagicMock(
            return_value=(
                torch.randn(1, 8, 16),
                torch.randn(1, 6, 16),
                torch.ones(1, 8),
            )
        )
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.model = MagicMock(config=MagicMock(patch_size=1, patch_size_t=1))
        model.model.rope = None
        model.model.return_value = (
            torch.randn(1, 10, 128),
            torch.randn(1, 6, 8),
        )

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2, 2),
            "conditioning_latents": torch.randn(1, 128, 1, 1, 2),
            "audio_noisy_latents": torch.randn(1, 8, 6, 8),
            "audio_latents": torch.randn(1, 8, 6, 8),
            "encoder_hidden_states": torch.randn(1, 12, 16),
            "encoder_attention_mask": torch.ones(1, 12),
            "timesteps": torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]]),
        }

        with patch(
            "simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_latents",
            side_effect=[torch.randn(1, 8, 128), torch.randn(1, 2, 128)],
        ):
            with patch(
                "simpletuner.helpers.models.ltxvideo2.model.pack_ltx2_audio_latents",
                return_value=torch.randn(1, 6, 8),
            ):
                with patch(
                    "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_latents",
                    return_value=torch.randn(1, 128, 2, 2, 2),
                ):
                    with patch(
                        "simpletuner.helpers.models.ltxvideo2.model.unpack_ltx2_audio_latents",
                        return_value=torch.randn(1, 8, 6, 8),
                    ):
                        model.model_predict(prepared_batch)

        transformer_kwargs = model.model.call_args.kwargs
        expected = torch.tensor([[0.0, 0.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], expected))


class TestWanFrameAdjustment(unittest.TestCase):
    """Test Wan frame constraint (frames % 8 == 1)"""

    def test_valid_frame_count_unchanged(self):
        """Frame counts satisfying frames % 8 == 1 should be unchanged"""
        valid_counts = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81]
        for count in valid_counts:
            with self.subTest(frames=count):
                result = Wan.adjust_video_frames(count)
                self.assertEqual(result, count)
                self.assertEqual(result % 8, 1)

    def test_invalid_frame_count_rounds_down(self):
        """Invalid frame counts should round down to nearest valid value"""
        test_cases = [
            (119, 113),  # 119 -> 113
            (50, 49),  # 50 -> 49
            (100, 97),  # 100 -> 97
        ]
        for input_frames, expected_frames in test_cases:
            with self.subTest(input_frames=input_frames):
                result = Wan.adjust_video_frames(input_frames)
                self.assertEqual(result, expected_frames)
                self.assertEqual(result % 8, 1)

    def test_adjustment_idempotent(self):
        """Applying adjustment twice should give same result"""
        for frames in [50, 119, 100]:
            adjusted_once = Wan.adjust_video_frames(frames)
            adjusted_twice = Wan.adjust_video_frames(adjusted_once)
            self.assertEqual(adjusted_once, adjusted_twice)

    def test_self_flow_support_tracks_expand_timestep_flavour(self):
        model = Wan.__new__(Wan)
        model._wan_expand_timesteps = False
        self.assertFalse(model.supports_crepa_self_flow())
        model._wan_expand_timesteps = True
        self.assertTrue(model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_builds_mixed_student_and_clean_teacher_views(self):
        model = Wan.__new__(Wan)
        model._wan_expand_timesteps = True
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=MagicMock(patch_size=(1, 2, 2)))
        model.unwrap_model = lambda wrapped=None, model_obj=None: wrapped if wrapped is not None else model_obj
        alt_sigmas = torch.tensor([0.8], dtype=torch.float32)
        alt_timesteps = torch.tensor([800.0], dtype=torch.float32)
        model.sample_flow_sigmas = MagicMock(return_value=(alt_sigmas, alt_timesteps))

        batch = {
            "latents": torch.zeros(1, 1, 2, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 1, 2, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor(
            [[[[0.2, 0.7], [0.9, 0.1]], [[0.4, 0.6], [0.8, 0.3]]]],
            dtype=torch.float32,
        )

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 8))
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 4, 4))
        self.assertEqual(result["crepa_teacher_sigmas"].shape, (1, 1, 1, 1, 1))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        self.assertEqual(result["crepa_teacher_noisy_latents"].shape, (1, 1, 2, 4, 4))

        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertEqual(result["crepa_teacher_timesteps"].item(), 200.0)
        self.assertTrue(torch.allclose(result["crepa_teacher_noisy_latents"], torch.full_like(result["latents"], 0.2)))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        model = Wan.__new__(Wan)
        model._wan_expand_timesteps = True
        model.config = MagicMock(weight_dtype=torch.float32, tread_config=None, twinflow_enabled=False)
        model._apply_i2v_conditioning_to_kwargs = MagicMock()
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        predicted = torch.randn(1, 1, 2, 4, 4)
        captured = torch.randn(1, 2, 4, 8)
        model.model = MagicMock(return_value=(predicted, captured))

        tokenwise_timesteps = torch.tensor([[200.0, 800.0, 200.0, 800.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 1, 2, 4, 4),
            "encoder_hidden_states": torch.randn(1, 4, 8),
            "timesteps": tokenwise_timesteps,
            "latents": torch.randn(1, 1, 2, 4, 4),
            "crepa_capture_block_index": 11,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["model_prediction"], predicted)
        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_state_layer"], 11)
        self.assertTrue(transformer_kwargs["output_hidden_states"])
        model._apply_i2v_conditioning_to_kwargs.assert_called_once()


class TestLTXVideoFrameAdjustment(unittest.TestCase):
    """Test LTXVideo frame constraint (frames % 8 == 1)"""

    def test_valid_frame_count_unchanged(self):
        """Frame counts satisfying frames % 8 == 1 should be unchanged"""
        valid_counts = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
        for count in valid_counts:
            with self.subTest(frames=count):
                result = LTXVideo.adjust_video_frames(count)
                self.assertEqual(result, count)
                self.assertEqual(result % 8, 1)

    def test_invalid_frame_count_rounds_down(self):
        """Invalid frame counts should round down to nearest valid value"""
        test_cases = [
            (119, 113),  # 119 -> 113
            (50, 49),  # 50 -> 49
            (100, 97),  # 100 -> 97
        ]
        for input_frames, expected_frames in test_cases:
            with self.subTest(input_frames=input_frames):
                result = LTXVideo.adjust_video_frames(input_frames)
                self.assertEqual(result, expected_frames)
                self.assertEqual(result % 8, 1)

    def test_adjustment_idempotent(self):
        """Applying adjustment twice should give same result"""
        for frames in [50, 119, 100]:
            adjusted_once = LTXVideo.adjust_video_frames(frames)
            adjusted_twice = LTXVideo.adjust_video_frames(adjusted_once)
            self.assertEqual(adjusted_once, adjusted_twice)

    def test_model_supports_crepa_self_flow(self):
        model = LTXVideo.__new__(LTXVideo)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        model = LTXVideo.__new__(LTXVideo)
        model.config = MagicMock(weight_dtype=torch.float32, framerate=25, twinflow_enabled=False)
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True

        predicted = torch.randn(1, 8, 128)
        captured = torch.randn(1, 2, 4, 16)
        model.model = MagicMock(return_value=(predicted, captured))

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2, 4),
            "encoder_hidden_states": torch.randn(1, 77, 4096),
            "encoder_attention_mask": torch.ones(1, 77),
            "timesteps": tokenwise_timesteps,
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.ltxvideo.model.pack_ltx_latents", return_value=torch.randn(1, 8, 128)):
            with patch(
                "simpletuner.helpers.models.ltxvideo.model.unpack_ltx_latents",
                return_value=torch.randn(1, 128, 2, 2, 4),
            ):
                result = model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_state_layer"], 7)
        self.assertTrue(transformer_kwargs["output_hidden_states"])


class TestSanaVideoFrameAdjustment(unittest.TestCase):
    """Test SanaVideo frame constraint (frames % 8 == 1)"""

    def test_valid_frame_count_unchanged(self):
        """Frame counts satisfying frames % 8 == 1 should be unchanged"""
        valid_counts = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81]
        for count in valid_counts:
            with self.subTest(frames=count):
                result = SanaVideo.adjust_video_frames(count)
                self.assertEqual(result, count)
                self.assertEqual(result % 8, 1)

    def test_invalid_frame_count_rounds_down(self):
        """Invalid frame counts should round down to nearest valid value"""
        test_cases = [
            (119, 113),  # 119 -> 113
            (50, 49),  # 50 -> 49
            (100, 97),  # 100 -> 97
        ]
        for input_frames, expected_frames in test_cases:
            with self.subTest(input_frames=input_frames):
                result = SanaVideo.adjust_video_frames(input_frames)
                self.assertEqual(result, expected_frames)
                self.assertEqual(result % 8, 1)

    def test_adjustment_idempotent(self):
        """Applying adjustment twice should give same result"""
        for frames in [50, 119, 100]:
            adjusted_once = SanaVideo.adjust_video_frames(frames)
            adjusted_twice = SanaVideo.adjust_video_frames(adjusted_once)
            self.assertEqual(adjusted_once, adjusted_twice)

    def test_model_supports_crepa_self_flow(self):
        model = SanaVideo.__new__(SanaVideo)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        model = SanaVideo.__new__(SanaVideo)
        model.config = MagicMock(weight_dtype=torch.float32)
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True

        predicted = torch.randn(1, 16, 2, 4, 4)
        captured = torch.randn(1, 2, 16, 32)
        model.model = MagicMock(return_value=(predicted, captured))

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 2, 2, 4),
            "encoder_hidden_states": torch.randn(1, 77, 64),
            "encoder_attention_mask": torch.ones(1, 77),
            "timesteps": tokenwise_timesteps,
            "crepa_capture_block_index": 7,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_state_layer"], 7)
        self.assertTrue(transformer_kwargs["output_hidden_states"])


class TestLongCatVideoFrameAdjustment(unittest.TestCase):
    """Test LongCatVideo frame constraint (frames % 8 == 1)"""

    def test_valid_frame_count_unchanged(self):
        """Frame counts satisfying frames % 8 == 1 should be unchanged"""
        valid_counts = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81]
        for count in valid_counts:
            with self.subTest(frames=count):
                result = LongCatVideo.adjust_video_frames(count)
                self.assertEqual(result, count)
                self.assertEqual(result % 8, 1)

    def test_invalid_frame_count_rounds_down(self):
        """Invalid frame counts should round down to nearest valid value"""
        test_cases = [
            (119, 113),  # 119 -> 113
            (50, 49),  # 50 -> 49
            (100, 97),  # 100 -> 97
        ]
        for input_frames, expected_frames in test_cases:
            with self.subTest(input_frames=input_frames):
                result = LongCatVideo.adjust_video_frames(input_frames)
                self.assertEqual(result, expected_frames)
                self.assertEqual(result % 8, 1)

    def test_adjustment_idempotent(self):
        """Applying adjustment twice should give same result"""
        for frames in [50, 119, 100]:
            adjusted_once = LongCatVideo.adjust_video_frames(frames)
            adjusted_twice = LongCatVideo.adjust_video_frames(adjusted_once)
            self.assertEqual(adjusted_once, adjusted_twice)

    def test_model_supports_crepa_self_flow(self):
        model = LongCatVideo.__new__(LongCatVideo)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_builds_framewise_student_and_teacher_views(self):
        model = LongCatVideo.__new__(LongCatVideo)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=MagicMock(patch_size=(1, 2, 2)))
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        alt_sigmas = torch.tensor([0.8], dtype=torch.float32)
        alt_timesteps = torch.tensor([800.0], dtype=torch.float32)
        model.sample_flow_sigmas = MagicMock(return_value=(alt_sigmas, alt_timesteps))

        batch = {
            "latents": torch.zeros(1, 16, 4, 2, 2, dtype=torch.float32),
            "input_noise": torch.ones(1, 16, 4, 2, 2, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[0.2, 0.7, 0.1, 0.9]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["sigmas"].shape, (1, 1, 4, 1, 1))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertEqual(result["crepa_teacher_timesteps"].item(), 200.0)
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_returns_crepa_hidden_states_from_buffer_capture_override(self):
        model = LongCatVideo.__new__(LongCatVideo)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, twinflow_enabled=False)
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        model.model = MagicMock()
        model.model.return_value = (torch.randn(1, 16, 2, 2, 2),)

        def fake_forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = torch.randn(1, 2, 4, 8)
            return (torch.randn(1, 16, 2, 2, 2),)

        model.model.side_effect = fake_forward

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 4, 8),
            "encoder_attention_mask": torch.ones(1, 4),
            "timesteps": torch.tensor([100.0]),
            "crepa_capture_block_index": 7,
        }

        result = model.model_predict(prepared_batch)

        self.assertIsNotNone(result["crepa_hidden_states"])
        self.assertEqual(result["crepa_hidden_states"].shape, (1, 2, 4, 8))
        self.assertEqual(result["hidden_states_buffer"].capture_layers, {7})

    def test_model_predict_trims_conditioning_tokens_from_crepa_hidden_states(self):
        model = LongCatVideo.__new__(LongCatVideo)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, twinflow_enabled=False)
        model._new_hidden_state_buffer = MagicMock(return_value=None)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True

        def fake_forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
            return (torch.randn(1, 16, 4, 2, 2),)

        model.model = MagicMock(side_effect=fake_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 2, 2),
            "encoder_hidden_states": torch.randn(1, 4, 8),
            "encoder_attention_mask": torch.ones(1, 4),
            "timesteps": torch.tensor([100.0]),
            "conditioning_latent_count": 1,
            "crepa_capture_block_index": 7,
        }

        result = model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"][:, :, 0], torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])))

    def test_model_predict_preserves_existing_capture_layers_on_shared_buffer(self):
        model = LongCatVideo.__new__(LongCatVideo)
        model.accelerator = MagicMock(device=torch.device("cpu"))
        model.config = MagicMock(weight_dtype=torch.float32, twinflow_enabled=False)
        existing_buffer = HiddenStateBuffer()
        existing_buffer.capture_layers = {1, 2}
        model._new_hidden_state_buffer = MagicMock(return_value=existing_buffer)
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model.crepa_regularizer.wants_hidden_states.return_value = True

        def fake_forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = torch.randn(1, 8, 4)
            return (torch.randn(1, 16, 2, 2, 2),)

        model.model = MagicMock(side_effect=fake_forward)
        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 4, 8),
            "encoder_attention_mask": torch.ones(1, 4),
            "timesteps": torch.tensor([100.0]),
            "crepa_capture_block_index": 7,
        }

        model.model_predict(prepared_batch)

        self.assertEqual(existing_buffer.capture_layers, {1, 2})


class TestAdjustVideoFramesForModel(unittest.TestCase):
    """Test the adjust_video_frames_for_model helper function"""

    def test_none_model_returns_unchanged(self):
        """When model_class is None, video should be unchanged"""
        video = np.random.rand(100, 480, 640, 3)
        adjusted, original, was_adjusted = adjust_video_frames_for_model(video, None)
        np.testing.assert_array_equal(adjusted, video)
        self.assertEqual(original, 100)
        self.assertFalse(was_adjusted)

    def test_model_without_method_returns_unchanged(self):
        """When model doesn't have adjust_video_frames, video should be unchanged"""
        video = np.random.rand(100, 480, 640, 3)
        mock_model = MagicMock(spec=[])
        adjusted, original, was_adjusted = adjust_video_frames_for_model(video, mock_model)
        np.testing.assert_array_equal(adjusted, video)
        self.assertEqual(original, 100)
        self.assertFalse(was_adjusted)

    def test_no_adjustment_needed(self):
        """When video already satisfies constraint, should be unchanged"""
        video = np.random.rand(49, 480, 640, 3)
        mock_model = MagicMock()
        mock_model.adjust_video_frames = MagicMock(return_value=49)
        adjusted, original, was_adjusted = adjust_video_frames_for_model(video, mock_model)
        np.testing.assert_array_equal(adjusted, video)
        self.assertEqual(original, 49)
        self.assertFalse(was_adjusted)

    def test_adjustment_trims_video(self):
        """When adjustment is needed, video should be trimmed"""
        video = np.random.rand(119, 480, 640, 3)
        mock_model = MagicMock()
        mock_model.adjust_video_frames = MagicMock(return_value=113)
        adjusted, original, was_adjusted = adjust_video_frames_for_model(video, mock_model)
        self.assertEqual(adjusted.shape[0], 113)
        self.assertEqual(original, 119)
        self.assertTrue(was_adjusted)
        # First 113 frames should match
        np.testing.assert_array_equal(adjusted, video[:113])

    def test_ltxvideo2_adjustment_via_helper(self):
        """Test adjustment via helper with actual LTXVideo2 model"""
        video = np.random.rand(119, 480, 640, 3)
        adjusted, original, was_adjusted = adjust_video_frames_for_model(video, LTXVideo2)
        self.assertEqual(adjusted.shape[0], 113)
        self.assertEqual(original, 119)
        self.assertTrue(was_adjusted)

    def test_adjustment_preserves_shape_except_frames(self):
        """Adjustment should only change frame dimension"""
        video = np.random.rand(100, 480, 640, 3)
        adjusted, _, _ = adjust_video_frames_for_model(video, LTXVideo2)
        self.assertEqual(adjusted.shape[1], 480)  # Height unchanged
        self.assertEqual(adjusted.shape[2], 640)  # Width unchanged
        self.assertEqual(adjusted.shape[3], 3)  # Channels unchanged

    def test_adjustment_preserves_frame_order(self):
        """Adjustment should preserve the order of frames"""
        # Create video with unique values per frame
        video = np.zeros((119, 10, 10, 3))
        for i in range(119):
            video[i] = i
        adjusted, _, _ = adjust_video_frames_for_model(video, LTXVideo2)
        # Check first 113 frames match
        np.testing.assert_array_equal(adjusted, video[:113])


class TestFrameAdjustmentIntegration(unittest.TestCase):
    """Integration tests for frame adjustment across the system"""

    def test_ltxvideo2_class_method_accessible(self):
        """LTXVideo2.adjust_video_frames should be accessible as class method"""
        self.assertTrue(hasattr(LTXVideo2, "adjust_video_frames"))
        self.assertTrue(callable(LTXVideo2.adjust_video_frames))

    def test_adjustment_idempotent(self):
        """Applying adjustment twice should give same result as once"""
        for frames in [50, 119, 100, 200]:
            adjusted_once = LTXVideo2.adjust_video_frames(frames)
            adjusted_twice = LTXVideo2.adjust_video_frames(adjusted_once)
            self.assertEqual(adjusted_once, adjusted_twice)

    def test_different_video_dimensions(self):
        """Adjustment should work with different video dimensions"""
        dimensions = [
            (119, 480, 640, 3),
            (119, 960, 544, 3),
            (119, 512, 512, 3),
            (119, 1920, 1080, 3),
        ]
        for shape in dimensions:
            with self.subTest(shape=shape):
                video = np.random.rand(*shape)
                adjusted, original, was_adjusted = adjust_video_frames_for_model(video, LTXVideo2)
                self.assertEqual(adjusted.shape[0], 113)
                self.assertEqual(adjusted.shape[1:], video.shape[1:])


if __name__ == "__main__":
    unittest.main()
