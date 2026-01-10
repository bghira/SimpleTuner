import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from simpletuner.helpers.image_manipulation.load import adjust_video_frames_for_model
from simpletuner.helpers.models.common import VideoModelFoundation
from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2


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
