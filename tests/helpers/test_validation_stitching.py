import unittest
from unittest.mock import patch

from PIL import Image

from simpletuner.helpers.training.validation import Validation


class TestValidationStitching(unittest.TestCase):
    def setUp(self):
        self.validation = Validation.__new__(Validation)

    def test_stitch_three_images_repeats_static_media_to_match_video_length(self):
        left = Image.new("RGB", (10, 10), color="red")
        middle_frames = [
            Image.new("RGB", (8, 10), color="green"),
            Image.new("RGB", (8, 10), color="blue"),
        ]
        right = Image.new("RGB", (6, 10), color="yellow")

        stitched = self.validation.stitch_three_images(left, middle_frames, right, labels=["L", "M", "R"])

        self.assertIsInstance(stitched, list)
        self.assertEqual(len(stitched), 2)
        expected_size = (10 + 5 + 8 + 5 + 6, 10)
        for frame in stitched:
            self.assertEqual(frame.size, expected_size)

    def test_stitch_three_images_combines_multi_condition_left_input(self):
        left_conditions = [
            Image.new("RGB", (10, 12), color="red"),
            Image.new("RGB", (12, 12), color="blue"),
        ]
        middle = Image.new("RGB", (8, 12), color="green")
        right = Image.new("RGB", (6, 12), color="yellow")

        stitched = self.validation.stitch_three_images(left_conditions, middle, right)

        self.assertIsInstance(stitched, Image.Image)
        self.assertEqual(stitched.size, (10 + 12 + 5 + 8 + 5 + 6, 12))

    @patch("simpletuner.helpers.training.validation.StateTracker.get_global_step", return_value=321)
    def test_ema_comparison_labels_hot_weights_when_display_is_unlabelled(self, _global_step):
        labels = self.validation._ema_comparison_labels(display_has_checkpoint_label=False)

        self.assertEqual(labels, ["step 321", "EMA"])

    @patch("simpletuner.helpers.training.validation.StateTracker.get_global_step", return_value=321)
    def test_ema_comparison_keeps_existing_hot_weight_label(self, _global_step):
        labels = self.validation._ema_comparison_labels(display_has_checkpoint_label=True)

        self.assertEqual(labels, [None, "EMA"])


if __name__ == "__main__":
    unittest.main()
