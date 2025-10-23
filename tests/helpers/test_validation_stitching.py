import unittest

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


if __name__ == "__main__":
    unittest.main()
