import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from simpletuner.helpers.image_manipulation.load import load_image


class LoadImageTests(unittest.TestCase):
    def test_load_image_reuses_unchanged_trainingsample_decode(self):
        calls = []
        decoded = np.zeros((3, 4, 3), dtype=np.uint8)

        def imdecode_py(_img_data, flags):
            calls.append(flags)
            if flags == -1:
                return decoded
            raise AssertionError("load_image should not decode RGB after unchanged decode succeeds")

        fake_tsr = SimpleNamespace(imdecode_py=imdecode_py)

        with patch("simpletuner.helpers.image_manipulation.load.tsr", fake_tsr):
            image = load_image(b"jpeg bytes")

        self.assertEqual(calls, [-1])
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (4, 3))

    @staticmethod
    def _png(array, mode):
        buffer = BytesIO()
        Image.fromarray(array, mode).save(buffer, "PNG")
        return buffer.getvalue()

    def test_a_grey_image_with_alpha_loads_as_the_rgb_its_rgba_twin_does(self):
        grey_alpha = np.random.default_rng(0).integers(0, 256, (6, 5, 2), dtype=np.uint8)
        rgba = np.concatenate([grey_alpha[..., :1]] * 3 + [grey_alpha[..., 1:]], axis=2)
        from simpletuner.helpers.image_manipulation import load

        with patch.object(load.tsr, "imdecode_py", wraps=load.tsr.imdecode_py) as decode:
            image = load_image(self._png(grey_alpha, "LA"))
        self.assertEqual([call.args[1] for call in decode.call_args_list], [-1])
        self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(np.array(image), np.array(load_image(self._png(rgba, "RGBA"))))

    def test_load_image_restores_a_missing_jpeg_end_marker(self):
        buffer = BytesIO()
        Image.new("RGB", (32, 24), "blue").save(buffer, "JPEG")
        jpeg = buffer.getvalue()
        self.assertEqual(jpeg[-2:], b"\xff\xd9")
        expected = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"))
        with patch("simpletuner.helpers.image_manipulation.load.tsr.imdecode_py", return_value=None):
            image = load_image(BytesIO(jpeg[:-2]))
        np.testing.assert_array_equal(np.asarray(image), expected)

    def test_load_image_still_reports_an_undecodable_jpeg(self):
        with patch("simpletuner.helpers.image_manipulation.load.tsr.imdecode_py", return_value=None):
            with self.assertRaises(Image.UnidentifiedImageError):
                load_image(b"\xff\xd8invalid header")


if __name__ == "__main__":
    unittest.main()
