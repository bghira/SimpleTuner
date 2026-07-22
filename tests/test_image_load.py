import unittest
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


if __name__ == "__main__":
    unittest.main()
