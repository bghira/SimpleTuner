import unittest

from simpletuner.helpers.data_backend.factory import init_backend_config


class TestLTXVideo2FrameValidation(unittest.TestCase):
    def test_ltxvideo2_requires_num_frames_mod_8_eq_1(self):
        backend = {
            "id": "ltx2-video",
            "dataset_type": "video",
            "video": {
                "num_frames": 48,
                "min_frames": 48,
            },
        }
        args = {
            "model_family": "ltxvideo2",
            "model_flavour": "dev",
            "framerate": 25,
        }

        with self.assertRaises(ValueError) as ctx:
            init_backend_config(backend, args, accelerator=None)

        self.assertIn("frame_count % 8 == 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
