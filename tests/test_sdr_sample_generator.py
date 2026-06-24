import unittest

import numpy as np
from PIL import Image

from simpletuner.helpers.data_generation.sample_generator import (
    GENERATOR_REGISTRY,
    SampleGenerator,
    SDRDownsampleSampleGenerator,
)


class TestSDRDownsampleSampleGenerator(unittest.TestCase):
    def test_logc3_compression_matches_ltx_constants(self):
        values = np.array([0.0, SDRDownsampleSampleGenerator.CUT, 1.0], dtype=np.float32)

        result = SDRDownsampleSampleGenerator._compress_logc3(values)

        expected = np.array(
            [
                SDRDownsampleSampleGenerator.F,
                SDRDownsampleSampleGenerator.C
                * np.log10(
                    SDRDownsampleSampleGenerator.A * SDRDownsampleSampleGenerator.CUT + SDRDownsampleSampleGenerator.B
                )
                + SDRDownsampleSampleGenerator.D,
                SDRDownsampleSampleGenerator.C * np.log10(SDRDownsampleSampleGenerator.A + SDRDownsampleSampleGenerator.B)
                + SDRDownsampleSampleGenerator.D,
            ],
            dtype=np.float32,
        )
        expected = np.clip(expected, 0.0, 1.0)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_transform_batch_generates_logc3_rgb_image(self):
        source = Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8), mode="RGB")
        generator = SDRDownsampleSampleGenerator({"type": "logc3_sdr"})

        result = generator.transform_batch([source], ["sample.png"], [{}], None)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].mode, "RGB")
        self.assertEqual(result[0].size, source.size)

        expected_value = int(
            SDRDownsampleSampleGenerator._compress_logc3(np.array([128.0 / 255.0], dtype=np.float32))[0] * 255.0 + 0.5
        )
        self.assertEqual(np.asarray(result[0])[0, 0, 0], expected_value)

    def test_sdr_default_clamps_rec709_ldr_values(self):
        source = np.array([[[0.25, 1.25, -0.5]]], dtype=np.float32)
        generator = SDRDownsampleSampleGenerator({"type": "sdr"})

        result = generator.transform_batch([source], ["sample.npy"], [{}], None)

        np.testing.assert_array_equal(np.asarray(result[0])[0, 0], np.array([64, 255, 0], dtype=np.uint8))

    def test_transform_batch_accepts_nested_params_from_webui(self):
        source = Image.fromarray(np.full((1, 1, 3), 64, dtype=np.uint8), mode="RGB")
        generator = SDRDownsampleSampleGenerator({"type": "sdr", "params": {"transform": "srgb", "input_scale": 1.0}})

        result = generator.transform_batch([source], ["sample.png"], [{}], None)

        expected_value = int(SDRDownsampleSampleGenerator._linear_to_srgb(np.array([64.0 / 255.0]))[0] * 255.0 + 0.5)
        self.assertEqual(np.asarray(result[0])[0, 0, 0], expected_value)

    def test_generator_registry_aliases(self):
        for alias in ("sdr", "sdr_downsample", "logc3_sdr", "hdr_to_sdr"):
            self.assertIs(GENERATOR_REGISTRY[alias], SDRDownsampleSampleGenerator)
            generator = SampleGenerator.from_backend({"conditioning_config": {"type": alias}})
            self.assertIsInstance(generator, SDRDownsampleSampleGenerator)

    def test_video_arrays_are_rejected(self):
        generator = SDRDownsampleSampleGenerator({"type": "sdr"})
        video = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        with self.assertLogs("SampleGenerator", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "expects image samples"):
                generator.transform_batch([video], ["sample.mp4"], [{}], None)


if __name__ == "__main__":
    unittest.main()
