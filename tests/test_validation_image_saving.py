import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from simpletuner.helpers.training import validation_images


class ValidationImageSavingTests(unittest.TestCase):
    def test_multiple_resolutions_use_generated_image_count_with_prompt_banners(self):
        for image_count in (1, 2):
            with self.subTest(image_count=image_count), tempfile.TemporaryDirectory() as directory:
                config = SimpleNamespace(
                    num_eval_images=25,
                    num_validation_images=image_count,
                    validation_image_format="png",
                    validation_image_quality=90,
                )
                resolutions = [(64, 64), (128, 128)]
                generated_resolutions = [resolution for resolution in resolutions for _ in range(image_count)]
                images = [Image.new("RGB", (width, height + 47)) for width, height in generated_resolutions]
                with (
                    patch.object(validation_images.StateTracker, "get_global_step", return_value=24),
                    patch.object(validation_images, "record_validation_media") as record,
                ):
                    validation_images.save_images(directory, {"portrait": images}, "portrait", resolutions, config)
                expected_names = []
                for index, (width, height) in enumerate(generated_resolutions):
                    filename = f"step_24_portrait_{index}_{width}x{height}.png"
                    expected_names.append(filename)
                    with Image.open(Path(directory, filename)) as saved:
                        self.assertEqual(saved.size, (width, height + 47))
                self.assertCountEqual([path.name for path in Path(directory).iterdir()], expected_names)
                self.assertEqual(
                    [call.kwargs["resolution"] for call in record.call_args_list],
                    [f"{width}x{height}" for width, height in generated_resolutions],
                )


if __name__ == "__main__":
    unittest.main()
