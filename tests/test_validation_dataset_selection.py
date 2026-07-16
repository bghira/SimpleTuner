import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.helpers.training.validation import (
    _validation_input_is_configured,
    _validation_reference_prompt_metadata,
    retrieve_validation_edit_images,
    retrieve_validation_images,
)


class EvalDatasetSelectionTests(unittest.TestCase):
    def _build_base_model_mock(self):
        model = MagicMock()
        model.requires_validation_edit_captions.return_value = False
        model.requires_validation_i2v_samples.return_value = False
        model.requires_conditioning_validation_inputs.return_value = False
        model.conditioning_validation_dataset_type.return_value = "image"
        model.validation_image_input_edge_length.return_value = None
        model.requires_s2v_validation_inputs.return_value = False
        return model

    def _build_args(self, eval_dataset_id):
        return SimpleNamespace(
            eval_dataset_id=eval_dataset_id,
            controlnet=False,
            control=False,
            num_eval_images=4,
        )

    def test_missing_eval_dataset_raises_for_standard_validation(self):
        model = self._build_base_model_mock()
        args = self._build_args(eval_dataset_id="missing-dataset")
        available_backends = {"existing": {"id": "existing"}}

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=available_backends),
        ):
            with self.assertRaisesRegex(ValueError, "missing-dataset"):
                retrieve_validation_images()

    def test_missing_eval_dataset_raises_for_edit_validation(self):
        model = self._build_base_model_mock()
        model.requires_validation_edit_captions.return_value = True
        args = self._build_args(eval_dataset_id="unknown-edit")
        available_backends = {"present": {"id": "present"}}

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=available_backends),
        ):
            with self.assertRaisesRegex(ValueError, "unknown-edit"):
                retrieve_validation_edit_images()

    def test_i2v_with_validation_using_datasets_uses_image_backend(self):
        """
        When validation_using_datasets is True for an i2v model, it should use
        simple image datasets rather than requiring conditioning dataset pairing.
        """
        model = self._build_base_model_mock()
        model.requires_validation_i2v_samples.return_value = True
        model.requires_conditioning_validation_inputs.return_value = True
        args = SimpleNamespace(
            eval_dataset_id=None,
            controlnet=False,
            control=False,
            num_eval_images=4,
            validation_using_datasets=True,
        )

        mock_sampler = MagicMock()
        mock_sampler.retrieve_validation_set.return_value = [
            ("shortname1", "prompt1", "/path/to/image1.jpg", None),
        ]

        image_backends = {
            "my-image-dataset": {
                "id": "my-image-dataset",
                "dataset_type": "image",
                "config": {},
                "sampler": mock_sampler,
            }
        }

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=image_backends),
            patch(
                "simpletuner.helpers.training.validation.StateTracker.get_data_backend",
                side_effect=lambda k: image_backends.get(k, {}),
            ),
            patch("simpletuner.helpers.training.validation._normalise_validation_sample", side_effect=lambda x, **_: x),
        ):
            result = retrieve_validation_images()
            self.assertEqual(len(result), 1)
            mock_sampler.retrieve_validation_set.assert_called_once_with(batch_size=4)

    def test_i2v_without_validation_using_datasets_routes_to_edit_images(self):
        """
        When validation_using_datasets is False for an i2v model, it should
        route to retrieve_validation_edit_images which requires conditioning datasets.
        """
        model = self._build_base_model_mock()
        model.requires_validation_i2v_samples.return_value = True
        model.requires_conditioning_validation_inputs.return_value = True
        args = SimpleNamespace(
            eval_dataset_id=None,
            controlnet=False,
            control=False,
            num_eval_images=4,
            validation_using_datasets=False,
        )

        image_backends = {"present": {"id": "present"}}

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=image_backends),
            patch("simpletuner.helpers.training.validation.StateTracker.get_conditioning_datasets", return_value=None),
        ):
            # Should return empty because no conditioning datasets are linked
            result = retrieve_validation_images()
            self.assertEqual(result, [])

    def test_validation_input_uses_standalone_images_without_dataset_backend(self):
        model = self._build_base_model_mock()
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "input.png"
            Image.new("RGB", (32, 24), color=(128, 64, 32)).save(image_path)
            args = SimpleNamespace(
                eval_dataset_id=None,
                controlnet=False,
                control=False,
                num_eval_images=4,
                validation_using_datasets=False,
                validation_input=f'[{{"path": "{image_path}", "prompt": "move the camera forward"}}]',
            )

            with (
                patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
                patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
                patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends") as mock_get_backends,
            ):
                result = retrieve_validation_images()

            self.assertEqual(len(result), 1)
            validation_prompt = result[0]
            self.assertTrue(validation_prompt.shortname.startswith("validation_input_0_input"))
            self.assertEqual(validation_prompt.prompt, "move the camera forward")
            self.assertEqual(validation_prompt.image_path, str(image_path))
            self.assertEqual(validation_prompt.conditioning.size, (32, 24))
            mock_get_backends.assert_not_called()

    def test_empty_validation_input_json_list_is_unset(self):
        self.assertFalse(_validation_input_is_configured(SimpleNamespace(validation_input="[]")))
        self.assertFalse(_validation_input_is_configured(SimpleNamespace(validation_input="  []  ")))

    def test_empty_validation_input_json_list_uses_dataset_selection(self):
        model = self._build_base_model_mock()
        args = SimpleNamespace(
            eval_dataset_id=None,
            controlnet=False,
            control=False,
            num_eval_images=4,
            validation_using_datasets=False,
            validation_input="[]",
        )

        mock_sampler = MagicMock()
        mock_sampler.retrieve_validation_set.return_value = [
            ("shortname1", "prompt1", "/path/to/image1.jpg", Image.new("RGB", (16, 16))),
        ]
        image_backends = {
            "my-image-dataset": {
                "id": "my-image-dataset",
                "dataset_type": "image",
                "config": {},
                "sampler": mock_sampler,
            }
        }

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch("simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=image_backends),
            patch(
                "simpletuner.helpers.training.validation.StateTracker.get_data_backend",
                side_effect=lambda k: image_backends.get(k, {}),
            ),
        ):
            result = retrieve_validation_images()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].prompt, "prompt1")
        mock_sampler.retrieve_validation_set.assert_called_once_with(batch_size=4)

    def test_validation_input_requires_json_list(self):
        model = self._build_base_model_mock()
        args = SimpleNamespace(validation_input='{"path": "/tmp/input.png", "prompt": "prompt"}')

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            with self.assertRaisesRegex(ValueError, "must be a list"):
                retrieve_validation_images()

    def test_validation_input_requires_non_empty_prompt(self):
        model = self._build_base_model_mock()
        args = SimpleNamespace(validation_input='[{"path": "/tmp/input.png", "prompt": ""}]')

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            with self.assertRaisesRegex(ValueError, "non-empty"):
                retrieve_validation_images()

    def test_validation_reference_metadata_includes_direct_conditioning_pixels(self):
        image = Image.new("RGB", (16, 8), color=(255, 0, 0))
        metadata = _validation_reference_prompt_metadata(image, image_path="/tmp/input.png")

        self.assertEqual(metadata["image_path"], "/tmp/input.png")
        self.assertEqual(metadata["image_paths"], ["/tmp/input.png"])
        self.assertIn("conditioning_pixel_values", metadata)
        self.assertEqual(tuple(metadata["conditioning_pixel_values"].shape), (3, 8, 16))


if __name__ == "__main__":
    unittest.main()
