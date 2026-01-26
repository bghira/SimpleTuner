import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.validation import retrieve_validation_edit_images, retrieve_validation_images


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
            patch("simpletuner.helpers.training.validation._normalise_validation_sample", side_effect=lambda x: x),
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


if __name__ == "__main__":
    unittest.main()
