import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.validation import (
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

        with patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model), patch(
            "simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args
        ), patch(
            "simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=available_backends
        ):
            with self.assertRaisesRegex(ValueError, "missing-dataset"):
                retrieve_validation_images()

    def test_missing_eval_dataset_raises_for_edit_validation(self):
        model = self._build_base_model_mock()
        model.requires_validation_edit_captions.return_value = True
        args = self._build_args(eval_dataset_id="unknown-edit")
        available_backends = {"present": {"id": "present"}}

        with patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model), patch(
            "simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args
        ), patch(
            "simpletuner.helpers.training.validation.StateTracker.get_data_backends", return_value=available_backends
        ):
            with self.assertRaisesRegex(ValueError, "unknown-edit"):
                retrieve_validation_edit_images()


if __name__ == "__main__":
    unittest.main()

