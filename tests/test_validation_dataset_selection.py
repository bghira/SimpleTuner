import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import (
    Evaluation,
    _validation_input_is_configured,
    _validation_reference_prompt_metadata,
    reset_eval_datasets,
    retrieve_eval_images,
    retrieve_validation_edit_images,
    retrieve_validation_images,
    retrieve_validation_s2v_samples,
)


class EvalDatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        registry = patch.object(StateTracker, "datasets", {})
        registry.start()
        self.addCleanup(registry.stop)

    def _register(self, id, *, sampler=None, root=None, config=None):
        dataset = DatasetRuntime(id, None, SimpleNamespace(instance_data_dir=root), config or {})
        dataset.sampler = sampler
        StateTracker.register_dataset(dataset)
        return dataset

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

    def test_evaluation_uses_the_registered_runtime_through_collation(self):
        sampler = MagicMock()
        sampler.__iter__.return_value = iter([0])
        sampler.__len__.return_value = 1
        dataset = self._register("held-out", sampler=sampler, config={"dataset_type": "eval"})
        dataset.training_dataset = [{"loaded": True}]
        self._register("training")
        with patch("simpletuner.helpers.training.collate.collate_fn", return_value={"prepared": True}) as collate:
            result = retrieve_eval_images("held-out")
        self.assertEqual(result, {"prepared": True})
        collate.assert_called_once_with([{"loaded": True}], dataset=dataset)
        self.assertEqual(Evaluation.total_eval_batches(None), 1)
        reset_eval_datasets()
        sampler._reset_buckets.assert_called_once_with(raise_exhaustion_signal=False)

    def test_s2v_validation_resolves_audio_from_the_registered_binding(self):
        model = self._build_base_model_mock()
        model.requires_s2v_validation_inputs.return_value = True
        args = self._build_args(eval_dataset_id="video")
        sampler = MagicMock()
        image = Image.new("RGB", (8, 8))
        sampler.retrieve_validation_set.return_value = [("clip", "prompt", "clip.mp4", image)]
        with TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "clip.flac"
            audio_file.touch()
            video = self._register("video", sampler=sampler, config={"dataset_type": "video"})
            audio = self._register("audio", config={"dataset_type": "audio", "instance_data_dir": tmpdir})
            video.audio_datasets = (audio,)

            with (
                patch.object(StateTracker, "get_model", return_value=model),
                patch.object(StateTracker, "get_args", return_value=args),
            ):
                result = retrieve_validation_s2v_samples()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].conditioning, {"image": image, "audio_path": str(audio_file)})

    def test_missing_eval_dataset_raises_for_standard_validation(self):
        model = self._build_base_model_mock()
        args = self._build_args(eval_dataset_id="missing-dataset")
        self._register("existing")

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            with self.assertRaisesRegex(ValueError, "missing-dataset"):
                retrieve_validation_images()

    def test_missing_eval_dataset_raises_for_edit_validation(self):
        model = self._build_base_model_mock()
        model.requires_validation_edit_captions.return_value = True
        args = self._build_args(eval_dataset_id="unknown-edit")
        self._register("present")

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            with self.assertRaisesRegex(ValueError, "unknown-edit"):
                retrieve_validation_edit_images()

    def test_edit_validation_relativizes_absolute_sample_against_relative_dataset_root(self):
        model = self._build_base_model_mock()
        model.requires_validation_edit_captions.return_value = True
        args = self._build_args(eval_dataset_id=None)
        dataset_root = Path("datasets/source")
        sample_path = str(dataset_root.resolve() / "clip.mp4")
        sampler = MagicMock()
        sampler.metadata_backend.instance_data_dir = str(dataset_root)
        sampler.retrieve_validation_set.return_value = [("clip", "prompt", sample_path, None)]
        conditioning_sample = MagicMock()
        conditioning_sampler = MagicMock()
        conditioning_sampler.get_conditioning_sample.return_value = conditioning_sample
        source = self._register("source", sampler=sampler, root=str(dataset_root))
        source.conditioning_datasets = (self._register("conditioning", sampler=conditioning_sampler),)

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            result = retrieve_validation_edit_images()

        self.assertEqual(len(result), 1)
        conditioning_sampler.get_conditioning_sample.assert_called_once_with(
            "clip.mp4", paired_source=("source", sample_path)
        )

        conditioning_sampler.reset_mock()
        native_ref = "webshart://3/152/shard.parquet/000000000152.jpg"
        sampler.retrieve_validation_set.return_value = [("clip", "prompt", native_ref, None)]
        with (
            patch.object(StateTracker, "get_model", return_value=model),
            patch.object(StateTracker, "get_args", return_value=args),
        ):
            result = retrieve_validation_edit_images()
        self.assertEqual(len(result), 1)
        conditioning_sampler.get_conditioning_sample.assert_called_once_with(
            native_ref, paired_source=("source", native_ref)
        )

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

        self._register("my-image-dataset", sampler=mock_sampler)

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
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

        self._register("present")

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
        ):
            # Should return empty because no conditioning datasets are linked
            result = retrieve_validation_images()
            self.assertEqual(result, [])

    def test_required_s2v_validation_takes_precedence_over_i2v_pairing(self):
        model = self._build_base_model_mock()
        model.requires_validation_i2v_samples.return_value = True
        model.requires_s2v_validation_inputs.return_value = True
        args = SimpleNamespace(validation_input=None, validation_using_datasets=False)
        expected = [MagicMock()]

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
            patch(
                "simpletuner.helpers.training.validation.retrieve_validation_s2v_samples",
                return_value=expected,
            ) as mock_s2v,
            patch("simpletuner.helpers.training.validation.retrieve_validation_edit_images") as mock_edit,
        ):
            result = retrieve_validation_images()

        self.assertEqual(result, expected)
        mock_s2v.assert_called_once_with()
        mock_edit.assert_not_called()

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
                patch("simpletuner.helpers.training.validation.StateTracker.get_datasets") as mock_get_datasets,
            ):
                result = retrieve_validation_images()

            self.assertEqual(len(result), 1)
            validation_prompt = result[0]
            self.assertTrue(validation_prompt.shortname.startswith("validation_input_0_input"))
            self.assertEqual(validation_prompt.prompt, "move the camera forward")
            self.assertEqual(validation_prompt.image_path, str(image_path))
            self.assertEqual(validation_prompt.conditioning.size, (32, 24))
            mock_get_datasets.assert_not_called()

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
        self._register("my-image-dataset", sampler=mock_sampler)

        with (
            patch("simpletuner.helpers.training.validation.StateTracker.get_model", return_value=model),
            patch("simpletuner.helpers.training.validation.StateTracker.get_args", return_value=args),
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
