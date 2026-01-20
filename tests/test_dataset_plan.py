import sys
import types
import unittest

import torch

if not hasattr(torch, "cuda"):
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
if "torch.distributed" not in sys.modules:
    dist_stub = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    )
    sys.modules["torch.distributed"] = dist_stub
    torch.distributed = dist_stub  # type: ignore[attr-defined]

try:
    from simpletuner.helpers.distillation.common import DistillationBase
    from simpletuner.helpers.distillation.registry import DistillationRegistry
    from simpletuner.simpletuner_sdk.server.services.dataset_plan import _is_immediately_available, compute_validations
except ImportError as exc:  # pragma: no cover - optional dependency guard
    compute_validations = None
    _is_immediately_available = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

if IMPORT_ERROR is None:

    class _CaptionOnlyDistiller(DistillationBase):
        """Stub distiller used to exercise dataset requirement validation."""

        def __init__(self, *args, **kwargs):
            pass

else:  # pragma: no cover - optional dependency guard

    class _CaptionOnlyDistiller:  # type: ignore[override]
        pass


class StrictI2VDatasetValidationTest(unittest.TestCase):
    """Verify dataset validation requirements for strict I2V flavours."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
            raise unittest.SkipTest(f"compute_validations import failed: {IMPORT_ERROR}")
        # Ensure the model registry is populated so lookups succeed.
        import simpletuner.helpers.models.all  # noqa: F401

        cls._registry_snapshot = (
            DistillationRegistry._registry.copy(),
            DistillationRegistry._metadata.copy(),
            DistillationRegistry._requirement_profiles.copy(),
        )
        DistillationRegistry.register(
            "caption_generator",
            _CaptionOnlyDistiller,
            data_requirements=["caption"],
            is_data_generator=True,
        )

    @classmethod
    def tearDownClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover
            return
        DistillationRegistry._registry = cls._registry_snapshot[0]
        DistillationRegistry._metadata = cls._registry_snapshot[1]
        DistillationRegistry._requirement_profiles = cls._registry_snapshot[2]

    def _base_datasets(self):
        return [
            {"id": "image-dataset", "dataset_type": "image", "type": "local"},
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]

    def _collect_errors(self, validations):
        return [message.message for message in validations if message.level == "error"]

    def test_wan_i2v_requires_video_dataset(self):
        datasets = self._base_datasets()
        validations = compute_validations(
            datasets,
            blueprints=[],
            model_family="wan",
            model_flavour="i2v-14b-2.1",
        )
        errors = self._collect_errors(validations)
        self.assertTrue(any("video dataset" in msg for msg in errors))

    def test_wan_i2v_with_video_dataset_passes(self):
        datasets = self._base_datasets()
        datasets.append({"id": "video-dataset", "dataset_type": "video", "type": "local"})
        validations = compute_validations(
            datasets,
            blueprints=[],
            model_family="wan",
            model_flavour="i2v-14b-2.1",
        )
        errors = self._collect_errors(validations)
        self.assertFalse(any("video dataset" in msg for msg in errors))

    def test_non_strict_video_flavour_allows_image_dataset(self):
        datasets = self._base_datasets()
        validations = compute_validations(
            datasets,
            blueprints=[],
            model_family="wan",
            model_flavour="t2v-480p-1.3b-2.1",
        )
        errors = self._collect_errors(validations)
        self.assertFalse(any("video dataset" in msg for msg in errors))

    def test_caption_only_distiller_skips_image_requirement(self):
        datasets = [
            {"id": "captions", "dataset_type": "caption", "type": "local"},
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]
        validations = compute_validations(
            datasets,
            blueprints=[],
            distillation_method="caption_generator",
        )
        errors = self._collect_errors(validations)
        self.assertFalse(any("image dataset" in msg for msg in errors))

    def test_caption_csv_backend_rejected(self):
        datasets = [
            {"id": "captions", "dataset_type": "caption", "type": "csv"},
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]
        validations = compute_validations(
            datasets,
            blueprints=[],
            distillation_method="caption_generator",
        )
        errors = self._collect_errors(validations)
        self.assertTrue(any("Caption datasets cannot use CSV backends" in msg for msg in errors))


class IsImmediatelyAvailableTest(unittest.TestCase):
    """Tests for the _is_immediately_available helper function."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
            raise unittest.SkipTest(f"_is_immediately_available import failed: {IMPORT_ERROR}")

    def test_empty_dataset_is_immediately_available(self):
        """Dataset with no scheduling fields should be immediately available."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local"}
        self.assertTrue(_is_immediately_available(dataset))

    def test_start_epoch_none_is_immediate(self):
        """Explicit None for start_epoch should be immediate."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": None}
        self.assertTrue(_is_immediately_available(dataset))

    def test_start_step_none_is_immediate(self):
        """Explicit None for start_step should be immediate."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_step": None}
        self.assertTrue(_is_immediately_available(dataset))

    def test_start_epoch_1_is_immediate(self):
        """start_epoch=1 means available from epoch 1 (immediate)."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 1}
        self.assertTrue(_is_immediately_available(dataset))

    def test_start_step_0_is_immediate(self):
        """start_step=0 means available from step 0 (immediate)."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_step": 0}
        self.assertTrue(_is_immediately_available(dataset))

    def test_start_epoch_greater_than_1_not_immediate(self):
        """start_epoch > 1 means delayed."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 5}
        self.assertFalse(_is_immediately_available(dataset))

    def test_start_step_greater_than_0_not_immediate(self):
        """start_step > 0 means delayed."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_step": 100}
        self.assertFalse(_is_immediately_available(dataset))

    def test_both_immediate_values(self):
        """Both start_epoch=1 and start_step=0 should be immediate."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 1, "start_step": 0}
        self.assertTrue(_is_immediately_available(dataset))

    def test_epoch_delayed_step_immediate(self):
        """If start_epoch > 1 even with start_step=0, not immediate."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 3, "start_step": 0}
        self.assertFalse(_is_immediately_available(dataset))

    def test_epoch_immediate_step_delayed(self):
        """If start_step > 0 even with start_epoch=1, not immediate."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 1, "start_step": 50}
        self.assertFalse(_is_immediately_available(dataset))

    def test_negative_start_epoch_is_immediate(self):
        """Negative start_epoch is treated as <= 1 (immediate)."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": -1}
        self.assertTrue(_is_immediately_available(dataset))

    def test_negative_start_step_is_immediate(self):
        """Negative start_step is treated as <= 0 (immediate)."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_step": -10}
        self.assertTrue(_is_immediately_available(dataset))

    def test_float_start_epoch(self):
        """Float values for start_epoch should work."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_epoch": 2.5}
        self.assertFalse(_is_immediately_available(dataset))

    def test_float_start_step(self):
        """Float values for start_step should work."""
        dataset = {"id": "test", "dataset_type": "image", "type": "local", "start_step": 0.5}
        self.assertFalse(_is_immediately_available(dataset))


class DatasetSchedulingValidationTest(unittest.TestCase):
    """Tests for dataset scheduling validation in compute_validations."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
            raise unittest.SkipTest(f"compute_validations import failed: {IMPORT_ERROR}")
        import simpletuner.helpers.models.all  # noqa: F401

    def _base_datasets(self):
        return [
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]

    def _collect_errors(self, validations):
        return [message.message for message in validations if message.level == "error"]

    def _collect_by_field(self, validations, field):
        return [message for message in validations if message.field == field]

    def test_immediate_dataset_passes_validation(self):
        """Single immediate dataset should pass scheduling validation."""
        datasets = self._base_datasets()
        datasets.append({"id": "image-dataset", "dataset_type": "image", "type": "local"})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_all_delayed_datasets_fails_validation(self):
        """All datasets with delayed scheduling should fail validation."""
        datasets = self._base_datasets()
        datasets.append({"id": "delayed-1", "dataset_type": "image", "type": "local", "start_epoch": 5})
        datasets.append({"id": "delayed-2", "dataset_type": "image", "type": "local", "start_step": 100})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 1)
        self.assertIn("at least one dataset must be available", scheduling_errors[0].message)

    def test_one_immediate_among_delayed_passes(self):
        """One immediate dataset among delayed ones should pass."""
        datasets = self._base_datasets()
        datasets.append({"id": "immediate", "dataset_type": "image", "type": "local"})
        datasets.append({"id": "delayed-1", "dataset_type": "image", "type": "local", "start_epoch": 5})
        datasets.append({"id": "delayed-2", "dataset_type": "video", "type": "local", "start_step": 100})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_disabled_immediate_does_not_count(self):
        """Disabled immediate dataset should not satisfy the requirement."""
        datasets = self._base_datasets()
        datasets.append({"id": "immediate-disabled", "dataset_type": "image", "type": "local", "disabled": True})
        datasets.append({"id": "delayed", "dataset_type": "image", "type": "local", "start_epoch": 5})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 1)

    def test_text_embeds_not_counted_for_scheduling(self):
        """text_embeds datasets should not be counted for scheduling validation."""
        datasets = [
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
            {"id": "delayed", "dataset_type": "image", "type": "local", "start_epoch": 5},
        ]
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 1)

    def test_image_embeds_not_counted_for_scheduling(self):
        """image_embeds datasets should not be counted for scheduling validation."""
        datasets = [
            {"id": "image-embeds", "dataset_type": "image_embeds", "type": "local"},
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
            {"id": "delayed", "dataset_type": "image", "type": "local", "start_epoch": 5},
        ]
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 1)

    def test_video_dataset_immediate_passes(self):
        """Video dataset that is immediate should pass."""
        datasets = self._base_datasets()
        datasets.append({"id": "video", "dataset_type": "video", "type": "local"})
        validations = compute_validations(datasets, blueprints=[], model_family="wan", model_flavour="t2v-480p-1.3b-2.1")
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_audio_dataset_immediate_passes(self):
        """Audio dataset that is immediate should pass."""
        datasets = self._base_datasets()
        datasets.append({"id": "audio", "dataset_type": "audio", "type": "local"})
        validations = compute_validations(datasets, blueprints=[], model_family="ace_step")
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_start_epoch_1_is_immediate(self):
        """start_epoch=1 should be treated as immediate."""
        datasets = self._base_datasets()
        datasets.append({"id": "image", "dataset_type": "image", "type": "local", "start_epoch": 1})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_start_step_0_is_immediate(self):
        """start_step=0 should be treated as immediate."""
        datasets = self._base_datasets()
        datasets.append({"id": "image", "dataset_type": "image", "type": "local", "start_step": 0})
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)

    def test_no_training_datasets_no_scheduling_error(self):
        """If there are no training datasets, scheduling validation should not trigger."""
        datasets = [
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]
        validations = compute_validations(datasets, blueprints=[])
        scheduling_errors = self._collect_by_field(validations, "scheduling")
        self.assertEqual(len(scheduling_errors), 0)


class AudioOnlyDatasetValidationTest(unittest.TestCase):
    """Tests for audio-only dataset validation in compute_validations."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
            raise unittest.SkipTest(f"compute_validations import failed: {IMPORT_ERROR}")
        import simpletuner.helpers.models.all  # noqa: F401

    def _base_datasets(self):
        return [
            {"id": "text-embeds", "dataset_type": "text_embeds", "type": "local", "default": True},
        ]

    def _collect_errors(self, validations):
        return [message.message for message in validations if message.level == "error"]

    def test_ltxvideo2_implicit_audio_only_allows_audio_dataset(self):
        """LTX-2 with only audio datasets (no video/image) should implicitly be audio-only mode."""
        datasets = self._base_datasets()
        datasets.append({"id": "audio", "dataset_type": "audio", "type": "local"})
        validations = compute_validations(datasets, blueprints=[], model_family="ltxvideo2")
        errors = self._collect_errors(validations)
        # Should NOT require video/image when only audio datasets present
        self.assertFalse(any("image or video" in msg for msg in errors))

    def test_ltxvideo2_explicit_audio_only_allows_audio_dataset(self):
        """LTX-2 with explicit audio_only flag should not require video or image datasets."""
        datasets = self._base_datasets()
        datasets.append({"id": "audio", "dataset_type": "audio", "type": "local", "audio": {"audio_only": True}})
        validations = compute_validations(datasets, blueprints=[], model_family="ltxvideo2")
        errors = self._collect_errors(validations)
        self.assertFalse(any("image or video" in msg for msg in errors))

    def test_ltxvideo2_audio_with_video_passes(self):
        """LTX-2 with audio AND video datasets should pass (joint training)."""
        datasets = self._base_datasets()
        datasets.append({"id": "audio", "dataset_type": "audio", "type": "local"})
        datasets.append({"id": "video", "dataset_type": "video", "type": "local"})
        validations = compute_validations(datasets, blueprints=[], model_family="ltxvideo2")
        errors = self._collect_errors(validations)
        self.assertFalse(any("image or video" in msg for msg in errors))

    def test_non_audio_model_still_requires_images(self):
        """Non-audio models should still require image datasets even with audio_only flag."""
        datasets = self._base_datasets()
        datasets.append({"id": "audio", "dataset_type": "audio", "type": "local", "audio": {"audio_only": True}})
        # Using pixart which doesn't support audio-only
        validations = compute_validations(datasets, blueprints=[], model_family="pixart")
        errors = self._collect_errors(validations)
        self.assertTrue(any("image dataset" in msg for msg in errors))


if __name__ == "__main__":
    unittest.main()
