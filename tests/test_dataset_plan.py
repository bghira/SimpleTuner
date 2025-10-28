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
    from simpletuner.simpletuner_sdk.server.services.dataset_plan import compute_validations
except ImportError as exc:  # pragma: no cover - optional dependency guard
    compute_validations = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class StrictI2VDatasetValidationTest(unittest.TestCase):
    """Verify dataset validation requirements for strict I2V flavours."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
            raise unittest.SkipTest(f"compute_validations import failed: {IMPORT_ERROR}")
        # Ensure the model registry is populated so lookups succeed.
        import simpletuner.helpers.models.all  # noqa: F401

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


if __name__ == "__main__":
    unittest.main()
