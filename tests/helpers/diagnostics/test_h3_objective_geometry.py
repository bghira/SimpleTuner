import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from simpletuner.diagnostics.h3_objective_geometry import (
    GeometryPoint,
    _describe_batch,
    _first_raw_batch,
    _forbid_text_encoding,
    cosine_similarity,
    pca_coordinates,
    sample_vector,
    trajectory_metrics,
)


class H3ObjectiveGeometryTests(unittest.TestCase):
    def test_first_batch_uses_the_training_schedule_and_bound_text_cache(self):
        from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.training.state_tracker import StateTracker

        settings = SimpleNamespace(seed=42, data_backend_sampling="uniform")
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(
                text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION,
                unpack_text_embeddings_from_cache=lambda record: record,
                requires_text_embed_image_context=lambda: False,
            )
            model.encode_text_batch = lambda *args, **kwargs: _forbid_text_encoding(model, *args, **kwargs)
            backend = LocalDataBackend(None, "text")
            cache = TextEmbeddingCache(
                "text",
                backend,
                [],
                [],
                None,
                model=model,
                cache_dir=directory,
                text_cache_disable=True,
            )
            self.addCleanup(cache.close)
            record = {"prompt": "caption", "metadata": {}}
            backend.torch_save({"prompt_embeds": torch.ones(1, 1, 1)}, cache.hash_prompt_with_path(record))
            metadata = SimpleNamespace(view_revision="accepted-v1", requires_complete_preparation=True, epoch_provider=None)
            active = DatasetRuntime("active", None, metadata, {"dataset_type": "image", "start_step": 1})
            inactive = DatasetRuntime("inactive", None, metadata, {"dataset_type": "image", "start_step": 2})
            active.text_embed_cache = cache

            def batch():
                embeds = cache.read_training_batch([record])[0]
                yield {"data_backend_id": active.id, "filepaths": ["sample.png"], **embeds}

            active.training_dataloader = batch()
            inactive.training_dataloader = iter([{"data_backend_id": "inactive"}])
            trainer = SimpleNamespace(_training_datasets=lambda: {"active": active, "inactive": inactive})
            with patch.object(StateTracker, "get_args", return_value=settings):
                raw, dataset = _first_raw_batch(trainer)
            self.assertIs(dataset, active)
            torch.testing.assert_close(raw["prompt_embeds"], torch.ones(1, 1, 1))
            self.assertEqual(
                _describe_batch(dataset, raw),
                {
                    "data_backend_id": "active",
                    "view_revision": "accepted-v1",
                    "text_cache_mode": "read_existing",
                    "filepaths": ["sample.png"],
                },
            )
            backend.delete(cache.hash_prompt_with_path(record))
            with self.assertRaisesRegex(RuntimeError, "missed the prepared text embedding cache"):
                next(batch())

    def test_cosine_similarity_detects_sign_flip(self):
        value = torch.tensor([1.0, -2.0, 3.0])

        self.assertAlmostEqual(cosine_similarity(value, value), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(value, -value), -1.0, places=6)

    def test_trajectory_metrics_exposes_target_scale_and_adapter_residual(self):
        normal_target = torch.tensor([1.0, 0.0])
        anyflow_target = torch.tensor([-2.0, 0.0])
        base_prediction = torch.tensor([0.5, 0.0])
        drift_reference = torch.tensor([1.0, 0.0])
        adapter_prediction = torch.tensor([3.0, 0.0])
        point = GeometryPoint(
            timestep=500.0,
            sigma=0.5,
            model_timestep=0.5,
            r_timestep=0.75,
            r_sigma=0.25,
            anyflow_weight=1.0,
            drift_weight=0.5,
            sft_weight=1.0,
            normal_target=normal_target,
            anyflow_target=anyflow_target,
            flowmap_objective_target=torch.tensor([-1.0, 0.0]),
            base_prediction=base_prediction,
            drift_reference_prediction=drift_reference,
            normal_batch={},
            prepared_batch={},
        )

        metrics = trajectory_metrics(
            adapter_label="bad-1000",
            point=point,
            adapter_prediction=adapter_prediction,
            normal_adapter_prediction=torch.tensor([1.5, 0.0]),
        )

        self.assertEqual(metrics["cos_anyflow_normal_target"], -1.0)
        self.assertEqual(metrics["anyflow_normal_target_norm_ratio"], 2.0)
        self.assertEqual(metrics["adapter_residual_norm"], 2.0)
        self.assertEqual(metrics["adapter_residual_base_norm_ratio"], 4.0)
        self.assertEqual(metrics["normal_adapter_residual_norm"], 1.0)
        self.assertEqual(metrics["interval"], 0.25)

    def test_sample_vector_is_deterministic_and_bounded(self):
        tensor = torch.arange(100, dtype=torch.float32)

        first = sample_vector(tensor, 10)
        second = sample_vector(tensor, 10)

        self.assertEqual(first.shape, (10,))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first[-1], 99.0)

    def test_pca_coordinates_returns_two_finite_axes(self):
        coordinates = pca_coordinates(
            [
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
            ]
        )

        self.assertEqual(coordinates.shape, (3, 2))
        self.assertTrue(all(math.isfinite(value) for value in coordinates.reshape(-1)))


if __name__ == "__main__":
    unittest.main()
