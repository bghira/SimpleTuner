import math
import unittest

import numpy as np
import torch

from simpletuner.diagnostics.h3_objective_geometry import (
    GeometryPoint,
    cosine_similarity,
    pca_coordinates,
    sample_vector,
    trajectory_metrics,
)


class H3ObjectiveGeometryTests(unittest.TestCase):
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
