import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.benchmark_attention_precision import metrics, row_metrics
from scripts.plot_attention_precision_rows import distribution


class RowMetricsTests(unittest.TestCase):
    def test_matches_individual_row_metrics_in_batch_head_sequence_order(self):
        torch.manual_seed(46)
        reference = torch.randn(2, 3, 5, 7, dtype=torch.float64)
        got = reference + 0.1 * torch.randn_like(reference)
        result = row_metrics(got, reference)
        self.assertEqual(result["shape"], [2, 3, 5])
        for index, (actual, expected) in enumerate(zip(got.reshape(-1, 7), reference.reshape(-1, 7))):
            global_metrics = metrics(actual, expected)
            for name in ("finite", "reference_norm", "relative_error", "log10_relative_error", "max_absolute_error"):
                self.assertAlmostEqual(result[name][index], global_metrics[name])

    def test_tiny_zero_and_nonfinite_rows_are_preserved_without_json_nan(self):
        reference = torch.tensor([[1e-250, -2e-250], [0, 0], [1, 2], [1, 2]], dtype=torch.float64)
        got = torch.tensor([[2e-250, -4e-250], [1, 0], [1, 2], [math.inf, 2]], dtype=torch.float64)
        result = row_metrics(got, reference)
        self.assertGreater(result["reference_norm"][0], 0)
        self.assertEqual(result["relative_error"], [1.0, None, 0.0, None])
        self.assertEqual(result["log10_relative_error"], [0.0, None, None, None])
        self.assertEqual(result["finite"], [True, True, True, False])
        json.dumps(result, allow_nan=False)
        values, zeros, undefined = distribution({"row_metrics": {"dq": result}}, "dq")
        self.assertEqual((values, zeros, undefined), ([0.0], 1, 2))

    def test_log_relative_error_survives_overflow_in_the_ratio(self):
        result = row_metrics(torch.tensor([[1e200]], dtype=torch.float64), torch.tensor([[1e-200]], dtype=torch.float64))
        self.assertIsNone(result["relative_error"][0])
        self.assertAlmostEqual(result["log10_relative_error"][0], 400)
        values, zeros, undefined = distribution({"row_metrics": {"dq": result}}, "dq")
        self.assertEqual((values, zeros, undefined), ([400.0], 0, 0))

    def test_underflowed_ratio_is_not_counted_as_exact_zero_error(self):
        result = row_metrics(
            torch.tensor([[1e200, 1e-200]], dtype=torch.float64),
            torch.tensor([[1e200, 0]], dtype=torch.float64),
        )
        self.assertEqual(result["relative_error"][0], 0)
        self.assertAlmostEqual(result["log10_relative_error"][0], -400)
        values, zeros, undefined = distribution({"row_metrics": {"dq": result}}, "dq")
        self.assertEqual((values, zeros, undefined), ([-400.0], 0, 0))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Requires optional plots dependency")
    def test_violin_plot_handles_constant_exact_and_undefined_rows(self):
        from scripts.plot_attention_precision_rows import render

        data = row_metrics(torch.tensor([[1.0], [2.0], [0.0]]), torch.tensor([[1.0], [1.0], [0.0]]))
        report = {
            "device": "CPU test fixture",
            "rows": [
                dict(
                    backend="math",
                    status="ok",
                    requested_scale=100,
                    max_absolute_logit=110,
                    row_metrics={name: data for name in ("dq", "dk", "dv")},
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.svg"
            render(report, output)
            self.assertIn("<svg", output.read_text())
