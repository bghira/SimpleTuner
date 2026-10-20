import unittest

import torch

from scripts.benchmark_attention_length_precision import sweep


class LengthPrecisionTests(unittest.TestCase):
    def test_cpu_sweep_matches_fp64_for_both_dtypes_and_visibility_modes(self):
        rows = sweep(
            lengths=[7, 8],
            dtypes=["fp16", "bf16"],
            modes=["dense", "causal", "padding"],
            backends=["math"],
            device="cpu",
            dim=8,
            logit_scale=10,
        )
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(dtype=row["dtype"], mode=row["mode"], length=row["sequence_length"]):
                self.assertEqual(row["status"], "ok", row.get("error"))
                for name in ("out", "dq", "dk", "dv"):
                    self.assertTrue(row["metrics"][name]["finite"])
                    self.assertLess(row["metrics"][name]["relative_error"], 0.02)

    def test_unavailable_backend_is_reported_without_losing_other_results(self):
        rows = sweep(
            lengths=[5],
            dtypes=["bf16"],
            modes=["dense"],
            backends=["kohaku-fa", "math"],
            device="cpu",
            dim=8,
            logit_scale=10,
        )
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("CUDA", rows[0]["error"])
        self.assertEqual(rows[1]["status"], "ok")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "Requires CUDA FlexAttention with FP16/BF16 support",
)
class FlexLengthPrecisionTests(unittest.TestCase):
    def test_compiled_flex_at_aligned_and_unaligned_lengths(self):
        rows = sweep(
            lengths=[128, 129],
            dtypes=["fp16", "bf16"],
            modes=["dense", "causal", "padding"],
            backends=["flex"],
            device="cuda",
            dim=64,
            logit_scale=10,
        )
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(dtype=row["dtype"], mode=row["mode"], length=row["sequence_length"]):
                self.assertEqual(row["status"], "ok", row.get("error"))
                for name in ("out", "dq", "dk", "dv"):
                    self.assertTrue(row["metrics"][name]["finite"])
                    self.assertLess(row["metrics"][name]["relative_error"], 0.04)
