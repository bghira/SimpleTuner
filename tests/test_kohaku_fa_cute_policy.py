"""Integration guards for the synchronized CuTe dispatch and precision contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.training.kohaku_fa_cute import _ops, api, fa4_sm80
from simpletuner.helpers.training.kohaku_fa_cute.determinism import check_atomic_backward


class CuTePolicyTests(unittest.TestCase):
    def test_atomic_backward_checks_live_strict_and_warning_policies(self):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            check_atomic_backward()
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(RuntimeError, "atomic_backward"):
                check_atomic_backward()
            torch.use_deterministic_algorithms(True, warn_only=True)
            with self.assertWarnsRegex(UserWarning, "atomic_backward"):
                check_atomic_backward()
            torch.use_deterministic_algorithms(False)
            check_atomic_backward()
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)

    def test_atomic_guards_precede_optional_imports_and_tensor_access(self):
        from simpletuner.helpers.training.kohaku_fa_cute.fa4_hopper._attention import _Attention
        from simpletuner.helpers.training.kohaku_fa_cute.fa4_sm80._attention import _backward

        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True):
            for operation in (lambda: _backward(*([None] * 11)), lambda: _Attention.backward(None, None)):
                with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "atomic_backward"):
                    operation()

    def test_optional_schedule_requires_the_entire_qualified_stack(self):
        qualified = {
            "flash-attn-4": "0.0.0",
            "nvidia-cutlass-dsl": "4.8.1",
            "apache-tvm-ffi": "0.1.15.post0",
            "quack-kernels": "0.6.5",
            "torch-c-dlpack-ext": "0.1.5",
        }
        from simpletuner.helpers.training.kohaku_fa_cute.fa4_sm80 import _private_api

        try:
            with (
                patch.object(fa4_sm80.sys, "version_info", (3, 13)),
                patch.object(torch, "__version__", "2.15.0+cu132"),
                patch.object(torch.version, "cuda", "13.2"),
                patch.object(_private_api, "interface_matches", return_value=True) as sources,
            ):
                for package in qualified:
                    versions = qualified | {package: "unqualified"}
                    fa4_sm80.available.cache_clear()
                    with patch.object(fa4_sm80, "version", side_effect=versions.__getitem__):
                        if package == "flash-attn-4":
                            self.assertTrue(fa4_sm80.available())
                        else:
                            self.assertFalse(fa4_sm80.available(), package)
                fa4_sm80.available.cache_clear()
                with patch.object(fa4_sm80, "version", side_effect=qualified.__getitem__):
                    self.assertTrue(fa4_sm80.available())
                    sources.assert_called()
                    fa4_sm80.available.cache_clear()
                    sources.return_value = False
                    self.assertFalse(fa4_sm80.available())
        finally:
            fa4_sm80.available.cache_clear()

    def test_selected_optional_paths_and_their_errors_reach_all_public_apis(self):
        q = SimpleNamespace(ndim=4, shape=(1, 8, 4096, 128), device=torch.device("cuda", 0))
        for operation in (
            api.attention,
            api.kohaku_fa_scaled_dot_product_attention,
            api.automatic_kohaku_fa_scaled_dot_product_attention,
        ):
            with (
                self.subTest(operation=operation.__name__),
                patch.object(api, "_validate"),
                patch.object(api, "_try_rtx4090_s1024", return_value=None),
                patch.object(fa4_sm80, "compatible", return_value=True),
                patch.object(fa4_sm80, "attention", return_value="selected") as selected,
                patch.object(api, "extension") as extension,
            ):
                extension.cache_info.return_value = SimpleNamespace(currsize=0)
                self.assertEqual(operation(q, q, q), "selected")
                extension.assert_not_called()
                selected.side_effect = RuntimeError("selected kernel failed")
                with self.assertRaisesRegex(RuntimeError, "selected kernel failed"):
                    operation(q, q, q)

    def test_fake_outputs_preserve_stats_mask_tiles_and_grouped_head_shapes(self):
        q = torch.empty(2, 6, 129, 128, dtype=torch.bfloat16)
        k = torch.empty(2, 2, 257, 128, dtype=q.dtype)
        mask = torch.empty(129, 257, dtype=torch.bool).expand(2, 6, -1, -1)
        output, maximum, logsum, blocks = _ops.forward_fake(q, k, k, mask, 0.125, False)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.dtype, q.dtype)
        self.assertEqual(maximum.shape, q.shape[:-1])
        self.assertEqual(logsum.dtype, torch.float32)
        self.assertEqual(blocks.shape, (1, 1, 3, 5))

    def test_compiled_cpu_automatic_path_preserves_outputs_and_gradients(self):
        q = torch.randn(1, 2, 5, 16, requires_grad=True)
        k = torch.randn(1, 2, 7, 16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=-0.2)
        actual = torch.compile(api.automatic_kohaku_fa_scaled_dot_product_attention, backend="eager", fullgraph=True)(
            q, k, v, scale=-0.2
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for got, want in zip(torch.autograd.grad(actual.sum(), (q, k, v)), torch.autograd.grad(reference.sum(), (q, k, v))):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


class CuTeRoundingReferenceTests(unittest.TestCase):
    def test_single_visible_causal_row_has_exact_zero_query_gradient(self):
        from scripts.benchmark_attention_precision import rounded_reference

        torch.manual_seed(1428)
        inputs = [torch.randn(1, 2, 7, 64, dtype=torch.float16) for _ in range(3)]
        expected = rounded_reference(inputs, torch.randn_like(inputs[0]), causal=True, scale=-0.1)
        self.assertEqual(expected[1][..., 0, :].count_nonzero().item(), 0)

    def test_fp16_probability_underflow_remains_part_of_the_operand_contract(self):
        from scripts.benchmark_attention_precision import fp64_reference, rounded_reference

        q = torch.ones(1, 1, 1, 1, dtype=torch.float16)
        k = torch.tensor([[[[0.0], [-20.0]]]], dtype=q.dtype)
        v = torch.zeros_like(k)
        dout = torch.full_like(q, 10000)
        mathematical = fp64_reference((q, k, v), dout, scale=1.0)
        rounded = rounded_reference((q, k, v), dout, scale=1.0)
        self.assertGreater(mathematical[3][0, 0, 1, 0].item(), 1e-5)
        self.assertEqual(rounded[3][0, 0, 1, 0].item(), 0)

    def test_saturated_probability_rounds_before_derivative_products(self):
        from scripts.benchmark_attention_precision import rounded_reference

        q = torch.tensor([[[[10.0]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[10.0], [5.0]]]], dtype=q.dtype)
        v = torch.tensor([[[[1.0], [-1.0]]]], dtype=q.dtype)
        _, _, dk, dv = rounded_reference((q, k, v), torch.ones_like(q), scale=1.0)
        self.assertEqual(dk[0, 0, 0, 0].item(), 0)
        self.assertGreater(dk[0, 0, 1, 0].abs().item(), 0)
        self.assertEqual(dv[0, 0, 0, 0].item(), 1)


@unittest.skipUnless(torch.cuda.is_available(), "Requires the CuTe CUDA kernels")
class CuTeRoundingKernelTests(unittest.TestCase):
    def test_directed_operand_rounding_matches_the_saved_output_contract(self):
        from scripts.benchmark_attention_precision import rounded_reference

        for dtype in (torch.float16, torch.bfloat16):
            for saturated in (False, True):
                with self.subTest(dtype=dtype, saturated=saturated):
                    q = torch.zeros(1, 1, 1, 64, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 1, 2, 64, device="cuda", dtype=dtype)
                    v = torch.zeros_like(k)
                    if saturated:
                        q[..., 0], k[..., 0, 0], k[..., 1, 0] = 10, 10, 5
                        v[..., 0, 0], v[..., 1, 0] = 1, -1
                        dout = torch.ones_like(q)
                    else:
                        q[..., 0], k[..., 1, 0] = 1, -20
                        dout = torch.full_like(q, 10000)
                    inputs = [value.requires_grad_() for value in (q, k, v)]
                    output = api.attention(*inputs, scale=1.0)
                    gradients = torch.autograd.grad(output, inputs, dout)
                    expected = rounded_reference(inputs, dout, scale=1.0, output=output)
                    for got, want in zip((output, *gradients), expected):
                        torch.testing.assert_close(got.double(), want, rtol=0.01, atol=1e-30)

    def test_negative_strided_views_preserve_logical_values_and_base_gradients(self):
        from scripts.benchmark_attention_precision import fp64_reference, metrics

        torch.manual_seed(42013)
        for dtype in (torch.float16, torch.bfloat16):
            for operation in (api.attention, api.kohaku_fa_scaled_dot_product_attention):
                with self.subTest(dtype=dtype, operation=operation.__name__):
                    bases = [torch.randn(1, 2, 17, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
                    inputs = [torch.ops.aten._neg_view.default(value[..., ::2]) for value in bases]
                    dout = torch.ops.aten._neg_view.default(torch.randn_like(inputs[0]))
                    output = operation(*inputs)
                    gradients = torch.autograd.grad(output, bases, dout)
                    reference = fp64_reference(inputs, dout)
                    expected = [reference[0]]
                    for base, logical in zip(bases, reference[1:]):
                        full = torch.zeros_like(base, dtype=logical.dtype)
                        full[..., ::2] = -logical
                        expected.append(full)
                    for got, want in zip((output, *gradients), expected):
                        error = metrics(got, want)
                        self.assertTrue(error["finite"])
                        self.assertLess(error["relative_error"], 0.01 if dtype == torch.bfloat16 else 0.005)


if __name__ == "__main__":
    unittest.main()
