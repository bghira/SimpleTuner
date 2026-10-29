"""Directed numerical and launch-grid regressions for the vendored CuTe kernels."""

import math
import unittest
from unittest.mock import patch

import torch

from scripts.benchmark_attention_precision import fp64_reference, metrics
from scripts.benchmark_attention_regressions import assess, cases, logits_statistics, make_case


def run_attention(inputs, dout, *, mask=None, causal=False, scale=None):
    from simpletuner.helpers.training.kohaku_fa_cute import attention

    variables = [value.detach().requires_grad_() for value in inputs]
    out = attention(*variables, mask=mask, causal=causal, scale=scale)
    return out, *torch.autograd.grad(out, variables, dout)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class TrackerRegressionTests(unittest.TestCase):
    run_attention = staticmethod(run_attention)

    def assert_reference(self, values, reference):
        for name, actual, expected in zip(("out", "dq", "dk", "dv"), values, reference):
            for region in (slice(None), slice(0, 1), slice(-1, None)):
                got, want = actual[..., region, :], expected[..., region, :]
                result = metrics(got, want)
                self.assertTrue(result["finite"], (name, region, result))
                if result["reference_norm"]:
                    self.assertIsNotNone(result["relative_error"])
                    self.assertLess(
                        result["relative_error"], 0.03 if actual.dtype == torch.bfloat16 else 0.01, (name, region, result)
                    )
                else:
                    self.assertEqual(got.count_nonzero().item(), 0)

    def test_live_kv_views_ignore_capacity_tail(self):
        torch.manual_seed(1567)
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.randn(1, 4, 96, 64, device="cuda", dtype=dtype)
            storage = [torch.randn(1, 4, 256, 64, device="cuda", dtype=dtype) for _ in range(2)]
            clean = [value.clone() for value in storage]
            dout = torch.randn_like(q)
            # Exercise the same cache in both length orders and then restore.
            for length in (256, 96, 192, 200, 255, 64, 256):
                for poison in (1234.0, float("nan")):
                    with self.subTest(dtype=dtype, length=length, poison=str(poison)):
                        for value, original in zip(storage, clean):
                            value.copy_(original)
                            value[..., length:, :] = poison
                        inputs = (q, *(value[..., :length, :] for value in storage))
                        reference = fp64_reference(inputs, dout)
                        actual = self.run_attention(inputs, dout)
                        self.assert_reference(actual, reference)
                        for value in storage:
                            tail = value[..., length:, :]
                            if math.isnan(poison):
                                self.assertTrue(torch.isnan(tail).all())
                            else:
                                self.assertTrue((tail == poison).all())

    def test_cached_strides_and_batch_changes(self):
        torch.manual_seed(1570)
        for dtype in (torch.float16, torch.bfloat16):
            for batch, length, layout in (
                (2, 256, "bhsd"),
                (2, 256, "bshd"),
                (2, 128, "bhsd"),
                (1, 256, "bhsd"),
                (2, 256, "bshd"),
                (2, 256, "bhsd"),
            ):
                with self.subTest(dtype=dtype, batch=batch, length=length, layout=layout):
                    inputs = [torch.randn(batch, 4, length, 64, device="cuda", dtype=dtype) for _ in range(3)]
                    if layout == "bshd":
                        inputs = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in inputs]
                    dout = torch.randn_like(inputs[0])
                    self.assert_reference(self.run_attention(inputs, dout), fp64_reference(inputs, dout))

    def test_nonpositive_scales_with_masks(self):
        torch.manual_seed(1428)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 2, 129, 128, device="cuda", dtype=dtype) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            mask = torch.arange(129, device="cuda")[None, :] < 97
            mask = mask.expand(129, -1).clone()
            mask[0] = False
            for scale in (0.0, -0.1):
                for causal in (False, True):
                    for masked in (False, True):
                        with self.subTest(dtype=dtype, scale=scale, causal=causal, masked=masked):
                            selected = mask if masked else None
                            actual = self.run_attention(inputs, dout, mask=selected, causal=causal, scale=scale)
                            reference = fp64_reference(inputs, dout, selected, causal, scale)
                            self.assert_reference(actual, reference)
                            if scale == 0:
                                self.assertEqual(actual[1].count_nonzero().item(), 0)
                                self.assertEqual(actual[2].count_nonzero().item(), 0)

    def test_single_head_single_query(self):
        torch.manual_seed(1071)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = [torch.randn(1, 1, length, 64, device="cuda", dtype=dtype) for length in (1, 128, 128)]
            dout = torch.randn_like(inputs[0])
            self.assert_reference(self.run_attention(inputs, dout), fp64_reference(inputs, dout))

    def test_single_query_gqa_statistics(self):
        from simpletuner.helpers.training.kohaku_fa_cute import native

        torch.manual_seed(1083)
        for heads, kvheads in ((8, 2), (32, 8)):
            inputs = [torch.randn(1, heads, 1, 128, device="cuda", dtype=torch.bfloat16)]
            inputs += [torch.randn(1, kvheads, 300, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            dout = torch.randn_like(inputs[0])
            expected = fp64_reference(inputs, dout)
            self.assert_reference(self.run_attention(inputs, dout), expected)
            scale = 128**-0.5
            native.extension().forward(*inputs, None, scale, False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, maximum, logsum, _ = native.extension().forward(*inputs, None, scale, False)
            for _ in range(3):
                maximum.fill_(float("nan"))
                logsum.fill_(float("nan"))
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(maximum).all())
                self.assertTrue(torch.isfinite(logsum).all())
                keys = inputs[1].double().repeat_interleave(heads // kvheads, dim=1)
                scores = inputs[0].double() @ keys.transpose(-1, -2)
                recovered = maximum.double() * scale + logsum.double() * math.log(2)
                torch.testing.assert_close(recovered, (scores * scale).logsumexp(-1), atol=1e-3, rtol=1e-5)
                torch.testing.assert_close(out.double(), expected[0], atol=0.004, rtol=0.03)

    def test_zero_stride_output_gradient(self):
        torch.manual_seed(100)
        inputs = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        expanded = torch.tensor(0.25, device="cuda", dtype=torch.bfloat16).expand_as(inputs[0])
        self.assertEqual(expanded.stride(), (0, 0, 0, 0))
        reference = fp64_reference(inputs, expanded)
        self.assert_reference(self.run_attention(inputs, expanded), reference)
        self.assert_reference(self.run_attention(inputs, expanded.contiguous()), reference)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class LaunchLimitTests(unittest.TestCase):
    def run_attention(self, inputs, dout, *, mask=None):
        return run_attention(inputs, dout, mask=mask)

    def test_grid_limits(self):
        # One visible key gives an exact oracle without a huge FP64 score matrix.
        shapes = (
            (1, 65534, 1, 64),
            (1, 65535, 1, 64),
            (1, 65536, 1, 64),
            (256, 256, 1, 64),
            (65536, 1, 1, 64),
            (1, 65536, 129, 8),
            (256, 256, 1, 257),
        )
        for batch, heads, queries, dimension in shapes:
            with self.subTest(batch=batch, heads=heads, queries=queries, dimension=dimension):
                inputs = [
                    torch.randn(batch, heads, length, dimension, device="cuda", dtype=torch.float16)
                    for length in (queries, 1, 1)
                ]
                dout = torch.ones_like(inputs[0])
                out, dq, dk, dv = self.run_attention(inputs, dout)
                torch.testing.assert_close(out, inputs[2].expand_as(out), atol=0, rtol=0)
                self.assertEqual(dq.count_nonzero().item(), 0)
                self.assertEqual(dk.count_nonzero().item(), 0)
                torch.testing.assert_close(dv, dout.sum(2, keepdim=True), atol=0, rtol=0)

        # Query and KV launches can lie on opposite sides of the grid limit.
        q = torch.randn(256, 256, 1, 64, device="cuda", dtype=torch.float16)
        k, v = [torch.randn(256, 1, 1, 64, device="cuda", dtype=torch.float16) for _ in range(2)]
        out, dq, dk, dv = self.run_attention((q, k, v), torch.ones_like(q))
        torch.testing.assert_close(out, v.expand_as(q), atol=0, rtol=0)
        self.assertEqual(dq.count_nonzero().item(), 0)
        self.assertEqual(dk.count_nonzero().item(), 0)
        torch.testing.assert_close(dv, torch.full_like(v, 256), atol=0, rtol=0)

    def test_masked_grid_limits(self):
        batch, heads = 256, 256
        inputs = [torch.randn(batch, heads, 1, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        mask = (torch.arange(batch * heads, device="cuda").reshape(batch, heads, 1, 1) % 2) == 0
        dout = torch.ones_like(inputs[0])
        out, dq, dk, dv = self.run_attention(inputs, dout, mask=mask)
        torch.testing.assert_close(out, inputs[2].masked_fill(~mask, 0), atol=0, rtol=0)
        self.assertEqual(dq.count_nonzero().item(), 0)
        self.assertEqual(dk.count_nonzero().item(), 0)
        torch.testing.assert_close(dv, dout.masked_fill(~mask, 0), atol=0, rtol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(10, 0), (10, 3)},
    "Requires datacentre Blackwell",
)
class BlackwellLaunchLimitTests(unittest.TestCase):
    def check_singleton_key(self, dimension, configuration, *, wide_aux=False):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, blackwell_register_aux, kernels, native

        torch.manual_seed(1573)
        q, k, v = [torch.randn(256, 256, 1, dimension, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        dout = torch.ones_like(q)
        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            with patch.object(kernels, "kernel_configuration", configuration):
                if wide_aux:
                    # Exercise the wide worker without allocating S=512 at every head.
                    def worker(causal, fallback, base_vector):
                        return blackwell_register_aux.JointBackward(causal=causal, base_vector=base_vector)

                    with patch.object(blackwell_register_aux, "AdaptiveBackward", worker):
                        inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                        out = attention(*inputs)
                        gradients = torch.autograd.grad(out, inputs, dout)
                else:
                    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                    out = attention(*inputs)
                    gradients = torch.autograd.grad(out, inputs, dout)
            torch.testing.assert_close(out, v, atol=0, rtol=0)
            self.assertEqual(gradients[0].count_nonzero().item(), 0)
            self.assertEqual(gradients[1].count_nonzero().item(), 0)
            torch.testing.assert_close(gradients[2], dout, atol=0, rtol=0)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_native_grid(self):
        for dimension in (64, 512):

            def configuration(device, mode, descriptors):
                selected = (32, min(dimension, 256), 4, False, mode != "forward", True, False, False, 64)
                return (*selected, True) if mode != "forward" and dimension == 512 else selected

            with self.subTest(dimension=dimension):
                self.check_singleton_key(dimension, configuration)

    def test_pipeline_grid(self):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        for dimension in (64, 512):

            def configuration(device, mode, descriptors):
                return kernels._SM100_FUSED_CONFIGURATIONS[(dimension, mode)]

            with self.subTest(dimension=dimension):
                self.check_singleton_key(dimension, configuration)

    def test_register_aux_grid(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("The adaptive wide worker is selected only on SM103")
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        def configuration(device, mode, descriptors):
            return kernels._SM103_FUSED_CONFIGURATIONS[(512, mode)]

        self.check_singleton_key(512, configuration, wide_aux=True)

    def test_split_forward_grid(self):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        # Explicit scratch selects the split forward even though normal dispatch
        # reserves it for low-occupancy Ada/Hopper shapes.
        q = torch.zeros(1, 65536, 3, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(1, 65536, 1, 64, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.full_like(q, float("nan"))
        maximum = torch.full(q.shape[:-1], float("nan"), device=q.device, dtype=torch.float32)
        logsum = maximum.clone()
        partial = torch.empty(*q.shape[:-1], 2, 64, device=q.device, dtype=torch.float32)
        stats = torch.empty(*q.shape[:-1], 2, 2, device=q.device, dtype=torch.float32)
        mask = torch.empty(1, device=q.device, dtype=torch.int32)
        blocks = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.int32)
        tensors = (q, k, v, mask, out, maximum, logsum, partial, stats, out, out, out, blocks, maximum)
        torch.cuda.synchronize()
        function = kernels.compile_kernel(q.device.index, "forward", False, False, tuple(map(kernels.descriptor, tensors)))
        function(*tensors, 0.125)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, v.expand_as(q), atol=0, rtol=0)
        torch.testing.assert_close(maximum, torch.zeros_like(maximum), atol=0, rtol=0)
        torch.testing.assert_close(logsum, torch.zeros_like(logsum), atol=0, rtol=0)


class AdversarialFixtureTests(unittest.TestCase):
    def test_negative_statistics_and_tiny_reported_gradient(self):
        for case in cases(extended=False):
            with self.subTest(case=case.name):
                inputs, dout, mask = make_case(case, device="cpu", dtype=torch.bfloat16)
                stats = logits_statistics(inputs, mask, scale=case.scale)
                if case.name != "zero_shift_control":
                    self.assertLess(stats["lse_min"], -88.7)
                else:
                    self.assertGreater(stats["lse_min"], 0)
                expected = fp64_reference(inputs, dout, mask, scale=case.scale)
                self.assertTrue(all(torch.isfinite(value).all() for value in expected))
                if case.literal:
                    norm = metrics(expected[1], expected[1])["reference_norm"]
                    self.assertGreater(norm, 1e-12)
                    self.assertLess(norm, 1e-7)

    def test_constant_logit_shift_preserves_forward_and_gradients(self):
        # Inputs with an exactly representable, constant key component isolate
        # softmax shift invariance from input-rounding and cancellation effects.
        q = torch.tensor([[[[8.0, 0.25], [8.0, -0.5]]]], dtype=torch.float64)
        k = torch.tensor([[[[0.0, 1.0], [0.0, -1.0], [0.0, 0.5]]]], dtype=torch.float64)
        v = torch.tensor([[[[1.0, 0.0], [-1.0, 0.5], [0.0, -0.25]]]], dtype=torch.float64)
        dout = torch.ones_like(q)
        expected = fp64_reference((q, k, v), dout, scale=0.125)
        shifted = k.clone()
        shifted[..., 0] = -128
        for actual, want in zip(fp64_reference((q, shifted, v), dout, scale=0.125), expected):
            torch.testing.assert_close(actual, want, atol=2e-13, rtol=2e-13)

    def test_comparator_rejects_lost_tiny_gradients_and_lost_tail_rows(self):
        reference = torch.full((1, 1, 2, 8), 1e-22, dtype=torch.float64)
        actual = torch.zeros_like(reference)
        result = assess((actual,) * 4, (reference,) * 4, torch.bfloat16)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["tensors"]["dq"]["relative_error"], 1)
        reference = torch.ones(1, 1, 10001, 1, dtype=torch.float64)
        actual = reference.clone()
        actual[..., -1, :] = 0
        result = assess((actual,) * 4, (reference,) * 4, torch.bfloat16)
        self.assertLess(result["tensors"]["dq"]["relative_error"], 0.05)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertFalse(result["tensors"]["dq"]["edge_rows"]["last"]["passed"])


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)},
    "Requires supported CuTe CUDA kernels",
)
class AdversarialKernelTests(unittest.TestCase):
    def assert_reference(self, actual, expected, dtype, case):
        for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
            for region in (slice(None), slice(0, 1), slice(-1, None)):
                value, reference = got[..., region, :], want[..., region, :]
                result = metrics(value, reference)
                self.assertTrue(result["finite"], (case, name, region, result))
                if result["reference_norm"]:
                    self.assertLess(
                        result["relative_error"], 0.05 if dtype == torch.bfloat16 else 0.015, (case, name, region, result)
                    )
                else:
                    self.assertEqual(value.count_nonzero().item(), 0)

    def test_reported_cudnn_inputs_and_negative_lse_tails(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dtype in (torch.bfloat16, torch.float16):
            for case in cases(extended=False):
                with self.subTest(dtype=dtype, case=case.name):
                    inputs, dout, mask = make_case(case, device="cuda", dtype=dtype)
                    expected = fp64_reference(inputs, dout, mask, scale=case.scale)
                    variables = [value.detach().requires_grad_() for value in inputs]
                    output = attention(*variables, mask=mask, scale=case.scale)
                    actual = output, *torch.autograd.grad(output, variables, dout)
                    self.assert_reference(actual, expected, dtype, case.name)

    def test_partial_tiles_preserve_negative_unscaled_maxima(self):
        import math

        from simpletuner.helpers.training.kohaku_fa_cute import native

        selected = {"issue1072", "kv_tail_64", "kv_tail_192"}
        for dtype in (torch.bfloat16, torch.float16):
            for case in (case for case in cases(extended=False) if case.name in selected):
                with self.subTest(dtype=dtype, case=case.name):
                    inputs, _, _ = make_case(case, device="cuda", dtype=dtype)
                    scale = case.dim**-0.5
                    _, maximum, logsum, _ = native.extension().forward(*inputs, None, scale, False)
                    scores = inputs[0].double() @ inputs[1].double().transpose(-1, -2)
                    self.assertLess(maximum.max().item(), 0)
                    self.assertTrue(torch.isfinite(logsum).all())
                    torch.testing.assert_close(maximum.double(), scores.amax(-1), atol=1e-3, rtol=1e-6)
                    reconstructed = maximum.double() * scale + logsum.double() * math.log(2)
                    torch.testing.assert_close(reconstructed, (scores * scale).logsumexp(-1), atol=1e-3, rtol=1e-6)

    def test_compiled_and_graph_replay_read_changed_negative_lse_inputs(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        dtype = torch.bfloat16
        for case in (case for case in cases(extended=False) if case.name in {"issue1072", "kv_tail_192"}):
            with self.subTest(case=case.name):
                inputs, dout, _ = make_case(case, device="cuda", dtype=dtype)
                variables = [value.detach().requires_grad_() for value in inputs]

                def run(operation=attention):
                    output = operation(*variables)
                    return output, *torch.autograd.grad(output, variables, dout)

                compiled = torch.compile(attention, fullgraph=True)
                expected = fp64_reference(variables, dout)
                self.assert_reference(run(compiled), expected, dtype, case.name)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                for seed in (124, 125):
                    changed, changed_dout, _ = make_case(case, device="cuda", dtype=dtype, seed=seed)
                    with torch.no_grad():
                        for target, source in zip((*variables, dout), (*changed, changed_dout)):
                            target.copy_(source)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assert_reference(actual, fp64_reference(variables, dout), dtype, case.name)
