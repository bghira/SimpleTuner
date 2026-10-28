import itertools
import unittest
from unittest.mock import patch

import torch

from scripts.benchmark_attention_precision import fp64_reference, make_inputs
from tests import test_kohaku_fa_cute as legacy_tests


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Requires Hopper SM90",
)
class ClefHopperForwardTests(unittest.TestCase):
    def test_pipeline_matches_serial_statistics_outputs_and_gradients(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, hopper_forward, kernels, native

        def serial(mode, causal, masked, *configuration, vector):
            return kernels.Attention(mode, causal, masked, 128, *configuration[1:], vector=vector)

        def evaluate(inputs, dout, scale):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            q, k, v = [tensor.detach().requires_grad_(True) for tensor in inputs]
            out = attention(q, k, v, causal=True, scale=scale)
            gradients = torch.autograd.grad(out, (q, k, v), dout)
            statistics = native.extension().forward(q, k, v, None, scale, True)[1:3]
            return out, *gradients, *statistics

        cases = ((1, 1, 257), (1, 63, 63), (1, 127, 127), (2, 129, 129), (1, 257, 257), (1, 129, 257), (1, 257, 129))
        try:
            for batch, queries, keys in cases:
                for scale in (0.0625, -0.125, 0.0):
                    with self.subTest(batch=batch, queries=queries, keys=keys, scale=scale):
                        torch.manual_seed(509)
                        q = torch.randn(batch, 24, queries, 256, device="cuda", dtype=torch.bfloat16)
                        k = torch.randn(batch, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                        v = torch.randn(batch, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                        dout = torch.randn_like(q)
                        with patch.object(hopper_forward, "HopperForwardSpecialized", serial):
                            expected = evaluate((q, k, v), dout, scale)
                        actual = evaluate((q, k, v), dout, scale)
                        # The denominator now reduces across lanes in the
                        # epilogue. Its FP32 reassociation can change BF16 ties.
                        for got, want in zip(actual[:4], expected[:4]):
                            torch.testing.assert_close(got, want, atol=0.02, rtol=0.02)
                        self.assertTrue(torch.equal(actual[4], expected[4]))
                        torch.testing.assert_close(actual[5], expected[5], atol=1e-5, rtol=1e-5)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_infinite_values_preserve_serial_output(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, hopper_forward, kernels, native

        def serial(mode, causal, masked, *configuration, vector):
            return kernels.Attention(mode, causal, masked, 128, *configuration[1:], vector=vector)

        def evaluate(q, k, v):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            return attention(q, k, v, causal=True)

        try:
            for queries, keys in ((1, 1), (129, 3)):
                with self.subTest(queries=queries, keys=keys):
                    q = torch.zeros(1, 24, queries, 256, device="cuda", dtype=torch.bfloat16)
                    k = torch.zeros(1, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                    v = torch.zeros(1, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                    v[:, :, 0, 0] = float("inf")
                    v[:, :, 0, 1] = -float("inf")
                    v[:, :, 0, 2] = float("nan")
                    v[:, :, 0, 3] = 1
                    with patch.object(hopper_forward, "HopperForwardSpecialized", serial):
                        expected = evaluate(q, k, v)
                    actual = evaluate(q, k, v)
                    self.assertTrue(torch.isposinf(expected[..., 0]).all().item())
                    self.assertTrue(torch.isneginf(expected[..., 1]).all().item())
                    self.assertTrue(torch.isnan(expected[..., 2]).all().item())
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_causal_gqa_matches_fp64(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for scale in (0.0625, 0.125, -0.125, 0.0, 1e-30, -1e-30):
            with self.subTest(scale=scale):
                torch.manual_seed(510)
                q = torch.randn(1, 24, 137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(1, 4, 193, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn(1, 193, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout, causal=True, scale=scale)
                out = attention(q, k, v, causal=True, scale=scale)
                actual = out, *torch.autograd.grad(out, (q, k, v), dout)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_causal_gqa_graph_reads_changed_inputs(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            q = torch.randn(1, 24, 257, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 4, 269, 256, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 269, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            attention(q, k, v, causal=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = attention(q, k, v, causal=True)
            for seed in (511, 512):
                torch.manual_seed(seed)
                for tensor in (q, k, v):
                    tensor.copy_(torch.randn_like(tensor))
                graph.replay()
                torch.cuda.synchronize()
                eager = attention(q, k, v, causal=True)
                self.assertTrue(torch.equal(actual, eager))
                expected = fp64_reference((q, k, v), torch.zeros_like(q), causal=True, scale=0.0625)[0]
                torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.02)

        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "Requires Ada SM89",
)
class ClefForwardOverlapTests(unittest.TestCase):
    def test_causal_gqa_tails_strides_statistics_and_gradients_match_serial(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, kernels, native

        original = kernels.Attention.__init__

        def serial(program, *args, **kwargs):
            kwargs["forward_overlap"] = False
            original(program, *args, **kwargs)

        selected = []

        def observe(program, *args, **kwargs):
            selected.append(kwargs.get("forward_overlap", False))
            original(program, *args, **kwargs)

        def evaluate(inputs, dout, scale):
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()
            q, k, v = [t.detach().requires_grad_(True) for t in inputs]
            out = attention(q, k, v, causal=True, scale=scale)
            gradients = torch.autograd.grad(out, (q, k, v), dout)
            statistics = native.extension().forward(q, k, v, None, scale, True)[1:3]
            return out, *gradients, *statistics

        cases = ((1, 127, 127), (1, 128, 128), (2, 129, 129), (1, 257, 257), (1, 129, 257), (1, 257, 129))
        try:
            for batch, queries, keys in cases:
                for scale in (0.0625, -0.125, 0.0):
                    with self.subTest(batch=batch, queries=queries, keys=keys, scale=scale):
                        torch.manual_seed(419)
                        q = torch.randn(batch, 16, queries, 256, device="cuda", dtype=torch.bfloat16)
                        k = torch.randn(batch, 4, keys, 256, device="cuda", dtype=torch.bfloat16)
                        v = torch.randn(batch, keys, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                        dout = torch.randn_like(q)
                        with patch.object(kernels.Attention, "__init__", serial):
                            expected = evaluate((q, k, v), dout, scale)
                        selected.clear()
                        with patch.object(kernels.Attention, "__init__", observe):
                            actual = evaluate((q, k, v), dout, scale)
                        self.assertIn(True, selected)
                        for got, want in zip(actual, expected):
                            self.assertTrue(torch.equal(got, want))
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()

    def test_causal_gqa_matches_fp64(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        torch.manual_seed(422)
        q = torch.randn(1, 16, 137, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 4, 193, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 193, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        dout = torch.randn_like(q)
        expected = fp64_reference((q, k, v), dout, causal=True, scale=-0.125)
        out = attention(q, k, v, causal=True, scale=-0.125)
        actual = out, *torch.autograd.grad(out, (q, k, v), dout)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_causal_gqa_graph_reads_changed_inputs(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention, kernels, native

        original = kernels.Attention.__init__

        def serial(program, *args, **kwargs):
            kwargs["forward_overlap"] = False
            original(program, *args, **kwargs)

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        try:
            q = torch.randn(1, 16, 257, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 4, 269, 256, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 269, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
            attention(q, k, v, causal=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = attention(q, k, v, causal=True)
            for seed in (420, 421):
                torch.manual_seed(seed)
                for tensor in (q, k, v):
                    tensor.copy_(torch.randn_like(tensor))
                graph.replay()
                torch.cuda.synchronize()
                with patch.object(kernels.Attention, "__init__", serial):
                    native.extension().clear_cache()
                    kernels._compile_kernel.cache_clear()
                    expected = attention(q, k, v, causal=True)
                self.assertTrue(torch.equal(actual, expected))
        finally:
            native.extension().clear_cache()
            kernels._compile_kernel.cache_clear()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(8, 9), (9, 0)},
    "Requires Ada or Hopper short-query forward kernels",
)
class ShortQueryForwardTests(unittest.TestCase):
    compare = legacy_tests.CuTeAttentionTests.compare

    def test_sm89_split_forward_compiled_dynamic_query_boundaries(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        function = torch.compile(attention, fullgraph=True, dynamic=True)
        for queries in (31, 33, 65, 257, 12):
            with self.subTest(queries=queries):
                torch.manual_seed(817)
                q = torch.randn(1, queries, 16, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                k, v = [torch.randn(1, 257, 16, 64, device="cuda", dtype=q.dtype).transpose(1, 2) for _ in range(2)]
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = function(*inputs)
                gradients = torch.autograd.grad(output, inputs, dout)
                for got, want in zip((output, *gradients), expected):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_sm89_short_query_split_forward_matches_fp64_and_saved_statistics(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from simpletuner.helpers.training.kohaku_fa_cute import native

        extension = native.extension()
        extension.clear_cache()
        try:
            for dtype, queries, keys, scale in itertools.product(
                (torch.float16, torch.bfloat16), (1, 3, 12, 32, 64, 128, 129, 256), (256, 787), (0.125, -0.125, 0.0)
            ):
                with self.subTest(dtype=dtype, queries=queries, keys=keys, scale=scale):
                    torch.manual_seed(813)
                    q = torch.randn(1, queries, 16, 64, device="cuda", dtype=dtype).transpose(1, 2)
                    k, v = [torch.randn(1, keys, 16, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)]
                    dout = torch.randn_like(q)

                    def run():
                        out, maximum, logsum, blocks = extension.forward(q, k, v, None, scale, False)
                        gradients = extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, scale, False)
                        return out, *gradients

                    run()
                    run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for seed in (814, 815):
                        torch.manual_seed(seed)
                        for tensor in (q, k, v, dout):
                            tensor.copy_(torch.randn_like(tensor))
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = fp64_reference((q, k, v), dout, scale=scale)
                        tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
        finally:
            extension.clear_cache()

    def serial_forward_backward(self, inputs, dout, scale=None):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels, native

        q, k, v = inputs
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        out = torch.empty_like(q)
        maximum = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        logsum = torch.empty_like(maximum)
        mask = torch.empty(1, device=q.device, dtype=torch.int32)
        blocks = torch.empty(1, 1, 1, 1, device=q.device, dtype=torch.int32)
        tensors = (q, k, v, mask, out, maximum, logsum, out, maximum, out, out, out, blocks, maximum)
        function = kernels.compile_kernel(q.device.index, "forward", False, False, tuple(map(kernels.descriptor, tensors)))
        function(*tensors, scale)
        gradients = native.extension().backward(q, k, v, None, maximum, logsum, dout, blocks, out, scale, False)
        return out, *gradients

    def test_sm89_short_query_split_forward_saturated_and_offset_logits(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from simpletuner.helpers.training.kohaku_fa_cute import attention, native

        native.extension().clear_cache()
        try:
            for dtype, logit_scale, offset in itertools.product(
                (torch.float16, torch.bfloat16), (10, 1000, 1e5, 1e7), (0, 100)
            ):
                with self.subTest(dtype=dtype, logit_scale=logit_scale, offset=offset):
                    q, k, v, dout = make_inputs(logit_scale, device="cuda", dtype=dtype, seq=787, dim=64, seed=0, heads=16)
                    q, dout = q[:, :, :3].contiguous(), dout[:, :, :3].contiguous()
                    k = (k.float() + offset).to(dtype)
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs)
                    actual = (output, *torch.autograd.grad(output, inputs, dout))
                    expected = fp64_reference(inputs, dout)
                    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                    torch.testing.assert_close(actual[0].double(), expected[0], atol=tolerance, rtol=tolerance)
                    # Compare serial gradients to isolate the split forward from existing high-logit QK rounding.
                    serial = self.serial_forward_backward(inputs, dout)
                    for got, want in zip(actual[1:], serial[1:]):
                        torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
            for dtype in (torch.float16, torch.bfloat16):
                q, k, v, dout = make_inputs(1e5, device="cuda", dtype=dtype, seq=787, dim=64, seed=0, heads=16)
                q, dout = q[:, :, :3].contiguous(), torch.ones_like(dout[:, :, :3])
                v.fill_(1)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = attention(*inputs)
                dq, dk, _ = torch.autograd.grad(output, inputs, dout)
                self.assertEqual(dq.count_nonzero().item(), 0)
                self.assertEqual(dk.count_nonzero().item(), 0)
            for dtype, gap in itertools.product((torch.float16, torch.bfloat16), (1, 5)):
                with self.subTest(dtype=dtype, saturated_gap=gap):
                    q = torch.zeros(1, 16, 3, 64, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 16, 787, 64, device="cuda", dtype=dtype)
                    v, dout = torch.zeros_like(k), torch.zeros_like(q)
                    q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 10 - gap, 10
                    v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs, scale=1.0)
                    actual = (output, *torch.autograd.grad(output, inputs, dout))
                    expected = fp64_reference(inputs, dout, scale=1.0)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.double(), want, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.02)
        finally:
            native.extension().clear_cache()

    def test_sm89_short_query_split_forward_fallback_boundaries(self):
        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Requires Ada or Hopper short-query forward kernels.")
        from unittest.mock import patch

        from simpletuner.helpers.training.kohaku_fa_cute import native

        extension = native.extension()
        extension.clear_cache()
        compiler = patch.object(native, "compile_kernel", wraps=native.compile_kernel)
        compiled = compiler.start()
        try:
            # Keep the warm native cache valid while crossing each selection boundary.
            for queries, keys, heads, kvheads, mode in (
                (3, 255, 16, 16, "dense"),
                (3, 256, 16, 16, "dense"),
                (3, 257, 16, 16, "dense"),
                (31, 257, 16, 16, "dense"),
                (32, 257, 16, 16, "dense"),
                (33, 257, 16, 16, "dense"),
                (63, 257, 16, 16, "dense"),
                (64, 257, 16, 16, "dense"),
                (65, 257, 16, 16, "dense"),
                (127, 257, 16, 16, "dense"),
                (128, 257, 16, 16, "dense"),
                (129, 257, 16, 16, "dense"),
                (255, 257, 16, 16, "dense"),
                (256, 257, 16, 16, "dense"),
                (257, 257, 16, 16, "dense"),
                (3, 16384, 16, 16, "dense"),
                (3, 16385, 16, 16, "dense"),
                (3, 257, 16, 4, "dense"),
                (3, 257, 2, 2, "dense"),
                (3, 257, 16, 16, "causal"),
                (3, 257, 16, 16, "masked"),
                (3, 257, 16, 16, "strided"),
                (3, 256, 16, 16, "dense"),
            ):
                with self.subTest(queries=queries, keys=keys, heads=heads, kvheads=kvheads, mode=mode):
                    torch.manual_seed(816)
                    step = 2 if mode == "strided" else 1
                    q = torch.randn(1, queries, heads, 64 * step, device="cuda", dtype=torch.bfloat16)[
                        ..., ::step
                    ].transpose(1, 2)
                    k, v = [
                        torch.randn(1, keys, kvheads, 64 * step, device="cuda", dtype=q.dtype)[..., ::step].transpose(1, 2)
                        for _ in range(2)
                    ]
                    mask = torch.ones(1, 1, queries, keys, device="cuda", dtype=torch.bool) if mode == "masked" else None
                    if mask is not None:
                        mask[..., 0, :] = False
                    previous = compiled.call_count
                    self.compare(q, k, v, mask=mask, causal=mode == "causal")
                    forward_compiles = [
                        call.args for call in compiled.call_args_list[previous:] if call.args[0] == "forward"
                    ]
                    split = (
                        queries <= 256 and 256 <= keys <= 16384 and heads == kvheads == 16 and mode in ("dense", "strided")
                    )
                    for arguments in forward_compiles:
                        self.assertEqual(arguments[1][7].ndim == 5, split)
                    if queries == 3 and keys == 257 and heads == kvheads == 16 and mode == "dense":
                        self.assertEqual(len(forward_compiles), 1)
        finally:
            compiler.stop()
            extension.clear_cache()
