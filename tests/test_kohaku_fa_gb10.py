"""GB10 wide-head and short-sequence attention regressions."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from tests import test_kohaku_fa_cute as legacy_tests


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1),
    "Requires GB10 SM121",
)
class GB10AttentionTests(unittest.TestCase):
    compare = legacy_tests.CuTeAttentionTests.compare

    def test_training_dispatch_matches_reference_in_eager_and_compiled_calls(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from scripts.benchmark_attention_regressions import assess
        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

        for mechanism, dimension, causal in itertools.product(("kohaku-fa", "kohaku-fa-auto"), (256, 512), (False, True)):
            with self.subTest(mechanism=mechanism, dimension=dimension, causal=causal):
                torch.manual_seed(129)
                q = torch.randn(1, 4, 65, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                k, v = [
                    torch.randn(1, 2, 127, dimension, device="cuda", dtype=q.dtype, requires_grad=True) for _ in range(2)
                ]
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout, causal=causal, scale=-0.125)
                try:
                    AttentionBackendController.apply(SimpleNamespace(attention_mechanism=mechanism), AttentionPhase.TRAIN)
                    self.assertEqual(AttentionBackendController.active_backend(), mechanism)
                    eager = torch.nn.functional.scaled_dot_product_attention
                    for function in (eager, torch.compile(eager, fullgraph=True)):
                        output = function(q, k, v, is_causal=causal, scale=-0.125, enable_gqa=True)
                        actual = output, *torch.autograd.grad(output, (q, k, v), dout)
                        result = assess(actual, expected, q.dtype)
                        self.assertEqual(result["status"], "pass", result)
                finally:
                    AttentionBackendController.restore_default()

    def test_sm121_wide_head_tails_masks_and_layouts(self):
        for dtype, mode in itertools.product(
            (torch.float16, torch.bfloat16), ("dense", "causal", "gqa", "mask", "padding", "projection", "strided")
        ):
            with self.subTest(dtype=dtype, mode=mode):
                torch.manual_seed(121)
                q = torch.randn(1, 4, 19, 512, device="cuda", dtype=dtype)
                k, v = [torch.randn(1, 2 if mode == "gqa" else 4, 37, 512, device="cuda", dtype=dtype) for _ in range(2)]
                mask = None
                if mode == "mask":
                    mask = torch.rand(1, 1, 19, 37, device="cuda") > 0.25
                    mask[..., 0, :] = False
                elif mode == "padding":
                    mask = torch.arange(37, device="cuda") < 29
                elif mode == "projection":
                    q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                elif mode == "strided":
                    q, k, v = [torch.stack((x, x), -1).flatten(-2)[..., ::2] for x in (q, k, v)]
                self.compare(q, k, v, mask, mode == "causal", scale=-0.125, compiled=mode == "projection")

    def test_sm121_wide_graph_reads_changed_inputs(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from simpletuner.helpers.training.kohaku_fa_cute import native

        extension = native.extension()
        q = torch.randn(1, 2, 33, 512, device="cuda", dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, 65, 512, device="cuda", dtype=q.dtype) for _ in range(2)]
        dout = torch.randn_like(q)

        def run():
            out, maximum, logsum, blocks = extension.forward(q, k, v, None, -0.125, False)
            gradients = extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, -0.125, False)
            return out, *gradients

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for seed in (122, 123):
            torch.manual_seed(seed)
            for tensor in (q, k, v, dout):
                tensor.copy_(torch.randn_like(tensor))
            graph.replay()
            torch.cuda.synchronize()
            expected = fp64_reference((q, k, v), dout, scale=-0.125)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)

    def test_sm121_short_forward_reuses_cache_across_both_length_boundaries(self):
        from unittest.mock import patch

        from simpletuner.helpers.training.kohaku_fa_cute import native

        for dimension, dtype, causal in itertools.product((256, 512), (torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dimension=dimension, dtype=dtype, causal=causal):
                native.extension().clear_cache()
                with patch.object(native, "compile_kernel", wraps=native.compile_kernel) as compiler:
                    for queries, keys in ((128, 128), (128, 129), (129, 128), (65, 127), (128, 128)):
                        q = torch.randn(1, 2, queries, dimension, device="cuda", dtype=dtype)
                        k, v = [torch.randn(1, 2, keys, dimension, device="cuda", dtype=dtype) for _ in range(2)]
                        self.compare(q, k, v, causal=causal, scale=-0.125)
                        self.assertEqual(compiler.call_count, 2)

    def test_sm121_short_forward_graph_preserves_statistics_and_aliases(self):
        from scripts.benchmark_attention_precision import fp64_reference, metrics
        from simpletuner.helpers.training.kohaku_fa_cute import native

        extension = native.extension()
        for dimension, dtype, causal in itertools.product((256, 512), (torch.float16, torch.bfloat16), (False, True)):
            with self.subTest(dimension=dimension, dtype=dtype, causal=causal):
                q = torch.randn(1, 19, 4, dimension, device="cuda", dtype=dtype).transpose(1, 2)
                k, v = [torch.randn(1, 37, 2, dimension, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)]
                dout = torch.randn_like(q)

                def run():
                    out, maximum, logsum, blocks = extension.forward(q, k, v, None, -0.125, causal)
                    gradients = extension.backward(q, k, v, None, maximum, logsum, dout, blocks, out, -0.125, causal)
                    return out, maximum, logsum, *gradients

                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for seed in (125, 126):
                    torch.manual_seed(seed)
                    for tensor in (q, k, v, dout):
                        tensor.copy_(torch.randn_like(tensor))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = fp64_reference((q, k, v), dout, causal=causal, scale=-0.125)
                    for got, want in zip((actual[0], *actual[3:]), expected):
                        self.assertLess(metrics(got, want)["relative_error"], 0.01)
                    keys = k.double().repeat_interleave(2, dim=1)
                    logits = q.double() @ keys.transpose(-1, -2)
                    if causal:
                        visible = torch.arange(37, device="cuda")[None, :] <= torch.arange(19, device="cuda")[:, None]
                        logits.masked_fill_(~visible, torch.inf)
                    maximum = (-logits).amax(-1)
                    torch.testing.assert_close(actual[1].double(), maximum, atol=0.001, rtol=0.001)
                    logsum = torch.exp2((-logits - maximum[..., None]) * 0.125 * 1.4426950408889634).sum(-1).log2()
                    torch.testing.assert_close(actual[2].double(), logsum, atol=0.001, rtol=0.001)

    def test_sm121_short_forward_keeps_negative_maxima_and_tiny_gradients(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from scripts.benchmark_attention_regressions import Case, assess, make_case
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dimension, dtype in itertools.product((256, 512), (torch.float16, torch.bfloat16)):
            cases = [
                Case("short_negative", queries=queries, keys=keys, dim=dimension, scale=scale)
                for queries, keys, scale in ((1, 1, None), (3, 63, None), (65, 127, None), (128, 128, -0.125), (19, 37, 0))
            ]
            cases.append(Case("padded_literal", queries=2, keys=2, dim=8, heads=1, kvheads=1, literal=True, scale=8**-0.5))
            cases.append(
                Case("padded_literal_parallel", queries=2, keys=2, dim=8, heads=1, kvheads=1, literal=True, scale=8**-0.5)
            )
            cases.append(
                Case(
                    "padded_literal_parallel_causal",
                    queries=2,
                    keys=2,
                    dim=8,
                    heads=1,
                    kvheads=1,
                    literal=True,
                    scale=8**-0.5,
                    causal=True,
                )
            )
            for case in cases:
                with self.subTest(dimension=dimension, dtype=dtype, case=case):
                    inputs, dout, _ = make_case(case, device="cuda", dtype=dtype)
                    if case.literal:
                        inputs = [torch.nn.functional.pad(value, (0, dimension - 8)) for value in inputs]
                        dout = torch.nn.functional.pad(dout, (0, dimension - 8))
                        if "_parallel" in case.name:
                            inputs = [value.repeat(1, 32, 64, 1) for value in inputs]
                            dout = dout.repeat(1, 32, 64, 1)
                    expected = fp64_reference(inputs, dout, scale=case.scale, causal=case.causal)
                    variables = [value.requires_grad_() for value in inputs]
                    output = attention(*variables, scale=case.scale, causal=case.causal)
                    actual = output, *torch.autograd.grad(output, variables, dout)
                    result = assess(actual, expected, dtype)
                    self.assertEqual(result["status"], "pass", result)

    def test_sm121_short_forward_parallel_tiles_preserve_tail_gradients(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from scripts.benchmark_attention_regressions import assess
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dimension, dtype, causal, (batch, heads) in itertools.product(
            (256, 512), (torch.float16, torch.bfloat16), (False, True), ((1, 16), (1, 32), (4, 16))
        ):
            with self.subTest(dimension=dimension, dtype=dtype, causal=causal, batch=batch, heads=heads):
                torch.manual_seed(128)
                queries, keys = (65, 127) if causal else (128, 128)
                q = torch.randn(batch, heads, queries, dimension, device="cuda", dtype=dtype)
                k, v = [torch.randn(batch, heads, keys, dimension, device="cuda", dtype=dtype) for _ in range(2)]
                dout = torch.randn_like(q)
                expected = fp64_reference((q, k, v), dout, causal=causal, scale=-0.125)
                variables = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = attention(*variables, causal=causal, scale=-0.125)
                actual = output, *torch.autograd.grad(output, variables, dout)
                result = assess(actual, expected, dtype)
                self.assertEqual(result["status"], "pass", result)
