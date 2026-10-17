import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(8, 9), (9, 0), (12, 0)},
    "Requires Ada, Hopper or RTX Blackwell",
)
class CuTeAttentionTests(unittest.TestCase):
    def test_warm_dispatch_does_not_reenter_python_kernels(self):
        import cProfile
        from pathlib import Path
        from types import CodeType

        from simpletuner.helpers.training.kohaku_fa_cute import api

        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(q)
        mask = torch.ones(17, 17, device="cuda", dtype=torch.bool)
        for fn in (api.kohaku_fa_scaled_dot_product_attention, api.automatic_kohaku_fa_scaled_dot_product_attention):
            for selected_mask in (None, mask):
                with self.subTest(backend=fn.__name__, masked=selected_mask is not None):

                    def run():
                        out = fn(q, k, v, attn_mask=selected_mask)
                        return torch.autograd.grad(out, (q, k, v), dout)

                    run()
                    torch.cuda.synchronize()
                    profile = cProfile.Profile()
                    with profile:
                        run()
                    torch.cuda.synchronize()
                    callbacks = [
                        entry.code.co_name
                        for entry in profile.getstats()
                        if isinstance(entry.code, CodeType) and "cutlass" in Path(entry.code.co_filename).parts
                    ]
                    self.assertEqual(callbacks, [])

    def test_automatic_unsupported_first_call_does_not_build_extension(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from simpletuner.helpers.training.kohaku_fa_cute import api

        original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(api, "extension", side_effect=RuntimeError("Unexpected extension build")) as builder:
            builder.cache_info.return_value = SimpleNamespace(currsize=0)
            for dtype, dropout in [(torch.float32, 0.0), (torch.bfloat16, 0.1)]:
                q = torch.randn(1, 2, 17, 64, device="cuda", dtype=dtype, requires_grad=True)
                torch.manual_seed(101)
                actual = api.automatic_kohaku_fa_scaled_dot_product_attention(q, q, q, dropout_p=dropout)
                torch.manual_seed(101)
                expected = original(q, q, q, dropout_p=dropout)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    torch.autograd.grad(actual.sum(), q)[0], torch.autograd.grad(expected.sum(), q)[0], atol=0, rtol=0
                )
                builder.assert_not_called()

    def test_automatic_training_routes_supported_and_native_calls_in_fullgraph(self):
        from types import SimpleNamespace

        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
        from simpletuner.helpers.training.kohaku_fa_cute import scaled_dot_product_attention as kohaku

        original = torch.nn.functional.scaled_dot_product_attention

        def native_call(q, k, v, **kwargs):
            return original(q, k, v, **kwargs)

        compiled_native = torch.compile(native_call, fullgraph=True)
        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.randn(17, 17, device="cuda")
        try:
            AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa-auto"), AttentionPhase.TRAIN)
            for compiled in (False, True):
                fn = torch.nn.functional.scaled_dot_product_attention
                native = compiled_native if compiled else original
                if compiled:
                    fn = torch.compile(fn, fullgraph=True)
                for tensors, kwargs, reference in [
                    ((q, k, v), {"scale": -0.2}, kohaku),
                    ((q.float(), k.float(), v.float()), {"attn_mask": mask, "scale": 0.2}, native),
                    ((q, k, v), {"attn_mask": mask.to(q.dtype), "scale": 0.2}, native),
                    ((q[:, :, :0], k, v), {}, native),
                    ((q, k, v[..., :32]), {}, native),
                ]:
                    with self.subTest(compiled=compiled, dtype=tensors[0].dtype, shapes=[t.shape for t in tensors]):
                        actual_inputs = [t.detach().requires_grad_() for t in tensors]
                        ref_inputs = [t.detach().requires_grad_() for t in tensors]
                        actual = fn(*actual_inputs, **kwargs)
                        expected = reference(*ref_inputs, **kwargs)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        dout = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad(actual, actual_inputs, dout)
                        expected_grads = torch.autograd.grad(expected, ref_inputs, dout)
                        for got, want in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(got, want, atol=0, rtol=0)
        finally:
            AttentionBackendController.restore_default()
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_native_sdpa_rejects_unsupported_features(self):
        from simpletuner.helpers.training.kohaku_fa_cute import scaled_dot_product_attention

        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.ones(17, 17, device="cuda", dtype=torch.bool)
        cases = [
            ((q, k, v), {"dropout_p": 0.1}, "dropout"),
            ((q, k, v), {"attn_mask": mask, "is_causal": True}, "folded"),
            ((q, k, v), {"attn_mask": mask.float()}, "boolean"),
            ((q, k[:, :1], v[:, :1]), {}, "enable_gqa"),
            ((q.float(), k.float(), v.float()), {}, "FP32"),
            ((q.cpu(), k.cpu(), v.cpu()), {}, "CUDA"),
            ((q.squeeze(0), k, v), {}, "4D"),
            ((q, k, v[:, :, :16]), {}, "K/V shapes"),
            ((q[:, :, :0], k, v), {}, "nonempty"),
        ]
        for inputs, kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                scaled_dot_product_attention(*inputs, **kwargs)

    def test_native_dispatch_concurrent_streams_and_cache_release(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import tvm_ffi

        from simpletuner.helpers.training.kohaku_fa_cute import attention
        from simpletuner.helpers.training.kohaku_fa_cute.native import extension

        tensors = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        inputs = [tensor.detach().requires_grad_() for tensor in tensors[:3]]
        output = attention(*inputs, scale=-0.125)
        expected = (output, *torch.autograd.grad(output, inputs, tensors[3]))
        torch.cuda.synchronize()
        native = extension()
        native.clear_cache()
        barrier = Barrier(2)

        def run():
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream):
                inputs = [tensor.detach().clone().requires_grad_() for tensor in tensors[:3]]
                output = attention(*inputs, scale=-0.125)
                gradients = torch.autograd.grad(output, inputs, tensors[3])
            stream.synchronize()
            return output, *gradients

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: run(), range(2)))
        for result in results:
            for got, want in zip(result, expected):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
        self.assertGreater(native.cache_size(), 0)
        self.assertLessEqual(native.cache_size(), 128)
        self.assertFalse(
            any(name.startswith("simpletuner.cute.native.") for name in tvm_ffi.registry.list_global_func_names())
        )
        native.clear_cache()
        self.assertEqual(native.cache_size(), 0)

    def test_compile_after_backend_selection_in_fresh_process(self):
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            """
            from types import SimpleNamespace
            import torch
            from scripts.benchmark_attention_precision import fp64_reference
            from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

            AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa"), AttentionPhase.TRAIN)
            torch.overrides.get_testing_overrides.cache_clear()
            q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros(1, 1, 2, 128, device="cuda", dtype=q.dtype)
            v = torch.zeros_like(k)
            q[..., 0] = 10
            k[0, 0, :, 0] = torch.tensor([10, 5], device="cuda", dtype=q.dtype)
            v[0, 0, :, 0] = torch.tensor([1, -1], device="cuda", dtype=q.dtype)
            dout = torch.zeros_like(q)
            dout[..., 0] = 1
            expected = fp64_reference((q, k, v), dout, scale=1.0)
            inputs = [t.requires_grad_() for t in (q, k, v)]
            fn = torch.compile(torch.nn.functional.scaled_dot_product_attention, fullgraph=True)
            out = fn(*inputs, scale=1.0)
            gradients = torch.autograd.grad(out, inputs, dout)
            assert gradients[0][..., 0].item() > 0
            for actual, reference in zip((out, *gradients), expected):
                torch.testing.assert_close(actual.double(), reference, rtol=0.03, atol=0)
            AttentionBackendController.restore_default()
            """
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_simpletuner_and_diffusers_dispatch(self):
        from types import SimpleNamespace

        from diffusers.models.attention_dispatch import dispatch_attention_fn

        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        native = torch.nn.functional.scaled_dot_product_attention
        expected = native(q, k, v)
        try:
            AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa"), AttentionPhase.TRAIN)
            compiled = torch.compile(torch.nn.functional.scaled_dot_product_attention, fullgraph=True)
            direct = compiled(q, k, v)
            dispatched = dispatch_attention_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            for output in (direct, dispatched):
                torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
                grads = torch.autograd.grad(output.sum(), (q, k, v))
                self.assertTrue(all(torch.isfinite(x).all() for x in grads))
        finally:
            AttentionBackendController.restore_default()
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, native)

    def compare(self, q, k, v, mask=None, causal=False, scale=None, compiled=False):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        refs = [x.detach().double().requires_grad_() for x in (q, k, v)]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                *refs, attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=q.shape[1] != k.shape[1]
            )
        dout = torch.randn_like(q)
        ref_grads = torch.autograd.grad(expected, refs, dout.double())
        inputs = [x.detach().requires_grad_() for x in (q, k, v)]
        if compiled:
            torch.compiler.reset()
        function = torch.compile(attention, fullgraph=True) if compiled else attention
        actual = function(*inputs, mask=mask, causal=causal, scale=scale)
        grads = torch.autograd.grad(actual, inputs, dout)
        tolerance = 0.02 if q.dtype == torch.bfloat16 else 0.003
        for got, want in zip((actual, *grads), (expected, *ref_grads)):
            torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
        if mask is not None and mask.ndim == 4 and not mask[..., 0, :].any():
            self.assertEqual(actual[..., 0, :].count_nonzero().item(), 0)
            self.assertEqual(grads[0][..., 0, :].count_nonzero().item(), 0)

    def test_forward_backward_tails_masks_gqa_and_compile(self):
        for dtype in (torch.float16, torch.bfloat16):
            for mode in ("dense", "mask", "padding", "causal", "gqa", "strided"):
                with self.subTest(dtype=dtype, mode=mode):
                    torch.manual_seed(41)
                    q = torch.randn(2, 4, 33, 64, device="cuda", dtype=dtype)
                    k, v = [torch.randn(2, 2 if mode == "gqa" else 4, 41, 64, device="cuda", dtype=dtype) for _ in range(2)]
                    mask = None
                    if mode == "mask":
                        mask = torch.rand(1, 1, 33, 41, device="cuda") > 0.25
                        mask[..., 0, :] = False
                    if mode == "padding":
                        mask = torch.arange(41, device="cuda") < 29
                    if mode == "strided":
                        q, k, v = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
                    self.compare(q, k, v, mask, mode == "causal", scale=0.3, compiled=True)

    def test_head_dimensions(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (1, 17, 96, 128, 256, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    torch.manual_seed(42)
                    q = torch.randn(1, 1, 17, dim, device="cuda", dtype=dtype)
                    k, v = [torch.randn(1, 1, 35, dim, device="cuda", dtype=dtype) for _ in range(2)]
                    self.compare(q, k, v)

    def test_preserves_projection_layout_with_compiled_backward(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dim in (64, 128, 256):
            with self.subTest(dim=dim):
                torch.manual_seed(87)
                inputs = [
                    torch.randn(1, 137, 2, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
                    for _ in range(3)
                ]
                torch.compiler.reset()
                output = torch.compile(attention, fullgraph=True)(*inputs)
                gradients = torch.autograd.grad(output, inputs, torch.randn_like(output))
                for tensor in (output, *gradients):
                    self.assertEqual(tensor.stride(), inputs[0].stride())
                self.compare(*inputs, compiled=True)

    def test_wide_head_tails_masks_gqa_and_strides(self):
        for dim in (129, 192, 257, 384, 511):
            with self.subTest(dim=dim):
                torch.manual_seed(81)
                q = torch.randn(1, 4, 19, dim * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
                k, v = [torch.randn(1, 2, 37, dim + 1, device="cuda", dtype=torch.bfloat16)[..., 1:] for _ in range(2)]
                mask = torch.rand(1, 1, 19, 37, device="cuda") > 0.2
                mask[..., 0, :] = False
                self.compare(q, k, v, mask=mask, scale=-0.1)

    def test_causal_keys_beyond_query_length(self):
        for dim in (64, 128):
            for queries, keys in ((32, 193), (64, 193), (65, 193)):
                with self.subTest(dim=dim, queries=queries, keys=keys):
                    torch.manual_seed(88)
                    q = torch.randn(1, 2, queries, dim, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(1, 2, keys, dim, device="cuda", dtype=q.dtype) for _ in range(2)]
                    self.compare(q, k, v, causal=True)

    def test_large_logits_key_and_value_gradients(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dtype in (torch.float16, torch.bfloat16):
            generator = torch.Generator(device="cuda").manual_seed(2)
            centers = torch.randn(1, 2, 8, 64, device="cuda", generator=generator)
            pick = torch.randint(8, (1, 2, 269), device="cuda", generator=generator)
            base = centers.gather(2, pick[..., None].expand(-1, -1, -1, 64))
            q, k = [
                (400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=generator))).to(dtype)
                for _ in range(2)
            ]
            v, dout = [torch.randn(base.shape, device="cuda", generator=generator).to(dtype) for _ in range(2)]
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    _, *expected_grads = fp64_reference((q, k, v), dout, causal=causal)
                    inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                    actual = attention(*inputs, causal=causal)
                    grads = torch.autograd.grad(actual, inputs, dout)
                    for got, want, bound in zip(grads[1:], expected_grads[1:], (0.995, 0.9999)):
                        cosine = torch.nn.functional.cosine_similarity(got.double().flatten(), want.flatten(), dim=0)
                        self.assertGreater(cosine.item(), bound)

    def test_wide_head_gradient_cancellation(self):
        from scripts.benchmark_attention_precision import fp64_reference, make_inputs
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q, k, v, dout = make_inputs(1e6, device="cuda", dtype=dtype, seq=256, dim=256, seed=0)
                _, *expected_grads = fp64_reference((q, k, v), dout)
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                output = attention(*inputs)
                grads = torch.autograd.grad(output, inputs, dout)
                for got, want in zip(grads[:2], expected_grads[:2]):
                    self.assertTrue(torch.isfinite(got).all())
                    relative_error = torch.linalg.vector_norm(got.double() - want) / torch.linalg.vector_norm(want)
                    cosine = torch.nn.functional.cosine_similarity(got.double().flatten(), want.flatten(), dim=0)
                    self.assertLess(relative_error.item(), 0.3)
                    self.assertGreater(cosine.item(), 0.995)

    def test_saturated_softmax_gradient_sign_and_magnitude(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        if torch.cuda.get_device_capability() not in {(8, 9), (9, 0)}:
            self.skipTest("Centered backward requires Ada or Hopper.")
        for dtype in (torch.bfloat16, torch.float16):
            for dim in (1, 17, 64, 80, 96, 128, 160, 192, 224, 256, 257, 384, 511, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    q = torch.zeros(1, 1, 1, dim, device="cuda", dtype=dtype)
                    k = torch.zeros(1, 1, 2, dim, device="cuda", dtype=dtype)
                    v = torch.zeros_like(k)
                    dout = torch.zeros_like(q)
                    q[..., 0] = 10
                    k[..., 0] = torch.tensor([10, 5 if dtype == torch.bfloat16 else 9], device="cuda", dtype=dtype)
                    v[..., 0] = torch.tensor([1, -1], device="cuda", dtype=dtype)
                    dout[..., 0] = 1
                    reference = fp64_reference((q, k, v), dout, scale=1.0)
                    inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                    output = attention(*inputs, scale=1.0)
                    gradients = torch.autograd.grad(output, inputs, dout)
                    self.assertGreater(gradients[0][..., 0].item(), 0)
                    for actual, expected in zip((output, *gradients), reference):
                        torch.testing.assert_close(
                            actual.double(), expected, atol=1e-7 if dtype == torch.float16 else 0, rtol=0.02
                        )

    def test_long_dense_tails_gqa_and_compile(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(21)
                q = torch.randn(1, 1537, 4, 128, device="cuda", dtype=dtype).transpose(1, 2)
                k, v = [torch.randn(1, 1601, 2, 128, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2)]
                self.compare(q, k, v, scale=-0.125, compiled=True)

    def test_long_dense_saturated_gradient(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        q = torch.zeros(1, 1537, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        k = torch.zeros(1, 1601, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        v, dout = torch.zeros_like(k), torch.zeros_like(q)
        q[..., 0], k[..., 0], k[:, :, 0, 0] = 10, 5, 10
        v[..., 0], v[:, :, 0, 0], dout[..., 0] = -1, 1, 1
        expected = fp64_reference((q, k, v), dout, scale=1.0)
        inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
        output = attention(*inputs, scale=1.0)
        actual = (output, *torch.autograd.grad(output, inputs, dout))
        self.assertGreater(actual[1][..., 0].min().item(), 0)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.double(), want, atol=0, rtol=0.03)

    def test_long_masked_runs_keep_tma_stages_synchronized(self):
        from scripts.benchmark_attention_precision import fp64_reference
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dim in (64, 128):
            with self.subTest(dim=dim):
                torch.manual_seed(100)
                q, k, v, dout = [torch.randn(1, 2, 4096, dim, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
                segments = torch.arange(4096, device="cuda") // 256
                mask = segments[:, None] == segments[None, :]
                inputs = [tensor.requires_grad_() for tensor in (q, k, v)]
                expected = fp64_reference(inputs, dout, mask)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        output = attention(*inputs, mask=mask)
                        gradients = torch.autograd.grad(output, inputs, dout)
                torch.cuda.current_stream().wait_stream(stream)
                for actual, reference in zip((output, *gradients), expected):
                    torch.testing.assert_close(actual.double(), reference, atol=0.02, rtol=0.02)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = attention(*inputs, mask=mask)
                    gradients = torch.autograd.grad(output, inputs, dout)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                for actual, reference in zip((output, *gradients), expected):
                    torch.testing.assert_close(actual.double(), reference, atol=0.02, rtol=0.02)

    def test_single_query_and_key_tma_layouts(self):
        for dim in (64, 128, 256):
            for queries, keys, batch, heads, kvheads, bshd, causal in (
                (1, 2, 1, 1, 1, False, False),
                (2, 1, 1, 1, 1, False, False),
                (1, 1, 2, 2, 1, True, False),
                (1, 65, 1, 2, 2, True, True),
            ):
                with self.subTest(dim=dim, queries=queries, keys=keys, bshd=bshd, causal=causal):
                    torch.manual_seed(97)
                    q = torch.randn(batch, heads, queries, dim, device="cuda", dtype=torch.bfloat16)
                    k, v = [torch.randn(batch, kvheads, keys, dim, device="cuda", dtype=q.dtype) for _ in range(2)]
                    if bshd:
                        q, k, v = [tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (q, k, v)]
                    self.compare(q, k, v, causal=causal)

    def test_centered_backward_compilation(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 80, 128, 192, 256, 512):
                with self.subTest(dtype=dtype, dim=dim):
                    torch.manual_seed(85)
                    q = torch.randn(1, 4, 67, dim, device="cuda", dtype=dtype)
                    k, v = [torch.randn(1, 2, 83, dim, device="cuda", dtype=dtype) for _ in range(2)]
                    mask = torch.arange(67, device="cuda")[:, None] // 19 == torch.arange(83, device="cuda")[None, :] // 19
                    self.compare(q, k, v, mask=mask, compiled=True)

    def test_block_masks_and_broadcast_axes(self):
        torch.manual_seed(82)
        q = torch.randn(2, 4, 137, 64, device="cuda", dtype=torch.bfloat16)
        k, v = [torch.randn(2, 2, 149, 64, device="cuda", dtype=q.dtype) for _ in range(2)]
        queries = torch.arange(137, device="cuda")[:, None]
        keys = torch.arange(149, device="cuda")[None, :]
        masks = {
            "segments": (queries // 43 == keys // 43)[None, None],
            "padding": (keys < 91)[None, None],
            "query": (queries < 81)[None, None],
            "full": torch.ones(1, 1, 1, 1, device="cuda", dtype=torch.bool),
            "empty": torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.bool),
            "strided": (torch.rand(2, 4, 274, 298, device="cuda") > 0.4)[..., ::2, ::2],
        }
        for name, mask in masks.items():
            with self.subTest(mask=name):
                self.compare(q, k, v, mask=mask, compiled=True)

    def test_production_model_attention(self):
        from types import SimpleNamespace

        from simpletuner.helpers.models.ideogram.transformer import Attention, Ideogram4MRoPE
        from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Attention, MiniMaxH3RotaryPosEmbed
        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

        torch.manual_seed(83)
        positions = torch.randint(16, (2, 137, 3), device="cuda")
        segments = torch.arange(137, device="cuda")[None].expand(2, -1) // 43
        for name in ("ideogram", "minimax"):
            with self.subTest(model=name):
                if name == "ideogram":
                    model = Attention(512, 2).cuda().bfloat16()
                    cos, sin = [t.bfloat16() for t in Ideogram4MRoPE(256, 5000000, (24, 20, 20)).cuda()(positions)]
                    arguments = (segments, cos, sin)
                    hidden = 512
                else:
                    model = MiniMaxH3Attention(256, 4, 64).cuda().bfloat16()
                    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=8).cuda()(positions)
                    arguments = (rope, (segments[:, :, None] == segments[:, None, :])[:, None])
                    hidden = 256
                x = torch.randn(2, 137, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                dout = torch.randn_like(x)
                variables = (x, *tuple(model.parameters()))
                expected = model(x, *arguments)
                ref_grads = torch.autograd.grad(expected, variables, dout)
                try:
                    AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa"), AttentionPhase.TRAIN)
                    actual = model(x, *arguments)
                    grads = torch.autograd.grad(actual, variables, dout)
                    for got, want in zip((actual, *grads), (expected, *ref_grads)):
                        error = (got.float() - want.float()).norm()
                        self.assertLessEqual(error.item(), 0.02 * want.float().norm().item() + 1e-5)
                finally:
                    AttentionBackendController.restore_default()

    def test_zero_negative_scale_and_offset_views(self):
        q = torch.randn(1, 2, 17, 65, device="cuda", dtype=torch.bfloat16)[..., 1:]
        k, v = [torch.randn(1, 2, 33, 65, device="cuda", dtype=torch.bfloat16)[..., 1:] for _ in range(2)]
        for scale in (0.0, -0.25):
            with self.subTest(scale=scale):
                self.compare(q, k, v, scale=scale)

    def test_optimizer_steps_with_checkpointing_and_lora(self):
        from copy import deepcopy
        from types import SimpleNamespace

        from peft import LoraConfig, get_peft_model
        from torch.utils.checkpoint import checkpoint

        from simpletuner.helpers.models.ideogram.transformer import Attention, Ideogram4MRoPE
        from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Attention, MiniMaxH3RotaryPosEmbed
        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase

        for family in ("ideogram", "minimax"):
            for lora in (False, True):
                for reentrant in (False, True):
                    with self.subTest(family=family, lora=lora, reentrant=reentrant):
                        torch.manual_seed(86)
                        hidden = 512 if family == "ideogram" else 256
                        positions = torch.randint(16, (1, 137, 3), device="cuda")
                        segments = torch.arange(137, device="cuda")[None] // 43
                        if family == "ideogram":
                            model = Attention(hidden, 2).cuda().train()
                            cos, sin = [t.bfloat16() for t in Ideogram4MRoPE(256, 5000000, (24, 20, 20)).cuda()(positions)]
                            arguments = (segments, cos, sin)
                        else:
                            model = MiniMaxH3Attention(hidden, 2, 128).cuda().train()
                            rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16).cuda()(positions)
                            arguments = (rope, (segments[:, :, None] == segments[:, None, :])[:, None])
                        if lora:
                            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, target_modules="all-linear"))
                        x = torch.randn(1, 137, hidden, device="cuda", requires_grad=True)
                        target = torch.randn_like(x)
                        initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                        trained, losses, predictions = [], [], []
                        try:
                            for backend in ("diffusers", "kohaku-fa"):
                                current = deepcopy(model)
                                optimizer = torch.optim.AdamW((p for p in current.parameters() if p.requires_grad), lr=1e-3)
                                AttentionBackendController.apply(
                                    SimpleNamespace(attention_mechanism=backend), AttentionPhase.TRAIN
                                )
                                history = []
                                for _ in range(4):
                                    optimizer.zero_grad(set_to_none=True)
                                    x.grad = None
                                    with torch.autocast("cuda", dtype=torch.bfloat16):
                                        output = checkpoint(current, x, *arguments, use_reentrant=reentrant)
                                        loss = (output.float() - target).square().mean()
                                    loss.backward()
                                    self.assertTrue(torch.isfinite(loss))
                                    for parameter in current.parameters():
                                        if parameter.requires_grad:
                                            self.assertIsNotNone(parameter.grad)
                                            self.assertTrue(torch.isfinite(parameter.grad).all())
                                    optimizer.step()
                                    history.append(loss.detach())
                                trained.append(
                                    {name: p.detach().clone() for name, p in current.named_parameters() if p.requires_grad}
                                )
                                losses.append(torch.stack(history))
                                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                    predictions.append(current(x, *arguments))
                        finally:
                            AttentionBackendController.restore_default()
                        torch.testing.assert_close(losses[1], losses[0], atol=0.001, rtol=0.01)
                        updates = [
                            torch.cat([(parameters[name] - initial[name]).flatten() for name in initial])
                            for parameters in trained
                        ]
                        self.assertGreater(updates[1].norm().item(), 0)
                        self.assertLess((updates[1] - updates[0]).norm().item(), 0.05 * updates[0].norm().item())
                        self.assertLess((predictions[1] - predictions[0]).norm().item(), 0.02 * predictions[0].norm().item())
                        self.assertLess(losses[1][-1].item(), losses[1][0].item())

    def test_cuda_graph_current_stream_and_gradients(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        inputs = [torch.randn(1, 2, 33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = attention(*inputs)
                grads = torch.autograd.grad(output.sum(), inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = attention(*inputs)
            grads = torch.autograd.grad(output.sum(), inputs)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(all(torch.isfinite(x).all() for x in (output, *grads)))
        reference = torch.nn.functional.scaled_dot_product_attention(*inputs)
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    def test_cuda_graph_reclassifies_changed_masks(self):
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        for dim in (64, 128, 256):
            torch.manual_seed(84)
            inputs = [torch.randn(1, 2, 137, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
            dout = torch.randn_like(inputs[0])
            ids = torch.arange(137, device="cuda") // 43
            mask = (ids[:, None] == ids[None, :])[None, None]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = attention(*inputs, mask=mask)
                    grads = torch.autograd.grad(output, inputs, dout)
            torch.cuda.current_stream().wait_stream(stream)
            del output, grads
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = attention(*inputs, mask=mask)
                grads = torch.autograd.grad(output, inputs, dout)
            for visible in (False, True):
                mask.fill_(visible)
                graph.replay()
                reference_inputs = [x.detach().double().requires_grad_() for x in inputs]
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, attn_mask=mask)
                ref_grads = torch.autograd.grad(expected, reference_inputs, dout.double())
                for got, want in zip((output, *grads), (expected, *ref_grads)):
                    torch.testing.assert_close(got.double(), want, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
