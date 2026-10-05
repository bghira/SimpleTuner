import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

from simpletuner.helpers.training.attention_backend import (
    AttentionBackendController,
    AttentionPhase,
    _automatic_kohaku_fa_sdpa,
    _kohaku_fa_sdpa,
    get_kohaku_fa_unavailable_reason,
)


class KohakuFAIntegrationTests(unittest.TestCase):
    def tearDown(self):
        AttentionBackendController.restore_default()

    def test_cli_and_webui_default_to_automatic_attention(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import field_registry

        self.assertEqual(field_registry.get_field("attention_mechanism").default_value, "kohaku-fa-auto")
        self.assertEqual(get_default_config()["attention_mechanism"], "kohaku-fa-auto")

    def test_implicit_default_on_cpu_preserves_native_attention(self):
        original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(torch.cuda, "is_available", return_value=False):
            AttentionBackendController.apply(SimpleNamespace(), AttentionPhase.TRAIN)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
        self.assertEqual(AttentionBackendController.active_backend(), "kohaku-fa-auto")

    def test_default_keeps_native_attention_on_uncentered_architectures(self):
        original = torch.nn.functional.scaled_dot_product_attention
        for capability in ((10, 0), (10, 3), (12, 0)):
            with (
                self.subTest(capability=capability),
                patch("simpletuner.helpers.training.attention_backend.get_kohaku_fa_unavailable_reason", return_value=None),
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
            ):
                AttentionBackendController.apply(SimpleNamespace(), AttentionPhase.TRAIN)
                self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
                self.assertEqual(AttentionBackendController.active_backend(), "kohaku-fa-auto")

    def test_implicit_default_routes_training_and_restores_evaluation(self):
        original = torch.nn.functional.scaled_dot_product_attention
        kernel = Mock()
        module = SimpleNamespace(automatic_scaled_dot_product_attention=kernel)
        with (
            patch("simpletuner.helpers.training.attention_backend.get_kohaku_fa_unavailable_reason", return_value=None),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.dict("sys.modules", {"simpletuner.helpers.training.kohaku_fa_cute": module}),
        ):
            for phase in (AttentionPhase.TRAIN, AttentionPhase.EVAL, AttentionPhase.TRAIN):
                AttentionBackendController.apply(SimpleNamespace(), phase)
                self.assertEqual(AttentionBackendController.active_backend(), "kohaku-fa-auto")
                if phase == AttentionPhase.TRAIN:
                    self.assertIs(torch.nn.functional.scaled_dot_product_attention, kernel)
                else:
                    self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_non_cuda_host_rejected_without_importing_kernels(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                AttentionBackendController.apply(
                    SimpleNamespace(attention_mechanism="kohaku-fa"),
                    AttentionPhase.TRAIN,
                )

    def test_automatic_mode_on_cpu_preserves_native_forward_and_backward(self):
        original = torch.nn.functional.scaled_dot_product_attention
        with patch.object(torch.cuda, "is_available", return_value=False):
            AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa-auto"), AttentionPhase.TRAIN)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
        self.assertEqual(AttentionBackendController.active_backend(), "kohaku-fa-auto")
        q = torch.randn(1, 2, 3, 16, requires_grad=True)
        mask = torch.randn(3, 3)
        direct = original(q, q, q, attn_mask=mask, scale=-0.1)
        out = dispatch_attention_fn(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2), attn_mask=mask, scale=-0.1)
        torch.testing.assert_close(out.transpose(1, 2), direct, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), q)[0], torch.autograd.grad(direct.sum(), q)[0])

    def test_automatic_mode_preserves_unsupported_model_features(self):
        original = torch.nn.functional.scaled_dot_product_attention
        for settings in [{"context_parallel_size": 2}, {"minimax_h3_sparse_attention": "flex"}]:
            with (
                self.subTest(settings=settings),
                patch("simpletuner.helpers.training.attention_backend.get_kohaku_fa_unavailable_reason") as availability,
            ):
                AttentionBackendController.apply(
                    SimpleNamespace(attention_mechanism="kohaku-fa-auto", **settings), AttentionPhase.TRAIN
                )
                self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
                availability.assert_not_called()

    def test_automatic_adapter_routes_calls_and_does_not_hide_kernel_errors(self):
        def tensor(heads=2, dtype=torch.bfloat16, device="cuda", dim=64):
            return SimpleNamespace(ndim=4, device=torch.device(device), dtype=dtype, shape=(1, heads, 5, dim))

        q, k, v = tensor(4), tensor(), tensor()
        kernel = Mock(return_value="kohaku")
        arguments = {"scale": -0.2, "enable_gqa": True}
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(AttentionBackendController, "_call_original_sdpa", return_value="native") as native,
        ):
            self.assertEqual(_automatic_kohaku_fa_sdpa(kernel, {(9, 0)}, q, k, v, **arguments), "kohaku")
            kernel.assert_called_once_with(q, k, v, None, 0.0, False, **arguments)
            native.assert_not_called()
            for tensors, extra in [
                ((q, k, v), {"dropout_p": 0.1}),
                ((q, k, v), {"attn_mask": tensor(dtype=torch.float32)}),
                ((q, k, v), {"attn_mask": tensor(dtype=torch.bool), "is_causal": True}),
                ((tensor(dtype=torch.float32),) * 3, {}),
                ((tensor(device="cpu"),) * 3, {}),
                ((tensor(device="mps"),) * 3, {}),
                ((tensor(dim=513),) * 3, {}),
            ]:
                with self.subTest(tensors=tensors, extra=extra):
                    self.assertEqual(_automatic_kohaku_fa_sdpa(kernel, {(9, 0)}, *tensors, **(arguments | extra)), "native")
            kernel.side_effect = RuntimeError("kernel failed")
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                _automatic_kohaku_fa_sdpa(kernel, {(9, 0)}, q, k, v, **arguments)
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            with patch.object(AttentionBackendController, "_call_original_sdpa", return_value="native"):
                self.assertEqual(_automatic_kohaku_fa_sdpa(kernel, {(9, 0)}, q, k, v, **arguments), "native")

    def test_automatic_mode_restores_eval_and_reenables_training(self):
        original = torch.nn.functional.scaled_dot_product_attention
        kernel = Mock()
        module = SimpleNamespace(scaled_dot_product_attention=kernel, automatic_scaled_dot_product_attention=kernel)
        with (
            patch("simpletuner.helpers.training.attention_backend.get_kohaku_fa_unavailable_reason", return_value=None),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.dict("sys.modules", {"simpletuner.helpers.training.kohaku_fa_cute": module}),
        ):
            for phase in (AttentionPhase.TRAIN, AttentionPhase.EVAL, AttentionPhase.TRAIN):
                AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa-auto"), phase)
                if phase == AttentionPhase.TRAIN:
                    self.assertIsNot(torch.nn.functional.scaled_dot_product_attention, original)
                else:
                    self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)
                self.assertEqual(AttentionBackendController._active_phase, phase)
            AttentionBackendController.restore_default()
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_automatic_adapter_cpu_branch_compiles_without_loading_cuda_kernels(self):
        from functools import partial

        kernel = Mock(side_effect=AssertionError("CUDA kernel reached on CPU"))
        fn = torch.compile(partial(_automatic_kohaku_fa_sdpa, kernel, {(9, 0)}), backend="eager", fullgraph=True)
        q = torch.randn(1, 2, 3, 16, requires_grad=True)
        out = fn(q, q, q, scale=-0.2)
        native = torch.nn.functional.scaled_dot_product_attention(q, q, q, scale=-0.2)
        torch.testing.assert_close(out, native, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), q)[0], torch.autograd.grad(native.sum(), q)[0])
        kernel.assert_not_called()

    def test_unsupported_architectures_rejected(self):
        for capability in [(7, 5), (8, 0), (12, 1)]:
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
            ):
                self.assertIn("unsupported", get_kohaku_fa_unavailable_reason())

    def test_architecture_specific_dependencies(self):
        for capability, packages in [
            ((8, 9), {"nvidia-cutlass-dsl": "4.8", "apache-tvm-ffi": "0.1.14", "ninja": "1.11"}),
            ((9, 0), {"nvidia-cutlass-dsl": "4.8", "apache-tvm-ffi": "0.1.14", "ninja": "1.11"}),
            ((12, 0), {"nvidia-cutlass-dsl": "4.8", "apache-tvm-ffi": "0.1.14", "ninja": "1.11"}),
            ((10, 0), {"triton": "3.7"}),
        ]:
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
                patch("importlib.metadata.version", side_effect=packages.__getitem__),
            ):
                self.assertIsNone(get_kohaku_fa_unavailable_reason())

    def test_cute_requires_native_dlpack_pytorch_api(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(torch, "__version__", "2.10.0+cu128"),
        ):
            self.assertIn("2.11", get_kohaku_fa_unavailable_reason())

    def test_context_parallel_and_sparse_attention_rejected(self):
        for settings in [
            {"context_parallel_size": 2},
            {"minimax_h3_sparse_attention": "flex"},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AttentionBackendController.apply(
                    SimpleNamespace(attention_mechanism="kohaku-fa", **settings),
                    AttentionPhase.TRAIN,
                )

    def test_direct_sdpa_and_diffusers_dispatch_route_and_restore(self):
        original = torch.nn.functional.scaled_dot_product_attention
        q = torch.randn(1, 2, 3, 16, requires_grad=True)
        k = torch.randn(1, 2, 5, 16, requires_grad=True)
        v = torch.randn(1, 2, 5, 16, requires_grad=True)
        kernel = Mock(side_effect=lambda q, k, v, **kwargs: original(q, k, v, attn_mask=kwargs["mask"]))
        module = SimpleNamespace(attention=kernel)

        def cpu_adapter(attention, query, key, value, attn_mask=None, *args, **kwargs):
            return attention(query, key, value, mask=attn_mask)

        with (
            patch(
                "simpletuner.helpers.training.attention_backend.get_kohaku_fa_unavailable_reason",
                return_value=None,
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.dict("sys.modules", {"simpletuner.helpers.training.kohakufa": module}),
            patch(
                "simpletuner.helpers.training.attention_backend._kohaku_fa_sdpa",
                side_effect=cpu_adapter,
            ),
        ):
            for phase in (AttentionPhase.TRAIN, AttentionPhase.EVAL):
                AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa"), phase)
                self.assertEqual(AttentionBackendController._active_phase, phase)
                self.assertEqual(
                    torch.nn.functional.scaled_dot_product_attention.__name__, "kohaku_fa_scaled_dot_product_attention"
                )
                torch.overrides.get_testing_overrides.cache_clear()
                try:
                    self.assertIn(torch.nn.functional.scaled_dot_product_attention, torch.overrides.get_testing_overrides())
                finally:
                    torch.overrides.get_testing_overrides.cache_clear()
                direct = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                dispatched = dispatch_attention_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                torch.testing.assert_close(direct, dispatched.transpose(1, 2))
                gradients = torch.autograd.grad(dispatched.sum(), (q, k, v))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertEqual(kernel.call_count, 4)
            AttentionBackendController.apply(SimpleNamespace(attention_mechanism="native-math"), AttentionPhase.TRAIN)
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_adapter_preserves_mask_causal_scale_and_gqa(self):
        def tensor(heads=2, dtype=torch.bfloat16):
            return SimpleNamespace(
                ndim=4,
                device=torch.device("cuda", 0),
                dtype=dtype,
                shape=(1, heads, 5, 64),
            )

        q, k, v = tensor(4), tensor(), tensor()
        mask = tensor(dtype=torch.bool)
        kernel = Mock(return_value="output")
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            result = _kohaku_fa_sdpa(kernel, q, k, v, mask, scale=0.2, enable_gqa=True)
            self.assertEqual(result, "output")
            kernel.assert_called_once_with(q, k, v, mask=mask, causal=False, scale=0.2)
            for kwargs, message in [
                ({"dropout_p": 0.1}, "dropout"),
                ({}, "enable_gqa"),
                (
                    {"enable_gqa": True, "attn_mask": tensor(dtype=torch.float32)},
                    "boolean",
                ),
                ({"enable_gqa": True, "attn_mask": mask, "is_causal": True}, "folded"),
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                    _kohaku_fa_sdpa(kernel, q, k, v, **kwargs)
            with self.assertRaisesRegex(ValueError, "FP32"):
                _kohaku_fa_sdpa(
                    kernel,
                    tensor(dtype=torch.float32),
                    tensor(dtype=torch.float32),
                    tensor(dtype=torch.float32),
                )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(10, 0), (10, 3)},
    "Requires sm_100/sm_103",
)
class KohakuFAKernelTests(unittest.TestCase):
    def tearDown(self):
        AttentionBackendController.restore_default()

    def test_mask_tile_lists_preserve_visible_columns_and_classes(self):
        from simpletuner.helpers.training.kohakufa.mask import _csr

        listed = torch.tensor([[[[False, True, False], [True, False, True], [False, False, False]]]], device="cuda")
        classes = torch.tensor([[[[0, 3, 0], [1, 0, 2], [0, 0, 0]]]], device="cuda")
        compiled = torch.compile(_csr, fullgraph=True)
        start, count, entries, packed_classes = compiled(listed, classes, 2, 4)
        expected_columns = [[1], [0, 2], [0]] * 8
        expected_classes = [[3], [1, 2], [0]] * 8
        for row, (columns, row_classes) in enumerate(zip(expected_columns, expected_classes)):
            offset, length = int(start[row]), int(count[row])
            self.assertEqual(entries[offset : offset + length].tolist(), columns)
            self.assertEqual(packed_classes[offset : offset + length].tolist(), row_classes)

    def test_large_logits_preserve_key_and_value_gradients(self):
        generator = torch.Generator(device="cuda").manual_seed(2)
        centers = torch.randn(1, 2, 8, 64, device="cuda", generator=generator)
        pick = torch.randint(8, (1, 2, 269), device="cuda", generator=generator)
        base = centers.gather(2, pick[..., None].expand(-1, -1, -1, 64))
        q, k = [(400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=generator))).half() for _ in range(2)]
        v, dout = [torch.randn(base.shape, device="cuda", generator=generator).half() for _ in range(2)]
        for causal in (False, True):
            with self.subTest(causal=causal):
                ref_inputs = [t.double().requires_grad_() for t in (q, k, v)]
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    reference = torch.nn.functional.scaled_dot_product_attention(*ref_inputs, is_causal=causal)
                ref_grad = torch.autograd.grad(reference, ref_inputs, dout.double())
                AttentionBackendController.apply(SimpleNamespace(attention_mechanism="kohaku-fa"), AttentionPhase.TRAIN)
                inputs = [t.detach().requires_grad_() for t in (q, k, v)]
                out = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=causal)
                grads = torch.autograd.grad(out, inputs, dout)
                for got, want, bound in zip(grads[1:], ref_grad[1:], (0.995, 0.9999)):
                    cosine = torch.nn.functional.cosine_similarity(got.double().flatten(), want.flatten(), dim=0)
                    self.assertGreater(cosine.item(), bound)
                AttentionBackendController.restore_default()

    def test_forward_backward_masks_gqa_and_compile(self):
        for dtype in (torch.float16, torch.bfloat16):
            for mode in ("dense", "mask", "padding", "causal", "gqa"):
                with self.subTest(dtype=dtype, mode=mode):
                    torch.manual_seed(7)
                    q = torch.randn(1, 4, 33, 64, device="cuda", dtype=dtype, requires_grad=True)
                    k, v = [
                        torch.randn(
                            1,
                            2 if mode == "gqa" else 4,
                            41,
                            64,
                            device="cuda",
                            dtype=dtype,
                            requires_grad=True,
                        )
                        for _ in range(2)
                    ]
                    mask = torch.rand(1, 1, 33, 41, device="cuda") > 0.25 if mode == "mask" else None
                    if mask is not None:
                        mask[..., 0, :] = False
                    if mode == "padding":
                        mask = torch.arange(41, device="cuda") < 35
                    kwargs = dict(
                        attn_mask=mask,
                        is_causal=mode == "causal",
                        enable_gqa=mode == "gqa",
                    )
                    ref_inputs = [t.detach().double().requires_grad_() for t in (q, k, v)]
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        reference = torch.nn.functional.scaled_dot_product_attention(*ref_inputs, **kwargs)
                    dout = torch.randn_like(q)
                    ref_grad = torch.autograd.grad(reference, ref_inputs, dout.double())
                    AttentionBackendController.apply(
                        SimpleNamespace(attention_mechanism="kohaku-fa"),
                        AttentionPhase.TRAIN,
                    )
                    fn = torch.compile(torch.nn.functional.scaled_dot_product_attention, fullgraph=True)
                    out = fn(q, k, v, **kwargs)
                    grads = torch.autograd.grad(out, (q, k, v), dout)
                    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
                    for got, want in zip((out, *grads), (reference, *ref_grad)):
                        torch.testing.assert_close(got.double(), want, atol=tolerance, rtol=tolerance)
                    AttentionBackendController.restore_default()


if __name__ == "__main__":
    unittest.main()
