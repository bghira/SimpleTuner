import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.training import attention_backend as attention_backend_module
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase, PackedAttentionBackend


class TestAttentionBackendPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._reset_controller()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self._reset_controller()

    def _reset_controller(self):
        AttentionBackendController.restore_default()
        AttentionBackendController._active_backend = None
        AttentionBackendController._active_phase = AttentionPhase.TRAIN
        AttentionBackendController._sla_cache = {}
        AttentionBackendController._sla_settings = None
        AttentionBackendController._optimizer = None
        AttentionBackendController._parameter_sink = None
        AttentionBackendController._sink_param_ids = set()
        AttentionBackendController._optimizer_param_ids = set()
        AttentionBackendController._sla_state_store = {}
        AttentionBackendController._diffusers_backend_context = None
        AttentionBackendController._diffusers_backend_name = None
        AttentionBackendController._metal_flash_attention_health_checked = False
        attention_backend_module.get_packed_attention_backend.cache_clear()
        attention_backend_module.get_metal_flash_attention_unavailable_reason.cache_clear()
        attention_backend_module._metal_flash_attention_runtime_error.cache_clear()

    def test_metal_flash_attention_availability_false_when_package_missing(self):
        def fake_import(name):
            if name == "pytorch_custom_op_ffi":
                raise ImportError(name)
            return __import__(name)

        with patch.object(attention_backend_module.importlib, "import_module", side_effect=fake_import):
            self.assertFalse(attention_backend_module.is_metal_flash_attention_available())

    def test_metal_flash_attention_availability_false_when_runtime_parity_fails(self):
        package = SimpleNamespace(is_metal_sdpa_available=lambda: True)
        extension = SimpleNamespace(
            metal_flash_attention_autograd=lambda query, key, value, is_causal=False, scale=0.0: query
        )

        def fake_import(name):
            if name == "pytorch_custom_op_ffi":
                return package
            if name == "metal_sdpa_extension":
                return extension
            return __import__(name)

        with (
            patch.object(attention_backend_module.importlib, "import_module", side_effect=fake_import),
            patch.object(attention_backend_module, "_metal_flash_attention_runtime_error", return_value="parity failed"),
        ):
            self.assertFalse(attention_backend_module.is_metal_flash_attention_available())
            self.assertIn(
                "parity failed",
                attention_backend_module.get_metal_flash_attention_unavailable_reason(),
            )

    def test_metal_flash_attention_installs_sdpa_wrapper(self):
        calls = []

        def fake_metal_sdpa(query, key, value, is_causal=False, scale=0.0):
            calls.append((query, key, value, is_causal, scale))
            return query + 1

        extension = SimpleNamespace(metal_flash_attention_autograd=fake_metal_sdpa)
        config = type("Config", (object,), {"attention_mechanism": "metal-flash-attention"})()

        with (
            patch.object(AttentionBackendController, "_load_metal_flash_attention_extension", return_value=extension),
            patch.object(AttentionBackendController, "_metal_flash_attention_should_fallback", return_value=False),
        ):
            AttentionBackendController.apply(config, AttentionPhase.TRAIN)
            query = torch.zeros(1, 1, 2, 4)
            key = torch.zeros_like(query)
            value = torch.zeros_like(query)
            output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, scale=0.5)

        self.assertEqual(AttentionBackendController._active_backend, "metal-flash-attention")
        self.assertTrue(torch.equal(output, torch.ones_like(query)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][3])
        self.assertEqual(calls[0][4], 0.5)

    def test_metal_flash_attention_wrapper_falls_back_when_unsupported(self):
        calls = []

        def fake_metal_sdpa(query, key, value, is_causal=False, scale=0.0):
            calls.append((query, key, value, is_causal, scale))
            return query + 1

        def fake_original(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            return query + 2

        extension = SimpleNamespace(metal_flash_attention_autograd=fake_metal_sdpa)
        config = type("Config", (object,), {"attention_mechanism": "metal-flash-attention"})()

        with (
            patch.object(AttentionBackendController, "_load_metal_flash_attention_extension", return_value=extension),
            patch.object(AttentionBackendController, "_metal_flash_attention_should_fallback", return_value=True),
        ):
            AttentionBackendController.apply(config, AttentionPhase.TRAIN)
            query = torch.zeros(1, 1, 2, 4)
            key = torch.zeros_like(query)
            value = torch.zeros_like(query)
            with patch.object(torch.nn.functional, "scaled_dot_product_attention_sdpa", fake_original, create=True):
                output = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.1)

        self.assertTrue(torch.equal(output, torch.full_like(query, 2)))
        self.assertEqual(calls, [])

    def test_call_original_sdpa_uses_keyword_only_arguments(self):
        query = torch.zeros(1, 1, 2, 4)

        output = AttentionBackendController._call_original_sdpa(
            query,
            query,
            query,
            None,
            0.0,
            False,
            None,
            False,
        )

        self.assertEqual(output.shape, query.shape)

    def test_metal_flash_attention_runtime_check_reports_failure(self):
        result = SimpleNamespace(returncode=1, stdout="extension crashed", stderr="")

        with patch.object(attention_backend_module.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "runtime check failed"):
                AttentionBackendController._check_metal_flash_attention_runtime()

        self.assertFalse(AttentionBackendController._metal_flash_attention_health_checked)

    def test_metal_flash_attention_fallback_conditions(self):
        query = torch.zeros(1, 1, 2, 4)

        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                query,
                query,
                query,
                None,
                0.0,
                False,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                query,
                query,
                query,
                None,
                0.1,
                False,
                False,
            )
        )

    @unittest.skipUnless(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "Requires MPS.",
    )
    def test_metal_flash_attention_dispatches_only_verified_fp32_mps_shape(self):
        query = torch.zeros(1, 4, 64, 64, device="mps", dtype=torch.float32)
        transposed_query = torch.zeros(1, 64, 4, 64, device="mps", dtype=torch.float32).transpose(1, 2)
        single_head_query = torch.zeros(1, 1, 8, 16, device="mps", dtype=torch.float32)
        fp16_query = torch.zeros_like(query, dtype=torch.float16)
        mask = torch.zeros(1, 1, 64, 64, device="mps", dtype=torch.bool)
        two_dimensional_query = torch.zeros(2, 4, device="mps", dtype=torch.float32)

        self.assertFalse(
            AttentionBackendController._metal_flash_attention_should_fallback(
                query,
                query,
                query,
                None,
                0.0,
                False,
                False,
            )
        )
        self.assertFalse(
            AttentionBackendController._metal_flash_attention_should_fallback(
                transposed_query,
                transposed_query,
                transposed_query,
                None,
                0.0,
                False,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                single_head_query,
                single_head_query,
                single_head_query,
                None,
                0.0,
                False,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                fp16_query,
                fp16_query,
                fp16_query,
                None,
                0.0,
                False,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                query,
                query,
                query,
                mask,
                0.0,
                False,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                query,
                query,
                query,
                None,
                0.0,
                True,
                False,
            )
        )
        self.assertTrue(
            AttentionBackendController._metal_flash_attention_should_fallback(
                two_dimensional_query,
                two_dimensional_query,
                two_dimensional_query,
                None,
                0.0,
                False,
                False,
            )
        )

    def test_save_checkpoint_no_state(self):
        AttentionBackendController._active_backend = "sla"
        AttentionBackendController.on_save_checkpoint(self.tmpdir, is_main_process=True)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "sla_attention.pt")))

    def test_round_trip_serialization(self):
        sample_state = {"proj_l.weight": torch.ones(1, 1), "proj_l.bias": torch.zeros(1)}
        AttentionBackendController._sla_state_store = {(128, "bf16"): sample_state}
        AttentionBackendController._sla_settings = {
            "topk": 0.2,
            "feature_map": "softmax",
            "blkq": 64,
            "blkk": 64,
        }
        AttentionBackendController._active_backend = "sla"

        AttentionBackendController.on_save_checkpoint(self.tmpdir, is_main_process=True)
        payload_path = os.path.join(self.tmpdir, "sla_attention.pt")
        self.assertTrue(os.path.exists(payload_path))

        AttentionBackendController._sla_state_store = {}
        AttentionBackendController._sla_settings = None
        AttentionBackendController.on_load_checkpoint(self.tmpdir)

        self.assertIn((128, "bf16"), AttentionBackendController._sla_state_store)
        restored = AttentionBackendController._sla_state_store[(128, "bf16")]
        self.assertTrue(torch.equal(restored["proj_l.weight"], sample_state["proj_l.weight"]))
        self.assertTrue(torch.equal(restored["proj_l.bias"], sample_state["proj_l.bias"]))
        self.assertEqual(
            AttentionBackendController._sla_settings,
            {
                "topk": 0.2,
                "feature_map": "softmax",
                "blkq": 64,
                "blkk": 64,
            },
        )

    def test_force_proj_fp32(self):
        class DummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_l = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

        module = DummyModule()
        AttentionBackendController._force_proj_fp32(module)
        self.assertEqual(module.proj_l.weight.dtype, torch.float32)
        self.assertEqual(module.proj_l.bias.dtype, torch.float32)

    def test_enable_diffusers_backend_context(self):
        if not attention_backend_module._DIFFUSERS_BACKEND_ALIASES:
            self.skipTest("Diffusers attention backend helpers unavailable in this environment.")

        config = type("Config", (object,), {"attention_mechanism": "native-math"})()
        AttentionBackendController.apply(config, AttentionPhase.TRAIN)
        self.assertEqual(AttentionBackendController._diffusers_backend_name, "native-math")
        AttentionBackendController.restore_default()
        self.assertIsNone(AttentionBackendController._diffusers_backend_name)

    def test_packed_backend_dispatches_fixed_qkvpacked(self):
        class DummyKernel:
            @staticmethod
            def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False):
                return qkv[:, :, 0]

        backend = PackedAttentionBackend("dummy", DummyKernel())
        qkv = torch.randn(2, 4, 3, 2, 8)

        output = backend.qkvpacked(qkv)

        self.assertTrue(torch.equal(output, qkv[:, :, 0]))

    def test_packed_backend_dispatches_varlen_qkvpacked_with_bool_mask(self):
        calls = {}

        class DummyKernel:
            @staticmethod
            def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False):
                return qkv[:, :, 0]

            @staticmethod
            def flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False
            ):
                calls["qkv"] = qkv
                calls["cu_seqlens"] = cu_seqlens
                calls["max_seqlen"] = max_seqlen
                return qkv[:, 0]

        backend = PackedAttentionBackend("dummy", DummyKernel())
        qkv = torch.arange(2 * 4 * 3 * 1 * 2, dtype=torch.float32).view(2, 4, 3, 1, 2)
        mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)

        output = backend.qkvpacked(qkv, attention_mask=mask)

        expected = torch.zeros(2, 4, 1, 2)
        expected[mask] = qkv[:, :, 0][mask]
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(torch.equal(calls["cu_seqlens"].cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertEqual(calls["max_seqlen"], 3)
        self.assertEqual(calls["qkv"].shape, (5, 3, 1, 2))

    def test_packed_backend_dispatches_varlen_unpacked_when_qkvpacked_unavailable(self):
        calls = {}

        class DummyKernel:
            @staticmethod
            def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False):
                return qkv[:, :, 0]

            @staticmethod
            def flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            ):
                calls["query"] = query
                calls["key"] = key
                calls["value"] = value
                calls["cu_seqlens_q"] = cu_seqlens_q
                calls["cu_seqlens_k"] = cu_seqlens_k
                calls["max_seqlen_q"] = max_seqlen_q
                calls["max_seqlen_k"] = max_seqlen_k
                return query

        backend = PackedAttentionBackend("dummy", DummyKernel())
        qkv = torch.arange(2 * 4 * 3 * 1 * 2, dtype=torch.float32).view(2, 4, 3, 1, 2)
        mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)

        output = backend.qkvpacked(qkv, attention_mask=mask)

        expected = torch.zeros(2, 4, 1, 2)
        expected[mask] = qkv[:, :, 0][mask]
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(torch.equal(calls["cu_seqlens_q"].cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(calls["cu_seqlens_k"].cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertEqual(calls["max_seqlen_q"], 3)
        self.assertEqual(calls["max_seqlen_k"], 3)
        self.assertEqual(calls["query"].shape, (5, 1, 2))
        self.assertTrue(torch.equal(calls["key"], qkv[:, :, 1][mask]))
        self.assertTrue(torch.equal(calls["value"], qkv[:, :, 2][mask]))

    def test_packed_backend_varlen_unpacked_supports_fa3_signature_without_dropout(self):
        calls = {}

        class DummyKernel:
            @staticmethod
            def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False):
                return qkv[:, :, 0]

            @staticmethod
            def flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                seqused_q=None,
                seqused_k=None,
                softmax_scale=None,
                causal=False,
            ):
                calls["seqused_q"] = seqused_q
                calls["seqused_k"] = seqused_k
                calls["softmax_scale"] = softmax_scale
                calls["causal"] = causal
                return value

        backend = PackedAttentionBackend("dummy-fa3", DummyKernel())
        qkv = torch.arange(2 * 4 * 3 * 1 * 2, dtype=torch.float32).view(2, 4, 3, 1, 2)
        mask = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)

        output = backend.qkvpacked(qkv, attention_mask=mask, causal=True, softmax_scale=0.5)

        expected = torch.zeros(2, 4, 1, 2)
        expected[mask] = qkv[:, :, 2][mask]
        self.assertTrue(torch.equal(output, expected))
        self.assertIsNone(calls["seqused_q"])
        self.assertIsNone(calls["seqused_k"])
        self.assertEqual(calls["softmax_scale"], 0.5)
        self.assertTrue(calls["causal"])

    def test_auto_packed_backend_prefers_fa2_hub_for_varlen_qkvpacked(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            self.assertEqual(
                attention_backend_module._select_packed_backend(None, require_varlen_qkvpacked=True),
                "flash2-hub",
            )
            self.assertEqual(attention_backend_module._select_packed_backend(None), "flash3-hub")

    def test_packed_backend_accepts_config_fa2_hub_varlen_alias(self):
        self.assertEqual(
            attention_backend_module._select_packed_backend("flash-attn-varlen-hub"),
            "flash-attn-varlen-hub",
        )
