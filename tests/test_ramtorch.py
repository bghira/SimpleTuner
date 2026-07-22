import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from simpletuner.helpers.configuration import cmd_args
from simpletuner.helpers.utils import ramtorch as ramtorch_utils


class _StubLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, skip_init=False):
        super().__init__(in_features, out_features, bias=bias, device="cpu", dtype=dtype)
        self._device_arg = device
        self._skip_init = skip_init


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.block = nn.Sequential(nn.Linear(2, 2), nn.ReLU())


class RamTorchUtilsTests(unittest.TestCase):
    def _build_stub_imports(self, replace_all_fn):
        return {
            "Linear": _StubLinear,
            "replace_all": replace_all_fn,
            "broadcast_zero_params": None,
            "create_zero_param_groups": None,
            "setup_grad_sharding_hooks": None,
        }

    def test_replace_linear_with_target_patterns(self):
        model = _SimpleModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(
                model, device="cuda", target_patterns=["linear1", "block.*"]
            )

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 0)
        self.assertIsInstance(model.linear1, _StubLinear)
        self.assertIsInstance(model.block[0], _StubLinear)
        self.assertIsInstance(model.block[1], nn.ReLU)
        self.assertTrue(getattr(model.linear1.weight, "is_ramtorch", False))
        self.assertTrue(getattr(model.block[0].weight, "is_ramtorch", False))

    def test_replace_linear_uses_replace_all_when_no_patterns(self):
        model = _SimpleModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", target_patterns=None)

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 0)
        self.assertIsInstance(model.linear1, _StubLinear)
        self.assertIsInstance(model.block[0], _StubLinear)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_replace_linear_with_percent_50(self):
        """Test that percent=50 replaces only half (rounded up) of eligible layers."""

        class _LargerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 2)
                self.linear2 = nn.Linear(2, 2)
                self.linear3 = nn.Linear(2, 2)
                self.linear4 = nn.Linear(2, 2)

        model = _LargerModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            # percent=50 should replace 2 of 4 layers (ceil(4 * 0.5) = 2)
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", percent=50)

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 0)  # Should not use replace_all
        # First 2 layers should be replaced
        self.assertIsInstance(model.linear1, _StubLinear)
        self.assertIsInstance(model.linear2, _StubLinear)
        # Last 2 layers should remain as nn.Linear
        self.assertIsInstance(model.linear3, nn.Linear)
        self.assertIsInstance(model.linear4, nn.Linear)

    def test_replace_linear_with_percent_100(self):
        """Test that percent=100 (or None) replaces all layers."""
        model = _SimpleModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            # percent=100 should behave the same as percent=None (all layers)
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", percent=100)

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 0)

    def test_replace_torchao_int8_linear_preserves_weight_subclass(self):
        try:
            from torchao.prototype.quantized_training import int8_weight_only_quantized_training
            from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight
            from torchao.quantization import quantize_
        except ImportError as exc:
            self.skipTest(f"TorchAO int8 training quantization is unavailable: {exc}")

        model = nn.Sequential(nn.Linear(16, 16))
        quantize_(model, int8_weight_only_quantized_training())
        model.requires_grad_(False)
        original_weight_type = type(model[0].weight)

        with patch.object(
            ramtorch_utils,
            "ensure_available",
            return_value=self._build_stub_imports(lambda mod, device=None: None),
        ):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cpu")

        self.assertEqual(replaced, 1)
        self.assertIsInstance(model[0], _StubLinear)
        self.assertIs(type(model[0].weight), original_weight_type)
        self.assertIsInstance(model[0].weight, Int8QuantizedTrainingLinearWeight)
        self.assertTrue(getattr(model[0].weight, "is_ramtorch", False))

    def test_replace_quanto_qlinear_preserves_weight_subclass(self):
        try:
            from optimum.quanto import freeze, qint8, quantize
            from optimum.quanto.nn.qlinear import QLinear
            from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
        except ImportError as exc:
            self.skipTest(f"Quanto int8 quantization is unavailable: {exc}")

        model = nn.Sequential(nn.Linear(16, 16))
        quantize(model, weights=qint8)
        freeze(model)
        self.assertIsInstance(model[0], QLinear)

        with patch.object(
            ramtorch_utils,
            "ensure_available",
            return_value=self._build_stub_imports(lambda mod, device=None: None),
        ):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cpu")

        self.assertEqual(replaced, 1)
        self.assertIsInstance(model[0], _StubLinear)
        self.assertIsInstance(model[0].weight, WeightQBytesTensor)
        self.assertTrue(getattr(model[0].weight, "is_ramtorch", False))

    def test_torchao_int8_ramtorch_backward_uses_dense_weight_view(self):
        try:
            from torchao.prototype.quantized_training import int8_weight_only_quantized_training
            from torchao.quantization import quantize_
        except ImportError as exc:
            self.skipTest(f"TorchAO int8 training quantization is unavailable: {exc}")

        from simpletuner.helpers.ramtorch.modules.linear import Linear as RamTorchLinear

        linear = nn.Linear(16, 16)
        quantize_(linear, int8_weight_only_quantized_training())

        ramtorch_linear = RamTorchLinear(16, 16, bias=True, device="cpu", dtype=linear.weight.dtype, skip_init=True)
        ramtorch_linear.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        ramtorch_linear.weight.is_ramtorch = True
        ramtorch_linear.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        ramtorch_linear.bias.is_ramtorch = True

        x = torch.randn(2, 16, requires_grad=True)
        loss = ramtorch_linear(x).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_manual_quantization_defers_base_ramtorch_until_after_quantization(self):
        from simpletuner.helpers.models.common import ModelFoundation

        model = SimpleNamespace(
            config=SimpleNamespace(
                ramtorch=True,
                quantize_via="accelerator",
                base_model_precision="int8-torchao",
            ),
            _ramtorch_enabled=lambda: True,
        )

        self.assertTrue(ModelFoundation._ramtorch_base_deferred_until_after_quantization(model))

        model.config.quantize_via = "pipeline"
        self.assertFalse(ModelFoundation._ramtorch_base_deferred_until_after_quantization(model))

        model.config.quantize_via = "accelerator"
        model.config.base_model_precision = "no_change"
        self.assertFalse(ModelFoundation._ramtorch_base_deferred_until_after_quantization(model))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_replace_linear_with_percent_0(self):
        """Test that percent=0 replaces no layers."""

        class _LargerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 2)
                self.linear2 = nn.Linear(2, 2)

        model = _LargerModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", percent=0)

        self.assertEqual(replaced, 0)
        self.assertEqual(replace_all_calls["count"], 0)
        self.assertIsInstance(model.linear1, nn.Linear)
        self.assertIsInstance(model.linear2, nn.Linear)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_replace_linear_with_percent_and_patterns(self):
        """Test that percent works together with target_patterns."""

        class _NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_a = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
                self.block_b = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        model = _NestedModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            # Only target block_a layers (2 layers), replace 50% = 1 layer
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(
                model, device="cuda", target_patterns=["block_a.*"], percent=50
            )

        self.assertEqual(replaced, 1)
        self.assertEqual(replace_all_calls["count"], 0)
        # First layer in block_a should be replaced
        self.assertIsInstance(model.block_a[0], _StubLinear)
        # Second layer in block_a should remain as nn.Linear
        self.assertIsInstance(model.block_a[1], nn.Linear)
        # block_b layers should all remain as nn.Linear (not in target patterns)
        self.assertIsInstance(model.block_b[0], nn.Linear)
        self.assertIsInstance(model.block_b[1], nn.Linear)

    def test_mark_ddp_ignore_params_marks_ramtorch_names(self):
        class _IgnoreModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = nn.Linear(2, 2)
                self.linear_b = nn.Linear(2, 2)
                setattr(self.linear_b.weight, "is_ramtorch", True)
                if self.linear_b.bias is not None:
                    setattr(self.linear_b.bias, "is_ramtorch", True)

        model = _IgnoreModel()
        ignored = ramtorch_utils.mark_ddp_ignore_params(model)
        self.assertEqual(ignored, 2)
        ignore_set = getattr(model, "_ddp_params_and_buffers_to_ignore", set())
        self.assertIn("linear_b.weight", ignore_set)
        self.assertIn("linear_b.bias", ignore_set)

    def test_prefetch_hooks_follow_ramtorch_module_order(self):
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        StateTracker.reset_ramtorch_prefetch_orders()

        class _PrefetchModule(nn.Module):
            is_ramtorch = True

            def __init__(self, label, calls):
                super().__init__()
                self.label = label
                self.calls = calls

            def prefetch_forward(self):
                self.calls.append(self.label)
                return True

            def forward(self, x):
                return x

        calls = []
        model = nn.Sequential(
            _PrefetchModule("first", calls),
            nn.ReLU(),
            _PrefetchModule("second", calls),
            _PrefetchModule("third", calls),
        )

        hooks = add_ramtorch_prefetch_hooks(model, component_label="sequential-test")
        try:
            self.assertGreaterEqual(len(hooks), 2)
            model(torch.ones(1))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(calls, ["second", "third"])

    def test_prefetch_hooks_learn_actual_execution_order(self):
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        class _PrefetchModule(nn.Module):
            is_ramtorch = True

            def __init__(self, label, calls):
                super().__init__()
                self.label = label
                self.calls = calls

            def prefetch_forward(self):
                self.calls.append(self.label)
                return True

            def forward(self, x):
                return x

        class _OutOfTraversalOrder(nn.Module):
            def __init__(self, calls):
                super().__init__()
                self.first = _PrefetchModule("first", calls)
                self.second = _PrefetchModule("second", calls)
                self.third = _PrefetchModule("third", calls)

            def forward(self, x):
                x = self.first(x)
                x = self.third(x)
                return self.second(x)

        calls = []
        model = _OutOfTraversalOrder(calls)
        StateTracker.reset_ramtorch_prefetch_orders()

        with patch.dict(
            "os.environ",
            {
                "SIMPLETUNER_RAMTORCH_PREFETCH_LEARNED_MIN_OBSERVATIONS": "2",
                "SIMPLETUNER_RAMTORCH_PREFETCH_LEARNED_MIN_CONFIDENCE": "0.5",
            },
        ):
            hooks = add_ramtorch_prefetch_hooks(model, component_label="out-of-order-test")
            try:
                for _ in range(4):
                    model(torch.ones(1))
            finally:
                for hook in hooks:
                    hook.remove()

        self.assertEqual(calls[:4], ["second", "third", "second", "third"])
        self.assertEqual(calls[4:], ["third", "second", "third", "second"])
        self.assertEqual(
            StateTracker.get_ramtorch_prefetch_successor("out-of-order-test", "first"),
            "third",
        )
        self.assertEqual(
            StateTracker.get_ramtorch_prefetch_successor("out-of-order-test", "third"),
            "second",
        )
        self.assertTrue(StateTracker.ramtorch_prefetch_disabled("out-of-order-test", "second"))

    def test_prefetch_hooks_preserve_tail_for_backward(self):
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        class _PrefetchModule(nn.Module):
            is_ramtorch = True

            def __init__(self, label, prefetch_calls, preserve_calls):
                super().__init__()
                self.label = label
                self.prefetch_calls = prefetch_calls
                self.preserve_calls = preserve_calls

            def prefetch_forward(self):
                self.prefetch_calls.append(self.label)
                return True

            def preserve_forward_for_backward(self, *, max_entries=2, max_bytes=0):
                self.preserve_calls.append((self.label, max_entries, max_bytes))
                return True

            def ramtorch_forward_bytes(self):
                return 1

            def forward(self, x):
                return x

        prefetch_calls = []
        preserve_calls = []
        model = nn.Sequential(
            _PrefetchModule("first", prefetch_calls, preserve_calls),
            _PrefetchModule("second", prefetch_calls, preserve_calls),
            _PrefetchModule("third", prefetch_calls, preserve_calls),
            _PrefetchModule("fourth", prefetch_calls, preserve_calls),
        )
        StateTracker.reset_ramtorch_prefetch_orders()

        with patch.dict("os.environ", {"SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MAX_ENTRIES": "2"}):
            hooks = add_ramtorch_prefetch_hooks(model, component_label="tail-preserve-test")
            try:
                model(torch.ones(1))
            finally:
                for hook in hooks:
                    hook.remove()

        self.assertEqual(prefetch_calls, ["second", "third", "fourth"])
        self.assertEqual(
            preserve_calls,
            [
                ("third", 2, 0),
                ("fourth", 2, 0),
            ],
        )

    def test_prefetch_hooks_skip_backward_preserve_when_free_vram_is_low(self):
        from simpletuner.helpers.ramtorch import profiling as ramtorch_profile
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        class _PrefetchModule(nn.Module):
            is_ramtorch = True
            device = torch.device("cuda", 0)

            def __init__(self, label, prefetch_calls, preserve_calls):
                super().__init__()
                self.label = label
                self.prefetch_calls = prefetch_calls
                self.preserve_calls = preserve_calls

            def prefetch_forward(self):
                self.prefetch_calls.append(self.label)
                return True

            def preserve_forward_for_backward(self, *, max_entries=2, max_bytes=0):
                self.preserve_calls.append((self.label, max_entries, max_bytes))
                return True

            def ramtorch_forward_bytes(self):
                return 1024

            def forward(self, x):
                return x

        prefetch_calls = []
        preserve_calls = []
        model = nn.Sequential(
            _PrefetchModule("first", prefetch_calls, preserve_calls),
            _PrefetchModule("second", prefetch_calls, preserve_calls),
            _PrefetchModule("third", prefetch_calls, preserve_calls),
        )
        StateTracker.reset_ramtorch_prefetch_orders()
        ramtorch_profile.reset_for_new_run()

        env = {
            "SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MAX_ENTRIES": "2",
            "SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MIN_FREE_RATIO": "0.20",
        }
        with (
            patch.dict("os.environ", env),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(10, 100)),
            patch("torch.cuda.device"),
        ):
            hooks = add_ramtorch_prefetch_hooks(model, component_label="tail-preserve-vram-test")
            try:
                model(torch.ones(1))
            finally:
                for hook in hooks:
                    hook.remove()

        self.assertEqual(prefetch_calls, ["second", "third"])
        self.assertEqual(preserve_calls, [])
        counters = ramtorch_profile.snapshot()["counters"]
        self.assertEqual(counters["backward_preserve_skipped_policy"], 2)
        self.assertEqual(counters["bytes_backward_preserve_skipped_policy"], 2048)

    def test_ramtorch_prefetch_order_state_round_trips(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        with tempfile.TemporaryDirectory() as tmp_dir:
            StateTracker.reset_ramtorch_prefetch_orders()
            StateTracker.configure_ramtorch_prefetch_component(
                "transformer",
                ["layers.0", "layers.2"],
            )
            for _ in range(3):
                StateTracker.record_ramtorch_prefetch_transition(
                    "transformer",
                    "layers.0",
                    "layers.2",
                )
            StateTracker.save_ramtorch_prefetch_orders(tmp_dir)

            StateTracker.reset_ramtorch_prefetch_orders()
            StateTracker.load_ramtorch_prefetch_orders(tmp_dir)

            self.assertEqual(
                StateTracker.get_ramtorch_prefetch_successor("transformer", "layers.0"),
                "layers.2",
            )

    def test_ramtorch_prefetch_order_resets_when_topology_changes(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        StateTracker.reset_ramtorch_prefetch_orders()
        StateTracker.configure_ramtorch_prefetch_component(
            "transformer",
            ["layers.0", "layers.2"],
        )
        for _ in range(3):
            StateTracker.record_ramtorch_prefetch_transition(
                "transformer",
                "layers.0",
                "layers.2",
            )
        self.assertEqual(
            StateTracker.get_ramtorch_prefetch_successor("transformer", "layers.0"),
            "layers.2",
        )

        StateTracker.configure_ramtorch_prefetch_component(
            "transformer",
            ["layers.0", "layers.1", "layers.2"],
        )

        self.assertIsNone(StateTracker.get_ramtorch_prefetch_successor("transformer", "layers.0"))
        component = StateTracker.get_ramtorch_prefetch_component("transformer")
        self.assertEqual(component.get("observations"), 0)
        self.assertEqual(component.get("successors"), {})

    def test_prefetch_hooks_decline_when_ramtorch_module_lacks_prefetch(self):
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        StateTracker.reset_ramtorch_prefetch_orders()

        class _MissingPrefetchModule(nn.Module):
            is_ramtorch = True

            def forward(self, x):
                return x

        model = nn.Sequential(_MissingPrefetchModule(), _MissingPrefetchModule())
        self.assertEqual(add_ramtorch_prefetch_hooks(model), [])

    def test_prefetch_hooks_decline_when_policy_is_sync(self):
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        StateTracker.reset_ramtorch_prefetch_orders()

        class _PrefetchModule(nn.Module):
            is_ramtorch = True

            def prefetch_forward(self):
                return True

            def forward(self, x):
                return x

        model = nn.Sequential(_PrefetchModule(), _PrefetchModule())
        with patch.dict("os.environ", {"SIMPLETUNER_RAMTORCH_PREFETCH_POLICY": "sync"}):
            self.assertEqual(add_ramtorch_prefetch_hooks(model), [])

    def test_bundled_linear_skips_frozen_weight_gradients(self):
        from simpletuner.helpers.ramtorch.modules.linear import Linear

        linear = Linear(3, 2, device="cpu")
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        x = torch.randn(4, 3, requires_grad=True)
        linear(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(linear.weight.grad)
        if linear.bias is not None:
            self.assertIsNone(linear.bias.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_bundled_linear_reuses_preserved_forward_weight_in_backward(self):
        from simpletuner.helpers.ramtorch import profiling as ramtorch_profile
        from simpletuner.helpers.ramtorch.modules.linear import Linear
        from simpletuner.helpers.ramtorch_extensions import add_ramtorch_prefetch_hooks
        from simpletuner.helpers.training.state_tracker import StateTracker

        device = torch.device("cuda", torch.cuda.current_device())
        model = nn.Sequential(
            Linear(4, 4, device=device),
            nn.SiLU(),
            Linear(4, 4, device=device),
            nn.SiLU(),
            Linear(4, 2, device=device),
        )
        StateTracker.reset_ramtorch_prefetch_orders()
        ramtorch_profile.reset_for_new_run()

        with patch.dict("os.environ", {"SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MAX_ENTRIES": "2"}):
            hooks = add_ramtorch_prefetch_hooks(model, component_label="linear-tail-preserve-test")
            try:
                x = torch.randn(8, 4, device=device, requires_grad=True)
                model(x).sum().backward()
                torch.cuda.synchronize(device)
            finally:
                for hook in hooks:
                    hook.remove()

        counters = ramtorch_profile.snapshot()["counters"]
        self.assertGreaterEqual(counters["backward_preserve_retained"], 1)
        self.assertGreaterEqual(counters["backward_preserve_hits"], 1)
        self.assertGreater(counters["bytes_backward_preserve_hit"], 0)


class RamTorchConfigTests(unittest.TestCase):
    def _base_args(self, tmp_dir: str) -> list[str]:
        data_config = Path(tmp_dir) / "backend.json"
        data_config.write_text("{}")
        return [
            f"--output_dir={tmp_dir}",
            "--model_type=full",
            "--optimizer=adamw_bf16",
            f"--data_backend_config={data_config}",
            "--model_family=sdxl",
            "--pretrained_model_name_or_path=stub-model",
        ]

    def test_group_offload_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_args = self._base_args(tmp_dir)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    cmd_args.parse_cmdline_args(["--ramtorch", "--enable_group_offload", *base_args], exit_on_error=True)

    def test_set_grads_to_none_forced_for_ramtorch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_args = self._base_args(tmp_dir)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                args = cmd_args.parse_cmdline_args(["--ramtorch", *base_args], exit_on_error=True)
        self.assertTrue(getattr(args, "set_grads_to_none", False))


class RamTorchTextEncoderEmbeddingTests(unittest.TestCase):
    """Test that embedding layers work correctly with ramtorch text encoders."""

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(),
        "CUDA not available",
    )
    def test_embedding_device_mismatch_with_ramtorch(self):
        """
        When ramtorch is applied to a text encoder model, only nn.Linear layers
        are converted to RamTorch (kept in CPU RAM). The nn.Embedding layers
        remain as regular PyTorch modules on CPU. If the caller sends input_ids
        to GPU, the embedding lookup fails with a device mismatch error.

        This test reproduces the issue seen in Chroma/T5 encoding:
        RuntimeError: Expected all tensors to be on the same device,
        but got index is on cuda:0, different from other tensors on cpu
        """
        import torch
        import torch.nn as nn

        # Simple model mimicking T5 structure: embedding + linear layers
        class SimpleTextEncoder(nn.Module):
            def __init__(self, vocab_size=100, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
                self.linear1 = nn.Linear(embed_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, embed_dim)

            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = SimpleTextEncoder()

        # Simulate what ramtorch does: convert Linear layers but not Embedding
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = {
            "Linear": _StubLinear,
            "replace_all": fake_replace_all,
            "broadcast_zero_params": None,
            "create_zero_param_groups": None,
            "setup_grad_sharding_hooks": None,
        }

        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", target_patterns=None)

        # Model stays on CPU (ramtorch doesn't move the model)
        # This is the state after ramtorch is applied to text encoders

        # Now simulate what the pipeline does: send input_ids to GPU
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        input_ids_cuda = input_ids.to("cuda")

        # This should fail because embed_tokens weights are on CPU
        # but input_ids are on CUDA
        with self.assertRaises(RuntimeError) as ctx:
            model(input_ids_cuda)

        self.assertIn("device", str(ctx.exception).lower())

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(),
        "CUDA not available",
    )
    def test_embedding_moved_to_gpu_with_ramtorch(self):
        """
        After ramtorch is applied to text encoder, embedding layers should be
        moved to GPU so that input_ids (which come from GPU) work correctly.
        This tests the fix for the device mismatch issue.
        """
        import torch
        import torch.nn as nn

        class SimpleTextEncoder(nn.Module):
            def __init__(self, vocab_size=100, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
                self.linear1 = nn.Linear(embed_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, embed_dim)

            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = SimpleTextEncoder()

        # Verify embedding starts on CPU
        self.assertEqual(model.embed_tokens.weight.device.type, "cpu")

        # Simulate ramtorch being applied - mark Linear layers as ramtorch
        # (In real usage, replace_linear_layers_with_ramtorch does this)
        for name, child in model.named_modules():
            if isinstance(child, nn.Linear):
                setattr(child.weight, "is_ramtorch", True)
                if child.bias is not None:
                    setattr(child.bias, "is_ramtorch", True)

        # Apply the fix: move non-ramtorch modules to GPU
        moved = ramtorch_utils.move_embeddings_to_device(model, "cuda")

        # Verify one module was moved (only the embedding, not the ramtorch linears)
        self.assertEqual(moved, 1)

        # Verify embedding is now on GPU
        self.assertEqual(model.embed_tokens.weight.device.type, "cuda")

        # Verify Linear layers stayed on CPU (they're ramtorch)
        self.assertEqual(model.linear1.weight.device.type, "cpu")
        self.assertEqual(model.linear2.weight.device.type, "cpu")

        # Test that embedding lookup works with GPU input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        input_ids_cuda = input_ids.to("cuda")

        # Embedding lookup should work (no device mismatch)
        embed_output = model.embed_tokens(input_ids_cuda)
        self.assertEqual(embed_output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
