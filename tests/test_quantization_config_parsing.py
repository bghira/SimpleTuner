import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.training.state_tracker import StateTracker


def _base_args():
    return [
        "--model_family=pixart",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


class TestQuantizationConfigParsing(unittest.TestCase):
    def setUp(self):
        self.original_args = StateTracker.get_args()
        self.original_sdnq_compile = os.environ.get("SDNQ_USE_TORCH_COMPILE")
        from simpletuner.helpers.training import sdnq_compile

        self.sdnq_compile = sdnq_compile
        self.original_configured_mode = sdnq_compile._CONFIGURED_MODE
        self.original_import_warning_emitted = sdnq_compile._IMPORT_WARNING_EMITTED
        sdnq_compile._CONFIGURED_MODE = None
        sdnq_compile._IMPORT_WARNING_EMITTED = False

    def tearDown(self):
        StateTracker.set_args(self.original_args)
        if self.original_sdnq_compile is None:
            os.environ.pop("SDNQ_USE_TORCH_COMPILE", None)
        else:
            os.environ["SDNQ_USE_TORCH_COMPILE"] = self.original_sdnq_compile
        self.sdnq_compile._CONFIGURED_MODE = self.original_configured_mode
        self.sdnq_compile._IMPORT_WARNING_EMITTED = self.original_import_warning_emitted

    def test_pipeline_quantize_via_rejects_manual_precision(self):
        args_list = _base_args() + ["--quantize_via=pipeline", "--base_model_precision=int8-sdnq"]
        with self.assertRaises(ValueError):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_quantization_config_requires_pipeline_compatible_base(self):
        qconfig = json.dumps({"unet": {"load_in_4bit": True}})
        args_list = _base_args() + [
            "--base_model_precision=int8-sdnq",
            f"--quantization_config={qconfig}",
        ]
        with self.assertRaises(ValueError):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_quantization_config_json_parses(self):
        qconfig = json.dumps({"unet": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}})
        args_list = _base_args() + [
            "--quantize_via=pipeline",
            "--base_model_precision=no_change",
            f"--quantization_config={qconfig}",
        ]
        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)
        self.assertIsInstance(args.quantization_config, dict)
        self.assertIn("unet", args.quantization_config)
        self.assertEqual(args.quantization_config["unet"]["bnb_4bit_quant_type"], "nf4")

    def test_pipeline_quantize_via_allows_torchao_preset(self):
        args_list = _base_args() + ["--quantize_via=pipeline", "--base_model_precision=int8-torchao"]
        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)
        self.assertEqual(args.base_model_precision, "int8-torchao")

    def test_pipeline_quantize_via_allows_quanto_preset(self):
        args_list = _base_args() + ["--quantize_via=pipeline", "--base_model_precision=int8-quanto"]
        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)
        self.assertEqual(args.base_model_precision, "int8-quanto")

    def test_sdnq_advanced_options_parse(self):
        args_list = _base_args() + [
            "--base_model_precision=fp8-sdnq",
            "--sdnq_quantized_matmul_dtype=float8_e4m3fn",
            "--sdnq_group_size=-1",
            "--sdnq_use_quantized_matmul=true",
            "--sdnq_compile_mode=compile",
            "--sdnq_modules_to_not_convert=proj_out,norm_out",
            '--sdnq_modules_dtype_dict={"minimum_6bit":["x_embedder"]}',
            '--sdnq_modules_quant_config={"attn":{"group_size":-1}}',
        ]
        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)
        self.assertEqual(args.base_model_precision, "fp8-sdnq")
        self.assertEqual(args.sdnq_quantized_matmul_dtype, "float8_e4m3fn")
        self.assertEqual(args.sdnq_group_size, -1)
        self.assertTrue(args.sdnq_use_quantized_matmul)
        self.assertEqual(args.sdnq_compile_mode, "compile")
        self.assertEqual(args.sdnq_modules_to_not_convert, ["proj_out", "norm_out"])
        self.assertEqual(args.sdnq_modules_dtype_dict, {"minimum_6bit": ["x_embedder"]})
        self.assertEqual(args.sdnq_modules_quant_config, {"attn": {"group_size": -1}})

    def test_sdnq_fake_mode_detection_is_best_effort(self):
        from simpletuner.helpers.training.sdnq_workarounds import _detect_fake_mode

        guards = ModuleType("torch._guards")
        guards.detect_fake_mode = lambda _tensor: (_ for _ in ()).throw(RuntimeError("fake-mode unavailable"))
        with patch.dict(sys.modules, {"torch._guards": guards}):
            self.assertIsNone(_detect_fake_mode(object()))

    def test_sdnq_hadamard_cache_does_not_leak_real_tensor_into_fake_mode(self):
        import sdnq.dequantizer as sdnq_dequantizer
        import sdnq.quant_utils as sdnq_quant_utils
        import torch
        from sdnq.training.tensor import SDNQTensor
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        from simpletuner.helpers.training.sdnq_workarounds import _PATCHED_FROM_FLOAT_ATTR, apply_sdnq_workarounds

        original_get_hadamard = sdnq_quant_utils.get_hadamard
        original_dequantizer_get_hadamard = sdnq_dequantizer.get_hadamard
        original_from_float = SDNQTensor.from_float
        original_from_float_patched = getattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, False)
        cache_key = (4, torch.device("cpu"), torch.float32)
        original_cached = sdnq_quant_utils.HADAMARD_MATRIX_CACHE.get(cache_key)
        try:
            sdnq_quant_utils.HADAMARD_MATRIX_CACHE[cache_key] = torch.eye(4)
            apply_sdnq_workarounds()
            with FakeTensorMode():
                result = sdnq_quant_utils.get_hadamard(4, dtype=torch.float32, device=torch.device("cpu"))
            self.assertIsInstance(result, FakeTensor)

            sentinel = object()
            with (
                patch.object(torch.compiler, "is_compiling", return_value=True),
                patch.object(sdnq_quant_utils, "build_hadamard", return_value=sentinel) as build,
            ):
                self.assertIs(sdnq_quant_utils.get_hadamard(4, dtype=torch.float32, device=torch.device("cpu")), sentinel)
            build.assert_called_once_with(4, dtype=torch.float32, device=torch.device("cpu"))
        finally:
            sdnq_quant_utils.get_hadamard = original_get_hadamard
            sdnq_dequantizer.get_hadamard = original_dequantizer_get_hadamard
            SDNQTensor.from_float = staticmethod(original_from_float)
            setattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, original_from_float_patched)
            if original_cached is None:
                sdnq_quant_utils.HADAMARD_MATRIX_CACHE.pop(cache_key, None)
            else:
                sdnq_quant_utils.HADAMARD_MATRIX_CACHE[cache_key] = original_cached

    def test_sdnq_installs_triton_allocator_only_for_null_default(self):
        from simpletuner.helpers.training.sdnq_workarounds import _TRITON_ALLOCATOR_ATTR, _install_triton_allocator

        class NullAllocator:
            pass

        state = SimpleNamespace(current=NullAllocator(), installed=None)
        allocation_module = ModuleType("triton.runtime._allocation")
        allocation_module.NullAllocator = NullAllocator
        allocation_module._allocator = SimpleNamespace(get=lambda: state.current)
        runtime_module = ModuleType("triton.runtime")
        runtime_module._allocation = allocation_module
        triton_module = ModuleType("triton")
        triton_module.runtime = runtime_module
        triton_module.set_allocator = lambda allocator: setattr(state, "installed", allocator)

        with patch.dict(
            sys.modules,
            {
                "triton": triton_module,
                "triton.runtime": runtime_module,
                "triton.runtime._allocation": allocation_module,
            },
        ):
            self.assertTrue(_install_triton_allocator())
            self.assertTrue(getattr(state.installed, _TRITON_ALLOCATOR_ATTR))
            state.current = object()
            self.assertFalse(_install_triton_allocator())

    def test_sdnq_scaled_mm_installs_allocator_in_execution_context(self):
        from simpletuner.helpers.training.sdnq_workarounds import (
            _PATCHED_SCALED_MM_ATTR,
            _patch_sdnq_scaled_mm_allocator_context,
        )

        calls = []
        custom_op = SimpleNamespace(_backend_fns={None: lambda value: calls.append(("backend", value)) or value})
        kernels_module = ModuleType("sdnq.kernels")
        scaled_mm_module = ModuleType("sdnq.kernels.triton_scaled_mm")
        scaled_mm_module.sdnq_scaled_mm = custom_op

        with (
            patch.dict(
                sys.modules,
                {
                    "sdnq.kernels": kernels_module,
                    "sdnq.kernels.triton_scaled_mm": scaled_mm_module,
                },
            ),
            patch("simpletuner.helpers.training.sdnq_workarounds._install_triton_allocator") as install,
        ):
            self.assertTrue(_patch_sdnq_scaled_mm_allocator_context())
            self.assertFalse(_patch_sdnq_scaled_mm_allocator_context())
            self.assertEqual(custom_op._backend_fns[None](17), 17)

        install.assert_called_once_with()
        self.assertEqual(calls, [("backend", 17)])
        self.assertTrue(getattr(custom_op._backend_fns[None], _PATCHED_SCALED_MM_ATTR))

    def test_sdnq_compile_mode_sets_env_before_import(self):
        from simpletuner.helpers.training.sdnq_compile import configure_sdnq_compile_mode

        StateTracker.set_args(SimpleNamespace(sdnq_compile_mode="eager"))
        os.environ.pop("SDNQ_USE_TORCH_COMPILE", None)
        with patch.dict("sys.modules", {"sdnq.common": None}, clear=False):
            del sys.modules["sdnq.common"]
            configure_sdnq_compile_mode()

        self.assertEqual(os.environ["SDNQ_USE_TORCH_COMPILE"], "0")

        StateTracker.set_args(SimpleNamespace(sdnq_compile_mode="compile"))
        configure_sdnq_compile_mode()

        self.assertEqual(os.environ["SDNQ_USE_TORCH_COMPILE"], "1")

    def test_sdnq_compile_mode_defaults_when_config_lacks_attribute(self):
        from simpletuner.helpers.training.sdnq_compile import configure_sdnq_compile_mode

        StateTracker.set_args(Mock())
        os.environ.pop("SDNQ_USE_TORCH_COMPILE", None)

        configure_sdnq_compile_mode()

        self.assertNotIn("SDNQ_USE_TORCH_COMPILE", os.environ)

    def test_sdnq_compile_mode_does_not_warn_after_prior_configuration(self):
        from simpletuner.helpers.training import sdnq_compile

        os.environ.pop("SDNQ_USE_TORCH_COMPILE", None)
        with patch.dict("sys.modules", {"sdnq.common": None}, clear=False):
            del sys.modules["sdnq.common"]
            sdnq_compile.configure_sdnq_compile_mode("compile")

        with (
            patch.dict("sys.modules", {"sdnq.common": ModuleType("sdnq.common")}, clear=False),
            patch.object(sdnq_compile.logger, "warning") as warning,
        ):
            sdnq_compile.configure_sdnq_compile_mode("compile")

        warning.assert_not_called()

    def test_convrot_loader_applies_sdnq_compile_mode_before_import(self):
        from simpletuner.helpers.models.z_image import quantized_loading

        class Dummy:
            pass

        dequantizer_module = ModuleType("sdnq.dequantizer")
        dequantizer_module.SDNQDequantizer = Dummy
        layers_module = ModuleType("sdnq.layers")
        layers_module.get_sdnq_wrapper_class = lambda module, forward: module
        forward_module = ModuleType("sdnq.training.forward")
        forward_module.get_forward_func = lambda *args, **kwargs: None
        tensor_module = ModuleType("sdnq.training.tensor")
        tensor_module.SDNQTensor = Dummy
        training_module = ModuleType("sdnq.training")
        sdnq_module = ModuleType("sdnq")

        modules = {
            "sdnq": sdnq_module,
            "sdnq.dequantizer": dequantizer_module,
            "sdnq.layers": layers_module,
            "sdnq.training": training_module,
            "sdnq.training.forward": forward_module,
            "sdnq.training.tensor": tensor_module,
        }
        with (
            patch.dict("sys.modules", modules),
            patch("simpletuner.helpers.training.sdnq_compile.configure_sdnq_compile_mode") as configure,
        ):
            quantized_loading._load_sdnq_training_symbols()

        configure.assert_called_once_with()

    def test_sdnq_compile_mode_configures_during_parse(self):
        os.environ.pop("SDNQ_USE_TORCH_COMPILE", None)
        args = parse_cmdline_args(
            input_args=_base_args() + ["--base_model_precision=int8-sdnq", "--sdnq_compile_mode=eager"],
            exit_on_error=False,
        )

        self.assertEqual(args.sdnq_compile_mode, "eager")
        self.assertEqual(os.environ["SDNQ_USE_TORCH_COMPILE"], "0")
