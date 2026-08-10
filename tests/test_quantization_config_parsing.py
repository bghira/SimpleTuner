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
