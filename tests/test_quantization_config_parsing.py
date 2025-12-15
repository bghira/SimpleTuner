import json
import unittest

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args


def _base_args():
    return [
        "--model_family=pixart_sigma",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


class TestQuantizationConfigParsing(unittest.TestCase):
    def test_pipeline_quantize_via_rejects_manual_precision(self):
        args_list = _base_args() + ["--quantize_via=pipeline", "--base_model_precision=int8-quanto"]
        with self.assertRaises(ValueError):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_quantization_config_requires_pipeline_compatible_base(self):
        qconfig = json.dumps({"unet": {"load_in_4bit": True}})
        args_list = _base_args() + [
            "--base_model_precision=int8-quanto",
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
