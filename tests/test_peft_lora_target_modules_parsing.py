"""Test peft_lora_target_modules parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args


def _base_args():
    return [
        "--model_family=pixart",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


class TestPeftLoraTargetModulesParsing(unittest.TestCase):
    def test_inline_target_modules_json(self):
        targets = ["to_k", "to_q", "to_v", "to_out.0"]
        config_json = json.dumps(targets)
        args_list = _base_args() + [f"--peft_lora_target_modules={config_json}"]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        self.assertIsNotNone(args)
        self.assertEqual(args.peft_lora_target_modules, targets)

    def test_file_target_modules_json(self):
        targets = ["to_k", "to_q"]
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            json.dump(targets, handle)
            temp_path = handle.name
        try:
            args_list = _base_args() + [f"--peft_lora_target_modules={temp_path}"]
            args = parse_cmdline_args(input_args=args_list, exit_on_error=True)
            self.assertEqual(args.peft_lora_target_modules, targets)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_target_modules_json(self):
        config_json = json.dumps({"target_modules": ["to_k"]})
        args_list = _base_args() + [f"--peft_lora_target_modules={config_json}"]

        with self.assertRaises(ValueError) as context:
            parse_cmdline_args(input_args=args_list, exit_on_error=True)

        self.assertIn("peft_lora_target_modules", str(context.exception))


if __name__ == "__main__":
    unittest.main()
