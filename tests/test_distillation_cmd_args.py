import unittest

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.distillation.common import validate_distillation_text_encoder_training


def _base_args():
    return [
        "--model_family=pixart",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


class DistillationCmdArgsTests(unittest.TestCase):
    def test_distillation_rejects_text_encoder_training(self):
        for method in ("lcm", "dcm", "dmd", "perflow", "flow_dpo"):
            with self.subTest(method=method):
                args_list = _base_args() + [f"--distillation_method={method}", "--train_text_encoder"]

                with self.assertRaisesRegex(ValueError, "train_text_encoder"):
                    parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_distillation_text_encoder_guard_covers_registry_methods(self):
        for method in ("lcm", "dcm", "dmd", "perflow", "self_forcing", "flow_dpo"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "train_text_encoder"):
                    validate_distillation_text_encoder_training(method, True)

    def test_invalid_flow_timesteps_mode_rejected_at_startup(self):
        args_list = _base_args() + ["--flow_timesteps_mode=sequential"]

        with self.assertRaisesRegex(ValueError, "flow_timesteps_mode"):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)


if __name__ == "__main__":
    unittest.main()
