import unittest

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args


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
        args_list = _base_args() + ["--distillation_method=flow_dpo", "--train_text_encoder"]

        with self.assertRaisesRegex(ValueError, "train_text_encoder"):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)


if __name__ == "__main__":
    unittest.main()
