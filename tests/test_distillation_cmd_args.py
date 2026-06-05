import unittest
from tempfile import NamedTemporaryFile

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
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
        for method in ("lcm", "dcm", "dmd", "perflow", "flow_dpo", "anyflow"):
            with self.subTest(method=method):
                args_list = _base_args() + [f"--distillation_method={method}", "--train_text_encoder"]

                with self.assertRaisesRegex(ValueError, "train_text_encoder"):
                    parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_distillation_text_encoder_guard_covers_registry_methods(self):
        for method in ("lcm", "dcm", "dmd", "perflow", "self_forcing", "flow_dpo", "anyflow"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "train_text_encoder"):
                    validate_distillation_text_encoder_training(method, True)

    def test_invalid_flow_timesteps_mode_rejected_at_startup(self):
        args_list = _base_args() + ["--flow_timesteps_mode=sequential"]

        with self.assertRaisesRegex(ValueError, "flow_timesteps_mode"):
            parse_cmdline_args(input_args=args_list, exit_on_error=True)

    def test_distillation_config_json_string_is_parsed(self):
        args_list = _base_args() + [
            "--distillation_method=anyflow",
            '--distillation_config={"anyflow":{"target_mode":"linear","teacher_rollout_steps":2}}',
        ]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=True)

        self.assertEqual(
            args.distillation_config,
            {"anyflow": {"target_mode": "linear", "teacher_rollout_steps": 2}},
        )

    def test_distillation_config_file_is_loaded(self):
        with NamedTemporaryFile("w", suffix=".json") as handle:
            handle.write('{"anyflow":{"target_mode":"linear","r_timestep_sampler":"zero"}}')
            handle.flush()

            args = parse_cmdline_args(
                input_args=_base_args() + ["--distillation_method=anyflow", f"--distillation_config={handle.name}"],
                exit_on_error=True,
            )

        self.assertEqual(
            args.distillation_config,
            {"anyflow": {"target_mode": "linear", "r_timestep_sampler": "zero"}},
        )

    def test_mapping_to_cli_args_preserves_distillation_config_mapping(self):
        cli_args = mapping_to_cli_args(
            {
                "distillation_method": "anyflow",
                "distillation_config": {"anyflow": {"target_mode": "linear"}},
            }
        )

        self.assertIn("--distillation_method=anyflow", cli_args)
        self.assertIn('--distillation_config={"anyflow": {"target_mode": "linear"}}', cli_args)


if __name__ == "__main__":
    unittest.main()
