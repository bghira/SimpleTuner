from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.training.default_settings.safety_check import safety_check
from simpletuner.helpers.training.reporting import normalize_report_to, report_to_contains, report_to_is_disabled

BASE_ARGS = [
    "--model_family=sdxl",
    "--output_dir=output",
    "--model_type=lora",
    "--optimizer=adamw_bf16",
    "--data_backend_config=config/multidatabackend.json",
]


class ReportToTests(unittest.TestCase):
    def test_parser_accepts_comma_separated_trackers(self):
        args = parse_cmdline_args(BASE_ARGS + ["--report_to=wandb,simpletuner"], exit_on_error=True)

        self.assertEqual(args.report_to, ["wandb", "simpletuner"])
        self.assertTrue(report_to_contains(args.report_to, "wandb"))
        self.assertTrue(report_to_contains(args.report_to, "simpletuner"))

    def test_parser_rejects_removed_all_value(self):
        with self.assertRaisesRegex(ValueError, "Unsupported --report_to value 'all'"):
            parse_cmdline_args(BASE_ARGS + ["--report_to=all"], exit_on_error=True)

    def test_parser_rejects_none_combined_with_tracker(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            parse_cmdline_args(BASE_ARGS + ["--report_to=none,wandb"], exit_on_error=True)

    def test_mapping_to_cli_args_preserves_report_to_list(self):
        cli_args = mapping_to_cli_args({"report_to": ["wandb", "simpletuner"]})

        self.assertEqual(cli_args, ["--report_to=wandb,simpletuner"])

    def test_normalize_report_to_deduplicates_and_detects_disabled(self):
        self.assertEqual(normalize_report_to("wandb, simpletuner, wandb"), ["wandb", "simpletuner"])
        self.assertEqual(normalize_report_to("none"), "none")
        self.assertTrue(report_to_is_disabled("none"))
        self.assertFalse(report_to_is_disabled(["wandb", "simpletuner"]))

    def test_safety_check_detects_wandb_in_multiple_trackers(self):
        args = SimpleNamespace(report_to=["wandb", "simpletuner"])

        with patch(
            "simpletuner.helpers.training.default_settings.safety_check.is_wandb_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(ImportError, "install wandb"):
                safety_check(args, accelerator=None)


if __name__ == "__main__":
    unittest.main()
