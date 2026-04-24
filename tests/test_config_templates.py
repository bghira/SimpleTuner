import unittest
from unittest.mock import patch

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.configuration.template_vars import render_modelspec_comment
from simpletuner.helpers.data_backend.factory import fill_variables_in_config_paths


class TestConfigTemplateExpansion(unittest.TestCase):
    def test_mapping_to_cli_args_expands_env_in_string_values(self):
        with patch.dict("os.environ", {"DATASET_CONFIG_NAME": "multidatabackend-dev"}, clear=False):
            cli_args = mapping_to_cli_args(
                {
                    "data_backend_config": "config/{env:DATASET_CONFIG_NAME}.json",
                    "output_dir": "/tmp/output",
                }
            )

        self.assertIn("--data_backend_config=config/multidatabackend-dev.json", cli_args)

    def test_fill_variables_in_config_paths_expands_env_and_builtin_tokens(self):
        with patch.dict("os.environ", {"DATA_ROOT": "/mnt/data"}, clear=False):
            result = fill_variables_in_config_paths(
                args={"model_family": "flux", "output_dir": "/tmp/run"},
                config=[
                    {
                        "id": "train-set",
                        "instance_data_dir": "{env:DATA_ROOT}/{model_family}/{id}",
                        "cache_dir_vae": "{output_dir}/cache/{id}",
                    }
                ],
            )

        self.assertEqual(result[0]["instance_data_dir"], "/mnt/data/flux/train-set")
        self.assertEqual(result[0]["cache_dir_vae"], "/tmp/run/cache/train-set")

    def test_modelspec_comment_runtime_placeholders_are_resolved(self):
        rendered = render_modelspec_comment(
            "step={current_step} epoch={current_epoch} ts={timestamp}",
            variables={
                "current_step": 1234,
                "current_epoch": 7,
                "timestamp": "2026-04-02T12:34:56+00:00",
            },
        )

        self.assertEqual(rendered, "step=1234 epoch=7 ts=2026-04-02T12:34:56+00:00")

    def test_modelspec_comment_preserves_runtime_tokens_until_save_time(self):
        rendered = render_modelspec_comment("step={current_step} epoch={current_epoch}")

        self.assertEqual(rendered, "step={current_step} epoch={current_epoch}")


if __name__ == "__main__":
    unittest.main()
