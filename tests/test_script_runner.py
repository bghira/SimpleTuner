import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.training import script_runner


class ScriptRunnerTests(unittest.TestCase):
    def test_build_script_command_requires_nonempty(self):
        with self.assertRaises(ValueError):
            script_runner.build_script_command("", lambda _: "")

    def test_build_script_command_unknown_placeholder_raises(self):
        with self.assertRaises(ValueError):
            script_runner.build_script_command("echo {missing}", lambda name: (_ for _ in ()).throw(KeyError(name)))

    @patch("simpletuner.helpers.training.script_runner.submit_script")
    def test_run_hook_script_formats_context(self, mock_submit):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = os.path.join(tmp_dir, "checkpoint-1")
            os.makedirs(checkpoint_dir)
            config = SimpleNamespace(
                output_dir=tmp_dir,
                tracker_run_name="run-123",
                tracker_project_name="proj-abc",
                model_family="flux",
                hub_model_id="org/model",
                validation_num_inference_steps=15,
                model_type="lora",
                lora_type="standard",
            )

            script_runner.run_hook_script(
                "echo {local_checkpoint_path} {remote_checkpoint_path} {tracker_run_name} {tracker_project_name} {model_family} {huggingface_path} {model_type} {lora_type} {global_step} {validation_num_inference_steps}",
                config=config,
                local_path=checkpoint_dir,
                remote_path="s3://remote/path",
                global_step=10,
            )

            mock_submit.assert_called_once()
            command = mock_submit.call_args[0][0]
            self.assertEqual(
                command,
                [
                    "echo",
                    checkpoint_dir,
                    "s3://remote/path",
                    "run-123",
                    "proj-abc",
                    "flux",
                    "org/model",
                    "lora",
                    "standard",
                    "10",
                    "15",
                ],
            )


if __name__ == "__main__":
    unittest.main()
