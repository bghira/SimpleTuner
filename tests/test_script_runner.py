import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from simpletuner.helpers.training import script_runner
from simpletuner.helpers.training.local_metrics import REPORT_FILENAME, LocalMetricsTracker, record_timestep_distribution
from simpletuner.helpers.training.validation_images import save_validation_image


class ScriptRunnerTests(unittest.TestCase):
    def test_hook_refreshes_report_before_script_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = SimpleNamespace(output_dir=directory, report_to="simpletuner")
            tracker = LocalMetricsTracker("upload-report", directory, "tests")
            tracker.store_init_configuration({"model_family": "flux"})
            tracker.log({"train_loss": 1.0}, step=943)
            tracker.log({"train_loss": 0.5}, step=1000)
            validation_dir = output_dir / "validation_images"
            validation_dir.mkdir()
            with patch("simpletuner.helpers.training.local_metrics._state_tracker_step", return_value=1000):
                save_validation_image(
                    Image.new("RGB", (8, 8)),
                    validation_dir,
                    "step_1000_sample",
                    config,
                    label="sample",
                    index=0,
                    resolution="8x8",
                )
            tracker.log({"train_loss": 0.4}, step=1002)
            record_timestep_distribution(config, [(1002, 250)])

            def report_payload():
                html = (output_dir / REPORT_FILENAME).read_text(encoding="utf-8")
                return json.loads(
                    html.split('<script id="training-metrics-data" type="application/json">')[1].split("</script>")[0]
                )

            self.assertEqual(report_payload()["run"]["last_step"], 943)
            submitted_reports = []
            with patch.object(
                script_runner, "submit_script", side_effect=lambda command: submitted_reports.append(report_payload())
            ) as submit:
                script_runner.run_hook_script(
                    "sync-report {global_step}",
                    config=config,
                    local_path=str(output_dir / "checkpoint-1000"),
                    global_step=1002,
                )

            submit.assert_called_once_with(["sync-report", "1002"])
            payload = submitted_reports[0]
            self.assertEqual(payload["run"]["last_step"], 1002)
            self.assertEqual([record["step"] for record in payload["records"]], [943, 1000, 1002])
            self.assertEqual(payload["media"][0]["path"], "validation_images/step_1000_sample.png")
            self.assertEqual(payload["media"][0]["step"], 1000)
            self.assertEqual(payload["timesteps"][0]["step"], 1002)

    def test_hook_does_not_create_report_when_local_metrics_are_disabled(self):
        for report_to in (None, "none", "wandb", "all"):
            with self.subTest(report_to=report_to), tempfile.TemporaryDirectory() as directory:
                config = SimpleNamespace(output_dir=directory, report_to=report_to)
                with patch.object(script_runner, "submit_script") as submit:
                    script_runner.run_hook_script("sync-report", config=config)
                submit.assert_called_once_with(["sync-report"])
                self.assertFalse((Path(directory) / REPORT_FILENAME).exists())

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
