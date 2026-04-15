import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.validation import Validation


class ValidationExternalScriptTests(unittest.TestCase):
    def test_command_formats_checkpoint_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = os.path.join(tmp_dir, "checkpoint-1")
            os.makedirs(checkpoint_dir)
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(
                validation_external_script="echo {local_checkpoint_path}",
                output_dir=tmp_dir,
            )

            with patch(
                "simpletuner.helpers.training.validation.CheckpointManager.get_latest_checkpoint",
                return_value="checkpoint-1",
            ):
                command = validation._build_external_validation_command()

            self.assertEqual(command, ["echo", checkpoint_dir])

    def test_command_requires_checkpoint_when_placeholder_used(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(
                validation_external_script="echo {local_checkpoint_path}",
                output_dir=tmp_dir,
            )

            with (
                patch(
                    "simpletuner.helpers.training.validation.CheckpointManager.get_latest_checkpoint",
                    return_value=None,
                ),
                self.assertRaisesRegex(ValueError, "local_checkpoint_path"),
            ):
                validation._build_external_validation_command()

    def test_command_rejects_unknown_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_template = "run {missing}"
            checkpoint_dir = os.path.join(tmp_dir, "checkpoint-5")
            os.makedirs(checkpoint_dir)
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(
                validation_external_script=invalid_template,
                output_dir=tmp_dir,
            )

            with (
                patch(
                    "simpletuner.helpers.training.validation.CheckpointManager.get_latest_checkpoint",
                    return_value="checkpoint-5",
                ),
                self.assertRaisesRegex(ValueError, "Unknown placeholder"),
            ):
                validation._build_external_validation_command()

    def test_command_formats_state_and_tracker_placeholders(self):
        validation = Validation.__new__(Validation)
        validation.global_step = 7
        validation.config = SimpleNamespace(
            validation_external_script="echo {global_step} {tracker_run_name} {tracker_project_name} {model_family}",
            tracker_run_name="run-123",
            tracker_project_name="proj-abc",
            model_family="flux",
        )

        command = validation._build_external_validation_command()

        self.assertEqual(command, ["echo", "7", "run-123", "proj-abc", "flux"])

    def test_global_step_placeholder_falls_back_to_state_tracker(self):
        validation = Validation.__new__(Validation)
        validation.global_step = None
        validation.config = SimpleNamespace(
            validation_external_script="echo {global_step}",
        )

        with patch("simpletuner.helpers.training.validation.StateTracker.get_global_step", return_value=12):
            command = validation._build_external_validation_command()

        self.assertEqual(command, ["echo", "12"])

    def test_model_family_placeholder_falls_back_to_state_tracker(self):
        validation = Validation.__new__(Validation)
        validation.global_step = None
        validation.config = SimpleNamespace(
            validation_external_script="echo {model_family}",
            model_family=None,
        )

        with patch("simpletuner.helpers.training.validation.StateTracker.get_model_family", return_value="sdxl"):
            command = validation._build_external_validation_command()

        self.assertEqual(command, ["echo", "sdxl"])

    def test_model_and_lora_type_placeholders(self):
        validation = Validation.__new__(Validation)
        validation.global_step = None
        validation.config = SimpleNamespace(
            validation_external_script="echo {model_type} {lora_type}",
            model_type="lora",
            lora_type="standard",
        )

        command = validation._build_external_validation_command()

        self.assertEqual(command, ["echo", "lora", "standard"])

    def test_validation_prefix_placeholder_uses_config(self):
        validation = Validation.__new__(Validation)
        validation.global_step = None
        validation.config = SimpleNamespace(
            validation_external_script="echo {validation_num_inference_steps}",
            validation_num_inference_steps=22,
        )

        command = validation._build_external_validation_command()

        self.assertEqual(command, ["echo", "22"])

    @patch("simpletuner.helpers.training.validation.StateTracker.get_webhook_handler", return_value=None)
    def test_run_validations_invokes_external_runner(self, _mock_webhook):
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(
            validation_method="external-script",
            validation_multigpu="single-gpu",
            gradient_accumulation_steps=1,
        )
        validation.accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
        validation.deepspeed = False
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.validation_prompt_metadata = {"validation_prompts": ["prompt-1"]}
        validation.validation_prompt_dict = {}
        validation.validation_video_paths = {}
        validation.eval_scores = {}
        validation.validation_images = None
        validation.evaluation_result = None
        validation.global_step = 1
        validation.global_resume_step = 0
        validation.current_epoch = 0
        validation.current_epoch_step = 0
        validation._run_external_validation = MagicMock()
        validation._use_distributed_validation = MagicMock(return_value=False)
        validation.should_perform_intermediary_validation = MagicMock(return_value=True)
        validation._update_state = MagicMock()

        validation.run_validations(step=1, validation_type="intermediary")

        validation._run_external_validation.assert_called_once_with(validation_type="intermediary", step=1)
        self.assertEqual(validation.validation_images, {})

    @patch("simpletuner.helpers.training.validation.StateTracker.get_webhook_handler", return_value=None)
    @patch("simpletuner.helpers.training.validation.reclaim_memory")
    def test_base_model_benchmark_uses_builtin_validation_when_external_script_configured(
        self, _mock_reclaim_memory, _mock_webhook
    ):
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(
            validation_method="external-script",
            validation_multigpu="single-gpu",
            gradient_accumulation_steps=1,
        )
        validation.accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
        validation.deepspeed = False
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.validation_prompt_metadata = {"validation_prompts": ["prompt-1"]}
        validation.validation_prompt_dict = {}
        validation.validation_video_paths = {}
        validation.eval_scores = {}
        validation.validation_images = None
        validation.validation_audios = None
        validation.evaluation_result = None
        validation.global_step = 0
        validation.global_resume_step = 0
        validation.current_epoch = 0
        validation.current_epoch_step = 0
        validation.validation_adapter_runs = []
        validation.model = SimpleNamespace(pipeline=object())
        validation._run_external_validation = MagicMock()
        validation._use_distributed_validation = MagicMock(return_value=False)
        validation.should_perform_intermediary_validation = MagicMock(return_value=False)
        validation._update_state = MagicMock()
        validation._check_abort = MagicMock()
        validation.setup_pipeline = MagicMock()
        validation.setup_scheduler = MagicMock()
        validation.finalize_validation = MagicMock()
        validation._publish_validation_artifacts = MagicMock()
        validation.clean_pipeline = MagicMock()

        validation.run_validations(step=0, validation_type="base_model")

        validation._run_external_validation.assert_not_called()
        validation.setup_pipeline.assert_called_once_with("base_model")
        validation.setup_scheduler.assert_called_once()
        validation.finalize_validation.assert_called_once_with("base_model")
        validation.clean_pipeline.assert_called_once()

    def test_run_external_validation_skips_when_checkpoint_placeholder_has_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(
                validation_external_script="echo {local_checkpoint_path}",
                validation_external_background=False,
                output_dir=tmp_dir,
            )

            with (
                patch(
                    "simpletuner.helpers.training.validation.CheckpointManager.get_latest_checkpoint",
                    return_value=None,
                ),
                patch("subprocess.run") as mock_run,
                self.assertLogs("Validation", level="WARNING") as logs,
            ):
                result = validation._run_external_validation(validation_type="intermediary", step=100)

        self.assertFalse(result)
        mock_run.assert_not_called()
        self.assertIn("Skipping external validation for intermediary at step 100", "\n".join(logs.output))

    def test_run_external_validation_background_uses_popen(self):
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(validation_external_background=True)
        validation._build_external_validation_command = MagicMock(return_value=["echo", "hi"])

        with (
            patch("subprocess.run") as mock_run,
            patch("subprocess.Popen") as mock_popen,
        ):
            validation._run_external_validation(validation_type="final", step=5)

        mock_popen.assert_called_once_with(["echo", "hi"])
        mock_run.assert_not_called()

    def test_run_external_validation_foreground_checks_exit_code(self):
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(validation_external_background=False)
        validation._build_external_validation_command = MagicMock(return_value=["echo", "hi"])

        with (
            patch("subprocess.run") as mock_run,
            patch("subprocess.Popen") as mock_popen,
        ):
            validation._run_external_validation(validation_type="final", step=5)

        mock_run.assert_called_once_with(["echo", "hi"], check=True)
        mock_popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
