import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from simpletuner.helpers.training.validation import Validation, ValidationAbortedException


class ValidationUploadHookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.validation = Validation.__new__(Validation)
        validation = self.validation
        validation.config = SimpleNamespace(
            output_dir=self.directory.name,
            post_upload_script='notify "{local_checkpoint_path}" "{remote_checkpoint_path}" {global_step}',
            validation_method="simpletuner-local",
        )
        validation.accelerator = SimpleNamespace(is_main_process=True)
        validation.publishing_manager = None
        validation.deepspeed = False
        validation.global_step = 7
        validation._pending_epoch_validation = None
        validation.validation_prompt_metadata = {"validation_prompts": ["a cat"]}
        validation.validation_video_paths = {}
        validation.validation_adapter_runs = [None]
        validation.evaluation_result = None
        validation.model = SimpleNamespace(pipeline=object())
        for name in (
            "_update_state",
            "_check_abort",
            "setup_pipeline",
            "setup_scheduler",
            "finalize_validation",
            "clean_pipeline",
            "_log_adapter_run",
        ):
            setattr(validation, name, MagicMock())
        validation._use_distributed_validation = MagicMock(return_value=False)
        validation.should_perform_intermediary_validation = MagicMock(return_value=False)
        validation._temporary_validation_adapters = MagicMock(side_effect=lambda _: nullcontext())
        validation.process_prompts = MagicMock(side_effect=self.generate)
        self.submit = self.enterContext(patch("simpletuner.helpers.training.script_runner.submit_script"))
        self.enterContext(
            patch("simpletuner.helpers.training.validation.StateTracker.get_webhook_handler", return_value=None)
        )
        self.enterContext(patch("simpletuner.helpers.training.validation.reclaim_memory"))

    def generate(self, **kwargs):
        artifact = Path(self.directory.name) / "validation.png"
        artifact.write_bytes(b"generated")
        kwargs["image_accumulator"]["prompt"] = [artifact]

    def run_validation(self, **kwargs):
        return self.validation.run_validations(step=7, force_evaluation=True, **kwargs)

    def test_manual_validation_runs_hook_without_provider_or_checkpoint_after_completion(self):
        def submitted(command):
            self.assertTrue((Path(self.directory.name) / "validation.png").exists())
            self.validation.finalize_validation.assert_called_once_with("intermediary")

        self.submit.side_effect = submitted
        self.run_validation()
        self.submit.assert_called_once_with(["notify", self.directory.name, "", "7"])

    def test_unconfigured_provider_runs_local_hook(self):
        self.validation.publishing_manager = MagicMock(configured=False)
        self.run_validation()
        self.submit.assert_called_once_with(["notify", self.directory.name, "", "7"])
        self.validation.publishing_manager.publish.assert_not_called()

    def test_scheduled_validation_runs_local_hook(self):
        self.validation.should_perform_intermediary_validation.return_value = True
        self.validation.run_validations(step=7)
        self.submit.assert_called_once_with(["notify", self.directory.name, "", "7"])

    def test_base_model_validation_runs_local_hook(self):
        self.validation.run_validations(step=0, validation_type="base_model")
        self.submit.assert_called_once_with(["notify", self.directory.name, "", "7"])

    def test_provider_hooks_keep_each_result_paths_without_extra_local_hook(self):
        self.validation.publishing_manager = MagicMock(configured=True)
        self.validation.publishing_manager.publish.return_value = [
            SimpleNamespace(uri="s3://bucket/run", artifact_path=self.directory.name),
            None,
            SimpleNamespace(uri="https://example.com/run", artifact_path=self.directory.name),
        ]
        self.run_validation()
        self.assertEqual(
            self.submit.call_args_list,
            [
                call(["notify", self.directory.name, "s3://bucket/run", "7"]),
                call(["notify", self.directory.name, "https://example.com/run", "7"]),
            ],
        )

    def test_provider_failure_does_not_run_local_hook(self):
        self.validation.publishing_manager = MagicMock(configured=True)
        self.validation.publishing_manager.publish.side_effect = RuntimeError("upload failed")
        self.run_validation()
        self.submit.assert_not_called()

    def test_provider_without_successful_results_does_not_run_hook(self):
        self.validation.publishing_manager = MagicMock(configured=True)
        self.validation.publishing_manager.publish.return_value = [None]
        self.run_validation()
        self.submit.assert_not_called()

    def test_skipped_validation_does_not_run_hook(self):
        self.run_validation(skip_execution=True)
        self.submit.assert_not_called()
        self.validation.process_prompts.assert_not_called()

    def test_missing_pipeline_does_not_run_hook(self):
        self.validation.model.pipeline = None
        self.run_validation()
        self.submit.assert_not_called()

    def test_generation_failure_or_abort_does_not_run_hook(self):
        for error in (RuntimeError("generation failed"), ValidationAbortedException("aborted")):
            with self.subTest(error=type(error).__name__):
                self.validation.process_prompts.side_effect = error
                with self.assertRaises(type(error)):
                    self.run_validation()
                self.submit.assert_not_called()

    def test_distributed_non_main_rank_does_not_run_hook(self):
        self.validation.accelerator.is_main_process = False
        self.validation._use_distributed_validation.return_value = True
        self.run_validation()
        self.validation.process_prompts.assert_called_once()
        self.submit.assert_not_called()

    def test_external_script_launch_does_not_run_hook(self):
        self.validation.config.validation_method = "external-script"
        self.validation._run_external_validation = MagicMock(return_value=True)
        self.run_validation()
        self.validation._run_external_validation.assert_called_once()
        self.submit.assert_not_called()

    def test_disabled_hook_does_not_submit(self):
        for template in (None, "", "None"):
            with self.subTest(template=template):
                self.validation.config.post_upload_script = template
                self.run_validation()
                self.submit.assert_not_called()
