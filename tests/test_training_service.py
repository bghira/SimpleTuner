import copy
import unittest
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services import training_service


class DummyValidation:
    def __init__(self, *, errors=None, warnings=None, suggestions=None, is_valid=True):
        self.errors = errors or []
        self.warnings = warnings or []
        self.suggestions = suggestions or []
        self.is_valid = is_valid


class DummyStore:
    def __init__(self, validation):
        self._validation = validation

    def validate_config(self, _config):
        return self._validation


class TrainingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_state = copy.deepcopy(training_service.APIState.state)
        self._save_state_patch = patch.object(training_service.APIState, "save_state", return_value=None)
        self._save_state_patch.start()

    def tearDown(self) -> None:
        training_service.APIState.state = self._saved_state
        self._save_state_patch.stop()

    def test_validate_training_config_requires_steps_or_epochs(self) -> None:
        store = DummyStore(DummyValidation())
        result = training_service.validate_training_config(
            store,
            {"--num_train_epochs": 0, "--max_train_steps": 0},
        )

        self.assertTrue(any("num_train_epochs" in error for error in result.errors))

    def test_validate_training_config_flags_constant_warmup(self) -> None:
        store = DummyStore(DummyValidation())
        result = training_service.validate_training_config(
            store,
            {"--lr_scheduler": "constant", "--lr_warmup_steps": 12},
        )

        self.assertTrue(any("Warmup steps" in error for error in result.errors))

    def test_start_training_job_updates_api_state(self) -> None:
        captured = {}

        def fake_submit(job_id, func, config):
            captured["job_id"] = job_id
            captured["func"] = func
            captured["config"] = config

        with patch.object(training_service.process_keeper, "submit_job", side_effect=fake_submit):
            job_id = training_service.start_training_job({"--foo": "bar"})

        self.assertEqual(job_id, captured["job_id"])
        self.assertIs(captured["func"], training_service.run_trainer_job)
        self.assertEqual(captured["config"]["__job_id__"], job_id)
        self.assertEqual(
            captured["config"]["--webhook_config"], training_service.DEFAULT_WEBHOOK_CONFIG
        )
        self.assertEqual(training_service.APIState.get_state("current_job_id"), job_id)
        self.assertEqual(training_service.APIState.get_state("training_status"), "starting")
        self.assertEqual(
            training_service.APIState.get_state("training_config")["--webhook_config"],
            training_service.DEFAULT_WEBHOOK_CONFIG,
        )

    def test_terminate_training_job_clears_state(self) -> None:
        terminated = {}

        def fake_terminate(job_id):
            terminated["job_id"] = job_id

        with patch.object(training_service.process_keeper, "terminate_process", side_effect=fake_terminate):
            training_service.APIState.set_state("current_job_id", "abc123")
            result = training_service.terminate_training_job("abc123", status="cancelled", clear_job_id=True)

        self.assertTrue(result)
        self.assertEqual(terminated["job_id"], "abc123")
        self.assertIsNone(training_service.APIState.get_state("current_job_id"))
        self.assertEqual(training_service.APIState.get_state("training_status"), "cancelled")


if __name__ == "__main__":
    unittest.main()
