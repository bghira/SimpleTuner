import copy

import pytest

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


@pytest.fixture
def api_state_guard(monkeypatch):
    saved_state = copy.deepcopy(training_service.APIState.state)
    monkeypatch.setattr(training_service.APIState, "save_state", lambda: None)
    yield
    training_service.APIState.state = saved_state


def test_validate_training_config_requires_steps_or_epochs():
    store = DummyStore(DummyValidation())
    result = training_service.validate_training_config(
        store,
        {"--num_train_epochs": 0, "--max_train_steps": 0},
    )

    assert any("num_train_epochs" in error for error in result.errors)


def test_validate_training_config_flags_constant_warmup():
    store = DummyStore(DummyValidation())
    result = training_service.validate_training_config(
        store,
        {"--lr_scheduler": "constant", "--lr_warmup_steps": 12},
    )

    assert any("Warmup steps" in error for error in result.errors)


def test_start_training_job_updates_api_state(monkeypatch, api_state_guard):
    captured = {}

    def fake_submit(job_id, func, config):
        captured["job_id"] = job_id
        captured["func"] = func
        captured["config"] = config

    monkeypatch.setattr(training_service.process_keeper, "submit_job", fake_submit)

    job_id = training_service.start_training_job({"--foo": "bar"})

    assert job_id == captured["job_id"]
    assert captured["func"] is training_service.run_trainer_job
    assert captured["config"]["__job_id__"] == job_id
    assert training_service.APIState.get_state("current_job_id") == job_id
    assert training_service.APIState.get_state("training_status") == "starting"


def test_terminate_training_job_clears_state(monkeypatch, api_state_guard):
    terminated = {}

    def fake_terminate(job_id):
        terminated["job_id"] = job_id

    monkeypatch.setattr(training_service.process_keeper, "terminate_process", fake_terminate)

    training_service.APIState.set_state("current_job_id", "abc123")
    result = training_service.terminate_training_job("abc123", status="cancelled", clear_job_id=True)

    assert result is True
    assert terminated["job_id"] == "abc123"
    assert training_service.APIState.get_state("current_job_id") is None
    assert training_service.APIState.get_state("training_status") == "cancelled"
