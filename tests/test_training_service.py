import copy
import unittest
from contextlib import ExitStack
from typing import Any, Dict, Optional
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services import training_service
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults


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


class DummyConfigStore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config else None

    def get_active_config(self) -> str:
        return "default"

    def load_config(self, name: str):
        if self._config is None:
            raise FileNotFoundError
        return dict(self._config), {}


def _mock_is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return False
    return text not in {"false", "0", "no", "off"}


class TrainingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_state = copy.deepcopy(training_service.APIState.state)
        self._save_state_patch = patch.object(training_service.APIState, "save_state", return_value=None)
        self._save_state_patch.start()

    def tearDown(self) -> None:
        training_service.APIState.state = self._saved_state
        self._save_state_patch.stop()

    def _build_bundle(
        self,
        form_data: Dict[str, Any],
        *,
        stored_config: Optional[Dict[str, Any]] = None,
        defaults: Optional[WebUIDefaults] = None,
    ):
        effective_defaults = defaults or WebUIDefaults(accelerate_overrides={"mode": "auto"})
        store = DummyConfigStore(stored_config)
        with ExitStack() as stack:
            stack.enter_context(patch.object(training_service, "get_webui_state", return_value=(None, effective_defaults)))
            stack.enter_context(patch.object(training_service, "get_config_store", return_value=store))
            stack.enter_context(
                patch.object(
                    training_service,
                    "get_all_field_defaults",
                    return_value={"--num_processes": 1, "--output_dir": "/base/output"},
                )
            )
            stack.enter_context(
                patch.object(
                    training_service,
                    "detect_gpu_inventory",
                    return_value={
                        "detected": True,
                        "backend": "cuda",
                        "devices": [{"index": 0}, {"index": 1}],
                        "count": 2,
                        "optimal_processes": 2,
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    training_service.ConfigsService,
                    "normalize_form_to_config",
                    side_effect=lambda form, *_, **__: dict(form),
                )
            )
            stack.enter_context(
                patch.object(
                    training_service.ConfigsService,
                    "_migrate_legacy_keys",
                    side_effect=lambda mapping: mapping,
                )
            )
            stack.enter_context(patch.object(training_service.ConfigsService, "_is_truthy", side_effect=_mock_is_truthy))
            stack.enter_context(
                patch.object(
                    training_service.ConfigsService,
                    "coerce_config_values_by_field",
                    side_effect=lambda config: config,
                )
            )
            stack.enter_context(patch.object(training_service.lazy_field_registry, "get_all_fields", return_value=[]))
            stack.enter_context(patch.object(training_service.lazy_field_registry, "get_field", return_value=None))
            return training_service.build_config_bundle(form_data)

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

    def test_build_config_bundle_uses_onboarding_accelerate_defaults(self) -> None:
        bundle = self._build_bundle({})

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 2)
        self.assertNotIn("num_processes", bundle.save_config)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "auto")
        self.assertNotIn("accelerate_strategy", bundle.save_config)
        self.assertNotIn("accelerate_visible_devices", bundle.complete_config)

    def test_build_config_bundle_prefers_config_over_onboarding(self) -> None:
        bundle = self._build_bundle({}, stored_config={"num_processes": 3})

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 3)
        saved = bundle.save_config.get("num_processes")
        self.assertIsNotNone(saved)
        self.assertEqual(int(saved), 3)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "auto")

    def test_build_config_bundle_prefers_form_over_config(self) -> None:
        bundle = self._build_bundle(
            {"--num_processes": "5"},
            stored_config={"num_processes": 3},
        )

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 5)
        self.assertEqual(bundle.config_dict.get("--num_processes"), "5")
        saved = bundle.save_config.get("num_processes")
        self.assertIsNotNone(saved)
        self.assertEqual(int(saved), 5)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "auto")

    def test_onboarding_defaults_skip_when_accelerate_config_present(self) -> None:
        bundle = self._build_bundle(
            {},
            stored_config={"--accelerate_config": "/tmp/accelerate.yaml"},
        )

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 1)
        self.assertNotIn("accelerate_strategy", bundle.complete_config)

    def test_manual_device_selection_sets_visible_devices(self) -> None:
        defaults = WebUIDefaults(
            accelerate_overrides={
                "mode": "manual",
                "device_ids": [0, 2],
                "manual_count": 2,
            }
        )
        bundle = self._build_bundle({}, defaults=defaults)

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 2)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "manual")
        self.assertEqual(bundle.complete_config.get("accelerate_visible_devices"), [0, 2])
        self.assertNotIn("accelerate_visible_devices", bundle.save_config)

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
        self.assertEqual(captured["config"]["--webhook_config"], training_service.DEFAULT_WEBHOOK_CONFIG)
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
