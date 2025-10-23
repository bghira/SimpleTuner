import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
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
                        "capabilities": {
                            "supports_cuda": True,
                            "supports_mps": False,
                            "supports_rocm": False,
                            "supports_deepspeed": True,
                            "supports_fsdp": True,
                        },
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

            # Mock field registry to return common trainer fields
            # This ensures the field validation check doesn't filter out legitimate fields
            class MockField:
                def __init__(self, name, arg_name, webui_only=False):
                    self.name = name
                    self.arg_name = arg_name
                    self.webui_only = webui_only

            mock_fields = [
                MockField("num_processes", "--num_processes"),
                MockField("output_dir", "--output_dir"),
                MockField("learning_rate", "--learning_rate"),
                MockField("model_family", "--model_family"),
            ]
            stack.enter_context(
                patch.object(training_service.lazy_field_registry, "get_all_fields", return_value=mock_fields)
            )
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

    def test_validate_training_config_checks_prompt_library_path(self) -> None:
        store = DummyStore(DummyValidation())
        result = training_service.validate_training_config(
            store,
            {"--user_prompt_library": "/not/a/real/file.json"},
        )

        self.assertTrue(any("User prompt library" in error for error in result.errors))

    def test_validate_training_config_accepts_existing_prompt_library(self) -> None:
        store = DummyStore(DummyValidation())
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "library.json"
            prompt_file.write_text(json.dumps({"a": "b"}), encoding="utf-8")
            result = training_service.validate_training_config(
                store,
                {"--user_prompt_library": str(prompt_file)},
            )

        self.assertFalse(any("User prompt library" in error for error in result.errors))

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

        with (
            patch.object(training_service, "get_webui_state", return_value=(None, WebUIDefaults())),
            patch.object(training_service.process_keeper, "submit_job", side_effect=fake_submit),
        ):
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

    def test_start_training_job_copies_prompt_library(self) -> None:
        captured = {}

        def fake_submit(job_id, func, config):
            captured["job_id"] = job_id
            captured["config"] = config

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(training_service, "get_webui_state", return_value=(None, WebUIDefaults())),
            patch.object(training_service.process_keeper, "submit_job", side_effect=fake_submit),
        ):
            source = Path(tmpdir) / "library.json"
            source.write_text(json.dumps({"prompt": "value"}), encoding="utf-8")
            job_id = training_service.start_training_job({"--user_prompt_library": str(source)})

        runtime_path = Path(captured["config"]["--user_prompt_library"])
        self.assertEqual(job_id, captured["job_id"])
        self.assertNotEqual(runtime_path, source)
        self.assertTrue(runtime_path.exists())
        self.assertEqual(json.loads(runtime_path.read_text(encoding="utf-8")), {"prompt": "value"})

    def test_start_training_job_missing_prompt_library_raises(self) -> None:
        with (
            patch.object(training_service, "get_webui_state", return_value=(None, WebUIDefaults())),
            patch.object(training_service.process_keeper, "submit_job", side_effect=AssertionError("should not submit")),
        ):
            with self.assertRaises(FileNotFoundError):
                training_service.start_training_job({"--user_prompt_library": "/missing/library.json"})

        self.assertIsNone(training_service.APIState.get_state("current_job_id"))

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

    def test_build_config_bundle_filters_webui_only_fields(self) -> None:
        """Test that webui_only fields are filtered from config_dict and don't appear in trainer configs."""

        # Create a mock field that has webui_only=True
        class MockField:
            def __init__(self, name, webui_only=False):
                self.name = name
                self.webui_only = webui_only
                self.validation_rules = []

        def mock_get_field(name):
            if name in ("webhook_config", "webhook_reporting_interval", "num_validation_images", "configs_dir"):
                return MockField(name, webui_only=True)
            return None

        form_data = {
            "--webhook_config": '[{"webhook_type": "raw", "callback_url": "https://example.com/webhook"}]',
            "--webhook_reporting_interval": "60",
            "--num_validation_images": "5",
            "--configs_dir": "/custom/configs",
            "--datasets_dir": "/some/datasets",
            "--learning_rate": "1e-4",
            "--output_dir": "/output",
        }

        with patch.object(training_service.lazy_field_registry, "get_field", side_effect=mock_get_field):
            bundle = self._build_bundle(form_data)

        # WebUI-only fields should NOT appear in save_config (this prevents them from being saved to disk)
        # This is the critical check - we don't want WebUI-only fields persisted to config files
        self.assertNotIn("webhook_config", bundle.save_config)
        self.assertNotIn("webhook_reporting_interval", bundle.save_config)
        self.assertNotIn("num_validation_images", bundle.save_config)
        self.assertNotIn("datasets_dir", bundle.save_config)

        # webhook_config IS in complete_config because it's injected at runtime by the WebUI for trainer use
        # This is intentional - the trainer needs the webhook config to send callbacks, but it shouldn't
        # be saved to disk config files
        self.assertIn("--webhook_config", bundle.complete_config)
        self.assertEqual(bundle.complete_config["--webhook_config"], training_service.DEFAULT_WEBHOOK_CONFIG)

        # Other WebUI-only fields should NOT appear in complete_config
        self.assertNotIn("--num_validation_images", bundle.complete_config)
        self.assertNotIn("num_validation_images", bundle.complete_config)
        self.assertNotIn("--datasets_dir", bundle.complete_config)
        self.assertNotIn("datasets_dir", bundle.complete_config)

        # Regular trainer fields should still be present
        self.assertIn("--learning_rate", bundle.complete_config)
        self.assertIn("--output_dir", bundle.complete_config)


if __name__ == "__main__":
    unittest.main()
