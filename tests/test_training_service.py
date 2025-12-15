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
        field_defaults: Optional[Dict[str, Any]] = None,
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
                    return_value=field_defaults or {"--num_processes": 1, "--output_dir": "/base/output"},
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
                MockField("adam_beta1", "--adam_beta1"),
                MockField("adam_beta2", "--adam_beta2"),
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

    def test_config_num_processes_ignored_when_strategy_selected(self) -> None:
        bundle = self._build_bundle({}, stored_config={"num_processes": 3})

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 2)
        self.assertNotIn("num_processes", bundle.save_config)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "auto")

    def test_form_num_processes_respected_in_hardware_mode(self) -> None:
        bundle = self._build_bundle(
            {"--num_processes": "5"},
            stored_config={"num_processes": 3},
            defaults=WebUIDefaults(accelerate_overrides={"mode": "hardware"}),
        )

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 5)
        self.assertEqual(bundle.config_dict.get("--num_processes"), "5")
        saved = bundle.save_config.get("num_processes")
        self.assertIsNotNone(saved)
        self.assertEqual(int(saved), 5)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "hardware")

    def test_onboarding_defaults_skip_when_accelerate_config_present(self) -> None:
        bundle = self._build_bundle(
            {},
            stored_config={"--accelerate_config": "/tmp/accelerate.yaml"},
        )

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 1)
        self.assertNotIn("accelerate_strategy", bundle.complete_config)

    def test_hardware_mode_delegates_to_field_default(self) -> None:
        defaults = WebUIDefaults(accelerate_overrides={"mode": "hardware"})
        bundle = self._build_bundle(
            {}, defaults=defaults, field_defaults={"--num_processes": 4, "--output_dir": "/base/output"}
        )

        value = bundle.complete_config.get("--num_processes")
        self.assertIsNotNone(value)
        self.assertEqual(int(value), 4)
        self.assertEqual(bundle.complete_config.get("accelerate_strategy"), "hardware")
        self.assertNotIn("num_processes", bundle.save_config)

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

    def test_start_training_job_keeps_accelerate_visible_devices(self) -> None:
        captured = {}

        def fake_submit(job_id, func, config):
            captured["job_id"] = job_id
            captured["config"] = config

        payload = {
            "accelerate_visible_devices": [1],
            "--num_processes": 1,
        }

        with (
            patch.object(training_service, "get_webui_state", return_value=(None, WebUIDefaults())),
            patch.object(training_service.process_keeper, "submit_job", side_effect=fake_submit),
        ):
            job_id = training_service.start_training_job(payload)

        self.assertEqual(job_id, captured["job_id"])
        self.assertEqual(captured["config"].get("accelerate_visible_devices"), [1])
        self.assertEqual(captured["config"].get("--num_processes"), 1)

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
            return True

        with patch.object(training_service.process_keeper, "terminate_process", side_effect=fake_terminate):
            training_service.APIState.set_state("current_job_id", "abc123")
            result = training_service.terminate_training_job("abc123", status="cancelled", clear_job_id=True)

        self.assertTrue(result)
        self.assertEqual(terminated["job_id"], "abc123")
        self.assertIsNone(training_service.APIState.get_state("current_job_id"))
        self.assertEqual(training_service.APIState.get_state("training_status"), "cancelled")

    def test_request_manual_validation_sends_command(self) -> None:
        training_service.APIState.set_state("current_job_id", "job-xyz")
        with patch.object(training_service.process_keeper, "send_process_command") as mock_send:
            job_id = training_service.request_manual_validation()

        self.assertEqual(job_id, "job-xyz")
        mock_send.assert_called_once_with("job-xyz", "trigger_validation", None)

    def test_request_manual_validation_without_job_raises(self) -> None:
        training_service.APIState.set_state("current_job_id", None)
        with self.assertRaises(RuntimeError):
            training_service.request_manual_validation()

    def test_request_manual_checkpoint_sends_command(self) -> None:
        training_service.APIState.set_state("current_job_id", "job-abc")
        with patch.object(training_service.process_keeper, "send_process_command") as mock_send:
            job_id = training_service.request_manual_checkpoint()

        self.assertEqual(job_id, "job-abc")
        mock_send.assert_called_once_with("job-abc", "trigger_checkpoint", None)

    def test_request_manual_checkpoint_without_job_raises(self) -> None:
        training_service.APIState.set_state("current_job_id", None)
        with self.assertRaises(RuntimeError):
            training_service.request_manual_checkpoint()

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

    def test_preserve_defaults_excludes_default_values(self) -> None:
        """
        Regression test: preserve_defaults should exclude fields matching default values from saved configs.

        Bug: The webUI frontend sends ALL fields in form submission (via appendConfigValuesToFormData),
        including fields the user didn't touch. This made preserve_defaults ineffective because every
        field looked "explicitly set" even though it was just carried forward from the saved config.

        Fix: Compare form values with saved config values to detect actual changes. Only fields that
        were actually modified should bypass preserve_defaults filtering.

        Expected: Fields matching defaults should be excluded unless explicitly changed by user.
        """
        # Set up field defaults that match what we'll put in stored_config
        field_defaults = {
            "--output_dir": "/base/output",
            "--learning_rate": 1e-4,  # Default learning rate
            "--model_family": "flux",  # Default model family
        }

        # Simulate a saved config where some fields match their defaults
        stored_config = {
            "learning_rate": 1e-4,  # Matches default
            "model_family": "flux",  # Matches default
            "output_dir": "/custom/output",  # Different from default - should be saved
        }

        # Enable preserve_defaults via WebUIDefaults
        defaults = WebUIDefaults(
            accelerate_overrides={"mode": "disabled"},
            auto_preserve_defaults=True,
        )

        # Simulate the real webUI behavior: form submission contains ALL fields (unchanged ones too)
        # This is what appendConfigValuesToFormData() does in production
        form_data = {
            "--learning_rate": 1e-4,  # In form, but unchanged from saved config
            "--model_family": "flux",  # In form, but unchanged from saved config
            "--output_dir": "/custom/output",  # In form, unchanged from saved config
        }

        # Build bundle with form containing all fields (like production)
        bundle = self._build_bundle(
            form_data,  # Form contains all fields, simulating webUI behavior
            stored_config=stored_config,
            defaults=defaults,
            field_defaults=field_defaults,
        )

        # Fields matching defaults AND unchanged from saved config should NOT be in save_config
        self.assertNotIn(
            "learning_rate", bundle.save_config, "learning_rate matches default and wasn't changed, should be excluded"
        )
        self.assertNotIn(
            "model_family", bundle.save_config, "model_family matches default and wasn't changed, should be excluded"
        )

        # Field with non-default value SHOULD be in save_config (even if unchanged)
        self.assertIn("output_dir", bundle.save_config, "output_dir differs from default and should be in save_config")
        self.assertEqual(bundle.save_config["output_dir"], "/custom/output")

        # All fields should still be in complete_config (for runtime use)
        self.assertIn("--learning_rate", bundle.complete_config)
        self.assertIn("--model_family", bundle.complete_config)
        self.assertIn("--output_dir", bundle.complete_config)

    def test_preserve_defaults_saves_changed_fields_even_if_default(self) -> None:
        """
        Test that fields explicitly changed to their default value ARE saved.

        If a user changes a field FROM a non-default value TO the default value,
        that's an explicit change and should be saved even with preserve_defaults=True.
        """
        field_defaults = {
            "--learning_rate": 1e-4,  # Default learning rate
        }

        # Config has non-default learning rate
        stored_config = {
            "learning_rate": 5e-5,  # Non-default value
        }

        defaults = WebUIDefaults(
            accelerate_overrides={"mode": "disabled"},
            auto_preserve_defaults=True,
        )

        # User changes learning_rate back to default value
        form_data = {
            "--learning_rate": 1e-4,  # Changed from 5e-5 to 1e-4 (the default)
        }

        bundle = self._build_bundle(
            form_data,
            stored_config=stored_config,
            defaults=defaults,
            field_defaults=field_defaults,
        )

        # Should be saved because user explicitly changed it (even though it's now the default)
        self.assertIn(
            "learning_rate",
            bundle.save_config,
            "learning_rate was explicitly changed and should be saved even though it matches default",
        )
        self.assertEqual(bundle.save_config["learning_rate"], 1e-4)

    def test_preserve_defaults_excludes_new_fields_with_default_values(self) -> None:
        """
        Test that new fields sent by frontend with default values are excluded.

        The frontend sends ALL fields (via appendConfigValuesToFormData), including fields
        that weren't in the original saved config. If these new fields have default values,
        they should be excluded (not treated as "newly added" fields).
        """
        field_defaults = {
            "--learning_rate": 1e-4,
            "--adam_beta1": 0.9,  # Default adam_beta1
            "--adam_beta2": 0.999,  # Default adam_beta2
        }

        # Minimal saved config - doesn't have adam_beta1 or adam_beta2
        stored_config = {
            "learning_rate": 5e-5,  # Non-default value
        }

        defaults = WebUIDefaults(
            accelerate_overrides={"mode": "disabled"},
            auto_preserve_defaults=True,
        )

        # Frontend sends ALL fields including ones not in saved config
        form_data = {
            "--learning_rate": 5e-5,  # Unchanged from saved config
            "--adam_beta1": 0.9,  # NOT in saved config, matches default
            "--adam_beta2": 0.999,  # NOT in saved config, matches default
        }

        bundle = self._build_bundle(
            form_data,
            stored_config=stored_config,
            defaults=defaults,
            field_defaults=field_defaults,
        )

        # learning_rate differs from default, should be saved
        self.assertIn("learning_rate", bundle.save_config)

        # adam_beta1 and adam_beta2 were NOT in saved config and match defaults, should be excluded
        self.assertNotIn(
            "adam_beta1", bundle.save_config, "adam_beta1 wasn't in saved config and matches default, should be excluded"
        )
        self.assertNotIn(
            "adam_beta2", bundle.save_config, "adam_beta2 wasn't in saved config and matches default, should be excluded"
        )


if __name__ == "__main__":
    unittest.main()
