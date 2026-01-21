import tempfile
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.validation_service import ValidationService, ValidationSeverity


class UIValidationConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_instances)

    def _restore_instances(self) -> None:
        ConfigStore._instances = self._instances_backup

    def test_full_model_type_rejects_quantized_base_precision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=Path(tmpdir), config_type="model")
            validation = store.validate_config(
                {
                    "--model_type": "full",
                    "--base_model_precision": "int8-quanto",
                }
            )

        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("base model" in error.lower() and "quant" in error.lower() for error in validation.errors),
            validation.errors,
        )

    def test_full_model_type_allows_no_change_base_precision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=Path(tmpdir), config_type="model")
            validation = store.validate_config(
                {
                    "--model_type": "full",
                    "--base_model_precision": "no_change",
                }
            )

        self.assertTrue(validation.is_valid)
        self.assertFalse(
            any("base model" in error.lower() and "quant" in error.lower() for error in validation.errors),
            validation.errors,
        )

    def test_lora_model_type_rejects_deepspeed_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=Path(tmpdir), config_type="model")
            validation = store.validate_config(
                {
                    "--model_type": "lora",
                    "--deepspeed_config": '{"zero_optimization": {"stage": 2}}',
                }
            )

        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("lora" in error.lower() and "deepspeed" in error.lower() for error in validation.errors),
            validation.errors,
        )


class ValidationServiceConstraintTests(unittest.TestCase):
    def test_validation_service_flags_full_quantization(self) -> None:
        service = ValidationService()
        result = service.validate_configuration(
            {
                "model_type": "full",
                "base_model_precision": "int8-quanto",
            },
            validate_paths=False,
            estimate_vram=False,
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any(msg.field == "base_model_precision" and msg.severity == ValidationSeverity.ERROR for msg in result.messages),
            [msg.to_dict() for msg in result.messages],
        )

    def test_validation_service_flags_lora_deepspeed(self) -> None:
        service = ValidationService()
        result = service.validate_configuration(
            {
                "model_type": "lora",
                "deepspeed_config": {"zero_optimization": {"stage": 2}},
            },
            validate_paths=False,
            estimate_vram=False,
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any(msg.field == "deepspeed_config" and msg.severity == ValidationSeverity.ERROR for msg in result.messages),
            [msg.to_dict() for msg in result.messages],
        )

    def test_validation_service_requires_s3_config_for_remote_resume(self) -> None:
        service = ValidationService()
        result = service.validate_configuration(
            {"resume_from_checkpoint": "s3://bucket/jobs/run/checkpoint-100"},
            validate_paths=True,
            estimate_vram=False,
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any(
                msg.field == "resume_from_checkpoint" and msg.severity == ValidationSeverity.ERROR
                for msg in result.messages
            ),
            [msg.to_dict() for msg in result.messages],
        )

    def test_validation_service_allows_remote_resume_with_s3_config(self) -> None:
        service = ValidationService()
        result = service.validate_configuration(
            {
                "resume_from_checkpoint": "s3://bucket/jobs/run/checkpoint-100",
                "publishing_config": [{"provider": "s3", "bucket": "bucket"}],
                "num_train_epochs": 1,
            },
            validate_paths=True,
            estimate_vram=False,
        )

        self.assertTrue(result.is_valid)
