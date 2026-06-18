import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SANITIZATION_PATH = REPO_ROOT / "simpletuner" / "helpers" / "configuration" / "sanitization.py"
TRAINER_PATH = REPO_ROOT / "simpletuner" / "helpers" / "training" / "trainer.py"


def _ensure_sanitization_import_stubs() -> None:
    if "torch" not in sys.modules:
        torch_stub = ModuleType("torch")
        torch_stub.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
        torch_stub.device = type("device", (), {})  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub

    if "numpy" not in sys.modules:
        numpy_stub = ModuleType("numpy")
        numpy_stub.generic = type("generic", (), {})  # type: ignore[attr-defined]
        numpy_stub.ndarray = type("ndarray", (), {})  # type: ignore[attr-defined]
        sys.modules["numpy"] = numpy_stub


def _load_sanitization_module():
    _ensure_sanitization_import_stubs()
    spec = importlib.util.spec_from_file_location("_test_sanitization_module", SANITIZATION_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive loader guard
        raise AssertionError(f"Unable to load module from {SANITIZATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingLaunchLoggingRegressionTests(unittest.TestCase):
    def test_trainer_imports_launch_log_sanitizer(self):
        tree = ast.parse(TRAINER_PATH.read_text())

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "simpletuner.helpers.configuration.sanitization":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn("sanitize_cli_args_for_public_logging", imported_names)

    def test_sanitize_cli_args_for_public_logging_strips_sensitive_payloads(self):
        sanitization = _load_sanitization_module()

        sanitized = sanitization.sanitize_cli_args_for_public_logging(
            [
                "accelerate",
                "launch",
                "--api_key=dummy-api-key",
                "--use_fsdp",
                "--publishing_config={\"bucket\":\"training\",\"access_key\":\"dummy-access-key\",\"secret_key\":\"dummy-secret-key\"}",
                "--webhook_config",
                "{\"url\":\"https://example.invalid/webhook\",\"auth_token\":\"dummy-auth-token\"}",
                "--some_json={\"safe\":true,\"nested\":{\"token\":\"redacted\"}}",
            ]
        )

        self.assertEqual(
            sanitized,
            [
                "accelerate",
                "launch",
                "--use_fsdp",
                "--some_json={\"nested\": {}, \"safe\": true}",
            ],
        )


if __name__ == "__main__":
    unittest.main()
