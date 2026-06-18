import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SANITIZATION_PATH = REPO_ROOT / "simpletuner" / "helpers" / "configuration" / "sanitization.py"
TRAINER_PATH = REPO_ROOT / "simpletuner" / "helpers" / "training" / "trainer.py"
SANITIZATION_MODULE = "simpletuner.helpers.configuration.sanitization"


def _load_sanitization_module() -> ModuleType:
    original_modules = {name: sys.modules.get(name) for name in ("torch", "numpy")}
    try:
        if original_modules["torch"] is None:
            torch_stub = ModuleType("torch")
            setattr(torch_stub, "dtype", type("dtype", (), {}))
            setattr(torch_stub, "device", type("device", (), {}))
            sys.modules["torch"] = torch_stub

        if original_modules["numpy"] is None:
            numpy_stub = ModuleType("numpy")
            setattr(numpy_stub, "generic", type("generic", (), {}))
            setattr(numpy_stub, "ndarray", type("ndarray", (), {}))
            sys.modules["numpy"] = numpy_stub

        spec = importlib.util.spec_from_file_location("_test_sanitization_module", SANITIZATION_PATH)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Unable to load module from {SANITIZATION_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class TrainingLaunchLoggingRegressionTests(unittest.TestCase):
    def test_trainer_imports_launch_log_sanitizer(self):
        tree = ast.parse(TRAINER_PATH.read_text())

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == SANITIZATION_MODULE:
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
                "--publishing_config="
                + json.dumps(
                    {
                        "bucket": "training",
                        "access_key": "dummy-access-key",
                        "secret_key": "dummy-secret-key",
                    },
                    separators=(",", ":"),
                ),
                "--webhook_config",
                json.dumps(
                    {"url": "https://example.invalid/webhook", "auth_token": "dummy-auth-token"},
                    separators=(",", ":"),
                ),
                "--some_json="
                + json.dumps(
                    {"safe": True, "nested": {"token": "redacted"}},
                    separators=(",", ":"),
                ),
            ]
        )

        self.assertEqual(sanitized[:3], ["accelerate", "launch", "--use_fsdp"])
        self.assertEqual(len(sanitized), 4, "expected accelerate, launch, --use_fsdp, plus the sanitized --some_json")
        self.assertTrue(sanitized[3].startswith("--some_json="))
        self.assertNotIn("--api_key=dummy-api-key", sanitized)
        self.assertFalse(any(arg.startswith("--publishing_config") for arg in sanitized))
        self.assertNotIn("--webhook_config", sanitized)
        self.assertEqual(
            json.loads(sanitized[3].split("=", 1)[1]),
            {"safe": True, "nested": {}},
        )

    def test_sanitize_cli_args_for_public_logging_leaves_invalid_json_values(self):
        sanitization = _load_sanitization_module()

        sanitized = sanitization.sanitize_cli_args_for_public_logging(
            ["accelerate", "launch", "--some_json={not-valid-json}"]
        )

        self.assertEqual(sanitized, ["accelerate", "launch", "--some_json={not-valid-json}"])


if __name__ == "__main__":
    unittest.main()
