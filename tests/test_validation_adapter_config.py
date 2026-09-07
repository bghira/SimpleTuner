import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.helpers.configuration.loader import load_config
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation_adapters import build_validation_adapter_runs
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService


class ValidationAdapterConfigTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.configs_dir = Path(self.tempdir.name).resolve() / "configs"
        self.configs_dir.mkdir()
        self.addCleanup(StateTracker.set_args, StateTracker.get_args())
        self.adapter = {"label": "Turbo adapter", "path": "org/turbo-adapter:adapter.safetensors"}

    def _load_webui_runs(self, value):
        normalized = ConfigsService.normalize_form_to_config(
            {"validation_adapter_config": value}, configs_dir=str(self.configs_dir)
        )
        persisted = json.loads(json.dumps(normalized))
        args = load_config(
            {
                "model_family": "pixart",
                "model_type": "lora",
                "optimizer": "adamw_bf16",
                "output_dir": "output",
                "data_backend_config": "config.json",
                "resume_from_checkpoint": "latest",
                **persisted,
            },
            exit_on_error=True,
        )
        return build_validation_adapter_runs(None, args.validation_adapter_config)

    def test_saved_file_config_matches_inline_json(self):
        layouts = {
            "single": self.adapter,
            "list": [self.adapter, {"label": "Style adapter", "path": "org/style-adapter"}],
            "runs": {"runs": [self.adapter]},
        }
        for layout, payload in layouts.items():
            path = self.configs_dir / f"validation-adapters-{layout}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            expected = build_validation_adapter_runs(None, payload)
            with self.subTest(layout=layout, source="inline"):
                self.assertEqual(self._load_webui_runs(json.dumps(payload)), expected)
            for source, value in (("absolute", str(path)), ("relative", path.name)):
                with self.subTest(layout=layout, source=source):
                    self.assertEqual(self._load_webui_runs(value), expected)

    def test_file_outside_config_directory_stays_loadable(self):
        path = Path(self.tempdir.name) / "external-adapters.json"
        path.write_text(json.dumps(self.adapter), encoding="utf-8")
        self.assertEqual(self._load_webui_runs(str(path)), build_validation_adapter_runs(None, self.adapter))

    def test_malformed_json_file_raises(self):
        path = self.configs_dir / "malformed-adapters.json"
        path.write_text('{"path":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Could not load --validation_adapter_config"):
            self._load_webui_runs(str(path))

    def test_invalid_adapter_entry_raises_for_file_and_inline(self):
        payload = {"label": "Missing adapter path"}
        path = self.configs_dir / "invalid-adapters.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        for value in (str(path), json.dumps(payload)):
            with self.subTest(source="file" if value == str(path) else "inline"):
                with self.assertRaisesRegex(ValueError, "Adapter run must include at least one adapter path"):
                    self._load_webui_runs(value)


if __name__ == "__main__":
    unittest.main()
