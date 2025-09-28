import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore


class ConfigStoreSaveBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.config_root = Path(self._tmpdir.name)

    def _read_json(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_save_config_strips_metadata_wrapper(self) -> None:
        store = ConfigStore(config_dir=self.config_root)

        payload = {
            "_metadata": {"name": "example", "description": "Example config"},
            "config": {
                "--model_family": "flux",
                "--model_type": "lora",
                "--output_dir": "output/example",
            },
        }

        store.save_config("example-env", payload, overwrite=False)

        config_path = store._get_config_path("example-env")
        self.assertTrue(config_path.exists())

        saved = self._read_json(config_path)
        self.assertIsInstance(saved, dict)
        self.assertNotIn("_metadata", saved)
        self.assertNotIn("config", saved)
        self.assertEqual(saved["--model_family"], "flux")
        self.assertEqual(saved["--model_type"], "lora")
        self.assertEqual(saved["--output_dir"], "output/example")

    def test_dataloader_save_preserves_list_payload(self) -> None:
        store = ConfigStore(config_dir=self.config_root / "dataloaders", config_type="dataloader")

        datasets = [
            {"id": "primary", "type": "local", "dataset_type": "image", "instance_data_dir": "data"}
        ]

        store.save_config("example-env", datasets, overwrite=False)

        config_path = store._get_config_path("example-env")
        self.assertTrue(config_path.exists())

        saved = self._read_json(config_path)
        self.assertIsInstance(saved, list)
        self.assertEqual(saved, datasets)


if __name__ == "__main__":
    unittest.main()
