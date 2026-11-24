import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService


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

    def test_worker_defaults_follow_local_environments_and_stay_out_of_exports(self):
        store = ConfigStore(config_dir=self.config_root)
        for name in ("default", "flat", "folder"):
            with self.subTest(name=name):
                if name == "folder":
                    (self.config_root / name).mkdir()
                store.save_trainer_config(name, {"--model_family": "flux"}, overwrite=True)
                store.save_worker_defaults(name, ["b", "a"])
                store.save_trainer_config(name, {"--model_family": "flux"}, overwrite=True)
                config, metadata = store.load_config(name)
                self.assertNotIn("default_worker_ids", config)
                self.assertEqual(metadata.default_worker_ids, ["b", "a"])
                self.assertEqual(store.copy_config(name, name + "-copy").default_worker_ids, ["b", "a"])
                self.assertEqual(store.rename_config(name + "-copy", name + "-new").default_worker_ids, ["b", "a"])
                self.assertEqual(store.load_config(name + "-new")[1].default_worker_ids, ["b", "a"])
                self.assertNotIn("default_worker_ids", store.export_config(name)["_metadata"])
                store.save_worker_defaults(name, [])
                self.assertEqual(store.load_config(name)[1].default_worker_ids, [])

    def test_worker_dataset_copy_preserves_flat_environment_until_enabled(self):
        service = ConfigsService()
        store = ConfigStore(config_dir=self.config_root)
        original = {"data_backend_config": "old-datasets.json"}
        datasets = [{"id": "worker-images", "type": "local", "dataset_type": "image"}]
        store.save_config("flat", original)
        with patch.object(ConfigsService, "_get_store", return_value=store):
            previous = service.get_config("flat")["config"]
            pulled = service.save_worker_dataloader_copy("flat", datasets)
            self.assertFalse(pulled["enabled"])
            self.assertEqual(service.get_config("flat")["config"], previous)
            with self.assertRaises(ConfigServiceError) as existing:
                service.save_worker_dataloader_copy("flat", [])
            self.assertEqual(existing.exception.status_code, 409)
            self.assertEqual(json.loads((self.config_root / pulled["path"]).read_text()), datasets)
            enabled = service.save_worker_dataloader_copy("flat")
            self.assertTrue(enabled["enabled"])
            result = service.get_config("flat")
            self.assertEqual(result["metadata"]["worker_dataloader_copy"], enabled)
            self.assertEqual(result["config"]["data_backend_config"], pulled["path"])
        self.assertTrue((self.config_root / "flat.json").is_file())

    def test_dataloader_save_preserves_list_payload(self) -> None:
        store = ConfigStore(config_dir=self.config_root / "dataloaders", config_type="dataloader")

        datasets = [{"id": "primary", "type": "local", "dataset_type": "image", "instance_data_dir": "data"}]

        store.save_config("example-env", datasets, overwrite=False)

        config_path = store._get_config_path("example-env")
        self.assertTrue(config_path.exists())

        saved = self._read_json(config_path)
        self.assertIsInstance(saved, list)
        self.assertEqual(saved, datasets)


if __name__ == "__main__":
    unittest.main()
