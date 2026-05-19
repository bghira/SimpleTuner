import json
import os
import tempfile
import unittest
from pathlib import Path

from simpletuner.cli.server import _ensure_server_state_dir
from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import get_default_config_dir, get_default_db_path


class TestCloudStatePaths(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self._previous = {
            "SIMPLETUNER_STATE_DIR": os.environ.get("SIMPLETUNER_STATE_DIR"),
            "SIMPLETUNER_CONFIG_DIR": os.environ.get("SIMPLETUNER_CONFIG_DIR"),
            "SIMPLETUNER_WEB_UI_CONFIG": os.environ.get("SIMPLETUNER_WEB_UI_CONFIG"),
        }

    def tearDown(self) -> None:
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        AsyncJobStore._instance = None
        self._tmpdir.cleanup()

    def test_state_dir_takes_precedence_over_config_dir(self) -> None:
        state_dir = self.tmp_path / "local-state"
        config_dir = self.tmp_path / "shared-config"
        os.environ["SIMPLETUNER_STATE_DIR"] = str(state_dir)
        os.environ["SIMPLETUNER_CONFIG_DIR"] = str(config_dir)

        self.assertEqual(get_default_config_dir(), state_dir)
        self.assertEqual(get_default_db_path("jobs.db"), state_dir / "cloud" / "jobs.db")

    def test_async_job_store_uses_state_dir_before_webui_configs_dir(self) -> None:
        state_dir = self.tmp_path / "local-state"
        shared_configs = self.tmp_path / "shared" / "config"
        webui_dir = self.tmp_path / "webui"
        webui_dir.mkdir()
        shared_configs.mkdir(parents=True)
        (webui_dir / "defaults.json").write_text(
            json.dumps({"configs_dir": str(shared_configs)}),
            encoding="utf-8",
        )
        os.environ["SIMPLETUNER_STATE_DIR"] = str(state_dir)
        os.environ["SIMPLETUNER_WEB_UI_CONFIG"] = str(webui_dir)

        store = AsyncJobStore()

        self.assertEqual(store.get_database_path(), state_dir / "cloud" / "jobs.db")

    def test_server_sets_local_state_dir_when_unset(self) -> None:
        os.environ.pop("SIMPLETUNER_STATE_DIR", None)

        _ensure_server_state_dir()

        self.assertEqual(Path(os.environ["SIMPLETUNER_STATE_DIR"]).expanduser(), get_default_config_dir())


if __name__ == "__main__":
    unittest.main()
