import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.git_config_service import (
    GIT_CONFIG_SERVICE,
    GitConfigError,
    SnapshotPreferences,
)
from simpletuner.simpletuner_sdk.server.services.git_repo_service import GIT_REPO_SERVICE
from simpletuner.simpletuner_sdk.server.services.tab_service import TabService


class GitRepoServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._config_store_instances = ConfigStore._instances.copy()
        ConfigStore._instances = {}

    def tearDown(self) -> None:
        ConfigStore._instances = self._config_store_instances

    def test_init_status_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            target = config_dir / "config.json"
            target.write_text(json.dumps({"foo": "bar"}))

            status = GIT_REPO_SERVICE.init_repo(config_dir)
            self.assertTrue(status.repo_present)

            commit_result = GIT_REPO_SERVICE.stage_and_commit(config_dir, [target], "initial commit", include_untracked=True)
            self.assertIn("commit_message", commit_result)

            history = GIT_REPO_SERVICE.log(config_dir, target)
            self.assertGreaterEqual(len(history), 1)

    def test_restore_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            file_path = config_dir / "config.json"
            file_path.write_text("one")
            GIT_REPO_SERVICE.init_repo(config_dir)
            GIT_REPO_SERVICE.stage_and_commit(config_dir, [file_path], "first", include_untracked=True)
            initial_commit = GIT_REPO_SERVICE.log(config_dir, file_path)[0]["commit"]

            file_path.write_text("two")
            GIT_REPO_SERVICE.restore_path(config_dir, file_path, commit=initial_commit)
            restored = file_path.read_text()
            self.assertEqual(restored, "one")


class GitConfigServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._config_store_instances = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        GIT_CONFIG_SERVICE._store_cache = {}

    def tearDown(self) -> None:
        ConfigStore._instances = self._config_store_instances
        GIT_CONFIG_SERVICE._store_cache = {}

    def test_snapshot_on_save_creates_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            store.save_config("env1", {"param": 1}, overwrite=True, metadata=None)
            GIT_CONFIG_SERVICE._store_cache["model"] = store
            GIT_REPO_SERVICE.init_repo(store.config_dir)
            GIT_REPO_SERVICE.stage_and_commit(
                store.config_dir, [store._get_config_path("env1")], "base", include_untracked=True
            )  # noqa: SLF001

            # Modify config and snapshot
            store.save_config("env1", {"param": 2}, overwrite=True, metadata=None)
            prefs = SnapshotPreferences(
                auto_commit=True, require_clean=False, include_untracked=True, push_on_snapshot=False
            )
            snapshot = GIT_CONFIG_SERVICE.snapshot_on_save("env1", "model", prefs, message="update")
            self.assertIsNotNone(snapshot)
            history = GIT_REPO_SERVICE.log(store.config_dir, store._get_config_path("env1"))  # noqa: SLF001
            self.assertGreaterEqual(len(history), 2)

    def test_require_clean_blocks_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            store.save_config("env1", {"param": 1}, overwrite=True, metadata=None)
            GIT_CONFIG_SERVICE._store_cache["model"] = store
            GIT_REPO_SERVICE.init_repo(store.config_dir)
            # Leave working tree dirty without committing
            prefs = SnapshotPreferences(
                auto_commit=False, require_clean=True, include_untracked=False, push_on_snapshot=False
            )
            status = GIT_REPO_SERVICE.discover_repo(store.config_dir)
            self.assertTrue(status.dirty_paths)
            with self.assertRaises(GitConfigError):
                GIT_CONFIG_SERVICE.snapshot_on_save("env1", "model", prefs, message="noop")


class TabServiceGitGatingTests(unittest.TestCase):
    def test_git_tab_gating(self) -> None:
        dummy_templates = SimpleNamespace(TemplateResponse=lambda request, name, context: {"name": name, "context": context})
        with patch("simpletuner.simpletuner_sdk.server.services.tab_service.WebUIStateStore") as mock_store_cls:
            defaults = SimpleNamespace(git_mirror_enabled=False)
            mock_store = mock_store_cls.return_value
            mock_store.load_defaults.return_value = defaults
            service = TabService(dummy_templates)  # type: ignore[arg-type]

            tabs = service.get_all_tabs()
            self.assertFalse(any(tab["name"] == "git_mirror" for tab in tabs))
            with self.assertRaises(HTTPException):
                service.get_tab_config("git_mirror")

            defaults.git_mirror_enabled = True
            tabs = service.get_all_tabs()
            self.assertTrue(any(tab["name"] == "git_mirror" for tab in tabs))
            config = service.get_tab_config("git_mirror")
            self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
