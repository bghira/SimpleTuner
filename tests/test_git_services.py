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

    def test_identity_status_and_setter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.json"
            config_file.write_text("{}")

            status = GIT_REPO_SERVICE.init_repo(config_dir)
            self.assertIn(status.identity_configured, [True, False])

            GIT_REPO_SERVICE.set_identity(config_dir, "Tester", "tester@example.com")
            status = GIT_REPO_SERVICE.discover_repo(config_dir)
            self.assertTrue(status.identity_configured)
            self.assertEqual(status.user_name, "Tester")
            self.assertEqual(status.user_email, "tester@example.com")


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


class GitRepoServiceBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._config_store_instances = ConfigStore._instances.copy()
        ConfigStore._instances = {}

    def tearDown(self) -> None:
        ConfigStore._instances = self._config_store_instances

    def test_switch_branch_blocked_when_dirty(self) -> None:
        from simpletuner.simpletuner_sdk.server.services.git_repo_service import GitRepoError

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.json"
            config_file.write_text("{}")

            GIT_REPO_SERVICE.init_repo(config_dir)
            GIT_REPO_SERVICE.stage_and_commit(config_dir, [config_file], "initial", include_untracked=True)

            # Dirty the tree
            config_file.write_text('{"dirty": true}')

            with self.assertRaises(GitRepoError) as ctx:
                GIT_REPO_SERVICE.create_or_switch_branch(config_dir, "new-branch", create=True)

            self.assertIn("uncommitted changes", ctx.exception.message.lower())

    def test_switch_branch_succeeds_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.json"
            config_file.write_text("{}")

            GIT_REPO_SERVICE.init_repo(config_dir)
            GIT_REPO_SERVICE.stage_and_commit(config_dir, [config_file], "initial", include_untracked=True)

            status = GIT_REPO_SERVICE.create_or_switch_branch(config_dir, "new-branch", create=True)
            self.assertEqual(status.branch, "new-branch")

    def test_switch_branch_allowed_when_dirty_outside_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_dir = repo_root / "configs"
            config_dir.mkdir()
            config_file = config_dir / "config.json"
            config_file.write_text("{}")
            outside_file = repo_root / "outside.txt"
            outside_file.write_text("initial")

            GIT_REPO_SERVICE.init_repo(repo_root)
            GIT_REPO_SERVICE.stage_and_commit(repo_root, [config_file, outside_file], "initial", include_untracked=True)

            # Dirty a file outside config_dir
            outside_file.write_text("dirty")

            # Should succeed since only files outside config_dir are dirty
            status = GIT_REPO_SERVICE.create_or_switch_branch(config_dir, "new-branch", create=True)
            self.assertEqual(status.branch, "new-branch")


class GitRoutesTests(unittest.TestCase):
    """Router-level tests for git API endpoints."""

    def setUp(self) -> None:
        self._config_store_instances = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        GIT_CONFIG_SERVICE._store_cache = {}

    def tearDown(self) -> None:
        ConfigStore._instances = self._config_store_instances
        GIT_CONFIG_SERVICE._store_cache = {}

    def test_git_status_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            GIT_CONFIG_SERVICE._store_cache["model"] = store

            from simpletuner.simpletuner_sdk.server.routes.git import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/git/status")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("git_available", data)
            self.assertIn("repo_present", data)

    def test_git_init_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            GIT_CONFIG_SERVICE._store_cache["model"] = store

            from simpletuner.simpletuner_sdk.server.routes.git import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post("/api/git/init", json={})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["repo_present"])

    def test_git_identity_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            GIT_CONFIG_SERVICE._store_cache["model"] = store
            GIT_REPO_SERVICE.init_repo(store.config_dir)

            from simpletuner.simpletuner_sdk.server.routes.git import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post(
                "/api/git/identity",
                json={"name": "Test User", "email": "test@example.com"},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["user_name"], "Test User")
            self.assertEqual(data["user_email"], "test@example.com")
            self.assertTrue(data["identity_configured"])

    def test_git_branch_blocked_when_dirty(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            GIT_CONFIG_SERVICE._store_cache["model"] = store

            config_file = Path(tmpdir) / "config.json"
            config_file.write_text("{}")
            GIT_REPO_SERVICE.init_repo(store.config_dir)
            GIT_REPO_SERVICE.stage_and_commit(store.config_dir, [config_file], "initial", include_untracked=True)

            # Dirty the tree
            config_file.write_text('{"dirty": true}')

            from simpletuner.simpletuner_sdk.server.routes.git import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post("/api/git/branch", json={"name": "new-branch", "create": True})
            self.assertEqual(response.status_code, 409)
            self.assertIn("uncommitted", response.json()["detail"].lower())

    def test_git_push_requires_opt_in(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(config_dir=tmpdir, config_type="model")
            GIT_CONFIG_SERVICE._store_cache["model"] = store
            GIT_REPO_SERVICE.init_repo(store.config_dir)

            from simpletuner.simpletuner_sdk.server.routes.git import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post("/api/git/push", json={"allow_remote": False})
            self.assertEqual(response.status_code, 400)
            self.assertIn("opt-in", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
