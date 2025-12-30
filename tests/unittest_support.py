"""Helper utilities for converting legacy pytest tests to unittest."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server import ServerMode, create_app


class _APIStateGuard:
    """Small helper to redirect API state persistence to a temp location."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self._original_state_file: Optional[str] = getattr(APIState, "state_file", None)

    def __enter__(self) -> None:  # pragma: no cover - trivial
        APIState.state_file = str(self.target)
        if hasattr(APIState, "_state_file_initialised"):
            APIState._state_file_initialised = False
        APIState.clear_state()

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        APIState.clear_state()
        if self._original_state_file is not None:
            APIState.state_file = self._original_state_file
            if hasattr(APIState, "_state_file_initialised"):
                APIState._state_file_initialised = False


class APITestEnvironmentMixin:
    """Mixin that provisions temp dirs, dataset plan path, and API state sandboxing."""

    def _setup_api_environment(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

        self._previous_dataset_plan = os.environ.get("SIMPLETUNER_DATASET_PLAN_PATH")
        dataset_plan = self.tmp_path / "dataset_plan.json"
        os.environ["SIMPLETUNER_DATASET_PLAN_PATH"] = str(dataset_plan)

        # Disable TQDM progress bars during tests
        self._previous_tqdm_disable = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"

        self._api_state_guard = _APIStateGuard(self.tmp_path / "api_state.json")
        self._api_state_guard.__enter__()

    def _teardown_api_environment(self) -> None:
        try:
            self._api_state_guard.__exit__(None, None, None)
            if self._previous_dataset_plan is not None:
                os.environ["SIMPLETUNER_DATASET_PLAN_PATH"] = self._previous_dataset_plan
            else:
                os.environ.pop("SIMPLETUNER_DATASET_PLAN_PATH", None)

            # Restore TQDM setting
            if self._previous_tqdm_disable is not None:
                os.environ["TQDM_DISABLE"] = self._previous_tqdm_disable
            else:
                os.environ.pop("TQDM_DISABLE", None)

            # Clean up cloud service singletons to prevent aiosqlite thread errors
            self._cleanup_cloud_services()
        finally:
            # Always clean up temp directory to avoid ResourceWarning
            if hasattr(self, "_tmpdir") and self._tmpdir is not None:
                self._tmpdir.cleanup()

    def _cleanup_cloud_services(self) -> None:
        """Clean up cloud service singletons between tests.

        This must clear ALL storage singletons to prevent test data from
        leaking to production databases. Properly closes connections to
        avoid ResourceWarning about unclosed aiosqlite connections.
        """
        import asyncio

        async def _async_cleanup():
            # Close AsyncSQLiteStore instances (aiosqlite connections)
            try:
                from simpletuner.simpletuner_sdk.server.services.cloud.storage.async_base import AsyncSQLiteStore

                await AsyncSQLiteStore.close_all_instances()
            except (ImportError, Exception):
                pass

            # Close AsyncJobStore
            try:
                from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore

                if AsyncJobStore._instance is not None:
                    await AsyncJobStore._instance.close()
                AsyncJobStore._instance = None
            except (ImportError, Exception):
                pass

        # Run async cleanup
        try:
            asyncio.run(_async_cleanup())
        except RuntimeError:
            # Event loop already running - can't use asyncio.run
            pass

        # Reset container (high-level service registry)
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.container import container

            container.reset()
        except ImportError:
            pass

        # Reset BaseSQLiteStore instances (synchronous sqlite3)
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

            # Close connections before clearing
            for instance in BaseSQLiteStore._instances.values():
                try:
                    if hasattr(instance, "close"):
                        instance.close()
                except Exception:
                    pass
            BaseSQLiteStore._instances.clear()
        except ImportError:
            pass

        # Reset BaseAuthStore instances
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.auth.stores.base import BaseAuthStore

            for instance in BaseAuthStore._instances.values():
                try:
                    if hasattr(instance, "close"):
                        instance.close()
                except Exception:
                    pass
            BaseAuthStore._instances.clear()
        except ImportError:
            pass

        # Reset QueueStore singleton
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.queue import QueueStore

            QueueStore._instance = None
        except ImportError:
            pass

        # Reset UserStore singleton
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.auth.user_store import UserStore

            UserStore._instance = None
        except ImportError:
            pass

        # Reset module-level store singletons
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.storage import (
                audit_store,
                idempotency_store,
                metrics_store,
                reservation_store,
            )

            audit_store._audit_store = None
            metrics_store._metrics_store = None
            idempotency_store._idempotency_store = None
            reservation_store._reservation_store = None
        except (ImportError, AttributeError):
            pass

    def create_test_client(self, mode: ServerMode) -> TestClient:
        app = create_app(mode=mode)
        return TestClient(app)

    @contextmanager
    def client_session(self, mode: ServerMode):
        with self.create_test_client(mode) as client:
            yield client

    @contextmanager
    def execution_mode(self, mode: str):
        previous = os.environ.get("SIMPLETUNER_EXECUTION_MODE")
        os.environ["SIMPLETUNER_EXECUTION_MODE"] = mode
        try:
            yield
        finally:
            if previous is not None:
                os.environ["SIMPLETUNER_EXECUTION_MODE"] = previous
            else:
                os.environ.pop("SIMPLETUNER_EXECUTION_MODE", None)


class APITestCase(APITestEnvironmentMixin):
    """Standard unittest TestCase that provisions API state sandbox."""

    def setUp(self) -> None:  # pragma: no cover - exercised indirectly
        super().setUp() if hasattr(super(), "setUp") else None
        self._setup_api_environment()

    def tearDown(self) -> None:  # pragma: no cover - exercised indirectly
        self._teardown_api_environment()
        super().tearDown() if hasattr(super(), "tearDown") else None


class AsyncAPITestCase(APITestEnvironmentMixin):
    """IsolatedAsyncioTestCase-compatible mixin for async FastAPI tests."""

    async def asyncSetUp(self) -> None:  # pragma: no cover - exercised indirectly
        await super().asyncSetUp() if hasattr(super(), "asyncSetUp") else None
        self._setup_api_environment()

    async def asyncTearDown(self) -> None:  # pragma: no cover - exercised indirectly
        self._teardown_api_environment()
        await super().asyncTearDown() if hasattr(super(), "asyncTearDown") else None


def run_async(coro):
    """Run a coroutine in a synchronous test helper."""

    return asyncio.run(coro)
