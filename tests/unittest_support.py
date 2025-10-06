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

        self._tmpdir.cleanup()

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
