import os
import tempfile
import unittest
from unittest.mock import AsyncMock

from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus
from simpletuner.simpletuner_sdk.server.services.cloud.exceptions import InvalidConfigError
from simpletuner.simpletuner_sdk.server.services.cloud.simpletuner_io_client import SimpleTunerIOClient
from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore


class TestSimpleTunerIOClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        os.environ["SIMPLETUNER_CONFIG_DIR"] = self.temp_dir
        BaseSQLiteStore._instances.clear()
        self.client = SimpleTunerIOClient()

    async def asyncTearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        BaseSQLiteStore._instances.clear()
        os.environ.pop("SIMPLETUNER_CONFIG_DIR", None)

    async def test_run_job_requires_max_runtime(self) -> None:
        with self.assertRaises(InvalidConfigError):
            await self.client.run_job(config={}, dataloader=[])

    async def test_run_job_uses_provider_config_runtime(self) -> None:
        await self.client._config_store.update("simpletuner_io", {"max_runtime_minutes": 120})
        self.client._request = AsyncMock(
            return_value={"id": "job-1", "status": "pending", "attempt_id": "attempt-1"}
        )

        job = await self.client.run_job(config={}, dataloader=[])

        self.assertEqual(job.job_id, "job-1")
        self.assertEqual(job.status, CloudJobStatus.PENDING)
        self.client._request.assert_awaited_once()
        payload = self.client._request.call_args.kwargs["json_body"]
        self.assertEqual(payload["max_runtime_minutes"], 120)
