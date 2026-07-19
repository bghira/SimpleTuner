"""Tests for remote worker training job execution."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from simpletuner.worker_agent import WorkerAgent, WorkerConfig


class WorkerAgentJobStartTestCase(unittest.IsolatedAsyncioTestCase):
    """Test worker job configuration handoff to the training process."""

    async def test_start_job_uses_json_config_path(self) -> None:
        """Verify the training process receives the dispatched JSON config."""
        job_id = f"test-{uuid.uuid4().hex}"
        config = {
            "model_family": "sdxl",
            "data_backend_config": "/worker/config/dataloader.json",
            "output_dir": "/worker/output",
        }
        event = {
            "type": "job_submit",
            "job_id": job_id,
            "config": config,
            "dataloader": None,
            "upload_endpoint": "/api/cloud/storage",
        }
        agent = WorkerAgent(
            WorkerConfig(
                orchestrator_url="https://orchestrator.example.com",
                worker_token="test-token",
                name="test-worker",
                persistent=True,
            )
        )
        job_dir = Path(f"/tmp/simpletuner_job_{job_id}")
        process = Mock()

        def close_background_coroutine(coroutine):
            """Close the mocked monitor coroutine to avoid leaking it.

            Args:
                coroutine: Monitor coroutine created by the worker.

            Returns:
                A mock task placeholder.
            """
            coroutine.close()
            return Mock()

        try:
            with (
                patch.object(agent, "_report_job_status", new=AsyncMock()) as report_status,
                patch("simpletuner.worker_agent.subprocess.Popen", return_value=process) as popen,
                patch("simpletuner.worker_agent.asyncio.create_task", side_effect=close_background_coroutine),
            ):
                await agent._start_job(event)

            config_path = job_dir / "config.json"
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), config)

            command = popen.call_args.args[0]
            process_env = popen.call_args.kwargs["env"]
            self.assertEqual(command, [sys.executable, "-m", "simpletuner.train"])
            self.assertEqual(process_env["CONFIG_PATH"], str(config_path))
            self.assertEqual(process_env["SIMPLETUNER_CONFIG_BACKEND"], "json")
            report_status.assert_awaited_once_with("starting")
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
