"""Tests for remote worker training job execution."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from simpletuner.worker_agent import WorkerAgent, WorkerConfig
from simpletuner.worker_protocol import WORKER_LOG_TAIL_SIZE


class WorkerAgentJobStartTestCase(unittest.IsolatedAsyncioTestCase):
    """Test worker job configuration handoff to the training process."""

    def test_worker_config_has_no_kubeflow_runtime_fields(self) -> None:
        """Verify Kubernetes integration does not change Worker configuration."""
        self.assertNotIn("provider", WorkerConfig.__dataclass_fields__)
        self.assertNotIn("bound_job_id", WorkerConfig.__dataclass_fields__)

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
            "provider": "kubeflow",
            "config": config,
            "dataloader": None,
            "upload_endpoint": "/api/cloud/storage",
            "upload_token": "test-upload-token",
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
        process = AsyncMock()

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
                patch(
                    "simpletuner.worker_agent.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=process,
                ) as create_process,
                patch("simpletuner.worker_agent.asyncio.create_task", side_effect=close_background_coroutine),
            ):
                await agent._start_job(event)

            config_path = job_dir / "config.json"
            from simpletuner.train import _build_signal_consumer

            signal_file = create_process.call_args.kwargs["env"]["SIMPLETUNER_ACCELERATE_SIGNAL_FILE"]
            checkpoint = _build_signal_consumer(signal_file, "manual_checkpoint")
            validation = _build_signal_consumer(signal_file, "manual_validation")

            def packet(job, action):
                return "data: " + json.dumps({"type": "job_manual_action", "job_id": job, "action": action})

            with patch.object(agent, "_is_process_running", return_value=True):
                await agent._handle_sse_line(packet("finished-job", "checkpoint"))
                self.assertFalse(checkpoint())
                await agent._handle_sse_line(packet(job_id, "validation"))
                await agent._handle_sse_line(packet(job_id, "validation"))
                self.assertFalse(checkpoint())
                self.assertTrue(validation())
                self.assertTrue(validation())
                self.assertFalse(validation())
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), config)

            command = create_process.await_args.args
            process_env = create_process.await_args.kwargs["env"]
            self.assertEqual(command, (sys.executable, "-m", "simpletuner.train"))
            self.assertEqual(process_env["CONFIG_PATH"], str(config_path))
            self.assertEqual(process_env["SIMPLETUNER_JOB_ID"], job_id)
            self.assertEqual(
                process_env["SIMPLETUNER_WORKER_STATUS_URL"],
                f"https://orchestrator.example.com/api/workers/job/{job_id}/status",
            )
            self.assertEqual(process_env["SIMPLETUNER_CONFIG_BACKEND"], "json")
            self.assertEqual(process_env["SIMPLETUNER_UPLOAD_TOKEN"], "test-upload-token")
            report_status.assert_awaited_once_with("starting")
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    async def test_start_job_materializes_dataloader_json(self) -> None:
        """Verify dispatched dataset content becomes the active config path."""
        for provider in ("kubeflow", "worker"):
            with self.subTest(provider=provider):
                await self._check_dataset_snapshot(provider)

    async def _check_dataset_snapshot(self, provider: str) -> None:
        job_id = f"test-{uuid.uuid4().hex}"
        event = {
            "type": "job_submit",
            "job_id": job_id,
            "provider": provider,
            "config": {"model_family": "sdxl"},
            "dataloader": [{"id": "dataset", "type": "local"}],
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

        def close_background_coroutine(coroutine):
            """Close a mocked monitor coroutine.

            Args:
                coroutine: Monitor coroutine created by the Worker.

            Returns:
                Mock task placeholder.
            """
            coroutine.close()
            return Mock()

        try:
            with (
                patch.object(agent, "_report_job_status", new=AsyncMock()),
                patch("simpletuner.worker_agent.subprocess.Popen"),
                patch(
                    "simpletuner.worker_agent.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock,
                    return_value=AsyncMock(),
                ),
                patch("simpletuner.worker_agent.asyncio.create_task", side_effect=close_background_coroutine),
            ):
                await agent._start_job(event)

            dataloader_path = job_dir / "dataloader.json"
            written_config = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(dataloader_path.read_text(encoding="utf-8")), event["dataloader"])
            self.assertEqual(written_config["data_backend_config"], str(dataloader_path))
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    async def test_start_legacy_job_preserves_yaml_dataloader(self) -> None:
        """Verify ordinary Workers retain their existing YAML launch contract."""
        job_id = f"test-{uuid.uuid4().hex}"
        event = {
            "type": "job_submit",
            "job_id": job_id,
            "config": {"model_family": "sdxl"},
            "dataloader": [{"id": "dataset", "type": "local"}],
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

        def close_background_coroutine(coroutine):
            """Close the mocked monitor coroutine to avoid leaking it.

            Args:
                coroutine: Monitor coroutine created by the Worker.

            Returns:
                A mock task placeholder.
            """
            coroutine.close()
            return Mock()

        try:
            with (
                patch.object(agent, "_report_job_status", new=AsyncMock()),
                patch("simpletuner.worker_agent.subprocess.Popen", return_value=Mock()) as popen,
                patch("simpletuner.worker_agent.asyncio.create_task", side_effect=close_background_coroutine),
            ):
                await agent._start_job(event)

            self.assertTrue((job_dir / "dataloader.yaml").exists())
            self.assertFalse((job_dir / "dataloader.json").exists())
            self.assertEqual(
                json.loads((job_dir / "config.json").read_text(encoding="utf-8")),
                event["config"],
            )
            self.assertEqual(popen.call_args.args[0], [sys.executable, "-m", "simpletuner.train"])
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    async def test_ephemeral_worker_exits_after_training_when_queue_is_empty(self) -> None:
        """Verify an ordinary ephemeral Worker keeps the legacy process path."""
        agent = WorkerAgent(
            WorkerConfig(
                orchestrator_url="https://orchestrator.example.com",
                worker_token="test-token",
                name="test-worker",
            )
        )
        process = Mock()
        process.stdout = None
        process.wait.return_value = 0
        agent.training_process = process
        agent.current_job = {"job_id": "job-1"}

        with (
            patch.object(agent, "_report_job_status", new=AsyncMock()) as report_status,
            patch.object(agent, "_check_queue", new=AsyncMock(return_value=False)) as check_queue,
        ):
            await agent._monitor_training(Path("/tmp"))

        self.assertTrue(agent.shutdown_requested)
        check_queue.assert_awaited_once_with()
        self.assertEqual(report_status.await_args_list[0].args, ("training",))
        self.assertEqual(report_status.await_args_list[1].args, ("completed",))

    async def test_worker_reports_bounded_output_even_after_process_exit(self) -> None:
        agent = WorkerAgent(
            WorkerConfig(
                orchestrator_url="https://orchestrator.example.com",
                worker_token="test-token",
                name="test-worker",
                persistent=True,
            )
        )
        agent.current_job = {"job_id": "job-1"}
        log_line = "x" * 1000
        output = (log_line + "\n") * 60 + "last training line\n"
        with subprocess.Popen(
            [sys.executable, "-c", "print('last training line')"],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            # A completed process can still have unread output in its pipe.
            process.wait(timeout=5)
            agent.training_process = process
            agent._log_tail = output[: -len("last training line\n")][-WORKER_LOG_TAIL_SIZE:]
            with patch("simpletuner.worker_agent.httpx.AsyncClient") as client:
                post = client.return_value.__aenter__.return_value.post
                await agent._monitor_training(Path("/tmp"))

            self.assertEqual(post.await_args.kwargs["json"]["status"], "completed")
            self.assertEqual(post.await_args.kwargs["json"]["log_tail"], output[-WORKER_LOG_TAIL_SIZE:])

    async def test_reconnect_preserves_training_log_capture_and_completion(self) -> None:
        agent = WorkerAgent(
            WorkerConfig(
                orchestrator_url="https://orchestrator.example.com",
                worker_token="test-token",
                name="test-worker",
                persistent=True,
            )
        )
        agent.current_job = {"job_id": "job-1"}
        agent.training_process = Mock(stdout=StringIO("training output after reconnect\n"))
        agent.training_process.wait.return_value = 0
        response = Mock()
        response.json.return_value = {
            "worker_id": "worker-1",
            "resume_job": {"job_id": "job-1", "status": "running"},
        }

        async def post_request(url, *, json, **kwargs):
            if json.get("status") == "training":
                await agent._register()
            return response

        with (
            patch("simpletuner.worker_agent.detect_gpu_info", return_value={}),
            patch("simpletuner.worker_agent.httpx.AsyncClient") as client,
        ):
            post = client.return_value.__aenter__.return_value.post
            post.side_effect = post_request
            await agent._monitor_training(Path("/tmp"))

        final_update = post.await_args.kwargs["json"]
        self.assertEqual(final_update.get("status"), "completed")
        self.assertEqual(final_update.get("log_tail"), "training output after reconnect\n")
        self.assertIsNone(agent.current_job)

    async def test_stop_current_job_awaits_async_subprocess(self) -> None:
        """Verify Kubeflow cancellation awaits the asyncio subprocess API."""
        agent = WorkerAgent(
            WorkerConfig(
                orchestrator_url="https://orchestrator.example.com",
                worker_token="test-token",
                name="test-worker",
            )
        )
        process = Mock()
        process.returncode = None
        process.wait = AsyncMock(return_value=0)
        agent.training_process = process
        agent.current_job = {"job_id": "job-1"}

        with patch.object(agent, "_report_job_status", new=AsyncMock()) as report_status:
            with patch.object(agent, "_is_async_process", return_value=True):
                await agent._stop_current_job()

        process.terminate.assert_called_once_with()
        process.wait.assert_awaited_once_with()
        report_status.assert_awaited_once_with("cancelled")
        self.assertIsNone(agent.training_process)
        self.assertIsNone(agent.current_job)


if __name__ == "__main__":
    unittest.main()
