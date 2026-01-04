"""
Worker agent for connecting to SimpleTuner orchestration panel.

Usage:
    # Via command line
    simpletuner worker \\
        --orchestrator-url https://panel.example.com \\
        --worker-token abc123...

    # Via environment variables
    SIMPLETUNER_ORCHESTRATOR_URL=https://panel.example.com \\
    SIMPLETUNER_WORKER_TOKEN=abc123... \\
    simpletuner worker
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    orchestrator_url: str
    worker_token: str
    name: str
    persistent: bool = False

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        url = os.environ.get("SIMPLETUNER_ORCHESTRATOR_URL")
        token = os.environ.get("SIMPLETUNER_WORKER_TOKEN")

        if not url or not token:
            raise ValueError("SIMPLETUNER_ORCHESTRATOR_URL and SIMPLETUNER_WORKER_TOKEN environment variables are required")

        return cls(
            orchestrator_url=url.rstrip("/"),
            worker_token=token,
            name=os.environ.get("SIMPLETUNER_WORKER_NAME", socket.gethostname()),
            persistent=os.environ.get("SIMPLETUNER_WORKER_PERSISTENT", "").lower() == "true",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WorkerConfig":
        return cls(
            orchestrator_url=args.orchestrator_url.rstrip("/"),
            worker_token=args.worker_token,
            name=args.name or socket.gethostname(),
            persistent=args.persistent,
        )


def detect_gpu_info() -> Dict[str, Any]:
    """Detect GPU information"""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "count": torch.cuda.device_count(),
                "driver": torch.version.cuda,
                "accelerator": "cuda",
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "name": "Apple Silicon",
                "vram_gb": None,
                "count": 1,
                "accelerator": "mps",
            }
    except ImportError:
        # torch is not installed; fall back to nvidia-smi detection
        pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            name, vram_mb = lines[0].split(", ")
            return {
                "name": name.strip(),
                "vram_gb": round(int(vram_mb) / 1024, 1),
                "count": len(lines),
                "accelerator": "cuda",
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not available or timed out
        pass

    return {"name": "Unknown", "vram_gb": None, "count": 0, "accelerator": None}


class WorkerAgent:
    """Agent that connects to orchestrator and executes training jobs"""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id: Optional[str] = None
        self.current_job: Optional[Dict[str, Any]] = None
        self.shutdown_requested = False
        self.training_process: Optional[subprocess.Popen] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def run(self):
        """Main entry point"""
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown_signal)

        logger.info(f"Worker agent starting: {self.config.name}")
        logger.info(f"Orchestrator URL: {self.config.orchestrator_url}")
        logger.info(f"Persistent: {self.config.persistent}")

        while not self.shutdown_requested:
            try:
                await self._register()
                await self._run_event_loop()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error("Invalid worker token, exiting")
                    break
                logger.error(f"HTTP error: {e}, reconnecting in 10s...")
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Worker error: {e}, reconnecting in 10s...")
                await asyncio.sleep(10)

        await self._cleanup()
        logger.info("Worker agent stopped")

    def _handle_shutdown_signal(self):
        """Handle SIGTERM/SIGINT for graceful shutdown"""
        logger.info("Shutdown signal received")
        self.shutdown_requested = True

        if self.training_process and self.training_process.poll() is None:
            logger.info("Sending SIGTERM to training process")
            self.training_process.terminate()

    async def _register(self):
        """Register with orchestrator"""
        gpu_info = detect_gpu_info()
        logger.info(f"Detected GPU: {gpu_info}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.orchestrator_url}/api/workers/register",
                headers={"X-Worker-Token": self.config.worker_token},
                json={
                    "name": self.config.name,
                    "gpu_info": gpu_info,
                    "persistent": self.config.persistent,
                    "current_job_id": self.current_job["job_id"] if self.current_job else None,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

        self.worker_id = data["worker_id"]
        logger.info(f"Registered as worker: {self.worker_id}")

        # Handle reconciliation instructions
        if data.get("resume_job"):
            logger.info(f"Resuming job: {data['resume_job']['job_id']}")
            self.current_job = data["resume_job"]
        elif data.get("abandon_job"):
            logger.info(f"Abandoning job: {data['abandon_job']}")
            await self._stop_current_job()

    async def _run_event_loop(self):
        """Connect to SSE stream and process events"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                async with client.stream(
                    "GET",
                    f"{self.config.orchestrator_url}/api/workers/stream",
                    headers={"X-Worker-Token": self.config.worker_token},
                    params={"worker_id": self.worker_id},
                ) as response:
                    response.raise_for_status()
                    logger.info("Connected to SSE stream")

                    async for line in response.aiter_lines():
                        if self.shutdown_requested:
                            break
                        await self._handle_sse_line(line)
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    # Heartbeat task cancellation is expected during shutdown
                    pass

    async def _handle_sse_line(self, line: str):
        """Process a line from SSE stream"""
        if not line or not line.startswith("data: "):
            return

        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            logger.warning(f"Invalid SSE data: {line}")
            return

        event_type = event.get("type")
        logger.debug(f"Received event: {event_type}")

        if event_type == "job_submit":
            await self._start_job(event)
        elif event_type == "job_cancel":
            await self._stop_current_job()
        elif event_type == "shutdown":
            logger.info(f"Shutdown requested: {event.get('reason', 'unknown')}")
            self.shutdown_requested = True
        elif event_type == "ping":
            pass  # Keepalive, no action needed

    async def _start_job(self, event: Dict[str, Any]):
        """Start a training job"""
        if self.current_job:
            logger.warning("Already running a job, ignoring new job request")
            return

        self.current_job = event
        job_id = event["job_id"]
        logger.info(f"Starting job: {job_id}")

        # Write config files to temp directory
        job_dir = Path(f"/tmp/simpletuner_job_{job_id}")
        job_dir.mkdir(exist_ok=True)

        config_path = job_dir / "config.yaml"
        dataloader_path = job_dir / "dataloader.yaml"

        # Import yaml here to avoid import at module level
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(event["config"], f)
        with open(dataloader_path, "w") as f:
            yaml.dump(event["dataloader"], f)

        # Set up environment
        env = os.environ.copy()
        env.update(
            {
                "SIMPLETUNER_UPLOAD_ENDPOINT": f"{self.config.orchestrator_url}{event['upload_endpoint']}",
                "SIMPLETUNER_UPLOAD_TOKEN": event.get("upload_token", ""),
            }
        )
        if event.get("hf_token"):
            env["HF_TOKEN"] = event["hf_token"]

        # Report starting status
        await self._report_job_status("starting")

        # Launch training subprocess
        # Use the same Python interpreter
        cmd = [
            sys.executable,
            "-m",
            "simpletuner.train",
            "--config",
            str(config_path),
            "--data_config",
            str(dataloader_path),
        ]

        logger.info(f"Launching training: {' '.join(cmd)}")
        self.training_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Monitor in background
        asyncio.create_task(self._monitor_training(job_dir))

    async def _monitor_training(self, job_dir: Path):
        """Monitor training process and report status"""
        process = self.training_process

        await self._report_job_status("training")

        # Stream output (could send to server for log streaming)
        while process.poll() is None:
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    # Parse progress from output if possible
                    logger.info(f"[training] {line.rstrip()}")
            await asyncio.sleep(0.1)

        exit_code = process.returncode
        logger.info(f"Training finished with exit code: {exit_code}")

        if exit_code == 0:
            await self._report_job_status("completed")
        else:
            await self._report_job_status("failed", error=f"Training exited with code {exit_code}")

        # Cleanup
        self.current_job = None
        self.training_process = None

        # Ephemeral workers check for more jobs or shut down
        if not self.config.persistent:
            has_more = await self._check_queue()
            if not has_more:
                logger.info("No more jobs queued, shutting down ephemeral worker")
                self.shutdown_requested = True

    async def _stop_current_job(self):
        """Stop the current training job"""
        if self.training_process and self.training_process.poll() is None:
            logger.info("Stopping current job")
            self.training_process.terminate()

            # Wait for graceful shutdown
            try:
                self.training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Training process did not stop, killing")
                self.training_process.kill()

        if self.current_job:
            await self._report_job_status("cancelled")

        self.current_job = None
        self.training_process = None

    async def _report_job_status(self, status: str, error: Optional[str] = None):
        """Report job status to orchestrator"""
        if not self.current_job:
            return

        job_id = self.current_job["job_id"]

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.config.orchestrator_url}/api/workers/job/{job_id}/status",
                    headers={"X-Worker-Token": self.config.worker_token},
                    json={
                        "status": status,
                        "error": error,
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.error(f"Failed to report job status: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while not self.shutdown_requested:
            await self._send_heartbeat()
            await asyncio.sleep(30)

    async def _send_heartbeat(self):
        """Send heartbeat to orchestrator"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.config.orchestrator_url}/api/workers/heartbeat",
                    headers={"X-Worker-Token": self.config.worker_token},
                    json={
                        "worker_id": self.worker_id,
                        "status": "busy" if self.current_job else "idle",
                        "current_job_id": self.current_job["job_id"] if self.current_job else None,
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

    async def _check_queue(self) -> bool:
        """Check if there are more jobs queued for this worker"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.orchestrator_url}/api/workers/queue",
                    headers={"X-Worker-Token": self.config.worker_token},
                    params={"worker_id": self.worker_id},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("has_jobs", False)
        except Exception as e:
            logger.warning(f"Queue check failed: {e}")
        return False

    async def _cleanup(self):
        """Cleanup on shutdown"""
        await self._stop_current_job()

        # Notify orchestrator we're shutting down
        if self.worker_id:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.config.orchestrator_url}/api/workers/disconnect",
                        headers={"X-Worker-Token": self.config.worker_token},
                        json={"worker_id": self.worker_id},
                        timeout=5,
                    )
            except Exception as e:
                logger.warning(f"Failed to notify disconnect: {e}")


def main():
    """Entry point for worker agent"""
    parser = argparse.ArgumentParser(description="SimpleTuner Worker Agent")
    parser.add_argument(
        "--orchestrator-url",
        help="URL of the orchestrator panel",
    )
    parser.add_argument(
        "--worker-token",
        help="Worker authentication token",
    )
    parser.add_argument(
        "--name",
        help="Worker name (defaults to hostname)",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Run as persistent worker (don't shutdown after job)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Get config from args or environment
    if args.orchestrator_url and args.worker_token:
        config = WorkerConfig.from_args(args)
    else:
        config = WorkerConfig.from_env()

    agent = WorkerAgent(config)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
