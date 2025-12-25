"""Cog helper utilities for running SimpleTuner training jobs without shell wrappers.

This module provides a small orchestration layer that:
- stages user-provided training data archives into a local dataset directory
- materializes a minimal data backend config pointing at that dataset
- loads a JSON training config (defaulting to config/config.json if present)
- launches SimpleTuner training via run_trainer_job
- packages the output directory for return to Cog callers
- captures webhook events and prints them to Cog logs
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import threading
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from simpletuner.helpers.configuration.loader import load_config
from simpletuner.helpers.training.trainer import run_trainer_job


class CogWebhookReceiver:
    """
    Local HTTP server that receives SimpleTuner webhook events and prints them
    to stdout for Cog's log capture system.
    """

    def __init__(self, port: int = 0):
        """
        Initialize the webhook receiver.

        Args:
            port: Port to listen on. 0 means pick a free port automatically.
        """
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """Return the actual port the server is listening on."""
        if self._server is None:
            raise RuntimeError("Server not started")
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        """Return the callback URL for SimpleTuner webhook config."""
        return f"http://127.0.0.1:{self.port}/webhook"

    def start(self) -> "CogWebhookReceiver":
        """Start the webhook receiver server in a background thread."""
        handler = self._create_handler()
        self._server = HTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[COG] Webhook receiver started on {self.url}")
        return self

    def stop(self) -> None:
        """Stop the webhook receiver server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[COG] Webhook receiver stopped")

    def __enter__(self) -> "CogWebhookReceiver":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _create_handler(self):
        """Create the HTTP request handler class."""

        class WebhookHandler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                # Suppress default HTTP logging
                pass

            def do_POST(self):
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode("utf-8"))
                    self._handle_event(data)
                    self.send_response(200)
                    self.end_headers()
                except Exception as e:
                    print(f"[COG] Webhook error: {e}")
                    self.send_response(500)
                    self.end_headers()

            def do_GET(self):
                # Health check endpoint
                self.send_response(200)
                self.end_headers()

            def _handle_event(self, data: dict):
                """Format and print the webhook event."""
                event_type = data.get("type", "unknown")

                if event_type == "lifecycle.stage":
                    self._handle_lifecycle_stage(data)
                elif event_type == "training.status":
                    self._handle_training_status(data)
                elif event_type == "training.checkpoint":
                    self._handle_checkpoint(data)
                elif event_type == "notification":
                    self._handle_notification(data)
                elif event_type == "error":
                    self._handle_error(data)
                else:
                    # Generic fallback
                    message = data.get("message", data.get("title", ""))
                    if message:
                        print(f"[EVENT] {event_type}: {message}")

            def _handle_lifecycle_stage(self, data: dict):
                """Handle lifecycle.stage events."""
                stage = data.get("stage", {})
                label = stage.get("label", stage.get("key", "unknown"))
                status = stage.get("status", "")
                message = data.get("message", "")

                progress = stage.get("progress", {})
                progress_str = ""
                if progress.get("percent") is not None:
                    progress_str = f" ({progress['percent']:.1f}%)"
                elif progress.get("current") is not None and progress.get("total") is not None:
                    progress_str = f" ({progress['current']}/{progress['total']})"

                status_icon = {"running": "⏳", "completed": "✓", "failed": "✗"}.get(status, "•")

                if message:
                    print(f"[STAGE] {status_icon} {label}{progress_str}: {message}")
                else:
                    print(f"[STAGE] {status_icon} {label}{progress_str}")

            def _handle_training_status(self, data: dict):
                """Handle training.status events."""
                status = data.get("status", "")
                message = data.get("message", "")

                step_info = ""
                if "step" in data:
                    step = data.get("step", 0)
                    total_steps = data.get("total_steps", 0)
                    if total_steps > 0:
                        pct = (step / total_steps) * 100
                        step_info = f"Step {step}/{total_steps} ({pct:.1f}%) "

                if message:
                    print(f"[TRAINING] {step_info}{message}")
                elif status:
                    print(f"[TRAINING] {step_info}Status: {status}")

            def _handle_checkpoint(self, data: dict):
                """Handle training.checkpoint events."""
                label = data.get("label", "Checkpoint saved")
                print(f"[CHECKPOINT] {label}")

            def _handle_notification(self, data: dict):
                """Handle notification events."""
                message = data.get("message", "")
                title = data.get("title", "")
                severity = data.get("severity", "info")

                prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO", "debug": "DEBUG"}.get(severity, "INFO")

                if title and message:
                    print(f"[{prefix}] {title}: {message}")
                elif message:
                    print(f"[{prefix}] {message}")

            def _handle_error(self, data: dict):
                """Handle error events."""
                message = data.get("message", "Unknown error")
                title = data.get("title", "Error")
                print(f"[ERROR] {title}: {message}")

        return WebhookHandler

    @staticmethod
    def build_webhook_config(callback_url: str) -> Dict[str, Any]:
        """
        Build a SimpleTuner webhook config dict for the Cog receiver.

        Args:
            callback_url: The URL to receive webhook events.

        Returns:
            A webhook config dict suitable for SimpleTuner.
        """
        return {
            "webhook_type": "raw",
            "callback_url": callback_url,
            "log_level": "info",
            "message_prefix": "cog",
        }


class SimpleTunerCogRunner:
    """Utility to prepare a Cog training run using SimpleTuner."""

    def __init__(
        self,
        *,
        dataset_root: Path | str = Path("datasets") / "cog",
        output_root: Path | str = Path("output") / "cog",
        config_root: Path | str = Path("config") / "cog",
        debug_log: Path | str = Path("debug.log"),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.config_root = Path(config_root)
        self.debug_log_path = Path(debug_log)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        dataset_archive: Optional[Path] = None,
        hf_token: Optional[str] = None,
        base_config_path: Optional[Path] = None,
        base_config_dict: Optional[Dict[str, Any]] = None,
        dataloader_config_path: Optional[Path] = None,
        dataloader_config_dict: Optional[Any] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        max_train_steps: Optional[int] = None,
        job_id: Optional[str] = None,
        webhook_config: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage data, build configs, and launch training via run_trainer_job.

        Args:
            dataset_archive: Zip/tar of training images. Not required if dataloader config is provided.
            base_config_path: Path to training config JSON.
            base_config_dict: Training config as a dict (alternative to base_config_path).
            dataloader_config_path: Path to multidatabackend config JSON.
            dataloader_config_dict: Multidatabackend config as a dict/list (alternative to path).

        Returns a result dict containing:
        - job_id
        - dataset_dir (None if using user-provided dataloader config)
        - output_dir
        - dataset_config_path
        - training_result (whatever run_trainer_job returns)
        """

        job = job_id or self._new_job_id()

        # Load base config from path or use provided dict
        if base_config_dict:
            base_config = base_config_dict
        else:
            base_config = self._load_base_config(base_config_path)

        merged_config = dict(base_config)
        if config_overrides:
            merged_config.update(config_overrides)
        if max_train_steps is not None:
            merged_config["--max_train_steps"] = max_train_steps

        output_dir = self.output_root / job
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use provided dataloader config, or auto-generate one from the images archive
        dataset_dir = None
        if dataloader_config_dict is not None:
            # Write the provided dict to a temp file
            dataset_config_path = self.config_root / f"{job}_multidatabackend.json"
            with dataset_config_path.open("w", encoding="utf-8") as handle:
                json.dump(dataloader_config_dict, handle, indent=2)
        elif dataloader_config_path:
            dataset_config_path = Path(dataloader_config_path)
            if not dataset_config_path.exists():
                raise FileNotFoundError(f"Dataloader config not found: {dataloader_config_path}")
        else:
            if not dataset_archive:
                raise ValueError("Either dataset_archive or dataloader config must be provided.")
            dataset_dir = self._stage_dataset(dataset_archive, job)
            dataset_config_path = self._write_dataset_config(job, dataset_dir, merged_config, output_dir)

        merged_config.setdefault("--output_dir", str(output_dir))
        merged_config["--data_backend_config"] = str(dataset_config_path)
        merged_config["__job_id__"] = job

        if webhook_config:
            merged_config["webhook_config"] = webhook_config

        self._apply_hf_token(hf_token)

        training_result = run_trainer_job(merged_config)

        return {
            "job_id": job,
            "dataset_dir": str(dataset_dir) if dataset_dir else None,
            "output_dir": str(output_dir),
            "dataset_config_path": str(dataset_config_path),
            "training_result": training_result,
        }

    def package_output(self, output_dir: Path, *, target: Path = Path("/tmp/output.zip")) -> Path:
        """Zip the output directory to a single archive for Cog return."""

        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()

        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in output_dir.rglob("*"):
                archive.write(path, arcname=path.relative_to(output_dir))

        return target

    def read_debug_log(self, *, max_bytes: int = 200_000) -> str:
        """Return up to the last `max_bytes` of debug.log."""

        if not self.debug_log_path.exists():
            return ""

        data = self.debug_log_path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode(errors="replace")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _new_job_id(self) -> str:
        return f"cog-{uuid.uuid4().hex[:10]}"

    def _stage_dataset(self, archive_path: Path, job_id: str) -> Path:
        """Extract a zip/tar archive into a job-scoped dataset directory."""

        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

        dest = self.dataset_root / job_id
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(archive_path):
            self._extract_zip(archive_path, dest)
        elif tarfile.is_tarfile(archive_path):
            self._extract_tar(archive_path, dest)
        else:
            raise ValueError("Unsupported dataset archive format. Use .zip or .tar.")

        return dest

    def _extract_zip(self, archive_path: Path, dest: Path) -> None:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                # Basic traversal guard
                target = (dest / member.filename).resolve()
                if dest not in target.parents and target != dest:
                    raise ValueError(f"Unsafe zip path detected: {member.filename}")
            archive.extractall(dest)

    def _extract_tar(self, archive_path: Path, dest: Path) -> None:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                target = (dest / member.name).resolve()
                if dest not in target.parents and target != dest:
                    raise ValueError(f"Unsafe tar path detected: {member.name}")
            archive.extractall(dest)

    def _load_base_config(self, candidate: Optional[Path]) -> Dict[str, Any]:
        """Load a JSON config mapping to feed run_trainer_job."""

        path = self._resolve_config_path(candidate)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Config at {path} must be a JSON object of CLI args.")
        return data

    def _resolve_config_path(self, candidate: Optional[Path]) -> Path:
        if candidate:
            resolved = Path(candidate)
            if not resolved.exists():
                raise FileNotFoundError(f"Config file not found: {candidate}")
            return resolved

        defaults = [
            Path("config") / "config.json",
            Path("config") / "config.json.example",
        ]
        for path in defaults:
            if path.exists():
                return path
        raise FileNotFoundError("No training config found. Provide a config JSON or create config/config.json.")

    def _write_dataset_config(
        self,
        job_id: str,
        dataset_dir: Path,
        base_config: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Materialize a minimal data backend config for the staged dataset."""

        resolution = base_config.get("--resolution") or base_config.get("resolution")
        resolution_type = base_config.get("--resolution_type") or base_config.get("resolution_type")

        dataset_entry: Dict[str, Any] = {
            "id": f"{job_id}-images",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": str(dataset_dir),
            "caption_strategy": "filename",
            "crop": True,
            "crop_style": "center",
        }
        if resolution is not None:
            dataset_entry["resolution"] = resolution
        if resolution_type is not None:
            dataset_entry["resolution_type"] = resolution_type

        dataset_entry["cache_dir_vae"] = str(output_dir / "vae-cache")

        config_path = self.config_root / f"{job_id}_multidatabackend.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump([dataset_entry], handle, indent=2)
        return config_path

    def _apply_hf_token(self, hf_token: Optional[str]) -> None:
        """Expose the HF token to downstream libraries via standard env vars."""

        if not hf_token:
            return

        for var in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN"):
            os.environ[var] = hf_token
