"""Cog helper utilities for running SimpleTuner training jobs without shell wrappers.

This module provides a small orchestration layer that:
- stages user-provided training data archives into a local dataset directory
- materializes a minimal data backend config pointing at that dataset
- loads a JSON training config (defaulting to config/config.json if present)
- launches SimpleTuner training via run_trainer_job
- packages the output directory for return to Cog callers
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from simpletuner.helpers.configuration.loader import load_config
from simpletuner.helpers.training.trainer import run_trainer_job


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
        dataset_archive: Path,
        hf_token: Optional[str] = None,
        base_config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        max_train_steps: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage data, build configs, and launch training via run_trainer_job.

        Returns a result dict containing:
        - job_id
        - dataset_dir
        - output_dir
        - dataset_config_path
        - training_result (whatever run_trainer_job returns)
        """

        job = job_id or self._new_job_id()
        dataset_dir = self._stage_dataset(dataset_archive, job)

        base_config = self._load_base_config(base_config_path)
        merged_config = dict(base_config)
        if config_overrides:
            merged_config.update(config_overrides)
        if max_train_steps is not None:
            merged_config["--max_train_steps"] = max_train_steps

        output_dir = self.output_root / job
        dataset_config_path = self._write_dataset_config(job, dataset_dir, merged_config, output_dir)

        merged_config.setdefault("--output_dir", str(output_dir))
        merged_config["--data_backend_config"] = str(dataset_config_path)
        merged_config["__job_id__"] = job

        self._apply_hf_token(hf_token)

        training_result = run_trainer_job(merged_config)

        return {
            "job_id": job,
            "dataset_dir": str(dataset_dir),
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
