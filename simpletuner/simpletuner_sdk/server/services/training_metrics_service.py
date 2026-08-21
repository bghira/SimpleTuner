from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import status

from simpletuner.helpers.training.local_metrics import (
    MANIFEST_FILENAME,
    MEDIA_FILENAME,
    METRICS_FILENAME,
    REPORT_FILENAME,
    downsample_records,
    read_manifest,
    read_media_records,
    read_metric_records,
)
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

SUPPORTED_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov", ".wav"}


class TrainingMetricsServiceError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class TrainingMetricsService:
    def _get_config_store(self) -> ConfigStore:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            return ConfigStore(config_dir=Path(defaults.configs_dir).expanduser(), config_type="model")
        return ConfigStore(config_type="model")

    def _environment(self, environment: str) -> tuple[dict[str, Any], Path]:
        store = self._get_config_store()
        try:
            config, _metadata = store.load_config(environment)
        except FileNotFoundError as exc:
            raise TrainingMetricsServiceError(f"Environment '{environment}' not found.", status.HTTP_404_NOT_FOUND) from exc
        output_dir = config.get("--output_dir") or config.get("output_dir")
        if not output_dir:
            raise TrainingMetricsServiceError(
                f"Environment '{environment}' does not have an output_dir configured.",
                status.HTTP_400_BAD_REQUEST,
            )
        return config, Path(os.path.expanduser(str(output_dir))).resolve()

    def list_runs(self) -> dict[str, Any]:
        store = self._get_config_store()
        runs = []
        for metadata in store.list_configs():
            environment = metadata.get("name")
            if not environment:
                continue
            try:
                config, output_dir = self._environment(environment)
                manifest = read_manifest(output_dir)
            except TrainingMetricsServiceError:
                continue
            if not manifest:
                continue
            runs.append(
                {
                    "environment": environment,
                    "run_name": manifest.get("run_name") or environment,
                    "project_name": manifest.get("project_name"),
                    "status": manifest.get("status", "unknown"),
                    "model_family": config.get("--model_family") or config.get("model_family"),
                    "last_step": manifest.get("last_step"),
                    "record_count": int(manifest.get("record_count", 0) or 0),
                    "metric_count": len(manifest.get("metric_names", [])),
                    "updated_at": manifest.get("updated_at"),
                    "has_report": (output_dir / REPORT_FILENAME).is_file(),
                }
            )
        runs.sort(key=lambda run: run.get("updated_at") or "", reverse=True)
        return {"runs": runs, "count": len(runs)}

    def get_run(
        self,
        environment: str,
        *,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        max_points: int = 2000,
        metric_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        config, output_dir = self._environment(environment)
        manifest = read_manifest(output_dir)
        if not manifest:
            raise TrainingMetricsServiceError(
                f"Environment '{environment}' has no SimpleTuner training metrics.",
                status.HTTP_404_NOT_FOUND,
            )

        requested_metrics = set(metric_names or [])
        records = []
        for record in read_metric_records(output_dir):
            step = int(record.get("step", 0))
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            if requested_metrics:
                record = dict(record)
                record["metrics"] = {
                    name: value for name, value in record.get("metrics", {}).items() if name in requested_metrics
                }
            records.append(record)
        records = downsample_records(records, max_points=max_points) if records else []

        media = []
        for record in read_media_records(output_dir):
            relative_path = record.get("path")
            if not isinstance(relative_path, str):
                continue
            path = (output_dir / relative_path).resolve()
            validation_root = (output_dir / "validation_images").resolve()
            if not path.is_file() or not path.is_relative_to(validation_root):
                continue
            enriched = dict(record)
            enriched["url"] = (
                f"/api/metrics/training/runs/{quote(environment, safe='')}/media/{quote(relative_path, safe='/')}"
            )
            media.append(enriched)

        return {
            "run": {
                **manifest,
                "environment": environment,
                "model_family": config.get("--model_family") or config.get("model_family"),
                "has_report": (output_dir / REPORT_FILENAME).is_file(),
            },
            "records": records,
            "media": media,
            "available_metrics": sorted(manifest.get("metric_names", [])),
            "raw_files": {
                "metrics": METRICS_FILENAME,
                "manifest": MANIFEST_FILENAME,
                "media": MEDIA_FILENAME,
                "report": REPORT_FILENAME,
            },
        }

    def media_path(self, environment: str, relative_path: str) -> Path:
        _config, output_dir = self._environment(environment)
        path = (output_dir / relative_path).resolve()
        validation_root = (output_dir / "validation_images").resolve()
        if not path.is_file() or not path.is_relative_to(validation_root):
            raise TrainingMetricsServiceError("Validation media not found.", status.HTTP_404_NOT_FOUND)
        if path.suffix.lower() not in SUPPORTED_MEDIA_SUFFIXES:
            raise TrainingMetricsServiceError("Unsupported validation media type.", status.HTTP_400_BAD_REQUEST)
        indexed_paths = {record.get("path") for record in read_media_records(output_dir)}
        if path.relative_to(output_dir).as_posix() not in indexed_paths:
            raise TrainingMetricsServiceError("Validation media is not indexed for this run.", status.HTTP_404_NOT_FOUND)
        return path

    def report_path(self, environment: str) -> Path:
        _config, output_dir = self._environment(environment)
        path = output_dir / REPORT_FILENAME
        if not path.is_file():
            raise TrainingMetricsServiceError("Training report not found.", status.HTTP_404_NOT_FOUND)
        return path


TRAINING_METRICS_SERVICE = TrainingMetricsService()
