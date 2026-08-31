from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from accelerate.tracking import GeneralTracker

from simpletuner.helpers.training.reporting import report_to_contains

SCHEMA_VERSION = 1
METRICS_FILENAME = "training_metrics.jsonl"
MANIFEST_FILENAME = "training_metrics.json"
MEDIA_FILENAME = "validation_media.jsonl"
REPORT_FILENAME = "training_report.html"
REPORT_REFRESH_INTERVAL = 100

_CONFIG_KEYS = (
    "model_family",
    "model_flavour",
    "model_type",
    "pretrained_model_name_or_path",
    "optimizer",
    "learning_rate",
    "lr_scheduler",
    "lr_warmup_steps",
    "train_batch_size",
    "gradient_accumulation_steps",
    "max_train_steps",
    "num_train_epochs",
    "seed",
    "resolution",
    "validation_resolution",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    content = path.read_text(encoding="utf-8")
    records = []
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            is_torn_final_write = index == len(lines) - 1 and not content.endswith("\n")
            if is_torn_final_write:
                break
            raise
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object in {path.name} at line {index + 1}.")
        records.append(record)
    return records


def read_manifest(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(output_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{MANIFEST_FILENAME} must contain a JSON object.")
    return payload


def read_metric_records(output_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return read_jsonl(Path(output_dir) / METRICS_FILENAME)


def read_media_records(output_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return read_jsonl(Path(output_dir) / MEDIA_FILENAME)


def downsample_records(records: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if max_points < 2:
        raise ValueError("max_points must be at least 2.")
    if len(records) <= max_points:
        return records

    last_index = len(records) - 1
    selected = {round(index * last_index / (max_points - 1)) for index in range(max_points)}
    return [records[index] for index in sorted(selected)]


def is_local_metrics_enabled(report_to: Any) -> bool:
    return report_to_contains(report_to, "simpletuner")


def _coerce_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif hasattr(value, "numel") and callable(value.numel) and value.numel() == 1:
        result = float(value.detach().item()) if hasattr(value, "detach") else float(value.item())
    elif hasattr(value, "item") and callable(value.item):
        try:
            result = float(value.item())
        except (TypeError, ValueError):
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _filtered_config(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values[key] for key in _CONFIG_KEYS if values.get(key) is not None}


def _media_for_report(output_dir: Path) -> list[dict[str, Any]]:
    media = []
    for record in read_media_records(output_dir):
        relative_path = record.get("path")
        if not isinstance(relative_path, str):
            continue
        path = (output_dir / relative_path).resolve()
        if path.is_file() and path.is_relative_to(output_dir.resolve()):
            media.append(record)
    return media


def _json_for_html(payload: Any) -> str:
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_static_report(output_dir: str | os.PathLike[str], max_points: int = 5000) -> Path:
    output_path = Path(output_dir).resolve()
    package_root = Path(__file__).resolve().parents[2]
    template_path = package_root / "templates" / "training_metrics_report.html"
    chart_script_path = package_root / "static" / "js" / "training_metrics_chart.js"
    chart_style_path = package_root / "static" / "css" / "training_metrics_report.css"

    records = downsample_records(read_metric_records(output_path), max_points=max_points)
    payload = {
        "run": read_manifest(output_path),
        "records": records,
        "media": _media_for_report(output_path),
    }
    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__SIMPLETUNER_REPORT_CSS__", chart_style_path.read_text(encoding="utf-8"))
    html = html.replace("__SIMPLETUNER_REPORT_DATA__", _json_for_html(payload))
    html = html.replace("__SIMPLETUNER_CHART_JS__", chart_script_path.read_text(encoding="utf-8"))
    report_path = output_path / REPORT_FILENAME
    temporary = report_path.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, report_path)
    return report_path


def record_validation_media(
    config: Any,
    media_path: str | os.PathLike[str],
    *,
    media_type: str,
    label: str,
    index: int,
    resolution: Optional[str] = None,
) -> None:
    if config is None or not is_local_metrics_enabled(getattr(config, "report_to", None)):
        return

    output_dir = Path(getattr(config, "output_dir")).expanduser().resolve()
    path = Path(media_path).resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError("Validation media for local metrics must be inside output_dir.")
    _append_json_line(
        output_dir / MEDIA_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _utc_now(),
            "step": int(_state_tracker_step()),
            "type": media_type,
            "label": str(label),
            "index": int(index),
            "resolution": resolution,
            "path": path.relative_to(output_dir).as_posix(),
        },
    )


def _state_tracker_step() -> int:
    from simpletuner.helpers.training.state_tracker import StateTracker

    return StateTracker.get_global_step()


class LocalMetricsTracker(GeneralTracker):
    name = "simpletuner"
    requires_logging_directory = False
    main_process_only = True

    def __init__(self, run_name: str, output_dir: str, project_name: str):
        super().__init__()
        self.run_name = run_name
        self.project_name = project_name
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / METRICS_FILENAME
        self.manifest_path = self.output_dir / MANIFEST_FILENAME
        self._manifest = read_manifest(self.output_dir)
        self._metric_names = set(self._manifest.get("metric_names", []))
        self._record_count = int(self._manifest.get("record_count", 0) or 0)

    @property
    def tracker(self):
        return self

    def store_init_configuration(self, values: dict[str, Any]) -> None:
        now = _utc_now()
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "project_name": self.project_name,
            "status": "running",
            "created_at": self._manifest.get("created_at", now),
            "updated_at": now,
            "last_step": self._manifest.get("last_step"),
            "record_count": self._record_count,
            "metric_names": sorted(self._metric_names),
            "config": _filtered_config(values),
        }
        self._write_manifest()
        render_static_report(self.output_dir)

    def log(self, values: dict[str, Any], step: Optional[int] = None, **kwargs) -> None:
        metrics = {}
        for key, value in values.items():
            scalar = _coerce_scalar(value)
            if scalar is not None:
                metrics[str(key)] = scalar
        if not metrics:
            return

        resolved_step = int(step) if step is not None else self._record_count
        _append_json_line(
            self.metrics_path,
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": _utc_now(),
                "step": resolved_step,
                "metrics": metrics,
            },
        )
        self._record_count += 1
        self._metric_names.update(metrics)
        self._manifest.update(
            {
                "status": "running",
                "updated_at": _utc_now(),
                "last_step": resolved_step,
                "record_count": self._record_count,
                "metric_names": sorted(self._metric_names),
            }
        )
        self._write_manifest()
        if self._record_count == 1 or self._record_count % REPORT_REFRESH_INTERVAL == 0:
            render_static_report(self.output_dir)

    def finish(self) -> None:
        self._manifest.update({"status": "completed", "updated_at": _utc_now()})
        self._write_manifest()
        render_static_report(self.output_dir)

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._manifest)
