from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BucketStageSnapshot:
    """Lightweight record describing a single stage in the bucket build pipeline."""

    name: str
    sample_count: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


@dataclass
class BucketEvent:
    """Structured record for removals or adjustments that impact available samples."""

    bucket: str
    reason: str
    removed: int
    details: Dict[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.time)


class BucketReport:
    """
    Collects lightweight telemetry while buckets are constructed so that user facing
    error messages can clearly explain why a dataset became empty.
    """

    def __init__(self, dataset_id: str, dataset_type: str) -> None:
        self.dataset_id = dataset_id
        self.dataset_type = dataset_type
        self.instance_data_dir: Optional[str] = None
        self.constraints: Dict[str, Any] = {}
        self.stage_order: List[str] = []
        self.stages: Dict[str, BucketStageSnapshot] = {}
        self.skip_counts: Counter[str] = Counter()
        self.total_processed: int = 0
        self.bucket_events: List[BucketEvent] = []
        self.notes: List[str] = []
        self.bucket_summaries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ---- basic configuration -------------------------------------------------
    def set_instance_data_dir(self, path: Optional[str]) -> None:
        with self._lock:
            if path:
                self.instance_data_dir = path

    def set_constraints(self, **constraints: Any) -> None:
        """Record constraints that influence bucket creation (min size, aspect ratio, etc.)."""
        with self._lock:
            for key, value in constraints.items():
                if value is not None:
                    self.constraints[key] = value

    def add_note(self, note: str) -> None:
        if not note:
            return
        with self._lock:
            self.notes.append(note)

    # ---- stage tracking ------------------------------------------------------
    def record_stage(self, name: str, sample_count: Optional[int] = None, **details: Any) -> None:
        with self._lock:
            snapshot = self.stages.get(name)
            if snapshot is None:
                snapshot = BucketStageSnapshot(name=name)
                self.stages[name] = snapshot
                self.stage_order.append(name)

            if sample_count is not None:
                snapshot.sample_count = sample_count
            if details:
                snapshot.details.update({k: v for k, v in details.items() if v is not None})
            snapshot.updated_at = time.time()

    def record_bucket_snapshot(self, name: str, bucket_indices: Dict[Any, Iterable[Any]]) -> None:
        """Capture a lightweight summary of bucket distribution."""
        if bucket_indices is None:
            return
        total_samples = 0
        bucket_sizes = []
        for bucket_key, samples in bucket_indices.items():
            try:
                size = len(samples)
            except TypeError:
                size = 0
            total_samples += size
            bucket_sizes.append((bucket_key, size))

        bucket_sizes.sort(key=lambda item: item[1], reverse=True)
        top_buckets = bucket_sizes[:3]

        with self._lock:
            self.bucket_summaries[name] = {
                "total_samples": total_samples,
                "bucket_count": len(bucket_sizes),
                "top_buckets": top_buckets,
            }
        self.record_stage(
            name,
            sample_count=total_samples,
            bucket_count=len(bucket_sizes),
            top_buckets=top_buckets,
        )

    # ---- statistics ----------------------------------------------------------
    def update_statistics(self, statistics: Dict[str, Any]) -> None:
        if not statistics:
            return
        skipped = statistics.get("skipped", {})
        with self._lock:
            if "total_processed" in statistics:
                self.total_processed = max(self.total_processed, int(statistics["total_processed"]))
            for reason, count in skipped.items():
                if count:
                    self.skip_counts[reason] += int(count)

    def record_bucket_event(self, bucket: str, reason: str, removed: int, **details: Any) -> None:
        if removed <= 0:
            return
        event = BucketEvent(
            bucket=bucket, reason=reason, removed=removed, details={k: v for k, v in details.items() if v is not None}
        )
        with self._lock:
            self.bucket_events.append(event)

    # ---- summary generation --------------------------------------------------
    def _format_skip_counts(self) -> Optional[str]:
        if not self.skip_counts:
            return None
        parts = []
        for reason, count in self.skip_counts.most_common():
            parts.append(f"{reason}={count}")
        return ", ".join(parts)

    def _derive_recommendations(self) -> List[str]:
        recommendations: List[str] = []
        discovered = self.stages.get("file_discovery")
        post_refresh = self.bucket_summaries.get("post_refresh")
        post_split = self.bucket_summaries.get("post_split")
        min_size = self.constraints.get("minimum_image_size")
        resolution_type = self.constraints.get("resolution_type")

        if discovered and (discovered.sample_count or 0) == 0:
            recommendations.append(
                "Check that 'instance_data_dir' points to a directory containing supported media files (image, video, or audio)."
            )
        if self.skip_counts.get("too_small") and min_size is not None:
            size_hint = f"{min_size}px" if resolution_type == "pixel" else f"{min_size}MP"
            recommendations.append(f"Lower 'minimum_image_size' (currently {size_hint}) or supply higher resolution media.")
        if self.skip_counts.get("metadata_missing"):
            recommendations.append("Ensure metadata cache files are accessible and not corrupted.")
        exhausted_by_batch = any(event for event in self.bucket_events if event.reason == "insufficient_for_batch")
        if exhausted_by_batch and self.constraints.get("effective_batch_size"):
            recommendations.append(
                "Reduce 'train_batch_size' or dataset repeats to satisfy the effective batch size requirement."
            )
        if self.skip_counts.get("too_long") and self.constraints.get("max_duration_seconds") is not None:
            recommendations.append(
                f"Increase 'audio.max_duration_seconds' (currently {self.constraints['max_duration_seconds']}) "
                "or trim long clips offline."
            )
        if self.skip_counts.get("no_audio"):
            recommendations.append(
                "Videos have no audio stream. Set 'audio.allow_zero_audio: true' to generate silent audio, "
                "or provide videos with audio tracks."
            )
        if self.skip_counts.get("processing_error"):
            recommendations.append("Some audio samples failed during processing. Check logs for detailed error messages.")
        if post_refresh and post_refresh["total_samples"] == 0 and not self.skip_counts:
            recommendations.append("Confirm skip filters (caption, quality, or custom filters) leave usable samples.")
        if post_split and post_split["total_samples"] == 0 and post_refresh and post_refresh["total_samples"] > 0:
            recommendations.append(
                "Increase dataset size or decrease effective batch size so each process receives samples."
            )

        return recommendations

    def format_empty_dataset_message(self) -> str:
        lines: List[str] = [
            f"Dataset '{self.dataset_id}' produced no usable samples.",
            f"dataset_type: {self.dataset_type}",
        ]
        if self.instance_data_dir:
            lines.append(f"instance_data_dir: {self.instance_data_dir}")

        if self.constraints:
            entries = []
            for key in (
                "minimum_image_size",
                "resolution_type",
                "minimum_aspect_ratio",
                "maximum_aspect_ratio",
                "effective_batch_size",
                "bucket_strategy",
                "duration_interval",
                "max_duration_seconds",
                "truncation_mode",
            ):
                if key in self.constraints:
                    entries.append(f"{key}={self.constraints[key]}")
            for key in ("train_batch_size", "repeats"):
                if key in self.constraints:
                    entries.append(f"{key}={self.constraints[key]}")
            if entries:
                lines.append("constraints: " + ", ".join(entries))

        for stage_name in self.stage_order:
            snapshot = self.stages.get(stage_name)
            if snapshot is None:
                continue
            details = snapshot.details.copy()
            bucket_summary = self.bucket_summaries.get(stage_name)
            if bucket_summary:
                details.setdefault("bucket_count", bucket_summary["bucket_count"])
                details.setdefault("top_buckets", bucket_summary["top_buckets"])
            detail_parts = []
            for key, value in details.items():
                if value in (None, [], {}):
                    continue
                if key == "top_buckets":
                    formatted = ", ".join(f"{bucket}:{count}" for bucket, count in value[:3])
                    if formatted:
                        detail_parts.append(f"{key}={formatted}")
                else:
                    detail_parts.append(f"{key}={value}")
            line = f"{stage_name}: {snapshot.sample_count}"
            if detail_parts:
                line += f" ({'; '.join(detail_parts)})"
            lines.append(line)

        if self.total_processed:
            lines.append(f"processed_files: {self.total_processed}")

        skip_summary = self._format_skip_counts()
        if skip_summary:
            lines.append(f"filtered_files: {skip_summary}")

        if self.bucket_events:
            for event in self.bucket_events[:5]:
                detail_str = ", ".join(f"{key}={value}" for key, value in event.details.items() if value is not None)
                if detail_str:
                    lines.append(
                        f"bucket_event[{event.bucket}]: removed={event.removed} reason={event.reason} ({detail_str})"
                    )
                else:
                    lines.append(f"bucket_event[{event.bucket}]: removed={event.removed} reason={event.reason}")
            if len(self.bucket_events) > 5:
                lines.append(f"... {len(self.bucket_events) - 5} additional bucket events omitted")

        recommendations = self._derive_recommendations()
        recommendations.extend(self.notes)
        if recommendations:
            lines.append("next_steps:")
            for idx, rec in enumerate(recommendations, start=1):
                lines.append(f"  {idx}. {rec}")

        return "\n".join(lines)
