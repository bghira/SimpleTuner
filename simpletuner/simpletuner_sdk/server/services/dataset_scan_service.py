"""Service for triggering standalone dataset metadata scans from the WebUI."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScanJob:
    job_id: str
    dataset_id: str
    status: ScanStatus = ScanStatus.PENDING
    current: int = 0
    total: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class ScanQueue:
    queue_id: str
    dataset_ids: List[str]
    jobs: Dict[str, ScanJob] = field(default_factory=dict)
    current_index: int = 0
    cancelled: bool = False


class _ScanAcceleratorStub:
    """Minimal accelerator stub for running metadata scans outside the trainer."""

    num_processes = 1
    process_index = 0
    is_main_process = True
    is_local_main_process = True
    device = "cpu"
    data_parallel_rank = 0
    data_parallel_shard_rank = 0

    def wait_for_everyone(self):
        pass

    @contextmanager
    def main_process_first(self):
        yield

    @contextmanager
    def split_between_processes(self, data, apply_padding=False):
        yield data


class _ScanArgsNamespace:
    """Minimal args namespace to satisfy StateTracker.get_args() during scan."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.aspect_bucket_worker_count = 4
        self.enable_multiprocessing = False
        self.delete_problematic_images = False
        self.delete_unwanted_images = False
        self.metadata_update_interval = 3600
        self.resolution = 1024
        self.resolution_type = "pixel"
        self.minimum_image_size = 0
        self.maximum_image_size = None
        self.target_downsample_size = None
        self.train_batch_size = 1
        self.caption_strategy = "filename"
        self.compress_disk_cache = False
        self.model_type = "full"
        self.model_family = None
        self.output_dir = "output/scan"
        self.disable_bucket_pruning = True
        self.ignore_missing_files = False
        self.controlnet = False
        self.allow_dataset_oversubscription = False
        self.data_backend_sampling = "uniform"
        self.framerate = 24
        self.gradient_accumulation_steps = 1
        if overrides:
            for k, v in overrides.items():
                setattr(self, k, v)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class DatasetScanService:
    """Manages standalone dataset metadata scans with progress broadcasting."""

    _lock = threading.Lock()
    _active_job: Optional[ScanJob] = None
    _active_queue: Optional[ScanQueue] = None
    _scan_thread: Optional[threading.Thread] = None

    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self._broadcast_fn = broadcast_fn

    def _broadcast(self, event_type: str, data: Dict[str, Any]):
        if self._broadcast_fn:
            self._broadcast_fn(data=data, event_type=event_type)

    @staticmethod
    def _get_resolved_webui_output_dir() -> Optional[str]:
        try:
            from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

            resolved = WebUIStateStore().get_defaults_bundle().get("resolved", {})
            return resolved.get("output_dir")
        except Exception:
            return None

    @classmethod
    def _resolve_scan_output_dir(cls, output_dir: str) -> str:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        return ConfigsService._resolve_under_base(
            cls._get_resolved_webui_output_dir(),
            output_dir,
        )

    @staticmethod
    def _convert_pixel_area(dataset_config: Dict[str, Any], scan_args: _ScanArgsNamespace) -> Dict[str, Any]:
        """Convert pixel_area resolution_type to area with megapixel values.

        Mirrors DataBackendFactory._handle_resolution_conversion() which normally
        runs during training init but is bypassed by the scan service.
        """
        resolution_type = dataset_config.get("resolution_type", scan_args.resolution_type)
        if resolution_type != "pixel_area":
            return dataset_config

        dataset_config = dataset_config.copy()
        pixel_edge = dataset_config.get("resolution", scan_args.resolution)
        if pixel_edge is None:
            return dataset_config

        pixel_edge = int(pixel_edge)
        dataset_config["resolution_type"] = "area"
        dataset_config["resolution"] = (pixel_edge * pixel_edge) / (1000**2)

        for key in ("maximum_image_size", "target_downsample_size", "minimum_image_size"):
            val = dataset_config.get(key)
            if val is not None and val > 0:
                dataset_config[key] = (val * val) / 1_000_000

        return dataset_config

    def _resolve_vae_cache_dir(self, dataset_config: Dict[str, Any], global_config: Dict[str, Any]) -> Optional[str]:
        """Resolve the cache_dir_vae path, expanding template variables."""
        vae_dir = dataset_config.get("cache_dir_vae")
        if not vae_dir:
            return None

        from simpletuner.helpers.configuration.template_vars import resolve_string_placeholders

        output_dir = global_config.get("output_dir", "")
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
        output_dir = self._resolve_scan_output_dir(output_dir)

        resolved = resolve_string_placeholders(
            vae_dir,
            variables={
                "output_dir": output_dir,
                "model_family": global_config.get("model_family", ""),
                "id": dataset_config.get("id", ""),
            },
        )
        return os.path.abspath(resolved) if resolved else None

    def clear_vae_cache(self, dataset_config: Dict[str, Any], global_config: Dict[str, Any]) -> Optional[str]:
        """Delete the VAE cache directory for a dataset. Returns the path cleared, or None."""
        vae_dir = self._resolve_vae_cache_dir(dataset_config, global_config)
        if not vae_dir or not os.path.isdir(vae_dir):
            return None

        shutil.rmtree(vae_dir)
        logger.info("Cleared VAE cache directory: %s", vae_dir)
        return vae_dir

    def clear_conditioning_cache(self, dataset_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
        """Delete auto-generated conditioning data directories for a dataset.

        Only removes directories that would be auto-generated by the duplicator
        (i.e. derived from the ``conditioning`` generators block).  Manually
        declared conditioning datasets referenced via ``conditioning_data`` are
        never touched.

        Returns the list of paths that were deleted.
        """
        generators = dataset_config.get("conditioning") or []
        if isinstance(generators, dict):
            generators = [generators]
        if not generators:
            return []

        source_id = dataset_config.get("id", "")

        # Derive cache_root from cache_dir_vae (same heuristic as the viewer)
        cache_root = None
        cache_dir_vae = dataset_config.get("cache_dir_vae", "")
        if cache_dir_vae:
            from simpletuner.helpers.configuration.template_vars import resolve_string_placeholders

            output_dir = global_config.get("output_dir", "")
            if not output_dir:
                output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
            output_dir = self._resolve_scan_output_dir(output_dir)

            resolved_vae = resolve_string_placeholders(
                cache_dir_vae,
                variables={
                    "output_dir": output_dir,
                    "model_family": global_config.get("model_family", ""),
                    "id": dataset_config.get("id", ""),
                },
            )
            if resolved_vae:
                resolved_vae = os.path.abspath(resolved_vae)
                parts = Path(resolved_vae).parts
                try:
                    vae_idx = parts.index("vae")
                    cache_root = str(Path(*parts[:vae_idx]))
                except (ValueError, TypeError):
                    pass

        # Also try global cache_dir directly
        if cache_root is None:
            raw_cache_dir = global_config.get("cache_dir", "")
            if raw_cache_dir:
                from simpletuner.helpers.configuration.template_vars import resolve_string_placeholders

                output_dir = global_config.get("output_dir", "")
                if not output_dir:
                    output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
                output_dir = self._resolve_scan_output_dir(output_dir)

                cache_root = resolve_string_placeholders(
                    raw_cache_dir,
                    variables={
                        "output_dir": output_dir,
                        "model_family": global_config.get("model_family", ""),
                    },
                )
                if cache_root:
                    cache_root = os.path.abspath(cache_root)

        if not cache_root:
            logger.warning("Cannot determine cache root to clear conditioning data")
            return []

        cleared = []
        for gen in generators:
            gen_type = gen.get("type", "")
            if not gen_type:
                continue
            gen_id = gen.get("id") or f"{source_id}_conditioning_{gen_type}"
            cond_dir = os.path.join(cache_root, "conditioning_data", gen_id)
            if os.path.isdir(cond_dir):
                shutil.rmtree(cond_dir)
                logger.info("Cleared auto-generated conditioning directory: %s", cond_dir)
                cleared.append(cond_dir)

        return cleared

    def scan_dataset(
        self,
        dataset_id: str,
        dataset_config: Dict[str, Any],
        global_config: Optional[Dict[str, Any]] = None,
        force_rescan: bool = False,
        clear_vae_cache: bool = False,
        clear_conditioning_cache: bool = False,
    ) -> str:
        """Start a background scan for a single dataset. Returns job_id."""
        # Prevent scanning while a cache job is active
        try:
            from .cache_job_service import get_cache_service

            if get_cache_service().get_active_status():
                raise RuntimeError("A cache job is in progress. Wait for it to finish first.")
        except ImportError:
            pass

        if clear_vae_cache:
            self.clear_vae_cache(dataset_config, global_config or {})
        if clear_conditioning_cache:
            self.clear_conditioning_cache(dataset_config, global_config or {})

        with self._lock:
            if self._active_job and self._active_job.status == ScanStatus.RUNNING:
                raise RuntimeError("A scan is already in progress. Cancel it first or wait.")

            job_id = str(uuid.uuid4())[:8]
            job = ScanJob(job_id=job_id, dataset_id=dataset_id)
            self._active_job = job

        self._scan_thread = threading.Thread(
            target=self._run_scan,
            args=(job, dataset_config, global_config or {}, force_rescan),
            daemon=True,
        )
        self._scan_thread.start()
        return job_id

    def scan_all(
        self,
        datasets: List[Dict[str, Any]],
        global_config: Optional[Dict[str, Any]] = None,
        force_rescan: bool = False,
    ) -> str:
        """Queue all datasets for sequential scanning. Returns queue_id."""
        try:
            from .cache_job_service import get_cache_service

            if get_cache_service().get_active_status():
                raise RuntimeError("A cache job is in progress. Wait for it to finish first.")
        except ImportError:
            pass

        with self._lock:
            if self._active_job and self._active_job.status == ScanStatus.RUNNING:
                raise RuntimeError("A scan is already in progress.")

            queue_id = str(uuid.uuid4())[:8]
            dataset_ids = [ds.get("id", "") for ds in datasets]
            queue = ScanQueue(queue_id=queue_id, dataset_ids=dataset_ids)

            for ds in datasets:
                ds_id = ds.get("id", "")
                job_id = str(uuid.uuid4())[:8]
                queue.jobs[ds_id] = ScanJob(job_id=job_id, dataset_id=ds_id, status=ScanStatus.PENDING)

            self._active_queue = queue

        self._scan_thread = threading.Thread(
            target=self._run_queue,
            args=(queue, datasets, global_config or {}, force_rescan),
            daemon=True,
        )
        self._scan_thread.start()
        return queue_id

    def get_scan_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._active_job
        if job and job.job_id == job_id:
            return self._job_to_dict(job)

        queue = self._active_queue
        if queue:
            for j in queue.jobs.values():
                if j.job_id == job_id:
                    return self._job_to_dict(j)
        return None

    def get_queue_status(self, queue_id: str) -> Optional[Dict[str, Any]]:
        queue = self._active_queue
        if not queue or queue.queue_id != queue_id:
            return None

        completed = sum(1 for j in queue.jobs.values() if j.status == ScanStatus.COMPLETED)
        return {
            "queue_id": queue.queue_id,
            "total": len(queue.dataset_ids),
            "completed": completed,
            "current_index": queue.current_index,
            "cancelled": queue.cancelled,
            "datasets": {ds_id: self._job_to_dict(job) for ds_id, job in queue.jobs.items()},
        }

    def get_active_status(self) -> Optional[Dict[str, Any]]:
        """Return status of whatever scan is currently active, if any."""
        job = self._active_job
        if job and job.status == ScanStatus.RUNNING:
            result = self._job_to_dict(job)
            queue = self._active_queue
            if queue and not queue.cancelled:
                result["queue_id"] = queue.queue_id
                result["queue_total"] = len(queue.dataset_ids)
                result["queue_completed"] = sum(1 for j in queue.jobs.values() if j.status == ScanStatus.COMPLETED)
            return result
        return None

    def cancel_scan(self) -> bool:
        """Cancel the active scan and any remaining queue."""
        with self._lock:
            if self._active_queue:
                self._active_queue.cancelled = True
            if self._active_job and self._active_job.status == ScanStatus.RUNNING:
                self._active_job.status = ScanStatus.CANCELLED
                self._active_job.finished_at = time.time()
                return True
        return False

    @staticmethod
    def _job_to_dict(job: ScanJob) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "dataset_id": job.dataset_id,
            "status": job.status.value,
            "current": job.current,
            "total": job.total,
            "error": job.error,
        }

    def _run_scan(
        self,
        job: ScanJob,
        dataset_config: Dict[str, Any],
        global_config: Dict[str, Any],
        force_rescan: bool,
    ):
        job.status = ScanStatus.RUNNING
        job.started_at = time.time()
        self._broadcast(
            "dataset_scan",
            {"job_id": job.job_id, "dataset_id": job.dataset_id, "status": "running"},
        )

        try:
            self._execute_scan(job, dataset_config, global_config, force_rescan)
            if job.status == ScanStatus.CANCELLED:
                return
            job.status = ScanStatus.COMPLETED
            job.finished_at = time.time()
            self._broadcast(
                "dataset_scan",
                {
                    "job_id": job.job_id,
                    "dataset_id": job.dataset_id,
                    "status": "completed",
                    "total": job.total,
                },
            )
        except Exception as e:
            if job.status == ScanStatus.CANCELLED:
                return
            logger.exception("Scan failed for %s", job.dataset_id)
            job.status = ScanStatus.FAILED
            job.error = str(e)
            job.finished_at = time.time()
            self._broadcast(
                "dataset_scan",
                {
                    "job_id": job.job_id,
                    "dataset_id": job.dataset_id,
                    "status": "failed",
                    "error": str(e),
                },
            )

    def _run_queue(
        self,
        queue: ScanQueue,
        datasets: List[Dict[str, Any]],
        global_config: Dict[str, Any],
        force_rescan: bool,
    ):
        for i, ds_config in enumerate(datasets):
            if queue.cancelled:
                for ds_id, j in queue.jobs.items():
                    if j.status == ScanStatus.PENDING:
                        j.status = ScanStatus.CANCELLED
                break

            ds_id = ds_config.get("id", "")
            queue.current_index = i
            job = queue.jobs.get(ds_id)
            if not job:
                continue

            with self._lock:
                self._active_job = job

            self._run_scan(job, ds_config, global_config, force_rescan)

            self._broadcast(
                "dataset_scan_queue",
                {
                    "queue_id": queue.queue_id,
                    "completed": i + 1,
                    "total": len(datasets),
                    "current_dataset_id": ds_id,
                },
            )

    def _execute_scan(
        self,
        job: ScanJob,
        dataset_config: Dict[str, Any],
        global_config: Dict[str, Any],
        force_rescan: bool,
    ):
        """Execute the actual metadata scan using existing infrastructure."""
        from simpletuner.helpers.data_backend.builders import create_backend_builder
        from simpletuner.helpers.data_backend.config import create_backend_config
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Build args namespace from global config and dataset config
        args_overrides = {}
        for key in (
            "resolution",
            "resolution_type",
            "minimum_image_size",
            "maximum_image_size",
            "target_downsample_size",
            "caption_strategy",
            "model_type",
            "model_family",
        ):
            if key in dataset_config:
                args_overrides[key] = dataset_config[key]
            elif key in global_config:
                args_overrides[key] = global_config[key]

        scan_args = _ScanArgsNamespace(args_overrides)
        scan_args.output_dir = self._resolve_scan_output_dir(scan_args.output_dir)
        accelerator = _ScanAcceleratorStub()

        # Apply pixel_area -> area conversion (normally done by DataBackendFactory)
        dataset_config = self._convert_pixel_area(dataset_config, scan_args)

        # Temporarily set StateTracker args for the scan
        original_args = StateTracker.get_args()
        StateTracker.set_args(scan_args)

        config_id = None
        try:
            # Create config object
            args_dict = {k: getattr(scan_args, k, None) for k in vars(scan_args)}
            config = create_backend_config(dataset_config, args_dict)
            config_id = config.id

            # Skip validation for scan-only operation - some fields may not be set
            # config.validate(args_dict)

            # Register dataset config with StateTracker so metadata backends
            # can read resolution_type, crop settings, etc.
            StateTracker.set_data_backend_config(config.id, config.to_dict()["config"])

            # Create data backend
            builder = create_backend_builder(config.backend_type, accelerator, scan_args)
            data_backend = builder.build(config)

            # Create metadata backend
            metadata_backend = builder.create_metadata_backend(config, data_backend, args_dict)

            # Define progress callback
            def on_progress(current: int, total: int):
                if job.status == ScanStatus.CANCELLED:
                    raise RuntimeError("Scan cancelled by user")
                job.current = current
                job.total = total
                # Throttle broadcasts to every 50 files or 2 seconds
                if current % 50 == 0 or current == total:
                    self._broadcast(
                        "dataset_scan",
                        {
                            "job_id": job.job_id,
                            "dataset_id": job.dataset_id,
                            "status": "running",
                            "current": current,
                            "total": total,
                        },
                    )

            # Run the scan
            metadata_backend.compute_aspect_ratio_bucket_indices(
                ignore_existing_cache=force_rescan,
                progress_callback=on_progress,
            )
        finally:
            # Restore original args and clean up registered config
            StateTracker.set_args(original_args)
            if config_id is not None:
                StateTracker.data_backends.pop(config_id, None)


# Module-level singleton
_scan_service: Optional[DatasetScanService] = None


def get_scan_service() -> DatasetScanService:
    global _scan_service
    if _scan_service is None:
        try:
            from .sse_manager import get_sse_manager

            sse = get_sse_manager()
            _scan_service = DatasetScanService(broadcast_fn=sse.broadcast_threadsafe)
        except Exception:
            _scan_service = DatasetScanService()
    return _scan_service
