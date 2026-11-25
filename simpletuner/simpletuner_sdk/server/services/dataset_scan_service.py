"""Service for triggering standalone dataset metadata scans from the WebUI."""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dataset_worker import dataset_worker

logger = logging.getLogger(__name__)


class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
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
    plan_scope: Optional[str] = None
    queue_id: Optional[str] = None
    queue_index: Optional[int] = None


@dataclass
class ScanQueue:
    queue_id: str
    dataset_ids: List[str]
    jobs: Dict[str, ScanJob] = field(default_factory=dict)
    current_index: int = 0
    cancelled: bool = False
    finished: bool = False
    plan_scope: Optional[str] = None


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


def _execute_scan_process(connection, cancelled, dataset_config, global_config, force_rescan, dataset_plan):
    def report(current, total):
        if cancelled.is_set():
            raise RuntimeError("Scan cancelled by user")
        if current % 50 == 0 or current == total:
            connection.send({"current": current, "total": total})

    with connection:
        if cancelled.is_set():
            return
        try:
            DatasetScanService()._execute_scan(
                dataset_config, global_config, force_rescan, report, dataset_plan=dataset_plan
            )
        except Exception as e:
            if not cancelled.is_set():
                logger.exception("Dataset metadata scan failed")
                connection.send({"error": str(e)})


class DatasetScanService:
    """Manages standalone dataset metadata scans with progress broadcasting."""

    _lock = threading.Lock()
    _active_job: Optional[ScanJob] = None
    _active_queue: Optional[ScanQueue] = None
    _scan_thread: Optional[threading.Thread] = None

    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self._broadcast_fn = broadcast_fn
        self._cancel_event = None

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

    @classmethod
    def _build_scan_args(cls, dataset_config: Dict[str, Any], global_config: Dict[str, Any]) -> _ScanArgsNamespace:
        configured = {key.removeprefix("--").replace("-", "_"): value for key, value in global_config.items()}
        scan_keys = (
            "resolution",
            "resolution_type",
            "minimum_image_size",
            "maximum_image_size",
            "target_downsample_size",
            "caption_strategy",
            "only_instance_prompt",
            "model_type",
            "model_family",
            "aspect_bucket_alignment",
            "aspect_bucket_rounding",
            "seed",
            "train_batch_size",
            "disable_bucket_pruning",
            "framerate",
            "cache_dir",
        )
        settings = {**configured, **dataset_config}
        args = _ScanArgsNamespace({key: settings[key] for key in scan_keys if key in settings})
        args.output_dir = cls._resolve_scan_output_dir(configured.get("output_dir") or args.output_dir)
        return args

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

    def _local_vae_clear_directory(self, dataset_config, global_config, dataset_plan):
        from .cache_job_service import resolve_image_artifact_entry

        store = resolve_image_artifact_entry(dataset_config, dataset_plan, "image_embeds") or dataset_config
        if store.get("type", "local") != "local":
            raise ValueError("Clearing latents during a rescan requires a Local artifact store.")
        cache_path = self._resolve_vae_cache_dir(dataset_config, global_config)
        if cache_path is None:
            return None
        directory = Path(cache_path).resolve()
        for item in dataset_plan or [dataset_config]:
            if item.get("type", "local") in {"local", "csv"}:
                root = item.get("csv_cache_dir") or item.get("instance_data_dir")
                for source in [root] if isinstance(root, str) else root or []:
                    if Path(source).resolve().is_relative_to(directory):
                        raise ValueError("The latent cache directory overlaps a dataset source directory.")
            if item.get("id") != dataset_config["id"]:
                other_cache = self._resolve_vae_cache_dir(item, global_config)
                if other_cache:
                    other_directory = Path(other_cache).resolve()
                    if other_directory.is_relative_to(directory) or directory.is_relative_to(other_directory):
                        raise ValueError("The latent cache directory overlaps another dataset's latent cache.")
        return directory

    def clear_vae_cache(self, dataset_config, global_config, *, dataset_plan=None) -> Optional[str]:
        """Clear generated latent files while keeping the store directory and other files."""
        directory = self._local_vae_clear_directory(dataset_config, global_config, dataset_plan)
        if directory is None or not directory.is_dir():
            return None

        for root, _, files in os.walk(directory):
            for name in files:
                if re.fullmatch(r"[0-9a-f]{64}\.pt", name):
                    (Path(root) / name).unlink(missing_ok=True)
        logger.info("Cleared VAE cache files from: %s", directory)
        return str(directory)

    @staticmethod
    def conditioning_cache_root(global_config, dataset_config):
        from .cache_job_service import CacheJobService

        args = CacheJobService._build_args(global_config)
        if template := CacheJobService._resolve_global_key(global_config, "cache_dir"):
            return Path(CacheJobService._resolve_cache_dir(template, global_config, dataset_config))
        return Path(args.output_dir) / "cache"

    def _conditioning_cache_directories(self, dataset_config, global_config, dataset_plan):
        from simpletuner.helpers.metadata.utils.duplicator import DatasetDuplicator

        generators = dataset_config.get("conditioning") or []
        if isinstance(generators, dict):
            generators = [generators]
        if not generators:
            return []

        cache_root = self.conditioning_cache_root(global_config, dataset_config).resolve()
        conditioning_root = cache_root / "conditioning_data"
        source_roots = [
            Path(config["instance_data_dir"]).resolve()
            for config in dataset_plan or [dataset_config]
            if config.get("type", "local") == "local" and config.get("instance_data_dir")
        ]
        paths = []
        for generator in generators:
            generator_type = generator.get("type")
            if not generator_type or generator.get("instance_data_dir"):
                continue
            dataset_id = DatasetDuplicator.generated_conditioning_id(dataset_config["id"], generator_type)
            path = Path(DatasetDuplicator.generated_conditioning_directory(cache_root, dataset_id))
            if path != path.resolve() or not path.is_relative_to(conditioning_root) or path == conditioning_root:
                raise ValueError("The generated conditioning directory is outside the conditioning cache.")
            if any(source.is_relative_to(path) for source in source_roots):
                raise ValueError("The generated conditioning cache overlaps a dataset source directory.")
            paths.append(path)
        return paths

    def clear_conditioning_cache(self, dataset_config, global_config, *, dataset_plan=None) -> List[str]:
        """Clear the cache-owned directories of configured conditioning generators."""
        cleared = []
        for path in self._conditioning_cache_directories(dataset_config, global_config, dataset_plan):
            if path.is_dir():
                shutil.rmtree(path)
                cleared.append(str(path))
                logger.info("Cleared auto-generated conditioning directory: %s", path)
        return cleared

    def scan_dataset(
        self,
        dataset_id: str,
        dataset_config: Dict[str, Any],
        global_config: Optional[Dict[str, Any]] = None,
        force_rescan: bool = False,
        clear_vae_cache: bool = False,
        clear_conditioning_cache: bool = False,
        *,
        dataset_plan=None,
        plan_scope=None,
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
            self._local_vae_clear_directory(dataset_config, global_config or {}, dataset_plan)
        if clear_conditioning_cache:
            self._conditioning_cache_directories(dataset_config, global_config or {}, dataset_plan)

        with self._lock:
            if self._scan_thread is not None and self._scan_thread.is_alive():
                raise RuntimeError("A scan is already in progress. Wait for it to finish.")

            job_id = str(uuid.uuid4())[:8]
            job = ScanJob(job_id=job_id, dataset_id=dataset_id, plan_scope=plan_scope)
            self._active_job = job
            self._active_queue = None

            self._scan_thread = threading.Thread(
                target=self._run_scan,
                args=(job, dataset_config, global_config or {}, force_rescan),
                kwargs={
                    "clear_vae_cache": clear_vae_cache,
                    "clear_conditioning_cache": clear_conditioning_cache,
                    "dataset_plan": dataset_plan,
                },
                daemon=True,
            )
            self._scan_thread.start()
        return job_id

    def scan_all(
        self,
        datasets: List[Dict[str, Any]],
        global_config: Optional[Dict[str, Any]] = None,
        force_rescan: bool = False,
        *,
        dataset_plan=None,
        plan_scope=None,
    ) -> str:
        """Queue all datasets for sequential scanning. Returns queue_id."""
        try:
            from .cache_job_service import get_cache_service

            if get_cache_service().get_active_status():
                raise RuntimeError("A cache job is in progress. Wait for it to finish first.")
        except ImportError:
            pass

        with self._lock:
            if self._scan_thread is not None and self._scan_thread.is_alive():
                raise RuntimeError("A scan is already in progress.")

            queue_id = str(uuid.uuid4())[:8]
            dataset_ids = [ds.get("id", "") for ds in datasets]
            queue = ScanQueue(queue_id=queue_id, dataset_ids=dataset_ids, plan_scope=plan_scope)

            for index, ds in enumerate(datasets):
                ds_id = ds.get("id", "")
                job_id = str(uuid.uuid4())[:8]
                queue.jobs[ds_id] = ScanJob(
                    job_id=job_id, dataset_id=ds_id, plan_scope=plan_scope, queue_id=queue_id, queue_index=index
                )

            self._active_queue = queue
            self._active_job = queue.jobs.get(dataset_ids[0]) if dataset_ids else None

            self._scan_thread = threading.Thread(
                target=self._run_queue,
                args=(queue, datasets, global_config or {}, force_rescan, dataset_plan or datasets),
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
            "plan_scope": queue.plan_scope,
            "active": not queue.finished,
            "total": len(queue.dataset_ids),
            "completed": completed,
            "current_index": queue.current_index,
            "cancelled": queue.cancelled,
            "datasets": {ds_id: self._job_to_dict(job) for ds_id, job in queue.jobs.items()},
        }

    def get_active_status(self) -> Optional[Dict[str, Any]]:
        """Return status of whatever scan is currently active, if any."""
        job = self._active_job
        if job and self._scan_thread is not None and self._scan_thread.is_alive():
            result = self._job_to_dict(job)
            queue = self._active_queue
            if queue and not queue.finished:
                result["queue_total"] = len(queue.dataset_ids)
                result["queue_completed"] = sum(1 for j in queue.jobs.values() if j.status == ScanStatus.COMPLETED)
            return result
        return None

    def cancel_scan(self, *, job_id=None, queue_id=None, plan_scope=None) -> bool:
        """Cancel the active scan and any remaining queue."""
        with self._lock:
            job, queue = self._active_job, self._active_queue
            if job is None or (job_id is not None and job.job_id != job_id):
                return False
            if plan_scope is not None and job.plan_scope != plan_scope:
                return False
            if queue_id is not None and (queue is None or queue.queue_id != queue_id):
                return False
            if queue is not None and not queue.finished:
                queue.cancelled = True
            elif job.status not in (ScanStatus.PENDING, ScanStatus.RUNNING):
                return False
            if job.status in (ScanStatus.PENDING, ScanStatus.RUNNING):
                job.status = ScanStatus.CANCELLING
                if self._cancel_event is not None:
                    self._cancel_event.set()
                self._broadcast("dataset_scan", self._job_to_dict(job))
            return True

    @staticmethod
    def _job_to_dict(job: ScanJob) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "plan_scope": job.plan_scope,
            "queue_id": job.queue_id,
            "queue_index": job.queue_index,
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
        *,
        clear_vae_cache: bool = False,
        clear_conditioning_cache: bool = False,
        dataset_plan=None,
    ):
        context = multiprocessing.get_context("spawn")
        with self._lock:
            if job.status == ScanStatus.CANCELLING or (self._active_queue and self._active_queue.cancelled):
                job.status = ScanStatus.CANCELLED
                job.finished_at = time.time()
                self._broadcast("dataset_scan", self._job_to_dict(job))
                return
            cancelled = self._cancel_event = context.Event()
            job.status = ScanStatus.RUNNING
            job.started_at = time.time()
            self._broadcast("dataset_scan", self._job_to_dict(job))

        try:
            with dataset_worker(
                context, _execute_scan_process, cancelled, dataset_config, global_config, force_rescan, dataset_plan
            ) as messages:
                for data in messages:
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    with self._lock:
                        if job.status == ScanStatus.CANCELLING:
                            continue
                        job.current, job.total = data["current"], data["total"]
                        self._broadcast("dataset_scan", self._job_to_dict(job))
            with self._lock:
                if job.status == ScanStatus.CANCELLING:
                    return
                if clear_vae_cache:
                    self.clear_vae_cache(dataset_config, global_config, dataset_plan=dataset_plan)
                if clear_conditioning_cache:
                    self.clear_conditioning_cache(dataset_config, global_config, dataset_plan=dataset_plan)
                job.status = ScanStatus.COMPLETED
        except Exception as e:
            with self._lock:
                if job.status == ScanStatus.CANCELLING:
                    return
                job.status = ScanStatus.FAILED
                job.error = str(e)
            logger.exception("Scan failed for %s", job.dataset_id)
        finally:
            with self._lock:
                self._cancel_event = None
                if job.status == ScanStatus.CANCELLING:
                    job.status = ScanStatus.CANCELLED
                job.finished_at = time.time()
                self._broadcast("dataset_scan", self._job_to_dict(job))

    def _run_queue(
        self,
        queue: ScanQueue,
        datasets: List[Dict[str, Any]],
        global_config: Dict[str, Any],
        force_rescan: bool,
        dataset_plan,
    ):
        for i, ds_config in enumerate(datasets):
            if queue.cancelled:
                for j in queue.jobs.values():
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

            self._run_scan(job, ds_config, global_config, force_rescan, dataset_plan=dataset_plan)

            self._broadcast(
                "dataset_scan_queue",
                {
                    "queue_id": queue.queue_id,
                    "plan_scope": queue.plan_scope,
                    "completed": i + 1,
                    "total": len(datasets),
                    "current_dataset_id": ds_id,
                },
            )
        with self._lock:
            queue.finished = True
            self._broadcast("dataset_scan_queue", self.get_queue_status(queue.queue_id))

    def _execute_scan(
        self,
        dataset_config: Dict[str, Any],
        global_config: Dict[str, Any],
        force_rescan: bool,
        on_progress,
        *,
        dataset_plan=None,
    ):
        """Execute the actual metadata scan using existing infrastructure."""
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.training.state_tracker import StateTracker

        scan_args = self._build_scan_args(dataset_config, global_config)
        accelerator = _ScanAcceleratorStub()

        StateTracker.set_args(scan_args)
        StateTracker.set_accelerator(accelerator)

        with DatasetRuntime.open_preparation(
            dataset_config, scan_args, accelerator, dataset_plan=dataset_plan or ()
        ) as dataset:
            dataset.prepare_view(force_rescan=force_rescan, progress_callback=on_progress)


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
