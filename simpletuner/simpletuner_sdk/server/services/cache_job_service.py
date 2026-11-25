"""Service for running cache operations (text embeds, VAE, conditioning) from the WebUI."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import uuid
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from .dataset_worker import dataset_worker

logger = logging.getLogger(__name__)

# Model families that do not use text embedding caches.
_NO_TEXT_EMBED_MODELS = frozenset({"heartmula", "omnigen"})


class CacheType(str, Enum):
    TEXT_EMBEDS = "text_embeds"
    VAE = "vae"
    CONDITIONING = "conditioning"


class CacheJobStatus(str, Enum):
    PENDING = "pending"
    LOADING_MODEL = "loading_model"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CacheJob:
    job_id: str
    dataset_id: str
    cache_type: str
    status: CacheJobStatus = CacheJobStatus.PENDING
    stage: str = ""
    current: int = 0
    total: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    plan_scope: Optional[str] = None


class _CacheAcceleratorStub:
    """Minimal accelerator stub with real device support for cache operations."""

    num_processes = 1
    process_index = 0
    local_process_index = 0
    is_main_process = True
    is_local_main_process = True
    data_parallel_rank = 0
    data_parallel_shard_rank = 0

    def __init__(self, device=None):
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

    def wait_for_everyone(self):
        pass

    @contextmanager
    def main_process_first(self):
        yield

    @contextmanager
    def split_between_processes(self, data, apply_padding=False):
        yield data

    def unwrap_model(self, model):
        return model

    def gather(self, tensor):
        return tensor


class _CacheArgsNamespace:
    """Args namespace built from the active training config.

    Accepts the full config dict and exposes values as attributes.
    Model and cache classes access a wide range of config attributes
    during ``__init__`` and setup methods, so unknown attributes fall
    back to ``None`` rather than raising ``AttributeError``.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        # Track which keys were explicitly set for __contains__
        self._explicit_keys: set = set()
        # Essential defaults that model/cache code expects
        defaults = {
            "aspect_bucket_worker_count": 4,
            "enable_multiprocessing": False,
            "delete_problematic_images": False,
            "delete_unwanted_images": False,
            "metadata_update_interval": 3600,
            "compress_disk_cache": False,
            "disable_bucket_pruning": True,
            "ignore_missing_files": False,
            "allow_dataset_oversubscription": False,
            "controlnet": False,
            "hash_filenames": True,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "data_backend_sampling": "uniform",
            "write_batch_size": 64,
            "read_batch_size": 25,
            "vae_batch_size": 4,
            "text_encoder_batch_size": 1,
            "vae_cache_ondemand": False,
            "i_know_what_i_am_doing": True,
            "resolution": 1024,
            "resolution_type": "pixel",
            "minimum_image_size": 0,
            "maximum_image_size": None,
            "target_downsample_size": None,
            "caption_strategy": "filename",
            "model_type": "full",
            "model_family": None,
            "output_dir": "output/cache",
            "framerate": 24,
        }
        for k, v in defaults.items():
            object.__setattr__(self, k, v)
            self._explicit_keys.add(k)
        # Override with config values (strip leading -- from CLI-style keys)
        for k, v in config_dict.items():
            clean_key = k.lstrip("-").replace("-", "_") if k.startswith("--") else k
            object.__setattr__(self, clean_key, v)
            self._explicit_keys.add(clean_key)

    def __getattr__(self, name):
        # Called only when normal lookup fails. Return None for public
        # attributes so model/cache code that probes optional config
        # values doesn't crash.  Private/dunder names still raise so
        # that Python internals (pickle, copy, etc.) behave correctly.
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def __contains__(self, key):
        return key in self._explicit_keys

    def get(self, key, default=None):
        val = getattr(self, key, None)
        return val if val is not None else default


def _execute_cache_process(connection, cancelled, job, config, global_config, dataset_plan):
    def send(data, event_type):
        connection.send(data)

    with connection:
        service = CacheJobService(broadcast_fn=send)
        service._cancel_event = cancelled
        service._run_cache_job(job, config, global_config, dataset_plan)


def resolve_image_artifact_entry(dataset, plan, role):
    store_id = dataset.get(role)
    if store_id is None:
        return None
    entry = next(
        (
            item
            for item in plan or ()
            if item.get("id") == store_id
            and item.get("dataset_type") == role
            and not (item.get("disabled") or item.get("disable"))
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Could not find {role} backend {store_id!r} in the standalone cache plan.")
    return entry


class CacheJobService:
    """Manages background cache operations with SSE progress broadcasting."""

    _lock = threading.Lock()
    _active_job: Optional[CacheJob] = None
    _thread: Optional[threading.Thread] = None

    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self._broadcast_fn = broadcast_fn
        self._cancel_event = None

    def _broadcast(self, event_type: str, data: Dict[str, Any]):
        if self._broadcast_fn:
            self._broadcast_fn(data=data, event_type=event_type)

    @classmethod
    def get_capabilities(
        cls,
        model_family: str,
        dataset_configs: List[Dict[str, Any]],
        global_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine available cache types without loading any models."""
        has_text_embeds = bool(model_family) and model_family not in _NO_TEXT_EMBED_MODELS
        has_vae = True

        conditioning_types: List[str] = []
        for ds in dataset_configs:
            if ds.get("conditioning_data"):
                if "image_embeds" not in conditioning_types:
                    conditioning_types.append("image_embeds")
            generators = ds.get("conditioning") or []
            if isinstance(generators, dict):
                generators = [generators]
            for gen in generators:
                gen_type = gen.get("type", "")
                if gen_type and gen_type not in conditioning_types:
                    conditioning_types.append(gen_type)

        if int(global_config.get("max_grounding_entities", 0) or 0) > 0:
            if "grounding" not in conditioning_types:
                conditioning_types.append("grounding")

        return {
            "text_embeds": has_text_embeds,
            "vae": has_vae,
            "conditioning_types": conditioning_types,
            "datasets": cls._dataset_cache_capabilities(dataset_configs, global_config, has_text_embeds),
        }

    @staticmethod
    def _dataset_cache_capabilities(plan, global_config, has_text_embeds):
        if not plan:
            return {}
        from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
        from simpletuner.helpers.data_backend.factory import _as_bool, init_backend_config, select_text_embed_backend

        args = _CacheArgsNamespace(global_config)
        entries = {}
        stores, default_id = {}, None
        store_error = None
        try:
            if has_text_embeds:
                stores, default_id = CacheJobService._text_embed_plan(plan, args)
        except ValueError as exc:
            store_error = str(exc)
        for dataset in plan:
            dataset_id = dataset.get("id")
            if not dataset_id or dataset.get("disabled") or dataset.get("disable"):
                continue
            try:
                role = ensure_dataset_type(dataset.get("dataset_type"), default=DatasetType.IMAGE)
            except ValueError:
                continue
            if role not in {
                DatasetType.IMAGE,
                DatasetType.VIDEO,
                DatasetType.AUDIO,
                DatasetType.EVAL,
                DatasetType.CONDITIONING,
                DatasetType.CAPTION,
            }:
                continue
            text = {"prepare": False}
            if has_text_embeds:
                try:
                    if store_error:
                        raise ValueError(store_error)
                    store = select_text_embed_backend(dataset, stores, default_id)
                    settings = init_backend_config(store, args, None)["config"]
                    mode, reason = "precomputed", None
                    if settings.get("text_cache_encode_only"):
                        mode, reason = "encode_only", "Encode-only training does not use stored text embeddings."
                    elif settings["text_cache_disable"]:
                        mode, reason = "read_existing", "Training can read existing text embeddings with writes disabled."
                    elif settings["text_cache_ondemand"]:
                        mode, reason = (
                            "on_demand",
                            "Training encodes and stores misses. You can also prepare embeddings now.",
                        )
                    text.update(
                        prepare=not settings["text_cache_disable"],
                        mode=mode,
                        reason=reason,
                        store_id=store["id"],
                        uses_s3_listing=store.get("type") == "aws" and settings.get("text_cache_use_listing", False),
                    )
                except ValueError as exc:
                    text["reason"] = str(exc)
            disabled = _as_bool(args.vae_cache_disable) or _as_bool(dataset.get("vae_cache_disable"))
            entries[dataset_id] = {
                "text_embeds": text,
                "vae": {
                    "prepare": not disabled and role is not DatasetType.CAPTION,
                    "reason": "VAE cache writes are disabled." if disabled else None,
                },
            }
        return entries

    @staticmethod
    def _check_training_running() -> bool:
        """Return True if a local training job is currently running."""
        try:
            import asyncio

            from .local_gpu_allocator import get_gpu_allocator

            allocator = get_gpu_allocator()
            job_repo = allocator._get_job_repo()

            async def _count():
                return await job_repo.count_running_local_jobs()

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    count = pool.submit(lambda: asyncio.run(_count())).result(timeout=5)
            except RuntimeError:
                count = asyncio.run(_count())
            return count > 0
        except Exception:
            logger.exception(
                "Failed to determine whether a training job is running; "
                "assuming training may be active and blocking cache job start."
            )
            return True

    def start_cache_job(
        self,
        dataset_id: str,
        cache_type: str,
        dataset_config: Dict[str, Any],
        global_config: Dict[str, Any],
        dataset_plan=None,
        *,
        plan_scope=None,
    ) -> str:
        """Start a background cache job for a single dataset. Returns job_id."""
        from .dataset_scan_service import get_scan_service

        if get_scan_service().get_active_status():
            raise RuntimeError("A metadata scan is in progress. Wait for it to finish first.")

        if self._check_training_running():
            raise RuntimeError("A training job is running. Wait for it to finish first.")

        with self._lock:
            if self._active_job and self._active_job.status in (
                CacheJobStatus.PENDING,
                CacheJobStatus.LOADING_MODEL,
                CacheJobStatus.RUNNING,
                CacheJobStatus.CANCELLING,
            ):
                raise RuntimeError("A cache job is already in progress.")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("The previous cache worker is still shutting down.")

            job_id = str(uuid.uuid4())[:8]
            job = CacheJob(job_id=job_id, dataset_id=dataset_id, cache_type=cache_type, plan_scope=plan_scope)
            self._active_job = job

            context = multiprocessing.get_context("spawn")
            self._cancel_event = context.Event()
            self._thread = threading.Thread(
                target=self._monitor_worker,
                args=(context, self._cancel_event, job, dataset_config, global_config, dataset_plan),
                daemon=True,
            )
            self._thread.start()
        return job_id

    def _monitor_worker(self, context, cancelled, job, dataset_config, global_config, dataset_plan):
        try:
            with dataset_worker(
                context, _execute_cache_process, cancelled, job, dataset_config, global_config, dataset_plan
            ) as messages:
                for data in messages:
                    with self._lock:
                        if job.status == CacheJobStatus.CANCELLING:
                            continue
                        job.status = CacheJobStatus(data["status"])
                        job.started_at = data.get("started_at")
                        job.finished_at = data.get("finished_at")
                        job.stage, job.current, job.total, job.error = (
                            data["stage"],
                            data["current"],
                            data["total"],
                            data["error"],
                        )
                        self._broadcast("dataset_cache", self._job_to_dict(job))
                if job.status not in (
                    CacheJobStatus.COMPLETED,
                    CacheJobStatus.FAILED,
                    CacheJobStatus.CANCELLED,
                    CacheJobStatus.CANCELLING,
                ):
                    raise RuntimeError("Cache worker exited before reporting completion.")
        except Exception as exc:
            logger.exception("Cache worker failed for %s", job.dataset_id)
            with self._lock:
                if job.status != CacheJobStatus.CANCELLING:
                    job.status, job.error = CacheJobStatus.FAILED, str(exc)
                    self._broadcast("dataset_cache", self._job_to_dict(job))
        finally:
            with self._lock:
                job.finished_at = time.time()
                if job.status == CacheJobStatus.CANCELLING:
                    job.status = CacheJobStatus.CANCELLED
                    job.stage = "Cancelled"
                    self._broadcast("dataset_cache", self._job_to_dict(job))

    def get_active_status(self, *, plan_scope=None) -> Optional[Dict[str, Any]]:
        job = self._active_job
        if job and plan_scope is not None and job.plan_scope != plan_scope:
            return None
        if job and (
            job.status
            in (CacheJobStatus.PENDING, CacheJobStatus.LOADING_MODEL, CacheJobStatus.RUNNING, CacheJobStatus.CANCELLING)
            or (self._thread is not None and self._thread.is_alive())
        ):
            return self._job_to_dict(job)
        return None

    def get_job_status(self, job_id: str, *, plan_scope=None) -> Optional[Dict[str, Any]]:
        job = self._active_job
        if job and job.job_id == job_id and (plan_scope is None or job.plan_scope == plan_scope):
            return self._job_to_dict(job)
        return None

    def cancel(self, *, job_id=None, plan_scope=None) -> bool:
        with self._lock:
            job = self._active_job
            if job is None or (job_id is not None and job.job_id != job_id):
                return False
            if plan_scope is not None and job.plan_scope != plan_scope:
                return False
            if job.status in (
                CacheJobStatus.PENDING,
                CacheJobStatus.LOADING_MODEL,
                CacheJobStatus.RUNNING,
            ):
                if self._cancel_event is not None:
                    self._cancel_event.set()
                if self._thread is not None and self._thread.is_alive():
                    job.status = CacheJobStatus.CANCELLING
                    job.stage = "Cancelling"
                else:
                    job.status = CacheJobStatus.CANCELLED
                    job.finished_at = time.time()
                self._broadcast("dataset_cache", self._job_to_dict(job))
                return True
        return False

    @staticmethod
    def _job_to_dict(job: CacheJob) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "dataset_id": job.dataset_id,
            "cache_type": job.cache_type,
            "status": job.status.value,
            "stage": job.stage,
            "current": job.current,
            "total": job.total,
            "error": job.error,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }

    # ------------------------------------------------------------------
    # Background thread entry points
    # ------------------------------------------------------------------

    def _run_cache_job(
        self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any], dataset_plan=None
    ):
        job.status = CacheJobStatus.LOADING_MODEL
        job.started_at = time.time()
        job.stage = "Loading model components"
        self._broadcast("dataset_cache", self._job_to_dict(job))

        try:
            self._check_cancelled(job)
            if job.cache_type == CacheType.TEXT_EMBEDS.value:
                self._execute_text_embed_cache(job, dataset_config, global_config, dataset_plan)
            elif job.cache_type == CacheType.VAE.value:
                self._execute_vae_cache(job, dataset_config, global_config, dataset_plan)
            elif job.cache_type == CacheType.CONDITIONING.value:
                self._execute_conditioning_cache(job, dataset_config, global_config, dataset_plan)
            else:
                raise ValueError(f"Unknown cache type: {job.cache_type}")

            if job.status == CacheJobStatus.CANCELLED:
                self._broadcast("dataset_cache", self._job_to_dict(job))
                return

            job.status = CacheJobStatus.COMPLETED
            job.finished_at = time.time()
            job.stage = "Complete"
            self._broadcast("dataset_cache", self._job_to_dict(job))

        except Exception as e:
            if job.status == CacheJobStatus.CANCELLED:
                self._broadcast("dataset_cache", self._job_to_dict(job))
                return
            logger.exception("Cache job failed for %s (%s)", job.dataset_id, job.cache_type)
            job.status = CacheJobStatus.FAILED
            job.error = str(e)
            job.finished_at = time.time()
            self._broadcast("dataset_cache", self._job_to_dict(job))

    def _check_cancelled(self, job: CacheJob):
        if self._cancel_event is not None and self._cancel_event.is_set():
            job.status = CacheJobStatus.CANCELLED
        if job.status == CacheJobStatus.CANCELLED:
            raise RuntimeError("Cache job cancelled by user")

    # ------------------------------------------------------------------
    # Shared setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_args(global_config: Dict[str, Any]) -> _CacheArgsNamespace:
        args = _CacheArgsNamespace(global_config)
        from .dataset_scan_service import DatasetScanService

        args.output_dir = DatasetScanService._resolve_scan_output_dir(args.output_dir or "output/cache")
        return args

    @staticmethod
    def _resolve_global_key(global_config: Dict[str, Any], key: str, default: str = "") -> str:
        """Look up a key in global_config, trying both plain and ``--``-prefixed forms."""
        val = global_config.get(key) or global_config.get(f"--{key}")
        return val if val else default

    @staticmethod
    def _resolve_cache_dir(
        template: str,
        global_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        *,
        absolute=True,
    ) -> Optional[str]:
        if not template:
            return None
        from simpletuner.helpers.configuration.template_vars import resolve_string_placeholders

        output_dir = CacheJobService._resolve_global_key(global_config, "output_dir")
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), ".simpletuner_output")
        from .dataset_scan_service import DatasetScanService

        output_dir = DatasetScanService._resolve_scan_output_dir(output_dir)

        resolved = resolve_string_placeholders(
            template,
            variables={
                "output_dir": output_dir,
                "model_family": global_config.get("model_family", ""),
                "id": dataset_config.get("id", ""),
            },
        )
        return (os.path.abspath(resolved) if absolute else resolved) if resolved else None

    def _create_dataset_runtime(self, dataset_config, args, accelerator, *, preparation_only=False):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        config = DatasetRuntime.preparation_config(dataset_config, args)
        return DatasetRuntime.from_config(config, args, accelerator, preparation_only=preparation_only)

    @staticmethod
    def _iter_prompt_records(model, runtime, *, batch_size=1, args=None, conditioning_datasets=None):
        from simpletuner.helpers.data_backend.dataset_runtime import TextRequestBuilder

        yield from runtime.iter_preparation_text_records(
            model,
            batch_size=batch_size,
            builder=TextRequestBuilder(model, runtime, args=args, conditioning_datasets=conditioning_datasets),
            on_missing_caption=lambda ref: logger.warning("Could not find caption for sample %s", ref),
        )

    def _load_model(self, global_config: Dict[str, Any], args, accelerator):
        from simpletuner.helpers.models.registry import ModelRegistry

        model_family = args.model_family
        if not model_family:
            raise ValueError("No model_family in active config")

        families = ModelRegistry.model_families()
        model_cls = families.get(model_family)
        if model_cls is None:
            raise ValueError(f"Unknown model family: {model_family}")

        return model_cls(args, accelerator)

    @staticmethod
    def _cleanup_gpu(*objects):
        """Delete provided objects and release GPU memory."""
        import gc

        for obj in objects:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cache execution per type
    # ------------------------------------------------------------------

    def _prepare_image_artifacts(self, job, runtime, binding, cache):
        setattr(runtime, binding, cache)
        job.status = CacheJobStatus.RUNNING
        job.stage = "Preparing latents" if binding == "vae_cache" else "Preparing conditioning embeddings"
        last_update = 0.0

        def progress(current, total):
            nonlocal last_update
            self._check_cancelled(job)
            job.current, job.total = current, total
            now = time.monotonic()
            if now - last_update >= 1 or current == total:
                self._broadcast("dataset_cache", self._job_to_dict(job))
                last_update = now

        processed = cache.prepare_dataset(runtime, progress_callback=progress)
        if processed == 0:
            raise ValueError("No accepted samples found. Run a metadata scan before preparing this cache.")

    @staticmethod
    def _image_artifact_store(dataset, plan, args, accelerator, source, role):
        from simpletuner.helpers.data_backend.builders import create_backend_builder
        from simpletuner.helpers.data_backend.config import create_backend_config

        entry = resolve_image_artifact_entry(dataset, plan, role)
        if entry is None:
            return source, {}
        config = create_backend_config(entry, vars(args))
        config.validate(vars(args))
        if config.backend_type not in {"local", "aws"}:
            raise ValueError(f"Standalone {role} preparation requires a persistent Local or AWS store.")
        store = create_backend_builder(config.backend_type, accelerator, args).build(config)
        return store, entry

    def _execute_vae_cache(
        self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any], dataset_plan=None
    ):
        from simpletuner.helpers.data_backend.factory import default_vae_cache_dir, init_backend_config
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        scope = ExitStack()
        model = None
        vae_cache = None

        try:
            self._check_cancelled(job)
            settings = init_backend_config(dataset_config, args, accelerator)["config"]
            if settings["vae_cache_disable"]:
                raise ValueError("The selected dataset has VAE cache writes disabled.")

            job.stage = "Loading VAE"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)
            model.load_vae()
            vae = getattr(model, "vae", None) or StateTracker.get_vae()

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            runtime = self._create_dataset_runtime(dataset_config, args, accelerator)
            scope.callback(runtime.close)

            dataset_type = runtime.config["dataset_type"]
            dataset_vae = model.get_vae_for_dataset_type(dataset_type)
            if dataset_vae is not None:
                vae = dataset_vae
            audio = runtime.config.get("audio") or {}
            source_id = runtime.config.get("source_dataset_id") or audio.get("source_dataset_id")
            needs_source = dataset_type == "conditioning" or audio.get("source_from_video")
            if source_id and needs_source:
                source_config = next(
                    (
                        item
                        for item in dataset_plan or ()
                        if item.get("id") == source_id and not (item.get("disabled") or item.get("disable"))
                    ),
                    None,
                )
                if source_config is None:
                    raise ValueError(f"Dataset {runtime.id} requires an enabled geometry source {source_id}.")
                runtime.source_dataset = self._create_dataset_runtime(
                    source_config, args, accelerator, preparation_only=True
                )
                scope.callback(runtime.source_dataset.close)

            cache_store, _ = self._image_artifact_store(
                dataset_config, dataset_plan, args, accelerator, runtime.data_backend, "image_embeds"
            )
            cache_dir_vae = self._resolve_cache_dir(
                dataset_config.get("cache_dir_vae") or default_vae_cache_dir(args, runtime.id, dataset_type),
                global_config,
                dataset_config,
                absolute=cache_store.type != "aws",
            )
            from simpletuner.helpers.caching.vae import VAECache

            vae_cache = VAECache(
                id=runtime.id,
                dataset_type=dataset_type,
                model=model,
                vae=vae,
                accelerator=accelerator,
                metadata_backend=runtime.metadata_backend,
                image_data_backend=runtime.data_backend,
                cache_data_backend=cache_store,
                instance_data_dir=runtime.config["instance_data_dir"],
                cache_dir=cache_dir_vae,
                vae_batch_size=dataset_config.get("vae_batch_size", args.vae_batch_size),
                write_batch_size=dataset_config.get("write_batch_size", args.write_batch_size),
                read_batch_size=dataset_config.get("read_batch_size", args.read_batch_size),
                process_queue_size=dataset_config.get("image_processing_batch_size", args.image_processing_batch_size or 16),
                hash_filenames=True,
                cache_inventory_dir=args.output_dir,
                dataset_config=runtime.config,
                dataset=runtime,
            )

            self._check_cancelled(job)

            self._prepare_image_artifacts(job, runtime, "vae_cache", vae_cache)

        finally:
            StateTracker.set_args(original_args)
            StateTracker.set_accelerator(original_accelerator)
            scope.close()
            del vae_cache, model
            self._cleanup_gpu()

    @staticmethod
    def _text_embed_plan(plan, args):
        from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
        from simpletuner.helpers.data_backend.factory import ensure_text_embed_backends_present

        entries = ensure_text_embed_backends_present(args, [dict(entry) for entry in plan])
        stores = {
            entry["id"]: entry
            for entry in entries
            if ensure_dataset_type(entry.get("dataset_type"), default=DatasetType.IMAGE) is DatasetType.TEXT_EMBEDS
            and not (entry.get("disable") or entry.get("disabled"))
        }
        return stores, next(entry["id"] for entry in stores.values() if entry.get("default"))

    def _text_embed_store(self, dataset, plan, global_config, args, accelerator):
        from simpletuner.helpers.data_backend.factory import (
            build_text_embed_store,
            init_backend_config,
            select_text_embed_backend,
        )

        stores, default_id = self._text_embed_plan(plan or [dataset], args)
        config = select_text_embed_backend(dataset, stores, default_id)
        settings = init_backend_config(config, args, accelerator)["config"]
        if settings["text_cache_disable"]:
            raise ValueError("The selected text embedding store has writes disabled.")
        store, directory = build_text_embed_store(config, args, accelerator)
        if store.type == "local":
            directory = self._resolve_cache_dir(directory, global_config, config)
        if directory is None:
            raise ValueError("No text embed cache directory configured for this dataset")
        return config, store, directory

    def _execute_text_embed_cache(
        self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any], dataset_plan=None
    ):
        from simpletuner.helpers.data_backend.factory import build_text_preparation_listing
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        scope = ExitStack()
        model = None
        text_cache = None

        try:
            self._check_cancelled(job)
            store_config, store, cache_dir_text = self._text_embed_store(
                dataset_config, dataset_plan, global_config, args, accelerator
            )

            job.stage = "Loading text encoders"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)
            model.load_text_encoder()
            model.load_text_tokenizer()

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            source_runtime = self._create_dataset_runtime(dataset_config, args, accelerator, preparation_only=True)
            scope.callback(source_runtime.close)

            from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache

            text_cache = TextEmbeddingCache(
                id=store_config["id"],
                data_backend=store,
                text_encoders=model.text_encoders,
                tokenizers=model.tokenizers,
                accelerator=accelerator,
                cache_dir=cache_dir_text,
                model_type=args.model_family,
                write_batch_size=int(store_config.get("write_batch_size", args.write_batch_size) or 64),
                text_encoder_batch_size=int(store_config.get("text_encoder_batch_size", args.text_encoder_batch_size) or 1),
                model=model,
                caption_filter_list=store_config.get("caption_filter_list"),
                preparation_listing=build_text_preparation_listing(store_config, store, cache_dir_text, args, accelerator),
            )

            scope.enter_context(closing(text_cache))
            self._check_cancelled(job)

            job.stage = "Collecting captions"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            batch_size = max(text_cache.read_batch_size, text_cache.text_encoder_batch_size)
            source_runtime.text_embed_cache = text_cache
            conditioning = None
            ids = dataset_config.get("conditioning_data")
            from simpletuner.helpers.data_backend.dataset_runtime import TextRequestBuilder

            if TextRequestBuilder.needs_conditioning_datasets(model, ids, args):
                conditioning = []
                available = {
                    item["id"]: item for item in dataset_plan or () if not (item.get("disabled") or item.get("disable"))
                }
                for ref_id in [ids] if isinstance(ids, str) else ids:
                    if ref_id not in available:
                        raise ValueError(f"Missing conditioning dataset {ref_id!r} in the standalone cache plan.")
                    ref_config = available[ref_id]
                    runtime = self._create_dataset_runtime(ref_config, args, accelerator, preparation_only=True)
                    scope.callback(runtime.close)
                    conditioning.append(runtime)
                source_runtime.conditioning_datasets = tuple(conditioning)
            prompt_records = self._iter_prompt_records(
                model, source_runtime, batch_size=batch_size, args=args, conditioning_datasets=conditioning
            )

            # Move text encoders to device
            from simpletuner.helpers.data_backend.factory import move_text_encoders

            move_text_encoders(args, model.text_encoders, accelerator.device, force_move=True)

            self._check_cancelled(job)

            job.status = CacheJobStatus.RUNNING
            job.stage = "Computing text embeddings"
            job.total = 0
            self._broadcast("dataset_cache", self._job_to_dict(job))

            _last_broadcast = [0.0]

            def _text_progress(current):
                self._check_cancelled(job)
                job.current = current
                now = time.time()
                if now - _last_broadcast[0] >= 1.0:
                    _last_broadcast[0] = now
                    self._broadcast("dataset_cache", self._job_to_dict(job))

            text_cache.encode_dropout_caption()
            job.total = text_cache.precompute_embeddings(
                prompt_records,
                batch_size=batch_size,
                progress_callback=_text_progress,
            )
            if not job.total:
                raise ValueError("No captions found for this dataset")
            self._broadcast("dataset_cache", self._job_to_dict(job))

        finally:
            try:
                scope.close()
            finally:
                StateTracker.set_args(original_args)
                StateTracker.set_accelerator(original_accelerator)
                del text_cache, model
                self._cleanup_gpu()

    def _execute_conditioning_cache(
        self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any], dataset_plan=None
    ):
        from simpletuner.helpers.data_backend.config.relationships import resolve_dataset_relationships
        from simpletuner.helpers.data_backend.factory import configure_model_data_signals
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        model = None

        try:
            self._check_cancelled(job)

            job.stage = "Loading model for conditioning embeds"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)
            plan = dataset_plan or [dataset_config]
            configure_model_data_signals(model, plan)

            if not model.requires_conditioning_image_embeds():
                raise ValueError(
                    f"Model family '{global_config.get('model_family')}' "
                    "does not require conditioning image embeds with this configuration"
                )

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            datasets = [dataset_config]
            if dataset_config.get("conditioning_data") and model.conditioning_image_embeds_use_reference_dataset():
                conditioning, _ = resolve_dataset_relationships(plan)
                target_ids = conditioning.get(dataset_config["id"], ())
                if target_ids:
                    by_id = {entry["id"]: entry for entry in plan}
                    datasets = [by_id[target_id] for target_id in dict.fromkeys(target_ids)]

            for dataset in datasets:
                self._check_cancelled(job)
                with closing(self._create_dataset_runtime(dataset, args, accelerator)) as runtime:
                    self._prepare_conditioning_embeddings(
                        job, runtime, dataset, plan, global_config, args, accelerator, model
                    )

        finally:
            StateTracker.set_args(original_args)
            StateTracker.set_accelerator(original_accelerator)
            del model
            self._cleanup_gpu()

    def _prepare_conditioning_embeddings(self, job, runtime, dataset, dataset_plan, global_config, args, accelerator, model):
        cache_store, store_config = self._image_artifact_store(
            dataset, dataset_plan, args, accelerator, runtime.data_backend, "conditioning_image_embeds"
        )
        store_directory = store_config.get("cache_dir")
        if store_directory == dataset.get("cache_dir_vae"):
            store_directory = None
        cache_dir = self._resolve_cache_dir(
            dataset.get("cache_dir_conditioning_image_embeds")
            or store_directory
            or self._resolve_global_key(global_config, "cache_dir_conditioning_image_embeds"),
            global_config,
            dataset,
            absolute=cache_store.type != "aws",
        )
        if not cache_dir:
            raise ValueError("No conditioning cache directory configured")

        from simpletuner.helpers.caching.image_embed import ImageEmbedCache

        embed_cache = ImageEmbedCache(
            id=runtime.id,
            dataset_type=runtime.config["dataset_type"],
            model=model,
            accelerator=accelerator,
            metadata_backend=runtime.metadata_backend,
            image_data_backend=runtime.data_backend,
            cache_data_backend=cache_store,
            instance_data_dir=runtime.config["instance_data_dir"],
            cache_dir=cache_dir,
            write_batch_size=dataset.get("write_batch_size", args.write_batch_size),
            read_batch_size=dataset.get("read_batch_size", args.read_batch_size),
            embed_batch_size=dataset.get(
                "conditioning_image_embed_batch_size", args.conditioning_image_embed_batch_size or args.vae_batch_size
            ),
            hash_filenames=True,
        )

        self._check_cancelled(job)

        self._prepare_image_artifacts(job, runtime, "conditioning_image_embed_cache", embed_cache)


# Module-level singleton
_cache_service: Optional[CacheJobService] = None


def get_cache_service() -> CacheJobService:
    global _cache_service
    if _cache_service is None:
        try:
            from .sse_manager import get_sse_manager

            sse = get_sse_manager()
            _cache_service = CacheJobService(broadcast_fn=sse.broadcast_threadsafe)
        except Exception:
            _cache_service = CacheJobService()
    return _cache_service
