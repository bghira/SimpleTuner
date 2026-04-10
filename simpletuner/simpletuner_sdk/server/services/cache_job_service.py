"""Service for running cache operations (text embeds, VAE, conditioning) from the WebUI."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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


class _CacheAcceleratorStub:
    """Minimal accelerator stub with real device support for cache operations."""

    num_processes = 1
    process_index = 0
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


class CacheJobService:
    """Manages background cache operations with SSE progress broadcasting."""

    _lock = threading.Lock()
    _active_job: Optional[CacheJob] = None
    _thread: Optional[threading.Thread] = None

    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self._broadcast_fn = broadcast_fn

    def _broadcast(self, event_type: str, data: Dict[str, Any]):
        if self._broadcast_fn:
            self._broadcast_fn(data=data, event_type=event_type)

    @staticmethod
    def get_capabilities(
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
        }

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
            ):
                raise RuntimeError("A cache job is already in progress.")

            job_id = str(uuid.uuid4())[:8]
            job = CacheJob(job_id=job_id, dataset_id=dataset_id, cache_type=cache_type)
            self._active_job = job

        self._thread = threading.Thread(
            target=self._run_cache_job,
            args=(job, dataset_config, global_config),
            daemon=True,
        )
        self._thread.start()
        return job_id

    def get_active_status(self) -> Optional[Dict[str, Any]]:
        job = self._active_job
        if job and job.status in (
            CacheJobStatus.PENDING,
            CacheJobStatus.LOADING_MODEL,
            CacheJobStatus.RUNNING,
        ):
            return self._job_to_dict(job)
        return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._active_job
        if job and job.job_id == job_id:
            return self._job_to_dict(job)
        return None

    def cancel(self) -> bool:
        with self._lock:
            if self._active_job and self._active_job.status in (
                CacheJobStatus.PENDING,
                CacheJobStatus.LOADING_MODEL,
                CacheJobStatus.RUNNING,
            ):
                self._active_job.status = CacheJobStatus.CANCELLED
                self._active_job.finished_at = time.time()
                self._broadcast("dataset_cache", self._job_to_dict(self._active_job))
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
        }

    # ------------------------------------------------------------------
    # Background thread entry points
    # ------------------------------------------------------------------

    def _run_cache_job(self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any]):
        job.status = CacheJobStatus.LOADING_MODEL
        job.started_at = time.time()
        job.stage = "Loading model components"
        self._broadcast("dataset_cache", self._job_to_dict(job))

        try:
            if job.cache_type == CacheType.TEXT_EMBEDS.value:
                self._execute_text_embed_cache(job, dataset_config, global_config)
            elif job.cache_type == CacheType.VAE.value:
                self._execute_vae_cache(job, dataset_config, global_config)
            elif job.cache_type == CacheType.CONDITIONING.value:
                self._execute_conditioning_cache(job, dataset_config, global_config)
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
        return os.path.abspath(resolved) if resolved else None

    def _create_data_backend_and_metadata(
        self,
        dataset_config: Dict[str, Any],
        args: _CacheArgsNamespace,
        accelerator: _CacheAcceleratorStub,
    ):
        """Create data backend and metadata backend for a single dataset.

        Returns ``(config, data_backend, metadata_backend, config_id)``
        where *config_id* must be cleaned up from StateTracker after use.
        """
        from simpletuner.helpers.data_backend.builders import create_backend_builder
        from simpletuner.helpers.data_backend.config import create_backend_config
        from simpletuner.helpers.training.state_tracker import StateTracker

        from .dataset_scan_service import DatasetScanService

        dataset_config = DatasetScanService._convert_pixel_area(dataset_config, args)

        args_dict = {}
        for k in vars(args):
            if not k.startswith("_"):
                args_dict[k] = getattr(args, k, None)

        config = create_backend_config(dataset_config, args_dict)
        StateTracker.set_data_backend_config(config.id, config.to_dict()["config"])

        builder = create_backend_builder(config.backend_type, accelerator, args)
        data_backend = builder.build(config)
        metadata_backend = builder.create_metadata_backend(config, data_backend, args_dict)

        return config, data_backend, metadata_backend, config.id

    @staticmethod
    def _collect_image_files_from_metadata(metadata_backend) -> List[str]:
        """Read image file paths from the existing bucket cache."""
        try:
            bucket_data = metadata_backend.read_cache()
            if not bucket_data:
                return []
            files = []
            for file_list in bucket_data.values():
                if isinstance(file_list, list):
                    files.extend(file_list)
            return files
        except Exception:
            return []

    @staticmethod
    def _collect_prompt_records(
        model,
        dataset_config: Dict[str, Any],
        data_backend,
        metadata_backend,
    ) -> List[Dict[str, Any]]:
        """Gather captions from the dataset and build prompt records.

        Mirrors the caption collection logic in
        ``DataBackendFactory._process_text_embed_deferred_queue()``.
        """
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.prompts import PromptHandler
        from simpletuner.helpers.utils.pathing import normalize_data_path

        caption_strategy = dataset_config.get("caption_strategy", "filename")
        prepend_instance_prompt = dataset_config.get("prepend_instance_prompt", False)
        instance_prompt = dataset_config.get("instance_prompt", "")
        use_captions = True
        if dataset_config.get("only_instance_prompt"):
            use_captions = False
        elif caption_strategy == "instanceprompt":
            use_captions = False

        captions, _missing, caption_image_paths = PromptHandler.get_all_captions(
            data_backend=data_backend,
            instance_data_dir=dataset_config.get("instance_data_dir", ""),
            prepend_instance_prompt=prepend_instance_prompt,
            instance_prompt=instance_prompt,
            use_captions=use_captions,
            caption_strategy=caption_strategy,
            return_image_paths=True,
            disable_multiline_split=dataset_config.get("disable_multiline_split", False),
        )

        key_type = model.text_embed_cache_key()
        dataset_id = dataset_config.get("id", "")
        dataset_root = dataset_config.get("instance_data_dir")

        prompt_records: List[Dict[str, Any]] = []
        for caption, image_path in zip(captions, caption_image_paths):
            image_path_str = str(image_path)
            normalized_identifier = normalize_data_path(image_path_str, dataset_root)
            metadata = {
                "image_path": image_path_str,
                "data_backend_id": dataset_id,
                "prompt": caption,
                "dataset_relative_path": normalized_identifier,
            }
            if key_type is TextEmbedCacheKey.DATASET_AND_FILENAME:
                key_value = f"{dataset_id}:{normalized_identifier}"
            elif key_type is TextEmbedCacheKey.FILENAME:
                key_value = normalize_data_path(image_path_str, None)
            else:
                key_value = caption
            prompt_records.append({"prompt": caption, "key": key_value, "metadata": metadata})

        # Add grounding entity labels if metadata has bbox_entities
        if metadata_backend is not None:
            for image_path in caption_image_paths:
                image_path_str = str(image_path)
                img_meta = metadata_backend.get_metadata_by_filepath(image_path_str)
                if not isinstance(img_meta, dict):
                    continue
                bbox_entities = img_meta.get("bbox_entities")
                if not bbox_entities:
                    continue
                normalized_identifier = normalize_data_path(image_path_str, dataset_root)
                for idx, entity in enumerate(bbox_entities):
                    label = entity.get("label", "")
                    if not label:
                        continue
                    entity_key = f"{normalized_identifier}__bbox_{idx}"
                    prompt_records.append({"prompt": label, "key": entity_key, "metadata": {"grounding_entity": True}})

        return prompt_records

    def _load_model(self, global_config: Dict[str, Any], args, accelerator):
        from simpletuner.helpers.models.registry import ModelRegistry

        model_family = global_config.get("model_family")
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

    def _execute_vae_cache(self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any]):
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        config_id = None
        model = None
        vae_cache = None

        try:
            self._check_cancelled(job)

            job.stage = "Loading VAE"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)
            model.load_vae()
            vae = getattr(model, "vae", None) or StateTracker.get_vae()

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            config, data_backend, metadata_backend, config_id = self._create_data_backend_and_metadata(
                dataset_config, args, accelerator
            )

            image_files = self._collect_image_files_from_metadata(metadata_backend)
            if not image_files:
                raise ValueError("No image files found in metadata cache. Run a metadata scan first.")

            cache_dir_vae = self._resolve_cache_dir(
                dataset_config.get("cache_dir_vae") or self._resolve_global_key(global_config, "cache_dir_vae"),
                global_config,
                dataset_config,
            )
            if not cache_dir_vae:
                raise ValueError("No VAE cache directory configured for this dataset")
            os.makedirs(cache_dir_vae, exist_ok=True)

            from simpletuner.helpers.caching.vae import VAECache

            dataset_type = str(dataset_config.get("dataset_type", "image")).lower()

            vae_cache = VAECache(
                id=dataset_config["id"],
                dataset_type=dataset_type,
                model=model,
                vae=vae,
                accelerator=accelerator,
                metadata_backend=metadata_backend,
                image_data_backend=data_backend,
                cache_data_backend=data_backend,
                instance_data_dir=dataset_config.get("instance_data_dir", ""),
                cache_dir=cache_dir_vae,
                vae_batch_size=int(global_config.get("vae_batch_size", 4) or 4),
                write_batch_size=int(global_config.get("write_batch_size", 64) or 64),
                read_batch_size=int(global_config.get("read_batch_size", 25) or 25),
                process_queue_size=int(global_config.get("process_queue_size", 16) or 16),
                hash_filenames=True,
            )

            self._check_cancelled(job)

            job.stage = "Discovering files"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            vae_cache.discover_all_files()
            vae_cache.build_vae_cache_filename_map(all_image_files=image_files)
            unprocessed = vae_cache.discover_unprocessed_files()

            if not unprocessed:
                logger.info("All %d files already cached for %s", len(image_files), job.dataset_id)
                job.stage = "All files already cached"
                return

            job.stage = "Initialising VAE"
            self._broadcast("dataset_cache", self._job_to_dict(job))
            vae_cache.init_vae()

            self._check_cancelled(job)

            job.status = CacheJobStatus.RUNNING
            job.stage = "Processing VAE cache"
            job.total = len(unprocessed)
            self._broadcast("dataset_cache", self._job_to_dict(job))

            _last_broadcast = [0.0]

            def _vae_progress(current, total):
                self._check_cancelled(job)
                job.current = current
                job.total = total
                now = time.time()
                if now - _last_broadcast[0] >= 1.0 or current >= total:
                    _last_broadcast[0] = now
                    self._broadcast("dataset_cache", self._job_to_dict(job))

            vae_cache.process_buckets(progress_callback=_vae_progress)

        finally:
            StateTracker.set_args(original_args)
            StateTracker.set_accelerator(original_accelerator)
            if config_id is not None:
                StateTracker.data_backends.pop(config_id, None)
            del vae_cache, model
            self._cleanup_gpu()

    def _execute_text_embed_cache(self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any]):
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        config_id = None
        model = None
        text_cache = None

        try:
            self._check_cancelled(job)

            job.stage = "Loading text encoders"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)
            model.load_text_encoder()
            model.load_text_tokenizer()

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            config, data_backend, metadata_backend, config_id = self._create_data_backend_and_metadata(
                dataset_config, args, accelerator
            )

            cache_dir_text = self._resolve_cache_dir(
                dataset_config.get("cache_dir_text") or self._resolve_global_key(global_config, "cache_dir_text", "cache"),
                global_config,
                dataset_config,
            )
            if not cache_dir_text:
                raise ValueError("No text embed cache directory configured for this dataset")
            os.makedirs(cache_dir_text, exist_ok=True)

            from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache

            text_cache = TextEmbeddingCache(
                id=dataset_config["id"],
                data_backend=data_backend,
                text_encoders=model.text_encoders,
                tokenizers=model.tokenizers,
                accelerator=accelerator,
                cache_dir=cache_dir_text,
                model_type=global_config.get("model_family", ""),
                write_batch_size=int(global_config.get("write_batch_size", 64) or 64),
                model=model,
            )

            self._check_cancelled(job)

            # Gather captions from the dataset
            job.stage = "Collecting captions"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            prompt_records = self._collect_prompt_records(model, dataset_config, data_backend, metadata_backend)
            if not prompt_records:
                raise ValueError("No captions found for this dataset")

            # Move text encoders to device
            from simpletuner.helpers.data_backend.factory import move_text_encoders

            move_text_encoders(args, model.text_encoders, accelerator.device, force_move=True)

            self._check_cancelled(job)

            job.status = CacheJobStatus.RUNNING
            job.stage = "Computing text embeddings"
            job.total = len(prompt_records)
            self._broadcast("dataset_cache", self._job_to_dict(job))

            _last_broadcast = [0.0]

            def _text_progress(current, total):
                self._check_cancelled(job)
                job.current = current
                job.total = total
                now = time.time()
                if now - _last_broadcast[0] >= 1.0 or current >= total:
                    _last_broadcast[0] = now
                    self._broadcast("dataset_cache", self._job_to_dict(job))

            text_cache.discover_all_files()
            text_cache.encode_dropout_caption()
            text_cache.compute_embeddings_for_prompts(
                prompt_records,
                return_concat=False,
                load_from_cache=False,
                progress_callback=_text_progress,
            )

        finally:
            StateTracker.set_args(original_args)
            StateTracker.set_accelerator(original_accelerator)
            if config_id is not None:
                StateTracker.data_backends.pop(config_id, None)
            del text_cache, model
            self._cleanup_gpu()

    def _execute_conditioning_cache(self, job: CacheJob, dataset_config: Dict[str, Any], global_config: Dict[str, Any]):
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = self._build_args(global_config)
        accelerator = _CacheAcceleratorStub()
        original_args = StateTracker.get_args()
        original_accelerator = StateTracker.get_accelerator()
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        config_id = None
        model = None
        embed_cache = None

        try:
            self._check_cancelled(job)

            job.stage = "Loading model for conditioning embeds"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            model = self._load_model(global_config, args, accelerator)

            if not model.requires_conditioning_image_embeds():
                raise ValueError(
                    f"Model family '{global_config.get('model_family')}' "
                    "does not require conditioning image embeds with this configuration"
                )

            self._check_cancelled(job)

            job.stage = "Setting up data backend"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            config, data_backend, metadata_backend, config_id = self._create_data_backend_and_metadata(
                dataset_config, args, accelerator
            )

            image_files = self._collect_image_files_from_metadata(metadata_backend)
            if not image_files:
                raise ValueError("No image files found in metadata cache. Run a metadata scan first.")

            cache_dir = self._resolve_cache_dir(
                dataset_config.get("cache_dir_conditioning_image_embeds")
                or self._resolve_global_key(global_config, "cache_dir_conditioning_image_embeds"),
                global_config,
                dataset_config,
            )
            if not cache_dir:
                raise ValueError("No conditioning cache directory configured")
            os.makedirs(cache_dir, exist_ok=True)

            from simpletuner.helpers.caching.image_embed import ImageEmbedCache

            dataset_type = str(dataset_config.get("dataset_type", "conditioning")).lower()

            embed_cache = ImageEmbedCache(
                id=dataset_config["id"],
                dataset_type=dataset_type,
                model=model,
                accelerator=accelerator,
                metadata_backend=metadata_backend,
                image_data_backend=data_backend,
                cache_data_backend=data_backend,
                instance_data_dir=dataset_config.get("instance_data_dir", ""),
                cache_dir=cache_dir,
                write_batch_size=int(global_config.get("write_batch_size", 64) or 64),
                read_batch_size=int(global_config.get("read_batch_size", 25) or 25),
                embed_batch_size=int(global_config.get("embed_batch_size", 4) or 4),
                hash_filenames=True,
            )

            self._check_cancelled(job)

            job.status = CacheJobStatus.RUNNING
            job.stage = "Computing conditioning embeddings"
            self._broadcast("dataset_cache", self._job_to_dict(job))

            embed_cache.discover_all_files()
            embed_cache.build_embed_filename_map(all_image_files=image_files)
            pending = embed_cache.discover_unprocessed_files()
            if pending:
                job.total = len(pending)
                self._broadcast("dataset_cache", self._job_to_dict(job))

                _last_broadcast = [0.0]

                def _cond_progress(current, total):
                    self._check_cancelled(job)
                    job.current = current
                    job.total = total
                    now = time.time()
                    if now - _last_broadcast[0] >= 1.0 or current >= total:
                        _last_broadcast[0] = now
                        self._broadcast("dataset_cache", self._job_to_dict(job))

                embed_cache.process_files(pending, progress_callback=_cond_progress)

        finally:
            StateTracker.set_args(original_args)
            StateTracker.set_accelerator(original_accelerator)
            if config_id is not None:
                StateTracker.data_backends.pop(config_id, None)
            del embed_cache, model
            self._cleanup_gpu()


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
