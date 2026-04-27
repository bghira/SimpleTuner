"""CaptionFlow job integration for local GPU captioning."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hardware_service import detect_gpu_inventory

logger = logging.getLogger(__name__)


DEFAULT_CAPTIONFLOW_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CAPTIONFLOW_PROMPT = (
    "You're a caption bot designed to output image descriptions. " "Describe what you see. Output only the caption."
)
LOCAL_FILESYSTEM_SOURCE = "local_filesystem"
HUGGINGFACE_SOURCE = "huggingface_datasets"
ORCHESTRATOR_READY_LOG_LINE = "Orchestrator ready for connections"


@dataclass
class CaptionFlowJobResult:
    """Result of a CaptionFlow job submission."""

    job_id: Optional[str]
    status: str
    allocated_gpus: Optional[List[int]] = None
    queue_position: Optional[int] = None
    reason: Optional[str] = None


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(lambda: asyncio.run(coro)).result(timeout=10)


def _detect_local_hardware() -> str:
    try:
        inventory = detect_gpu_inventory()
    except Exception:
        logger.debug("Failed to detect local hardware for captioning job", exc_info=True)
        return "local-gpu"

    total = int(inventory.get("total_gpus") or 0)
    if total <= 0:
        return "local-gpu"

    names = []
    for gpu in inventory.get("gpus") or []:
        name = str(gpu.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    if names:
        return f"{total}x {', '.join(names[:2])}"
    return f"{total}x local GPU"


def _coerce_positive_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def _parse_raw_captionflow_config(raw_config: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_config or not raw_config.strip():
        return None

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("PyYAML is required to parse raw CaptionFlow YAML configs") from exc
        try:
            parsed = yaml.safe_load(raw_config)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid raw CaptionFlow YAML config: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Raw CaptionFlow config must be a JSON or YAML object")

    orchestrator = parsed.get("orchestrator", parsed)
    if not isinstance(orchestrator, dict):
        raise ValueError("Raw CaptionFlow config must contain an orchestrator object")

    return copy.deepcopy(orchestrator)


def resolve_captioning_dataset_path(dataset_config: Dict[str, Any]) -> Path:
    """Resolve the local dataset root CaptionFlow can read."""
    dataset_type = str(dataset_config.get("dataset_type") or "image").lower()
    if dataset_type in {"text_embeds", "image_embeds", "audio"}:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported for CaptionFlow captioning")

    raw_path = dataset_config.get("instance_data_dir") or dataset_config.get("dataset_path")
    if not raw_path:
        raise ValueError("Dataset has no instance_data_dir configured")

    dataset_path = Path(str(raw_path)).expanduser()
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise ValueError(f"Dataset path does not exist or is not a directory: {dataset_path}")
    return dataset_path.resolve()


def _get_huggingface_block(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    block = dataset_config.get("huggingface")
    return block if isinstance(block, dict) else {}


def _resolve_huggingface_url_column(dataset_config: Dict[str, Any], hf_block: Dict[str, Any], image_column: str) -> str:
    explicit = (
        dataset_config.get("dataset_url_column")
        or dataset_config.get("url_column")
        or hf_block.get("dataset_url_column")
        or hf_block.get("url_column")
    )
    if explicit:
        return str(explicit)
    if "url" in image_column.lower():
        return image_column
    return ""


def resolve_captioning_dataset_source(dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Map a SimpleTuner dataset config to CaptionFlow's processor source config."""
    dataset_type = str(dataset_config.get("dataset_type") or "image").lower()
    if dataset_type in {"text_embeds", "image_embeds", "audio"}:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported for CaptionFlow captioning")

    backend_type = str(dataset_config.get("type") or "local").lower()
    if backend_type == "huggingface":
        hf_block = _get_huggingface_block(dataset_config)
        dataset_name = dataset_config.get("dataset_name") or hf_block.get("dataset_name")
        if not dataset_name:
            raise ValueError("HuggingFace dataset has no dataset_name configured")

        image_column = str(dataset_config.get("image_column") or hf_block.get("image_column") or "image")
        return {
            "source_type": HUGGINGFACE_SOURCE,
            "dataset_path": str(dataset_name),
            "dataset_name": dataset_config.get("id") or dataset_name,
            "dataset_config": dataset_config.get("dataset_config")
            or hf_block.get("dataset_config")
            or hf_block.get("config"),
            "dataset_split": dataset_config.get("split") or hf_block.get("split") or "train",
            "dataset_image_column": image_column,
            "dataset_url_column": _resolve_huggingface_url_column(dataset_config, hf_block, image_column),
            "revision": dataset_config.get("revision") or hf_block.get("revision"),
        }

    dataset_path = resolve_captioning_dataset_path(dataset_config)
    return {
        "source_type": LOCAL_FILESYSTEM_SOURCE,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_config.get("id") or dataset_path.name,
    }


def build_captionflow_runtime_config(
    *,
    job_id: str,
    dataset_id: str,
    dataset_config: Dict[str, Any],
    global_config: Dict[str, Any],
    request_config: Dict[str, Any],
    allocated_gpus: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Build the job config consumed by the process keeper subprocess."""
    source = resolve_captioning_dataset_source(dataset_config)
    output_dir = (
        request_config.get("output_dir") or global_config.get("--output_dir") or global_config.get("output_dir") or "output"
    )
    output_root = Path(str(output_dir)).expanduser()
    workspace = output_root / "captionflow" / job_id

    worker_count = _coerce_positive_int(request_config.get("worker_count"), default=1, maximum=32)
    batch_size = _coerce_positive_int(request_config.get("batch_size"), default=8, maximum=256)
    chunk_size = _coerce_positive_int(request_config.get("chunk_size"), default=256, maximum=10000)
    max_tokens = _coerce_positive_int(request_config.get("max_tokens"), default=256, maximum=4096)
    source_type = source["source_type"]
    export_textfiles = bool(request_config.get("export_textfiles", True)) and source_type == LOCAL_FILESYSTEM_SOURCE
    raw_orchestrator_config = _parse_raw_captionflow_config(request_config.get("raw_config"))

    return {
        "__job_id__": job_id,
        "output_dir": str(output_root),
        "workspace_dir": str(workspace),
        "dataset_id": dataset_id,
        "dataset_path": source["dataset_path"],
        "dataset_export_dir": source["dataset_path"] if source_type == LOCAL_FILESYSTEM_SOURCE else None,
        "dataset_name": source["dataset_name"],
        "source_type": source_type,
        "dataset_config": source.get("dataset_config"),
        "dataset_split": source.get("dataset_split"),
        "dataset_image_column": source.get("dataset_image_column"),
        "dataset_url_column": source.get("dataset_url_column"),
        "revision": source.get("revision"),
        "model": request_config.get("model") or DEFAULT_CAPTIONFLOW_MODEL,
        "prompt": request_config.get("prompt") or DEFAULT_CAPTIONFLOW_PROMPT,
        "output_field": request_config.get("output_field") or "captions",
        "worker_count": worker_count,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "max_tokens": max_tokens,
        "temperature": float(request_config.get("temperature", 0.7)),
        "top_p": float(request_config.get("top_p", 0.95)),
        "gpu_memory_utilization": float(request_config.get("gpu_memory_utilization", 0.92)),
        "export_textfiles": export_textfiles,
        "export_jsonl": bool(request_config.get("export_jsonl", source_type != LOCAL_FILESYSTEM_SOURCE)),
        "raw_orchestrator_config": raw_orchestrator_config,
        "allocated_gpus": allocated_gpus or [],
    }


def _queue_captionflow_job(
    *,
    job_id: str,
    runtime_config: Dict[str, Any],
    dataset_id: str,
    num_processes: int,
    preferred_gpus: Optional[List[int]],
    any_gpu: bool,
    user_id: Optional[int] = None,
    org_id: Optional[int] = None,
) -> CaptionFlowJobResult:
    from .cloud.base import CloudJobStatus, JobType, UnifiedJob
    from .cloud.storage.job_repository import get_job_repository

    async def _add_to_queue():
        job_repo = get_job_repository()
        now = datetime.now(timezone.utc).isoformat()
        stats = await job_repo.get_queue_stats()
        queue_depth = stats.get("queue_depth", 0)
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.LOCAL,
            provider="captionflow",
            status=CloudJobStatus.QUEUED.value,
            config_name=f"Captioning: {dataset_id}",
            created_at=now,
            queued_at=now,
            user_id=user_id,
            org_id=org_id,
            num_processes=num_processes,
            allocated_gpus=preferred_gpus if not any_gpu else None,
            queue_position=queue_depth + 1,
            output_url=runtime_config.get("workspace_dir"),
            hardware_type=_detect_local_hardware(),
            metadata={
                "handler": "captionflow",
                "runtime_config": runtime_config,
                "any_gpu": any_gpu,
                "dataset_id": dataset_id,
                "run_name": f"Captioning: {dataset_id}",
            },
        )
        await job_repo.add(job)
        return job.queue_position

    position = _run_async(_add_to_queue())
    return CaptionFlowJobResult(
        job_id=job_id,
        status="queued",
        queue_position=position,
        reason=f"Waiting for {num_processes} GPU(s) to become available",
    )


def start_captionflow_job(
    *,
    dataset_id: str,
    dataset_config: Dict[str, Any],
    global_config: Dict[str, Any],
    request_config: Dict[str, Any],
    no_wait: bool = False,
    any_gpu: bool = False,
    user_id: Optional[int] = None,
    org_id: Optional[int] = None,
) -> CaptionFlowJobResult:
    """Submit a CaptionFlow job through the local GPU queue."""
    from .local_gpu_allocator import get_gpu_allocator

    job_id = str(uuid.uuid4())[:8]
    worker_count = _coerce_positive_int(request_config.get("worker_count"), default=1, maximum=32)
    preferred_gpus = request_config.get("preferred_gpus")
    if not isinstance(preferred_gpus, list) or not preferred_gpus:
        preferred_gpus = None

    runtime_config = build_captionflow_runtime_config(
        job_id=job_id,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        global_config=global_config,
        request_config=request_config,
    )

    allocator = get_gpu_allocator()

    async def _check():
        return await allocator.can_allocate(
            required_count=worker_count,
            preferred_gpus=preferred_gpus,
            any_gpu=any_gpu,
            org_id=org_id,
        )

    can_start, gpus_to_use, reason = _run_async(_check())
    if not can_start:
        if no_wait:
            return CaptionFlowJobResult(
                job_id=None,
                status="rejected",
                reason=reason or "Required GPUs unavailable",
            )
        return _queue_captionflow_job(
            job_id=job_id,
            runtime_config=runtime_config,
            dataset_id=dataset_id,
            num_processes=worker_count,
            preferred_gpus=preferred_gpus,
            any_gpu=any_gpu,
            user_id=user_id,
            org_id=org_id,
        )

    runtime_config["allocated_gpus"] = gpus_to_use

    from .cloud.base import CloudJobStatus, JobType, UnifiedJob
    from .cloud.storage.job_repository import get_job_repository

    async def _create_and_allocate():
        job_repo = get_job_repository()
        now = datetime.now(timezone.utc).isoformat()
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.LOCAL,
            provider="captionflow",
            status=CloudJobStatus.RUNNING.value,
            config_name=f"Captioning: {dataset_id}",
            created_at=now,
            started_at=now,
            queued_at=now,
            user_id=user_id,
            org_id=org_id,
            num_processes=worker_count,
            allocated_gpus=gpus_to_use,
            output_url=runtime_config.get("workspace_dir"),
            hardware_type=_detect_local_hardware(),
            metadata={
                "handler": "captionflow",
                "runtime_config": runtime_config,
                "any_gpu": any_gpu,
                "dataset_id": dataset_id,
                "run_name": f"Captioning: {dataset_id}",
            },
        )
        await job_repo.add(job)
        if not await allocator.allocate(job_id, gpus_to_use):
            await job_repo.mark_failed(job_id, f"Failed to allocate GPUs {gpus_to_use}")
            raise RuntimeError(f"Failed to allocate GPUs {gpus_to_use}")

    _run_async(_create_and_allocate())

    from simpletuner.simpletuner_sdk import process_keeper

    process_keeper.submit_job(job_id, run_captionflow_job, runtime_config)
    _store_pid(job_id)

    return CaptionFlowJobResult(job_id=job_id, status="running", allocated_gpus=gpus_to_use)


def _store_pid(job_id: str) -> None:
    from simpletuner.simpletuner_sdk import process_keeper

    pid = process_keeper.get_process_pid(job_id)
    if not pid:
        return

    from .cloud.storage.job_repository import get_job_repository

    async def _update():
        job_repo = get_job_repository()
        job = await job_repo.get(job_id)
        if job:
            metadata = (job.metadata or {}).copy()
            metadata["pid"] = pid
            await job_repo.update(job_id, {"metadata": metadata})

    _run_async(_update())


def run_captionflow_job(config) -> Dict[str, Any]:
    """Process keeper entrypoint for a CaptionFlow job."""
    from simpletuner.simpletuner_sdk.server.services.captionflow_job_service import _run_captionflow_job_impl

    return _run_captionflow_job_impl(config)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _captionflow_orchestrator_startup_timeout() -> float:
    raw_timeout = os.environ.get("SIMPLETUNER_CAPTIONFLOW_ORCHESTRATOR_TIMEOUT", "180")
    try:
        return max(15.0, float(raw_timeout))
    except ValueError:
        return 180.0


def _captionflow_export_ready_timeout() -> float:
    raw_timeout = os.environ.get("SIMPLETUNER_CAPTIONFLOW_EXPORT_READY_TIMEOUT", "60")
    try:
        return max(1.0, float(raw_timeout))
    except ValueError:
        return 60.0


def _wait_for_orchestrator_ready(
    process: subprocess.Popen,
    log_path: Path,
    host: str,
    port: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(_process_failure_message("orchestrator", exit_code, log_path))

        if ORCHESTRATOR_READY_LOG_LINE in _tail_text_file(log_path):
            return
        time.sleep(0.5)

    detail = _extract_process_error_line(log_path)
    log_hint = f" See {log_path}."
    if detail:
        raise TimeoutError(
            f"CaptionFlow orchestrator did not report readiness on {host}:{port} within {timeout:.0f}s: "
            f"{detail}.{log_hint}"
        )
    raise TimeoutError(
        f"CaptionFlow orchestrator did not report readiness on {host}:{port} within {timeout:.0f}s.{log_hint}"
    )


def _terminate_processes(processes: List[subprocess.Popen], timeout: float = 10.0) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                process.kill()


def _tail_text_file(path: Path, max_bytes: int = 12000) -> str:
    if not path.exists():
        return ""

    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
            handle.readline()
        else:
            handle.seek(0)
        return handle.read().decode("utf-8", errors="replace")


def _extract_process_error_line(log_path: Path) -> str:
    tail = _tail_text_file(log_path)
    if not tail:
        return ""

    error_tokens = (
        "Traceback",
        "Error:",
        "Exception:",
        "ImportError:",
        "ModuleNotFoundError:",
        "RuntimeError:",
        "ValueError:",
        "ERROR",
        "CRITICAL",
    )
    for line in reversed(tail.splitlines()):
        stripped = line.strip()
        if stripped and any(token in stripped for token in error_tokens):
            return stripped[-2000:]
    for line in reversed(tail.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[-2000:]
    return ""


def _process_failure_message(process_name: str, returncode: int, log_path: Path) -> str:
    error_line = _extract_process_error_line(log_path)
    log_hint = f" See {log_path}."
    if error_line:
        return f"CaptionFlow {process_name} exited with status {returncode}: {error_line}.{log_hint}"
    return f"CaptionFlow {process_name} exited with status {returncode}.{log_hint}"


def _stop_orchestrator_before_export(process: subprocess.Popen, log_path: Path, timeout: float = 30.0) -> None:
    exit_code = process.poll()
    if exit_code is not None:
        if exit_code != 0:
            raise RuntimeError(_process_failure_message("orchestrator", exit_code, log_path))
        return

    process.send_signal(signal.SIGINT)
    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"CaptionFlow orchestrator did not checkpoint before export within {timeout:.0f}s. See {log_path}."
        ) from exc

    if exit_code != 0:
        raise RuntimeError(_process_failure_message("orchestrator", exit_code, log_path))


async def _load_captionflow_storage_contents(storage_dir: Path):
    from caption_flow.storage import StorageManager

    storage = StorageManager(storage_dir)
    await storage.initialize()
    return await storage.get_shard_contents("default")


def _captionflow_storage_not_ready_detail(contents) -> str:
    metadata = getattr(contents, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("error", "message"):
            value = metadata.get(key)
            if value:
                return str(value)
    rows = len(getattr(contents, "rows", []) or [])
    fields = list(getattr(contents, "output_fields", []) or [])
    return f"rows={rows}, output_fields={fields}, columns={list(getattr(contents, 'columns', []) or [])}"


def _wait_for_captionflow_storage_contents(storage_dir: Path, timeout: float):
    deadline = time.monotonic() + timeout
    last_detail = ""

    while time.monotonic() < deadline:
        contents = asyncio.run(_load_captionflow_storage_contents(storage_dir))
        if getattr(contents, "columns", None):
            return contents
        last_detail = _captionflow_storage_not_ready_detail(contents)
        time.sleep(1.0)

    detail = f": {last_detail}" if last_detail else ""
    raise RuntimeError(
        f"CaptionFlow completed, but no exportable caption columns were written within {timeout:.0f}s{detail}."
    )


def _prepare_captionflow_import_shims(workspace: Path) -> Path:
    """Create job-scoped shims for optional CaptionFlow imports we do not exercise."""
    shim_dir = workspace / "import_shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "cv2.py").write_text(
        "\n".join(
            [
                '"""SimpleTuner CaptionFlow import shim for unused WebDataset OpenCV paths."""',
                "IMREAD_COLOR = 1",
                "COLOR_BGR2RGB = 4",
                "",
                "def imdecode(*_args, **_kwargs):",
                "    raise RuntimeError('OpenCV is not available in the SimpleTuner captioning environment.')",
                "",
                "def cvtColor(*_args, **_kwargs):",
                "    raise RuntimeError('OpenCV is not available in the SimpleTuner captioning environment.')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return shim_dir


def _python_site_package_roots() -> List[Path]:
    roots = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.append(Path(value))
    fallback = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    roots.append(fallback)

    deduped = []
    seen = set()
    for root in roots:
        resolved = root.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)
    return deduped


def _discover_python_nvidia_library_dirs(site_roots: Optional[List[Path]] = None) -> List[Path]:
    """Find NVIDIA wheel library directories installed in the active Python environment."""
    library_dirs = []
    seen = set()
    for site_root in site_roots or _python_site_package_roots():
        nvidia_root = site_root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if not lib_dir.is_dir():
                continue
            key = str(lib_dir)
            if key in seen:
                continue
            seen.add(key)
            library_dirs.append(lib_dir)
    return library_dirs


def _prepare_cuda_library_shims(workspace: Path, library_dirs: List[Path]) -> Optional[Path]:
    cuda_lib_dir = workspace / "cuda_libs"
    cudart_candidates = []
    for lib_dir in library_dirs:
        cudart_candidates.extend(sorted(lib_dir.glob("libcudart.so.*")))

    if not cudart_candidates:
        return None

    cuda_lib_dir.mkdir(parents=True, exist_ok=True)
    shim_path = cuda_lib_dir / "libcudart.so"
    if not shim_path.exists():
        shim_path.symlink_to(cudart_candidates[0])
    return cuda_lib_dir


def _prepend_env_path(env: Dict[str, str], name: str, paths: List[Path]) -> None:
    values = [str(path) for path in paths if path]
    if not values:
        return

    existing = env.get(name)
    if existing:
        values.append(existing)
    env[name] = os.pathsep.join(values)


def _prepare_captionflow_subprocess_env(workspace: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["CAPTIONFLOW_LOG_DIR"] = str(workspace / "logs")

    import_shim_dir = _prepare_captionflow_import_shims(workspace)
    _prepend_env_path(env, "PYTHONPATH", [import_shim_dir])

    nvidia_library_dirs = _discover_python_nvidia_library_dirs()
    cuda_shim_dir = _prepare_cuda_library_shims(workspace, nvidia_library_dirs)
    ld_paths = ([cuda_shim_dir] if cuda_shim_dir else []) + nvidia_library_dirs
    _prepend_env_path(env, "LD_LIBRARY_PATH", ld_paths)
    return env


def _build_orchestrator_config(config, orchestrator_port: int, image_port: int, worker_token: str) -> Dict[str, Any]:
    workspace = Path(config.workspace_dir)
    storage_dir = workspace / "caption_data"
    checkpoint_dir = workspace / "checkpoints"
    source_type = str(config.source_type)
    if source_type == HUGGINGFACE_SOURCE:
        dataset_config = {
            "type": "huggingface",
            "processor_type": "huggingface_datasets",
            "dataset_path": config.dataset_path,
            "name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "dataset_image_column": config.dataset_image_column,
            "dataset_url_column": config.dataset_url_column,
        }
        if config.dataset_config:
            dataset_config["dataset_config"] = config.dataset_config
    else:
        dataset_config = {
            "type": "local_filesystem",
            "processor_type": "local_filesystem",
            "dataset_path": config.dataset_path,
            "name": config.dataset_name,
            "recursive": True,
            "http_bind_address": "127.0.0.1",
            "public_address": "127.0.0.1",
            "http_port": image_port,
        }

    raw_config = getattr(config, "raw_orchestrator_config", None)
    if isinstance(raw_config, dict):
        orchestrator = copy.deepcopy(raw_config)
    else:
        orchestrator = {
            "chunk_size": int(config.chunk_size),
            "chunks_per_request": 2,
            "chunk_buffer_multiplier": 3,
            "min_chunk_buffer": 2,
            "vllm": {
                "model": config.model,
                "tensor_parallel_size": 1,
                "max_model_len": 16384,
                "dtype": "float16",
                "gpu_memory_utilization": float(config.gpu_memory_utilization),
                "enforce_eager": True,
                "disable_mm_preprocessor_cache": True,
                "limit_mm_per_prompt": {"image": 1},
                "batch_size": int(config.batch_size),
                "sampling": {
                    "temperature": float(config.temperature),
                    "top_p": float(config.top_p),
                    "max_tokens": int(config.max_tokens),
                    "repetition_penalty": 1.05,
                    "skip_special_tokens": True,
                    "stop": ["<|end|>", "<|endoftext|>", "<|im_end|>"],
                },
                "inference_prompts": [config.prompt],
            },
        }

    orchestrator["host"] = "127.0.0.1"
    orchestrator["port"] = orchestrator_port
    orchestrator.pop("ssl", None)
    orchestrator["dataset"] = dataset_config
    orchestrator.setdefault("chunk_size", int(config.chunk_size))
    orchestrator.setdefault("chunks_per_request", 2)
    orchestrator.setdefault("chunk_buffer_multiplier", 3)
    orchestrator.setdefault("min_chunk_buffer", 2)
    orchestrator["storage"] = {
        **(orchestrator.get("storage") if isinstance(orchestrator.get("storage"), dict) else {}),
        "data_dir": str(storage_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }
    orchestrator["storage"].setdefault("checkpoint_interval", 1000)
    orchestrator["storage"].setdefault("caption_buffer_size", 100)
    orchestrator["storage"].setdefault("job_buffer_size", 100)
    orchestrator["storage"].setdefault("contributor_buffer_size", 10)
    orchestrator["auth"] = {
        "worker_tokens": [{"token": worker_token, "name": "SimpleTuner CaptionFlow Worker"}],
        "admin_tokens": [{"token": secrets.token_urlsafe(18), "name": "SimpleTuner"}],
    }

    return {"orchestrator": orchestrator}


def _run_captionflow_job_impl(config) -> Dict[str, Any]:
    workspace = Path(config.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    orchestrator_port = _find_free_port()
    image_port = _find_free_port()
    worker_token = secrets.token_urlsafe(24)
    config_path = workspace / "orchestrator.yaml"
    storage_dir = workspace / "caption_data"

    orchestrator_config = _build_orchestrator_config(config, orchestrator_port, image_port, worker_token)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(orchestrator_config, handle, indent=2)

    env = _prepare_captionflow_subprocess_env(workspace)

    orchestrator_cmd = [
        sys.executable,
        "-m",
        "caption_flow.cli",
        "orchestrator",
        "--config",
        str(config_path),
        "--no-ssl",
        "--vllm",
    ]
    print(f"Starting CaptionFlow orchestrator on 127.0.0.1:{orchestrator_port}")
    log_handles = []
    process_logs: Dict[subprocess.Popen, Path] = {}
    orchestrator_log = logs_dir / "orchestrator.log"
    orchestrator_handle = open(orchestrator_log, "ab", buffering=0)
    log_handles.append(orchestrator_handle)
    orchestrator = subprocess.Popen(orchestrator_cmd, env=env, stdout=orchestrator_handle, stderr=subprocess.STDOUT)
    process_logs[orchestrator] = orchestrator_log
    processes = [orchestrator]

    try:
        _wait_for_orchestrator_ready(
            orchestrator,
            orchestrator_log,
            "127.0.0.1",
            orchestrator_port,
            _captionflow_orchestrator_startup_timeout(),
        )

        gpus = list(config.allocated_gpus or [])
        worker_count = int(config.worker_count)
        if not gpus:
            gpus = list(range(worker_count))

        workers: List[subprocess.Popen] = []
        for idx, gpu_id in enumerate(gpus[:worker_count], start=1):
            worker_cmd = [
                sys.executable,
                "-m",
                "caption_flow.cli",
                "worker",
                "--server",
                f"ws://127.0.0.1:{orchestrator_port}",
                "--token",
                worker_token,
                "--name",
                f"SimpleTuner caption worker {idx}",
                "--vllm",
                "--gpu-id",
                str(gpu_id),
                "--when_finished",
                "shutdown",
                "--no-verify-ssl",
            ]
            print(f"Starting CaptionFlow worker {idx} on GPU {gpu_id}")
            worker_log = logs_dir / f"worker-{idx}.log"
            worker_handle = open(worker_log, "ab", buffering=0)
            log_handles.append(worker_handle)
            process = subprocess.Popen(worker_cmd, env=env, stdout=worker_handle, stderr=subprocess.STDOUT)
            process_logs[process] = worker_log
            workers.append(process)
            processes.append(process)

        while True:
            if config.should_abort():
                raise RuntimeError("CaptionFlow job aborted")

            orchestrator_exit = orchestrator.poll()
            if orchestrator_exit is not None:
                raise RuntimeError(_process_failure_message("orchestrator", orchestrator_exit, orchestrator_log))

            failed_worker = next((worker for worker in workers if worker.poll() not in (None, 0)), None)
            if failed_worker is not None:
                log_path = process_logs.get(failed_worker, logs_dir / "worker.log")
                raise RuntimeError(_process_failure_message("worker", failed_worker.returncode, log_path))

            if workers and all(worker.poll() == 0 for worker in workers):
                break

            time.sleep(2.0)

        _stop_orchestrator_before_export(orchestrator, orchestrator_log)

        exported = 0
        export_path = None
        from caption_flow.storage.exporter import StorageExporter

        contents = _wait_for_captionflow_storage_contents(storage_dir, _captionflow_export_ready_timeout())
        exporter = StorageExporter(contents)

        if bool(config.export_textfiles):
            dataset_export_dir = getattr(config, "dataset_export_dir", None) or config.dataset_path
            print(f"Exporting CaptionFlow text captions into {dataset_export_dir}")
            exported = exporter.to_txt(
                dataset_export_dir,
                filename_column="filename",
                export_column=config.output_field,
            )
            export_path = dataset_export_dir
        elif bool(config.export_jsonl):
            exports_dir = workspace / "exports"
            export_path = str(exports_dir / "captions.jsonl")
            print(f"Exporting CaptionFlow captions to {export_path}")
            exported = exporter.to_jsonl(export_path)

        return {
            "message": "CaptionFlow captioning completed.",
            "dataset_id": config.dataset_id,
            "workspace_dir": str(workspace),
            "exported": exported,
            "export_path": export_path,
        }
    finally:
        _terminate_processes(processes)
        for handle in log_handles:
            handle.close()
