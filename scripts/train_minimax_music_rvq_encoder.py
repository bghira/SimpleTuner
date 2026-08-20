#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletuner.helpers.models.minimaxmusic.encoders import _SEMANTIC_VOCAB_SIZE
from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV
from simpletuner.helpers.training.checkpointing import checkpoint
from simpletuner.helpers.training.custom_schedule import get_lr_scheduler
from simpletuner.helpers.training.multi_process import broadcast_object_from_main, should_log, split_across_processes
from simpletuner.helpers.training.optimizer_param import (
    create_optimizer_params_with_decay,
    create_optimizer_with_param_groups,
    optimizer_parameters,
)
from simpletuner.helpers.training.wrappers import unwrap_model

logger = logging.getLogger("MiniMaxMusicRVQTrainer")

DEFAULT_DATASET_REPO_ID = "bghira/minimax-music3-rvq-reverse-distillation"
DEFAULT_VAE_MODEL = "SimpleTuner/MiniMax-Music-3-Encoder"
DEFAULT_CODEBOOK_VOCAB_SIZES = (_SEMANTIC_VOCAB_SIZE, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
LATENT_CHANNELS = 128
SEMANTIC_FRAMES_PER_WINDOW = 128
LATENT_RATE_NUM = 441
LATENT_RATE_DEN = 128
DAV_HOP_SAMPLES = 512
DAV_SAMPLE_RATE = 44100
CHUNK_FRAMES = 200
CHUNK_HOP_FRAMES = 100
STITCHED_HOP_LATENTS = 345
NON_FIRST_CHUNK_OWNED_FROM = 25
RVQ_CACHE_FORMAT = "simpletuner-minimaxmusic-rvq-cache-v1"
MERT_CACHE_FORMAT = "simpletuner-minimaxmusic-mert-cache-v1"
MERT_ALIGNMENT_VERSION = "dav512-mert75-center-v1"
DEFAULT_MERT_MODEL = "m-a-p/MERT-v1-95M"
DEFAULT_MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_SAMPLE_RATE = 24000
MERT_FEATURE_RATE = 75.0
MERT_HIDDEN_SIZE = 768
MUP_ENCODER_SCOPE = "rvq-encoder-v1"
MUP_MERT_SCOPE = "rvq-mert-training-wrapper-v1"
MUP_DEPTH_SCOPE = "rvq-depth-decoder-v1"
MUP_MERT_DEPTH_SCOPE = "rvq-mert-depth-training-wrapper-v1"
CHECKPOINT_FORMAT = "simpletuner-minimaxmusic-rvq-encoder-v1"
HUB_CHECKPOINT_ALLOW_PATTERNS = (
    "rvq_encoder.safetensors",
    "rvq_encoder_config.json",
    "mup_base_shapes.bsh",
    "mup_base_shapes.bsh.meta.json",
)
HUB_TRUE_VALUES = {"1", "true", "yes", "on"}
HUB_FALSE_VALUES = {"0", "false", "no", "off", "none", "null"}
EVALUATION_FORMAT = "simpletuner-minimaxmusic-rvq-evaluation-v1"
EVALUATION_SECTION_START = "<!-- simpletuner-rvq-evaluation-start -->"
EVALUATION_SECTION_END = "<!-- simpletuner-rvq-evaluation-end -->"
PUBLIC_TEXT_LOCAL_IDENTITY_PATTERNS = (
    re.compile(r"/Users/", re.IGNORECASE),
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"/private/", re.IGNORECASE),
    re.compile(r"/tmp/", re.IGNORECASE),
    re.compile(r"/var/folders/", re.IGNORECASE),
    re.compile(r"/workspace/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"co-authored-by:", re.IGNORECASE),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not should_log():
        logger.setLevel("ERROR")


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("float", "fp")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "half", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; expected fp32, fp16, or bf16.")


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if text:
        return text[:120]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class RVQTraceRecord:
    shard_id: int
    shard_path: str
    sample_id: str
    audio_file: str
    tensor_file: str
    split: str
    emitted_frames: int
    sampling_rate: int
    codebook_vocab_sizes: tuple[int, ...]
    alignment: dict[str, Any]

    @property
    def cache_stem(self) -> str:
        return f"shard-{self.shard_id:06d}-{_safe_name(self.sample_id)}"


@dataclass(frozen=True)
class RVQEncoderConfig:
    latent_channels: int = LATENT_CHANNELS
    codebook_vocab_sizes: tuple[int, ...] = DEFAULT_CODEBOOK_VOCAB_SIZES
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_position_embeddings: int = SEMANTIC_FRAMES_PER_WINDOW
    conv_dilations: tuple[int, ...] = (1, 3, 9)
    mup: bool = False
    mup_output_mult: float = 1.0
    mup_readout_zero_init: bool = False
    mup_attention_multiplier: float = 8.0
    depth_decoder: bool = False
    depth_decoder_dim: int = 512
    depth_decoder_layers: int = 2
    depth_decoder_heads: int = 8
    depth_decoder_ff_mult: int = 4
    depth_decoder_dropout: float = 0.1


def _record_from_index_entry(entry: dict[str, Any]) -> RVQTraceRecord | None:
    jobs = (entry.get("manifest") or {}).get("jobs") or []
    if len(jobs) != 1:
        raise ValueError(f"Expected exactly one job in shard {entry.get('path')!r}, got {len(jobs)}.")
    job = dict(jobs[0])
    if job.get("status") not in {None, "succeeded"}:
        return None

    split = job.get("dataset_split") or entry.get("dataset_split") or (job.get("source_metadata") or {}).get("dataset_split")
    alignment = job.get("alignment") or entry.get("alignment") or {}
    codebook_vocab_sizes = tuple(int(value) for value in job.get("codebook_vocab_sizes", DEFAULT_CODEBOOK_VOCAB_SIZES))
    if len(codebook_vocab_sizes) != len(DEFAULT_CODEBOOK_VOCAB_SIZES):
        raise ValueError(f"Expected 8 codebook vocab sizes for {job.get('id')!r}; got {codebook_vocab_sizes}.")

    required = ("id", "audio_file", "tensor_file", "emitted_frames")
    missing = [name for name in required if job.get(name) in (None, "")]
    if missing:
        raise ValueError(f"Index entry for shard {entry.get('path')!r} is missing {missing}.")

    return RVQTraceRecord(
        shard_id=int(entry["shard_id"]),
        shard_path=str(entry["path"]),
        sample_id=str(job["id"]),
        audio_file=str(job["audio_file"]),
        tensor_file=str(job["tensor_file"]),
        split=str(split or "train"),
        emitted_frames=int(job["emitted_frames"]),
        sampling_rate=int(job.get("sampling_rate") or 44100),
        codebook_vocab_sizes=codebook_vocab_sizes,
        alignment=alignment,
    )


def _iter_index_entries(index_path: Path) -> Iterable[dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {index_path} at line {line_number}.") from exc


def resolve_index_files(args: argparse.Namespace) -> list[Path]:
    if args.index_file:
        return [Path(path).expanduser() for path in args.index_file]

    if args.index_dir:
        index_dir = Path(args.index_dir).expanduser()
        return sorted(index_dir.glob("*.jsonl"))

    repo_files = sorted(
        name
        for name in list_repo_files(args.dataset_repo_id, repo_type="dataset", revision=args.dataset_revision)
        if name.startswith("indexes/") and name.endswith(".jsonl")
    )
    if args.max_index_files:
        repo_files = repo_files[: args.max_index_files]
    return [
        Path(
            hf_hub_download(
                args.dataset_repo_id,
                repo_file,
                repo_type="dataset",
                revision=args.dataset_revision,
                cache_dir=args.hf_cache_dir,
            )
        )
        for repo_file in repo_files
    ]


def load_records(args: argparse.Namespace) -> tuple[list[RVQTraceRecord], list[RVQTraceRecord]]:
    index_files = resolve_index_files(args)
    if not index_files:
        raise ValueError("No MiniMax Music RVQ index files were found.")

    train_records: list[RVQTraceRecord] = []
    validation_records: list[RVQTraceRecord] = []
    train_split = str(args.train_split).lower()
    validation_split = str(args.validation_split).lower() if args.validation_split else None

    for index_file in index_files:
        for entry in _iter_index_entries(index_file):
            record = _record_from_index_entry(entry)
            if record is None:
                continue
            split = record.split.lower()
            if split == train_split:
                if args.max_train_records <= 0 or len(train_records) < args.max_train_records:
                    train_records.append(record)
            elif validation_split and split == validation_split:
                if args.max_validation_records <= 0 or len(validation_records) < args.max_validation_records:
                    validation_records.append(record)

    if not train_records:
        raise ValueError(f"No records matched train split {args.train_split!r}.")
    if not validation_records and args.validation_fraction > 0:
        rng = random.Random(args.seed)
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * args.validation_fraction)))
        validation_records = shuffled[:validation_count]
        train_records = shuffled[validation_count:]
    return train_records, validation_records


def load_records_for_accelerator(
    args: argparse.Namespace, accelerator: Accelerator
) -> tuple[list[RVQTraceRecord], list[RVQTraceRecord]]:
    if accelerator.is_main_process:
        records = load_records(args)
    else:
        records = ([], [])
    train_records, validation_records = broadcast_object_from_main(records)
    if not train_records:
        raise ValueError(f"No records matched train split {args.train_split!r}.")
    return train_records, validation_records


def _n_legacy_dit_windows(n_frames: int) -> int:
    return max(1, (n_frames - 1) // CHUNK_HOP_FRAMES)


def legacy_frame_latent_starts(n_frames: int) -> list[int]:
    starts: list[int] = []
    num_windows = _n_legacy_dit_windows(n_frames)
    for frame_index in range(n_frames + 1):
        chunk_index = min(max((frame_index - NON_FIRST_CHUNK_OWNED_FROM) // CHUNK_HOP_FRAMES, 0), num_windows - 1)
        local_frame = frame_index - chunk_index * CHUNK_HOP_FRAMES
        chunk_frames = min(CHUNK_FRAMES, n_frames - chunk_index * CHUNK_HOP_FRAMES)
        chunk_latents = chunk_frames * LATENT_RATE_NUM // LATENT_RATE_DEN
        local_latent = (local_frame * chunk_latents + chunk_frames - 1) // chunk_frames
        starts.append(chunk_index * STITCHED_HOP_LATENTS + local_latent)
    return starts


def chunk_stitching_frame_latent_starts(n_frames: int, chunks: list[dict[str, Any]]) -> list[int]:
    if not chunks:
        raise ValueError("chunk_stitching alignment is empty.")
    ordered = sorted(chunks, key=lambda item: int(item["chunk_index"]))
    starts = [0 for _ in range(n_frames + 1)]

    for index, chunk in enumerate(ordered):
        semantic_start = int(chunk["semantic_frame_start"])
        semantic_end = min(int(chunk["semantic_frame_end_exclusive"]), n_frames)
        owner_start = semantic_start if index == 0 else min(semantic_start + NON_FIRST_CHUNK_OWNED_FROM, n_frames)
        if index + 1 < len(ordered):
            next_start = int(ordered[index + 1]["semantic_frame_start"])
            owner_end = min(next_start + NON_FIRST_CHUNK_OWNED_FROM, n_frames)
        else:
            owner_end = semantic_end
        owner_start = max(owner_start, 0)
        owner_end = max(owner_end, owner_start)
        if owner_start == owner_end:
            continue

        stitched_start = int(chunk["stitched_flow_latent_start"])
        stitched_end = int(chunk["stitched_flow_latent_end_exclusive"])
        kept_latents = stitched_end - stitched_start
        owner_frames = owner_end - owner_start
        if kept_latents <= 0:
            raise ValueError(f"Invalid chunk_stitching entry with non-positive kept latent length: {chunk}.")
        for frame_index in range(owner_start, owner_end + 1):
            local_frame = frame_index - owner_start
            starts[frame_index] = stitched_start + (local_frame * kept_latents + owner_frames - 1) // owner_frames

    for frame_index in range(1, len(starts)):
        if starts[frame_index] < starts[frame_index - 1]:
            raise ValueError("chunk_stitching produced non-monotonic frame/latent boundaries.")
    return starts


def frame_latent_starts(n_frames: int, alignment: dict[str, Any]) -> tuple[list[int], str]:
    chunks = alignment.get("chunk_stitching") if isinstance(alignment, dict) else None
    if chunks:
        return chunk_stitching_frame_latent_starts(n_frames, chunks), "chunk_stitching"
    return legacy_frame_latent_starts(n_frames), "legacy_nominal"


def build_pool_matrix(bounds: list[int]) -> torch.Tensor:
    if len(bounds) < 2:
        raise ValueError("At least two frame/latent boundaries are required.")
    origin = bounds[0]
    local_bounds = [int(value) - origin for value in bounds]
    latent_count = local_bounds[-1]
    if latent_count <= 0:
        raise ValueError(f"Invalid latent bounds {bounds[:4]}...{bounds[-4:]}.")
    pool = torch.zeros((len(local_bounds) - 1, latent_count), dtype=torch.float32)
    for frame_index, (start, end) in enumerate(zip(local_bounds[:-1], local_bounds[1:])):
        if end <= start:
            raise ValueError(f"Frame {frame_index} has invalid latent span [{start}, {end}).")
        pool[frame_index, start:end] = 1.0 / float(end - start)
    return pool


def _cache_paths(cache_dir: Path, record: RVQTraceRecord) -> tuple[Path, Path]:
    shard_dir = cache_dir / f"{record.shard_id // 1000:05d}"
    return shard_dir / f"{record.cache_stem}.safetensors", shard_dir / f"{record.cache_stem}.json"


def _mert_cache_paths(cache_dir: Path, record: RVQTraceRecord) -> tuple[Path, Path]:
    shard_dir = cache_dir / f"{record.shard_id // 1000:05d}"
    return shard_dir / f"{record.cache_stem}.mert.safetensors", shard_dir / f"{record.cache_stem}.mert.json"


def mert_cache_layers(args: argparse.Namespace) -> tuple[int, ...]:
    layers = {int(args.mert_teacher_layer)}
    layers.update(int(layer) for layer in args.mert_cache_layer)
    if min(layers) < 0:
        raise ValueError("MERT cache layers must be non-negative.")
    return tuple(sorted(layers))


def rvq_frame_center_seconds(record: RVQTraceRecord) -> torch.Tensor:
    if record.sampling_rate != DAV_SAMPLE_RATE:
        raise ValueError(
            f"RVQ frame timing requires {DAV_SAMPLE_RATE} Hz DAV audio, got {record.sampling_rate} Hz for "
            f"{record.sample_id}."
        )
    starts, _ = frame_latent_starts(record.emitted_frames, record.alignment)
    boundaries = torch.tensor(starts, dtype=torch.float64)
    return (boundaries[:-1] + boundaries[1:]) * (DAV_HOP_SAMPLES / (2.0 * DAV_SAMPLE_RATE))


def interpolate_features_at_times(
    features: torch.Tensor,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    *,
    tolerance_seconds: float,
) -> torch.Tensor:
    if features.ndim != 2 or source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("MERT interpolation expects features [frames, dim] and one-dimensional time tensors.")
    if features.shape[0] != source_times.shape[0] or source_times.numel() < 2:
        raise ValueError("MERT interpolation requires at least two timestamped source features.")
    if not torch.all(source_times[1:] > source_times[:-1]):
        raise ValueError("MERT feature timestamps must be strictly increasing.")
    lower_gap = float(source_times[0] - target_times[0])
    upper_gap = float(target_times[-1] - source_times[-1])
    if lower_gap > tolerance_seconds or upper_gap > tolerance_seconds:
        raise ValueError(
            "MERT features do not cover the RVQ frame timeline: " f"lower gap={lower_gap:.6f}s, upper gap={upper_gap:.6f}s."
        )

    target_times = target_times.clamp(min=float(source_times[0]), max=float(source_times[-1]))
    right = torch.searchsorted(source_times, target_times, right=False).clamp(1, source_times.numel() - 1)
    left = right - 1
    left_times = source_times[left]
    right_times = source_times[right]
    weight = ((target_times - left_times) / (right_times - left_times)).to(dtype=features.dtype).unsqueeze(-1)
    return features[left] + (features[right] - features[left]) * weight


def _mert_cache_metadata(args: argparse.Namespace, record: RVQTraceRecord, layers: tuple[int, ...]) -> dict[str, Any]:
    return {
        "format": MERT_CACHE_FORMAT,
        "alignment_version": MERT_ALIGNMENT_VERSION,
        "dav_hop_samples": DAV_HOP_SAMPLES,
        "dav_sample_rate": DAV_SAMPLE_RATE,
        "model_name_or_path": args.mert_model_name_or_path,
        "revision": args.mert_revision,
        "layers": list(layers),
        "sample_rate": MERT_SAMPLE_RATE,
        "feature_rate": MERT_FEATURE_RATE,
        "hidden_size": MERT_HIDDEN_SIZE,
        "chunk_seconds": float(args.mert_chunk_seconds),
        "chunk_overlap_seconds": float(args.mert_chunk_overlap_seconds),
        "cache_dtype": args.mert_cache_dtype,
        "emitted_frames": int(record.emitted_frames),
    }


def _load_mert_cache_meta(cache_dir: Path, record: RVQTraceRecord) -> dict[str, Any]:
    _, meta_path = _mert_cache_paths(cache_dir, record)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing MERT cache metadata for {record.sample_id}: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != MERT_CACHE_FORMAT:
        raise ValueError(f"Unsupported MERT cache metadata format in {meta_path}.")
    return meta


def _mert_cache_satisfies_requirements(args: argparse.Namespace, cache_dir: Path, record: RVQTraceRecord) -> bool:
    tensor_path, meta_path = _mert_cache_paths(cache_dir, record)
    if args.rebuild_mert_cache or not tensor_path.is_file() or not meta_path.is_file():
        return False
    expected = _mert_cache_metadata(args, record, mert_cache_layers(args))
    try:
        actual = _load_mert_cache_meta(cache_dir, record)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    expected_layers = set(expected.pop("layers"))
    actual_layers = set(actual.get("layers", []))
    return expected_layers.issubset(actual_layers) and all(actual.get(key) == value for key, value in expected.items())


def _resolve_shard_path(args: argparse.Namespace, record: RVQTraceRecord) -> Path:
    shard_path = Path(record.shard_path)
    if shard_path.is_absolute() and shard_path.is_file():
        return shard_path
    if args.corpus_dir:
        local = Path(args.corpus_dir).expanduser() / record.shard_path
        if local.is_file():
            return local
    return Path(
        hf_hub_download(
            args.dataset_repo_id,
            record.shard_path,
            repo_type="dataset",
            revision=args.dataset_revision,
            cache_dir=args.hf_cache_dir,
        )
    )


def _load_dav_from_path(
    checkpoint_path: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    device: torch.device,
) -> MiniMaxMusic3DAV:
    def _has_diffusers_audio_vae(path: str) -> bool:
        if os.path.isfile(path):
            return False
        if os.path.isdir(path):
            checkpoint = Path(path)
            return (checkpoint / "config.json").is_file() or (checkpoint / "audio_vae" / "config.json").is_file()
        try:
            hf_hub_download(path, "audio_vae/config.json", repo_type="model", revision=revision, cache_dir=cache_dir)
            return True
        except (EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return False

    def _resolve_dav_checkpoint(path: str) -> str | None:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            dav_path = Path(path) / "dav.pth"
            return str(dav_path) if dav_path.is_file() else None
        try:
            return hf_hub_download(path, "dav.pth", repo_type="model", revision=revision, cache_dir=cache_dir)
        except (EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return None

    if _has_diffusers_audio_vae(checkpoint_path):
        if os.path.isdir(checkpoint_path) and (Path(checkpoint_path) / "config.json").is_file():
            vae = MiniMaxMusic3DAV.from_pretrained(checkpoint_path, torch_dtype=torch.float32, cache_dir=cache_dir)
        else:
            vae = MiniMaxMusic3DAV.from_pretrained(
                checkpoint_path,
                subfolder="audio_vae",
                torch_dtype=torch.float32,
                revision=revision,
                cache_dir=cache_dir,
            )
    elif (dav_checkpoint := _resolve_dav_checkpoint(checkpoint_path)) is not None:
        vae = MiniMaxMusic3DAV.from_original_dav(dav_checkpoint)
    else:
        raise RuntimeError(
            "MiniMax Music RVQ encoder training requires a DAV checkpoint with encode() support. "
            "Use SimpleTuner/MiniMax-Music-3-Encoder, MiniMaxAI/MiniMax-Music3, or a local dav.pth."
        )
    vae.eval()
    vae.requires_grad_(False)
    return vae.to(device=device, dtype=torch.float32)


def _read_zip_safetensors(
    zip_file: zipfile.ZipFile,
    tensor_member: str,
    *,
    required_names: tuple[str, ...],
    optional_names: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    with zip_file.open(tensor_member) as source, tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        shutil.copyfileobj(source, handle)
        handle.flush()
        with safe_open(handle.name, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            missing = [name for name in required_names if name not in available]
            if missing:
                raise KeyError(f"{tensor_member} does not contain required tensor(s): {missing}.")
            output = {name: tensors.get_tensor(name) for name in required_names}
            output.update({name: tensors.get_tensor(name) for name in optional_names if name in available})
            return output


def _decode_audio_member(zip_file: zipfile.ZipFile, audio_member: str, target_sample_rate: int) -> torch.Tensor:
    import soundfile as sf

    audio_bytes = zip_file.read(audio_member)
    waveform_np, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_np.T).contiguous()
    if int(sample_rate) != int(target_sample_rate):
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, int(sample_rate), int(target_sample_rate)).contiguous()
    return waveform


def _mert_chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    if chunk_samples <= 0:
        raise ValueError("MERT chunk length must be positive.")
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("MERT chunk overlap must be non-negative and shorter than the chunk.")
    if total_samples <= chunk_samples:
        return [0]
    hop = chunk_samples - overlap_samples
    starts = list(range(0, total_samples - chunk_samples + 1, hop))
    final_start = total_samples - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def extract_mert_features(
    waveform: torch.Tensor,
    *,
    processor: Any,
    model: nn.Module,
    layers: tuple[int, ...],
    chunk_seconds: float,
    overlap_seconds: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    if waveform.ndim != 2:
        raise ValueError(f"MERT waveform must have shape [channels, samples], got {tuple(waveform.shape)}.")
    if batch_size <= 0:
        raise ValueError("mert_cache_batch_size must be positive.")
    waveform = waveform.mean(dim=0).float().cpu()
    chunk_samples = round(chunk_seconds * MERT_SAMPLE_RATE)
    overlap_samples = round(overlap_seconds * MERT_SAMPLE_RATE)
    starts = _mert_chunk_starts(waveform.numel(), chunk_samples, overlap_samples)
    ends = [min(start + chunk_samples, waveform.numel()) for start in starts]
    boundaries = [(ends[index] + starts[index + 1]) / (2.0 * MERT_SAMPLE_RATE) for index in range(len(starts) - 1)]
    layer_features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    feature_times: list[torch.Tensor] = []

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        chunks = [waveform[start : min(start + chunk_samples, waveform.numel())].numpy() for start in batch_starts]
        inputs = processor(
            chunks,
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(device=device, dtype=torch.float32)
        attention_mask = inputs["attention_mask"].to(device=device)
        outputs = model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("MERT did not return hidden states with output_hidden_states=True.")
        if max(layers) >= len(hidden_states):
            raise ValueError(f"MERT exposes {len(hidden_states)} hidden states; requested layer {max(layers)}.")
        length_fn = getattr(model, "_get_feat_extract_output_lengths", None)
        if length_fn is None:
            raise RuntimeError("MERT model does not expose _get_feat_extract_output_lengths for exact chunk trimming.")
        feature_lengths = length_fn(attention_mask.sum(dim=-1)).to(device="cpu", dtype=torch.long)

        for local_index, absolute_start in enumerate(batch_starts):
            chunk_index = batch_start + local_index
            feature_count = int(feature_lengths[local_index].item())
            times = (
                absolute_start / MERT_SAMPLE_RATE
                + (torch.arange(feature_count, dtype=torch.float64) + 0.5) / MERT_FEATURE_RATE
            )
            keep = torch.ones(feature_count, dtype=torch.bool)
            if chunk_index > 0:
                keep &= times > boundaries[chunk_index - 1]
            if chunk_index < len(starts) - 1:
                keep &= times <= boundaries[chunk_index]
            feature_times.append(times[keep])
            for layer in layers:
                values = hidden_states[layer][local_index, :feature_count].detach().float().cpu()
                layer_features[layer].append(values[keep])

    times = torch.cat(feature_times)
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("MERT chunk trimming produced non-monotonic feature timestamps.")
    return {layer: torch.cat(values) for layer, values in layer_features.items()}, times


@torch.no_grad()
def _encode_record_to_mert_cache(
    args: argparse.Namespace,
    record: RVQTraceRecord,
    *,
    processor: Any,
    model: nn.Module,
    cache_dir: Path,
    device: torch.device,
) -> None:
    tensor_path, meta_path = _mert_cache_paths(cache_dir, record)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if _mert_cache_satisfies_requirements(args, cache_dir, record):
        return

    shard_path = _resolve_shard_path(args, record)
    with zipfile.ZipFile(shard_path) as archive:
        waveform = _decode_audio_member(archive, record.audio_file, MERT_SAMPLE_RATE)
    layers = mert_cache_layers(args)
    extracted, source_times = extract_mert_features(
        waveform,
        processor=processor,
        model=model,
        layers=layers,
        chunk_seconds=args.mert_chunk_seconds,
        overlap_seconds=args.mert_chunk_overlap_seconds,
        batch_size=args.mert_cache_batch_size,
        device=device,
    )
    target_times = rvq_frame_center_seconds(record)
    cache_dtype = dtype_from_name(args.mert_cache_dtype)
    aligned = {
        f"mert_layer_{layer}": interpolate_features_at_times(
            extracted[layer],
            source_times,
            target_times,
            tolerance_seconds=1.0 / MERT_FEATURE_RATE,
        )
        .to(dtype=cache_dtype)
        .contiguous()
        for layer in layers
    }
    for name, tensor in aligned.items():
        if tensor.shape != (record.emitted_frames, MERT_HIDDEN_SIZE):
            raise ValueError(f"{name} has unexpected aligned shape {tuple(tensor.shape)}.")

    tmp_tensor = tensor_path.with_suffix(".safetensors.tmp")
    save_safetensors_file(aligned, str(tmp_tensor))
    os.replace(tmp_tensor, tensor_path)
    meta = _mert_cache_metadata(args, record, layers)
    tmp_meta = meta_path.with_suffix(".json.tmp")
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_meta, meta_path)


@torch.no_grad()
def _encode_record_to_cache(
    args: argparse.Namespace,
    record: RVQTraceRecord,
    *,
    vae: MiniMaxMusic3DAV,
    cache_dir: Path,
    device: torch.device,
) -> None:
    tensor_path, meta_path = _cache_paths(cache_dir, record)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if _cache_satisfies_requirements(args, cache_dir, record):
        return

    shard_path = _resolve_shard_path(args, record)
    with zipfile.ZipFile(shard_path) as archive:
        source_tensors = _read_zip_safetensors(
            archive,
            record.tensor_file,
            required_names=("codes",),
            optional_names=("teacher_topk_ids", "teacher_topk_logits"),
        )
        codes = source_tensors["codes"].to(dtype=torch.int16).contiguous()
        waveform = _decode_audio_member(archive, record.audio_file, int(vae.config.sampling_rate))

    if codes.ndim != 2 or codes.shape[1] != len(record.codebook_vocab_sizes):
        raise ValueError(f"{record.tensor_file} codes must have shape [frames, 8], got {tuple(codes.shape)}.")

    teacher_topk_ids = source_tensors.get("teacher_topk_ids")
    teacher_topk_logits = source_tensors.get("teacher_topk_logits")
    has_teacher_topk = teacher_topk_ids is not None and teacher_topk_logits is not None
    if (teacher_topk_ids is None) != (teacher_topk_logits is None):
        raise ValueError(f"{record.tensor_file} must contain both teacher_topk_ids and teacher_topk_logits, or neither.")
    if args.teacher_kl_weight > 0 and not has_teacher_topk:
        raise ValueError(
            f"{record.tensor_file} is missing teacher_topk_ids/teacher_topk_logits required by "
            f"--teacher_kl_weight {args.teacher_kl_weight}. Set --teacher_kl_weight 0 or regenerate the corpus with "
            "teacher distributions."
        )
    if has_teacher_topk:
        expected_prefix = (codes.shape[0], codes.shape[1])
        if teacher_topk_ids.ndim != 3 or tuple(teacher_topk_ids.shape[:2]) != expected_prefix:
            raise ValueError(
                f"{record.tensor_file} teacher_topk_ids must have shape [code_rows, 8, topk], "
                f"got {tuple(teacher_topk_ids.shape)}."
            )
        if teacher_topk_logits.ndim != 3 or tuple(teacher_topk_logits.shape) != tuple(teacher_topk_ids.shape):
            raise ValueError(
                f"{record.tensor_file} teacher_topk_logits must match teacher_topk_ids shape "
                f"{tuple(teacher_topk_ids.shape)}, got {tuple(teacher_topk_logits.shape)}."
            )
        teacher_topk_ids = teacher_topk_ids.to(dtype=torch.int32).contiguous()
        teacher_topk_logits = teacher_topk_logits.contiguous()

    waveform = waveform.to(device=device, dtype=torch.float32).unsqueeze(0)
    latents = vae.encode(waveform).squeeze(0).transpose(0, 1).detach().cpu().contiguous()
    latents = latents.to(dtype=dtype_from_name(args.cache_latent_dtype))
    del waveform

    tmp_tensor = tensor_path.with_suffix(".safetensors.tmp")
    cache_tensors = {"latents": latents, "codes": codes}
    if has_teacher_topk:
        cache_tensors["teacher_topk_ids"] = teacher_topk_ids
        cache_tensors["teacher_topk_logits"] = teacher_topk_logits
    save_safetensors_file(cache_tensors, str(tmp_tensor))
    os.replace(tmp_tensor, tensor_path)

    starts, alignment_source = frame_latent_starts(record.emitted_frames, record.alignment)
    meta = {
        "format": RVQ_CACHE_FORMAT,
        "record": asdict(record),
        "latent_frames": int(latents.shape[0]),
        "latent_channels": int(latents.shape[1]),
        "code_frames": int(codes.shape[0]),
        "has_teacher_topk": bool(has_teacher_topk),
        "teacher_topk_k": int(teacher_topk_ids.shape[-1]) if has_teacher_topk else 0,
        "alignment_source": alignment_source,
        "mapped_latent_frames": int(starts[-1]),
        "cache_latent_dtype": str(latents.dtype).removeprefix("torch."),
    }
    tmp_meta = meta_path.with_suffix(".json.tmp")
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_meta, meta_path)


def ensure_latent_cache(
    args: argparse.Namespace,
    records: list[RVQTraceRecord],
    *,
    accelerator: Accelerator,
) -> None:
    cache_dir = Path(args.latent_cache_dir).expanduser()
    missing = [record for record in records if not _cache_satisfies_requirements(args, cache_dir, record)]
    if not missing:
        wait_for_all_processes(accelerator)
        return

    local_records = split_across_processes(accelerator, missing)
    if not local_records:
        wait_for_all_processes(accelerator)
        return
    vae = _load_dav_from_path(
        args.pretrained_vae_model_name_or_path,
        revision=args.vae_revision,
        cache_dir=args.hf_cache_dir,
        device=accelerator.device,
    )
    iterator = tqdm(
        local_records,
        desc="Caching DAV latents",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    for record in iterator:
        iterator.set_postfix_str(record.cache_stem[:32])
        _encode_record_to_cache(args, record, vae=vae, cache_dir=cache_dir, device=accelerator.device)
    del vae
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
    wait_for_all_processes(accelerator)


def ensure_mert_cache(
    args: argparse.Namespace,
    records: list[RVQTraceRecord],
    *,
    accelerator: Accelerator,
) -> None:
    if args.mert_alignment_weight <= 0:
        return
    cache_dir = Path(args.mert_cache_dir).expanduser()
    missing = [record for record in records if not _mert_cache_satisfies_requirements(args, cache_dir, record)]
    if not missing:
        wait_for_all_processes(accelerator)
        return

    from transformers import AutoFeatureExtractor, AutoModel

    local_records = split_across_processes(accelerator, missing)
    if not local_records:
        wait_for_all_processes(accelerator)
        return
    processor = AutoFeatureExtractor.from_pretrained(
        args.mert_model_name_or_path,
        revision=args.mert_revision,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.mert_model_name_or_path,
        revision=args.mert_revision,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval().requires_grad_(False)
    model.to(device=accelerator.device, dtype=torch.float32)
    iterator = tqdm(
        local_records,
        desc="Caching MERT features",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    for record in iterator:
        iterator.set_postfix_str(record.cache_stem[:32])
        _encode_record_to_mert_cache(
            args,
            record,
            processor=processor,
            model=model,
            cache_dir=cache_dir,
            device=accelerator.device,
        )
    del model
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
    wait_for_all_processes(accelerator)


def _load_cache_meta(cache_dir: Path, record: RVQTraceRecord) -> dict[str, Any]:
    _, meta_path = _cache_paths(cache_dir, record)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing RVQ cache metadata for {record.sample_id}: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != RVQ_CACHE_FORMAT:
        raise ValueError(f"Unsupported RVQ cache metadata format in {meta_path}.")
    return meta


def _cache_satisfies_requirements(args: argparse.Namespace, cache_dir: Path, record: RVQTraceRecord) -> bool:
    tensor_path, meta_path = _cache_paths(cache_dir, record)
    if args.rebuild_latent_cache or not tensor_path.is_file() or not meta_path.is_file():
        return False
    if args.teacher_kl_weight > 0:
        meta = _load_cache_meta(cache_dir, record)
        if not meta.get("has_teacher_topk", False):
            return False
    return True


def wait_for_all_processes(accelerator: Accelerator) -> None:
    if dist.is_available() and dist.is_initialized() and torch.backends.mps.is_available() and not torch.cuda.is_available():
        dist.barrier()
        return
    accelerator.wait_for_everyone()


def _normalize_hub_model_id(repo_id: str | None) -> str | None:
    if repo_id is None:
        return None
    cleaned = str(repo_id).strip()
    if not cleaned or cleaned.lower() in HUB_FALSE_VALUES:
        return None
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc:
        cleaned = parsed.path
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0].strip("/")
    if "huggingface.co/" in cleaned:
        cleaned = cleaned.split("huggingface.co/", 1)[1]
    if cleaned.startswith("huggingface.co"):
        cleaned = cleaned[len("huggingface.co") :].strip("/")
    for marker in ("/tree/", "/blob/", "/resolve/"):
        cleaned = cleaned.split(marker, 1)[0]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned or None


def resolve_hub_model_id(args: argparse.Namespace) -> str | None:
    push_value = getattr(args, "push_to_hub", False)
    if isinstance(push_value, str):
        cleaned = push_value.strip()
        normalized = cleaned.lower()
        if not cleaned or normalized in HUB_FALSE_VALUES:
            return None
        if normalized not in HUB_TRUE_VALUES:
            return _normalize_hub_model_id(cleaned)
        push_enabled = True
    else:
        push_enabled = bool(push_value)

    if not push_enabled:
        return None
    repo_id = _normalize_hub_model_id(getattr(args, "hub_model_id", None))
    if not repo_id:
        raise ValueError("--push_to_hub requires a repo id via --push_to_hub owner/repo or --hub_model_id owner/repo.")
    return repo_id


def checkpoint_path_in_repo(checkpoint_dir: Path, output_root: Path) -> str:
    try:
        relative_path = checkpoint_dir.resolve().relative_to(output_root.resolve())
    except ValueError:
        relative_path = Path(checkpoint_dir.name)
    return relative_path.as_posix()


@dataclass
class HubCheckpointUploader:
    accelerator: Accelerator
    api: HfApi | None
    repo_id: str
    output_root: Path

    def upload_checkpoint(self, checkpoint_dir: Path, *, global_step: int, final: bool) -> None:
        upload_error = None
        if self.accelerator.is_main_process:
            if self.api is None:
                upload_error = "Hub uploader was not initialized on the main process."
            else:
                try:
                    path_in_repo = checkpoint_path_in_repo(checkpoint_dir, self.output_root)
                    commit_message = (
                        "Add RVQ encoder final checkpoint" if final else f"Add RVQ encoder checkpoint {global_step}"
                    )
                    self.api.upload_folder(
                        repo_id=self.repo_id,
                        repo_type="model",
                        folder_path=str(checkpoint_dir),
                        path_in_repo=path_in_repo,
                        commit_message=commit_message,
                        allow_patterns=list(HUB_CHECKPOINT_ALLOW_PATTERNS),
                    )
                    logger.info(
                        "Uploaded RVQ checkpoint %s to https://huggingface.co/%s/tree/main/%s.",
                        checkpoint_dir.name,
                        self.repo_id,
                        path_in_repo,
                    )
                except Exception as exc:
                    upload_error = f"{type(exc).__name__}: {exc}"
        upload_error = broadcast_object_from_main(upload_error)
        if upload_error:
            raise RuntimeError(f"Failed to upload RVQ checkpoint to Hugging Face Hub: {upload_error}")


def prepare_hub_checkpoint_uploader(
    args: argparse.Namespace,
    accelerator: Accelerator,
    output_root: Path,
) -> HubCheckpointUploader | None:
    repo_id = resolve_hub_model_id(args)
    if repo_id is None:
        return None

    api = None
    create_error = None
    if accelerator.is_main_process:
        try:
            api = HfApi()
            repo_url = api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=bool(getattr(args, "model_card_private", False)),
            )
            repo_id = getattr(repo_url, "repo_id", repo_id)
            logger.info("RVQ checkpoint Hub uploads enabled for https://huggingface.co/%s.", repo_id)
        except Exception as exc:
            create_error = f"{type(exc).__name__}: {exc}"
    repo_id, create_error = broadcast_object_from_main((repo_id, create_error))
    if create_error:
        raise RuntimeError(f"Failed to prepare Hugging Face Hub repository: {create_error}")
    return HubCheckpointUploader(
        accelerator=accelerator,
        api=api,
        repo_id=repo_id,
        output_root=output_root,
    )


@dataclass(frozen=True)
class RVQWindow:
    record_index: int
    frame_start: int
    usable_frames: int


class RVQWindowDataset(Dataset):
    def __init__(
        self,
        records: list[RVQTraceRecord],
        *,
        cache_dir: str | Path,
        window_frames: int,
        window_stride: int,
        random_crop: bool,
        require_exact_alignment: bool,
        require_teacher_topk: bool = False,
        mert_cache_dir: str | Path | None = None,
        mert_teacher_layer: int = 9,
        require_mert_features: bool = False,
    ):
        self.records = records
        self.cache_dir = Path(cache_dir).expanduser()
        self.window_frames = int(window_frames)
        self.window_stride = int(window_stride)
        self.random_crop = bool(random_crop)
        self.require_teacher_topk = bool(require_teacher_topk)
        self.mert_cache_dir = Path(mert_cache_dir).expanduser() if mert_cache_dir is not None else None
        self.mert_teacher_layer = int(mert_teacher_layer)
        self.require_mert_features = bool(require_mert_features)
        self.windows: list[RVQWindow] = []

        if self.window_frames <= 0:
            raise ValueError("window_frames must be positive.")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if self.require_mert_features and self.mert_cache_dir is None:
            raise ValueError("mert_cache_dir is required when MERT features are enabled.")

        for record_index, record in enumerate(records):
            meta = _load_cache_meta(self.cache_dir, record)
            if require_exact_alignment and meta.get("alignment_source") != "chunk_stitching":
                continue
            if self.require_teacher_topk and not meta.get("has_teacher_topk", False):
                raise ValueError(
                    f"RVQ cache for {record.sample_id} is missing teacher_topk_ids/teacher_topk_logits. "
                    "Rebuild the latent cache or set --teacher_kl_weight 0 for CE-only training."
                )
            if self.require_mert_features:
                mert_meta = _load_mert_cache_meta(self.mert_cache_dir, record)
                if self.mert_teacher_layer not in mert_meta.get("layers", []):
                    raise ValueError(f"MERT cache for {record.sample_id} does not contain layer {self.mert_teacher_layer}.")
                if int(mert_meta.get("emitted_frames", 0)) < int(record.emitted_frames):
                    raise ValueError(
                        f"MERT cache for {record.sample_id} is truncated: "
                        f"{mert_meta.get('emitted_frames', 0)} < {record.emitted_frames}."
                    )
            code_offset = int(record.alignment.get("emitted_code_row_offset", 1) if record.alignment else 1)
            code_frames = int(meta["code_frames"])
            latent_frames = int(meta["latent_frames"])
            n_frames = min(int(record.emitted_frames), code_frames - code_offset)
            starts, _ = frame_latent_starts(n_frames, record.alignment)
            while n_frames > 0 and starts[n_frames] > latent_frames:
                n_frames -= 1
            if n_frames < self.window_frames:
                continue
            for frame_start in range(0, n_frames - self.window_frames + 1, self.window_stride):
                self.windows.append(RVQWindow(record_index, frame_start, n_frames))

        if not self.windows:
            raise ValueError("No trainable RVQ windows were found after applying cache/alignment constraints.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        if self.random_crop:
            frame_start = random.randint(0, window.usable_frames - self.window_frames)
        else:
            frame_start = window.frame_start
        record = self.records[window.record_index]
        tensor_path, _ = _cache_paths(self.cache_dir, record)
        code_offset = int(record.alignment.get("emitted_code_row_offset", 1) if record.alignment else 1)

        starts, _ = frame_latent_starts(window.usable_frames, record.alignment)
        bounds = starts[frame_start : frame_start + self.window_frames + 1]
        latent_start = int(bounds[0])
        latent_end = int(bounds[-1])
        target_start = frame_start + code_offset
        target_end = target_start + self.window_frames
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            latent_shape = tensors.get_slice("latents").get_shape()
            if latent_end > latent_shape[0]:
                raise ValueError(
                    f"Window {record.sample_id}:{frame_start} maps past cached latents "
                    f"({latent_end} > {latent_shape[0]})."
                )
            latents = tensors.get_slice("latents")[latent_start:latent_end].to(dtype=torch.float32)
            target = tensors.get_slice("codes")[target_start:target_end].to(dtype=torch.long)
            has_teacher_topk = "teacher_topk_ids" in keys and "teacher_topk_logits" in keys
            if self.require_teacher_topk and not has_teacher_topk:
                raise ValueError(
                    f"RVQ cache for {record.sample_id} is missing teacher_topk_ids/teacher_topk_logits. "
                    "Rebuild the latent cache or set --teacher_kl_weight 0 for CE-only training."
                )
            teacher_topk_ids = None
            teacher_topk_logits = None
            if has_teacher_topk:
                teacher_topk_ids = tensors.get_slice("teacher_topk_ids")[target_start:target_end].to(dtype=torch.long)
                teacher_topk_logits = tensors.get_slice("teacher_topk_logits")[target_start:target_end].to(
                    dtype=torch.float32
                )
        if target.shape != (self.window_frames, len(record.codebook_vocab_sizes)):
            raise ValueError(f"Target window has unexpected shape {tuple(target.shape)}.")
        sample = {
            "latents": latents,
            "pool": build_pool_matrix(bounds),
            "target": target,
        }
        if teacher_topk_ids is not None and teacher_topk_logits is not None:
            if teacher_topk_ids.shape[:2] != target.shape or teacher_topk_logits.shape != teacher_topk_ids.shape:
                raise ValueError(
                    f"Teacher top-k window for {record.sample_id}:{frame_start} has inconsistent shape: "
                    f"ids={tuple(teacher_topk_ids.shape)}, logits={tuple(teacher_topk_logits.shape)}, "
                    f"target={tuple(target.shape)}."
                )
            sample["teacher_topk_ids"] = teacher_topk_ids
            sample["teacher_topk_logits"] = teacher_topk_logits
        if self.require_mert_features:
            mert_tensor_path, _ = _mert_cache_paths(self.mert_cache_dir, record)
            with safe_open(str(mert_tensor_path), framework="pt", device="cpu") as tensors:
                key = f"mert_layer_{self.mert_teacher_layer}"
                shape = tensors.get_slice(key).get_shape()
                feature_end = frame_start + self.window_frames
                if feature_end > shape[0]:
                    raise ValueError(
                        f"MERT sidecar for {record.sample_id} does not cover window {frame_start}:{feature_end}."
                    )
                mert_features = tensors.get_slice(key)[frame_start:feature_end].to(dtype=torch.float32)
            if mert_features.shape != (self.window_frames, MERT_HIDDEN_SIZE):
                raise ValueError(f"MERT window has unexpected shape {tuple(mert_features.shape)}.")
            sample["mert_features"] = mert_features
        return sample


def collate_rvq_windows(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_latents = max(int(sample["latents"].shape[0]) for sample in samples)
    batch_size = len(samples)
    latent_channels = int(samples[0]["latents"].shape[1])
    window_frames = int(samples[0]["target"].shape[0])
    num_codebooks = int(samples[0]["target"].shape[1])
    teacher_presence = [("teacher_topk_ids" in sample, "teacher_topk_logits" in sample) for sample in samples]
    if any(has_ids != has_logits for has_ids, has_logits in teacher_presence):
        raise ValueError("RVQ samples must include both teacher_topk_ids and teacher_topk_logits, or neither.")
    has_teacher_topk = [has_ids and has_logits for has_ids, has_logits in teacher_presence]
    if any(has_teacher_topk) and not all(has_teacher_topk):
        raise ValueError("Cannot collate a mixed RVQ batch with teacher top-k tensors present for only some samples.")
    mert_presence = ["mert_features" in sample for sample in samples]
    if any(mert_presence) and not all(mert_presence):
        raise ValueError("Cannot collate a mixed RVQ batch with MERT features present for only some samples.")
    latents = torch.zeros((batch_size, max_latents, latent_channels), dtype=torch.float32)
    pool = torch.zeros((batch_size, window_frames, max_latents), dtype=torch.float32)
    target = torch.empty((batch_size, window_frames, num_codebooks), dtype=torch.long)
    teacher_topk_ids = None
    teacher_topk_logits = None
    if all(has_teacher_topk):
        topk = int(samples[0]["teacher_topk_ids"].shape[-1])
        teacher_topk_ids = torch.empty((batch_size, window_frames, num_codebooks, topk), dtype=torch.long)
        teacher_topk_logits = torch.empty((batch_size, window_frames, num_codebooks, topk), dtype=torch.float32)
    for index, sample in enumerate(samples):
        latent_count = sample["latents"].shape[0]
        latents[index, :latent_count] = sample["latents"]
        pool[index, :, :latent_count] = sample["pool"]
        target[index] = sample["target"]
        if teacher_topk_ids is not None and teacher_topk_logits is not None:
            teacher_topk_ids[index] = sample["teacher_topk_ids"]
            teacher_topk_logits[index] = sample["teacher_topk_logits"]
    batch = {"latents": latents, "pool": pool, "target": target}
    if teacher_topk_ids is not None and teacher_topk_logits is not None:
        batch["teacher_topk_ids"] = teacher_topk_ids
        batch["teacher_topk_logits"] = teacher_topk_logits
    if all(mert_presence):
        batch["mert_features"] = torch.stack([sample["mert_features"] for sample in samples])
    return batch


class RVQResBlock(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.gelu(self.norm(hidden_states)))
        return hidden_states + self.conv2(F.gelu(residual))


def _load_mup_package() -> SimpleNamespace:
    try:
        import mup
    except ImportError as exc:
        raise ImportError(
            "MiniMax Music RVQ encoder --mup requires the microsoft/mup package. Install it with `pip install mup`."
        ) from exc

    required = ("MuReadout", "MuAdam", "MuAdamW", "MuSGD", "set_base_shapes", "save_base_shapes")
    missing = [name for name in required if not hasattr(mup, name)]
    if missing:
        raise ImportError(f"The installed mup package is missing required API(s): {missing}.")
    return SimpleNamespace(
        module=mup,
        MuReadout=mup.MuReadout,
        MuAdam=mup.MuAdam,
        MuAdamW=mup.MuAdamW,
        MuSGD=mup.MuSGD,
        set_base_shapes=mup.set_base_shapes,
        save_base_shapes=mup.save_base_shapes,
    )


class RVQMuTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        attention_multiplier: float,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = int(d_model // num_heads)
        self.attention_multiplier = float(attention_multiplier)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_model * ff_mult)
        self.linear2 = nn.Linear(d_model * ff_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = hidden_states.shape
        return hidden_states.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(batch, frames, self.num_heads * self.head_dim)

    def _attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        scale = self.attention_multiplier / float(self.head_dim)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attention_probs = F.softmax(attention_scores.float(), dim=-1).to(query.dtype)
        attention_probs = self.attn_dropout(attention_probs)
        hidden_states = torch.matmul(attention_probs, value)
        return self.out_proj(self._merge_heads(hidden_states))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self._attention(self.norm1(hidden_states)))
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(hidden_states)))))
        return hidden_states + self.dropout(feedforward)


class RVQDepthDecoderLayer(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"depth_decoder_dim={dim} must be divisible by depth_decoder_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, dim * ff_mult)
        self.linear2 = nn.Linear(dim * ff_mult, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, depth, dim = hidden_states.shape
        return hidden_states.view(batch, depth, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, _, depth, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(batch, depth, self.num_heads * self.head_dim)

    def _attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        depth = hidden_states.shape[1]
        causal_mask = torch.ones((depth, depth), dtype=torch.bool, device=hidden_states.device).triu(1)
        attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)
        attention_probs = F.softmax(attention_scores.float(), dim=-1).to(query.dtype)
        attention_probs = self.attn_dropout(attention_probs)
        return self.out_proj(self._merge_heads(torch.matmul(attention_probs, value)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self._attention(self.norm1(hidden_states)))
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(hidden_states)))))
        return hidden_states + self.dropout(feedforward)


class MiniMaxMusicRVQDepthDecoder(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        if len(config.codebook_vocab_sizes) < 2:
            raise ValueError("The RVQ depth decoder requires one semantic and at least one acoustic codebook.")
        self.config = config
        if config.mup:
            MuReadout = _load_mup_package().MuReadout
            self.context_projection = MuReadout(
                config.d_model,
                config.depth_decoder_dim,
                bias=False,
                readout_zero_init=False,
                output_mult=config.mup_output_mult,
            )
        else:
            self.context_projection = nn.Linear(config.d_model, config.depth_decoder_dim, bias=False)
        self.prior_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, config.depth_decoder_dim) for vocab_size in config.codebook_vocab_sizes[:-1]
        )
        self.position = nn.Parameter(torch.zeros(1, len(config.codebook_vocab_sizes), config.depth_decoder_dim))
        self.layers = nn.ModuleList(
            RVQDepthDecoderLayer(
                dim=config.depth_decoder_dim,
                num_heads=config.depth_decoder_heads,
                ff_mult=config.depth_decoder_ff_mult,
                dropout=config.depth_decoder_dropout,
            )
            for _ in range(config.depth_decoder_layers)
        )
        self.norm = nn.LayerNorm(config.depth_decoder_dim)
        self.heads = nn.ModuleList(
            nn.Linear(config.depth_decoder_dim, vocab_size) for vocab_size in config.codebook_vocab_sizes[1:]
        )
        nn.init.normal_(self.position, std=0.02)

    def _embed_prior(self, codebook_index: int, codes: torch.Tensor) -> torch.Tensor:
        vocab_size = int(self.config.codebook_vocab_sizes[codebook_index])
        invalid = (codes < 0) & (codes != -100) | (codes >= vocab_size)
        if invalid.any():
            raise ValueError(f"Depth decoder prior codebook {codebook_index} contains an out-of-range token.")
        return self.prior_embeddings[codebook_index](codes.masked_fill(codes == -100, 0))

    def _decode(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden_states = sequence + self.position[:, : sequence.shape[1]].to(sequence.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    def forward(
        self,
        frame_context: torch.Tensor,
        semantic_codes: torch.Tensor,
        *,
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if frame_context.ndim != 3 or semantic_codes.shape != frame_context.shape[:2]:
            raise ValueError("Depth decoder expects frame context [batch, frames, dim] and semantic codes [batch, frames].")
        batch, frames, _ = frame_context.shape
        context = self.context_projection(frame_context).flatten(0, 1).unsqueeze(1)
        semantic = self._embed_prior(0, semantic_codes).flatten(0, 1).unsqueeze(1)

        if teacher_forcing_targets is not None:
            expected_shape = (batch, frames, len(self.config.codebook_vocab_sizes))
            if teacher_forcing_targets.shape != expected_shape:
                raise ValueError(
                    f"teacher_forcing_targets must have shape {expected_shape}, got {tuple(teacher_forcing_targets.shape)}."
                )
            priors = [semantic]
            for codebook_index in range(1, len(self.config.codebook_vocab_sizes) - 1):
                prior = self._embed_prior(codebook_index, teacher_forcing_targets[:, :, codebook_index])
                priors.append(prior.flatten(0, 1).unsqueeze(1))
            hidden_states = self._decode(torch.cat([context, *priors], dim=1))
            return [
                head(hidden_states[:, acoustic_index + 1]).view(batch, frames, -1)
                for acoustic_index, head in enumerate(self.heads)
            ]

        sequence = torch.cat([context, semantic], dim=1)
        logits = []
        for acoustic_index, head in enumerate(self.heads):
            hidden_states = self._decode(sequence)
            head_logits = head(hidden_states[:, -1]).view(batch, frames, -1)
            logits.append(head_logits)
            if acoustic_index + 1 < len(self.heads):
                selected = head_logits.argmax(dim=-1)
                prior = self._embed_prior(acoustic_index + 1, selected).flatten(0, 1).unsqueeze(1)
                sequence = torch.cat([sequence, prior], dim=1)
        return logits


class MiniMaxMusicRVQEncoder(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.conv_in = nn.Conv1d(config.latent_channels, config.d_model, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(RVQResBlock(config.d_model, dilation) for dilation in config.conv_dilations)
        self.position = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))
        if config.mup:
            self.transformer = nn.ModuleList(
                [
                    RVQMuTransformerEncoderLayer(
                        d_model=config.d_model,
                        num_heads=config.num_heads,
                        ff_mult=config.ff_mult,
                        dropout=config.dropout,
                        attention_multiplier=config.mup_attention_multiplier,
                    )
                    for _ in range(config.num_layers)
                ]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * config.ff_mult,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, config.num_layers)
        self.norm_out = nn.LayerNorm(config.d_model)
        readout_vocab_sizes = config.codebook_vocab_sizes[:1] if config.depth_decoder else config.codebook_vocab_sizes
        if config.mup:
            MuReadout = _load_mup_package().MuReadout
            readouts = nn.ModuleList(
                MuReadout(
                    config.d_model,
                    vocab_size,
                    readout_zero_init=config.mup_readout_zero_init,
                    output_mult=config.mup_output_mult,
                )
                for vocab_size in readout_vocab_sizes
            )
        else:
            readouts = nn.ModuleList(nn.Linear(config.d_model, vocab_size) for vocab_size in readout_vocab_sizes)
        self.heads = readouts
        self.depth_decoder = MiniMaxMusicRVQDepthDecoder(config) if config.depth_decoder else None
        nn.init.normal_(self.position, std=0.02)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward_features(
        self,
        latents: torch.Tensor,
        pool: torch.Tensor,
        *,
        capture_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if latents.ndim != 3:
            raise ValueError(f"latents must have shape [batch, latent_frames, channels], got {tuple(latents.shape)}.")
        if pool.ndim != 3:
            raise ValueError(f"pool must have shape [batch, semantic_frames, latent_frames], got {tuple(pool.shape)}.")
        semantic_frames = pool.shape[1]
        if semantic_frames > self.position.shape[1]:
            raise ValueError(
                f"Window has {semantic_frames} semantic frames but the model was built for "
                f"{self.position.shape[1]} max_position_embeddings."
            )
        hidden_states = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = torch.bmm(pool.to(hidden_states.dtype), hidden_states.transpose(1, 2))
        hidden_states = hidden_states + self.position[:, :semantic_frames].to(hidden_states.dtype)
        layers = self.transformer if isinstance(self.transformer, nn.ModuleList) else self.transformer.layers
        if capture_layer is not None and not 0 <= capture_layer < len(layers):
            raise ValueError(f"capture_layer must be in [0, {len(layers) - 1}], got {capture_layer}.")
        captured = None
        for layer_index, layer in enumerate(layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
            if layer_index == capture_layer:
                captured = hidden_states
        if isinstance(self.transformer, nn.TransformerEncoder) and self.transformer.norm is not None:
            hidden_states = self.transformer.norm(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states, captured

    def logits_from_features(
        self,
        hidden_states: torch.Tensor,
        *,
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        semantic_logits = self.heads[0](hidden_states)
        if self.depth_decoder is None:
            return [head(hidden_states) for head in self.heads]
        semantic_codes = (
            semantic_logits.argmax(dim=-1) if teacher_forcing_targets is None else teacher_forcing_targets[:, :, 0]
        )
        acoustic_logits = self.depth_decoder(
            hidden_states,
            semantic_codes,
            teacher_forcing_targets=teacher_forcing_targets,
        )
        return [semantic_logits, *acoustic_logits]

    def forward(
        self,
        latents: torch.Tensor,
        pool: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        hidden_states, _ = self.forward_features(latents, pool)
        return self.logits_from_features(hidden_states, teacher_forcing_targets=teacher_forcing_targets)


class RVQEncoderMERTTrainingModel(nn.Module):
    def __init__(self, encoder: MiniMaxMusicRVQEncoder, *, student_layer: int):
        super().__init__()
        if not 0 <= student_layer < encoder.config.num_layers:
            raise ValueError(f"mert_student_layer must be in [0, {encoder.config.num_layers - 1}], got {student_layer}.")
        self.encoder = encoder
        self.student_layer = int(student_layer)
        if encoder.config.mup:
            MuReadout = _load_mup_package().MuReadout
            self.mert_projection = MuReadout(
                encoder.config.d_model,
                MERT_HIDDEN_SIZE,
                bias=False,
                readout_zero_init=False,
                output_mult=encoder.config.mup_output_mult,
            )
        else:
            self.mert_projection = nn.Linear(encoder.config.d_model, MERT_HIDDEN_SIZE, bias=False)
        if torch.count_nonzero(self.mert_projection.weight).item() == 0:
            raise RuntimeError("MERT projection must have non-zero initialization for cosine alignment.")

    @property
    def config(self) -> RVQEncoderConfig:
        return self.encoder.config

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.enable_gradient_checkpointing()

    def forward(
        self,
        latents: torch.Tensor,
        pool: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        hidden_states, captured = self.encoder.forward_features(
            latents,
            pool,
            capture_layer=self.student_layer,
        )
        if captured is None:
            raise RuntimeError("The configured MERT student layer was not captured.")
        logits = self.encoder.logits_from_features(
            hidden_states,
            teacher_forcing_targets=teacher_forcing_targets,
        )
        return logits, self.mert_projection(captured)


def unwrap_training_encoder(model: nn.Module) -> MiniMaxMusicRVQEncoder:
    if isinstance(model, RVQEncoderMERTTrainingModel):
        return model.encoder
    if not isinstance(model, MiniMaxMusicRVQEncoder):
        raise TypeError(f"Unexpected RVQ training model type {type(model).__name__}.")
    return model


def rvq_head_losses(logits: list[torch.Tensor], target: torch.Tensor) -> list[torch.Tensor]:
    if len(logits) != target.shape[-1]:
        raise ValueError(f"Expected {target.shape[-1]} codebook heads, got {len(logits)}.")
    losses = []
    for index, head_logits in enumerate(logits):
        vocab_size = head_logits.shape[-1]
        if target[:, :, index].max().item() >= vocab_size:
            raise ValueError(f"Target codebook {index} contains token >= vocab size {vocab_size}.")
        losses.append(F.cross_entropy(head_logits.flatten(0, 1), target[:, :, index].flatten(), ignore_index=-100))
    return losses


def rvq_topk_kl_losses(
    logits: list[torch.Tensor],
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    *,
    temperature: float,
    target: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    if temperature <= 0:
        raise ValueError("teacher_kl_temperature must be positive.")
    if teacher_topk_ids.shape != teacher_topk_logits.shape:
        raise ValueError(
            f"teacher_topk_ids and teacher_topk_logits must have matching shapes, got "
            f"{tuple(teacher_topk_ids.shape)} and {tuple(teacher_topk_logits.shape)}."
        )
    if teacher_topk_ids.ndim != 4:
        raise ValueError(
            f"teacher_topk_ids must have shape [batch, frames, codebooks, topk], got {tuple(teacher_topk_ids.shape)}."
        )
    if len(logits) != teacher_topk_ids.shape[2]:
        raise ValueError(f"Expected teacher top-k for {len(logits)} heads, got {teacher_topk_ids.shape[2]}.")

    losses = []
    temperature_sq = temperature * temperature
    for index, head_logits in enumerate(logits):
        ids = teacher_topk_ids[:, :, index].to(device=head_logits.device, dtype=torch.long)
        teacher_logits = teacher_topk_logits[:, :, index].to(device=head_logits.device, dtype=torch.float32)
        vocab_size = head_logits.shape[-1]
        valid_ids = (ids >= 0) & (ids < vocab_size)
        frame_mask = valid_ids.any(dim=-1)
        if target is not None:
            frame_mask &= target[:, :, index].to(device=head_logits.device) != -100
        if not frame_mask.any():
            losses.append(head_logits.sum() * 0.0)
            continue

        ids = ids[frame_mask]
        valid_ids = valid_ids[frame_mask]
        teacher_logits = teacher_logits[frame_mask]
        student_logits = head_logits[frame_mask]
        safe_ids = ids.masked_fill(~valid_ids, 0)
        teacher_log_probs = F.log_softmax((teacher_logits / temperature).masked_fill(~valid_ids, -torch.inf), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1).gather(dim=-1, index=safe_ids)
        per_token = torch.where(valid_ids, teacher_probs * (teacher_log_probs - student_log_probs), 0.0)
        per_frame = per_token.sum(dim=-1) * temperature_sq
        losses.append(per_frame.mean())
    return losses


def rvq_loss(
    logits: list[torch.Tensor],
    target: torch.Tensor,
    *,
    teacher_topk_ids: torch.Tensor | None = None,
    teacher_topk_logits: torch.Tensor | None = None,
    teacher_kl_weight: float = 0.0,
    teacher_kl_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_losses = rvq_head_losses(logits, target)
    ce_loss = sum(ce_losses) / len(ce_losses)
    if teacher_kl_weight <= 0:
        return ce_loss, ce_loss, ce_loss.new_zeros(())
    if teacher_topk_ids is None or teacher_topk_logits is None:
        raise ValueError("teacher_topk_ids/teacher_topk_logits are required when teacher_kl_weight > 0.")
    kl_losses = rvq_topk_kl_losses(
        logits,
        teacher_topk_ids,
        teacher_topk_logits,
        temperature=teacher_kl_temperature,
        target=target,
    )
    kl_loss = sum(kl_losses) / len(kl_losses)
    return ce_loss + float(teacher_kl_weight) * kl_loss, ce_loss, kl_loss


def mert_cosine_alignment_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if projected.shape != target.shape:
        raise ValueError(
            f"Projected student and MERT teacher features must match, got {tuple(projected.shape)} and "
            f"{tuple(target.shape)}."
        )
    if projected.ndim != 3 or projected.shape[-1] != MERT_HIDDEN_SIZE:
        raise ValueError(f"MERT alignment features must have shape [batch, frames, {MERT_HIDDEN_SIZE}].")
    cosine = F.cosine_similarity(projected.float(), target.float(), dim=-1, eps=1e-8)
    return 1.0 - cosine.mean()


def mert_alignment_weight_at_step(
    base_weight: float,
    step: int,
    total_steps: int,
    *,
    decay_start: float,
    decay_end: float,
) -> float:
    if base_weight <= 0:
        return 0.0
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for MERT weight scheduling.")
    if not 0 <= decay_start < decay_end <= 1:
        raise ValueError("MERT decay must satisfy 0 <= decay_start < decay_end <= 1.")
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    if progress <= decay_start:
        return float(base_weight)
    if progress >= decay_end:
        return 0.0
    return float(base_weight) * (decay_end - progress) / (decay_end - decay_start)


def rvq_accuracy_counts(
    logits: list[torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic_correct = (logits[0].argmax(dim=-1) == target[:, :, 0]).sum()
    semantic_total = torch.tensor(target[:, :, 0].numel(), device=target.device)
    acoustic_correct = torch.zeros((), dtype=torch.long, device=target.device)
    acoustic_total = torch.zeros((), dtype=torch.long, device=target.device)
    for index in range(1, len(logits)):
        acoustic_correct = acoustic_correct + (logits[index].argmax(dim=-1) == target[:, :, index]).sum()
        acoustic_total = acoustic_total + torch.tensor(target[:, :, index].numel(), device=target.device)
    return semantic_correct, semantic_total, acoustic_correct, acoustic_total


def rvq_topk_head_accuracy_counts(
    logits: list[torch.Tensor], target: torch.Tensor, *, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(logits) != target.shape[-1]:
        raise ValueError(f"Expected {target.shape[-1]} codebook heads, got {len(logits)}.")
    correct = []
    totals = []
    for index, head_logits in enumerate(logits):
        head_target = target[:, :, index]
        valid = head_target != -100
        selected = head_logits.topk(min(top_k, head_logits.shape[-1]), dim=-1).indices
        matches = (selected == head_target.unsqueeze(-1)) & valid.unsqueeze(-1)
        correct.append(matches.any(dim=-1).sum())
        totals.append(valid.sum())
    return torch.stack(correct), torch.stack(totals)


def rvq_accuracy(logits: list[torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    semantic_correct, semantic_total, acoustic_correct, acoustic_total = rvq_accuracy_counts(logits, target)
    return semantic_correct.float() / semantic_total.float(), acoustic_correct.float() / acoustic_total.float()


def _validate_model_width(config: RVQEncoderConfig, *, label: str) -> None:
    if config.d_model <= 0:
        raise ValueError(f"{label} d_model must be positive.")
    if config.num_heads <= 0:
        raise ValueError(f"{label} num_heads must be positive.")
    if config.d_model % config.num_heads != 0:
        raise ValueError(f"{label} d_model={config.d_model} must be divisible by num_heads={config.num_heads}.")
    if config.depth_decoder:
        if config.depth_decoder_dim <= 0:
            raise ValueError(f"{label} depth_decoder_dim must be positive.")
        if config.depth_decoder_layers <= 0:
            raise ValueError(f"{label} depth_decoder_layers must be positive.")
        if config.depth_decoder_heads <= 0:
            raise ValueError(f"{label} depth_decoder_heads must be positive.")
        if config.depth_decoder_dim % config.depth_decoder_heads != 0:
            raise ValueError(
                f"{label} depth_decoder_dim={config.depth_decoder_dim} must be divisible by "
                f"depth_decoder_heads={config.depth_decoder_heads}."
            )
        if config.depth_decoder_ff_mult <= 0:
            raise ValueError(f"{label} depth_decoder_ff_mult must be positive.")
        if not 0 <= config.depth_decoder_dropout < 1:
            raise ValueError(f"{label} depth_decoder_dropout must be in [0, 1).")


def _infer_mup_heads(width: int, *, target_width: int, target_heads: int) -> int:
    if target_width % target_heads != 0:
        raise ValueError(f"Target d_model={target_width} must be divisible by heads={target_heads}.")
    target_head_dim = target_width // target_heads
    requested = max(1, round(width / target_head_dim))
    divisors = [value for value in range(1, width + 1) if width % value == 0]
    return min(divisors, key=lambda value: (abs(value - requested), value))


def _mup_shape_config(
    target: RVQEncoderConfig,
    *,
    d_model: int,
    num_heads: int,
    output_mult: float,
    readout_zero_init: bool,
    attention_multiplier: float,
) -> RVQEncoderConfig:
    return RVQEncoderConfig(
        latent_channels=target.latent_channels,
        codebook_vocab_sizes=target.codebook_vocab_sizes,
        d_model=int(d_model),
        num_layers=target.num_layers,
        num_heads=int(num_heads),
        ff_mult=target.ff_mult,
        dropout=target.dropout,
        max_position_embeddings=target.max_position_embeddings,
        conv_dilations=target.conv_dilations,
        mup=True,
        mup_output_mult=float(output_mult),
        mup_readout_zero_init=bool(readout_zero_init),
        mup_attention_multiplier=float(attention_multiplier),
        depth_decoder=target.depth_decoder,
        depth_decoder_dim=target.depth_decoder_dim,
        depth_decoder_layers=target.depth_decoder_layers,
        depth_decoder_heads=target.depth_decoder_heads,
        depth_decoder_ff_mult=target.depth_decoder_ff_mult,
        depth_decoder_dropout=target.depth_decoder_dropout,
    )


def _mup_shape_metadata_path(path: str | Path) -> Path:
    return Path(f"{path}.meta.json")


def _mup_shape_scope(model: nn.Module) -> str:
    encoder = model.encoder if isinstance(model, RVQEncoderMERTTrainingModel) else model
    if not isinstance(encoder, MiniMaxMusicRVQEncoder):
        raise TypeError(f"Unexpected RVQ model type {type(model).__name__} for muP shape metadata.")
    if isinstance(model, RVQEncoderMERTTrainingModel):
        return MUP_MERT_DEPTH_SCOPE if encoder.config.depth_decoder else MUP_MERT_SCOPE
    return MUP_DEPTH_SCOPE if encoder.config.depth_decoder else MUP_ENCODER_SCOPE


def _write_mup_shape_metadata(path: str | Path, model: nn.Module) -> None:
    metadata = {"scope": _mup_shape_scope(model)}
    encoder = model.encoder if isinstance(model, RVQEncoderMERTTrainingModel) else model
    if encoder.config.depth_decoder:
        metadata.update(
            {
                "depth_decoder_dim": encoder.config.depth_decoder_dim,
                "depth_decoder_layers": encoder.config.depth_decoder_layers,
                "depth_decoder_heads": encoder.config.depth_decoder_heads,
                "depth_decoder_ff_mult": encoder.config.depth_decoder_ff_mult,
            }
        )
    if isinstance(model, RVQEncoderMERTTrainingModel):
        metadata.update(
            {
                "mert_hidden_size": MERT_HIDDEN_SIZE,
                "mert_student_layer": model.student_layer,
            }
        )
    metadata_path = _mup_shape_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_mup_shape_metadata(path: str | Path, model: nn.Module) -> None:
    metadata_path = _mup_shape_metadata_path(path)
    expected_scope = _mup_shape_scope(model)
    if not metadata_path.is_file():
        encoder = model.encoder if isinstance(model, RVQEncoderMERTTrainingModel) else model
        if isinstance(model, RVQEncoderMERTTrainingModel) or encoder.config.depth_decoder:
            raise ValueError(
                f"This RVQ topology requires shape metadata at {metadata_path}. "
                "Encoder-only v1/v2 base shapes are incompatible; generate fresh base shapes."
            )
        return
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("scope") != expected_scope:
        raise ValueError(f"muTransfer base-shape scope {metadata.get('scope')!r} is incompatible with {expected_scope!r}.")
    if isinstance(model, RVQEncoderMERTTrainingModel):
        if int(metadata.get("mert_hidden_size", -1)) != MERT_HIDDEN_SIZE:
            raise ValueError("muTransfer MERT projection width does not match this trainer.")
        if int(metadata.get("mert_student_layer", -1)) != model.student_layer:
            raise ValueError("muTransfer MERT student layer does not match this run.")
    encoder = model.encoder if isinstance(model, RVQEncoderMERTTrainingModel) else model
    if encoder.config.depth_decoder:
        expected_depth = {
            "depth_decoder_dim": encoder.config.depth_decoder_dim,
            "depth_decoder_layers": encoder.config.depth_decoder_layers,
            "depth_decoder_heads": encoder.config.depth_decoder_heads,
            "depth_decoder_ff_mult": encoder.config.depth_decoder_ff_mult,
        }
        changed = [name for name, value in expected_depth.items() if int(metadata.get(name, -1)) != value]
        if changed:
            raise ValueError(f"muTransfer depth-decoder shape metadata does not match: {changed}.")


def _build_mup_shape_model(config: RVQEncoderConfig, args: argparse.Namespace) -> nn.Module:
    encoder = MiniMaxMusicRVQEncoder(config)
    if args.mert_alignment_weight > 0:
        return RVQEncoderMERTTrainingModel(encoder, student_layer=args.mert_student_layer)
    return encoder


def apply_mup_base_shapes(model: nn.Module, config: RVQEncoderConfig, args: argparse.Namespace) -> None:
    if not config.mup:
        return
    mup = _load_mup_package()
    if args.mup_base_shapes:
        _validate_mup_shape_metadata(args.mup_base_shapes, model)
        mup.set_base_shapes(model, args.mup_base_shapes)
        return

    base_width = int(args.mup_base_d_model)
    delta_width = int(args.mup_delta_d_model or base_width * 2)
    base_heads = int(
        args.mup_base_heads or _infer_mup_heads(base_width, target_width=config.d_model, target_heads=config.num_heads)
    )
    delta_heads = int(
        args.mup_delta_heads or _infer_mup_heads(delta_width, target_width=config.d_model, target_heads=config.num_heads)
    )
    base_config = _mup_shape_config(
        config,
        d_model=base_width,
        num_heads=base_heads,
        output_mult=args.mup_output_mult,
        readout_zero_init=args.mup_readout_zero_init,
        attention_multiplier=args.mup_attention_multiplier,
    )
    delta_config = _mup_shape_config(
        config,
        d_model=delta_width,
        num_heads=delta_heads,
        output_mult=args.mup_output_mult,
        readout_zero_init=args.mup_readout_zero_init,
        attention_multiplier=args.mup_attention_multiplier,
    )
    _validate_model_width(base_config, label="mup base")
    _validate_model_width(delta_config, label="mup delta")
    base_model = _build_mup_shape_model(base_config, args)
    delta_model = _build_mup_shape_model(delta_config, args)
    savefile = args.mup_save_base_shapes if should_log() else None
    mup.set_base_shapes(model, base_model, delta=delta_model, savefile=savefile or None)
    if savefile and Path(savefile).is_file():
        _write_mup_shape_metadata(savefile, model)
    del base_model, delta_model


def autocast_for_accelerator(accelerator: Accelerator):
    if accelerator.mixed_precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=accelerator.device.type, dtype=dtype)


def _optimizer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        optimizer_config=args.optimizer_config,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_release_gradients=False,
        prodigy_steps=None,
        use_fsdp=False,
        fsdp_enable=False,
        fsdp_version=2,
        fsdp_fallback_optimizer="torch-adamw",
    )


def create_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_class, optimizer_details = optimizer_parameters(args.optimizer, _optimizer_args(args))
    settings = dict(optimizer_details.get("default_settings", {}))
    settings["lr"] = args.learning_rate
    if args.weight_decay is not None:
        settings["weight_decay"] = args.weight_decay
    if args.mup:
        mup = _load_mup_package()
        supported = {
            "torch-adam": mup.MuAdam,
            "torch-adamw": mup.MuAdamW,
            "torch-sgd": mup.MuSGD,
        }
        if args.optimizer not in supported:
            raise ValueError(
                "--mup supports --optimizer torch-adam, torch-adamw, or torch-sgd in this script. "
                f"Received {args.optimizer!r}."
            )
        weight_decay = settings.get("weight_decay", 0.01)
        if args.use_optimizer_param_groups and weight_decay > 0:
            param_groups = create_optimizer_params_with_decay(
                model,
                weight_decay=weight_decay,
                learning_rate=settings["lr"],
            )
        else:
            param_groups = filter(lambda parameter: parameter.requires_grad, model.parameters())
        return supported[args.optimizer](param_groups, **settings)
    return create_optimizer_with_param_groups(
        model,
        optimizer_class,
        settings,
        use_parameter_groups=args.use_optimizer_param_groups,
    )


def _scheduler_args(args: argparse.Namespace, total_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_end=args.lr_end,
        lr_power=args.lr_power,
        lr_num_cycles=args.lr_num_cycles,
        max_train_steps=total_steps,
    )


def tracker_config(
    args: argparse.Namespace, parameter_count: int, training_parameter_count: int | None = None
) -> dict[str, Any]:
    return {
        "parameter_count": int(parameter_count),
        "training_parameter_count": int(training_parameter_count or parameter_count),
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "window_frames": args.window_frames,
        "window_stride": args.window_stride,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_end": args.lr_end,
        "lr_power": args.lr_power,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_kl_temperature": args.teacher_kl_temperature,
        "mert_alignment_weight": args.mert_alignment_weight,
        "mert_model_name_or_path": args.mert_model_name_or_path,
        "mert_revision": args.mert_revision,
        "mert_teacher_layer": args.mert_teacher_layer,
        "mert_student_layer": args.mert_student_layer,
        "mert_chunk_seconds": args.mert_chunk_seconds,
        "mert_chunk_overlap_seconds": args.mert_chunk_overlap_seconds,
        "mert_decay_start": args.mert_decay_start,
        "mert_decay_end": args.mert_decay_end,
        "depth_decoder": args.depth_decoder,
        "depth_decoder_dim": args.depth_decoder_dim,
        "depth_decoder_layers": args.depth_decoder_layers,
        "depth_decoder_heads": args.depth_decoder_heads,
        "depth_decoder_ff_mult": args.depth_decoder_ff_mult,
        "depth_decoder_dropout": args.depth_decoder_dropout,
        "mixed_precision": args.mixed_precision,
        "mup": args.mup,
        "mup_base_d_model": args.mup_base_d_model,
        "mup_delta_d_model": args.mup_delta_d_model,
        "mup_output_mult": args.mup_output_mult,
        "mup_readout_zero_init": args.mup_readout_zero_init,
        "mup_attention_multiplier": args.mup_attention_multiplier,
        "seed": args.seed,
    }


def init_trackers(
    args: argparse.Namespace,
    accelerator: Accelerator,
    parameter_count: int,
    training_parameter_count: int | None = None,
) -> None:
    if args.report_to == "none" or not accelerator.is_main_process:
        return
    init_kwargs = {args.report_to: {}}
    if args.tracker_run_name:
        init_kwargs[args.report_to]["name"] = args.tracker_run_name
    accelerator.init_trackers(
        args.tracker_project_name,
        config=tracker_config(args, parameter_count, training_parameter_count),
        init_kwargs=init_kwargs,
    )


def log_tracker_metrics(
    args: argparse.Namespace,
    accelerator: Accelerator,
    namespace: str,
    metrics: dict[str, Any],
    step: int,
) -> None:
    if args.report_to == "none" or not accelerator.is_main_process:
        return
    accelerator.log({f"{namespace}/{name}": value for name, value in metrics.items()}, step=step)


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    *,
    max_batches: int,
    teacher_kl_weight: float,
    teacher_kl_temperature: float,
    mert_alignment_weight: float = 0.0,
    mert_step: int = 0,
    mert_total_steps: int = 1,
    mert_decay_start: float = 0.7,
    mert_decay_end: float = 0.9,
) -> dict[str, float]:
    model.eval()
    encoder = unwrap_training_encoder(unwrap_model(accelerator, model))
    autoregressive_depth_metrics = bool(encoder.config.depth_decoder)
    loss_sum = torch.zeros((), device=accelerator.device)
    ce_loss_sum = torch.zeros((), device=accelerator.device)
    kl_loss_sum = torch.zeros((), device=accelerator.device)
    optimization_loss_sum = torch.zeros((), device=accelerator.device)
    mert_loss_sum = torch.zeros((), device=accelerator.device)
    loss_items = torch.zeros((), device=accelerator.device)
    top1_correct = None
    top1_total = None
    top5_correct = None
    top5_total = None
    teacher_forced_top1_correct = None
    teacher_forced_top1_total = None
    teacher_forced_top5_correct = None
    teacher_forced_top5_total = None

    for batch_index, batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        with autocast_for_accelerator(accelerator):
            if mert_alignment_weight > 0:
                logits, projected = model(batch["latents"], batch["pool"], batch["target"])
                mert_loss = mert_cosine_alignment_loss(projected, batch["mert_features"])
            else:
                logits = model(batch["latents"], batch["pool"], batch["target"])
                mert_loss = loss_sum.new_zeros(())
            loss, ce_loss, kl_loss = rvq_loss(
                logits,
                batch["target"],
                teacher_topk_ids=batch.get("teacher_topk_ids"),
                teacher_topk_logits=batch.get("teacher_topk_logits"),
                teacher_kl_weight=teacher_kl_weight,
                teacher_kl_temperature=teacher_kl_temperature,
            )
            active_mert_weight = mert_alignment_weight_at_step(
                mert_alignment_weight,
                mert_step,
                mert_total_steps,
                decay_start=mert_decay_start,
                decay_end=mert_decay_end,
            )
            optimization_loss = loss + active_mert_weight * mert_loss
            if autoregressive_depth_metrics:
                if mert_alignment_weight > 0:
                    metric_logits, _ = model(batch["latents"], batch["pool"])
                else:
                    metric_logits = model(batch["latents"], batch["pool"])
            else:
                metric_logits = logits
        batch_top1_correct, batch_top1_total = rvq_topk_head_accuracy_counts(metric_logits, batch["target"], top_k=1)
        batch_top5_correct, batch_top5_total = rvq_topk_head_accuracy_counts(metric_logits, batch["target"], top_k=5)
        loss_sum = loss_sum + loss.detach().float() * batch["target"].shape[0]
        ce_loss_sum = ce_loss_sum + ce_loss.detach().float() * batch["target"].shape[0]
        kl_loss_sum = kl_loss_sum + kl_loss.detach().float() * batch["target"].shape[0]
        optimization_loss_sum = optimization_loss_sum + optimization_loss.detach().float() * batch["target"].shape[0]
        mert_loss_sum = mert_loss_sum + mert_loss.detach().float() * batch["target"].shape[0]
        loss_items = loss_items + batch["target"].shape[0]
        if top1_correct is None:
            top1_correct = torch.zeros_like(batch_top1_correct, dtype=torch.float32)
            top1_total = torch.zeros_like(batch_top1_total, dtype=torch.float32)
            top5_correct = torch.zeros_like(batch_top5_correct, dtype=torch.float32)
            top5_total = torch.zeros_like(batch_top5_total, dtype=torch.float32)
        top1_correct = top1_correct + batch_top1_correct.float()
        top1_total = top1_total + batch_top1_total.float()
        top5_correct = top5_correct + batch_top5_correct.float()
        top5_total = top5_total + batch_top5_total.float()
        if autoregressive_depth_metrics:
            batch_teacher_top1_correct, batch_teacher_top1_total = rvq_topk_head_accuracy_counts(
                logits, batch["target"], top_k=1
            )
            batch_teacher_top5_correct, batch_teacher_top5_total = rvq_topk_head_accuracy_counts(
                logits, batch["target"], top_k=5
            )
            if teacher_forced_top1_correct is None:
                teacher_forced_top1_correct = torch.zeros_like(batch_teacher_top1_correct, dtype=torch.float32)
                teacher_forced_top1_total = torch.zeros_like(batch_teacher_top1_total, dtype=torch.float32)
                teacher_forced_top5_correct = torch.zeros_like(batch_teacher_top5_correct, dtype=torch.float32)
                teacher_forced_top5_total = torch.zeros_like(batch_teacher_top5_total, dtype=torch.float32)
            teacher_forced_top1_correct = teacher_forced_top1_correct + batch_teacher_top1_correct.float()
            teacher_forced_top1_total = teacher_forced_top1_total + batch_teacher_top1_total.float()
            teacher_forced_top5_correct = teacher_forced_top5_correct + batch_teacher_top5_correct.float()
            teacher_forced_top5_total = teacher_forced_top5_total + batch_teacher_top5_total.float()

    if top1_correct is None or top1_total is None or top5_correct is None or top5_total is None:
        raise ValueError("Evaluation dataloader produced no batches.")
    stat_parts = [
        torch.stack((loss_sum, ce_loss_sum, kl_loss_sum, optimization_loss_sum, mert_loss_sum, loss_items)),
        top1_correct,
        top1_total,
        top5_correct,
        top5_total,
    ]
    if autoregressive_depth_metrics:
        if any(
            value is None
            for value in (
                teacher_forced_top1_correct,
                teacher_forced_top1_total,
                teacher_forced_top5_correct,
                teacher_forced_top5_total,
            )
        ):
            raise RuntimeError("Teacher-forced depth metrics were not accumulated.")
        stat_parts.extend(
            (
                teacher_forced_top1_correct,
                teacher_forced_top1_total,
                teacher_forced_top5_correct,
                teacher_forced_top5_total,
            )
        )
    stats = torch.cat(tuple(stat_parts))
    gathered = accelerator.gather(stats).reshape(-1, stats.numel())
    totals = gathered.sum(dim=0)
    model.train()
    loss_count = max(float(totals[5].item()), 1.0)
    num_heads = int(top1_correct.numel())
    top1_correct_total = totals[6 : 6 + num_heads]
    top1_count_total = totals[6 + num_heads : 6 + 2 * num_heads]
    top5_correct_total = totals[6 + 2 * num_heads : 6 + 3 * num_heads]
    top5_count_total = totals[6 + 3 * num_heads : 6 + 4 * num_heads]
    metrics = {
        "loss": float(totals[0].item() / loss_count),
        "ce_loss": float(totals[1].item() / loss_count),
        "teacher_kl_loss": float(totals[2].item() / loss_count),
        "semantic_top1": float(top1_correct_total[0].item() / max(top1_count_total[0].item(), 1.0)),
        "semantic_top5": float(top5_correct_total[0].item() / max(top5_count_total[0].item(), 1.0)),
        "acoustic_top1": float(top1_correct_total[1:].sum().item() / max(top1_count_total[1:].sum().item(), 1.0)),
        "acoustic_top5": float(top5_correct_total[1:].sum().item() / max(top5_count_total[1:].sum().item(), 1.0)),
    }
    if mert_alignment_weight > 0:
        mert_loss_value = float(totals[4].item() / loss_count)
        metrics.update(
            {
                "optimization_loss": float(totals[3].item() / loss_count),
                "mert_alignment_loss": mert_loss_value,
                "mert_cosine": 1.0 - mert_loss_value,
                "mert_weight": active_mert_weight,
            }
        )
    for index in range(num_heads):
        metrics[f"head_{index}_top1"] = float(top1_correct_total[index].item() / max(top1_count_total[index].item(), 1.0))
        metrics[f"head_{index}_top5"] = float(top5_correct_total[index].item() / max(top5_count_total[index].item(), 1.0))
    if autoregressive_depth_metrics:
        teacher_offset = 6 + 4 * num_heads
        teacher_top1_correct_total = totals[teacher_offset : teacher_offset + num_heads]
        teacher_top1_count_total = totals[teacher_offset + num_heads : teacher_offset + 2 * num_heads]
        teacher_top5_correct_total = totals[teacher_offset + 2 * num_heads : teacher_offset + 3 * num_heads]
        teacher_top5_count_total = totals[teacher_offset + 3 * num_heads : teacher_offset + 4 * num_heads]
        metrics.update(
            {
                "teacher_forced_semantic_top1": float(
                    teacher_top1_correct_total[0].item() / max(teacher_top1_count_total[0].item(), 1.0)
                ),
                "teacher_forced_semantic_top5": float(
                    teacher_top5_correct_total[0].item() / max(teacher_top5_count_total[0].item(), 1.0)
                ),
                "teacher_forced_acoustic_top1": float(
                    teacher_top1_correct_total[1:].sum().item() / max(teacher_top1_count_total[1:].sum().item(), 1.0)
                ),
                "teacher_forced_acoustic_top5": float(
                    teacher_top5_correct_total[1:].sum().item() / max(teacher_top5_count_total[1:].sum().item(), 1.0)
                ),
            }
        )
        for index in range(num_heads):
            metrics[f"teacher_forced_head_{index}_top1"] = float(
                teacher_top1_correct_total[index].item() / max(teacher_top1_count_total[index].item(), 1.0)
            )
            metrics[f"teacher_forced_head_{index}_top5"] = float(
                teacher_top5_correct_total[index].item() / max(teacher_top5_count_total[index].item(), 1.0)
            )
    return metrics


def append_metrics_record(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def save_checkpoint(
    accelerator: Accelerator,
    args: argparse.Namespace,
    model: nn.Module,
    output_dir: Path,
    hub_uploader: HubCheckpointUploader | None,
    *,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    best_validation_loss: float | None,
    final: bool = False,
) -> Path:
    checkpoint_dir = output_dir / ("final" if final else f"checkpoint-{global_step}")
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        unwrapped = unwrap_model(accelerator, model)
        encoder = unwrap_training_encoder(unwrapped)
        state = {name: value.detach().cpu().contiguous() for name, value in encoder.state_dict().items()}
        save_safetensors_file(
            state,
            str(checkpoint_dir / "rvq_encoder.safetensors"),
            metadata={"format": CHECKPOINT_FORMAT},
        )
        with (checkpoint_dir / "rvq_encoder_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(encoder.config), handle, indent=2, sort_keys=True)
            handle.write("\n")
        if encoder.config.mup:
            mup = _load_mup_package()
            encoder_shapes = checkpoint_dir / "mup_base_shapes.bsh"
            mup.save_base_shapes(encoder, str(encoder_shapes))
            _write_mup_shape_metadata(encoder_shapes, encoder)
            if isinstance(unwrapped, RVQEncoderMERTTrainingModel):
                wrapper_shapes = checkpoint_dir / "mert_training_mup_base_shapes.bsh"
                mup.save_base_shapes(unwrapped, str(wrapper_shapes))
                _write_mup_shape_metadata(wrapper_shapes, unwrapped)
        with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": CHECKPOINT_FORMAT,
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "best_validation_loss": best_validation_loss,
                    "train_args": vars(args),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    wait_for_all_processes(accelerator)
    if hub_uploader is not None:
        hub_uploader.upload_checkpoint(checkpoint_dir, global_step=global_step, final=final)
    return checkpoint_dir


def load_trainer_state(checkpoint_dir: str | None) -> dict[str, Any]:
    if not checkpoint_dir:
        return {}
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.is_file():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_resume_training_topology(args: argparse.Namespace, trainer_state: dict[str, Any]) -> None:
    if not trainer_state:
        return
    previous = trainer_state.get("train_args") or {}
    previous_depth = bool(previous.get("depth_decoder", False))
    current_depth = bool(args.depth_decoder)
    if previous_depth != current_depth:
        raise ValueError("Checkpoint resume cannot change whether the autoregressive depth decoder is enabled.")
    if current_depth:
        depth_fields = (
            "depth_decoder_dim",
            "depth_decoder_layers",
            "depth_decoder_heads",
            "depth_decoder_ff_mult",
            "depth_decoder_dropout",
        )
        changed_depth = [name for name in depth_fields if previous.get(name) != getattr(args, name)]
        if changed_depth:
            raise ValueError(f"Checkpoint resume cannot change depth-decoder topology fields: {changed_depth}.")
    previous_mert = float(previous.get("mert_alignment_weight", 0.0)) > 0
    current_mert = args.mert_alignment_weight > 0
    if previous_mert != current_mert:
        raise ValueError("Checkpoint resume cannot change whether MERT alignment is enabled.")
    if not current_mert:
        return
    required = (
        "mert_model_name_or_path",
        "mert_revision",
        "mert_teacher_layer",
        "mert_student_layer",
    )
    changed = [name for name in required if previous.get(name) != getattr(args, name)]
    if changed:
        raise ValueError(f"Checkpoint resume cannot change MERT topology/provenance fields: {changed}.")


def build_dataloaders(
    args: argparse.Namespace,
    train_records: list[RVQTraceRecord],
    validation_records: list[RVQTraceRecord],
) -> tuple[DataLoader, DataLoader | None, RVQWindowDataset, RVQWindowDataset | None]:
    train_dataset = RVQWindowDataset(
        train_records,
        cache_dir=args.latent_cache_dir,
        window_frames=args.window_frames,
        window_stride=args.window_stride,
        random_crop=args.random_crop,
        require_exact_alignment=args.require_exact_alignment,
        require_teacher_topk=args.teacher_kl_weight > 0,
        mert_cache_dir=args.mert_cache_dir,
        mert_teacher_layer=args.mert_teacher_layer,
        require_mert_features=args.mert_alignment_weight > 0,
    )
    validation_dataset = None
    if validation_records:
        validation_dataset = RVQWindowDataset(
            validation_records,
            cache_dir=args.latent_cache_dir,
            window_frames=args.window_frames,
            window_stride=args.window_frames,
            random_crop=False,
            require_exact_alignment=args.require_exact_alignment,
            require_teacher_topk=args.teacher_kl_weight > 0,
            mert_cache_dir=args.mert_cache_dir,
            mert_teacher_layer=args.mert_teacher_layer,
            require_mert_features=args.mert_alignment_weight > 0,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "collate_fn": collate_rvq_windows,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
        loader_kwargs["persistent_workers"] = args.dataloader_persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=args.dataloader_drop_last,
        **loader_kwargs,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.validation_batch_size or args.train_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    return train_loader, validation_loader, train_dataset, validation_dataset


@dataclass(frozen=True)
class EvaluationCheckpoint:
    name: str
    path: Path
    step: int


def discover_evaluation_checkpoints(output_dir: str | Path) -> list[EvaluationCheckpoint]:
    root = Path(output_dir).expanduser()
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not (path / "rvq_encoder.safetensors").is_file():
            continue
        checkpoints.append(EvaluationCheckpoint(path.name, path, int(match.group(1))))

    final_path = root / "final"
    if (final_path / "rvq_encoder.safetensors").is_file():
        state = load_trainer_state(str(final_path))
        if "global_step" not in state:
            raise ValueError(f"Final checkpoint is missing global_step in {final_path / 'trainer_state.json'}.")
        checkpoints.append(EvaluationCheckpoint("final", final_path, int(state["global_step"])))

    checkpoints.sort(key=lambda checkpoint: (checkpoint.step, checkpoint.name == "final"))
    if not checkpoints:
        raise ValueError(f"No exported RVQ checkpoints were found under {root}.")
    return checkpoints


def load_evaluation_model(checkpoint: EvaluationCheckpoint) -> MiniMaxMusicRVQEncoder:
    config_path = checkpoint.path / "rvq_encoder_config.json"
    base_shapes_path = checkpoint.path / "mup_base_shapes.bsh"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint {checkpoint.name} is missing {config_path.name}.")
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    config_values["codebook_vocab_sizes"] = tuple(int(value) for value in config_values["codebook_vocab_sizes"])
    config_values["conv_dilations"] = tuple(int(value) for value in config_values["conv_dilations"])
    config = RVQEncoderConfig(**config_values)
    model = MiniMaxMusicRVQEncoder(config)
    if config.mup:
        if not base_shapes_path.is_file():
            raise FileNotFoundError(f"Checkpoint {checkpoint.name} is missing {base_shapes_path.name}.")
        _validate_mup_shape_metadata(base_shapes_path, model)
        _load_mup_package().set_base_shapes(model, str(base_shapes_path), rescale_params=False)
    state = load_safetensors_file(str(checkpoint.path / "rvq_encoder.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    return model


def build_validation_dataloader(
    args: argparse.Namespace,
    validation_records: list[RVQTraceRecord],
) -> tuple[DataLoader, RVQWindowDataset]:
    if not validation_records:
        raise ValueError(f"No records matched validation split {args.validation_split!r}.")
    dataset = RVQWindowDataset(
        validation_records,
        cache_dir=args.latent_cache_dir,
        window_frames=args.window_frames,
        window_stride=args.window_frames,
        random_crop=False,
        require_exact_alignment=args.require_exact_alignment,
        require_teacher_topk=args.teacher_kl_weight > 0,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "collate_fn": collate_rvq_windows,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
        loader_kwargs["persistent_workers"] = args.dataloader_persistent_workers
    dataloader = DataLoader(
        dataset,
        batch_size=args.validation_batch_size or args.train_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return dataloader, dataset


def parse_training_progress_log(path: str | Path, *, log_steps: int) -> list[dict[str, Any]]:
    if log_steps <= 0:
        raise ValueError("log_steps must be positive when parsing a training progress log.")
    text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    records: dict[int, dict[str, Any]] = {}
    metric_names = {
        "loss": "loss",
        "ce": "ce_loss",
        "kl": "teacher_kl_loss",
        "sem": "semantic_top1",
        "ac": "acoustic_top1",
        "lr": "learning_rate",
    }
    for line in text.splitlines():
        if "RVQ encoder steps:" not in line:
            continue
        progress_match = re.search(r"(\d+)/(\d+)", line)
        if progress_match is None:
            continue
        step = int(progress_match.group(1))
        if step == 0 or step % log_steps != 0:
            continue
        values = {name: float(value) for name, value in re.findall(r"\b(ac|ce|kl|loss|lr|sem)=([0-9.eE+-]+)", line)}
        if set(values) != set(metric_names):
            continue
        records[step] = {
            "type": "train",
            "step": step,
            **{target: values[source] for source, target in metric_names.items()},
        }
    if not records:
        raise ValueError(f"No RVQ training metrics were found in {path}.")
    return [records[step] for step in sorted(records)]


def load_training_history(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    metrics_path = output_dir / "training_metrics.jsonl"
    if metrics_path.is_file():
        records = []
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {metrics_path} at line {line_number}.") from exc
        return records
    if args.eval_training_log:
        return parse_training_progress_log(args.eval_training_log, log_steps=args.log_steps)
    return []


def _evaluation_csv_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "checkpoint",
        "step",
        "loss",
        "ce_loss",
        "teacher_kl_loss",
        "semantic_top1",
        "semantic_top5",
        "acoustic_top1",
        "acoustic_top5",
        "teacher_forced_semantic_top1",
        "teacher_forced_semantic_top5",
        "teacher_forced_acoustic_top1",
        "teacher_forced_acoustic_top5",
    ]
    head_fields = [f"head_{index}_top{k}" for index in range(8) for k in (1, 5)]
    teacher_head_fields = [f"teacher_forced_head_{index}_top{k}" for index in range(8) for k in (1, 5)]
    available = set().union(*(row.keys() for row in rows))
    return [name for name in preferred + head_fields + teacher_head_fields if name in available]


def write_evaluation_charts(
    output_dir: Path,
    rows: list[dict[str, Any]],
    training_history: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write RVQ evaluation plots.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in rows]
    charts = []

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(steps, [row["loss"] for row in rows], marker="o", markersize=3, label="total")
    axis.plot(steps, [row["ce_loss"] for row in rows], marker="o", markersize=3, label="hard CE")
    axis.plot(steps, [row["teacher_kl_loss"] for row in rows], marker="o", markersize=3, label="teacher KL")
    axis.set_title("Exact-alignment holdout loss by checkpoint")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "checkpoint-loss.png", dpi=180)
    plt.close(figure)
    charts.append("checkpoint-loss.png")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for metric, label in (
        ("semantic_top1", "semantic top-1"),
        ("semantic_top5", "semantic top-5"),
        ("acoustic_top1", "acoustic top-1"),
        ("acoustic_top5", "acoustic top-5"),
    ):
        axis.plot(steps, [row[metric] for row in rows], marker="o", markersize=3, label=label)
    axis.set_title("Exact-alignment holdout token accuracy")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "checkpoint-accuracy.png", dpi=180)
    plt.close(figure)
    charts.append("checkpoint-accuracy.png")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index in range(8):
        label = "semantic" if index == 0 else f"acoustic {index}"
        axis.plot(steps, [row[f"head_{index}_top1"] for row in rows], label=label)
    axis.set_title("Per-codebook holdout top-1 accuracy")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "codebook-top1.png", dpi=180)
    plt.close(figure)
    charts.append("codebook-top1.png")

    if all("teacher_forced_acoustic_top1" in row for row in rows):
        figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(steps, [row["acoustic_top1"] for row in rows], marker="o", markersize=3, label="free-running")
        axes[0].plot(
            steps,
            [row["teacher_forced_acoustic_top1"] for row in rows],
            marker="o",
            markersize=3,
            label="teacher-forced",
        )
        axes[0].set_ylabel("Acoustic top-1")
        axes[0].legend()
        for index in range(1, 8):
            gap = [row[f"teacher_forced_head_{index}_top1"] - row[f"head_{index}_top1"] for row in rows]
            axes[1].plot(steps, gap, label=f"acoustic {index}")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Teacher-forced minus free-running")
        axes[1].legend(ncol=2, fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle("Autoregressive depth-decoder exposure gap")
        figure.tight_layout()
        figure.savefig(output_dir / "depth-teacher-forcing-gap.png", dpi=180)
        plt.close(figure)
        charts.append("depth-teacher-forcing-gap.png")

    train_rows = [row for row in training_history if row.get("type") == "train"]
    if train_rows:
        train_steps = [int(row["step"]) for row in train_rows]
        figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(train_steps, [row["loss"] for row in train_rows], linewidth=0.8, label="total")
        axes[0].plot(train_steps, [row["ce_loss"] for row in train_rows], linewidth=0.8, label="hard CE")
        axes[0].plot(
            train_steps,
            [row["teacher_kl_loss"] for row in train_rows],
            linewidth=0.8,
            label="teacher KL",
        )
        axes[0].set_ylabel("Batch loss")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(train_steps, [row["semantic_top1"] for row in train_rows], linewidth=0.8, label="semantic")
        axes[1].plot(train_steps, [row["acoustic_top1"] for row in train_rows], linewidth=0.8, label="acoustic")
        axes[1].set_ylabel("Batch top-1")
        axes[1].legend(fontsize=8)
        axes[2].plot(train_steps, [row["learning_rate"] for row in train_rows], linewidth=0.8)
        axes[2].set_ylabel("First-group LR")
        axes[2].set_xlabel("Training step")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle("Recorded training statistics")
        figure.tight_layout()
        figure.savefig(output_dir / "training-history.png", dpi=180)
        plt.close(figure)
        charts.append("training-history.png")
        mert_rows = [row for row in train_rows if "mert_alignment_loss" in row]
        if mert_rows:
            mert_steps = [int(row["step"]) for row in mert_rows]
            figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(
                mert_steps,
                [row["mert_alignment_loss"] for row in mert_rows],
                linewidth=0.8,
                label="cosine loss",
            )
            axes[0].plot(
                mert_steps,
                [row["mert_cosine"] for row in mert_rows],
                linewidth=0.8,
                label="cosine similarity",
            )
            axes[0].set_ylabel("MERT alignment")
            axes[0].legend(fontsize=8)
            axes[1].plot(mert_steps, [row["mert_weight"] for row in mert_rows], linewidth=0.8)
            axes[1].set_ylabel("Loss weight")
            axes[1].set_xlabel("Training step")
            for axis in axes:
                axis.grid(alpha=0.25)
            figure.suptitle("MERT representation alignment")
            figure.tight_layout()
            figure.savefig(output_dir / "mert-alignment.png", dpi=180)
            plt.close(figure)
            charts.append("mert-alignment.png")
    return charts


def _summary_checkpoints(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    candidates = (
        ("lowest loss", min(rows, key=lambda row: row["loss"])),
        ("best semantic top-1", max(rows, key=lambda row: row["semantic_top1"])),
        ("best acoustic top-1", max(rows, key=lambda row: row["acoustic_top1"])),
        ("final", rows[-1]),
    )
    selected: dict[str, tuple[list[str], dict[str, Any]]] = {}
    for label, row in candidates:
        checkpoint = str(row["checkpoint"])
        if checkpoint not in selected:
            selected[checkpoint] = ([], row)
        selected[checkpoint][0].append(label)
    return [("; ".join(labels), row) for labels, row in selected.values()]


def render_evaluation_readme(
    rows: list[dict[str, Any]],
    *,
    eval_subdir: str,
    validation_records: int,
    validation_windows: int,
    charts: list[str],
) -> str:
    lines = [
        "## Offline Checkpoint Evaluation",
        "",
        f"Exact-alignment holdout: {validation_records:,} tracks, {validation_windows:,} windows.",
        "",
        "| Selection | Checkpoint | Step | Loss | Semantic top-1 | Semantic top-5 | Acoustic top-1 | Acoustic top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in _summary_checkpoints(rows):
        lines.append(
            f"| {label} | `{row['checkpoint']}` | {row['step']:,} | {row['loss']:.6f} | "
            f"{row['semantic_top1']:.4f} | {row['semantic_top5']:.4f} | "
            f"{row['acoustic_top1']:.4f} | {row['acoustic_top5']:.4f} |"
        )
    lines.extend(("", "Top-k accuracy measures exact token inclusion. It does not measure perceptual code equivalence.", ""))
    for chart in charts:
        title = chart.removesuffix(".png").replace("-", " ").title()
        lines.extend((f"### {title}", "", f"![{title}]({eval_subdir}/{chart})", ""))
    lines.extend(
        (
            f"Full data: [`checkpoint-metrics.csv`]({eval_subdir}/checkpoint-metrics.csv), "
            f"[`evaluation-metrics.json`]({eval_subdir}/evaluation-metrics.json).",
            "",
        )
    )
    return "\n".join(lines)


def update_evaluation_section(readme: str, section: str) -> str:
    managed = f"{EVALUATION_SECTION_START}\n{section.rstrip()}\n{EVALUATION_SECTION_END}"
    pattern = re.compile(
        rf"{re.escape(EVALUATION_SECTION_START)}.*?{re.escape(EVALUATION_SECTION_END)}",
        re.DOTALL,
    )
    if pattern.search(readme):
        return pattern.sub(managed, readme).rstrip() + "\n"
    return readme.rstrip() + "\n\n" + managed + "\n"


def assert_public_text_safe(*values: str) -> None:
    public_text = "\n".join(values)
    if any(pattern.search(public_text) for pattern in PUBLIC_TEXT_LOCAL_IDENTITY_PATTERNS):
        raise ValueError("Blocked: local machine identity was found in public text.")


def write_evaluation_artifacts(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    training_history: list[dict[str, Any]],
    *,
    validation_records: int,
    validation_windows: int,
) -> tuple[Path, str]:
    output_dir = Path(args.output_dir).expanduser()
    eval_subdir = Path(args.eval_output_subdir)
    if eval_subdir.is_absolute() or ".." in eval_subdir.parts or eval_subdir == Path("."):
        raise ValueError("eval_output_subdir must be a non-empty relative path without '..'.")
    eval_dir = output_dir / eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "checkpoint-metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_evaluation_csv_fields(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    charts = write_evaluation_charts(eval_dir, rows, training_history)
    payload = {
        "format": EVALUATION_FORMAT,
        "dataset_repo_id": args.dataset_repo_id,
        "validation_split": args.validation_split,
        "require_exact_alignment": args.require_exact_alignment,
        "validation_records": validation_records,
        "validation_windows": validation_windows,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_kl_temperature": args.teacher_kl_temperature,
        "checkpoints": rows,
        "training_history": training_history,
    }
    (eval_dir / "evaluation-metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    section = render_evaluation_readme(
        rows,
        eval_subdir=args.eval_output_subdir,
        validation_records=validation_records,
        validation_windows=validation_windows,
        charts=charts,
    )
    artifact_readme = render_evaluation_readme(
        rows,
        eval_subdir=".",
        validation_records=validation_records,
        validation_windows=validation_windows,
        charts=charts,
    )
    (eval_dir / "README.md").write_text(artifact_readme, encoding="utf-8")
    return eval_dir, section


def publish_evaluation_artifacts(args: argparse.Namespace, eval_dir: Path, section: str) -> None:
    repo_id = resolve_hub_model_id(args)
    if repo_id is None:
        return
    api = HfApi()
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
        readme = Path(readme_path).read_text(encoding="utf-8")
    except EntryNotFoundError:
        readme = ""
    updated_readme = update_evaluation_section(readme, section)
    commit_message = "Add RVQ encoder offline evaluation"
    text_artifacts = [
        updated_readme,
        commit_message,
        *(path.read_text(encoding="utf-8") for path in eval_dir.iterdir() if path.suffix in {".csv", ".json", ".md"}),
    ]
    assert_public_text_safe(*text_artifacts)
    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=BytesIO(updated_readme.encode("utf-8")))]
    for path in sorted(eval_dir.iterdir()):
        if path.is_file():
            operations.append(
                CommitOperationAdd(path_in_repo=f"{args.eval_output_subdir}/{path.name}", path_or_fileobj=str(path))
            )
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    )


def run_checkpoint_evaluation(args: argparse.Namespace) -> Path:
    configure_logging()
    output_dir = Path(args.output_dir).expanduser()
    checkpoints = discover_evaluation_checkpoints(output_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    _, validation_records = load_records_for_accelerator(args, accelerator)
    ensure_latent_cache(args, validation_records, accelerator=accelerator)
    validation_loader, validation_dataset = build_validation_dataloader(args, validation_records)
    validation_loader = accelerator.prepare(validation_loader)
    included_records = len({window.record_index for window in validation_dataset.windows})
    rows = []
    for checkpoint_info in checkpoints:
        model = load_evaluation_model(checkpoint_info)
        model = accelerator.prepare(model)
        metrics = evaluate(
            accelerator,
            model,
            validation_loader,
            max_batches=args.max_validation_batches,
            teacher_kl_weight=args.teacher_kl_weight,
            teacher_kl_temperature=args.teacher_kl_temperature,
        )
        row = {"checkpoint": checkpoint_info.name, "step": checkpoint_info.step, **metrics}
        rows.append(row)
        if accelerator.is_main_process:
            logger.info(
                "Evaluated %s: loss %.6f, semantic top1 %.4f, acoustic top1 %.4f",
                checkpoint_info.name,
                metrics["loss"],
                metrics["semantic_top1"],
                metrics["acoustic_top1"],
            )
        del model
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        wait_for_all_processes(accelerator)

    eval_dir = output_dir / args.eval_output_subdir
    if accelerator.is_main_process:
        training_history = load_training_history(args, output_dir)
        eval_dir, section = write_evaluation_artifacts(
            args,
            rows,
            training_history,
            validation_records=included_records,
            validation_windows=len(validation_dataset),
        )
        publish_evaluation_artifacts(args, eval_dir, section)
    wait_for_all_processes(accelerator)
    accelerator.end_training()
    return eval_dir


def train(args: argparse.Namespace) -> Path:
    configure_logging()
    if args.teacher_kl_weight < 0:
        raise ValueError("teacher_kl_weight must be non-negative.")
    if args.teacher_kl_temperature <= 0:
        raise ValueError("teacher_kl_temperature must be positive.")
    if args.mert_alignment_weight < 0:
        raise ValueError("mert_alignment_weight must be non-negative.")
    if args.mert_alignment_weight > 0:
        if not 0 <= args.mert_decay_start < args.mert_decay_end <= 1:
            raise ValueError("MERT decay must satisfy 0 <= mert_decay_start < mert_decay_end <= 1.")
        if not 0 < args.mert_chunk_seconds <= 10:
            raise ValueError("mert_chunk_seconds must be in (0, 10].")
        if not 0 <= args.mert_chunk_overlap_seconds < args.mert_chunk_seconds:
            raise ValueError("mert_chunk_overlap_seconds must be non-negative and shorter than the chunk.")
        if args.mert_cache_batch_size <= 0:
            raise ValueError("mert_cache_batch_size must be positive.")
        if not 0 <= args.mert_student_layer < args.layers:
            raise ValueError(f"mert_student_layer must be in [0, {args.layers - 1}].")
        mert_cache_layers(args)
    if args.mup and args.mup_base_d_model <= 0 and not args.mup_base_shapes:
        raise ValueError("mup_base_d_model must be positive when --mup is enabled without --mup_base_shapes.")
    if args.mup and args.mup_delta_d_model < 0:
        raise ValueError("mup_delta_d_model must be non-negative.")
    if args.mup and args.mup_output_mult <= 0:
        raise ValueError("mup_output_mult must be positive.")
    if args.mup and args.mup_attention_multiplier <= 0:
        raise ValueError("mup_attention_multiplier must be positive.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    set_seed(args.seed, device_specific=True)
    hub_uploader = prepare_hub_checkpoint_uploader(args, accelerator, output_dir)

    train_records, validation_records = load_records_for_accelerator(args, accelerator)
    all_records = train_records + validation_records
    if accelerator.is_main_process:
        logger.info("Loaded %s train records and %s validation records.", len(train_records), len(validation_records))
    ensure_latent_cache(args, all_records, accelerator=accelerator)
    ensure_mert_cache(args, all_records, accelerator=accelerator)

    train_loader, validation_loader, train_dataset, validation_dataset = build_dataloaders(
        args, train_records, validation_records
    )
    if accelerator.is_main_process:
        logger.info("RVQ train windows: %s", len(train_dataset))
        if validation_dataset is not None:
            logger.info("RVQ validation windows: %s", len(validation_dataset))

    encoder_config = RVQEncoderConfig(
        codebook_vocab_sizes=tuple(int(value) for value in train_records[0].codebook_vocab_sizes),
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_position_embeddings=args.window_frames,
        mup=args.mup,
        mup_output_mult=args.mup_output_mult,
        mup_readout_zero_init=args.mup_readout_zero_init,
        mup_attention_multiplier=args.mup_attention_multiplier,
        depth_decoder=args.depth_decoder,
        depth_decoder_dim=args.depth_decoder_dim,
        depth_decoder_layers=args.depth_decoder_layers,
        depth_decoder_heads=args.depth_decoder_heads,
        depth_decoder_ff_mult=args.depth_decoder_ff_mult,
        depth_decoder_dropout=args.depth_decoder_dropout,
    )
    _validate_model_width(encoder_config, label="target")
    encoder = MiniMaxMusicRVQEncoder(encoder_config)
    model: nn.Module
    if args.mert_alignment_weight > 0:
        model = RVQEncoderMERTTrainingModel(encoder, student_layer=args.mert_student_layer)
    else:
        model = encoder
    apply_mup_base_shapes(model, encoder_config, args)
    if args.mup:
        wait_for_all_processes(accelerator)
    init_trackers(
        args,
        accelerator,
        sum(parameter.numel() for parameter in encoder.parameters()),
        sum(parameter.numel() for parameter in model.parameters()),
    )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = create_optimizer(args, model)
    if validation_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model, optimizer, train_loader, validation_loader
        )

    steps_per_epoch = max(math.ceil(len(train_loader) / args.gradient_accumulation_steps), 1)
    total_steps = args.max_train_steps if args.max_train_steps > 0 else steps_per_epoch * args.num_train_epochs
    scheduler = get_lr_scheduler(_scheduler_args(args, total_steps), optimizer, accelerator, logger, global_step=0)
    scheduler = accelerator.prepare(scheduler)

    resume_state = load_trainer_state(args.resume_from_checkpoint)
    validate_resume_training_topology(args, resume_state)
    global_step = int(resume_state.get("global_step", 0))
    start_epoch = int(resume_state.get("epoch", 0))
    resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
    best_validation_loss = resume_state.get("best_validation_loss")
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        if accelerator.is_main_process:
            logger.info("Resumed from %s at step %s.", args.resume_from_checkpoint, global_step)

    progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="RVQ encoder steps",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    model.train()
    last_checkpoint: Path | None = None
    epoch = start_epoch
    while global_step < total_steps:
        active_loader = train_loader
        if resume_batch_in_epoch > 0:
            active_loader = accelerator.skip_first_batches(train_loader, resume_batch_in_epoch)
            resume_batch_in_epoch = 0

        for batch_index, batch in enumerate(active_loader):
            with accelerator.accumulate(model):
                with autocast_for_accelerator(accelerator):
                    if args.mert_alignment_weight > 0:
                        logits, projected = model(batch["latents"], batch["pool"], batch["target"])
                        mert_loss = mert_cosine_alignment_loss(projected, batch["mert_features"])
                    else:
                        logits = model(batch["latents"], batch["pool"], batch["target"])
                        mert_loss = batch["latents"].new_zeros(())
                    loss, ce_loss, kl_loss = rvq_loss(
                        logits,
                        batch["target"],
                        teacher_topk_ids=batch.get("teacher_topk_ids"),
                        teacher_topk_logits=batch.get("teacher_topk_logits"),
                        teacher_kl_weight=args.teacher_kl_weight,
                        teacher_kl_temperature=args.teacher_kl_temperature,
                    )
                    active_mert_weight = mert_alignment_weight_at_step(
                        args.mert_alignment_weight,
                        global_step + 1,
                        total_steps,
                        decay_start=args.mert_decay_start,
                        decay_end=args.mert_decay_end,
                    )
                    optimization_loss = loss + active_mert_weight * mert_loss
                accelerator.backward(optimization_loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            if global_step % args.log_steps == 0:
                sem_ok, sem_count, ac_ok, ac_count = rvq_accuracy_counts(logits, batch["target"])
                loss_items = torch.tensor(batch["target"].shape[0], device=accelerator.device, dtype=torch.float32)
                gathered = accelerator.gather(
                    torch.stack(
                        (
                            loss.detach().float() * loss_items,
                            ce_loss.detach().float() * loss_items,
                            kl_loss.detach().float() * loss_items,
                            optimization_loss.detach().float() * loss_items,
                            mert_loss.detach().float() * loss_items,
                            loss_items,
                            sem_ok.detach().float(),
                            sem_count.detach().float(),
                            ac_ok.detach().float(),
                            ac_count.detach().float(),
                        )
                    )
                ).reshape(-1, 10)
                if accelerator.is_main_process:
                    totals = gathered.sum(dim=0)
                    loss_count = max(float(totals[5].item()), 1.0)
                    semantic_count = max(float(totals[7].item()), 1.0)
                    acoustic_count = max(float(totals[9].item()), 1.0)
                    train_metrics = {
                        "type": "train",
                        "step": global_step,
                        "loss": float(totals[0].item() / loss_count),
                        "ce_loss": float(totals[1].item() / loss_count),
                        "teacher_kl_loss": float(totals[2].item() / loss_count),
                        "semantic_top1": float(totals[6].item() / semantic_count),
                        "acoustic_top1": float(totals[8].item() / acoustic_count),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    if args.depth_decoder:
                        train_metrics["teacher_forced_semantic_top1"] = train_metrics["semantic_top1"]
                        train_metrics["teacher_forced_acoustic_top1"] = train_metrics["acoustic_top1"]
                    if args.mert_alignment_weight > 0:
                        mert_loss_value = float(totals[4].item() / loss_count)
                        train_metrics.update(
                            {
                                "optimization_loss": float(totals[3].item() / loss_count),
                                "mert_alignment_loss": mert_loss_value,
                                "mert_cosine": 1.0 - mert_loss_value,
                                "mert_weight": active_mert_weight,
                            }
                        )
                    progress.set_postfix(
                        loss=f"{train_metrics['loss']:.4f}",
                        ce=f"{train_metrics['ce_loss']:.4f}",
                        kl=f"{train_metrics['teacher_kl_loss']:.4f}",
                        mert=f"{train_metrics.get('mert_cosine', 0.0):.3f}",
                        sem=f"{train_metrics['semantic_top1']:.3f}",
                        ac=f"{train_metrics['acoustic_top1']:.3f}",
                        lr=f"{train_metrics['learning_rate']:.2e}",
                    )
                    append_metrics_record(output_dir, train_metrics)
                    log_tracker_metrics(args, accelerator, "train", train_metrics, global_step)

            if validation_loader is not None and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                metrics = evaluate(
                    accelerator,
                    model,
                    validation_loader,
                    max_batches=args.max_validation_batches,
                    teacher_kl_weight=args.teacher_kl_weight,
                    teacher_kl_temperature=args.teacher_kl_temperature,
                    mert_alignment_weight=args.mert_alignment_weight,
                    mert_step=global_step,
                    mert_total_steps=total_steps,
                    mert_decay_start=args.mert_decay_start,
                    mert_decay_end=args.mert_decay_end,
                )
                if accelerator.is_main_process:
                    mert_summary = f", MERT cosine {metrics['mert_cosine']:.3f}" if args.mert_alignment_weight > 0 else ""
                    depth_summary = (
                        f", teacher-forced acoustic top1 {metrics['teacher_forced_acoustic_top1']:.3f}"
                        if args.depth_decoder
                        else ""
                    )
                    logger.info(
                        "Validation step %s: loss %.4f, CE %.4f, teacher KL %.4f, semantic top1 %.3f, "
                        "free-running acoustic top1 %.3f%s%s",
                        global_step,
                        metrics["loss"],
                        metrics["ce_loss"],
                        metrics["teacher_kl_loss"],
                        metrics["semantic_top1"],
                        metrics["acoustic_top1"],
                        mert_summary,
                        depth_summary,
                    )
                    append_metrics_record(output_dir, {"type": "validation", "step": global_step, **metrics})
                    log_tracker_metrics(args, accelerator, "validation", metrics, global_step)
                if best_validation_loss is None or metrics["loss"] < float(best_validation_loss):
                    best_validation_loss = metrics["loss"]
                    last_checkpoint = save_checkpoint(
                        accelerator,
                        args,
                        model,
                        output_dir / "best",
                        hub_uploader,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        best_validation_loss=best_validation_loss,
                    )

            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                last_checkpoint = save_checkpoint(
                    accelerator,
                    args,
                    model,
                    output_dir,
                    hub_uploader,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    best_validation_loss=best_validation_loss,
                )

            if global_step >= total_steps:
                break
        epoch += 1

    progress.close()
    final_checkpoint = save_checkpoint(
        accelerator,
        args,
        model,
        output_dir,
        hub_uploader,
        global_step=global_step,
        epoch=epoch,
        batch_in_epoch=0,
        best_validation_loss=best_validation_loss,
        final=True,
    )
    if accelerator.is_main_process:
        summary = {
            "final_checkpoint": str(final_checkpoint),
            "last_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
        }
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info("Saved final RVQ encoder checkpoint to %s.", final_checkpoint)
    accelerator.end_training()
    return final_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MiniMax Music 3 RVQ encoder from reverse-distillation traces.")
    parser.add_argument("--eval", action="store_true", help="Evaluate every exported checkpoint and write plots.")
    parser.add_argument("--eval_output_subdir", default="evaluation/v1")
    parser.add_argument("--eval_training_log", default=None, help="Trainer progress log used to recover batch metrics.")
    parser.add_argument("--dataset_repo_id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--index_file", action="append", default=None, help="Local index JSONL. May be repeated.")
    parser.add_argument("--index_dir", default=None, help="Directory containing local index JSONL files.")
    parser.add_argument("--corpus_dir", default=None, help="Local root containing data/... shard zip paths.")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--max_index_files", type=int, default=0)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--validation_split", default="holdout")
    parser.add_argument("--validation_fraction", type=float, default=0.0)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_validation_records", type=int, default=0)
    parser.add_argument("--require_exact_alignment", action="store_true")

    parser.add_argument("--pretrained_vae_model_name_or_path", default=DEFAULT_VAE_MODEL)
    parser.add_argument("--vae_revision", default=None)
    parser.add_argument("--latent_cache_dir", default="cache/vae/minimaxmusic-rvq-encoder")
    parser.add_argument("--cache_latent_dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--rebuild_latent_cache", action="store_true")
    parser.add_argument("--mert_cache_dir", default="cache/mert/minimaxmusic-rvq-encoder")
    parser.add_argument("--mert_cache_dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--rebuild_mert_cache", action="store_true")

    parser.add_argument("--output_dir", default="output/minimaxmusic-rvq-encoder")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--push_to_hub",
        nargs="?",
        const=True,
        default=False,
        help="Enable Hub uploads with --hub_model_id, or provide the target Hub model repo id directly.",
    )
    parser.add_argument("--hub_model_id", default=None, help="Target Hugging Face Hub model repo id.")
    parser.add_argument("--model_card_private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--validation_batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--dataloader_drop_last", action="store_true")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--dataloader_persistent_workers", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw_bf16")
    parser.add_argument("--optimizer_config", default=None)
    parser.add_argument("--optimizer_beta1", type=float, default=None)
    parser.add_argument("--optimizer_beta2", type=float, default=None)
    parser.add_argument("--use_optimizer_param_groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr_scheduler", default="polynomial")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_end", type=float, default=1e-7)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--max_validation_batches", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--report_to", choices=("none", "wandb"), default="none")
    parser.add_argument("--tracker_project_name", default="simpletuner-rvq-encoder")
    parser.add_argument("--tracker_run_name", default=None)
    parser.add_argument("--teacher_kl_weight", type=float, default=0.25)
    parser.add_argument("--teacher_kl_temperature", type=float, default=1.0)
    parser.add_argument("--mert_alignment_weight", type=float, default=0.0)
    parser.add_argument("--mert_model_name_or_path", default=DEFAULT_MERT_MODEL)
    parser.add_argument("--mert_revision", default=DEFAULT_MERT_REVISION)
    parser.add_argument("--mert_teacher_layer", type=int, default=9)
    parser.add_argument(
        "--mert_cache_layer",
        action="append",
        type=int,
        default=[],
        help="Additional MERT hidden-state layer to cache in the same pass. May be repeated.",
    )
    parser.add_argument("--mert_student_layer", type=int, default=4)
    parser.add_argument(
        "--mert_chunk_seconds",
        type=float,
        default=5.0,
        help=(
            "MERT cache-extraction chunk length. The 5-second default matches MERT training excerpts; "
            "larger values improve cache throughput but can reduce representation fidelity (maximum 10 seconds)."
        ),
    )
    parser.add_argument("--mert_chunk_overlap_seconds", type=float, default=1.0)
    parser.add_argument("--mert_cache_batch_size", type=int, default=8)
    parser.add_argument("--mert_decay_start", type=float, default=0.7)
    parser.add_argument("--mert_decay_end", type=float, default=0.9)

    parser.add_argument("--window_frames", type=int, default=SEMANTIC_FRAMES_PER_WINDOW)
    parser.add_argument("--window_stride", type=int, default=SEMANTIC_FRAMES_PER_WINDOW)
    parser.add_argument("--random_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--depth_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict acoustic codebooks causally from the semantic code and preceding acoustic codes.",
    )
    parser.add_argument("--depth_decoder_dim", type=int, default=512)
    parser.add_argument("--depth_decoder_layers", type=int, default=2)
    parser.add_argument("--depth_decoder_heads", type=int, default=8)
    parser.add_argument("--depth_decoder_ff_mult", type=int, default=4)
    parser.add_argument("--depth_decoder_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mup", action="store_true", help="Enable microsoft/mup parametrization for width transfer.")
    parser.add_argument("--mup_base_shapes", default=None, help="Existing mup .bsh file to load for muTransfer.")
    parser.add_argument("--mup_save_base_shapes", default=None, help="Optional path to save computed mup base shapes.")
    parser.add_argument("--mup_base_d_model", type=int, default=128)
    parser.add_argument("--mup_delta_d_model", type=int, default=0, help="0 means 2x --mup_base_d_model.")
    parser.add_argument("--mup_base_heads", type=int, default=0, help="0 infers heads from target head dimension.")
    parser.add_argument("--mup_delta_heads", type=int, default=0, help="0 infers heads from target head dimension.")
    parser.add_argument("--mup_output_mult", type=float, default=1.0)
    parser.add_argument("--mup_readout_zero_init", action="store_true")
    parser.add_argument("--mup_attention_multiplier", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval:
        run_checkpoint_evaluation(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
