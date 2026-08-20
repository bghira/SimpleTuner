#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simpletuner.helpers.models.minimaxmusic.reference_adapter import MiniMaxMusic3ReferenceAdapter

SOURCE_REPO = "webshart/suno-various-94k"
TARGET_REPO = "bghira/suno-various-94k-style-pairs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache paired source/target RVQ codes for MiniMax Music 3")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--target-repo", default=TARGET_REPO)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dav-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--clip-ids-csv", type=Path)
    return parser.parse_args()


def shard_path(repo_id: str, prefix: str, shard_index: int, cache_dir: str | None) -> Path:
    filename = f"shards/{prefix}-{shard_index:05d}.tar"
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=cache_dir))


def member_index(archive: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    samples: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        path = Path(member.name)
        suffix = path.suffix.lower()
        if suffix not in {".json", ".mp3"}:
            continue
        samples.setdefault(path.stem, {})[suffix] = member
    return samples


def read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Could not read tar member {member.name}")
    return handle.read()


def decode_audio(data: bytes) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T.copy()), sample_rate


def load_clip_ids_csv(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
            raise ValueError("clip IDs CSV must contain a clip_id column")
        clip_ids = tuple(row["clip_id"].strip() for row in reader if row["clip_id"].strip())
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("clip IDs CSV must contain non-empty unique clip IDs")
    return clip_ids


def main() -> None:
    args = parse_args()
    if args.shard_index < 0:
        raise ValueError("shard-index must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = shard_path(args.source_repo, "suno-various-94k", args.shard_index, args.cache_dir)
    target_path = shard_path(
        args.target_repo,
        "suno-various-94k-style-pairs",
        args.shard_index,
        args.cache_dir,
    )
    encoder = MiniMaxMusic3ReferenceAdapter.from_pretrained(cache_dir=args.cache_dir)
    with tarfile.open(source_path) as source_archive, tarfile.open(target_path) as target_archive:
        source_samples = member_index(source_archive)
        target_samples = member_index(target_archive)
        keys = sorted(target_samples)
        if args.clip_ids_csv is not None:
            keys = list(load_clip_ids_csv(args.clip_ids_csv))
            missing = sorted(set(keys).difference(source_samples, target_samples))
            if missing:
                raise ValueError(f"Requested clip IDs are absent from shard {args.shard_index}: {missing}")
        if args.max_samples is not None:
            keys = keys[: args.max_samples]
        for clip_id in tqdm(keys, desc=f"RVQ shard {args.shard_index:05d}"):
            output_path = args.output_dir / f"shard-{args.shard_index:05d}" / f"{clip_id}.safetensors"
            if output_path.exists():
                continue
            source = source_samples.get(clip_id)
            target = target_samples[clip_id]
            if source is None or set(source) != {".json", ".mp3"} or set(target) != {".json", ".mp3"}:
                raise ValueError(f"Incomplete source/target pair for clip {clip_id}")
            source_json = json.loads(read_member(source_archive, source[".json"]))
            target_json = json.loads(read_member(target_archive, target[".json"]))
            if source_json["id"] != clip_id or target_json["id"] != clip_id:
                raise ValueError(f"Clip ID mismatch for {clip_id}")
            source_audio, source_rate = decode_audio(read_member(source_archive, source[".mp3"]))
            target_audio, target_rate = decode_audio(read_member(target_archive, target[".mp3"]))
            source_codes = encoder.predict_codes(
                source_audio,
                source_rate,
                device=args.device,
                offload_after=False,
                dav_chunk_seconds=args.dav_chunk_seconds,
            )
            target_codes = encoder.predict_codes(
                target_audio,
                target_rate,
                device=args.device,
                offload_after=False,
                dav_chunk_seconds=args.dav_chunk_seconds,
            )
            metadata = {
                "clip_id": clip_id,
                "prompt": target_json["caption"],
                "lyrics": target_json["lyrics"],
                "source_caption": source_json["caption"],
                "target_style": json.dumps(target_json["target_style"], ensure_ascii=False),
                "source_repo": args.source_repo,
                "target_repo": args.target_repo,
                "shard_index": str(args.shard_index),
                "source_frames": str(source_codes.shape[0]),
                "target_frames": str(target_codes.shape[0]),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(".tmp")
            save_file(
                {
                    "reference_codes": source_codes.to(torch.int16).contiguous(),
                    "target_codes": target_codes.to(torch.int16).contiguous(),
                },
                str(temporary_path),
                metadata=metadata,
            )
            temporary_path.replace(output_path)


if __name__ == "__main__":
    main()
