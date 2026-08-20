#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0
"""Precompute substitute-RVQ-encoder codes for source tracks.

Stores per-track [frames, 8] int32 codes with per-book vocabulary offsets
already applied, plus track duration, so training crops map by time ratio.
Shardable across GPUs via --shard-index/--num-shards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RVQ codes for re-planner conditioning")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pair-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--raw-codes",
        action="store_true",
        help="store raw per-codebook indices without vocabulary offsets (required for SimpleTuner LM training)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within num-shards")
    sys.path.insert(0, str(args.collection_dir))
    from minimax_music3_reference_adapter import MiniMaxMusic3ReferenceAdapter

    device = torch.device(args.device)
    adapter = MiniMaxMusic3ReferenceAdapter.from_pretrained(cache_dir=args.cache_dir)
    vocab_sizes = list(adapter.rvq_encoder.config.codebook_vocab_sizes)
    books = len(vocab_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "meta.json"
    if args.shard_index == 0 and not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "codebook_vocab_sizes": vocab_sizes,
                    "books": books,
                    "total_vocab": sum(vocab_sizes),
                }
            )
        )

    pair_ids = [line for line in args.pair_ids_file.read_text().replace(",", "\n").split() if line]
    done = skipped = 0
    for index, pair_id in enumerate(pair_ids):
        if index % args.num_shards != args.shard_index:
            continue
        output_path = args.output_dir / f"{pair_id.replace('/', '__')}.pt"
        if output_path.exists():
            skipped += 1
            continue
        audio, sample_rate = sf.read(args.source_dir / pair_id, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy())
        codes = adapter.predict_codes(waveform, sample_rate, device=device, offload_after=False)
        if args.raw_codes:
            combined = codes.long().to(torch.int32)
        else:
            offsets = torch.tensor([0] + list(torch.tensor(vocab_sizes[:-1]).cumsum(0)), dtype=torch.long)[:books]
            combined = (codes.long() + offsets[None, :]).to(torch.int32)
        torch.save({"codes": combined, "duration": audio.shape[0] / sample_rate}, output_path)
        done += 1
        if done % 100 == 0:
            print(json.dumps({"shard": args.shard_index, "done": done, "skipped": skipped}), flush=True)
    print(json.dumps({"shard": args.shard_index, "done": done, "skipped": skipped, "finished": True}), flush=True)


if __name__ == "__main__":
    main()
