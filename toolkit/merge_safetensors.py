#!/usr/bin/env python3
"""
Merge Flux / Kontext sharded .safetensors into one file.

Usage:
  python merge_flux_shards.py \
      --src-dir /models/black-forest-labs/FLUX.1-Kontext-dev/transformer \
      --dst-file diffusion_pytorch_model-merged.safetensors
"""
import argparse
import glob
import os
from pathlib import Path

import torch  # only needed because we want tensors on save
from safetensors.torch import load_file, save_file


def merge_shards(src_dir: Path, dst_file: Path, pattern: str):
    shard_paths = sorted(src_dir.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards matching {pattern!r} in {src_dir}")

    print(f"Found {len(shard_paths)} shards:")
    for p in shard_paths:
        print(f"  • {p.name}")

    merged: dict[str, torch.Tensor] = {}

    for shard in shard_paths:
        print(f"→ Loading {shard.name}")
        shard_dict = load_file(shard)  # lazy-loads tensors
        # `load_file` returns dict[str, torch.Tensor] already on CPU
        overlap = merged.keys() & shard_dict.keys()
        if overlap:
            raise ValueError(f"Duplicate keys across shards: {sorted(overlap)[:5]} …")
        merged.update(shard_dict)

    print(f" ✓ Merged state-dict with {len(merged):,} tensors")

    print(f"→ Saving combined file to {dst_file}")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(dst_file))

    size_gb = dst_file.stat().st_size / (1024**3)
    print(f" ✓ Done – output size: {size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Merge sharded safetensors")
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Folder that holds the sharded *.safetensors files",
    )
    parser.add_argument(
        "--dst-file",
        type=Path,
        required=True,
        help="Path (including filename) for the merged safetensors",
    )
    parser.add_argument(
        "--pattern",
        default="diffusion_pytorch_model-*.safetensors",
        help="Glob pattern that matches the shards",
    )
    args = parser.parse_args()
    merge_shards(args.src_dir, args.dst_file, args.pattern)


if __name__ == "__main__":
    main()
