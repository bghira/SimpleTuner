"""Measure Webshart source metadata and image reads outside the trainer.

Run from the repository root with a normal dataset JSON configuration. Results
contain aggregate metrics only. The first image pass may reuse a configured
disk cache. Later passes reuse the same reader. Use AWS environment variables
for credentials.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import psutil

from simpletuner.helpers.data_backend.builders.webshart import WebshartBackendBuilder
from simpletuner.helpers.data_backend.config.image import ImageBackendConfig


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def benchmark(config, *, shards=1, samples=128, batch_size=32, repeats=2):
    started = time.perf_counter()
    backend = WebshartBackendBuilder(SimpleNamespace(is_local_main_process=False), {}).build(
        ImageBackendConfig.from_dict(config, {})
    )
    report = {
        "scope": "webshart_source_reads",
        "webshart_version": backend.webshart.__version__,
        "discovery_seconds": time.perf_counter() - started,
        "dataset_shards": backend.num_shards(),
    }
    started = time.perf_counter()
    paths = []
    rows = eligible = scanned = 0
    for shard in range(min(shards, backend.num_shards())):
        entries = backend.list_samples_in_shard(shard)
        rows += backend.dataset.get_shard_info(shard).get("num_rows", len(entries))
        eligible += len(entries)
        scanned += 1
        paths.extend(
            backend.sample_id(shard, entry["sample_idx"], entry["filename"]) for entry in entries[: samples - len(paths)]
        )
        if len(paths) >= samples:
            break
    report.update(
        metadata_seconds=time.perf_counter() - started,
        shards_scanned=scanned,
        physical_rows=rows,
        metadata_eligible_samples=eligible,
        sampled_images=len(paths),
        passes=[],
    )
    for index in range(repeats):
        started = time.perf_counter()
        decoded = 0
        for offset in range(0, len(paths), batch_size):
            keys, images = backend.read_image_batch(paths[offset : offset + batch_size])
            decoded += len(keys)
            del images
        elapsed = time.perf_counter() - started
        report["passes"].append(
            dict(
                reader_pass="first" if index == 0 else "reused",
                seconds=elapsed,
                decoded=decoded,
                decode_failures=len(paths) - decoded,
                images_per_second=decoded / elapsed if elapsed else 0,
            )
        )
    report["resident_bytes"] = psutil.Process().memory_info().rss
    if backend.format == "parquet":
        report["transport"] = backend.dataset.io_stats()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataset-id", help="required when the file contains multiple Webshart datasets")
    parser.add_argument("--shards", type=positive, default=1)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--batch-size", type=positive, default=32)
    parser.add_argument("--repeats", type=positive, default=2)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    datasets = payload if isinstance(payload, list) else [payload]
    selected = [
        d for d in datasets if d.get("type") == "webshart" and (args.dataset_id is None or d.get("id") == args.dataset_id)
    ]
    if len(selected) != 1:
        parser.error("select exactly one Webshart dataset with --dataset-id")
    result = benchmark(
        selected[0], shards=args.shards, samples=args.samples, batch_size=args.batch_size, repeats=args.repeats
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
