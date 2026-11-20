"""Measure native capture discovery and projected row reads in a fresh process.

Pass a credential object with url, access_id and secret_key on stdin for S3.
Local Parquet fixtures take a source directory and an optional manifest. Each
invocation uses a new reader and visits evenly spaced shards with no payload cache.
"""

import argparse
import gc
import json
import sys
import time
import tracemalloc
from urllib.parse import urlparse

import psutil
import webshart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--startup-budget-mib", type=int, default=256)
    parser.add_argument("--retained-budget-mib", type=int, default=32)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    s3_options = None
    if urlparse(args.source).scheme == "s3":
        if not args.manifest:
            parser.error("--manifest is required for an S3 source")
        credentials = json.load(sys.stdin)
        s3_options = {
            "region_name": "auto",
            "endpoint_url": credentials["url"],
            "aws_access_key_id": credentials["access_id"],
            "aws_secret_access_key": credentials["secret_key"],
        }
    process = psutil.Process()
    initial = process.memory_info().rss
    dataset = webshart.discover_parquet_dataset(
        args.source,
        manifest=args.manifest,
        metadata_cache_size=8,
        s3_options=s3_options,
    )
    count = min(args.shards, dataset.num_shards)
    gc.collect()
    baseline = process.memory_info().rss
    print(
        json.dumps(
            dict(
                stage="discovery",
                scope="webshart_catalogue_projection",
                webshart_version=webshart.__version__,
                source_revision=dataset.fingerprint,
                shards=dataset.num_shards,
                physical_rows=len(dataset),
                initial_rss=initial,
                catalogue_rss=baseline,
            )
        ),
        flush=True,
    )
    if baseline - initial > args.startup_budget_mib * 1024**2:
        raise RuntimeError("Catalogue startup exceeded the RSS budget")
    tracemalloc.start()
    warm = None
    min_size = max_size = None
    started = time.monotonic()
    for index in range(count):
        shard = index * (dataset.num_shards - 1) // max(1, count - 1)
        info = dataset.get_shard_info(shard)
        size = info["size"]
        min_size = size if min_size is None else min(min_size, size)
        max_size = size if max_size is None else max(max_size, size)
        dataset.read_metadata_rows(
            shard, range(min(4, info["num_rows"])), fields=["captions", "width", "height", "url_status"]
        )
        visited = index + 1
        if visited in (1, 10, 100, 1000, count):
            gc.collect()
            rss = process.memory_info().rss
            if visited == 10:
                warm = rss
            heap, peak_heap = tracemalloc.get_traced_memory()
            print(
                json.dumps(
                    dict(
                        stage="traversal",
                        visited=visited,
                        rss=rss,
                        python_heap=heap,
                        peak_python_heap=peak_heap,
                        transport=dataset.io_stats(),
                        open_readers=len(dataset._files),
                        hydrated_shards=len(dataset._metadata_cache),
                        smallest_shard=min_size,
                        largest_shard=max_size,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                ),
                flush=True,
            )
            if len(dataset._files) > 8 or dataset._metadata_cache:
                raise RuntimeError("Projected traversal exceeded its reader or metadata cache budget")
            if warm is not None and rss - warm > args.retained_budget_mib * 1024**2:
                raise RuntimeError("Projected traversal exceeded the post-warmup RSS budget")


if __name__ == "__main__":
    main()
