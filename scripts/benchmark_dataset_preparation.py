"""Measure preparation bookkeeping in fresh processes; no model or remote storage is used.

Run from the repository root with PYTHONPATH=. CPU timings include tracemalloc
when --trace is selected. The encoding sink normalizes requests but creates no
tensors or artifacts, so results do not represent text encoder throughput.
"""

import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker(args):
    import tracemalloc
    from types import SimpleNamespace
    from unittest.mock import patch

    import psutil

    from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
    from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
    from simpletuner.helpers.models.common import TextEmbedCacheKey
    from simpletuner.helpers.training.state_tracker import StateTracker

    counters = {"lookups": 0, "records_encoded": 0, "encode_calls": 0, "largest_encode_call": 0}

    def exists(_filename):
        counters["lookups"] += 1
        if args.lookup_delay_ms:
            time.sleep(args.lookup_delay_ms / 1000)
        return args.cache_state == "warm"

    backend = SimpleNamespace(
        id="benchmark",
        exists=exists,
        get_caption=lambda ref: f"Caption {ref}: " + "a descriptive caption with several details. " * 5,
    )
    metadata = SimpleNamespace(get_metadata_by_filepath=lambda ref: {})
    model = SimpleNamespace(
        text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION,
        uses_text_embeddings_cache=lambda: True,
        requires_text_embed_image_context=lambda: False,
    )
    runtime = DatasetRuntime("benchmark", backend, metadata, {"caption_strategy": "csv"})
    cache = object.__new__(TextEmbeddingCache)
    cache.id = "benchmark"
    cache.model = model
    cache.model_type = "benchmark"
    cache.data_backend = backend
    cache.cache_dir = "benchmark-cache"
    cache.rank_info = ""
    cache.key_type = TextEmbedCacheKey.CAPTION
    cache._requires_path_based_keys = False
    cache.batch_write_thread = SimpleNamespace(is_alive=lambda: True, join=lambda **kwargs: None)
    cache.debug_log = lambda *_: None
    cache.wait_for_pending_writes = lambda: None

    def encode_sink(records, **_kwargs):
        normalized = cache._normalize_prompt_records(records)
        counters["records_encoded"] += len(normalized)
        counters["encode_calls"] += 1
        counters["largest_encode_call"] = max(counters["largest_encode_call"], len(normalized))

    cache.compute_prompt_embeddings_with_model = encode_sink
    process = psutil.Process()
    gc.collect()
    rss_start = process.memory_info().rss
    rss_peak = [rss_start]
    stop = threading.Event()

    def sample_rss():
        while not stop.wait(0.005):
            rss_peak[0] = max(rss_peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss)
    sampler.start()
    if args.trace:
        tracemalloc.start()
    started = time.perf_counter()
    refs = (f"sample-{index:012d}.jpg" for index in range(args.samples))
    records = runtime.iter_text_embed_records(model, refs, init_backend={})
    try:
        if args.mode == "eager":
            records = list(records)
            inventory = (
                {cache.hash_prompt_with_path(record): False for record in records} if args.cache_state == "warm" else {}
            )
            with patch.object(StateTracker, "get_text_cache_files", return_value=inventory):
                cache.compute_embeddings_for_prompts(
                    records, return_concat=False, load_from_cache=False, split_between_processes=False
                )
        else:
            cache.precompute_embeddings(records, batch_size=args.batch_size)
        seconds = time.perf_counter() - started
        python_peak = tracemalloc.get_traced_memory()[1] if args.trace else None
        rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
        if args.trace:
            tracemalloc.stop()
    expected = args.samples if args.cache_state == "cold" else 0
    if counters["records_encoded"] != expected:
        raise AssertionError(f"Expected {expected} encoded records, got {counters['records_encoded']}")
    return {
        "mode": args.mode,
        "cache_state": args.cache_state,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "lookup_delay_ms": args.lookup_delay_ms,
        "tracemalloc": args.trace,
        "seconds": seconds,
        "records_per_second": args.samples / seconds,
        "python_peak_bytes": python_peak,
        "rss_peak_delta_bytes": max(0, rss_peak[0] - rss_start),
        **counters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "streaming"))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--counts", type=int, nargs="+", default=[2000, 20000, 100000])
    parser.add_argument("--cache-state", choices=("cold", "warm"), default="cold")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookup-delay-ms", type=float, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(worker(args)))
        return
    results = []
    for count in args.counts:
        for state in ("cold", "warm"):
            for mode in ("eager", "streaming"):
                command = [
                    sys.executable,
                    __file__,
                    "--mode",
                    mode,
                    "--samples",
                    str(count),
                    "--cache-state",
                    state,
                    "--batch-size",
                    str(args.batch_size),
                    "--lookup-delay-ms",
                    str(args.lookup_delay_ms),
                ]
                if args.trace:
                    command.append("--trace")
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                record = json.loads(result.stdout.splitlines()[-1])
                results.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"python": sys.version.split()[0], "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
