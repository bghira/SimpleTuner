"""Measure preparation bookkeeping in fresh processes; no model or remote storage is used.

Run from the repository root with PYTHONPATH=. CPU timings include tracemalloc
when --trace is selected. The encoding sink normalizes requests but creates no
tensors or artifacts, so results do not represent text encoder throughput.
Eager mode materializes the same requests that streaming mode passes incrementally.
"""

import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def worker(args):
    import tracemalloc
    from types import SimpleNamespace

    import psutil

    from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
    from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
    from simpletuner.helpers.metadata.preparation import PreparationView
    from simpletuner.helpers.models.common import TextEmbedCacheKey

    counters = {"lookups": 0, "records_encoded": 0, "encode_calls": 0, "largest_encode_call": 0}

    def exists(_filename):
        counters["lookups"] += 1
        if args.lookup_delay_ms:
            time.sleep(args.lookup_delay_ms / 1000)
        return args.cache_state == "warm" or (args.cache_state == "mixed" and counters["lookups"] % 2 == 0)

    backend = SimpleNamespace(
        id="benchmark",
        type="local",
        create_directory=lambda _: None,
        bulk_exists=lambda keys: [exists(key) for key in keys],
        get_caption=lambda ref: f"Caption {ref}: " + "a descriptive caption with several details. " * 5,
    )
    metadata = SimpleNamespace(instance_data_dir="", get_metadata_by_filepath=lambda ref: {})

    @contextmanager
    def open_preparation_view(*, worker, workers):
        refs = (f"sample-{index:012d}.jpg" for index in range(worker, args.samples, workers))
        yield PreparationView(refs, metadata)

    metadata.open_preparation_view = open_preparation_view
    model = SimpleNamespace(
        text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION,
        uses_text_embeddings_cache=lambda: True,
        requires_text_embed_image_context=lambda: False,
    )
    runtime = DatasetRuntime("benchmark", backend, metadata, {"caption_strategy": "csv"})
    cache = TextEmbeddingCache(
        backend.id, backend, [], [], None, model=model, model_type="benchmark", text_cache_disable=True
    )

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
    try:
        if args.trace:
            tracemalloc.start()
        started = time.perf_counter()
        records = runtime.iter_preparation_text_records(model, batch_size=args.batch_size)
        if args.mode == "eager":
            records = list(records)
        cache.precompute_embeddings(records, batch_size=args.batch_size)
        seconds = time.perf_counter() - started
        python_peak = tracemalloc.get_traced_memory()[1] if args.trace else None
        rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
        if args.trace:
            tracemalloc.stop()
        cache.close()
    expected = {"cold": args.samples, "warm": 0, "mixed": (args.samples + 1) // 2}[args.cache_state]
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
    parser.add_argument("--cache-state", choices=("cold", "warm", "mixed"), default="cold")
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
        for state in ("cold", "warm", "mixed"):
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
