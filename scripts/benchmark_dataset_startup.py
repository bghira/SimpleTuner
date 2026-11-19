"""Measure discovery and preparation on a fresh Local image fixture.

Run from the checkout with python -m scripts.benchmark_dataset_startup.
The fixture, caches, and output directory are temporary. The second pass
reopens the saved view in the same process. No model or artifacts are loaded.
For a revision comparison, run the same command in each checkout.
"""

import argparse
import json
import subprocess
import time
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil
from PIL import Image

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import _ScanAcceleratorStub, _ScanArgsNamespace


def benchmark(samples):
    process = psutil.Process()
    with TemporaryDirectory(prefix="simpletuner-dataset-benchmark-") as directory:
        root = Path(directory)
        images = root / "images"
        images.mkdir()
        for index in range(samples):
            size = (96, 64) if index % 2 else (64, 64)
            Image.new("RGB", size).save(images / f"{index:012d}.jpg")
        config = {
            "id": "benchmark",
            "type": "local",
            "instance_data_dir": str(images),
            "resolution": 64,
            "resolution_type": "pixel",
            "crop": False,
            "preserve_data_backend_cache": True,
        }
        args = _ScanArgsNamespace(
            {"output_dir": str(root), "resolution": 64, "aspect_bucket_alignment": 8, "model_family": "sd1x"}
        )
        accelerator = _ScanAcceleratorStub()
        original_args = StateTracker.get_args()
        StateTracker.set_args(args)
        try:
            for state in ("fresh", "reopen"):
                started = time.perf_counter()
                rss_start = process.memory_info().rss
                with closing(DatasetRuntime.from_config(config, args, accelerator)) as dataset:
                    opened = time.perf_counter()
                    dataset.prepare_view()
                    prepared = time.perf_counter()
                    _, accepted = dataset.read_page(limit=1)
                if accepted != samples:
                    raise RuntimeError(f"Expected {samples} accepted images, got {accepted}.")
                yield {
                    "scope": "local_provider_preparation",
                    "cache_state": state,
                    "accepted_samples": accepted,
                    "open_seconds": opened - started,
                    "prepare_seconds": prepared - opened,
                    "rss_start_bytes": rss_start,
                    "rss_end_bytes": process.memory_info().rss,
                }
        finally:
            StateTracker.set_args(original_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for record in benchmark(args.samples):
        print(json.dumps({"revision": revision, **record}), flush=True)


if __name__ == "__main__":
    main()
