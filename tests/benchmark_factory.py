#!/usr/bin/env python3
"""
Performance benchmark comparison script for old vs new factory implementations.

This script measures and compares:
- Initialization time for old vs new factory
- Memory usage before/after
- Backend creation time
- Total configuration processing time
- Memory efficiency
- CPU utilization

Usage:
    python benchmark_factory.py [--iterations N] [--config CONFIG_NAME]
"""

import argparse
import gc
import json
import logging
import os
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple
from unittest.mock import Mock

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FactoryBenchmark:
    """Benchmark suite for factory performance comparison."""

    def __init__(self, iterations: int = 5):
        """Initialize benchmark suite."""
        self.iterations = iterations
        self.test_dir = Path(__file__).parent
        self.fixtures_dir = self.test_dir / "fixtures" / "factory_golden"
        self.configs_dir = self.fixtures_dir / "configs"

        # Create temporary test environment
        self.temp_dir = Path(tempfile.mkdtemp(prefix="factory_benchmark_"))
        self.temp_images_dir = self.temp_dir / "images"
        self.temp_cache_dir = self.temp_dir / "cache"
        self.temp_conditioning_dir = self.temp_dir / "conditioning"

        self._setup_test_environment()

        # Results storage
        self.results = {"old_factory": {}, "new_factory": {}, "comparison": {}}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _setup_test_environment(self):
        """Set up test directories and files."""
        # Create test directories
        for dir_path in [self.temp_images_dir, self.temp_cache_dir, self.temp_conditioning_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create dummy test files
        for i in range(10):  # More files for realistic benchmarking
            image_file = self.temp_images_dir / f"test_image_{i}.jpg"
            image_file.write_text(f"dummy image content {i}")

            caption_file = self.temp_images_dir / f"test_image_{i}.txt"
            caption_file.write_text(f"test caption {i}")

        for i in range(5):
            cond_file = self.temp_conditioning_dir / f"test_cond_{i}.jpg"
            cond_file.write_text(f"dummy conditioning content {i}")

            caption_file = self.temp_conditioning_dir / f"test_cond_{i}.txt"
            caption_file.write_text(f"conditioning caption {i}")

    def _create_mock_args(self):
        """Create mock arguments for testing."""
        mock_args = Mock()
        mock_args.model_type = "sdxl"
        mock_args.pretrained_model_name_or_path = "test/model"
        mock_args.revision = "main"
        mock_args.variant = None
        mock_args.mixed_precision = "bf16"
        mock_args.resolution = 1024
        mock_args.resolution_type = "pixel"
        mock_args.aspect_bucket_rounding = 2
        mock_args.maximum_image_size = 2048
        mock_args.target_downsample_size = 1024
        mock_args.train_batch_size = 1
        mock_args.caption_dropout_probability = 0.0
        mock_args.cache_dir_text = str(self.temp_cache_dir)
        mock_args.vae_cache_scan_behaviour = "recreate"
        mock_args.delete_problematic_images = False
        mock_args.skip_file_discovery = "none"
        mock_args.disable_xformers = True
        mock_args.crop = True
        mock_args.crop_aspect = "preserve"
        mock_args.crop_style = "centre"
        mock_args.minimum_image_size = 0
        mock_args.maximum_image_size = 2048
        mock_args.data_backend_config = []
        return mock_args

    def _load_and_prepare_config(self, config_name: str):
        """Load and prepare a config file for benchmarking."""
        config_path = self.configs_dir / config_name
        with open(config_path, "r") as f:
            config = json.load(f)

        # Update paths to use test directories
        for backend_config in config:
            if backend_config.get("instance_data_dir"):
                if "main" in backend_config["id"] or "dataset" in backend_config["id"]:
                    backend_config["instance_data_dir"] = str(self.temp_images_dir)
                elif "conditioning" in backend_config["id"]:
                    backend_config["instance_data_dir"] = str(self.temp_conditioning_dir)

            if backend_config.get("cache_dir"):
                backend_config["cache_dir"] = str(self.temp_cache_dir / backend_config["id"])

        return config

    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

    def measure_cpu_usage(self) -> float:
        """Measure current CPU usage."""
        return psutil.cpu_percent(interval=0.1)

    def benchmark_factory_implementation(self, config: List[Dict], args: Mock) -> Dict[str, any]:
        """Benchmark the factory implementation."""

        results = {"times": [], "memory_before": [], "memory_after": [], "memory_peak": [], "cpu_usage": [], "errors": []}

        for iteration in range(self.iterations):

            # Force garbage collection before measurement
            gc.collect()

            # using standard factory

            # Measure initial state
            memory_before = self.measure_memory_usage()
            results["memory_before"].append(memory_before)

            # Start memory tracking
            tracemalloc.start()

            try:
                # Start timing
                start_time = time.perf_counter()

                # Force module reload to ensure we get the right implementation
                if "simpletuner.helpers.data_backend.factory" in os.sys.modules:
                    import importlib

                    importlib.reload(os.sys.modules["simpletuner.helpers.data_backend.factory"])

                # Import and run the factory
                from simpletuner.helpers.data_backend.factory import configure_multi_databackend

                # Set config on args
                args.data_backend_config = config

                # Execute the factory function
                result = configure_multi_databackend(
                    args, Mock(), None, None, Mock()  # accelerator  # text_encoders  # tokenizers  # model
                )

                # End timing
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                results["times"].append(execution_time)

                # Measure memory after execution
                memory_after = self.measure_memory_usage()
                results["memory_after"].append(memory_after)

                # Get peak memory usage
                current, peak = tracemalloc.get_traced_memory()
                results["memory_peak"].append(peak / 1024 / 1024)  # Convert to MB

                # Measure CPU usage
                cpu_usage = self.measure_cpu_usage()
                results["cpu_usage"].append(cpu_usage)

            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            finally:
                # Stop memory tracking
                tracemalloc.stop()

            # Small delay between iterations
            time.sleep(0.1)

        return results

    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0}

        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "stdev": stdev(values) if len(values) > 1 else 0,
        }

    def benchmark_config(self, config_name: str) -> Dict[str, any]:
        """Benchmark factory implementation with a specific config."""

        config = self._load_and_prepare_config(config_name)
        args = self._create_mock_args()

        # benchmark factory
        factory_results = self.benchmark_factory_implementation(config, args)

        # Calculate statistics
        factory_stats = {
            "time": self.calculate_statistics(factory_results["times"]),
            "memory_rss": self.calculate_statistics([m["rss_mb"] for m in factory_results["memory_after"]]),
            "memory_peak": self.calculate_statistics(factory_results["memory_peak"]),
            "cpu_usage": self.calculate_statistics(factory_results["cpu_usage"]),
            "error_count": len(factory_results["errors"]),
        }

        return {"config_name": config_name, "factory": {"raw_results": factory_results, "statistics": factory_stats}}

    def run_full_benchmark(self, config_names: List[str] = None) -> Dict[str, any]:
        """Run complete benchmark suite."""
        if config_names is None:
            # Default configs to benchmark
            config_names = ["minimal_local_config.json", "multi_backend_dependencies.json"]

        full_results = {}

        for config_name in config_names:
            if (self.configs_dir / config_name).exists():
                try:
                    results = self.benchmark_config(config_name)
                    full_results[config_name] = results
                except Exception as e:
                    logger.error(f"Failed to benchmark {config_name}: {e}")
                    full_results[config_name] = {"error": str(e)}
            else:
                logger.warning(f"Config file not found: {config_name}")

        return full_results

    def print_summary_report(self, results: Dict[str, any]):
        """Print a summary report of benchmark results."""
        print("\n" + "=" * 80)
        print("FACTORY PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        for config_name, config_results in results.items():
            if "error" in config_results:
                print(f"\n{config_name}: ERROR - {config_results['error']}")
                continue

            print(f"\n{config_name}:")
            print("-" * len(config_name))

            factory_stats = config_results["factory"]["statistics"]

            # Execution Time
            print(f"Execution Time: {factory_stats['time']['mean']:.4f}s ± {factory_stats['time']['stdev']:.4f}s")

            # Memory Usage
            print(
                f"Memory Usage (RSS): {factory_stats['memory_rss']['mean']:.2f}MB ± {factory_stats['memory_rss']['stdev']:.2f}MB"
            )

            # Peak Memory
            print(
                f"Peak Memory: {factory_stats['memory_peak']['mean']:.2f}MB ± {factory_stats['memory_peak']['stdev']:.2f}MB"
            )

            # CPU Usage
            print(f"CPU Usage: {factory_stats['cpu_usage']['mean']:.1f}% ± {factory_stats['cpu_usage']['stdev']:.1f}%")

            # Error Counts
            print(f"Errors: {factory_stats['error_count']} errors")

        print("\n" + "=" * 80)

    def save_detailed_results(self, results: Dict[str, any], output_file: str = None):
        """Save detailed results to a JSON file."""
        if output_file is None:
            output_file = f"factory_benchmark_results_{int(time.time())}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark factory implementations")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per test (default: 5)")
    parser.add_argument("--config", type=str, help="Specific config to test (default: test all)")
    parser.add_argument("--output", type=str, help="Output file for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    configs_to_test = None
    if args.config:
        configs_to_test = [args.config]

    with FactoryBenchmark(iterations=args.iterations) as benchmark:
        try:
            results = benchmark.run_full_benchmark(configs_to_test)
            benchmark.print_summary_report(results)

            if args.output:
                benchmark.save_detailed_results(results, args.output)
            else:
                benchmark.save_detailed_results(results)

        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()
