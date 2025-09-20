#!/usr/bin/env python3
"""
Factory Performance Benchmark

Benchmarks factory performance across various configuration sizes and complexity levels.
"""

import json
import logging
import os
import psutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# Add the SimpleTuner root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpletuner.helpers.models.common import ModelFoundation


class MockArgs:
    """Mock args object for factory testing."""

    def __init__(self, data_backend_config: str, model_family: str = "flux"):
        self.data_backend_config = data_backend_config
        self.model_family = model_family
        self.mixed_precision = "bf16"
        self.model_type = "full"
        self.cache_dir = "/tmp/benchmark_cache"
        self.pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
        self.controlnet_model_name_or_path = None
        self.revision = None
        self.variant = None
        self.minimum_image_size = None
        self.maximum_image_size = None
        self.target_downsample_size = None
        self.resolution_type = "pixel"
        self.resolution = 1024
        self.crop = False
        self.crop_aspect = "square"
        self.crop_style = "random"
        self.caption_strategy = "filename"
        self.instance_prompt = ""
        self.prepend_instance_prompt = False
        self.only_instance_prompt = False
        self.instance_data_dir = "/tmp/data"
        self.class_data_dir = None
        self.num_class_images = 0
        self.preserve_data_backend_cache = False
        self.write_batch_size = 128
        self.debug_aspect_buckets = False

        # Default values that might be referenced
        self.tracker_project_name = "benchmark"
        self.tracker_run_name = "test"
        self.output_dir = "/tmp/output"
        self.seed = 42
        self.push_to_hub = False
        self.hub_model_id = None

        # Additional attributes that may be referenced by conditioning
        self.controlnet = False
        self.controlnet_model_name_or_path = None
        self.caption_dropout_probability = 0.1
        self.metadata_update_interval = 65
        self.conditioning_multidataset_sampling = "uniform"


class MockModelFoundation(ModelFoundation):
    """Mock model foundation for testing."""

    def __init__(self):
        # Set up the basic attributes needed by the factory
        self.model_type = "full"
        self.model_family = "flux"

        # Create mock objects for required components
        self.vae = MagicMock()
        self.unet = MagicMock()
        self.text_encoder_1 = MagicMock()
        self.text_encoder_2 = MagicMock()
        self.transformer = MagicMock()

        # Mock the methods that might be called
        self.vae.config = MagicMock()
        self.vae.config.latent_channels = 16

        # Set up required method calls
        self.get_vae_sample_size = MagicMock(return_value=64)
        self.load_models = MagicMock()

    def _encode_prompts(self, *args, **kwargs):
        """Mock implementation of abstract method."""
        return MagicMock(), MagicMock()

    def convert_negative_text_embed_for_pipeline(self, *args, **kwargs):
        """Mock implementation of abstract method."""
        return MagicMock()

    def convert_text_embed_for_pipeline(self, *args, **kwargs):
        """Mock implementation of abstract method."""
        return MagicMock()

    def model_predict(self, *args, **kwargs):
        """Mock implementation of abstract method."""
        return MagicMock()


class BenchmarkResult:
    """Stores results from a single benchmark run."""

    def __init__(self, factory_type: str, config_size: int, backend_count: int):
        self.factory_type = factory_type
        self.config_size = config_size
        self.backend_count = backend_count
        self.execution_time = 0.0
        self.peak_memory_mb = 0.0
        self.initialization_time = 0.0
        self.configuration_time = 0.0
        self.build_time = 0.0
        self.error = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for analysis."""
        return {
            "factory_type": self.factory_type,
            "config_size": self.config_size,
            "backend_count": self.backend_count,
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "initialization_time": self.initialization_time,
            "configuration_time": self.configuration_time,
            "build_time": self.build_time,
            "error": self.error,
        }


class FactoryBenchmark:
    """Main benchmark runner for factory performance comparison."""

    def __init__(self, iterations: int = 3, verbose: bool = False):
        self.iterations = iterations
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("FactoryBenchmark")

    def create_test_configs(self) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
        """Create test configurations of various sizes and complexities."""
        configs = {}

        # Simple 2-backend config
        simple_config = [
            {
                "id": "simple-dataset",
                "type": "local",
                "instance_data_dir": "/tmp/data",
                "metadata_backend": "discovery",
                "caption_strategy": "filename",
                "cache_dir_vae": "/tmp/cache/vae",
                "resolution": 512,
            },
            {
                "id": "text-embed-cache",
                "dataset_type": "text_embeds",
                "default": True,
                "type": "local",
                "cache_dir": "/tmp/cache/text",
            },
        ]
        configs["simple_2_backends"] = ("Simple 2-backend config", simple_config)

        # Medium 5-backend config
        medium_config = []
        for i in range(4):
            medium_config.append(
                {
                    "id": f"dataset-{i}",
                    "type": "local",
                    "instance_data_dir": f"/tmp/data/{i}",
                    "metadata_backend": "discovery",
                    "caption_strategy": "filename",
                    "cache_dir_vae": f"/tmp/cache/vae/{i}",
                    "resolution": 512 + (i * 256),
                    "repeats": i + 1,
                    "probability": 1.0 / (i + 1),
                    "crop": i % 2 == 0,
                    "minimum_image_size": 256,
                    "maximum_image_size": 2048,
                }
            )
        medium_config.append(
            {
                "id": "text-embed-cache",
                "dataset_type": "text_embeds",
                "default": True,
                "type": "local",
                "cache_dir": "/tmp/cache/text",
            }
        )
        configs["medium_5_backends"] = ("Medium 5-backend config", medium_config)

        # Complex 10+ backend config with dependencies and conditioning
        complex_config = []

        # Add main datasets
        for i in range(8):
            backend_config = {
                "id": f"complex-dataset-{i}",
                "type": "local",
                "instance_data_dir": f"/tmp/data/complex/{i}",
                "metadata_backend": "parquet" if i % 2 == 0 else "discovery",
                "caption_strategy": "filename",
                "cache_dir_vae": f"/tmp/cache/vae/complex/{i}",
                "resolution": 512 + (i * 128),
                "repeats": i + 1,
                "probability": 1.0 / (i + 1),
                "crop": i % 2 == 0,
                "crop_aspect": "square" if i % 3 == 0 else "random",
                "crop_style": "center" if i % 4 == 0 else "random",
                "minimum_image_size": 256,
                "maximum_image_size": 2048 + (i * 256),
                "target_downsample_size": 1.0 + (i * 0.25),
                "resolution_type": "pixel_area" if i % 2 == 0 else "pixel",
                "scan_for_errors": i % 3 == 0,
                "preserve_data_backend_cache": i % 4 == 0,
                "vae_cache_clear_each_epoch": i % 5 == 0,
            }

            # Add parquet config for parquet backends
            if i % 2 == 0:
                backend_config["parquet"] = {
                    "path": f"/tmp/data/complex/{i}/metadata.parquet",
                    "filename_column": "filename",
                    "caption_column": "caption",
                    "fallback_caption_column": "fallback_caption",
                    "width_column": "width",
                    "height_column": "height",
                    "aspect_ratio_column": "aspect_ratio",
                }

            # Add conditioning config to some backends
            if i >= 4:
                backend_config["conditioning"] = {
                    "type": "depth",
                    "conditioning_image_column": "conditioning_image",
                    "conditioning_data_root": f"/tmp/conditioning/{i}",
                }

            complex_config.append(backend_config)

        # Add reference datasets with dependencies
        complex_config.append(
            {
                "id": "reference-dataset",
                "type": "local",
                "instance_data_dir": "/tmp/data/reference",
                "metadata_backend": "discovery",
                "caption_strategy": "filename",
                "cache_dir_vae": "/tmp/cache/vae/reference",
                "resolution": 1024,
                "reference_dataset": "complex-dataset-0",
                "crop": True,
                "probability": 0.1,
            }
        )

        # Add text and image embed backends
        complex_config.extend(
            [
                {
                    "id": "text-embed-cache",
                    "dataset_type": "text_embeds",
                    "default": True,
                    "type": "local",
                    "cache_dir": "/tmp/cache/text",
                    "caption_filter_list": "/tmp/filter_list.txt",
                },
                {
                    "id": "image-embed-cache",
                    "dataset_type": "image_embeds",
                    "type": "local",
                    "cache_dir": "/tmp/cache/image_embeds",
                },
            ]
        )

        configs["complex_10_backends"] = ("Complex 10+ backend config", complex_config)

        return configs

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
        except RuntimeError:
            try:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except:
                return 0.0

    def benchmark_factory(self, factory_type: str, config_data: List[Dict[str, Any]], config_name: str) -> BenchmarkResult:
        """Benchmark a specific factory implementation."""
        result = BenchmarkResult(factory_type, len(json.dumps(config_data)), len(config_data))

        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config_data, f, indent=2)
                config_path = f.name

            # Start memory tracking
            tracemalloc.start()
            start_memory = self._get_memory_usage()
            start_time = time.time()

            # Create mock dependencies
            mock_accelerator = MagicMock()
            mock_accelerator.is_main_process = True
            mock_text_encoders = [MagicMock(), MagicMock()]
            mock_tokenizers = [MagicMock(), MagicMock()]
            mock_model = MockModelFoundation()

            # Mock StateTracker to avoid initialization issues
            from simpletuner.helpers.training.state_tracker import StateTracker

            StateTracker._accelerator = mock_accelerator
            # Also provide the get_accelerator method
            StateTracker.get_accelerator = MagicMock(return_value=mock_accelerator)

            args = MockArgs(config_path)

            if factory_type == "original":
                # Test original factory

                # Import and test original factory functions
                from simpletuner.helpers.data_backend.factory import (
                    sort_dataset_configs_by_dependencies,
                    fill_variables_in_config_paths,
                )

                init_start = time.time()

                # Load and process config (equivalent to factory initialization)
                with open(config_path, "r") as f:
                    data_backend_config = json.load(f)

                config_start = time.time()
                data_backend_config = sort_dataset_configs_by_dependencies(data_backend_config)
                data_backend_config = fill_variables_in_config_paths(args, data_backend_config)
                result.configuration_time = time.time() - config_start

                # For original factory, we can only test configuration processing
                # since full factory setup requires actual filesystem and model setup
                build_start = time.time()
                # Simulate some processing work equivalent to backend creation
                for backend in data_backend_config:
                    # Just iterate and do basic processing to simulate work
                    backend_id = backend.get("id", "unknown")
                    backend_type = backend.get("type", "unknown")
                result.build_time = time.time() - build_start
                result.initialization_time = time.time() - init_start

            else:
                # Test new factory

                from simpletuner.helpers.data_backend.factory import FactoryRegistry

                init_start = time.time()
                factory = FactoryRegistry(args, mock_accelerator, mock_text_encoders, mock_tokenizers, mock_model)
                result.initialization_time = time.time() - init_start

                config_start = time.time()
                data_backend_config = factory.load_configuration()
                result.configuration_time = time.time() - config_start

                build_start = time.time()
                # Test only configuration processing, not full backend building
                # which would require filesystem access
                data_backend_config = factory.process_conditioning_datasets(data_backend_config)

                # Simulate the configuration validation and processing that would happen
                for backend in data_backend_config:
                    backend_id = backend.get("id", "unknown")
                    backend_type = backend.get("type", "unknown")

                result.build_time = time.time() - build_start

                # Get metrics from the new factory
                if hasattr(factory, "metrics"):
                    result.peak_memory_mb = factory.metrics["memory_usage"]["peak"]
                    # Update timing from factory metrics if available
                    if factory.metrics.get("configuration_time"):
                        result.configuration_time = factory.metrics["configuration_time"]

            result.execution_time = time.time() - start_time

            # Update peak memory if not set by factory
            if result.peak_memory_mb == 0:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    result.peak_memory_mb = peak / 1024 / 1024
                except:
                    result.peak_memory_mb = self._get_memory_usage()

            tracemalloc.stop()

            # Clean up
            os.unlink(config_path)

        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error benchmarking {factory_type} factory: {e}")

            # Clean up on error
            try:
                os.unlink(config_path)
            except:
                pass

        return result

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""

        test_configs = self.create_test_configs()

        for config_key, (config_name, config_data) in test_configs.items():
            self.logger.info(f"\nTesting {config_name} ({len(config_data)} backends)...")

            # Test each factory type multiple times
            for factory_type in ["original", "new"]:
                iteration_results = []

                for iteration in range(self.iterations):

                    result = self.benchmark_factory(factory_type, config_data, config_name)
                    iteration_results.append(result)
                    self.results.append(result)

                    # Brief pause between iterations
                    time.sleep(0.1)

                # Log summary for this factory type
                if iteration_results and not iteration_results[0].error:
                    times = [r.execution_time for r in iteration_results if not r.error]
                    memories = [r.peak_memory_mb for r in iteration_results if not r.error]

                    if times:
                        avg_time = statistics.mean(times)
                        std_time = statistics.stdev(times) if len(times) > 1 else 0
                        avg_memory = statistics.mean(memories)

                        self.logger.info(
                            f"  {factory_type}: {avg_time:.4f}s ± {std_time:.4f}s, " f"memory: {avg_memory:.2f}MB"
                        )

        return self.results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate comparison report."""

        analysis = {"summary": {}, "detailed_results": [], "performance_comparison": {}, "recommendations": []}

        # Group results by configuration and factory type
        results_by_config = {}
        for result in self.results:
            if result.error:
                continue

            config_key = f"{result.backend_count}_backends"
            if config_key not in results_by_config:
                results_by_config[config_key] = {"original": [], "new": []}

            results_by_config[config_key][result.factory_type].append(result)

        # Analyze each configuration
        for config_key, factory_results in results_by_config.items():
            if not factory_results["original"] or not factory_results["new"]:
                continue

            original_times = [r.execution_time for r in factory_results["original"]]
            new_times = [r.execution_time for r in factory_results["new"]]
            original_memory = [r.peak_memory_mb for r in factory_results["original"]]
            new_memory = [r.peak_memory_mb for r in factory_results["new"]]

            original_avg_time = statistics.mean(original_times)
            new_avg_time = statistics.mean(new_times)
            original_avg_memory = statistics.mean(original_memory)
            new_avg_memory = statistics.mean(new_memory)

            time_improvement = ((original_avg_time - new_avg_time) / original_avg_time) * 100
            memory_change = ((new_avg_memory - original_avg_memory) / original_avg_memory) * 100

            config_analysis = {
                "config": config_key,
                "original": {
                    "avg_time": original_avg_time,
                    "avg_memory": original_avg_memory,
                    "std_time": statistics.stdev(original_times) if len(original_times) > 1 else 0,
                },
                "new": {
                    "avg_time": new_avg_time,
                    "avg_memory": new_avg_memory,
                    "std_time": statistics.stdev(new_times) if len(new_times) > 1 else 0,
                },
                "improvement": {
                    "time_percent": time_improvement,
                    "memory_percent": memory_change,
                    "faster": time_improvement > 0,
                },
            }

            analysis["performance_comparison"][config_key] = config_analysis

            self.logger.info(f"\n{config_key.replace('_', ' ').title()}:")
            self.logger.info(f"  Original: {original_avg_time:.4f}s, {original_avg_memory:.2f}MB")
            self.logger.info(f"  New:      {new_avg_time:.4f}s, {new_avg_memory:.2f}MB")
            self.logger.info(f"  Time:     {time_improvement:+.1f}% {'(faster)' if time_improvement > 0 else '(slower)'}")
            self.logger.info(f"  Memory:   {memory_change:+.1f}% {'(more)' if memory_change > 0 else '(less)'}")

        # Generate recommendations
        recommendations = []

        # Check for significant regressions
        regression_configs = [
            config
            for config, data in analysis["performance_comparison"].items()
            if data["improvement"]["time_percent"] < -5  # More than 5% slower
        ]

        if regression_configs:
            recommendations.append(f"Performance regression detected in: {', '.join(regression_configs)}")
        else:
            recommendations.append("No significant performance regressions detected")

        # Check memory usage
        memory_increases = [
            config
            for config, data in analysis["performance_comparison"].items()
            if data["improvement"]["memory_percent"] > 10  # More than 10% memory increase
        ]

        if memory_increases:
            recommendations.append(f"Significant memory increase in: {', '.join(memory_increases)}")
        else:
            recommendations.append("Memory usage is within acceptable bounds")

        # Check for errors
        error_count = len([r for r in self.results if r.error])
        if error_count > 0:
            recommendations.append(f"Found {error_count} errors during benchmarking - review required")
        else:
            recommendations.append("All benchmark tests completed successfully")

        analysis["recommendations"] = recommendations
        analysis["detailed_results"] = [r.to_dict() for r in self.results]

        return analysis

    def save_results(self, analysis: Dict[str, Any], output_file: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = os.path.join(os.path.dirname(__file__), output_file)

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        return output_path


def main():
    """Main benchmark execution."""
    # Parse environment variables
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", "3"))
    verbose = bool(os.environ.get("BENCHMARK_VERBOSE", ""))

    print("Factory Performance Benchmark")
    print("=" * 50)
    print(f"Iterations per test: {iterations}")
    print(f"Verbose output: {verbose}")
    print()

    # Run benchmark
    benchmark = FactoryBenchmark(iterations=iterations, verbose=verbose)
    results = benchmark.run_benchmarks()

    # Analyze and save results
    analysis = benchmark.analyze_results()
    output_file = benchmark.save_results(analysis)

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)

    for recommendation in analysis["recommendations"]:
        print(f"• {recommendation}")

    print(f"\nDetailed results saved to: {output_file}")

    # Determine overall readiness
    has_regressions = any(data["improvement"]["time_percent"] < -5 for data in analysis["performance_comparison"].values())

    has_errors = any(r["error"] for r in analysis["detailed_results"])

    if has_errors:
        print("\n❌ BENCHMARK FAILED: Errors detected during testing")
        return 1
    elif has_regressions:
        print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED: Review required before migration")
        return 2
    else:
        print("\n✅ PERFORMANCE BENCHMARKS PASSED: Ready for migration")
        return 0


if __name__ == "__main__":
    sys.exit(main())
