"""Memory debugging utilities for tracking RAM usage."""

import gc
import logging
import sys
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


def get_ram_usage_gb():
    """Get current process RAM usage in GB."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except:
        return 0.0


def analyze_cpu_tensors(min_size_mb: float = 10.0):
    """
    Analyze all CPU tensors in memory.

    Args:
        min_size_mb: Only report tensors larger than this size.

    Returns:
        Dict with analysis results.
    """
    gc.collect()

    results = {
        "total_cpu_mb": 0,
        "total_gpu_mb": 0,
        "pinned_mb": 0,
        "by_dtype": defaultdict(float),
        "by_shape": defaultdict(float),
        "large_tensors": [],
    }

    seen_ids = set()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size_mb = obj.numel() * obj.element_size() / (1024 * 1024)
                dtype_str = str(obj.dtype)

                if obj.device.type == "cpu":
                    results["total_cpu_mb"] += size_mb
                    results["by_dtype"][dtype_str] += size_mb

                    if obj.is_pinned():
                        results["pinned_mb"] += size_mb

                    if size_mb >= min_size_mb:
                        shape_str = str(tuple(obj.shape))
                        results["by_shape"][shape_str] += size_mb
                        results["large_tensors"].append(
                            {
                                "shape": tuple(obj.shape),
                                "dtype": dtype_str,
                                "size_mb": size_mb,
                                "pinned": obj.is_pinned(),
                                "requires_grad": obj.requires_grad,
                            }
                        )
                else:
                    results["total_gpu_mb"] += size_mb
        except:
            pass

    # Sort large tensors by size
    results["large_tensors"].sort(key=lambda x: x["size_mb"], reverse=True)

    return results


def log_memory_summary(label: str = ""):
    """Log a summary of current memory usage."""
    ram_gb = get_ram_usage_gb()

    analysis = analyze_cpu_tensors(min_size_mb=100.0)

    logger.info(f"=== Memory Summary{' (' + label + ')' if label else ''} ===")
    logger.info(f"Process RAM: {ram_gb:.2f} GB")
    logger.info(f"CPU tensors: {analysis['total_cpu_mb'] / 1024:.2f} GB (pinned: {analysis['pinned_mb'] / 1024:.2f} GB)")
    logger.info(f"GPU tensors: {analysis['total_gpu_mb'] / 1024:.2f} GB")

    if analysis["by_dtype"]:
        logger.info("By dtype:")
        for dtype, size_mb in sorted(analysis["by_dtype"].items(), key=lambda x: -x[1]):
            logger.info(f"  {dtype}: {size_mb / 1024:.2f} GB")

    if analysis["large_tensors"]:
        logger.info(f"Large CPU tensors (>100MB, top 10):")
        for i, t in enumerate(analysis["large_tensors"][:10]):
            logger.info(
                f"  {t['shape']}: {t['size_mb']:.1f} MB ({t['dtype']}, pinned={t['pinned']}, grad={t['requires_grad']})"
            )

    return analysis
