"""
Metrics management commands.

Handles metrics status, prometheus, health, and cost-related commands.
"""

import json

from .cloud.api import cloud_api_request


def cmd_metrics(args) -> int:
    """Manage metrics and monitoring."""
    metrics_action = getattr(args, "metrics_action", None)

    if metrics_action == "status":
        return _metrics_status(args)
    elif metrics_action == "health":
        return _metrics_health(args)
    elif metrics_action == "prometheus":
        return _metrics_prometheus(args)
    elif metrics_action == "costs":
        return _metrics_costs(args)
    elif metrics_action == "usage":
        return _metrics_usage(args)
    else:
        print("Error: Unknown metrics action. Use 'simpletuner metrics --help'.")
        return 1


def _metrics_status(args) -> int:
    """Show metrics system status."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/metrics/status")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Metrics System Status")
    print("=" * 50)

    enabled = result.get("enabled", False)
    print(f"Status: {'[+] Enabled' if enabled else '[-] Disabled'}")

    if result.get("prometheus_enabled"):
        print(f"Prometheus: [+] Enabled at {result.get('prometheus_endpoint', '/metrics')}")
    else:
        print("Prometheus: [-] Disabled")

    if result.get("collectors"):
        print("\nActive Collectors:")
        for collector in result.get("collectors", []):
            print(f"  - {collector}")

    return 0


def _metrics_health(args) -> int:
    """Show system health metrics."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/metrics/health")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("System Health")
    print("=" * 50)

    # Overall status
    healthy = result.get("healthy", False)
    print(f"Overall: {'[+] HEALTHY' if healthy else '[X] UNHEALTHY'}")

    # Component health
    components = result.get("components", {})
    if components:
        print("\nComponents:")
        for name, status in components.items():
            if isinstance(status, dict):
                comp_healthy = status.get("healthy", False)
                comp_status = "[+]" if comp_healthy else "[X]"
                detail = status.get("detail", "")
                print(f"  {comp_status} {name}: {detail}")
            else:
                print(f"  {name}: {status}")

    # Resource usage
    resources = result.get("resources", {})
    if resources:
        print("\nResources:")
        if "cpu_percent" in resources:
            print(f"  CPU: {resources['cpu_percent']:.1f}%")
        if "memory_percent" in resources:
            print(f"  Memory: {resources['memory_percent']:.1f}%")
        if "disk_percent" in resources:
            print(f"  Disk: {resources['disk_percent']:.1f}%")

    return 0 if healthy else 1


def _metrics_prometheus(args) -> int:
    """Show Prometheus metrics endpoint info or raw metrics."""
    raw = getattr(args, "raw", False)
    output_format = getattr(args, "format", "table")

    if raw:
        result = cloud_api_request("GET", "/api/metrics/prometheus")
        # Prometheus format is plain text
        print(result.get("metrics", "# No metrics available"))
        return 0

    result = cloud_api_request("GET", "/api/metrics/prometheus/info")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Prometheus Metrics")
    print("=" * 50)
    print(f"Endpoint: {result.get('endpoint', '/api/metrics/prometheus')}")
    print(f"Enabled: {'Yes' if result.get('enabled') else 'No'}")

    families = result.get("metric_families", [])
    if families:
        print(f"\nMetric Families ({len(families)}):")
        for family in families[:20]:  # Limit to first 20
            print(f"  - {family}")
        if len(families) > 20:
            print(f"  ... and {len(families) - 20} more")

    return 0


def _metrics_costs(args) -> int:
    """Show cost metrics and estimates."""
    output_format = getattr(args, "format", "table")
    period = getattr(args, "period", "month")

    result = cloud_api_request("GET", f"/api/metrics/costs?period={period}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Cost Metrics ({period})")
    print("=" * 50)

    # Summary
    summary = result.get("summary", {})
    if summary:
        total = summary.get("total_cost", 0)
        currency = summary.get("currency", "USD")
        print(f"\nTotal Cost: {currency} {total:.2f}")

        if summary.get("estimated_monthly"):
            print(f"Estimated Monthly: {currency} {summary['estimated_monthly']:.2f}")

    # By provider
    by_provider = result.get("by_provider", {})
    if by_provider:
        print("\nBy Provider:")
        for provider, cost in by_provider.items():
            print(f"  {provider}: ${cost:.2f}")

    # By job type
    by_type = result.get("by_job_type", {})
    if by_type:
        print("\nBy Job Type:")
        for job_type, cost in by_type.items():
            print(f"  {job_type}: ${cost:.2f}")

    # Recent jobs
    recent = result.get("recent_jobs", [])
    if recent:
        print("\nRecent Job Costs:")
        for job in recent[:5]:
            print(f"  {job.get('job_id', 'unknown')}: ${job.get('cost', 0):.2f}")

    return 0


def _metrics_usage(args) -> int:
    """Show resource usage metrics."""
    output_format = getattr(args, "format", "table")
    period = getattr(args, "period", "day")

    result = cloud_api_request("GET", f"/api/metrics/usage?period={period}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Resource Usage ({period})")
    print("=" * 50)

    # GPU hours
    gpu = result.get("gpu", {})
    if gpu:
        print("\nGPU Usage:")
        print(f"  Total Hours: {gpu.get('total_hours', 0):.2f}")
        if gpu.get("by_type"):
            print("  By GPU Type:")
            for gpu_type, hours in gpu.get("by_type", {}).items():
                print(f"    {gpu_type}: {hours:.2f} hours")

    # Storage
    storage = result.get("storage", {})
    if storage:
        print("\nStorage:")
        print(f"  Used: {_format_size(storage.get('used_bytes', 0))}")
        print(f"  Quota: {_format_size(storage.get('quota_bytes', 0))}")
        if storage.get("quota_bytes"):
            usage_pct = (storage.get("used_bytes", 0) / storage.get("quota_bytes", 1)) * 100
            print(f"  Usage: {usage_pct:.1f}%")

    # Jobs
    jobs = result.get("jobs", {})
    if jobs:
        print("\nJobs:")
        print(f"  Submitted: {jobs.get('submitted', 0)}")
        print(f"  Completed: {jobs.get('completed', 0)}")
        print(f"  Failed: {jobs.get('failed', 0)}")

    return 0


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
