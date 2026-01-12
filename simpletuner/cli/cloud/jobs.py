"""
Cloud job management commands.

Handles submit, list, cancel, delete, retry, logs, get, and status commands.
"""

import json
import sys
import time
from typing import Any, Dict

from ..common import format_cost, format_duration
from .api import cloud_api_request, format_job_status


def cmd_cloud_submit(args) -> int:
    """Submit a training job to the cloud."""
    config_name = getattr(args, "config", None)
    provider = getattr(args, "provider", "replicate")
    idempotency_key = getattr(args, "idempotency_key", None)
    dry_run = getattr(args, "dry_run", False)

    if not config_name:
        print("Error: Config name is required.")
        print("Usage: simpletuner cloud submit <config_name>")
        return 1

    if dry_run:
        return _cloud_submit_dry_run(config_name, provider)

    request_data: Dict[str, Any] = {
        "config_name_to_load": config_name,
    }
    if idempotency_key:
        request_data["idempotency_key"] = idempotency_key

    print(f"Submitting job with config '{config_name}' to {provider}...")

    result = cloud_api_request(
        "POST",
        f"/api/cloud/jobs/submit?provider={provider}",
        data=request_data,
    )

    if result.get("success"):
        job_id = result.get("job_id", "unknown")
        status = result.get("status", "unknown")
        print("Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")

        if result.get("idempotent_hit"):
            print("  Note: Returned existing job (matched idempotency key)")

        if result.get("data_uploaded"):
            print("  Data: Uploaded")

        if result.get("cost_limit_warning"):
            print(f"  Warning: {result['cost_limit_warning']}")

        for warning in result.get("quota_warnings", []):
            print(f"  Warning: {warning}")

        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1


def _cloud_submit_dry_run(config_name: str, provider: str) -> int:
    """Preview cloud job submission without actually submitting."""
    print(f"[DRY RUN] Previewing cloud job submission for config '{config_name}'...")
    print(f"          Provider: {provider}")
    print()

    # Get config details
    try:
        config_result = cloud_api_request("GET", f"/api/configs/{config_name}")
    except SystemExit:
        print("Error: Could not fetch config details.")
        return 1

    config = config_result.get("config", config_result)
    if not config:
        print(f"Error: Config '{config_name}' not found.")
        return 1

    # Display config summary
    print("Configuration:")
    print("=" * 50)
    print(f"  Name:            {config_name}")
    print(f"  Model Type:      {config.get('model_type', '-')}")
    print(f"  Model Family:    {config.get('model_family', '-')}")
    print(f"  Resolution:      {config.get('resolution', '-')}")
    print(f"  Train Batch:     {config.get('train_batch_size', '-')}")
    print(f"  Gradient Accum:  {config.get('gradient_accumulation_steps', '-')}")
    print(f"  Max Steps:       {config.get('max_train_steps', '-')}")
    print(f"  Learning Rate:   {config.get('learning_rate', '-')}")
    print()

    # Check provider token
    try:
        validate_result = cloud_api_request("GET", f"/api/cloud/{provider}/validate")
        if validate_result.get("valid"):
            user_info = validate_result.get("user_info", {})
            username = user_info.get("username", "unknown")
            print(f"Provider Authentication:")
            print("=" * 50)
            print(f"  [{provider.upper()}] Authenticated as: {username}")
            print()
        else:
            print(f"Provider Authentication:")
            print("=" * 50)
            print(f"  [X] {provider.upper()}: {validate_result.get('error', 'Not authenticated')}")
            print()
            print("Set up credentials with:")
            print(f"  simpletuner cloud config set-token --provider {provider}")
            return 1
    except SystemExit:
        pass

    # Get dataloader config from the config
    dataloader_config = config.get("dataloader_config", [])

    # Preview data upload
    if dataloader_config:
        try:
            preview_result = cloud_api_request(
                "POST",
                "/api/cloud/data-consent/preview",
                data=dataloader_config,
            )

            print("Data Upload Preview:")
            print("=" * 50)

            consent_mode = preview_result.get("consent_mode", "deny")
            requires_upload = preview_result.get("requires_upload", False)
            datasets = preview_result.get("datasets", [])
            total_files = preview_result.get("total_files", 0)
            total_size_mb = preview_result.get("total_size_mb", 0.0)

            print(f"  Consent Mode:    {consent_mode}")
            print(f"  Requires Upload: {'Yes' if requires_upload else 'No'}")

            if requires_upload and datasets:
                print(f"  Total Files:     {total_files}")
                print(f"  Total Size:      {total_size_mb:.2f} MB")
                print()
                print("  Datasets:")
                for ds in datasets:
                    ds_id = ds.get("id", "unknown")
                    ds_files = ds.get("file_count", 0)
                    ds_size = ds.get("total_size_mb", 0.0)
                    print(f"    - {ds_id}: {ds_files} files ({ds_size:.2f} MB)")
            elif preview_result.get("message"):
                print(f"  Message:         {preview_result['message']}")
            print()
        except SystemExit:
            pass

    # Get provider config for hardware info
    try:
        provider_config = cloud_api_request("GET", f"/api/cloud/providers/{provider}/config")
        config_data = provider_config.get("config", {})
        hardware_info = config_data.get("hardware_info", {})

        if hardware_info:
            print("Available Hardware:")
            print("=" * 50)
            for hw_id, hw_data in hardware_info.items():
                name = hw_data.get("name", hw_id)
                cost = hw_data.get("cost_per_second", 0)
                print(f"  {hw_id}: {name} (${cost:.4f}/sec)")
            print()
    except SystemExit:
        pass

    # Cost estimation (simplified)
    try:
        max_steps = config.get("max_train_steps", 0)
        if max_steps:
            # Rough estimate: assume ~1 second per step on typical hardware
            estimated_seconds = max_steps * 1.5  # conservative estimate
            estimated_cost_low = estimated_seconds * 0.001  # cheap hardware
            estimated_cost_high = estimated_seconds * 0.002  # expensive hardware

            print("Cost Estimate (rough):")
            print("=" * 50)
            print(f"  Max Steps:       {max_steps}")
            print(f"  Est. Duration:   {format_duration(estimated_seconds)}")
            print(f"  Est. Cost:       ${estimated_cost_low:.2f} - ${estimated_cost_high:.2f}")
            print()
    except Exception:
        pass

    print("-" * 50)
    print("To submit this job, run without --dry-run:")
    print(f"  simpletuner cloud jobs submit {config_name} --provider {provider}")

    return 0


def cmd_cloud_list(args) -> int:
    """List cloud training jobs."""
    limit = getattr(args, "limit", 20)
    status_filter = getattr(args, "status", None)
    provider = getattr(args, "provider", None)
    sync = getattr(args, "sync", False)
    output_format = getattr(args, "format", "table")

    params = [f"limit={limit}"]
    if status_filter:
        params.append(f"status={status_filter}")
    if provider:
        params.append(f"provider={provider}")
    if sync:
        params.append("sync_active=true")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/cloud/jobs?{query}")

    jobs = result.get("jobs", [])

    if not jobs:
        print("No jobs found.")
        return 0

    if output_format == "json":
        print(json.dumps(jobs, indent=2))
        return 0

    print(f"{'Job ID':<14} {'Status':<12} {'Config':<25} {'Provider':<10} {'Cost':<8} {'Duration':<10}")
    print("-" * 85)

    for job in jobs:
        job_id = job.get("job_id", "")[:12]
        status = job.get("status", "unknown")
        config_name = (job.get("config_name") or "unnamed")[:24]
        job_provider = job.get("provider") or "local"
        cost = format_cost(job.get("cost_usd"))
        duration = format_duration(job.get("duration_seconds"))

        print(f"{job_id:<14} {format_job_status(status):<12} {config_name:<25} {job_provider:<10} {cost:<8} {duration:<10}")

    print(f"\nTotal: {len(jobs)} jobs")
    return 0


def cmd_cloud_cancel(args) -> int:
    """Cancel a running cloud job."""
    job_id = getattr(args, "job_id", None)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner cloud cancel <job_id>")
        return 1

    print(f"Cancelling job {job_id}...")

    result = cloud_api_request("POST", f"/api/cloud/jobs/{job_id}/cancel")

    if result.get("success"):
        print(f"Job {job_id} cancelled successfully.")
        return 0
    else:
        print(f"Error: Failed to cancel job {job_id}")
        return 1


def cmd_cloud_delete(args) -> int:
    """Delete a job from local history."""
    job_id = getattr(args, "job_id", None)
    force = getattr(args, "force", False)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner cloud delete <job_id>")
        return 1

    if not force:
        confirm = input(f"Delete job {job_id}? This cannot be undone. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    print(f"Deleting job {job_id}...")

    result = cloud_api_request("DELETE", f"/api/cloud/jobs/{job_id}")

    if result.get("success"):
        print(f"Job {job_id} deleted successfully.")
        return 0
    else:
        print(f"Error: Failed to delete job {job_id}")
        return 1


def cmd_cloud_retry(args) -> int:
    """Retry a failed job by resubmitting with the same configuration."""
    job_id = getattr(args, "job_id", None)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner cloud retry <job_id>")
        return 1

    job_result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}")
    job = job_result.get("job", {})

    if not job:
        print(f"Error: Job {job_id} not found.")
        return 1

    status = job.get("status", "")
    if status not in ("failed", "cancelled"):
        print(f"Error: Can only retry failed or cancelled jobs. Job status is '{status}'.")
        return 1

    config_name = job.get("config_name")
    if not config_name:
        print("Error: Job has no associated config name. Cannot retry.")
        return 1

    provider = job.get("provider", "replicate")

    print(f"Retrying job with config '{config_name}' on {provider}...")

    request_data = {"config_name_to_load": config_name}
    result = cloud_api_request(
        "POST",
        f"/api/cloud/jobs/submit?provider={provider}",
        data=request_data,
    )

    if result.get("success"):
        new_job_id = result.get("job_id", "unknown")
        print("Job resubmitted successfully!")
        print(f"  New Job ID: {new_job_id}")
        print(f"  Status: {result.get('status', 'unknown')}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1


def cmd_cloud_logs(args) -> int:
    """Fetch logs for a cloud job."""
    job_id = getattr(args, "job_id", None)
    follow = getattr(args, "follow", False)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner cloud logs <job_id>")
        return 1

    if follow:
        return _cloud_logs_follow(job_id)

    result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}/logs")

    logs = result.get("logs", "")

    if not logs:
        print("No logs available for this job.")
        return 0

    print(logs)
    return 0


def _cloud_logs_follow(job_id: str) -> int:
    """Follow job logs in real-time by polling."""
    print(f"Following logs for job {job_id}... (Ctrl+C to stop)")
    print("-" * 60)

    last_log_length = 0
    terminal_states = {"completed", "failed", "cancelled"}

    try:
        while True:
            job_result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}")
            job = job_result.get("job", {})

            if not job:
                print(f"\nError: Job {job_id} not found.")
                return 1

            log_result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}/logs")
            logs = log_result.get("logs", "")

            if len(logs) > last_log_length:
                new_content = logs[last_log_length:]
                sys.stdout.write(new_content)
                sys.stdout.flush()
                last_log_length = len(logs)

            status = job.get("status", "")
            if status in terminal_states:
                print(f"\n\n--- Job {status} ---")
                break

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nStopped following logs.")
        return 0
    except Exception as e:
        print(f"\nError following logs: {e}")
        return 1

    return 0


def cmd_cloud_get(args) -> int:
    """Get details for a specific job."""
    job_id = getattr(args, "job_id", None)
    output_format = getattr(args, "format", "table")

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner cloud get <job_id>")
        return 1

    result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}")
    job = result.get("job", {})

    if output_format == "json":
        print(json.dumps(job, indent=2))
        return 0

    print(f"Job: {job.get('job_id', 'unknown')}")
    print("=" * 50)
    print(f"Status:      {format_job_status(job.get('status', 'unknown'))}")
    print(f"Config:      {job.get('config_name') or 'unnamed'}")
    print(f"Provider:    {job.get('provider') or 'local'}")
    print(f"Type:        {job.get('job_type', 'unknown')}")
    print(f"Created:     {job.get('created_at', '-')}")

    if job.get("started_at"):
        print(f"Started:     {job['started_at']}")
    if job.get("finished_at"):
        print(f"Finished:    {job['finished_at']}")

    print(f"Duration:    {format_duration(job.get('duration_seconds'))}")
    print(f"Cost:        {format_cost(job.get('cost_usd'))}")

    if job.get("error_message"):
        print(f"\nError: {job['error_message']}")

    if job.get("queue_position"):
        print(f"\nQueue Position: {job['queue_position']}")
        if job.get("estimated_wait_seconds"):
            print(f"Estimated Wait: {format_duration(job['estimated_wait_seconds'])}")

    return 0


def cmd_cloud_status(args) -> int:
    """Get cloud system status."""
    include_replicate = getattr(args, "replicate", False)
    output_format = getattr(args, "format", "table")

    health_params = "include_replicate=true" if include_replicate else ""
    health = cloud_api_request("GET", f"/api/cloud/health?{health_params}")

    try:
        system_status = cloud_api_request("GET", "/api/cloud/replicate/status")
    except SystemExit:
        system_status = {"error": "Could not fetch Replicate status"}

    if output_format == "json":
        print(json.dumps({"health": health, "system": system_status}, indent=2))
        return 0

    print("SimpleTuner Cloud Status")
    print("=" * 50)

    status_emoji = {"healthy": "[+]", "degraded": "[!]", "unhealthy": "[X]"}
    overall = health.get("status", "unknown")
    print(f"\nOverall Status: {status_emoji.get(overall, '[?]')} {overall.upper()}")

    if health.get("uptime_seconds"):
        uptime = format_duration(health["uptime_seconds"])
        print(f"Uptime: {uptime}")

    components = health.get("components", [])
    if components:
        print("\nComponents:")
        for comp in components:
            name = comp.get("name", "unknown")
            comp_status = comp.get("status", "unknown")
            message = comp.get("message", "")
            latency = comp.get("latency_ms")

            latency_str = f" ({latency:.0f}ms)" if latency else ""
            print(f"  {status_emoji.get(comp_status, '[?]')} {name}: {message}{latency_str}")

    if not system_status.get("error"):
        print("\nReplicate Status:")
        if system_status.get("operational"):
            print("  [+] All systems operational")
        else:
            incidents = system_status.get("ongoing_incidents", [])
            maintenances = system_status.get("in_progress_maintenances", [])
            if incidents:
                print(f"  [!] {len(incidents)} ongoing incident(s)")
            if maintenances:
                print(f"  [~] {len(maintenances)} maintenance(s) in progress")

        scheduled = system_status.get("scheduled_maintenances", [])
        if scheduled:
            print(f"  [i] {len(scheduled)} scheduled maintenance(s)")

    return 0
