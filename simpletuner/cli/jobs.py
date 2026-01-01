"""
Local job queue management commands.

Handles list, cancel, submit, delete, retry, logs, get, status, and approval for local jobs.
"""

import json
import sys
import time

from .cloud.api import cloud_api_request


def cmd_jobs(args) -> int:
    """Manage local job queue."""
    jobs_action = getattr(args, "jobs_action", None)

    # Default to list if no subcommand given
    if jobs_action is None or jobs_action == "list":
        return _jobs_list(args)
    elif jobs_action == "submit":
        return _jobs_submit(args)
    elif jobs_action == "cancel":
        return _jobs_cancel(args)
    elif jobs_action == "delete":
        return _jobs_delete(args)
    elif jobs_action == "retry":
        return _jobs_retry(args)
    elif jobs_action == "logs":
        return _jobs_logs(args)
    elif jobs_action == "get":
        return _jobs_get(args)
    elif jobs_action == "status":
        return _jobs_status(args)
    elif jobs_action == "purge":
        return _jobs_purge(args)
    elif jobs_action == "approval":
        return _jobs_approval(args)
    else:
        print("Error: Unknown jobs action. Use 'simpletuner jobs --help'.")
        return 1


# --- Approval Subcommand ---


def _jobs_approval(args) -> int:
    """Manage job approval workflow."""
    approval_action = getattr(args, "approval_action", None)

    if approval_action == "list":
        return _approval_list(args)
    elif approval_action == "pending":
        return _approval_pending(args)
    elif approval_action == "approve":
        return _approval_approve(args)
    elif approval_action == "reject":
        return _approval_reject(args)
    elif approval_action == "rules":
        return _approval_rules(args)
    else:
        print("Error: Unknown approval action. Use 'simpletuner jobs approval --help'.")
        return 1


def _approval_list(args) -> int:
    """List approval requests."""
    status_filter = getattr(args, "status", None)
    output_format = getattr(args, "format", "table")
    limit = getattr(args, "limit", 50)

    params = [f"limit={limit}"]
    if status_filter:
        params.append(f"status={status_filter}")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/approvals?{query}")
    approvals = result.get("approvals", [])

    if not approvals:
        print("No approval requests found.")
        return 0

    if output_format == "json":
        print(json.dumps(approvals, indent=2))
        return 0

    print(f"{'ID':<6} {'Job ID':<14} {'Status':<12} {'Config':<25} {'Requested':<20}")
    print("-" * 85)

    for a in approvals:
        approval_id = a.get("id", "-")
        job_id = (a.get("job_id") or "-")[:12]
        status = a.get("status", "unknown")
        config_name = (a.get("config_name") or "unnamed")[:24]
        requested_at = a.get("requested_at", "-")[:19]

        print(f"{approval_id:<6} {job_id:<14} {status:<12} {config_name:<25} {requested_at:<20}")

    return 0


def _approval_pending(args) -> int:
    """List only pending approval requests."""
    output_format = getattr(args, "format", "table")
    limit = getattr(args, "limit", 50)

    result = cloud_api_request("GET", f"/api/approvals?status=pending&limit={limit}")
    approvals = result.get("approvals", [])

    if not approvals:
        print("No pending approval requests.")
        return 0

    if output_format == "json":
        print(json.dumps(approvals, indent=2))
        return 0

    print(f"Pending Approvals ({len(approvals)})")
    print("=" * 70)

    for a in approvals:
        approval_id = a.get("id", "-")
        job_id = a.get("job_id") or "-"
        config_name = a.get("config_name") or "unnamed"
        reason = a.get("reason") or "No reason provided"
        requested_at = a.get("requested_at", "-")

        print(f"\n  ID: {approval_id}  |  Job: {job_id}")
        print(f"  Config: {config_name}")
        print(f"  Reason: {reason}")
        print(f"  Requested: {requested_at}")

    print("\n" + "-" * 70)
    print("Use 'simpletuner jobs approval approve <id>' to approve")
    print("Use 'simpletuner jobs approval reject <id>' to reject")

    return 0


def _approval_approve(args) -> int:
    """Approve a pending request."""
    approval_id = getattr(args, "approval_id", None)
    reason = getattr(args, "reason", None)

    if not approval_id:
        print("Error: Approval ID is required.")
        return 1

    data = {}
    if reason:
        data["reason"] = reason

    result = cloud_api_request(
        "POST",
        f"/api/approvals/{approval_id}/approve",
        data=data if data else None,
    )

    if result.get("success"):
        print(f"Approval {approval_id} approved.")
        if result.get("job_id"):
            print(f"  Job {result['job_id']} is now queued for execution.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to approve')}")
        return 1


def _approval_reject(args) -> int:
    """Reject a pending request."""
    approval_id = getattr(args, "approval_id", None)
    reason = getattr(args, "reason", None)

    if not approval_id:
        print("Error: Approval ID is required.")
        return 1

    data = {}
    if reason:
        data["reason"] = reason

    result = cloud_api_request(
        "POST",
        f"/api/approvals/{approval_id}/reject",
        data=data if data else None,
    )

    if result.get("success"):
        print(f"Approval {approval_id} rejected.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to reject')}")
        return 1


def _approval_rules(args) -> int:
    """List or manage approval rules."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/approvals/rules")
    rules = result.get("rules", [])

    if not rules:
        print("No approval rules configured.")
        print("\nApproval rules determine when jobs require manual approval before running.")
        return 0

    if output_format == "json":
        print(json.dumps(rules, indent=2))
        return 0

    print("Approval Rules")
    print("=" * 60)

    for rule in rules:
        rule_id = rule.get("id", "-")
        rule_type = rule.get("type", "unknown")
        condition = rule.get("condition", {})
        enabled = rule.get("enabled", True)

        status_icon = "[+]" if enabled else "[-]"

        print(f"\n{status_icon} Rule {rule_id}: {rule_type}")

        if isinstance(condition, dict):
            for key, value in condition.items():
                print(f"    {key}: {value}")
        else:
            print(f"    Condition: {condition}")

    return 0


def _format_duration(seconds) -> str:
    """Format seconds into a human-readable duration."""
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


def _format_job_status(status: str) -> str:
    """Format job status with indicator."""
    status_map = {
        "pending": "[.] pending",
        "queued": "[~] queued",
        "running": "[>] running",
        "completed": "[+] completed",
        "failed": "[X] failed",
        "cancelled": "[-] cancelled",
    }
    return status_map.get(status, f"[?] {status}")


def _jobs_list(args) -> int:
    """List local jobs."""
    limit = getattr(args, "limit", 20)
    status_filter = getattr(args, "status", None)
    output_format = getattr(args, "format", "table")
    output_fields = getattr(args, "output", None)

    params = [f"limit={limit}"]
    if status_filter:
        params.append(f"status={status_filter}")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/cloud/jobs?{query}")

    jobs = result.get("jobs", [])

    if not jobs:
        print("No local jobs found.")
        return 0

    if output_format == "json":
        if output_fields:
            fields = [f.strip() for f in output_fields.split(",")]
            filtered_jobs = [{k: j.get(k) for k in fields} for j in jobs]
            print(json.dumps(filtered_jobs, indent=2))
        else:
            print(json.dumps(jobs, indent=2))
        return 0

    # Custom field output
    if output_fields:
        fields = [f.strip() for f in output_fields.split(",")]
        _print_custom_fields(jobs, fields)
        return 0

    # Default table output
    print(f"{'Job ID':<14} {'Status':<15} {'Config':<20} {'Run Name':<25} {'Duration':<10}")
    print("-" * 90)

    for job in jobs:
        job_id = job.get("job_id", "")[:12]
        status = job.get("status", "unknown")
        config_name = (job.get("config_name") or "unnamed")[:19]
        run_name = _get_run_name(job)[:24]
        duration = _format_duration(job.get("duration_seconds"))

        print(f"{job_id:<14} {_format_job_status(status):<15} {config_name:<20} {run_name:<25} {duration:<10}")

    print(f"\nTotal: {len(jobs)} jobs")
    return 0


def _get_run_name(job: dict) -> str:
    """Extract run_name from job metadata."""
    metadata = job.get("metadata", {})
    return metadata.get("run_name", "-") if metadata else "-"


def _print_custom_fields(jobs: list, fields: list) -> None:
    """Print jobs with custom field selection."""
    # Field display config: (width, formatter)
    field_config = {
        "job_id": (14, lambda v: (v or "")[:12]),
        "status": (15, lambda v: _format_job_status(v or "unknown")),
        "config_name": (20, lambda v: (v or "unnamed")[:19]),
        "run_name": (25, lambda v: (v or "-")[:24]),
        "duration_seconds": (10, lambda v: _format_duration(v)),
        "duration": (10, lambda v: _format_duration(v)),
        "created_at": (20, lambda v: (v or "-")[:19]),
        "started_at": (20, lambda v: (v or "-")[:19]),
        "finished_at": (20, lambda v: (v or "-")[:19]),
        "job_type": (12, lambda v: v or "training"),
        "error_message": (40, lambda v: (v or "-")[:39]),
        "queue_position": (8, lambda v: str(v) if v is not None else "-"),
        "user_id": (10, lambda v: str(v) if v is not None else "-"),
    }

    # Build header
    header_parts = []
    for field in fields:
        width = field_config.get(field, (20, str))[0]
        header_parts.append(f"{field:<{width}}")
    print(" ".join(header_parts))
    print("-" * (sum(field_config.get(f, (20, str))[0] + 1 for f in fields) - 1))

    # Print rows
    for job in jobs:
        row_parts = []
        for field in fields:
            width, formatter = field_config.get(field, (20, lambda v: str(v) if v is not None else "-"))
            # Handle special fields
            if field == "run_name":
                value = _get_run_name(job)
            elif field == "duration" and job.get(field) is None:
                value = job.get("duration_seconds")
            else:
                value = job.get(field)
            formatted = formatter(value)
            row_parts.append(f"{formatted:<{width}}")
        print(" ".join(row_parts))

    print(f"\nTotal: {len(jobs)} jobs")


def _jobs_submit(args) -> int:
    """Submit a local training job."""
    config_name = getattr(args, "config", None)
    dry_run = getattr(args, "dry_run", False)
    no_wait = getattr(args, "no_wait", False)
    any_gpu = getattr(args, "any_gpu", False)

    if not config_name:
        print("Error: Config name is required.")
        print("Usage: simpletuner jobs submit <config_name>")
        return 1

    if dry_run:
        return _jobs_submit_dry_run(config_name, any_gpu=any_gpu)

    request_data = {
        "config_name": config_name,
        "no_wait": no_wait,
        "any_gpu": any_gpu,
    }

    print(f"Submitting local job with config '{config_name}'...")

    result = cloud_api_request("POST", "/api/queue/submit", data=request_data)

    if result.get("success"):
        job_id = result.get("job_id", "unknown")
        status = result.get("status", "unknown")
        allocated_gpus = result.get("allocated_gpus")
        queue_position = result.get("queue_position")

        print("Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")
        if allocated_gpus:
            print(f"  GPUs:   {allocated_gpus}")
        if queue_position is not None:
            print(f"  Queue Position: {queue_position}")
        return 0
    else:
        error = result.get("error") or result.get("detail") or result.get("reason") or "Unknown error"
        print(f"Error: {error}")
        return 1


def _jobs_submit_dry_run(config_name: str, *, any_gpu: bool = False) -> int:
    """Preview what would be submitted without actually submitting."""
    print(f"[DRY RUN] Previewing job submission for config '{config_name}'...")
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
    print(f"  Output Dir:      {config.get('output_dir', '-')}")
    print(f"  Resolution:      {config.get('resolution', '-')}")
    print(f"  Train Batch:     {config.get('train_batch_size', '-')}")
    print(f"  Gradient Accum:  {config.get('gradient_accumulation_steps', '-')}")
    print(f"  Max Steps:       {config.get('max_train_steps', '-')}")
    print(f"  Learning Rate:   {config.get('learning_rate', '-')}")
    print()

    # Get GPU requirements from config
    num_processes = config.get("num_processes") or config.get("--num_processes") or 1
    try:
        num_processes = int(num_processes)
    except (TypeError, ValueError):
        num_processes = 1
    configured_gpus = config.get("accelerate_visible_devices") or config.get("--accelerate_visible_devices")

    print("GPU Requirements:")
    print("=" * 50)
    print(f"  num_processes:   {num_processes}")
    print(f"  configured_gpus: {configured_gpus or 'auto'}")
    if any_gpu:
        print(f"  --any-gpu:       enabled (will use any available GPUs)")
    print()

    # Get GPU status
    try:
        gpu_result = cloud_api_request("GET", "/api/queue/local/gpu-status")
        total_gpus = gpu_result.get("total_gpus", 0)
        allocated_gpus = gpu_result.get("allocated_gpus", [])
        available_gpus = gpu_result.get("available_gpus", [])
        devices = gpu_result.get("devices", [])
        running_local_jobs = gpu_result.get("running_local_jobs", 0)

        print("Available GPUs:")
        print("=" * 50)
        for device in devices:
            idx = device.get("index", 0)
            name = device.get("name", f"GPU {idx}")
            allocated = device.get("allocated", False)
            job_id = device.get("job_id")
            status = f"[allocated - job {job_id}]" if allocated else "[available]"
            print(f"  GPU {idx}: {name} {status}")
        if not devices:
            print("  No GPUs detected")
        print()

        print("GPU Status:")
        print("=" * 50)
        print(f"  Total GPUs:         {total_gpus}")
        print(f"  Available GPUs:     {len(available_gpus)}")
        print(f"  Running local jobs: {running_local_jobs}")

        # Determine if job can start
        can_start = len(available_gpus) >= num_processes
        if can_start:
            gpus_to_use = available_gpus[:num_processes]
            print(f"  [+] Job can start immediately using GPUs: {gpus_to_use}")
        else:
            print(f"  [.] Job will be queued (need {num_processes} GPUs, {len(available_gpus)} available)")
        print()
    except SystemExit:
        print("GPU Status:")
        print("=" * 50)
        print("  Unable to fetch GPU status")
        print()

    # Get queue status
    try:
        queue_result = cloud_api_request("GET", "/api/queue/stats")
        queue_depth = queue_result.get("queue_depth", 0)
        running = queue_result.get("running", 0)
        max_concurrent = queue_result.get("max_concurrent", 5)

        print("Queue Status:")
        print("=" * 50)
        print(f"  Queue Depth:     {queue_depth}")
        print(f"  Running:         {running} / {max_concurrent}")

        if running >= max_concurrent:
            print(f"  Note:            Job will be queued (at capacity)")
        elif queue_depth > 0:
            print(f"  Note:            Job will be queued behind {queue_depth} other(s)")
        else:
            print(f"  Note:            Job will start immediately")
        print()
    except SystemExit:
        pass

    # Validate config
    try:
        validation_result = cloud_api_request("POST", f"/api/configs/{config_name}/validate")
        is_valid = validation_result.get("valid", True)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])

        print("Validation:")
        print("=" * 50)
        if is_valid and not errors:
            print("  [+] Config is valid")
        else:
            print("  [X] Config has errors:")
            for err in errors:
                print(f"      - {err}")

        if warnings:
            print("  Warnings:")
            for warn in warnings:
                print(f"      - {warn}")
        print()
    except SystemExit:
        pass

    print("-" * 50)
    print("To submit this job, run without --dry-run:")
    print(f"  simpletuner jobs submit {config_name}")

    return 0


def _jobs_cancel(args) -> int:
    """Cancel a local job or bulk cancel by status."""
    job_id = getattr(args, "job_id", None)
    status_filter = getattr(args, "status", None)
    cancel_all = getattr(args, "all", False)
    force = getattr(args, "force", False)

    # Bulk cancel mode
    if cancel_all or status_filter:
        if cancel_all:
            statuses = ["running", "pending", "queued"]
        else:
            statuses = [status_filter]

        # Fetch jobs matching the statuses
        jobs_to_cancel = []
        for status in statuses:
            offset = 0
            while True:
                result = cloud_api_request("GET", f"/api/cloud/jobs?status={status}&limit=500&offset={offset}")
                batch = result.get("jobs", [])
                if not batch:
                    break
                jobs_to_cancel.extend(batch)
                if len(batch) < 500:
                    break
                offset += 500

        if not jobs_to_cancel:
            print(f"No jobs found with status: {', '.join(statuses)}")
            return 0

        print(f"Found {len(jobs_to_cancel)} job(s) to cancel:")
        for status in statuses:
            count = sum(1 for j in jobs_to_cancel if j.get("status") == status)
            if count > 0:
                print(f"  {status}: {count}")

        if not force:
            confirm = input("\nCancel these jobs? [y/N] ")
            if confirm.lower() not in ("y", "yes"):
                print("Cancelled.")
                return 0

        print("\nCancelling jobs...")
        cancelled = 0
        failed = 0

        for job in jobs_to_cancel:
            jid = job.get("job_id")
            if not jid:
                continue

            result = cloud_api_request("POST", f"/api/cloud/jobs/{jid}/cancel")
            if result.get("success"):
                cancelled += 1
            else:
                failed += 1
                print(f"  Failed to cancel {jid}: {result.get('detail', 'Unknown error')}")

        print(f"\nCancelled {cancelled} job(s)")
        if failed > 0:
            print(f"Failed to cancel {failed} job(s)")
            return 1

        return 0

    # Single job cancel mode
    if not job_id:
        print("Error: Job ID is required, or use --all or --status to bulk cancel.")
        print("Usage: simpletuner jobs cancel <job_id>")
        print("       simpletuner jobs cancel --status running")
        print("       simpletuner jobs cancel --all")
        return 1

    print(f"Cancelling job {job_id}...")

    result = cloud_api_request("POST", f"/api/cloud/jobs/{job_id}/cancel")

    if result.get("success"):
        print(f"Job {job_id} cancelled successfully.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to cancel job')}")
        return 1


def _jobs_delete(args) -> int:
    """Delete a job from the queue."""
    job_id = getattr(args, "job_id", None)
    force = getattr(args, "force", False)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner jobs delete <job_id>")
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
        print(f"Error: {result.get('detail', 'Failed to delete job')}")
        return 1


def _jobs_retry(args) -> int:
    """Retry a failed or cancelled job."""
    job_id = getattr(args, "job_id", None)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner jobs retry <job_id>")
        return 1

    result = cloud_api_request("POST", f"/api/cloud/jobs/{job_id}/retry")

    if result.get("success"):
        new_job_id = result.get("job_id", "unknown")
        print("Job resubmitted successfully!")
        print(f"  New Job ID: {new_job_id}")
        print(f"  Status: {result.get('status', 'unknown')}")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to retry job')}")
        return 1


def _jobs_logs(args) -> int:
    """Fetch logs for a job."""
    job_id = getattr(args, "job_id", None)
    follow = getattr(args, "follow", False)

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner jobs logs <job_id>")
        return 1

    if follow:
        return _jobs_logs_follow(job_id)

    result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}/logs")

    logs = result.get("logs", "")

    if not logs:
        print("No logs available for this job.")
        return 0

    print(logs)
    return 0


def _jobs_logs_follow(job_id: str) -> int:
    """Follow job logs in real-time via SSE streaming."""
    import os
    import ssl
    import urllib.error
    import urllib.request

    from .cloud.api import get_cloud_server_url

    print(f"Following logs for job {job_id}... (Ctrl+C to stop)")
    print("-" * 60)

    base_url = get_cloud_server_url()
    stream_url = f"{base_url}/api/cloud/jobs/{job_id}/logs/stream"

    # Build request with auth
    req = urllib.request.Request(stream_url)
    req.add_header("Accept", "text/event-stream")
    req.add_header("Cache-Control", "no-cache")

    api_key = os.environ.get("SIMPLETUNER_API_KEY")
    if api_key:
        req.add_header("X-API-Key", api_key)

    # Handle SSL
    ssl_context = None
    if stream_url.startswith("https"):
        if os.environ.get("SIMPLETUNER_SSL_NO_VERIFY") == "true":
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=None, context=ssl_context) as response:
            buffer = ""
            while True:
                chunk = response.read(1)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8", errors="replace")

                # Process complete SSE messages
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    for line in message.split("\n"):
                        if line.startswith("data: "):
                            data = line[6:]
                            print(data)
                        elif line.startswith("event: done"):
                            # Next data line contains the final status
                            pass
                        elif line == "event: done":
                            continue

                    # Check for done event
                    if "event: done" in message:
                        # Extract status from the data line
                        for line in message.split("\n"):
                            if line.startswith("data: "):
                                status = line[6:]
                                if status != "stream_end":
                                    print(f"\n--- Job {status} ---")
                        return 0

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Error: Job {job_id} not found.")
        else:
            print(f"Error: HTTP {e.code} - {e.reason}")
        return 1
    except KeyboardInterrupt:
        print("\n\nStopped following logs.")
        return 0
    except Exception as e:
        print(f"\nError following logs: {e}")
        return 1

    return 0


def _jobs_get(args) -> int:
    """Get details for a specific job."""
    job_id = getattr(args, "job_id", None)
    output_format = getattr(args, "format", "table")

    if not job_id:
        print("Error: Job ID is required.")
        print("Usage: simpletuner jobs get <job_id>")
        return 1

    result = cloud_api_request("GET", f"/api/cloud/jobs/{job_id}")
    job = result.get("job", result)

    if output_format == "json":
        print(json.dumps(job, indent=2))
        return 0

    print(f"Job: {job.get('job_id', 'unknown')}")
    print("=" * 50)
    print(f"Status:      {_format_job_status(job.get('status', 'unknown'))}")
    print(f"Config:      {job.get('config_name') or 'unnamed'}")
    print(f"Type:        {job.get('job_type', 'training')}")
    print(f"Created:     {job.get('created_at', '-')}")

    if job.get("started_at"):
        print(f"Started:     {job['started_at']}")
    if job.get("finished_at"):
        print(f"Finished:    {job['finished_at']}")

    print(f"Duration:    {_format_duration(job.get('duration_seconds'))}")

    if job.get("error_message"):
        print(f"\nError: {job['error_message']}")

    if job.get("queue_position"):
        print(f"\nQueue Position: {job['queue_position']}")

    return 0


def _jobs_status(args) -> int:
    """Get job queue status."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/queue/stats")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Job Queue Status")
    print("=" * 50)

    by_status = result.get("by_status", {})
    print(f"\nQueue Depth:     {result.get('queue_depth', 0)}")
    print(f"Running:         {result.get('running', 0)}")
    print(f"Pending:         {by_status.get('pending', 0)}")
    print(f"Completed:       {by_status.get('completed', 0)}")
    print(f"Failed:          {by_status.get('failed', 0)}")

    avg_wait = result.get("avg_wait_seconds")
    if avg_wait is not None:
        print(f"\nAvg Wait Time:   {_format_duration(avg_wait)}")

    print(f"\nConcurrency Limits:")
    print(f"  Max Concurrent:      {result.get('max_concurrent', 5)}")
    print(f"  User Max Concurrent: {result.get('user_max_concurrent', 2)}")
    print(f"  Team Max Concurrent: {result.get('team_max_concurrent', 10)}")
    print(f"  Fair Share:          {'Enabled' if result.get('enable_fair_share') else 'Disabled'}")

    return 0


def _jobs_purge(args) -> int:
    """Bulk delete jobs by status."""
    statuses = getattr(args, "status", None) or []
    purge_all = getattr(args, "all", False)
    force = getattr(args, "force", False)

    if purge_all:
        statuses = ["pending", "queued", "completed", "failed", "cancelled"]
    elif not statuses:
        print("Error: Specify --status or --all")
        print("Usage: simpletuner jobs purge --status failed --status cancelled")
        print("       simpletuner jobs purge --all")
        return 1

    # Fetch jobs matching the statuses (paginate if needed)
    jobs_to_delete = []
    for status in statuses:
        offset = 0
        while True:
            result = cloud_api_request("GET", f"/api/cloud/jobs?status={status}&limit=500&offset={offset}")
            batch = result.get("jobs", [])
            if not batch:
                break
            jobs_to_delete.extend(batch)
            if len(batch) < 500:
                break
            offset += 500

    if not jobs_to_delete:
        print(f"No jobs found with status: {', '.join(statuses)}")
        return 0

    print(f"Found {len(jobs_to_delete)} job(s) to delete:")
    for status in statuses:
        count = sum(1 for j in jobs_to_delete if j.get("status") == status)
        if count > 0:
            print(f"  {status}: {count}")

    if not force:
        confirm = input("\nDelete these jobs? This cannot be undone. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    print("\nDeleting jobs...")
    deleted = 0
    failed = 0

    for job in jobs_to_delete:
        job_id = job.get("job_id")
        job_status = job.get("status")
        if not job_id:
            continue

        # Cancel pending/queued jobs first
        if job_status in ("pending", "queued"):
            cancel_result = cloud_api_request("POST", f"/api/cloud/jobs/{job_id}/cancel")
            if not cancel_result.get("success"):
                failed += 1
                print(f"  Failed to cancel {job_id}: {cancel_result.get('detail', 'Unknown error')}")
                continue

        result = cloud_api_request("DELETE", f"/api/cloud/jobs/{job_id}")
        if result.get("success"):
            deleted += 1
        else:
            failed += 1
            print(f"  Failed to delete {job_id}: {result.get('detail', 'Unknown error')}")

    print(f"\nDeleted {deleted} job(s)")
    if failed > 0:
        print(f"Failed to delete {failed} job(s)")
        return 1

    return 0
