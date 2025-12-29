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
        print(json.dumps(jobs, indent=2))
        return 0

    print(f"{'Job ID':<14} {'Status':<15} {'Config':<30} {'Duration':<10}")
    print("-" * 75)

    for job in jobs:
        job_id = job.get("job_id", "")[:12]
        status = job.get("status", "unknown")
        config_name = (job.get("config_name") or "unnamed")[:29]
        duration = _format_duration(job.get("duration_seconds"))

        print(f"{job_id:<14} {_format_job_status(status):<15} {config_name:<30} {duration:<10}")

    print(f"\nTotal: {len(jobs)} jobs")
    return 0


def _jobs_submit(args) -> int:
    """Submit a local training job."""
    config_name = getattr(args, "config", None)
    dry_run = getattr(args, "dry_run", False)

    if not config_name:
        print("Error: Config name is required.")
        print("Usage: simpletuner jobs submit <config_name>")
        return 1

    if dry_run:
        return _jobs_submit_dry_run(config_name)

    request_data = {
        "config_name": config_name,
    }

    print(f"Submitting local job with config '{config_name}'...")

    result = cloud_api_request("POST", "/api/cloud/jobs/submit", data=request_data)

    if result.get("success"):
        job_id = result.get("job_id", "unknown")
        status = result.get("status", "unknown")
        print("Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")
        return 0
    else:
        print(f"Error: {result.get('error', result.get('detail', 'Unknown error'))}")
        return 1


def _jobs_submit_dry_run(config_name: str) -> int:
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
