"""
Cloud approval workflow commands.

Handles approval list, pending, approve, reject, and rules commands.
"""

import json

from .api import cloud_api_request


def cmd_cloud_approval(args) -> int:
    """Manage job approval workflow."""
    approval_action = getattr(args, "approval_action", None)

    if approval_action == "list":
        return cmd_cloud_approval_list(args)
    elif approval_action == "pending":
        return cmd_cloud_approval_pending(args)
    elif approval_action == "approve":
        return cmd_cloud_approval_approve(args)
    elif approval_action == "reject":
        return cmd_cloud_approval_reject(args)
    elif approval_action == "rules":
        return cmd_cloud_approval_rules(args)
    else:
        print("Error: Unknown approval action. Use 'simpletuner cloud approval --help'.")
        return 1


def cmd_cloud_approval_list(args) -> int:
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


def cmd_cloud_approval_pending(args) -> int:
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
    print("Use 'simpletuner cloud approval approve <id>' to approve")
    print("Use 'simpletuner cloud approval reject <id>' to reject")

    return 0


def cmd_cloud_approval_approve(args) -> int:
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


def cmd_cloud_approval_reject(args) -> int:
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


def cmd_cloud_approval_rules(args) -> int:
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
