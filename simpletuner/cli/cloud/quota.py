"""
Cloud quota management commands.

Handles quota list, create, delete, status, and types commands.
"""

import json

from .api import cloud_api_request


def cmd_cloud_quota(args) -> int:
    """Manage cloud quotas."""
    quota_action = getattr(args, "quota_action", None)

    if quota_action == "list":
        return cmd_cloud_quota_list(args)
    elif quota_action == "create":
        return cmd_cloud_quota_create(args)
    elif quota_action == "delete":
        return cmd_cloud_quota_delete(args)
    elif quota_action == "status":
        return cmd_cloud_quota_status(args)
    elif quota_action == "types":
        return cmd_cloud_quota_types(args)
    else:
        print("Error: Unknown quota action. Use 'simpletuner cloud quota --help'.")
        return 1


def cmd_cloud_quota_list(args) -> int:
    """List configured quotas."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/quotas")
    quotas = result.get("quotas", [])

    if not quotas:
        print("No quotas configured.")
        return 0

    if output_format == "json":
        print(json.dumps(quotas, indent=2))
        return 0

    print(f"{'ID':<6} {'Type':<20} {'Limit':<12} {'Action':<12} {'Scope':<15}")
    print("-" * 75)

    for q in quotas:
        quota_id = q.get("id", "-")
        quota_type = q.get("quota_type", "unknown")
        limit_value = q.get("limit_value", 0)
        action = q.get("action", "block")
        scope = _format_quota_scope(q)

        print(f"{quota_id:<6} {quota_type:<20} {limit_value:<12} {action:<12} {scope:<15}")

    return 0


def _format_quota_scope(quota: dict) -> str:
    """Format the scope (user, team, org) of a quota."""
    if quota.get("org_id"):
        return f"org:{quota['org_id']}"
    elif quota.get("team_id"):
        return f"team:{quota['team_id']}"
    elif quota.get("user_id"):
        return f"user:{quota['user_id']}"
    else:
        return "global"


def cmd_cloud_quota_create(args) -> int:
    """Create a new quota."""
    quota_type = getattr(args, "type", None)
    limit_value = getattr(args, "limit", None)
    action = getattr(args, "action", "block")
    user_id = getattr(args, "user_id", None)
    team_id = getattr(args, "team_id", None)
    org_id = getattr(args, "org_id", None)

    if not quota_type or limit_value is None:
        print("Error: Both --type and --limit are required.")
        return 1

    data = {
        "quota_type": quota_type,
        "limit_value": float(limit_value),
        "action": action,
    }

    if user_id:
        data["user_id"] = int(user_id)
    if team_id:
        data["team_id"] = int(team_id)
    if org_id:
        data["org_id"] = int(org_id)

    result = cloud_api_request("POST", "/api/quotas", data=data)

    if result.get("quota"):
        print(f"Quota created: {quota_type} = {limit_value} ({action})")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to create quota')}")
        return 1


def cmd_cloud_quota_delete(args) -> int:
    """Delete a quota."""
    quota_id = getattr(args, "quota_id", None)
    force = getattr(args, "force", False)

    if not quota_id:
        print("Error: Quota ID is required.")
        return 1

    if not force:
        confirm = input(f"Delete quota {quota_id}? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/quotas/{quota_id}")

    if result.get("success"):
        print(f"Quota {quota_id} deleted.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to delete quota')}")
        return 1


def cmd_cloud_quota_status(args) -> int:
    """Show current quota usage status."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/quotas/status")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Quota Usage Status")
    print("=" * 60)

    quotas = result.get("quotas", [])
    if not quotas:
        print("No quotas configured.")
        return 0

    for q in quotas:
        quota_type = q.get("quota_type", "unknown")
        limit_value = q.get("limit_value", 0)
        current = q.get("current_usage", 0)
        percentage = (current / limit_value * 100) if limit_value > 0 else 0
        action = q.get("action", "block")

        bar_width = 30
        filled = int(bar_width * min(percentage, 100) / 100)
        bar = "#" * filled + "-" * (bar_width - filled)

        status = "OK"
        if percentage >= 100:
            status = "EXCEEDED"
        elif percentage >= 80:
            status = "WARNING"

        print(f"\n{quota_type} ({action}):")
        print(f"  [{bar}] {percentage:.1f}%")
        print(f"  {current:.2f} / {limit_value:.2f} - {status}")

    return 0


def cmd_cloud_quota_types(args) -> int:
    """List available quota types."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/quotas/types")
    types = result.get("types", [])

    if output_format == "json":
        print(json.dumps(types, indent=2))
        return 0

    print("Available Quota Types:")
    print("=" * 50)

    for qt in types:
        name = qt.get("name", "unknown")
        description = qt.get("description", "")
        unit = qt.get("unit", "")
        print(f"\n  {name}")
        print(f"    Description: {description}")
        print(f"    Unit: {unit}")

    return 0
