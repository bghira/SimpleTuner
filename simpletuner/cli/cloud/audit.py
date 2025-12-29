"""
Cloud audit log CLI commands.

Provides access to the audit log for viewing security events and chain verification.
"""

import json

from .api import cloud_api_request


def cmd_cloud_audit(args) -> int:
    """Access the audit log."""
    audit_action = getattr(args, "audit_action", None)

    if audit_action == "list":
        return _audit_list(args)
    elif audit_action == "stats":
        return _audit_stats(args)
    elif audit_action == "types":
        return _audit_types(args)
    elif audit_action == "user":
        return _audit_user(args)
    elif audit_action == "security":
        return _audit_security(args)
    elif audit_action == "verify":
        return _audit_verify(args)
    else:
        print("Error: Unknown audit action. Use 'simpletuner cloud audit --help'.")
        return 1


def _audit_list(args) -> int:
    """List audit log entries."""
    event_type = getattr(args, "event_type", None)
    actor_id = getattr(args, "actor_id", None)
    target_type = getattr(args, "target_type", None)
    target_id = getattr(args, "target_id", None)
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    limit = getattr(args, "limit", 50)
    offset = getattr(args, "offset", 0)
    output_format = getattr(args, "format", "table")

    params = [f"limit={limit}", f"offset={offset}"]
    if event_type:
        params.append(f"event_type={event_type}")
    if actor_id:
        params.append(f"actor_id={actor_id}")
    if target_type:
        params.append(f"target_type={target_type}")
    if target_id:
        params.append(f"target_id={target_id}")
    if since:
        params.append(f"since={since}")
    if until:
        params.append(f"until={until}")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/audit?{query}")

    entries = result.get("entries", [])
    has_more = result.get("has_more", False)

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not entries:
        print("No audit entries found.")
        return 0

    print(f"{'ID':<8} {'Timestamp':<20} {'Type':<20} {'Actor':<15} {'Action':<20}")
    print("-" * 90)

    for entry in entries:
        entry_id = entry.get("id", "-")
        timestamp = entry.get("timestamp", "-")[:19]
        event_type = entry.get("event_type", "unknown")[:19]
        actor = entry.get("actor_username") or f"user:{entry.get('actor_id', '-')}"
        actor = actor[:14]
        action = entry.get("action", "-")[:19]

        print(f"{entry_id:<8} {timestamp:<20} {event_type:<20} {actor:<15} {action:<20}")

    if has_more:
        print(f"\n... more entries available (use --offset {offset + limit})")

    return 0


def _audit_stats(args) -> int:
    """Get audit log statistics."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/audit/stats")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Audit Log Statistics")
    print("=" * 50)
    print(f"Total Entries:   {result.get('total_entries', 0)}")
    print(f"Last 24 Hours:   {result.get('last_24h', 0)}")
    print(f"First Entry:     {result.get('first_entry') or '-'}")
    print(f"Last Entry:      {result.get('last_entry') or '-'}")

    by_type = result.get("by_type", {})
    if by_type:
        print("\nBy Event Type:")
        for event_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {event_type:<30} {count}")

    return 0


def _audit_types(args) -> int:
    """List available audit event types."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/audit/types")

    event_types = result.get("event_types", [])

    if output_format == "json":
        print(json.dumps(event_types, indent=2))
        return 0

    if not event_types:
        print("No event types found.")
        return 0

    print("Available Audit Event Types")
    print("=" * 50)

    for et in event_types:
        value = et.get("value", "-")
        name = et.get("name", "-")
        print(f"  {value:<35} ({name})")

    return 0


def _audit_user(args) -> int:
    """Get audit log entries for a specific user."""
    user_id = getattr(args, "user_id", None)
    limit = getattr(args, "limit", 50)
    offset = getattr(args, "offset", 0)
    output_format = getattr(args, "format", "table")

    if not user_id:
        print("Error: User ID is required.")
        return 1

    params = [f"limit={limit}", f"offset={offset}"]
    query = "&".join(params)

    result = cloud_api_request("GET", f"/api/audit/user/{user_id}?{query}")

    entries = result.get("entries", [])
    has_more = result.get("has_more", False)

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not entries:
        print(f"No audit entries found for user {user_id}.")
        return 0

    print(f"Audit Log for User {user_id}")
    print("=" * 90)
    print(f"{'ID':<8} {'Timestamp':<20} {'Type':<20} {'Action':<25} {'Target':<15}")
    print("-" * 90)

    for entry in entries:
        entry_id = entry.get("id", "-")
        timestamp = entry.get("timestamp", "-")[:19]
        event_type = entry.get("event_type", "unknown")[:19]
        action = entry.get("action", "-")[:24]
        target = entry.get("target_type") or "-"
        if entry.get("target_id"):
            target = f"{target}:{entry.get('target_id')}"
        target = target[:14]

        print(f"{entry_id:<8} {timestamp:<20} {event_type:<20} {action:<25} {target:<15}")

    if has_more:
        print(f"\n... more entries available (use --offset {offset + limit})")

    return 0


def _audit_security(args) -> int:
    """Get security-related audit events."""
    limit = getattr(args, "limit", 50)
    offset = getattr(args, "offset", 0)
    output_format = getattr(args, "format", "table")

    params = [f"limit={limit}", f"offset={offset}"]
    query = "&".join(params)

    result = cloud_api_request("GET", f"/api/audit/security?{query}")

    entries = result.get("entries", [])
    has_more = result.get("has_more", False)

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not entries:
        print("No security events found.")
        return 0

    print("Security Events")
    print("=" * 100)
    print(f"{'ID':<8} {'Timestamp':<20} {'Type':<25} {'Actor':<15} {'IP':<15} {'Action':<15}")
    print("-" * 100)

    for entry in entries:
        entry_id = entry.get("id", "-")
        timestamp = entry.get("timestamp", "-")[:19]
        event_type = entry.get("event_type", "unknown")[:24]
        actor = entry.get("actor_username") or "-"
        actor = actor[:14]
        ip = entry.get("actor_ip") or "-"
        ip = ip[:14]
        action = entry.get("action", "-")[:14]

        print(f"{entry_id:<8} {timestamp:<20} {event_type:<25} {actor:<15} {ip:<15} {action:<15}")

    if has_more:
        print(f"\n... more entries available (use --offset {offset + limit})")

    return 0


def _audit_verify(args) -> int:
    """Verify audit log chain integrity."""
    start_id = getattr(args, "start_id", None)
    end_id = getattr(args, "end_id", None)
    output_format = getattr(args, "format", "table")

    params = []
    if start_id:
        params.append(f"start_id={start_id}")
    if end_id:
        params.append(f"end_id={end_id}")

    query = "&".join(params) if params else ""
    endpoint = "/api/audit/verify"
    if query:
        endpoint += f"?{query}"

    result = cloud_api_request("GET", endpoint)

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    valid = result.get("valid", False)
    entries_checked = result.get("entries_checked", 0)
    broken_links = result.get("broken_links", [])

    print("Audit Log Chain Verification")
    print("=" * 50)
    print(f"Valid:           {'Yes' if valid else 'NO - CHAIN BROKEN!'}")
    print(f"Entries Checked: {entries_checked}")
    print(f"First ID:        {result.get('first_id') or '-'}")
    print(f"Last ID:         {result.get('last_id') or '-'}")

    if broken_links:
        print(f"\nBroken Links ({len(broken_links)}):")
        for link in broken_links[:10]:
            print(f"  Entry {link.get('id')}: {link.get('reason', 'unknown reason')}")
        if len(broken_links) > 10:
            print(f"  ... and {len(broken_links) - 10} more")

    if not valid:
        return 1

    return 0
