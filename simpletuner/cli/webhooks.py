"""
Webhook management commands.

Handles webhook list, create, update, delete, test, and history commands.
"""

import json

from .cloud.api import cloud_api_request


def cmd_webhooks(args) -> int:
    """Manage webhooks."""
    webhook_action = getattr(args, "webhook_action", None)

    if webhook_action == "list":
        return _webhook_list(args)
    elif webhook_action == "get":
        return _webhook_get(args)
    elif webhook_action == "create":
        return _webhook_create(args)
    elif webhook_action == "update":
        return _webhook_update(args)
    elif webhook_action == "delete":
        return _webhook_delete(args)
    elif webhook_action == "test":
        return _webhook_test(args)
    elif webhook_action == "history":
        return _webhook_history(args)
    elif webhook_action == "events":
        return _webhook_events(args)
    else:
        print("Error: Unknown webhook action. Use 'simpletuner webhooks --help'.")
        return 1


def _webhook_list(args) -> int:
    """List configured webhooks."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/webhooks")
    webhooks = result.get("webhooks", [])

    if not webhooks:
        print("No webhooks configured.")
        return 0

    if output_format == "json":
        print(json.dumps(webhooks, indent=2))
        return 0

    print(f"{'ID':<6} {'Name':<20} {'URL':<40} {'Events':<15} {'Enabled':<8}")
    print("-" * 95)

    for wh in webhooks:
        wh_id = wh.get("id", "-")
        name = wh.get("name", "-")[:19]
        url = wh.get("url", "-")[:39]
        events = len(wh.get("events", []))
        enabled = "yes" if wh.get("enabled") else "no"

        print(f"{wh_id:<6} {name:<20} {url:<40} {events:<15} {enabled:<8}")

    return 0


def _webhook_get(args) -> int:
    """Get details of a specific webhook."""
    webhook_id = getattr(args, "webhook_id", None)
    output_format = getattr(args, "format", "table")

    if not webhook_id:
        print("Error: Webhook ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/webhooks/{webhook_id}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Webhook: {result.get('name')}")
    print("=" * 50)
    print(f"  ID: {result.get('id')}")
    print(f"  URL: {result.get('url')}")
    print(f"  Enabled: {'yes' if result.get('enabled') else 'no'}")
    print(f"  Secret: {'configured' if result.get('has_secret') else 'not configured'}")

    events = result.get("events", [])
    if events:
        print(f"  Events ({len(events)}):")
        for event in events:
            print(f"    - {event}")

    if result.get("created_at"):
        print(f"  Created: {result.get('created_at')}")
    if result.get("last_triggered"):
        print(f"  Last Triggered: {result.get('last_triggered')}")

    return 0


def _webhook_create(args) -> int:
    """Create a new webhook."""
    name = getattr(args, "name", None)
    url = getattr(args, "url", None)
    events = getattr(args, "events", None)
    secret = getattr(args, "secret", None)
    enabled = getattr(args, "enabled", True)

    if not name or not url:
        print("Error: Both --name and --url are required.")
        return 1

    data = {
        "name": name,
        "url": url,
        "enabled": enabled,
    }

    if events:
        data["events"] = events
    if secret:
        data["secret"] = secret

    result = cloud_api_request("POST", "/api/webhooks", data=data)

    if result.get("webhook") or result.get("id"):
        wh = result.get("webhook", result)
        print(f"Webhook created successfully!")
        print(f"  ID: {wh.get('id')}")
        print(f"  Name: {wh.get('name')}")
        print(f"  URL: {wh.get('url')}")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to create webhook')}")
        return 1


def _webhook_update(args) -> int:
    """Update an existing webhook."""
    webhook_id = getattr(args, "webhook_id", None)
    name = getattr(args, "name", None)
    url = getattr(args, "url", None)
    events = getattr(args, "events", None)
    secret = getattr(args, "secret", None)
    enabled = getattr(args, "enabled", None)

    if not webhook_id:
        print("Error: Webhook ID is required.")
        return 1

    data = {}
    if name is not None:
        data["name"] = name
    if url is not None:
        data["url"] = url
    if events is not None:
        data["events"] = events
    if secret is not None:
        data["secret"] = secret
    if enabled is not None:
        data["enabled"] = enabled

    if not data:
        print("Error: No updates provided.")
        return 1

    result = cloud_api_request("PATCH", f"/api/webhooks/{webhook_id}", data=data)

    if result.get("webhook") or result.get("id"):
        print(f"Webhook {webhook_id} updated successfully.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to update webhook')}")
        return 1


def _webhook_delete(args) -> int:
    """Delete a webhook."""
    webhook_id = getattr(args, "webhook_id", None)
    force = getattr(args, "force", False)

    if not webhook_id:
        print("Error: Webhook ID is required.")
        return 1

    if not force:
        confirm = input(f"Delete webhook {webhook_id}? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/webhooks/{webhook_id}")

    if result.get("success"):
        print(f"Webhook {webhook_id} deleted.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to delete webhook')}")
        return 1


def _webhook_test(args) -> int:
    """Send a test event to a webhook."""
    webhook_id = getattr(args, "webhook_id", None)
    event_type = getattr(args, "event_type", "test")

    if not webhook_id:
        print("Error: Webhook ID is required.")
        return 1

    data = {"event_type": event_type}

    result = cloud_api_request("POST", f"/api/webhooks/{webhook_id}/test", data=data)

    if result.get("success"):
        print(f"Test event sent successfully!")
        print(f"  Event Type: {event_type}")
        if result.get("response_status"):
            print(f"  Response Status: {result.get('response_status')}")
        if result.get("response_time_ms"):
            print(f"  Response Time: {result.get('response_time_ms')}ms")
        return 0
    else:
        print(f"Test failed: {result.get('detail', result.get('error', 'Unknown error'))}")
        return 1


def _webhook_history(args) -> int:
    """Show webhook delivery history."""
    webhook_id = getattr(args, "webhook_id", None)
    limit = getattr(args, "limit", 20)
    output_format = getattr(args, "format", "table")

    endpoint = "/api/webhooks/history"
    if webhook_id:
        endpoint = f"/api/webhooks/{webhook_id}/history"
    endpoint += f"?limit={limit}"

    result = cloud_api_request("GET", endpoint)
    history = result.get("deliveries", [])

    if not history:
        print("No delivery history found.")
        return 0

    if output_format == "json":
        print(json.dumps(history, indent=2))
        return 0

    print(f"{'Time':<20} {'Webhook':<15} {'Event':<20} {'Status':<10} {'Duration':<10}")
    print("-" * 80)

    for h in history:
        time = h.get("timestamp", "-")[:19]
        webhook = str(h.get("webhook_id", "-"))[:14]
        event = h.get("event_type", "-")[:19]
        status = "[+]" if h.get("success") else "[X]"
        status += f" {h.get('response_status', '-')}"
        duration = f"{h.get('response_time_ms', 0)}ms"

        print(f"{time:<20} {webhook:<15} {event:<20} {status:<10} {duration:<10}")

    return 0


def _webhook_events(args) -> int:
    """List available webhook event types."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/webhooks/events")
    events = result.get("events", [])

    if not events:
        print("No event types available.")
        return 0

    if output_format == "json":
        print(json.dumps(events, indent=2))
        return 0

    print("Available Webhook Event Types")
    print("=" * 50)

    for event in events:
        if isinstance(event, dict):
            name = event.get("name", "unknown")
            description = event.get("description", "")
            print(f"\n  {name}")
            if description:
                print(f"    {description}")
        else:
            print(f"  - {event}")

    return 0
