"""
Cloud notification channel CLI commands.

Provides access to notification channels, preferences, and delivery history.
"""

import json

from .api import cloud_api_request, format_bool


def cmd_cloud_notifications(args) -> int:
    """Manage notification channels and preferences."""
    notif_action = getattr(args, "notif_action", None)

    if notif_action == "channels":
        return _notif_channels(args)
    elif notif_action == "channel-get":
        return _notif_channel_get(args)
    elif notif_action == "channel-create":
        return _notif_channel_create(args)
    elif notif_action == "channel-update":
        return _notif_channel_update(args)
    elif notif_action == "channel-delete":
        return _notif_channel_delete(args)
    elif notif_action == "channel-test":
        return _notif_channel_test(args)
    elif notif_action == "preferences":
        return _notif_preferences(args)
    elif notif_action == "preference-create":
        return _notif_preference_create(args)
    elif notif_action == "preference-delete":
        return _notif_preference_delete(args)
    elif notif_action == "events":
        return _notif_events(args)
    elif notif_action == "presets":
        return _notif_presets(args)
    elif notif_action == "preset-get":
        return _notif_preset_get(args)
    elif notif_action == "history":
        return _notif_history(args)
    elif notif_action == "status":
        return _notif_status(args)
    elif notif_action == "skip":
        return _notif_skip(args)
    else:
        print("Error: Unknown notifications action. Use 'simpletuner cloud notifications --help'.")
        return 1


# --- Channels ---


def _notif_channels(args) -> int:
    """List notification channels."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/cloud/notifications/channels")

    channels = result.get("channels", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not channels:
        print("No notification channels configured.")
        return 0

    print("Notification Channels")
    print("=" * 70)
    print(f"{'ID':<6} {'Name':<20} {'Type':<10} {'Enabled':<8} {'Host/URL':<25}")
    print("-" * 70)

    for ch in channels:
        ch_id = ch.get("id", "-")
        name = ch.get("name", "unknown")[:19]
        ch_type = ch.get("channel_type", "unknown")[:9]
        enabled = format_bool(ch.get("is_enabled", False))

        host_or_url = ch.get("smtp_host") or ch.get("webhook_url") or "-"
        host_or_url = host_or_url[:24]

        print(f"{ch_id:<6} {name:<20} {ch_type:<10} {enabled:<8} {host_or_url:<25}")

    return 0


def _notif_channel_get(args) -> int:
    """Get a channel by ID."""
    channel_id = getattr(args, "channel_id", None)
    output_format = getattr(args, "format", "table")

    if not channel_id:
        print("Error: Channel ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/cloud/notifications/channels/{channel_id}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Channel {result.get('id', channel_id)}")
    print("=" * 50)
    print(f"Name:        {result.get('name', 'unknown')}")
    print(f"Type:        {result.get('channel_type', '-')}")
    print(f"Enabled:     {format_bool(result.get('is_enabled', False))}")

    if result.get("smtp_host"):
        print(f"\nSMTP Settings:")
        print(f"  Host:      {result.get('smtp_host')}")
        print(f"  Port:      {result.get('smtp_port', 587)}")
        print(f"  Username:  {result.get('smtp_username') or '-'}")
        print(f"  TLS:       {format_bool(result.get('smtp_use_tls', True))}")
        print(f"  From:      {result.get('smtp_from_address') or '-'}")

    if result.get("webhook_url"):
        print(f"\nWebhook Settings:")
        print(f"  URL:       {result.get('webhook_url')}")

    if result.get("imap_enabled"):
        print(f"\nIMAP Settings:")
        print(f"  Host:      {result.get('imap_host')}")

    return 0


def _notif_channel_create(args) -> int:
    """Create a notification channel."""
    channel_type = getattr(args, "type", None)
    name = getattr(args, "name", None)

    if not channel_type or not name:
        print("Error: --type and --name are required.")
        return 1

    data = {
        "channel_type": channel_type,
        "name": name,
        "is_enabled": getattr(args, "enabled", True),
    }

    # SMTP settings
    if getattr(args, "smtp_host", None):
        data["smtp_host"] = args.smtp_host
    if getattr(args, "smtp_port", None):
        data["smtp_port"] = args.smtp_port
    if getattr(args, "smtp_username", None):
        data["smtp_username"] = args.smtp_username
    if getattr(args, "smtp_password", None):
        data["smtp_password"] = args.smtp_password
    if getattr(args, "smtp_use_tls", None) is not None:
        data["smtp_use_tls"] = args.smtp_use_tls
    if getattr(args, "smtp_from_address", None):
        data["smtp_from_address"] = args.smtp_from_address
    if getattr(args, "smtp_from_name", None):
        data["smtp_from_name"] = args.smtp_from_name

    # Webhook settings
    if getattr(args, "webhook_url", None):
        data["webhook_url"] = args.webhook_url
    if getattr(args, "webhook_secret", None):
        data["webhook_secret"] = args.webhook_secret

    # IMAP settings
    if getattr(args, "imap_enabled", None):
        data["imap_enabled"] = args.imap_enabled
    if getattr(args, "imap_host", None):
        data["imap_host"] = args.imap_host
    if getattr(args, "imap_port", None):
        data["imap_port"] = args.imap_port
    if getattr(args, "imap_username", None):
        data["imap_username"] = args.imap_username
    if getattr(args, "imap_password", None):
        data["imap_password"] = args.imap_password

    result = cloud_api_request("POST", "/api/cloud/notifications/channels", data=data)

    print(f"Channel created:")
    print(f"  ID:   {result.get('id', '-')}")
    print(f"  Name: {result.get('name', name)}")
    print(f"  Type: {result.get('channel_type', channel_type)}")
    return 0


def _notif_channel_update(args) -> int:
    """Update a notification channel."""
    channel_id = getattr(args, "channel_id", None)

    if not channel_id:
        print("Error: Channel ID is required.")
        return 1

    data = {}
    if getattr(args, "name", None):
        data["name"] = args.name
    if getattr(args, "enabled", None) is not None:
        data["is_enabled"] = args.enabled
    if getattr(args, "smtp_host", None):
        data["smtp_host"] = args.smtp_host
    if getattr(args, "smtp_port", None):
        data["smtp_port"] = args.smtp_port
    if getattr(args, "smtp_username", None):
        data["smtp_username"] = args.smtp_username
    if getattr(args, "smtp_password", None):
        data["smtp_password"] = args.smtp_password
    if getattr(args, "smtp_use_tls", None) is not None:
        data["smtp_use_tls"] = args.smtp_use_tls
    if getattr(args, "smtp_from_address", None):
        data["smtp_from_address"] = args.smtp_from_address
    if getattr(args, "webhook_url", None):
        data["webhook_url"] = args.webhook_url

    if not data:
        print("Error: No updates specified.")
        return 1

    cloud_api_request("PATCH", f"/api/cloud/notifications/channels/{channel_id}", data=data)

    print(f"Channel {channel_id} updated.")
    return 0


def _notif_channel_delete(args) -> int:
    """Delete a notification channel."""
    channel_id = getattr(args, "channel_id", None)

    if not channel_id:
        print("Error: Channel ID is required.")
        return 1

    cloud_api_request("DELETE", f"/api/cloud/notifications/channels/{channel_id}")

    print(f"Channel {channel_id} deleted.")
    return 0


def _notif_channel_test(args) -> int:
    """Test a notification channel's connectivity."""
    channel_id = getattr(args, "channel_id", None)

    if not channel_id:
        print("Error: Channel ID is required.")
        return 1

    result = cloud_api_request("POST", f"/api/cloud/notifications/channels/{channel_id}/test")

    if result.get("success"):
        latency = result.get("latency_ms")
        if latency:
            print(f"Channel {channel_id} test successful (latency: {latency:.0f}ms)")
        else:
            print(f"Channel {channel_id} test successful")
        return 0
    else:
        error = result.get("error", "Unknown error")
        print(f"Channel {channel_id} test FAILED: {error}")
        return 1


# --- Preferences ---


def _notif_preferences(args) -> int:
    """List notification preferences."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/cloud/notifications/preferences")

    preferences = result.get("preferences", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not preferences:
        print("No notification preferences configured.")
        return 0

    print("Notification Preferences")
    print("=" * 70)
    print(f"{'ID':<6} {'Event Type':<25} {'Channel':<10} {'Enabled':<8} {'Severity':<10}")
    print("-" * 70)

    for pref in preferences:
        pref_id = pref.get("id", "-")
        event_type = pref.get("event_type", "unknown")[:24]
        channel_id = pref.get("channel_id", "-")
        enabled = format_bool(pref.get("is_enabled", False))
        severity = pref.get("min_severity", "info")

        print(f"{pref_id:<6} {event_type:<25} {channel_id:<10} {enabled:<8} {severity:<10}")

    return 0


def _notif_preference_create(args) -> int:
    """Create a notification preference."""
    event_type = getattr(args, "event_type", None)
    channel_id = getattr(args, "channel_id", None)

    if not event_type or not channel_id:
        print("Error: --event-type and --channel-id are required.")
        return 1

    data = {
        "event_type": event_type,
        "channel_id": channel_id,
        "is_enabled": getattr(args, "enabled", True),
        "min_severity": getattr(args, "min_severity", "info"),
    }

    recipients = getattr(args, "recipients", None)
    if recipients:
        data["recipients"] = recipients.split(",")

    result = cloud_api_request("POST", "/api/cloud/notifications/preferences", data=data)

    print(f"Preference created:")
    print(f"  ID:         {result.get('id', '-')}")
    print(f"  Event Type: {result.get('event_type', event_type)}")
    print(f"  Channel:    {result.get('channel_id', channel_id)}")
    return 0


def _notif_preference_delete(args) -> int:
    """Delete a notification preference."""
    preference_id = getattr(args, "preference_id", None)

    if not preference_id:
        print("Error: Preference ID is required.")
        return 1

    cloud_api_request("DELETE", f"/api/cloud/notifications/preferences/{preference_id}")

    print(f"Preference {preference_id} deleted.")
    return 0


# --- Metadata ---


def _notif_events(args) -> int:
    """List available notification event types."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/cloud/notifications/events")

    event_types = result.get("event_types", [])

    if output_format == "json":
        print(json.dumps(event_types, indent=2))
        return 0

    if not event_types:
        print("No event types found.")
        return 0

    print("Available Notification Event Types")
    print("=" * 70)

    current_category = None
    for et in event_types:
        category = et.get("category", "Other")
        if category != current_category:
            print(f"\n{category}:")
            current_category = category

        event_id = et.get("id", "-")
        name = et.get("name", "-")
        desc = et.get("description", "")[:40]

        print(f"  {event_id:<25} {name:<20} {desc}")

    return 0


def _notif_presets(args) -> int:
    """List email provider presets."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/cloud/notifications/presets")

    presets = result.get("presets", [])

    if output_format == "json":
        print(json.dumps(presets, indent=2))
        return 0

    if not presets:
        print("No presets found.")
        return 0

    print("Email Provider Presets")
    print("=" * 70)
    print(f"{'ID':<12} {'Name':<25} {'SMTP Host':<25} {'Port':<6} {'TLS':<5}")
    print("-" * 70)

    for preset in presets:
        preset_id = preset.get("id", "-")
        name = preset.get("name", "-")[:24]
        host = preset.get("smtp_host", "-")[:24]
        port = preset.get("smtp_port", 587)
        tls = format_bool(preset.get("smtp_use_tls", True))

        print(f"{preset_id:<12} {name:<25} {host:<25} {port:<6} {tls:<5}")

    return 0


def _notif_preset_get(args) -> int:
    """Get an email provider preset by ID."""
    preset_id = getattr(args, "preset_id", None)
    output_format = getattr(args, "format", "table")

    if not preset_id:
        print("Error: Preset ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/cloud/notifications/presets/{preset_id}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Email Preset: {result.get('name', preset_id)}")
    print("=" * 50)
    print(f"ID:        {result.get('id', preset_id)}")
    print(f"SMTP Host: {result.get('smtp_host', '-')}")
    print(f"SMTP Port: {result.get('smtp_port', 587)}")
    print(f"TLS:       {format_bool(result.get('smtp_use_tls', True))}")

    return 0


# --- History & Status ---


def _notif_history(args) -> int:
    """Get notification delivery history."""
    limit = getattr(args, "limit", 50)
    channel_id = getattr(args, "channel_id", None)
    output_format = getattr(args, "format", "table")

    params = [f"limit={limit}"]
    if channel_id:
        params.append(f"channel_id={channel_id}")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/cloud/notifications/history?{query}")

    entries = result.get("entries", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    if not entries:
        print("No delivery history found.")
        return 0

    print("Notification Delivery History")
    print("=" * 90)
    print(f"{'ID':<6} {'Sent At':<20} {'Event':<20} {'Channel':<8} {'Recipient':<20} {'Status':<10}")
    print("-" * 90)

    for entry in entries:
        entry_id = entry.get("id", "-")
        sent_at = entry.get("sent_at", "-")[:19]
        event = entry.get("event_type", "-")[:19]
        channel = entry.get("channel_id", "-")
        recipient = entry.get("recipient", "-")[:19]
        status = entry.get("delivery_status", "-")[:9]

        print(f"{entry_id:<6} {sent_at:<20} {event:<20} {channel:<8} {recipient:<20} {status:<10}")

    return 0


def _notif_status(args) -> int:
    """Get notification system status."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/cloud/notifications/status")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Notification System Status")
    print("=" * 50)
    print(f"Initialized: {format_bool(result.get('initialized', False))}")

    channels = result.get("channels", {})
    print(f"\nChannels:")
    print(f"  Total:   {channels.get('total', 0)}")
    print(f"  Enabled: {channels.get('enabled', 0)}")

    by_type = channels.get("by_type", {})
    if by_type:
        print(f"  By Type:")
        for ch_type, count in by_type.items():
            print(f"    {ch_type}: {count}")

    handlers = result.get("response_handlers", {})
    print(f"\nResponse Handlers:")
    print(f"  Total:   {handlers.get('total', 0)}")
    print(f"  Running: {handlers.get('running', 0)}")

    stats = result.get("stats", {})
    if stats:
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return 0


def _notif_skip(args) -> int:
    """Dismiss the notification setup prompt."""
    cloud_api_request("POST", "/api/cloud/notifications/skip")

    print("Notification setup prompt dismissed.")
    return 0
