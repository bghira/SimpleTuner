"""
Cloud provider configuration commands.

Handles config show, set-token, delete-token, and set commands.
"""

import json
from typing import Any

from .api import cloud_api_request


def cmd_cloud_config(args) -> int:
    """Manage cloud provider configuration."""
    config_action = getattr(args, "config_action", None)

    if config_action == "show":
        return cmd_cloud_config_show(args)
    elif config_action == "set-token":
        return cmd_cloud_config_set_token(args)
    elif config_action == "delete-token":
        return cmd_cloud_config_delete_token(args)
    elif config_action == "set":
        return cmd_cloud_config_set(args)
    else:
        print("Error: Unknown config action. Use 'simpletuner cloud config --help'.")
        return 1


def cmd_cloud_config_show(args) -> int:
    """Show provider configuration."""
    provider = getattr(args, "provider", "replicate")
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", f"/api/cloud/providers/{provider}/config")
    config = result.get("config", {})

    if output_format == "json":
        print(json.dumps(config, indent=2))
        return 0

    print(f"Configuration for {provider}:")
    print("=" * 50)

    display_config = dict(config)
    for key in ("api_token", "secret", "password", "access_token", "refresh_token"):
        if key in display_config and display_config[key]:
            display_config[key] = "***"

    for key, value in sorted(display_config.items()):
        if value is not None:
            print(f"  {key}: {value}")

    return 0


def cmd_cloud_config_set_token(args) -> int:
    """Set provider API token."""
    provider = getattr(args, "provider", "replicate")
    token = getattr(args, "token", None)

    if not token:
        import getpass

        token = getpass.getpass(f"Enter {provider} API token: ")

    if not token:
        print("Error: Token cannot be empty.")
        return 1

    if provider == "replicate":
        result = cloud_api_request(
            "PUT",
            "/api/cloud/providers/replicate/token",
            data={"api_token": token},
        )
    elif provider == "simpletuner_io":
        result = cloud_api_request(
            "PUT",
            "/api/cloud/providers/simpletuner_io/token",
            data={"api_token": token},
        )
    else:
        print(f"Error: Token management not supported for provider '{provider}'.")
        return 1

    if result.get("success"):
        print("API token saved successfully.")
        if result.get("file_path"):
            print(f"  Saved to: {result['file_path']}")
        if result.get("error"):
            print(f"  Warning: {result['error']}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Failed to save token')}")
        return 1


def cmd_cloud_config_delete_token(args) -> int:
    """Delete provider API token."""
    provider = getattr(args, "provider", "replicate")

    if provider == "replicate":
        result = cloud_api_request("DELETE", "/api/cloud/providers/replicate/token")
    elif provider == "simpletuner_io":
        result = cloud_api_request("DELETE", "/api/cloud/providers/simpletuner_io/token")
    else:
        print(f"Error: Token management not supported for provider '{provider}'.")
        return 1

    if result.get("success"):
        print("API token deleted successfully.")
        return 0
    else:
        print(f"Error: {result.get('error', 'Failed to delete token')}")
        return 1


def cmd_cloud_config_set(args) -> int:
    """Set a provider configuration option."""
    provider = getattr(args, "provider", "replicate")
    key = getattr(args, "key", None)
    value = getattr(args, "value", None)

    if not key:
        print("Error: Configuration key is required.")
        return 1

    parsed_value: Any = value
    if value is not None:
        if value.lower() == "true":
            parsed_value = True
        elif value.lower() == "false":
            parsed_value = False
        elif value.lower() in ("null", "none"):
            parsed_value = None
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    pass

    update_data = {key: parsed_value}

    result = cloud_api_request(
        "PUT",
        f"/api/cloud/providers/{provider}/config",
        data=update_data,
    )

    if result.get("config") is not None:
        print(f"Configuration updated: {key} = {parsed_value}")
        return 0
    else:
        print("Error: Failed to update configuration")
        return 1
