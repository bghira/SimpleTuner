"""
Backup management commands.

Handles backup list, create, restore, get, and delete commands.
"""

import json

from .cloud.api import cloud_api_request


def cmd_backup(args) -> int:
    """Manage backups."""
    backup_action = getattr(args, "backup_action", None)

    if backup_action == "list":
        return _backup_list(args)
    elif backup_action == "create":
        return _backup_create(args)
    elif backup_action == "get":
        return _backup_get(args)
    elif backup_action == "restore":
        return _backup_restore(args)
    elif backup_action == "delete":
        return _backup_delete(args)
    else:
        print("Error: Unknown backup action. Use 'simpletuner backup --help'.")
        return 1


def _backup_list(args) -> int:
    """List available backups."""
    output_format = getattr(args, "format", "table")
    limit = getattr(args, "limit", 50)

    result = cloud_api_request("GET", f"/api/backup?limit={limit}")
    backups = result.get("backups", [])

    if not backups:
        print("No backups found.")
        return 0

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"{'ID':<25} {'Name':<20} {'Size':<12} {'Components':<25} {'Created':<20}")
    print("-" * 102)

    for b in backups:
        backup_id = b.get("id", "-")[:24]
        name = b.get("name", "-")[:19]
        size = _format_size(b.get("size_bytes", 0))
        components = ", ".join(b.get("components", []))[:24]
        created = b.get("created_at", "-")[:19]

        print(f"{backup_id:<25} {name:<20} {size:<12} {components:<25} {created:<20}")

    print(f"\nTotal size: {_format_size(result.get('total_size_bytes', 0))}")
    print(f"Backup directory: {result.get('backup_dir', 'data/backups')}")
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


def _backup_create(args) -> int:
    """Create a new backup."""
    name = getattr(args, "name", None)
    description = getattr(args, "description", None)
    components = getattr(args, "components", None)

    data = {}
    if name:
        data["name"] = name
    if description:
        data["description"] = description
    if components:
        data["components"] = components

    result = cloud_api_request("POST", "/api/backup", data=data)

    if result.get("success"):
        print(f"Backup created successfully!")
        print(f"  ID: {result.get('backup_id')}")
        print(f"  Path: {result.get('path')}")
        print(f"  Size: {_format_size(result.get('size_bytes', 0))}")
        print(f"  Components: {', '.join(result.get('components', []))}")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to create backup')}")
        return 1


def _backup_get(args) -> int:
    """Get details of a specific backup."""
    backup_id = getattr(args, "backup_id", None)
    output_format = getattr(args, "format", "table")

    if not backup_id:
        print("Error: Backup ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/backup/{backup_id}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Backup: {result.get('id')}")
    print("=" * 50)
    print(f"  Name: {result.get('name')}")
    print(f"  Created: {result.get('created_at')}")
    print(f"  Size: {_format_size(result.get('size_bytes', 0))}")
    print(f"  Components: {', '.join(result.get('components', []))}")
    if result.get("description"):
        print(f"  Description: {result.get('description')}")
    if result.get("created_by"):
        print(f"  Created by: User {result.get('created_by')}")

    return 0


def _backup_restore(args) -> int:
    """Restore from a backup."""
    backup_id = getattr(args, "backup_id", None)
    components = getattr(args, "components", None)
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)

    if not backup_id:
        print("Error: Backup ID is required.")
        return 1

    if not force and not dry_run:
        confirm = input(f"Restore from backup {backup_id}? This will overwrite existing data. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    data = {
        "backup_id": backup_id,
        "dry_run": dry_run,
    }
    if components:
        data["components"] = components

    result = cloud_api_request("POST", f"/api/backup/{backup_id}/restore", data=data)

    if result.get("success"):
        if result.get("dry_run"):
            print("Dry run - no changes made.")
            print(f"Would restore: {', '.join(result.get('restored_components', []))}")
        else:
            print("Restore completed successfully!")
            print(f"Restored: {', '.join(result.get('restored_components', []))}")

        warnings = result.get("warnings", [])
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to restore backup')}")
        return 1


def _backup_delete(args) -> int:
    """Delete a backup."""
    backup_id = getattr(args, "backup_id", None)
    force = getattr(args, "force", False)

    if not backup_id:
        print("Error: Backup ID is required.")
        return 1

    if not force:
        confirm = input(f"Delete backup {backup_id}? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/backup/{backup_id}")

    if result.get("success"):
        print(f"Backup {backup_id} deleted.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Failed to delete backup')}")
        return 1
