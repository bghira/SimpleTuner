"""
Database management commands.

Handles database health, verify, vacuum, migrations, and stats commands.
"""

import json

from .cloud.api import cloud_api_request


def cmd_database(args) -> int:
    """Manage databases."""
    db_action = getattr(args, "db_action", None)

    if db_action == "health":
        return _db_health(args)
    elif db_action == "verify":
        return _db_verify(args)
    elif db_action == "vacuum":
        return _db_vacuum(args)
    elif db_action == "migrations":
        return _db_migrations(args)
    elif db_action == "migrate":
        return _db_migrate(args)
    elif db_action == "stats":
        return _db_stats(args)
    else:
        print("Error: Unknown database action. Use 'simpletuner database --help'.")
        return 1


def _db_health(args) -> int:
    """Check database health."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/database/health")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    healthy = result.get("healthy", False)
    status = "[+] HEALTHY" if healthy else "[X] UNHEALTHY"
    print(f"Database Health: {status}")
    print("=" * 60)

    print(f"\nTotal Size: {_format_size(result.get('total_size_bytes', 0))}")

    print("\nDatabases:")
    for db in result.get("databases", []):
        exists = "[+]" if db.get("exists") else "[-]"
        name = db.get("name", "unknown")
        size = _format_size(db.get("size_bytes", 0)) if db.get("exists") else "N/A"
        tables = db.get("table_count", 0)
        print(f"  {exists} {name:<15} {size:<12} ({tables} tables)")

    issues = result.get("issues", [])
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  [!] {issue}")

    return 0 if healthy else 1


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


def _db_verify(args) -> int:
    """Verify database integrity."""
    database = getattr(args, "database", None)
    output_format = getattr(args, "format", "table")

    if not database:
        print("Error: Database name is required.")
        return 1

    result = cloud_api_request("GET", f"/api/database/verify/{database}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    valid = result.get("valid", False)
    status = "[+] VALID" if valid else "[X] INVALID"
    print(f"Database: {database}")
    print(f"Integrity: {status}")
    print(f"Integrity Check: {result.get('integrity_check', 'unknown')}")

    fk_issues = result.get("foreign_key_check", [])
    if fk_issues:
        print("\nForeign Key Issues:")
        for fk in fk_issues:
            print(f"  - {fk}")

    issues = result.get("issues", [])
    if issues:
        print("\nOther Issues:")
        for issue in issues:
            print(f"  - {issue}")

    return 0 if valid else 1


def _db_vacuum(args) -> int:
    """Vacuum a database to reclaim space."""
    database = getattr(args, "database", None)
    force = getattr(args, "force", False)

    if not database:
        print("Error: Database name is required.")
        return 1

    if not force:
        confirm = input(f"Vacuum database '{database}'? This may briefly lock the database. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("POST", f"/api/database/vacuum/{database}")

    if result.get("success"):
        print(f"Vacuum completed for {database}!")
        print(f"  Size before: {_format_size(result.get('size_before', 0))}")
        print(f"  Size after:  {_format_size(result.get('size_after', 0))}")
        print(f"  Freed:       {_format_size(result.get('freed_bytes', 0))}")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Vacuum failed')}")
        return 1


def _db_migrations(args) -> int:
    """Show migration status."""
    database = getattr(args, "database", "auth")
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", f"/api/database/migrations?database={database}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"Migration Status: {database}")
    print("=" * 60)
    print(f"Current Version: {result.get('current_version', 'None')}")

    applied = result.get("applied_migrations", [])
    if applied:
        print("\nApplied Migrations:")
        for m in applied:
            print(f"  [+] {m.get('version')} - {m.get('name')}")
            if m.get("applied_at"):
                print(f"      Applied: {m.get('applied_at')}")

    pending = result.get("pending_migrations", [])
    if pending:
        print("\nPending Migrations:")
        for m in pending:
            print(f"  [-] {m.get('version')} - {m.get('name')}")
    elif not applied:
        print("\nNo migrations found.")
    else:
        print("\nNo pending migrations.")

    return 0


def _db_migrate(args) -> int:
    """Run pending migrations."""
    database = getattr(args, "database", "auth")
    dry_run = getattr(args, "dry_run", False)
    force = getattr(args, "force", False)

    if not force and not dry_run:
        confirm = input(f"Run migrations for '{database}'? Make a backup first! [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    params = f"?database={database}"
    if dry_run:
        params += "&dry_run=true"

    result = cloud_api_request("POST", f"/api/database/migrations/run{params}")

    if result.get("success"):
        if result.get("dry_run"):
            print("Dry run - no changes made.")
            pending = result.get("pending", [])
            if pending:
                print(f"Would run {len(pending)} migration(s):")
                for m in pending:
                    print(f"  - {m}")
            else:
                print("No pending migrations.")
        else:
            migrations = result.get("migrations_run", [])
            if migrations:
                print(f"Ran {len(migrations)} migration(s):")
                for m in migrations:
                    print(f"  [+] {m}")
            else:
                print("No migrations to run.")
        return 0
    else:
        print(f"Error: {result.get('detail', 'Migration failed')}")
        return 1


def _db_stats(args) -> int:
    """Show database statistics."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/database/stats")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Database Statistics")
    print("=" * 60)

    for db_name, stats in result.items():
        if not stats.get("exists"):
            print(f"\n{db_name}: not found")
            continue

        print(f"\n{db_name}:")
        print(f"  Size: {_format_size(stats.get('size_bytes', 0))}")
        print(f"  Pages: {stats.get('page_count', 0)} x {stats.get('page_size', 0)} bytes")
        print(f"  Free pages: {stats.get('freelist_count', 0)}")

        tables = stats.get("tables", {})
        if tables:
            print("  Tables:")
            for table_name, table_stats in tables.items():
                if "error" in table_stats:
                    print(f"    - {table_name}: {table_stats['error']}")
                else:
                    print(f"    - {table_name}: {table_stats.get('row_count', 0)} rows")

    return 0
