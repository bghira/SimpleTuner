"""
SimpleTuner CLI - Command-line interface for SimpleTuner.

This module provides the main entry point and argument parser for the CLI.
"""

import os

# Skip torch import for fast CLI startup (we're always rank 0 in CLI mode)
os.environ.setdefault("SIMPLETUNER_SKIP_TORCH", "1")

import argparse

import simpletuner.helpers.log_format  # noqa: F401

from .common import get_version


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="simpletuner",
        description="SimpleTuner - Fine-tune diffusion models with ease",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"SimpleTuner {get_version()}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        metavar="<command>",
        help="Available commands",
    )

    # --- train command ---
    _add_train_parser(subparsers)

    # --- examples command ---
    _add_examples_parser(subparsers)

    # --- configure command ---
    _add_configure_parser(subparsers)

    # --- server command ---
    _add_server_parser(subparsers)

    # --- cloud command ---
    from .cloud import add_cloud_parser

    add_cloud_parser(subparsers)

    # --- jobs command (local job queue) ---
    _add_jobs_parser(subparsers)

    # --- quota command ---
    _add_quota_parser(subparsers)

    # --- notifications command ---
    _add_notifications_parser(subparsers)

    # --- auth command ---
    from .auth import add_auth_parser

    add_auth_parser(subparsers)

    # --- backup command ---
    _add_backup_parser(subparsers)

    # --- database command ---
    _add_database_parser(subparsers)

    # --- metrics command ---
    _add_metrics_parser(subparsers)

    # --- webhooks command ---
    _add_webhooks_parser(subparsers)

    return parser


def _add_train_parser(subparsers):
    """Add the train command parser."""
    from . import train

    train_parser = subparsers.add_parser(
        "train",
        help="Run training",
        description="Run SimpleTuner training with a configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner train                           # Use config.json in current directory
  simpletuner train --example sdxl-lora       # Use a built-in example
  simpletuner train --env my-config           # Use a named environment
  simpletuner train example=sdxl-lora         # Alternative syntax
""",
    )
    train_parser.set_defaults(func=train.cmd_train)

    train_group = train_parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "--example",
        "-e",
        help="Run training with a built-in example",
    )
    train_group.add_argument(
        "--env",
        help="Run training with a named environment",
    )

    train_parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments passed to train.py (e.g., key=value pairs)",
    )


def _add_examples_parser(subparsers):
    """Add the examples command parser."""
    from . import examples

    examples_parser = subparsers.add_parser(
        "examples",
        help="Manage training examples",
        description="List and copy built-in training examples.",
    )
    examples_parser.set_defaults(func=examples.cmd_examples)

    examples_subparsers = examples_parser.add_subparsers(
        dest="action",
        title="Actions",
        metavar="<action>",
    )

    # examples list
    list_parser = examples_subparsers.add_parser(
        "list",
        help="List available examples",
    )
    list_parser.set_defaults(action="list")

    # examples copy
    copy_parser = examples_subparsers.add_parser(
        "copy",
        help="Copy an example to the current directory",
    )
    copy_parser.set_defaults(action="copy")
    copy_parser.add_argument("name", help="Name of the example to copy")
    copy_parser.add_argument(
        "--dest",
        "-d",
        default=".",
        help="Destination directory (default: current directory)",
    )


def _add_configure_parser(subparsers):
    """Add the configure command parser."""
    from . import configure

    configure_parser = subparsers.add_parser(
        "configure",
        help="Run the configuration wizard",
        description="Interactive configuration wizard for SimpleTuner.",
    )
    configure_parser.set_defaults(func=configure.cmd_configure)
    configure_parser.add_argument(
        "output_file",
        nargs="?",
        default="config.json",
        help="Output configuration file (default: config.json)",
    )


def _add_server_parser(subparsers):
    """Add the server command parser."""
    from . import server

    server_parser = subparsers.add_parser(
        "server",
        help="Start the web server",
        description="Start the SimpleTuner web UI and API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  unified   - Both API and callback server (default)
  trainer   - API server only
  callback  - Callback server only

Examples:
  simpletuner server                           # Start on default port 8001
  simpletuner server --port 8080               # Start on custom port
  simpletuner server --ssl                     # Enable HTTPS with auto-generated cert
  simpletuner server --env my-config           # Auto-start training with config
""",
    )
    server_parser.set_defaults(func=server.cmd_server)

    server_parser.add_argument(
        "--host",
        "-H",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to bind to (default: 8001 for trainer/unified, 8002 for callback)",
    )
    server_parser.add_argument(
        "--reload",
        "-r",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    server_parser.add_argument(
        "--mode",
        "-m",
        choices=["trainer", "callback", "unified"],
        default="unified",
        help="Server mode (default: unified)",
    )
    server_parser.add_argument(
        "--ssl",
        action="store_true",
        help="Enable HTTPS (generates self-signed certificate if needed)",
    )
    server_parser.add_argument(
        "--ssl-key",
        help="Path to SSL private key file",
    )
    server_parser.add_argument(
        "--ssl-cert",
        help="Path to SSL certificate file",
    )
    server_parser.add_argument(
        "--ssl-no-verify",
        action="store_true",
        help="Disable SSL certificate verification for callbacks",
    )
    server_parser.add_argument(
        "--env",
        "-e",
        help="Environment to auto-start training with",
    )


def _add_jobs_parser(subparsers):
    """Add the jobs command parser for local job queue."""
    from . import jobs

    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage local job queue",
        description="Submit, monitor, and manage local training jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner jobs list
  simpletuner jobs submit my-config
  simpletuner jobs logs <job_id> --follow
  simpletuner jobs cancel <job_id>
""",
    )
    jobs_parser.set_defaults(func=jobs.cmd_jobs)

    jobs_subparsers = jobs_parser.add_subparsers(
        dest="jobs_action",
        title="Job commands",
        metavar="<action>",
    )

    # list
    list_parser = jobs_subparsers.add_parser(
        "list",
        help="List local jobs",
        description="List jobs in the local queue.",
    )
    list_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)",
    )
    list_parser.add_argument(
        "--status",
        "-s",
        choices=["pending", "queued", "running", "completed", "failed", "cancelled"],
        help="Filter by job status",
    )
    list_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Comma-separated list of fields to output (e.g., job_id,status,config_name,created_at)",
    )

    # submit
    submit_parser = jobs_subparsers.add_parser(
        "submit",
        help="Submit a training job",
        description="Submit a training configuration to the local queue.",
    )
    submit_parser.add_argument("config", help="Configuration name to run")
    submit_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be submitted without actually submitting",
    )

    # cancel
    cancel_parser = jobs_subparsers.add_parser(
        "cancel",
        help="Cancel a job",
        description="Cancel one or more jobs that are currently running or queued.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner jobs cancel <job_id>
  simpletuner jobs cancel --status running
  simpletuner jobs cancel --status pending
  simpletuner jobs cancel --all
""",
    )
    cancel_parser.add_argument("job_id", nargs="?", help="Job ID to cancel")
    cancel_parser.add_argument(
        "--status",
        "-s",
        choices=["running", "pending", "queued"],
        help="Cancel all jobs with this status",
    )
    cancel_parser.add_argument(
        "--all",
        action="store_true",
        help="Cancel all running and pending jobs",
    )
    cancel_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt for bulk cancel",
    )

    # delete
    delete_parser = jobs_subparsers.add_parser(
        "delete",
        help="Delete a job",
        description="Remove a job from the queue.",
    )
    delete_parser.add_argument("job_id", help="Job ID to delete")
    delete_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # retry
    retry_parser = jobs_subparsers.add_parser(
        "retry",
        help="Retry a failed job",
        description="Resubmit a failed or cancelled job.",
    )
    retry_parser.add_argument("job_id", help="Job ID to retry")

    # logs
    logs_parser = jobs_subparsers.add_parser(
        "logs",
        help="Fetch job logs",
        description="View training logs from a job.",
    )
    logs_parser.add_argument("job_id", help="Job ID to get logs for")
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow logs in real-time",
    )

    # get
    get_parser = jobs_subparsers.add_parser(
        "get",
        help="Get job details",
        description="Show detailed information about a job.",
    )
    get_parser.add_argument("job_id", help="Job ID to get details for")
    get_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # status
    status_parser = jobs_subparsers.add_parser(
        "status",
        help="Get queue status",
        description="Show job queue status summary.",
    )
    status_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # purge
    purge_parser = jobs_subparsers.add_parser(
        "purge",
        help="Bulk delete jobs by status",
        description="Delete multiple jobs matching a status filter.",
    )
    purge_parser.add_argument(
        "--status",
        "-s",
        choices=["pending", "queued", "completed", "failed", "cancelled"],
        action="append",
        help="Status(es) to purge (can specify multiple, e.g., -s failed -s cancelled)",
    )
    purge_parser.add_argument(
        "--all",
        action="store_true",
        help="Purge all non-running jobs (pending, queued, completed, failed, cancelled)",
    )
    purge_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # approval subcommand group
    approval_parser = jobs_subparsers.add_parser(
        "approval",
        help="Manage job approval workflow",
        description="View and manage job approval requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner jobs approval pending
  simpletuner jobs approval approve 123 --reason "Approved"
  simpletuner jobs approval reject 123 --reason "Cost too high"
  simpletuner jobs approval rules
""",
    )

    approval_subparsers = approval_parser.add_subparsers(
        dest="approval_action",
        title="Approval commands",
        metavar="<action>",
    )

    # approval list
    approval_list = approval_subparsers.add_parser(
        "list",
        help="List approval requests",
        description="List all approval requests with optional filtering.",
    )
    approval_list.add_argument(
        "--status",
        choices=["pending", "approved", "rejected"],
        help="Filter by approval status",
    )
    approval_list.add_argument("--limit", "-l", type=int, default=50)
    approval_list.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # approval pending
    approval_pending = approval_subparsers.add_parser(
        "pending",
        help="List pending approval requests",
        description="Show only pending approval requests.",
    )
    approval_pending.add_argument("--limit", "-l", type=int, default=50)
    approval_pending.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # approval approve
    approval_approve = approval_subparsers.add_parser(
        "approve",
        help="Approve a pending request",
        description="Approve a job that is waiting for approval.",
    )
    approval_approve.add_argument("approval_id", type=int, help="Approval request ID")
    approval_approve.add_argument("--reason", "-r", help="Reason for approval")

    # approval reject
    approval_reject = approval_subparsers.add_parser(
        "reject",
        help="Reject a pending request",
        description="Reject a job that is waiting for approval.",
    )
    approval_reject.add_argument("approval_id", type=int, help="Approval request ID")
    approval_reject.add_argument("--reason", "-r", help="Reason for rejection")

    # approval rules
    approval_rules = approval_subparsers.add_parser(
        "rules",
        help="List approval rules",
        description="Show configured approval rules.",
    )
    approval_rules.add_argument("--format", "-f", choices=["table", "json"], default="table")


def _add_quota_parser(subparsers):
    """Add the quota command parser."""
    from . import quota

    quota_parser = subparsers.add_parser(
        "quota",
        help="Manage usage quotas",
        description="Configure and monitor usage quotas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner quota list
  simpletuner quota status
  simpletuner quota create --type concurrent_jobs --limit 5
""",
    )
    quota_parser.set_defaults(func=quota.cmd_quota)

    quota_subparsers = quota_parser.add_subparsers(
        dest="quota_action",
        title="Quota commands",
        metavar="<action>",
    )

    # list
    list_parser = quota_subparsers.add_parser("list", help="List configured quotas")
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # create
    create_parser = quota_subparsers.add_parser("create", help="Create a quota")
    create_parser.add_argument("--type", "-t", required=True, help="Quota type")
    create_parser.add_argument("--limit", "-l", type=float, required=True, help="Limit value")
    create_parser.add_argument(
        "--action",
        "-a",
        choices=["block", "warn", "require_approval"],
        default="block",
        help="Action when quota exceeded (default: block)",
    )
    create_parser.add_argument("--user-id", type=int, help="Apply to specific user")
    create_parser.add_argument("--team-id", type=int, help="Apply to specific team")
    create_parser.add_argument("--org-id", type=int, help="Apply to specific organization")

    # delete
    delete_parser = quota_subparsers.add_parser("delete", help="Delete a quota")
    delete_parser.add_argument("quota_id", type=int, help="Quota ID to delete")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # status
    status_parser = quota_subparsers.add_parser("status", help="Show quota usage status")
    status_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # types
    types_parser = quota_subparsers.add_parser("types", help="List available quota types")
    types_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")


def _add_notifications_parser(subparsers):
    """Add the notifications command parser."""
    from . import notifications

    notif_parser = subparsers.add_parser(
        "notifications",
        help="Manage notification channels",
        description="Configure notification channels (email, webhook, slack) and preferences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner notifications channels
  simpletuner notifications status
  simpletuner notifications channel-create --type email --name "My Email"
""",
    )
    notif_parser.set_defaults(func=notifications.cmd_notifications)

    notif_subparsers = notif_parser.add_subparsers(
        dest="notif_action",
        title="Notification commands",
        metavar="<action>",
    )

    # channels
    channels_parser = notif_subparsers.add_parser("channels", help="List notification channels")
    channels_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # channel-get
    channel_get = notif_subparsers.add_parser("channel-get", help="Get a channel by ID")
    channel_get.add_argument("channel_id", type=int, help="Channel ID")
    channel_get.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # channel-create
    channel_create = notif_subparsers.add_parser("channel-create", help="Create a channel")
    channel_create.add_argument("--type", "-t", required=True, choices=["email", "webhook", "slack"])
    channel_create.add_argument("--name", "-n", required=True, help="Channel name")
    channel_create.add_argument("--enabled", action="store_true", default=True)
    channel_create.add_argument("--smtp-host", help="SMTP host")
    channel_create.add_argument("--smtp-port", type=int, default=587)
    channel_create.add_argument("--smtp-username", help="SMTP username")
    channel_create.add_argument("--smtp-password", help="SMTP password")
    channel_create.add_argument("--smtp-use-tls", action="store_true", default=True)
    channel_create.add_argument("--smtp-from-address", help="From email address")
    channel_create.add_argument("--smtp-from-name", help="From name")
    channel_create.add_argument("--webhook-url", help="Webhook URL")
    channel_create.add_argument("--webhook-secret", help="Webhook secret")

    # channel-update
    channel_update = notif_subparsers.add_parser("channel-update", help="Update a channel")
    channel_update.add_argument("channel_id", type=int, help="Channel ID")
    channel_update.add_argument("--name", "-n", help="New name")
    channel_update.add_argument("--enabled", type=bool, help="Enable/disable")

    # channel-delete
    channel_delete = notif_subparsers.add_parser("channel-delete", help="Delete a channel")
    channel_delete.add_argument("channel_id", type=int, help="Channel ID")

    # channel-test
    channel_test = notif_subparsers.add_parser("channel-test", help="Test a channel")
    channel_test.add_argument("channel_id", type=int, help="Channel ID")

    # preferences
    prefs_parser = notif_subparsers.add_parser("preferences", help="List preferences")
    prefs_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # preference-create
    pref_create = notif_subparsers.add_parser("preference-create", help="Create a preference")
    pref_create.add_argument("--event-type", "-e", required=True, help="Event type")
    pref_create.add_argument("--channel-id", "-c", type=int, required=True, help="Channel ID")
    pref_create.add_argument("--recipients", "-r", help="Comma-separated recipients")
    pref_create.add_argument("--min-severity", choices=["info", "warning", "error", "critical"], default="info")
    pref_create.add_argument("--enabled", action="store_true", default=True)

    # preference-delete
    pref_delete = notif_subparsers.add_parser("preference-delete", help="Delete a preference")
    pref_delete.add_argument("preference_id", type=int, help="Preference ID")

    # events
    events_parser = notif_subparsers.add_parser("events", help="List available event types")
    events_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # presets
    presets_parser = notif_subparsers.add_parser("presets", help="List email provider presets")
    presets_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # preset-get
    preset_get = notif_subparsers.add_parser("preset-get", help="Get an email preset")
    preset_get.add_argument("preset_id", help="Preset ID (gmail, outlook, ses)")
    preset_get.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # history
    history_parser = notif_subparsers.add_parser("history", help="Get delivery history")
    history_parser.add_argument("--limit", "-l", type=int, default=50)
    history_parser.add_argument("--channel-id", "-c", type=int, help="Filter by channel")
    history_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # status
    status_parser = notif_subparsers.add_parser("status", help="Get system status")
    status_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # skip
    notif_subparsers.add_parser("skip", help="Dismiss the setup prompt")


def _add_backup_parser(subparsers):
    """Add the backup command parser."""
    from . import backup

    backup_parser = subparsers.add_parser(
        "backup",
        help="Manage backups",
        description="Create, list, and restore backups of SimpleTuner data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner backup list
  simpletuner backup create --name "pre-upgrade"
  simpletuner backup restore <backup_id> --dry-run
""",
    )
    backup_parser.set_defaults(func=backup.cmd_backup)

    backup_subparsers = backup_parser.add_subparsers(
        dest="backup_action",
        title="Backup commands",
        metavar="<action>",
    )

    # list
    list_parser = backup_subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument("--limit", "-l", type=int, default=50, help="Maximum backups to show")
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # create
    create_parser = backup_subparsers.add_parser("create", help="Create a new backup")
    create_parser.add_argument("--name", "-n", help="Backup name")
    create_parser.add_argument("--description", "-d", help="Backup description")
    create_parser.add_argument(
        "--components",
        "-c",
        nargs="+",
        choices=["jobs", "auth", "config", "approvals", "notifications"],
        help="Components to backup (default: jobs, auth, config)",
    )

    # get
    get_parser = backup_subparsers.add_parser("get", help="Get backup details")
    get_parser.add_argument("backup_id", help="Backup ID")
    get_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # restore
    restore_parser = backup_subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument(
        "--components",
        "-c",
        nargs="+",
        help="Components to restore (default: all from backup)",
    )
    restore_parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without restoring")
    restore_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # delete
    delete_parser = backup_subparsers.add_parser("delete", help="Delete a backup")
    delete_parser.add_argument("backup_id", help="Backup ID to delete")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")


def _add_database_parser(subparsers):
    """Add the database command parser."""
    from . import database

    db_parser = subparsers.add_parser(
        "database",
        help="Manage databases",
        description="Database health checks, verification, and maintenance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner database health
  simpletuner database verify auth
  simpletuner database vacuum jobs
  simpletuner database migrations --database auth
""",
    )
    db_parser.set_defaults(func=database.cmd_database)

    db_subparsers = db_parser.add_subparsers(
        dest="db_action",
        title="Database commands",
        metavar="<action>",
    )

    # health
    health_parser = db_subparsers.add_parser("health", help="Check database health")
    health_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # verify
    verify_parser = db_subparsers.add_parser("verify", help="Verify database integrity")
    verify_parser.add_argument("database", choices=["jobs", "auth", "approvals", "notifications", "webui"])
    verify_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # vacuum
    vacuum_parser = db_subparsers.add_parser("vacuum", help="Vacuum a database")
    vacuum_parser.add_argument("database", choices=["jobs", "auth", "approvals", "notifications", "webui"])
    vacuum_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # migrations
    migrations_parser = db_subparsers.add_parser("migrations", help="Show migration status")
    migrations_parser.add_argument("--database", "-d", default="auth", help="Database to check")
    migrations_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # migrate
    migrate_parser = db_subparsers.add_parser("migrate", help="Run pending migrations")
    migrate_parser.add_argument("--database", "-d", default="auth", help="Database to migrate")
    migrate_parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without running")
    migrate_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # stats
    stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")


def _add_metrics_parser(subparsers):
    """Add the metrics command parser."""
    from . import metrics

    metrics_parser = subparsers.add_parser(
        "metrics",
        help="View metrics and monitoring",
        description="System metrics, health checks, and cost tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner metrics status
  simpletuner metrics health
  simpletuner metrics costs --period month
  simpletuner metrics prometheus --raw
""",
    )
    metrics_parser.set_defaults(func=metrics.cmd_metrics)

    metrics_subparsers = metrics_parser.add_subparsers(
        dest="metrics_action",
        title="Metrics commands",
        metavar="<action>",
    )

    # status
    status_parser = metrics_subparsers.add_parser("status", help="Show metrics system status")
    status_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # health
    health_parser = metrics_subparsers.add_parser("health", help="Show system health")
    health_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # prometheus
    prom_parser = metrics_subparsers.add_parser("prometheus", help="Prometheus metrics")
    prom_parser.add_argument("--raw", "-r", action="store_true", help="Show raw Prometheus format")
    prom_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # costs
    costs_parser = metrics_subparsers.add_parser("costs", help="Show cost metrics")
    costs_parser.add_argument("--period", "-p", choices=["day", "week", "month"], default="month")
    costs_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # usage
    usage_parser = metrics_subparsers.add_parser("usage", help="Show resource usage")
    usage_parser.add_argument("--period", "-p", choices=["day", "week", "month"], default="day")
    usage_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")


def _add_webhooks_parser(subparsers):
    """Add the webhooks command parser."""
    from . import webhooks

    webhooks_parser = subparsers.add_parser(
        "webhooks",
        help="Manage webhooks",
        description="Configure webhook endpoints for event notifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner webhooks list
  simpletuner webhooks create --name "My Hook" --url https://example.com/hook
  simpletuner webhooks test 1
  simpletuner webhooks history
""",
    )
    webhooks_parser.set_defaults(func=webhooks.cmd_webhooks)

    webhooks_subparsers = webhooks_parser.add_subparsers(
        dest="webhook_action",
        title="Webhook commands",
        metavar="<action>",
    )

    # list
    list_parser = webhooks_subparsers.add_parser("list", help="List webhooks")
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # get
    get_parser = webhooks_subparsers.add_parser("get", help="Get webhook details")
    get_parser.add_argument("webhook_id", type=int, help="Webhook ID")
    get_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # create
    create_parser = webhooks_subparsers.add_parser("create", help="Create a webhook")
    create_parser.add_argument("--name", "-n", required=True, help="Webhook name")
    create_parser.add_argument("--url", "-u", required=True, help="Webhook URL")
    create_parser.add_argument("--events", "-e", nargs="+", help="Event types to subscribe to")
    create_parser.add_argument("--secret", "-s", help="Webhook signing secret")
    create_parser.add_argument("--disabled", action="store_true", help="Create in disabled state")

    # update
    update_parser = webhooks_subparsers.add_parser("update", help="Update a webhook")
    update_parser.add_argument("webhook_id", type=int, help="Webhook ID")
    update_parser.add_argument("--name", "-n", help="New name")
    update_parser.add_argument("--url", "-u", help="New URL")
    update_parser.add_argument("--events", "-e", nargs="+", help="New event types")
    update_parser.add_argument("--secret", "-s", help="New secret")
    update_parser.add_argument("--enable", action="store_true", help="Enable webhook")
    update_parser.add_argument("--disable", action="store_true", help="Disable webhook")

    # delete
    delete_parser = webhooks_subparsers.add_parser("delete", help="Delete a webhook")
    delete_parser.add_argument("webhook_id", type=int, help="Webhook ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # test
    test_parser = webhooks_subparsers.add_parser("test", help="Send test event")
    test_parser.add_argument("webhook_id", type=int, help="Webhook ID")
    test_parser.add_argument("--event-type", "-e", default="test", help="Event type to send")

    # history
    history_parser = webhooks_subparsers.add_parser("history", help="Show delivery history")
    history_parser.add_argument("--webhook-id", "-w", type=int, help="Filter by webhook")
    history_parser.add_argument("--limit", "-l", type=int, default=20, help="Maximum entries")
    history_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # events
    events_parser = webhooks_subparsers.add_parser("events", help="List available event types")
    events_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")


def main() -> int:
    """Main entry point for the SimpleTuner CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1
