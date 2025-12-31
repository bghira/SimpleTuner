"""
Cloud training CLI commands.

This module provides the command dispatcher and parser configuration
for all cloud-related subcommands.
"""

import argparse

from . import config, cost_limit, jobs


def add_cloud_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the cloud command parser and all subcommands."""
    cloud_parser = subparsers.add_parser(
        "cloud",
        help="Manage cloud training",
        description="Cloud provider configuration and cloud training jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner cloud jobs submit my-config
  simpletuner cloud jobs list --status running
  simpletuner cloud jobs logs <job_id> --follow
  simpletuner cloud config show
  simpletuner cloud cost-limit set --amount 50
""",
    )
    cloud_parser.set_defaults(func=cmd_cloud)

    cloud_subparsers = cloud_parser.add_subparsers(
        dest="cloud_action",
        title="Cloud commands",
        metavar="<command>",
    )

    # --- Jobs Subcommand Group ---
    _add_jobs_parser(cloud_subparsers)

    # --- Configuration Commands ---
    _add_config_parser(cloud_subparsers)
    _add_cost_limit_parser(cloud_subparsers)

    # --- Status Command (cloud-level) ---
    _add_status_parser(cloud_subparsers)

    return cloud_parser


def cmd_cloud(args) -> int:
    """Main cloud command dispatcher."""
    cloud_action = getattr(args, "cloud_action", None)

    if cloud_action is None:
        print("Error: Please specify a cloud command.")
        print("Usage: simpletuner cloud <command>")
        print("\nRun 'simpletuner cloud --help' for available commands.")
        return 1

    # Jobs subcommand group
    if cloud_action == "jobs":
        return _dispatch_cloud_jobs(args)

    # Configuration commands
    elif cloud_action == "config":
        return config.cmd_cloud_config(args)
    elif cloud_action == "cost-limit":
        return cost_limit.cmd_cloud_cost_limit(args)

    # Status command
    elif cloud_action == "status":
        return jobs.cmd_cloud_status(args)

    else:
        print(f"Error: Unknown cloud command '{cloud_action}'")
        return 1


def _dispatch_cloud_jobs(args) -> int:
    """Dispatch cloud jobs subcommands."""
    jobs_action = getattr(args, "cloud_jobs_action", None)

    if jobs_action is None:
        print("Error: Please specify a jobs command.")
        print("Usage: simpletuner cloud jobs <command>")
        print("\nRun 'simpletuner cloud jobs --help' for available commands.")
        return 1

    if jobs_action == "submit":
        return jobs.cmd_cloud_submit(args)
    elif jobs_action == "list":
        return jobs.cmd_cloud_list(args)
    elif jobs_action == "cancel":
        return jobs.cmd_cloud_cancel(args)
    elif jobs_action == "delete":
        return jobs.cmd_cloud_delete(args)
    elif jobs_action == "retry":
        return jobs.cmd_cloud_retry(args)
    elif jobs_action == "logs":
        return jobs.cmd_cloud_logs(args)
    elif jobs_action == "get":
        return jobs.cmd_cloud_get(args)
    else:
        print(f"Error: Unknown cloud jobs command '{jobs_action}'")
        return 1


# --- Parser Builders ---


def _add_jobs_parser(subparsers):
    """Add jobs subcommand group parser."""
    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage cloud training jobs",
        description="Submit, monitor, and manage cloud training jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner cloud jobs submit my-config
  simpletuner cloud jobs list --status running
  simpletuner cloud jobs logs <job_id> --follow
  simpletuner cloud jobs cancel <job_id>
""",
    )

    jobs_subparsers = jobs_parser.add_subparsers(
        dest="cloud_jobs_action",
        title="Job commands",
        metavar="<action>",
    )

    # submit
    submit_parser = jobs_subparsers.add_parser(
        "submit",
        help="Submit a training job to the cloud",
        description="Submit a training configuration to run on a cloud provider.",
    )
    submit_parser.add_argument("config", help="Configuration name to run")
    submit_parser.add_argument(
        "--provider",
        "-p",
        default="replicate",
        help="Cloud provider (default: replicate)",
    )
    submit_parser.add_argument(
        "--idempotency-key",
        help="Idempotency key to prevent duplicate submissions",
    )
    submit_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be uploaded and submitted without actually doing it",
    )

    # list
    list_parser = jobs_subparsers.add_parser(
        "list",
        help="List cloud training jobs",
        description="List training jobs with optional filtering.",
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
        "--provider",
        "-p",
        help="Filter by cloud provider",
    )
    list_parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync active job statuses with provider before listing",
    )
    list_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # cancel
    cancel_parser = jobs_subparsers.add_parser(
        "cancel",
        help="Cancel a running cloud job",
        description="Cancel a job that is currently running or queued.",
    )
    cancel_parser.add_argument("job_id", help="Job ID to cancel")

    # delete
    delete_parser = jobs_subparsers.add_parser(
        "delete",
        help="Delete a job from local history",
        description="Remove a job from the local job database.",
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
        help="Retry a failed or cancelled job",
        description="Resubmit a job with the same configuration.",
    )
    retry_parser.add_argument("job_id", help="Job ID to retry")

    # logs
    logs_parser = jobs_subparsers.add_parser(
        "logs",
        help="Fetch logs for a cloud job",
        description="View training logs from a cloud job.",
    )
    logs_parser.add_argument("job_id", help="Job ID to get logs for")
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow logs in real-time (poll for updates)",
    )

    # get
    get_parser = jobs_subparsers.add_parser(
        "get",
        help="Get details for a specific job",
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


def _add_status_parser(subparsers):
    """Add status subcommand parser."""
    parser = subparsers.add_parser(
        "status",
        help="Get cloud system status",
        description="Check the health of the cloud system and providers.",
    )
    parser.add_argument(
        "--replicate",
        action="store_true",
        help="Include Replicate API status",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def _add_config_parser(subparsers):
    """Add config subcommand parser."""
    config_parser = subparsers.add_parser(
        "config",
        help="Manage provider configuration",
        description="View and modify cloud provider settings.",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_action",
        title="Config commands",
        metavar="<action>",
    )

    # show
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show provider configuration",
    )
    show_parser.add_argument("--provider", "-p", default="replicate")
    show_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # set-token
    token_parser = config_subparsers.add_parser(
        "set-token",
        help="Set provider API token",
    )
    token_parser.add_argument("--provider", "-p", default="replicate")
    token_parser.add_argument("--token", "-t", help="API token (will prompt if not provided)")

    # delete-token
    del_token_parser = config_subparsers.add_parser(
        "delete-token",
        help="Delete provider API token",
    )
    del_token_parser.add_argument("--provider", "-p", default="replicate")

    # set
    set_parser = config_subparsers.add_parser(
        "set",
        help="Set a configuration option",
    )
    set_parser.add_argument("--provider", "-p", default="replicate")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", nargs="?", help="Configuration value")


def _add_cost_limit_parser(subparsers):
    """Add cost-limit subcommand parser."""
    limit_parser = subparsers.add_parser(
        "cost-limit",
        help="Manage cost limits",
        description="Configure spending limits for cloud providers.",
    )
    limit_subparsers = limit_parser.add_subparsers(
        dest="limit_action",
        title="Cost limit commands",
        metavar="<action>",
    )

    # show
    show_parser = limit_subparsers.add_parser("show", help="Show current cost limit")
    show_parser.add_argument("--provider", "-p", default="replicate")

    # set
    set_parser = limit_subparsers.add_parser("set", help="Set cost limit")
    set_parser.add_argument("--provider", "-p", default="replicate")
    set_parser.add_argument("--amount", "-a", type=float, required=True, help="Limit amount in USD")
    set_parser.add_argument(
        "--period",
        choices=["daily", "weekly", "monthly"],
        default="monthly",
        help="Limit period (default: monthly)",
    )
    set_parser.add_argument(
        "--action",
        choices=["warn", "block", "require_approval"],
        default="warn",
        help="Action when limit reached (default: warn)",
    )

    # disable
    disable_parser = limit_subparsers.add_parser("disable", help="Disable cost limit")
    disable_parser.add_argument("--provider", "-p", default="replicate")
