"""
Authentication and user management CLI commands.

Provides user management, organization/team management, and audit logging.
"""

import argparse

from . import audit, orgs, users


def add_auth_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the auth command parser and all subcommands."""
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage authentication, users, orgs, and audit",
        description="Authentication, user management, organizations, and audit logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner auth status
  simpletuner auth users list
  simpletuner auth users me
  simpletuner auth orgs list
  simpletuner auth audit list
""",
    )
    auth_parser.set_defaults(func=cmd_auth)

    auth_subparsers = auth_parser.add_subparsers(
        dest="auth_action",
        title="Auth commands",
        metavar="<command>",
    )

    # --- Status Command ---
    status_parser = auth_subparsers.add_parser(
        "status",
        help="Check authentication configuration",
        description="Check if authentication is configured and show server status.",
    )
    status_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # --- Setup Command ---
    setup_parser = auth_subparsers.add_parser(
        "setup",
        help="Bootstrap first admin user (headless setup)",
        description="Create the first admin user for headless/automated deployments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  simpletuner auth setup --email admin@example.com --username admin
  simpletuner auth setup --email admin@example.com --username admin --password mypass123
  simpletuner auth setup --email admin@example.com --username admin --display-name "Admin User"
""",
    )
    setup_parser.add_argument("--email", "-e", required=True, help="Admin email address")
    setup_parser.add_argument("--username", "-u", required=True, help="Admin username (3-50 chars, alphanumeric)")
    setup_parser.add_argument("--password", "-p", help="Admin password (min 8 chars, prompts if not provided)")
    setup_parser.add_argument("--display-name", "-d", help="Display name (optional)")
    setup_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # --- Users Commands ---
    _add_users_parser(auth_subparsers)

    # --- Orgs Commands ---
    _add_orgs_parser(auth_subparsers)

    # --- Audit Commands ---
    _add_audit_parser(auth_subparsers)

    return auth_parser


def cmd_auth(args) -> int:
    """Main auth command dispatcher."""
    auth_action = getattr(args, "auth_action", None)

    if auth_action is None:
        print("Error: Please specify an auth command.")
        print("Usage: simpletuner auth <command>")
        print("\nRun 'simpletuner auth --help' for available commands.")
        return 1

    if auth_action == "status":
        return _auth_status(args)
    elif auth_action == "setup":
        return _auth_setup(args)
    elif auth_action == "users":
        return users.cmd_users(args)
    elif auth_action == "orgs":
        return orgs.cmd_orgs(args)
    elif auth_action == "audit":
        return audit.cmd_audit(args)
    else:
        print(f"Error: Unknown auth command '{auth_action}'")
        return 1


def _auth_status(args) -> int:
    """Check authentication configuration and server status."""
    import json as json_mod

    from ..cloud.api import cloud_api_request, get_cloud_server_url

    output_format = getattr(args, "format", "table")

    # First check if server is reachable
    server_url = get_cloud_server_url()
    if output_format != "json":
        print(f"Checking server at {server_url}...")
        print()

    try:
        result = cloud_api_request("GET", "/api/users/meta/auth-status")
    except SystemExit:
        # cloud_api_request exits on error, but we want to handle connection errors gracefully
        return 1

    if output_format == "json":
        print(json_mod.dumps(result, indent=2))
        return 0

    in_use = result.get("in_use", False)
    user_count = result.get("user_count", 0)
    has_providers = result.get("has_external_providers", False)

    print("Authentication Status")
    print("=" * 50)
    print()

    if in_use:
        print("[+] Authentication is ENABLED")
        print()
        print(f"    Users configured:     {user_count}")
        print(f"    External providers:   {'Yes' if has_providers else 'No'}")
        print()
        print("To manage users:     simpletuner auth users list")
        print("To view your info:   simpletuner auth users me")
    else:
        print("[-] Authentication is NOT configured")
        print()
        print("    The server is running in open mode.")
        print("    All API endpoints are accessible without authentication.")
        print()
        print("To set up authentication:")
        print("  Option 1: CLI setup (recommended for headless)")
        print("    simpletuner auth setup")
        print()
        print("  Option 2: Create users directly")
        print("    simpletuner auth users create --email admin@example.com --username admin --admin")

    return 0


def _auth_setup(args) -> int:
    """Bootstrap first admin user for headless deployments."""
    import getpass
    import json as json_mod

    from ..cloud.api import cloud_api_request, get_cloud_server_url

    email = getattr(args, "email", None)
    username = getattr(args, "username", None)
    password = getattr(args, "password", None)
    display_name = getattr(args, "display_name", None)
    output_format = getattr(args, "format", "table")

    # Validate required fields
    if not email or not username:
        print("Error: --email and --username are required.")
        return 1

    # Prompt for password if not provided
    if not password:
        password = getpass.getpass("Enter admin password (min 8 chars): ")
        if not password:
            print("Error: Password cannot be empty.")
            return 1
        if len(password) < 8:
            print("Error: Password must be at least 8 characters.")
            return 1
        # Confirm password
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Error: Passwords do not match.")
            return 1

    # Check setup status first
    server_url = get_cloud_server_url()
    if output_format != "json":
        print(f"Checking server at {server_url}...")

    try:
        status_result = cloud_api_request("GET", "/api/cloud/setup/status")
    except SystemExit:
        return 1

    needs_setup = status_result.get("needs_setup", False)
    if not needs_setup:
        user_count = status_result.get("user_count", 0)
        if output_format == "json":
            print(json_mod.dumps({"success": False, "error": "Setup already completed", "user_count": user_count}))
        else:
            print()
            print("Error: Setup already completed.")
            print(f"  {user_count} user(s) already exist.")
            print()
            print("Use 'simpletuner auth users create' to add more users.")
        return 1

    # Create first admin
    if output_format != "json":
        print(f"Creating admin user '{username}'...")

    data = {
        "email": email,
        "username": username,
        "password": password,
    }
    if display_name:
        data["display_name"] = display_name

    try:
        result = cloud_api_request("POST", "/api/cloud/setup/first-admin", data=data)
    except SystemExit:
        return 1

    if output_format == "json":
        print(json_mod.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    if result.get("success"):
        user = result.get("user", {})
        print()
        print("[+] Admin account created successfully!")
        print()
        print(f"  Username:  {user.get('username', username)}")
        print(f"  Email:     {user.get('email', email)}")
        print(f"  User ID:   {user.get('id', '-')}")
        print(f"  Admin:     Yes")
        print()
        print("Authentication is now enabled. You can:")
        print("  - Log in via the web UI")
        print("  - Create more users: simpletuner auth users create --email ... --username ...")
        print("  - Check status: simpletuner auth status")
        return 0
    else:
        print(f"Error: {result.get('message', 'Failed to create admin')}")
        return 1


def _add_users_parser(subparsers):
    """Add users subcommand parser."""
    users_parser = subparsers.add_parser(
        "users",
        help="Manage users and authentication",
        description="User management, levels, permissions, and credentials.",
    )
    users_subparsers = users_parser.add_subparsers(
        dest="users_action",
        title="User commands",
        metavar="<action>",
    )

    # list
    list_parser = users_subparsers.add_parser("list", help="List all users")
    list_parser.add_argument("--include-inactive", action="store_true")
    list_parser.add_argument("--limit", "-l", type=int, default=100)
    list_parser.add_argument("--offset", "-o", type=int, default=0)
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # me
    me_parser = users_subparsers.add_parser("me", help="Get current user info")
    me_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # get
    get_parser = users_subparsers.add_parser("get", help="Get user by ID")
    get_parser.add_argument("user_id", type=int, help="User ID")
    get_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # create
    create_parser = users_subparsers.add_parser("create", help="Create a new user")
    create_parser.add_argument("--email", required=True)
    create_parser.add_argument("--username", required=True)
    create_parser.add_argument("--password")
    create_parser.add_argument("--display-name")
    create_parser.add_argument("--admin", action="store_true")
    create_parser.add_argument("--levels", help="Comma-separated level names")

    # update
    update_parser = users_subparsers.add_parser("update", help="Update a user")
    update_parser.add_argument("user_id", type=int)
    update_parser.add_argument("--email")
    update_parser.add_argument("--username")
    update_parser.add_argument("--display-name")
    update_parser.add_argument("--password")
    update_parser.add_argument("--active", type=lambda x: x.lower() == "true")
    update_parser.add_argument("--admin", type=lambda x: x.lower() == "true")

    # delete
    delete_parser = users_subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("user_id", type=int)
    delete_parser.add_argument("--force", "-f", action="store_true")

    # deactivate / activate
    deact_parser = users_subparsers.add_parser("deactivate", help="Deactivate a user")
    deact_parser.add_argument("user_id", type=int)

    act_parser = users_subparsers.add_parser("activate", help="Activate a user")
    act_parser.add_argument("user_id", type=int)

    # levels
    levels_parser = users_subparsers.add_parser("levels", help="Manage levels/roles")
    levels_subparsers = levels_parser.add_subparsers(
        dest="levels_action",
        title="Levels commands",
        metavar="<action>",
    )

    levels_list = levels_subparsers.add_parser("list", help="List all levels")
    levels_list.add_argument("--format", "-f", choices=["table", "json"], default="table")

    levels_create = levels_subparsers.add_parser("create", help="Create a level")
    levels_create.add_argument("--name", required=True)
    levels_create.add_argument("--description", default="")
    levels_create.add_argument("--priority", type=int, default=0)
    levels_create.add_argument("--permissions", help="Comma-separated permission names")

    levels_update = levels_subparsers.add_parser("update", help="Update a level")
    levels_update.add_argument("level_id", type=int)
    levels_update.add_argument("--name")
    levels_update.add_argument("--description")
    levels_update.add_argument("--priority", type=int)
    levels_update.add_argument("--permissions", help="Comma-separated permission names")

    levels_delete = levels_subparsers.add_parser("delete", help="Delete a level")
    levels_delete.add_argument("level_id", type=int)

    # assign-level / remove-level
    assign_parser = users_subparsers.add_parser("assign-level", help="Assign level to user")
    assign_parser.add_argument("user_id", type=int)
    assign_parser.add_argument("level_name")

    remove_level_parser = users_subparsers.add_parser("remove-level", help="Remove level from user")
    remove_level_parser.add_argument("user_id", type=int)
    remove_level_parser.add_argument("level_name")

    # permissions
    perms_parser = users_subparsers.add_parser("permissions", help="List available permissions")
    perms_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # set-permission / remove-permission
    set_perm = users_subparsers.add_parser("set-permission", help="Set permission override")
    set_perm.add_argument("user_id", type=int)
    set_perm.add_argument("permission_name")
    set_perm.add_argument("--granted", action="store_true")
    set_perm.add_argument("--denied", action="store_true")

    remove_perm = users_subparsers.add_parser("remove-permission", help="Remove permission override")
    remove_perm.add_argument("user_id", type=int)
    remove_perm.add_argument("permission_name")

    # credentials
    creds_parser = users_subparsers.add_parser("credentials", help="List user credentials")
    creds_parser.add_argument("user_id", type=int)
    creds_parser.add_argument("--provider")
    creds_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # set-credential
    set_cred = users_subparsers.add_parser("set-credential", help="Set a credential")
    set_cred.add_argument("user_id", type=int)
    set_cred.add_argument("--provider", required=True)
    set_cred.add_argument("--name", required=True)
    set_cred.add_argument("--value")
    set_cred.add_argument("--description")

    # delete-credential
    del_cred = users_subparsers.add_parser("delete-credential", help="Delete a credential")
    del_cred.add_argument("user_id", type=int)
    del_cred.add_argument("provider")
    del_cred.add_argument("credential_name")

    # rotate-credentials
    rotate_parser = users_subparsers.add_parser("rotate-credentials", help="Rotate credentials")
    rotate_parser.add_argument("user_id", type=int)
    rotate_parser.add_argument("--provider")
    rotate_parser.add_argument("--reason")

    # stale-credentials
    stale_parser = users_subparsers.add_parser("stale-credentials", help="List stale credentials")
    stale_parser.add_argument("user_id", type=int)
    stale_parser.add_argument("--days", type=int, default=90)

    # check-stale-credentials
    check_stale = users_subparsers.add_parser("check-stale-credentials", help="Check all for stale")
    check_stale.add_argument("--days", type=int, default=90)

    # rules
    rules_parser = users_subparsers.add_parser("rules", help="Manage resource rules")
    rules_subparsers = rules_parser.add_subparsers(
        dest="rules_action",
        title="Rules commands",
        metavar="<action>",
    )

    rules_list = rules_subparsers.add_parser("list", help="List resource rules")
    rules_list.add_argument("--type")
    rules_list.add_argument("--format", "-f", choices=["table", "json"], default="table")

    rules_create = rules_subparsers.add_parser("create", help="Create a rule")
    rules_create.add_argument("--name", required=True)
    rules_create.add_argument("--type", required=True, dest="type")
    rules_create.add_argument("--pattern", required=True)
    rules_create.add_argument("--action", default="allow", choices=["allow", "deny"])
    rules_create.add_argument("--priority", type=int, default=0)
    rules_create.add_argument("--description", default="")

    rules_update = rules_subparsers.add_parser("update", help="Update a rule")
    rules_update.add_argument("rule_id", type=int)
    rules_update.add_argument("--name")
    rules_update.add_argument("--pattern")
    rules_update.add_argument("--action", choices=["allow", "deny"])
    rules_update.add_argument("--priority", type=int)

    rules_delete = rules_subparsers.add_parser("delete", help="Delete a rule")
    rules_delete.add_argument("rule_id", type=int)

    rules_assign = rules_subparsers.add_parser("assign", help="Assign rule to level")
    rules_assign.add_argument("level_id", type=int)
    rules_assign.add_argument("rule_id", type=int)

    rules_remove = rules_subparsers.add_parser("remove", help="Remove rule from level")
    rules_remove.add_argument("level_id", type=int)
    rules_remove.add_argument("rule_id", type=int)

    # auth-status
    auth_status = users_subparsers.add_parser("auth-status", help="Check auth status")
    auth_status.add_argument("--format", "-f", choices=["table", "json"], default="table")


def _add_orgs_parser(subparsers):
    """Add orgs subcommand parser."""
    orgs_parser = subparsers.add_parser(
        "orgs",
        help="Manage organizations and teams",
        description="Organization and team management.",
    )
    orgs_subparsers = orgs_parser.add_subparsers(
        dest="orgs_action",
        title="Organization commands",
        metavar="<action>",
    )

    # list
    list_parser = orgs_subparsers.add_parser("list", help="List all organizations")
    list_parser.add_argument("--include-inactive", action="store_true")
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # get
    get_parser = orgs_subparsers.add_parser("get", help="Get organization by ID")
    get_parser.add_argument("org_id", type=int)
    get_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # create
    create_parser = orgs_subparsers.add_parser("create", help="Create an organization")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--slug", required=True)
    create_parser.add_argument("--description", default="")
    create_parser.add_argument("--settings", help="JSON settings")

    # update
    update_parser = orgs_subparsers.add_parser("update", help="Update an organization")
    update_parser.add_argument("org_id", type=int)
    update_parser.add_argument("--name")
    update_parser.add_argument("--description")
    update_parser.add_argument("--active", type=lambda x: x.lower() == "true")
    update_parser.add_argument("--settings", help="JSON settings")

    # delete
    delete_parser = orgs_subparsers.add_parser("delete", help="Delete an organization")
    delete_parser.add_argument("org_id", type=int)
    delete_parser.add_argument("--force", "-f", action="store_true")

    # members
    members_parser = orgs_subparsers.add_parser("members", help="List org members")
    members_parser.add_argument("org_id", type=int)
    members_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # add-member / remove-member
    add_member = orgs_subparsers.add_parser("add-member", help="Add member to org")
    add_member.add_argument("org_id", type=int)
    add_member.add_argument("user_id", type=int)

    remove_member = orgs_subparsers.add_parser("remove-member", help="Remove member from org")
    remove_member.add_argument("org_id", type=int)
    remove_member.add_argument("user_id", type=int)

    # quotas
    quotas_parser = orgs_subparsers.add_parser("quotas", help="List org quotas")
    quotas_parser.add_argument("org_id", type=int)
    quotas_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # set-quota
    set_quota = orgs_subparsers.add_parser("set-quota", help="Set org quota")
    set_quota.add_argument("org_id", type=int)
    set_quota.add_argument("--type", required=True)
    set_quota.add_argument("--limit", type=float, required=True)
    set_quota.add_argument("--action", default="block", choices=["block", "warn", "notify"])

    # teams
    teams_parser = orgs_subparsers.add_parser("teams", help="List teams in org")
    teams_parser.add_argument("org_id", type=int)
    teams_parser.add_argument("--include-inactive", action="store_true")
    teams_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # team-create
    team_create = orgs_subparsers.add_parser("team-create", help="Create a team")
    team_create.add_argument("org_id", type=int)
    team_create.add_argument("--name", required=True)
    team_create.add_argument("--slug", required=True)
    team_create.add_argument("--description", default="")
    team_create.add_argument("--settings", help="JSON settings")

    # team-get
    team_get = orgs_subparsers.add_parser("team-get", help="Get team by ID")
    team_get.add_argument("org_id", type=int)
    team_get.add_argument("team_id", type=int)
    team_get.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # team-update
    team_update = orgs_subparsers.add_parser("team-update", help="Update a team")
    team_update.add_argument("org_id", type=int)
    team_update.add_argument("team_id", type=int)
    team_update.add_argument("--name")
    team_update.add_argument("--description")
    team_update.add_argument("--active", type=lambda x: x.lower() == "true")
    team_update.add_argument("--settings", help="JSON settings")

    # team-delete
    team_delete = orgs_subparsers.add_parser("team-delete", help="Delete a team")
    team_delete.add_argument("org_id", type=int)
    team_delete.add_argument("team_id", type=int)
    team_delete.add_argument("--force", "-f", action="store_true")

    # team-members
    team_members = orgs_subparsers.add_parser("team-members", help="List team members")
    team_members.add_argument("org_id", type=int)
    team_members.add_argument("team_id", type=int)
    team_members.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # team-add-member
    team_add = orgs_subparsers.add_parser("team-add-member", help="Add member to team")
    team_add.add_argument("org_id", type=int)
    team_add.add_argument("team_id", type=int)
    team_add.add_argument("--user-id", type=int, required=True)
    team_add.add_argument("--role", default="member", choices=["member", "lead", "admin"])

    # team-remove-member
    team_remove = orgs_subparsers.add_parser("team-remove-member", help="Remove member from team")
    team_remove.add_argument("org_id", type=int)
    team_remove.add_argument("team_id", type=int)
    team_remove.add_argument("user_id", type=int)

    # team-quotas
    team_quotas = orgs_subparsers.add_parser("team-quotas", help="List team quotas")
    team_quotas.add_argument("org_id", type=int)
    team_quotas.add_argument("team_id", type=int)
    team_quotas.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # team-set-quota
    team_set_quota = orgs_subparsers.add_parser("team-set-quota", help="Set team quota")
    team_set_quota.add_argument("org_id", type=int)
    team_set_quota.add_argument("team_id", type=int)
    team_set_quota.add_argument("--type", required=True)
    team_set_quota.add_argument("--limit", type=float, required=True)
    team_set_quota.add_argument("--action", default="block", choices=["block", "warn", "notify"])


def _add_audit_parser(subparsers):
    """Add audit subcommand parser."""
    audit_parser = subparsers.add_parser(
        "audit",
        help="Access audit logs",
        description="View and analyze audit log entries.",
    )
    audit_subparsers = audit_parser.add_subparsers(
        dest="audit_action",
        title="Audit commands",
        metavar="<action>",
    )

    # list
    list_parser = audit_subparsers.add_parser("list", help="List audit entries")
    list_parser.add_argument("--event-type")
    list_parser.add_argument("--actor-id", type=int)
    list_parser.add_argument("--target-type")
    list_parser.add_argument("--target-id", type=int)
    list_parser.add_argument("--since", help="ISO date")
    list_parser.add_argument("--until", help="ISO date")
    list_parser.add_argument("--limit", "-l", type=int, default=50)
    list_parser.add_argument("--offset", "-o", type=int, default=0)
    list_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # stats
    stats_parser = audit_subparsers.add_parser("stats", help="Audit log statistics")
    stats_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # types
    types_parser = audit_subparsers.add_parser("types", help="List event types")
    types_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # user
    user_parser = audit_subparsers.add_parser("user", help="Entries for a user")
    user_parser.add_argument("user_id", type=int)
    user_parser.add_argument("--limit", "-l", type=int, default=50)
    user_parser.add_argument("--offset", "-o", type=int, default=0)
    user_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # security
    security_parser = audit_subparsers.add_parser("security", help="Security events")
    security_parser.add_argument("--limit", "-l", type=int, default=50)
    security_parser.add_argument("--offset", "-o", type=int, default=0)
    security_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")

    # verify
    verify_parser = audit_subparsers.add_parser("verify", help="Verify chain integrity")
    verify_parser.add_argument("--start-id", type=int)
    verify_parser.add_argument("--end-id", type=int)
    verify_parser.add_argument("--format", "-f", choices=["table", "json"], default="table")
