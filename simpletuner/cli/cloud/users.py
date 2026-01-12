"""
Cloud user management commands.

Handles user CRUD, levels, permissions, credentials, and resource rules.
"""

import json

from .api import cloud_api_request, format_bool


def cmd_cloud_users(args) -> int:
    """Manage users and authentication."""
    users_action = getattr(args, "users_action", None)

    if users_action == "list":
        return _users_list(args)
    elif users_action == "me":
        return _users_me(args)
    elif users_action == "get":
        return _users_get(args)
    elif users_action == "create":
        return _users_create(args)
    elif users_action == "update":
        return _users_update(args)
    elif users_action == "delete":
        return _users_delete(args)
    elif users_action == "deactivate":
        return _users_deactivate(args)
    elif users_action == "activate":
        return _users_activate(args)
    elif users_action == "levels":
        return _users_levels(args)
    elif users_action == "assign-level":
        return _users_assign_level(args)
    elif users_action == "remove-level":
        return _users_remove_level(args)
    elif users_action == "permissions":
        return _users_permissions(args)
    elif users_action == "set-permission":
        return _users_set_permission(args)
    elif users_action == "remove-permission":
        return _users_remove_permission(args)
    elif users_action == "credentials":
        return _users_credentials(args)
    elif users_action == "set-credential":
        return _users_set_credential(args)
    elif users_action == "delete-credential":
        return _users_delete_credential(args)
    elif users_action == "rotate-credentials":
        return _users_rotate_credentials(args)
    elif users_action == "stale-credentials":
        return _users_stale_credentials(args)
    elif users_action == "check-stale-credentials":
        return _users_check_stale_credentials(args)
    elif users_action == "rules":
        return _users_rules(args)
    elif users_action == "auth-status":
        return _users_auth_status(args)
    else:
        print("Error: Unknown users action. Use 'simpletuner cloud users --help'.")
        return 1


# --- User CRUD ---


def _users_list(args) -> int:
    """List all users."""
    include_inactive = getattr(args, "include_inactive", False)
    limit = getattr(args, "limit", 100)
    offset = getattr(args, "offset", 0)
    output_format = getattr(args, "format", "table")

    params = [f"limit={limit}", f"offset={offset}"]
    if include_inactive:
        params.append("include_inactive=true")

    query = "&".join(params)
    result = cloud_api_request("GET", f"/api/users?{query}")

    if isinstance(result, list):
        users = result
    else:
        users = result.get("users", result)

    if not users:
        print("No users found.")
        return 0

    if output_format == "json":
        print(json.dumps(users, indent=2))
        return 0

    print(f"{'ID':<6} {'Username':<20} {'Email':<30} {'Admin':<6} {'Active':<6}")
    print("-" * 75)

    for user in users:
        user_id = user.get("id", "-")
        username = user.get("username", "unknown")[:19]
        email = user.get("email", "")[:29]
        is_admin = format_bool(user.get("is_admin", False))
        is_active = format_bool(user.get("is_active", True))

        print(f"{user_id:<6} {username:<20} {email:<30} {is_admin:<6} {is_active:<6}")

    return 0


def _users_me(args) -> int:
    """Get current user info."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/users/me")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Current User")
    print("=" * 50)
    print(f"ID:           {result.get('id', '-')}")
    print(f"Username:     {result.get('username', 'unknown')}")
    print(f"Email:        {result.get('email', '-')}")
    print(f"Display Name: {result.get('display_name') or '-'}")
    print(f"Admin:        {format_bool(result.get('is_admin', False))}")
    print(f"Active:       {format_bool(result.get('is_active', True))}")
    print(f"Verified:     {format_bool(result.get('email_verified', False))}")
    print(f"Auth Provider:{result.get('auth_provider', 'local')}")
    print(f"Created:      {result.get('created_at', '-')}")
    print(f"Last Login:   {result.get('last_login_at') or '-'}")

    levels = result.get("levels", [])
    if levels:
        level_names = ", ".join(lvl.get("name", "?") for lvl in levels)
        print(f"Levels:       {level_names}")

    permissions = result.get("permissions", [])
    if permissions:
        print(f"Permissions:  {len(permissions)} total")

    return 0


def _users_get(args) -> int:
    """Get a specific user by ID."""
    user_id = getattr(args, "user_id", None)
    output_format = getattr(args, "format", "table")

    if not user_id:
        print("Error: User ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/users/{user_id}")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print(f"User {result.get('id', user_id)}")
    print("=" * 50)
    print(f"Username:     {result.get('username', 'unknown')}")
    print(f"Email:        {result.get('email', '-')}")
    print(f"Display Name: {result.get('display_name') or '-'}")
    print(f"Admin:        {format_bool(result.get('is_admin', False))}")
    print(f"Active:       {format_bool(result.get('is_active', True))}")
    print(f"Verified:     {format_bool(result.get('email_verified', False))}")
    print(f"Auth Provider:{result.get('auth_provider', 'local')}")
    print(f"Created:      {result.get('created_at', '-')}")
    print(f"Last Login:   {result.get('last_login_at') or '-'}")

    levels = result.get("levels", [])
    if levels:
        print("\nLevels:")
        for level in levels:
            print(f"  - {level.get('name', '?')} (priority: {level.get('priority', 0)})")

    permissions = result.get("permissions", [])
    if permissions:
        print(f"\nPermissions ({len(permissions)}):")
        for perm in sorted(permissions)[:10]:
            print(f"  - {perm}")
        if len(permissions) > 10:
            print(f"  ... and {len(permissions) - 10} more")

    return 0


def _users_create(args) -> int:
    """Create a new user."""
    email = getattr(args, "email", None)
    username = getattr(args, "username", None)
    password = getattr(args, "password", None)
    display_name = getattr(args, "display_name", None)
    is_admin = getattr(args, "admin", False)
    levels = getattr(args, "levels", None)

    if not email or not username:
        print("Error: --email and --username are required.")
        return 1

    data = {
        "email": email,
        "username": username,
        "is_admin": is_admin,
    }
    if password:
        data["password"] = password
    if display_name:
        data["display_name"] = display_name
    if levels:
        data["level_names"] = levels.split(",")

    result = cloud_api_request("POST", "/api/users", data=data)

    print(f"User created successfully:")
    print(f"  ID:       {result.get('id', '-')}")
    print(f"  Username: {result.get('username', username)}")
    print(f"  Email:    {result.get('email', email)}")
    return 0


def _users_update(args) -> int:
    """Update a user."""
    user_id = getattr(args, "user_id", None)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    data = {}
    if getattr(args, "email", None):
        data["email"] = args.email
    if getattr(args, "username", None):
        data["username"] = args.username
    if getattr(args, "display_name", None):
        data["display_name"] = args.display_name
    if getattr(args, "password", None):
        data["password"] = args.password
    if getattr(args, "active", None) is not None:
        data["is_active"] = args.active
    if getattr(args, "admin", None) is not None:
        data["is_admin"] = args.admin

    if not data:
        print("Error: No updates specified.")
        return 1

    result = cloud_api_request("PATCH", f"/api/users/{user_id}", data=data)

    print(f"User {user_id} updated successfully.")
    return 0


def _users_delete(args) -> int:
    """Delete a user."""
    user_id = getattr(args, "user_id", None)
    force = getattr(args, "force", False)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    if not force:
        confirm = input(f"Delete user {user_id}? This cannot be undone. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/users/{user_id}")

    if result.get("success"):
        print(f"User {user_id} deleted.")
    else:
        print(f"Error: {result.get('detail', 'Failed to delete user')}")
        return 1
    return 0


def _users_deactivate(args) -> int:
    """Deactivate a user."""
    user_id = getattr(args, "user_id", None)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    result = cloud_api_request("POST", f"/api/users/{user_id}/deactivate")

    if result.get("success"):
        print(f"User {user_id} deactivated.")
    else:
        print(f"Error: {result.get('detail', 'Failed to deactivate user')}")
        return 1
    return 0


def _users_activate(args) -> int:
    """Activate a user."""
    user_id = getattr(args, "user_id", None)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    result = cloud_api_request("POST", f"/api/users/{user_id}/activate")

    if result.get("success"):
        print(f"User {user_id} activated.")
    else:
        print(f"Error: {result.get('detail', 'Failed to activate user')}")
        return 1
    return 0


# --- Levels ---


def _users_levels(args) -> int:
    """Manage levels/roles."""
    levels_action = getattr(args, "levels_action", "list")
    output_format = getattr(args, "format", "table")

    if levels_action == "list":
        result = cloud_api_request("GET", "/api/users/meta/levels")

        if isinstance(result, list):
            levels = result
        else:
            levels = result.get("levels", result)

        if not levels:
            print("No levels found.")
            return 0

        if output_format == "json":
            print(json.dumps(levels, indent=2))
            return 0

        print(f"{'ID':<6} {'Name':<20} {'Priority':<10} {'System':<8} {'Permissions':<10}")
        print("-" * 60)

        for level in levels:
            level_id = level.get("id", "-")
            name = level.get("name", "unknown")[:19]
            priority = level.get("priority", 0)
            is_system = format_bool(level.get("is_system", False))
            perm_count = len(level.get("permissions", []))

            print(f"{level_id:<6} {name:<20} {priority:<10} {is_system:<8} {perm_count:<10}")

        return 0

    elif levels_action == "create":
        name = getattr(args, "name", None)
        description = getattr(args, "description", "")
        priority = getattr(args, "priority", 0)
        permissions = getattr(args, "permissions", None)

        if not name:
            print("Error: --name is required.")
            return 1

        data = {
            "name": name,
            "description": description,
            "priority": priority,
        }
        if permissions:
            data["permission_names"] = permissions.split(",")

        result = cloud_api_request("POST", "/api/users/levels", data=data)

        print(f"Level created: {result.get('name', name)}")
        return 0

    elif levels_action == "update":
        level_id = getattr(args, "level_id", None)

        if not level_id:
            print("Error: Level ID is required.")
            return 1

        data = {}
        if getattr(args, "name", None):
            data["name"] = args.name
        if getattr(args, "description", None):
            data["description"] = args.description
        if getattr(args, "priority", None) is not None:
            data["priority"] = args.priority
        if getattr(args, "permissions", None):
            data["permission_names"] = args.permissions.split(",")

        if not data:
            print("Error: No updates specified.")
            return 1

        result = cloud_api_request("PUT", f"/api/users/levels/{level_id}", data=data)
        print(f"Level {level_id} updated.")
        return 0

    elif levels_action == "delete":
        level_id = getattr(args, "level_id", None)

        if not level_id:
            print("Error: Level ID is required.")
            return 1

        result = cloud_api_request("DELETE", f"/api/users/levels/{level_id}")
        print(f"Level {level_id} deleted.")
        return 0

    else:
        print("Error: Unknown levels action.")
        return 1


def _users_assign_level(args) -> int:
    """Assign a level to a user."""
    user_id = getattr(args, "user_id", None)
    level_name = getattr(args, "level_name", None)

    if not user_id or not level_name:
        print("Error: User ID and level name are required.")
        return 1

    result = cloud_api_request(
        "POST",
        f"/api/users/{user_id}/levels",
        data={"level_name": level_name},
    )

    if result.get("success"):
        print(f"Level '{level_name}' assigned to user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to assign level')}")
        return 1
    return 0


def _users_remove_level(args) -> int:
    """Remove a level from a user."""
    user_id = getattr(args, "user_id", None)
    level_name = getattr(args, "level_name", None)

    if not user_id or not level_name:
        print("Error: User ID and level name are required.")
        return 1

    result = cloud_api_request("DELETE", f"/api/users/{user_id}/levels/{level_name}")

    if result.get("success"):
        print(f"Level '{level_name}' removed from user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to remove level')}")
        return 1
    return 0


# --- Permissions ---


def _users_permissions(args) -> int:
    """List available permissions."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/users/meta/permissions")

    if isinstance(result, list):
        permissions = result
    else:
        permissions = result.get("permissions", result)

    if not permissions:
        print("No permissions found.")
        return 0

    if output_format == "json":
        print(json.dumps(permissions, indent=2))
        return 0

    print(f"{'ID':<6} {'Name':<30} {'Category':<15} {'Description':<30}")
    print("-" * 85)

    for perm in permissions:
        perm_id = perm.get("id", "-")
        name = perm.get("name", "unknown")[:29]
        category = perm.get("category", "")[:14]
        description = (perm.get("description") or "")[:29]

        print(f"{perm_id:<6} {name:<30} {category:<15} {description:<30}")

    return 0


def _users_set_permission(args) -> int:
    """Set a permission override for a user."""
    user_id = getattr(args, "user_id", None)
    permission_name = getattr(args, "permission_name", None)
    granted = getattr(args, "granted", None)
    denied = getattr(args, "denied", None)

    if not user_id or not permission_name:
        print("Error: User ID and permission name are required.")
        return 1

    if granted is None and denied is None:
        print("Error: Must specify --granted or --denied.")
        return 1

    is_granted = granted if granted is not None else not denied

    result = cloud_api_request(
        "POST",
        f"/api/users/{user_id}/permissions",
        data={"permission_name": permission_name, "granted": is_granted},
    )

    if result.get("success"):
        action = "granted" if is_granted else "denied"
        print(f"Permission '{permission_name}' {action} for user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to set permission')}")
        return 1
    return 0


def _users_remove_permission(args) -> int:
    """Remove a permission override from a user."""
    user_id = getattr(args, "user_id", None)
    permission_name = getattr(args, "permission_name", None)

    if not user_id or not permission_name:
        print("Error: User ID and permission name are required.")
        return 1

    result = cloud_api_request("DELETE", f"/api/users/{user_id}/permissions/{permission_name}")

    if result.get("success"):
        print(f"Permission override '{permission_name}' removed from user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to remove permission')}")
        return 1
    return 0


# --- Credentials ---


def _users_credentials(args) -> int:
    """List user credentials."""
    user_id = getattr(args, "user_id", None)
    provider = getattr(args, "provider", None)
    output_format = getattr(args, "format", "table")

    if not user_id:
        print("Error: User ID is required.")
        return 1

    params = []
    if provider:
        params.append(f"provider={provider}")

    query = "&".join(params) if params else ""
    endpoint = f"/api/users/{user_id}/credentials"
    if query:
        endpoint += f"?{query}"

    result = cloud_api_request("GET", endpoint)

    if isinstance(result, list):
        credentials = result
    else:
        credentials = result.get("credentials", result)

    if not credentials:
        print("No credentials found.")
        return 0

    if output_format == "json":
        print(json.dumps(credentials, indent=2))
        return 0

    print(f"{'ID':<6} {'Provider':<15} {'Name':<20} {'Active':<8} {'Updated':<20}")
    print("-" * 75)

    for cred in credentials:
        cred_id = cred.get("id", "-")
        provider_name = cred.get("provider", "unknown")[:14]
        cred_name = cred.get("credential_name", "")[:19]
        is_active = format_bool(cred.get("is_active", True))
        updated_at = (cred.get("updated_at") or "-")[:19]

        print(f"{cred_id:<6} {provider_name:<15} {cred_name:<20} {is_active:<8} {updated_at:<20}")

    return 0


def _users_set_credential(args) -> int:
    """Set a credential for a user."""
    user_id = getattr(args, "user_id", None)
    provider = getattr(args, "provider", None)
    name = getattr(args, "name", None)
    value = getattr(args, "value", None)
    description = getattr(args, "description", None)

    if not user_id or not provider or not name:
        print("Error: --user-id, --provider, and --name are required.")
        return 1

    if not value:
        import getpass

        value = getpass.getpass(f"Enter credential value: ")

    data = {
        "provider": provider,
        "credential_name": name,
        "value": value,
    }
    if description:
        data["description"] = description

    result = cloud_api_request("POST", f"/api/users/{user_id}/credentials", data=data)

    if result.get("success"):
        print(f"Credential {provider}/{name} set for user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to set credential')}")
        return 1
    return 0


def _users_delete_credential(args) -> int:
    """Delete a credential."""
    user_id = getattr(args, "user_id", None)
    provider = getattr(args, "provider", None)
    credential_name = getattr(args, "credential_name", None)

    if not user_id or not provider or not credential_name:
        print("Error: User ID, provider, and credential name are required.")
        return 1

    result = cloud_api_request("DELETE", f"/api/users/{user_id}/credentials/{provider}/{credential_name}")

    if result.get("success"):
        print(f"Credential {provider}/{credential_name} deleted for user {user_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to delete credential')}")
        return 1
    return 0


def _users_rotate_credentials(args) -> int:
    """Rotate credentials for a user."""
    user_id = getattr(args, "user_id", None)
    provider = getattr(args, "provider", None)
    reason = getattr(args, "reason", None)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    data = {}
    if provider:
        data["provider"] = provider
    if reason:
        data["reason"] = reason

    result = cloud_api_request(
        "POST",
        f"/api/users/{user_id}/credentials/rotate",
        data=data if data else None,
    )

    rotated_count = result.get("rotated_count", 0)
    print(f"Rotated {rotated_count} credential(s) for user {user_id}.")
    print(result.get("message", ""))
    return 0


def _users_stale_credentials(args) -> int:
    """List stale credentials for a user."""
    user_id = getattr(args, "user_id", None)
    days = getattr(args, "days", 90)

    if not user_id:
        print("Error: User ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/users/{user_id}/credentials/stale?days={days}")

    stale_count = result.get("stale_count", 0)
    print(f"Found {stale_count} stale credential(s) (older than {days} days)")

    credentials = result.get("credentials", [])
    if credentials:
        for cred in credentials:
            print(f"  - {cred.get('provider')}/{cred.get('credential_name')} ({cred.get('days_old')} days old)")

    return 0


def _users_check_stale_credentials(args) -> int:
    """Check all users for stale credentials."""
    days = getattr(args, "days", 90)

    result = cloud_api_request("POST", f"/api/users/credentials/check-stale?days={days}")

    total = result.get("total_stale", 0)
    print(f"Total stale credentials: {total} (older than {days} days)")

    users = result.get("users_with_stale", [])
    if users:
        print("\nUsers with stale credentials:")
        for u in users:
            print(f"  - {u.get('username')} (user {u.get('user_id')}): {u.get('stale_count')} stale")

    return 0


# --- Resource Rules ---


def _users_rules(args) -> int:
    """Manage resource rules."""
    rules_action = getattr(args, "rules_action", "list")
    output_format = getattr(args, "format", "table")

    if rules_action == "list":
        resource_type = getattr(args, "type", None)

        params = []
        if resource_type:
            params.append(f"resource_type={resource_type}")

        query = "&".join(params) if params else ""
        endpoint = "/api/users/resource-rules"
        if query:
            endpoint += f"?{query}"

        result = cloud_api_request("GET", endpoint)

        if isinstance(result, list):
            rules = result
        else:
            rules = result.get("rules", result)

        if not rules:
            print("No resource rules found.")
            return 0

        if output_format == "json":
            print(json.dumps(rules, indent=2))
            return 0

        print(f"{'ID':<6} {'Name':<20} {'Type':<12} {'Pattern':<20} {'Action':<8}")
        print("-" * 70)

        for rule in rules:
            rule_id = rule.get("id", "-")
            name = rule.get("name", "unknown")[:19]
            rtype = rule.get("resource_type", "")[:11]
            pattern = rule.get("pattern", "")[:19]
            action = rule.get("action", "allow")

            print(f"{rule_id:<6} {name:<20} {rtype:<12} {pattern:<20} {action:<8}")

        return 0

    elif rules_action == "create":
        name = getattr(args, "name", None)
        resource_type = getattr(args, "type", None)
        pattern = getattr(args, "pattern", None)
        action = getattr(args, "action", "allow")
        priority = getattr(args, "priority", 0)
        description = getattr(args, "description", "")

        if not name or not resource_type or not pattern:
            print("Error: --name, --type, and --pattern are required.")
            return 1

        data = {
            "name": name,
            "resource_type": resource_type,
            "pattern": pattern,
            "action": action,
            "priority": priority,
            "description": description,
        }

        result = cloud_api_request("POST", "/api/users/resource-rules", data=data)

        print(f"Rule created: {result.get('name', name)} (ID: {result.get('id', '-')})")
        return 0

    elif rules_action == "update":
        rule_id = getattr(args, "rule_id", None)

        if not rule_id:
            print("Error: Rule ID is required.")
            return 1

        data = {}
        if getattr(args, "name", None):
            data["name"] = args.name
        if getattr(args, "pattern", None):
            data["pattern"] = args.pattern
        if getattr(args, "action", None):
            data["action"] = args.action
        if getattr(args, "priority", None) is not None:
            data["priority"] = args.priority

        if not data:
            print("Error: No updates specified.")
            return 1

        result = cloud_api_request("PUT", f"/api/users/resource-rules/{rule_id}", data=data)
        print(f"Rule {rule_id} updated.")
        return 0

    elif rules_action == "delete":
        rule_id = getattr(args, "rule_id", None)

        if not rule_id:
            print("Error: Rule ID is required.")
            return 1

        result = cloud_api_request("DELETE", f"/api/users/resource-rules/{rule_id}")
        print(f"Rule {rule_id} deleted.")
        return 0

    elif rules_action == "assign":
        level_id = getattr(args, "level_id", None)
        rule_id = getattr(args, "rule_id", None)

        if not level_id or not rule_id:
            print("Error: Level ID and rule ID are required.")
            return 1

        result = cloud_api_request("POST", f"/api/users/levels/{level_id}/rules/{rule_id}")
        print(f"Rule {rule_id} assigned to level {level_id}.")
        return 0

    elif rules_action == "remove":
        level_id = getattr(args, "level_id", None)
        rule_id = getattr(args, "rule_id", None)

        if not level_id or not rule_id:
            print("Error: Level ID and rule ID are required.")
            return 1

        result = cloud_api_request("DELETE", f"/api/users/levels/{level_id}/rules/{rule_id}")
        print(f"Rule {rule_id} removed from level {level_id}.")
        return 0

    else:
        print("Error: Unknown rules action.")
        return 1


# --- Auth Status ---


def _users_auth_status(args) -> int:
    """Check if authentication is in use."""
    output_format = getattr(args, "format", "table")

    result = cloud_api_request("GET", "/api/users/meta/auth-status")

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    print("Authentication Status")
    print("=" * 40)
    print(f"In Use:             {format_bool(result.get('in_use', False))}")
    print(f"User Count:         {result.get('user_count', 0)}")
    print(f"External Providers: {format_bool(result.get('has_external_providers', False))}")

    return 0
