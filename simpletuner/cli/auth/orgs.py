"""
Organization and team management commands.

Handles organization CRUD, team CRUD, memberships, and quotas.
"""

import json

from ..cloud.api import cloud_api_request, format_bool


def cmd_orgs(args) -> int:
    """Manage organizations and teams."""
    orgs_action = getattr(args, "orgs_action", None)

    if orgs_action == "list":
        return _orgs_list(args)
    elif orgs_action == "get":
        return _orgs_get(args)
    elif orgs_action == "create":
        return _orgs_create(args)
    elif orgs_action == "update":
        return _orgs_update(args)
    elif orgs_action == "delete":
        return _orgs_delete(args)
    elif orgs_action == "members":
        return _orgs_members(args)
    elif orgs_action == "add-member":
        return _orgs_add_member(args)
    elif orgs_action == "remove-member":
        return _orgs_remove_member(args)
    elif orgs_action == "quotas":
        return _orgs_quotas(args)
    elif orgs_action == "set-quota":
        return _orgs_set_quota(args)
    elif orgs_action == "teams":
        return _orgs_teams(args)
    elif orgs_action == "team-create":
        return _orgs_team_create(args)
    elif orgs_action == "team-get":
        return _orgs_team_get(args)
    elif orgs_action == "team-update":
        return _orgs_team_update(args)
    elif orgs_action == "team-delete":
        return _orgs_team_delete(args)
    elif orgs_action == "team-members":
        return _orgs_team_members(args)
    elif orgs_action == "team-add-member":
        return _orgs_team_add_member(args)
    elif orgs_action == "team-remove-member":
        return _orgs_team_remove_member(args)
    elif orgs_action == "team-quotas":
        return _orgs_team_quotas(args)
    elif orgs_action == "team-set-quota":
        return _orgs_team_set_quota(args)
    else:
        print("Error: Unknown orgs action. Use 'simpletuner auth orgs --help'.")
        return 1


# --- Organization CRUD ---


def _orgs_list(args) -> int:
    """List all organizations."""
    include_inactive = getattr(args, "include_inactive", False)
    output_format = getattr(args, "format", "table")

    params = []
    if include_inactive:
        params.append("include_inactive=true")

    query = "&".join(params) if params else ""
    endpoint = "/api/orgs"
    if query:
        endpoint += f"?{query}"

    result = cloud_api_request("GET", endpoint)

    if isinstance(result, list):
        orgs = result
    else:
        orgs = result.get("organizations", result)

    if not orgs:
        print("No organizations found.")
        return 0

    if output_format == "json":
        print(json.dumps(orgs, indent=2))
        return 0

    print(f"{'ID':<6} {'Name':<25} {'Slug':<20} {'Active':<8}")
    print("-" * 65)

    for org in orgs:
        org_id = org.get("id", "-")
        name = org.get("name", "unknown")[:24]
        slug = org.get("slug", "")[:19]
        is_active = format_bool(org.get("is_active", True))

        print(f"{org_id:<6} {name:<25} {slug:<20} {is_active:<8}")

    return 0


def _orgs_get(args) -> int:
    """Get an organization by ID."""
    org_id = getattr(args, "org_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}")

    org = result.get("organization", result)

    if output_format == "json":
        print(json.dumps(org, indent=2))
        return 0

    print(f"Organization {org.get('id', org_id)}")
    print("=" * 50)
    print(f"Name:        {org.get('name', 'unknown')}")
    print(f"Slug:        {org.get('slug', '-')}")
    print(f"Description: {org.get('description') or '-'}")
    print(f"Active:      {format_bool(org.get('is_active', True))}")
    print(f"Created:     {org.get('created_at', '-')}")

    settings = org.get("settings", {})
    if settings:
        print(f"Settings:    {json.dumps(settings)}")

    return 0


def _orgs_create(args) -> int:
    """Create a new organization."""
    name = getattr(args, "name", None)
    slug = getattr(args, "slug", None)
    description = getattr(args, "description", "")
    settings_str = getattr(args, "settings", None)

    if not name or not slug:
        print("Error: --name and --slug are required.")
        return 1

    data = {
        "name": name,
        "slug": slug,
        "description": description,
    }
    if settings_str:
        data["settings"] = json.loads(settings_str)

    result = cloud_api_request("POST", "/api/orgs", data=data)

    org = result.get("organization", result)
    print(f"Organization created:")
    print(f"  ID:   {org.get('id', '-')}")
    print(f"  Name: {org.get('name', name)}")
    print(f"  Slug: {org.get('slug', slug)}")
    return 0


def _orgs_update(args) -> int:
    """Update an organization."""
    org_id = getattr(args, "org_id", None)

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    data = {}
    if getattr(args, "name", None):
        data["name"] = args.name
    if getattr(args, "description", None):
        data["description"] = args.description
    if getattr(args, "active", None) is not None:
        data["is_active"] = args.active
    settings_str = getattr(args, "settings", None)
    if settings_str:
        data["settings"] = json.loads(settings_str)

    if not data:
        print("Error: No updates specified.")
        return 1

    cloud_api_request("PATCH", f"/api/orgs/{org_id}", data=data)

    print(f"Organization {org_id} updated.")
    return 0


def _orgs_delete(args) -> int:
    """Delete an organization."""
    org_id = getattr(args, "org_id", None)
    force = getattr(args, "force", False)

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    if not force:
        confirm = input(f"Delete organization {org_id}? This will delete all teams. [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/orgs/{org_id}")

    if result.get("success"):
        print(f"Organization {org_id} deleted.")
    else:
        print(f"Error: {result.get('detail', 'Failed to delete organization')}")
        return 1
    return 0


# --- Organization Members ---


def _orgs_members(args) -> int:
    """List organization members."""
    org_id = getattr(args, "org_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}/members")

    members = result.get("members", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    org = result.get("organization", {})
    print(f"Members of {org.get('name', f'Organization {org_id}')}")
    print("=" * 50)

    if not members:
        print("No members found.")
        return 0

    print(f"{'ID':<6} {'Username':<20} {'Email':<30} {'Admin':<6}")
    print("-" * 65)

    for member in members:
        member_id = member.get("id", "-")
        username = member.get("username", "unknown")[:19]
        email = member.get("email", "")[:29]
        is_admin = format_bool(member.get("is_admin", False))

        print(f"{member_id:<6} {username:<20} {email:<30} {is_admin:<6}")

    return 0


def _orgs_add_member(args) -> int:
    """Add a member to an organization."""
    org_id = getattr(args, "org_id", None)
    user_id = getattr(args, "user_id", None)

    if not org_id or not user_id:
        print("Error: Organization ID and user ID are required.")
        return 1

    result = cloud_api_request("POST", f"/api/orgs/{org_id}/members/{user_id}")

    if result.get("success"):
        print(f"User {user_id} added to organization {org_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to add member')}")
        return 1
    return 0


def _orgs_remove_member(args) -> int:
    """Remove a member from an organization."""
    org_id = getattr(args, "org_id", None)
    user_id = getattr(args, "user_id", None)

    if not org_id or not user_id:
        print("Error: Organization ID and user ID are required.")
        return 1

    result = cloud_api_request("DELETE", f"/api/orgs/{org_id}/members/{user_id}")

    if result.get("success"):
        print(f"User {user_id} removed from organization {org_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to remove member')}")
        return 1
    return 0


# --- Organization Quotas ---


def _orgs_quotas(args) -> int:
    """List organization quotas."""
    org_id = getattr(args, "org_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}/quotas")

    quotas = result.get("quotas", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    org = result.get("organization", {})
    print(f"Quotas for {org.get('name', f'Organization {org_id}')}")
    print("=" * 50)

    if not quotas:
        print("No quotas configured.")
        return 0

    print(f"{'Type':<20} {'Limit':<12} {'Action':<15}")
    print("-" * 50)

    for quota in quotas:
        qtype = quota.get("quota_type", "unknown")
        limit = quota.get("limit_value", "-")
        action = quota.get("action", "block")

        print(f"{qtype:<20} {limit:<12} {action:<15}")

    return 0


def _orgs_set_quota(args) -> int:
    """Set an organization quota."""
    org_id = getattr(args, "org_id", None)
    quota_type = getattr(args, "type", None)
    limit_value = getattr(args, "limit", None)
    action = getattr(args, "action", "block")

    if not org_id or not quota_type or limit_value is None:
        print("Error: --org-id, --type, and --limit are required.")
        return 1

    data = {
        "quota_type": quota_type,
        "limit_value": limit_value,
        "action": action,
    }

    result = cloud_api_request("POST", f"/api/orgs/{org_id}/quotas", data=data)

    if result.get("success"):
        print(f"Quota {quota_type} = {limit_value} ({action}) set for organization {org_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to set quota')}")
        return 1
    return 0


# --- Team CRUD ---


def _orgs_teams(args) -> int:
    """List teams in an organization."""
    org_id = getattr(args, "org_id", None)
    include_inactive = getattr(args, "include_inactive", False)
    output_format = getattr(args, "format", "table")

    if not org_id:
        print("Error: Organization ID is required.")
        return 1

    params = []
    if include_inactive:
        params.append("include_inactive=true")

    query = "&".join(params) if params else ""
    endpoint = f"/api/orgs/{org_id}/teams"
    if query:
        endpoint += f"?{query}"

    result = cloud_api_request("GET", endpoint)

    teams = result.get("teams", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    org = result.get("organization", {})
    print(f"Teams in {org.get('name', f'Organization {org_id}')}")
    print("=" * 50)

    if not teams:
        print("No teams found.")
        return 0

    print(f"{'ID':<6} {'Name':<25} {'Slug':<20} {'Active':<8}")
    print("-" * 65)

    for team in teams:
        team_id = team.get("id", "-")
        name = team.get("name", "unknown")[:24]
        slug = team.get("slug", "")[:19]
        is_active = format_bool(team.get("is_active", True))

        print(f"{team_id:<6} {name:<25} {slug:<20} {is_active:<8}")

    return 0


def _orgs_team_create(args) -> int:
    """Create a team in an organization."""
    org_id = getattr(args, "org_id", None)
    name = getattr(args, "name", None)
    slug = getattr(args, "slug", None)
    description = getattr(args, "description", "")
    settings_str = getattr(args, "settings", None)

    if not org_id or not name or not slug:
        print("Error: --org-id, --name, and --slug are required.")
        return 1

    data = {
        "name": name,
        "slug": slug,
        "description": description,
    }
    if settings_str:
        data["settings"] = json.loads(settings_str)

    result = cloud_api_request("POST", f"/api/orgs/{org_id}/teams", data=data)

    team = result.get("team", result)
    print(f"Team created:")
    print(f"  ID:   {team.get('id', '-')}")
    print(f"  Name: {team.get('name', name)}")
    print(f"  Slug: {team.get('slug', slug)}")
    return 0


def _orgs_team_get(args) -> int:
    """Get a team by ID."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id or not team_id:
        print("Error: Organization ID and team ID are required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}/teams/{team_id}")

    team = result.get("team", result)

    if output_format == "json":
        print(json.dumps(team, indent=2))
        return 0

    print(f"Team {team.get('id', team_id)}")
    print("=" * 50)
    print(f"Name:        {team.get('name', 'unknown')}")
    print(f"Slug:        {team.get('slug', '-')}")
    print(f"Description: {team.get('description') or '-'}")
    print(f"Active:      {format_bool(team.get('is_active', True))}")
    print(f"Created:     {team.get('created_at', '-')}")

    settings = team.get("settings", {})
    if settings:
        print(f"Settings:    {json.dumps(settings)}")

    return 0


def _orgs_team_update(args) -> int:
    """Update a team."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)

    if not org_id or not team_id:
        print("Error: Organization ID and team ID are required.")
        return 1

    data = {}
    if getattr(args, "name", None):
        data["name"] = args.name
    if getattr(args, "description", None):
        data["description"] = args.description
    if getattr(args, "active", None) is not None:
        data["is_active"] = args.active
    settings_str = getattr(args, "settings", None)
    if settings_str:
        data["settings"] = json.loads(settings_str)

    if not data:
        print("Error: No updates specified.")
        return 1

    cloud_api_request("PATCH", f"/api/orgs/{org_id}/teams/{team_id}", data=data)

    print(f"Team {team_id} updated.")
    return 0


def _orgs_team_delete(args) -> int:
    """Delete a team."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    force = getattr(args, "force", False)

    if not org_id or not team_id:
        print("Error: Organization ID and team ID are required.")
        return 1

    if not force:
        confirm = input(f"Delete team {team_id}? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = cloud_api_request("DELETE", f"/api/orgs/{org_id}/teams/{team_id}")

    if result.get("success"):
        print(f"Team {team_id} deleted.")
    else:
        print(f"Error: {result.get('detail', 'Failed to delete team')}")
        return 1
    return 0


# --- Team Members ---


def _orgs_team_members(args) -> int:
    """List team members."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id or not team_id:
        print("Error: Organization ID and team ID are required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}/teams/{team_id}/members")

    members = result.get("members", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    team = result.get("team", {})
    print(f"Members of {team.get('name', f'Team {team_id}')}")
    print("=" * 50)

    if not members:
        print("No members found.")
        return 0

    print(f"{'User ID':<10} {'Username':<20} {'Role':<12}")
    print("-" * 45)

    for member in members:
        user_id = member.get("user_id", "-")
        username = member.get("username", "unknown")[:19]
        role = member.get("role", "member")

        print(f"{user_id:<10} {username:<20} {role:<12}")

    return 0


def _orgs_team_add_member(args) -> int:
    """Add a member to a team."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    user_id = getattr(args, "user_id", None)
    role = getattr(args, "role", "member")

    if not org_id or not team_id or not user_id:
        print("Error: --org-id, --team-id, and --user-id are required.")
        return 1

    data = {
        "user_id": user_id,
        "role": role,
    }

    result = cloud_api_request("POST", f"/api/orgs/{org_id}/teams/{team_id}/members", data=data)

    if result.get("success"):
        print(f"User {user_id} added to team {team_id} as {role}.")
    else:
        print(f"Error: {result.get('message', result.get('detail', 'Failed to add member'))}")
        return 1
    return 0


def _orgs_team_remove_member(args) -> int:
    """Remove a member from a team."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    user_id = getattr(args, "user_id", None)

    if not org_id or not team_id or not user_id:
        print("Error: Organization ID, team ID, and user ID are required.")
        return 1

    result = cloud_api_request("DELETE", f"/api/orgs/{org_id}/teams/{team_id}/members/{user_id}")

    if result.get("success"):
        print(f"User {user_id} removed from team {team_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to remove member')}")
        return 1
    return 0


# --- Team Quotas ---


def _orgs_team_quotas(args) -> int:
    """List team quotas."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    output_format = getattr(args, "format", "table")

    if not org_id or not team_id:
        print("Error: Organization ID and team ID are required.")
        return 1

    result = cloud_api_request("GET", f"/api/orgs/{org_id}/teams/{team_id}/quotas")

    quotas = result.get("quotas", [])

    if output_format == "json":
        print(json.dumps(result, indent=2))
        return 0

    team = result.get("team", {})
    print(f"Quotas for {team.get('name', f'Team {team_id}')}")
    print("=" * 50)

    if not quotas:
        print("No quotas configured.")
        return 0

    print(f"{'Type':<20} {'Limit':<12} {'Action':<15}")
    print("-" * 50)

    for quota in quotas:
        qtype = quota.get("quota_type", "unknown")
        limit = quota.get("limit_value", "-")
        action = quota.get("action", "block")

        print(f"{qtype:<20} {limit:<12} {action:<15}")

    return 0


def _orgs_team_set_quota(args) -> int:
    """Set a team quota."""
    org_id = getattr(args, "org_id", None)
    team_id = getattr(args, "team_id", None)
    quota_type = getattr(args, "type", None)
    limit_value = getattr(args, "limit", None)
    action = getattr(args, "action", "block")

    if not org_id or not team_id or not quota_type or limit_value is None:
        print("Error: --org-id, --team-id, --type, and --limit are required.")
        return 1

    data = {
        "quota_type": quota_type,
        "limit_value": limit_value,
        "action": action,
    }

    result = cloud_api_request("POST", f"/api/orgs/{org_id}/teams/{team_id}/quotas", data=data)

    if result.get("success"):
        print(f"Quota {quota_type} = {limit_value} ({action}) set for team {team_id}.")
    else:
        print(f"Error: {result.get('detail', 'Failed to set quota')}")
        return 1
    return 0
