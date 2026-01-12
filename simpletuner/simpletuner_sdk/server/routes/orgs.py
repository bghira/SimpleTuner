"""Organization and team management API routes.

Provides endpoints for managing organizations, teams, and their quotas.
Organizations and teams use a ceiling model for quota enforcement.

NOTE: This module was moved from routes/cloud/orgs.py to become a top-level
global route, as organizations are a global concept in SimpleTuner.
"""

import logging
import re

# Forward reference for type hints in helper functions
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.cloud.auth import User, get_current_user, require_permission

if TYPE_CHECKING:
    from ..services.cloud.auth.models import User as UserType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orgs", tags=["organizations"])


# ==================== Request/Response Models ====================


class CreateOrgRequest(BaseModel):
    """Request to create an organization."""

    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z0-9-]+$")
    description: str = Field(default="", max_length=500)
    settings: Dict[str, Any] = Field(default_factory=dict)


class UpdateOrgRequest(BaseModel):
    """Request to update an organization."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class CreateTeamRequest(BaseModel):
    """Request to create a team."""

    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z0-9-]+$")
    description: str = Field(default="", max_length=500)
    settings: Dict[str, Any] = Field(default_factory=dict)


class UpdateTeamRequest(BaseModel):
    """Request to update a team."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class TeamMembershipRequest(BaseModel):
    """Request to add/update team membership."""

    user_id: int
    role: str = Field(default="member", pattern=r"^(member|lead|admin)$")


class SetOrgQuotaRequest(BaseModel):
    """Request to set an organization quota ceiling."""

    quota_type: str = Field(
        ..., pattern=r"^(concurrent_jobs|jobs_per_day|jobs_per_hour|cost_daily|cost_monthly|local_gpus)$"
    )
    limit_value: float = Field(..., gt=0)
    action: str = Field(default="block", pattern=r"^(block|warn|require_approval)$")


class SetTeamQuotaRequest(BaseModel):
    """Request to set a team quota ceiling."""

    quota_type: str = Field(
        ..., pattern=r"^(concurrent_jobs|jobs_per_day|jobs_per_hour|cost_daily|cost_monthly|local_gpus)$"
    )
    limit_value: float = Field(..., gt=0)
    action: str = Field(default="block", pattern=r"^(block|warn|require_approval)$")


# ==================== Access Control Helpers ====================


def can_access_org(user: "User", org_id: int) -> bool:
    """Check if user can access the specified organization.

    Access is granted if:
    - User is a system admin (has admin.users permission)
    - User belongs to the organization (user.org_id == org_id)

    Args:
        user: The requesting user
        org_id: The organization ID being accessed

    Returns:
        True if access is allowed, False otherwise
    """
    # System admins can access any org
    if "admin.users" in user.effective_permissions:
        return True

    # Users can access their own org
    return user.org_id == org_id


def require_org_access(user: "User", org_id: int) -> None:
    """Require that user can access the specified organization.

    Raises HTTPException 403 if access is denied.
    """
    if not can_access_org(user, org_id):
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this organization",
        )


def can_manage_role(actor_role: str, target_role: str) -> bool:
    """Check if actor_role can manage target_role based on hierarchy.

    Rules:
    - admin: Can manage all roles (member, lead, admin)
    - lead: Can manage members only
    - member: Cannot manage anyone
    """
    if actor_role == "admin":
        return True
    if actor_role == "lead":
        return target_role == "member"
    return False


def get_user_team_role(members: list, user_id: int) -> Optional[str]:
    """Get a user's role in a team from the members list."""
    for m in members:
        if m.get("id") == user_id:
            return m.get("role")
    return None


# ==================== Organization Routes ====================


@router.get("")
async def list_organizations(
    include_inactive: bool = Query(False),
    user=Depends(require_permission("org.view")),
):
    """List all organizations with enriched data.

    Returns organizations with team_count and member_count included
    to avoid N+1 queries on the frontend.

    Requires org.view permission.
    """
    from ..services.cloud.container import get_user_store

    user_store = get_user_store()
    orgs = await user_store.list_organizations(include_inactive=include_inactive)

    # Enrich with counts to avoid N+1 queries on frontend
    enriched_orgs = []
    for org in orgs:
        org_dict = org.to_dict()
        teams = await user_store.list_teams(org.id, include_inactive=include_inactive)
        users = await user_store.list_users(org_id=org.id)
        org_dict["team_count"] = len(teams)
        org_dict["member_count"] = len(users)
        enriched_orgs.append(org_dict)

    return {"organizations": enriched_orgs}


@router.post("")
async def create_organization(
    request: CreateOrgRequest,
    user=Depends(require_permission("org.create")),
):
    """Create a new organization.

    Requires org.create permission.
    """
    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    # Check if slug already exists
    existing = await user_store.get_organization_by_slug(request.slug)
    if existing:
        raise HTTPException(400, f"Organization with slug '{request.slug}' already exists")

    org = await user_store.create_organization(
        name=request.name,
        slug=request.slug,
        description=request.description,
        settings=request.settings,
    )

    logger.info(f"Organization created: {org.slug} by user {user.id}")
    return {"organization": org.to_dict()}


@router.get("/{org_id}")
async def get_organization(
    org_id: int,
    user=Depends(require_permission("org.view")),
):
    """Get an organization by ID.

    Requires org.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()
    org = await user_store.get_organization(org_id)

    if not org:
        raise HTTPException(404, "Organization not found")

    return {"organization": org.to_dict()}


@router.patch("/{org_id}")
async def update_organization(
    org_id: int,
    request: UpdateOrgRequest,
    user=Depends(require_permission("org.edit")),
):
    """Update an organization.

    Requires org.edit permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    # Build update dict
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.is_active is not None:
        updates["is_active"] = 1 if request.is_active else 0
    if request.settings is not None:
        import json

        updates["settings"] = json.dumps(request.settings)

    if updates:
        updated = await user_store.update_organization(org_id, updates)
        if updated:
            logger.info(f"Organization {org_id} updated by user {user.id}")
            # Refresh the org object
            org = await user_store.get_organization(org_id)

    return {"organization": org.to_dict() if org else {}}


@router.delete("/{org_id}")
async def delete_organization(
    org_id: int,
    user=Depends(require_permission("org.delete")),
):
    """Delete an organization and all its teams.

    Requires org.delete permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    # Delete the organization (this should cascade to teams)
    success = await user_store.delete_organization(org_id)
    if not success:
        raise HTTPException(500, "Failed to delete organization")

    logger.info(f"Organization {org_id} deleted by user {user.id}")
    return {"success": True, "message": "Organization deleted"}


# ==================== Organization Membership Routes ====================


@router.get("/{org_id}/members")
async def list_org_members(
    org_id: int,
    user=Depends(require_permission("org.view")),
):
    """List all members of an organization.

    Requires org.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    # Get users in this org
    users = await user_store.list_users(org_id=org_id)

    return {
        "organization": org.to_dict(),
        "members": [u.to_dict() for u in users],
    }


@router.post("/{org_id}/members/{user_id}")
async def add_user_to_org(
    org_id: int,
    target_user_id: int,
    user=Depends(require_permission("org.manage.members")),
):
    """Add a user to an organization.

    Requires org.manage.members permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    success = await user_store.set_user_organization(target_user_id, org_id)
    if not success:
        raise HTTPException(400, "Failed to add user to organization")

    logger.info(f"User {target_user_id} added to org {org_id} by user {user.id}")
    return {"success": True, "message": "User added to organization"}


@router.delete("/{org_id}/members/{target_user_id}")
async def remove_user_from_org(
    org_id: int,
    target_user_id: int,
    user=Depends(require_permission("org.manage.members")),
):
    """Remove a user from an organization.

    Requires org.manage.members permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    success = await user_store.set_user_organization(target_user_id, None)
    if not success:
        raise HTTPException(400, "Failed to remove user from organization")

    logger.info(f"User {target_user_id} removed from org {org_id} by user {user.id}")
    return {"success": True, "message": "User removed from organization"}


# ==================== Organization Quota Routes ====================


@router.get("/{org_id}/quotas")
async def get_org_quotas(
    org_id: int,
    user=Depends(require_permission("org.view")),
):
    """Get quota ceilings for an organization.

    Requires org.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    quotas = await user_store.get_org_quotas(org_id)

    return {
        "organization": org.to_dict(),
        "quotas": [q.to_dict() for q in quotas],
    }


@router.post("/{org_id}/quotas")
async def set_org_quota(
    org_id: int,
    request: SetOrgQuotaRequest,
    user=Depends(require_permission("org.manage.quotas")),
):
    """Set a quota ceiling for an organization.

    This sets an absolute ceiling that applies to all users and teams
    within the organization, regardless of any higher limits they may have.

    Requires org.manage.quotas permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.auth.quotas import QuotaAction, QuotaType
    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    quota_type = QuotaType(request.quota_type)
    action = QuotaAction(request.action)

    # Set the org quota
    await user_store.set_quota(
        quota_type=quota_type,
        limit_value=request.limit_value,
        action=action,
        org_id=org_id,
    )

    logger.info(f"Org {org_id} quota {quota_type.value} set to {request.limit_value} by user {user.id}")
    return {"success": True, "message": f"Organization quota ceiling set: {quota_type.value} = {request.limit_value}"}


# ==================== Team Routes ====================


@router.get("/{org_id}/teams")
async def list_teams(
    org_id: int,
    include_inactive: bool = Query(False),
    user=Depends(require_permission("team.view")),
):
    """List all teams in an organization.

    Requires team.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    teams = await user_store.list_teams(org_id, include_inactive=include_inactive)

    return {
        "organization": org.to_dict(),
        "teams": [t.to_dict() for t in teams],
    }


@router.post("/{org_id}/teams")
async def create_team(
    org_id: int,
    request: CreateTeamRequest,
    user=Depends(require_permission("team.create")),
):
    """Create a new team in an organization.

    Requires team.create permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    org = await user_store.get_organization(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    team = await user_store.create_team(
        org_id=org_id,
        name=request.name,
        slug=request.slug,
        description=request.description,
        settings=request.settings,
    )

    logger.info(f"Team created: {org.slug}/{team.slug} by user {user.id}")
    return {"team": team.to_dict()}


@router.get("/{org_id}/teams/{team_id}")
async def get_team(
    org_id: int,
    team_id: int,
    user=Depends(require_permission("team.view")),
):
    """Get a team by ID.

    Requires team.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    return {"team": team.to_dict()}


@router.patch("/{org_id}/teams/{team_id}")
async def update_team(
    org_id: int,
    team_id: int,
    request: UpdateTeamRequest,
    user=Depends(require_permission("team.edit")),
):
    """Update a team.

    Requires team.edit permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    # Build update dict
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.is_active is not None:
        updates["is_active"] = 1 if request.is_active else 0
    if request.settings is not None:
        import json

        updates["settings"] = json.dumps(request.settings)

    if updates:
        updated = await user_store.update_team(team_id, updates)
        if updated:
            logger.info(f"Team {team_id} updated by user {user.id}")
            team = await user_store.get_team(team_id)

    return {"team": team.to_dict() if team else {}}


@router.delete("/{org_id}/teams/{team_id}")
async def delete_team(
    org_id: int,
    team_id: int,
    user=Depends(require_permission("team.delete")),
):
    """Delete a team.

    Requires team.delete permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    success = await user_store.delete_team(team_id)
    if not success:
        raise HTTPException(500, "Failed to delete team")

    logger.info(f"Team {team_id} deleted by user {user.id}")
    return {"success": True, "message": "Team deleted"}


# ==================== Team Membership Routes ====================


@router.get("/{org_id}/teams/{team_id}/members")
async def list_team_members(
    org_id: int,
    team_id: int,
    user=Depends(require_permission("team.view")),
):
    """List members of a team.

    Requires team.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    members = await user_store.get_team_members(team_id)
    return {
        "team": team.to_dict(),
        "members": members,
    }


@router.post("/{org_id}/teams/{team_id}/members")
async def add_team_member(
    org_id: int,
    team_id: int,
    request: TeamMembershipRequest,
    user=Depends(require_permission("team.manage.members")),
):
    """Add a user to a team.

    Requires team.manage.members permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    success = await user_store.add_user_to_team(
        user_id=request.user_id,
        team_id=team_id,
        role=request.role,
    )

    if not success:
        return {"success": False, "message": "User already a team member"}

    logger.info(f"User {request.user_id} added to team {team_id} by user {user.id}")
    return {"success": True, "message": "User added to team"}


@router.patch("/{org_id}/teams/{team_id}/members/{target_user_id}")
async def update_team_member_role(
    org_id: int,
    team_id: int,
    target_user_id: int,
    request: TeamMembershipRequest,
    user=Depends(require_permission("team.manage.members")),
):
    """Update a team member's role.

    Roles: member, lead, admin
    - member: Basic team membership
    - lead: Can manage team members (but not leads or admins)
    - admin: Full team administration

    Requires team.manage.members permission, membership in the organization
    (or system admin), AND appropriate team role.
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    members = await user_store.get_team_members(team_id)

    # Get target user's current role
    target_current_role = get_user_team_role(members, target_user_id)
    if not target_current_role:
        raise HTTPException(404, "User is not a member of this team")

    # Get requesting user's role in this team (if they're a member)
    actor_role = get_user_team_role(members, user.id)

    # Check role-based permission: actor must be able to manage both the current and new role
    if actor_role:
        if not can_manage_role(actor_role, target_current_role):
            raise HTTPException(403, f"Your role ({actor_role}) cannot modify users with role '{target_current_role}'")
        if not can_manage_role(actor_role, request.role):
            raise HTTPException(403, f"Your role ({actor_role}) cannot assign role '{request.role}'")
    # If actor is not a team member but has team.manage.members permission (e.g., org admin),
    # they can proceed - the global permission takes precedence

    success = await user_store.update_team_member_role(target_user_id, team_id, request.role)
    if not success:
        raise HTTPException(400, "Failed to update role")

    logger.info(f"User {target_user_id} role updated to {request.role} in team {team_id} by user {user.id}")
    return {"success": True, "message": f"Role updated to {request.role}"}


@router.delete("/{org_id}/teams/{team_id}/members/{target_user_id}")
async def remove_team_member(
    org_id: int,
    team_id: int,
    target_user_id: int,
    user=Depends(require_permission("team.manage.members")),
):
    """Remove a user from a team.

    Requires team.manage.members permission, membership in the organization
    (or system admin), AND appropriate team role.
    Leads can remove members; admins can remove anyone.
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    members = await user_store.get_team_members(team_id)

    # Get target user's role
    target_role = get_user_team_role(members, target_user_id)
    if not target_role:
        raise HTTPException(404, "User is not a member of this team")

    # Get requesting user's role in this team (if they're a member)
    actor_role = get_user_team_role(members, user.id)

    # Check role-based permission
    if actor_role:
        if not can_manage_role(actor_role, target_role):
            raise HTTPException(403, f"Your role ({actor_role}) cannot remove users with role '{target_role}'")
    # If actor is not a team member but has team.manage.members permission (e.g., org admin),
    # they can proceed - the global permission takes precedence

    success = await user_store.remove_user_from_team(target_user_id, team_id)
    if not success:
        raise HTTPException(400, "Failed to remove user from team")

    logger.info(f"User {target_user_id} removed from team {team_id} by user {user.id}")
    return {"success": True, "message": "User removed from team"}


# ==================== Team Quota Routes ====================


@router.get("/{org_id}/teams/{team_id}/quotas")
async def get_team_quotas(
    org_id: int,
    team_id: int,
    user=Depends(require_permission("team.view")),
):
    """Get quota ceilings for a team.

    Requires team.view permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    quotas = await user_store.get_team_quotas([team_id])

    return {
        "team": team.to_dict(),
        "quotas": [q.to_dict() for q in quotas],
    }


@router.post("/{org_id}/teams/{team_id}/quotas")
async def set_team_quota(
    org_id: int,
    team_id: int,
    request: SetTeamQuotaRequest,
    user=Depends(require_permission("team.manage.quotas")),
):
    """Set a quota ceiling for a team.

    Team quotas act as ceilings for team members, but are themselves
    bounded by the organization's quota ceiling. If the requested limit_value
    exceeds the org's ceiling for the same quota_type, returns HTTP 400.

    Args:
        org_id: Organization ID owning the team
        team_id: Team ID to set quota for
        request: QuotaRequest with quota_type, limit_value, and action

    Raises:
        HTTPException(400): If team quota exceeds org ceiling
        HTTPException(404): If team not found or doesn't belong to org

    Requires team.manage.quotas permission and membership in the organization
    (or system admin).
    """
    require_org_access(user, org_id)

    from ..services.cloud.auth.quotas import QuotaAction, QuotaType
    from ..services.cloud.container import get_user_store

    user_store = get_user_store()

    team = await user_store.get_team(team_id)
    if not team or team.org_id != org_id:
        raise HTTPException(404, "Team not found")

    quota_type = QuotaType(request.quota_type)
    action = QuotaAction(request.action)

    # Check if team quota exceeds org ceiling
    org_quotas = await user_store.get_org_quotas(org_id)
    org_limit = next((q for q in org_quotas if q.quota_type == quota_type), None)

    if org_limit and request.limit_value > org_limit.limit_value:
        raise HTTPException(
            400, f"Team quota ({request.limit_value}) cannot exceed organization ceiling ({org_limit.limit_value})"
        )

    # Set the team quota
    await user_store.set_quota(
        quota_type=quota_type,
        limit_value=request.limit_value,
        action=action,
        team_id=team_id,
    )

    logger.info(f"Team {team_id} quota {quota_type.value} set to {request.limit_value} by user {user.id}")
    return {"success": True, "message": f"Team quota ceiling set: {quota_type.value} = {request.limit_value}"}
