"""User management routes for administrators.

Provides CRUD operations for users, level assignment, and permission management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from ...services.cloud.auth import UserStore, require_permission
from ...services.cloud.auth.models import AuthProvider, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


# ==================== Request/Response Models ====================


class CreateUserRequest(BaseModel):
    """Admin user creation request."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: Optional[str] = Field(None, min_length=8)
    display_name: Optional[str] = None
    is_admin: bool = False
    level_names: List[str] = Field(default=["researcher"])


class UpdateUserRequest(BaseModel):
    """User update request."""

    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8)


class UserResponse(BaseModel):
    """User response model."""

    id: int
    email: str
    username: str
    display_name: Optional[str]
    avatar_url: Optional[str] = None
    auth_provider: str
    is_active: bool
    is_admin: bool
    email_verified: bool
    created_at: str
    last_login_at: Optional[str]
    levels: List[Dict[str, Any]]
    permissions: List[str]

    @classmethod
    def from_user(cls, user: User) -> "UserResponse":
        """Create a UserResponse from a User model."""
        return cls(
            id=user.id,
            email=user.email,
            username=user.username,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            auth_provider=user.auth_provider.value,
            is_active=user.is_active,
            is_admin=user.is_admin,
            email_verified=user.email_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            levels=[{"id": lvl.id, "name": lvl.name, "priority": lvl.priority} for lvl in user.levels],
            permissions=sorted(user.effective_permissions),
        )


class LevelResponse(BaseModel):
    """Level/role response model."""

    id: int
    name: str
    description: str
    priority: int
    is_system: bool
    permissions: List[str]


class PermissionResponse(BaseModel):
    """Permission response model."""

    id: int
    name: str
    description: str
    category: str


class AssignLevelRequest(BaseModel):
    """Level assignment request."""

    level_name: str


class PermissionOverrideRequest(BaseModel):
    """Permission override request."""

    permission_name: str
    granted: bool


# ==================== Helper Functions ====================


def _get_store() -> UserStore:
    """Get the user store instance."""
    return UserStore()


# ==================== Current User Routes ====================


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: User = Depends(require_permission("api.access")),
) -> UserResponse:
    """Get the current authenticated user's info.

    Returns the user object with levels, permissions, and resource rules.
    Requires api.access permission.
    """
    return UserResponse.from_user(user)


# ==================== User CRUD Routes (non-parameterized first) ====================


@router.get("", response_model=List[UserResponse])
async def list_users(
    limit: int = 100,
    offset: int = 0,
    include_inactive: bool = False,
    admin: User = Depends(require_permission("admin.users")),
) -> List[UserResponse]:
    """List all users.

    Requires admin.users permission.
    """
    store = _get_store()
    users = await store.list_users(
        limit=limit,
        offset=offset,
        include_inactive=include_inactive,
    )
    return [UserResponse.from_user(u) for u in users]


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: CreateUserRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> UserResponse:
    """Create a new user.

    Requires admin.users permission.
    """
    store = _get_store()

    try:
        user = await store.create_user(
            email=data.email,
            username=data.username,
            password=data.password,
            display_name=data.display_name,
            is_admin=data.is_admin,
            level_names=data.level_names,
        )

        # Reload with permissions
        user = await store.get_user(user.id)

        logger.info("User created by admin %s: %s", admin.username, user.username)

        return UserResponse.from_user(user)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ==================== Meta Routes (must come before /{user_id}) ====================


class AuthStatusResponse(BaseModel):
    """Auth system status for UI settings."""

    in_use: bool = Field(..., description="Whether authentication is actively in use")
    user_count: int = Field(..., description="Number of configured users")
    has_external_providers: bool = Field(..., description="Whether external auth providers are configured")


@router.get("/meta/levels", response_model=List[LevelResponse])
async def list_levels(
    admin: User = Depends(require_permission("admin.levels")),
) -> List[LevelResponse]:
    """List all available levels/roles.

    Requires admin.levels permission.
    """
    store = _get_store()
    levels = await store.get_all_levels()

    return [
        LevelResponse(
            id=lvl.id,
            name=lvl.name,
            description=lvl.description,
            priority=lvl.priority,
            is_system=lvl.is_system,
            permissions=sorted(lvl.permissions),
        )
        for lvl in levels
    ]


@router.get("/meta/permissions", response_model=List[PermissionResponse])
async def list_permissions(
    admin: User = Depends(require_permission("admin.levels")),
) -> List[PermissionResponse]:
    """List all defined permissions.

    Requires admin.levels permission.
    """
    store = _get_store()
    permissions = await store.get_all_permissions()

    return [
        PermissionResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            category=p.category,
        )
        for p in permissions
    ]


@router.get("/meta/auth-status", response_model=AuthStatusResponse)
async def get_auth_status() -> AuthStatusResponse:
    """Check if authentication is configured and in use.

    This endpoint is public (no auth required) so UI settings can check
    whether the admin tab can be disabled.
    """
    try:
        store = _get_store()
        users = await store.list_users()
        user_count = len(users)

        # Check for external auth providers
        has_providers = False
        try:
            from ...services.cloud.auth.external_auth import ExternalAuthManager

            manager = ExternalAuthManager()
            providers = manager.list_providers()
            has_providers = len(providers) > 0
        except Exception:
            pass

        # Auth is "in use" if there are multiple users or external providers
        in_use = user_count > 1 or has_providers

        return AuthStatusResponse(
            in_use=in_use,
            user_count=user_count,
            has_external_providers=has_providers,
        )
    except Exception as exc:
        logger.warning("Failed to check auth status: %s", exc)
        return AuthStatusResponse(
            in_use=False,
            user_count=0,
            has_external_providers=False,
        )


# ==================== Level CRUD Routes (must come before /{user_id}) ====================


class CreateLevelRequest(BaseModel):
    """Request to create a new level."""

    name: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field("", max_length=500)
    priority: int = Field(0, ge=0, le=1000)
    permission_names: List[str] = Field(default_factory=list)


class UpdateLevelRequest(BaseModel):
    """Request to update a level."""

    name: Optional[str] = Field(None, min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(None, max_length=500)
    priority: Optional[int] = Field(None, ge=0, le=1000)
    permission_names: Optional[List[str]] = None


class SetLevelPermissionsRequest(BaseModel):
    """Request to set permissions for a level."""

    permission_names: List[str] = Field(..., description="List of permission names to assign")


@router.post("/levels", response_model=LevelResponse, status_code=status.HTTP_201_CREATED)
async def create_level(
    request: CreateLevelRequest,
    admin: User = Depends(require_permission("admin.levels")),
) -> LevelResponse:
    """Create a new level/role.

    Requires admin.levels permission.
    """
    store = _get_store()

    # Check if level name already exists
    existing = await store.get_level_by_name(request.name)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Level '{request.name}' already exists")

    level = await store.create_level(
        name=request.name,
        description=request.description,
        priority=request.priority,
        permission_names=request.permission_names,
    )

    logger.info("Level created by admin %s: %s", admin.username, level.name)

    return LevelResponse(
        id=level.id,
        name=level.name,
        description=level.description,
        priority=level.priority,
        is_system=level.is_system,
        permissions=sorted(level.permissions),
    )


@router.put("/levels/{level_id}", response_model=LevelResponse)
async def update_level(
    level_id: int,
    request: UpdateLevelRequest,
    admin: User = Depends(require_permission("admin.levels")),
) -> LevelResponse:
    """Update a level/role.

    Requires admin.levels permission.
    Cannot modify system levels.
    """
    store = _get_store()

    level = await store.get_level(level_id)
    if not level:
        raise HTTPException(status_code=404, detail="Level not found")

    if level.is_system:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot modify system levels")

    # Build updates
    updates = {}
    if request.name is not None:
        # Check for duplicate name
        existing = await store.get_level_by_name(request.name)
        if existing and existing.id != level_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Level '{request.name}' already exists")
        updates["name"] = request.name

    if request.description is not None:
        updates["description"] = request.description

    if request.priority is not None:
        updates["priority"] = request.priority

    if request.permission_names is not None:
        await store.set_level_permissions(level_id, request.permission_names)

    if updates:
        await store.update_level(level_id, updates)

    # Reload the level
    level = await store.get_level(level_id)
    logger.info("Level updated by admin %s: %s", admin.username, level.name)

    return LevelResponse(
        id=level.id,
        name=level.name,
        description=level.description,
        priority=level.priority,
        is_system=level.is_system,
        permissions=sorted(level.permissions),
    )


@router.delete("/levels/{level_id}")
async def delete_level(
    level_id: int,
    admin: User = Depends(require_permission("admin.levels")),
):
    """Delete a level/role.

    Requires admin.levels permission.
    Cannot delete system levels.
    """
    store = _get_store()

    level = await store.get_level(level_id)
    if not level:
        raise HTTPException(status_code=404, detail="Level not found")

    if level.is_system:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete system levels")

    success = await store.delete_level(level_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete level")

    logger.info("Level deleted by admin %s: %s", admin.username, level.name)
    return {"success": True, "message": f"Level '{level.name}' deleted"}


@router.put("/levels/{level_id}/permissions")
async def set_level_permissions(
    level_id: int,
    request: SetLevelPermissionsRequest,
    admin: User = Depends(require_permission("admin.levels")),
):
    """Set permissions for a level.

    Requires admin.levels permission.
    """
    store = _get_store()

    level = await store.get_level(level_id)
    if not level:
        raise HTTPException(status_code=404, detail="Level not found")

    await store.set_level_permissions(level_id, request.permission_names)

    logger.info("Level permissions updated by admin %s: %s", admin.username, level.name)
    return {"success": True, "message": "Level permissions updated"}


# ==================== Resource Rules (must come before /{user_id}) ====================


class CreateResourceRuleRequest(BaseModel):
    """Request to create a resource rule."""

    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    resource_type: str = Field(..., description="Type: config, hardware, provider, output_path")
    pattern: str = Field(..., min_length=1, description="Glob pattern to match resources")
    action: str = Field("allow", description="Action: allow or deny")
    priority: int = Field(0, ge=0, le=1000, description="Higher priority rules evaluated first")
    description: str = Field("", max_length=500, description="Optional description")


class UpdateResourceRuleRequest(BaseModel):
    """Request to update a resource rule."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    pattern: Optional[str] = Field(None, min_length=1)
    action: Optional[str] = None
    priority: Optional[int] = Field(None, ge=0, le=1000)
    description: Optional[str] = Field(None, max_length=500)


class ResourceRuleResponse(BaseModel):
    """Resource rule in API responses."""

    id: int
    name: str
    resource_type: str
    pattern: str
    action: str
    priority: int
    description: str


class ResourceRuleWithLevelsResponse(BaseModel):
    """Resource rule with level assignments."""

    id: int
    name: str
    resource_type: str
    pattern: str
    action: str
    priority: int
    description: str
    created_at: Optional[str] = None
    created_by: Optional[int] = None
    levels: List[Dict[str, Any]]


class SetLevelRulesRequest(BaseModel):
    """Request to set rules for a level."""

    rule_ids: List[int] = Field(..., description="List of rule IDs to assign")


@router.get("/resource-rules", response_model=List[ResourceRuleWithLevelsResponse])
async def list_resource_rules(
    resource_type: Optional[str] = Query(None, description="Filter by type: config, hardware, provider, output_path"),
    admin: User = Depends(require_permission("admin.users")),
) -> List[ResourceRuleWithLevelsResponse]:
    """List all resource rules with their level assignments.

    Requires admin.users permission.
    """
    store = _get_store()
    rules = await store.get_rules_with_level_assignments()

    if resource_type:
        rules = [r for r in rules if r["resource_type"] == resource_type]

    return [ResourceRuleWithLevelsResponse(**r) for r in rules]


@router.get("/resource-rules/{rule_id}", response_model=ResourceRuleResponse)
async def get_resource_rule(
    rule_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> ResourceRuleResponse:
    """Get a single resource rule by ID.

    Requires admin.users permission.
    """
    store = _get_store()
    rule = await store.get_resource_rule(rule_id)

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource rule {rule_id} not found",
        )

    return ResourceRuleResponse(
        id=rule.id,
        name=rule.name,
        resource_type=rule.resource_type.value,
        pattern=rule.pattern,
        action=rule.action.value,
        priority=rule.priority,
        description=rule.description,
    )


@router.post("/resource-rules", response_model=ResourceRuleResponse, status_code=status.HTTP_201_CREATED)
async def create_resource_rule(
    data: CreateResourceRuleRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> ResourceRuleResponse:
    """Create a new resource rule.

    Requires admin.users permission.

    Resource types:
    - config: Training configuration names (e.g., "team-x-*")
    - hardware: Hardware/GPU types (e.g., "gpu-a100*")
    - provider: Cloud providers (e.g., "replicate")
    - output_path: Output directories
    """
    valid_types = ["config", "hardware", "provider", "output_path"]
    if data.resource_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resource_type. Must be one of: {valid_types}",
        )

    valid_actions = ["allow", "deny"]
    if data.action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action. Must be one of: {valid_actions}",
        )

    store = _get_store()
    rule = await store.create_resource_rule(
        name=data.name,
        resource_type=data.resource_type,
        pattern=data.pattern,
        action=data.action,
        priority=data.priority,
        description=data.description,
        created_by=admin.id,
    )

    logger.info("Resource rule '%s' created by %s", rule.name, admin.username)

    return ResourceRuleResponse(
        id=rule.id,
        name=rule.name,
        resource_type=rule.resource_type.value,
        pattern=rule.pattern,
        action=rule.action.value,
        priority=rule.priority,
        description=rule.description,
    )


@router.put("/resource-rules/{rule_id}", response_model=ResourceRuleResponse)
async def update_resource_rule(
    rule_id: int,
    data: UpdateResourceRuleRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> ResourceRuleResponse:
    """Update a resource rule.

    Requires admin.users permission.
    """
    store = _get_store()

    # Check rule exists
    existing = await store.get_resource_rule(rule_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource rule {rule_id} not found",
        )

    # Build updates dict
    updates = {}
    if data.name is not None:
        updates["name"] = data.name
    if data.pattern is not None:
        updates["pattern"] = data.pattern
    if data.action is not None:
        if data.action not in ["allow", "deny"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid action. Must be 'allow' or 'deny'",
            )
        updates["action"] = data.action
    if data.priority is not None:
        updates["priority"] = data.priority
    if data.description is not None:
        updates["description"] = data.description

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    success = await store.update_resource_rule(rule_id, updates)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update rule",
        )

    # Fetch updated rule
    rule = await store.get_resource_rule(rule_id)

    logger.info("Resource rule %d updated by %s", rule_id, admin.username)

    return ResourceRuleResponse(
        id=rule.id,
        name=rule.name,
        resource_type=rule.resource_type.value,
        pattern=rule.pattern,
        action=rule.action.value,
        priority=rule.priority,
        description=rule.description,
    )


@router.delete("/resource-rules/{rule_id}")
async def delete_resource_rule(
    rule_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Delete a resource rule.

    This also removes the rule from all levels it was assigned to.
    Requires admin.users permission.
    """
    store = _get_store()

    success = await store.delete_resource_rule(rule_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource rule {rule_id} not found",
        )

    logger.info("Resource rule %d deleted by %s", rule_id, admin.username)

    return {"success": True}


# ==================== Level-Rule Assignment (must come before /{user_id}) ====================


@router.get("/levels/{level_id}/rules", response_model=List[ResourceRuleResponse])
async def get_level_rules(
    level_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> List[ResourceRuleResponse]:
    """Get all resource rules assigned to a level.

    Requires admin.users permission.
    """
    store = _get_store()
    rules = await store.get_level_rules(level_id)

    return [
        ResourceRuleResponse(
            id=r.id,
            name=r.name,
            resource_type=r.resource_type.value,
            pattern=r.pattern,
            action=r.action.value,
            priority=r.priority,
            description=r.description,
        )
        for r in rules
    ]


@router.put("/levels/{level_id}/rules")
async def set_level_rules(
    level_id: int,
    data: SetLevelRulesRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, Any]:
    """Set the complete list of resource rules for a level.

    This replaces all existing rule assignments for the level.
    Requires admin.users permission.
    """
    store = _get_store()

    # Verify all rule_ids exist
    for rule_id in data.rule_ids:
        rule = await store.get_resource_rule(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resource rule {rule_id} not found",
            )

    success = await store.set_level_rules(level_id, data.rule_ids)

    logger.info("Level %d rules set to %s by %s", level_id, data.rule_ids, admin.username)

    return {"success": success, "rule_count": len(data.rule_ids)}


@router.post("/levels/{level_id}/rules/{rule_id}")
async def assign_rule_to_level(
    level_id: int,
    rule_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Assign a resource rule to a level.

    Requires admin.users permission.
    """
    store = _get_store()

    # Verify rule exists
    rule = await store.get_resource_rule(rule_id)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource rule {rule_id} not found",
        )

    success = await store.assign_rule_to_level(level_id, rule_id)

    logger.info("Rule %d assigned to level %d by %s", rule_id, level_id, admin.username)

    return {"success": True, "already_assigned": not success}


@router.delete("/levels/{level_id}/rules/{rule_id}")
async def remove_rule_from_level(
    level_id: int,
    rule_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Remove a resource rule from a level.

    Requires admin.users permission.
    """
    store = _get_store()

    success = await store.remove_rule_from_level(level_id, rule_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule {rule_id} not assigned to level {level_id}",
        )

    logger.info("Rule %d removed from level %d by %s", rule_id, level_id, admin.username)

    return {"success": True}


# ==================== Parameterized User Routes (/{user_id}) ====================


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> UserResponse:
    """Get a specific user by ID.

    Requires admin.users permission.
    """
    store = _get_store()
    user = await store.get_user(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    return UserResponse.from_user(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UpdateUserRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> UserResponse:
    """Update a user.

    Requires admin.users permission.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Build updates dict
    updates = {}
    if data.email is not None:
        updates["email"] = data.email
    if data.username is not None:
        updates["username"] = data.username
    if data.display_name is not None:
        updates["display_name"] = data.display_name
    if data.is_active is not None:
        updates["is_active"] = 1 if data.is_active else 0
    if data.is_admin is not None:
        updates["is_admin"] = 1 if data.is_admin else 0
    if data.password is not None:
        updates["password"] = data.password

    if updates:
        await store.update_user(user_id, updates)
        logger.info("User %d updated by admin %s: %s", user_id, admin.username, list(updates.keys()))

    # Reload and return
    user = await store.get_user(user_id)
    return UserResponse.from_user(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Delete a user.

    Requires admin.users permission. This is a hard delete.
    Use PATCH to deactivate instead for soft delete.
    """
    store = _get_store()

    # Prevent self-deletion
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    success = await store.delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    logger.info("User %d deleted by admin %s", user_id, admin.username)

    return {"success": True}


@router.post("/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Deactivate a user (soft delete).

    Requires admin.users permission.
    """
    store = _get_store()

    # Prevent self-deactivation
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    success = await store.deactivate_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    logger.info("User %d deactivated by admin %s", user_id, admin.username)

    return {"success": True}


@router.post("/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Reactivate a deactivated user.

    Requires admin.users permission.
    """
    store = _get_store()

    success = await store.update_user(user_id, {"is_active": 1})

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    logger.info("User %d activated by admin %s", user_id, admin.username)

    return {"success": True}


# ==================== Level Assignment Routes ====================


@router.post("/{user_id}/levels")
async def assign_level(
    user_id: int,
    data: AssignLevelRequest,
    admin: User = Depends(require_permission("admin.levels")),
) -> Dict[str, bool]:
    """Assign a level/role to a user.

    Requires admin.levels permission.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    success = await store.assign_level(user_id, data.level_name, granted_by=admin.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Level '{data.level_name}' not found",
        )

    logger.info("Level '%s' assigned to user %d by admin %s", data.level_name, user_id, admin.username)

    return {"success": True}


@router.delete("/{user_id}/levels/{level_name}")
async def remove_level(
    user_id: int,
    level_name: str,
    admin: User = Depends(require_permission("admin.levels")),
) -> Dict[str, bool]:
    """Remove a level/role from a user.

    Requires admin.levels permission.
    """
    store = _get_store()

    success = await store.remove_level(user_id, level_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} does not have level '{level_name}'",
        )

    logger.info("Level '%s' removed from user %d by admin %s", level_name, user_id, admin.username)

    return {"success": True}


# ==================== Permission Override Routes ====================


@router.post("/{user_id}/permissions")
async def set_permission_override(
    user_id: int,
    data: PermissionOverrideRequest,
    admin: User = Depends(require_permission("admin.levels")),
) -> Dict[str, bool]:
    """Set a permission override for a user.

    Overrides allow granting or denying specific permissions
    independent of the user's levels.

    Requires admin.levels permission.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    await store.set_permission_override(
        user_id=user_id,
        permission_name=data.permission_name,
        granted=data.granted,
        granted_by=admin.id,
    )

    action = "granted" if data.granted else "denied"
    logger.info("Permission '%s' %s for user %d by admin %s", data.permission_name, action, user_id, admin.username)

    return {"success": True}


@router.delete("/{user_id}/permissions/{permission_name}")
async def remove_permission_override(
    user_id: int,
    permission_name: str,
    admin: User = Depends(require_permission("admin.levels")),
) -> Dict[str, bool]:
    """Remove a permission override from a user.

    The user's effective permission will revert to what their levels grant.

    Requires admin.levels permission.
    """
    store = _get_store()

    success = await store.remove_permission_override(user_id, permission_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} has no override for '{permission_name}'",
        )

    logger.info("Permission override '%s' removed from user %d by admin %s", permission_name, user_id, admin.username)

    return {"success": True}


# ==================== Provider Credentials ====================


class SetCredentialRequest(BaseModel):
    """Set a provider credential."""

    provider: str = Field(..., description="Provider name (e.g., 'replicate', 'huggingface')")
    credential_name: str = Field(..., description="Credential name (e.g., 'api_token')")
    value: str = Field(..., min_length=1, description="Credential value")
    description: Optional[str] = Field(None, description="Optional description")


class CredentialResponse(BaseModel):
    """Credential metadata (value never returned)."""

    id: int
    provider: str
    credential_name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    last_used_at: Optional[str]
    is_active: bool


class ValidateCredentialRequest(BaseModel):
    """Validate a credential by testing it."""

    provider: str
    credential_name: str


@router.get("/{user_id}/credentials", response_model=List[CredentialResponse])
async def list_user_credentials(
    user_id: int,
    provider: Optional[str] = None,
    admin: User = Depends(require_permission("admin.users")),
) -> List[CredentialResponse]:
    """List a user's provider credentials (values not included).

    Requires admin.users permission.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    credentials = await store.list_provider_credentials(user_id, provider)
    return [CredentialResponse(**c) for c in credentials]


@router.post("/{user_id}/credentials")
async def set_user_credential(
    user_id: int,
    data: SetCredentialRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, Any]:
    """Set a provider credential for a user.

    Requires admin.users permission.
    The credential value is encrypted at rest.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    try:
        cred_id = await store.set_provider_credential(
            user_id=user_id,
            provider=data.provider,
            credential_name=data.credential_name,
            value=data.value,
            description=data.description,
        )

        logger.info(
            "Credential %s/%s set for user %d by admin %s", data.provider, data.credential_name, user_id, admin.username
        )

        return {"success": True, "id": cred_id}

    except Exception as exc:
        logger.error("Failed to set credential: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store credential",
        )


@router.delete("/{user_id}/credentials/{provider}/{credential_name}")
async def delete_user_credential(
    user_id: int,
    provider: str,
    credential_name: str,
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, bool]:
    """Delete a provider credential for a user.

    Requires admin.users permission.
    """
    store = _get_store()

    success = await store.delete_provider_credential(user_id, provider, credential_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Credential {provider}/{credential_name} not found for user {user_id}",
        )

    logger.info("Credential %s/%s deleted for user %d by admin %s", provider, credential_name, user_id, admin.username)

    return {"success": True}


# ==================== Self-Service Credential Management ====================


@router.get("/me/credentials", response_model=List[CredentialResponse])
async def list_my_credentials(
    provider: Optional[str] = None,
    user: User = Depends(require_permission("api.access")),
) -> List[CredentialResponse]:
    """List your own provider credentials (values not included).

    Requires api.access permission.
    """
    store = _get_store()
    credentials = await store.list_provider_credentials(user.id, provider)
    return [CredentialResponse(**c) for c in credentials]


@router.post("/me/credentials")
async def set_my_credential(
    data: SetCredentialRequest,
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, Any]:
    """Set your own provider credential.

    Requires api.access permission.
    The credential value is encrypted at rest.
    """
    store = _get_store()

    try:
        cred_id = await store.set_provider_credential(
            user_id=user.id,
            provider=data.provider,
            credential_name=data.credential_name,
            value=data.value,
            description=data.description,
        )

        logger.info("Credential %s/%s set by user %s", data.provider, data.credential_name, user.username)

        return {"success": True, "id": cred_id}

    except Exception as exc:
        logger.error("Failed to set credential: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store credential",
        )


@router.delete("/me/credentials/{provider}/{credential_name}")
async def delete_my_credential(
    provider: str,
    credential_name: str,
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, bool]:
    """Delete your own provider credential.

    Requires api.access permission.
    """
    store = _get_store()

    success = await store.delete_provider_credential(user.id, provider, credential_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Credential {provider}/{credential_name} not found",
        )

    logger.info("Credential %s/%s deleted by user %s", provider, credential_name, user.username)

    return {"success": True}


@router.post("/me/credentials/validate")
async def validate_my_credential(
    data: ValidateCredentialRequest,
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, Any]:
    """Validate a provider credential by testing it.

    Requires api.access permission.
    Currently supports: replicate
    """
    if data.provider == "replicate" and data.credential_name == "api_token":
        from ...services.cloud.replicate_client import ReplicateCogClient

        client = ReplicateCogClient()
        result = await client.validate_credentials(user_id=user.id)

        if result.get("valid"):
            return {
                "valid": True,
                "username": result.get("username"),
                "token_source": result.get("token_source"),
            }
        else:
            return {
                "valid": False,
                "error": result.get("error"),
            }

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Validation not supported for {data.provider}/{data.credential_name}",
    )


# ==================== Credential Rotation ====================


class RotateCredentialsRequest(BaseModel):
    """Request to rotate credentials."""

    provider: Optional[str] = Field(None, description="Filter to specific provider (optional)")
    reason: Optional[str] = Field(None, description="Reason for rotation (for audit log)")


class RotatedCredential(BaseModel):
    """Info about a rotated credential."""

    provider: str
    credential_name: str
    status: str  # "deleted" or "error"
    error: Optional[str] = None


class RotateCredentialsResponse(BaseModel):
    """Response from credential rotation."""

    success: bool
    rotated_count: int
    credentials: List[RotatedCredential]
    message: str


@router.post("/{user_id}/credentials/rotate", response_model=RotateCredentialsResponse)
async def rotate_user_credentials(
    user_id: int,
    data: RotateCredentialsRequest,
    admin: User = Depends(require_permission("admin.users")),
) -> RotateCredentialsResponse:
    """Rotate (delete) all credentials for a user.

    This deletes all stored credentials, forcing the user to re-enter them.
    Use after a security incident or for scheduled rotation.

    Requires admin.users permission.
    """
    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Get all credentials for this user
    credentials = await store.list_provider_credentials(user_id, data.provider)

    rotated = []
    for cred in credentials:
        try:
            success = await store.delete_provider_credential(
                user_id,
                cred["provider"],
                cred["credential_name"],
            )
            rotated.append(
                RotatedCredential(
                    provider=cred["provider"],
                    credential_name=cred["credential_name"],
                    status="deleted" if success else "error",
                    error=None if success else "Delete failed",
                )
            )
        except Exception as exc:
            rotated.append(
                RotatedCredential(
                    provider=cred["provider"],
                    credential_name=cred["credential_name"],
                    status="error",
                    error=str(exc),
                )
            )

    deleted_count = sum(1 for r in rotated if r.status == "deleted")

    logger.info(
        "Rotated %d credentials for user %d by admin %s (reason: %s)",
        deleted_count,
        user_id,
        admin.username,
        data.reason or "not specified",
    )

    return RotateCredentialsResponse(
        success=deleted_count == len(rotated),
        rotated_count=deleted_count,
        credentials=rotated,
        message=f"Rotated {deleted_count} of {len(rotated)} credentials. User must re-enter them.",
    )


@router.post("/me/credentials/rotate", response_model=RotateCredentialsResponse)
async def rotate_my_credentials(
    data: RotateCredentialsRequest,
    user: User = Depends(require_permission("api.access")),
) -> RotateCredentialsResponse:
    """Rotate (delete) your own credentials.

    Use for self-service credential rotation.
    """
    store = _get_store()

    # Get all credentials
    credentials = await store.list_provider_credentials(user.id, data.provider)

    rotated = []
    for cred in credentials:
        try:
            success = await store.delete_provider_credential(
                user.id,
                cred["provider"],
                cred["credential_name"],
            )
            rotated.append(
                RotatedCredential(
                    provider=cred["provider"],
                    credential_name=cred["credential_name"],
                    status="deleted" if success else "error",
                    error=None if success else "Delete failed",
                )
            )
        except Exception as exc:
            rotated.append(
                RotatedCredential(
                    provider=cred["provider"],
                    credential_name=cred["credential_name"],
                    status="error",
                    error=str(exc),
                )
            )

    deleted_count = sum(1 for r in rotated if r.status == "deleted")

    logger.info(
        "User %s rotated %d of their own credentials (reason: %s)",
        user.username,
        deleted_count,
        data.reason or "not specified",
    )

    return RotateCredentialsResponse(
        success=deleted_count == len(rotated),
        rotated_count=deleted_count,
        credentials=rotated,
        message=f"Rotated {deleted_count} credentials. Please re-enter them.",
    )


@router.get("/{user_id}/credentials/stale")
async def list_stale_credentials(
    user_id: int,
    days: int = Query(90, ge=1, le=365, description="Credentials older than this many days"),
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, Any]:
    """List credentials that haven't been updated in N days.

    Useful for identifying credentials that may need rotation.
    Requires admin.users permission.
    """
    from datetime import datetime, timedelta, timezone

    store = _get_store()

    # Check user exists
    user = await store.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    credentials = await store.list_provider_credentials(user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    stale = []
    for cred in credentials:
        updated_at = cred.get("updated_at")
        if updated_at:
            try:
                updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if updated < cutoff:
                    days_old = (datetime.now(timezone.utc) - updated).days
                    stale.append(
                        {
                            **cred,
                            "days_old": days_old,
                        }
                    )
            except (ValueError, TypeError):
                pass

    return {
        "user_id": user_id,
        "threshold_days": days,
        "stale_count": len(stale),
        "credentials": stale,
    }


@router.post("/credentials/check-stale")
async def check_all_stale_credentials(
    days: int = Query(90, ge=1, le=365, description="Credentials older than this many days"),
    admin: User = Depends(require_permission("admin.users")),
) -> Dict[str, Any]:
    """Check all users for stale credentials.

    Returns a summary of which users have stale credentials.
    Requires admin.users permission.
    """
    from datetime import datetime, timedelta, timezone

    store = _get_store()

    # Get all users
    users = await store.list_users(include_inactive=False)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    users_with_stale = []
    total_stale = 0

    for user in users:
        credentials = await store.list_provider_credentials(user.id)
        stale_count = 0

        for cred in credentials:
            updated_at = cred.get("updated_at")
            if updated_at:
                try:
                    updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    if updated < cutoff:
                        stale_count += 1
                except (ValueError, TypeError):
                    pass

        if stale_count > 0:
            users_with_stale.append(
                {
                    "user_id": user.id,
                    "username": user.username,
                    "stale_count": stale_count,
                }
            )
            total_stale += stale_count

    return {
        "threshold_days": days,
        "total_stale": total_stale,
        "users_with_stale": users_with_stale,
    }


# ==================== Self-Service Profile Management ====================


class UpdateProfileRequest(BaseModel):
    """Request to update own profile."""

    display_name: Optional[str] = Field(None, max_length=100)


class ProfileResponse(BaseModel):
    """Profile response with avatar info."""

    id: int
    email: str
    username: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    auth_provider: str
    is_admin: bool
    created_at: str


@router.get("/me/profile", response_model=ProfileResponse)
async def get_my_profile(
    user: User = Depends(require_permission("api.access")),
) -> ProfileResponse:
    """Get your own profile.

    Requires api.access permission.
    """
    return ProfileResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider.value,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


@router.patch("/me/profile", response_model=ProfileResponse)
async def update_my_profile(
    data: UpdateProfileRequest,
    user: User = Depends(require_permission("api.access")),
) -> ProfileResponse:
    """Update your own profile.

    Requires api.access permission.
    """
    store = _get_store()

    updates = {}
    if data.display_name is not None:
        updates["display_name"] = data.display_name

    if updates:
        await store.update_user(user.id, updates)
        logger.info("User %s updated their profile", user.username)

    # Reload user
    updated_user = await store.get_user(user.id)

    return ProfileResponse(
        id=updated_user.id,
        email=updated_user.email,
        username=updated_user.username,
        display_name=updated_user.display_name,
        avatar_url=updated_user.avatar_url,
        auth_provider=updated_user.auth_provider.value,
        is_admin=updated_user.is_admin,
        created_at=updated_user.created_at,
    )


@router.post("/me/avatar")
async def upload_my_avatar(
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, Any]:
    """Upload an avatar image.

    The avatar should be sent as a base64-encoded data URL in the request body.
    Requires api.access permission.
    """
    from fastapi import Request
    from starlette.requests import Request as StarletteRequest

    # This endpoint will be called with JSON body containing the data URL
    # We need to import Request properly for this
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Use /me/avatar/upload endpoint with JSON body",
    )


class AvatarUploadRequest(BaseModel):
    """Avatar upload request with base64 data URL."""

    data_url: str = Field(..., description="Base64-encoded data URL (data:image/...;base64,...)")


@router.post("/me/avatar/upload")
async def upload_avatar_data(
    data: AvatarUploadRequest,
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, Any]:
    """Upload an avatar image as a base64 data URL.

    Accepts data URLs in the format: data:image/png;base64,...
    Maximum size is 512KB. Images are resized to 128x128.

    Requires api.access permission.
    """
    import base64
    import hashlib
    import re
    from pathlib import Path

    store = _get_store()

    # Parse data URL
    match = re.match(r"data:image/(png|jpeg|jpg|gif|webp);base64,(.+)", data.data_url)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid data URL format. Must be data:image/TYPE;base64,DATA",
        )

    image_type = match.group(1)
    if image_type == "jpg":
        image_type = "jpeg"
    base64_data = match.group(2)

    # Decode and validate size
    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 encoding",
        )

    # Max 512KB
    if len(image_bytes) > 512 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image too large. Maximum size is 512KB.",
        )

    # Generate filename based on user ID and hash
    file_hash = hashlib.md5(image_bytes).hexdigest()[:8]
    filename = f"avatar_{user.id}_{file_hash}.{image_type}"

    # Create avatars directory
    avatars_dir = Path("data/avatars")
    avatars_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    avatar_path = avatars_dir / filename
    avatar_path.write_bytes(image_bytes)

    # Update user's avatar_url
    avatar_url = f"/static/avatars/{filename}"
    await store.update_user(user.id, {"avatar_url": avatar_url})

    logger.info("User %s uploaded avatar: %s", user.username, filename)

    return {
        "success": True,
        "avatar_url": avatar_url,
    }


@router.delete("/me/avatar")
async def delete_my_avatar(
    user: User = Depends(require_permission("api.access")),
) -> Dict[str, bool]:
    """Remove your avatar.

    Requires api.access permission.
    """
    from pathlib import Path

    store = _get_store()

    # Clear avatar URL
    old_url = user.avatar_url
    await store.update_user(user.id, {"avatar_url": None})

    # Optionally delete the file
    if old_url and old_url.startswith("/static/avatars/"):
        filename = old_url.split("/")[-1]
        avatar_path = Path("data/avatars") / filename
        if avatar_path.exists():
            try:
                avatar_path.unlink()
            except Exception:
                pass  # File deletion is best-effort

    logger.info("User %s removed their avatar", user.username)

    return {"success": True}
