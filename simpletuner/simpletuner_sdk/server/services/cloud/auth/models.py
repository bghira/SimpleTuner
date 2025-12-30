"""Data models for authentication and authorization.

Defines User, APIKey, Permission, Level, and ResourceRule models
used throughout the auth system.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AuthProvider(str, Enum):
    """Authentication provider types."""

    LOCAL = "local"
    OIDC = "oidc"
    LDAP = "ldap"


class ResourceType(str, Enum):
    """Types of resources that can have access rules."""

    CONFIG = "config"  # Training configuration names
    HARDWARE = "hardware"  # Hardware/GPU types
    PROVIDER = "provider"  # Cloud providers
    OUTPUT_PATH = "output_path"  # Output directories


class RuleAction(str, Enum):
    """Actions for resource rules."""

    ALLOW = "allow"  # Explicitly allow matching resources
    DENY = "deny"  # Explicitly deny matching resources


@dataclass
class Organization:
    """An organization that contains teams and users.

    Organizations provide the top-level quota ceiling. All users and teams
    within an organization are bound by its limits, regardless of any
    higher limits assigned at the team or user level.

    The org limit acts as an absolute ceiling:
    - If org allows 100 concurrent jobs, no user can exceed that
    - Even if a team lead assigns a user 200 concurrent jobs, org ceiling applies
    """

    id: int
    name: str
    slug: str  # URL-safe identifier, unique
    description: str = ""
    is_active: bool = True
    created_at: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Organization":
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            name=data["name"],
            slug=data["slug"],
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", ""),
            settings=data.get("settings", {}),
        )


@dataclass
class Team:
    """A team within an organization.

    Teams group users for fair-share scheduling and quota management.
    Team limits act as a ceiling below the organization level:
    - Org ceiling > Team ceiling > User/Level limits

    Users can belong to multiple teams for cross-functional work,
    but quota limits use the team with the highest limit for each quota type.
    """

    id: int
    org_id: int  # Parent organization
    name: str
    slug: str  # URL-safe identifier, unique within org
    description: str = ""
    is_active: bool = True
    created_at: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Team":
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            org_id=data["org_id"],
            name=data["name"],
            slug=data["slug"],
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", ""),
            settings=data.get("settings", {}),
        )


@dataclass
class ResourceRule:
    """A rule for resource-based access control.

    Rules allow restricting users to specific resources based on patterns.
    For example, restricting a team to configs matching "team-x-*".

    Rules are assigned to levels. Users inherit rules from their levels.
    Uses "most permissive wins" logic: if any matching ALLOW rule exists,
    access is granted regardless of DENY rules.

    Examples:
        # Allow only team-x configs
        ResourceRule(name="Team X Configs", resource_type=ResourceType.CONFIG,
                     pattern="team-x-*", action=RuleAction.ALLOW)

        # Deny expensive hardware
        ResourceRule(name="Block A100", resource_type=ResourceType.HARDWARE,
                     pattern="gpu-a100*", action=RuleAction.DENY)
    """

    id: int
    name: str  # Human-readable name for the rule
    resource_type: ResourceType
    pattern: str  # Glob pattern (e.g., "team-x-*", "*.json", "gpu-*")
    action: RuleAction
    priority: int = 0  # Higher priority evaluated first
    description: str = ""

    # Optional: restrict to specific permission context
    # e.g., only apply this rule for job.submit, not job.view
    applies_to_permissions: Optional[List[str]] = None

    def matches(self, resource_name: str) -> bool:
        """Check if this rule matches a resource name."""
        return fnmatch.fnmatch(resource_name, self.pattern)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "resource_type": self.resource_type.value,
            "pattern": self.pattern,
            "action": self.action.value,
            "priority": self.priority,
            "description": self.description,
            "applies_to_permissions": self.applies_to_permissions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceRule":
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            resource_type=ResourceType(data["resource_type"]),
            pattern=data["pattern"],
            action=RuleAction(data.get("action", "allow")),
            priority=data.get("priority", 0),
            description=data.get("description", ""),
            applies_to_permissions=data.get("applies_to_permissions"),
        )


@dataclass
class Permission:
    """A single permission that can be granted to users or levels.

    Permissions follow a hierarchical naming convention:
    - resource.action (e.g., "job.submit", "job.cancel")
    - resource.scope.action (e.g., "job.own.cancel", "job.all.cancel")
    """

    id: int
    name: str
    description: str
    category: str = "general"  # For UI grouping

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Permission):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False


@dataclass
class UserLevel:
    """A role/level that groups permissions.

    Users can have multiple levels. Permissions are combined additively.
    Higher priority levels take precedence for conflicting settings.
    """

    id: int
    name: str
    description: str
    priority: int = 0  # Higher = more privileged
    is_system: bool = False  # System levels can't be deleted

    # Permissions granted by this level (populated when loaded)
    permissions: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, UserLevel):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False


@dataclass
class APIKey:
    """An API key for programmatic access.

    API keys are associated with a user and have a name for identification.
    The actual key is only shown once at creation time.
    """

    id: int
    user_id: int
    name: str
    key_prefix: str  # First 8 chars for identification (e.g., "st_abc123")
    created_at: str
    key_hash: Optional[str] = None  # Omitted in create response, present when reading from DB
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool = True

    # Permissions can be scoped per key (subset of user's permissions)
    scoped_permissions: Optional[Set[str]] = None

    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(expires.tzinfo) > expires
        except (ValueError, TypeError):
            return False


@dataclass
class User:
    """A user account.

    Users can authenticate via local password, OIDC, or LDAP.
    They have levels (roles) that grant permissions, plus optional
    per-user permission overrides and resource access rules.

    Users belong to an organization (optional) and can be members of
    multiple teams within that organization. Quota limits are resolved
    using a ceiling model:
    - Org limit is the absolute ceiling
    - Team limits further constrain within org ceiling
    - User/level limits are bounded by team and org ceilings
    """

    id: int
    email: str
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    # Organization and team membership
    org_id: Optional[int] = None  # Parent organization (ceiling provider)
    organization: Optional[Organization] = field(default=None, repr=False)
    teams: List[Team] = field(default_factory=list)  # Team memberships

    # Authentication
    auth_provider: AuthProvider = AuthProvider.LOCAL
    password_hash: Optional[str] = None  # Only for local auth
    external_id: Optional[str] = None  # OIDC/LDAP subject ID

    # Status
    is_active: bool = True
    is_admin: bool = False  # Shortcut for full admin access
    email_verified: bool = False

    # Timestamps
    created_at: str = ""
    last_login_at: Optional[str] = None

    # Authorization (populated when loaded with permissions)
    levels: List[UserLevel] = field(default_factory=list)
    permission_overrides: Dict[str, bool] = field(default_factory=dict)
    # Combined effective permissions (calculated)
    _effective_permissions: Optional[Set[str]] = field(default=None, repr=False)

    # Resource-based access control (populated from levels)
    resource_rules: List[ResourceRule] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_permissions(self) -> Set[str]:
        """Get the combined set of effective permissions.

        Combines permissions from all levels, then applies user-specific overrides.
        """
        if self._effective_permissions is not None:
            return self._effective_permissions

        # Admin has all permissions
        if self.is_admin:
            return {"*"}

        # Combine permissions from all levels
        perms = set()
        for level in self.levels:
            perms.update(level.permissions)

        # Apply overrides
        for perm, granted in self.permission_overrides.items():
            if granted:
                perms.add(perm)
            else:
                perms.discard(perm)

        return perms

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission.

        Supports wildcard matching:
        - "*" grants all permissions
        - "job.*" grants all job permissions
        """
        if not self.is_active:
            return False

        effective = self.effective_permissions

        # Direct match or admin wildcard
        if permission in effective or "*" in effective:
            return True

        # Check hierarchical wildcards (e.g., "job.*" matches "job.submit")
        parts = permission.split(".")
        for i in range(len(parts)):
            wildcard = ".".join(parts[: i + 1]) + ".*"
            if wildcard in effective:
                return True

        return False

    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if user has all of the specified permissions."""
        return all(self.has_permission(p) for p in permissions)

    def can_access_resource(
        self,
        resource_type: ResourceType,
        resource_name: str,
        permission_context: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if user can access a specific resource.

        Uses "most permissive wins" logic: if ANY matching rule allows access,
        access is granted regardless of deny rules.

        Args:
            resource_type: Type of resource being accessed
            resource_name: Name/identifier of the resource
            permission_context: Optional permission being exercised (e.g., "job.submit")

        Returns:
            (True, None) if access is allowed
            (False, "reason") if access is denied

        Examples:
            # Check if user can submit with config "team-x-training"
            allowed, reason = user.can_access_resource(
                ResourceType.CONFIG,
                "team-x-training",
                permission_context="job.submit"
            )
        """
        # Admins bypass all resource restrictions
        if self.is_admin:
            return True, None

        # No rules = allow by default (no restrictions configured)
        if not self.resource_rules:
            return True, None

        # Filter rules by resource type
        applicable_rules = [r for r in self.resource_rules if r.resource_type == resource_type]

        if not applicable_rules:
            # No rules for this resource type = allow by default
            return True, None

        # Filter by permission context if applicable
        if permission_context:
            applicable_rules = [
                r
                for r in applicable_rules
                if r.applies_to_permissions is None or permission_context in r.applies_to_permissions
            ]

        if not applicable_rules:
            return True, None

        # Most permissive wins: check if ANY rule allows access
        matching_allow = None
        matching_deny = None
        deny_reason = None

        for rule in applicable_rules:
            if rule.matches(resource_name):
                if rule.action == RuleAction.ALLOW:
                    # Found an allow rule - immediately grant access
                    matching_allow = rule
                    break
                elif rule.action == RuleAction.DENY and matching_deny is None:
                    # Track first deny for error message if no allow found
                    matching_deny = rule
                    deny_reason = f"Access denied by rule: {rule.description or rule.pattern}"

        # If any allow rule matched, grant access
        if matching_allow is not None:
            return True, None

        # If a deny rule matched with no allow, deny access
        if matching_deny is not None:
            return False, deny_reason

        # No matching rules = deny by default (user has rules but none match)
        return False, f"No rule grants access to {resource_type.value} '{resource_name}'"

    def get_resource_rules_for_type(self, resource_type: ResourceType) -> List[ResourceRule]:
        """Get all rules for a specific resource type."""
        return [r for r in self.resource_rules if r.resource_type == resource_type]

    @property
    def highest_level(self) -> Optional[UserLevel]:
        """Get the user's highest priority level."""
        if not self.levels:
            return None
        return max(self.levels, key=lambda lvl: lvl.priority)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for API responses.

        Args:
            include_sensitive: If True, include password_hash and other sensitive fields.
        """
        result = {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "display_name": self.display_name or self.username,
            "avatar_url": self.avatar_url,
            "auth_provider": self.auth_provider.value,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "email_verified": self.email_verified,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "levels": [{"id": lvl.id, "name": lvl.name, "priority": lvl.priority} for lvl in self.levels],
            "permissions": sorted(self.effective_permissions),
            "resource_rules": [r.to_dict() for r in self.resource_rules],
            # Organization and team membership
            "org_id": self.org_id,
            "organization": self.organization.to_dict() if self.organization else None,
            "teams": [t.to_dict() for t in self.teams],
        }

        if include_sensitive:
            result["password_hash"] = self.password_hash
            result["external_id"] = self.external_id
            result["metadata"] = self.metadata

        return result


# Default permissions
DEFAULT_PERMISSIONS = [
    # Job permissions
    Permission(1, "job.submit", "Submit training jobs", "jobs"),
    Permission(2, "job.view.own", "View own jobs", "jobs"),
    Permission(3, "job.view.all", "View all users' jobs", "jobs"),
    Permission(4, "job.cancel.own", "Cancel own jobs", "jobs"),
    Permission(5, "job.cancel.all", "Cancel any user's jobs", "jobs"),
    Permission(6, "job.delete.own", "Delete own job history", "jobs"),
    Permission(7, "job.delete.all", "Delete any job history", "jobs"),
    Permission(8, "job.priority.high", "Submit high priority jobs", "jobs"),
    Permission(9, "job.bypass.queue", "Skip the job queue", "jobs"),
    Permission(10, "job.bypass.approval", "Skip approval workflow", "jobs"),
    Permission(11, "job.bypass.quota", "Bypass quota limits", "jobs"),
    # Config permissions
    Permission(20, "config.view", "View training configurations", "config"),
    Permission(21, "config.create", "Create new configurations", "config"),
    Permission(22, "config.edit.own", "Edit own configurations", "config"),
    Permission(23, "config.edit.all", "Edit any configuration", "config"),
    Permission(24, "config.delete.own", "Delete own configurations", "config"),
    Permission(25, "config.delete.all", "Delete any configuration", "config"),
    # Queue permissions
    Permission(30, "queue.view", "View queue status", "queue"),
    Permission(31, "queue.view.all", "View all users' queue entries", "queue"),
    Permission(32, "queue.cancel.own", "Cancel own queued jobs", "queue"),
    Permission(33, "queue.cancel.all", "Cancel any queued job", "queue"),
    Permission(34, "queue.approve", "Approve/reject blocked jobs", "queue"),
    Permission(35, "queue.manage", "Manage queue settings", "queue"),
    # Quota permissions
    Permission(40, "quota.view", "View quota settings", "quota"),
    Permission(41, "quota.manage", "Create/edit/delete quotas", "quota"),
    # Admin permissions
    Permission(50, "admin.users", "Manage users", "admin"),
    Permission(51, "admin.levels", "Manage roles/levels", "admin"),
    Permission(52, "admin.quotas", "Manage quotas", "admin"),
    Permission(53, "admin.approve", "Approve pending jobs", "admin"),
    Permission(54, "admin.config", "Change provider/system config", "admin"),
    Permission(55, "admin.audit", "View audit logs", "admin"),
    Permission(56, "admin.queue", "Manage job queue", "admin"),
    # Organization permissions
    Permission(60, "org.view", "View organization details", "org"),
    Permission(61, "org.create", "Create organizations", "org"),
    Permission(62, "org.edit", "Edit organization settings", "org"),
    Permission(63, "org.delete", "Delete organizations", "org"),
    Permission(64, "org.manage.members", "Manage organization members", "org"),
    Permission(65, "org.manage.quotas", "Set organization quota ceilings", "org"),
    # Team permissions
    Permission(80, "team.view", "View team details", "team"),
    Permission(81, "team.create", "Create teams in organization", "team"),
    Permission(82, "team.edit", "Edit team settings", "team"),
    Permission(83, "team.delete", "Delete teams", "team"),
    Permission(84, "team.manage.members", "Manage team members", "team"),
    Permission(85, "team.manage.quotas", "Set team quota limits", "team"),
    # API permissions
    Permission(70, "api.access", "Access API with API keys", "api"),
    Permission(71, "api.keys.own", "Manage own API keys", "api"),
    Permission(72, "api.keys.all", "Manage all API keys", "api"),
]

# Default levels with their permissions
DEFAULT_LEVELS = [
    UserLevel(
        id=1,
        name="viewer",
        description="Can view own jobs only",
        priority=0,
        is_system=True,
        permissions={"job.view.own", "config.view"},
    ),
    UserLevel(
        id=2,
        name="researcher",
        description="Can submit and manage own jobs",
        priority=10,
        is_system=True,
        permissions={
            "job.submit",
            "job.view.own",
            "job.cancel.own",
            "job.delete.own",
            "config.view",
            "config.create",
            "config.edit.own",
            "config.delete.own",
            "queue.view",
            "queue.cancel.own",
            "api.access",
            "api.keys.own",
        },
    ),
    UserLevel(
        id=3,
        name="lead",
        description="Can view all jobs, approve requests, and manage teams",
        priority=50,
        is_system=True,
        permissions={
            "job.submit",
            "job.view.own",
            "job.view.all",
            "job.cancel.own",
            "job.priority.high",
            "job.delete.own",
            "config.view",
            "config.create",
            "config.edit.own",
            "config.delete.own",
            "queue.view",
            "queue.view.all",
            "queue.cancel.own",
            "queue.approve",
            "quota.view",
            "admin.approve",
            "admin.audit",
            "api.access",
            "api.keys.own",
            # Organization and team management
            "org.view",
            "team.view",
            "team.create",
            "team.edit",
            "team.manage.members",
            "team.manage.quotas",
        },
    ),
    UserLevel(
        id=4,
        name="admin",
        description="Full administrative access",
        priority=100,
        is_system=True,
        permissions={"*"},  # All permissions
    ),
]
