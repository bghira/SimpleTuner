"""Authentication and authorization module for cloud training.

Provides user management, API key authentication, session handling,
RBAC, and quota management.
"""

from .middleware import (
    AuthMiddleware,
    get_current_user,
    get_current_user_ws,
    get_optional_user,
    require_any_permission,
    require_permission,
)
from .models import APIKey, Permission, ResourceRule, ResourceType, RuleAction, User, UserLevel
from .password import PasswordHasher
from .quotas import Quota, QuotaAction, QuotaChecker, QuotaStatus, QuotaType
from .user_store import UserStore

# Backwards compatibility alias - UserStore already has async methods
AsyncUserStore = UserStore

__all__ = [
    # Models
    "User",
    "APIKey",
    "Permission",
    "UserLevel",
    "ResourceRule",
    "ResourceType",
    "RuleAction",
    # Store
    "UserStore",
    "AsyncUserStore",
    # Password
    "PasswordHasher",
    # Middleware
    "AuthMiddleware",
    "get_current_user",
    "get_current_user_ws",
    "get_optional_user",
    "require_permission",
    "require_any_permission",
    # Quotas
    "Quota",
    "QuotaAction",
    "QuotaChecker",
    "QuotaStatus",
    "QuotaType",
]
