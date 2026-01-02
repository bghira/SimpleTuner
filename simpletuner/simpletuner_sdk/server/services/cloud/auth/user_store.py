"""User storage facade composing specialized stores.

This module provides the UserStore class which serves as a unified
interface to all user-related storage operations. It composes specialized
stores for different concerns:

- UserCrudStore: Core user CRUD and authentication
- PermissionStore: Levels, permissions, overrides, resource rules
- SessionStore: Web UI session management
- OAuthStateStore: OAuth/OIDC CSRF protection
- APIKeyStore: API key management
- QuotaStore: Quota management at all scopes
- OrgStore: Organization CRUD
- TeamStore: Team CRUD and membership
- CredentialStore: Per-user provider credentials (API keys for external services)

For simple use cases, you can use the individual stores directly.
UserStore provides a unified API for code that needs multiple concerns.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    APIKey,
    AuthProvider,
    Organization,
    Permission,
    ResourceRule,
    ResourceType,
    RuleAction,
    Team,
    User,
    UserLevel,
)
from .stores import (
    APIKeyStore,
    CredentialStore,
    OAuthStateStore,
    OrgStore,
    PermissionStore,
    QuotaStore,
    SessionStore,
    TeamStore,
    UserCrudStore,
    get_default_db_path,
)

logger = logging.getLogger(__name__)

# Schema version for user tables
USER_SCHEMA_VERSION = 7


class UserStore:
    """Facade for user storage operations.

    Composes specialized stores to provide a unified API for:
    - User CRUD and authentication
    - Levels and permissions
    - Sessions and OAuth state
    - API keys
    - Quotas
    - Organizations and teams
    - Provider credentials

    Use the singleton pattern via direct instantiation.
    """

    _instance: Optional["UserStore"] = None
    _init_lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "UserStore":
        """Singleton pattern."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._pending_db_path = db_path
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        pending = getattr(self, "_pending_db_path", None)
        if pending is not None:
            db_path = pending
            self._pending_db_path = None

        if db_path is None:
            db_path = self._resolve_db_path()

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        # Initialize component stores with shared db path
        self._users = UserCrudStore(self._db_path)
        self._permissions = PermissionStore(self._db_path)
        self._sessions = SessionStore(self._db_path)
        self._oauth = OAuthStateStore(self._db_path)
        self._api_keys = APIKeyStore(self._db_path)
        self._quotas = QuotaStore(self._db_path)
        self._orgs = OrgStore(self._db_path)
        self._teams = TeamStore(self._db_path)
        self._credentials = CredentialStore(self._db_path)

        # Initialize database schema
        self._init_db()
        self._initialized = True

    def _resolve_db_path(self) -> Path:
        """Resolve database path from AsyncJobStore or default."""
        try:
            from ..async_job_store import AsyncJobStore

            if AsyncJobStore._instance is not None:
                return AsyncJobStore._instance._db_path
        except Exception:
            pass
        return get_default_db_path()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with proper settings."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level="IMMEDIATE",
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema using component stores."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Check if users table exists (for legacy migration detection)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            users_table_exists = cursor.fetchone() is not None

            # Schema version tracking (must exist before migrations)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_schema_version (
                    version INTEGER PRIMARY KEY
                )
            """
            )

            # Check current version
            cursor.execute("SELECT version FROM user_schema_version LIMIT 1")
            row = cursor.fetchone()

            # If users table exists but no version row, this is a legacy database
            # Assume version 4 (before org_id was added)
            if users_table_exists and row is None:
                current_version = 4
                logger.info("Legacy user database detected, setting version to 4")
            else:
                current_version = row["version"] if row else 0

            # Run migrations before initializing schemas (to add missing columns)
            if current_version > 0 and current_version < USER_SCHEMA_VERSION:
                self._run_migrations(cursor, current_version, USER_SCHEMA_VERSION)

            # Safety check: ensure required columns exist even if version is already current
            # (handles case where version was set without column being added)
            if users_table_exists:
                self._ensure_org_id_column(cursor)
            self._ensure_provider_credentials_columns(cursor)

            # Initialize schemas from component stores
            self._users._init_schema(cursor)
            self._permissions._init_schema(cursor)
            self._sessions._init_schema(cursor)
            self._oauth._init_schema(cursor)
            self._api_keys._init_schema(cursor)
            self._quotas._init_schema(cursor)
            self._orgs._init_schema(cursor)
            self._teams._init_schema(cursor)
            self._credentials._init_schema(cursor)

            # Update or insert schema version
            if row is None:
                cursor.execute(
                    "INSERT INTO user_schema_version (version) VALUES (?)",
                    (USER_SCHEMA_VERSION,),
                )
                # Seed default data on first init
                self._permissions.seed_defaults(cursor)
            elif current_version < USER_SCHEMA_VERSION:
                cursor.execute(
                    "UPDATE user_schema_version SET version = ?",
                    (USER_SCHEMA_VERSION,),
                )

            conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize user database: %s", exc, exc_info=True)
            raise
        finally:
            conn.close()

    def _run_migrations(self, cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations for user tables."""
        logger.info("Running user schema migrations from v%d to v%d", from_version, to_version)

        # v4 -> v5: Add org_id column to users table
        if from_version < 5 <= to_version:
            self._ensure_org_id_column(cursor)

        # v6 -> v7: Ensure provider_credentials has all required columns
        if from_version < 7 <= to_version:
            self._ensure_provider_credentials_columns(cursor)

    def _ensure_org_id_column(self, cursor) -> None:
        """Ensure org_id column exists in users table.

        Called both from migrations and as a safety check before _init_schema
        in case the version was incorrectly set without the column being added.
        """
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        if "org_id" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN org_id INTEGER")
            logger.info("Added org_id column to users table")

    def _ensure_provider_credentials_columns(self, cursor) -> None:
        """Ensure provider_credentials table has all required columns.

        Called from migrations to add missing columns to existing tables.
        """
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='provider_credentials'")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(provider_credentials)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        required_columns = {
            "credential_value": "TEXT NOT NULL DEFAULT ''",
            "created_at": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT",
            "last_used_at": "TEXT",
            "is_active": "INTEGER NOT NULL DEFAULT 1",
        }

        for column, definition in required_columns.items():
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE provider_credentials ADD COLUMN {column} {definition}")
                logger.info("Added %s column to provider_credentials table", column)

    # ==================== User Operations ====================

    async def create_user(
        self,
        email: str,
        username: str,
        password: Optional[str] = None,
        display_name: Optional[str] = None,
        auth_provider: AuthProvider = AuthProvider.LOCAL,
        external_id: Optional[str] = None,
        is_admin: bool = False,
        level_names: Optional[List[str]] = None,
    ) -> User:
        """Create a new user with optional level assignments."""
        user = await self._users.create(
            email=email,
            username=username,
            password=password,
            display_name=display_name,
            auth_provider=auth_provider,
            external_id=external_id,
            is_admin=is_admin,
        )

        # Assign levels
        levels_to_assign = level_names if level_names else ["researcher"]
        for level_name in levels_to_assign:
            await self._permissions.assign_level(user.id, level_name)

        return user

    async def get_user(self, user_id: int, include_permissions: bool = True) -> Optional[User]:
        """Get a user by ID, optionally loading permissions."""
        user = await self._users.get(user_id)
        if user and include_permissions:
            user.permissions = await self._permissions.get_user_permissions(user_id, user.is_admin)
            user.levels = await self._permissions.get_user_levels(user_id)
        return user

    async def get_user_by_email(self, email: str, include_permissions: bool = True) -> Optional[User]:
        """Get a user by email."""
        user = await self._users.get_by_email(email)
        if user and include_permissions:
            user.permissions = await self._permissions.get_user_permissions(user.id, user.is_admin)
            user.levels = await self._permissions.get_user_levels(user.id)
        return user

    async def get_user_by_username(self, username: str, include_permissions: bool = True) -> Optional[User]:
        """Get a user by username."""
        user = await self._users.get_by_username(username)
        if user and include_permissions:
            user.permissions = await self._permissions.get_user_permissions(user.id, user.is_admin)
            user.levels = await self._permissions.get_user_levels(user.id)
        return user

    async def get_user_by_external_id(self, external_id: str, provider: AuthProvider) -> Optional[User]:
        """Get a user by external ID and provider."""
        return await self._users.get_by_external_id(external_id, provider)

    async def authenticate_local(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate a user with username/email and password."""
        user = await self._users.authenticate_local(username_or_email, password)
        if user:
            user.permissions = await self._permissions.get_user_permissions(user.id, user.is_admin)
            user.levels = await self._permissions.get_user_levels(user.id)
        return user

    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        include_inactive: bool = False,
        org_id: Optional[int] = None,
    ) -> List[User]:
        """List all users with pagination."""
        users = await self._users.list(limit, offset, include_inactive, org_id)
        for user in users:
            user.permissions = await self._permissions.get_user_permissions(user.id, user.is_admin)
            user.levels = await self._permissions.get_user_levels(user.id)
        return users

    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update a user's fields."""
        return await self._users.update(user_id, updates)

    async def update_last_login(self, user_id: int) -> bool:
        """Update a user's last login timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        return await self._users.update(user_id, {"last_login_at": now})

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user and invalidate all their sessions."""
        await self._sessions.delete_user_sessions(user_id)
        return await self._users.delete(user_id)

    async def get_user_count(self) -> int:
        """Get total user count."""
        return await self._users.get_count()

    async def has_any_users(self) -> bool:
        """Check if any users exist."""
        return await self._users.has_any_users()

    async def needs_first_admin_setup(self) -> bool:
        """Check if first-admin setup is needed.

        Returns True if:
        1. No users exist, OR
        2. Only the auto-created 'local' user exists

        This allows first-admin setup to work even after the middleware
        has auto-created the local placeholder user.
        """
        users = await self.list_users(limit=2)
        if not users:
            return True
        if len(users) == 1 and users[0].username == "local":
            return True
        return False

    # ==================== API Key Operations ====================

    async def create_api_key(
        self,
        user_id: int,
        name: str,
        expires_in_days: Optional[int] = None,
        scoped_permissions: Optional[Set[str]] = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key."""
        return await self._api_keys.create(user_id, name, expires_in_days, scoped_permissions)

    async def authenticate_api_key(self, raw_key: str) -> Optional[Tuple[User, APIKey]]:
        """Authenticate using an API key.

        Returns:
            Tuple of (User, APIKey) if valid, None otherwise
        """
        result = await self._api_keys.authenticate(raw_key)
        if not result:
            return None

        # Fetch full User and APIKey objects
        user = await self.get_user(result["user_id"])
        if not user:
            return None

        api_key = await self._api_keys.get(result["api_key_id"])
        if not api_key:
            return None

        return user, api_key

    async def list_api_keys(self, user_id: int) -> List[APIKey]:
        """List all API keys for a user."""
        return await self._api_keys.list_for_user(user_id)

    async def revoke_api_key(self, key_id: int, user_id: Optional[int] = None) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key ID to revoke
            user_id: User ID (for authorization check). If None, revokes any key (admin mode).
        """
        return await self._api_keys.revoke(key_id, user_id)

    async def delete_api_key(self, key_id: int, user_id: int) -> bool:
        """Delete an API key permanently.

        Args:
            key_id: Key ID to delete
            user_id: User ID (for authorization check)
        """
        return await self._api_keys.delete(key_id, user_id)

    async def cleanup_expired_api_keys(self) -> int:
        """Remove expired API keys."""
        return await self._api_keys.cleanup_expired()

    # ==================== Session Operations ====================

    async def create_session(
        self,
        user_id: int,
        duration_hours: int = 24,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Create a new session."""
        return await self._sessions.create(user_id, duration_hours, ip_address, user_agent)

    async def get_session_user(self, session_id: str) -> Optional[User]:
        """Get the user for a valid session."""
        user_id = await self._sessions.get_user_id(session_id)
        if user_id:
            return await self.get_user(user_id)
        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return await self._sessions.delete(session_id)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        return await self._sessions.cleanup_expired()

    # ==================== OAuth State Operations ====================

    async def create_oauth_state(
        self,
        provider: str,
        redirect_uri: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new OAuth state."""
        return await self._oauth.create(provider, redirect_uri, metadata)

    async def get_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Get OAuth state data."""
        return await self._oauth.get(state)

    async def consume_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Consume (validate and delete) an OAuth state."""
        return await self._oauth.consume(state)

    async def cleanup_expired_oauth_states(self) -> int:
        """Remove expired OAuth states."""
        return await self._oauth.cleanup_expired()

    # ==================== Level/Permission Operations ====================

    async def get_levels(self) -> List[UserLevel]:
        """Get all defined levels."""
        return await self._permissions.get_all_levels()

    async def get_all_levels(self) -> List[UserLevel]:
        """Get all defined levels (alias for get_levels)."""
        return await self._permissions.get_all_levels()

    async def get_level(self, level_id: int) -> Optional[UserLevel]:
        """Get a level by ID."""
        return await self._permissions.get_level(level_id)

    async def get_level_by_name(self, name: str) -> Optional[UserLevel]:
        """Get a level by name."""
        return await self._permissions.get_level_by_name(name)

    async def create_level(
        self, name: str, description: str, priority: int = 0, permissions: Optional[Set[str]] = None
    ) -> UserLevel:
        """Create a new level."""
        return await self._permissions.create_level(name, description, priority, permissions)

    async def update_level(self, level_id: int, updates: Dict[str, Any]) -> bool:
        """Update a level."""
        return await self._permissions.update_level(level_id, updates)

    async def delete_level(self, level_id: int) -> bool:
        """Delete a level."""
        return await self._permissions.delete_level(level_id)

    async def set_level_permissions(self, level_id: int, permission_names: Set[str]) -> None:
        """Set permissions for a level."""
        return await self._permissions.set_level_permissions(level_id, permission_names)

    async def get_permissions(self) -> List[Permission]:
        """Get all defined permissions."""
        return await self._permissions.get_all_permissions()

    async def get_all_permissions(self) -> List[Permission]:
        """Get all defined permissions (alias)."""
        return await self._permissions.get_all_permissions()

    async def get_user_levels(self, user_id: int) -> List[UserLevel]:
        """Get levels assigned to a user."""
        return await self._permissions.get_user_levels(user_id)

    async def assign_level_to_user(self, user_id: int, level_name: str, granted_by: Optional[int] = None) -> bool:
        """Assign a level to a user."""
        return await self._permissions.assign_level(user_id, level_name, granted_by)

    async def assign_level(self, user_id: int, level_name: str, granted_by: Optional[int] = None) -> bool:
        """Assign a level to a user (alias)."""
        return await self._permissions.assign_level(user_id, level_name, granted_by)

    async def remove_level_from_user(self, user_id: int, level_name: str) -> bool:
        """Remove a level from a user."""
        return await self._permissions.remove_level(user_id, level_name)

    async def remove_level(self, user_id: int, level_name: str) -> bool:
        """Remove a level from a user (alias)."""
        return await self._permissions.remove_level(user_id, level_name)

    async def override_user_permission(
        self,
        user_id: int,
        permission_name: str,
        granted: bool = True,
        granted_by: Optional[int] = None,
    ) -> bool:
        """Override a specific permission for a user."""
        return await self._permissions.override_permission(user_id, permission_name, granted, granted_by)

    async def set_permission_override(
        self,
        user_id: int,
        permission_name: str,
        granted: bool = True,
        granted_by: Optional[int] = None,
    ) -> bool:
        """Set a permission override for a user (alias)."""
        return await self._permissions.override_permission(user_id, permission_name, granted, granted_by)

    async def remove_user_permission_override(self, user_id: int, permission_name: str) -> bool:
        """Remove a permission override."""
        return await self._permissions.remove_permission_override(user_id, permission_name)

    async def remove_permission_override(self, user_id: int, permission_name: str) -> bool:
        """Remove a permission override (alias)."""
        return await self._permissions.remove_permission_override(user_id, permission_name)

    async def get_user_resource_rules(self, user_id: int) -> List[ResourceRule]:
        """Get resource rules for a user."""
        return await self._permissions.get_user_resource_rules(user_id)

    async def get_rules_with_level_assignments(self) -> List[Dict[str, Any]]:
        """Get all resource rules with their level assignments."""
        return await self._permissions.get_rules_with_level_assignments()

    async def get_resource_rule(self, rule_id: int) -> Optional[ResourceRule]:
        """Get a resource rule by ID."""
        return await self._permissions.get_resource_rule(rule_id)

    async def create_resource_rule(
        self,
        name: str,
        resource_type: ResourceType,
        pattern: str,
        action: RuleAction = RuleAction.ALLOW,
        priority: int = 0,
        description: Optional[str] = None,
        created_by: Optional[int] = None,
    ) -> int:
        """Create a resource rule."""
        return await self._permissions.create_resource_rule(
            name, resource_type, pattern, action, priority, description, created_by
        )

    async def update_resource_rule(self, rule_id: int, updates: Dict[str, Any]) -> bool:
        """Update a resource rule."""
        return await self._permissions.update_resource_rule(rule_id, updates)

    async def delete_resource_rule(self, rule_id: int) -> bool:
        """Delete a resource rule."""
        return await self._permissions.delete_resource_rule(rule_id)

    async def get_level_rules(self, level_id: int) -> List[ResourceRule]:
        """Get resource rules assigned to a level."""
        return await self._permissions.get_level_rules(level_id)

    async def set_level_rules(self, level_id: int, rule_ids: List[int]) -> bool:
        """Set rules for a level."""
        return await self._permissions.set_level_rules(level_id, rule_ids)

    async def assign_rule_to_level(self, level_id: int, rule_id: int) -> bool:
        """Assign a rule to a level by ID."""
        # Get level name first
        level = await self._permissions.get_level(level_id)
        if not level:
            return False
        return await self._permissions.assign_rule_to_level(level.name, rule_id)

    async def remove_rule_from_level(self, level_id: int, rule_id: int) -> bool:
        """Remove a rule from a level."""
        return await self._permissions.remove_rule_from_level(level_id, rule_id)

    # ==================== Quota Operations ====================

    async def set_quota(
        self,
        quota_type,
        limit_value: float,
        action=None,
        user_id: Optional[int] = None,
        level_id: Optional[int] = None,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
        created_by: Optional[int] = None,
    ) -> int:
        """Set a quota."""
        quota_type_str = quota_type.value if hasattr(quota_type, "value") else str(quota_type)
        action_str = action.value if hasattr(action, "value") else (str(action) if action else "block")
        return await self._quotas.set(
            quota_type_str, limit_value, action_str, user_id, level_id, org_id, team_id, created_by
        )

    async def delete_quota(self, quota_id: int) -> bool:
        """Delete a quota."""
        return await self._quotas.delete(quota_id)

    async def get_user_quotas(self, user_id: int) -> List[Any]:
        """Get quotas specific to a user."""
        return await self._quotas.get_for_user(user_id)

    async def get_level_quotas(self, level_ids: List[int]) -> List[Any]:
        """Get quotas for levels."""
        return await self._quotas.get_for_levels(level_ids)

    async def get_global_quotas(self) -> List[Any]:
        """Get global default quotas."""
        return await self._quotas.get_global()

    async def list_all_quotas(self) -> List[Dict[str, Any]]:
        """List all quotas with scope info."""
        return await self._quotas.list_all()

    async def get_org_quotas(self, org_id: int) -> List[Any]:
        """Get quotas for an organization."""
        return await self._quotas.get_for_org(org_id)

    async def get_team_quotas(self, team_ids: List[int]) -> List[Any]:
        """Get quotas for teams."""
        return await self._quotas.get_for_teams(team_ids)

    async def get_quotas(self, user_id: int) -> Dict[str, Any]:
        """Get effective quota settings for a user as a simple dict.

        Aggregates user, level, and global quotas into a dict with keys like
        'max_concurrent_jobs'. Returns defaults if no quotas are configured.
        """
        result = {
            "max_concurrent_jobs": 5,
            "max_daily_jobs": 100,
            "max_monthly_cost": None,
        }

        try:
            # Get quotas from all scopes
            user_quotas = await self.get_user_quotas(user_id)
            global_quotas = await self.get_global_quotas()

            # Get user's level IDs
            user = await self.get_user(user_id)
            level_quotas = []
            if user and user.level_ids:
                level_quotas = await self.get_level_quotas(list(user.level_ids))

            # Merge quotas - user overrides level overrides global
            all_quotas = global_quotas + level_quotas + user_quotas
            for quota in all_quotas:
                quota_type = getattr(quota, "quota_type", None) or quota.get("quota_type")
                limit_value = getattr(quota, "limit_value", None) or quota.get("limit_value")

                if quota_type == "concurrent_jobs" and limit_value is not None:
                    result["max_concurrent_jobs"] = int(limit_value)
                elif quota_type == "daily_jobs" and limit_value is not None:
                    result["max_daily_jobs"] = int(limit_value)
                elif quota_type == "monthly_cost" and limit_value is not None:
                    result["max_monthly_cost"] = float(limit_value)
        except Exception:
            pass

        return result

    # ==================== Organization Operations ====================

    async def create_organization(
        self,
        name: str,
        slug: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Create a new organization."""
        return await self._orgs.create(name, slug, description, settings)

    async def get_organization(self, org_id: int) -> Optional[Organization]:
        """Get an organization by ID."""
        return await self._orgs.get(org_id)

    async def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get an organization by slug."""
        return await self._orgs.get_by_slug(slug)

    async def list_organizations(self, include_inactive: bool = False) -> List[Organization]:
        """List all organizations."""
        return await self._orgs.list(include_inactive)

    async def update_organization(self, org_id: int, updates: Dict[str, Any]) -> bool:
        """Update an organization."""
        return await self._orgs.update(org_id, updates)

    async def delete_organization(self, org_id: int) -> bool:
        """Delete an organization."""
        return await self._orgs.delete(org_id)

    # ==================== Team Operations ====================

    async def create_team(
        self,
        org_id: int,
        name: str,
        slug: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Team:
        """Create a new team."""
        return await self._teams.create(org_id, name, slug, description, settings)

    async def get_team(self, team_id: int) -> Optional[Team]:
        """Get a team by ID."""
        return await self._teams.get(team_id)

    async def update_team(self, team_id: int, updates: Dict[str, Any]) -> bool:
        """Update a team's fields."""
        return await self._teams.update(team_id, updates)

    async def delete_team(self, team_id: int) -> bool:
        """Delete a team."""
        return await self._teams.delete(team_id)

    async def list_teams(self, org_id: int, include_inactive: bool = False) -> List[Team]:
        """List teams in an organization."""
        return await self._teams.list_for_org(org_id, include_inactive)

    async def get_user_teams(self, user_id: int) -> List[Team]:
        """Get all teams a user belongs to."""
        return await self._teams.get_user_teams(user_id)

    async def add_user_to_team(self, user_id: int, team_id: int, role: str = "member") -> bool:
        """Add a user to a team."""
        return await self._teams.add_user(user_id, team_id, role)

    async def remove_user_from_team(self, user_id: int, team_id: int) -> bool:
        """Remove a user from a team."""
        return await self._teams.remove_user(user_id, team_id)

    async def get_team_members(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all members of a team."""
        return await self._teams.get_members(team_id)

    async def set_user_organization(self, user_id: int, org_id: Optional[int]) -> bool:
        """Set a user's organization membership."""
        return await self._users.set_organization(user_id, org_id)

    # ==================== Provider Credential Operations ====================

    async def set_provider_credential(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
        credential_value: str,
    ) -> int:
        """Set or update a provider credential for a user."""
        return await self._credentials.set(user_id, provider, credential_name, credential_value)

    async def get_provider_credential(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> Optional[str]:
        """Get a provider credential for a user."""
        return await self._credentials.get(user_id, provider, credential_name)

    async def mark_credential_used(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Update the last_used_at timestamp for a credential."""
        return await self._credentials.mark_used(user_id, provider, credential_name)

    async def list_provider_credentials(
        self,
        user_id: int,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List credentials for a user (metadata only, no values)."""
        return await self._credentials.list_for_user(user_id, provider)

    async def delete_provider_credential(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Delete a provider credential."""
        return await self._credentials.delete(user_id, provider, credential_name)

    # ==================== Utility ====================

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._init_lock:
            cls._instance = None


# Backwards compatibility alias
AsyncUserStore = UserStore
