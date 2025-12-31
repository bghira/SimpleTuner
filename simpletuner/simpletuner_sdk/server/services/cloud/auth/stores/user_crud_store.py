"""Core user CRUD and authentication store.

Handles user creation, retrieval, updates, deletion, and local authentication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import AuthProvider, User
from ..password import get_password_hasher
from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class UserCrudStore(BaseAuthStore):
    """Store for core user CRUD operations.

    Handles user creation, retrieval, updates, deletion,
    and local password authentication.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the users table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT,
                avatar_url TEXT,
                auth_provider TEXT NOT NULL DEFAULT 'local',
                password_hash TEXT,
                external_id TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_admin INTEGER NOT NULL DEFAULT 0,
                email_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login_at TEXT,
                org_id INTEGER,
                metadata TEXT DEFAULT '{}'
            )
        """
        )

        # Ensure columns exist for legacy databases (migrations)
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        if "org_id" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN org_id INTEGER")
            logger.info("Added org_id column to users table (in-schema migration)")
        if "avatar_url" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
            logger.info("Added avatar_url column to users table (in-schema migration)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id)")

    async def create(
        self,
        email: str,
        username: str,
        password: Optional[str] = None,
        display_name: Optional[str] = None,
        auth_provider: AuthProvider = AuthProvider.LOCAL,
        external_id: Optional[str] = None,
        is_admin: bool = False,
        org_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user.

        Args:
            email: User's email address (unique)
            username: Username for login (unique)
            password: Password for local auth (required if auth_provider is LOCAL)
            display_name: Display name (defaults to username)
            auth_provider: Authentication provider (local, oidc, ldap)
            external_id: External provider's user ID (for OIDC/LDAP)
            is_admin: If True, grant full admin access
            org_id: Organization ID to associate user with
            metadata: Additional metadata dict

        Returns:
            The created User object

        Raises:
            ValueError: If email/username already exists or password missing for local auth
        """
        if auth_provider == AuthProvider.LOCAL and not password:
            raise ValueError("Password required for local authentication")

        password_hash = None
        if password:
            hasher = get_password_hasher()
            password_hash = hasher.hash(password)

        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Check for existing email/username
                cursor.execute(
                    "SELECT id FROM users WHERE email = ? OR username = ?",
                    (email, username),
                )
                if cursor.fetchone():
                    raise ValueError(f"User with email '{email}' or username '{username}' already exists")

                cursor.execute(
                    """
                    INSERT INTO users (
                        email, username, display_name, auth_provider,
                        password_hash, external_id, is_active, is_admin,
                        email_verified, created_at, org_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, 0, ?, ?, ?)
                    """,
                    (
                        email,
                        username,
                        display_name or username,
                        auth_provider.value,
                        password_hash,
                        external_id,
                        1 if is_admin else 0,
                        now,
                        org_id,
                        json.dumps(metadata or {}),
                    ),
                )
                user_id = cursor.lastrowid
                conn.commit()

                return User(
                    id=user_id,
                    email=email,
                    username=username,
                    display_name=display_name or username,
                    auth_provider=auth_provider,
                    password_hash=password_hash,
                    external_id=external_id,
                    is_active=True,
                    is_admin=is_admin,
                    created_at=now,
                    org_id=org_id,
                    metadata=metadata or {},
                )
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _create)

    async def get(self, user_id: int) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_user(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email.

        Args:
            email: User email

        Returns:
            User if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_user(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get a user by username.

        Args:
            username: Username

        Returns:
            User if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_user(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_by_external_id(self, external_id: str, provider: AuthProvider) -> Optional[User]:
        """Get a user by external ID and provider.

        Args:
            external_id: External provider's user ID
            provider: Authentication provider

        Returns:
            User if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM users WHERE external_id = ? AND auth_provider = ?",
                    (external_id, provider.value),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_user(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def authenticate_local(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate a user with username/email and password.

        Args:
            username_or_email: Username or email
            password: Password to verify

        Returns:
            User if credentials are valid, None otherwise
        """
        loop = asyncio.get_running_loop()
        hasher = get_password_hasher()

        def _auth():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM users
                    WHERE (email = ? OR username = ?)
                      AND auth_provider = 'local'
                      AND is_active = 1
                    """,
                    (username_or_email, username_or_email),
                )
                row = cursor.fetchone()

                if not row or not row["password_hash"]:
                    return None

                if not hasher.verify(row["password_hash"], password):
                    return None

                user = self._row_to_user(row)

                # Update last login
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user.id))

                # Check if password needs rehashing
                if hasher.needs_rehash(row["password_hash"]):
                    new_hash = hasher.hash(password)
                    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user.id))

                conn.commit()
                user.last_login_at = now
                return user
            finally:
                conn.close()

        return await loop.run_in_executor(None, _auth)

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        include_inactive: bool = False,
        org_id: Optional[int] = None,
    ) -> List[User]:
        """List users with pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            include_inactive: Include inactive users
            org_id: Filter by organization ID

        Returns:
            List of User objects
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE 1=1"
                params: List[Any] = []

                if not include_inactive:
                    query += " AND is_active = 1"
                if org_id is not None:
                    query += " AND org_id = ?"
                    params.append(org_id)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                return [self._row_to_user(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def update(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update a user's fields.

        Args:
            user_id: User ID
            updates: Dict of field -> value to update

        Returns:
            True if updated, False if not found
        """
        if not updates:
            return False

        loop = asyncio.get_running_loop()

        def _update():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Handle password specially
                if "password" in updates:
                    hasher = get_password_hasher()
                    updates["password_hash"] = hasher.hash(updates.pop("password"))

                # Handle metadata specially
                if "metadata" in updates:
                    updates["metadata"] = json.dumps(updates["metadata"])

                set_clauses = []
                values = []
                for key, value in updates.items():
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                values.append(user_id)

                query = f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete(self, user_id: int) -> bool:
        """Delete a user.

        Args:
            user_id: User ID to delete

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def set_organization(self, user_id: int, org_id: Optional[int]) -> bool:
        """Set a user's organization membership.

        Args:
            user_id: User to update
            org_id: Organization ID (or None to remove from org)

        Returns:
            True if updated
        """
        return await self.update(user_id, {"org_id": org_id})

    async def get_count(self, include_inactive: bool = False) -> int:
        """Get total user count.

        Args:
            include_inactive: Include inactive users

        Returns:
            User count
        """
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if include_inactive:
                    cursor.execute("SELECT COUNT(*) FROM users")
                else:
                    cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                return cursor.fetchone()[0]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def has_any_users(self) -> bool:
        """Check if any users exist (for first-run setup).

        Returns:
            True if any users exist
        """
        count = await self.get_count(include_inactive=True)
        return count > 0

    def _row_to_user(self, row) -> User:
        """Convert a database row to a User object."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        org_id = None
        try:
            org_id = row["org_id"]
        except (IndexError, KeyError):
            pass

        avatar_url = None
        try:
            avatar_url = row["avatar_url"]
        except (IndexError, KeyError):
            pass

        return User(
            id=row["id"],
            email=row["email"],
            username=row["username"],
            display_name=row["display_name"],
            avatar_url=avatar_url,
            auth_provider=AuthProvider(row["auth_provider"]),
            password_hash=row["password_hash"],
            external_id=row["external_id"],
            is_active=bool(row["is_active"]),
            is_admin=bool(row["is_admin"]),
            email_verified=bool(row["email_verified"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            org_id=org_id,
            metadata=metadata,
        )


# Singleton accessor
_instance: Optional[UserCrudStore] = None


def get_user_crud_store() -> UserCrudStore:
    """Get the singleton UserCrudStore instance."""
    global _instance
    if _instance is None:
        _instance = UserCrudStore()
    return _instance
