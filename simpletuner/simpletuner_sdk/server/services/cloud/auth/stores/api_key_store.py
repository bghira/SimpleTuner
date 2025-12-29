"""API key management store.

Handles creation, authentication, and management of API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from ..models import APIKey
from ..password import get_api_key_generator, get_password_hasher
from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class APIKeyStore(BaseAuthStore):
    """Store for API key management.

    Handles secure creation, authentication, and lifecycle
    management of API keys for programmatic access.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the api_keys table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                expires_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                scoped_permissions TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)")

    async def create(
        self,
        user_id: int,
        name: str,
        expires_in_days: Optional[int] = None,
        scoped_permissions: Optional[Set[str]] = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key for a user.

        Args:
            user_id: User to create key for
            name: Human-readable name for the key
            expires_in_days: Days until expiration (None for no expiry)
            scoped_permissions: Optional set of permission names to scope key to

        Returns:
            Tuple of (APIKey metadata object, raw key string)
            The raw key is only returned once and cannot be retrieved later.
        """
        generator = get_api_key_generator()
        hasher = get_password_hasher()

        raw_key = generator.generate()
        key_prefix = raw_key[:8]
        key_hash = hasher.hash(raw_key)

        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            expires_at = now + timedelta(days=expires_in_days)

        loop = asyncio.get_running_loop()

        scoped_json = json.dumps(list(scoped_permissions)) if scoped_permissions else None

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO api_keys
                    (user_id, name, key_prefix, key_hash, created_at, expires_at, is_active, scoped_permissions)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        user_id,
                        name,
                        key_prefix,
                        key_hash,
                        now.isoformat(),
                        expires_at.isoformat() if expires_at else None,
                        scoped_json,
                    ),
                )
                key_id = cursor.lastrowid
                conn.commit()
                return key_id
            finally:
                conn.close()

        async with self._lock:
            key_id = await loop.run_in_executor(None, _create)

        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name,
            key_prefix=key_prefix,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat() if expires_at else None,
            is_active=True,
            scoped_permissions=scoped_permissions,
        )

        return api_key, raw_key

    async def authenticate(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate using an API key.

        Args:
            raw_key: The raw API key string

        Returns:
            Dict with user_id and scoped_permissions if valid, None otherwise
        """
        hasher = get_password_hasher()
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _auth():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Get all active, non-expired keys
                cursor.execute(
                    """
                    SELECT id, user_id, key_hash, scoped_permissions
                    FROM api_keys
                    WHERE is_active = 1
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (now,),
                )

                for row in cursor.fetchall():
                    if hasher.verify(row["key_hash"], raw_key):
                        # Update last used timestamp
                        cursor.execute(
                            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                            (now, row["id"]),
                        )
                        conn.commit()

                        scoped = None
                        if row["scoped_permissions"]:
                            try:
                                scoped = set(json.loads(row["scoped_permissions"]))
                            except json.JSONDecodeError:
                                pass

                        return {
                            "user_id": row["user_id"],
                            "api_key_id": row["id"],
                            "scoped_permissions": scoped,
                        }
                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _auth)

    async def list_for_user(self, user_id: int) -> List[APIKey]:
        """List all API keys for a user.

        Args:
            user_id: User to list keys for

        Returns:
            List of APIKey objects (without hash)
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, user_id, name, key_prefix, created_at,
                           last_used_at, expires_at, is_active, scoped_permissions
                    FROM api_keys
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )

                keys = []
                for row in cursor.fetchall():
                    scoped = None
                    if row["scoped_permissions"]:
                        try:
                            scoped = set(json.loads(row["scoped_permissions"]))
                        except json.JSONDecodeError:
                            pass

                    keys.append(
                        APIKey(
                            id=row["id"],
                            user_id=row["user_id"],
                            name=row["name"],
                            key_prefix=row["key_prefix"],
                            created_at=row["created_at"],
                            last_used_at=row["last_used_at"],
                            expires_at=row["expires_at"],
                            is_active=bool(row["is_active"]),
                            scoped_permissions=scoped,
                        )
                    )
                return keys
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def revoke(self, key_id: int, user_id: int) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key ID to revoke
            user_id: User ID (for authorization check)

        Returns:
            True if revoked, False if not found or not owned by user
        """
        loop = asyncio.get_running_loop()

        def _revoke():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE api_keys SET is_active = 0
                    WHERE id = ? AND user_id = ?
                    """,
                    (key_id, user_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _revoke)

    async def delete(self, key_id: int, user_id: int) -> bool:
        """Delete an API key permanently.

        Args:
            key_id: Key ID to delete
            user_id: User ID (for authorization check)

        Returns:
            True if deleted, False if not found or not owned by user
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM api_keys WHERE id = ? AND user_id = ?",
                    (key_id, user_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def cleanup_expired(self) -> int:
        """Remove expired API keys.

        Returns:
            Number of keys cleaned up
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _cleanup():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM api_keys WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _cleanup)


# Singleton accessor
_instance: Optional[APIKeyStore] = None


def get_api_key_store() -> APIKeyStore:
    """Get the singleton APIKeyStore instance."""
    global _instance
    if _instance is None:
        _instance = APIKeyStore()
    return _instance
