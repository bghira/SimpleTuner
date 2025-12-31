"""Provider credential store.

Handles storage and retrieval of per-user provider credentials
(e.g., Replicate API tokens, HuggingFace tokens).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class CredentialStore(BaseAuthStore):
    """Store for per-user provider credentials.

    Allows users to store their own API keys for cloud providers,
    overriding global credentials when present.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the provider_credentials table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                credential_name TEXT NOT NULL,
                credential_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                last_used_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, provider, credential_name)
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider_creds_user ON provider_credentials(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_provider_creds_provider ON provider_credentials(provider)")

    async def set(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
        credential_value: str,
    ) -> int:
        """Set or update a provider credential for a user.

        Args:
            user_id: User to set credential for
            provider: Provider name (e.g., "replicate", "huggingface")
            credential_name: Credential name (e.g., "api_token")
            credential_value: The credential value

        Returns:
            Credential ID
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _set():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO provider_credentials
                    (user_id, provider, credential_name, credential_value, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(user_id, provider, credential_name) DO UPDATE SET
                        credential_value = excluded.credential_value,
                        updated_at = ?,
                        is_active = 1
                    """,
                    (user_id, provider, credential_name, credential_value, now, now),
                )
                cred_id = cursor.lastrowid
                conn.commit()
                return cred_id
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _set)

    async def get(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> Optional[str]:
        """Get a provider credential for a user.

        Args:
            user_id: User ID
            provider: Provider name
            credential_name: Credential name

        Returns:
            Credential value if found and active, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT credential_value
                    FROM provider_credentials
                    WHERE user_id = ? AND provider = ? AND credential_name = ? AND is_active = 1
                    """,
                    (user_id, provider, credential_name),
                )
                row = cursor.fetchone()
                return row["credential_value"] if row else None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def mark_used(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Update the last_used_at timestamp for a credential.

        Args:
            user_id: User ID
            provider: Provider name
            credential_name: Credential name

        Returns:
            True if updated, False if not found
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _mark():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE provider_credentials
                    SET last_used_at = ?
                    WHERE user_id = ? AND provider = ? AND credential_name = ? AND is_active = 1
                    """,
                    (now, user_id, provider, credential_name),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        return await loop.run_in_executor(None, _mark)

    async def list_for_user(
        self,
        user_id: int,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List credentials for a user.

        Args:
            user_id: User ID
            provider: Optional provider filter

        Returns:
            List of credential metadata (without values)
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if provider:
                    cursor.execute(
                        """
                        SELECT id, provider, credential_name, created_at, updated_at, last_used_at, is_active
                        FROM provider_credentials
                        WHERE user_id = ? AND provider = ?
                        ORDER BY provider, credential_name
                        """,
                        (user_id, provider),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, provider, credential_name, created_at, updated_at, last_used_at, is_active
                        FROM provider_credentials
                        WHERE user_id = ?
                        ORDER BY provider, credential_name
                        """,
                        (user_id,),
                    )

                return [
                    {
                        "id": row["id"],
                        "provider": row["provider"],
                        "credential_name": row["credential_name"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "last_used_at": row["last_used_at"],
                        "is_active": bool(row["is_active"]),
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def delete(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Delete a provider credential.

        Args:
            user_id: User ID
            provider: Provider name
            credential_name: Credential name

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM provider_credentials
                    WHERE user_id = ? AND provider = ? AND credential_name = ?
                    """,
                    (user_id, provider, credential_name),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def deactivate(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Deactivate a credential without deleting it.

        Args:
            user_id: User ID
            provider: Provider name
            credential_name: Credential name

        Returns:
            True if deactivated, False if not found
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _deactivate():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE provider_credentials
                    SET is_active = 0, updated_at = ?
                    WHERE user_id = ? AND provider = ? AND credential_name = ?
                    """,
                    (now, user_id, provider, credential_name),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _deactivate)


# Singleton accessor
_instance: Optional[CredentialStore] = None


def get_credential_store() -> CredentialStore:
    """Get the singleton CredentialStore instance."""
    global _instance
    if _instance is None:
        _instance = CredentialStore()
    return _instance
