"""Idempotency key storage for job submission deduplication.

Provides storage and lookup of idempotency keys to prevent duplicate
job submissions from network retries or client errors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from .async_base import AsyncSQLiteStore

logger = logging.getLogger(__name__)


class IdempotencyStore(AsyncSQLiteStore):
    """Store for idempotency key tracking.

    Idempotency keys allow clients to safely retry job submission requests.
    If a request with the same key is received within the TTL, the original
    job_id is returned instead of creating a duplicate job.
    """

    async def _init_schema(self) -> None:
        """Initialize the idempotency_keys table."""
        conn = await self._get_connection()

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                idempotency_key TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                user_id INTEGER,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at)")
        await conn.commit()

    async def check(self, key: str, user_id: Optional[int] = None) -> Optional[str]:
        """Check if an idempotency key exists and return the associated job_id.

        Also cleans up expired keys as a side effect.

        Args:
            key: The idempotency key to check
            user_id: Optional user ID for scoping (not currently used for lookup)

        Returns:
            job_id if key exists and is valid, None otherwise
        """
        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            # Clean up expired keys
            await conn.execute("DELETE FROM idempotency_keys WHERE expires_at < ?", (now,))

        # Check for existing key
        row = await self.fetch_one(
            "SELECT job_id FROM idempotency_keys WHERE idempotency_key = ? AND expires_at > ?",
            (key, now),
        )
        return row["job_id"] if row else None

    async def store(
        self,
        key: str,
        job_id: str,
        user_id: Optional[int] = None,
        ttl_hours: int = 24,
    ) -> None:
        """Store an idempotency key for deduplication.

        Args:
            key: The idempotency key
            job_id: The associated job ID
            user_id: Optional user ID for scoping
            ttl_hours: Time-to-live in hours (default 24)
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)

        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO idempotency_keys
                (idempotency_key, job_id, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, job_id, user_id, now.isoformat(), expires_at.isoformat()),
            )

    async def cleanup_expired(self) -> int:
        """Remove expired idempotency keys.

        Returns:
            Number of keys deleted
        """
        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM idempotency_keys WHERE expires_at < ?", (now,))
            return cursor.rowcount


# Singleton access
_instance: Optional[IdempotencyStore] = None


async def get_idempotency_store() -> IdempotencyStore:
    """Get the singleton IdempotencyStore instance."""
    global _instance
    if _instance is None:
        _instance = await IdempotencyStore.get_instance()
    return _instance
