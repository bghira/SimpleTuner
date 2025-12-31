"""Job slot reservation storage for atomic quota enforcement.

Provides reservation-based locking to prevent race conditions where
multiple concurrent requests pass quota checks before any job is created.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from .async_base import AsyncSQLiteStore

logger = logging.getLogger(__name__)


class ReservationStore(AsyncSQLiteStore):
    """Store for job slot reservations.

    Reservations provide a two-phase approach to quota enforcement:
    1. reserve_slot() - Atomically check quota and reserve a slot
    2. consume() or release() - Mark the reservation as used or cancelled

    This prevents the race condition where two requests both pass the quota
    check before either creates a job.
    """

    async def _init_schema(self) -> None:
        """Initialize the job_reservations table."""
        conn = await self._get_connection()

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_reservations (
                reservation_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                consumed INTEGER DEFAULT 0
            )
        """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_reservations_user ON job_reservations(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_reservations_expires ON job_reservations(expires_at)")
        await conn.commit()

    async def reserve_slot(
        self,
        user_id: int,
        max_concurrent: int,
        ttl_seconds: int = 300,
    ) -> Optional[str]:
        """Atomically reserve a job slot if quota allows.

        This counts both active jobs and active reservations to determine
        if the user is within their quota.

        Args:
            user_id: User requesting the slot
            max_concurrent: Maximum concurrent jobs allowed for user
            ttl_seconds: Reservation TTL in seconds (default 5 minutes)

        Returns:
            reservation_id if slot reserved, None if quota exceeded
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)

        async with self.transaction() as conn:
            # Clean up expired reservations first
            await conn.execute(
                "DELETE FROM job_reservations WHERE expires_at < ?",
                (now.isoformat(),),
            )

            # Count active jobs for this user
            # Note: This requires the jobs table to exist in the same DB
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE user_id = ? AND status IN ('pending', 'uploading', 'queued', 'running')
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            active_jobs = row[0] if row else 0

            # Count active (unconsumed) reservations
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM job_reservations
                WHERE user_id = ? AND expires_at > ? AND consumed = 0
                """,
                (user_id, now.isoformat()),
            )
            row = await cursor.fetchone()
            active_reservations = row[0] if row else 0

            total_active = active_jobs + active_reservations

            if total_active >= max_concurrent:
                logger.debug(
                    "Quota exceeded for user %d: %d active jobs + %d reservations >= %d max",
                    user_id,
                    active_jobs,
                    active_reservations,
                    max_concurrent,
                )
                return None

            # Create reservation
            reservation_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO job_reservations
                (reservation_id, user_id, created_at, expires_at, consumed)
                VALUES (?, ?, ?, ?, 0)
                """,
                (reservation_id, user_id, now.isoformat(), expires_at.isoformat()),
            )

            logger.debug(
                "Reserved slot %s for user %d (expires %s)",
                reservation_id,
                user_id,
                expires_at.isoformat(),
            )
            return reservation_id

    async def consume(self, reservation_id: str) -> bool:
        """Mark a reservation as consumed (job was created successfully).

        Args:
            reservation_id: The reservation to consume

        Returns:
            True if consumed, False if not found or already consumed
        """
        async with self.transaction() as conn:
            cursor = await conn.execute(
                """
                UPDATE job_reservations
                SET consumed = 1
                WHERE reservation_id = ? AND consumed = 0
                """,
                (reservation_id,),
            )
            success = cursor.rowcount > 0

        if success:
            logger.debug("Consumed reservation %s", reservation_id)
        else:
            logger.warning("Failed to consume reservation %s (not found or already consumed)", reservation_id)

        return success

    async def release(self, reservation_id: str) -> bool:
        """Release a reservation (job submission failed).

        This removes the reservation entirely, freeing the slot.

        Args:
            reservation_id: The reservation to release

        Returns:
            True if released, False if not found
        """
        async with self.transaction() as conn:
            cursor = await conn.execute(
                "DELETE FROM job_reservations WHERE reservation_id = ?",
                (reservation_id,),
            )
            success = cursor.rowcount > 0

        if success:
            logger.debug("Released reservation %s", reservation_id)

        return success

    async def cleanup_expired(self) -> int:
        """Remove expired reservations.

        Returns:
            Number of reservations deleted
        """
        now = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM job_reservations WHERE expires_at < ?", (now,))
            return cursor.rowcount

    async def get_active_count(self, user_id: int) -> int:
        """Get the count of active (unconsumed, unexpired) reservations for a user.

        Args:
            user_id: User to check

        Returns:
            Number of active reservations
        """
        now = datetime.now(timezone.utc).isoformat()
        row = await self.fetch_one(
            """
            SELECT COUNT(*) FROM job_reservations
            WHERE user_id = ? AND expires_at > ? AND consumed = 0
            """,
            (user_id, now),
        )
        return row[0] if row else 0


# Singleton access
_instance: Optional[ReservationStore] = None


async def get_reservation_store() -> ReservationStore:
    """Get the singleton ReservationStore instance."""
    global _instance
    if _instance is None:
        _instance = await ReservationStore.get_instance()
    return _instance
