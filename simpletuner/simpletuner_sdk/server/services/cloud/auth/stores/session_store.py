"""Session management store.

Handles web UI sessions for authenticated users.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class SessionStore(BaseAuthStore):
    """Store for session management.

    Handles web UI sessions with secure token generation,
    expiration, and cleanup.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the sessions table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

    async def create(
        self,
        user_id: int,
        duration_hours: int = 24,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Create a new session for a user.

        Args:
            user_id: ID of the authenticated user
            duration_hours: Session duration in hours
            ip_address: Client IP address
            user_agent: Client user agent string

        Returns:
            Session token
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=duration_hours)

        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (id, user_id, created_at, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, user_id, now.isoformat(), expires_at.isoformat(), ip_address, user_agent),
                )
                conn.commit()
            finally:
                conn.close()

        async with self._lock:
            await loop.run_in_executor(None, _create)

        return session_id

    async def get_user_id(self, session_id: str) -> Optional[int]:
        """Get the user ID for a valid session.

        Args:
            session_id: The session token

        Returns:
            User ID if session is valid, None otherwise
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id FROM sessions
                    WHERE id = ? AND expires_at > ?
                    """,
                    (session_id, now),
                )
                row = cursor.fetchone()
                return row["user_id"] if row else None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_expired_session_user_id(self, session_id: str) -> Optional[int]:
        """Get the user ID for an expired session.

        Used to detect when someone tries to use an expired session.
        Does NOT return user_id for valid sessions (use get_user_id for that).

        Args:
            session_id: The session token

        Returns:
            User ID if session exists but is expired, None otherwise
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id FROM sessions
                    WHERE id = ? AND expires_at <= ?
                    """,
                    (session_id, now),
                )
                row = cursor.fetchone()
                return row["user_id"] if row else None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def delete(self, session_id: str) -> bool:
        """Delete a session (logout).

        Args:
            session_id: The session token to delete

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def delete_user_sessions(self, user_id: int) -> int:
        """Delete all sessions for a user.

        Args:
            user_id: User ID to delete sessions for

        Returns:
            Number of sessions deleted
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _cleanup():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _cleanup)

    async def extend(self, session_id: str, additional_hours: int = 24) -> bool:
        """Extend a session's expiration time.

        Args:
            session_id: The session token
            additional_hours: Hours to add to expiration

        Returns:
            True if extended, False if session not found
        """
        now = datetime.now(timezone.utc)
        new_expires = now + timedelta(hours=additional_hours)
        loop = asyncio.get_running_loop()

        def _extend():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE sessions SET expires_at = ?
                    WHERE id = ? AND expires_at > ?
                    """,
                    (new_expires.isoformat(), session_id, now.isoformat()),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _extend)


# Singleton accessor
_instance: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get the singleton SessionStore instance."""
    global _instance
    if _instance is None:
        _instance = SessionStore()
    return _instance
