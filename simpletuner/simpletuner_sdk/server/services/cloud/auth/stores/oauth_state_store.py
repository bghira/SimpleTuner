"""OAuth state management store.

Handles CSRF protection for OAuth/OIDC flows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .base import BaseAuthStore

logger = logging.getLogger(__name__)

# Default OAuth state expiration in seconds (10 minutes)
OAUTH_STATE_EXPIRATION_SECONDS = 600


class OAuthStateStore(BaseAuthStore):
    """Store for OAuth state management.

    Handles CSRF protection for OAuth/OIDC authentication flows
    across multiple workers and server restarts.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the oauth_states table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_states_expires ON oauth_states(expires_at)")

    async def create(
        self,
        provider: str,
        redirect_uri: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = OAUTH_STATE_EXPIRATION_SECONDS,
    ) -> str:
        """Create a new OAuth state for CSRF protection.

        Args:
            provider: OAuth provider name (e.g., 'oidc', 'ldap')
            redirect_uri: URI to redirect to after auth
            metadata: Optional additional metadata to store
            ttl_seconds: Time-to-live in seconds

        Returns:
            The generated state token
        """
        state = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)

        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO oauth_states (state, provider, redirect_uri, created_at, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state,
                        provider,
                        redirect_uri,
                        now.isoformat(),
                        expires_at.isoformat(),
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        async with self._lock:
            await loop.run_in_executor(None, _create)

        return state

    async def get(self, state: str) -> Optional[Dict[str, Any]]:
        """Get OAuth state data without consuming it.

        Args:
            state: The state token to look up

        Returns:
            Dict with provider, redirect_uri, metadata, or None if not found/expired
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT provider, redirect_uri, metadata
                    FROM oauth_states
                    WHERE state = ? AND expires_at > ?
                    """,
                    (state, now),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        pass

                return {
                    "provider": row["provider"],
                    "redirect_uri": row["redirect_uri"],
                    "metadata": metadata,
                }
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def consume(self, state: str) -> Optional[Dict[str, Any]]:
        """Consume (validate and delete) an OAuth state.

        This is the primary method for validating OAuth callbacks.
        The state is deleted after retrieval to prevent replay.

        Args:
            state: The state token from the OAuth callback

        Returns:
            Dict with provider, redirect_uri, metadata, or None if invalid/expired
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _consume():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Get the state first
                cursor.execute(
                    """
                    SELECT provider, redirect_uri, metadata
                    FROM oauth_states
                    WHERE state = ? AND expires_at > ?
                    """,
                    (state, now),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                # Delete it to prevent replay
                cursor.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
                conn.commit()

                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        pass

                return {
                    "provider": row["provider"],
                    "redirect_uri": row["redirect_uri"],
                    "metadata": metadata,
                }
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _consume)

    async def cleanup_expired(self) -> int:
        """Remove expired OAuth states.

        Returns:
            Number of states cleaned up
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _cleanup():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM oauth_states WHERE expires_at < ?", (now,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _cleanup)


# Singleton accessor
_instance: Optional[OAuthStateStore] = None


def get_oauth_state_store() -> OAuthStateStore:
    """Get the singleton OAuthStateStore instance."""
    global _instance
    if _instance is None:
        _instance = OAuthStateStore()
    return _instance
