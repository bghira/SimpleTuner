"""Quota management store.

Handles quota storage and retrieval at user, level, org, team, and global scopes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class QuotaStore(BaseAuthStore):
    """Store for quota management.

    Handles CRUD operations for quotas at various scopes:
    - User: Specific to one user
    - Level: Applies to all users with a level
    - Organization: Ceiling for an org
    - Team: Ceiling for a team
    - Global: Default fallback
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the quotas table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quotas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quota_type TEXT NOT NULL,
                limit_value REAL NOT NULL,
                action TEXT NOT NULL DEFAULT 'block',
                user_id INTEGER,
                level_id INTEGER,
                org_id INTEGER,
                team_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (level_id) REFERENCES levels(id) ON DELETE CASCADE,
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
            )
        """
        )

        # Migrate legacy quotas table: add org_id and team_id columns if missing
        cursor.execute("PRAGMA table_info(quotas)")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        if "org_id" not in existing_columns:
            cursor.execute("ALTER TABLE quotas ADD COLUMN org_id INTEGER")
            logger.info("Added org_id column to quotas table")
        if "team_id" not in existing_columns:
            cursor.execute("ALTER TABLE quotas ADD COLUMN team_id INTEGER")
            logger.info("Added team_id column to quotas table")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotas_user ON quotas(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotas_level ON quotas(level_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotas_org ON quotas(org_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotas_team ON quotas(team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotas_type ON quotas(quota_type)")

    async def set(
        self,
        quota_type: str,
        limit_value: float,
        action: str = "block",
        user_id: Optional[int] = None,
        level_id: Optional[int] = None,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
        created_by: Optional[int] = None,
    ) -> int:
        """Set a quota (upsert).

        Args:
            quota_type: Type of quota (cost_monthly, concurrent_jobs, etc.)
            limit_value: The limit value
            action: Action when exceeded (block, warn, require_approval)
            user_id: If set, applies to specific user
            level_id: If set, applies to users with this level
            org_id: If set, applies as org ceiling
            team_id: If set, applies as team ceiling
            created_by: User ID of admin creating the quota

        Returns:
            The quota ID

        Note: Exactly one of user_id, level_id, org_id, team_id should be set,
        or all None for global default.
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _set():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Check if quota already exists for this scope
                if user_id is not None:
                    cursor.execute(
                        "SELECT id FROM quotas WHERE quota_type = ? AND user_id = ?",
                        (quota_type, user_id),
                    )
                elif level_id is not None:
                    cursor.execute(
                        "SELECT id FROM quotas WHERE quota_type = ? AND level_id = ? AND user_id IS NULL",
                        (quota_type, level_id),
                    )
                elif org_id is not None:
                    cursor.execute(
                        "SELECT id FROM quotas WHERE quota_type = ? AND org_id = ? AND team_id IS NULL",
                        (quota_type, org_id),
                    )
                elif team_id is not None:
                    cursor.execute(
                        "SELECT id FROM quotas WHERE quota_type = ? AND team_id = ?",
                        (quota_type, team_id),
                    )
                else:
                    cursor.execute(
                        """SELECT id FROM quotas WHERE quota_type = ?
                           AND user_id IS NULL AND level_id IS NULL
                           AND org_id IS NULL AND team_id IS NULL""",
                        (quota_type,),
                    )

                existing = cursor.fetchone()

                if existing:
                    cursor.execute(
                        """
                        UPDATE quotas SET limit_value = ?, action = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (limit_value, action, now, existing["id"]),
                    )
                    quota_id = existing["id"]
                else:
                    cursor.execute(
                        """
                        INSERT INTO quotas
                        (quota_type, limit_value, action, user_id, level_id, org_id, team_id, created_at, updated_at, created_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (quota_type, limit_value, action, user_id, level_id, org_id, team_id, now, now, created_by),
                    )
                    quota_id = cursor.lastrowid

                conn.commit()
                return quota_id
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _set)

    async def delete(self, quota_id: int) -> bool:
        """Delete a quota.

        Args:
            quota_id: Quota ID to delete

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM quotas WHERE id = ?", (quota_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def get_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get quotas specific to a user.

        Args:
            user_id: User ID

        Returns:
            List of quota dicts
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM quotas WHERE user_id = ?", (user_id,))
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_for_levels(self, level_ids: List[int]) -> List[Dict[str, Any]]:
        """Get quotas for a list of levels.

        Args:
            level_ids: List of level IDs

        Returns:
            List of quota dicts ordered by level priority
        """
        if not level_ids:
            return []

        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(level_ids))
                cursor.execute(
                    f"""
                    SELECT q.*, l.priority
                    FROM quotas q
                    JOIN levels l ON q.level_id = l.id
                    WHERE q.level_id IN ({placeholders}) AND q.user_id IS NULL
                    ORDER BY l.priority DESC
                    """,
                    level_ids,
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_for_org(self, org_id: int) -> List[Dict[str, Any]]:
        """Get quotas for an organization.

        Args:
            org_id: Organization ID

        Returns:
            List of quota dicts
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM quotas WHERE org_id = ? AND team_id IS NULL",
                    (org_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_for_teams(self, team_ids: List[int]) -> List[Dict[str, Any]]:
        """Get quotas for a list of teams.

        Args:
            team_ids: List of team IDs

        Returns:
            List of quota dicts
        """
        if not team_ids:
            return []

        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(team_ids))
                cursor.execute(
                    f"SELECT * FROM quotas WHERE team_id IN ({placeholders})",
                    team_ids,
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_global(self) -> List[Dict[str, Any]]:
        """Get global default quotas.

        Returns:
            List of quota dicts
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM quotas
                    WHERE user_id IS NULL AND level_id IS NULL
                      AND org_id IS NULL AND team_id IS NULL
                    """
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all quotas with scope information.

        Returns:
            List of quota dicts with scope metadata
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT q.*,
                           u.username as user_name,
                           l.name as level_name,
                           o.name as org_name,
                           t.name as team_name
                    FROM quotas q
                    LEFT JOIN users u ON q.user_id = u.id
                    LEFT JOIN levels l ON q.level_id = l.id
                    LEFT JOIN organizations o ON q.org_id = o.id
                    LEFT JOIN teams t ON q.team_id = t.id
                    ORDER BY q.quota_type, q.user_id, q.level_id, q.org_id, q.team_id
                    """
                )

                quotas = []
                for row in cursor.fetchall():
                    scope = "global"
                    scope_name = None
                    if row["user_id"]:
                        scope = "user"
                        scope_name = row["user_name"]
                    elif row["level_id"]:
                        scope = "level"
                        scope_name = row["level_name"]
                    elif row["team_id"]:
                        scope = "team"
                        scope_name = row["team_name"]
                    elif row["org_id"]:
                        scope = "org"
                        scope_name = row["org_name"]

                    quotas.append(
                        {
                            "id": row["id"],
                            "quota_type": row["quota_type"],
                            "limit_value": row["limit_value"],
                            "action": row["action"],
                            "scope": scope,
                            "scope_name": scope_name,
                            "user_id": row["user_id"],
                            "level_id": row["level_id"],
                            "org_id": row["org_id"],
                            "team_id": row["team_id"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    )
                return quotas
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)


# Singleton accessor
_instance: Optional[QuotaStore] = None


def get_quota_store() -> QuotaStore:
    """Get the singleton QuotaStore instance."""
    global _instance
    if _instance is None:
        _instance = QuotaStore()
    return _instance
