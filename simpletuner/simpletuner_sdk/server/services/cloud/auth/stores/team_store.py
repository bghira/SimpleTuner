"""Team management store.

Handles team CRUD and membership operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import Team
from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class TeamStore(BaseAuthStore):
    """Store for team management.

    Handles CRUD operations for teams and user membership.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the teams and user_teams tables schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                slug TEXT NOT NULL,
                description TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                settings TEXT DEFAULT '{}',
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
                UNIQUE (org_id, slug)
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_slug ON teams(slug)")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_teams (
                user_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                role TEXT DEFAULT 'member',
                joined_at TEXT NOT NULL,
                PRIMARY KEY (user_id, team_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_teams_user ON user_teams(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_teams_team ON user_teams(team_id)")

    async def create(
        self,
        org_id: int,
        name: str,
        slug: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Team:
        """Create a new team in an organization.

        Args:
            org_id: Parent organization ID
            name: Team name
            slug: URL-safe identifier (unique within org)
            description: Team description
            settings: Optional settings dict

        Returns:
            The created Team object
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO teams (org_id, name, slug, description, is_active, created_at, settings)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (org_id, name, slug, description, now, json.dumps(settings or {})),
                )
                team_id = cursor.lastrowid
                conn.commit()

                return Team(
                    id=team_id,
                    org_id=org_id,
                    name=name,
                    slug=slug,
                    description=description,
                    is_active=True,
                    created_at=now,
                    settings=settings or {},
                )
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _create)

    async def get(self, team_id: int) -> Optional[Team]:
        """Get a team by ID.

        Args:
            team_id: Team ID

        Returns:
            Team if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_team(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def list_for_org(self, org_id: int, include_inactive: bool = False) -> List[Team]:
        """List teams in an organization.

        Args:
            org_id: Organization ID
            include_inactive: Include inactive teams

        Returns:
            List of Team objects
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if include_inactive:
                    cursor.execute("SELECT * FROM teams WHERE org_id = ? ORDER BY name", (org_id,))
                else:
                    cursor.execute(
                        "SELECT * FROM teams WHERE org_id = ? AND is_active = 1 ORDER BY name",
                        (org_id,),
                    )
                return [self._row_to_team(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def get_user_teams(self, user_id: int) -> List[Team]:
        """Get all teams a user belongs to.

        Args:
            user_id: User ID

        Returns:
            List of Team objects
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT t.* FROM teams t
                    JOIN user_teams ut ON t.id = ut.team_id
                    WHERE ut.user_id = ? AND t.is_active = 1
                    ORDER BY t.name
                    """,
                    (user_id,),
                )
                return [self._row_to_team(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def add_user(self, user_id: int, team_id: int, role: str = "member") -> bool:
        """Add a user to a team.

        Args:
            user_id: User to add
            team_id: Team to add user to
            role: Team role (member, lead, etc.)

        Returns:
            True if added, False if already a member
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _add():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO user_teams (user_id, team_id, role, joined_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, team_id, role, now),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _add)

    async def remove_user(self, user_id: int, team_id: int) -> bool:
        """Remove a user from a team.

        Args:
            user_id: User to remove
            team_id: Team to remove user from

        Returns:
            True if removed, False if not a member
        """
        loop = asyncio.get_running_loop()

        def _remove():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM user_teams WHERE user_id = ? AND team_id = ?",
                    (user_id, team_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _remove)

    async def get_members(self, team_id: int) -> List[Dict[str, Any]]:
        """Get all members of a team.

        Args:
            team_id: Team to get members for

        Returns:
            List of dicts with user info and their team role
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT u.id, u.email, u.username, u.display_name, u.is_active,
                           u.is_admin, ut.role, ut.joined_at
                    FROM users u
                    INNER JOIN user_teams ut ON u.id = ut.user_id
                    WHERE ut.team_id = ?
                    ORDER BY ut.joined_at DESC
                    """,
                    (team_id,),
                )
                return [
                    {
                        "id": row["id"],
                        "email": row["email"],
                        "username": row["username"],
                        "display_name": row["display_name"],
                        "is_active": bool(row["is_active"]),
                        "is_admin": bool(row["is_admin"]),
                        "role": row["role"],
                        "joined_at": row["joined_at"],
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def update_user_role(self, user_id: int, team_id: int, role: str) -> bool:
        """Update a user's role in a team.

        Args:
            user_id: User to update
            team_id: Team to update in
            role: New role

        Returns:
            True if updated, False if not a member
        """
        loop = asyncio.get_running_loop()

        def _update():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE user_teams SET role = ?
                    WHERE user_id = ? AND team_id = ?
                    """,
                    (role, user_id, team_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def update(self, team_id: int, updates: Dict[str, Any]) -> bool:
        """Update a team's fields.

        Args:
            team_id: Team ID to update
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

                # Handle settings specially
                if "settings" in updates:
                    updates["settings"] = json.dumps(updates["settings"])

                set_clauses = []
                values = []
                for key, value in updates.items():
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                values.append(team_id)

                query = f"UPDATE teams SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete(self, team_id: int) -> bool:
        """Delete a team.

        Args:
            team_id: Team ID to delete

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM teams WHERE id = ?", (team_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    def _row_to_team(self, row) -> Team:
        """Convert a database row to a Team."""
        settings = {}
        if row["settings"]:
            try:
                settings = json.loads(row["settings"])
            except json.JSONDecodeError:
                pass

        return Team(
            id=row["id"],
            org_id=row["org_id"],
            name=row["name"],
            slug=row["slug"],
            description=row["description"] or "",
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            settings=settings,
        )


# Singleton accessor
_instance: Optional[TeamStore] = None


def get_team_store() -> TeamStore:
    """Get the singleton TeamStore instance."""
    global _instance
    if _instance is None:
        _instance = TeamStore()
    return _instance
