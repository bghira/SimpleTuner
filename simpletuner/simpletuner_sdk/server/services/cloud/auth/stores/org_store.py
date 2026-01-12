"""Organization management store.

Handles organization CRUD operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import Organization
from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class OrgStore(BaseAuthStore):
    """Store for organization management.

    Handles CRUD operations for organizations.
    """

    def _init_schema(self, cursor) -> None:
        """Initialize the organizations table schema."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                description TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                settings TEXT DEFAULT '{}'
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug)")

    async def create(
        self,
        name: str,
        slug: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Create a new organization.

        Args:
            name: Organization name
            slug: URL-safe unique identifier
            description: Organization description
            settings: Optional settings dict

        Returns:
            The created Organization object
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO organizations (name, slug, description, is_active, created_at, settings)
                    VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (name, slug, description, now, json.dumps(settings or {})),
                )
                org_id = cursor.lastrowid
                conn.commit()

                return Organization(
                    id=org_id,
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

    async def get(self, org_id: int) -> Optional[Organization]:
        """Get an organization by ID.

        Args:
            org_id: Organization ID

        Returns:
            Organization if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_org(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get an organization by slug.

        Args:
            slug: Organization slug

        Returns:
            Organization if found, None otherwise
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM organizations WHERE slug = ?", (slug,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_org(row)
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def list(self, include_inactive: bool = False) -> List[Organization]:
        """List all organizations.

        Args:
            include_inactive: Include inactive organizations

        Returns:
            List of Organization objects
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if include_inactive:
                    cursor.execute("SELECT * FROM organizations ORDER BY name")
                else:
                    cursor.execute("SELECT * FROM organizations WHERE is_active = 1 ORDER BY name")

                return [self._row_to_org(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def update(self, org_id: int, updates: Dict[str, Any]) -> bool:
        """Update an organization.

        Args:
            org_id: Organization ID
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
                set_clauses = []
                values = []
                for key, value in updates.items():
                    if key == "settings":
                        set_clauses.append("settings = ?")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                values.append(org_id)

                query = f"UPDATE organizations SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete(self, org_id: int) -> bool:
        """Delete an organization and its teams.

        Args:
            org_id: Organization ID

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Delete teams first (foreign key constraint)
                cursor.execute("DELETE FROM teams WHERE org_id = ?", (org_id,))
                # Remove org association from users
                cursor.execute("UPDATE users SET org_id = NULL WHERE org_id = ?", (org_id,))
                # Delete org
                cursor.execute("DELETE FROM organizations WHERE id = ?", (org_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    def _row_to_org(self, row) -> Organization:
        """Convert a database row to an Organization."""
        settings = {}
        if row["settings"]:
            try:
                settings = json.loads(row["settings"])
            except json.JSONDecodeError:
                pass

        return Organization(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            description=row["description"] or "",
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            settings=settings,
        )


# Singleton accessor
_instance: Optional[OrgStore] = None


def get_org_store() -> OrgStore:
    """Get the singleton OrgStore instance."""
    global _instance
    if _instance is None:
        _instance = OrgStore()
    return _instance
