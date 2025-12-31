"""Permission and level management store.

Handles levels (roles), permissions, user-level assignments,
and permission overrides.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..models import DEFAULT_LEVELS, DEFAULT_PERMISSIONS, Permission, ResourceRule, ResourceType, RuleAction, UserLevel
from .base import BaseAuthStore

logger = logging.getLogger(__name__)


class PermissionStore(BaseAuthStore):
    """Store for permission and level management.

    Handles:
    - Level (role) definitions
    - Permission definitions
    - User-level assignments
    - Per-user permission overrides
    - Resource-based access rules
    """

    def _init_schema(self, cursor) -> None:
        """Initialize permission-related tables."""
        # Permissions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general'
            )
        """
        )

        # Levels (roles) table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS levels (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                is_system INTEGER NOT NULL DEFAULT 0
            )
        """
        )

        # Level-Permission linking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS level_permissions (
                level_id INTEGER NOT NULL,
                permission_name TEXT NOT NULL,
                PRIMARY KEY (level_id, permission_name),
                FOREIGN KEY (level_id) REFERENCES levels(id) ON DELETE CASCADE
            )
        """
        )

        # User-Level linking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_levels (
                user_id INTEGER NOT NULL,
                level_id INTEGER NOT NULL,
                granted_at TEXT NOT NULL,
                granted_by INTEGER,
                PRIMARY KEY (user_id, level_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (level_id) REFERENCES levels(id) ON DELETE CASCADE
            )
        """
        )

        # User-Permission overrides table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id INTEGER NOT NULL,
                permission_name TEXT NOT NULL,
                granted INTEGER NOT NULL DEFAULT 1,
                granted_at TEXT NOT NULL,
                granted_by INTEGER,
                PRIMARY KEY (user_id, permission_name),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """
        )

        # Resource rules table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                pattern TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'allow',
                priority INTEGER NOT NULL DEFAULT 0,
                description TEXT,
                created_at TEXT NOT NULL,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resource_rules_type ON resource_rules(resource_type)")

        # Level-ResourceRule linking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS level_resource_rules (
                level_id INTEGER NOT NULL,
                rule_id INTEGER NOT NULL,
                PRIMARY KEY (level_id, rule_id),
                FOREIGN KEY (level_id) REFERENCES levels(id) ON DELETE CASCADE,
                FOREIGN KEY (rule_id) REFERENCES resource_rules(id) ON DELETE CASCADE
            )
        """
        )

    def seed_defaults(self, cursor) -> None:
        """Seed default permissions and levels."""
        # Insert default permissions
        for perm in DEFAULT_PERMISSIONS:
            cursor.execute(
                """
                INSERT OR IGNORE INTO permissions (id, name, description, category)
                VALUES (?, ?, ?, ?)
                """,
                (perm.id, perm.name, perm.description, perm.category),
            )

        # Insert default levels
        for level in DEFAULT_LEVELS:
            cursor.execute(
                """
                INSERT OR IGNORE INTO levels (id, name, description, priority, is_system)
                VALUES (?, ?, ?, ?, ?)
                """,
                (level.id, level.name, level.description, level.priority, 1 if level.is_system else 0),
            )

            # Insert level permissions
            for perm_name in level.permissions:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO level_permissions (level_id, permission_name)
                    VALUES (?, ?)
                    """,
                    (level.id, perm_name),
                )

        logger.info("Seeded default permissions and levels")

    async def get_all_levels(self) -> List[UserLevel]:
        """Get all defined levels.

        Returns:
            List of UserLevel objects
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM levels ORDER BY priority DESC")
                levels = []
                for row in cursor.fetchall():
                    # Get permissions for this level
                    cursor.execute(
                        "SELECT permission_name FROM level_permissions WHERE level_id = ?",
                        (row["id"],),
                    )
                    perms = {r["permission_name"] for r in cursor.fetchall()}

                    levels.append(
                        UserLevel(
                            id=row["id"],
                            name=row["name"],
                            description=row["description"],
                            priority=row["priority"],
                            permissions=perms,
                            is_system=bool(row["is_system"]),
                        )
                    )
                return levels
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_level(self, level_id: int) -> Optional[UserLevel]:
        """Get a level by ID."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM levels WHERE id = ?", (level_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                cursor.execute(
                    "SELECT permission_name FROM level_permissions WHERE level_id = ?",
                    (level_id,),
                )
                perms = {r["permission_name"] for r in cursor.fetchall()}
                return UserLevel(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    priority=row["priority"],
                    permissions=perms,
                    is_system=bool(row["is_system"]),
                )
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_level_by_name(self, name: str) -> Optional[UserLevel]:
        """Get a level by name."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM levels WHERE name = ?", (name,))
                row = cursor.fetchone()
                if not row:
                    return None
                cursor.execute(
                    "SELECT permission_name FROM level_permissions WHERE level_id = ?",
                    (row["id"],),
                )
                perms = {r["permission_name"] for r in cursor.fetchall()}
                return UserLevel(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    priority=row["priority"],
                    permissions=perms,
                    is_system=bool(row["is_system"]),
                )
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def create_level(
        self, name: str, description: str, priority: int = 0, permissions: Optional[Set[str]] = None
    ) -> UserLevel:
        """Create a new level."""
        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO levels (name, description, priority, is_system) VALUES (?, ?, ?, 0)",
                    (name, description, priority),
                )
                level_id = cursor.lastrowid
                if permissions:
                    for perm in permissions:
                        cursor.execute(
                            "INSERT OR IGNORE INTO level_permissions (level_id, permission_name) VALUES (?, ?)",
                            (level_id, perm),
                        )
                conn.commit()
                return UserLevel(
                    id=level_id,
                    name=name,
                    description=description,
                    priority=priority,
                    permissions=permissions or set(),
                    is_system=False,
                )
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _create)

    async def update_level(self, level_id: int, updates: Dict[str, Any]) -> bool:
        """Update a level."""
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
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                values.append(level_id)
                query = f"UPDATE levels SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete_level(self, level_id: int) -> bool:
        """Delete a level (non-system only)."""
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM levels WHERE id = ? AND is_system = 0", (level_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def set_level_permissions(self, level_id: int, permission_names: Set[str]) -> None:
        """Set permissions for a level (replaces existing)."""
        loop = asyncio.get_running_loop()

        def _set():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM level_permissions WHERE level_id = ?", (level_id,))
                for perm in permission_names:
                    cursor.execute(
                        "INSERT INTO level_permissions (level_id, permission_name) VALUES (?, ?)",
                        (level_id, perm),
                    )
                conn.commit()
            finally:
                conn.close()

        async with self._lock:
            await loop.run_in_executor(None, _set)

    async def get_all_permissions(self) -> List[Permission]:
        """Get all defined permissions.

        Returns:
            List of Permission objects
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM permissions ORDER BY category, name")
                return [
                    Permission(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        category=row["category"],
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_user_levels(self, user_id: int) -> List[UserLevel]:
        """Get all levels assigned to a user.

        Args:
            user_id: User ID

        Returns:
            List of UserLevel objects
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT l.* FROM levels l
                    JOIN user_levels ul ON l.id = ul.level_id
                    WHERE ul.user_id = ?
                    ORDER BY l.priority DESC
                    """,
                    (user_id,),
                )
                levels = []
                for row in cursor.fetchall():
                    cursor.execute(
                        "SELECT permission_name FROM level_permissions WHERE level_id = ?",
                        (row["id"],),
                    )
                    perms = {r["permission_name"] for r in cursor.fetchall()}
                    levels.append(
                        UserLevel(
                            id=row["id"],
                            name=row["name"],
                            description=row["description"],
                            priority=row["priority"],
                            permissions=perms,
                            is_system=bool(row["is_system"]),
                        )
                    )
                return levels
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_user_permissions(self, user_id: int, is_admin: bool = False) -> Set[str]:
        """Get effective permissions for a user.

        Combines permissions from all assigned levels plus any overrides.
        Admins get all permissions.

        Args:
            user_id: User ID
            is_admin: Whether user is admin

        Returns:
            Set of permission names
        """
        if is_admin:
            # Admins get all permissions
            all_perms = await self.get_all_permissions()
            return {p.name for p in all_perms}

        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Get permissions from levels
                cursor.execute(
                    """
                    SELECT DISTINCT lp.permission_name
                    FROM level_permissions lp
                    JOIN user_levels ul ON lp.level_id = ul.level_id
                    WHERE ul.user_id = ?
                    """,
                    (user_id,),
                )
                perms = {row["permission_name"] for row in cursor.fetchall()}

                # Apply overrides
                cursor.execute(
                    "SELECT permission_name, granted FROM user_permissions WHERE user_id = ?",
                    (user_id,),
                )
                for row in cursor.fetchall():
                    if row["granted"]:
                        perms.add(row["permission_name"])
                    else:
                        perms.discard(row["permission_name"])

                return perms
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def assign_level(
        self,
        user_id: int,
        level_name: str,
        granted_by: Optional[int] = None,
    ) -> bool:
        """Assign a level to a user.

        Args:
            user_id: User to assign level to
            level_name: Name of level to assign
            granted_by: ID of user granting the level

        Returns:
            True if assigned, False if level not found or already assigned
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _assign():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM levels WHERE name = ?", (level_name,))
                level_row = cursor.fetchone()
                if not level_row:
                    return False

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO user_levels (user_id, level_id, granted_at, granted_by)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, level_row["id"], now, granted_by),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _assign)

    async def remove_level(self, user_id: int, level_name: str) -> bool:
        """Remove a level from a user.

        Args:
            user_id: User to remove level from
            level_name: Name of level to remove

        Returns:
            True if removed, False if not assigned
        """
        loop = asyncio.get_running_loop()

        def _remove():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM levels WHERE name = ?", (level_name,))
                level_row = cursor.fetchone()
                if not level_row:
                    return False

                cursor.execute(
                    "DELETE FROM user_levels WHERE user_id = ? AND level_id = ?",
                    (user_id, level_row["id"]),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _remove)

    async def override_permission(
        self,
        user_id: int,
        permission_name: str,
        granted: bool = True,
        granted_by: Optional[int] = None,
    ) -> bool:
        """Override a specific permission for a user.

        Args:
            user_id: User to override permission for
            permission_name: Name of permission
            granted: True to grant, False to revoke
            granted_by: ID of user making the override

        Returns:
            True if override was set
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _override():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO user_permissions
                    (user_id, permission_name, granted, granted_at, granted_by)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, permission_name, 1 if granted else 0, now, granted_by),
                )
                conn.commit()
                return True
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _override)

    async def remove_permission_override(self, user_id: int, permission_name: str) -> bool:
        """Remove a permission override for a user.

        Args:
            user_id: User to remove override for
            permission_name: Name of permission

        Returns:
            True if removed, False if not found
        """
        loop = asyncio.get_running_loop()

        def _remove():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM user_permissions WHERE user_id = ? AND permission_name = ?",
                    (user_id, permission_name),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _remove)

    async def create_resource_rule(
        self,
        name: str,
        resource_type: ResourceType,
        pattern: str,
        action: RuleAction = RuleAction.ALLOW,
        priority: int = 0,
        description: Optional[str] = None,
        created_by: Optional[int] = None,
    ) -> int:
        """Create a resource access rule.

        Args:
            name: Rule name
            resource_type: Type of resource (config, dataset, model, etc.)
            pattern: Glob pattern to match resources
            action: Allow or deny
            priority: Higher priority rules evaluated first
            description: Rule description
            created_by: User creating the rule

        Returns:
            Rule ID
        """
        now = datetime.now(timezone.utc).isoformat()
        loop = asyncio.get_running_loop()

        def _create():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO resource_rules
                    (name, resource_type, pattern, action, priority, description, created_at, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, resource_type.value, pattern, action.value, priority, description, now, created_by),
                )
                rule_id = cursor.lastrowid
                conn.commit()
                return rule_id
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _create)

    async def get_user_resource_rules(self, user_id: int) -> List[ResourceRule]:
        """Get resource rules for a user based on their levels.

        Args:
            user_id: User ID

        Returns:
            List of ResourceRule objects ordered by priority
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT rr.*
                    FROM resource_rules rr
                    JOIN level_resource_rules lrr ON rr.id = lrr.rule_id
                    JOIN user_levels ul ON lrr.level_id = ul.level_id
                    WHERE ul.user_id = ?
                    ORDER BY rr.priority DESC
                    """,
                    (user_id,),
                )
                return [
                    ResourceRule(
                        id=row["id"],
                        name=row["name"],
                        resource_type=ResourceType(row["resource_type"]),
                        pattern=row["pattern"],
                        action=RuleAction(row["action"]),
                        priority=row["priority"],
                        description=row["description"],
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def assign_rule_to_level(self, level_name: str, rule_id: int) -> bool:
        """Assign a resource rule to a level.

        Args:
            level_name: Level name
            rule_id: Rule ID

        Returns:
            True if assigned
        """
        loop = asyncio.get_running_loop()

        def _assign():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM levels WHERE name = ?", (level_name,))
                level_row = cursor.fetchone()
                if not level_row:
                    return False

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO level_resource_rules (level_id, rule_id)
                    VALUES (?, ?)
                    """,
                    (level_row["id"], rule_id),
                )
                conn.commit()
                return True
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _assign)

    async def get_rules_with_level_assignments(self) -> List[Dict[str, Any]]:
        """Get all resource rules with their level assignments.

        Returns:
            List of dicts with rule info and assigned levels
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Get all resource rules
                cursor.execute(
                    """
                    SELECT * FROM resource_rules ORDER BY priority DESC, name
                """
                )
                rules = []
                for row in cursor.fetchall():
                    rule_id = row["id"]
                    # Get levels assigned to this rule
                    cursor.execute(
                        """
                        SELECT l.id, l.name, l.description, l.priority
                        FROM levels l
                        JOIN level_resource_rules lrr ON l.id = lrr.level_id
                        WHERE lrr.rule_id = ?
                        ORDER BY l.priority DESC
                    """,
                        (rule_id,),
                    )
                    levels = [
                        {
                            "id": lvl["id"],
                            "name": lvl["name"],
                            "description": lvl["description"],
                            "priority": lvl["priority"],
                        }
                        for lvl in cursor.fetchall()
                    ]
                    rules.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "resource_type": row["resource_type"],
                            "pattern": row["pattern"],
                            "action": row["action"],
                            "priority": row["priority"],
                            "description": row["description"],
                            "created_at": row["created_at"],
                            "created_by": row["created_by"],
                            "levels": levels,
                        }
                    )
                return rules
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_resource_rule(self, rule_id: int) -> Optional[ResourceRule]:
        """Get a resource rule by ID."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM resource_rules WHERE id = ?", (rule_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return ResourceRule(
                    id=row["id"],
                    name=row["name"],
                    resource_type=ResourceType(row["resource_type"]),
                    pattern=row["pattern"],
                    action=RuleAction(row["action"]),
                    priority=row["priority"],
                    description=row["description"],
                )
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def update_resource_rule(self, rule_id: int, updates: Dict[str, Any]) -> bool:
        """Update a resource rule."""
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
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                values.append(rule_id)
                query = f"UPDATE resource_rules SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete_resource_rule(self, rule_id: int) -> bool:
        """Delete a resource rule."""
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM resource_rules WHERE id = ?", (rule_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def get_level_rules(self, level_id: int) -> List[ResourceRule]:
        """Get all resource rules assigned to a level."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT rr.* FROM resource_rules rr
                    JOIN level_resource_rules lrr ON rr.id = lrr.rule_id
                    WHERE lrr.level_id = ?
                    ORDER BY rr.priority DESC
                """,
                    (level_id,),
                )
                return [
                    ResourceRule(
                        id=row["id"],
                        name=row["name"],
                        resource_type=ResourceType(row["resource_type"]),
                        pattern=row["pattern"],
                        action=RuleAction(row["action"]),
                        priority=row["priority"],
                        description=row["description"],
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def set_level_rules(self, level_id: int, rule_ids: List[int]) -> bool:
        """Set rules for a level (replaces existing)."""
        loop = asyncio.get_running_loop()

        def _set():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM level_resource_rules WHERE level_id = ?", (level_id,))
                for rule_id in rule_ids:
                    cursor.execute(
                        "INSERT INTO level_resource_rules (level_id, rule_id) VALUES (?, ?)",
                        (level_id, rule_id),
                    )
                conn.commit()
                return True
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _set)

    async def remove_rule_from_level(self, level_id: int, rule_id: int) -> bool:
        """Remove a rule from a level."""
        loop = asyncio.get_running_loop()

        def _remove():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM level_resource_rules WHERE level_id = ? AND rule_id = ?",
                    (level_id, rule_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _remove)


# Singleton accessor
_instance: Optional[PermissionStore] = None


def get_permission_store() -> PermissionStore:
    """Get the singleton PermissionStore instance."""
    global _instance
    if _instance is None:
        _instance = PermissionStore()
    return _instance
