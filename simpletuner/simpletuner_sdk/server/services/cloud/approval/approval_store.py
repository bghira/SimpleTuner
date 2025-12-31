"""SQLite-based storage for approval rules and requests."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ApprovalRequest, ApprovalRule, ApprovalStatus, RuleCondition

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class ApprovalStore:
    """SQLite-based storage for approval workflow.

    Manages approval rules and pending requests.
    Thread-safe with WAL mode for concurrent access.
    """

    _instance: Optional["ApprovalStore"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "ApprovalStore":
        """Singleton pattern - one approval store per process."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the approval store."""
        if getattr(self, "_initialized", False):
            return

        if db_path is None:
            from ..container import get_job_store

            job_store = get_job_store()
            db_path = job_store._db_path.parent / "approval.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check current schema version
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
            """
        )

        row = cursor.execute("SELECT version FROM schema_version").fetchone()
        current_version = row["version"] if row else 0

        if current_version < SCHEMA_VERSION:
            self._run_migrations(cursor, current_version, SCHEMA_VERSION)
            cursor.execute("DELETE FROM schema_version")
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()

    def _run_migrations(self, cursor: sqlite3.Cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        if from_version < 1 <= to_version:
            # Approval rules table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    condition TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 0,
                    applies_to_provider TEXT,
                    applies_to_level TEXT,
                    exempt_levels TEXT DEFAULT '[]',
                    required_approver_level TEXT DEFAULT 'lead',
                    created_at TEXT NOT NULL,
                    created_by INTEGER
                )
                """
            )

            # Approval requests table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    rule_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    reason TEXT DEFAULT '',
                    provider TEXT DEFAULT 'replicate',
                    config_name TEXT,
                    estimated_cost REAL DEFAULT 0.0,
                    hardware_type TEXT,
                    reviewed_by INTEGER,
                    reviewed_at TEXT,
                    review_notes TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    notification_sent INTEGER DEFAULT 0,
                    notification_sent_at TEXT,
                    FOREIGN KEY (rule_id) REFERENCES approval_rules(id)
                )
                """
            )

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON approval_requests(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_user ON approval_requests(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_job ON approval_requests(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_active ON approval_rules(is_active)")

            logger.info("Created approval schema v%d", SCHEMA_VERSION)

    # --- Rule Management ---

    async def create_rule(self, rule: ApprovalRule) -> int:
        """Create a new approval rule."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()
        exempt_json = json.dumps(rule.exempt_levels)

        cursor.execute(
            """
            INSERT INTO approval_rules (
                name, description, condition, threshold, is_active, priority,
                applies_to_provider, applies_to_level, exempt_levels,
                required_approver_level, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule.name,
                rule.description,
                rule.condition.value,
                rule.threshold,
                1 if rule.is_active else 0,
                rule.priority,
                rule.applies_to_provider,
                rule.applies_to_level,
                exempt_json,
                rule.required_approver_level,
                now,
                rule.created_by,
            ),
        )
        conn.commit()

        return cursor.lastrowid

    async def get_rule(self, rule_id: int) -> Optional[ApprovalRule]:
        """Get an approval rule by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM approval_rules WHERE id = ?", (rule_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_rule(row)

    async def update_rule(self, rule_id: int, updates: Dict[str, Any]) -> bool:
        """Update an approval rule."""
        if not updates:
            return True

        conn = self._get_connection()
        cursor = conn.cursor()

        # Handle special fields
        if "condition" in updates and isinstance(updates["condition"], RuleCondition):
            updates["condition"] = updates["condition"].value
        if "is_active" in updates:
            updates["is_active"] = 1 if updates["is_active"] else 0
        if "exempt_levels" in updates:
            updates["exempt_levels"] = json.dumps(updates["exempt_levels"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [rule_id]

        cursor.execute(f"UPDATE approval_rules SET {set_clause} WHERE id = ?", values)
        conn.commit()

        return cursor.rowcount > 0

    async def delete_rule(self, rule_id: int) -> bool:
        """Delete an approval rule."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM approval_rules WHERE id = ?", (rule_id,))
        conn.commit()

        return cursor.rowcount > 0

    async def list_rules(
        self,
        active_only: bool = False,
        condition: Optional[RuleCondition] = None,
    ) -> List[ApprovalRule]:
        """List approval rules."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM approval_rules WHERE 1=1"
        params: List[Any] = []

        if active_only:
            query += " AND is_active = 1"

        if condition:
            query += " AND condition = ?"
            params.append(condition.value)

        query += " ORDER BY priority DESC, id ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_rule(row) for row in rows]

    # --- Request Management ---

    async def create_request(
        self,
        job_id: str,
        user_id: int,
        rule_id: int,
        reason: str,
        provider: str = "replicate",
        config_name: Optional[str] = None,
        estimated_cost: float = 0.0,
        hardware_type: Optional[str] = None,
        expires_hours: int = 24,
    ) -> ApprovalRequest:
        """Create a new approval request."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=expires_hours)

        cursor.execute(
            """
            INSERT INTO approval_requests (
                job_id, user_id, rule_id, status, reason, provider,
                config_name, estimated_cost, hardware_type,
                created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                user_id,
                rule_id,
                ApprovalStatus.PENDING.value,
                reason,
                provider,
                config_name,
                estimated_cost,
                hardware_type,
                now.isoformat(),
                expires.isoformat(),
            ),
        )
        conn.commit()

        return ApprovalRequest(
            id=cursor.lastrowid,
            job_id=job_id,
            user_id=user_id,
            rule_id=rule_id,
            status=ApprovalStatus.PENDING,
            reason=reason,
            provider=provider,
            config_name=config_name,
            estimated_cost=estimated_cost,
            hardware_type=hardware_type,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )

    async def get_request(self, request_id: int) -> Optional[ApprovalRequest]:
        """Get an approval request by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM approval_requests WHERE id = ?", (request_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_request(row)

    async def get_request_by_job_id(self, job_id: str) -> Optional[ApprovalRequest]:
        """Get the approval request for a job."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM approval_requests WHERE job_id = ? ORDER BY id DESC LIMIT 1",
            (job_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_request(row)

    async def approve_request(
        self,
        request_id: int,
        reviewed_by: int,
        notes: Optional[str] = None,
    ) -> bool:
        """Approve a pending request."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE approval_requests
            SET status = ?, reviewed_by = ?, reviewed_at = ?, review_notes = ?
            WHERE id = ? AND status = ?
            """,
            (
                ApprovalStatus.APPROVED.value,
                reviewed_by,
                now,
                notes,
                request_id,
                ApprovalStatus.PENDING.value,
            ),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def reject_request(
        self,
        request_id: int,
        reviewed_by: int,
        reason: str,
    ) -> bool:
        """Reject a pending request."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE approval_requests
            SET status = ?, reviewed_by = ?, reviewed_at = ?, review_notes = ?
            WHERE id = ? AND status = ?
            """,
            (
                ApprovalStatus.REJECTED.value,
                reviewed_by,
                now,
                reason,
                request_id,
                ApprovalStatus.PENDING.value,
            ),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def cancel_request(self, request_id: int) -> bool:
        """Cancel a pending request (by the requester)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE approval_requests
            SET status = ?
            WHERE id = ? AND status = ?
            """,
            (ApprovalStatus.CANCELLED.value, request_id, ApprovalStatus.PENDING.value),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def list_requests(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ApprovalStatus] = None,
        user_id: Optional[int] = None,
        pending_only: bool = False,
    ) -> List[ApprovalRequest]:
        """List approval requests."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM approval_requests WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.value)
        elif pending_only:
            query += " AND status = ?"
            params.append(ApprovalStatus.PENDING.value)

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_request(row) for row in rows]

    async def get_pending_count(self) -> int:
        """Get count of pending approval requests."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) as cnt FROM approval_requests WHERE status = ?",
            (ApprovalStatus.PENDING.value,),
        )
        return cursor.fetchone()["cnt"]

    async def mark_notification_sent(self, request_id: int) -> bool:
        """Mark that a notification was sent for a request."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE approval_requests
            SET notification_sent = 1, notification_sent_at = ?
            WHERE id = ?
            """,
            (now, request_id),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def get_unsent_notifications(self) -> List[ApprovalRequest]:
        """Get pending requests that haven't had notifications sent."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM approval_requests
            WHERE status = ? AND notification_sent = 0
            ORDER BY created_at ASC
            """,
            (ApprovalStatus.PENDING.value,),
        )
        rows = cursor.fetchall()

        return [self._row_to_request(row) for row in rows]

    async def expire_old_requests(self) -> int:
        """Expire requests past their expiration time."""
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE approval_requests
            SET status = ?
            WHERE status = ? AND expires_at IS NOT NULL AND expires_at < ?
            """,
            (ApprovalStatus.EXPIRED.value, ApprovalStatus.PENDING.value, now),
        )
        conn.commit()

        expired = cursor.rowcount
        if expired > 0:
            logger.info("Expired %d approval requests", expired)

        return expired

    def _row_to_rule(self, row: sqlite3.Row) -> ApprovalRule:
        """Convert a database row to an ApprovalRule."""
        exempt_levels = json.loads(row["exempt_levels"]) if row["exempt_levels"] else []

        return ApprovalRule(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            condition=RuleCondition(row["condition"]),
            threshold=row["threshold"],
            is_active=bool(row["is_active"]),
            priority=row["priority"] or 0,
            applies_to_provider=row["applies_to_provider"],
            applies_to_level=row["applies_to_level"],
            exempt_levels=exempt_levels,
            required_approver_level=row["required_approver_level"] or "lead",
            created_at=row["created_at"],
            created_by=row["created_by"],
        )

    def _row_to_request(self, row: sqlite3.Row) -> ApprovalRequest:
        """Convert a database row to an ApprovalRequest."""
        return ApprovalRequest(
            id=row["id"],
            job_id=row["job_id"],
            user_id=row["user_id"],
            rule_id=row["rule_id"],
            status=ApprovalStatus(row["status"]),
            reason=row["reason"] or "",
            provider=row["provider"],
            config_name=row["config_name"],
            estimated_cost=row["estimated_cost"] or 0.0,
            hardware_type=row["hardware_type"],
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            review_notes=row["review_notes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            notification_sent=bool(row["notification_sent"]),
            notification_sent_at=row["notification_sent_at"],
        )
