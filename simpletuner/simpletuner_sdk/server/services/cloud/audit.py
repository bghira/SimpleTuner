"""Audit logging with tamper-evident storage.

Provides:
- Immutable audit log entries
- Cryptographic hash chaining for tamper detection
- Structured event types for security-relevant operations
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Categories of auditable events."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_SESSION_EXPIRED = "auth.session.expired"
    AUTH_API_KEY_USED = "auth.api_key.used"

    # User management
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LEVEL_CHANGED = "user.level.changed"
    USER_PERMISSION_CHANGED = "user.permission.changed"

    # API key management
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"

    # Credential management
    CREDENTIAL_CREATED = "credential.created"
    CREDENTIAL_DELETED = "credential.deleted"
    CREDENTIAL_USED = "credential.used"

    # Job operations
    JOB_SUBMITTED = "job.submitted"
    JOB_CANCELLED = "job.cancelled"
    JOB_APPROVED = "job.approved"
    JOB_REJECTED = "job.rejected"

    # Quota operations
    QUOTA_EXCEEDED = "quota.exceeded"
    QUOTA_CHANGED = "quota.changed"

    # Configuration changes
    CONFIG_CHANGED = "config.changed"
    PROVIDER_CONFIGURED = "provider.configured"

    # Security events
    PERMISSION_DENIED = "security.permission_denied"
    RATE_LIMITED = "security.rate_limited"
    SUSPICIOUS_ACTIVITY = "security.suspicious"


@dataclass
class AuditEntry:
    """An immutable audit log entry."""

    id: int
    timestamp: str
    event_type: str
    actor_id: Optional[int]  # User ID who performed the action
    actor_username: Optional[str]
    actor_ip: Optional[str]
    target_type: Optional[str]  # e.g., "user", "job", "config"
    target_id: Optional[str]  # ID of the affected resource
    action: str  # Human-readable action description
    details: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""  # Hash of previous entry for chain integrity
    entry_hash: str = ""  # Hash of this entry


class AuditStore:
    """Persists audit logs with tamper-evident storage.

    Features:
    - Append-only storage (no updates/deletes)
    - Hash chaining for integrity verification
    - Optional HMAC signing with secret key
    """

    _instance: Optional["AuditStore"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "AuditStore":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return

        self._db_path = db_path or Path.home() / ".simpletuner" / "audit.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        # Get signing key from environment or generate one
        self._signing_key = self._get_or_create_signing_key()

        self._init_db()
        self._initialized = True

    def _get_or_create_signing_key(self) -> bytes:
        """Get or create the HMAC signing key."""
        # Check environment variable first
        env_key = os.environ.get("SIMPLETUNER_AUDIT_KEY")
        if env_key:
            return env_key.encode("utf-8")

        # Check key file
        key_path = self._db_path.parent / "audit.key"
        if key_path.exists():
            return key_path.read_bytes()

        # Generate new key
        import secrets

        key = secrets.token_bytes(32)
        try:
            key_path.write_bytes(key)
            key_path.chmod(0o600)
            logger.info("Generated new audit signing key")
        except Exception as exc:
            logger.warning("Could not save audit key: %s", exc)

        return key

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Initialize the audit database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Audit log table - append-only
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id INTEGER,
                    actor_username TEXT,
                    actor_ip TEXT,
                    target_type TEXT,
                    target_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
            """
            )

            # Indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_type, target_id)")

            conn.commit()
        finally:
            conn.close()

    def _compute_entry_hash(
        self,
        timestamp: str,
        event_type: str,
        actor_id: Optional[int],
        action: str,
        details: str,
        previous_hash: str,
    ) -> str:
        """Compute HMAC hash for an entry."""
        data = f"{timestamp}|{event_type}|{actor_id}|{action}|{details}|{previous_hash}"
        return hmac.new(
            self._signing_key,
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _get_last_hash(self, conn: sqlite3.Connection) -> str:
        """Get the hash of the most recent entry."""
        cursor = conn.cursor()
        cursor.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row["entry_hash"] if row else "genesis"

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        actor_id: Optional[int] = None,
        actor_username: Optional[str] = None,
        actor_ip: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log an audit event.

        Args:
            event_type: Category of the event
            action: Human-readable description
            actor_id: User ID who performed the action
            actor_username: Username for display
            actor_ip: IP address of the actor
            target_type: Type of resource affected
            target_id: ID of resource affected
            details: Additional structured data

        Returns:
            The audit entry ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {})

        loop = asyncio.get_running_loop()

        def _log():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Get previous hash for chain integrity
                previous_hash = self._get_last_hash(conn)

                # Compute this entry's hash
                entry_hash = self._compute_entry_hash(
                    timestamp,
                    event_type.value,
                    actor_id,
                    action,
                    details_json,
                    previous_hash,
                )

                cursor.execute(
                    """
                    INSERT INTO audit_log (
                        timestamp, event_type, actor_id, actor_username, actor_ip,
                        target_type, target_id, action, details, previous_hash, entry_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        event_type.value,
                        actor_id,
                        actor_username,
                        actor_ip,
                        target_type,
                        target_id,
                        action,
                        details_json,
                        previous_hash,
                        entry_hash,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

        async with self._lock:
            entry_id = await loop.run_in_executor(None, _log)

        # Forward to webhook if configured (fire-and-forget)
        asyncio.create_task(
            self._forward_to_webhook(
                AuditEntry(
                    id=entry_id,
                    timestamp=timestamp,
                    event_type=event_type.value,
                    actor_id=actor_id,
                    actor_username=actor_username,
                    actor_ip=actor_ip,
                    target_type=target_type,
                    target_id=target_id,
                    action=action,
                    details=details or {},
                )
            )
        )

        return entry_id

    async def _forward_to_webhook(self, entry: AuditEntry) -> None:
        """Forward audit entry to configured webhook (Elasticsearch/SIEM)."""
        try:
            from ..webui_state import WebUIStateStore

            defaults = WebUIStateStore().load_defaults()
            webhook_url = getattr(defaults, "audit_export_webhook_url", None)
            auth_token = getattr(defaults, "audit_export_auth_token", None)
            security_only = getattr(defaults, "audit_export_security_only", False)

            if not webhook_url:
                return

            # Filter security-only if configured
            if security_only:
                security_types = {
                    "auth.login.failed",
                    "security.permission_denied",
                    "security.rate_limited",
                    "security.suspicious",
                }
                if entry.event_type not in security_types:
                    return

            # Prepare payload
            payload = {
                "@timestamp": entry.timestamp,
                "event": {
                    "kind": "event",
                    "category": ["audit"],
                    "type": [entry.event_type.split(".")[0]],
                    "action": entry.action,
                },
                "simpletuner": {
                    "audit_id": entry.id,
                    "event_type": entry.event_type,
                    "actor_id": entry.actor_id,
                    "actor_username": entry.actor_username,
                    "actor_ip": entry.actor_ip,
                    "target_type": entry.target_type,
                    "target_id": entry.target_id,
                    "details": entry.details,
                },
            }

            headers = {"Content-Type": "application/json"}
            if auth_token:
                # Support both Bearer and ApiKey formats
                if auth_token.startswith("ApiKey ") or auth_token.startswith("Bearer "):
                    headers["Authorization"] = auth_token
                else:
                    headers["Authorization"] = f"ApiKey {auth_token}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                )
                if response.status_code >= 400:
                    logger.warning(
                        "Audit webhook failed: %s %s",
                        response.status_code,
                        response.text[:200],
                    )

        except Exception as exc:
            # Don't let webhook failures affect audit logging
            logger.debug("Audit webhook forward failed: %s", exc)

    async def query(
        self,
        event_types: Optional[List[str]] = None,
        actor_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit log entries.

        Args:
            event_types: Filter by event type(s)
            actor_id: Filter by actor
            target_type: Filter by target type
            target_id: Filter by target ID
            since: Start timestamp (ISO format)
            until: End timestamp (ISO format)
            limit: Max entries to return
            offset: Pagination offset

        Returns:
            List of matching audit entries
        """
        loop = asyncio.get_running_loop()

        def _query():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = "SELECT * FROM audit_log WHERE 1=1"
                params = []

                if event_types:
                    placeholders = ",".join("?" * len(event_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(event_types)

                if actor_id is not None:
                    query += " AND actor_id = ?"
                    params.append(actor_id)

                if target_type:
                    query += " AND target_type = ?"
                    params.append(target_type)

                if target_id:
                    query += " AND target_id = ?"
                    params.append(target_id)

                if since:
                    query += " AND timestamp >= ?"
                    params.append(since)

                if until:
                    query += " AND timestamp <= ?"
                    params.append(until)

                query += " ORDER BY id DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)

                return [
                    AuditEntry(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        event_type=row["event_type"],
                        actor_id=row["actor_id"],
                        actor_username=row["actor_username"],
                        actor_ip=row["actor_ip"],
                        target_type=row["target_type"],
                        target_id=row["target_id"],
                        action=row["action"],
                        details=json.loads(row["details"]),
                        previous_hash=row["previous_hash"],
                        entry_hash=row["entry_hash"],
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _query)

    async def verify_chain(
        self,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Verify the integrity of the audit log chain.

        Args:
            start_id: Start from this entry ID (default: beginning)
            end_id: End at this entry ID (default: most recent)

        Returns:
            Verification result with any broken links
        """
        loop = asyncio.get_running_loop()

        def _verify():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = "SELECT * FROM audit_log"
                params = []

                if start_id is not None or end_id is not None:
                    conditions = []
                    if start_id is not None:
                        conditions.append("id >= ?")
                        params.append(start_id)
                    if end_id is not None:
                        conditions.append("id <= ?")
                        params.append(end_id)
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY id ASC"
                cursor.execute(query, params)

                entries = cursor.fetchall()
                if not entries:
                    return {"valid": True, "entries_checked": 0, "broken_links": []}

                broken_links = []
                expected_previous = "genesis"

                for row in entries:
                    # Check previous hash link
                    if row["previous_hash"] != expected_previous:
                        broken_links.append(
                            {
                                "id": row["id"],
                                "expected_previous": expected_previous,
                                "actual_previous": row["previous_hash"],
                            }
                        )

                    # Verify this entry's hash
                    computed_hash = self._compute_entry_hash(
                        row["timestamp"],
                        row["event_type"],
                        row["actor_id"],
                        row["action"],
                        row["details"],
                        row["previous_hash"],
                    )

                    if computed_hash != row["entry_hash"]:
                        broken_links.append(
                            {
                                "id": row["id"],
                                "error": "hash_mismatch",
                                "expected_hash": computed_hash,
                                "actual_hash": row["entry_hash"],
                            }
                        )

                    expected_previous = row["entry_hash"]

                return {
                    "valid": len(broken_links) == 0,
                    "entries_checked": len(entries),
                    "broken_links": broken_links,
                    "first_id": entries[0]["id"] if entries else None,
                    "last_id": entries[-1]["id"] if entries else None,
                }
            finally:
                conn.close()

        return await loop.run_in_executor(None, _verify)

    async def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        loop = asyncio.get_running_loop()

        def _stats():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Total entries
                cursor.execute("SELECT COUNT(*) as count FROM audit_log")
                total = cursor.fetchone()["count"]

                # Entries by type
                cursor.execute(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM audit_log
                    GROUP BY event_type
                    ORDER BY count DESC
                """
                )
                by_type = {row["event_type"]: row["count"] for row in cursor.fetchall()}

                # Recent activity (last 24 hours)
                cursor.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM audit_log
                    WHERE timestamp >= datetime('now', '-24 hours')
                """
                )
                recent = cursor.fetchone()["count"]

                # First and last entry
                cursor.execute("SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM audit_log")
                row = cursor.fetchone()

                return {
                    "total_entries": total,
                    "by_type": by_type,
                    "last_24h": recent,
                    "first_entry": row["first"],
                    "last_entry": row["last"],
                }
            finally:
                conn.close()

        return await loop.run_in_executor(None, _stats)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._init_lock:
            cls._instance = None


# Singleton accessor
_audit_store: Optional[AuditStore] = None


def get_audit_store() -> AuditStore:
    """Get the global audit store instance."""
    global _audit_store
    if _audit_store is None:
        _audit_store = AuditStore()
    return _audit_store


# Convenience function for quick logging
async def audit_log(
    event_type: AuditEventType,
    action: str,
    **kwargs,
) -> int:
    """Log an audit event (convenience wrapper)."""
    store = get_audit_store()
    return await store.log(event_type, action, **kwargs)
