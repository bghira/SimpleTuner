"""SQLite-based storage for notification channels, preferences, and delivery logs."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ChannelConfig, NotificationLog, NotificationPreference, PendingResponse
from .protocols import ChannelType, DeliveryStatus, NotificationEventType

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class NotificationStore:
    """SQLite-based storage for notification system.

    Manages notification channels, preferences, delivery logs, and pending responses.
    Thread-safe with WAL mode for concurrent access.
    """

    _instance: Optional["NotificationStore"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "NotificationStore":
        """Singleton pattern - one notification store per process."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the notification store."""
        if getattr(self, "_initialized", False):
            return

        if db_path is None:
            from ..container import get_job_store

            job_store = get_job_store()
            db_path = job_store._db_path.parent / "notification.db"

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
            # Notification channels table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    is_enabled INTEGER DEFAULT 1,
                    config_json TEXT DEFAULT '{}',

                    -- SMTP settings
                    smtp_host TEXT,
                    smtp_port INTEGER DEFAULT 587,
                    smtp_username TEXT,
                    smtp_password_key TEXT,
                    smtp_use_tls INTEGER DEFAULT 1,
                    smtp_from_address TEXT,
                    smtp_from_name TEXT,

                    -- Webhook settings
                    webhook_url TEXT,
                    webhook_secret_key TEXT,

                    -- IMAP settings
                    imap_enabled INTEGER DEFAULT 0,
                    imap_host TEXT,
                    imap_port INTEGER DEFAULT 993,
                    imap_username TEXT,
                    imap_password_key TEXT,
                    imap_use_ssl INTEGER DEFAULT 1,
                    imap_folder TEXT DEFAULT 'INBOX',

                    -- Metadata
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """
            )

            # Notification preferences table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    event_type TEXT NOT NULL,
                    channel_id INTEGER NOT NULL,
                    is_enabled INTEGER DEFAULT 1,
                    recipients_json TEXT DEFAULT '[]',
                    min_severity TEXT DEFAULT 'info',
                    FOREIGN KEY (channel_id) REFERENCES notification_channels(id)
                        ON DELETE CASCADE
                )
                """
            )

            # Notification log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    job_id TEXT,
                    user_id INTEGER,
                    recipient TEXT NOT NULL,
                    delivery_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    sent_at TEXT NOT NULL,
                    delivered_at TEXT,
                    provider_message_id TEXT,
                    FOREIGN KEY (channel_id) REFERENCES notification_channels(id)
                        ON DELETE CASCADE
                )
                """
            )

            # Pending responses table (for email reply correlation)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_token TEXT UNIQUE NOT NULL,
                    approval_request_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    authorized_senders TEXT DEFAULT '[]',
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (channel_id) REFERENCES notification_channels(id)
                        ON DELETE CASCADE
                )
                """
            )

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_type ON notification_channels(channel_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_enabled ON notification_channels(is_enabled)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prefs_event ON notification_preferences(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prefs_channel ON notification_preferences(channel_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_status ON notification_log(delivery_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_job ON notification_log(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_sent ON notification_log(sent_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_token ON pending_responses(response_token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_expires ON pending_responses(expires_at)")

            logger.info("Created notification schema v%d", SCHEMA_VERSION)

    # --- Channel Management ---

    async def create_channel(self, channel: ChannelConfig) -> int:
        """Create a new notification channel.

        Args:
            channel: Channel configuration

        Returns:
            ID of the created channel
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            INSERT INTO notification_channels (
                channel_type, name, is_enabled, config_json,
                smtp_host, smtp_port, smtp_username, smtp_password_key,
                smtp_use_tls, smtp_from_address, smtp_from_name,
                webhook_url, webhook_secret_key,
                imap_enabled, imap_host, imap_port, imap_username,
                imap_password_key, imap_use_ssl, imap_folder,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel.channel_type.value,
                channel.name,
                1 if channel.is_enabled else 0,
                json.dumps(channel.config),
                channel.smtp_host,
                channel.smtp_port,
                channel.smtp_username,
                channel.smtp_password_key,
                1 if channel.smtp_use_tls else 0,
                channel.smtp_from_address,
                channel.smtp_from_name,
                channel.webhook_url,
                channel.webhook_secret_key,
                1 if channel.imap_enabled else 0,
                channel.imap_host,
                channel.imap_port,
                channel.imap_username,
                channel.imap_password_key,
                1 if channel.imap_use_ssl else 0,
                channel.imap_folder,
                now,
                now,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    async def get_channel(self, channel_id: int) -> Optional[ChannelConfig]:
        """Get a channel by ID.

        Args:
            channel_id: Channel ID

        Returns:
            Channel configuration or None if not found
        """
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM notification_channels WHERE id = ?", (channel_id,)).fetchone()

        if not row:
            return None

        return self._row_to_channel(row)

    async def update_channel(self, channel_id: int, updates: Dict[str, Any]) -> bool:
        """Update a channel.

        Args:
            channel_id: Channel ID
            updates: Fields to update

        Returns:
            True if channel was updated
        """
        conn = self._get_connection()

        # Build update query dynamically
        allowed_fields = {
            "name",
            "is_enabled",
            "config",
            "smtp_host",
            "smtp_port",
            "smtp_username",
            "smtp_password_key",
            "smtp_use_tls",
            "smtp_from_address",
            "smtp_from_name",
            "webhook_url",
            "webhook_secret_key",
            "imap_enabled",
            "imap_host",
            "imap_port",
            "imap_username",
            "imap_password_key",
            "imap_use_ssl",
            "imap_folder",
        }

        set_clauses = []
        values = []
        for field, value in updates.items():
            if field not in allowed_fields:
                continue

            if field == "config":
                set_clauses.append("config_json = ?")
                values.append(json.dumps(value))
            elif field in ("is_enabled", "smtp_use_tls", "imap_enabled", "imap_use_ssl"):
                set_clauses.append(f"{field} = ?")
                values.append(1 if value else 0)
            else:
                set_clauses.append(f"{field} = ?")
                values.append(value)

        if not set_clauses:
            return False

        set_clauses.append("updated_at = ?")
        values.append(datetime.now(timezone.utc).isoformat())
        values.append(channel_id)

        result = conn.execute(
            f"UPDATE notification_channels SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        conn.commit()
        return result.rowcount > 0

    async def delete_channel(self, channel_id: int) -> bool:
        """Delete a channel.

        Args:
            channel_id: Channel ID

        Returns:
            True if channel was deleted
        """
        conn = self._get_connection()
        result = conn.execute("DELETE FROM notification_channels WHERE id = ?", (channel_id,))
        conn.commit()
        return result.rowcount > 0

    async def list_channels(
        self, enabled_only: bool = False, channel_type: Optional[ChannelType] = None
    ) -> List[ChannelConfig]:
        """List notification channels.

        Args:
            enabled_only: Only return enabled channels
            channel_type: Filter by channel type

        Returns:
            List of channel configurations
        """
        conn = self._get_connection()

        query = "SELECT * FROM notification_channels WHERE 1=1"
        params: List[Any] = []

        if enabled_only:
            query += " AND is_enabled = 1"
        if channel_type:
            query += " AND channel_type = ?"
            params.append(channel_type.value)

        query += " ORDER BY created_at DESC"

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_channel(row) for row in rows]

    def _row_to_channel(self, row: sqlite3.Row) -> ChannelConfig:
        """Convert a database row to ChannelConfig."""
        return ChannelConfig(
            id=row["id"],
            channel_type=ChannelType(row["channel_type"]),
            name=row["name"],
            is_enabled=bool(row["is_enabled"]),
            config=json.loads(row["config_json"] or "{}"),
            smtp_host=row["smtp_host"],
            smtp_port=row["smtp_port"],
            smtp_username=row["smtp_username"],
            smtp_password_key=row["smtp_password_key"],
            smtp_use_tls=bool(row["smtp_use_tls"]),
            smtp_from_address=row["smtp_from_address"],
            smtp_from_name=row["smtp_from_name"],
            webhook_url=row["webhook_url"],
            webhook_secret_key=row["webhook_secret_key"],
            imap_enabled=bool(row["imap_enabled"]),
            imap_host=row["imap_host"],
            imap_port=row["imap_port"],
            imap_username=row["imap_username"],
            imap_password_key=row["imap_password_key"],
            imap_use_ssl=bool(row["imap_use_ssl"]),
            imap_folder=row["imap_folder"] or "INBOX",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # --- Preference Management ---

    async def set_preference(self, preference: NotificationPreference) -> int:
        """Create or update a notification preference.

        Args:
            preference: Preference to save

        Returns:
            ID of the preference
        """
        conn = self._get_connection()

        if preference.id > 0:
            # Update existing
            conn.execute(
                """
                UPDATE notification_preferences
                SET is_enabled = ?, recipients_json = ?, min_severity = ?
                WHERE id = ?
                """,
                (
                    1 if preference.is_enabled else 0,
                    json.dumps(preference.recipients),
                    preference.min_severity,
                    preference.id,
                ),
            )
            conn.commit()
            return preference.id

        # Create new
        cursor = conn.execute(
            """
            INSERT INTO notification_preferences (
                user_id, event_type, channel_id, is_enabled,
                recipients_json, min_severity
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                preference.user_id,
                preference.event_type.value,
                preference.channel_id,
                1 if preference.is_enabled else 0,
                json.dumps(preference.recipients),
                preference.min_severity,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    async def get_preferences(self, user_id: Optional[int] = None) -> List[NotificationPreference]:
        """Get notification preferences.

        Args:
            user_id: Filter by user ID (None returns global preferences)

        Returns:
            List of preferences
        """
        conn = self._get_connection()

        if user_id is not None:
            rows = conn.execute("SELECT * FROM notification_preferences WHERE user_id = ?", (user_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM notification_preferences WHERE user_id IS NULL").fetchall()

        return [self._row_to_preference(row) for row in rows]

    async def get_preferences_for_event(self, event_type: NotificationEventType) -> List[NotificationPreference]:
        """Get preferences for a specific event type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of preferences for this event
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT p.* FROM notification_preferences p
            JOIN notification_channels c ON p.channel_id = c.id
            WHERE p.event_type = ? AND p.is_enabled = 1 AND c.is_enabled = 1
            """,
            (event_type.value,),
        ).fetchall()
        return [self._row_to_preference(row) for row in rows]

    async def delete_preference(self, preference_id: int) -> bool:
        """Delete a preference.

        Args:
            preference_id: Preference ID

        Returns:
            True if deleted
        """
        conn = self._get_connection()
        result = conn.execute("DELETE FROM notification_preferences WHERE id = ?", (preference_id,))
        conn.commit()
        return result.rowcount > 0

    def _row_to_preference(self, row: sqlite3.Row) -> NotificationPreference:
        """Convert a database row to NotificationPreference."""
        return NotificationPreference(
            id=row["id"],
            user_id=row["user_id"],
            event_type=NotificationEventType(row["event_type"]),
            channel_id=row["channel_id"],
            is_enabled=bool(row["is_enabled"]),
            recipients=json.loads(row["recipients_json"] or "[]"),
            min_severity=row["min_severity"],
        )

    # --- Delivery Log ---

    async def log_delivery(
        self,
        channel_id: int,
        event_type: NotificationEventType,
        recipient: str,
        status: DeliveryStatus,
        job_id: Optional[str] = None,
        user_id: Optional[int] = None,
        error_message: Optional[str] = None,
        provider_message_id: Optional[str] = None,
    ) -> int:
        """Log a notification delivery attempt.

        Args:
            channel_id: Channel used
            event_type: Type of event
            recipient: Recipient address
            status: Delivery status
            job_id: Associated job ID
            user_id: Associated user ID
            error_message: Error message if failed
            provider_message_id: Message ID from provider

        Returns:
            ID of the log entry
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            INSERT INTO notification_log (
                channel_id, event_type, job_id, user_id, recipient,
                delivery_status, error_message, sent_at, provider_message_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                event_type.value,
                job_id,
                user_id,
                recipient,
                status.value,
                error_message,
                now,
                provider_message_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    async def update_delivery_status(
        self,
        log_id: int,
        status: DeliveryStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update delivery status.

        Args:
            log_id: Log entry ID
            status: New status
            error_message: Error message if failed

        Returns:
            True if updated
        """
        conn = self._get_connection()

        if status == DeliveryStatus.DELIVERED:
            result = conn.execute(
                """
                UPDATE notification_log
                SET delivery_status = ?, delivered_at = ?
                WHERE id = ?
                """,
                (status.value, datetime.now(timezone.utc).isoformat(), log_id),
            )
        else:
            result = conn.execute(
                """
                UPDATE notification_log
                SET delivery_status = ?, error_message = ?
                WHERE id = ?
                """,
                (status.value, error_message, log_id),
            )
        conn.commit()
        return result.rowcount > 0

    async def get_delivery_history(
        self,
        limit: int = 50,
        offset: int = 0,
        job_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        status: Optional[DeliveryStatus] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get delivery history with channel names.

        Args:
            limit: Maximum number of records
            offset: Number of records to skip for pagination
            job_id: Filter by job ID
            channel_id: Filter by channel
            status: Filter by status
            start_date: Filter entries on or after this date (ISO format)
            end_date: Filter entries on or before this date (ISO format)

        Returns:
            List of log entries with channel_name included
        """
        conn = self._get_connection()

        query = """
            SELECT l.*, c.name as channel_name
            FROM notification_log l
            LEFT JOIN notification_channels c ON l.channel_id = c.id
            WHERE 1=1
        """
        params: List[Any] = []

        if job_id:
            query += " AND l.job_id = ?"
            params.append(job_id)
        if channel_id:
            query += " AND l.channel_id = ?"
            params.append(channel_id)
        if status:
            query += " AND l.delivery_status = ?"
            params.append(status.value)
        if start_date:
            query += " AND l.sent_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND l.sent_at <= ?"
            params.append(end_date)

        query += " ORDER BY l.sent_at DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

        rows = conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            log = self._row_to_log(row)
            entry = log.to_dict()
            entry["channel_name"] = row["channel_name"]
            result.append(entry)
        return result

    def _row_to_log(self, row: sqlite3.Row) -> NotificationLog:
        """Convert a database row to NotificationLog."""
        return NotificationLog(
            id=row["id"],
            channel_id=row["channel_id"],
            event_type=row["event_type"],
            job_id=row["job_id"],
            user_id=row["user_id"],
            recipient=row["recipient"],
            delivery_status=row["delivery_status"],
            error_message=row["error_message"],
            sent_at=row["sent_at"],
            delivered_at=row["delivered_at"],
            provider_message_id=row["provider_message_id"],
        )

    async def cleanup_old_logs(self, days: int = 30) -> int:
        """Delete old log entries.

        Args:
            days: Delete entries older than this many days

        Returns:
            Number of deleted entries
        """
        conn = self._get_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        result = conn.execute("DELETE FROM notification_log WHERE sent_at < ?", (cutoff,))
        conn.commit()
        return result.rowcount

    # --- Pending Responses ---

    async def create_pending_response(
        self,
        response_token: str,
        approval_request_id: int,
        channel_id: int,
        authorized_senders: List[str],
        expires_hours: int = 24,
    ) -> int:
        """Create a pending response record for email reply tracking.

        Args:
            response_token: Unique token for correlation
            approval_request_id: Associated approval request
            channel_id: Channel that sent the notification
            authorized_senders: Email addresses authorized to respond
            expires_hours: Hours until expiration

        Returns:
            ID of the pending response
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=expires_hours)

        cursor = conn.execute(
            """
            INSERT INTO pending_responses (
                response_token, approval_request_id, channel_id,
                authorized_senders, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                response_token,
                approval_request_id,
                channel_id,
                json.dumps(authorized_senders),
                expires_at.isoformat(),
                now.isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    async def get_pending_by_token(self, response_token: str) -> Optional[PendingResponse]:
        """Get a pending response by token.

        Args:
            response_token: Response token

        Returns:
            PendingResponse or None if not found or expired
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        row = conn.execute(
            """
            SELECT * FROM pending_responses
            WHERE response_token = ? AND expires_at > ?
            """,
            (response_token, now),
        ).fetchone()

        if not row:
            return None

        return PendingResponse(
            id=row["id"],
            response_token=row["response_token"],
            approval_request_id=row["approval_request_id"],
            channel_id=row["channel_id"],
            authorized_senders=json.loads(row["authorized_senders"] or "[]"),
            expires_at=row["expires_at"],
            created_at=row["created_at"],
        )

    async def delete_pending_response(self, response_token: str) -> bool:
        """Delete a pending response after it's been processed.

        Args:
            response_token: Response token

        Returns:
            True if deleted
        """
        conn = self._get_connection()
        result = conn.execute("DELETE FROM pending_responses WHERE response_token = ?", (response_token,))
        conn.commit()
        return result.rowcount > 0

    async def expire_old_responses(self) -> int:
        """Delete expired pending responses.

        Returns:
            Number of deleted responses
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        result = conn.execute("DELETE FROM pending_responses WHERE expires_at < ?", (now,))
        conn.commit()
        return result.rowcount

    # --- Statistics ---

    async def get_channel_stats(self, channel_id: int, days: int = 30) -> Dict[str, Any]:
        """Get delivery statistics for a channel.

        Args:
            channel_id: Channel ID
            days: Number of days to include

        Returns:
            Statistics dict
        """
        conn = self._get_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN delivery_status = 'delivered' THEN 1 ELSE 0 END) as delivered,
                SUM(CASE WHEN delivery_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN delivery_status = 'bounced' THEN 1 ELSE 0 END) as bounced
            FROM notification_log
            WHERE channel_id = ? AND sent_at > ?
            """,
            (channel_id, cutoff),
        ).fetchone()

        return {
            "total": row["total"] or 0,
            "delivered": row["delivered"] or 0,
            "failed": row["failed"] or 0,
            "bounced": row["bounced"] or 0,
            "success_rate": ((row["delivered"] or 0) / row["total"] * 100 if row["total"] else 0),
        }

    async def get_overall_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get overall notification statistics.

        Args:
            days: Number of days to include

        Returns:
            Statistics dict
        """
        conn = self._get_connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN delivery_status = 'delivered' THEN 1 ELSE 0 END) as delivered,
                SUM(CASE WHEN delivery_status = 'failed' THEN 1 ELSE 0 END) as failed,
                COUNT(DISTINCT channel_id) as channels_used,
                COUNT(DISTINCT event_type) as event_types
            FROM notification_log
            WHERE sent_at > ?
            """,
            (cutoff,),
        ).fetchone()

        return {
            "total": row["total"] or 0,
            "delivered": row["delivered"] or 0,
            "failed": row["failed"] or 0,
            "channels_used": row["channels_used"] or 0,
            "event_types": row["event_types"] or 0,
            "success_rate": ((row["delivered"] or 0) / row["total"] * 100 if row["total"] else 0),
        }

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
            except Exception:
                pass
            del self._local.connection

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Properly closes the connection before resetting to avoid ResourceWarning.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None
