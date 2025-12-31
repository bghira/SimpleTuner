"""Upload progress tracking storage.

This module handles tracking of data upload progress during
job submission, using SQLite for multi-worker safety.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseSQLiteStore, get_default_db_path

logger = logging.getLogger(__name__)


class UploadProgressStore(BaseSQLiteStore):
    """Storage for upload progress tracking.

    Provides thread/worker-safe progress tracking for data uploads
    during cloud job submission.
    """

    def _get_default_db_path(self) -> Path:
        return get_default_db_path("jobs.db")

    def _init_schema(self) -> None:
        """Initialize the upload_progress table schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_progress (
                    upload_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    current INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0,
                    percent REAL NOT NULL DEFAULT 0,
                    message TEXT,
                    done INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_upload_progress_updated_at
                ON upload_progress(updated_at)
            """
            )

            conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize upload_progress schema: %s", exc)
            raise
        finally:
            conn.close()

    def update(
        self,
        upload_id: str,
        stage: str,
        current: int,
        total: int,
        message: Optional[str] = None,
        done: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Update upload progress state.

        Args:
            upload_id: Unique identifier for this upload.
            stage: Current stage (e.g., "packaging", "uploading").
            current: Current progress value.
            total: Total progress value.
            message: Optional progress message.
            done: Whether upload is complete.
            error: Optional error message.
        """
        percent = round((current / total * 100) if total > 0 else 0, 1)
        updated_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO upload_progress
                    (upload_id, stage, current, total, percent, message, done, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(upload_id) DO UPDATE SET
                    stage = excluded.stage,
                    current = excluded.current,
                    total = excluded.total,
                    percent = excluded.percent,
                    message = excluded.message,
                    done = excluded.done,
                    error = excluded.error,
                    updated_at = excluded.updated_at
            """,
                (upload_id, stage, current, total, percent, message, 1 if done else 0, error, updated_at),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to write upload progress for %s: %s", upload_id, exc)
        finally:
            conn.close()

    def get(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload progress state.

        Args:
            upload_id: Unique identifier for this upload.

        Returns:
            Progress dictionary or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT stage, current, total, percent, message, done, error, updated_at
                FROM upload_progress
                WHERE upload_id = ?
            """,
                (upload_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "stage": row["stage"],
                "current": row["current"],
                "total": row["total"],
                "percent": row["percent"],
                "message": row["message"],
                "done": bool(row["done"]),
                "error": row["error"],
                "updated_at": row["updated_at"],
            }
        except Exception as exc:
            logger.warning("Failed to read upload progress for %s: %s", upload_id, exc)
            return None
        finally:
            conn.close()

    def delete(self, upload_id: str) -> bool:
        """Remove upload progress entry.

        Args:
            upload_id: Unique identifier for this upload.

        Returns:
            True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM upload_progress WHERE upload_id = ?", (upload_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as exc:
            logger.warning("Failed to cleanup upload progress for %s: %s", upload_id, exc)
            return False
        finally:
            conn.close()

    def cleanup_stale(self, max_age_minutes: int = 60) -> int:
        """Remove stale upload progress entries.

        Args:
            max_age_minutes: Maximum age in minutes before cleanup.

        Returns:
            Number of entries removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        cutoff_iso = cutoff.isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM upload_progress
                WHERE updated_at < ?
            """,
                (cutoff_iso,),
            )
            conn.commit()
            return cursor.rowcount
        except Exception as exc:
            logger.warning("Failed to cleanup stale upload progress: %s", exc)
            return 0
        finally:
            conn.close()


# Singleton accessor
_upload_progress_store: Optional[UploadProgressStore] = None


def get_upload_progress_store() -> UploadProgressStore:
    """Get the singleton UploadProgressStore instance."""
    global _upload_progress_store
    if _upload_progress_store is None:
        _upload_progress_store = UploadProgressStore()
    return _upload_progress_store
