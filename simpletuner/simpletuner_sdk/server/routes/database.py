"""Database health and management endpoints.

Provides endpoints for database health checks, verification, and management.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.cloud.auth import require_permission
from ..services.cloud.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/database", tags=["database"])


# --- Request/Response Models ---


class DatabaseInfo(BaseModel):
    """Information about a database."""

    name: str
    path: str
    exists: bool
    size_bytes: int = 0
    last_modified: Optional[str] = None
    table_count: int = 0
    tables: List[str] = Field(default_factory=list)


class DatabaseHealthResponse(BaseModel):
    """Response for database health check."""

    healthy: bool
    databases: List[DatabaseInfo]
    total_size_bytes: int
    issues: List[str] = Field(default_factory=list)


class DatabaseVerifyResponse(BaseModel):
    """Response for database verification."""

    database: str
    valid: bool
    integrity_check: str
    foreign_key_check: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class DatabaseVacuumResponse(BaseModel):
    """Response for database vacuum operation."""

    database: str
    size_before: int
    size_after: int
    freed_bytes: int
    success: bool


class MigrationInfo(BaseModel):
    """Information about a database migration."""

    version: str
    name: str
    applied_at: Optional[str] = None
    status: str  # "applied", "pending", "failed"


class MigrationStatusResponse(BaseModel):
    """Response for migration status."""

    current_version: Optional[str]
    pending_migrations: List[MigrationInfo]
    applied_migrations: List[MigrationInfo]


# --- Database paths ---

DATABASE_PATHS = {
    "jobs": Path("data/cloud_jobs.db"),
    "auth": Path("data/auth.db"),
    "approvals": Path("data/approvals.db"),
    "notifications": Path("data/notifications.db"),
    "webui": Path("data/webui.db"),
}


# --- Helper Functions ---


def _get_db_info(name: str, path: Path) -> DatabaseInfo:
    """Get information about a database."""
    if not path.exists():
        return DatabaseInfo(
            name=name,
            path=str(path),
            exists=False,
        )

    stat = path.stat()
    tables = []
    table_count = 0

    try:
        import sqlite3

        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        table_count = len(tables)
        conn.close()
    except Exception as exc:
        logger.warning("Could not read tables from %s: %s", path, exc)

    return DatabaseInfo(
        name=name,
        path=str(path),
        exists=True,
        size_bytes=stat.st_size,
        last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        table_count=table_count,
        tables=tables,
    )


# --- Endpoints ---


@router.get("/health", response_model=DatabaseHealthResponse)
async def health_check(
    user: User = Depends(require_permission("admin.database")),
) -> DatabaseHealthResponse:
    """Check health of all databases.

    Verifies that all expected databases exist and are accessible.

    Requires admin.database permission.
    """
    databases = []
    total_size = 0
    issues = []

    for name, path in DATABASE_PATHS.items():
        info = _get_db_info(name, path)
        databases.append(info)

        if info.exists:
            total_size += info.size_bytes

            # Check if we can open the database
            try:
                import sqlite3

                conn = sqlite3.connect(str(path))
                conn.execute("SELECT 1")
                conn.close()
            except Exception as exc:
                issues.append(f"{name}: Cannot open database - {exc}")

        else:
            # Not all databases are required, so just note which ones don't exist
            if name in ("jobs", "auth"):
                issues.append(f"{name}: Database does not exist (may need initialization)")

    healthy = len([i for i in issues if "Cannot open" in i]) == 0

    return DatabaseHealthResponse(
        healthy=healthy,
        databases=databases,
        total_size_bytes=total_size,
        issues=issues,
    )


@router.get("/verify/{database}", response_model=DatabaseVerifyResponse)
async def verify_database(
    database: str,
    user: User = Depends(require_permission("admin.database")),
) -> DatabaseVerifyResponse:
    """Verify database integrity.

    Runs SQLite integrity and foreign key checks on the specified database.

    Requires admin.database permission.
    """
    if database not in DATABASE_PATHS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown database: {database}. Available: {list(DATABASE_PATHS.keys())}",
        )

    path = DATABASE_PATHS[database]
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database does not exist: {database}",
        )

    issues = []

    try:
        import sqlite3

        conn = sqlite3.connect(str(path))

        # Run integrity check
        cursor = conn.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]

        if integrity_result != "ok":
            issues.append(f"Integrity check failed: {integrity_result}")

        # Run foreign key check
        cursor = conn.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        fk_strings = []
        for row in fk_issues:
            fk_strings.append(f"Table {row[0]}: row {row[1]} references missing {row[2]}.{row[3]}")
            issues.append(f"Foreign key violation in {row[0]}")

        conn.close()

        return DatabaseVerifyResponse(
            database=database,
            valid=len(issues) == 0,
            integrity_check=integrity_result,
            foreign_key_check=fk_strings,
            issues=issues,
        )

    except Exception as exc:
        logger.error("Database verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {exc}",
        )


@router.post("/vacuum/{database}", response_model=DatabaseVacuumResponse)
async def vacuum_database(
    database: str,
    user: User = Depends(require_permission("admin.database")),
) -> DatabaseVacuumResponse:
    """Vacuum a database to reclaim space.

    This operation may take a while for large databases and will briefly
    lock the database for writes.

    Requires admin.database permission.
    """
    if database not in DATABASE_PATHS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown database: {database}. Available: {list(DATABASE_PATHS.keys())}",
        )

    path = DATABASE_PATHS[database]
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database does not exist: {database}",
        )

    size_before = path.stat().st_size

    try:
        import sqlite3

        conn = sqlite3.connect(str(path))
        conn.execute("VACUUM")
        conn.close()

        size_after = path.stat().st_size

        logger.info(
            "Vacuumed database %s: %d -> %d bytes (freed %d)", database, size_before, size_after, size_before - size_after
        )

        return DatabaseVacuumResponse(
            database=database,
            size_before=size_before,
            size_after=size_after,
            freed_bytes=size_before - size_after,
            success=True,
        )

    except Exception as exc:
        logger.error("Vacuum failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vacuum failed: {exc}",
        )


@router.get("/migrations", response_model=MigrationStatusResponse)
async def get_migration_status(
    database: str = Query("auth", description="Database to check migrations for"),
    user: User = Depends(require_permission("admin.database")),
) -> MigrationStatusResponse:
    """Get database migration status.

    Shows which migrations have been applied and which are pending.

    Requires admin.database permission.
    """
    if database not in DATABASE_PATHS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown database: {database}",
        )

    path = DATABASE_PATHS[database]

    applied = []
    pending = []
    current_version = None

    if path.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(path))

            # Check if migrations table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
            if cursor.fetchone():
                cursor = conn.execute("SELECT version, name, applied_at FROM schema_migrations ORDER BY version")
                for row in cursor.fetchall():
                    applied.append(
                        MigrationInfo(
                            version=row[0],
                            name=row[1] or f"migration_{row[0]}",
                            applied_at=row[2],
                            status="applied",
                        )
                    )
                    current_version = row[0]

            conn.close()

        except Exception as exc:
            logger.warning("Could not read migration status: %s", exc)

    # Check for pending migrations in migrations directory
    migrations_dir = Path(f"migrations/{database}")
    if migrations_dir.exists():
        applied_versions = {m.version for m in applied}

        for migration_file in sorted(migrations_dir.glob("*.sql")):
            version = migration_file.stem.split("_")[0]
            name = migration_file.stem

            if version not in applied_versions:
                pending.append(
                    MigrationInfo(
                        version=version,
                        name=name,
                        status="pending",
                    )
                )

    return MigrationStatusResponse(
        current_version=current_version,
        pending_migrations=pending,
        applied_migrations=applied,
    )


@router.post("/migrations/run")
async def run_migrations(
    database: str = Query("auth", description="Database to run migrations for"),
    dry_run: bool = Query(False, description="If true, only show what would be run"),
    user: User = Depends(require_permission("admin.database")),
) -> Dict[str, Any]:
    """Run pending database migrations.

    WARNING: This modifies the database schema. Make a backup first.

    Requires admin.database permission.
    """
    if database not in DATABASE_PATHS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown database: {database}",
        )

    migrations_dir = Path(f"migrations/{database}")
    if not migrations_dir.exists():
        return {
            "success": True,
            "message": "No migrations directory found",
            "migrations_run": [],
        }

    path = DATABASE_PATHS[database]

    # Get applied migrations
    applied_versions = set()
    if path.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(path))
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
            if cursor.fetchone():
                cursor = conn.execute("SELECT version FROM schema_migrations")
                applied_versions = {row[0] for row in cursor.fetchall()}
            conn.close()
        except Exception:
            pass

    # Find pending migrations
    pending = []
    for migration_file in sorted(migrations_dir.glob("*.sql")):
        version = migration_file.stem.split("_")[0]
        if version not in applied_versions:
            pending.append(migration_file)

    if not pending:
        return {
            "success": True,
            "message": "No pending migrations",
            "migrations_run": [],
        }

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "message": f"Would run {len(pending)} migrations",
            "pending": [p.name for p in pending],
        }

    # Run migrations
    import sqlite3

    conn = sqlite3.connect(str(path))

    # Ensure migrations table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            name TEXT,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    migrations_run = []
    try:
        for migration_file in pending:
            version = migration_file.stem.split("_")[0]
            name = migration_file.stem

            logger.info("Running migration: %s", name)

            sql = migration_file.read_text()
            conn.executescript(sql)

            conn.execute("INSERT INTO schema_migrations (version, name) VALUES (?, ?)", (version, name))
            conn.commit()

            migrations_run.append(name)

        logger.info("Ran %d migrations on %s", len(migrations_run), database)

        return {
            "success": True,
            "message": f"Ran {len(migrations_run)} migrations",
            "migrations_run": migrations_run,
        }

    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {exc}. Rolled back.",
        )

    finally:
        conn.close()


@router.get("/stats")
async def get_database_stats(
    user: User = Depends(require_permission("admin.database")),
) -> Dict[str, Any]:
    """Get statistics about all databases.

    Requires admin.database permission.
    """
    stats = {}

    for name, path in DATABASE_PATHS.items():
        if not path.exists():
            stats[name] = {"exists": False}
            continue

        try:
            import sqlite3

            conn = sqlite3.connect(str(path))

            db_stats = {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "tables": {},
            }

            # Get table names
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]

            # Get row counts for each table
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    db_stats["tables"][table] = {"row_count": count}
                except Exception:
                    db_stats["tables"][table] = {"error": "Could not count rows"}

            # Get page count and page size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]

            db_stats["page_count"] = page_count
            db_stats["page_size"] = page_size
            db_stats["freelist_count"] = conn.execute("PRAGMA freelist_count").fetchone()[0]

            conn.close()
            stats[name] = db_stats

        except Exception as exc:
            stats[name] = {"exists": True, "error": str(exc)}

    return stats
