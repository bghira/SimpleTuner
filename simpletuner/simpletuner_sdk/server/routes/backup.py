"""Backup and restore endpoints for SimpleTuner data.

Provides endpoints for creating, listing, and restoring backups
of SimpleTuner databases and configuration.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.cloud.auth import require_permission
from ..services.cloud.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backup", tags=["backup"])

# Default backup location
BACKUP_DIR = Path(os.environ.get("SIMPLETUNER_BACKUP_DIR", "data/backups"))


# --- Request/Response Models ---


class BackupInfo(BaseModel):
    """Information about a backup."""

    id: str
    name: str
    created_at: str
    size_bytes: int
    components: List[str]
    description: Optional[str] = None
    created_by: Optional[int] = None


class BackupListResponse(BaseModel):
    """Response for listing backups."""

    backups: List[BackupInfo]
    total_size_bytes: int
    backup_dir: str


class CreateBackupRequest(BaseModel):
    """Request to create a backup."""

    name: Optional[str] = Field(None, description="Optional name for the backup")
    description: Optional[str] = Field(None, description="Optional description")
    components: List[str] = Field(
        default=["jobs", "auth", "config"], description="Components to backup: jobs, auth, config, approvals, notifications"
    )


class CreateBackupResponse(BaseModel):
    """Response from creating a backup."""

    success: bool
    backup_id: str
    path: str
    size_bytes: int
    components: List[str]


class RestoreBackupRequest(BaseModel):
    """Request to restore from a backup."""

    backup_id: str
    components: Optional[List[str]] = Field(None, description="Components to restore (default: all from backup)")
    dry_run: bool = Field(False, description="If true, only validate backup without restoring")


class RestoreBackupResponse(BaseModel):
    """Response from restoring a backup."""

    success: bool
    restored_components: List[str]
    dry_run: bool
    warnings: List[str] = Field(default_factory=list)


# --- Helper Functions ---


def _get_backup_dir() -> Path:
    """Get the backup directory, creating it if necessary."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def _generate_backup_id() -> str:
    """Generate a unique backup ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _get_backup_info(backup_path: Path) -> Optional[BackupInfo]:
    """Get information about a backup."""
    if not backup_path.exists() or not backup_path.is_dir():
        return None

    # Calculate total size
    total_size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())

    # Determine components from directory contents
    components = []
    component_files = {
        "jobs": "jobs.db",
        "auth": "auth.db",
        "config": "config.json",
        "approvals": "approvals.db",
        "notifications": "notifications.db",
    }
    for comp, filename in component_files.items():
        if (backup_path / filename).exists():
            components.append(comp)

    # Read metadata if it exists
    metadata_file = backup_path / "metadata.json"
    description = None
    created_by = None
    name = backup_path.name

    if metadata_file.exists():
        try:
            import json

            with open(metadata_file) as f:
                metadata = json.load(f)
                description = metadata.get("description")
                created_by = metadata.get("created_by")
                name = metadata.get("name", name)
        except Exception:
            pass

    # Get creation time from directory mtime
    created_at = datetime.fromtimestamp(backup_path.stat().st_mtime, tz=timezone.utc).isoformat()

    return BackupInfo(
        id=backup_path.name,
        name=name,
        created_at=created_at,
        size_bytes=total_size,
        components=components,
        description=description,
        created_by=created_by,
    )


# --- Endpoints ---


@router.get("", response_model=BackupListResponse)
async def list_backups(
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(require_permission("admin.backup")),
) -> BackupListResponse:
    """List available backups.

    Requires admin.backup permission.
    """
    backup_dir = _get_backup_dir()

    backups = []
    total_size = 0

    for path in sorted(backup_dir.iterdir(), reverse=True):
        if path.is_dir():
            info = _get_backup_info(path)
            if info:
                backups.append(info)
                total_size += info.size_bytes

                if len(backups) >= limit:
                    break

    return BackupListResponse(
        backups=backups,
        total_size_bytes=total_size,
        backup_dir=str(backup_dir),
    )


@router.post("", response_model=CreateBackupResponse)
async def create_backup(
    request: CreateBackupRequest,
    user: User = Depends(require_permission("admin.backup")),
) -> CreateBackupResponse:
    """Create a new backup.

    Components that can be backed up:
    - jobs: Job history and cloud job data
    - auth: User accounts, levels, permissions
    - config: Training configurations and settings
    - approvals: Approval rules and requests
    - notifications: Notification channels and preferences

    Requires admin.backup permission.
    """
    import json

    backup_dir = _get_backup_dir()
    backup_id = _generate_backup_id()

    if request.name:
        backup_id = f"{backup_id}_{request.name.replace(' ', '_')[:30]}"

    backup_path = backup_dir / backup_id
    backup_path.mkdir(parents=True, exist_ok=True)

    # Database paths (adjust as needed for your project)
    db_sources = {
        "jobs": Path("data/cloud_jobs.db"),
        "auth": Path("data/auth.db"),
        "approvals": Path("data/approvals.db"),
        "notifications": Path("data/notifications.db"),
    }

    backed_up = []

    try:
        for component in request.components:
            if component in db_sources:
                src = db_sources[component]
                if src.exists():
                    dst = backup_path / f"{component}.db"
                    shutil.copy2(src, dst)
                    backed_up.append(component)
                    logger.info("Backed up %s to %s", component, dst)

            elif component == "config":
                # Backup configuration files
                config_sources = [
                    Path("config"),
                    Path("data/webui_state.json"),
                ]
                config_backup = backup_path / "config"
                config_backup.mkdir(exist_ok=True)

                for config_src in config_sources:
                    if config_src.exists():
                        if config_src.is_dir():
                            shutil.copytree(config_src, config_backup / config_src.name, dirs_exist_ok=True)
                        else:
                            shutil.copy2(config_src, config_backup / config_src.name)

                backed_up.append("config")
                logger.info("Backed up config to %s", config_backup)

        # Write metadata
        metadata = {
            "backup_id": backup_id,
            "name": request.name or backup_id,
            "description": request.description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.id,
            "components": backed_up,
        }
        with open(backup_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())

        logger.info("Backup %s created by user %s (%d bytes)", backup_id, user.username, total_size)

        return CreateBackupResponse(
            success=True,
            backup_id=backup_id,
            path=str(backup_path),
            size_bytes=total_size,
            components=backed_up,
        )

    except Exception as exc:
        # Clean up on failure
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)
        logger.error("Backup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backup failed: {exc}",
        )


@router.get("/{backup_id}", response_model=BackupInfo)
async def get_backup(
    backup_id: str,
    user: User = Depends(require_permission("admin.backup")),
) -> BackupInfo:
    """Get information about a specific backup.

    Requires admin.backup permission.
    """
    # Validate backup_id to prevent path traversal
    if ".." in backup_id or "/" in backup_id or "\\" in backup_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid backup ID",
        )

    backup_dir = _get_backup_dir()
    backup_path = backup_dir / backup_id

    info = _get_backup_info(backup_path)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backup not found: {backup_id}",
        )

    return info


@router.post("/{backup_id}/restore", response_model=RestoreBackupResponse)
async def restore_backup(
    backup_id: str,
    request: RestoreBackupRequest,
    user: User = Depends(require_permission("admin.backup")),
) -> RestoreBackupResponse:
    """Restore from a backup.

    WARNING: This will overwrite existing data for the selected components.
    Use dry_run=true to validate the backup first.

    Requires admin.backup permission.
    """
    # Validate backup_id
    if ".." in backup_id or "/" in backup_id or "\\" in backup_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid backup ID",
        )

    backup_dir = _get_backup_dir()
    backup_path = backup_dir / backup_id

    if not backup_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backup not found: {backup_id}",
        )

    info = _get_backup_info(backup_path)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid backup format",
        )

    # Determine which components to restore
    components_to_restore = request.components or info.components

    # Validate all requested components exist in backup
    missing = set(components_to_restore) - set(info.components)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Components not in backup: {missing}",
        )

    warnings = []

    if request.dry_run:
        logger.info("Dry run: would restore %s from backup %s", components_to_restore, backup_id)
        return RestoreBackupResponse(
            success=True,
            restored_components=components_to_restore,
            dry_run=True,
            warnings=["Dry run - no changes made"],
        )

    # Database paths
    db_destinations = {
        "jobs": Path("data/cloud_jobs.db"),
        "auth": Path("data/auth.db"),
        "approvals": Path("data/approvals.db"),
        "notifications": Path("data/notifications.db"),
    }

    restored = []

    try:
        for component in components_to_restore:
            if component in db_destinations:
                src = backup_path / f"{component}.db"
                dst = db_destinations[component]

                if src.exists():
                    # Create backup of current data before overwriting
                    if dst.exists():
                        pre_restore_backup = dst.with_suffix(".db.pre_restore")
                        shutil.copy2(dst, pre_restore_backup)
                        warnings.append(f"Existing {component}.db backed up to {pre_restore_backup.name}")

                    # Restore
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    restored.append(component)
                    logger.info("Restored %s from backup %s", component, backup_id)

            elif component == "config":
                config_src = backup_path / "config"
                if config_src.exists():
                    # Restore config files
                    for item in config_src.iterdir():
                        if item.name == "config" and item.is_dir():
                            dst = Path("config")
                            shutil.copytree(item, dst, dirs_exist_ok=True)
                        elif item.name == "webui_state.json":
                            dst = Path("data/webui_state.json")
                            shutil.copy2(item, dst)

                    restored.append("config")
                    logger.info("Restored config from backup %s", backup_id)

        logger.info("Backup %s restored by user %s: %s", backup_id, user.username, restored)

        return RestoreBackupResponse(
            success=True,
            restored_components=restored,
            dry_run=False,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("Restore failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Restore failed: {exc}",
        )


@router.delete("/{backup_id}")
async def delete_backup(
    backup_id: str,
    user: User = Depends(require_permission("admin.backup")),
) -> Dict[str, Any]:
    """Delete a backup.

    Requires admin.backup permission.
    """
    # Validate backup_id
    if ".." in backup_id or "/" in backup_id or "\\" in backup_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid backup ID",
        )

    backup_dir = _get_backup_dir()
    backup_path = backup_dir / backup_id

    if not backup_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backup not found: {backup_id}",
        )

    try:
        shutil.rmtree(backup_path)
        logger.info("Backup %s deleted by user %s", backup_id, user.username)
        return {"success": True, "deleted": backup_id}

    except Exception as exc:
        logger.error("Failed to delete backup: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete backup: {exc}",
        )
