"""Async job storage facade composing specialized stores.

This module provides the AsyncJobStore class which serves as a unified
interface to all job-related storage operations. It composes specialized
stores for different concerns:

- JobRepository: Job CRUD operations
- IdempotencyStore: Deduplication via idempotency keys
- ReservationStore: Quota enforcement via slot reservations
- AuditStore: Audit logging
- MetricsStore: Aggregated metrics
- ProviderConfigStore: Provider configuration
- UploadProgressStore: Upload progress tracking
- BackupManager: Database backup/restore

For simple use cases, you can use the individual stores directly.
AsyncJobStore provides a unified API for code that needs multiple concerns.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CloudJobStatus, JobType, UnifiedJob
from .storage import (
    AuditStore,
    BackupManager,
    IdempotencyStore,
    JobRepository,
    MetricsStore,
    ProviderConfigStore,
    ReservationStore,
    UploadProgressStore,
    get_default_db_path,
)

logger = logging.getLogger(__name__)


class AsyncJobStore:
    """Facade for async job storage operations.

    Composes specialized stores to provide a unified API for:
    - Job CRUD operations
    - Idempotency key tracking
    - Job slot reservations
    - Audit logging
    - Metrics aggregation
    - Provider configuration
    - Upload progress tracking
    - Database backup/restore

    Use get_instance() for async initialization.
    """

    _instance: Optional["AsyncJobStore"] = None
    _init_lock: Optional[asyncio.Lock] = None
    _lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the store.

        Note: Use get_instance() for async initialization.
        """
        self._config_dir = Path(config_dir) if config_dir else self._resolve_config_dir()
        self._db_path = self._config_dir / "cloud" / "jobs.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Component stores (initialized lazily in _ensure_initialized)
        self._jobs: Optional[JobRepository] = None
        self._idempotency: Optional[IdempotencyStore] = None
        self._reservations: Optional[ReservationStore] = None
        self._audit: Optional[AuditStore] = None
        self._metrics: Optional[MetricsStore] = None
        self._provider_config: Optional[ProviderConfigStore] = None
        self._upload_progress: Optional[UploadProgressStore] = None
        self._backup: Optional[BackupManager] = None

        self._initialized = False

    @classmethod
    async def get_instance(cls, config_dir: Optional[Path] = None) -> "AsyncJobStore":
        """Get or create the singleton instance with async initialization."""
        # Fast path - instance already exists
        if cls._instance is not None:
            return cls._instance

        # Get current event loop
        current_loop = asyncio.get_running_loop()

        # Recreate lock if it's bound to a different loop (common in tests)
        if cls._init_lock is None or cls._lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._lock_loop = current_loop

        async with cls._init_lock:
            if cls._instance is None:
                instance = cls(config_dir)
                await instance._ensure_initialized()
                cls._instance = instance
            return cls._instance

    def _resolve_config_dir(self) -> Path:
        """Resolve config directory from WebUIStateStore or default.

        Returns the SimpleTuner root directory (e.g. /notebooks/simpletuner
        or ~/.simpletuner). The database path is then constructed as
        config_dir / "cloud" / "jobs.db".
        """
        try:
            from .webui_state import WebUIStateStore

            store = WebUIStateStore()
            defaults = store.load_defaults()
            if defaults.configs_dir:
                return Path(defaults.configs_dir)
            return store.base_dir.parent
        except Exception:
            from .storage.base import get_default_config_dir

            return get_default_config_dir()

    async def _ensure_initialized(self) -> None:
        """Ensure all component stores are initialized."""
        if self._initialized:
            return

        # Initialize component stores with shared db path
        self._jobs = JobRepository(self._db_path)
        self._provider_config = ProviderConfigStore(self._db_path)
        self._upload_progress = UploadProgressStore(self._db_path)

        # Async stores need explicit initialization
        self._idempotency = IdempotencyStore(self._db_path)
        await self._idempotency.ensure_initialized()

        self._reservations = ReservationStore(self._db_path)
        await self._reservations.ensure_initialized()

        self._audit = AuditStore(self._db_path)
        await self._audit.ensure_initialized()

        self._metrics = MetricsStore(self._db_path)
        await self._metrics.ensure_initialized()

        self._backup = BackupManager(self._db_path)

        self._initialized = True

    # ==================== Idempotency ====================

    async def check_idempotency_key(self, key: str, user_id: Optional[int] = None) -> Optional[str]:
        """Check if an idempotency key exists and return the associated job_id."""
        return await self._idempotency.check(key, user_id)

    async def store_idempotency_key(self, key: str, job_id: str, user_id: Optional[int] = None, ttl_hours: int = 24) -> None:
        """Store an idempotency key for deduplication."""
        await self._idempotency.store(key, job_id, user_id, ttl_hours)

    # ==================== Reservations ====================

    async def reserve_job_slot(self, user_id: int, max_concurrent: int, ttl_seconds: int = 300) -> Optional[str]:
        """Atomically reserve a job slot if quota allows."""
        return await self._reservations.reserve_slot(user_id, max_concurrent, ttl_seconds)

    async def consume_reservation(self, reservation_id: str) -> bool:
        """Mark a reservation as consumed (job was created)."""
        return await self._reservations.consume(reservation_id)

    async def release_reservation(self, reservation_id: str) -> bool:
        """Release a reservation (job submission failed)."""
        return await self._reservations.release(reservation_id)

    # ==================== Job CRUD ====================

    async def add_job(self, job: UnifiedJob) -> None:
        """Add a new job."""
        await self._jobs.add(job)

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing job."""
        return await self._jobs.update(job_id, updates)

    async def get_job(self, job_id: str) -> Optional[UnifiedJob]:
        """Get a job by ID."""
        return await self._jobs.get(job_id)

    async def get_job_by_upload_token(self, upload_token: str) -> Optional[UnifiedJob]:
        """Get a job by upload token."""
        return await self._jobs.get_by_upload_token(upload_token)

    async def list_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        job_type: Optional[JobType] = None,
        status: Optional[str] = None,
        user_id: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> List[UnifiedJob]:
        """List jobs with filtering."""
        return await self._jobs.list(limit, offset, job_type, status, user_id, provider)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        return await self._jobs.delete(job_id)

    async def cleanup_old_jobs(self, retention_days: int = 90) -> int:
        """Remove old jobs."""
        return await self._jobs.cleanup_old(retention_days)

    # ==================== Metrics ====================

    async def get_job_history_for_chart(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get job history aggregated by day."""
        return await self._metrics.get_daily_breakdown(days)

    async def get_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary metrics."""
        return await self._metrics.get_summary(days)

    # ==================== Provider Config ====================

    async def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get provider configuration."""
        return await self._provider_config.get(provider)

    async def save_provider_config(self, provider: str, config: Dict[str, Any]) -> None:
        """Save provider configuration."""
        await self._provider_config.save(provider, config)

    # ==================== Audit Log ====================

    async def log_audit_event(
        self,
        action: str,
        job_id: Optional[str] = None,
        provider: Optional[str] = None,
        config_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        user_ip: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Log an audit event."""
        await self._audit.log_event(action, job_id, provider, config_name, details, user_ip, user_id)

    async def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit entries."""
        return await self._audit.get_entries(limit)

    async def cleanup_audit_log(self, max_age_days: int = 90) -> int:
        """Remove old audit entries."""
        return await self._audit.cleanup(max_age_days)

    # ==================== Upload Progress ====================
    # Note: These are sync methods because they're called from sync callbacks

    def update_upload_progress(
        self,
        upload_id: str,
        stage: str,
        current: int,
        total: int,
        message: Optional[str] = None,
        done: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Update upload progress (sync - called from callbacks)."""
        self._upload_progress.update(upload_id, stage, current, total, message, done, error)

    def get_upload_progress(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload progress (sync)."""
        return self._upload_progress.get(upload_id)

    def cleanup_upload_progress(self, upload_id: str) -> bool:
        """Remove upload progress entry (sync)."""
        return self._upload_progress.delete(upload_id)

    def cleanup_stale_upload_progress(self, max_age_minutes: int = 60) -> int:
        """Remove stale upload progress entries (sync)."""
        return self._upload_progress.cleanup_stale(max_age_minutes)

    # ==================== Backup ====================

    async def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database."""
        return await self._backup.backup(backup_path)

    async def restore(self, backup_path: Path) -> bool:
        """Restore the database from a backup."""
        success = await self._backup.restore(backup_path)
        if success:
            # Reset singleton so next get_instance re-initializes
            AsyncJobStore._instance = None
        return success

    def list_backups(self) -> List[Path]:
        """List available backup files."""
        return self._backup.list_backups()

    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return await self._backup.get_database_info()

    # ==================== Lifecycle ====================

    async def close(self) -> None:
        """Close all database connections."""
        if self._idempotency:
            await self._idempotency.close()
        if self._reservations:
            await self._reservations.close()
        if self._audit:
            await self._audit.close()
        if self._metrics:
            await self._metrics.close()
        if self._backup:
            await self._backup.close()

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        # Get current event loop and ensure lock is valid
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._lock_loop = current_loop

        async with cls._init_lock:
            if cls._instance:
                await cls._instance.close()
            cls._instance = None

    def get_database_path(self) -> Path:
        """Get database path."""
        return self._db_path

    # ==================== Sync Compatibility ====================

    @classmethod
    def get_instance_sync(cls, config_dir: Optional[Path] = None) -> "AsyncJobStore":
        """Get instance synchronously (for backward compatibility).

        Prefer get_instance() in async contexts.
        """
        # Check if there's a running loop
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            # In async context - return existing instance or raise
            if cls._instance is not None:
                return cls._instance
            raise RuntimeError(
                "Cannot get_instance_sync from async context without existing instance. "
                "Use 'await AsyncJobStore.get_instance()' instead."
            )
        else:
            # No running loop - safe to use asyncio.run
            return asyncio.run(cls.get_instance(config_dir))
