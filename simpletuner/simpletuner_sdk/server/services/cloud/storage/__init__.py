"""Storage layer for cloud services.

This package provides separated repository classes for different data domains:

Core Infrastructure:
- BaseSQLiteStore: Sync base class with run_in_executor for async
- AsyncSQLiteStore: Native async base class using aiosqlite

Domain Stores:
- JobRepository: Job CRUD operations
- ProviderConfigStore: Provider-specific configuration
- UploadProgressStore: Upload progress tracking
- IdempotencyStore: Idempotency key tracking for deduplication
- ReservationStore: Job slot reservations for quota enforcement
- AuditStore: Audit log storage
- MetricsStore: Aggregated job metrics
- BackupManager: Database backup/restore operations

Store Architecture
------------------
All stores use AsyncSQLiteStore for native async database access.
Use the get_*_store() factory functions for singleton access.

Example::

    from .storage import get_idempotency_store, get_audit_store

    async def my_handler():
        idem_store = await get_idempotency_store()
        existing = await idem_store.check(key)
"""

from .async_base import AsyncSQLiteStore, AsyncStoreMixin
from .audit_store import AuditStore, get_audit_store
from .backup_manager import BackupManager, get_backup_manager
from .base import BaseSQLiteStore, get_default_db_path
from .idempotency_store import IdempotencyStore, get_idempotency_store
from .job_repository import JobRepository, get_job_repository
from .metrics_store import MetricsStore, get_metrics_store
from .provider_config_store import ProviderConfigStore, get_provider_config_store
from .reservation_store import ReservationStore, get_reservation_store
from .upload_progress_store import UploadProgressStore, get_upload_progress_store

__all__ = [
    # Base classes
    "AsyncSQLiteStore",
    "AsyncStoreMixin",
    "BaseSQLiteStore",
    "get_default_db_path",
    # Domain stores
    "JobRepository",
    "get_job_repository",
    "ProviderConfigStore",
    "get_provider_config_store",
    "UploadProgressStore",
    "get_upload_progress_store",
    "IdempotencyStore",
    "get_idempotency_store",
    "ReservationStore",
    "get_reservation_store",
    "AuditStore",
    "get_audit_store",
    "MetricsStore",
    "get_metrics_store",
    "BackupManager",
    "get_backup_manager",
]
