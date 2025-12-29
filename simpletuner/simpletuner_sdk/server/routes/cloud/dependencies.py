"""FastAPI dependency providers for cloud routes.

Provides dependency injection for job store and related services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...services.cloud import AsyncJobStore


def get_job_store() -> "AsyncJobStore":
    """Get the singleton AsyncJobStore instance via DI container.

    This function can be used directly or with FastAPI's Depends():
        store = get_job_store()  # Direct call
        async def handler(store: AsyncJobStore = Depends(get_job_store)):  # Injection

    For testing, use the container to override:
        from ...services.cloud.container import set_job_store
        set_job_store(mock_store)
    """
    from ...services.cloud.container import get_job_store as _get_store

    return _get_store()


async def get_async_job_store() -> "AsyncJobStore":
    """Get the async job store for atomic operations.

    The async store provides:
    - Idempotency key checking/storage
    - Atomic job reservation for quota enforcement
    - Native async database operations
    """
    from ...services.cloud.async_job_store import AsyncJobStore

    return await AsyncJobStore.get_instance()
