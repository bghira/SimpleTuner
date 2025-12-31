"""Protocol definitions for cloud services.

Protocols define the contracts that services must fulfill, enabling:
- Loose coupling between components
- Easy testing with mock implementations
- Clear documentation of service capabilities
- Type checking without inheritance

Usage:
    def process_job(store: JobStoreProtocol) -> None:
        # Works with any implementation that satisfies the protocol
        job = store.get_job("123")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .base import CloudJobInfo, CloudJobStatus, JobType, UnifiedJob


@runtime_checkable
class JobStoreProtocol(Protocol):
    """Protocol for job storage operations.

    Implementations handle persistence of job records.
    """

    async def get_job(self, job_id: str) -> Optional[UnifiedJob]:
        """Get a job by ID."""
        ...

    async def add_job(self, job: UnifiedJob) -> bool:
        """Add a new job."""
        ...

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing job."""
        ...

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        ...

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        job_type: Optional[JobType] = None,
        status: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[UnifiedJob]:
        """List jobs with optional filtering."""
        ...

    async def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a provider."""
        ...

    def log_audit_event(
        self,
        action: str,
        job_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an audit event."""
        ...


@runtime_checkable
class ProviderClientProtocol(Protocol):
    """Protocol for cloud provider clients.

    Implementations communicate with specific cloud providers.
    """

    async def run_job(
        self,
        config: Dict[str, Any],
        dataloader: List[Dict[str, Any]],
        data_archive_url: Optional[str] = None,
        webhook_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        hub_model_id: Optional[str] = None,
        lycoris_config: Optional[Dict[str, Any]] = None,
    ) -> CloudJobInfo:
        """Submit a job to the provider."""
        ...

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        """Get current status of a job."""
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        ...

    async def get_job_logs(self, job_id: str) -> str:
        """Get logs for a job."""
        ...

    async def list_jobs(self, limit: int = 100) -> List[CloudJobInfo]:
        """List recent jobs from the provider."""
        ...


# NOTE: QueueStoreProtocol moved to queue/protocol.py
# Import from there for the full interface with scheduling methods.
# This re-export is kept for backwards compatibility.
from .queue.protocol import QueueStoreProtocol


@runtime_checkable
class UserStoreProtocol(Protocol):
    """Protocol for user storage operations."""

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get a user by ID."""
        ...

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get a user by username."""
        ...

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
    ) -> int:
        """Create a new user. Returns user ID."""
        ...

    def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update a user."""
        ...

    def get_quotas(self, user_id: int) -> Dict[str, Any]:
        """Get quota settings for a user."""
        ...


@runtime_checkable
class SecretsProviderProtocol(Protocol):
    """Protocol for secrets management."""

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret by key."""
        ...

    def set_secret(self, key: str, value: str) -> bool:
        """Set a secret."""
        ...

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        ...

    def list_secrets(self) -> List[str]:
        """List available secret keys."""
        ...


@runtime_checkable
class UploadServiceProtocol(Protocol):
    """Protocol for file upload services."""

    def has_local_data(self, dataloader_config: List[Dict[str, Any]]) -> bool:
        """Check if dataloader config references local data."""
        ...

    def estimate_upload_size(self, dataloader_config: List[Dict[str, Any]]) -> int:
        """Estimate total upload size in bytes."""
        ...

    async def package_and_upload(
        self,
        dataloader_config: List[Dict[str, Any]],
        detailed_progress_callback: Optional[Any] = None,
    ) -> str:
        """Package and upload data. Returns URL."""
        ...


@runtime_checkable
class MetricsServiceProtocol(Protocol):
    """Protocol for metrics collection."""

    def record_job_submitted(self, provider: str, config_name: Optional[str] = None) -> None:
        """Record a job submission."""
        ...

    def record_job_completed(
        self,
        provider: str,
        duration_seconds: float,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record a job completion."""
        ...

    def record_job_failed(self, provider: str, error: Optional[str] = None) -> None:
        """Record a job failure."""
        ...

    async def get_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get metrics summary."""
        ...


class EventEmitterProtocol(Protocol):
    """Protocol for event emission."""

    def emit(
        self,
        event_type: str,
        job_id: Optional[str] = None,
        message: Optional[str] = None,
        severity: str = "info",
        **kwargs,
    ) -> None:
        """Emit an event."""
        ...

    def subscribe(
        self,
        event_type: str,
        handler: Any,
    ) -> None:
        """Subscribe to an event type."""
        ...


# Type aliases for cleaner function signatures
JobStore = JobStoreProtocol
ProviderClient = ProviderClientProtocol
QueueStore = QueueStoreProtocol
UserStore = UserStoreProtocol
