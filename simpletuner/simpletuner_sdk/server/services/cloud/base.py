"""Base classes and data models for cloud training services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CloudJobStatus(str, Enum):
    """Status of a cloud training job.

    Use this enum instead of magic strings for status comparisons.
    For external API responses, use from_external() to normalize spellings.
    """

    PENDING = "pending"
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_external(cls, status: str) -> "CloudJobStatus":
        """Convert external provider status strings to CloudJobStatus.

        Handles variations like:
        - "canceled" (American) -> CANCELLED
        - "succeeded" -> COMPLETED
        - "starting" -> QUEUED
        - "processing" -> RUNNING

        Args:
            status: Status string from external API (e.g., Replicate, Cog)

        Returns:
            Normalized CloudJobStatus enum value
        """
        # Normalize to lowercase for comparison
        status_lower = status.lower().strip()

        # Direct matches
        try:
            return cls(status_lower)
        except ValueError:
            pass

        # External API mappings (Replicate/Cog use different terminology)
        external_map = {
            # American vs British spelling
            "canceled": cls.CANCELLED,
            # Replicate-specific
            "starting": cls.QUEUED,
            "processing": cls.RUNNING,
            "succeeded": cls.COMPLETED,
            # Cog lifecycle events
            "error": cls.FAILED,
        }

        return external_map.get(status_lower, cls.PENDING)

    @classmethod
    def is_terminal(cls, status: "CloudJobStatus") -> bool:
        """Check if a status is terminal (job finished).

        Args:
            status: CloudJobStatus to check

        Returns:
            True if status is COMPLETED, FAILED, or CANCELLED
        """
        return status in (cls.COMPLETED, cls.FAILED, cls.CANCELLED)

    @classmethod
    def terminal_values(cls) -> set:
        """Get set of terminal status string values.

        Returns:
            Set of terminal status strings for SQL queries
        """
        return {cls.COMPLETED.value, cls.FAILED.value, cls.CANCELLED.value}

    @classmethod
    def active_values(cls) -> set:
        """Get set of active (non-terminal) status string values.

        Returns:
            Set of active status strings for SQL queries
        """
        return {
            cls.PENDING.value,
            cls.UPLOADING.value,
            cls.QUEUED.value,
            cls.RUNNING.value,
        }


class JobType(str, Enum):
    """Type of training job."""

    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class CloudJobInfo:
    """Information about a cloud training job from the provider."""

    job_id: str
    provider: str
    status: CloudJobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    cost_usd: Optional[float] = None
    hardware_type: Optional[str] = None
    logs_url: Optional[str] = None
    output_url: Optional[str] = None
    config_name: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedJob:
    """Unified job representation for both local and cloud jobs.

    This is the primary domain model for jobs. It encapsulates:
    - Job state and metadata
    - Queue/scheduling state (formerly in QueueStore)
    - Business rules for state transitions
    - Access control logic
    - Derived properties (duration, terminal status)
    """

    job_id: str
    job_type: JobType
    provider: Optional[str]  # None for local, "replicate" for cloud
    status: str
    config_name: Optional[str]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    cost_usd: Optional[float] = None  # Cloud jobs only
    hardware_type: Optional[str] = None
    error_message: Optional[str] = None
    output_url: Optional[str] = None  # Where the trained model was saved
    upload_token: Optional[str] = None  # Per-job token for S3 upload authentication
    user_id: Optional[int] = None  # User who submitted the job
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Queue/scheduling fields (merged from QueueStore)
    priority: int = 10  # Higher = more urgent
    priority_override: Optional[int] = None  # Manual override
    queue_position: int = 0  # 0 = running or next
    queued_at: Optional[str] = None  # When job entered queue
    requires_approval: bool = False  # Needs admin approval
    approval_id: Optional[int] = None  # Linked approval record
    attempt: int = 1  # Current retry attempt
    max_attempts: int = 3  # Maximum retries allowed
    team_id: Optional[str] = None  # Team for quota tracking
    org_id: Optional[int] = None  # Organization for quota tracking
    estimated_cost: float = 0.0  # Pre-execution cost estimate
    allocated_gpus: Optional[List[int]] = None  # GPU indices for local jobs
    num_processes: int = 1  # Number of GPUs/processes needed

    # --- Business Logic Methods ---

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal (finished) state."""
        return self.status in CloudJobStatus.terminal_values()

    @property
    def is_active(self) -> bool:
        """Check if job is actively running or queued."""
        return self.status in CloudJobStatus.active_values()

    @property
    def is_cloud(self) -> bool:
        """Check if this is a cloud job."""
        return self.job_type == JobType.CLOUD

    @property
    def is_local(self) -> bool:
        """Check if this is a local job."""
        return self.job_type == JobType.LOCAL

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds.

        Returns None if job hasn't started or completed.
        """
        if not self.started_at:
            return None

        end_time_str = self.completed_at or datetime.now(timezone.utc).isoformat()
        try:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
            return (end - start).total_seconds()
        except (ValueError, TypeError):
            return None

    def can_cancel(self) -> bool:
        """Check if the job can be cancelled."""
        cancellable = {
            CloudJobStatus.PENDING.value,
            CloudJobStatus.UPLOADING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        }
        return self.status in cancellable

    def can_delete(self) -> bool:
        """Check if the job can be deleted."""
        return self.is_terminal

    def can_access(
        self,
        user_id: Optional[int],
        has_view_all: bool = False,
        is_single_user_mode: bool = False,
    ) -> bool:
        """Check if a user can access this job.

        Args:
            user_id: The user attempting access
            has_view_all: Whether the user has job.view.all permission
            is_single_user_mode: Whether the server is in single-user mode

        Returns:
            True if access is allowed
        """
        if has_view_all:
            return True
        if user_id is None:
            # Only allow unauthenticated access in single-user mode
            return is_single_user_mode
        return self.user_id == user_id

    def can_cancel_by(self, user_id: Optional[int], has_cancel_all: bool, has_cancel_own: bool) -> bool:
        """Check if a user can cancel this job.

        Args:
            user_id: The user attempting to cancel
            has_cancel_all: Whether user has job.cancel.all permission
            has_cancel_own: Whether user has job.cancel.own permission

        Returns:
            True if cancellation is allowed
        """
        if not self.can_cancel():
            return False
        if has_cancel_all:
            return True
        is_own = self.user_id == user_id
        return is_own and has_cancel_own

    def can_delete_by(self, user_id: Optional[int], has_delete_all: bool, has_delete_own: bool) -> bool:
        """Check if a user can delete this job.

        Args:
            user_id: The user attempting to delete
            has_delete_all: Whether user has job.delete.all permission
            has_delete_own: Whether user has job.delete.own permission

        Returns:
            True if deletion is allowed
        """
        if not self.can_delete():
            return False
        if has_delete_all:
            return True
        is_own = self.user_id == user_id
        return is_own and has_delete_own

    def validate_transition(self, new_status: str) -> Optional[str]:
        """Validate a state transition.

        Args:
            new_status: The proposed new status

        Returns:
            Error message if transition is invalid, None if valid
        """
        # Can't transition from terminal states (except by admin override)
        if self.is_terminal:
            return f"Cannot change status from terminal state '{self.status}'"

        # Define valid transitions
        valid_transitions = {
            CloudJobStatus.PENDING.value: {
                CloudJobStatus.UPLOADING.value,
                CloudJobStatus.QUEUED.value,
                CloudJobStatus.RUNNING.value,
                CloudJobStatus.CANCELLED.value,
                CloudJobStatus.FAILED.value,
            },
            CloudJobStatus.UPLOADING.value: {
                CloudJobStatus.QUEUED.value,
                CloudJobStatus.RUNNING.value,
                CloudJobStatus.CANCELLED.value,
                CloudJobStatus.FAILED.value,
            },
            CloudJobStatus.QUEUED.value: {
                CloudJobStatus.RUNNING.value,
                CloudJobStatus.CANCELLED.value,
                CloudJobStatus.FAILED.value,
            },
            CloudJobStatus.RUNNING.value: {
                CloudJobStatus.COMPLETED.value,
                CloudJobStatus.CANCELLED.value,
                CloudJobStatus.FAILED.value,
            },
        }

        allowed = valid_transitions.get(self.status, set())
        if new_status not in allowed:
            return f"Invalid transition from '{self.status}' to '{new_status}'"

        return None

    # --- Factory Methods ---

    @classmethod
    def from_cloud_job(cls, cloud_job: CloudJobInfo) -> "UnifiedJob":
        """Create a UnifiedJob from a CloudJobInfo."""
        return cls(
            job_id=cloud_job.job_id,
            job_type=JobType.CLOUD,
            provider=cloud_job.provider,
            status=cloud_job.status.value,
            config_name=cloud_job.config_name,
            created_at=cloud_job.created_at,
            started_at=cloud_job.started_at,
            completed_at=cloud_job.completed_at,
            cost_usd=cloud_job.cost_usd,
            hardware_type=cloud_job.hardware_type,
            error_message=cloud_job.error_message,
            output_url=cloud_job.output_url,
            metadata=cloud_job.metadata,
        )

    @classmethod
    def create_local(
        cls,
        job_id: str,
        config_name: Optional[str] = None,
        hardware_type: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> "UnifiedJob":
        """Create a new local job entry."""
        return cls(
            job_id=job_id,
            job_type=JobType.LOCAL,
            provider=None,
            status=CloudJobStatus.PENDING.value,
            config_name=config_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            hardware_type=hardware_type,
            user_id=user_id,
        )

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "provider": self.provider,
            "status": self.status,
            "config_name": self.config_name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cost_usd": self.cost_usd,
            "hardware_type": self.hardware_type,
            "error_message": self.error_message,
            "output_url": self.output_url,
            "upload_token": self.upload_token,
            "user_id": self.user_id,
            "metadata": self.metadata,
            # Derived properties for convenience
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedJob":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            job_type=JobType(data["job_type"]),
            provider=data.get("provider"),
            status=data["status"],
            config_name=data.get("config_name"),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            cost_usd=data.get("cost_usd"),
            hardware_type=data.get("hardware_type"),
            error_message=data.get("error_message"),
            output_url=data.get("output_url"),
            upload_token=data.get("upload_token"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DashboardMetrics:
    """Aggregated metrics for the cloud dashboard."""

    credit_balance: Optional[float] = None
    estimated_jobs_remaining: Optional[int] = None
    total_cost_30d: float = 0.0
    job_count_30d: int = 0
    avg_job_duration_seconds: Optional[float] = None
    jobs_by_status: Dict[str, int] = field(default_factory=dict)
    cost_by_day: List[Dict[str, Any]] = field(default_factory=list)


class CloudTrainerService(ABC):
    """Abstract base class for cloud training providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'replicate')."""
        ...

    @property
    @abstractmethod
    def supports_cost_tracking(self) -> bool:
        """Whether this provider supports cost tracking."""
        ...

    @property
    @abstractmethod
    def supports_live_logs(self) -> bool:
        """Whether this provider supports live log streaming."""
        ...

    @abstractmethod
    async def validate_credentials(self) -> Dict[str, Any]:
        """Validate credentials and return user info or error."""
        ...

    @abstractmethod
    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        """List recent jobs from this provider."""
        ...

    @abstractmethod
    async def run_job(
        self,
        config: Dict[str, Any],
        dataloader: List[Dict[str, Any]],
        data_archive_url: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> CloudJobInfo:
        """Submit a new training job."""
        ...

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if successful."""
        ...

    @abstractmethod
    async def get_job_logs(self, job_id: str) -> str:
        """Fetch logs for a job."""
        ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        """Get current status of a job."""
        ...


class CloudUploadBackend(ABC):
    """Abstract base class for cloud upload backends."""

    @abstractmethod
    async def upload_archive(
        self,
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Upload an archive and return the public URL.

        Args:
            local_path: Path to the local archive file
            progress_callback: Optional callback(bytes_uploaded, total_bytes)

        Returns:
            Public URL to the uploaded file
        """
        ...
