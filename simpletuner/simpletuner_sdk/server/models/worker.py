"""Worker data models for GPU worker management."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class WorkerType(str, Enum):
    """Type of GPU worker."""

    EPHEMERAL = "ephemeral"  # Cloud-launched, shuts down after job
    PERSISTENT = "persistent"  # Always-on, user-managed


class WorkerStatus(str, Enum):
    """Status of a GPU worker."""

    CONNECTING = "connecting"  # Created but not yet registered
    IDLE = "idle"  # Ready to accept jobs
    BUSY = "busy"  # Currently running a job
    OFFLINE = "offline"  # Disconnected, may reconnect
    DRAINING = "draining"  # Finishing job, then shutdown


@dataclass
class Worker:
    """A GPU worker that can execute training jobs.

    Workers are registered with the server and can execute training jobs.
    They can be ephemeral (cloud-launched, shut down after job) or persistent
    (always-on, user-managed).
    """

    worker_id: str
    name: str
    worker_type: WorkerType
    status: WorkerStatus
    token_hash: str  # SHA256 hash of worker token
    user_id: int  # Owner
    gpu_info: Dict[str, Any] = field(default_factory=dict)  # {"name": "A100", "vram_gb": 80, "count": 1}
    provider: Optional[str] = None  # "runpod", "vast", None for manual
    labels: Dict[str, str] = field(default_factory=dict)
    current_job_id: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def persistent(self) -> bool:
        """Check if this is a persistent worker."""
        return self.worker_type == WorkerType.PERSISTENT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "worker_type": self.worker_type.value,
            "status": self.status.value,
            "user_id": self.user_id,
            "gpu_info": self.gpu_info,
            "provider": self.provider,
            "labels": self.labels,
            "current_job_id": self.current_job_id,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Worker":
        """Create from dictionary."""
        last_heartbeat = None
        if data.get("last_heartbeat"):
            try:
                last_heartbeat = datetime.fromisoformat(data["last_heartbeat"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        created_at = datetime.now(timezone.utc)
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            worker_id=data["worker_id"],
            name=data["name"],
            worker_type=WorkerType(data["worker_type"]),
            status=WorkerStatus(data["status"]),
            token_hash=data["token_hash"],
            user_id=data["user_id"],
            gpu_info=data.get("gpu_info", {}),
            provider=data.get("provider"),
            labels=data.get("labels", {}),
            current_job_id=data.get("current_job_id"),
            last_heartbeat=last_heartbeat,
            created_at=created_at,
        )
