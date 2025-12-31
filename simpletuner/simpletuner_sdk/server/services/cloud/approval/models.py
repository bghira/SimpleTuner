"""Data models for approval workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"  # Waiting for review
    APPROVED = "approved"  # Approved by authorized user
    REJECTED = "rejected"  # Rejected with reason
    EXPIRED = "expired"  # Auto-expired after timeout
    CANCELLED = "cancelled"  # Cancelled by requester


class RuleCondition(str, Enum):
    """Types of conditions that can trigger approval."""

    COST_EXCEEDS = "cost_exceeds"  # Estimated cost > threshold
    HARDWARE_TYPE = "hardware_type"  # Specific hardware requested
    PROVIDER = "provider"  # Specific cloud provider
    USER_LEVEL = "user_level"  # User's level is below threshold
    DAILY_JOBS_EXCEED = "daily_jobs_exceed"  # User's daily job count > threshold
    FIRST_JOB = "first_job"  # User's first job ever
    CONFIG_PATTERN = "config_pattern"  # Config name matches pattern


@dataclass
class ApprovalRule:
    """A rule that determines when approval is required.

    Rules are evaluated in priority order. First matching rule determines
    whether approval is needed.
    """

    id: int
    name: str
    description: str
    condition: RuleCondition
    threshold: str  # Condition-specific value (e.g., "100.00" for cost)
    is_active: bool = True
    priority: int = 0  # Higher priority rules evaluated first

    # Optional restrictions
    applies_to_provider: Optional[str] = None  # Only for specific provider
    applies_to_level: Optional[str] = None  # Only for users with this level
    exempt_levels: List[str] = field(default_factory=list)  # Levels exempt from this rule

    # Who can approve
    required_approver_level: str = "lead"  # Minimum level to approve

    # Metadata
    created_at: str = ""
    created_by: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition.value,
            "threshold": self.threshold,
            "is_active": self.is_active,
            "priority": self.priority,
            "applies_to_provider": self.applies_to_provider,
            "applies_to_level": self.applies_to_level,
            "exempt_levels": self.exempt_levels,
            "required_approver_level": self.required_approver_level,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRule":
        """Create from dictionary."""
        condition = data.get("condition", RuleCondition.COST_EXCEEDS.value)
        if isinstance(condition, str):
            condition = RuleCondition(condition)

        return cls(
            id=data.get("id", 0),
            name=data["name"],
            description=data.get("description", ""),
            condition=condition,
            threshold=str(data.get("threshold", "")),
            is_active=data.get("is_active", True),
            priority=data.get("priority", 0),
            applies_to_provider=data.get("applies_to_provider"),
            applies_to_level=data.get("applies_to_level"),
            exempt_levels=data.get("exempt_levels", []),
            required_approver_level=data.get("required_approver_level", "lead"),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by"),
        )


@dataclass
class ApprovalRequest:
    """A request for approval to run a job.

    Created when a job submission triggers an approval rule.
    """

    id: int
    job_id: str  # The job waiting for approval
    user_id: int  # User who submitted the job
    rule_id: int  # Rule that triggered the approval requirement

    status: ApprovalStatus = ApprovalStatus.PENDING
    reason: str = ""  # Why approval is needed

    # Job context
    provider: str = "replicate"
    config_name: Optional[str] = None
    estimated_cost: float = 0.0
    hardware_type: Optional[str] = None

    # Approval/rejection details
    reviewed_by: Optional[int] = None  # User who reviewed
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None

    # Timestamps
    created_at: str = ""
    expires_at: Optional[str] = None  # Auto-expire if not reviewed

    # Notification tracking
    notification_sent: bool = False
    notification_sent_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "rule_id": self.rule_id,
            "status": self.status.value,
            "reason": self.reason,
            "provider": self.provider,
            "config_name": self.config_name,
            "estimated_cost": self.estimated_cost,
            "hardware_type": self.hardware_type,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "review_notes": self.review_notes,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "notification_sent": self.notification_sent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        """Create from dictionary."""
        status = data.get("status", ApprovalStatus.PENDING.value)
        if isinstance(status, str):
            status = ApprovalStatus(status)

        return cls(
            id=data.get("id", 0),
            job_id=data["job_id"],
            user_id=data["user_id"],
            rule_id=data.get("rule_id", 0),
            status=status,
            reason=data.get("reason", ""),
            provider=data.get("provider", "replicate"),
            config_name=data.get("config_name"),
            estimated_cost=data.get("estimated_cost", 0.0),
            hardware_type=data.get("hardware_type"),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            review_notes=data.get("review_notes"),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at"),
            notification_sent=data.get("notification_sent", False),
            notification_sent_at=data.get("notification_sent_at"),
        )

    @property
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(expires.tzinfo) > expires
        except (ValueError, TypeError):
            return False


# Default approval rules for new installations
DEFAULT_RULES = [
    ApprovalRule(
        id=1,
        name="High Cost Jobs",
        description="Jobs estimated to cost more than $50 require approval",
        condition=RuleCondition.COST_EXCEEDS,
        threshold="50.00",
        priority=10,
        exempt_levels=["admin"],
        required_approver_level="lead",
    ),
    ApprovalRule(
        id=2,
        name="First Job Review",
        description="First job from new users requires approval",
        condition=RuleCondition.FIRST_JOB,
        threshold="true",
        priority=20,
        exempt_levels=["admin", "lead"],
        required_approver_level="lead",
    ),
]
