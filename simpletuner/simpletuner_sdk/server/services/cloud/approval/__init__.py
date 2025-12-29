"""Approval workflow for cloud training.

Provides rules-based approval requirements, request tracking, and notifications.
"""

from .approval_store import ApprovalStore
from .models import ApprovalRequest, ApprovalRule, ApprovalStatus, RuleCondition
from .rules_engine import ApprovalRulesEngine

# Backwards compatibility alias - ApprovalStore already has async methods
AsyncApprovalStore = ApprovalStore

__all__ = [
    # Models
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalStatus",
    "RuleCondition",
    # Store
    "ApprovalStore",
    "AsyncApprovalStore",
    # Engine
    "ApprovalRulesEngine",
]
