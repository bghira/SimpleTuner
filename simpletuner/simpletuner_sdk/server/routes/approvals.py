"""Approval workflow endpoints for training jobs.

NOTE: This module was moved from routes/cloud/approval.py to become a top-level
global route, as approvals apply to all jobs, not just cloud jobs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.cloud.approval import (
    ApprovalRequest,
    ApprovalRule,
    ApprovalRulesEngine,
    ApprovalStatus,
    ApprovalStore,
    RuleCondition,
)
from ..services.cloud.auth import get_current_user, require_permission
from ..services.cloud.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


# --- Request/Response Models ---


class RuleCreateRequest(BaseModel):
    """Request to create an approval rule."""

    name: str = Field(min_length=1)
    description: str = ""
    condition: str
    threshold: str
    is_active: bool = True
    priority: int = 0
    applies_to_provider: Optional[str] = None
    applies_to_level: Optional[str] = None
    exempt_levels: List[str] = Field(default_factory=list)
    required_approver_level: str = "lead"


class RuleResponse(BaseModel):
    """Response for an approval rule."""

    id: int
    name: str
    description: str
    condition: str
    threshold: str
    is_active: bool
    priority: int
    applies_to_provider: Optional[str] = None
    applies_to_level: Optional[str] = None
    exempt_levels: List[str] = Field(default_factory=list)
    required_approver_level: str
    created_at: str
    created_by: Optional[int] = None


class RuleListResponse(BaseModel):
    """Response for listing rules."""

    rules: List[RuleResponse]


class RequestResponse(BaseModel):
    """Response for an approval request."""

    id: int
    job_id: str
    user_id: int
    rule_id: int
    status: str
    reason: str
    provider: str
    config_name: Optional[str] = None
    estimated_cost: float
    hardware_type: Optional[str] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None


class RequestListResponse(BaseModel):
    """Response for listing requests."""

    requests: List[RequestResponse]
    pending_count: int = 0


class ApproveRejectRequest(BaseModel):
    """Request to approve or reject."""

    notes: Optional[str] = None
    reason: Optional[str] = None  # For rejection


class BulkApproveRequest(BaseModel):
    """Request to bulk approve requests."""

    request_ids: List[int] = Field(min_length=1, max_length=100)
    notes: Optional[str] = None


class BulkRejectRequest(BaseModel):
    """Request to bulk reject requests."""

    request_ids: List[int] = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1)


class BulkActionResult(BaseModel):
    """Result of a single bulk action item."""

    request_id: int
    success: bool
    error: Optional[str] = None


class BulkActionResponse(BaseModel):
    """Response for bulk approve/reject."""

    total: int
    succeeded: int
    failed: int
    results: List[BulkActionResult]


# --- Rule Endpoints ---


@router.get("/rules", response_model=RuleListResponse)
async def list_rules(
    active_only: bool = Query(False),
    user: User = Depends(require_permission("admin.approve")),
) -> RuleListResponse:
    """List approval rules."""
    store = ApprovalStore()
    rules = await store.list_rules(active_only=active_only)

    return RuleListResponse(rules=[_rule_to_response(r) for r in rules])


@router.post("/rules", response_model=RuleResponse)
async def create_rule(
    request: RuleCreateRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> RuleResponse:
    """Create a new approval rule."""
    try:
        condition = RuleCondition(request.condition)
    except ValueError:
        valid = [c.value for c in RuleCondition]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid condition: {request.condition}. Valid: {valid}",
        )

    rule = ApprovalRule(
        id=0,
        name=request.name,
        description=request.description,
        condition=condition,
        threshold=request.threshold,
        is_active=request.is_active,
        priority=request.priority,
        applies_to_provider=request.applies_to_provider,
        applies_to_level=request.applies_to_level,
        exempt_levels=request.exempt_levels,
        required_approver_level=request.required_approver_level,
        created_by=user.id,
    )

    store = ApprovalStore()
    rule_id = await store.create_rule(rule)
    rule.id = rule_id

    return _rule_to_response(rule)


@router.get("/rules/{rule_id}", response_model=RuleResponse)
async def get_rule(
    rule_id: int,
    user: User = Depends(require_permission("admin.approve")),
) -> RuleResponse:
    """Get a specific approval rule."""
    store = ApprovalStore()
    rule = await store.get_rule(rule_id)

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    return _rule_to_response(rule)


@router.patch("/rules/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: int,
    request: RuleCreateRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> RuleResponse:
    """Update an approval rule."""
    store = ApprovalStore()
    rule = await store.get_rule(rule_id)

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    try:
        condition = RuleCondition(request.condition)
    except ValueError:
        valid = [c.value for c in RuleCondition]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid condition: {request.condition}. Valid: {valid}",
        )

    updates = {
        "name": request.name,
        "description": request.description,
        "condition": condition.value,
        "threshold": request.threshold,
        "is_active": request.is_active,
        "priority": request.priority,
        "applies_to_provider": request.applies_to_provider,
        "applies_to_level": request.applies_to_level,
        "exempt_levels": request.exempt_levels,
        "required_approver_level": request.required_approver_level,
    }

    await store.update_rule(rule_id, updates)

    updated = await store.get_rule(rule_id)
    return _rule_to_response(updated)


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: int,
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, Any]:
    """Delete an approval rule."""
    store = ApprovalStore()

    success = await store.delete_rule(rule_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule not found: {rule_id}",
        )

    return {"success": True, "rule_id": rule_id}


@router.get("/conditions")
async def list_conditions(
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, Any]:
    """List available rule conditions."""
    return {
        "conditions": [
            {
                "value": c.value,
                "description": _condition_description(c),
            }
            for c in RuleCondition
        ]
    }


# --- Request Endpoints ---


@router.get("/requests", response_model=RequestListResponse)
async def list_requests(
    status_filter: Optional[str] = Query(None, alias="status"),
    pending_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_permission("admin.approve")),
) -> RequestListResponse:
    """List approval requests."""
    store = ApprovalStore()

    approval_status = None
    if status_filter:
        try:
            approval_status = ApprovalStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    requests = await store.list_requests(
        limit=limit,
        offset=offset,
        status=approval_status,
        pending_only=pending_only,
    )

    pending_count = await store.get_pending_count()

    return RequestListResponse(
        requests=[_request_to_response(r) for r in requests],
        pending_count=pending_count,
    )


@router.get("/requests/pending/count")
async def get_pending_count(
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, int]:
    """Get count of pending approval requests."""
    store = ApprovalStore()
    count = await store.get_pending_count()
    return {"pending_count": count}


@router.get("/requests/{request_id}", response_model=RequestResponse)
async def get_request(
    request_id: int,
    user: User = Depends(require_permission("admin.approve")),
) -> RequestResponse:
    """Get a specific approval request."""
    store = ApprovalStore()
    request = await store.get_request(request_id)

    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request not found: {request_id}",
        )

    return _request_to_response(request)


@router.get("/requests/job/{job_id}", response_model=RequestResponse)
async def get_request_by_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> RequestResponse:
    """Get the approval request for a job."""
    store = ApprovalStore()
    request = await store.get_request_by_job_id(job_id)

    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No approval request for job: {job_id}",
        )

    # Check access: own request or has approval permission
    if request.user_id != user.id and not user.has_permission("admin.approve"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied",
        )

    return _request_to_response(request)


@router.post("/requests/{request_id}/approve")
async def approve_request(
    request_id: int,
    body: ApproveRejectRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, Any]:
    """Approve a pending request."""
    from .cloud._shared import get_job_store

    engine = ApprovalRulesEngine()
    user_levels = [lvl.name for lvl in user.levels]

    success, error = await engine.approve(
        request_id=request_id,
        approver_id=user.id,
        approver_levels=user_levels,
        notes=body.notes,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to approve request",
        )

    # Get the request to find the job
    store = ApprovalStore()
    request = await store.get_request(request_id)

    # Unblock the job in the queue if using queue system
    if request:
        try:
            from ..services.cloud.background_tasks import get_queue_scheduler

            scheduler = get_queue_scheduler()
            if scheduler:
                await scheduler.approve_job(request.job_id, request_id)
        except Exception as exc:
            logger.warning("Failed to unblock job in queue: %s", exc)

    return {
        "success": True,
        "request_id": request_id,
        "approved_by": user.id,
        "job_id": request.job_id if request else None,
    }


@router.post("/requests/{request_id}/reject")
async def reject_request(
    request_id: int,
    body: ApproveRejectRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, Any]:
    """Reject a pending request."""
    if not body.reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejection reason is required",
        )

    engine = ApprovalRulesEngine()
    user_levels = [lvl.name for lvl in user.levels]

    success, error = await engine.reject(
        request_id=request_id,
        approver_id=user.id,
        approver_levels=user_levels,
        reason=body.reason,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to reject request",
        )

    # Get the request to find the job
    store = ApprovalStore()
    request = await store.get_request(request_id)

    # Reject the job in the queue if using queue system
    if request:
        try:
            from ..services.cloud.background_tasks import get_queue_scheduler

            scheduler = get_queue_scheduler()
            if scheduler:
                await scheduler.reject_job(request.job_id, body.reason)
        except Exception as exc:
            logger.warning("Failed to reject job in queue: %s", exc)

    return {
        "success": True,
        "request_id": request_id,
        "rejected_by": user.id,
        "reason": body.reason,
        "job_id": request.job_id if request else None,
    }


@router.post("/requests/{request_id}/cancel")
async def cancel_request(
    request_id: int,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Cancel an approval request (by the requester)."""
    store = ApprovalStore()
    request = await store.get_request(request_id)

    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request not found: {request_id}",
        )

    # Only the requester or admin can cancel
    if request.user_id != user.id and not user.has_permission("admin.approve"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied",
        )

    success = await store.cancel_request(request_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request already processed",
        )

    return {"success": True, "request_id": request_id}


@router.post("/requests/bulk-approve", response_model=BulkActionResponse)
async def bulk_approve_requests(
    body: BulkApproveRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> BulkActionResponse:
    """Bulk approve multiple pending requests."""
    engine = ApprovalRulesEngine()
    user_levels = [lvl.name for lvl in user.levels]

    results: List[BulkActionResult] = []
    succeeded = 0

    for request_id in body.request_ids:
        success, error = await engine.approve(
            request_id=request_id,
            approver_id=user.id,
            approver_levels=user_levels,
            notes=body.notes,
        )

        if success:
            succeeded += 1
            # Unblock the job in the queue
            try:
                store = ApprovalStore()
                request = await store.get_request(request_id)
                if request:
                    from ..services.cloud.background_tasks import get_queue_scheduler

                    scheduler = get_queue_scheduler()
                    if scheduler:
                        await scheduler.approve_job(request.job_id, request_id)
            except Exception as exc:
                logger.warning("Failed to unblock job for request %d: %s", request_id, exc)

        results.append(
            BulkActionResult(
                request_id=request_id,
                success=success,
                error=error,
            )
        )

    return BulkActionResponse(
        total=len(body.request_ids),
        succeeded=succeeded,
        failed=len(body.request_ids) - succeeded,
        results=results,
    )


@router.post("/requests/bulk-reject", response_model=BulkActionResponse)
async def bulk_reject_requests(
    body: BulkRejectRequest,
    user: User = Depends(require_permission("admin.approve")),
) -> BulkActionResponse:
    """Bulk reject multiple pending requests."""
    engine = ApprovalRulesEngine()
    user_levels = [lvl.name for lvl in user.levels]

    results: List[BulkActionResult] = []
    succeeded = 0

    for request_id in body.request_ids:
        success, error = await engine.reject(
            request_id=request_id,
            approver_id=user.id,
            approver_levels=user_levels,
            reason=body.reason,
        )

        if success:
            succeeded += 1
            # Reject the job in the queue
            try:
                store = ApprovalStore()
                request = await store.get_request(request_id)
                if request:
                    from ..services.cloud.background_tasks import get_queue_scheduler

                    scheduler = get_queue_scheduler()
                    if scheduler:
                        await scheduler.reject_job(request.job_id, body.reason)
            except Exception as exc:
                logger.warning("Failed to reject job for request %d: %s", request_id, exc)

        results.append(
            BulkActionResult(
                request_id=request_id,
                success=success,
                error=error,
            )
        )

    return BulkActionResponse(
        total=len(body.request_ids),
        succeeded=succeeded,
        failed=len(body.request_ids) - succeeded,
        results=results,
    )


@router.get("/me", response_model=RequestListResponse)
async def get_my_requests(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
) -> RequestListResponse:
    """Get the current user's approval requests."""
    store = ApprovalStore()

    approval_status = None
    if status_filter:
        try:
            approval_status = ApprovalStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    requests = await store.list_requests(
        limit=limit,
        status=approval_status,
        user_id=user.id,
    )

    pending_count = len([r for r in requests if r.status == ApprovalStatus.PENDING])

    return RequestListResponse(
        requests=[_request_to_response(r) for r in requests],
        pending_count=pending_count,
    )


@router.post("/expire")
async def expire_old_requests(
    user: User = Depends(require_permission("admin.approve")),
) -> Dict[str, int]:
    """Manually expire old pending requests."""
    store = ApprovalStore()
    expired = await store.expire_old_requests()
    return {"expired": expired}


# --- Helper Functions ---


def _rule_to_response(rule: ApprovalRule) -> RuleResponse:
    """Convert an ApprovalRule to response model."""
    return RuleResponse(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        condition=rule.condition.value,
        threshold=rule.threshold,
        is_active=rule.is_active,
        priority=rule.priority,
        applies_to_provider=rule.applies_to_provider,
        applies_to_level=rule.applies_to_level,
        exempt_levels=rule.exempt_levels,
        required_approver_level=rule.required_approver_level,
        created_at=rule.created_at,
        created_by=rule.created_by,
    )


def _request_to_response(request: ApprovalRequest) -> RequestResponse:
    """Convert an ApprovalRequest to response model."""
    return RequestResponse(
        id=request.id,
        job_id=request.job_id,
        user_id=request.user_id,
        rule_id=request.rule_id,
        status=request.status.value,
        reason=request.reason,
        provider=request.provider,
        config_name=request.config_name,
        estimated_cost=request.estimated_cost,
        hardware_type=request.hardware_type,
        reviewed_by=request.reviewed_by,
        reviewed_at=request.reviewed_at,
        review_notes=request.review_notes,
        created_at=request.created_at,
        expires_at=request.expires_at,
    )


def _condition_description(condition: RuleCondition) -> str:
    """Get human-readable description for a condition."""
    descriptions = {
        RuleCondition.COST_EXCEEDS: "Estimated cost exceeds threshold (in USD)",
        RuleCondition.HARDWARE_TYPE: "Specific hardware type is requested",
        RuleCondition.PROVIDER: "Specific cloud provider is used",
        RuleCondition.USER_LEVEL: "User's level is below specified level",
        RuleCondition.DAILY_JOBS_EXCEED: "User's daily job count exceeds threshold",
        RuleCondition.FIRST_JOB: "This is the user's first job",
        RuleCondition.CONFIG_PATTERN: "Config name matches pattern (glob or regex)",
    }
    return descriptions.get(condition, "")
