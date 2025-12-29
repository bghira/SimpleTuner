"""Quota management endpoints for cloud training."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...services.cloud.auth import QuotaAction, QuotaChecker, QuotaStatus, QuotaType, get_current_user, require_permission
from ...services.cloud.auth.models import User
from ...services.cloud.auth.user_store import UserStore
from ._shared import get_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quotas", tags=["quotas"])


# --- Request/Response Models ---


class QuotaRequest(BaseModel):
    """Request to create or update a quota."""

    quota_type: str
    limit_value: float = Field(gt=0)
    action: str = "block"
    user_id: Optional[int] = None
    level_id: Optional[int] = None


class QuotaResponse(BaseModel):
    """Response for a single quota."""

    id: int
    quota_type: str
    limit_value: float
    action: str
    user_id: Optional[int] = None
    level_id: Optional[int] = None
    is_global: bool = False


class QuotaListResponse(BaseModel):
    """Response for listing quotas."""

    quotas: List[QuotaResponse]


class QuotaStatusResponse(BaseModel):
    """Response for quota status check."""

    quota_type: str
    limit: float
    current: float
    action: str
    is_exceeded: bool
    is_warning: bool
    percent_used: float
    message: Optional[str] = None
    source: str


class UserQuotasResponse(BaseModel):
    """Response for user quota status."""

    user_id: int
    statuses: List[QuotaStatusResponse]
    can_submit: bool
    blocking_reasons: List[str] = Field(default_factory=list)


# --- Endpoints ---


@router.get("", response_model=QuotaListResponse)
async def list_quotas(
    user: User = Depends(require_permission("quota.view")),
) -> QuotaListResponse:
    """List all configured quotas (global, level, and user-specific)."""
    user_store = UserStore()
    quotas = await user_store.list_all_quotas()

    return QuotaListResponse(
        quotas=[
            QuotaResponse(
                id=q.id,
                quota_type=q.quota_type.value,
                limit_value=q.limit_value,
                action=q.action.value,
                user_id=q.user_id,
                level_id=q.level_id,
                is_global=q.is_global,
            )
            for q in quotas
        ]
    )


@router.post("", response_model=QuotaResponse)
async def create_quota(
    request: QuotaRequest,
    user: User = Depends(require_permission("quota.manage")),
) -> QuotaResponse:
    """Create or update a quota.

    If user_id is set, applies to that specific user.
    If level_id is set, applies to all users with that level.
    If neither is set, this is a global default quota.
    """
    try:
        quota_type = QuotaType(request.quota_type)
    except ValueError:
        valid_types = [t.value for t in QuotaType]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid quota_type: {request.quota_type}. Valid types: {valid_types}",
        )

    try:
        action = QuotaAction(request.action)
    except ValueError:
        valid_actions = [a.value for a in QuotaAction]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: {request.action}. Valid actions: {valid_actions}",
        )

    user_store = UserStore()
    quota_id = await user_store.set_quota(
        quota_type=quota_type,
        limit_value=request.limit_value,
        action=action,
        user_id=request.user_id,
        level_id=request.level_id,
    )

    return QuotaResponse(
        id=quota_id,
        quota_type=request.quota_type,
        limit_value=request.limit_value,
        action=request.action,
        user_id=request.user_id,
        level_id=request.level_id,
        is_global=request.user_id is None and request.level_id is None,
    )


@router.delete("/{quota_id}")
async def delete_quota(
    quota_id: int,
    user: User = Depends(require_permission("quota.manage")),
) -> Dict[str, Any]:
    """Delete a quota."""
    user_store = UserStore()
    success = await user_store.delete_quota(quota_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quota not found: {quota_id}",
        )

    return {"success": True, "quota_id": quota_id}


@router.get("/types")
async def list_quota_types(
    user: User = Depends(require_permission("quota.view")),
) -> Dict[str, Any]:
    """List available quota types and actions."""
    return {
        "quota_types": [{"value": t.value, "description": _quota_type_description(t)} for t in QuotaType],
        "actions": [{"value": a.value, "description": _quota_action_description(a)} for a in QuotaAction],
    }


@router.get("/user/{user_id}", response_model=UserQuotasResponse)
async def get_user_quotas(
    user_id: int,
    user: User = Depends(require_permission("quota.view")),
) -> UserQuotasResponse:
    """Get current quota status for a user."""
    user_store = UserStore()
    job_store = get_job_store()

    target_user = await user_store.get_user(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    quota_checker = QuotaChecker(job_store, user_store)
    statuses = await quota_checker.check_all_quotas(user_id)
    can_submit, blocking_statuses = await quota_checker.can_submit_job(user_id)

    blocking_reasons = [s.message for s in blocking_statuses if s.message and s.is_exceeded]

    return UserQuotasResponse(
        user_id=user_id,
        statuses=[
            QuotaStatusResponse(
                quota_type=s.quota_type.value,
                limit=s.limit_value,
                current=s.current_value,
                action=s.action.value,
                is_exceeded=s.is_exceeded,
                is_warning=s.is_warning,
                percent_used=s.percent_used,
                message=s.message,
                source=s.source,
            )
            for s in statuses
        ],
        can_submit=can_submit,
        blocking_reasons=blocking_reasons,
    )


@router.get("/me", response_model=UserQuotasResponse)
async def get_my_quotas(
    user: User = Depends(get_current_user),
) -> UserQuotasResponse:
    """Get current quota status for the authenticated user."""
    user_store = UserStore()
    job_store = get_job_store()

    quota_checker = QuotaChecker(job_store, user_store)
    statuses = await quota_checker.check_all_quotas(user.id)
    can_submit, blocking_statuses = await quota_checker.can_submit_job(user.id)

    blocking_reasons = [s.message for s in blocking_statuses if s.message and s.is_exceeded]

    return UserQuotasResponse(
        user_id=user.id,
        statuses=[
            QuotaStatusResponse(
                quota_type=s.quota_type.value,
                limit=s.limit_value,
                current=s.current_value,
                action=s.action.value,
                is_exceeded=s.is_exceeded,
                is_warning=s.is_warning,
                percent_used=s.percent_used,
                message=s.message,
                source=s.source,
            )
            for s in statuses
        ],
        can_submit=can_submit,
        blocking_reasons=blocking_reasons,
    )


def _quota_type_description(qt: QuotaType) -> str:
    """Get human-readable description for quota type."""
    descriptions = {
        QuotaType.COST_MONTHLY: "Maximum USD spend per month",
        QuotaType.COST_DAILY: "Maximum USD spend per day",
        QuotaType.CONCURRENT_JOBS: "Maximum running jobs at once",
        QuotaType.JOBS_PER_DAY: "Maximum job submissions per day",
        QuotaType.JOBS_PER_HOUR: "Maximum job submissions per hour",
    }
    return descriptions.get(qt, "")


def _quota_action_description(qa: QuotaAction) -> str:
    """Get human-readable description for quota action."""
    descriptions = {
        QuotaAction.BLOCK: "Block the action when quota is exceeded",
        QuotaAction.WARN: "Allow but show warning when quota is exceeded",
        QuotaAction.REQUIRE_APPROVAL: "Require admin approval when quota is exceeded",
    }
    return descriptions.get(qa, "")
