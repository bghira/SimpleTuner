"""Audit log endpoints for viewing security events.

NOTE: This module was moved from routes/cloud/audit.py to become a top-level
global route, as audit logging is a global concept in SimpleTuner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..services.cloud.audit import AuditEntry, AuditEventType, get_audit_store
from ..services.cloud.auth import require_permission
from ..services.cloud.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])


# --- Response Models ---


class AuditEntryResponse(BaseModel):
    """Response for a single audit entry."""

    id: int
    timestamp: str
    event_type: str
    actor_id: Optional[int] = None
    actor_username: Optional[str] = None
    actor_ip: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditListResponse(BaseModel):
    """Response for audit log listing."""

    entries: List[AuditEntryResponse]
    total: int = 0
    has_more: bool = False


class AuditStatsResponse(BaseModel):
    """Response for audit statistics."""

    total_entries: int
    by_type: Dict[str, int]
    last_24h: int
    first_entry: Optional[str] = None
    last_entry: Optional[str] = None


class VerifyResponse(BaseModel):
    """Response for chain verification."""

    valid: bool
    entries_checked: int
    broken_links: List[Dict[str, Any]] = Field(default_factory=list)
    first_id: Optional[int] = None
    last_id: Optional[int] = None


class EventTypesResponse(BaseModel):
    """Response for event type listing."""

    event_types: List[Dict[str, str]]


class AuditExportConfig(BaseModel):
    """Configuration for exporting audit logs to SIEM."""

    format: str = "json"  # json or csv
    webhook_url: Optional[str] = None
    auth_token: Optional[str] = None
    security_only: bool = False


# --- Endpoints ---


@router.get("", response_model=AuditListResponse)
async def list_audit_entries(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    actor_id: Optional[int] = Query(None, description="Filter by actor user ID"),
    target_type: Optional[str] = Query(None, description="Filter by target type"),
    target_id: Optional[str] = Query(None, description="Filter by target ID"),
    since: Optional[str] = Query(None, description="Start timestamp (ISO format)"),
    until: Optional[str] = Query(None, description="End timestamp (ISO format)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_permission("admin.audit")),
) -> AuditListResponse:
    """List audit log entries.

    Requires admin.audit permission.
    """
    store = get_audit_store()

    event_types = [event_type] if event_type else None

    entries = await store.query(
        event_types=event_types,
        actor_id=actor_id,
        target_type=target_type,
        target_id=target_id,
        since=since,
        until=until,
        limit=limit + 1,  # Fetch one extra to check has_more
        offset=offset,
    )

    has_more = len(entries) > limit
    if has_more:
        entries = entries[:limit]

    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                id=e.id,
                timestamp=e.timestamp,
                event_type=e.event_type,
                actor_id=e.actor_id,
                actor_username=e.actor_username,
                actor_ip=e.actor_ip,
                target_type=e.target_type,
                target_id=e.target_id,
                action=e.action,
                details=e.details,
            )
            for e in entries
        ],
        total=len(entries),
        has_more=has_more,
    )


@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_stats(
    user: User = Depends(require_permission("admin.audit")),
) -> AuditStatsResponse:
    """Get audit log statistics.

    Requires admin.audit permission.
    """
    store = get_audit_store()
    stats = await store.get_stats()

    return AuditStatsResponse(**stats)


@router.get("/verify", response_model=VerifyResponse)
async def verify_audit_chain(
    start_id: Optional[int] = Query(None, description="Start from this entry ID"),
    end_id: Optional[int] = Query(None, description="End at this entry ID"),
    user: User = Depends(require_permission("admin.audit")),
) -> VerifyResponse:
    """Verify the integrity of the audit log chain.

    Checks that each entry's hash matches its content and
    links correctly to the previous entry.

    Requires admin.audit permission.
    """
    store = get_audit_store()
    result = await store.verify_chain(start_id=start_id, end_id=end_id)

    return VerifyResponse(**result)


@router.get("/types", response_model=EventTypesResponse)
async def list_event_types(
    user: User = Depends(require_permission("admin.audit")),
) -> EventTypesResponse:
    """List available audit event types.

    Requires admin.audit permission.
    """
    return EventTypesResponse(event_types=[{"value": et.value, "name": et.name} for et in AuditEventType])


@router.get("/user/{user_id}", response_model=AuditListResponse)
async def get_user_audit_log(
    user_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_permission("admin.audit")),
) -> AuditListResponse:
    """Get audit entries for a specific user.

    Shows both actions performed by the user and actions targeting the user.

    Requires admin.audit permission.
    """
    store = get_audit_store()

    # Get entries where user is actor
    actor_entries = await store.query(
        actor_id=user_id,
        limit=limit,
        offset=offset,
    )

    # Get entries where user is target
    target_entries = await store.query(
        target_type="user",
        target_id=str(user_id),
        limit=limit,
        offset=offset,
    )

    # Combine and deduplicate
    seen_ids = set()
    combined = []
    for e in actor_entries + target_entries:
        if e.id not in seen_ids:
            seen_ids.add(e.id)
            combined.append(e)

    # Sort by timestamp descending
    combined.sort(key=lambda x: x.timestamp, reverse=True)
    combined = combined[:limit]

    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                id=e.id,
                timestamp=e.timestamp,
                event_type=e.event_type,
                actor_id=e.actor_id,
                actor_username=e.actor_username,
                actor_ip=e.actor_ip,
                target_type=e.target_type,
                target_id=e.target_id,
                action=e.action,
                details=e.details,
            )
            for e in combined
        ],
        total=len(combined),
        has_more=len(combined) == limit,
    )


@router.get("/security", response_model=AuditListResponse)
async def get_security_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_permission("admin.audit")),
) -> AuditListResponse:
    """Get security-related audit events.

    Includes failed logins, permission denials, and suspicious activity.

    Requires admin.audit permission.
    """
    store = get_audit_store()

    security_types = [
        AuditEventType.AUTH_LOGIN_FAILED.value,
        AuditEventType.PERMISSION_DENIED.value,
        AuditEventType.RATE_LIMITED.value,
        AuditEventType.SUSPICIOUS_ACTIVITY.value,
    ]

    entries = await store.query(
        event_types=security_types,
        limit=limit + 1,
        offset=offset,
    )

    has_more = len(entries) > limit
    if has_more:
        entries = entries[:limit]

    return AuditListResponse(
        entries=[
            AuditEntryResponse(
                id=e.id,
                timestamp=e.timestamp,
                event_type=e.event_type,
                actor_id=e.actor_id,
                actor_username=e.actor_username,
                actor_ip=e.actor_ip,
                target_type=e.target_type,
                target_id=e.target_id,
                action=e.action,
                details=e.details,
            )
            for e in entries
        ],
        total=len(entries),
        has_more=has_more,
    )


@router.get("/export-config", response_model=AuditExportConfig)
async def get_export_config(
    user: User = Depends(require_permission("admin.audit")),
) -> AuditExportConfig:
    """Get audit export configuration.

    Requires admin.audit permission.
    """
    from ..services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    return AuditExportConfig(
        format=defaults.audit_export_format,
        webhook_url=defaults.audit_export_webhook_url,
        auth_token=defaults.audit_export_auth_token,
        security_only=defaults.audit_export_security_only,
    )


@router.post("/export-config", response_model=AuditExportConfig)
async def save_export_config(
    config: AuditExportConfig,
    user: User = Depends(require_permission("admin.audit")),
) -> AuditExportConfig:
    """Save audit export configuration for SIEM integration.

    Configures how audit logs are exported to external systems.

    Requires admin.audit permission.
    """
    from fastapi import HTTPException, status

    from ..services.webui_state import WebUIStateStore

    # Validate format
    if config.format not in {"json", "csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format: {config.format}. Must be 'json' or 'csv'.",
        )

    store = WebUIStateStore()
    defaults = store.load_defaults()

    defaults.audit_export_format = config.format
    defaults.audit_export_webhook_url = config.webhook_url.strip() if config.webhook_url else None
    defaults.audit_export_auth_token = config.auth_token.strip() if config.auth_token else None
    defaults.audit_export_security_only = config.security_only

    store.save_defaults(defaults)

    logger.info("Audit export config updated by user %s", user.id)

    return AuditExportConfig(
        format=defaults.audit_export_format,
        webhook_url=defaults.audit_export_webhook_url,
        auth_token=defaults.audit_export_auth_token,
        security_only=defaults.audit_export_security_only,
    )
