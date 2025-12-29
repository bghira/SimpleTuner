"""Pydantic request/response models for notification endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# --- Channel Models ---


class ChannelCreateRequest(BaseModel):
    """Request to create a notification channel."""

    channel_type: str = Field(..., description="Channel type: email, webhook, slack")
    name: str = Field(..., min_length=1, max_length=100)
    is_enabled: bool = True

    # SMTP settings (for email)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None  # Will be stored in secrets
    smtp_use_tls: bool = True
    smtp_from_address: Optional[str] = None
    smtp_from_name: Optional[str] = None

    # Webhook settings
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None  # Will be stored in secrets

    # IMAP settings (for email responses)
    imap_enabled: bool = False
    imap_host: Optional[str] = None
    imap_port: int = 993
    imap_username: Optional[str] = None
    imap_password: Optional[str] = None  # Will be stored in secrets
    imap_use_ssl: bool = True
    imap_folder: str = "INBOX"


class ChannelUpdateRequest(BaseModel):
    """Request to update a notification channel."""

    name: Optional[str] = None
    is_enabled: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: Optional[bool] = None
    smtp_from_address: Optional[str] = None
    smtp_from_name: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    imap_enabled: Optional[bool] = None
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_username: Optional[str] = None
    imap_password: Optional[str] = None
    imap_use_ssl: Optional[bool] = None
    imap_folder: Optional[str] = None


class ChannelResponse(BaseModel):
    """Response with channel details."""

    id: int
    channel_type: str
    name: str
    is_enabled: bool
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_use_tls: Optional[bool] = None
    smtp_from_address: Optional[str] = None
    smtp_from_name: Optional[str] = None
    webhook_url: Optional[str] = None
    imap_enabled: Optional[bool] = None
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_username: Optional[str] = None
    imap_use_ssl: Optional[bool] = None
    imap_folder: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ChannelsListResponse(BaseModel):
    """Response with list of channels."""

    channels: List[ChannelResponse]
    total: int


class TestConnectionResponse(BaseModel):
    """Response from testing a channel connection."""

    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None


# --- Preset Models ---


class PresetsListResponse(BaseModel):
    """Response with available email presets."""

    presets: List[Dict[str, Any]]


class PresetDetailResponse(BaseModel):
    """Response with preset details."""

    id: str
    name: str
    smtp_host: str
    smtp_port: int
    smtp_use_tls: bool
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_use_ssl: Optional[bool] = None
    instructions: Optional[str] = None
    docs_url: Optional[str] = None


# --- Preference Models ---


class PreferenceRequest(BaseModel):
    """Request to set a notification preference."""

    event_type: str = Field(..., description="Event type to configure")
    channel_id: int = Field(..., description="Channel to use for this event")
    is_enabled: bool = True
    recipients: List[str] = Field(default_factory=list)
    min_severity: str = "info"


class PreferenceResponse(BaseModel):
    """Response with preference details."""

    id: int
    user_id: Optional[int]
    event_type: str
    channel_id: int
    is_enabled: bool
    recipients: List[str]
    min_severity: str


class PreferencesListResponse(BaseModel):
    """Response with list of preferences."""

    preferences: List[PreferenceResponse]


# --- Event Type Models ---


class EventTypesResponse(BaseModel):
    """Response with available event types."""

    event_types: List[Dict[str, Any]]


# --- Delivery History Models ---


class DeliveryLogEntry(BaseModel):
    """A delivery log entry."""

    id: int
    channel_id: int
    event_type: str
    job_id: Optional[str]
    user_id: Optional[int]
    recipient: str
    delivery_status: str
    error_message: Optional[str]
    sent_at: Optional[str]
    delivered_at: Optional[str]


class DeliveryHistoryResponse(BaseModel):
    """Response with delivery history."""

    entries: List[DeliveryLogEntry]
    total: int


# --- Status Models ---


class NotificationStatusResponse(BaseModel):
    """Response with notification system status."""

    initialized: bool
    channels: Dict[str, Any]
    response_handlers: Dict[str, Any]
    stats: Dict[str, Any]
