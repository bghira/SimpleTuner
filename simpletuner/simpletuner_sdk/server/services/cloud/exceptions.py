"""Exception hierarchy for cloud training services.

Provides structured exceptions that map to appropriate HTTP status codes.
Use these instead of raw HTTPException for better error handling and consistency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CloudError(Exception):
    """Base exception for all cloud-related errors.

    Subclasses should set:
    - status_code: HTTP status code to return
    - error_code: Machine-readable error identifier
    """

    status_code: int = 500
    error_code: str = "cloud_error"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for JSON response."""
        result = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# --- Authentication & Authorization Errors (401, 403) ---


class AuthenticationError(CloudError):
    """Authentication failed (invalid or missing credentials)."""

    status_code = 401
    error_code = "authentication_error"


class InvalidTokenError(AuthenticationError):
    """API token is invalid or expired."""

    error_code = "invalid_token"


class MissingTokenError(AuthenticationError):
    """Required API token is not configured."""

    error_code = "missing_token"


class InvalidSignatureError(AuthenticationError):
    """Webhook signature verification failed."""

    error_code = "invalid_signature"


class AuthorizationError(CloudError):
    """User lacks permission for the requested operation."""

    status_code = 403
    error_code = "authorization_error"


class PermissionDeniedError(AuthorizationError):
    """User does not have the required permission."""

    error_code = "permission_denied"

    def __init__(self, permission: str, **kwargs):
        super().__init__(f"Permission denied: {permission}", **kwargs)
        self.details["required_permission"] = permission


class IPNotAllowedError(AuthorizationError):
    """Request from IP address not in allowlist."""

    error_code = "ip_not_allowed"


# --- Validation Errors (400) ---


class ValidationError(CloudError):
    """Input validation failed."""

    status_code = 400
    error_code = "validation_error"


class InvalidInputError(ValidationError):
    """Input value is invalid."""

    error_code = "invalid_input"

    def __init__(self, field: str, message: str, **kwargs):
        super().__init__(f"Invalid {field}: {message}", **kwargs)
        self.details["field"] = field


class InvalidConfigError(ValidationError):
    """Configuration is invalid or incomplete."""

    error_code = "invalid_config"


class InvalidPayloadError(ValidationError):
    """Request payload is malformed or invalid."""

    error_code = "invalid_payload"


# --- Not Found Errors (404) ---


class NotFoundError(CloudError):
    """Requested resource was not found."""

    status_code = 404
    error_code = "not_found"


class JobNotFoundError(NotFoundError):
    """Job with the specified ID was not found."""

    error_code = "job_not_found"

    def __init__(self, job_id: str, **kwargs):
        super().__init__(f"Job not found: {job_id}", **kwargs)
        self.details["job_id"] = job_id


class UserNotFoundError(NotFoundError):
    """User with the specified ID was not found."""

    error_code = "user_not_found"

    def __init__(self, user_id: str, **kwargs):
        super().__init__(f"User not found: {user_id}", **kwargs)
        self.details["user_id"] = user_id


class ConfigNotFoundError(NotFoundError):
    """Configuration was not found."""

    error_code = "config_not_found"


# --- Conflict Errors (409) ---


class ConflictError(CloudError):
    """Operation conflicts with current state."""

    status_code = 409
    error_code = "conflict"


class JobStateError(ConflictError):
    """Job is in a state that does not allow the requested operation."""

    error_code = "invalid_job_state"

    def __init__(self, job_id: str, current_state: str, allowed_states: list, **kwargs):
        super().__init__(
            f"Job {job_id} is in state '{current_state}', " f"operation requires one of: {allowed_states}",
            **kwargs,
        )
        self.details.update(
            {
                "job_id": job_id,
                "current_state": current_state,
                "allowed_states": allowed_states,
            }
        )


class DuplicateError(ConflictError):
    """Resource already exists."""

    error_code = "duplicate"


# --- Rate Limiting Errors (429) ---


class RateLimitError(CloudError):
    """Rate limit exceeded."""

    status_code = 429
    error_code = "rate_limit_exceeded"

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if retry_after:
            self.details["retry_after_seconds"] = retry_after


# --- Quota Errors (429 or 403) ---


class QuotaExceededError(CloudError):
    """User or system quota has been exceeded."""

    status_code = 429
    error_code = "quota_exceeded"

    def __init__(
        self,
        quota_type: str,
        limit: float,
        current: float,
        **kwargs,
    ):
        super().__init__(
            f"{quota_type} quota exceeded: {current} / {limit}",
            **kwargs,
        )
        self.details.update(
            {
                "quota_type": quota_type,
                "limit": limit,
                "current": current,
            }
        )


class CostLimitExceededError(QuotaExceededError):
    """Spending limit has been exceeded."""

    error_code = "cost_limit_exceeded"

    def __init__(self, limit: float, current: float, period: str, **kwargs):
        super().__init__(
            quota_type="spending",
            limit=limit,
            current=current,
            **kwargs,
        )
        self.details["period"] = period


class JobLimitExceededError(QuotaExceededError):
    """Concurrent job limit has been exceeded."""

    error_code = "job_limit_exceeded"


# --- Provider Errors (502, 503) ---


class ProviderError(CloudError):
    """Error communicating with cloud provider."""

    status_code = 502
    error_code = "provider_error"

    def __init__(self, provider: str, message: str, **kwargs):
        super().__init__(f"{provider}: {message}", **kwargs)
        self.details["provider"] = provider


class ProviderUnavailableError(ProviderError):
    """Cloud provider is temporarily unavailable."""

    status_code = 503
    error_code = "provider_unavailable"


class ProviderTimeoutError(ProviderError):
    """Request to cloud provider timed out."""

    error_code = "provider_timeout"


class ProviderAPIError(ProviderError):
    """Cloud provider returned an error response."""

    error_code = "provider_api_error"

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(provider, message, **kwargs)
        if status_code:
            self.details["provider_status_code"] = status_code


# --- Internal Errors (500) ---


class InternalError(CloudError):
    """Internal server error."""

    status_code = 500
    error_code = "internal_error"


class DatabaseError(InternalError):
    """Database operation failed."""

    error_code = "database_error"


class ConfigurationError(InternalError):
    """Server configuration error."""

    error_code = "configuration_error"


# --- Service Unavailable (503) ---


class ServiceUnavailableError(CloudError):
    """Service is temporarily unavailable."""

    status_code = 503
    error_code = "service_unavailable"


class NotReadyError(ServiceUnavailableError):
    """Service is not ready to handle requests."""

    error_code = "not_ready"


# --- Helper function to convert CloudError to HTTPException ---


def to_http_exception(error: CloudError):
    """Convert a CloudError to a FastAPI HTTPException.

    Usage:
        try:
            do_something()
        except CloudError as e:
            raise to_http_exception(e)

    Or use the exception handler registered below.
    """
    from fastapi import HTTPException

    return HTTPException(
        status_code=error.status_code,
        detail=error.to_dict(),
    )


def register_exception_handlers(app):
    """Register exception handlers for CloudError hierarchy.

    Call this in your FastAPI app setup:
        from services.cloud.exceptions import register_exception_handlers
        register_exception_handlers(app)

    This automatically converts CloudError exceptions to proper HTTP responses.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(CloudError)
    async def cloud_error_handler(request: Request, exc: CloudError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
