"""Security middleware for SimpleTuner WebUI.

This module provides comprehensive security features including:
- CORS configuration
- Security headers
- Rate limiting with per-endpoint configuration
- Audit logging for security events
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit configuration for a route pattern."""

    pattern: Pattern[str]
    calls: int
    period: int  # seconds
    methods: Optional[List[str]] = None  # None = all methods

    def matches(self, path: str, method: str) -> bool:
        """Check if this rule matches the request."""
        if not self.pattern.match(path):
            return False
        if self.methods and method.upper() not in self.methods:
            return False
        return True


# Default rate limit rules - more restrictive for sensitive endpoints
# Note: Unauthenticated requests use DEFAULT_ANONYMOUS_RATE_LIMIT (below)
# Authenticated users get much higher limits via AUTHENTICATED_USER_MULTIPLIER
DEFAULT_RATE_LIMIT_RULES: List[Tuple[str, int, int, Optional[List[str]]]] = [
    # Authentication - moderate limits (allow retries for typos)
    (r"^/api/auth/login$", 15, 60, ["POST"]),  # 15 login attempts/min
    (r"^/api/auth/register$", 5, 60, ["POST"]),  # 5 user registrations/min
    (r"^/api/auth/api-keys$", 10, 60, ["POST"]),  # 10 key creations/min
    # Password change - strict limit to prevent brute force
    (r"^/api/users/me/password$", 5, 300, ["PUT"]),  # 5 attempts per 5 minutes
    # Job submission - moderate limits
    (r"^/api/cloud/jobs$", 20, 60, ["POST"]),  # 20 job submissions/min
    (r"^/api/cloud/jobs/.+/cancel$", 30, 60, ["POST"]),  # 30 cancellations/min
    # Webhooks - higher limits (already has its own limiter)
    (r"^/api/webhooks/", 100, 60, None),
    # S3 uploads - moderate limits
    (r"^/api/cloud/storage/", 50, 60, None),
    # Quotas - moderate limits
    (r"^/api/quotas/", 30, 60, None),
]

# Default rate limit for unauthenticated/anonymous requests
# Set high enough to allow page loads (which trigger 10+ API calls each)
DEFAULT_ANONYMOUS_RATE_LIMIT = 180  # calls per period

# Multiplier for authenticated users (they get this many times more requests)
# e.g., if anonymous limit is 60/min, authenticated users get 600/min
AUTHENTICATED_USER_MULTIPLIER = 10


def setup_cors_middleware(app: FastAPI) -> None:
    """Configure CORS middleware with secure defaults.

    Args:
        app: FastAPI application instance
    """
    # Get allowed origins from environment
    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

    # In production, don't allow wildcard origins
    if os.getenv("PRODUCTION", "false").lower() == "true":
        # Remove any wildcard origins
        allowed_origins = [origin for origin in allowed_origins if origin != "*"]
        if not allowed_origins:
            logger.warning("No allowed origins configured for production! Defaulting to localhost only.")
            allowed_origins = ["http://localhost:8000"]

    logger.info(f"Configuring CORS with allowed origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net",  # Allow HTMX, Alpine.js, Bootstrap
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",  # Allow inline styles, Bootstrap CSS, Font Awesome
            "img-src 'self' data: blob:",
            "media-src 'self' data: blob:",  # Allow video/audio from same origin, data URIs, and blob URLs
            "font-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",  # Allow Bootstrap and Font Awesome fonts
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Feature Policy / Permissions Policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-endpoint configuration and audit logging.

    Features:
    - Per-endpoint rate limits via pattern matching
    - Default fallback rate limit for unmatched routes
    - Audit logging for rate limit events
    - Proper proxy IP handling (X-Forwarded-For)
    - Thread-safe request tracking
    - Automatic cleanup of stale entries
    """

    def __init__(
        self,
        app: ASGIApp,
        calls: int = 100,
        period: int = 60,
        exclude_paths: Optional[List[str]] = None,
        rules: Optional[List[Tuple[str, int, int, Optional[List[str]]]]] = None,
        enable_audit: bool = True,
    ):
        super().__init__(app)
        self.default_calls = calls
        self.default_period = period
        self.exclude_paths = exclude_paths or [
            # Health and monitoring
            "/health",
            "/api/events/stream",
            "/api/events",
            "/static/",
            # UI state (read-only, not security-sensitive)
            "/api/cloud/hints",
            "/api/cloud/users/me",
            "/api/cloud/providers",
            "/api/webui/state",
            "/api/webui/ui-state/",
            # Read-only data endpoints (high frequency during UI interaction)
            "/api/configs/",  # Config listing
            "/api/fields/",  # Field metadata
            "/api/datasets/blueprints",
            "/api/datasets/plan",
            "/api/prompt-libraries/",
            "/api/caption-filters",
            "/api/training/status",
            # Auth status checks (needed during login/onboarding flow)
            "/api/auth/setup/status",
            "/api/auth/check",
            "/api/auth/me",
            "/api/users/meta/auth-status",
        ]
        self.enable_audit = enable_audit
        self._lock = threading.Lock()

        # Per-key tracking: key -> list of (timestamp, rule_key)
        self._requests: Dict[str, List[Tuple[float, str]]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Clean up every minute

        # Compile rate limit rules
        self._rules: List[RateLimitRule] = []
        rule_defs = rules if rules is not None else DEFAULT_RATE_LIMIT_RULES
        for pattern, rule_calls, rule_period, methods in rule_defs:
            self._rules.append(
                RateLimitRule(
                    pattern=re.compile(pattern),
                    calls=rule_calls,
                    period=rule_period,
                    methods=methods,
                )
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxy headers."""
        # Check X-Forwarded-For header (standard proxy header)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _is_authenticated(self, request: Request) -> bool:
        """Check if request appears to be from an authenticated user.

        This is a lightweight check - we just verify presence of auth tokens,
        not their validity (that happens in the route handlers).
        """
        # Check for session cookie (SimpleTuner uses "simpletuner_session")
        if request.cookies.get("simpletuner_session"):
            return True

        # Check for API key in header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") or auth_header.startswith("ApiKey "):
            return True

        # Check for X-API-Key header (standard SimpleTuner API key header)
        if request.headers.get("X-API-Key"):
            return True

        # Check for API key in query params (some endpoints allow this)
        if request.query_params.get("api_key"):
            return True

        return False

    def _find_matching_rule(self, path: str, method: str) -> Optional[RateLimitRule]:
        """Find the first matching rate limit rule."""
        for rule in self._rules:
            if rule.matches(path, method):
                return rule
        return None

    def _get_rate_limit_key(self, client_ip: str, rule: Optional[RateLimitRule]) -> str:
        """Generate a unique key for rate limit tracking."""
        if rule:
            # Per-rule tracking
            return f"{client_ip}:{rule.pattern.pattern}"
        return f"{client_ip}:default"

    def _cleanup_stale_entries(self, now: float) -> None:
        """Remove stale entries periodically."""
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        max_period = max(self.default_period, max((r.period for r in self._rules), default=60))
        cutoff = now - max_period * 2

        keys_to_remove = []
        for key, timestamps in self._requests.items():
            self._requests[key] = [(ts, rk) for ts, rk in timestamps if ts > cutoff]
            if not self._requests[key]:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._requests[key]

    def _check_rate_limit(
        self,
        client_ip: str,
        path: str,
        method: str,
        is_authenticated: bool = False,
    ) -> Tuple[bool, int, int, int]:
        """Check if request is allowed and return limit info.

        Args:
            client_ip: Client IP address
            path: Request path
            method: HTTP method
            is_authenticated: Whether the request has authentication credentials

        Returns:
            (allowed, limit, remaining, reset_time)
        """
        now = time.time()
        rule = self._find_matching_rule(path, method)
        rate_key = self._get_rate_limit_key(client_ip, rule)

        # Determine base limit
        if rule:
            limit = rule.calls
        else:
            limit = self.default_calls

        # Apply authenticated user multiplier for non-rule-matched requests
        # (specific rules like login attempts should stay strict)
        if is_authenticated and not rule:
            limit = limit * AUTHENTICATED_USER_MULTIPLIER

        period = rule.period if rule else self.default_period
        cutoff = now - period

        with self._lock:
            self._cleanup_stale_entries(now)

            if rate_key not in self._requests:
                self._requests[rate_key] = []

            # Filter to current window
            self._requests[rate_key] = [(ts, rk) for ts, rk in self._requests[rate_key] if ts > cutoff]

            current_count = len(self._requests[rate_key])

            if current_count >= limit:
                return False, limit, 0, int(now + period)

            # Record this request
            self._requests[rate_key].append((now, rate_key))
            remaining = limit - current_count - 1

            return True, limit, remaining, int(now + period)

    async def _log_rate_limit_event(
        self,
        request: Request,
        client_ip: str,
        path: str,
    ) -> None:
        """Log rate limit event to audit log."""
        if not self.enable_audit:
            return

        try:
            from ..services.cloud.audit import AuditEventType, audit_log

            await audit_log(
                event_type=AuditEventType.RATE_LIMITED,
                action=f"Rate limit exceeded for {path}",
                actor_ip=client_ip,
                target_type="endpoint",
                target_id=path,
                details={
                    "method": request.method,
                    "user_agent": request.headers.get("User-Agent", ""),
                },
            )
        except Exception as exc:
            logger.debug("Failed to log rate limit event: %s", exc)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Skip rate limiting for excluded paths
        if any(path.startswith(excl) for excl in self.exclude_paths):
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Skip rate limiting for localhost in development
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)

        # Check if user appears authenticated (higher rate limits)
        is_authenticated = self._is_authenticated(request)

        allowed, limit, remaining, reset_time = self._check_rate_limit(client_ip, path, request.method, is_authenticated)

        if not allowed:
            # Log the rate limit event
            asyncio.create_task(self._log_rate_limit_event(request, client_ip, path))

            logger.warning("Rate limit exceeded: ip=%s path=%s method=%s", client_ip, path, request.method)

            # Return proper JSON response (middleware can't raise HTTPException)
            retry_after = max(1, reset_time - int(time.time()))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response


def setup_security_middleware(app: FastAPI) -> None:
    """Setup all security middleware.

    Args:
        app: FastAPI application instance
    """
    # Setup CORS
    setup_cors_middleware(app)

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add rate limiting
    # Default base rate is for anonymous users; authenticated users get AUTHENTICATED_USER_MULTIPLIER times more
    rate_limit_calls = int(os.getenv("RATE_LIMIT_CALLS", str(DEFAULT_ANONYMOUS_RATE_LIMIT)))
    rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

    if rate_limit_calls > 0:  # Allow disabling with 0
        app.add_middleware(RateLimitMiddleware, calls=rate_limit_calls, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit_calls} calls/{rate_limit_period}s (anonymous), "
            f"{rate_limit_calls * AUTHENTICATED_USER_MULTIPLIER} calls/{rate_limit_period}s (authenticated)"
        )

    logger.info("Security middleware configured successfully")
