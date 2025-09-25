"""Security middleware for SimpleTuner WebUI.

This module provides comprehensive security features including:
- CORS configuration
- Security headers
- Rate limiting
- Request validation
"""

from __future__ import annotations

import os
import logging
import time
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


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
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com",  # Allow HTMX and Alpine.js
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles for now
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Feature Policy / Permissions Policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(
        self,
        app: ASGIApp,
        calls: int = 100,
        period: int = 60,
        exclude_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.exclude_paths = exclude_paths or ["/health", "/api/events/stream"]
        self.clients: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host

        # Clean old entries
        now = time.time()
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip]
            if timestamp > now - self.period
        ]

        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        # Record this request
        self.clients[client_ip].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - len(self.clients[client_ip]))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))

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
    rate_limit_calls = int(os.getenv("RATE_LIMIT_CALLS", "100"))
    rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

    if rate_limit_calls > 0:  # Allow disabling with 0
        app.add_middleware(
            RateLimitMiddleware,
            calls=rate_limit_calls,
            period=rate_limit_period
        )
        logger.info(f"Rate limiting enabled: {rate_limit_calls} calls per {rate_limit_period} seconds")

    logger.info("Security middleware configured successfully")