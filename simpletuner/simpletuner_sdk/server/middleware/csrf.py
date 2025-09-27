"""CSRF protection middleware for HTMX and regular requests."""

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

# CSRF token header name for HTMX
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"
CSRF_TOKEN_LENGTH = 32

# Exempt paths that don't need CSRF protection
CSRF_EXEMPT_PATHS = {
    "/api/events",  # SSE endpoint
    "/api/training/events",  # SSE endpoint
    "/health",
    "/api/health",
}


class CSRFMiddleware:
    """CSRF protection middleware for FastAPI."""

    def __init__(self, app, secret_key: str):
        self.app = app
        self.secret_key = secret_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # Skip CSRF for exempt paths
        if request.url.path in CSRF_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        # Skip CSRF for GET, HEAD, OPTIONS requests
        if request.method in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return

        # For state-changing requests, verify CSRF token
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            if not await self._verify_csrf_token(request):
                response = HTMLResponse(
                    content="""
                    <div class="alert alert-danger">
                        <h6><i class="fas fa-exclamation-triangle"></i> Security Error</h6>
                        <p>CSRF token validation failed. Please refresh the page and try again.</p>
                    </div>
                    """,
                    status_code=status.HTTP_403_FORBIDDEN,
                    headers={"HX-Retarget": "#validation-results", "HX-Reswap": "innerHTML"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)

    async def _verify_csrf_token(self, request: Request) -> bool:
        """Verify CSRF token from header or form data."""
        # Get token from cookie
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not cookie_token:
            logger.warning("No CSRF cookie found")
            return False

        # Get token from header (HTMX) or form data
        header_token = request.headers.get(CSRF_HEADER_NAME)

        if not header_token:
            # Try to get from form data for regular form submissions
            if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                try:
                    form = await request.form()
                    header_token = form.get("csrf_token")
                except Exception:
                    pass

        if not header_token:
            logger.warning("No CSRF token in request")
            return False

        # Compare tokens
        return secrets.compare_digest(cookie_token, header_token)


def generate_csrf_token() -> str:
    """Generate a new CSRF token."""
    return secrets.token_urlsafe(CSRF_TOKEN_LENGTH)


async def get_csrf_token(request: Request, response: Response) -> str:
    """Get or create CSRF token for the current session."""
    token = request.cookies.get(CSRF_COOKIE_NAME)

    if not token:
        token = generate_csrf_token()
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            max_age=3600 * 24,  # 24 hours
        )

    return token
