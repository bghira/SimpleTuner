"""Error handling middleware for SimpleTuner WebUI.

This middleware provides consistent error handling across the application,
with context-aware formatting for different types of requests.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""

    async def dispatch(self, request: Request, call_next):
        """Handle requests and catch errors."""
        # Generate or extract request ID for correlation
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log the error with request ID
            logger.error(
                f"Unhandled error in request {request_id}: {str(e)}",
                exc_info=True,
                extra={"request_id": request_id, "path": request.url.path},
            )
            # Return generic error response
            return self._create_error_response(
                request,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="An unexpected error occurred",
                request_id=request_id,
            )

    def _create_error_response(self, request: Request, status_code: int, message: str, request_id: str) -> Response:
        """Create appropriate error response based on request type."""
        # Check if this is an HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"

        if is_htmx:
            # Return HTML fragment for HTMX
            return HTMLResponse(
                content=f'<div class="alert alert-danger" role="alert">{message}</div>', status_code=status_code
            )
        else:
            # Return JSON for API requests
            return JSONResponse(
                status_code=status_code, content={"error": message, "status_code": status_code, "request_id": request_id}
            )


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Log the error
        logger.warning(
            f"HTTP exception in request {request_id}: {exc.status_code} - {exc.detail}",
            extra={"request_id": request_id, "path": request.url.path, "status_code": exc.status_code},
        )

        # Check if this is an HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"

        if is_htmx:
            # Return HTML error fragment
            error_html = _get_htmx_error_html(exc.status_code, exc.detail)
            return HTMLResponse(content=error_html, status_code=exc.status_code)
        else:
            # Return JSON error response
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail, "status_code": exc.status_code, "request_id": request_id},
            )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Extract field errors
        errors = []
        for error in exc.errors():
            field_name = ".".join(str(loc) for loc in error["loc"])
            errors.append({"field": field_name, "message": error["msg"], "type": error["type"]})

        logger.warning(
            f"Validation error in request {request_id}: {errors}", extra={"request_id": request_id, "path": request.url.path}
        )

        # Check if this is an HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"

        if is_htmx:
            # Return HTML with field errors
            error_html = _get_validation_error_html(errors)
            return HTMLResponse(content=error_html, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
        else:
            # Return JSON validation errors
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"error": "Validation failed", "errors": errors, "request_id": request_id},
            )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> Response:
        """Handle general exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Log the full traceback
        logger.error(
            f"Unhandled exception in request {request_id}: {str(exc)}",
            exc_info=True,
            extra={"request_id": request_id, "path": request.url.path, "exception_type": type(exc).__name__},
        )

        # Create user-friendly error message
        if app.debug:
            # In debug mode, include more details
            error_message = f"{type(exc).__name__}: {str(exc)}"
            details = traceback.format_exc()
        else:
            # In production, use generic message
            error_message = "An unexpected error occurred"
            details = None

        # Check if this is an HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"

        if is_htmx:
            # Return HTML error fragment
            error_html = _get_htmx_error_html(status.HTTP_500_INTERNAL_SERVER_ERROR, error_message, details)
            return HTMLResponse(content=error_html, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Return JSON error response
            content = {
                "error": error_message,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "request_id": request_id,
            }
            if details:
                content["details"] = details

            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)


def _get_htmx_error_html(status_code: int, message: str, details: Optional[str] = None) -> str:
    """Generate HTML error fragment for HTMX requests.

    Args:
        status_code: HTTP status code
        message: Error message
        details: Optional error details (for debug mode)

    Returns:
        HTML string
    """
    severity = "danger" if status_code >= 500 else "warning"

    html = f"""
    <div class="alert alert-{severity} alert-dismissible fade show" role="alert">
        <strong>Error {status_code}:</strong> {message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
    """

    if details:
        html += f"""
        <details class="mt-2">
            <summary>Error Details</summary>
            <pre class="bg-light p-2 rounded"><code>{details}</code></pre>
        </details>
        """

    return html


def _get_validation_error_html(errors: List[Dict[str, Any]]) -> str:
    """Generate HTML for validation errors.

    Args:
        errors: List of validation errors

    Returns:
        HTML string
    """
    error_items = []
    for error in errors:
        field = error["field"]
        message = error["message"]
        error_items.append(f"<li><strong>{field}:</strong> {message}</li>")

    return f"""
    <div class="alert alert-danger alert-dismissible fade show" role="alert">
        <strong>Validation Error:</strong>
        <ul class="mb-0">
            {"".join(error_items)}
        </ul>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
    """


def setup_error_middleware(app: FastAPI) -> None:
    """Setup error handling middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # Add error handler middleware
    app.add_middleware(ErrorHandlerMiddleware)

    # Setup exception handlers
    setup_error_handlers(app)

    logger.info("Error handling middleware configured")
