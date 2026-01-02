"""Authentication middleware for FastAPI.

Provides session and API key authentication with dependency injection
for protected routes. Supports enterprise features like internal network
bypass and proxy header trust.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Callable, List, Optional, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from .models import APIKey, User
from .user_store import UserStore

logger = logging.getLogger(__name__)


async def _log_permission_denied(
    permission: str,
    user: Optional[User],
    client_ip: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Log a permission denied event to the audit log."""
    try:
        from ..audit import AuditEventType, audit_log

        await audit_log(
            event_type=AuditEventType.PERMISSION_DENIED,
            action=f"Permission denied: {permission}",
            actor_id=user.id if user else None,
            actor_username=user.username if user else None,
            actor_ip=client_ip,
            target_type="permission",
            target_id=permission,
            details={
                "path": path,
                "required_permission": permission,
            },
        )
    except Exception as exc:
        logger.debug("Failed to log permission denied event: %s", exc)


def _raise_permission_denied(
    permission: str,
    user: Optional[User] = None,
    client_ip: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Log and raise a permission denied exception.

    Logs the event asynchronously (fire-and-forget) and raises HTTPException.
    """
    # Fire off audit log asynchronously
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_log_permission_denied(permission, user, client_ip, path))
    except RuntimeError:
        # No running loop - skip async logging
        pass

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Permission denied: {permission}",
    )


# Lazy import to avoid circular dependencies
_enterprise_config = None


def _get_enterprise_config():
    """Lazy load enterprise config."""
    global _enterprise_config
    if _enterprise_config is None:
        try:
            from ....config.enterprise import get_enterprise_config

            _enterprise_config = get_enterprise_config()
        except ImportError:
            _enterprise_config = None
    return _enterprise_config


# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Session cookie name
SESSION_COOKIE_NAME = "simpletuner_session"


class AuthContext:
    """Context object containing authentication information.

    Provides a unified interface for accessing the current user,
    regardless of how they authenticated (session or API key).

    Supports enterprise features:
    - is_internal_bypass: True if auth was bypassed for internal network
    - client_ip: The resolved client IP (considering proxy headers)
    """

    def __init__(
        self,
        user: Optional[User] = None,
        api_key: Optional[APIKey] = None,
        session_id: Optional[str] = None,
        is_authenticated: bool = False,
        is_internal_bypass: bool = False,
        client_ip: Optional[str] = None,
    ):
        self.user = user
        self.api_key = api_key
        self.session_id = session_id
        self.is_authenticated = is_authenticated
        self.is_internal_bypass = is_internal_bypass
        self.client_ip = client_ip

    @property
    def user_id(self) -> Optional[int]:
        """Get the user ID if authenticated."""
        return self.user.id if self.user else None

    @property
    def username(self) -> Optional[str]:
        """Get the username if authenticated."""
        return self.user.username if self.user else None

    @property
    def is_api_request(self) -> bool:
        """Check if this is an API key authenticated request."""
        return self.api_key is not None

    def has_permission(self, permission: str) -> bool:
        """Check if the user has a specific permission."""
        if not self.user:
            return False

        # If using scoped API key, check scope first
        if self.api_key and self.api_key.scoped_permissions is not None:
            if permission not in self.api_key.scoped_permissions and "*" not in self.api_key.scoped_permissions:
                return False

        return self.user.has_permission(permission)

    def require_permission(self, permission: str) -> None:
        """Raise HTTPException if user lacks the permission."""
        if not self.has_permission(permission):
            _raise_permission_denied(permission, self.user, self.client_ip)


class AuthMiddleware:
    """Middleware that extracts authentication from requests.

    Supports both session cookies and API key authentication.
    Does NOT enforce authentication - use get_current_user for that.

    In single-user mode (no users configured), automatically creates and
    authenticates as a local admin user.
    """

    def __init__(self, store: Optional[UserStore] = None):
        self._store = store
        self._single_user_checked = False
        self._single_user_mode = False
        self._local_admin: Optional[User] = None

    @property
    def store(self) -> UserStore:
        """Lazy load the user store."""
        if self._store is None:
            self._store = UserStore()
        return self._store

    def clear_single_user_cache(self) -> None:
        """Clear the single-user mode cache, forcing re-evaluation on next request."""
        self._single_user_checked = False
        self._single_user_mode = False
        self._local_admin = None

    async def _ensure_single_user_mode(self) -> Optional[User]:
        """Check for single-user mode and ensure a local admin exists.

        Single-user mode is enabled when:
        1. No users exist in the database, OR
        2. Only the auto-created 'local' user exists

        Returns the local admin user if in single-user mode, None otherwise.
        """
        if self._single_user_checked:
            return self._local_admin if self._single_user_mode else None

        self._single_user_checked = True

        try:
            has_users = await self.store.has_any_users()

            if not has_users:
                # Create a local admin user
                logger.info("Single-user mode: creating local admin user")
                user = await self.store.create_user(
                    email="local@localhost",
                    username="local",
                    password="local",  # Not used for auth in single-user mode
                    display_name="Local Admin",
                    is_admin=True,
                    level_names=["admin"],
                )
                # Reload with permissions
                user = await self.store.get_user(user.id)
                self._single_user_mode = True
                self._local_admin = user
                logger.info("Single-user mode enabled with local admin user")
                return user

            # Check if only the local user exists
            users = await self.store.list_users(limit=2)
            if len(users) == 1 and users[0].username == "local":
                self._single_user_mode = True
                self._local_admin = users[0]
                logger.debug("Single-user mode: using existing local admin")
                return users[0]

            # Multiple users or non-local user exists - not single-user mode
            self._single_user_mode = False
            self._local_admin = None
            return None

        except Exception as exc:
            logger.warning("Failed to check single-user mode: %s", exc)
            self._single_user_mode = False
            return None

    async def __call__(self, request: Request) -> AuthContext:
        """Extract authentication context from request.

        Checks for:
        1. Internal network bypass (if enterprise config allows)
        2. API key authentication
        3. Session cookie

        Returns an AuthContext with user info if authenticated.
        """
        # Get client IP (considering proxy headers if enterprise config allows)
        client_ip = self._get_client_ip(request)

        # Check for internal network bypass
        enterprise = _get_enterprise_config()
        if enterprise and enterprise.should_bypass_auth(client_ip, request.url.path):
            logger.debug("Auth bypassed for internal network: %s (path: %s)", client_ip, request.url.path)
            return AuthContext(
                is_authenticated=True,
                is_internal_bypass=True,
                client_ip=client_ip,
            )

        # Try API key authentication first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            result = await self.store.authenticate_api_key(api_key)
            if result:
                user, key = result
                logger.debug("Authenticated via API key: %s (user: %s)", key.key_prefix, user.username)
                return AuthContext(
                    user=user,
                    api_key=key,
                    is_authenticated=True,
                    client_ip=client_ip,
                )
            else:
                logger.debug("Invalid API key provided")
                # Don't error here - let route decide if auth is required

        # Try session cookie
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        if session_id:
            user = await self.store.get_session_user(session_id)
            if user:
                logger.debug("Authenticated via session: %s", user.username)
                return AuthContext(
                    user=user,
                    session_id=session_id,
                    is_authenticated=True,
                    client_ip=client_ip,
                )
            else:
                logger.debug("Invalid or expired session")

        # Check for single-user mode (local development without auth setup)
        local_admin = await self._ensure_single_user_mode()
        if local_admin:
            logger.debug("Authenticated via single-user mode as: %s", local_admin.username)
            return AuthContext(
                user=local_admin,
                is_authenticated=True,
                client_ip=client_ip,
            )

        # No authentication
        return AuthContext(client_ip=client_ip)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, considering enterprise proxy configuration."""
        enterprise = _get_enterprise_config()

        # Get direct connection IP
        direct_ip = request.client.host if request.client else "unknown"

        if enterprise:
            # Use enterprise config to resolve real IP
            forwarded_for = request.headers.get("X-Forwarded-For")
            real_ip = request.headers.get("X-Real-IP")
            return enterprise.get_client_ip(direct_ip, forwarded_for, real_ip)

        # Fallback: simple forwarded header check
        return get_client_ip(request)


# Global middleware instance (lazy initialized)
_auth_middleware: Optional[AuthMiddleware] = None


def get_auth_middleware() -> AuthMiddleware:
    """Get the global auth middleware instance."""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = AuthMiddleware()
    return _auth_middleware


def invalidate_single_user_mode() -> None:
    """Invalidate single-user mode cache, forcing re-check on next request.

    Call this when a real user is created or the placeholder user is deleted.
    """
    if _auth_middleware is not None:
        _auth_middleware.clear_single_user_cache()


async def get_auth_context(request: Request) -> AuthContext:
    """FastAPI dependency that provides AuthContext.

    Usage:
        @router.get("/protected")
        async def protected_route(auth: AuthContext = Depends(get_auth_context)):
            if auth.is_authenticated:
                return {"user": auth.username}
            return {"user": "anonymous"}
    """
    middleware = get_auth_middleware()
    return await middleware(request)


async def get_current_user(request: Request) -> User:
    """FastAPI dependency that requires an authenticated user.

    Raises 401 if not authenticated.

    Usage:
        @router.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user": user.username}
    """
    auth = await get_auth_context(request)
    if not auth.is_authenticated or not auth.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth.user


async def get_optional_user(request: Request) -> Optional[User]:
    """FastAPI dependency that optionally returns a user.

    Returns None if not authenticated (no error raised).

    Usage:
        @router.get("/public")
        async def public_route(user: Optional[User] = Depends(get_optional_user)):
            if user:
                return {"user": user.username}
            return {"user": "anonymous"}
    """
    auth = await get_auth_context(request)
    return auth.user


def require_permission(permission: str) -> Callable:
    """Dependency factory that requires a specific permission.

    Usage:
        @router.post("/admin/users")
        async def create_user(user: User = Depends(require_permission("admin.users"))):
            # Only users with admin.users permission can access
            pass
    """

    async def _check_permission(request: Request) -> User:
        auth = await get_auth_context(request)
        user = auth.user
        if not auth.is_authenticated or not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.has_permission(permission):
            _raise_permission_denied(permission, user, auth.client_ip, str(request.url.path))
        return user

    return _check_permission


def require_any_permission(permissions: List[str]) -> Callable:
    """Dependency factory that requires any of the specified permissions.

    Usage:
        @router.get("/jobs")
        async def list_jobs(
            user: User = Depends(require_any_permission(["job.view.own", "job.view.all"]))
        ):
            pass
    """

    async def _check_permission(request: Request) -> User:
        auth = await get_auth_context(request)
        user = auth.user
        if not auth.is_authenticated or not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.has_any_permission(permissions):
            perm_str = ", ".join(permissions)
            _raise_permission_denied(f"one of [{perm_str}]", user, auth.client_ip, str(request.url.path))
        return user

    return _check_permission


def require_admin(request: Request) -> Callable:
    """Dependency that requires admin access.

    Usage:
        @router.delete("/users/{user_id}")
        async def delete_user(user: User = Depends(require_admin)):
            pass
    """
    return require_permission("*")


class PermissionChecker:
    """Helper class for checking permissions in route handlers.

    Usage:
        checker = PermissionChecker(user)
        if checker.can("job.submit"):
            # do something
        checker.require("admin.users")  # raises 403 if not allowed
    """

    def __init__(self, user: User, api_key: Optional[APIKey] = None):
        self.user = user
        self.api_key = api_key

    def can(self, permission: str) -> bool:
        """Check if user has permission."""
        # If using scoped API key, check scope first
        if self.api_key and self.api_key.scoped_permissions is not None:
            if permission not in self.api_key.scoped_permissions and "*" not in self.api_key.scoped_permissions:
                return False

        return self.user.has_permission(permission)

    def can_any(self, permissions: List[str]) -> bool:
        """Check if user has any of the permissions."""
        return any(self.can(p) for p in permissions)

    def can_all(self, permissions: List[str]) -> bool:
        """Check if user has all of the permissions."""
        return all(self.can(p) for p in permissions)

    def require(self, permission: str, client_ip: Optional[str] = None, path: Optional[str] = None) -> None:
        """Require a permission, raising 403 if not allowed."""
        if not self.can(permission):
            _raise_permission_denied(permission, self.user, client_ip, path)

    def require_any(self, permissions: List[str], client_ip: Optional[str] = None, path: Optional[str] = None) -> None:
        """Require any of the permissions, raising 403 if none allowed."""
        if not self.can_any(permissions):
            perm_str = ", ".join(permissions)
            _raise_permission_denied(f"one of [{perm_str}]", self.user, client_ip, path)


def get_client_ip(request: Request) -> str:
    """Extract the client IP address from a request.

    Handles X-Forwarded-For for reverse proxies.
    """
    # Check for forwarded header (reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()

    # Check for real IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection
    if request.client:
        return request.client.host

    return "unknown"


# Reset function for testing
def reset_auth_middleware() -> None:
    """Reset the global auth middleware (for testing)."""
    global _auth_middleware
    _auth_middleware = None
