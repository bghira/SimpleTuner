"""Authentication routes for cloud training.

Provides login, logout, session management, and API key operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from ...services.cloud.audit import AuditEventType, audit_log
from ...services.cloud.auth import UserStore, get_current_user, get_optional_user, require_permission
from ...services.cloud.auth.middleware import SESSION_COOKIE_NAME, get_client_ip
from ...services.cloud.auth.models import AuthProvider, User
from .users import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ==================== Request/Response Models ====================


class LoginRequest(BaseModel):
    """Login request body."""

    username: str = Field(..., min_length=1, description="Username or email")
    password: str = Field(..., min_length=1, description="Password")
    remember_me: bool = Field(False, description="Extended session duration")


class LoginResponse(BaseModel):
    """Login response."""

    success: bool
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = None


class CreateAPIKeyRequest(BaseModel):
    """API key creation request."""

    name: str = Field(..., min_length=1, max_length=100, description="Key name for identification")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Days until expiration")
    scoped_permissions: Optional[List[str]] = Field(None, description="Limit key to specific permissions")


class APIKeyResponse(BaseModel):
    """API key in responses (without the full key)."""

    id: int
    name: str
    key_prefix: str
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool
    scoped_permissions: Optional[List[str]] = None


class APIKeyCreatedResponse(BaseModel):
    """Response when creating a new API key (includes full key)."""

    key: str  # Full key, shown only once
    key_info: APIKeyResponse


class SetupStatusResponse(BaseModel):
    """First-run setup status."""

    needs_setup: bool
    has_admin: bool
    user_count: int


class FirstRunSetupRequest(BaseModel):
    """First admin user creation."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = None


# ==================== Helper Functions ====================


def _get_store() -> UserStore:
    """Get the user store instance."""
    return UserStore()


# ==================== Setup Routes ====================


@router.get("/setup/status", response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """Check if first-run setup is needed.

    Returns whether any users exist and if an admin has been created.
    Use this to determine if the setup wizard should be shown.
    """
    store = _get_store()
    user_count = await store.get_user_count()

    # Check if there's at least one admin
    users = await store.list_users(limit=1)
    has_admin = any(u.is_admin for u in users)

    return SetupStatusResponse(
        needs_setup=user_count == 0,
        has_admin=has_admin,
        user_count=user_count,
    )


@router.post("/setup/first-admin", response_model=LoginResponse)
async def create_first_admin(
    request: Request,
    response: Response,
    data: FirstRunSetupRequest,
) -> LoginResponse:
    """Create the first admin user during initial setup.

    This endpoint is only available when no users exist.
    Creates an admin user and logs them in automatically.
    """
    store = _get_store()

    # Check if setup is still needed
    if await store.has_any_users():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Setup already completed. Use normal registration.",
        )

    try:
        # Create the admin user
        user = await store.create_user(
            email=data.email,
            username=data.username,
            password=data.password,
            display_name=data.display_name,
            is_admin=True,
            level_names=["admin"],
        )

        # Create session and log in
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")[:500]
        session_id = await store.create_session(
            user_id=user.id,
            ip_address=client_ip,
            user_agent=user_agent,
            expires_hours=24 * 30,  # 30 days for first admin
        )

        # Set session cookie
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            max_age=60 * 60 * 24 * 30,  # 30 days
        )

        # Reload user with permissions
        user = await store.get_user(user.id)

        logger.info("First admin user created: %s", user.username)

        return LoginResponse(
            success=True,
            user=user.to_dict(),
            message="Admin account created successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ==================== Session Routes ====================


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    data: LoginRequest,
) -> LoginResponse:
    """Authenticate with username/email and password.

    On success, sets a session cookie and returns user info.
    """
    store = _get_store()

    client_ip = get_client_ip(request)
    user = await store.authenticate_local(data.username, data.password)
    if not user:
        logger.warning("Failed login attempt for: %s", data.username)
        # Audit failed login
        await audit_log(
            AuditEventType.AUTH_LOGIN_FAILED,
            f"Failed login attempt for '{data.username}'",
            actor_ip=client_ip,
            details={"username": data.username},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Create session
    user_agent = request.headers.get("User-Agent", "")[:500]
    expires_hours = 24 * 30 if data.remember_me else 24  # 30 days or 24 hours

    session_id = await store.create_session(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
        expires_hours=expires_hours,
    )

    # Set session cookie
    max_age = 60 * 60 * 24 * 30 if data.remember_me else 60 * 60 * 24
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=max_age,
    )

    # Audit successful login
    await audit_log(
        AuditEventType.AUTH_LOGIN_SUCCESS,
        f"User '{user.username}' logged in",
        actor_id=user.id,
        actor_username=user.username,
        actor_ip=client_ip,
    )

    logger.info("User logged in: %s from %s", user.username, client_ip)

    return LoginResponse(
        success=True,
        user=user.to_dict(),
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, bool]:
    """End the current session.

    Clears the session cookie and invalidates the session server-side.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    client_ip = get_client_ip(request)

    if session_id:
        store = _get_store()
        await store.delete_session(session_id)
        if user:
            await audit_log(
                AuditEventType.AUTH_LOGOUT,
                f"User '{user.username}' logged out",
                actor_id=user.id,
                actor_username=user.username,
                actor_ip=client_ip,
            )
            logger.info("User logged out: %s", user.username)

    # Clear the cookie
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )

    return {"success": True}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: User = Depends(get_current_user),
) -> UserResponse:
    """Get the current authenticated user's information."""
    return UserResponse.from_user(user)


@router.get("/check")
async def check_auth(
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Check if the current request is authenticated.

    Returns authentication status and basic user info if authenticated.
    Useful for frontend to check session validity without a full user fetch.
    """
    if user:
        return {
            "authenticated": True,
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
        }
    return {"authenticated": False}


# ==================== API Key Routes ====================


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    user: User = Depends(get_current_user),
) -> List[APIKeyResponse]:
    """List all API keys for the current user."""
    store = _get_store()
    keys = await store.list_api_keys(user.id)

    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            is_active=k.is_active,
            scoped_permissions=list(k.scoped_permissions) if k.scoped_permissions else None,
        )
        for k in keys
    ]


@router.post("/api-keys", response_model=APIKeyCreatedResponse)
async def create_api_key(
    data: CreateAPIKeyRequest,
    user: User = Depends(get_current_user),
) -> APIKeyCreatedResponse:
    """Create a new API key.

    The full key is returned only once in this response.
    Store it securely - it cannot be retrieved again.
    """
    # Check permission
    if not user.has_permission("api.keys.own"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: api.keys.own",
        )

    # Calculate expiration
    expires_at = None
    if data.expires_days:
        from datetime import timedelta

        expires_at = (datetime.now(timezone.utc) + timedelta(days=data.expires_days)).isoformat()

    # Validate scoped permissions
    scoped_permissions = None
    if data.scoped_permissions:
        # User can only scope to permissions they have
        user_perms = user.effective_permissions
        if "*" not in user_perms:
            invalid_perms = set(data.scoped_permissions) - user_perms
            if invalid_perms:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot scope to permissions you don't have: {invalid_perms}",
                )
        scoped_permissions = set(data.scoped_permissions)

    store = _get_store()
    full_key, api_key = await store.create_api_key(
        user_id=user.id,
        name=data.name,
        expires_at=expires_at,
        scoped_permissions=scoped_permissions,
    )

    logger.info("API key created: %s for user %s", api_key.key_prefix, user.username)

    return APIKeyCreatedResponse(
        key=full_key,
        key_info=APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            created_at=api_key.created_at,
            last_used_at=api_key.last_used_at,
            expires_at=api_key.expires_at,
            is_active=api_key.is_active,
            scoped_permissions=list(scoped_permissions) if scoped_permissions else None,
        ),
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    user: User = Depends(get_current_user),
) -> Dict[str, bool]:
    """Revoke an API key.

    Users can revoke their own keys. Admins with api.keys.all can revoke any key.
    """
    store = _get_store()

    # Check if user owns the key or is admin
    if user.has_permission("api.keys.all"):
        # Admin can revoke any key
        success = await store.revoke_api_key(key_id)
    else:
        # User can only revoke own keys
        success = await store.revoke_api_key(key_id, user_id=user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or already revoked",
        )

    logger.info("API key revoked: %d by user %s", key_id, user.username)

    return {"success": True}


# ==================== User Registration (if enabled) ====================


@router.post("/register", response_model=LoginResponse)
async def register(
    request: Request,
    response: Response,
    data: RegisterRequest,
) -> LoginResponse:
    """Register a new user account.

    Note: This endpoint may be disabled in production environments.
    Check /setup/status for registration availability.
    """
    store = _get_store()

    # Check if registration is allowed (you might want to make this configurable)
    # For now, allow registration only if there are existing users (post-setup)
    if not await store.has_any_users():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /setup/first-admin for initial setup",
        )

    try:
        user = await store.create_user(
            email=data.email,
            username=data.username,
            password=data.password,
            display_name=data.display_name,
            level_names=["researcher"],  # Default level for new users
        )

        # Auto-login after registration
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")[:500]
        session_id = await store.create_session(
            user_id=user.id,
            ip_address=client_ip,
            user_agent=user_agent,
        )

        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            max_age=60 * 60 * 24,  # 24 hours
        )

        # Reload user with permissions
        user = await store.get_user(user.id)

        logger.info("New user registered: %s", user.username)

        return LoginResponse(
            success=True,
            user=user.to_dict(),
            message="Account created successfully",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ==================== Password Management ====================


class ChangePasswordRequest(BaseModel):
    """Password change request."""

    current_password: str
    new_password: str = Field(..., min_length=8)


@router.post("/change-password")
async def change_password(
    data: ChangePasswordRequest,
    user: User = Depends(get_current_user),
) -> Dict[str, bool]:
    """Change the current user's password."""
    if user.auth_provider != AuthProvider.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password change not supported for {user.auth_provider.value} accounts",
        )

    store = _get_store()

    # Verify current password
    verified_user = await store.authenticate_local(user.username, data.current_password)
    if not verified_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    # Update password
    await store.update_user(user.id, {"password": data.new_password})

    logger.info("Password changed for user: %s", user.username)

    return {"success": True}
