"""External authentication endpoints for OIDC and LDAP."""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ...services.cloud.auth import get_current_user, require_permission
from ...services.cloud.auth.middleware import SESSION_COOKIE_NAME
from ...services.cloud.auth.models import User
from ...services.cloud.auth.providers import AuthProviderManager
from ...services.cloud.auth.providers.base import ProviderConfig
from ...services.cloud.auth.user_store import UserStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external-auth", tags=["external-auth"])


# --- Request/Response Models ---


class ProviderConfigRequest(BaseModel):
    """Request to configure an auth provider."""

    name: str = Field(min_length=1)
    provider_type: str = Field(pattern="^(oidc|ldap)$")
    enabled: bool = True
    auto_create_users: bool = True
    default_levels: List[str] = Field(default_factory=lambda: ["researcher"])
    level_mapping: Dict[str, List[str]] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Response for a provider."""

    name: str
    type: str
    enabled: bool
    auto_create_users: bool


class ProviderListResponse(BaseModel):
    """Response for listing providers."""

    providers: List[ProviderResponse]


class LDAPLoginRequest(BaseModel):
    """Request for LDAP login."""

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    provider: Optional[str] = None  # Optional provider name


class ExternalLoginResponse(BaseModel):
    """Response for external login."""

    success: bool
    message: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class OIDCStartResponse(BaseModel):
    """Response for OIDC flow start."""

    auth_url: str
    state: str


# --- Provider Management Endpoints ---


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    user: User = Depends(require_permission("admin.config")),
) -> ProviderListResponse:
    """List configured authentication providers."""
    manager = AuthProviderManager()
    providers = manager.list_providers()

    return ProviderListResponse(
        providers=[
            ProviderResponse(
                name=p["name"],
                type=p["type"],
                enabled=p["enabled"],
                auto_create_users=p["auto_create_users"],
            )
            for p in providers
        ]
    )


@router.post("/providers", response_model=ProviderResponse)
async def create_provider(
    request: ProviderConfigRequest,
    user: User = Depends(require_permission("admin.config")),
) -> ProviderResponse:
    """Create a new authentication provider."""
    manager = AuthProviderManager()

    config = ProviderConfig(
        name=request.name,
        provider_type=request.provider_type,
        enabled=request.enabled,
        auto_create_users=request.auto_create_users,
        default_levels=request.default_levels,
        level_mapping=request.level_mapping,
        config=request.config,
    )

    success = manager.add_provider(config)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider already exists: {request.name}",
        )

    return ProviderResponse(
        name=request.name,
        type=request.provider_type,
        enabled=request.enabled,
        auto_create_users=request.auto_create_users,
    )


@router.patch("/providers/{name}")
async def update_provider(
    name: str,
    request: ProviderConfigRequest,
    user: User = Depends(require_permission("admin.config")),
) -> Dict[str, Any]:
    """Update an authentication provider."""
    manager = AuthProviderManager()

    updates = {
        "enabled": request.enabled,
        "auto_create_users": request.auto_create_users,
        "default_levels": request.default_levels,
        "level_mapping": request.level_mapping,
        "config": request.config,
    }

    success = manager.update_provider(name, updates)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not found: {name}",
        )

    return {"success": True, "name": name}


@router.delete("/providers/{name}")
async def delete_provider(
    name: str,
    user: User = Depends(require_permission("admin.config")),
) -> Dict[str, Any]:
    """Delete an authentication provider."""
    manager = AuthProviderManager()

    success = manager.remove_provider(name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not found: {name}",
        )

    return {"success": True, "name": name}


@router.post("/providers/{name}/test")
async def test_provider(
    name: str,
    user: User = Depends(require_permission("admin.config")),
) -> Dict[str, Any]:
    """Test connection to an authentication provider."""
    import time

    manager = AuthProviderManager()

    provider = manager.get_provider(name)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not found: {name}",
        )

    start_time = time.perf_counter()
    success, error = await provider.test_connection()
    latency_ms = round((time.perf_counter() - start_time) * 1000)

    return {
        "success": success,
        "error": error,
        "provider": name,
        "latency_ms": latency_ms,
    }


# --- OIDC Flow Endpoints ---


@router.get("/oidc/start")
async def start_oidc_flow(
    provider: str = Query(..., description="OIDC provider name"),
    redirect_uri: str = Query(..., description="Callback URL"),
) -> OIDCStartResponse:
    """Start OIDC authentication flow.

    Returns the authorization URL to redirect the user to.
    """
    manager = AuthProviderManager()
    oidc_provider = manager.get_provider(provider)

    if not oidc_provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not found: {provider}",
        )

    if not oidc_provider.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider disabled: {provider}",
        )

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)

    try:
        auth_url = await oidc_provider.get_auth_url(redirect_uri, state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    # Store state in database for validation (works across workers/restarts)
    user_store = UserStore()
    stored = await user_store.create_oauth_state(
        state=state,
        provider=provider,
        redirect_uri=redirect_uri,
    )
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store OAuth state",
        )

    return OIDCStartResponse(auth_url=auth_url, state=state)


@router.get("/oidc/callback")
async def oidc_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
) -> ExternalLoginResponse:
    """Handle OIDC callback.

    Exchanges the authorization code for tokens and creates/updates the user.
    """
    # Validate and consume state from database (atomic operation)
    user_store = UserStore()
    state_data = await user_store.consume_oauth_state(state)

    if not state_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state",
        )

    provider_name = state_data["provider"]
    redirect_uri = state_data["redirect_uri"]

    manager = AuthProviderManager()
    provider = manager.get_provider(provider_name)

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider not found: {provider_name}",
        )

    # Exchange code for tokens and get user info
    success, external_user, error = await provider.authenticate(
        {
            "code": code,
            "redirect_uri": redirect_uri,
        }
    )

    if not success or not external_user:
        return ExternalLoginResponse(
            success=False,
            message=error or "Authentication failed",
        )

    # Provision local user
    user_store = UserStore()
    user = await manager.provision_user(external_user, user_store)

    if not user:
        return ExternalLoginResponse(
            success=False,
            message="User provisioning disabled or failed",
        )

    # Create session
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    session_id = await user_store.create_session(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    # Update last login
    await user_store.update_last_login(user.id)

    return ExternalLoginResponse(
        success=True,
        message="Login successful",
        user=user.to_dict(),
        session_id=session_id,
    )


@router.get("/oidc/callback-redirect")
async def oidc_callback_redirect(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    frontend_url: str = Query("/", description="URL to redirect to after login"),
) -> RedirectResponse:
    """Handle OIDC callback with redirect to frontend.

    Like /oidc/callback but redirects to the frontend with session cookie set.
    """
    result = await oidc_callback(request, code, state)

    if not result.success:
        return RedirectResponse(
            url=f"{frontend_url}?error={result.message}",
            status_code=302,
        )

    response = RedirectResponse(url=frontend_url, status_code=302)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        result.session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,
    )

    return response


# --- LDAP Login Endpoints ---


@router.post("/ldap/login", response_model=ExternalLoginResponse)
async def ldap_login(
    request_body: LDAPLoginRequest,
    request: Request,
) -> ExternalLoginResponse:
    """Authenticate with LDAP.

    Tries all configured LDAP providers if provider is not specified.
    """
    manager = AuthProviderManager()

    success, external_user, error = await manager.authenticate_ldap(
        username=request_body.username,
        password=request_body.password,
        provider_name=request_body.provider,
    )

    if not success or not external_user:
        return ExternalLoginResponse(
            success=False,
            message=error or "Authentication failed",
        )

    # Provision local user
    user_store = UserStore()
    user = await manager.provision_user(external_user, user_store)

    if not user:
        return ExternalLoginResponse(
            success=False,
            message="User provisioning disabled or failed",
        )

    # Create session
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    session_id = await user_store.create_session(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    # Update last login
    await user_store.update_last_login(user.id)

    return ExternalLoginResponse(
        success=True,
        message="Login successful",
        user=user.to_dict(),
        session_id=session_id,
    )


# --- Utility Endpoints ---


@router.get("/available")
async def get_available_providers() -> Dict[str, Any]:
    """Get available external authentication providers.

    Public endpoint for login UI to know which providers are available.
    """
    manager = AuthProviderManager()

    oidc_providers = manager.get_oidc_providers()
    ldap_providers = manager.get_ldap_providers()

    return {
        "oidc": [{"name": p.name} for p in oidc_providers],
        "ldap": [{"name": p.name} for p in ldap_providers],
        "has_external_auth": bool(oidc_providers or ldap_providers),
    }
