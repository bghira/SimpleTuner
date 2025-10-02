"""HuggingFace Hub publishing routes for SimpleTuner WebUI."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.publishing_service import (
    PUBLISHING_SERVICE,
    PublishingServiceError,
)

router = APIRouter(prefix="/api/publishing", tags=["publishing"])


class RepositoryCheckRequest(BaseModel):
    """Request model for repository availability check."""

    repo_id: str = Field(..., description="Repository ID in format 'username/repo-name'")


class LicenseRequest(BaseModel):
    """Request model for getting license for a model family."""

    model_family: str = Field(..., description="Model family (e.g., flux, sdxl, sd3)")


class TokenSaveRequest(BaseModel):
    """Request model for saving HuggingFace token."""

    token: str = Field(..., description="HuggingFace access token", min_length=1)


def _call_service(func, *args, **kwargs):
    """Execute a service call and translate domain errors to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except PublishingServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("/token/validate")
async def validate_token() -> Dict[str, Any]:
    """
    Validate the HuggingFace Hub token stored in ~/.cache/huggingface/token.

    Returns:
        Dictionary with validation status and user information.
    """
    return _call_service(PUBLISHING_SERVICE.validate_token)


@router.post("/repository/check")
async def check_repository(request: RepositoryCheckRequest) -> Dict[str, Any]:
    """
    Check if a repository exists and is available for creation.

    Args:
        request: Repository check request with repo_id.

    Returns:
        Dictionary with repository availability status.
    """
    return _call_service(PUBLISHING_SERVICE.check_repository, request.repo_id)


@router.get("/namespaces")
async def get_namespaces() -> Dict[str, Any]:
    """
    Get available namespaces (user and organizations) for repository creation.

    Returns:
        Dictionary with username, organizations list, and combined namespaces.
    """
    return _call_service(PUBLISHING_SERVICE.get_user_organizations)


@router.post("/license")
async def get_license(request: LicenseRequest) -> Dict[str, str]:
    """
    Get appropriate license for a model family.

    Args:
        request: License request with model_family.

    Returns:
        Dictionary with license identifier.
    """
    license_id = PUBLISHING_SERVICE.get_license_for_model(request.model_family)
    return {"license": license_id, "model_family": request.model_family}


@router.post("/token/save")
async def save_token(request: TokenSaveRequest) -> Dict[str, Any]:
    """
    Save HuggingFace Hub token to ~/.cache/huggingface/token.

    Args:
        request: Token save request with token string.

    Returns:
        Dictionary with validation status and user information.
    """
    return _call_service(PUBLISHING_SERVICE.save_token, request.token)


@router.post("/token/logout")
async def logout() -> Dict[str, Any]:
    """
    Remove HuggingFace Hub token (logout).

    Returns:
        Dictionary with logout status.
    """
    return _call_service(PUBLISHING_SERVICE.logout)
