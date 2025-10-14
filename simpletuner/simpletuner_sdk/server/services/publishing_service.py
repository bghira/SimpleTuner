"""Service helpers for HuggingFace Hub publishing operations."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import status
from huggingface_hub import HfApi, HfFolder


class PublishingServiceError(Exception):
    """Domain error raised when publishing service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class PublishingService:
    """Coordinator for HuggingFace Hub publishing operations."""

    # License mapping based on model family
    LICENSE_MAP = {
        "flux": "flux-1-dev-non-commercial-license",
        "sdxl": "creativeml-openrail-m",
        "sd1x": "creativeml-openrail-m",
        "sd2x": "creativeml-openrail-m",
        "sd3": "stabilityai-ai-community",
    }
    DEFAULT_LICENSE = "apache-2.0"

    # Cache duration in seconds (5 minutes)
    CACHE_TTL = 300

    def __init__(self) -> None:
        self._api: Optional[HfApi] = None
        self._orgs_cache: Optional[Dict[str, Any]] = None
        self._orgs_cache_time: float = 0

    @property
    def api(self) -> HfApi:
        """Get or create HfApi instance."""
        if self._api is None:
            self._api = HfApi()
        return self._api

    def validate_token(self) -> Dict[str, Any]:
        """
        Validate HuggingFace token from ~/.cache/huggingface/token.

        Returns:
            Dictionary with validation status and user information.
        """
        try:
            # Try to get token from HfFolder (standard location)
            token = HfFolder.get_token()

            # If not found, try direct file read
            if not token:
                token_path = Path.home() / ".cache" / "huggingface" / "token"
                if token_path.exists():
                    with open(token_path, "r") as f:
                        token = f.read().strip()

            if not token:
                return {
                    "valid": False,
                    "message": "No HuggingFace token found. Please run 'huggingface-cli login'.",
                }

            # Validate token by trying to get user info
            try:
                user_info = self.api.whoami(token=token)
                return {
                    "valid": True,
                    "username": user_info.get("name"),
                    "message": "Token is valid",
                }
            except Exception as e:
                return {
                    "valid": False,
                    "message": f"Token is invalid or expired: {str(e)}",
                }

        except Exception as e:
            raise PublishingServiceError(
                f"Failed to validate token: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def save_token(self, token: str) -> Dict[str, Any]:
        """
        Save HuggingFace token to ~/.cache/huggingface/token.

        Args:
            token: The HuggingFace access token.

        Returns:
            Dictionary with validation status and user information.
        """
        try:
            # Validate token first
            try:
                user_info = self.api.whoami(token=token)
            except Exception as e:
                raise PublishingServiceError(f"Invalid token: {str(e)}", status.HTTP_400_BAD_REQUEST) from e

            # Save token to standard location
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token.strip())
            token_path.chmod(0o600)  # Secure permissions

            # Also save via HfFolder for compatibility
            HfFolder.save_token(token.strip())

            return {
                "valid": True,
                "username": user_info.get("name"),
                "message": "Token saved successfully",
            }

        except PublishingServiceError:
            raise
        except Exception as e:
            raise PublishingServiceError(
                f"Failed to save token: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def logout(self) -> Dict[str, Any]:
        """
        Remove HuggingFace token from ~/.cache/huggingface/token.

        Returns:
            Dictionary with logout status.
        """
        try:
            # Remove token file
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                token_path.unlink()

            # Also delete via HfFolder for compatibility
            try:
                HfFolder.delete_token()
            except Exception:
                pass  # May not exist, that's ok

            return {"success": True, "message": "Logged out successfully"}

        except Exception as e:
            raise PublishingServiceError(
                f"Failed to logout: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def check_repository(self, repo_id: str) -> Dict[str, Any]:
        """
        Check if a repository exists and is available.

        Args:
            repo_id: The repository ID in format "username/repo-name".

        Returns:
            Dictionary with repository status and information.
        """
        if not repo_id or "/" not in repo_id:
            raise PublishingServiceError(
                "Invalid repository ID. Must be in format 'username/repo-name'.",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            token = HfFolder.get_token()
            if not token:
                token_path = Path.home() / ".cache" / "huggingface" / "token"
                if token_path.exists():
                    with open(token_path, "r") as f:
                        token = f.read().strip()

            # Try to get repo info
            try:
                repo_info = self.api.repo_info(repo_id=repo_id, token=token)
                return {
                    "exists": True,
                    "available": False,
                    "message": f"Repository '{repo_id}' already exists",
                    "repo_id": repo_info.id,
                    "private": getattr(repo_info, "private", None),
                }
            except Exception:
                # Repository doesn't exist, which means it's available
                return {
                    "exists": False,
                    "available": True,
                    "message": f"Repository name '{repo_id}' is available",
                }

        except Exception as e:
            raise PublishingServiceError(
                f"Failed to check repository: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def get_user_organizations(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get list of organizations the user belongs to for namespace selection.

        Uses a 5-minute cache to avoid hitting HuggingFace API rate limits.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            Dictionary with user's username and list of organizations.
        """
        # Check if we have a valid cache
        current_time = time.time()
        cache_age = current_time - self._orgs_cache_time

        if not force_refresh and self._orgs_cache is not None and cache_age < self.CACHE_TTL:
            return self._orgs_cache

        try:
            token = HfFolder.get_token()
            if not token:
                token_path = Path.home() / ".cache" / "huggingface" / "token"
                if token_path.exists():
                    with open(token_path, "r") as f:
                        token = f.read().strip()

            if not token:
                raise PublishingServiceError(
                    "No HuggingFace token found. Please run 'huggingface-cli login'.",
                    status.HTTP_401_UNAUTHORIZED,
                )

            # Get user info from HuggingFace API
            user_info = self.api.whoami(token=token)
            username = user_info.get("name")

            # Get organizations
            organizations: List[str] = []
            orgs_info = user_info.get("orgs", [])
            if orgs_info:
                organizations = [org.get("name") for org in orgs_info if org.get("name")]

            result = {
                "username": username,
                "organizations": organizations,
                "namespaces": [username] + organizations,
            }

            # Update cache
            self._orgs_cache = result
            self._orgs_cache_time = current_time

            return result

        except PublishingServiceError:
            raise
        except Exception as e:
            raise PublishingServiceError(
                f"Failed to get user organizations: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def get_license_for_model(self, model_family: str) -> str:
        """
        Get the appropriate license for a model family.

        Args:
            model_family: The model family (e.g., "flux", "sdxl", "sd3").

        Returns:
            License identifier string.
        """
        if not model_family:
            return self.DEFAULT_LICENSE

        # Normalize model family for lookup
        model_family_lower = model_family.lower().strip()

        # Check exact match first
        if model_family_lower in self.LICENSE_MAP:
            return self.LICENSE_MAP[model_family_lower]

        # Check for partial matches (e.g., "sd-1-5" should match "sd1x")
        if any(key in model_family_lower for key in ["sd1", "sd-1"]):
            return self.LICENSE_MAP.get("sd1x", self.DEFAULT_LICENSE)
        if any(key in model_family_lower for key in ["sd2", "sd-2"]):
            return self.LICENSE_MAP.get("sd2x", self.DEFAULT_LICENSE)
        if any(key in model_family_lower for key in ["sd3", "sd-3"]):
            return self.LICENSE_MAP.get("sd3", self.DEFAULT_LICENSE)
        if "sdxl" in model_family_lower or "sd-xl" in model_family_lower:
            return self.LICENSE_MAP.get("sdxl", self.DEFAULT_LICENSE)
        if "flux" in model_family_lower:
            return self.LICENSE_MAP.get("flux", self.DEFAULT_LICENSE)

        return self.DEFAULT_LICENSE


# Singleton instance used by routes
PUBLISHING_SERVICE = PublishingService()
