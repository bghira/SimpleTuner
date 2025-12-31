"""Base classes for external authentication providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExternalUser:
    """User information from an external auth provider.

    Common representation for users across OIDC, LDAP, etc.
    """

    external_id: str  # Unique ID from the provider (sub, dn, etc.)
    email: str
    username: str
    display_name: Optional[str] = None

    # Provider info
    provider_type: str = ""  # "oidc" or "ldap"
    provider_name: str = ""  # Instance name (e.g., "keycloak", "corporate-ldap")

    # Groups/roles from the provider
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    # Raw claims/attributes from provider
    raw_attributes: Dict[str, Any] = field(default_factory=dict)

    # Email verification status from provider
    email_verified: bool = False

    def get_suggested_levels(self, level_mapping: Dict[str, List[str]]) -> List[str]:
        """Get suggested SimpleTuner levels based on provider groups/roles.

        Args:
            level_mapping: Dict mapping SimpleTuner level names to provider group/role names

        Returns:
            List of SimpleTuner level names the user should have
        """
        user_memberships = set(self.groups + self.roles)
        levels = []

        for level_name, provider_values in level_mapping.items():
            for pv in provider_values:
                if pv in user_memberships:
                    levels.append(level_name)
                    break

        return levels if levels else ["researcher"]  # Default to researcher


@dataclass
class ProviderConfig:
    """Configuration for an external auth provider."""

    name: str  # Unique name for this provider instance
    provider_type: str  # "oidc" or "ldap"
    enabled: bool = True

    # Level/role mapping
    level_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Auto-provisioning settings
    auto_create_users: bool = True
    default_levels: List[str] = field(default_factory=lambda: ["researcher"])

    # Provider-specific config
    config: Dict[str, Any] = field(default_factory=dict)


class AuthProviderBase(ABC):
    """Base class for authentication providers.

    Subclasses implement specific auth protocols (OIDC, LDAP, etc.).
    """

    def __init__(self, config: ProviderConfig):
        """Initialize with provider configuration."""
        self.config = config
        self.name = config.name
        self.enabled = config.enabled

    @abstractmethod
    async def authenticate(
        self,
        credentials: Dict[str, Any],
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Authenticate a user with the provider.

        Args:
            credentials: Provider-specific credentials
                - OIDC: {"code": "...", "redirect_uri": "..."} or {"token": "..."}
                - LDAP: {"username": "...", "password": "..."}

        Returns:
            Tuple of:
                - success: Whether authentication succeeded
                - user: ExternalUser if successful
                - error: Error message if failed
        """
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> Tuple[bool, Optional[ExternalUser]]:
        """Validate an access/session token.

        Args:
            token: The token to validate

        Returns:
            Tuple of (is_valid, user_info)
        """
        pass

    @abstractmethod
    async def get_auth_url(self, redirect_uri: str, state: str) -> str:
        """Get the URL to redirect users to for authentication.

        Only applicable for OAuth/OIDC providers.

        Args:
            redirect_uri: Where to redirect after auth
            state: State parameter for CSRF protection

        Returns:
            URL to redirect to
        """
        pass

    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
        """Refresh an access token.

        Args:
            refresh_token: The refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token) or (None, None) if failed
        """
        pass

    def get_suggested_levels(self, user: ExternalUser) -> List[str]:
        """Get suggested SimpleTuner levels for a user.

        Uses the configured level_mapping.
        """
        mapping = self.config.level_mapping
        if not mapping:
            return self.config.default_levels

        return user.get_suggested_levels(mapping) or self.config.default_levels

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test the connection to the auth provider.

        Returns:
            Tuple of (success, error_message)
        """
        return True, None
