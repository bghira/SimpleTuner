"""Manager for external authentication providers.

Handles provider registration, lookup, and user provisioning.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import AuthProviderBase, ExternalUser, ProviderConfig
from .ldap import LDAPProvider
from .oidc import OIDCProvider

logger = logging.getLogger(__name__)


# Provider type registry
PROVIDER_TYPES = {
    "oidc": OIDCProvider,
    "ldap": LDAPProvider,
}


class AuthProviderManager:
    """Manages external authentication providers.

    Handles:
    - Loading provider configurations
    - Creating provider instances
    - Routing auth requests to correct provider
    - User auto-provisioning
    """

    _instance: Optional["AuthProviderManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AuthProviderManager":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self):
        """Initialize the provider manager."""
        if getattr(self, "_initialized", False):
            return

        self._providers: Dict[str, AuthProviderBase] = {}
        self._configs: Dict[str, ProviderConfig] = {}
        self._config_file: Optional[Path] = None

        self._initialized = True

    def configure(self, config_file: Optional[Path] = None) -> None:
        """Load configuration from file or environment.

        Args:
            config_file: Path to JSON config file
        """
        self._config_file = config_file
        self._providers.clear()
        self._configs.clear()

        # Try to load from file
        if config_file and config_file.exists():
            self._load_config_file(config_file)

        # Also check for environment-based config
        self._load_env_config()

    def _load_config_file(self, config_file: Path) -> None:
        """Load provider configs from JSON file."""
        try:
            with open(config_file) as f:
                data = json.load(f)

            providers = data.get("auth_providers", [])
            for provider_data in providers:
                config = ProviderConfig(
                    name=provider_data["name"],
                    provider_type=provider_data["type"],
                    enabled=provider_data.get("enabled", True),
                    level_mapping=provider_data.get("level_mapping", {}),
                    auto_create_users=provider_data.get("auto_create_users", True),
                    default_levels=provider_data.get("default_levels", ["researcher"]),
                    config=provider_data.get("config", {}),
                )
                self._register_provider(config)

            logger.info("Loaded %d auth providers from %s", len(providers), config_file)

        except Exception as exc:
            logger.error("Failed to load auth config: %s", exc)

    def _load_env_config(self) -> None:
        """Load provider configs from environment variables.

        Supports:
        - SIMPLETUNER_OIDC_ISSUER, SIMPLETUNER_OIDC_CLIENT_ID, etc.
        - SIMPLETUNER_LDAP_SERVER, SIMPLETUNER_LDAP_BASE_DN, etc.
        """
        import os

        # Check for OIDC config
        oidc_issuer = os.environ.get("SIMPLETUNER_OIDC_ISSUER")
        oidc_client_id = os.environ.get("SIMPLETUNER_OIDC_CLIENT_ID")

        if oidc_issuer and oidc_client_id:
            config = ProviderConfig(
                name="oidc-env",
                provider_type="oidc",
                enabled=True,
                config={
                    "issuer": oidc_issuer,
                    "client_id": oidc_client_id,
                    "client_secret": os.environ.get("SIMPLETUNER_OIDC_CLIENT_SECRET"),
                    "scopes": os.environ.get("SIMPLETUNER_OIDC_SCOPES", "openid email profile").split(),
                },
            )
            self._register_provider(config)
            logger.info("Loaded OIDC provider from environment")

        # Check for LDAP config
        ldap_server = os.environ.get("SIMPLETUNER_LDAP_SERVER")
        ldap_base_dn = os.environ.get("SIMPLETUNER_LDAP_BASE_DN")

        if ldap_server and ldap_base_dn:
            config = ProviderConfig(
                name="ldap-env",
                provider_type="ldap",
                enabled=True,
                config={
                    "server": ldap_server,
                    "base_dn": ldap_base_dn,
                    "bind_dn": os.environ.get("SIMPLETUNER_LDAP_BIND_DN"),
                    "bind_password": os.environ.get("SIMPLETUNER_LDAP_BIND_PASSWORD"),
                    "user_search_filter": os.environ.get(
                        "SIMPLETUNER_LDAP_USER_FILTER",
                        "(uid={username})",
                    ),
                },
            )
            self._register_provider(config)
            logger.info("Loaded LDAP provider from environment")

    def _register_provider(self, config: ProviderConfig) -> None:
        """Register a provider from config."""
        if config.provider_type not in PROVIDER_TYPES:
            logger.error("Unknown provider type: %s", config.provider_type)
            return

        provider_class = PROVIDER_TYPES[config.provider_type]
        try:
            provider = provider_class(config)
            self._providers[config.name] = provider
            self._configs[config.name] = config
        except Exception as exc:
            logger.error("Failed to create provider %s: %s", config.name, exc)

    def get_provider(self, name: str) -> Optional[AuthProviderBase]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_enabled_providers(self) -> List[AuthProviderBase]:
        """Get all enabled providers."""
        return [p for p in self._providers.values() if p.enabled]

    def get_oidc_providers(self) -> List[AuthProviderBase]:
        """Get all OIDC providers."""
        return [p for p in self._providers.values() if p.enabled and isinstance(p, OIDCProvider)]

    def get_ldap_providers(self) -> List[AuthProviderBase]:
        """Get all LDAP providers."""
        return [p for p in self._providers.values() if p.enabled and isinstance(p, LDAPProvider)]

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers."""
        return [
            {
                "name": config.name,
                "type": config.provider_type,
                "enabled": config.enabled,
                "auto_create_users": config.auto_create_users,
            }
            for config in self._configs.values()
        ]

    async def authenticate(
        self,
        provider_name: str,
        credentials: Dict[str, Any],
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Authenticate using a specific provider.

        Args:
            provider_name: Name of the provider to use
            credentials: Provider-specific credentials

        Returns:
            Tuple of (success, external_user, error)
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return False, None, f"Unknown provider: {provider_name}"

        if not provider.enabled:
            return False, None, f"Provider disabled: {provider_name}"

        return await provider.authenticate(credentials)

    async def authenticate_ldap(
        self,
        username: str,
        password: str,
        provider_name: Optional[str] = None,
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Authenticate with LDAP.

        If provider_name is not specified, tries all LDAP providers.
        """
        if provider_name:
            return await self.authenticate(
                provider_name,
                {"username": username, "password": password},
            )

        # Try all LDAP providers
        ldap_providers = self.get_ldap_providers()
        if not ldap_providers:
            return False, None, "No LDAP providers configured"

        for provider in ldap_providers:
            success, user, error = await provider.authenticate({"username": username, "password": password})
            if success:
                return True, user, None

        return False, None, "Authentication failed with all LDAP providers"

    async def provision_user(
        self,
        external_user: ExternalUser,
        user_store,
    ) -> Optional[Any]:
        """Create or update a local user from external user info.

        Args:
            external_user: User info from external provider
            user_store: UserStore instance

        Returns:
            The local User object, or None if provisioning disabled
        """
        provider_name = external_user.provider_name
        config = self._configs.get(provider_name)

        if not config or not config.auto_create_users:
            return None

        provider = self._providers.get(provider_name)
        if not provider:
            return None

        # Check if user already exists
        existing = await user_store.get_user_by_external_id(
            external_user.external_id,
            external_user.provider_type,
        )

        if existing:
            # Update existing user
            await user_store.update_user(
                existing.id,
                {
                    "email": external_user.email,
                    "display_name": external_user.display_name,
                    "email_verified": external_user.email_verified,
                },
            )
            return await user_store.get_user(existing.id)

        # Create new user
        suggested_levels = provider.get_suggested_levels(external_user)

        user = await user_store.create_external_user(
            email=external_user.email,
            username=external_user.username,
            display_name=external_user.display_name,
            external_id=external_user.external_id,
            auth_provider=external_user.provider_type,
            level_names=suggested_levels,
            email_verified=external_user.email_verified,
        )

        logger.info(
            "Provisioned user %s from %s with levels %s",
            user.username,
            provider_name,
            suggested_levels,
        )

        return user

    async def test_all_providers(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Test connections to all providers.

        Returns:
            Dict mapping provider name to (success, error) tuple
        """
        results = {}
        for name, provider in self._providers.items():
            success, error = await provider.test_connection()
            results[name] = (success, error)
        return results

    def add_provider(self, config: ProviderConfig) -> bool:
        """Add a new provider configuration.

        Args:
            config: Provider configuration

        Returns:
            True if added successfully
        """
        if config.name in self._configs:
            return False

        self._register_provider(config)

        # Save to config file if configured
        if self._config_file:
            self._save_config()

        return True

    def remove_provider(self, name: str) -> bool:
        """Remove a provider.

        Args:
            name: Provider name

        Returns:
            True if removed
        """
        if name not in self._configs:
            return False

        del self._providers[name]
        del self._configs[name]

        if self._config_file:
            self._save_config()

        return True

    def update_provider(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update a provider configuration.

        Args:
            name: Provider name
            updates: Fields to update

        Returns:
            True if updated
        """
        if name not in self._configs:
            return False

        config = self._configs[name]

        # Update config fields
        if "enabled" in updates:
            config.enabled = updates["enabled"]
        if "auto_create_users" in updates:
            config.auto_create_users = updates["auto_create_users"]
        if "level_mapping" in updates:
            config.level_mapping = updates["level_mapping"]
        if "default_levels" in updates:
            config.default_levels = updates["default_levels"]
        if "config" in updates:
            config.config.update(updates["config"])

        # Recreate provider with new config
        self._register_provider(config)

        if self._config_file:
            self._save_config()

        return True

    def _save_config(self) -> None:
        """Save current config to file."""
        if not self._config_file:
            return

        data = {
            "auth_providers": [
                {
                    "name": config.name,
                    "type": config.provider_type,
                    "enabled": config.enabled,
                    "auto_create_users": config.auto_create_users,
                    "level_mapping": config.level_mapping,
                    "default_levels": config.default_levels,
                    "config": config.config,
                }
                for config in self._configs.values()
            ]
        }

        try:
            with open(self._config_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save auth config: %s", exc)
