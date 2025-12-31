"""Credential resolution with user-specific override support.

Implements the credential lookup chain:
1. Per-user credentials (from database)
2. Global credentials (from SecretsManager)

This allows users to provide their own API keys while falling back
to shared/organization keys when not configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .auth.user_store import UserStore

logger = logging.getLogger(__name__)


class CredentialResolver:
    """Resolves provider credentials with user-specific override support."""

    # Well-known credential names
    REPLICATE_API_TOKEN = "api_token"
    HF_TOKEN = "api_token"

    def __init__(self, user_store: Optional["UserStore"] = None):
        """Initialize the resolver.

        Args:
            user_store: UserStore instance for per-user credentials.
                       If not provided, will use singleton.
        """
        self._user_store = user_store
        self._secrets = None

    def _get_user_store(self) -> "UserStore":
        """Get or create the user store."""
        if self._user_store is None:
            from .auth.user_store import UserStore

            self._user_store = UserStore()
        return self._user_store

    def _get_secrets_manager(self):
        """Get or create the secrets manager."""
        if self._secrets is None:
            from .secrets import get_secrets_manager

            self._secrets = get_secrets_manager()
        return self._secrets

    async def get_replicate_token(self, user_id: Optional[int] = None) -> Optional[str]:
        """Get Replicate API token with user override support.

        Args:
            user_id: Optional user ID for per-user credentials

        Returns:
            API token (user's if available, otherwise global), or None
        """
        return await self.get_credential(
            provider="replicate",
            credential_name=self.REPLICATE_API_TOKEN,
            user_id=user_id,
            global_key="REPLICATE_API_TOKEN",
        )

    async def get_hf_token(self, user_id: Optional[int] = None) -> Optional[str]:
        """Get HuggingFace token with user override support.

        Args:
            user_id: Optional user ID for per-user credentials

        Returns:
            HF token (user's if available, otherwise global), or None
        """
        return await self.get_credential(
            provider="huggingface",
            credential_name=self.HF_TOKEN,
            user_id=user_id,
            global_key="HF_TOKEN",
        )

    async def get_credential(
        self,
        provider: str,
        credential_name: str,
        user_id: Optional[int] = None,
        global_key: Optional[str] = None,
    ) -> Optional[str]:
        """Get a credential with user override support.

        Lookup chain:
        1. User-specific credential (if user_id provided)
        2. Global credential from SecretsManager

        Args:
            provider: Provider name (e.g., "replicate", "huggingface")
            credential_name: Credential name (e.g., "api_token")
            user_id: Optional user ID for per-user lookup
            global_key: Optional key for global secrets lookup

        Returns:
            Credential value, or None if not found
        """
        # Try user-specific credential first
        if user_id is not None:
            try:
                user_store = self._get_user_store()
                user_cred = await user_store.get_provider_credential(
                    user_id=user_id,
                    provider=provider,
                    credential_name=credential_name,
                )
                if user_cred:
                    logger.debug("Using user-specific %s/%s credential for user %d", provider, credential_name, user_id)
                    # Update last_used timestamp
                    await user_store.mark_credential_used(user_id, provider, credential_name)
                    return user_cred
            except Exception as exc:
                logger.warning("Failed to get user credential %s/%s: %s", provider, credential_name, exc)

        # Fall back to global credential
        if global_key:
            secrets = self._get_secrets_manager()
            global_cred = secrets.get(global_key)
            if global_cred:
                logger.debug("Using global %s credential", global_key)
                return global_cred

        return None

    async def has_user_credential(
        self,
        user_id: int,
        provider: str,
        credential_name: str,
    ) -> bool:
        """Check if a user has a specific credential configured.

        Args:
            user_id: User ID
            provider: Provider name
            credential_name: Credential name

        Returns:
            True if user has the credential
        """
        try:
            user_store = self._get_user_store()
            creds = await user_store.list_provider_credentials(user_id, provider)
            return any(c["credential_name"] == credential_name and c["is_active"] for c in creds)
        except Exception:
            return False

    async def get_credential_source(
        self,
        provider: str,
        credential_name: str,
        user_id: Optional[int] = None,
        global_key: Optional[str] = None,
    ) -> Optional[str]:
        """Determine where a credential would come from.

        Args:
            provider: Provider name
            credential_name: Credential name
            user_id: Optional user ID
            global_key: Optional global secrets key

        Returns:
            "user", "global", or None if no credential available
        """
        if user_id is not None:
            if await self.has_user_credential(user_id, provider, credential_name):
                return "user"

        if global_key:
            secrets = self._get_secrets_manager()
            if secrets.get(global_key):
                return "global"

        return None


# Singleton instance
_resolver: Optional[CredentialResolver] = None


def get_credential_resolver() -> CredentialResolver:
    """Get the global CredentialResolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = CredentialResolver()
    return _resolver


def reset_credential_resolver() -> None:
    """Reset the global resolver (for testing)."""
    global _resolver
    _resolver = None
