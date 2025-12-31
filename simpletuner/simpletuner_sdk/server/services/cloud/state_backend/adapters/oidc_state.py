"""OIDC State Store Adapter.

Manages OAuth/OIDC flow state using the pluggable state backend.
Replaces in-memory _pending_states dict with persistent, multi-worker safe storage.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..protocols import StateBackendProtocol


@dataclass
class OIDCPendingState:
    """Pending OIDC authentication state.

    Stored during OAuth flow initiation, retrieved on callback.
    """

    state: str  # Random state parameter
    nonce: str  # Random nonce for ID token validation
    provider_id: str  # Provider configuration ID
    redirect_uri: str  # Where to redirect after auth
    code_verifier: Optional[str] = None  # PKCE code verifier
    created_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "state": self.state,
            "nonce": self.nonce,
            "provider_id": self.provider_id,
            "redirect_uri": self.redirect_uri,
            "code_verifier": self.code_verifier,
            "created_at": self.created_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OIDCPendingState":
        """Deserialize from dictionary."""
        return cls(
            state=data["state"],
            nonce=data["nonce"],
            provider_id=data["provider_id"],
            redirect_uri=data["redirect_uri"],
            code_verifier=data.get("code_verifier"),
            created_at=data.get("created_at", time.time()),
            extra=data.get("extra", {}),
        )


@dataclass
class OIDCDiscoveryDocument:
    """Cached OIDC discovery document."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    scopes_supported: list[str] = field(default_factory=list)
    response_types_supported: list[str] = field(default_factory=list)
    claims_supported: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "userinfo_endpoint": self.userinfo_endpoint,
            "jwks_uri": self.jwks_uri,
            "scopes_supported": self.scopes_supported,
            "response_types_supported": self.response_types_supported,
            "claims_supported": self.claims_supported,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OIDCDiscoveryDocument":
        """Deserialize from dictionary."""
        return cls(
            issuer=data["issuer"],
            authorization_endpoint=data["authorization_endpoint"],
            token_endpoint=data["token_endpoint"],
            userinfo_endpoint=data.get("userinfo_endpoint"),
            jwks_uri=data.get("jwks_uri"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", []),
            claims_supported=data.get("claims_supported", []),
            raw=data.get("raw", {}),
        )


class OIDCStateStore:
    """OIDC state management using pluggable state backend.

    Manages:
        - Pending authentication states (short TTL, consumed on callback)
        - Discovery document cache (medium TTL)
        - JWKS cache (medium TTL)

    Example:
        backend = await get_state_backend()
        store = OIDCStateStore(backend)

        # Create pending state for auth initiation
        pending = await store.create_pending_state(
            provider_id="google",
            redirect_uri="/callback",
            use_pkce=True,
        )

        # On callback, retrieve and consume
        pending = await store.consume_pending_state(state_param)
        if pending:
            # Validate nonce in ID token
            pass

        # Cache discovery document
        await store.cache_discovery(issuer, discovery_doc)
        doc = await store.get_discovery(issuer)
    """

    # TTL defaults
    PENDING_STATE_TTL = 600  # 10 minutes for auth flow
    DISCOVERY_TTL = 3600  # 1 hour for discovery docs
    JWKS_TTL = 3600  # 1 hour for JWKS

    def __init__(
        self,
        backend: StateBackendProtocol,
        pending_ttl: int = PENDING_STATE_TTL,
        discovery_ttl: int = DISCOVERY_TTL,
        jwks_ttl: int = JWKS_TTL,
        key_prefix: str = "oidc:",
    ):
        """Initialize OIDC state store.

        Args:
            backend: State backend instance.
            pending_ttl: TTL for pending auth states.
            discovery_ttl: TTL for discovery document cache.
            jwks_ttl: TTL for JWKS cache.
            key_prefix: Prefix for all OIDC keys.
        """
        self._backend = backend
        self._pending_ttl = pending_ttl
        self._discovery_ttl = discovery_ttl
        self._jwks_ttl = jwks_ttl
        self._key_prefix = key_prefix

    def _pending_key(self, state: str) -> str:
        """Get key for pending state."""
        return f"{self._key_prefix}pending:{state}"

    def _discovery_key(self, issuer: str) -> str:
        """Get key for discovery document."""
        # Normalize issuer URL
        issuer = issuer.rstrip("/")
        return f"{self._key_prefix}discovery:{issuer}"

    def _jwks_key(self, jwks_uri: str) -> str:
        """Get key for JWKS."""
        return f"{self._key_prefix}jwks:{jwks_uri}"

    @staticmethod
    def generate_state() -> str:
        """Generate cryptographically secure state parameter."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_nonce() -> str:
        """Generate cryptographically secure nonce."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_code_verifier() -> str:
        """Generate PKCE code verifier."""
        return secrets.token_urlsafe(64)

    async def create_pending_state(
        self,
        provider_id: str,
        redirect_uri: str,
        use_pkce: bool = True,
        extra: Optional[dict[str, Any]] = None,
    ) -> OIDCPendingState:
        """Create and store a pending authentication state.

        Args:
            provider_id: Provider configuration ID.
            redirect_uri: Post-auth redirect URI.
            use_pkce: Whether to use PKCE (recommended).
            extra: Additional data to store with state.

        Returns:
            OIDCPendingState with generated parameters.
        """
        pending = OIDCPendingState(
            state=self.generate_state(),
            nonce=self.generate_nonce(),
            provider_id=provider_id,
            redirect_uri=redirect_uri,
            code_verifier=self.generate_code_verifier() if use_pkce else None,
            extra=extra or {},
        )

        # Store as JSON
        key = self._pending_key(pending.state)
        data = json.dumps(pending.to_dict()).encode()
        await self._backend.set(key, data, ttl=self._pending_ttl)

        return pending

    async def get_pending_state(self, state: str) -> Optional[OIDCPendingState]:
        """Get pending state without consuming it.

        Args:
            state: State parameter from callback.

        Returns:
            OIDCPendingState if found, None otherwise.
        """
        key = self._pending_key(state)
        data = await self._backend.get(key)

        if data is None:
            return None

        try:
            parsed = json.loads(data.decode())
            return OIDCPendingState.from_dict(parsed)
        except (json.JSONDecodeError, KeyError):
            # Invalid data, clean it up
            await self._backend.delete(key)
            return None

    async def consume_pending_state(self, state: str) -> Optional[OIDCPendingState]:
        """Retrieve and delete pending state (one-time use).

        Args:
            state: State parameter from callback.

        Returns:
            OIDCPendingState if found, None otherwise.
        """
        pending = await self.get_pending_state(state)

        if pending:
            # Delete after retrieval (consume)
            key = self._pending_key(state)
            await self._backend.delete(key)

        return pending

    async def delete_pending_state(self, state: str) -> bool:
        """Delete a pending state.

        Args:
            state: State parameter.

        Returns:
            True if existed and was deleted.
        """
        key = self._pending_key(state)
        return await self._backend.delete(key)

    async def cache_discovery(
        self,
        issuer: str,
        document: OIDCDiscoveryDocument,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache OIDC discovery document.

        Args:
            issuer: Issuer URL.
            document: Discovery document.
            ttl: Optional TTL override.
        """
        key = self._discovery_key(issuer)
        data = json.dumps(document.to_dict()).encode()
        await self._backend.set(key, data, ttl=ttl or self._discovery_ttl)

    async def get_discovery(self, issuer: str) -> Optional[OIDCDiscoveryDocument]:
        """Get cached discovery document.

        Args:
            issuer: Issuer URL.

        Returns:
            OIDCDiscoveryDocument if cached, None otherwise.
        """
        key = self._discovery_key(issuer)
        data = await self._backend.get(key)

        if data is None:
            return None

        try:
            parsed = json.loads(data.decode())
            return OIDCDiscoveryDocument.from_dict(parsed)
        except (json.JSONDecodeError, KeyError):
            await self._backend.delete(key)
            return None

    async def invalidate_discovery(self, issuer: str) -> bool:
        """Invalidate cached discovery document.

        Args:
            issuer: Issuer URL.

        Returns:
            True if was cached.
        """
        key = self._discovery_key(issuer)
        return await self._backend.delete(key)

    async def cache_jwks(
        self,
        jwks_uri: str,
        jwks: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache JWKS.

        Args:
            jwks_uri: JWKS URI.
            jwks: JWKS document.
            ttl: Optional TTL override.
        """
        key = self._jwks_key(jwks_uri)
        data = json.dumps(jwks).encode()
        await self._backend.set(key, data, ttl=ttl or self._jwks_ttl)

    async def get_jwks(self, jwks_uri: str) -> Optional[dict[str, Any]]:
        """Get cached JWKS.

        Args:
            jwks_uri: JWKS URI.

        Returns:
            JWKS dict if cached, None otherwise.
        """
        key = self._jwks_key(jwks_uri)
        data = await self._backend.get(key)

        if data is None:
            return None

        try:
            return json.loads(data.decode())
        except json.JSONDecodeError:
            await self._backend.delete(key)
            return None

    async def invalidate_jwks(self, jwks_uri: str) -> bool:
        """Invalidate cached JWKS.

        Args:
            jwks_uri: JWKS URI.

        Returns:
            True if was cached.
        """
        key = self._jwks_key(jwks_uri)
        return await self._backend.delete(key)

    async def clear_all(self) -> int:
        """Clear all OIDC state (pending, discovery, JWKS).

        Returns:
            Number of keys deleted.
        """
        return await self._backend.delete_prefix(self._key_prefix)

    async def clear_pending_states(self) -> int:
        """Clear all pending states.

        Returns:
            Number of states deleted.
        """
        return await self._backend.delete_prefix(f"{self._key_prefix}pending:")

    async def clear_discovery_cache(self) -> int:
        """Clear discovery cache.

        Returns:
            Number of entries deleted.
        """
        return await self._backend.delete_prefix(f"{self._key_prefix}discovery:")

    async def clear_jwks_cache(self) -> int:
        """Clear JWKS cache.

        Returns:
            Number of entries deleted.
        """
        return await self._backend.delete_prefix(f"{self._key_prefix}jwks:")
