"""OpenID Connect (OIDC) authentication provider.

Supports authentication with OIDC-compliant identity providers like
Keycloak, Auth0, Okta, Azure AD, Google, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from .base import AuthProviderBase, ExternalUser, ProviderConfig

logger = logging.getLogger(__name__)


class OIDCProvider(AuthProviderBase):
    """OpenID Connect authentication provider.

    Configuration options (in config.config):
        - issuer: OIDC issuer URL (e.g., https://keycloak.example.com/realms/myrealm)
        - client_id: OAuth client ID
        - client_secret: OAuth client secret (optional for public clients)
        - scopes: List of scopes to request (default: ["openid", "email", "profile"])
        - groups_claim: Claim name for groups (default: "groups")
        - roles_claim: Claim name for roles (default: "roles")
        - username_claim: Claim for username (default: "preferred_username")
        - verify_ssl: Verify SSL certificates (default: True)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        # Accept both "issuer" and "issuer_url" for compatibility
        self._issuer = (config.config.get("issuer") or config.config.get("issuer_url", "")).rstrip("/")
        self._client_id = config.config.get("client_id", "")
        self._client_secret = config.config.get("client_secret")
        self._scopes = config.config.get("scopes", ["openid", "email", "profile"])
        self._groups_claim = config.config.get("groups_claim", "groups")
        self._roles_claim = config.config.get("roles_claim", "roles")
        self._username_claim = config.config.get("username_claim", "preferred_username")
        self._verify_ssl = config.config.get("verify_ssl", True)

        # Discovery endpoints (populated on first use)
        self._discovery: Optional[Dict[str, Any]] = None
        self._jwks: Optional[Dict[str, Any]] = None
        self._jwks_updated: float = 0

    async def _get_discovery(self) -> Dict[str, Any]:
        """Fetch OIDC discovery document."""
        if self._discovery is not None:
            return self._discovery

        import aiohttp

        discovery_url = f"{self._issuer}/.well-known/openid-configuration"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    discovery_url,
                    ssl=self._verify_ssl,
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Discovery failed: {response.status}")
                    self._discovery = await response.json()
                    return self._discovery
        except Exception as exc:
            logger.error("OIDC discovery failed for %s: %s", self._issuer, exc)
            raise

    async def _get_jwks(self) -> Dict[str, Any]:
        """Fetch JWKS for token validation."""
        # Refresh JWKS every 5 minutes
        if self._jwks is not None and time.time() - self._jwks_updated < 300:
            return self._jwks

        import aiohttp

        discovery = await self._get_discovery()
        jwks_uri = discovery.get("jwks_uri")

        if not jwks_uri:
            raise ValueError("No jwks_uri in discovery document")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    jwks_uri,
                    ssl=self._verify_ssl,
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"JWKS fetch failed: {response.status}")
                    self._jwks = await response.json()
                    self._jwks_updated = time.time()
                    return self._jwks
        except Exception as exc:
            logger.error("JWKS fetch failed: %s", exc)
            raise

    async def authenticate(
        self,
        credentials: Dict[str, Any],
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Authenticate using authorization code or access token.

        Args:
            credentials: Either:
                - {"code": "...", "redirect_uri": "..."} for auth code flow
                - {"token": "..."} for token validation

        Returns:
            Tuple of (success, user, error)
        """
        if "code" in credentials:
            return await self._authenticate_with_code(
                credentials["code"],
                credentials.get("redirect_uri", ""),
            )
        elif "token" in credentials:
            valid, user = await self.validate_token(credentials["token"])
            if valid and user:
                return True, user, None
            return False, None, "Invalid token"
        else:
            return False, None, "No code or token provided"

    async def _authenticate_with_code(
        self,
        code: str,
        redirect_uri: str,
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Exchange authorization code for tokens."""
        import aiohttp

        try:
            discovery = await self._get_discovery()
            token_endpoint = discovery.get("token_endpoint")

            if not token_endpoint:
                return False, None, "No token_endpoint in discovery"

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self._client_id,
            }

            if self._client_secret:
                data["client_secret"] = self._client_secret

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_endpoint,
                    data=data,
                    ssl=self._verify_ssl,
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        logger.error("Token exchange failed: %s - %s", response.status, error_body)
                        return False, None, f"Token exchange failed: {response.status}"

                    token_response = await response.json()

            # Validate ID token and extract user info
            id_token = token_response.get("id_token")
            access_token = token_response.get("access_token")

            if id_token:
                claims = await self._decode_jwt(id_token)
                if claims:
                    user = self._claims_to_user(claims)
                    return True, user, None

            # Fallback to userinfo endpoint
            if access_token:
                valid, user = await self.validate_token(access_token)
                if valid and user:
                    return True, user, None

            return False, None, "Failed to extract user info from tokens"

        except Exception as exc:
            logger.error("OIDC auth failed: %s", exc, exc_info=True)
            return False, None, str(exc)

    async def validate_token(self, token: str) -> Tuple[bool, Optional[ExternalUser]]:
        """Validate an access token and get user info."""
        import aiohttp

        try:
            # Try to decode as JWT first
            claims = await self._decode_jwt(token)
            if claims:
                return True, self._claims_to_user(claims)

            # Fallback to userinfo endpoint
            discovery = await self._get_discovery()
            userinfo_endpoint = discovery.get("userinfo_endpoint")

            if not userinfo_endpoint:
                return False, None

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    userinfo_endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                    ssl=self._verify_ssl,
                ) as response:
                    if response.status != 200:
                        return False, None
                    claims = await response.json()
                    return True, self._claims_to_user(claims)

        except Exception as exc:
            logger.warning("Token validation failed: %s", exc)
            return False, None

    async def _decode_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate a JWT token."""
        try:
            import jwt
        except ImportError:
            logger.warning("PyJWT not installed, cannot validate JWTs")
            return None

        try:
            # Get header to find the key ID
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            # Get JWKS and find the key
            jwks = await self._get_jwks()
            key = None

            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key = jwt.algorithms.RSAAlgorithm.from_jwk(k)
                    break

            if not key:
                logger.warning("Key %s not found in JWKS", kid)
                return None

            # Decode and validate
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256", "RS384", "RS512"],
                audience=self._client_id,
                issuer=self._issuer,
            )

            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.warning("Invalid token: %s", exc)
            return None
        except Exception as exc:
            logger.warning("JWT decode error: %s", exc)
            return None

    def _claims_to_user(self, claims: Dict[str, Any]) -> ExternalUser:
        """Convert JWT claims to ExternalUser."""
        external_id = claims.get("sub", "")
        email = claims.get("email", "")
        username = claims.get(self._username_claim) or claims.get("email", "").split("@")[0]
        display_name = claims.get("name") or claims.get("given_name")

        groups = claims.get(self._groups_claim, [])
        if isinstance(groups, str):
            groups = [groups]

        roles = claims.get(self._roles_claim, [])
        if isinstance(roles, str):
            roles = [roles]

        return ExternalUser(
            external_id=external_id,
            email=email,
            username=username,
            display_name=display_name,
            provider_type="oidc",
            provider_name=self.name,
            groups=groups,
            roles=roles,
            raw_attributes=claims,
            email_verified=claims.get("email_verified", False),
        )

    async def get_auth_url(self, redirect_uri: str, state: str) -> str:
        """Get the OIDC authorization URL."""
        discovery = await self._get_discovery()
        auth_endpoint = discovery.get("authorization_endpoint")

        if not auth_endpoint:
            raise ValueError("No authorization_endpoint in discovery")

        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self._scopes),
            "state": state,
        }

        return f"{auth_endpoint}?{urlencode(params)}"

    async def refresh_token(self, refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
        """Refresh tokens using refresh token."""
        import aiohttp

        try:
            discovery = await self._get_discovery()
            token_endpoint = discovery.get("token_endpoint")

            if not token_endpoint:
                return None, None

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self._client_id,
            }

            if self._client_secret:
                data["client_secret"] = self._client_secret

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_endpoint,
                    data=data,
                    ssl=self._verify_ssl,
                ) as response:
                    if response.status != 200:
                        return None, None

                    token_response = await response.json()
                    return (
                        token_response.get("access_token"),
                        token_response.get("refresh_token"),
                    )

        except Exception as exc:
            logger.error("Token refresh failed: %s", exc)
            return None, None

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test connection to OIDC provider."""
        try:
            await self._get_discovery()
            await self._get_jwks()
            return True, None
        except Exception as exc:
            return False, str(exc)
