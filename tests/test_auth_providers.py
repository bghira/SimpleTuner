"""Tests for OIDC and LDAP authentication providers.

Tests cover:
- ExternalUser model and level mapping
- ProviderConfig model
- OIDCProvider JWT validation and claims parsing
- LDAPProvider user search and authentication
- AuthProviderManager registration and routing
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Check if ldap3 is available
try:
    import ldap3

    _LDAP3_AVAILABLE = True
except ImportError:
    _LDAP3_AVAILABLE = False

# Suppress expected error logs during tests
logging.getLogger("simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc").setLevel(logging.CRITICAL)
logging.getLogger("simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap").setLevel(logging.CRITICAL)
logging.getLogger("simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager").setLevel(logging.CRITICAL)

from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.base import (
    AuthProviderBase,
    ExternalUser,
    ProviderConfig,
)


class TestExternalUser(unittest.TestCase):
    """Tests for ExternalUser dataclass."""

    def test_create_external_user(self):
        """Test creating an external user."""
        user = ExternalUser(
            external_id="user-123",
            email="test@example.com",
            username="testuser",
            display_name="Test User",
            provider_type="oidc",
            provider_name="keycloak",
        )

        self.assertEqual(user.external_id, "user-123")
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.display_name, "Test User")
        self.assertEqual(user.provider_type, "oidc")
        self.assertEqual(user.provider_name, "keycloak")

    def test_external_user_defaults(self):
        """Test external user default values."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
        )

        self.assertIsNone(user.display_name)
        self.assertEqual(user.provider_type, "")
        self.assertEqual(user.provider_name, "")
        self.assertEqual(user.groups, [])
        self.assertEqual(user.roles, [])
        self.assertEqual(user.raw_attributes, {})
        self.assertFalse(user.email_verified)

    def test_external_user_with_groups_and_roles(self):
        """Test external user with groups and roles."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["developers", "ml-team"],
            roles=["admin", "viewer"],
        )

        self.assertEqual(len(user.groups), 2)
        self.assertIn("developers", user.groups)
        self.assertIn("ml-team", user.groups)
        self.assertEqual(len(user.roles), 2)
        self.assertIn("admin", user.roles)

    def test_external_user_email_verified(self):
        """Test external user email verified flag."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            email_verified=True,
        )

        self.assertTrue(user.email_verified)


class TestExternalUserLevelMapping(unittest.TestCase):
    """Tests for ExternalUser.get_suggested_levels."""

    def test_no_mapping_returns_default(self):
        """Test empty mapping returns default researcher level."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["any-group"],
        )

        levels = user.get_suggested_levels({})
        self.assertEqual(levels, ["researcher"])

    def test_group_matches_single_level(self):
        """Test group matching returns correct level."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["admins"],
        )

        level_mapping = {
            "admin": ["admins", "administrators"],
            "researcher": ["users"],
        }

        levels = user.get_suggested_levels(level_mapping)
        self.assertEqual(levels, ["admin"])

    def test_role_matches_level(self):
        """Test role matching returns correct level."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            roles=["power-user"],
        )

        level_mapping = {
            "lead": ["power-user", "team-lead"],
        }

        levels = user.get_suggested_levels(level_mapping)
        self.assertEqual(levels, ["lead"])

    def test_multiple_matches_returns_all(self):
        """Test multiple matching groups returns all levels."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["admins", "developers"],
        )

        level_mapping = {
            "admin": ["admins"],
            "lead": ["developers"],
        }

        levels = user.get_suggested_levels(level_mapping)
        self.assertIn("admin", levels)
        self.assertIn("lead", levels)

    def test_no_match_returns_default(self):
        """Test no matching groups returns default."""
        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["random-group"],
        )

        level_mapping = {
            "admin": ["admins"],
            "lead": ["developers"],
        }

        levels = user.get_suggested_levels(level_mapping)
        self.assertEqual(levels, ["researcher"])


class TestProviderConfig(unittest.TestCase):
    """Tests for ProviderConfig dataclass."""

    def test_create_oidc_config(self):
        """Test creating OIDC provider config."""
        config = ProviderConfig(
            name="keycloak",
            provider_type="oidc",
            config={
                "issuer": "https://keycloak.example.com/realms/test",
                "client_id": "my-app",
                "client_secret": "secret123",
            },
        )

        self.assertEqual(config.name, "keycloak")
        self.assertEqual(config.provider_type, "oidc")
        self.assertTrue(config.enabled)
        self.assertEqual(config.config["issuer"], "https://keycloak.example.com/realms/test")

    def test_create_ldap_config(self):
        """Test creating LDAP provider config."""
        config = ProviderConfig(
            name="corporate-ldap",
            provider_type="ldap",
            config={
                "server": "ldap://ldap.corp.com:389",
                "base_dn": "dc=corp,dc=com",
                "bind_dn": "cn=admin,dc=corp,dc=com",
            },
        )

        self.assertEqual(config.name, "corporate-ldap")
        self.assertEqual(config.provider_type, "ldap")
        self.assertEqual(config.config["server"], "ldap://ldap.corp.com:389")

    def test_config_defaults(self):
        """Test provider config default values."""
        config = ProviderConfig(
            name="test",
            provider_type="oidc",
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.level_mapping, {})
        self.assertTrue(config.auto_create_users)
        self.assertEqual(config.default_levels, ["researcher"])
        self.assertEqual(config.config, {})

    def test_config_with_level_mapping(self):
        """Test provider config with level mapping."""
        config = ProviderConfig(
            name="keycloak",
            provider_type="oidc",
            level_mapping={
                "admin": ["keycloak-admins"],
                "lead": ["team-leads"],
            },
        )

        self.assertEqual(len(config.level_mapping), 2)
        self.assertIn("admin", config.level_mapping)

    def test_config_disabled(self):
        """Test disabled provider config."""
        config = ProviderConfig(
            name="disabled-provider",
            provider_type="ldap",
            enabled=False,
        )

        self.assertFalse(config.enabled)


class TestOIDCProvider(unittest.TestCase):
    """Tests for OIDCProvider."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        self.config = ProviderConfig(
            name="test-oidc",
            provider_type="oidc",
            config={
                "issuer": "https://auth.example.com",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        self.provider = OIDCProvider(self.config)

    def test_provider_initialization(self):
        """Test OIDC provider initialization."""
        self.assertEqual(self.provider.name, "test-oidc")
        self.assertEqual(self.provider._issuer, "https://auth.example.com")
        self.assertEqual(self.provider._client_id, "test-client")
        self.assertEqual(self.provider._client_secret, "test-secret")

    def test_default_scopes(self):
        """Test default OIDC scopes."""
        self.assertEqual(self.provider._scopes, ["openid", "email", "profile"])

    def test_custom_scopes(self):
        """Test custom OIDC scopes."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="custom",
            provider_type="oidc",
            config={
                "issuer": "https://auth.example.com",
                "client_id": "client",
                "scopes": ["openid", "email", "groups"],
            },
        )
        provider = OIDCProvider(config)
        self.assertIn("groups", provider._scopes)

    def test_claims_to_user(self):
        """Test converting JWT claims to ExternalUser."""
        claims = {
            "sub": "user-12345",
            "email": "user@example.com",
            "preferred_username": "jdoe",
            "name": "John Doe",
            "email_verified": True,
            "groups": ["developers", "ml-team"],
        }

        user = self.provider._claims_to_user(claims)

        self.assertEqual(user.external_id, "user-12345")
        self.assertEqual(user.email, "user@example.com")
        self.assertEqual(user.username, "jdoe")
        self.assertEqual(user.display_name, "John Doe")
        self.assertTrue(user.email_verified)
        self.assertEqual(len(user.groups), 2)
        self.assertIn("developers", user.groups)

    def test_claims_to_user_string_groups(self):
        """Test claims with string groups (not list)."""
        claims = {
            "sub": "user-1",
            "email": "user@test.com",
            "groups": "single-group",
        }

        user = self.provider._claims_to_user(claims)
        self.assertEqual(user.groups, ["single-group"])

    def test_claims_to_user_fallback_username(self):
        """Test username fallback to email prefix."""
        claims = {
            "sub": "user-1",
            "email": "john.doe@example.com",
        }

        user = self.provider._claims_to_user(claims)
        self.assertEqual(user.username, "john.doe")

    def test_claims_to_user_custom_username_claim(self):
        """Test custom username claim."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="custom",
            provider_type="oidc",
            config={
                "issuer": "https://auth.example.com",
                "client_id": "client",
                "username_claim": "upn",
            },
        )
        provider = OIDCProvider(config)

        claims = {
            "sub": "user-1",
            "email": "user@test.com",
            "upn": "custom_username",
        }

        user = provider._claims_to_user(claims)
        self.assertEqual(user.username, "custom_username")


class TestOIDCProviderAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for OIDCProvider."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        self.config = ProviderConfig(
            name="test-oidc",
            provider_type="oidc",
            config={
                "issuer": "https://auth.example.com",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        self.provider = OIDCProvider(self.config)

    async def test_authenticate_missing_credentials(self):
        """Test authentication with missing credentials."""
        success, user, error = await self.provider.authenticate({})
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "No code or token provided")

    @patch("aiohttp.ClientSession")
    async def test_get_discovery(self, mock_session_class):
        """Test fetching OIDC discovery document."""
        discovery_doc = {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
            "userinfo_endpoint": "https://auth.example.com/userinfo",
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=discovery_doc)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        # Clear cached discovery
        self.provider._discovery = None

        result = await self.provider._get_discovery()

        self.assertEqual(result["issuer"], "https://auth.example.com")
        self.assertEqual(result["token_endpoint"], "https://auth.example.com/token")

    @patch("aiohttp.ClientSession")
    async def test_get_discovery_failure(self, mock_session_class):
        """Test discovery failure."""
        mock_response = MagicMock()
        mock_response.status = 404

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        # Clear cached discovery
        self.provider._discovery = None

        with self.assertRaises(ValueError) as ctx:
            await self.provider._get_discovery()

        self.assertIn("Discovery failed: 404", str(ctx.exception))

    async def test_get_auth_url(self):
        """Test generating authorization URL."""
        self.provider._discovery = {
            "authorization_endpoint": "https://auth.example.com/authorize",
        }

        url = await self.provider.get_auth_url(
            redirect_uri="https://app.example.com/callback",
            state="random-state-123",
        )

        self.assertIn("https://auth.example.com/authorize", url)
        self.assertIn("client_id=test-client", url)
        self.assertIn("redirect_uri=https", url)
        self.assertIn("state=random-state-123", url)
        self.assertIn("response_type=code", url)

    async def test_get_auth_url_no_endpoint(self):
        """Test auth URL when no endpoint in discovery."""
        self.provider._discovery = {}

        with self.assertRaises(ValueError) as ctx:
            await self.provider.get_auth_url("http://test", "state")

        self.assertIn("No authorization_endpoint", str(ctx.exception))


class TestLDAPProvider(unittest.TestCase):
    """Tests for LDAPProvider."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        self.config = ProviderConfig(
            name="test-ldap",
            provider_type="ldap",
            config={
                "server": "ldap://ldap.example.com:389",
                "base_dn": "dc=example,dc=com",
                "bind_dn": "cn=admin,dc=example,dc=com",
                "bind_password": "admin-password",
            },
        )
        self.provider = LDAPProvider(self.config)

    def test_provider_initialization(self):
        """Test LDAP provider initialization."""
        self.assertEqual(self.provider.name, "test-ldap")
        self.assertEqual(self.provider._server, "ldap://ldap.example.com:389")
        self.assertEqual(self.provider._base_dn, "dc=example,dc=com")
        self.assertEqual(self.provider._bind_dn, "cn=admin,dc=example,dc=com")

    def test_default_search_filter(self):
        """Test default user search filter."""
        self.assertEqual(self.provider._user_search_filter, "(uid={username})")

    def test_custom_search_filter(self):
        """Test custom user search filter."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        config = ProviderConfig(
            name="ad",
            provider_type="ldap",
            config={
                "server": "ldap://ad.corp.com",
                "base_dn": "dc=corp,dc=com",
                "user_search_filter": "(sAMAccountName={username})",
            },
        )
        provider = LDAPProvider(config)
        self.assertEqual(provider._user_search_filter, "(sAMAccountName={username})")

    def test_default_attributes(self):
        """Test default attribute mappings."""
        self.assertEqual(self.provider._email_attr, "mail")
        self.assertEqual(self.provider._username_attr, "uid")
        self.assertEqual(self.provider._display_name_attr, "cn")
        self.assertEqual(self.provider._group_name_attr, "cn")

    def test_custom_attributes(self):
        """Test custom attribute mappings."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        config = ProviderConfig(
            name="ad",
            provider_type="ldap",
            config={
                "server": "ldap://ad.corp.com",
                "base_dn": "dc=corp,dc=com",
                "email_attribute": "userPrincipalName",
                "username_attribute": "sAMAccountName",
                "display_name_attribute": "displayName",
            },
        )
        provider = LDAPProvider(config)

        self.assertEqual(provider._email_attr, "userPrincipalName")
        self.assertEqual(provider._username_attr, "sAMAccountName")
        self.assertEqual(provider._display_name_attr, "displayName")

    def test_attrs_to_user(self):
        """Test converting LDAP attributes to ExternalUser."""
        user_dn = "uid=jdoe,ou=users,dc=example,dc=com"
        attrs = {
            "mail": "john.doe@example.com",
            "uid": "jdoe",
            "cn": "John Doe",
        }
        groups = ["developers", "admins"]

        user = self.provider._attrs_to_user(user_dn, attrs, groups)

        self.assertEqual(user.external_id, user_dn)
        self.assertEqual(user.email, "john.doe@example.com")
        self.assertEqual(user.username, "jdoe")
        self.assertEqual(user.display_name, "John Doe")
        self.assertEqual(user.provider_type, "ldap")
        self.assertEqual(len(user.groups), 2)

    def test_attrs_to_user_fallback_username(self):
        """Test username fallback to email prefix."""
        user_dn = "uid=jdoe,ou=users,dc=example,dc=com"
        attrs = {"mail": "john.doe@example.com"}

        user = self.provider._attrs_to_user(user_dn, attrs, [])

        self.assertEqual(user.username, "john.doe")


class TestLDAPProviderAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for LDAPProvider."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        self.config = ProviderConfig(
            name="test-ldap",
            provider_type="ldap",
            config={
                "server": "ldap://ldap.example.com:389",
                "base_dn": "dc=example,dc=com",
                "bind_dn": "cn=admin,dc=example,dc=com",
                "bind_password": "admin-password",
            },
        )
        self.provider = LDAPProvider(self.config)

    async def test_authenticate_missing_username(self):
        """Test authentication with missing username."""
        success, user, error = await self.provider.authenticate({"password": "pass"})
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "Username and password required")

    async def test_authenticate_missing_password(self):
        """Test authentication with missing password."""
        success, user, error = await self.provider.authenticate({"username": "user"})
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "Username and password required")

    async def test_validate_token_returns_false(self):
        """Test that LDAP validate_token always returns False."""
        valid, user = await self.provider.validate_token("any-token")
        self.assertFalse(valid)
        self.assertIsNone(user)

    async def test_get_auth_url_not_implemented(self):
        """Test that LDAP get_auth_url raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            await self.provider.get_auth_url("http://test", "state")

    async def test_refresh_token_returns_none(self):
        """Test that LDAP refresh_token returns None."""
        access, refresh = await self.provider.refresh_token("any-token")
        self.assertIsNone(access)
        self.assertIsNone(refresh)


class TestAuthProviderManager(unittest.TestCase):
    """Tests for AuthProviderManager."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        # Reset singleton for testing
        AuthProviderManager._instance = None
        self.manager = AuthProviderManager()

    def tearDown(self):
        """Clean up after tests."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        AuthProviderManager._instance = None

    def test_singleton_pattern(self):
        """Test manager is a singleton."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        manager2 = AuthProviderManager()
        self.assertIs(self.manager, manager2)

    def test_add_oidc_provider(self):
        """Test adding an OIDC provider."""
        config = ProviderConfig(
            name="keycloak",
            provider_type="oidc",
            config={
                "issuer": "https://keycloak.example.com",
                "client_id": "my-app",
            },
        )

        result = self.manager.add_provider(config)
        self.assertTrue(result)

        provider = self.manager.get_provider("keycloak")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.name, "keycloak")

    def test_add_ldap_provider(self):
        """Test adding an LDAP provider."""
        config = ProviderConfig(
            name="corp-ldap",
            provider_type="ldap",
            config={
                "server": "ldap://ldap.corp.com",
                "base_dn": "dc=corp,dc=com",
            },
        )

        result = self.manager.add_provider(config)
        self.assertTrue(result)

        provider = self.manager.get_provider("corp-ldap")
        self.assertIsNotNone(provider)

    def test_add_duplicate_provider_fails(self):
        """Test adding duplicate provider fails."""
        config = ProviderConfig(
            name="test",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )

        self.manager.add_provider(config)
        result = self.manager.add_provider(config)
        self.assertFalse(result)

    def test_remove_provider(self):
        """Test removing a provider."""
        config = ProviderConfig(
            name="to-remove",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )

        self.manager.add_provider(config)
        self.assertIsNotNone(self.manager.get_provider("to-remove"))

        result = self.manager.remove_provider("to-remove")
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_provider("to-remove"))

    def test_remove_nonexistent_provider(self):
        """Test removing nonexistent provider fails."""
        result = self.manager.remove_provider("nonexistent")
        self.assertFalse(result)

    def test_get_enabled_providers(self):
        """Test getting enabled providers."""
        config1 = ProviderConfig(
            name="enabled",
            provider_type="oidc",
            enabled=True,
            config={"issuer": "https://test.com", "client_id": "id"},
        )
        config2 = ProviderConfig(
            name="disabled",
            provider_type="oidc",
            enabled=False,
            config={"issuer": "https://test2.com", "client_id": "id"},
        )

        self.manager.add_provider(config1)
        self.manager.add_provider(config2)

        enabled = self.manager.get_enabled_providers()
        self.assertEqual(len(enabled), 1)
        self.assertEqual(enabled[0].name, "enabled")

    def test_get_oidc_providers(self):
        """Test getting OIDC providers only."""
        oidc_config = ProviderConfig(
            name="oidc",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )
        ldap_config = ProviderConfig(
            name="ldap",
            provider_type="ldap",
            config={"server": "ldap://test.com", "base_dn": "dc=test"},
        )

        self.manager.add_provider(oidc_config)
        self.manager.add_provider(ldap_config)

        oidc_providers = self.manager.get_oidc_providers()
        self.assertEqual(len(oidc_providers), 1)
        self.assertEqual(oidc_providers[0].name, "oidc")

    def test_get_ldap_providers(self):
        """Test getting LDAP providers only."""
        oidc_config = ProviderConfig(
            name="oidc",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )
        ldap_config = ProviderConfig(
            name="ldap",
            provider_type="ldap",
            config={"server": "ldap://test.com", "base_dn": "dc=test"},
        )

        self.manager.add_provider(oidc_config)
        self.manager.add_provider(ldap_config)

        ldap_providers = self.manager.get_ldap_providers()
        self.assertEqual(len(ldap_providers), 1)
        self.assertEqual(ldap_providers[0].name, "ldap")

    def test_list_providers(self):
        """Test listing all providers."""
        config1 = ProviderConfig(
            name="provider1",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )
        config2 = ProviderConfig(
            name="provider2",
            provider_type="ldap",
            config={"server": "ldap://test.com", "base_dn": "dc=test"},
        )

        self.manager.add_provider(config1)
        self.manager.add_provider(config2)

        providers = self.manager.list_providers()
        self.assertEqual(len(providers), 2)

        names = {p["name"] for p in providers}
        self.assertIn("provider1", names)
        self.assertIn("provider2", names)

    def test_update_provider(self):
        """Test updating provider configuration."""
        config = ProviderConfig(
            name="updateable",
            provider_type="oidc",
            enabled=True,
            config={"issuer": "https://test.com", "client_id": "id"},
        )

        self.manager.add_provider(config)

        result = self.manager.update_provider("updateable", {"enabled": False})
        self.assertTrue(result)

        # Provider should now be disabled
        enabled = self.manager.get_enabled_providers()
        names = [p.name for p in enabled]
        self.assertNotIn("updateable", names)

    def test_update_nonexistent_provider(self):
        """Test updating nonexistent provider fails."""
        result = self.manager.update_provider("nonexistent", {"enabled": False})
        self.assertFalse(result)


class TestAuthProviderManagerAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for AuthProviderManager."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        AuthProviderManager._instance = None
        self.manager = AuthProviderManager()

    def tearDown(self):
        """Clean up after tests."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        AuthProviderManager._instance = None

    async def test_authenticate_unknown_provider(self):
        """Test authentication with unknown provider."""
        success, user, error = await self.manager.authenticate(
            "unknown",
            {"username": "test", "password": "pass"},
        )

        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertIn("Unknown provider", error)

    async def test_authenticate_disabled_provider(self):
        """Test authentication with disabled provider."""
        config = ProviderConfig(
            name="disabled",
            provider_type="ldap",
            enabled=False,
            config={"server": "ldap://test.com", "base_dn": "dc=test"},
        )

        self.manager.add_provider(config)

        success, user, error = await self.manager.authenticate(
            "disabled",
            {"username": "test", "password": "pass"},
        )

        self.assertFalse(success)
        self.assertIn("Provider disabled", error)

    async def test_authenticate_ldap_no_providers(self):
        """Test LDAP auth when no providers configured."""
        success, user, error = await self.manager.authenticate_ldap(
            "testuser",
            "password",
        )

        self.assertFalse(success)
        self.assertIn("No LDAP providers configured", error)


class TestAuthProviderManagerConfig(unittest.TestCase):
    """Tests for AuthProviderManager configuration loading."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        AuthProviderManager._instance = None
        self.manager = AuthProviderManager()

    def tearDown(self):
        """Clean up after tests."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        AuthProviderManager._instance = None

    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "auth_providers": [
                {
                    "name": "keycloak",
                    "type": "oidc",
                    "enabled": True,
                    "auto_create_users": True,
                    "config": {
                        "issuer": "https://keycloak.example.com/realms/test",
                        "client_id": "my-app",
                        "client_secret": "secret",
                    },
                },
                {
                    "name": "corp-ldap",
                    "type": "ldap",
                    "enabled": True,
                    "config": {
                        "server": "ldap://ldap.corp.com",
                        "base_dn": "dc=corp,dc=com",
                    },
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            self.manager.configure(config_path)

            providers = self.manager.list_providers()
            self.assertEqual(len(providers), 2)

            keycloak = self.manager.get_provider("keycloak")
            self.assertIsNotNone(keycloak)

            ldap = self.manager.get_provider("corp-ldap")
            self.assertIsNotNone(ldap)

        finally:
            config_path.unlink()

    def test_load_config_with_level_mapping(self):
        """Test loading configuration with level mapping."""
        config_data = {
            "auth_providers": [
                {
                    "name": "keycloak",
                    "type": "oidc",
                    "level_mapping": {
                        "admin": ["kc-admins"],
                        "lead": ["kc-leads"],
                        "researcher": ["kc-users"],
                    },
                    "default_levels": ["researcher"],
                    "config": {
                        "issuer": "https://keycloak.example.com",
                        "client_id": "app",
                    },
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            self.manager.configure(config_path)

            provider = self.manager.get_provider("keycloak")
            self.assertIsNotNone(provider)

            # Test level mapping
            user = ExternalUser(
                external_id="user-1",
                email="user@test.com",
                username="user",
                groups=["kc-admins"],
                provider_type="oidc",
                provider_name="keycloak",
            )

            levels = provider.get_suggested_levels(user)
            self.assertIn("admin", levels)

        finally:
            config_path.unlink()

    @patch.dict(
        "os.environ",
        {
            "SIMPLETUNER_OIDC_ISSUER": "https://env-oidc.example.com",
            "SIMPLETUNER_OIDC_CLIENT_ID": "env-client",
            "SIMPLETUNER_OIDC_CLIENT_SECRET": "env-secret",
        },
    )
    def test_load_config_from_env_oidc(self):
        """Test loading OIDC config from environment."""
        self.manager.configure()

        provider = self.manager.get_provider("oidc-env")
        self.assertIsNotNone(provider)
        self.assertEqual(provider._issuer, "https://env-oidc.example.com")

    @patch.dict(
        "os.environ",
        {
            "SIMPLETUNER_LDAP_SERVER": "ldap://env-ldap.example.com",
            "SIMPLETUNER_LDAP_BASE_DN": "dc=env,dc=com",
            "SIMPLETUNER_LDAP_BIND_DN": "cn=admin,dc=env,dc=com",
        },
    )
    def test_load_config_from_env_ldap(self):
        """Test loading LDAP config from environment."""
        self.manager.configure()

        provider = self.manager.get_provider("ldap-env")
        self.assertIsNotNone(provider)
        self.assertEqual(provider._server, "ldap://env-ldap.example.com")


class TestAuthProviderBase(unittest.TestCase):
    """Tests for AuthProviderBase abstract class."""

    def test_get_suggested_levels_with_mapping(self):
        """Test get_suggested_levels uses mapping."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="test",
            provider_type="oidc",
            level_mapping={
                "admin": ["admins-group"],
                "lead": ["leads-group"],
            },
            default_levels=["researcher"],
            config={"issuer": "https://test.com", "client_id": "id"},
        )

        provider = OIDCProvider(config)

        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
            groups=["admins-group"],
        )

        levels = provider.get_suggested_levels(user)
        self.assertIn("admin", levels)

    def test_get_suggested_levels_default(self):
        """Test get_suggested_levels returns defaults when no mapping."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="test",
            provider_type="oidc",
            level_mapping={},
            default_levels=["custom-default"],
            config={"issuer": "https://test.com", "client_id": "id"},
        )

        provider = OIDCProvider(config)

        user = ExternalUser(
            external_id="user-1",
            email="user@test.com",
            username="user",
        )

        levels = provider.get_suggested_levels(user)
        self.assertEqual(levels, ["custom-default"])


class TestProviderTypes(unittest.TestCase):
    """Tests for provider type registry."""

    def test_provider_types_registered(self):
        """Test both provider types are registered."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import PROVIDER_TYPES
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        self.assertIn("oidc", PROVIDER_TYPES)
        self.assertIn("ldap", PROVIDER_TYPES)
        self.assertEqual(PROVIDER_TYPES["oidc"], OIDCProvider)
        self.assertEqual(PROVIDER_TYPES["ldap"], LDAPProvider)


class TestOIDCProviderJWT(unittest.IsolatedAsyncioTestCase):
    """Tests for OIDC JWT handling."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        self.config = ProviderConfig(
            name="test-oidc",
            provider_type="oidc",
            config={
                "issuer": "https://auth.example.com",
                "client_id": "test-client",
                "client_secret": "test-secret",
            },
        )
        self.provider = OIDCProvider(self.config)

    @patch("aiohttp.ClientSession")
    async def test_get_jwks(self, mock_session_class):
        """Test fetching JWKS."""
        self.provider._discovery = {
            "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
        }

        jwks_data = {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "key-1",
                    "use": "sig",
                    "n": "test-n",
                    "e": "AQAB",
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=jwks_data)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        # Clear cached JWKS
        self.provider._jwks = None
        self.provider._jwks_updated = 0

        result = await self.provider._get_jwks()

        self.assertIn("keys", result)
        self.assertEqual(len(result["keys"]), 1)
        self.assertEqual(result["keys"][0]["kid"], "key-1")

    async def test_get_jwks_no_uri(self):
        """Test JWKS fetch when no URI in discovery."""
        self.provider._discovery = {}
        self.provider._jwks = None

        with self.assertRaises(ValueError) as ctx:
            await self.provider._get_jwks()

        self.assertIn("No jwks_uri", str(ctx.exception))

    async def test_decode_jwt_no_pyjwt(self):
        """Test JWT decode when PyJWT not available."""
        with patch.dict("sys.modules", {"jwt": None}):
            # Force re-evaluation of import
            import importlib

            result = await self.provider._decode_jwt("invalid-token")
            # Should return None without crashing when jwt import fails
            self.assertIsNone(result)


@unittest.skipUnless(_LDAP3_AVAILABLE, "ldap3 package not installed")
class TestLDAPProviderSearch(unittest.IsolatedAsyncioTestCase):
    """Tests for LDAP search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        self.config = ProviderConfig(
            name="test-ldap",
            provider_type="ldap",
            config={
                "server": "ldap://ldap.example.com:389",
                "base_dn": "dc=example,dc=com",
                "bind_dn": "cn=admin,dc=example,dc=com",
                "bind_password": "admin-password",
            },
        )
        self.provider = LDAPProvider(self.config)

    @patch("simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap.LDAPProvider._find_user")
    @patch("simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap.LDAPProvider._get_user_groups")
    async def test_authenticate_user_not_found(self, mock_groups, mock_find):
        """Test authentication when user not found."""
        mock_find.return_value = (None, {})

        success, user, error = await self.provider.authenticate(
            {
                "username": "nonexistent",
                "password": "password",
            }
        )

        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "User not found")


class TestSSLConfiguration(unittest.TestCase):
    """Tests for SSL/TLS configuration."""

    def test_oidc_verify_ssl_default(self):
        """Test OIDC verify_ssl defaults to True."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="test",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id"},
        )
        provider = OIDCProvider(config)
        self.assertTrue(provider._verify_ssl)

    def test_oidc_verify_ssl_disabled(self):
        """Test OIDC verify_ssl can be disabled."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.oidc import OIDCProvider

        config = ProviderConfig(
            name="test",
            provider_type="oidc",
            config={"issuer": "https://test.com", "client_id": "id", "verify_ssl": False},
        )
        provider = OIDCProvider(config)
        self.assertFalse(provider._verify_ssl)

    def test_ldap_ssl_options(self):
        """Test LDAP SSL options."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider

        config = ProviderConfig(
            name="test",
            provider_type="ldap",
            config={
                "server": "ldaps://ldap.example.com:636",
                "base_dn": "dc=test",
                "use_ssl": True,
                "start_tls": False,
                "verify_ssl": False,
            },
        )
        provider = LDAPProvider(config)

        self.assertTrue(provider._use_ssl)
        self.assertFalse(provider._start_tls)
        self.assertFalse(provider._verify_ssl)


if __name__ == "__main__":
    unittest.main()
