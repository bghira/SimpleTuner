"""End-to-end LDAP authentication tests using local OpenLDAP container.

Requirements:
    - OpenLDAP running on localhost:389
    - Admin: cn=admin,dc=example,dc=com / SecretPassword123
    - Users: alice, bob, charlie (password: UserPassword123)
"""

import asyncio
import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.base import ProviderConfig
from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.ldap import LDAPProvider


class TestLDAPE2E(unittest.TestCase):
    """End-to-end tests for LDAP authentication against local OpenLDAP."""

    @classmethod
    def setUpClass(cls):
        """Set up LDAP provider with OpenLDAP configuration."""
        cls.config = ProviderConfig(
            name="test-openldap",
            provider_type="ldap",
            enabled=True,
            config={
                "server": "ldap://localhost:389",
                "base_dn": "dc=example,dc=com",
                "bind_dn": "cn=admin,dc=example,dc=com",
                "bind_password": "SecretPassword123",
                "user_search_filter": "(uid={username})",
                "user_search_base": "ou=users,dc=example,dc=com",  # Enterprise: users in ou=users
                "group_search_filter": "(uniqueMember={user_dn})",  # groupOfUniqueNames uses uniqueMember
                "group_search_base": "ou=groups,dc=example,dc=com",
                "email_attribute": "mail",
                "username_attribute": "uid",
                "display_name_attribute": "displayName",  # Use displayName for cleaner names
                "use_ssl": False,
                "start_tls": False,
            },
        )
        cls.provider = LDAPProvider(cls.config)
        cls.user_password = "UserPassword123"

    def test_connection(self):
        """Test LDAP connection to OpenLDAP server."""
        success, error = asyncio.run(self.provider.test_connection())
        self.assertTrue(success, f"Connection failed: {error}")
        self.assertIsNone(error)

    def test_authenticate_alice(self):
        """Test authentication for user alice."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "alice",
                    "password": self.user_password,
                }
            )
        )
        self.assertTrue(success, f"Auth failed: {error}")
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.display_name, "Alice Engineer")
        self.assertEqual(user.provider_type, "ldap")
        self.assertEqual(user.external_id, "cn=alice,ou=users,dc=example,dc=com")
        print(f"\nalice: dn={user.external_id}, display_name={user.display_name}")

    def test_authenticate_bob(self):
        """Test authentication for user bob (operations group)."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "bob",
                    "password": self.user_password,
                }
            )
        )
        self.assertTrue(success, f"Auth failed: {error}")
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "bob")
        self.assertEqual(user.email, "bob@example.com")
        self.assertIn("operations", user.groups)
        print(f"\nbob: email={user.email}, groups={user.groups}")

    def test_authenticate_charlie(self):
        """Test authentication for user charlie (managers group)."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "charlie",
                    "password": self.user_password,
                }
            )
        )
        self.assertTrue(success, f"Auth failed: {error}")
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "charlie")
        self.assertEqual(user.email, "charlie@example.com")
        self.assertIn("managers", user.groups)
        print(f"\ncharlie: email={user.email}, groups={user.groups}")

    def test_invalid_password(self):
        """Test authentication with wrong password."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "alice",
                    "password": "wrongpassword",
                }
            )
        )
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "Invalid credentials")

    def test_nonexistent_user(self):
        """Test authentication with non-existent user."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "nonexistent",
                    "password": "somepassword",
                }
            )
        )
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "User not found")

    def test_empty_credentials(self):
        """Test authentication with empty credentials."""
        success, user, error = asyncio.run(
            self.provider.authenticate(
                {
                    "username": "",
                    "password": "",
                }
            )
        )
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertEqual(error, "Username and password required")


class TestAuthProviderManager(unittest.TestCase):
    """Test the AuthProviderManager with LDAP configuration."""

    def setUp(self):
        """Reset manager singleton before each test."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        # Reset singleton for clean state
        AuthProviderManager._instance = None

    def test_env_config_ldap(self):
        """Test that LDAP provider is loaded from environment variables."""
        import os

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        # Set environment variables
        os.environ["SIMPLETUNER_LDAP_SERVER"] = "ldap://localhost:389"
        os.environ["SIMPLETUNER_LDAP_BASE_DN"] = "dc=example,dc=com"
        os.environ["SIMPLETUNER_LDAP_BIND_DN"] = "cn=admin,dc=example,dc=com"
        os.environ["SIMPLETUNER_LDAP_BIND_PASSWORD"] = "SecretPassword123"
        os.environ["SIMPLETUNER_LDAP_USER_FILTER"] = "(uid={username})"

        try:
            manager = AuthProviderManager()
            manager.configure()

            # Should have loaded ldap-env provider
            providers = manager.get_ldap_providers()
            self.assertEqual(len(providers), 1)
            self.assertEqual(providers[0].name, "ldap-env")

            # Test connection
            success, error = asyncio.run(providers[0].test_connection())
            self.assertTrue(success, f"Connection failed: {error}")

        finally:
            # Clean up env vars
            for key in [
                "SIMPLETUNER_LDAP_SERVER",
                "SIMPLETUNER_LDAP_BASE_DN",
                "SIMPLETUNER_LDAP_BIND_DN",
                "SIMPLETUNER_LDAP_BIND_PASSWORD",
                "SIMPLETUNER_LDAP_USER_FILTER",
            ]:
                os.environ.pop(key, None)

    def test_authenticate_ldap_via_manager(self):
        """Test authentication through the AuthProviderManager."""
        import os

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.providers.manager import AuthProviderManager

        # Configure via environment
        os.environ["SIMPLETUNER_LDAP_SERVER"] = "ldap://localhost:389"
        os.environ["SIMPLETUNER_LDAP_BASE_DN"] = "ou=users,dc=example,dc=com"
        os.environ["SIMPLETUNER_LDAP_BIND_DN"] = "cn=admin,dc=example,dc=com"
        os.environ["SIMPLETUNER_LDAP_BIND_PASSWORD"] = "SecretPassword123"

        try:
            manager = AuthProviderManager()
            manager.configure()

            # Authenticate bob
            success, user, error = asyncio.run(manager.authenticate_ldap("bob", "UserPassword123"))
            self.assertTrue(success, f"Auth failed: {error}")
            self.assertIsNotNone(user)
            self.assertEqual(user.username, "bob")

        finally:
            for key in [
                "SIMPLETUNER_LDAP_SERVER",
                "SIMPLETUNER_LDAP_BASE_DN",
                "SIMPLETUNER_LDAP_BIND_DN",
                "SIMPLETUNER_LDAP_BIND_PASSWORD",
            ]:
                os.environ.pop(key, None)


if __name__ == "__main__":
    # Check if ldap3 is installed
    try:
        import ldap3
    except ImportError:
        print("ldap3 package not installed. Install with: pip install ldap3")
        sys.exit(1)

    unittest.main(verbosity=2)
