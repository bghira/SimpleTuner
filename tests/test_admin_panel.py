"""Tests for admin panel functionality.

Tests cover:
- User model and permission system
- Level/role management
- Resource-based access control (RBAC)
- Organization and team models
- API key management
- Permission hierarchy and wildcards
"""

from __future__ import annotations

import unittest
from dataclasses import field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import (
    DEFAULT_LEVELS,
    DEFAULT_PERMISSIONS,
    APIKey,
    AuthProvider,
    Organization,
    Permission,
    ResourceRule,
    ResourceType,
    RuleAction,
    Team,
    User,
    UserLevel,
)


class TestPermission(unittest.TestCase):
    """Tests for Permission model."""

    def test_create_permission(self):
        """Test creating a permission."""
        perm = Permission(
            id=1,
            name="job.submit",
            description="Submit training jobs",
            category="jobs",
        )

        self.assertEqual(perm.id, 1)
        self.assertEqual(perm.name, "job.submit")
        self.assertEqual(perm.category, "jobs")

    def test_permission_equality(self):
        """Test permission equality by name."""
        perm1 = Permission(id=1, name="job.submit", description="A", category="a")
        perm2 = Permission(id=2, name="job.submit", description="B", category="b")
        perm3 = Permission(id=3, name="job.cancel", description="C", category="c")

        self.assertEqual(perm1, perm2)  # Same name
        self.assertNotEqual(perm1, perm3)  # Different name

    def test_permission_string_equality(self):
        """Test permission equality with string."""
        perm = Permission(id=1, name="job.submit", description="", category="")

        self.assertEqual(perm, "job.submit")
        self.assertNotEqual(perm, "job.cancel")

    def test_permission_hash(self):
        """Test permission hashing for sets."""
        perm1 = Permission(id=1, name="job.submit", description="", category="")
        perm2 = Permission(id=2, name="job.submit", description="", category="")

        perms = {perm1, perm2}
        self.assertEqual(len(perms), 1)  # Same name = same hash

    def test_default_permissions_exist(self):
        """Test default permissions are defined."""
        self.assertGreater(len(DEFAULT_PERMISSIONS), 0)

        # Check key permissions exist
        perm_names = {p.name for p in DEFAULT_PERMISSIONS}
        self.assertIn("job.submit", perm_names)
        self.assertIn("admin.users", perm_names)
        self.assertIn("api.access", perm_names)


class TestUserLevel(unittest.TestCase):
    """Tests for UserLevel model."""

    def test_create_level(self):
        """Test creating a user level."""
        level = UserLevel(
            id=1,
            name="researcher",
            description="Can submit jobs",
            priority=10,
            is_system=False,
            permissions={"job.submit", "job.view.own"},
        )

        self.assertEqual(level.name, "researcher")
        self.assertEqual(level.priority, 10)
        self.assertIn("job.submit", level.permissions)

    def test_level_equality(self):
        """Test level equality by name."""
        level1 = UserLevel(id=1, name="admin", description="A", priority=100)
        level2 = UserLevel(id=2, name="admin", description="B", priority=50)
        level3 = UserLevel(id=3, name="viewer", description="C", priority=0)

        self.assertEqual(level1, level2)  # Same name
        self.assertNotEqual(level1, level3)

    def test_level_string_equality(self):
        """Test level equality with string."""
        level = UserLevel(id=1, name="admin", description="", priority=0)

        self.assertEqual(level, "admin")
        self.assertNotEqual(level, "viewer")

    def test_default_levels_exist(self):
        """Test default levels are defined."""
        self.assertGreater(len(DEFAULT_LEVELS), 0)

        level_names = {l.name for l in DEFAULT_LEVELS}
        self.assertIn("viewer", level_names)
        self.assertIn("researcher", level_names)
        self.assertIn("admin", level_names)

    def test_admin_level_has_wildcard(self):
        """Test admin level has wildcard permission."""
        admin_level = next(l for l in DEFAULT_LEVELS if l.name == "admin")
        self.assertIn("*", admin_level.permissions)


class TestUser(unittest.TestCase):
    """Tests for User model."""

    def _create_user(self, **kwargs) -> User:
        """Create a test user."""
        defaults = {
            "id": 1,
            "email": "test@example.com",
            "username": "testuser",
            "is_active": True,
            "is_admin": False,
        }
        defaults.update(kwargs)
        return User(**defaults)

    def _create_level(self, name: str, permissions: Set[str]) -> UserLevel:
        """Create a test level."""
        return UserLevel(
            id=1,
            name=name,
            description=f"{name} level",
            priority=10,
            permissions=permissions,
        )

    def test_create_user(self):
        """Test creating a user."""
        user = self._create_user()

        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.username, "testuser")
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_admin)

    def test_admin_has_all_permissions(self):
        """Test admin user has all permissions."""
        user = self._create_user(is_admin=True)

        self.assertEqual(user.effective_permissions, {"*"})
        self.assertTrue(user.has_permission("anything"))
        self.assertTrue(user.has_permission("deeply.nested.permission"))

    def test_permissions_from_levels(self):
        """Test permissions are inherited from levels."""
        level = self._create_level("researcher", {"job.submit", "job.view.own"})
        user = self._create_user(levels=[level])

        perms = user.effective_permissions

        self.assertIn("job.submit", perms)
        self.assertIn("job.view.own", perms)
        self.assertNotIn("admin.users", perms)

    def test_multiple_levels_combined(self):
        """Test permissions from multiple levels are combined."""
        level1 = self._create_level("basic", {"job.view.own"})
        level2 = self._create_level("submitter", {"job.submit"})
        user = self._create_user(levels=[level1, level2])

        perms = user.effective_permissions

        self.assertIn("job.view.own", perms)
        self.assertIn("job.submit", perms)

    def test_permission_overrides(self):
        """Test per-user permission overrides."""
        level = self._create_level("basic", {"job.view.own", "job.submit"})
        user = self._create_user(
            levels=[level],
            permission_overrides={
                "job.submit": False,  # Revoke from level
                "job.cancel.own": True,  # Grant extra
            },
        )

        perms = user.effective_permissions

        self.assertIn("job.view.own", perms)  # From level
        self.assertNotIn("job.submit", perms)  # Revoked
        self.assertIn("job.cancel.own", perms)  # Added

    def test_has_permission_direct(self):
        """Test has_permission with direct match."""
        level = self._create_level("basic", {"job.submit"})
        user = self._create_user(levels=[level])

        self.assertTrue(user.has_permission("job.submit"))
        self.assertFalse(user.has_permission("job.cancel"))

    def test_has_permission_wildcard(self):
        """Test has_permission with wildcard matching."""
        level = self._create_level("jobs_manager", {"job.*"})
        user = self._create_user(levels=[level])

        self.assertTrue(user.has_permission("job.submit"))
        self.assertTrue(user.has_permission("job.cancel.all"))
        self.assertFalse(user.has_permission("admin.users"))

    def test_inactive_user_no_permissions(self):
        """Test inactive user has no permissions."""
        level = self._create_level("admin", {"*"})
        user = self._create_user(is_active=False, levels=[level])

        self.assertFalse(user.has_permission("job.submit"))
        self.assertFalse(user.has_permission("anything"))

    def test_has_any_permission(self):
        """Test has_any_permission checks multiple."""
        level = self._create_level("basic", {"job.submit", "job.view.own"})
        user = self._create_user(levels=[level])

        self.assertTrue(user.has_any_permission(["job.submit", "admin.users"]))
        self.assertFalse(user.has_any_permission(["admin.users", "admin.config"]))

    def test_has_all_permissions(self):
        """Test has_all_permissions checks all."""
        level = self._create_level("basic", {"job.submit", "job.view.own"})
        user = self._create_user(levels=[level])

        self.assertTrue(user.has_all_permissions(["job.submit", "job.view.own"]))
        self.assertFalse(user.has_all_permissions(["job.submit", "admin.users"]))

    def test_highest_level(self):
        """Test getting highest priority level."""
        low = UserLevel(id=1, name="low", description="", priority=10)
        high = UserLevel(id=2, name="high", description="", priority=50)
        user = self._create_user(levels=[low, high])

        self.assertEqual(user.highest_level, high)

    def test_to_dict(self):
        """Test user serialization to dict."""
        user = self._create_user()
        d = user.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["email"], "test@example.com")
        self.assertIn("permissions", d)
        self.assertNotIn("password_hash", d)  # Excluded by default

    def test_to_dict_with_sensitive(self):
        """Test user serialization with sensitive data."""
        user = self._create_user(password_hash="secret_hash")
        d = user.to_dict(include_sensitive=True)

        self.assertEqual(d["password_hash"], "secret_hash")


class TestResourceRule(unittest.TestCase):
    """Tests for ResourceRule and RBAC."""

    def _create_rule(
        self,
        resource_type: ResourceType = ResourceType.CONFIG,
        pattern: str = "test-*",
        action: RuleAction = RuleAction.ALLOW,
        **kwargs,
    ) -> ResourceRule:
        """Create a test resource rule."""
        return ResourceRule(
            id=1,
            name="Test Rule",
            resource_type=resource_type,
            pattern=pattern,
            action=action,
            **kwargs,
        )

    def test_create_rule(self):
        """Test creating a resource rule."""
        rule = self._create_rule()

        self.assertEqual(rule.resource_type, ResourceType.CONFIG)
        self.assertEqual(rule.pattern, "test-*")
        self.assertEqual(rule.action, RuleAction.ALLOW)

    def test_rule_matches_glob(self):
        """Test rule glob pattern matching."""
        rule = self._create_rule(pattern="team-x-*")

        self.assertTrue(rule.matches("team-x-training"))
        self.assertTrue(rule.matches("team-x-eval"))
        self.assertFalse(rule.matches("team-y-training"))

    def test_rule_matches_exact(self):
        """Test rule exact pattern matching."""
        rule = self._create_rule(pattern="specific-config")

        self.assertTrue(rule.matches("specific-config"))
        self.assertFalse(rule.matches("specific-config-extra"))

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = self._create_rule()
        d = rule.to_dict()

        self.assertEqual(d["resource_type"], "config")
        self.assertEqual(d["action"], "allow")

    def test_rule_from_dict(self):
        """Test rule deserialization."""
        data = {
            "id": 1,
            "name": "Test",
            "resource_type": "hardware",
            "pattern": "gpu-*",
            "action": "deny",
            "priority": 10,
        }

        rule = ResourceRule.from_dict(data)

        self.assertEqual(rule.resource_type, ResourceType.HARDWARE)
        self.assertEqual(rule.action, RuleAction.DENY)
        self.assertEqual(rule.priority, 10)


class TestUserResourceAccess(unittest.TestCase):
    """Tests for user resource-based access control."""

    def _create_user_with_rules(self, rules: List[ResourceRule], is_admin: bool = False) -> User:
        """Create a user with resource rules."""
        return User(
            id=1,
            email="test@example.com",
            username="testuser",
            is_active=True,
            is_admin=is_admin,
            resource_rules=rules,
        )

    def test_admin_bypasses_rules(self):
        """Test admin users bypass resource restrictions."""
        deny_rule = ResourceRule(
            id=1,
            name="Deny All",
            resource_type=ResourceType.CONFIG,
            pattern="*",
            action=RuleAction.DENY,
        )
        user = self._create_user_with_rules([deny_rule], is_admin=True)

        allowed, reason = user.can_access_resource(ResourceType.CONFIG, "anything")

        self.assertTrue(allowed)

    def test_no_rules_allows_access(self):
        """Test access is allowed when no rules are defined."""
        user = self._create_user_with_rules([])

        allowed, reason = user.can_access_resource(ResourceType.CONFIG, "any-config")

        self.assertTrue(allowed)

    def test_allow_rule_grants_access(self):
        """Test ALLOW rule grants access."""
        rule = ResourceRule(
            id=1,
            name="Allow Team X",
            resource_type=ResourceType.CONFIG,
            pattern="team-x-*",
            action=RuleAction.ALLOW,
        )
        user = self._create_user_with_rules([rule])

        allowed, reason = user.can_access_resource(ResourceType.CONFIG, "team-x-training")

        self.assertTrue(allowed)

    def test_deny_rule_blocks_access(self):
        """Test DENY rule blocks access when no ALLOW matches."""
        rule = ResourceRule(
            id=1,
            name="Deny Expensive",
            resource_type=ResourceType.HARDWARE,
            pattern="gpu-a100*",
            action=RuleAction.DENY,
        )
        user = self._create_user_with_rules([rule])

        allowed, reason = user.can_access_resource(ResourceType.HARDWARE, "gpu-a100-80gb")

        self.assertFalse(allowed)
        self.assertIn("Access denied", reason)

    def test_allow_wins_over_deny(self):
        """Test ALLOW rule takes precedence over DENY (most permissive wins)."""
        deny_rule = ResourceRule(
            id=1,
            name="Deny All",
            resource_type=ResourceType.CONFIG,
            pattern="*",
            action=RuleAction.DENY,
        )
        allow_rule = ResourceRule(
            id=2,
            name="Allow Team X",
            resource_type=ResourceType.CONFIG,
            pattern="team-x-*",
            action=RuleAction.ALLOW,
        )
        user = self._create_user_with_rules([deny_rule, allow_rule])

        allowed, reason = user.can_access_resource(ResourceType.CONFIG, "team-x-training")

        self.assertTrue(allowed)

    def test_no_matching_rule_denies(self):
        """Test access denied when no rules match."""
        rule = ResourceRule(
            id=1,
            name="Allow Team X",
            resource_type=ResourceType.CONFIG,
            pattern="team-x-*",
            action=RuleAction.ALLOW,
        )
        user = self._create_user_with_rules([rule])

        allowed, reason = user.can_access_resource(ResourceType.CONFIG, "team-y-training")

        self.assertFalse(allowed)
        self.assertIn("No rule grants access", reason)

    def test_rules_filter_by_type(self):
        """Test rules only apply to their resource type."""
        config_rule = ResourceRule(
            id=1,
            name="Allow Config",
            resource_type=ResourceType.CONFIG,
            pattern="*",
            action=RuleAction.ALLOW,
        )
        user = self._create_user_with_rules([config_rule])

        # Config should be allowed
        allowed, _ = user.can_access_resource(ResourceType.CONFIG, "any-config")
        self.assertTrue(allowed)

        # Hardware has no rules, so allowed by default
        allowed, _ = user.can_access_resource(ResourceType.HARDWARE, "gpu-a100")
        self.assertTrue(allowed)

    def test_get_rules_for_type(self):
        """Test filtering rules by resource type."""
        config_rule = ResourceRule(
            id=1, name="Config", resource_type=ResourceType.CONFIG, pattern="*", action=RuleAction.ALLOW
        )
        hardware_rule = ResourceRule(
            id=2, name="Hardware", resource_type=ResourceType.HARDWARE, pattern="*", action=RuleAction.ALLOW
        )
        user = self._create_user_with_rules([config_rule, hardware_rule])

        config_rules = user.get_resource_rules_for_type(ResourceType.CONFIG)
        self.assertEqual(len(config_rules), 1)
        self.assertEqual(config_rules[0].name, "Config")


class TestOrganization(unittest.TestCase):
    """Tests for Organization model."""

    def test_create_organization(self):
        """Test creating an organization."""
        org = Organization(
            id=1,
            name="Acme Corp",
            slug="acme-corp",
            description="Main organization",
            is_active=True,
        )

        self.assertEqual(org.name, "Acme Corp")
        self.assertEqual(org.slug, "acme-corp")
        self.assertTrue(org.is_active)

    def test_org_to_dict(self):
        """Test organization serialization."""
        org = Organization(id=1, name="Acme", slug="acme", description="Test")
        d = org.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["name"], "Acme")
        self.assertIn("settings", d)

    def test_org_from_dict(self):
        """Test organization deserialization."""
        data = {
            "id": 5,
            "name": "Test Org",
            "slug": "test-org",
            "description": "Testing",
            "is_active": True,
        }

        org = Organization.from_dict(data)

        self.assertEqual(org.id, 5)
        self.assertEqual(org.slug, "test-org")


class TestTeam(unittest.TestCase):
    """Tests for Team model."""

    def test_create_team(self):
        """Test creating a team."""
        team = Team(
            id=1,
            org_id=1,
            name="Engineering",
            slug="engineering",
            description="Engineering team",
        )

        self.assertEqual(team.name, "Engineering")
        self.assertEqual(team.org_id, 1)

    def test_team_to_dict(self):
        """Test team serialization."""
        team = Team(id=1, org_id=1, name="Dev", slug="dev", description="Developers")
        d = team.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["org_id"], 1)

    def test_team_from_dict(self):
        """Test team deserialization."""
        data = {
            "id": 3,
            "org_id": 1,
            "name": "QA Team",
            "slug": "qa-team",
            "is_active": True,
        }

        team = Team.from_dict(data)

        self.assertEqual(team.id, 3)
        self.assertEqual(team.name, "QA Team")


class TestAPIKey(unittest.TestCase):
    """Tests for API key model."""

    def test_create_api_key(self):
        """Test creating an API key."""
        key = APIKey(
            id=1,
            user_id=1,
            name="My API Key",
            key_prefix="st_abc123",
            key_hash="hashed_value",
            created_at="2024-01-15T10:00:00Z",
            is_active=True,
        )

        self.assertEqual(key.name, "My API Key")
        self.assertEqual(key.key_prefix, "st_abc123")
        self.assertTrue(key.is_active)

    def test_api_key_not_expired(self):
        """Test non-expired API key."""
        key = APIKey(
            id=1,
            user_id=1,
            name="Key",
            key_prefix="st_",
            key_hash="hash",
            created_at="2024-01-15T10:00:00Z",
            expires_at=None,  # Never expires
        )

        self.assertFalse(key.is_expired())

    def test_api_key_expired(self):
        """Test expired API key."""
        key = APIKey(
            id=1,
            user_id=1,
            name="Key",
            key_prefix="st_",
            key_hash="hash",
            created_at="2024-01-15T10:00:00Z",
            expires_at="2024-01-01T00:00:00Z",  # Already past
        )

        self.assertTrue(key.is_expired())

    def test_scoped_permissions(self):
        """Test API key with scoped permissions."""
        key = APIKey(
            id=1,
            user_id=1,
            name="Limited Key",
            key_prefix="st_",
            key_hash="hash",
            created_at="2024-01-15T10:00:00Z",
            scoped_permissions={"job.view.own", "config.view"},
        )

        self.assertIn("job.view.own", key.scoped_permissions)
        self.assertNotIn("admin.users", key.scoped_permissions)

    def test_api_key_without_hash(self):
        """Test creating an API key without key_hash (as returned from create)."""
        key = APIKey(
            id=1,
            user_id=1,
            name="New Key",
            key_prefix="st_abc123",
            created_at="2024-01-15T10:00:00Z",
        )

        self.assertEqual(key.name, "New Key")
        self.assertIsNone(key.key_hash)
        self.assertTrue(key.is_active)


class TestAuthProvider(unittest.TestCase):
    """Tests for AuthProvider enum."""

    def test_auth_providers(self):
        """Test auth provider values."""
        self.assertEqual(AuthProvider.LOCAL.value, "local")
        self.assertEqual(AuthProvider.OIDC.value, "oidc")
        self.assertEqual(AuthProvider.LDAP.value, "ldap")

    def test_user_with_provider(self):
        """Test user with different auth providers."""
        local_user = User(
            id=1,
            email="local@example.com",
            username="local",
            auth_provider=AuthProvider.LOCAL,
        )
        oidc_user = User(
            id=2,
            email="oidc@example.com",
            username="oidc",
            auth_provider=AuthProvider.OIDC,
            external_id="oidc-subject-123",
        )

        self.assertEqual(local_user.auth_provider, AuthProvider.LOCAL)
        self.assertEqual(oidc_user.auth_provider, AuthProvider.OIDC)
        self.assertEqual(oidc_user.external_id, "oidc-subject-123")


class TestResourceTypes(unittest.TestCase):
    """Tests for ResourceType enum."""

    def test_resource_types(self):
        """Test resource type values."""
        self.assertEqual(ResourceType.CONFIG.value, "config")
        self.assertEqual(ResourceType.HARDWARE.value, "hardware")
        self.assertEqual(ResourceType.PROVIDER.value, "provider")
        self.assertEqual(ResourceType.OUTPUT_PATH.value, "output_path")


class TestRuleActions(unittest.TestCase):
    """Tests for RuleAction enum."""

    def test_rule_actions(self):
        """Test rule action values."""
        self.assertEqual(RuleAction.ALLOW.value, "allow")
        self.assertEqual(RuleAction.DENY.value, "deny")


class TestPermissionHierarchy(unittest.TestCase):
    """Tests for permission hierarchy and wildcards."""

    def test_single_level_wildcard(self):
        """Test single-level wildcard (e.g., 'job.*')."""
        level = UserLevel(
            id=1,
            name="jobs",
            description="",
            priority=10,
            permissions={"job.*"},
        )
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            levels=[level],
        )

        self.assertTrue(user.has_permission("job.submit"))
        self.assertTrue(user.has_permission("job.cancel"))
        self.assertFalse(user.has_permission("config.view"))

    def test_nested_wildcard(self):
        """Test nested wildcard (e.g., 'admin.*')."""
        level = UserLevel(
            id=1,
            name="admin",
            description="",
            priority=100,
            permissions={"admin.*"},
        )
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            levels=[level],
        )

        self.assertTrue(user.has_permission("admin.users"))
        self.assertTrue(user.has_permission("admin.config"))
        self.assertFalse(user.has_permission("job.submit"))

    def test_full_wildcard(self):
        """Test full wildcard (*)."""
        level = UserLevel(
            id=1,
            name="superadmin",
            description="",
            priority=1000,
            permissions={"*"},
        )
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            levels=[level],
        )

        self.assertTrue(user.has_permission("anything"))
        self.assertTrue(user.has_permission("deeply.nested.permission"))


class TestUserOrganizationMembership(unittest.TestCase):
    """Tests for user organization and team membership."""

    def test_user_with_organization(self):
        """Test user with organization membership."""
        org = Organization(id=1, name="Acme", slug="acme")
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            org_id=1,
            organization=org,
        )

        self.assertEqual(user.org_id, 1)
        self.assertIsNotNone(user.organization)
        self.assertEqual(user.organization.name, "Acme")

    def test_user_with_teams(self):
        """Test user with team memberships."""
        team1 = Team(id=1, org_id=1, name="Dev", slug="dev")
        team2 = Team(id=2, org_id=1, name="QA", slug="qa")
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            org_id=1,
            teams=[team1, team2],
        )

        self.assertEqual(len(user.teams), 2)
        team_names = {t.name for t in user.teams}
        self.assertIn("Dev", team_names)
        self.assertIn("QA", team_names)

    def test_user_to_dict_includes_org_and_teams(self):
        """Test serialization includes org and team info."""
        org = Organization(id=1, name="Acme", slug="acme")
        team = Team(id=1, org_id=1, name="Dev", slug="dev")
        user = User(
            id=1,
            email="test@example.com",
            username="test",
            org_id=1,
            organization=org,
            teams=[team],
        )

        d = user.to_dict()

        self.assertEqual(d["org_id"], 1)
        self.assertIsNotNone(d["organization"])
        self.assertEqual(d["organization"]["name"], "Acme")
        self.assertEqual(len(d["teams"]), 1)


if __name__ == "__main__":
    unittest.main()
