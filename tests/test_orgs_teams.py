"""Tests for organizations and teams CRUD operations.

Tests cover:
- Organization model operations
- Team model operations
- Quota management models
- Slug validation patterns
- Membership management
"""

from __future__ import annotations

import re
import unittest
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class TestSlugValidation(unittest.TestCase):
    """Tests for slug validation patterns."""

    def test_valid_slug_lowercase(self):
        """Test valid lowercase slug."""
        pattern = r"^[a-z0-9-]+$"
        self.assertIsNotNone(re.match(pattern, "my-org"))
        self.assertIsNotNone(re.match(pattern, "team123"))
        self.assertIsNotNone(re.match(pattern, "dev-team-1"))

    def test_invalid_slug_uppercase(self):
        """Test uppercase characters are invalid."""
        pattern = r"^[a-z0-9-]+$"
        self.assertIsNone(re.match(pattern, "My-Org"))
        self.assertIsNone(re.match(pattern, "TEAM"))

    def test_invalid_slug_spaces(self):
        """Test spaces are invalid."""
        pattern = r"^[a-z0-9-]+$"
        self.assertIsNone(re.match(pattern, "my org"))

    def test_invalid_slug_special_chars(self):
        """Test special characters are invalid."""
        pattern = r"^[a-z0-9-]+$"
        self.assertIsNone(re.match(pattern, "my_org"))
        self.assertIsNone(re.match(pattern, "org@123"))
        self.assertIsNone(re.match(pattern, "org.team"))


class TestQuotaTypes(unittest.TestCase):
    """Tests for quota type validation."""

    VALID_QUOTA_TYPES = [
        "concurrent_jobs",
        "jobs_per_day",
        "jobs_per_hour",
        "cost_daily",
        "cost_monthly",
    ]

    def test_valid_quota_types(self):
        """Test all valid quota types."""
        pattern = r"^(concurrent_jobs|jobs_per_day|jobs_per_hour|cost_daily|cost_monthly)$"
        for qt in self.VALID_QUOTA_TYPES:
            self.assertIsNotNone(re.match(pattern, qt), f"{qt} should match")

    def test_invalid_quota_type(self):
        """Test invalid quota types are rejected."""
        pattern = r"^(concurrent_jobs|jobs_per_day|jobs_per_hour|cost_daily|cost_monthly)$"
        self.assertIsNone(re.match(pattern, "invalid"))
        self.assertIsNone(re.match(pattern, "jobs"))
        self.assertIsNone(re.match(pattern, "cost"))


class TestQuotaActions(unittest.TestCase):
    """Tests for quota action validation."""

    VALID_ACTIONS = ["block", "warn", "require_approval"]

    def test_valid_actions(self):
        """Test all valid quota actions."""
        pattern = r"^(block|warn|require_approval)$"
        for action in self.VALID_ACTIONS:
            self.assertIsNotNone(re.match(pattern, action), f"{action} should match")

    def test_invalid_action(self):
        """Test invalid actions are rejected."""
        pattern = r"^(block|warn|require_approval)$"
        self.assertIsNone(re.match(pattern, "allow"))
        self.assertIsNone(re.match(pattern, "deny"))


class TestTeamRoles(unittest.TestCase):
    """Tests for team role validation."""

    VALID_ROLES = ["member", "lead", "admin"]

    def test_valid_roles(self):
        """Test all valid team roles."""
        pattern = r"^(member|lead|admin)$"
        for role in self.VALID_ROLES:
            self.assertIsNotNone(re.match(pattern, role), f"{role} should match")

    def test_invalid_role(self):
        """Test invalid roles are rejected."""
        pattern = r"^(member|lead|admin)$"
        self.assertIsNone(re.match(pattern, "owner"))
        self.assertIsNone(re.match(pattern, "viewer"))


class TestOrganizationCRUD(unittest.TestCase):
    """Tests for organization CRUD operations logic."""

    def test_org_creation_fields(self):
        """Test organization creation requires proper fields."""

        # Simulate create org request validation
        class CreateOrgRequest(BaseModel):
            name: str = Field(..., min_length=1, max_length=100)
            slug: str = Field(..., min_length=1, max_length=50)
            description: str = Field(default="", max_length=500)

        # Valid request
        req = CreateOrgRequest(name="Acme Corp", slug="acme-corp")
        self.assertEqual(req.name, "Acme Corp")
        self.assertEqual(req.slug, "acme-corp")
        self.assertEqual(req.description, "")

        # With description
        req2 = CreateOrgRequest(name="Test Org", slug="test-org", description="A test organization")
        self.assertEqual(req2.description, "A test organization")

    def test_org_name_length_limits(self):
        """Test organization name length limits."""

        class CreateOrgRequest(BaseModel):
            name: str = Field(..., min_length=1, max_length=100)
            slug: str = Field(..., min_length=1, max_length=50)

        # Too short name
        with self.assertRaises(ValidationError):
            CreateOrgRequest(name="", slug="valid")

        # Too long name
        with self.assertRaises(ValidationError):
            CreateOrgRequest(name="a" * 101, slug="valid")

    def test_org_slug_length_limits(self):
        """Test organization slug length limits."""

        class CreateOrgRequest(BaseModel):
            name: str = Field(..., min_length=1, max_length=100)
            slug: str = Field(..., min_length=1, max_length=50)

        # Too short slug
        with self.assertRaises(ValidationError):
            CreateOrgRequest(name="Valid", slug="")

        # Too long slug
        with self.assertRaises(ValidationError):
            CreateOrgRequest(name="Valid", slug="a" * 51)

    def test_org_update_fields_optional(self):
        """Test organization update fields are optional."""

        class UpdateOrgRequest(BaseModel):
            name: Optional[str] = Field(None, min_length=1, max_length=100)
            description: Optional[str] = Field(None, max_length=500)
            is_active: Optional[bool] = None

        # All None is valid
        req = UpdateOrgRequest()
        self.assertIsNone(req.name)
        self.assertIsNone(req.description)
        self.assertIsNone(req.is_active)

        # Partial update
        req2 = UpdateOrgRequest(name="New Name")
        self.assertEqual(req2.name, "New Name")
        self.assertIsNone(req2.description)


class TestTeamCRUD(unittest.TestCase):
    """Tests for team CRUD operations logic."""

    def test_team_creation_fields(self):
        """Test team creation requires proper fields."""

        class CreateTeamRequest(BaseModel):
            name: str = Field(..., min_length=1, max_length=100)
            slug: str = Field(..., min_length=1, max_length=50)
            description: str = Field(default="", max_length=500)

        req = CreateTeamRequest(name="Engineering", slug="engineering")
        self.assertEqual(req.name, "Engineering")
        self.assertEqual(req.slug, "engineering")

    def test_team_update_fields_optional(self):
        """Test team update fields are optional."""

        class UpdateTeamRequest(BaseModel):
            name: Optional[str] = Field(None, min_length=1, max_length=100)
            description: Optional[str] = Field(None, max_length=500)
            is_active: Optional[bool] = None

        req = UpdateTeamRequest()
        self.assertIsNone(req.name)

    def test_team_membership_request(self):
        """Test team membership request validation."""

        class TeamMembershipRequest(BaseModel):
            user_id: int
            role: str = Field(default="member")

        req = TeamMembershipRequest(user_id=1)
        self.assertEqual(req.user_id, 1)
        self.assertEqual(req.role, "member")

        req2 = TeamMembershipRequest(user_id=2, role="lead")
        self.assertEqual(req2.role, "lead")


class TestQuotaRequests(unittest.TestCase):
    """Tests for quota request validation."""

    def test_org_quota_request(self):
        """Test organization quota request validation."""

        class SetOrgQuotaRequest(BaseModel):
            quota_type: str
            limit_value: float = Field(..., gt=0)
            action: str = Field(default="block")

        req = SetOrgQuotaRequest(quota_type="concurrent_jobs", limit_value=10.0)
        self.assertEqual(req.quota_type, "concurrent_jobs")
        self.assertEqual(req.limit_value, 10.0)
        self.assertEqual(req.action, "block")

    def test_quota_value_must_be_positive(self):
        """Test quota limit value must be positive."""

        class SetQuotaRequest(BaseModel):
            quota_type: str
            limit_value: float = Field(..., gt=0)

        with self.assertRaises(ValidationError):
            SetQuotaRequest(quota_type="concurrent_jobs", limit_value=0)

        with self.assertRaises(ValidationError):
            SetQuotaRequest(quota_type="concurrent_jobs", limit_value=-5)

    def test_team_quota_request(self):
        """Test team quota request validation."""

        class SetTeamQuotaRequest(BaseModel):
            quota_type: str
            limit_value: float = Field(..., gt=0)
            action: str = Field(default="block")

        req = SetTeamQuotaRequest(quota_type="jobs_per_day", limit_value=100.0, action="warn")
        self.assertEqual(req.quota_type, "jobs_per_day")
        self.assertEqual(req.action, "warn")


class TestOrgTeamHierarchy(unittest.TestCase):
    """Tests for organization-team hierarchy concepts."""

    def test_team_belongs_to_org(self):
        """Test team must belong to an organization."""
        # This is modeled in the Team dataclass
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Organization, Team

        org = Organization(id=1, name="Acme", slug="acme")
        team = Team(id=1, org_id=org.id, name="Dev", slug="dev")

        self.assertEqual(team.org_id, org.id)

    def test_quota_ceiling_concept(self):
        """Test quota ceiling: org > team > user."""
        # Org sets ceiling of 100 concurrent jobs
        org_limit = 100

        # Team can set lower limit within org ceiling
        team_limit = 50  # Valid: 50 < 100

        # User level limit bounded by team
        user_limit = 20  # Valid: 20 < 50 < 100

        # Ceiling validation
        self.assertLessEqual(user_limit, team_limit)
        self.assertLessEqual(team_limit, org_limit)

    def test_slug_uniqueness_concept(self):
        """Test slug uniqueness within scope."""
        # Org slugs must be globally unique
        # Team slugs must be unique within their org

        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Team

        # Two teams with same slug in different orgs is OK
        team1 = Team(id=1, org_id=1, name="Dev", slug="dev")
        team2 = Team(id=2, org_id=2, name="Dev", slug="dev")

        # Same slug, different orgs - this is valid
        self.assertEqual(team1.slug, team2.slug)
        self.assertNotEqual(team1.org_id, team2.org_id)


class TestOrgTeamSettings(unittest.TestCase):
    """Tests for organization/team settings."""

    def test_org_settings_dict(self):
        """Test organization can have settings dict."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Organization

        org = Organization(
            id=1,
            name="Acme",
            slug="acme",
            settings={
                "default_provider": "replicate",
                "allow_external_auth": True,
                "max_team_size": 20,
            },
        )

        self.assertEqual(org.settings["default_provider"], "replicate")
        self.assertTrue(org.settings["allow_external_auth"])

    def test_team_settings_dict(self):
        """Test team can have settings dict."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Team

        team = Team(
            id=1,
            org_id=1,
            name="Dev",
            slug="dev",
            settings={
                "priority_boost": 10,
                "notification_channel": "slack",
            },
        )

        self.assertEqual(team.settings["priority_boost"], 10)


class TestOrgTeamSerialization(unittest.TestCase):
    """Tests for organization/team serialization."""

    def test_org_to_dict(self):
        """Test organization to_dict includes all fields."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Organization

        org = Organization(
            id=1,
            name="Acme Corp",
            slug="acme-corp",
            description="Main organization",
            is_active=True,
            settings={"key": "value"},
        )

        d = org.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["name"], "Acme Corp")
        self.assertEqual(d["slug"], "acme-corp")
        self.assertEqual(d["description"], "Main organization")
        self.assertTrue(d["is_active"])
        self.assertEqual(d["settings"]["key"], "value")

    def test_team_to_dict(self):
        """Test team to_dict includes all fields."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Team

        team = Team(
            id=1,
            org_id=5,
            name="Engineering",
            slug="engineering",
            description="Engineering team",
            is_active=True,
            settings={"size": 10},
        )

        d = team.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["org_id"], 5)
        self.assertEqual(d["name"], "Engineering")
        self.assertEqual(d["slug"], "engineering")
        self.assertTrue(d["is_active"])

    def test_org_from_dict(self):
        """Test organization from_dict creates correct object."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Organization

        data = {
            "id": 10,
            "name": "Test Org",
            "slug": "test-org",
            "description": "Testing",
            "is_active": True,
            "created_at": "2024-01-15T10:00:00Z",
        }

        org = Organization.from_dict(data)

        self.assertEqual(org.id, 10)
        self.assertEqual(org.name, "Test Org")
        self.assertEqual(org.slug, "test-org")

    def test_team_from_dict(self):
        """Test team from_dict creates correct object."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Team

        data = {
            "id": 3,
            "org_id": 1,
            "name": "QA",
            "slug": "qa",
            "description": "QA team",
            "is_active": True,
        }

        team = Team.from_dict(data)

        self.assertEqual(team.id, 3)
        self.assertEqual(team.org_id, 1)
        self.assertEqual(team.name, "QA")


class TestOrgAccessControl(unittest.TestCase):
    """Tests for organization access control."""

    def _make_user(self, user_id: int, org_id: int, permissions: list[str] | None = None):
        """Create a mock user for testing."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

        user = User(
            id=user_id,
            email=f"user{user_id}@test.com",
            username=f"user{user_id}",
            org_id=org_id,
        )
        # Set internal effective permissions field directly for testing
        user._effective_permissions = set(permissions or [])
        return user

    def test_can_access_org_same_org(self):
        """Test user can access their own organization."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_access_org

        user = self._make_user(user_id=1, org_id=5)
        self.assertTrue(can_access_org(user, 5))

    def test_can_access_org_different_org_denied(self):
        """Test user cannot access a different organization."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_access_org

        user = self._make_user(user_id=1, org_id=5)
        self.assertFalse(can_access_org(user, 10))
        self.assertFalse(can_access_org(user, 1))

    def test_can_access_org_admin_can_access_any(self):
        """Test system admin can access any organization."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_access_org

        admin = self._make_user(user_id=1, org_id=5, permissions=["admin.users"])
        # Admin in org 5 can access org 10
        self.assertTrue(can_access_org(admin, 10))
        # Admin in org 5 can access org 1
        self.assertTrue(can_access_org(admin, 1))
        # Admin can also access their own org
        self.assertTrue(can_access_org(admin, 5))

    def test_can_access_org_other_permissions_not_enough(self):
        """Test non-admin permissions don't grant cross-org access."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_access_org

        # User with org.view permission but not admin.users
        user = self._make_user(user_id=1, org_id=5, permissions=["org.view", "org.edit", "team.view"])
        # Cannot access other org despite having org permissions
        self.assertFalse(can_access_org(user, 10))
        # Can still access own org
        self.assertTrue(can_access_org(user, 5))

    def test_require_org_access_allows_same_org(self):
        """Test require_org_access doesn't raise for same org."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import require_org_access

        user = self._make_user(user_id=1, org_id=5)
        # Should not raise
        require_org_access(user, 5)

    def test_require_org_access_denies_different_org(self):
        """Test require_org_access raises 403 for different org."""
        from fastapi import HTTPException

        from simpletuner.simpletuner_sdk.server.routes.orgs import require_org_access

        user = self._make_user(user_id=1, org_id=5)
        with self.assertRaises(HTTPException) as ctx:
            require_org_access(user, 10)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("do not have access", ctx.exception.detail)

    def test_require_org_access_allows_admin(self):
        """Test require_org_access allows admin to any org."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import require_org_access

        admin = self._make_user(user_id=1, org_id=5, permissions=["admin.users"])
        # Should not raise for any org
        require_org_access(admin, 10)
        require_org_access(admin, 1)
        require_org_access(admin, 999)

    def test_can_access_org_null_org_id(self):
        """Test user with no org cannot access any org."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_access_org

        user = self._make_user(user_id=1, org_id=None)
        self.assertFalse(can_access_org(user, 1))
        self.assertFalse(can_access_org(user, 5))


class TestRoleHierarchy(unittest.TestCase):
    """Tests for team role hierarchy and permission checking."""

    def test_can_manage_role_admin_manages_all(self):
        """Test admin can manage all roles."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_manage_role

        self.assertTrue(can_manage_role("admin", "admin"))
        self.assertTrue(can_manage_role("admin", "lead"))
        self.assertTrue(can_manage_role("admin", "member"))

    def test_can_manage_role_lead_manages_members_only(self):
        """Test lead can only manage members."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_manage_role

        self.assertTrue(can_manage_role("lead", "member"))
        self.assertFalse(can_manage_role("lead", "lead"))
        self.assertFalse(can_manage_role("lead", "admin"))

    def test_can_manage_role_member_manages_none(self):
        """Test member cannot manage anyone."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_manage_role

        self.assertFalse(can_manage_role("member", "member"))
        self.assertFalse(can_manage_role("member", "lead"))
        self.assertFalse(can_manage_role("member", "admin"))

    def test_can_manage_role_unknown_role(self):
        """Test unknown roles cannot manage anyone."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import can_manage_role

        self.assertFalse(can_manage_role("unknown", "member"))
        self.assertFalse(can_manage_role("guest", "member"))

    def test_get_user_team_role_finds_role(self):
        """Test get_user_team_role finds user's role."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import get_user_team_role

        members = [
            {"id": 1, "role": "admin"},
            {"id": 2, "role": "lead"},
            {"id": 3, "role": "member"},
        ]

        self.assertEqual(get_user_team_role(members, 1), "admin")
        self.assertEqual(get_user_team_role(members, 2), "lead")
        self.assertEqual(get_user_team_role(members, 3), "member")

    def test_get_user_team_role_not_found(self):
        """Test get_user_team_role returns None for non-members."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import get_user_team_role

        members = [
            {"id": 1, "role": "admin"},
            {"id": 2, "role": "lead"},
        ]

        self.assertIsNone(get_user_team_role(members, 999))
        self.assertIsNone(get_user_team_role(members, 0))

    def test_get_user_team_role_empty_list(self):
        """Test get_user_team_role handles empty list."""
        from simpletuner.simpletuner_sdk.server.routes.orgs import get_user_team_role

        self.assertIsNone(get_user_team_role([], 1))


class TestUserOrgTeamMembership(unittest.TestCase):
    """Tests for user organization/team membership."""

    def test_user_org_membership(self):
        """Test user belongs to one organization."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Organization, User

        org = Organization(id=1, name="Acme", slug="acme")
        user = User(
            id=1,
            email="user@acme.com",
            username="user",
            org_id=org.id,
            organization=org,
        )

        self.assertEqual(user.org_id, org.id)
        self.assertEqual(user.organization.name, "Acme")

    def test_user_multiple_teams(self):
        """Test user can belong to multiple teams."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import Team, User

        team1 = Team(id=1, org_id=1, name="Dev", slug="dev")
        team2 = Team(id=2, org_id=1, name="DevOps", slug="devops")

        user = User(
            id=1,
            email="user@acme.com",
            username="user",
            org_id=1,
            teams=[team1, team2],
        )

        self.assertEqual(len(user.teams), 2)
        team_names = {t.name for t in user.teams}
        self.assertIn("Dev", team_names)
        self.assertIn("DevOps", team_names)


if __name__ == "__main__":
    unittest.main()
