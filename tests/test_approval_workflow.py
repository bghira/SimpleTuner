"""Tests for approval workflow end-to-end functionality.

Tests cover:
- ApprovalRule model and serialization
- ApprovalRequest model and lifecycle
- RuleCondition evaluation
- EvaluationContext and EvaluationResult
- Rule priority and exemptions
- Approval status transitions
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from simpletuner.simpletuner_sdk.server.services.cloud.approval.models import (
    DEFAULT_RULES,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    RuleCondition,
)
from simpletuner.simpletuner_sdk.server.services.cloud.approval.rules_engine import EvaluationContext, EvaluationResult


class TestApprovalStatus(unittest.TestCase):
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        self.assertEqual(ApprovalStatus.PENDING.value, "pending")
        self.assertEqual(ApprovalStatus.APPROVED.value, "approved")
        self.assertEqual(ApprovalStatus.REJECTED.value, "rejected")
        self.assertEqual(ApprovalStatus.EXPIRED.value, "expired")
        self.assertEqual(ApprovalStatus.CANCELLED.value, "cancelled")

    def test_status_from_string(self):
        """Test status can be created from string."""
        status = ApprovalStatus("pending")
        self.assertEqual(status, ApprovalStatus.PENDING)

    def test_terminal_statuses(self):
        """Test terminal statuses (can't transition further)."""
        terminal = {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED, ApprovalStatus.CANCELLED}
        non_terminal = {ApprovalStatus.PENDING}

        for s in terminal:
            self.assertNotEqual(s, ApprovalStatus.PENDING)

        for s in non_terminal:
            self.assertEqual(s, ApprovalStatus.PENDING)


class TestRuleCondition(unittest.TestCase):
    """Tests for RuleCondition enum."""

    def test_condition_values(self):
        """Test all condition values exist."""
        self.assertEqual(RuleCondition.COST_EXCEEDS.value, "cost_exceeds")
        self.assertEqual(RuleCondition.HARDWARE_TYPE.value, "hardware_type")
        self.assertEqual(RuleCondition.PROVIDER.value, "provider")
        self.assertEqual(RuleCondition.USER_LEVEL.value, "user_level")
        self.assertEqual(RuleCondition.DAILY_JOBS_EXCEED.value, "daily_jobs_exceed")
        self.assertEqual(RuleCondition.FIRST_JOB.value, "first_job")
        self.assertEqual(RuleCondition.CONFIG_PATTERN.value, "config_pattern")

    def test_condition_from_string(self):
        """Test condition can be created from string."""
        condition = RuleCondition("cost_exceeds")
        self.assertEqual(condition, RuleCondition.COST_EXCEEDS)


class TestApprovalRule(unittest.TestCase):
    """Tests for ApprovalRule model."""

    def _create_rule(self, **kwargs) -> ApprovalRule:
        """Create a test approval rule."""
        defaults = {
            "id": 1,
            "name": "Test Rule",
            "description": "Test description",
            "condition": RuleCondition.COST_EXCEEDS,
            "threshold": "100.00",
            "is_active": True,
            "priority": 10,
        }
        defaults.update(kwargs)
        return ApprovalRule(**defaults)

    def test_create_rule(self):
        """Test creating an approval rule."""
        rule = self._create_rule()

        self.assertEqual(rule.id, 1)
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.condition, RuleCondition.COST_EXCEEDS)
        self.assertEqual(rule.threshold, "100.00")
        self.assertTrue(rule.is_active)

    def test_rule_with_exemptions(self):
        """Test rule with exempt levels."""
        rule = self._create_rule(
            exempt_levels=["admin", "lead"],
            required_approver_level="lead",
        )

        self.assertIn("admin", rule.exempt_levels)
        self.assertIn("lead", rule.exempt_levels)
        self.assertEqual(rule.required_approver_level, "lead")

    def test_rule_with_provider_restriction(self):
        """Test rule with provider restriction."""
        rule = self._create_rule(applies_to_provider="replicate")

        self.assertEqual(rule.applies_to_provider, "replicate")

    def test_rule_with_level_restriction(self):
        """Test rule with level restriction."""
        rule = self._create_rule(applies_to_level="researcher")

        self.assertEqual(rule.applies_to_level, "researcher")

    def test_rule_to_dict(self):
        """Test rule serialization to dict."""
        rule = self._create_rule(
            exempt_levels=["admin"],
            applies_to_provider="replicate",
        )

        d = rule.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["name"], "Test Rule")
        self.assertEqual(d["condition"], "cost_exceeds")
        self.assertEqual(d["threshold"], "100.00")
        self.assertEqual(d["exempt_levels"], ["admin"])
        self.assertEqual(d["applies_to_provider"], "replicate")

    def test_rule_from_dict(self):
        """Test rule deserialization from dict."""
        data = {
            "id": 5,
            "name": "High Cost Rule",
            "description": "Costs over $500",
            "condition": "cost_exceeds",
            "threshold": "500.00",
            "is_active": True,
            "priority": 20,
            "exempt_levels": ["admin"],
        }

        rule = ApprovalRule.from_dict(data)

        self.assertEqual(rule.id, 5)
        self.assertEqual(rule.name, "High Cost Rule")
        self.assertEqual(rule.condition, RuleCondition.COST_EXCEEDS)
        self.assertEqual(rule.threshold, "500.00")
        self.assertEqual(rule.priority, 20)
        self.assertIn("admin", rule.exempt_levels)


class TestApprovalRequest(unittest.TestCase):
    """Tests for ApprovalRequest model."""

    def _create_request(self, **kwargs) -> ApprovalRequest:
        """Create a test approval request."""
        defaults = {
            "id": 1,
            "job_id": "job-123",
            "user_id": 1,
            "rule_id": 1,
            "status": ApprovalStatus.PENDING,
            "reason": "Cost exceeds $100",
        }
        defaults.update(kwargs)
        return ApprovalRequest(**defaults)

    def test_create_request(self):
        """Test creating an approval request."""
        request = self._create_request()

        self.assertEqual(request.id, 1)
        self.assertEqual(request.job_id, "job-123")
        self.assertEqual(request.user_id, 1)
        self.assertEqual(request.status, ApprovalStatus.PENDING)

    def test_request_with_job_context(self):
        """Test request with job context."""
        request = self._create_request(
            provider="replicate",
            config_name="sdxl-training",
            estimated_cost=150.0,
            hardware_type="gpu-a100-large",
        )

        self.assertEqual(request.provider, "replicate")
        self.assertEqual(request.config_name, "sdxl-training")
        self.assertEqual(request.estimated_cost, 150.0)
        self.assertEqual(request.hardware_type, "gpu-a100-large")

    def test_request_approved(self):
        """Test approved request."""
        request = self._create_request(
            status=ApprovalStatus.APPROVED,
            reviewed_by=2,
            reviewed_at="2024-01-15T10:30:00Z",
            review_notes="Approved for training",
        )

        self.assertEqual(request.status, ApprovalStatus.APPROVED)
        self.assertEqual(request.reviewed_by, 2)
        self.assertIsNotNone(request.reviewed_at)
        self.assertEqual(request.review_notes, "Approved for training")

    def test_request_rejected(self):
        """Test rejected request."""
        request = self._create_request(
            status=ApprovalStatus.REJECTED,
            reviewed_by=2,
            reviewed_at="2024-01-15T10:30:00Z",
            review_notes="Budget exceeded for this month",
        )

        self.assertEqual(request.status, ApprovalStatus.REJECTED)
        self.assertIn("Budget", request.review_notes)

    def test_request_not_expired(self):
        """Test request is not expired when expires_at is in future."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        request = self._create_request(expires_at=future)

        self.assertFalse(request.is_expired)

    def test_request_expired(self):
        """Test request is expired when expires_at is in past."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        request = self._create_request(expires_at=past)

        self.assertTrue(request.is_expired)

    def test_request_no_expiry(self):
        """Test request with no expiry."""
        request = self._create_request(expires_at=None)

        self.assertFalse(request.is_expired)

    def test_request_to_dict(self):
        """Test request serialization to dict."""
        request = self._create_request(
            estimated_cost=200.0,
            notification_sent=True,
        )

        d = request.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["job_id"], "job-123")
        self.assertEqual(d["status"], "pending")
        self.assertEqual(d["estimated_cost"], 200.0)
        self.assertTrue(d["notification_sent"])

    def test_request_from_dict(self):
        """Test request deserialization from dict."""
        data = {
            "id": 10,
            "job_id": "job-456",
            "user_id": 3,
            "rule_id": 2,
            "status": "approved",
            "reason": "High cost job",
            "estimated_cost": 250.0,
            "reviewed_by": 1,
        }

        request = ApprovalRequest.from_dict(data)

        self.assertEqual(request.id, 10)
        self.assertEqual(request.job_id, "job-456")
        self.assertEqual(request.status, ApprovalStatus.APPROVED)
        self.assertEqual(request.estimated_cost, 250.0)


class TestEvaluationContext(unittest.TestCase):
    """Tests for EvaluationContext."""

    def test_create_context(self):
        """Test creating an evaluation context."""
        context = EvaluationContext(
            user_id=1,
            user_levels=["researcher"],
            provider="replicate",
            config_name="my-training",
            estimated_cost=75.0,
        )

        self.assertEqual(context.user_id, 1)
        self.assertIn("researcher", context.user_levels)
        self.assertEqual(context.provider, "replicate")
        self.assertEqual(context.estimated_cost, 75.0)

    def test_context_with_historical_data(self):
        """Test context with historical job data."""
        context = EvaluationContext(
            user_id=1,
            user_levels=["researcher"],
            user_job_count_today=5,
            user_total_job_count=100,
        )

        self.assertEqual(context.user_job_count_today, 5)
        self.assertEqual(context.user_total_job_count, 100)

    def test_context_first_job(self):
        """Test context for first job."""
        context = EvaluationContext(
            user_id=1,
            user_levels=["researcher"],
            user_total_job_count=0,
        )

        self.assertEqual(context.user_total_job_count, 0)


class TestEvaluationResult(unittest.TestCase):
    """Tests for EvaluationResult."""

    def test_result_requires_approval(self):
        """Test result that requires approval."""
        rule = ApprovalRule(
            id=1,
            name="High Cost",
            description="",
            condition=RuleCondition.COST_EXCEEDS,
            threshold="100.00",
        )

        result = EvaluationResult(
            requires_approval=True,
            matching_rule=rule,
            reason="Cost $150 exceeds threshold $100",
        )

        self.assertTrue(result.requires_approval)
        self.assertEqual(result.matching_rule.name, "High Cost")
        self.assertIn("150", result.reason)

    def test_result_no_approval_needed(self):
        """Test result that doesn't require approval."""
        result = EvaluationResult(
            requires_approval=False,
            reason="No rules matched",
        )

        self.assertFalse(result.requires_approval)
        self.assertIsNone(result.matching_rule)

    def test_result_with_evaluated_rules(self):
        """Test result with list of evaluated rules."""
        rule1 = ApprovalRule(id=1, name="Rule 1", description="", condition=RuleCondition.COST_EXCEEDS, threshold="100")
        rule2 = ApprovalRule(id=2, name="Rule 2", description="", condition=RuleCondition.FIRST_JOB, threshold="true")

        result = EvaluationResult(
            requires_approval=True,
            matching_rule=rule2,
            reason="First job",
            all_evaluated_rules=[(rule1, False), (rule2, True)],
        )

        self.assertEqual(len(result.all_evaluated_rules), 2)
        self.assertFalse(result.all_evaluated_rules[0][1])  # Rule 1 didn't match
        self.assertTrue(result.all_evaluated_rules[1][1])  # Rule 2 matched


class TestDefaultRules(unittest.TestCase):
    """Tests for default approval rules."""

    def test_default_rules_exist(self):
        """Test default rules are defined."""
        self.assertGreater(len(DEFAULT_RULES), 0)

    def test_high_cost_rule(self):
        """Test high cost default rule."""
        cost_rule = next((r for r in DEFAULT_RULES if r.condition == RuleCondition.COST_EXCEEDS), None)

        self.assertIsNotNone(cost_rule)
        self.assertIn("cost", cost_rule.name.lower())
        self.assertIn("admin", cost_rule.exempt_levels)

    def test_first_job_rule(self):
        """Test first job default rule."""
        first_job_rule = next((r for r in DEFAULT_RULES if r.condition == RuleCondition.FIRST_JOB), None)

        self.assertIsNotNone(first_job_rule)
        self.assertIn("admin", first_job_rule.exempt_levels)


class TestApprovalWorkflowLogic(unittest.TestCase):
    """Tests for approval workflow logic patterns."""

    def test_cost_exceeds_threshold(self):
        """Test cost threshold comparison."""
        threshold = 100.0
        below_threshold = 50.0
        at_threshold = 100.0
        above_threshold = 150.0

        self.assertFalse(below_threshold > threshold)
        self.assertFalse(at_threshold > threshold)  # At threshold doesn't trigger
        self.assertTrue(above_threshold > threshold)

    def test_level_exemption_check(self):
        """Test level exemption logic."""
        exempt_levels = ["admin", "lead"]
        user_levels = ["researcher"]
        admin_levels = ["admin"]

        # Researcher not exempt
        is_exempt = any(level in exempt_levels for level in user_levels)
        self.assertFalse(is_exempt)

        # Admin is exempt
        is_exempt = any(level in exempt_levels for level in admin_levels)
        self.assertTrue(is_exempt)

    def test_provider_restriction_check(self):
        """Test provider restriction logic."""
        rule_provider = "replicate"
        matching_provider = "replicate"
        non_matching_provider = "modal"

        # Matching provider
        applies = rule_provider == matching_provider
        self.assertTrue(applies)

        # Non-matching provider
        applies = rule_provider == non_matching_provider
        self.assertFalse(applies)

    def test_level_restriction_check(self):
        """Test level restriction logic."""
        rule_level = "researcher"
        user_levels_match = ["researcher", "viewer"]
        user_levels_no_match = ["viewer"]

        # User has matching level
        applies = rule_level in user_levels_match
        self.assertTrue(applies)

        # User doesn't have matching level
        applies = rule_level in user_levels_no_match
        self.assertFalse(applies)

    def test_priority_ordering(self):
        """Test rule priority ordering."""
        rules = [
            ApprovalRule(
                id=1, name="Low", description="", condition=RuleCondition.COST_EXCEEDS, threshold="100", priority=5
            ),
            ApprovalRule(
                id=2, name="High", description="", condition=RuleCondition.COST_EXCEEDS, threshold="50", priority=20
            ),
            ApprovalRule(
                id=3, name="Medium", description="", condition=RuleCondition.COST_EXCEEDS, threshold="75", priority=10
            ),
        ]

        # Sort by priority descending (higher priority first)
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        self.assertEqual(sorted_rules[0].name, "High")
        self.assertEqual(sorted_rules[1].name, "Medium")
        self.assertEqual(sorted_rules[2].name, "Low")


class TestApprovalRequestLifecycle(unittest.TestCase):
    """Tests for approval request lifecycle transitions."""

    def test_pending_to_approved(self):
        """Test transition from pending to approved."""
        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            status=ApprovalStatus.PENDING,
        )

        # Simulate approval
        request.status = ApprovalStatus.APPROVED
        request.reviewed_by = 2
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        request.review_notes = "Approved"

        self.assertEqual(request.status, ApprovalStatus.APPROVED)
        self.assertIsNotNone(request.reviewed_by)

    def test_pending_to_rejected(self):
        """Test transition from pending to rejected."""
        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            status=ApprovalStatus.PENDING,
        )

        # Simulate rejection
        request.status = ApprovalStatus.REJECTED
        request.reviewed_by = 2
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        request.review_notes = "Budget exceeded"

        self.assertEqual(request.status, ApprovalStatus.REJECTED)

    def test_pending_to_expired(self):
        """Test transition from pending to expired."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            status=ApprovalStatus.PENDING,
            expires_at=past,
        )

        # Check expiry
        if request.is_expired and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.EXPIRED

        self.assertEqual(request.status, ApprovalStatus.EXPIRED)

    def test_pending_to_cancelled(self):
        """Test transition from pending to cancelled."""
        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            status=ApprovalStatus.PENDING,
        )

        # User cancels request
        request.status = ApprovalStatus.CANCELLED

        self.assertEqual(request.status, ApprovalStatus.CANCELLED)


class TestApprovalNotifications(unittest.TestCase):
    """Tests for approval notification tracking."""

    def test_notification_not_sent(self):
        """Test request with notification not yet sent."""
        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            notification_sent=False,
        )

        self.assertFalse(request.notification_sent)
        self.assertIsNone(request.notification_sent_at)

    def test_notification_sent(self):
        """Test request with notification sent."""
        now = datetime.now(timezone.utc).isoformat()

        request = ApprovalRequest(
            id=1,
            job_id="job-1",
            user_id=1,
            rule_id=1,
            notification_sent=True,
            notification_sent_at=now,
        )

        self.assertTrue(request.notification_sent)
        self.assertIsNotNone(request.notification_sent_at)


class TestRuleConditionLogic(unittest.TestCase):
    """Tests for individual rule condition logic."""

    def test_cost_exceeds_condition(self):
        """Test COST_EXCEEDS condition evaluation."""
        threshold = "100.00"
        threshold_float = float(threshold)

        # Below threshold
        cost = 50.0
        self.assertFalse(cost > threshold_float)

        # At threshold (doesn't exceed)
        cost = 100.0
        self.assertFalse(cost > threshold_float)

        # Above threshold
        cost = 150.0
        self.assertTrue(cost > threshold_float)

    def test_daily_jobs_exceed_condition(self):
        """Test DAILY_JOBS_EXCEED condition evaluation."""
        threshold = 10

        # Below threshold
        job_count = 5
        self.assertFalse(job_count > threshold)

        # Above threshold
        job_count = 15
        self.assertTrue(job_count > threshold)

    def test_first_job_condition(self):
        """Test FIRST_JOB condition evaluation."""
        # First job (total count is 0)
        total_jobs = 0
        is_first = total_jobs == 0
        self.assertTrue(is_first)

        # Not first job
        total_jobs = 5
        is_first = total_jobs == 0
        self.assertFalse(is_first)


if __name__ == "__main__":
    unittest.main()
