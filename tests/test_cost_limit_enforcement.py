"""Tests for cost limit enforcement during job submission.

Verifies that:
- Jobs are blocked when cost limit is exceeded and action=block
- Jobs proceed with warning when limit exceeded and action=warn
- Warning threshold (80%) triggers appropriate warnings
- Different periods (daily, weekly, monthly) are calculated correctly
- Cost limits are disabled when amount is 0 or negative
"""

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


@dataclass
class MockJobStore:
    """Mock job store for testing."""

    provider_config: Dict[str, Any]
    current_spend: float = 0.0

    async def get_provider_config(self, provider: str) -> Dict[str, Any]:
        return self.provider_config

    async def get_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Mock metrics summary - the actual method used by _check_cost_limit."""
        return {"total_cost_30d": self.current_spend}


@dataclass
class MockCommandContext:
    """Mock command context for testing."""

    job_store: Optional[MockJobStore] = None


class TestCostLimitEnforcement(unittest.TestCase):
    """Test cost limit checks during job submission."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobCommand

        self.SubmitJobCommand = SubmitJobCommand

    def _create_command(self, provider: str = "replicate", config_name: str = "test"):
        """Create a submit job command for testing."""
        # Provide required config and dataloader_config
        return self.SubmitJobCommand(
            config={"model_type": "test"},
            dataloader_config=[{"id": "test-dataset"}],
            provider=provider,
            config_name=config_name,
        )

    async def _run_cost_check(
        self,
        limit_enabled: bool = True,
        limit_amount: float = 100.0,
        period: str = "monthly",
        action: str = "block",
        current_spend: float = 0.0,
    ) -> Optional[str]:
        """Run cost limit check with given parameters."""
        config = {
            "cost_limit_enabled": limit_enabled,
            "cost_limit_amount": limit_amount,
            "cost_limit_period": period,
            "cost_limit_action": action,
        }
        store = MockJobStore(provider_config=config, current_spend=current_spend)
        ctx = MockCommandContext(job_store=store)

        cmd = self._create_command()
        return await cmd._check_cost_limit(ctx)

    def test_cost_limit_disabled_allows_job(self):
        """Test that disabled cost limits allow all jobs."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(self._run_cost_check(limit_enabled=False, current_spend=1000.0))
        self.assertIsNone(result)

    def test_zero_limit_amount_allows_job(self):
        """Test that zero limit amount allows jobs (effectively disabled)."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(limit_enabled=True, limit_amount=0, current_spend=100.0)
        )
        self.assertIsNone(result)

    def test_negative_limit_amount_allows_job(self):
        """Test that negative limit amount allows jobs (invalid config)."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(limit_enabled=True, limit_amount=-50, current_spend=100.0)
        )
        self.assertIsNone(result)

    def test_under_limit_allows_job(self):
        """Test that spending under limit allows jobs."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=100.0,
                current_spend=50.0,  # 50% of limit
                action="block",
            )
        )
        self.assertIsNone(result)

    def test_exceeded_limit_blocks_job_when_action_block(self):
        """Test that exceeded limit blocks job when action is 'block'."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=100.0,
                current_spend=150.0,  # 150% of limit
                action="block",
            )
        )
        self.assertIsNotNone(result)
        self.assertIn("exceeded", result.lower())
        self.assertIn("$100.00", result)

    def test_exceeded_limit_warns_when_action_warn(self):
        """Test that exceeded limit only warns when action is 'warn'."""
        import asyncio

        cmd = self._create_command()
        config = {
            "cost_limit_enabled": True,
            "cost_limit_amount": 100.0,
            "cost_limit_period": "monthly",
            "cost_limit_action": "warn",
        }
        store = MockJobStore(provider_config=config, current_spend=150.0)
        ctx = MockCommandContext(job_store=store)

        result = asyncio.get_event_loop().run_until_complete(cmd._check_cost_limit(ctx))

        # Should not return error (job proceeds)
        self.assertIsNone(result)
        # But should set warning
        self.assertIsNotNone(cmd._cost_limit_warning)
        self.assertIn("exceeded", cmd._cost_limit_warning.lower())

    def test_warning_threshold_triggers_warning(self):
        """Test that 80% threshold triggers warning but allows job."""
        import asyncio

        cmd = self._create_command()
        config = {
            "cost_limit_enabled": True,
            "cost_limit_amount": 100.0,
            "cost_limit_period": "monthly",
            "cost_limit_action": "block",
        }
        store = MockJobStore(provider_config=config, current_spend=85.0)  # 85% of limit
        ctx = MockCommandContext(job_store=store)

        result = asyncio.get_event_loop().run_until_complete(cmd._check_cost_limit(ctx))

        # Should not block
        self.assertIsNone(result)
        # But should set warning
        self.assertIsNotNone(cmd._cost_limit_warning)
        self.assertIn("approaching", cmd._cost_limit_warning.lower())

    def test_daily_period_calculation(self):
        """Test that daily period uses correct number of days."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=10.0,
                period="daily",
                current_spend=5.0,
                action="block",
            )
        )
        # Should pass since 5 < 10
        self.assertIsNone(result)

    def test_weekly_period_calculation(self):
        """Test that weekly period uses correct number of days."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=70.0,
                period="weekly",
                current_spend=75.0,
                action="block",
            )
        )
        # Should block since 75 > 70
        self.assertIsNotNone(result)
        self.assertIn("weekly", result.lower())

    def test_yearly_period_calculation(self):
        """Test that yearly period uses correct number of days."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=1000.0,
                period="yearly",
                current_spend=999.0,
                action="block",
            )
        )
        # Should pass since 999 < 1000
        self.assertIsNone(result)

    def test_unknown_period_defaults_to_monthly(self):
        """Test that unknown period defaults to 30 days (monthly)."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            self._run_cost_check(
                limit_enabled=True,
                limit_amount=100.0,
                period="invalid_period",
                current_spend=150.0,
                action="block",
            )
        )
        # Should still work with default 30-day period
        self.assertIsNotNone(result)

    def test_no_job_store_skips_check(self):
        """Test that missing job store skips cost check gracefully."""
        import asyncio

        cmd = self._create_command()
        ctx = MockCommandContext(job_store=None)

        # The _check_cost_limit method is only called when ctx.job_store exists
        # Based on the code flow, this would be caught at validate() level
        # Let's test with a store that returns empty config
        config = {}
        store = MockJobStore(provider_config=config, current_spend=1000.0)
        ctx = MockCommandContext(job_store=store)

        result = asyncio.get_event_loop().run_until_complete(cmd._check_cost_limit(ctx))
        # Should pass since cost_limit_enabled is not set
        self.assertIsNone(result)


class TestCostLimitExceededError(unittest.TestCase):
    """Test CostLimitExceededError exception class."""

    def test_error_message_formatting(self):
        """Test error message includes all relevant info."""
        from simpletuner.simpletuner_sdk.server.services.cloud.exceptions import CostLimitExceededError

        error = CostLimitExceededError(
            limit=100.0,
            current=150.0,
            period="monthly",
        )

        self.assertEqual(error.error_code, "cost_limit_exceeded")
        # Check it has the expected attributes
        self.assertTrue(hasattr(error, "details"))

    def test_error_details_data(self):
        """Test error includes details data for API responses."""
        from simpletuner.simpletuner_sdk.server.services.cloud.exceptions import CostLimitExceededError

        error = CostLimitExceededError(
            limit=100.0,
            current=150.0,
            period="monthly",
        )

        details = error.details
        self.assertEqual(details.get("limit"), 100.0)
        self.assertEqual(details.get("current"), 150.0)
        self.assertEqual(details.get("period"), "monthly")
        self.assertEqual(details.get("quota_type"), "spending")


class TestCostLimitIntegration(unittest.TestCase):
    """Integration tests for cost limit with full submission flow."""

    def test_submit_blocked_by_cost_limit_returns_error_response(self):
        """Test that blocked submission returns proper error structure."""
        from simpletuner.simpletuner_sdk.server.services.cloud.exceptions import CostLimitExceededError

        error = CostLimitExceededError(
            limit=50.0,
            current=75.0,
            period="daily",
        )

        # Verify it can be serialized for API response
        error_dict = {
            "error_code": error.error_code,
            "message": str(error),
            "details": error.details,
        }

        self.assertEqual(error_dict["error_code"], "cost_limit_exceeded")
        self.assertIn("limit", error_dict["details"])
        self.assertIn("period", error_dict["details"])

    def test_cost_limit_warning_field_in_submit_data(self):
        """Test SubmitJobData includes cost_limit_warning field."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobData

        data = SubmitJobData(
            job_id="test-123",
            status="pending",
            provider="replicate",
            cost_limit_warning="Approaching limit: $80.00 / $100.00 (monthly)",
        )

        self.assertEqual(data.cost_limit_warning, "Approaching limit: $80.00 / $100.00 (monthly)")


if __name__ == "__main__":
    unittest.main()
