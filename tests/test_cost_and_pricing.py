"""Unit tests for cost limits and hardware pricing."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob


class TestHardwarePricingConfig(unittest.IsolatedAsyncioTestCase):
    """Test cases for configurable hardware pricing."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        # Clear hardware cache
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import clear_hardware_info_cache

        clear_hardware_info_cache()

        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create job store with async initialization
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        await self.store.close()
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore singletons
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clear hardware cache
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import clear_hardware_info_cache

        clear_hardware_info_cache()

    async def test_default_hardware_info(self) -> None:
        """Test that default hardware info is used when not configured."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            DEFAULT_HARDWARE_INFO,
            get_hardware_info_async,
        )

        hardware = await get_hardware_info_async(self.store)
        self.assertEqual(hardware, DEFAULT_HARDWARE_INFO)
        self.assertIn("gpu-l40s", hardware)
        self.assertIn("gpu-a100-large", hardware)

    async def test_configured_hardware_info(self) -> None:
        """Test that configured hardware info overrides defaults."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            clear_hardware_info_cache,
            get_hardware_info_async,
        )

        # Configure custom hardware info
        custom_hardware = {
            "gpu-h100": {"name": "H100 (80GB)", "cost_per_second": 0.002},
            "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.001},
        }
        await self.store.save_provider_config("replicate", {"hardware_info": custom_hardware})

        # Clear cache to pick up new config
        clear_hardware_info_cache()

        hardware = await get_hardware_info_async(self.store)
        self.assertEqual(hardware, custom_hardware)
        self.assertIn("gpu-h100", hardware)

    async def test_default_hardware_cost_per_hour(self) -> None:
        """Test getting default hardware cost per hour."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import get_default_hardware_cost_per_hour

        cost = await get_default_hardware_cost_per_hour(self.store)
        # L40S at $0.000975/sec = $3.51/hr
        self.assertAlmostEqual(cost, 3.51, places=2)

    async def test_configured_hardware_cost_per_hour(self) -> None:
        """Test that configured hardware cost is used."""
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
            clear_hardware_info_cache,
            get_default_hardware_cost_per_hour,
        )

        # Configure custom L40S price
        custom_hardware = {
            "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.001},  # $3.60/hr
        }
        await self.store.save_provider_config("replicate", {"hardware_info": custom_hardware})
        clear_hardware_info_cache()

        cost = await get_default_hardware_cost_per_hour(self.store)
        self.assertAlmostEqual(cost, 3.60, places=2)


class TestCostLimitCheck(unittest.IsolatedAsyncioTestCase):
    """Test cases for cost limit checking via SubmitJobCommand."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        from unittest.mock import AsyncMock, MagicMock

        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create job store with async initialization
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

        # Create a mock CommandContext
        self.ctx = MagicMock()
        self.ctx.job_store = self.store
        self.ctx.user_id = "test-user"

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        await self.store.close()
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore singletons
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_cost_limit_disabled(self) -> None:
        """Test that disabled cost limit returns None (no error)."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobCommand

        cmd = SubmitJobCommand(
            config={"model_type": "test"},
            dataloader_config=[{"id": "test"}],
            provider="replicate",
            config_name="test-config",
        )
        result = await cmd._check_cost_limit(self.ctx)
        # When disabled, returns None (no block)
        self.assertIsNone(result)

    async def test_cost_limit_exceeded_blocks(self) -> None:
        """Test that exceeded cost limit with block action returns error."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobCommand

        # Configure cost limit
        await self.store.save_provider_config(
            "replicate",
            {
                "cost_limit_enabled": True,
                "cost_limit_amount": 10.0,
                "cost_limit_period": "daily",
                "cost_limit_action": "block",
            },
        )

        # Add job with cost that exceeds limit
        job = UnifiedJob(
            job_id="expensive-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="expensive-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            cost_usd=15.0,
        )
        await self.store.add_job(job)

        cmd = SubmitJobCommand(
            config={"model_type": "test"},
            dataloader_config=[{"id": "test"}],
            provider="replicate",
            config_name="test-config",
        )
        result = await cmd._check_cost_limit(self.ctx)

        # Should return error message when exceeded and action=block
        self.assertIsNotNone(result)
        self.assertIn("Cost limit exceeded", result)

    async def test_cost_limit_exceeded_warn_no_block(self) -> None:
        """Test that exceeded cost limit with warn action doesn't block."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobCommand

        # Configure cost limit with warn action
        await self.store.save_provider_config(
            "replicate",
            {
                "cost_limit_enabled": True,
                "cost_limit_amount": 10.0,
                "cost_limit_period": "daily",
                "cost_limit_action": "warn",
            },
        )

        # Add job with cost that exceeds limit
        job = UnifiedJob(
            job_id="expensive-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="expensive-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            cost_usd=15.0,
        )
        await self.store.add_job(job)

        cmd = SubmitJobCommand(
            config={"model_type": "test"},
            dataloader_config=[{"id": "test"}],
            provider="replicate",
            config_name="test-config",
        )
        result = await cmd._check_cost_limit(self.ctx)

        # Should not block (returns None) but sets warning
        self.assertIsNone(result)
        self.assertIsNotNone(cmd._cost_limit_warning)
        self.assertIn("exceeded", cmd._cost_limit_warning.lower())

    async def test_cost_limit_warning_threshold(self) -> None:
        """Test warning when approaching limit (80% threshold)."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import SubmitJobCommand

        # Configure cost limit
        await self.store.save_provider_config(
            "replicate",
            {
                "cost_limit_enabled": True,
                "cost_limit_amount": 10.0,
                "cost_limit_period": "daily",
                "cost_limit_action": "block",
            },
        )

        # Add job with cost at 85% of limit (should trigger warning)
        job = UnifiedJob(
            job_id="warning-job",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.COMPLETED.value,
            config_name="warning-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            cost_usd=8.5,
        )
        await self.store.add_job(job)

        cmd = SubmitJobCommand(
            config={"model_type": "test"},
            dataloader_config=[{"id": "test"}],
            provider="replicate",
            config_name="test-config",
        )
        result = await cmd._check_cost_limit(self.ctx)

        # Should not block (under limit) but set warning (over 80%)
        self.assertIsNone(result)
        self.assertIsNotNone(cmd._cost_limit_warning)
        self.assertIn("Approaching limit", cmd._cost_limit_warning)


class TestGPUPricingInterface(unittest.TestCase):
    """Test cases for the GPUPricing interface."""

    def test_replicate_default_pricing(self) -> None:
        """Test Replicate default hardware options."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_replicate_pricing

        pricing = get_replicate_pricing()
        self.assertEqual(pricing.provider_name, "replicate")

        options = pricing.get_hardware_options()
        self.assertGreater(len(options), 0)

        # Check L40S exists
        l40s = pricing.get_hardware_by_id("gpu-l40s")
        self.assertIsNotNone(l40s)
        self.assertEqual(l40s.name, "L40S (48GB)")
        self.assertGreater(l40s.cost_per_second, 0)

    def test_hardware_option_cost_calculations(self) -> None:
        """Test hardware cost calculations."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import HardwareOption

        hw = HardwareOption(
            id="test-gpu",
            name="Test GPU",
            cost_per_second=0.001,  # $0.001/second
        )

        # Test cost per minute
        self.assertAlmostEqual(hw.cost_per_minute, 0.06, places=4)

        # Test cost per hour
        self.assertAlmostEqual(hw.cost_per_hour, 3.60, places=2)

    def test_configurable_pricing_override(self) -> None:
        """Test that pricing can be overridden."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_replicate_pricing

        pricing = get_replicate_pricing()

        # Apply custom config
        pricing.configure(
            {
                "gpu-custom": {
                    "name": "Custom GPU",
                    "cost_per_second": 0.002,
                    "memory_gb": 64,
                }
            }
        )

        # Should now have custom hardware
        custom = pricing.get_hardware_by_id("gpu-custom")
        self.assertIsNotNone(custom)
        self.assertEqual(custom.name, "Custom GPU")
        self.assertEqual(custom.memory_gb, 64)

        # L40S should no longer exist (replaced by custom config)
        l40s = pricing.get_hardware_by_id("gpu-l40s")
        self.assertIsNone(l40s)

        # Clear config to restore defaults
        pricing.clear_configuration()
        l40s = pricing.get_hardware_by_id("gpu-l40s")
        self.assertIsNotNone(l40s)

    def test_calculate_cost(self) -> None:
        """Test job cost calculation."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_replicate_pricing

        pricing = get_replicate_pricing()

        # Calculate cost for 1 hour on L40S
        cost = pricing.calculate_cost("gpu-l40s", 3600)
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)

        # Unknown hardware returns None
        cost = pricing.calculate_cost("unknown-gpu", 3600)
        self.assertIsNone(cost)

    def test_estimate_cost(self) -> None:
        """Test job cost estimation."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_replicate_pricing

        pricing = get_replicate_pricing()

        # Estimate cost for 60 minutes on L40S
        estimate = pricing.estimate_cost("gpu-l40s", 60)
        self.assertIsNotNone(estimate)
        self.assertIn("estimated", estimate)
        self.assertIn("min", estimate)
        self.assertIn("max", estimate)
        self.assertIn("currency", estimate)

        # min < estimated < max
        self.assertLess(estimate["min"], estimate["estimated"])
        self.assertLess(estimate["estimated"], estimate["max"])

    def test_simpletuner_io_pricing_provider(self) -> None:
        """Test SimpleTuner.io pricing provider (coming soon)."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_simpletuner_io_pricing

        pricing = get_simpletuner_io_pricing()
        self.assertEqual(pricing.provider_name, "simpletuner_io")

        options = pricing.get_hardware_options()
        self.assertGreater(len(options), 0)

        # Coming soon provider should have unavailable hardware
        default = pricing.get_default_hardware()
        self.assertFalse(default.available)

    def test_pricing_registry(self) -> None:
        """Test pricing provider registry."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider, list_pricing_providers

        providers = list_pricing_providers()
        self.assertIn("replicate", providers)
        self.assertIn("simpletuner_io", providers)

        replicate = get_pricing_provider("replicate")
        self.assertIsNotNone(replicate)
        self.assertEqual(replicate.provider_name, "replicate")

        simpletuner_io = get_pricing_provider("simpletuner_io")
        self.assertIsNotNone(simpletuner_io)
        self.assertEqual(simpletuner_io.provider_name, "simpletuner_io")

        unknown = get_pricing_provider("unknown")
        self.assertIsNone(unknown)


if __name__ == "__main__":
    unittest.main()
