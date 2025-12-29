"""Tests for cost estimation and GPU pricing.

Tests the pricing module accuracy:
- HardwareOption calculations
- GPUPricingProvider interface
- ConfigurablePricingProvider with overrides
- Cost calculation formulas
- Estimation accuracy bounds
"""

import unittest
from typing import List


class TestHardwareOption(unittest.TestCase):
    """Test HardwareOption dataclass."""

    def test_cost_per_hour_calculation(self):
        """Test cost_per_hour is calculated correctly from cost_per_second."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import HardwareOption

        hw = HardwareOption(
            id="test-gpu",
            name="Test GPU",
            cost_per_second=0.001,  # $0.001 per second
        )

        # $0.001/s * 3600s/h = $3.60/h
        self.assertAlmostEqual(hw.cost_per_hour, 3.60, places=6)

    def test_cost_per_minute_calculation(self):
        """Test cost_per_minute is calculated correctly."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import HardwareOption

        hw = HardwareOption(
            id="test-gpu",
            name="Test GPU",
            cost_per_second=0.001,
        )

        # $0.001/s * 60s/m = $0.06/m
        self.assertAlmostEqual(hw.cost_per_minute, 0.06, places=6)

    def test_replicate_l40s_pricing(self):
        """Test Replicate L40S pricing matches known values."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import HardwareOption

        # Replicate L40S pricing as of implementation
        hw = HardwareOption(
            id="gpu-l40s",
            name="L40S (48GB)",
            cost_per_second=0.000975,
            memory_gb=48,
        )

        # ~$3.51/hour
        self.assertAlmostEqual(hw.cost_per_hour, 3.51, places=2)

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import HardwareOption

        hw = HardwareOption(
            id="test-gpu",
            name="Test GPU",
            cost_per_second=0.001,
            memory_gb=48,
            gpu_count=2,
            available=True,
        )

        d = hw.to_dict()

        self.assertEqual(d["id"], "test-gpu")
        self.assertEqual(d["name"], "Test GPU")
        self.assertEqual(d["cost_per_second"], 0.001)
        self.assertEqual(d["cost_per_hour"], 3.6)
        self.assertEqual(d["memory_gb"], 48)
        self.assertEqual(d["gpu_count"], 2)
        self.assertTrue(d["available"])


class TestConfigurablePricingProvider(unittest.TestCase):
    """Test ConfigurablePricingProvider."""

    def test_default_hardware_options(self):
        """Test getting default hardware options."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
                "gpu-b": {"name": "GPU B", "cost_per_second": 0.002},
            },
            default_hardware_id="gpu-a",
        )

        options = provider.get_hardware_options()
        self.assertEqual(len(options), 2)

        ids = [o.id for o in options]
        self.assertIn("gpu-a", ids)
        self.assertIn("gpu-b", ids)

    def test_get_default_hardware(self):
        """Test getting default hardware option."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
                "gpu-b": {"name": "GPU B", "cost_per_second": 0.002},
            },
            default_hardware_id="gpu-b",
        )

        default = provider.get_default_hardware()
        self.assertEqual(default.id, "gpu-b")
        self.assertEqual(default.cost_per_second, 0.002)

    def test_get_hardware_by_id(self):
        """Test getting hardware by ID."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        hw = provider.get_hardware_by_id("gpu-a")
        self.assertIsNotNone(hw)
        self.assertEqual(hw.name, "GPU A")

        # Non-existent
        hw = provider.get_hardware_by_id("nonexistent")
        self.assertIsNone(hw)

    def test_configuration_override(self):
        """Test applying configuration overrides."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        # Override with different pricing
        provider.configure(
            {
                "gpu-a": {"name": "GPU A (Custom)", "cost_per_second": 0.002},
                "gpu-c": {"name": "GPU C", "cost_per_second": 0.003},
            }
        )

        options = provider.get_hardware_options()
        self.assertEqual(len(options), 2)

        hw_a = provider.get_hardware_by_id("gpu-a")
        self.assertEqual(hw_a.name, "GPU A (Custom)")
        self.assertEqual(hw_a.cost_per_second, 0.002)

        hw_c = provider.get_hardware_by_id("gpu-c")
        self.assertIsNotNone(hw_c)
        self.assertEqual(hw_c.cost_per_second, 0.003)

    def test_clear_configuration(self):
        """Test clearing configuration reverts to defaults."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        # Override
        provider.configure(
            {
                "gpu-a": {"name": "GPU A (Custom)", "cost_per_second": 0.999},
            }
        )

        # Clear
        provider.clear_configuration()

        hw = provider.get_hardware_by_id("gpu-a")
        self.assertEqual(hw.name, "GPU A")
        self.assertEqual(hw.cost_per_second, 0.001)


class TestCostCalculation(unittest.TestCase):
    """Test cost calculation methods."""

    def test_calculate_cost_exact(self):
        """Test exact cost calculation for a job."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        # 1000 seconds at $0.001/s = $1.00
        cost = provider.calculate_cost("gpu-a", 1000)
        self.assertAlmostEqual(cost, 1.00, places=6)

    def test_calculate_cost_one_hour(self):
        """Test cost calculation for one hour."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-l40s": {
                    "name": "L40S",
                    "cost_per_second": 0.000975,
                },
            },
            default_hardware_id="gpu-l40s",
        )

        # 1 hour = 3600 seconds
        cost = provider.calculate_cost("gpu-l40s", 3600)
        self.assertAlmostEqual(cost, 3.51, places=2)

    def test_calculate_cost_unknown_hardware(self):
        """Test calculate_cost returns None for unknown hardware."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={},
            default_hardware_id="gpu-a",
        )

        cost = provider.calculate_cost("nonexistent", 1000)
        self.assertIsNone(cost)

    def test_estimate_cost_range(self):
        """Test cost estimation returns a range."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        # Estimate for 60 minutes
        # $0.001/s * 60s/m = $0.06/m * 60m = $3.60 estimated
        estimate = provider.estimate_cost("gpu-a", 60)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate["estimated"], 3.60, places=2)

        # Min should be 20% less
        self.assertAlmostEqual(estimate["min"], 3.60 * 0.8, places=2)

        # Max should be 50% more
        self.assertAlmostEqual(estimate["max"], 3.60 * 1.5, places=2)

        self.assertEqual(estimate["currency"], "USD")

    def test_estimate_cost_bounds_accuracy(self):
        """Test estimation bounds are reasonable."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-l40s": {
                    "name": "L40S",
                    "cost_per_second": 0.000975,
                },
            },
            default_hardware_id="gpu-l40s",
        )

        # Typical training job: 30 minutes
        estimate = provider.estimate_cost("gpu-l40s", 30)

        # Estimated cost: 30 min * $0.000975 * 60 = $1.755
        self.assertAlmostEqual(estimate["estimated"], 1.755, places=2)

        # Bounds check
        self.assertLess(estimate["min"], estimate["estimated"])
        self.assertGreater(estimate["max"], estimate["estimated"])

        # Min should be reasonable (not too low)
        self.assertGreater(estimate["min"], 0)

    def test_estimate_short_job(self):
        """Test estimation for very short job."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.001},
            },
            default_hardware_id="gpu-a",
        )

        # 1 minute job
        estimate = provider.estimate_cost("gpu-a", 1)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate["estimated"], 0.06, places=4)

    def test_estimate_long_job(self):
        """Test estimation for long job (24 hours)."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-l40s": {
                    "name": "L40S",
                    "cost_per_second": 0.000975,
                },
            },
            default_hardware_id="gpu-l40s",
        )

        # 24 hours = 1440 minutes
        estimate = provider.estimate_cost("gpu-l40s", 1440)

        # Expected: 1440 * 60 * 0.000975 = $84.24
        self.assertAlmostEqual(estimate["estimated"], 84.24, places=2)


class TestReplicatePricing(unittest.TestCase):
    """Test Replicate-specific pricing configuration."""

    def test_replicate_provider_exists(self):
        """Test Replicate pricing provider is registered."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider, list_pricing_providers

        providers = list_pricing_providers()
        self.assertIn("replicate", providers)

        provider = get_pricing_provider("replicate")
        self.assertIsNotNone(provider)

    def test_replicate_l40s_pricing(self):
        """Test Replicate L40S pricing."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider

        provider = get_pricing_provider("replicate")
        hw = provider.get_hardware_by_id("gpu-l40s")

        self.assertIsNotNone(hw)
        self.assertEqual(hw.memory_gb, 48)
        # L40S costs ~$0.000975/s = ~$3.51/hour
        self.assertAlmostEqual(hw.cost_per_hour, 3.51, places=2)

    def test_replicate_a100_pricing(self):
        """Test Replicate A100 pricing."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider

        provider = get_pricing_provider("replicate")
        hw = provider.get_hardware_by_id("gpu-a100-large")

        self.assertIsNotNone(hw)
        self.assertEqual(hw.memory_gb, 80)
        # A100 costs ~$0.0014/s = ~$5.04/hour
        self.assertAlmostEqual(hw.cost_per_hour, 5.04, places=2)

    def test_replicate_default_hardware(self):
        """Test Replicate default hardware is L40S."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider

        provider = get_pricing_provider("replicate")
        default = provider.get_default_hardware()

        self.assertEqual(default.id, "gpu-l40s")


class TestPricingProviderRegistry(unittest.TestCase):
    """Test pricing provider registry functions."""

    def test_list_providers(self):
        """Test listing registered providers."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import list_pricing_providers

        providers = list_pricing_providers()
        self.assertIsInstance(providers, list)
        self.assertIn("replicate", providers)
        self.assertIn("simpletuner_io", providers)

    def test_get_nonexistent_provider(self):
        """Test getting non-existent provider returns None."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import get_pricing_provider

        provider = get_pricing_provider("nonexistent_provider")
        self.assertIsNone(provider)

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import (
            ConfigurablePricingProvider,
            get_pricing_provider,
            register_pricing_provider,
        )

        # Create custom provider
        custom = ConfigurablePricingProvider(
            provider_name="custom_test",
            default_hardware={
                "gpu-custom": {"name": "Custom GPU", "cost_per_second": 0.005},
            },
            default_hardware_id="gpu-custom",
        )

        # Register it
        register_pricing_provider(custom)

        # Retrieve it
        retrieved = get_pricing_provider("custom_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.provider_name, "custom_test")


class TestCostFormatting(unittest.TestCase):
    """Test cost formatting and rounding."""

    def test_estimate_rounding(self):
        """Test estimate values are rounded appropriately."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "gpu-a": {"name": "GPU A", "cost_per_second": 0.0001234567},
            },
            default_hardware_id="gpu-a",
        )

        estimate = provider.estimate_cost("gpu-a", 10)

        # Values should be rounded to 4 decimal places
        self.assertEqual(len(str(estimate["estimated"]).split(".")[-1]), 4)

    def test_zero_cost_hardware(self):
        """Test handling of zero-cost hardware."""
        from simpletuner.simpletuner_sdk.server.services.cloud.pricing import ConfigurablePricingProvider

        provider = ConfigurablePricingProvider(
            provider_name="test",
            default_hardware={
                "free-tier": {"name": "Free Tier", "cost_per_second": 0.0},
            },
            default_hardware_id="free-tier",
        )

        cost = provider.calculate_cost("free-tier", 1000)
        self.assertEqual(cost, 0.0)

        estimate = provider.estimate_cost("free-tier", 60)
        self.assertEqual(estimate["estimated"], 0.0)
        self.assertEqual(estimate["min"], 0.0)
        self.assertEqual(estimate["max"], 0.0)


if __name__ == "__main__":
    unittest.main()
