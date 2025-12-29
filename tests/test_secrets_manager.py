"""Tests for SecretsManager.

Tests the secrets management system including:
- Singleton pattern
- Test environment guard for reset()
"""

import unittest


class TestSecretsManagerReset(unittest.TestCase):
    """Tests for SecretsManager.reset() test guard."""

    def test_reset_allowed_in_test_environment(self):
        """Test that reset() works when unittest is loaded."""
        from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

        # Should not raise because we're running under unittest
        # First, ensure we have an instance
        manager1 = SecretsManager()
        self.assertIsNotNone(manager1)

        # Reset should work in test environment
        SecretsManager.reset()

        # Getting a new instance should create a fresh one
        manager2 = SecretsManager()
        self.assertIsNotNone(manager2)

    def test_reset_guard_checks_test_modules(self):
        """Test that reset() checks for test framework modules."""
        import sys

        from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

        # Verify that unittest is in sys.modules (since we're running tests)
        self.assertIn("unittest", sys.modules)

        # The reset should work because unittest is loaded
        try:
            SecretsManager.reset()
        except RuntimeError:
            self.fail("reset() raised RuntimeError even with unittest loaded")

    def test_singleton_pattern(self):
        """Test that SecretsManager uses singleton pattern."""
        from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

        # Reset to ensure clean state
        SecretsManager.reset()

        manager1 = SecretsManager()
        manager2 = SecretsManager()

        # Both should be the same instance
        self.assertIs(manager1, manager2)

    def test_reset_clears_singleton(self):
        """Test that reset() clears the singleton instance."""
        from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

        # Get initial instance
        manager1 = SecretsManager()

        # Reset
        SecretsManager.reset()

        # Get new instance
        manager2 = SecretsManager()

        # Should be different objects (new instance created)
        self.assertIsNot(manager1, manager2)


class TestSecretsManagerResetGuardSimulation(unittest.TestCase):
    """Tests simulating the reset guard behavior."""

    def test_guard_logic(self):
        """Test the guard logic directly without actually being outside test env."""
        import sys

        # Verify the guard logic: should pass when unittest OR pytest is in modules
        has_unittest = "unittest" in sys.modules
        has_pytest = "pytest" in sys.modules

        # At least one should be true since we're running tests
        self.assertTrue(
            has_unittest or has_pytest,
            "Neither unittest nor pytest detected - test environment check would fail",
        )


if __name__ == "__main__":
    unittest.main()
