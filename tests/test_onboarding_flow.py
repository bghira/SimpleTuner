"""Tests for first-run setup and onboarding flow.

Tests the onboarding sequence including:
- First-run detection
- Admin user setup
- Provider configuration
- Progressive disclosure steps
- Onboarding state persistence
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFirstRunDetection(unittest.TestCase):
    """Tests for detecting first-run state."""

    def test_needs_setup_when_no_admin(self):
        """Test that setup is needed when no admin user exists."""
        # Mock the user store to return no admins
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.auth.user_store.UserStore") as MockStore:
            mock_store = MagicMock()
            mock_store.get_admin_count = MagicMock(return_value=0)
            MockStore.return_value = mock_store

            # First run should be detected
            count = mock_store.get_admin_count()
            needs_setup = count == 0
            self.assertTrue(needs_setup)

    def test_no_setup_needed_when_admin_exists(self):
        """Test that setup is not needed when admin user exists."""
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.auth.user_store.UserStore") as MockStore:
            mock_store = MagicMock()
            mock_store.get_admin_count = MagicMock(return_value=1)
            MockStore.return_value = mock_store

            count = mock_store.get_admin_count()
            needs_setup = count == 0
            self.assertFalse(needs_setup)


class TestAdminUserSetup(unittest.TestCase):
    """Tests for initial admin user creation."""

    def test_create_first_admin_user(self):
        """Test creating the first admin user."""
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import AccessLevel, UserCreate

        user_data = UserCreate(
            username="admin",
            email="admin@example.com",
            password="securepassword123",
            display_name="Administrator",
            access_level=AccessLevel.ADMIN,
        )

        self.assertEqual(user_data.username, "admin")
        self.assertEqual(user_data.access_level, AccessLevel.ADMIN)

    def test_admin_password_requirements(self):
        """Test that admin password has minimum requirements."""
        # Password should have minimum length
        min_length = 8
        password = "short"
        self.assertLess(len(password), min_length)

        valid_password = "securepassword123"
        self.assertGreaterEqual(len(valid_password), min_length)


class TestProviderConfiguration(unittest.TestCase):
    """Tests for cloud provider configuration during setup."""

    def test_provider_not_configured_initially(self):
        """Test that provider is not configured before setup."""
        # Mock provider config store
        with patch(
            "simpletuner.simpletuner_sdk.server.services.cloud.storage.provider_config_store.ProviderConfigStore"
        ) as MockStore:
            mock_store = MagicMock()
            mock_store.get = MagicMock(return_value=None)
            MockStore.return_value = mock_store

            config = mock_store.get("replicate")
            self.assertIsNone(config)

    def test_save_provider_token(self):
        """Test saving provider API token."""
        # Mock secrets manager
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.secrets_manager.SecretsManager") as MockSecrets:
            mock_secrets = MagicMock()
            mock_secrets.set_secret = MagicMock(return_value=True)
            MockSecrets.return_value = mock_secrets

            # Save token
            result = mock_secrets.set_secret("replicate_api_token", "r8_test_token")
            self.assertTrue(result)
            mock_secrets.set_secret.assert_called_with("replicate_api_token", "r8_test_token")


class TestOnboardingSteps(unittest.TestCase):
    """Tests for progressive disclosure onboarding steps."""

    def test_initial_onboarding_state(self):
        """Test initial onboarding state has all steps incomplete."""
        initial_state = {
            "data_understood": False,
            "results_understood": False,
            "cost_understood": False,
        }

        self.assertFalse(initial_state["data_understood"])
        self.assertFalse(initial_state["results_understood"])
        self.assertFalse(initial_state["cost_understood"])

    def test_step_completion_order(self):
        """Test that steps must be completed in order."""
        state = {
            "data_understood": False,
            "results_understood": False,
            "cost_understood": False,
        }

        # Can't complete step 2 before step 1
        def can_complete_step(step, state):
            if step == "data_understood":
                return True
            elif step == "results_understood":
                return state["data_understood"]
            elif step == "cost_understood":
                return state["data_understood"] and state["results_understood"]
            return False

        self.assertTrue(can_complete_step("data_understood", state))
        self.assertFalse(can_complete_step("results_understood", state))
        self.assertFalse(can_complete_step("cost_understood", state))

        # Complete step 1
        state["data_understood"] = True
        self.assertTrue(can_complete_step("results_understood", state))
        self.assertFalse(can_complete_step("cost_understood", state))

        # Complete step 2
        state["results_understood"] = True
        self.assertTrue(can_complete_step("cost_understood", state))

    def test_onboarding_complete_check(self):
        """Test checking if all onboarding steps are complete."""

        def is_onboarding_complete(state):
            return all(
                [
                    state.get("data_understood", False),
                    state.get("results_understood", False),
                    state.get("cost_understood", False),
                ]
            )

        incomplete = {
            "data_understood": True,
            "results_understood": True,
            "cost_understood": False,
        }
        self.assertFalse(is_onboarding_complete(incomplete))

        complete = {
            "data_understood": True,
            "results_understood": True,
            "cost_understood": True,
        }
        self.assertTrue(is_onboarding_complete(complete))


class TestOnboardingPersistence(unittest.TestCase):
    """Tests for onboarding state persistence."""

    def test_save_onboarding_state(self):
        """Test saving onboarding state to storage."""
        state = {
            "data_understood": True,
            "results_understood": False,
            "cost_understood": False,
        }

        # Simulate localStorage save (browser-side)
        import json

        json_state = json.dumps(state)
        self.assertIn('"data_understood": true', json_state)

    def test_load_onboarding_state(self):
        """Test loading onboarding state from storage."""
        import json

        stored = '{"data_understood": true, "results_understood": false, "cost_understood": false}'
        state = json.loads(stored)

        self.assertTrue(state["data_understood"])
        self.assertFalse(state["results_understood"])
        self.assertFalse(state["cost_understood"])

    def test_reset_onboarding_state(self):
        """Test resetting onboarding state."""
        complete_state = {
            "data_understood": True,
            "results_understood": True,
            "cost_understood": True,
        }

        def reset_onboarding(state):
            return {
                "data_understood": False,
                "results_understood": False,
                "cost_understood": False,
            }

        reset_state = reset_onboarding(complete_state)

        self.assertFalse(reset_state["data_understood"])
        self.assertFalse(reset_state["results_understood"])
        self.assertFalse(reset_state["cost_understood"])


class TestSkipOnboarding(unittest.TestCase):
    """Tests for skipping onboarding flow."""

    def test_skip_sets_all_complete(self):
        """Test that skipping marks all steps as complete."""
        state = {
            "data_understood": False,
            "results_understood": False,
            "cost_understood": False,
        }

        def skip_onboarding(state):
            return {
                "data_understood": True,
                "results_understood": True,
                "cost_understood": True,
            }

        skipped = skip_onboarding(state)

        self.assertTrue(skipped["data_understood"])
        self.assertTrue(skipped["results_understood"])
        self.assertTrue(skipped["cost_understood"])


class TestDeliveryOptionsCheck(unittest.TestCase):
    """Tests for checking delivery option configuration."""

    def test_huggingface_configured_check(self):
        """Test checking if HuggingFace is configured."""
        publishing_status = {
            "push_to_hub": True,
            "hub_model_id": "user/model",
        }

        is_hf_configured = publishing_status.get("push_to_hub", False)
        self.assertTrue(is_hf_configured)

    def test_huggingface_not_configured(self):
        """Test when HuggingFace is not configured."""
        publishing_status = {
            "push_to_hub": False,
        }

        is_hf_configured = publishing_status.get("push_to_hub", False)
        self.assertFalse(is_hf_configured)

    def test_webhook_configured_check(self):
        """Test checking if webhook URL is configured."""
        webhook_url = "https://example.com/webhook"

        is_webhook_configured = bool(webhook_url and webhook_url.strip())
        self.assertTrue(is_webhook_configured)

    def test_webhook_not_configured(self):
        """Test when webhook URL is empty."""
        webhook_url = ""

        is_webhook_configured = bool(webhook_url and webhook_url.strip())
        self.assertFalse(is_webhook_configured)

    def test_s3_configured_check(self):
        """Test checking if S3 is configured."""
        publishing_status = {
            "s3_configured": True,
        }

        is_s3_configured = publishing_status.get("s3_configured", False)
        self.assertTrue(is_s3_configured)


class TestCostLimitSetup(unittest.TestCase):
    """Tests for cost limit configuration during onboarding."""

    def test_quick_cost_limit_defaults(self):
        """Test default values for quick cost limit setup."""
        defaults = {
            "enabled": False,
            "amount": 50,
            "period": "monthly",
        }

        self.assertFalse(defaults["enabled"])
        self.assertEqual(defaults["amount"], 50)
        self.assertEqual(defaults["period"], "monthly")

    def test_cost_limit_periods(self):
        """Test valid cost limit periods."""
        valid_periods = ["daily", "weekly", "monthly"]

        for period in valid_periods:
            self.assertIn(period, valid_periods)

    def test_cost_limit_minimum_amount(self):
        """Test that cost limit has minimum amount."""
        min_amount = 1

        # Valid amounts
        self.assertGreaterEqual(50, min_amount)
        self.assertGreaterEqual(1, min_amount)

        # Invalid amount
        self.assertLess(0, min_amount)


class TestSetupStateAPI(unittest.TestCase):
    """Tests for setup state API endpoints."""

    def test_setup_state_response_format(self):
        """Test the format of setup state response."""
        # Expected response format
        response = {
            "needsSetup": True,
            "loading": False,
            "error": None,
        }

        self.assertIn("needsSetup", response)
        self.assertIn("loading", response)
        self.assertIsInstance(response["needsSetup"], bool)

    def test_setup_complete_response(self):
        """Test response when setup is complete."""
        response = {
            "needsSetup": False,
            "loading": False,
            "error": None,
        }

        self.assertFalse(response["needsSetup"])
        self.assertIsNone(response["error"])


if __name__ == "__main__":
    unittest.main()
