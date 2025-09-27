"""End-to-end tests for WebUI critical user flows."""

import pytest
from selenium.webdriver.common.by import By

from tests.pages.trainer_page import BasicConfigTab, DatasetsTab, ModelConfigTab, TrainerPage, TrainingConfigTab


class TestBasicConfigurationFlow:
    """Test basic configuration and save flow."""

    @pytest.mark.e2e
    def test_save_basic_configuration(self, driver, setup_test_environment):
        """Test saving basic configuration updates WebUI defaults."""
        # Initialize page objects
        trainer_page = TrainerPage(driver)
        basic_tab = BasicConfigTab(driver)

        # Navigate to trainer
        trainer_page.navigate_to_trainer()

        # Fill in basic configuration
        basic_tab.set_configs_dir("/test/configs")
        basic_tab.set_model_name("test-model")
        basic_tab.set_output_dir("/test/output")
        basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")

        # Save configuration
        basic_tab.save_changes()

        # Check for success toast
        toast_message = trainer_page.get_toast_message()
        assert toast_message is not None
        assert "saved" in toast_message.lower() or "success" in toast_message.lower()

        # Verify values are persisted
        driver.refresh()
        trainer_page.wait_for_htmx()

        assert basic_tab.get_configs_dir() == "/test/configs"
        assert basic_tab.get_output_dir() == "/test/output"

    @pytest.mark.e2e
    def test_configuration_validation(self, driver):
        """Test configuration validation on incomplete form."""
        trainer_page = TrainerPage(driver)
        basic_tab = BasicConfigTab(driver)

        trainer_page.navigate_to_trainer()

        # Clear required fields
        basic_tab.set_model_name("")
        basic_tab.set_output_dir("")

        # Try to start training
        trainer_page.start_training()

        # Should show invalid configuration
        assert not trainer_page.is_config_valid()
        toast_message = trainer_page.get_toast_message()
        assert toast_message is not None
        assert "invalid" in toast_message.lower() or "required" in toast_message.lower()

    @pytest.mark.e2e
    def test_form_fields_maintain_independent_values(self, driver):
        """Regression test for form field synchronization bug.

        This test ensures that each form field maintains its own value
        independently, preventing the issue where all fields would sync
        to the same value due to shared Alpine.js x-model bindings.
        """
        trainer_page = TrainerPage(driver)
        basic_tab = BasicConfigTab(driver)

        trainer_page.navigate_to_trainer()

        # Clear any existing values first
        basic_tab.set_configs_dir("")
        basic_tab.set_model_name("")
        basic_tab.set_output_dir("")
        basic_tab.set_base_model("")

        # Type different values in each field
        test_values = {
            "configs_dir": "/unique/configs/path",
            "model_name": "unique-model-name",
            "output_dir": "/unique/output/path",
            "base_model": "unique-base-model",
        }

        basic_tab.set_configs_dir(test_values["configs_dir"])
        basic_tab.set_model_name(test_values["model_name"])
        basic_tab.set_output_dir(test_values["output_dir"])
        basic_tab.set_base_model(test_values["base_model"])

        # Verify each field has retained its unique value
        assert basic_tab.get_configs_dir() == test_values["configs_dir"]
        assert basic_tab.get_model_name() == test_values["model_name"]
        assert basic_tab.get_output_dir() == test_values["output_dir"]
        assert basic_tab.get_base_model() == test_values["base_model"]

        # Also verify by typing in reverse order
        basic_tab.set_base_model("reverse-base-model")
        basic_tab.set_output_dir("/reverse/output")
        basic_tab.set_model_name("reverse-model")
        basic_tab.set_configs_dir("/reverse/configs")

        # All fields should still have independent values
        assert basic_tab.get_configs_dir() == "/reverse/configs"
        assert basic_tab.get_model_name() == "reverse-model"
        assert basic_tab.get_output_dir() == "/reverse/output"
        assert basic_tab.get_base_model() == "reverse-base-model"


class TestTrainingWorkflow:
    """Test complete training workflow."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_configure_and_start_training(self, driver, setup_test_environment):
        """Test configuring all settings and starting training."""
        # Page objects
        trainer_page = TrainerPage(driver)
        basic_tab = BasicConfigTab(driver)
        model_tab = ModelConfigTab(driver)
        training_tab = TrainingConfigTab(driver)

        # Navigate to trainer
        trainer_page.navigate_to_trainer()

        # Configure basic settings
        basic_tab.set_model_name("flux-test-model")
        basic_tab.set_output_dir("/tmp/test-output")
        basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")
        basic_tab.save_changes()

        # Configure model settings
        trainer_page.switch_to_model_tab()
        model_tab.select_model_family("flux")
        model_tab.set_lora_rank("16")
        model_tab.set_lora_alpha("32")

        # Configure training settings
        trainer_page.switch_to_training_tab()
        training_tab.set_learning_rate("0.0001")
        training_tab.set_batch_size("1")
        training_tab.set_num_epochs("10")
        training_tab.select_mixed_precision("bf16")

        # Save all configurations
        trainer_page.save_configuration()

        # Verify configuration is valid
        assert trainer_page.is_config_valid()

        # Start training
        trainer_page.start_training()

        # Verify training status changes
        assert trainer_page.get_training_status() in ["running", "idle"]

        # Stop training (cleanup)
        if trainer_page.get_training_status() == "running":
            trainer_page.stop_training()


class TestDatasetManagement:
    """Test dataset management functionality."""

    @pytest.mark.e2e
    def test_add_and_remove_dataset(self, driver):
        """Test adding and removing datasets."""
        trainer_page = TrainerPage(driver)
        datasets_tab = DatasetsTab(driver)

        trainer_page.navigate_to_trainer()
        trainer_page.switch_to_datasets_tab()

        # Get initial dataset count
        initial_count = datasets_tab.get_dataset_count()

        # Add a new dataset
        datasets_tab.add_dataset()
        datasets_tab.fill_dataset_modal("Test Dataset", "/test/dataset/path")
        datasets_tab.save_dataset_modal()

        # Verify dataset was added
        new_count = datasets_tab.get_dataset_count()
        assert new_count == initial_count + 1

        # Delete the dataset
        datasets_tab.delete_dataset(0)

        # Verify dataset was removed
        final_count = datasets_tab.get_dataset_count()
        assert final_count == initial_count


class TestTabNavigation:
    """Test tab navigation functionality."""

    @pytest.mark.e2e
    def test_all_tabs_load(self, driver):
        """Test that all tabs load without errors."""
        trainer_page = TrainerPage(driver)

        trainer_page.navigate_to_trainer()

        # Test each tab
        tabs_to_test = [
            ("basic", trainer_page.switch_to_basic_tab),
            ("model", trainer_page.switch_to_model_tab),
            ("training", trainer_page.switch_to_training_tab),
            ("advanced", trainer_page.switch_to_advanced_tab),
            ("datasets", trainer_page.switch_to_datasets_tab),
            ("environments", trainer_page.switch_to_environments_tab),
        ]

        for tab_name, switch_method in tabs_to_test:
            switch_method()
            # Verify tab content is visible
            assert driver.find_element(By.ID, f"tab-{tab_name}").is_displayed()


class TestToastNotifications:
    """Test toast notification behavior."""

    @pytest.mark.e2e
    def test_toast_positioning(self, driver):
        """Test that toast appears at bottom of header."""
        trainer_page = TrainerPage(driver)
        basic_tab = BasicConfigTab(driver)

        trainer_page.navigate_to_trainer()

        # Trigger a toast by saving
        basic_tab.save_changes()

        # Get toast position
        toast_container = driver.find_element(By.CSS_SELECTOR, ".toast-container")
        toast_top = driver.execute_script("return arguments[0].offsetTop;", toast_container)

        # Toast should be at 60px (bottom of header)
        assert toast_top == 60

        # Dismiss toast
        trainer_page.dismiss_toast()


class TestResponsiveDesign:
    """Test responsive design functionality."""

    @pytest.mark.e2e
    @pytest.mark.chrome_only
    def test_mobile_viewport(self, driver):
        """Test UI adapts to mobile viewport."""
        trainer_page = TrainerPage(driver)

        # Set mobile viewport
        driver.set_window_size(375, 812)  # iPhone X size

        trainer_page.navigate_to_trainer()

        # Verify mobile menu is visible (if implemented)
        # This is a placeholder - actual implementation would check for mobile-specific UI

        # Reset viewport
        driver.set_window_size(1920, 1080)


class TestOnboardingFlow:
    """Test onboarding flow for new users."""

    @pytest.mark.e2e
    def test_first_time_user_onboarding(self, driver, mock_webui_state):
        """Test onboarding overlay appears for new users."""
        # Clear any existing state
        defaults_file = mock_webui_state / "defaults.json"
        if defaults_file.exists():
            defaults_file.unlink()

        trainer_page = TrainerPage(driver)
        trainer_page.navigate_to_trainer()

        # Check for onboarding overlay
        onboarding_overlay = driver.find_element(By.CSS_SELECTOR, '[x-show="onboardingRequired"]')
        assert onboarding_overlay.is_displayed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
