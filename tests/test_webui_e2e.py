"""End-to-end tests for WebUI critical user flows (unittest edition)."""

from __future__ import annotations

import unittest

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tests.pages.trainer_page import BasicConfigTab, DatasetsTab, ModelConfigTab, TrainerPage, TrainingConfigTab
from tests.webui_test_base import WebUITestCase


class _TrainerPageMixin:
    def _trainer_page(self, driver):
        return TrainerPage(driver, base_url=self.base_url)


class BasicConfigurationFlowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test basic configuration and save flow."""

    def test_save_basic_configuration(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            basic_tab.set_configs_dir(str(self.config_dir))
            basic_tab.set_model_name("test-model")
            basic_tab.set_output_dir("/test/output")
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")

            basic_tab.save_changes()
            toast_message = trainer_page.get_toast_message()
            print('toast after save:', toast_message)
            self.assertIsNotNone(toast_message)
            self.assertTrue("saved" in toast_message.lower() or "success" in toast_message.lower())

            driver.refresh()
            trainer_page.wait_for_htmx()

            self.assertEqual(basic_tab.get_configs_dir(), str(self.config_dir))
            self.assertEqual(basic_tab.get_output_dir(), "/test/output")

        self.for_each_browser("test_save_basic_configuration", scenario)

    def test_configuration_validation(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")
            basic_tab.set_model_name("")
            basic_tab.set_output_dir("")
            trainer_page.start_training()

            self.assertFalse(trainer_page.is_config_valid())
            toast_message = trainer_page.get_toast_message()
            self.assertIsNotNone(toast_message)
            self.assertTrue("invalid" in toast_message.lower() or "required" in toast_message.lower())

        self.for_each_browser("test_configuration_validation", scenario)

    def test_form_fields_maintain_independent_values(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            basic_tab.set_configs_dir("")
            basic_tab.set_model_name("")
            basic_tab.set_output_dir("")

            test_values = {
                "configs_dir": "/unique/configs/path",
                "model_name": "unique-model-name",
                "output_dir": "/unique/output/path",
                "logging_dir": "/unique/logging/path",
            }

            basic_tab.set_configs_dir(test_values["configs_dir"])
            basic_tab.set_model_name(test_values["model_name"])
            basic_tab.set_output_dir(test_values["output_dir"])
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            basic_tab.set_base_model(test_values["logging_dir"])

            self.assertEqual(basic_tab.get_configs_dir(), test_values["configs_dir"])
            self.assertEqual(basic_tab.get_model_name(), test_values["model_name"])
            self.assertEqual(basic_tab.get_output_dir(), test_values["output_dir"])
            self.assertEqual(basic_tab.get_base_model(), test_values["logging_dir"])

            basic_tab.set_base_model("reverse-base-model")
            basic_tab.set_output_dir("/reverse/output")
            basic_tab.set_model_name("reverse-model")
            basic_tab.set_configs_dir("/reverse/configs")

            self.assertEqual(basic_tab.get_configs_dir(), "/reverse/configs")
            self.assertEqual(basic_tab.get_model_name(), "reverse-model")
            self.assertEqual(basic_tab.get_output_dir(), "/reverse/output")
            self.assertEqual(basic_tab.get_base_model(), "reverse-base-model")

        self.for_each_browser("test_form_fields_maintain_independent_values", scenario)


class TrainingWorkflowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test configuring and starting training."""

    def test_configure_and_start_training(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)
            model_tab = ModelConfigTab(driver, base_url=self.base_url)
            training_tab = TrainingConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")
            basic_tab.set_model_name("flux-test-model")
            basic_tab.set_output_dir("/tmp/test-output")
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")
            model_tab.select_model_family("flux")

            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            training_tab.set_learning_rate("0.0001")
            training_tab.set_batch_size("1")
            training_tab.set_num_epochs("10")
            training_tab.select_mixed_precision("bf16")

            trainer_page.save_configuration()
            self.assertTrue(trainer_page.is_config_valid())

            trainer_page.start_training()
            status = trainer_page.get_training_status()
            self.assertIn(status, ["running", "idle", "training", None])

            if status in ["running", "training"]:
                trainer_page.stop_training()

        self.for_each_browser("test_configure_and_start_training", scenario)


class DatasetManagementTestCase(_TrainerPageMixin, WebUITestCase):
    """Test dataset management functionality."""

    def test_add_and_remove_dataset(self) -> None:
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            datasets_tab = DatasetsTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            initial_count = datasets_tab.get_dataset_count()
            datasets_tab.add_dataset()
            datasets_tab.fill_latest_dataset("/test/dataset/path")
            datasets_tab.save_datasets()

            # Give the UI a moment to surface feedback
            WebDriverWait(driver, 5).until(lambda d: trainer_page.get_toast_message() is not None)
            trainer_page.dismiss_toast()

            new_count = datasets_tab.get_dataset_count()
            self.assertGreaterEqual(new_count, initial_count + 1)

            datasets_tab.delete_dataset(0)
            final_count = datasets_tab.get_dataset_count()
            self.assertEqual(final_count, initial_count)

        self.for_each_browser("test_add_and_remove_dataset", scenario)


class TabNavigationTestCase(_TrainerPageMixin, WebUITestCase):
    """Test tab navigation functionality."""

    def test_all_tabs_load(self) -> None:
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)

            # Temporarily reduce wait timeout for faster test execution
            original_wait_time = trainer_page.wait._timeout
            trainer_page.wait._timeout = 3

            tabs_to_test = [
                ("basic", trainer_page.switch_to_basic_tab),
                ("model", trainer_page.switch_to_model_tab),
                ("training", trainer_page.switch_to_training_tab),
                ("advanced", trainer_page.switch_to_advanced_tab),
                ("datasets", trainer_page.switch_to_datasets_tab),
                ("environments", trainer_page.switch_to_environments_tab),
            ]

            for tab_name, switch_method in tabs_to_test:
                # Switch to tab
                switch_method()

                # Verify tab content is visible using a fast custom check
                selector = trainer_page.TAB_SELECTORS.get(tab_name, f"#tab-content #{tab_name}-tab-content")

                # Wait for element to be present and visible
                is_visible = driver.execute_script(f"""
                    const selector = '{selector}';
                    const el = document.querySelector(selector);
                    return el && el.offsetParent !== null && el.offsetHeight > 0;
                """)

                if not is_visible:
                    # If not immediately visible, wait a bit
                    wait = WebDriverWait(driver, 2)

                    def element_is_visible(driver):
                        try:
                            return driver.execute_script(f"""
                                const el = document.querySelector('{selector}');
                                return el && el.offsetParent !== null && el.offsetHeight > 0;
                            """)
                        except:
                            return False

                    self.assertTrue(wait.until(element_is_visible), f"Tab {tab_name} failed to load")

            # Restore original timeout
            trainer_page.wait._timeout = original_wait_time

        self.for_each_browser("test_all_tabs_load", scenario)


class ToastNotificationsTestCase(_TrainerPageMixin, WebUITestCase):
    """Test toast notification behavior."""

    def test_toast_positioning(self) -> None:
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")
            basic_tab.set_output_dir("/tmp/toast-position")
            basic_tab.save_changes()

            toast_container = driver.find_element(By.CSS_SELECTOR, ".toast-container")
            toast_top = driver.execute_script("return arguments[0].offsetTop;", toast_container)
            self.assertEqual(toast_top, 60)
            trainer_page.dismiss_toast()

        self.for_each_browser("test_toast_positioning", scenario)


class ResponsiveDesignTestCase(_TrainerPageMixin, WebUITestCase):
    """Test responsive design functionality."""

    def test_mobile_viewport(self) -> None:
        self.seed_defaults()

        def scenario(driver, browser):
            if browser != "chrome":
                self.skipTest("Chrome only scenario")
            trainer_page = self._trainer_page(driver)
            driver.set_window_size(375, 812)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            driver.set_window_size(1920, 1080)

        self.for_each_browser("test_mobile_viewport", scenario)


class OnboardingFlowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test onboarding flow for new users."""

    def setUp(self) -> None:
        super().setUp()
        # Ensure a clean slate for onboarding test
        onboarding_file = self.state_dir / "onboarding.json"
        onboarding_file.unlink(missing_ok=True)

        # Also remove defaults.json to ensure onboarding is triggered
        defaults_file = self.state_dir / "defaults.json"
        defaults_file.unlink(missing_ok=True)

    def test_first_time_user_onboarding(self) -> None:
        # Do not seed defaults so onboarding overlay is required.

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()

            # Wait for the page to fully load and check for overlay
            try:
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.onboarding-overlay'))
                )
            except TimeoutException:
                # If overlay not found, check if page loaded properly
                self.fail("Onboarding overlay not found - page may have cached state from previous tests")

            onboarding_overlay = driver.find_element(By.CSS_SELECTOR, '.onboarding-overlay')
            self.assertTrue(onboarding_overlay.is_displayed())

        self.for_each_browser("test_first_time_user_onboarding", scenario)


if __name__ == "__main__":
    unittest.main()
