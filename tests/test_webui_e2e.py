"""End-to-end tests for WebUI critical user flows (unittest edition)."""

from __future__ import annotations

import unittest

from selenium.common.exceptions import TimeoutException
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

    MAX_BROWSERS = 1

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
            basic_tab.set_base_model("jimmycarter/LibreFlux-SimpleTuner")

            basic_tab.save_changes()
            toast_message = trainer_page.get_toast_message()
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
        """Test that form fields maintain their values when switching tabs.

        NOTE: This test is currently failing due to a WebUI bug where form values
        are cleared when switching tabs. This is a real issue that needs to be
        fixed in the WebUI implementation, not a test problem.
        """
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            # Clear initial values
            basic_tab.set_model_name("")
            basic_tab.set_output_dir("")

            test_values = {
                "model_name": "unique-model-name",
                "output_dir": "/unique/output/path",
            }

            # Set initial values in basic tab
            basic_tab.set_model_name(test_values["model_name"])
            basic_tab.set_output_dir(test_values["output_dir"])

            # Wait for the model_name field to have the expected value
            WebDriverWait(driver, 5).until(
                lambda d: basic_tab.get_model_name() == test_values["model_name"],
                message=f"Model name field did not update to {test_values['model_name']}",
            )

            # Switch to a different tab
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")

            # Switch back to basic tab to check values
            trainer_page.switch_to_basic_tab()
            trainer_page.wait_for_tab("basic")

            # Wait for HTMX to finish loading the tab content
            trainer_page.wait_for_htmx()

            # Wait for the form elements to be present after tab switch
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='tracker_project_name']")),
                message="Model name input field not found after switching back to basic tab",
            )

            # Give the JavaScript time to restore form values
            import time

            time.sleep(0.5)

            # Verify values were preserved after tab switch
            self.assertEqual(basic_tab.get_model_name(), test_values["model_name"])
            self.assertEqual(basic_tab.get_output_dir(), test_values["output_dir"])

            # Now modify the values
            basic_tab.set_output_dir("/reverse/output")
            basic_tab.set_model_name("reverse-model")

            # Wait for model_name to update
            WebDriverWait(driver, 5).until(
                lambda d: basic_tab.get_model_name() == "reverse-model",
                message="Model name field did not update to reverse-model",
            )

            # Switch to another tab and back to test preservation
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            trainer_page.switch_to_basic_tab()
            trainer_page.wait_for_tab("basic")

            # Wait for form to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='tracker_project_name']")),
                message="Model name field not found after final tab switch",
            )

            # Give time for value restoration
            time.sleep(0.5)

            # Verify the modified values were preserved
            self.assertEqual(basic_tab.get_model_name(), "reverse-model")
            self.assertEqual(basic_tab.get_output_dir(), "/reverse/output")

        self.for_each_browser("test_form_fields_maintain_independent_values", scenario)


class TrainingWorkflowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test configuring and starting training."""

    MAX_BROWSERS = 1

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
            basic_tab.set_base_model("jimmycarter/LibreFlux-SimpleTuner")
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
            self.assertIn(status, ["running", "idle", "training", "validation", None])

            if status in ["running", "training"]:
                trainer_page.stop_training()

        self.for_each_browser("test_configure_and_start_training", scenario)

    def test_training_failure_updates_ui_state(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            basic_tab.set_model_name("failure-job")
            basic_tab.set_output_dir("/tmp/failure-output")

            trainer_page.start_training()

            trainer_page.wait_for_training_active(timeout=10)

            WebDriverWait(driver, 5).until(
                lambda d: bool(d.execute_script("return !!window.__simpletunerTestCallbackHook;"))
            )

            dispatch_script = (
                "window.dispatchEvent(new CustomEvent('simpletuner:test:callback',"
                " { detail: { category: 'status', payload: arguments[0] } }));"
            )

            driver.execute_script(
                dispatch_script,
                {
                    "type": "training.status",
                    "status": "starting",
                    "severity": "info",
                    "message": "Training is starting",
                    "job_id": "harness-job",
                    "data": {"status": "starting", "job_id": "harness-job"},
                },
            )

            driver.execute_script(
                dispatch_script,
                {
                    "type": "training.status",
                    "status": "failed",
                    "severity": "error",
                    "message": "Caption file not found",
                    "job_id": "harness-job",
                    "data": {"status": "failed", "job_id": "harness-job"},
                },
            )

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "const runBtn=document.getElementById('runBtn');"
                    "const cancelBtn=document.getElementById('cancelBtn');"
                    "return !!(runBtn && cancelBtn && !runBtn.disabled && cancelBtn.disabled);"
                )
            )

            trainer_state = driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "return store ? { isTraining: !!store.isTraining, showTrainingProgress: !!store.showTrainingProgress } : null;"
            )
            self.assertIsNotNone(trainer_state)
            self.assertFalse(trainer_state.get("isTraining"))
            self.assertFalse(trainer_state.get("showTrainingProgress"))

            self.assertEqual(
                driver.execute_script(
                    "return document.body && document.body.dataset.trainingActive ? document.body.dataset.trainingActive : 'false';"
                ),
                "false",
            )

            status_text = driver.execute_script(
                "const el = document.getElementById('training-status');"
                "return el ? (el.textContent || '').toLowerCase() : '';"
            )
            self.assertNotIn("running", status_text)

            trainer_page.wait_for_training_inactive(timeout=10)

        self.for_each_browser("test_training_failure_updates_ui_state", scenario)


class DatasetManagementTestCase(_TrainerPageMixin, WebUITestCase):
    """Test dataset management functionality."""

    MAX_BROWSERS = 1

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
                ("datasets", trainer_page.switch_to_datasets_tab),
                ("environments", trainer_page.switch_to_environments_tab),
            ]

            for tab_name, switch_method in tabs_to_test:
                # Switch to tab
                switch_method()

                # Verify tab content is visible using a fast custom check
                selector = trainer_page.TAB_SELECTORS.get(tab_name, f"#tab-content #{tab_name}-tab-content")

                # Wait for element to be present and visible
                is_visible = driver.execute_script(
                    f"""
                    const selector = '{selector}';
                    const el = document.querySelector(selector);
                    return el && el.offsetParent !== null && el.offsetHeight > 0;
                """
                )

                if not is_visible:
                    # If not immediately visible, wait a bit
                    wait = WebDriverWait(driver, 2)

                    def element_is_visible(driver):
                        try:
                            return driver.execute_script(
                                f"""
                                const el = document.querySelector('{selector}');
                                return el && el.offsetParent !== null && el.offsetHeight > 0;
                            """
                            )
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

        # Reset the server's onboarding state via API
        import requests

        response = requests.post(f"{self.base_url}/api/webui/onboarding/reset")
        response.raise_for_status()

        # Clear any defaults as well
        # Note: We're not manipulating directories directly since the server
        # is running in a separate process with its own environment

    def test_first_time_user_onboarding(self) -> None:
        # Do not seed defaults so onboarding overlay is required.

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()

            # Wait for overlay to be visible (wait for opacity transition)
            try:
                onboarding_overlay = WebDriverWait(driver, 5).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, ".onboarding-overlay"))
                )
                self.assertTrue(onboarding_overlay.is_displayed())
            except TimeoutException:
                # If still not visible, check what's wrong
                try:
                    overlay = driver.find_element(By.CSS_SELECTOR, ".onboarding-overlay")
                    print(f"DEBUG: Overlay found but not visible, style: {overlay.get_attribute('style')}")
                except:
                    print("DEBUG: Overlay element not found at all")
                self.fail("Onboarding overlay not visible after waiting")

        self.for_each_browser("test_first_time_user_onboarding", scenario)


if __name__ == "__main__":
    unittest.main()
