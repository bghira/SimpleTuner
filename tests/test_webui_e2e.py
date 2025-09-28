"""End-to-end tests for WebUI critical user flows (unittest edition)."""

from __future__ import annotations

import unittest

from selenium.webdriver.common.by import By

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

            basic_tab.set_configs_dir(str(self.config_dir))
            basic_tab.set_model_name("test-model")
            basic_tab.set_output_dir("/test/output")
            basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")

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

            basic_tab.set_configs_dir("")
            basic_tab.set_model_name("")
            basic_tab.set_output_dir("")
            basic_tab.set_base_model("")

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

            self.assertEqual(basic_tab.get_configs_dir(), test_values["configs_dir"])
            self.assertEqual(basic_tab.get_model_name(), test_values["model_name"])
            self.assertEqual(basic_tab.get_output_dir(), test_values["output_dir"])
            self.assertEqual(basic_tab.get_base_model(), test_values["base_model"])

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
            basic_tab.set_model_name("flux-test-model")
            basic_tab.set_output_dir("/tmp/test-output")
            basic_tab.set_base_model("black-forest-labs/FLUX.1-dev")
            basic_tab.save_changes()

            trainer_page.switch_to_model_tab()
            model_tab.select_model_family("flux")
            model_tab.set_lora_rank("16")
            model_tab.set_lora_alpha("32")

            trainer_page.switch_to_training_tab()
            training_tab.set_learning_rate("0.0001")
            training_tab.set_batch_size("1")
            training_tab.set_num_epochs("10")
            training_tab.select_mixed_precision("bf16")

            trainer_page.save_configuration()
            self.assertTrue(trainer_page.is_config_valid())

            trainer_page.start_training()
            self.assertIn(trainer_page.get_training_status(), ["running", "idle", None])

            if trainer_page.get_training_status() == "running":
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
            trainer_page.switch_to_datasets_tab()

            initial_count = datasets_tab.get_dataset_count()
            datasets_tab.add_dataset()
            datasets_tab.fill_dataset_modal("Test Dataset", "/test/dataset/path")
            datasets_tab.save_dataset_modal()

            new_count = datasets_tab.get_dataset_count()
            self.assertEqual(new_count, initial_count + 1)

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
                self.assertTrue(driver.find_element(By.ID, f"tab-{tab_name}").is_displayed())

        self.for_each_browser("test_all_tabs_load", scenario)


class ToastNotificationsTestCase(_TrainerPageMixin, WebUITestCase):
    """Test toast notification behavior."""

    def test_toast_positioning(self) -> None:
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
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
            driver.set_window_size(1920, 1080)

        self.for_each_browser("test_mobile_viewport", scenario)


class OnboardingFlowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test onboarding flow for new users."""

    def test_first_time_user_onboarding(self) -> None:
        # Do not seed defaults so onboarding overlay is required.

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            onboarding_overlay = driver.find_element(By.CSS_SELECTOR, '[x-show="onboardingRequired"]')
            self.assertTrue(onboarding_overlay.is_displayed())

        self.for_each_browser("test_first_time_user_onboarding", scenario)


if __name__ == "__main__":
    unittest.main()
