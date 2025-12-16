"""End-to-end tests for WebUI critical user flows (unittest edition)."""

from __future__ import annotations

import json
import time
import unittest

import requests
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

            trainer_page.wait_for_htmx()

            active_env = (
                driver.execute_script(
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "return store && store.activeEnvironment ? store.activeEnvironment : 'default';"
                )
                or "default"
            )

            config_response = requests.get(f"{self.base_url}/api/configs/{active_env}", timeout=5)
            config_response.raise_for_status()
            config_payload = config_response.json()
            config_body = config_payload.get("config") if isinstance(config_payload, dict) else {}
            self.assertEqual(config_body.get("--output_dir"), "/test/output")

            state_response = requests.get(f"{self.base_url}/api/webui/state", timeout=5)
            state_response.raise_for_status()
            state_payload = state_response.json()
            defaults = state_payload.get("defaults") if isinstance(state_payload, dict) else {}
            self.assertEqual(defaults.get("configs_dir"), str(self.config_dir))
            self.assertEqual(defaults.get("output_dir"), "/test/output")

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

    def test_config_json_modal_reflects_blank_fields(self) -> None:
        """Ensure the Config JSON modal reflects cleared text fields after saving."""

        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            def refresh_active_config() -> bool:
                return driver.execute_async_script(
                    "const done = arguments[0];"
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (!store || typeof store.fetchActiveEnvironmentConfig !== 'function') { done(false); return; }"
                    "try {"
                    "  const result = store.fetchActiveEnvironmentConfig();"
                    "  if (result && typeof result.then === 'function') {"
                    "    result.then(() => done(true)).catch(() => done(false));"
                    "  } else {"
                    "    done(true);"
                    "  }"
                    "} catch (err) {"
                    "  console.error('refresh_active_config failed', err);"
                    "  done(false);"
                    "}"
                )

            config_path = self.config_dir / "test-config" / "config.json"
            disk_payload_initial = json.loads(config_path.read_text(encoding="utf-8"))

            trainer_page.open_config_json_modal()
            try:
                driver.execute_script(
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (store && typeof store.resetConfigJsonDraft === 'function') {"
                    "  store.resetConfigJsonDraft();"
                    "}"
                )
                time.sleep(0.05)
                modal_initial = json.loads(trainer_page.get_config_json_text() or "{}")
            finally:
                trainer_page.close_config_json_modal()

            self.assertTrue(
                "pretrained_model_name_or_path" in disk_payload_initial
                or "--pretrained_model_name_or_path" in disk_payload_initial
            )
            self.assertIn("pretrained_model_name_or_path", modal_initial)

            def read_modal_payload():
                trainer_page.open_config_json_modal()
                try:
                    driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "if (store && typeof store.resetConfigJsonDraft === 'function') {"
                        "  store.resetConfigJsonDraft();"
                        "}"
                    )
                    time.sleep(0.05)
                    text = trainer_page.get_config_json_text()
                finally:
                    trainer_page.close_config_json_modal()
                return json.loads(text or "{}")

            first_payload = read_modal_payload()
            self.assertIn("pretrained_model_name_or_path", first_payload)

            basic_tab.set_base_model("")
            basic_tab.save_changes()
            trainer_page.wait_for_htmx()
            self.assertTrue(refresh_active_config(), "Failed to refresh active environment config after clearing field")

            disk_payload = {}
            for _ in range(12):
                disk_payload = json.loads(config_path.read_text(encoding="utf-8"))
                if (
                    "pretrained_model_name_or_path" not in disk_payload
                    and "--pretrained_model_name_or_path" not in disk_payload
                ):
                    break
                time.sleep(0.5)
            self.assertFalse(
                "pretrained_model_name_or_path" in disk_payload or "--pretrained_model_name_or_path" in disk_payload,
            )

            second_payload = {}
            for _ in range(12):
                second_payload = read_modal_payload()
                if "pretrained_model_name_or_path" not in second_payload:
                    break
                time.sleep(0.5)

            self.assertNotIn("pretrained_model_name_or_path", second_payload)

        self.for_each_browser("test_config_json_modal_reflects_blank_fields", scenario)


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

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "return !!(window.eventHandler && typeof window.eventHandler.notifyTrainingState === 'function');"
                )
            )

            WebDriverWait(driver, 5).until(lambda d: d.execute_script("return !!window.__simpletunerLifecycleTestHook;"))

            def send_lifecycle_stage(stage_status, *, key="model_loading", label="Model Loading", percent=50):
                driver.execute_script(
                    """
                    window.__simpletunerLifecycleTestHook && window.__simpletunerLifecycleTestHook({
                        type: 'lifecycle.stage',
                        job_id: 'harness-job',
                        stage: {
                            key: arguments[1],
                            label: arguments[2],
                            status: arguments[0],
                            progress: {
                                current: 1,
                                total: 2,
                                percent: arguments[3]
                            }
                        }
                    });
                    """,
                    stage_status,
                    key,
                    label,
                    percent,
                )

            with self.subTest("running_status_clears_completed_lifecycle_progress"):
                lifecycle_check = driver.execute_script(
                    """
                    return (function() {
                        const handler = window.eventHandler;
                        if (!handler || typeof handler.notifyTrainingState !== 'function') {
                            return { hasHandler: false };
                        }

                        let container = document.getElementById('progressBars');
                        const hadContainer = !!container;
                        let previousChildren = [];

                        if (!container) {
                            container = document.createElement('div');
                            container.id = 'progressBars';
                            const dock = document.querySelector('.event-dock-body') || document.body;
                            dock.appendChild(container);
                        } else {
                            previousChildren = Array.from(container.children).map(child => child.cloneNode(true));
                        }

                        container.innerHTML = '';

                        const addItem = (current, total) => {
                            const item = document.createElement('div');
                            item.className = 'progress-item';
                            item.dataset.current = String(current);
                            item.dataset.total = String(total);
                            container.appendChild(item);
                        };

                        addItem(3, 3); // Completed lifecycle task
                        addItem(1, 3); // In-progress lifecycle task

                        handler.notifyTrainingState('running', { job_id: 'selenium-job' }, {});

                        const remaining = Array.from(container.querySelectorAll('.progress-item')).map(item => ({
                            current: item.dataset.current,
                            total: item.dataset.total
                        }));

                        handler.notifyTrainingState('idle', { job_id: 'selenium-job' }, { force: true });

                        if (hadContainer) {
                            container.innerHTML = '';
                            previousChildren.forEach(child => container.appendChild(child));
                        } else if (container && container.parentNode) {
                            container.parentNode.removeChild(container);
                        }

                        return { hasHandler: true, remaining };
                    })();
                    """
                )
                self.assertTrue(lifecycle_check.get("hasHandler"), "Event handler should be initialised on the trainer page")
                remaining_progress = lifecycle_check.get("remaining", [])
                self.assertEqual(
                    len(remaining_progress),
                    1,
                    "Running status should remove completed lifecycle progress entries while preserving ongoing ones",
                )
                self.assertEqual(remaining_progress[0].get("current"), "1")
                self.assertEqual(remaining_progress[0].get("total"), "3")

            with self.subTest("lifecycle_stage_completion_schedules_removal"):
                lifecycle_stage_check = driver.execute_script(
                    """
                    return (function() {
                        const handler = window.eventHandler;
                        if (!handler || typeof handler.parseStructuredData !== 'function') {
                            return { hasHandler: false };
                        }

                        let container = document.getElementById('progressBars');
                        const hadContainer = !!container;
                        if (!container) {
                            container = document.createElement('div');
                            container.id = 'progressBars';
                            const dock = document.querySelector('.event-dock-body') || document.body;
                            dock.appendChild(container);
                        }

                        container.innerHTML = '';

                        handler.parseStructuredData({
                            message_type: 'lifecycle.stage',
                            stage: {
                                key: 'dataset_indexing',
                                label: 'Dataset Indexing',
                                status: 'completed',
                                progress: {
                                    current: 1,
                                    total: 10,
                                    percent: 10
                                }
                            }
                        });

                        const item = container.querySelector('.progress-item');
                        const result = {
                            hasHandler: true,
                            itemExists: !!item,
                            lifecycleStatus: item ? item.dataset.lifecycleStatus : null,
                            removalScheduled: item ? item.dataset.removalScheduled === 'true' : false
                        };

                        container.innerHTML = '';
                        if (!hadContainer && container.parentNode) {
                            container.parentNode.removeChild(container);
                        }

                        return result;
                    })();
                    """
                )
                self.assertTrue(lifecycle_stage_check.get("hasHandler"), "Event handler should handle lifecycle stages")
                self.assertTrue(
                    lifecycle_stage_check.get("itemExists"), "Lifecycle stage event should create a progress entry"
                )
                self.assertEqual(lifecycle_stage_check.get("lifecycleStatus"), "completed")
                self.assertTrue(
                    lifecycle_stage_check.get("removalScheduled"),
                    "Completed lifecycle stages should schedule progress removal even if percent < 100",
                )

            with self.subTest("lifecycle_component_shows_only_current_stage"):
                send_lifecycle_stage("running", key="dataset_indexing", label="Dataset Indexing", percent=25)
                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script(
                        "return document.querySelectorAll('#training-status .startup-progress-item').length;"
                    )
                    == 1
                )
                first_label = driver.execute_script(
                    "const el=document.querySelector('#training-status .startup-progress-item .fw-semibold');"
                    "return el ? el.textContent.trim() : '';"
                )
                self.assertIn("Dataset Indexing", first_label)

                send_lifecycle_stage("completed", key="dataset_indexing", label="Dataset Indexing", percent=100)
                WebDriverWait(driver, 5).until(
                    lambda d: bool(
                        d.execute_script(
                            "const badge=document.querySelector('#training-status .startup-progress-item .badge');"
                            "return badge && badge.textContent.includes('Completed');"
                        )
                    )
                )

                send_lifecycle_stage("running", key="model_loading", label="Model Loading", percent=10)
                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script(
                        "return document.querySelectorAll('#training-status .startup-progress-item').length;"
                    )
                    == 1
                )
                updated_label = driver.execute_script(
                    "const el=document.querySelector('#training-status .startup-progress-item .fw-semibold');"
                    "return el ? el.textContent.trim() : '';"
                )
                self.assertIn("Model Loading", updated_label)

            with self.subTest("lifecycle_component_clears_after_running_status"):
                driver.execute_script(
                    dispatch_script,
                    {
                        "type": "training.status",
                        "status": "running",
                        "severity": "info",
                        "message": "Training is running",
                        "job_id": "harness-job",
                        "data": {"status": "running", "job_id": "harness-job"},
                    },
                )

                # Lifecycle UI should re-render once another lifecycle event fires after a running status update.
                send_lifecycle_stage("running", key="refresh_after_running", label="Refresh After Running", percent=30)
                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script(
                        "return !!document.querySelector('#training-status .startup-progress-alert');"
                    )
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


class DatasetBuilderViewModeTestCase(_TrainerPageMixin, WebUITestCase):
    """Test dataset builder view mode switching and new UI features."""

    MAX_BROWSERS = 1

    def test_view_mode_toggle(self) -> None:
        """Test switching between list and grid view modes."""
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            datasets_tab = DatasetsTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            # Add a dataset first
            datasets_tab.add_dataset("image")

            # Default should be list view
            self.assertEqual(datasets_tab.get_view_mode(), "list")

            # Switch to grid view
            datasets_tab.switch_to_grid_view()
            self.assertEqual(datasets_tab.get_view_mode(), "cards")

            # Switch back to list view
            datasets_tab.switch_to_list_view()
            self.assertEqual(datasets_tab.get_view_mode(), "list")

        self.for_each_browser("test_view_mode_toggle", scenario)

    def test_dataset_modal(self) -> None:
        """Test opening and closing the dataset configuration modal."""
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            datasets_tab = DatasetsTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            # Add a dataset
            datasets_tab.add_dataset("image")

            # Open modal via JavaScript
            datasets_tab.open_dataset_modal_by_js(0)
            self.assertTrue(datasets_tab.is_modal_open())

            # Default tab should be basic
            self.assertEqual(datasets_tab.get_modal_tab(), "basic")

            # Switch tabs
            datasets_tab.switch_modal_tab("storage")
            self.assertEqual(datasets_tab.get_modal_tab(), "storage")

            datasets_tab.switch_modal_tab("advanced")
            self.assertEqual(datasets_tab.get_modal_tab(), "advanced")

            # Close modal
            datasets_tab.close_dataset_modal()
            self.assertFalse(datasets_tab.is_modal_open())

        self.for_each_browser("test_dataset_modal", scenario)

    def test_dataset_search(self) -> None:
        """Test searching/filtering datasets."""
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            datasets_tab = DatasetsTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            # Add multiple datasets
            datasets_tab.add_dataset("image")
            datasets_tab.add_dataset("video")
            datasets_tab.add_dataset("text_embeds")

            # All should be visible initially (auto text embed + 3 added)
            self.assertEqual(datasets_tab.get_filtered_dataset_count(), 4)

            # Search for "image"
            datasets_tab.search_datasets("image")
            self.assertEqual(datasets_tab.get_filtered_dataset_count(), 1)

            # Clear search
            datasets_tab.clear_dataset_search()
            self.assertEqual(datasets_tab.get_filtered_dataset_count(), 4)

            # Search for "video"
            datasets_tab.search_datasets("video")
            self.assertEqual(datasets_tab.get_filtered_dataset_count(), 1)

        self.for_each_browser("test_dataset_search", scenario)


class DatasetWizardUiSmokeTestCase(_TrainerPageMixin, WebUITestCase):
    """Lightweight UI check for dataset wizard Alpine state."""

    MAX_BROWSERS = 1

    def test_dataset_wizard_initializes_modal_state(self) -> None:
        """Dataset wizard should expose new folder/upload state without Alpine errors."""
        datasets_root = self.home_path / "datasets"
        datasets_root.mkdir(parents=True, exist_ok=True)
        self.seed_defaults(datasets_dir=datasets_root)

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            trainer_page.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[title='Add a dataset to the current configuration']"))
            ).click()

            trainer_page.wait.until(lambda d: d.execute_script("return !!window.datasetWizardComponentInstance"))

            state = driver.execute_script(
                """
                const comp = window.datasetWizardComponentInstance;
                if (!comp) { return { ready: false }; }
                try {
                    comp.openNewFolderDialog();
                    const showNewFolder = comp.showNewFolderInput === true;
                    comp.cancelNewFolder();
                    comp.openUploadModal();
                    const uploadOpen = comp.uploadModalOpen === true;
                    comp.closeUploadModal();
                    const hasFields = ['showNewFolderInput','newFolderName','newFolderError','uploadModalOpen','selectedUploadFiles','captionModalOpen','captionStatus','pendingCaptions']
                        .every(key => key in comp && typeof comp[key] !== 'undefined');
                    return { ready: true, hasFields, showNewFolder, uploadOpen };
                } catch (err) {
                    return { ready: false, error: String(err) };
                }
                """
            )

            self.assertTrue(state.get("ready"), state)
            self.assertTrue(state.get("hasFields"), state)
            self.assertTrue(state.get("showNewFolder"), state)
            self.assertTrue(state.get("uploadOpen"), state)

            try:
                logs = driver.get_log("browser")
            except Exception:
                logs = []
            for entry in logs:
                message = entry.get("message", "")
                self.assertNotIn("Alpine Expression Error", message)
                self.assertNotIn("is not defined", message)

        self.for_each_browser("test_dataset_wizard_initializes_modal_state", scenario)


if __name__ == "__main__":
    unittest.main()
