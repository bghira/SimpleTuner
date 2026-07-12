"""End-to-end tests for WebUI critical user flows (unittest edition)."""

from __future__ import annotations

import json
import time
import unittest

import requests
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from tests.pages.trainer_page import BasicConfigTab, DatasetsTab, ModelConfigTab, TrainerPage, TrainingConfigTab
from tests.webui_test_base import WebUITestCase


class _TrainerPageMixin:
    def _trainer_page(self, driver):
        return TrainerPage(driver, base_url=self.base_url)

    def _show_configured_cloud_dashboard(self, driver, *, load_jobs: bool = False) -> list[str]:
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script(
                "const el = document.querySelector('#cloud-tab-content');"
                "return !!(el && window.Alpine && window.Alpine.$data && window.Alpine.$data(el));"
            )
        )
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script(
                "const comp = window.Alpine.$data(document.querySelector('#cloud-tab-content'));"
                "return comp && comp.providersLoading === false;"
            )
        )
        driver.execute_script(
            """
            const comp = window.Alpine.$data(document.querySelector('#cloud-tab-content'));
            comp.providers = [{ id: 'replicate', name: 'Replicate', configured: true }];
            comp.isActiveProviderConfigured = () => true;
            comp.getActiveProvider = () => ({ id: 'replicate', name: 'Replicate', configured: true });
            comp.activeProvider = 'replicate';
            comp.onboarding = {
                data_understood: true,
                results_understood: true,
                cost_understood: true,
            };
            comp.hints = { dataloader_dismissed: false, git_dismissed: true };
            comp.webhookUrl = '';
            comp.savedWebhookUrl = '';
            comp.publishingStatus = {
                loading: false,
                hf_configured: false,
                hf_token_valid: false,
                hf_username: null,
                hub_model_id: null,
                push_to_hub: false,
                s3_configured: false,
                local_upload_available: false,
                local_upload_dir: null,
                message: null,
            };
            """,
        )

        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script(
                "const el = document.querySelector('.cloud-dashboard');" "return el && el.offsetParent !== null;"
            )
        )
        if not load_jobs:
            return []

        result = driver.execute_async_script(
            """
            const done = arguments[0];
            const comp = window.Alpine.$data(document.querySelector('#cloud-tab-content'));
            if (!comp || typeof comp.loadJobs !== 'function') {
                done({ error: 'Cloud jobs loader is not available' });
                return;
            }
            Promise.resolve(comp.loadJobs())
                .then(() => done({
                    names: (comp.jobs || []).map((job) => job.metadata?.tracker_run_name || null),
                }))
                .catch((error) => done({ error: String(error) }));
            """
        )
        if not isinstance(result, dict):
            raise AssertionError(f"Unexpected Cloud jobs loader result: {result!r}")
        if result.get("error"):
            raise AssertionError(result["error"])
        return result.get("names") or []


class BasicConfigurationFlowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test basic configuration and save flow."""

    MAX_BROWSERS = 1

    def test_ez_mode_resume_checkpoint_options(self) -> None:
        output_dir = self.home_path / "ez-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoint-100").mkdir()
        (output_dir / "checkpoint-200").mkdir()

        env_dir = self.config_dir / "resume-env"
        env_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "--model_family": "flux",
            "--model_type": "lora",
            "--model_flavour": "libreflux",
            "--pretrained_model_name_or_path": "jimmycarter/LibreFlux-SimpleTuner",
            "--output_dir": str(output_dir),
            "--data_backend_config": str(env_dir / "multidatabackend.json"),
            "--job_id": "resume-test",
            "--report_to": "none",
        }
        (env_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
        (env_dir / "multidatabackend.json").write_text("[]", encoding="utf-8")
        self.seed_defaults(active_config="resume-env", output_dir=str(output_dir))

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            select_locator = (By.CSS_SELECTOR, ".ez-mode-form select[x-model='resume_from_checkpoint']")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(select_locator),
                message="EZ mode resume_from_checkpoint select not found",
            )

            def has_checkpoint_options(_driver):
                select = _driver.find_element(*select_locator)
                options = select.find_elements(By.TAG_NAME, "option")
                values = {opt.get_attribute("value") for opt in options}
                return "checkpoint-200" in values and "checkpoint-100" in values

            WebDriverWait(driver, 10).until(
                has_checkpoint_options,
                message="EZ mode resume_from_checkpoint options did not include checkpoints",
            )

            select = driver.find_element(*select_locator)
            self.assertEqual(select.get_attribute("value"), "latest")

        self.for_each_browser("test_ez_mode_resume_checkpoint_options", scenario)

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


class MetricsGpuHealthDashboardTestCase(_TrainerPageMixin, WebUITestCase):
    """Test GPU health dashboard toggles."""

    MAX_BROWSERS = 1

    def test_gpu_history_toggle_series(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_metrics_tab()
            trainer_page.wait_for_tab("metrics")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#metrics-tab-content .gpu-dashboard"))
            )

            temp_toggle = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "#metrics-tab-content input[data-testid='gpu-history-temp-toggle']")
                )
            )
            fan_toggle = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "#metrics-tab-content input[data-testid='gpu-history-fan-toggle']")
                )
            )
            history_container = driver.find_element(By.CSS_SELECTOR, "#metrics-tab-content .history-chart-container")

            self.assertEqual(history_container.get_attribute("data-history-series"), "Temp+Fan")

            temp_toggle.click()
            WebDriverWait(driver, 5).until(lambda d: history_container.get_attribute("data-history-series") == "Fan")

            fan_toggle.click()
            WebDriverWait(driver, 5).until(lambda d: history_container.get_attribute("data-history-series") == "None")

            temp_toggle.click()
            WebDriverWait(driver, 5).until(lambda d: history_container.get_attribute("data-history-series") == "Temp")

        self.for_each_browser("test_gpu_history_toggle_series", scenario)


class EventDockUptimeTestCase(_TrainerPageMixin, WebUITestCase):
    """Test connection uptime tooltip updates."""

    MAX_BROWSERS = 1

    def test_connection_uptime_tooltip_updates(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "if (store) {"
                "  store.connectionStatus = 'connected';"
                "  store.serverUptimeSeconds = 5;"
                "  store.serverUptimeCapturedAt = Date.now();"
                "  store.serverUptimeTick = Date.now();"
                "}"
            )

            def get_tooltip_text(active_driver):
                return active_driver.execute_script(
                    "const el = document.querySelector('.connection-uptime-tooltip');"
                    "return el ? el.textContent.trim() : '';"
                )

            WebDriverWait(driver, 5).until(lambda d: "Server uptime:" in get_tooltip_text(d))
            initial_text = get_tooltip_text(driver)

            driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "if (store && store.serverUptimeCapturedAt) {"
                "  store.serverUptimeTick = store.serverUptimeCapturedAt + 3000;"
                "}"
            )

            WebDriverWait(driver, 5).until(lambda d: get_tooltip_text(d) != initial_text)

        self.for_each_browser("test_connection_uptime_tooltip_updates", scenario)


class FormDirtyStateFlowTestCase(_TrainerPageMixin, WebUITestCase):
    """Test form dirty state transitions across Easy Mode and full form tabs."""

    MAX_BROWSERS = 1

    def test_save_button_dirty_state(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)
            training_tab = TrainingConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            def wait_for_save_state(expected_dirty: bool) -> None:
                WebDriverWait(driver, 10).until(
                    lambda d: d.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "const btn = document.querySelector(\"button[aria-label='Save configuration']\");"
                        "if (!store || !btn) { return false; }"
                        "return Boolean(store.formDirty) === arguments[0]"
                        "  && btn.disabled === !arguments[0]"
                        "  && btn.classList.contains('is-active') === arguments[0];",
                        expected_dirty,
                    )
                )

            def save_via_ui() -> None:
                result = driver.execute_async_script(
                    "const done = arguments[0];"
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (!store || typeof store.doSaveConfig !== 'function') { done(false); return; }"
                    "const maybePromise = store.doSaveConfig({ preserveDefaults: false, createBackup: false });"
                    "if (maybePromise && typeof maybePromise.then === 'function') {"
                    "  maybePromise.then(() => done(true)).catch((err) => { console.error('doSaveConfig failed', err); done(false); });"
                    "} else { done(true); }"
                )
                self.assertTrue(result)

            def wait_for_clean_after_debounce() -> None:
                """Wait for clean state, then wait out debounce window and verify still clean."""
                import time

                wait_for_save_state(False)
                # Wait out the debounce window (default 500ms + buffer)
                time.sleep(0.7)
                # Re-verify still clean - catches delayed dirty flips from validation
                wait_for_save_state(False)

            wait_for_save_state(False)

            # Easy Mode change enables Save
            basic_tab.set_output_dir("/tmp/dirty-easy-mode")
            wait_for_save_state(True)

            save_via_ui()
            wait_for_clean_after_debounce()

            # Full form change enables Save
            trainer_page.switch_to_training_tab()
            training_tab.set_num_epochs(5)  # num_train_epochs is on training tab
            wait_for_save_state(True)

            save_via_ui()
            wait_for_clean_after_debounce()

            # Tab switch + edit still enables Save
            trainer_page.switch_to_model_tab()
            trainer_page.switch_to_training_tab()
            training_tab.set_learning_rate(0.0005)
            wait_for_save_state(True)

            save_via_ui()
            wait_for_clean_after_debounce()

            # Save clears, new edits re-enable
            trainer_page.switch_to_basic_tab()
            basic_tab.set_output_dir("/tmp/dirty-easy-mode-2")
            wait_for_save_state(True)

        self.for_each_browser("test_save_button_dirty_state", scenario)


class TrainingEpochsStepsValidationTestCase(_TrainerPageMixin, WebUITestCase):
    """Test cross-field validation for epochs and max steps stays in sync."""

    MAX_BROWSERS = 1

    def test_epochs_steps_cross_validation(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            training_tab = TrainingConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            trainer_page.wait_for_htmx()

            training_tab.set_max_train_steps(2000)
            training_tab.set_num_epochs(1)

            def feedback_text(active_driver):
                return active_driver.execute_script(
                    "const el = document.getElementById('field-feedback-max_train_steps');"
                    "return el ? el.textContent.trim() : '';"
                )

            WebDriverWait(driver, 10).until(lambda d: "cannot both be set" in feedback_text(d))

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const btn = document.querySelector(\"button[aria-label='Save configuration']\");"
                    "return btn ? btn.disabled : null;"
                )
            )

            training_tab.set_num_epochs(0)

            WebDriverWait(driver, 10).until(lambda d: feedback_text(d) == "")

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const btn = document.querySelector(\"button[aria-label='Save configuration']\");"
                    "return btn ? !btn.disabled : null;"
                )
            )

        self.for_each_browser("test_epochs_steps_cross_validation", scenario)


class ValidationPromptLibraryLayoutTestCase(_TrainerPageMixin, WebUITestCase):
    """Test full-mode Validation prompt library modal layout."""

    MAX_BROWSERS = 1

    def test_full_mode_prompt_library_modal_contains_prompt_row(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            driver.set_window_size(1280, 900)
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            driver.execute_script("localStorage.setItem('st_validation_hints', JSON.stringify({ ez_mode: false }));")
            trainer_page.wait_for_tab("validation")
            trainer_page.wait_for_htmx()

            trigger_selector = "#prompt-library-button-user_prompt_library"
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const button = document.querySelector(arguments[0]);"
                    "return Boolean(button && button.offsetParent !== null && button.dataset.promptLibraryInit === 'true');",
                    trigger_selector,
                ),
                message="Prompt library trigger was not visible and initialised in Validation full mode",
            )

            driver.execute_script(
                "const button = document.querySelector(arguments[0]);"
                "button.scrollIntoView({ block: 'center', inline: 'nearest' });"
                "button.click();",
                trigger_selector,
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const modal = document.querySelector('#promptLibraryModal');"
                    "const prompt = modal && modal.querySelector('.prompt-library-prompt');"
                    "return Boolean(modal && prompt && getComputedStyle(modal).display !== 'none');"
                ),
                message="Prompt library modal did not open with an editable prompt row",
            )

            metrics = driver.execute_script(
                """
                const rect = (element) => {
                    const r = element.getBoundingClientRect();
                    return {
                        left: r.left,
                        right: r.right,
                        top: r.top,
                        bottom: r.bottom,
                        width: r.width,
                        height: r.height
                    };
                };
                const modal = document.querySelector('#promptLibraryModal');
                const dialog = modal.querySelector('.modal-dialog');
                const body = modal.querySelector('.modal-body');
                const rows = modal.querySelector('#prompt-library-rows');
                const row = modal.querySelector('.prompt-library-row');
                const prompt = modal.querySelector('.prompt-library-prompt');
                const modalStyle = getComputedStyle(modal);
                return {
                    modalDisplay: modalStyle.display,
                    modalPosition: modalStyle.position,
                    dialog: rect(dialog),
                    body: rect(body),
                    rows: rect(rows),
                    row: rect(row),
                    prompt: rect(prompt),
                    rowsClientWidth: rows.clientWidth,
                    rowsScrollWidth: rows.scrollWidth
                };
                """
            )

            tolerance = 1.5
            self.assertEqual(metrics["modalDisplay"], "flex")
            self.assertEqual(metrics["modalPosition"], "fixed")
            self.assertLessEqual(metrics["rowsScrollWidth"], metrics["rowsClientWidth"] + tolerance, metrics)
            self.assertGreaterEqual(metrics["row"]["left"], metrics["body"]["left"] - tolerance, metrics)
            self.assertLessEqual(metrics["row"]["right"], metrics["body"]["right"] + tolerance, metrics)
            self.assertGreaterEqual(metrics["prompt"]["left"], metrics["body"]["left"] - tolerance, metrics)
            self.assertLessEqual(metrics["prompt"]["right"], metrics["body"]["right"] + tolerance, metrics)

        self.for_each_browser("test_full_mode_prompt_library_modal_contains_prompt_row", scenario)


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

            with self.subTest("notification_error_does_not_stop_training"):
                driver.execute_script(
                    """
                    if (window.eventHandler && typeof window.eventHandler.processProcessKeeperEvents === 'function') {
                        window.eventHandler.processProcessKeeperEvents([{
                            id: 'notif-err-1',
                            type: 'notification',
                            severity: 'error',
                            message: 'Upload failed due to quota',
                            job_id: 'harness-job',
                            data: { status: 'uploading_model' }
                        }]);
                    }
                    """
                )

                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script("return document.body && document.body.dataset.trainingActive === 'true';")
                )

            with self.subTest("notification_error_with_failure_status_stops_training"):
                driver.execute_script(
                    """
                    if (window.eventHandler && typeof window.eventHandler.processProcessKeeperEvents === 'function') {
                        window.eventHandler.processProcessKeeperEvents([{
                            id: 'notif-err-2',
                            type: 'notification',
                            severity: 'error',
                            message: 'Training failed',
                            job_id: 'harness-job',
                            data: { status: 'failed' }
                        }]);
                    }
                    """
                )

                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script("return document.body && document.body.dataset.trainingActive === 'false';")
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


class DatasetCaptioningTabSmokeTestCase(_TrainerPageMixin, WebUITestCase):
    """Dataset Captioning sub-tab should initialize without Alpine wiring errors."""

    MAX_BROWSERS = 1

    def test_captioning_subtab_initializes(self) -> None:
        dataset_dir = self.home_path / "caption-images"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        default_env_dir = self.config_dir / "default"
        (default_env_dir / "multidatabackend.json").write_text(
            json.dumps(
                [
                    {
                        "id": "caption-smoke",
                        "type": "local",
                        "dataset_type": "image",
                        "instance_data_dir": str(dataset_dir),
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.seed_defaults()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")

            captioning_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(@class, 'dataset-subtab-btn') and contains(., 'Captioning')]")
                )
            )
            captioning_button.click()

            def captioning_panel_ready(active_driver):
                return active_driver.execute_script(
                    """
                    const root = document.querySelector('[x-data="datasetCaptioningComponent()"]');
                    if (!root) return false;
                    const text = root.innerText || '';
                    return text.includes('Captioning')
                        && (
                            text.includes("pip install 'simpletuner[captioning]'")
                            || text.includes('No image datasets are available')
                            || !!root.querySelector('button[type="submit"]')
                        );
                    """
                )

            self.assertTrue(
                WebDriverWait(driver, 10).until(captioning_panel_ready),
                "Captioning sub-tab did not initialize",
            )

            raw_buttons = [
                button
                for button in driver.find_elements(By.XPATH, "//button[contains(., 'Raw Config')]")
                if button.is_displayed()
            ]
            if raw_buttons:
                raw_buttons[0].click()
                WebDriverWait(driver, 5).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, 'textarea[x-model="rawConfig"]')),
                    message="Captioning raw config textarea did not become visible",
                )
            else:
                self.assertIn("pip install 'simpletuner[captioning]'", driver.page_source)

            scroll_state = driver.execute_async_script(
                """
                const done = arguments[0];
                const root = document.querySelector('[x-data="datasetCaptioningComponent()"]');
                const comp = root && window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : null;
                if (!root || !comp) {
                    done({ ready: false, reason: 'captioning component not found' });
                    return;
                }
                if (comp.statusPollTimer) {
                    clearInterval(comp.statusPollTimer);
                    comp.statusPollTimer = null;
                }
                comp.loading = false;
                comp.capabilities = { ready: true, installed: true, version: 'test' };
                comp.datasets = [{ dataset_id: 'caption-smoke', total_files: 1, config: { dataset_type: 'image' } }];
                comp.selectedDatasetId = 'caption-smoke';
                comp.captionJobs = [{ job_id: 'job-scroll', status: 'running', config_name: 'Captioning' }];
                comp.activeJobId = 'job-scroll';
                comp.autoScrollLogs = true;
                comp.jobLogs = Array.from({ length: 80 }, (_, idx) => `caption log line ${idx}`).join('\\n');
                comp.$nextTick(() => {
                    const viewer = root.querySelector('[x-ref="captioningLogViewer"]');
                    if (!viewer) {
                        done({ ready: false, reason: 'log viewer not rendered' });
                        return;
                    }
                    viewer.style.height = '80px';
                    viewer.style.maxHeight = '80px';
                    comp.scrollLogsAfterUpdate(true);
                    setTimeout(() => {
                        done({
                            ready: true,
                            scrollTop: viewer.scrollTop,
                            scrollHeight: viewer.scrollHeight,
                            clientHeight: viewer.clientHeight,
                            atBottom: viewer.scrollTop + viewer.clientHeight >= viewer.scrollHeight - 2
                        });
                    }, 100);
                });
                """
            )
            self.assertTrue(scroll_state.get("ready"), scroll_state)
            self.assertTrue(scroll_state.get("atBottom"), scroll_state)

        self.for_each_browser("test_captioning_subtab_initializes", scenario)


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

            # Dismiss any visible toast that might block the button
            trainer_page.dismiss_toast()

            trainer_page.wait.until(lambda d: d.execute_script("return !!window.datasetWizardComponentInstance"))
            opened = driver.execute_async_script(
                """
                const done = arguments[0];
                const root = document.querySelector('#datasets-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : null;
                if (!comp) {
                    done(false);
                    return;
                }
                Promise.resolve(comp.openWizard())
                    .then(() => done(true))
                    .catch((err) => done(String(err)));
                """
            )
            self.assertTrue(opened)

            state = driver.execute_script(
                """
                const root = document.querySelector('#datasets-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : window.datasetWizardComponentInstance;
                if (!comp) { return { ready: false }; }
                try {
                    comp.openNewFolderDialog();
                    const showNewFolder = comp.showNewFolderInput === true;
                    comp.cancelNewFolder();
                    comp.openUploadModal();
                    const uploadOpen = comp.uploadModalOpen === true;
                    comp.closeUploadModal();
                    const hasFields = ['activeSubTab','showNewFolderInput','newFolderName','newFolderError','uploadModalOpen','selectedUploadFiles','captionModalOpen','captionStatus','pendingCaptions']
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

            step_ready = driver.execute_script(
                """
                const root = document.querySelector('#datasets-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : window.datasetWizardComponentInstance;
                if (!comp) { return null; }
                const captionsStep = comp.getStepNumber('captions');
                if (!captionsStep) { return null; }
                comp.wizardStep = captionsStep;
                if (typeof comp.updateWizardTitle === 'function') {
                    comp.updateWizardTitle();
                }
                comp.selectedBackend = 'local';
                comp.currentDataset.type = 'local';
                comp.currentDataset.caption_strategy = 'parquet';
                return captionsStep;
                """
            )
            self.assertIsNotNone(step_ready)

            trainer_page.wait.until(
                lambda d: d.execute_script(
                    "const pathInput = document.querySelector('input[x-model=\"currentDataset.parquet.path\"]');"
                    "const filenameInput = document.querySelector('input[x-model=\"currentDataset.parquet.filename_column\"]');"
                    "const captionInput = document.querySelector('input[x-model=\"currentDataset.parquet.caption_column\"]');"
                    "const extToggle = document.querySelector('#identifierExt');"
                    "return !!(pathInput && filenameInput && captionInput && extToggle);"
                )
            )

            parquet_state = driver.execute_script(
                """
                const filenameInput = document.querySelector('input[x-model="currentDataset.parquet.filename_column"]');
                const captionInput = document.querySelector('input[x-model="currentDataset.parquet.caption_column"]');
                const extToggle = document.querySelector('#identifierExt');
                return {
                    filenameValue: filenameInput ? filenameInput.value : null,
                    captionValue: captionInput ? captionInput.value : null,
                    extChecked: extToggle ? extToggle.checked : null
                };
                """
            )
            self.assertEqual(parquet_state.get("filenameValue"), "id")
            self.assertEqual(parquet_state.get("captionValue"), "caption")
            self.assertFalse(parquet_state.get("extChecked"))

            try:
                logs = driver.get_log("browser")
            except Exception:
                logs = []
            for entry in logs:
                message = entry.get("message", "")
                self.assertNotIn("Alpine Expression Error", message)
                self.assertNotIn("is not defined", message)

        self.for_each_browser("test_dataset_wizard_initializes_modal_state", scenario)

    def test_dataset_wizard_name_step_enables_next_after_typing(self) -> None:
        """The Name step should react to Dataset ID input when no primary dataset exists yet."""
        datasets_root = self.home_path / "datasets"
        datasets_root.mkdir(parents=True, exist_ok=True)
        self.seed_defaults(datasets_dir=datasets_root)

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_datasets_tab()
            trainer_page.wait_for_tab("datasets")
            trainer_page.dismiss_toast()

            trainer_page.wait.until(lambda d: d.execute_script("return !!window.datasetWizardComponentInstance"))
            opened = driver.execute_async_script(
                """
                const done = arguments[0];
                const root = document.querySelector('#datasets-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : null;
                if (!comp) {
                    done(false);
                    return;
                }
                Promise.resolve(comp.openWizard())
                    .then(() => done(true))
                    .catch((err) => done(String(err)));
                """
            )
            self.assertTrue(opened)

            next_button = trainer_page.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".dataset-wizard-modal button[aria-label='Continue to next step']")
                )
            )
            self.assertFalse(next_button.is_enabled())

            trainer_page.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".dataset-wizard-modal input[x-model='currentDataset.id']"))
            )
            driver.execute_script(
                """
                const input = document.querySelector(".dataset-wizard-modal input[x-model='currentDataset.id']");
                input.value = 'primary-images';
                input.dispatchEvent(new Event('input', { bubbles: true }));
                """
            )

            trainer_page.wait.until(
                lambda d: d.execute_script(
                    """
                    const button = document.querySelector(".dataset-wizard-modal button[aria-label='Continue to next step']");
                    return button && !button.disabled;
                    """
                )
            )

            state = driver.execute_script(
                """
                const root = document.querySelector('#datasets-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(root) : window.datasetWizardComponentInstance;
                return {
                    canProceed: comp ? comp.canProceed === true : null,
                    hasPrimaryDatasetAvailable: comp ? comp.hasPrimaryDatasetAvailable === true : null,
                    regularisationChecked: !!document.querySelector('#regularisationDatasetToggle')?.checked,
                    infoVisible: (() => {
                        const alert = document.querySelector('.dataset-wizard-modal .alert-info');
                        if (!alert) { return null; }
                        return window.getComputedStyle(alert).display !== 'none';
                    })()
                };
                """
            )
            self.assertTrue(state.get("canProceed"), state)
            self.assertFalse(state.get("hasPrimaryDatasetAvailable"), state)
            self.assertFalse(state.get("regularisationChecked"), state)
            self.assertTrue(state.get("infoVisible"), state)

        self.for_each_browser("test_dataset_wizard_name_step_enables_next_after_typing", scenario)


class CloudTabVisibilityTestCase(_TrainerPageMixin, WebUITestCase):
    """Ensure the Cloud tab default matches UI Settings."""

    MAX_BROWSERS = 1

    def test_cloud_tab_visible_when_default_setting_omitted(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            cloud_tab = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".tab-btn[data-tab='cloud']"))
            )
            self.assertTrue(cloud_tab.is_displayed())

            settings_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='ui_settings']"))
            )
            settings_tab.click()
            trainer_page.wait_for_tab("ui_settings")

            checkbox = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "ui-cloud-tab-enabled")))
            self.assertTrue(checkbox.is_selected())

        self.for_each_browser("test_cloud_tab_visible_when_default_setting_omitted", scenario)


class CloudUploadProgressTestCase(_TrainerPageMixin, WebUITestCase):
    """Test cloud upload progress wiring for SSE updates."""

    MAX_BROWSERS = 1

    def test_upload_progress_uses_webhooks_endpoint(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)

            cloud_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='cloud']"))
            )
            cloud_tab.click()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "cloud-tab-content")))
            trainer_page.wait_for_htmx()

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return !!(window.Alpine && document.querySelector('#cloud-tab-content'));")
            )

            started = driver.execute_script(
                """
                window.__cloudTestOriginalEventSource = window.EventSource;
                window.__cloudTestEventSourceUrl = null;
                window.__cloudTestEventSourceInstance = null;
                window.EventSource = function(url) {
                    window.__cloudTestEventSourceUrl = url;
                    const instance = { close: function() {}, onmessage: null, onerror: null };
                    window.__cloudTestEventSourceInstance = instance;
                    return instance;
                };
                const el = document.querySelector('#cloud-tab-content');
                const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;
                if (!comp || typeof comp.startUploadProgress !== 'function') {
                    return false;
                }
                comp.startUploadProgress('upload-test-1');
                return true;
                """
            )
            self.assertTrue(started)

            url = driver.execute_script("return window.__cloudTestEventSourceUrl;")
            self.assertEqual(url, "/api/webhooks/upload/progress/upload-test-1")

            driver.execute_script(
                """
                const instance = window.__cloudTestEventSourceInstance;
                if (instance && instance.onmessage) {
                    instance.onmessage({
                        data: JSON.stringify({
                            stage: 'uploading',
                            current: 512,
                            total: 1024,
                            percent: 50,
                            message: 'Uploading...'
                        })
                    });
                }
                """
            )

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "const el = document.querySelector('#cloud-tab-content');"
                    "const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;"
                    "return comp && comp.uploadProgress && comp.uploadProgress.message === 'Uploading...';"
                )
            )

            driver.execute_script(
                "if (window.__cloudTestOriginalEventSource) { window.EventSource = window.__cloudTestOriginalEventSource; }"
            )

        self.for_each_browser("test_upload_progress_uses_webhooks_endpoint", scenario)


class CloudHardwareProfileSubmitTestCase(_TrainerPageMixin, WebUITestCase):
    """Test Replicate hardware profile modal persistence and submission."""

    MAX_BROWSERS = 1

    def _open_cloud_tab(self, driver) -> None:
        trainer_page = self._trainer_page(driver)
        trainer_page.navigate_to_trainer()
        self.dismiss_onboarding(driver)

        cloud_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='cloud']"))
        )
        cloud_tab.click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "cloud-tab-content")))
        trainer_page.wait_for_htmx()
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return !!(window.Alpine && document.querySelector('#cloud-tab-content'));")
        )

    def _install_cloud_modal_harness(self, driver) -> bool:
        return driver.execute_script(
            """
            const el = document.querySelector('#cloud-tab-content');
            const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;
            if (!comp) { return false; }

            comp.activeProvider = 'replicate';
            comp.providers = [{
                id: 'replicate',
                name: 'Replicate',
                hardware_profile: 'l40s-x4',
                hardware_profiles: [
                    { id: 'h100', label: 'H100', hardware_type: 'H100', cost_per_hour: 5.49, cost_per_second: 0.001525 },
                    { id: 'h100-x8', label: '8x H100', hardware_type: '8x H100', cost_per_hour: 43.92, cost_per_second: 0.0122 },
                    { id: 'l40s', label: 'L40S', hardware_type: 'L40S', cost_per_hour: 3.50, cost_per_second: 0.000972222 },
                    { id: 'l40s-x4', label: '4x L40S', hardware_type: '4x L40S', cost_per_hour: 14.00, cost_per_second: 0.003888888 },
                    { id: 'l40s-x8', label: '8x L40S', hardware_type: '8x L40S', cost_per_hour: 28.00, cost_per_second: 0.007777776 }
                ]
            }];
            comp.providerConfig = { config: { hardware_profile: 'l40s-x4' }, cost_limit_enabled: false };
            comp.selectedConfigName = 'default';
            comp.webhookUrl = '';
            comp.quickSubmitMode = false;
            comp.isActiveProviderConfigured = () => true;
            comp.getActiveProvider = () => ({ id: 'replicate', name: 'Replicate' });
            comp.loadDataUploadPreview = async () => {
                comp.preSubmitModal.dataUploadPreview = {
                    requires_upload: false,
                    datasets: [],
                    total_files: 0,
                    total_size_mb: 0
                };
                comp.preSubmitModal.dataConsentConfirmed = true;
            };
            comp.loadCostEstimate = async () => {
                comp.preSubmitModal.costEstimate = {
                    has_estimate: true,
                    estimated_cost_usd: 1.23,
                    hardware_cost_per_hour: 4.0
                };
            };
            comp.loadConfigPreview = async () => {
                comp.preSubmitModal.configPreview = {};
                comp.preSubmitModal.dataloaderPreview = [];
            };
            comp.checkWebhookReachability = async () => {
                comp.preSubmitModal.webhookCheck = {
                    tested: true,
                    testing: false,
                    success: true,
                    error: null,
                    skipped: false
                };
            };
            comp.loadJobs = () => {};
            comp.loadCostLimitStatus = () => {};
            comp.submitJobToProvider = async (payload) => {
                window.__lastCloudSubmitPayload = payload;
                return { success: true, job_id: 'cloud-hardware-e2e' };
            };
            window.__cloudHardwareComponentReady = true;
            return true;
            """
        )

    def _open_modal(self, driver) -> None:
        opened = driver.execute_async_script(
            """
            const done = arguments[0];
            const el = document.querySelector('#cloud-tab-content');
            const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;
            if (!comp || typeof comp.openPreSubmitModal !== 'function') { done(false); return; }
            comp.openPreSubmitModal().then(() => done(true)).catch((err) => {
                console.error('openPreSubmitModal failed', err);
                done(false);
            });
            """
        )
        self.assertTrue(opened)
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".cloud-submit-modal.show")))
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-testid='replicate-hardware-profile-select']"))
        )

    def test_replicate_hardware_profile_persists_and_submits(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            self._open_cloud_tab(driver)
            driver.execute_script("localStorage.removeItem('cloud_replicate_hardware_profile');")
            self.assertTrue(self._install_cloud_modal_harness(driver))

            self._open_modal(driver)
            select_el = driver.find_element(By.CSS_SELECTOR, "[data-testid='replicate-hardware-profile-select']")
            self.assertEqual(select_el.get_attribute("value"), "l40s-x4")

            Select(select_el).select_by_value("h100-x8")
            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script("return localStorage.getItem('cloud_replicate_hardware_profile');") == "h100-x8"
            )

            driver.execute_script(
                "const comp = Alpine.$data(document.querySelector('#cloud-tab-content')); comp.closePreSubmitModal();"
            )
            self._open_modal(driver)
            select_el = driver.find_element(By.CSS_SELECTOR, "[data-testid='replicate-hardware-profile-select']")
            self.assertEqual(select_el.get_attribute("value"), "h100-x8")

            driver.refresh()
            self._open_cloud_tab(driver)
            self.assertTrue(self._install_cloud_modal_harness(driver))
            self._open_modal(driver)
            select_el = driver.find_element(By.CSS_SELECTOR, "[data-testid='replicate-hardware-profile-select']")
            self.assertEqual(select_el.get_attribute("value"), "h100-x8")

            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='cloud-submit-next-button']"))
            )
            next_button.click()
            submit_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='cloud-submit-final-button']"))
            )
            submit_button.click()

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script("return window.__lastCloudSubmitPayload?.hardware_profile;") == "h100-x8"
            )

        self.for_each_browser("test_replicate_hardware_profile_persists_and_submits", scenario)

    def test_replicate_settings_hardware_buttons_update_cost_and_persist(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            self._open_cloud_tab(driver)
            driver.execute_script("localStorage.setItem('cloud_replicate_hardware_profile', 'h100-x8');")
            self.assertTrue(self._install_cloud_modal_harness(driver))

            driver.execute_script(
                """
                const comp = Alpine.$data(document.querySelector('#cloud-tab-content'));
                comp.preSubmitModal.hardwareProfile = 'h100-x8';
                comp.showSettingsPanel = true;
                """
            )

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "return Array.from(document.querySelectorAll('[data-testid=\"cloud-settings-hardware-profile\"]'))"
                    ".filter((button) => button.offsetParent !== null).length === 2;"
                )
            )
            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "const panel = document.querySelector('.cloud-settings-panel');"
                    "return panel && panel.innerText.includes('$43.92/hr') && "
                    "panel.innerText.includes('$0.012200/sec');"
                )
            )

            pressed_before = driver.execute_script(
                """
                return Array.from(document.querySelectorAll('[data-testid="cloud-settings-hardware-profile"]'))
                    .filter((button) => button.offsetParent !== null)
                    .map((button) => ({ text: button.innerText.trim(), pressed: button.getAttribute('aria-pressed') }));
                """
            )
            self.assertIn({"text": "H100", "pressed": "true"}, pressed_before)
            self.assertIn({"text": "L40S", "pressed": "false"}, pressed_before)

            clicked = driver.execute_script(
                """
                const button = Array.from(document.querySelectorAll('[data-testid="cloud-settings-hardware-profile"]'))
                    .find((candidate) => candidate.offsetParent !== null && candidate.innerText.trim() === 'L40S');
                if (!button) { return false; }
                button.click();
                return true;
                """
            )
            self.assertTrue(clicked)

            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script("return localStorage.getItem('cloud_replicate_hardware_profile');") == "l40s-x8"
            )
            WebDriverWait(driver, 5).until(
                lambda d: d.execute_script(
                    "const panel = document.querySelector('.cloud-settings-panel');"
                    "return panel && panel.innerText.includes('$28.00/hr') && "
                    "panel.innerText.includes('$0.007778/sec');"
                )
            )

            pressed_after = driver.execute_script(
                """
                return Array.from(document.querySelectorAll('[data-testid="cloud-settings-hardware-profile"]'))
                    .filter((button) => button.offsetParent !== null)
                    .map((button) => ({ text: button.innerText.trim(), pressed: button.getAttribute('aria-pressed') }));
                """
            )
            self.assertIn({"text": "H100", "pressed": "false"}, pressed_after)
            self.assertIn({"text": "L40S", "pressed": "true"}, pressed_after)

        self.for_each_browser("test_replicate_settings_hardware_buttons_update_cost_and_persist", scenario)


class CloudUploadStatusTestCase(_TrainerPageMixin, WebUITestCase):
    """Ensure uploading jobs appear in the cloud job list."""

    MAX_BROWSERS = 1

    def _seed_cloud_job(self, job_id: str, status: str, run_name: str, error_message: str | None = None) -> None:
        from datetime import datetime, timezone

        from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
        from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import get_default_config_dir
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository

        async def _seed() -> None:
            for config_dir in {self.home_path / ".simpletuner", self.config_dir, get_default_config_dir()}:
                config_dir.mkdir(parents=True, exist_ok=True)
                await AsyncJobStore.reset_instance()
                JobRepository.reset_instance()
                store = await AsyncJobStore.get_instance(config_dir=config_dir)
                job = UnifiedJob(
                    job_id=job_id,
                    job_type=JobType.CLOUD,
                    provider="replicate",
                    status=status,
                    config_name="test-config",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    error_message=error_message,
                    metadata={"tracker_run_name": run_name},
                )
                await store.add_job(job)

        import asyncio

        asyncio.run(_seed())

    def test_uploading_job_visible_in_cloud_list(self) -> None:
        self.with_sample_environment()
        secrets_path = self.home_path / ".simpletuner" / "secrets.json"
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        secrets_path.write_text(json.dumps({"REPLICATE_API_TOKEN": "r8_test_dummy"}), encoding="utf-8")

        self._seed_cloud_job("upload-test-1", "uploading", "Upload Smoke")
        self._seed_cloud_job("upload-failed-1", "failed", "Upload Failed", "Upload failed")

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)

            driver.execute_script(
                "localStorage.setItem('cloud_onboarding', JSON.stringify({"
                "data_understood: true, results_understood: true, cost_understood: true}));"
            )

            cloud_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='cloud']"))
            )
            cloud_tab.click()

            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "cloud-tab-content")))
            trainer_page.wait_for_htmx()

            loaded_job_names = self._show_configured_cloud_dashboard(driver, load_jobs=True)
            self.assertIn("Upload Smoke", loaded_job_names)
            self.assertIn("Upload Failed", loaded_job_names)

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const el = document.querySelector('#cloud-tab-content');"
                    "const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;"
                    "if (!comp || !Array.isArray(comp.jobs)) { return false; }"
                    "const names = comp.jobs.map(job => job.metadata?.tracker_run_name);"
                    "return names.includes('Upload Smoke') && names.includes('Upload Failed');"
                )
            )

            upload_status = driver.execute_script(
                "const el = document.querySelector('#cloud-tab-content');"
                "const comp = window.Alpine && window.Alpine.$data ? window.Alpine.$data(el) : null;"
                "const job = comp.jobs.find(j => j.metadata?.tracker_run_name === 'Upload Smoke');"
                "return job ? job.status : null;"
            )
            self.assertEqual(upload_status, "uploading")

        self.for_each_browser("test_uploading_job_visible_in_cloud_list", scenario)


class CloudWebhookDraftInputTestCase(_TrainerPageMixin, WebUITestCase):
    """Ensure the Cloud checklist webhook input remains editable before save."""

    MAX_BROWSERS = 1

    def test_webhook_hint_input_stays_visible_while_typing_unsaved_draft(self) -> None:
        self.with_sample_environment()
        secrets_path = self.home_path / ".simpletuner" / "secrets.json"
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        secrets_path.write_text(json.dumps({"REPLICATE_API_TOKEN": "r8_test_dummy"}), encoding="utf-8")

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)

            driver.execute_script(
                "localStorage.removeItem('cloud_hints');"
                "localStorage.setItem('cloud_onboarding', JSON.stringify({"
                "data_understood: true, results_understood: true, cost_understood: true}));"
            )

            cloud_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='cloud']"))
            )
            cloud_tab.click()

            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "cloud-tab-content")))
            trainer_page.wait_for_htmx()

            self._show_configured_cloud_dashboard(driver)

            input_selector = ".cloud-setup-checklist .webhook-setup input[type='url']"
            webhook_input = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, input_selector))
            )
            webhook_input.send_keys("h")

            WebDriverWait(driver, 5).until(
                lambda d: d.find_element(By.CSS_SELECTOR, input_selector).get_attribute("value") == "h"
            )

            input_count = driver.execute_script(
                "return document.querySelectorAll(arguments[0]).length;",
                input_selector,
            )
            self.assertEqual(input_count, 1)

            has_output_destination = driver.execute_script(
                "const el = document.querySelector('#cloud-tab-content');"
                "if (!el) { throw new Error('Cloud tab content not found'); }"
                "if (!window.Alpine || !window.Alpine.$data) { throw new Error('Alpine is not initialized'); }"
                "const comp = window.Alpine.$data(el);"
                "if (!comp) { throw new Error('Cloud Alpine component not found'); }"
                "return comp.hasOutputDestination;"
            )
            self.assertIs(has_output_destination, False)

        self.for_each_browser("test_webhook_hint_input_stays_visible_while_typing_unsaved_draft", scenario)


class EasyModeFormDirtyTestCase(_TrainerPageMixin, WebUITestCase):
    """Test that Easy Mode field changes correctly enable the save button.

    Regression test for: Easy Mode events must use .stop modifier to prevent
    bubbling to form handlers that reset formDirty state.
    """

    MAX_BROWSERS = 1

    def test_easy_mode_enables_save_on_direct_load(self) -> None:
        """Editing Easy Mode fields on direct page load should enable save button."""
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            trainer_page.wait_for_htmx()

            # Wait for Easy Mode component to initialize
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.querySelector('.ez-mode-form select') !== null;")
            )

            # Verify formDirty starts false
            initial_dirty = driver.execute_script(
                "const store = window.Alpine?.store?.('trainer');" "return store ? store.formDirty : null;"
            )
            self.assertFalse(initial_dirty, "formDirty should start as false")

            # Change model type in Easy Mode (click the Full Model radio)
            driver.execute_script(
                """
                const fullModelRadio = document.querySelector('input[type="radio"][value="full"]');
                if (fullModelRadio) {
                    fullModelRadio.click();
                }
                """
            )

            # Wait a moment for Alpine reactivity
            time.sleep(0.2)

            # Verify formDirty is now true
            dirty_after_change = driver.execute_script(
                "const store = window.Alpine?.store?.('trainer');" "return store ? store.formDirty : null;"
            )
            self.assertTrue(dirty_after_change, "formDirty should be true after Easy Mode change")

            # Verify save button is enabled
            save_button_disabled = driver.execute_script(
                "const btn = document.querySelector('.header-actions .trainer-action-btn');"
                "return btn ? btn.disabled : null;"
            )
            self.assertFalse(save_button_disabled, "Save button should be enabled after Easy Mode change")

        self.for_each_browser("test_easy_mode_enables_save_on_direct_load", scenario)

    def test_easy_mode_select_enables_save(self) -> None:
        """Changing Easy Mode select dropdowns should enable save button."""
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            trainer_page.wait_for_htmx()

            # Wait for Easy Mode model family select
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.querySelector('.ez-mode-form select') !== null;")
            )

            # Change a select in Easy Mode (base model precision)
            driver.execute_script(
                """
                const precisionSelect = document.querySelector('.ez-mode-form select[x-model="base_model_precision"]');
                if (precisionSelect) {
                    precisionSelect.value = 'int8-quanto';
                    precisionSelect.dispatchEvent(new Event('change', { bubbles: false }));
                }
                """
            )

            time.sleep(0.2)

            # Verify formDirty is true
            dirty_state = driver.execute_script(
                "const store = window.Alpine?.store?.('trainer');" "return store ? store.formDirty : null;"
            )
            self.assertTrue(dirty_state, "formDirty should be true after Easy Mode select change")

        self.for_each_browser("test_easy_mode_select_enables_save", scenario)

    def test_main_form_still_enables_save(self) -> None:
        """Regular form fields should still enable save button (sanity check)."""
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)
            basic_tab = BasicConfigTab(driver, base_url=self.base_url)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")
            trainer_page.wait_for_htmx()

            # Change a regular form field
            basic_tab.set_output_dir("/new/output/path")

            time.sleep(0.2)

            # Verify formDirty is true
            dirty_state = driver.execute_script(
                "const store = window.Alpine?.store?.('trainer');" "return store ? store.formDirty : null;"
            )
            self.assertTrue(dirty_state, "formDirty should be true after main form change")

        self.for_each_browser("test_main_form_still_enables_save", scenario)


class EasyModeOptimizerSyncTestCase(_TrainerPageMixin, WebUITestCase):
    """Test that Easy Mode optimizer controls reflect full form values."""

    MAX_BROWSERS = 1

    def test_easy_mode_optimizer_syncs_from_full_form(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            trainer_page.wait_for_htmx()

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "return document.querySelector('.ez-mode-form select[x-model=\"optimizer\"]') !== null;"
                )
            )

            def get_optimizer_values(active_driver):
                return active_driver.execute_script(
                    "const ez = document.querySelector('.ez-mode-form select[x-model=\"optimizer\"]');"
                    "const full = document.getElementById('optimizer');"
                    "return { ez: ez ? ez.value : null, full: full ? full.value : null };"
                )

            WebDriverWait(driver, 10).until(lambda d: get_optimizer_values(d)["full"])

            WebDriverWait(driver, 10).until(lambda d: get_optimizer_values(d)["ez"] == get_optimizer_values(d)["full"])

            new_value = driver.execute_script(
                """
                const fullSelect = document.getElementById('optimizer');
                if (!fullSelect || !fullSelect.options || fullSelect.options.length === 0) {
                    return null;
                }
                const options = Array.from(fullSelect.options);
                const current = fullSelect.value;
                const next = options.find(opt => opt.value && opt.value !== current) || options[0];
                fullSelect.value = next.value;
                fullSelect.dispatchEvent(new Event('change', { bubbles: true }));
                fullSelect.dispatchEvent(new Event('input', { bubbles: true }));
                return next.value;
                """
            )
            self.assertTrue(new_value, "Expected to find optimizer options in full form.")

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "const ez = document.querySelector('.ez-mode-form select[x-model=\"optimizer\"]');"
                    "return ez ? ez.value : null;"
                )
                == new_value
            )

        self.for_each_browser("test_easy_mode_optimizer_syncs_from_full_form", scenario)

    def test_easy_mode_max_grad_norm_uses_full_form_default(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            trainer_page.wait_for_htmx()

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "return document.querySelector('.ez-mode-form input[x-model\\\\.number=\"max_grad_norm\"]') !== null;"
                )
            )

            def get_grad_norm_values(active_driver):
                return active_driver.execute_script(
                    """
                    const ez = document.querySelector('.ez-mode-form input[x-model\\\\.number="max_grad_norm"]');
                    const full = document.getElementById('max_grad_norm');
                    return { ez: ez ? ez.value : null, full: full ? full.value : null };
                    """
                )

            WebDriverWait(driver, 10).until(lambda d: get_grad_norm_values(d)["full"])
            WebDriverWait(driver, 10).until(lambda d: get_grad_norm_values(d)["ez"])

            values = get_grad_norm_values(driver)
            self.assertEqual(float(values["full"]), 2.0)
            self.assertEqual(float(values["ez"]), 2.0)

        self.for_each_browser("test_easy_mode_max_grad_norm_uses_full_form_default", scenario)

    def test_easy_mode_optimizer_preset_selection_updates_from_full_form_batch_size(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            trainer_page.wait_for_htmx()

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    """
                    return Array.from(document.querySelectorAll('.ez-mode-form .optimizer-preset-card'))
                        .some(card => card.textContent.includes('Moderate'));
                    """
                )
            )

            driver.execute_script(
                """
                const moderateCard = Array.from(document.querySelectorAll('.ez-mode-form .optimizer-preset-card'))
                    .find(card => card.textContent.includes('Moderate'));
                moderateCard.click();
                """
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    """
                    const component = window.Alpine?.$data(document.getElementById('training-tab-content'));
                    return component?.selectedOptimizerPreset === 'moderate';
                    """
                )
            )

            driver.execute_script(
                """
                const batchSizeInput = document.getElementById('train_batch_size');
                batchSizeInput.value = '3';
                batchSizeInput.dispatchEvent(new Event('input', { bubbles: true }));
                batchSizeInput.dispatchEvent(new Event('change', { bubbles: true }));
                """
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    """
                    const component = window.Alpine?.$data(document.getElementById('training-tab-content'));
                    const selectedCards = document.querySelectorAll('.ez-mode-form .optimizer-preset-card.selected');
                    return component?.train_batch_size === 3
                        && component?.selectedOptimizerPreset === null
                        && selectedCards.length === 0;
                    """
                )
            )

        self.for_each_browser(
            "test_easy_mode_optimizer_preset_selection_updates_from_full_form_batch_size",
            scenario,
        )

    def test_full_form_optimizer_presets_button_applies_preset(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.switch_to_training_tab()
            trainer_page.wait_for_tab("training")
            trainer_page.wait_for_htmx()

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.querySelector('.ez-mode-footer button') !== null;")
            )

            driver.execute_script(
                """
                const switchButton = Array.from(document.querySelectorAll('.ez-mode-footer button'))
                    .find(button => button.textContent.includes('Switch to Full Form'));
                if (switchButton) {
                    switchButton.click();
                }
                """
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    "return document.querySelector('#section-optimizer_config .optimizer-presets-btn') !== null;"
                )
            )

            driver.execute_script("document.querySelector('#section-optimizer_config .optimizer-presets-btn').click();")

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(
                    """
                    return Array.from(document.querySelectorAll('.optimizer-presets-modal .optimizer-preset-card'))
                        .some(card => card.textContent.includes('Moderate'));
                    """
                )
            )

            driver.execute_script(
                """
                const moderateCard = Array.from(document.querySelectorAll('.optimizer-presets-modal .optimizer-preset-card'))
                    .find(card => card.textContent.includes('Moderate'));
                moderateCard.click();
                document.querySelector('.optimizer-presets-modal .modal-footer .btn-primary').click();
                """
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.getElementById('learning_rate')?.value === '0.0001';")
            )

            values = driver.execute_script(
                """
                const store = window.Alpine?.store?.('trainer');
                return {
                    learningRate: document.getElementById('learning_rate')?.value,
                    optimizer: document.getElementById('optimizer')?.value,
                    trainBatchSize: store?.activeEnvironmentConfig?.['--train_batch_size'],
                    gradAccum: store?.activeEnvironmentConfig?.['--gradient_accumulation_steps'],
                    dirty: store?.formDirty
                };
                """
            )

            self.assertEqual(values["learningRate"], "0.0001")
            self.assertEqual(values["optimizer"], "adamw_bf16")
            self.assertEqual(values["trainBatchSize"], 2)
            self.assertEqual(values["gradAccum"], 1)
            self.assertTrue(values["dirty"])

        self.for_each_browser("test_full_form_optimizer_presets_button_applies_preset", scenario)


class SaveAsNewConfigurationTestCase(_TrainerPageMixin, WebUITestCase):
    """Regression coverage for the top-bar "Save As..." config cloning flow.

    Previously Save As captured only the inputs materialized in the DOM, so any
    field on an unvisited tab was dropped and the new config came back as mostly
    defaults. It must now clone the complete loaded config (including a live
    edit) and repoint name-derived identity fields to the new name.
    """

    MAX_BROWSERS = 1

    def _seed_source_config(self) -> None:
        env_dir = self.config_dir / "rank32-src"
        env_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "--model_family": "flux",
            "--model_type": "lora",
            "--model_flavour": "libreflux",
            "--pretrained_model_name_or_path": "jimmycarter/LibreFlux-SimpleTuner",
            "--lora_rank": "32",
            "--output_dir": "output/rank32-src",
            "--tracker_run_name": "rank32-src-run",
            "--data_backend_config": str(env_dir / "multidatabackend.json"),
            "--job_id": "rank32-src",
            "--report_to": "none",
        }
        (env_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
        (env_dir / "multidatabackend.json").write_text("[]", encoding="utf-8")
        self.seed_defaults(active_config="rank32-src", output_dir="output/rank32-src")

    def test_save_as_preserves_config_and_repoints_identity(self) -> None:
        self._seed_source_config()

        def scenario(driver, _browser):
            trainer_page = self._trainer_page(driver)

            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)
            trainer_page.wait_for_tab("basic")

            # Confirm the source config loaded before we edit it.
            trainer_page.switch_to_model_tab()
            trainer_page.wait_for_tab("model")
            WebDriverWait(driver, 10).until(
                lambda d: d.find_element(By.ID, "lora_rank").get_attribute("value") in ("32", "32.0"),
                message="lora_rank did not populate from the loaded config",
            )

            # The user's scenario: bump rank 32 -> 64 in the form.
            driver.execute_script(
                "const el = document.getElementById('lora_rank');"
                "el.value = arguments[0];"
                "el.dispatchEvent(new Event('input', { bubbles: true }));"
                "el.dispatchEvent(new Event('change', { bubbles: true }));",
                "64",
            )

            # Trigger Save As through the real header dropdown. window.prompt is
            # overridden because Selenium cannot interact with the native dialog. The
            # ".json" suffix exercises name sanitization: the backend strips it, so the
            # UI must switch to the sanitized "rank64-clone", not "rank64-clone.json".
            driver.execute_script("window.prompt = function () { return 'rank64-clone.json'; };")
            toggle = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".config-selector__toggle-wrapper button")),
                message="config selector toggle not clickable",
            )
            toggle.click()
            save_as = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//div[contains(@class, 'config-selector')]//a[contains(normalize-space(.), 'Save As')]")
                ),
                message="Save As menu item not clickable",
            )
            save_as.click()

            def _config_created(_driver):
                response = requests.get(f"{self.base_url}/api/configs/rank64-clone", timeout=5)
                return response if response.status_code == 200 else False

            config_response = WebDriverWait(driver, 10).until(
                _config_created,
                message="Save As did not create the new configuration",
            )
            body = config_response.json().get("config") or {}

            # The UI must switch to the sanitized name, not the raw "rank64-clone.json".
            active_env = driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "return store ? store.activeEnvironment : null;"
            )
            self.assertEqual(active_env, "rank64-clone")

            # Live edit captured.
            self.assertIn(body.get("--lora_rank"), ("64", "64.0"))
            # Untouched fields preserved (would be absent/default under the old bug).
            self.assertEqual(body.get("--model_flavour"), "libreflux")
            self.assertEqual(
                body.get("--pretrained_model_name_or_path"),
                "jimmycarter/LibreFlux-SimpleTuner",
            )
            # Identity fields repointed to the new name, not the source's.
            output_dir = body.get("--output_dir") or ""
            self.assertIn("rank64-clone", output_dir)
            self.assertNotIn("rank32-src", output_dir)
            tracker_run = body.get("--tracker_run_name") or ""
            self.assertIn("rank64-clone", tracker_run)
            self.assertNotIn("rank32-src", tracker_run)

            # The original config is untouched.
            source_response = requests.get(f"{self.base_url}/api/configs/rank32-src", timeout=5)
            source_response.raise_for_status()
            source_body = source_response.json().get("config") or {}
            self.assertIn(source_body.get("--lora_rank"), ("32", "32.0"))
            self.assertEqual(source_body.get("--output_dir"), "output/rank32-src")

        self.for_each_browser("test_save_as_preserves_config_and_repoints_identity", scenario)


if __name__ == "__main__":
    unittest.main()
