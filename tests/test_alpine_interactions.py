"""Alpine.js interaction tests using unittest."""

from __future__ import annotations

import os
import time
import unittest
from datetime import datetime

from selenium.common.exceptions import (
    JavascriptException,
    UnexpectedAlertPresentException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tests.webui_test_base import WebUITestCase


class AlpineComponentIsolationTestCase(WebUITestCase):
    """Tests that ensure Alpine components do not leak state.

    Performance optimizations:
    - Set TEST_FAST_MODE=true to skip thorough but slow tests
    - Set TEST_PERF_LOG=true to see timing for each subtest
    - Set SELENIUM_BROWSERS=chrome to test only Chrome
    - Dynamic waits instead of fixed sleeps
    - Batched JavaScript execution where possible
    """

    FAST_MODE = os.environ.get('TEST_FAST_MODE', '').lower() == 'true'
    PERFORMANCE_LOG = os.environ.get('TEST_PERF_LOG', '').lower() == 'true'

    # Override browser selection for faster testing
    if os.environ.get('TEST_SINGLE_BROWSER', '').lower() == 'true':
        BROWSERS = ['chrome']

    def _navigate(self, driver) -> None:
        driver.get(f"{self.base_url}/web/trainer")
        WebDriverWait(driver, 10).until(lambda d: d.execute_script("return window.Alpine !== undefined"))
        self.dismiss_onboarding(driver)

    def _wait_for_alpine_update(self, driver, timeout=0.5):
        """Wait for Alpine.js to finish updating the DOM."""
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script(
                    "return !document.querySelector('[x-transition]') || "
                    "Array.from(document.querySelectorAll('[x-transition]')).every(el => "
                    "!el.classList.contains('x-transition-enter') && "
                    "!el.classList.contains('x-transition-leave'));"
                )
            )
        except:
            # If wait times out, continue anyway
            pass

    def _batch_alpine_data(self, driver, elements):
        """Retrieve Alpine data for multiple elements in a single call."""
        return driver.execute_script(
            """
            return arguments[0].map(el => {
                try {
                    return Alpine.$data(el);
                } catch(e) {
                    return null;
                }
            });
            """,
            elements
        )

    def _log_performance(self, operation: str, duration: float) -> None:
        """Log performance metrics if enabled."""
        if self.PERFORMANCE_LOG:
            print(f"[PERF] {operation}: {duration:.3f}s")

    def test_alpine_component_suite(self) -> None:
        self.seed_defaults()

        def scenario(driver, _browser):
            suite_start = time.time()
            self._navigate(driver)
            self.dismiss_onboarding(driver)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "trainer-form")))

            with self.subTest("x_data_initialization"):
                start_time = time.time()
                field_checks = driver.execute_script(
                    """
                    return Array.from(document.querySelectorAll('.mb-3[x-data]')).map((el, idx) => {
                        const data = Alpine.$data(el);
                        const missing = [];
                        if (!data || typeof data !== 'object') {
                            return {idx, hasData: false, missingProps: ['value','error','validating','isValid']};
                        }
                        for (const prop of ['value','error','validating','isValid']) {
                            if (!(prop in data)) { missing.push(prop); }
                        }
                        return {idx, hasData: true, missingProps: missing};
                    });
                    """
                )
                for check in field_checks:
                    self.assertTrue(check["hasData"], f"Field {check['idx']} missing Alpine data")
                    self.assertFalse(check["missingProps"], f"Field {check['idx']} missing: {check['missingProps']}")
                self._log_performance("x_data_initialization", time.time() - start_time)

            with self.subTest("stores_isolation"):
                start_time = time.time()
                store_results = driver.execute_script(
                    """
                    const names = Alpine.store ? Object.keys(Alpine.store) : [];
                    return names.map(name => {
                        let value = null;
                        try {
                            value = Alpine.store(name);
                        } catch (err) {
                            value = undefined;
                        }
                        const ok = value === null || typeof value === 'object';
                        return {name, ok};
                    });
                    """
                )
                for result in store_results:
                    self.assertTrue(result["ok"], f"Store {result['name']} returned non-object value")
                self._log_performance("stores_isolation", time.time() - start_time)

            with self.subTest("x_model_not_present"):
                start_time = time.time()
                violations = driver.execute_script(
                    """
                    const allowedIds = new Set(['createBackupCheck', 'mergeEnvironmentToggleHidden']);
                    const allowedModels = new Set(['createBackupOption', 'mergeEnabledString']);
                    return Array.from(document.querySelectorAll('input, select, textarea'))
                        .filter(el => !el.closest('.onboarding-overlay'))
                        .filter(el => !el.closest('.save-dataset-overlay, .save-config-overlay'))
                        .map(el => {
                            const id = el.id || '';
                            const model = el.getAttribute('x-model');
                            if (!model) { return null; }
                            if (allowedIds.has(id)) { return null; }
                            if (allowedModels.has(model)) { return null; }
                            return id || '<unnamed>';
                        })
                        .filter(Boolean);
                    """
                )
                self.assertFalse(violations, f"Unexpected x-model bindings present: {violations}")
                self._log_performance("x_model_not_present", time.time() - start_time)

            with self.subTest("data_binding_independence"):
                start_time = time.time()
                def _locate(selector: str):
                    return driver.execute_script("return document.querySelector(arguments[0]);", selector)

                job_id_input = _locate("input[name='job_id']") or _locate("input[name='--job_id']")
                backend_input = _locate("input[name='--data_backend_config']")
                if not job_id_input or not backend_input:
                    self.skipTest("Basic config inputs not present")
                # Batch element queries
                components = driver.execute_script(
                    """
                    const job = arguments[0].closest('[x-data]');
                    const backend = arguments[1].closest('[x-data]');
                    const jobData = Alpine.$data(job);
                    if (jobData) {
                        jobData.value = 'test-value';
                        jobData.error = 'test-error';
                    }
                    return {
                        job: job,
                        backend: backend,
                        backendData: Alpine.$data(backend),
                        areEqual: job === backend
                    };
                    """,
                    job_id_input,
                    backend_input
                )
                self.assertFalse(components['areEqual'])
                backend_data = components['backendData']
                self.assertNotEqual(backend_data.get("value"), "test-value")
                self.assertNotEqual(backend_data.get("error"), "test-error")
                self._log_performance("data_binding_independence", time.time() - start_time)

            with self.subTest("form_field_scopes_are_isolated"):
                start_time = time.time()
                impacted = driver.execute_script(
                    """
                    const components = Array.from(document.querySelectorAll('[x-data]'));
                    if (components.length < 2) { return []; }
                    const first = Alpine.$data(components[0]);
                    if (!first || typeof first !== 'object' || !('value' in first)) {
                        return [];
                    }
                    const original = first.value;
                    first.value = 'modified-value';
                    const affected = [];
                    for (let idx = 1; idx < components.length; idx++) {
                        const data = Alpine.$data(components[idx]);
                        if (data && typeof data === 'object' && data.value === 'modified-value') {
                            affected.push(idx);
                        }
                    }
                    first.value = original;
                    return affected;
                    """
                )
                self.assertFalse(impacted, f"Other Alpine scopes were affected: {impacted}")
                self._log_performance("form_field_scopes_are_isolated", time.time() - start_time)

            with self.subTest("event_handling"):
                start_time = time.time()
                # Limit to first 3 elements in fast mode
                max_elements = 1 if self.FAST_MODE else 3
                clickable_elements = driver.find_elements(By.CSS_SELECTOR, "[\\@click]")
                for element in clickable_elements[:max_elements]:
                    parent_with_data = driver.execute_script(
                        "return arguments[0].closest('[x-data]');",
                        element,
                    )
                    if not parent_with_data:
                        continue
                    before_data = driver.execute_script(
                        "return JSON.stringify(Alpine.$data(arguments[0]));",
                        parent_with_data,
                    )
                    try:
                        driver.execute_script(
                            "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                            element,
                        )
                        if not element.is_enabled() or not element.is_displayed():
                            continue
                        element.click()
                        self._wait_for_alpine_update(driver, 0.3)
                        try:
                            after_data = driver.execute_script(
                                "return JSON.stringify(Alpine.$data(arguments[0]));",
                                parent_with_data,
                            )
                        except UnexpectedAlertPresentException:
                            alert = driver.switch_to.alert
                            alert.dismiss()
                            after_data = driver.execute_script(
                                "return JSON.stringify(Alpine.$data(arguments[0]));",
                                parent_with_data,
                            )
                        click_handler = element.get_attribute("@click") or ""
                        if "!" in click_handler:
                            self.assertNotEqual(before_data, after_data)
                    except (JavascriptException, WebDriverException):
                        continue
                self._log_performance("event_handling", time.time() - start_time)

            with self.subTest("x_show_sections"):
                start_time = time.time()
                collapsible_sections = driver.find_elements(By.CSS_SELECTOR, "[x-show='expanded']")
                # Skip in fast mode or limit sections
                sections_to_test = [] if self.FAST_MODE else collapsible_sections[:2]
                for section in sections_to_test:
                    parent = driver.execute_script(
                        "return arguments[0].closest('[x-data]');",
                        section,
                    )
                    if not parent:
                        continue
                    driver.execute_script(
                        """
                        const component = Alpine.$data(arguments[0]);
                        if (component && component.expanded !== undefined) {
                            component.expanded = !component.expanded;
                        }
                        """,
                        parent,
                    )
                    self._wait_for_alpine_update(driver, 0.3)
                    is_visible = section.is_displayed()
                    alpine_state = driver.execute_script(
                        "const component = Alpine.$data(arguments[0]); return component ? component.expanded : null;",
                        parent,
                    )
                    self.assertEqual(is_visible, alpine_state)
                self._log_performance("x_show_sections", time.time() - start_time)

            with self.subTest("component_cleanup"):
                start_time = time.time()
                if self.FAST_MODE:
                    self.skipTest("Skipping component cleanup in fast mode")
                self.dismiss_onboarding(driver)
                initial_count = driver.execute_script(
                    """
                    let count = 0;
                    document.querySelectorAll('[x-data]').forEach(el => {
                        if (Alpine.$data(el)) count++;
                    });
                    return count;
                    """
                )
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='model']"))).click()
                self._wait_for_alpine_update(driver)
                self.dismiss_onboarding(driver)
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".tab-btn[data-tab='basic']"))).click()
                self._wait_for_alpine_update(driver)
                final_count = driver.execute_script(
                    """
                    let count = 0;
                    document.querySelectorAll('[x-data]').forEach(el => {
                        if (Alpine.$data(el)) count++;
                    });
                    return count;
                    """
                )
                self.assertLessEqual(abs(final_count - initial_count), 1)
                self._log_performance("component_cleanup", time.time() - start_time)

            with self.subTest("transition_toggles"):
                start_time = time.time()
                if self.FAST_MODE:
                    self.skipTest("Skipping transition toggles in fast mode")
                transition_elements = driver.find_elements(By.CSS_SELECTOR, "[x-transition]")
                if not transition_elements:
                    self.skipTest("No transition elements present")
                elem = transition_elements[0]
                parent = driver.execute_script("return arguments[0].closest('[x-data]');", elem)
                if parent:
                    initial_visible = elem.is_displayed()
                    toggled = driver.execute_script(
                        """
                        const component = Alpine.$data(arguments[0]);
                        if (!component) { return null; }
                        const showProp = Object.keys(component).find(key => key.includes('show') || key.includes('expanded'));
                        if (!showProp) { return null; }
                        component[showProp] = !component[showProp];
                        return showProp;
                        """,
                        parent,
                    )
                    if toggled is None:
                        self.skipTest("No togglable property found for transition element")
                    self._wait_for_alpine_update(driver, 0.5)
                    final_visible = elem.is_displayed()
                    self.assertNotEqual(initial_visible, final_visible)
                self._log_performance("transition_toggles", time.time() - start_time)

            self._log_performance("complete_suite", time.time() - suite_start)

        self.for_each_browser("test_alpine_component_suite", scenario)

if __name__ == "__main__":
    unittest.main()
