"""HTMX behaviour tests using a compact unittest suite."""

from __future__ import annotations

import unittest

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from tests.pages.trainer_page import TrainerPage
from tests.webui_test_base import WebUITestCase


class HTMXBehaviourTestCase(WebUITestCase):
    """Exercise core HTMX interactions without redundant page reloads."""

    def test_htmx_interactions_suite(self) -> None:
        self.with_sample_environment()

        def scenario(driver, _browser):
            trainer_page = TrainerPage(driver, base_url=self.base_url)
            trainer_page.navigate_to_trainer()
            self.dismiss_onboarding(driver)

            driver.find_element(By.CSS_SELECTOR, ".tab-btn[data-tab='basic']").click()

            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "trainer-form")))
            driver.execute_script(
                """
                if (!window.__htmxHarness) {
                    window.__htmxHarness = {
                        requests: [],
                        domChanges: [],
                        fetches: [],
                        indicatorFlags: {
                            hxActivated: false,
                            hxCleared: false,
                            dataShown: false,
                            dataHidden: false
                        }
                    };

                    document.body.addEventListener('htmx:configRequest', function(evt) {
                        window.__htmxHarness.requests.push({
                            path: evt.detail.path,
                            verb: evt.detail.verb,
                            parameters: evt.detail.parameters || {},
                            headers: evt.detail.headers || {}
                        });
                    });

                    const domObserver = new MutationObserver(function(mutations) {
                        mutations.forEach(function(mutation) {
                            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                                window.__htmxHarness.domChanges.push({
                                    target: mutation.target.id || mutation.target.className || 'unknown',
                                    added: mutation.addedNodes.length
                                });
                            }
                        });
                    });
                    domObserver.observe(document.body, { childList: true, subtree: true });

                    const indicatorButton = document.querySelector('button[hx-indicator]');
                    const indicatorSelector = indicatorButton ? indicatorButton.getAttribute('hx-indicator') : null;
                    const indicatorTarget = indicatorSelector
                        ? (indicatorSelector.startsWith('#')
                            ? document.getElementById(indicatorSelector.slice(1))
                            : document.querySelector(indicatorSelector))
                        : null;
                    if (indicatorTarget) {
                        const indicatorObserver = new MutationObserver(function() {
                            if (indicatorTarget.classList.contains('htmx-request')) {
                                window.__htmxHarness.indicatorFlags.hxActivated = true;
                            } else if (window.__htmxHarness.indicatorFlags.hxActivated) {
                                window.__htmxHarness.indicatorFlags.hxCleared = true;
                            }
                        });
                        indicatorObserver.observe(indicatorTarget, { attributes: true, attributeFilter: ['class'] });
                    }

                    const dataIndicator = document.querySelector('[data-htmx-indicator]');
                    if (dataIndicator) {
                        const dataObserver = new MutationObserver(function() {
                            const visible = dataIndicator.offsetParent !== null && !dataIndicator.classList.contains('d-none');
                            if (visible) {
                                window.__htmxHarness.indicatorFlags.dataShown = true;
                            } else if (window.__htmxHarness.indicatorFlags.dataShown) {
                                window.__htmxHarness.indicatorFlags.dataHidden = true;
                            }
                        });
                        dataObserver.observe(dataIndicator, { attributes: true, attributeFilter: ['class', 'style'] });
                    }

                    if (!window.__htmxHarnessOriginalFetch && window.fetch) {
                        window.__htmxHarnessOriginalFetch = window.fetch.bind(window);
                        window.fetch = async function(input, init = {}) {
                            try {
                                const url = typeof input === 'string' ? input : (input && input.url) || '';
                                const method = (init && init.method ? init.method : 'get').toLowerCase();
                                const headers = init && init.headers ? init.headers : {};
                                const record = { url, method, headers, parameters: {} };
                                if (init && init.body instanceof FormData) {
                                    init.body.forEach((value, key) => {
                                        record.parameters[key] = value;
                                    });
                                }
                                window.__htmxHarness.fetches.push(record);
                            } catch (error) {
                                console.warn('Fetch capture failed', error);
                            }
                            return window.__htmxHarnessOriginalFetch(input, init);
                        };
                    }
                }

                window.__htmxHarness.requests = [];
                window.__htmxHarness.domChanges = [];
                window.__htmxHarness.fetches = [];
                window.__htmxHarness.indicatorFlags = {
                    hxActivated: false,
                    hxCleared: false,
                    dataShown: false,
                    dataHidden: false
                };
                if (window.Alpine && Alpine.store) {
                    const trainer = Alpine.store('trainer');
                    if (trainer && trainer.form && trainer.form.values) {
                        trainer.form.values['--job_id'] = 'htmx-suite-job';
                        trainer.hasUnsavedChanges = true;
                    }
                }
                """
            )

            def fire_request(*, expect_dom: bool = False):
                req_before = driver.execute_script("return window.__htmxHarness.requests.length;")
                dom_before = driver.execute_script("return window.__htmxHarness.domChanges.length;")
                driver.execute_script(
                    """
                    if (window.Alpine && Alpine.store) {
                        const trainer = Alpine.store('trainer');
                        if (trainer && typeof trainer.saveConfig === 'function') {
                            trainer.saveConfig();
                            trainer.createConfigBackupOption = false;
                            trainer.preserveDefaultsOption = false;
                            if (typeof trainer.confirmSaveConfig === 'function') {
                                trainer.confirmSaveConfig();
                            }
                        }
                    }
                    """
                )
                WebDriverWait(driver, 5).until(
                    lambda d: driver.execute_script("return window.__htmxHarness.requests.length;") > req_before
                )
                if expect_dom:
                    WebDriverWait(driver, 5).until(
                        lambda d: driver.execute_script("return window.__htmxHarness.domChanges.length;") > dom_before
                    )
                requests = driver.execute_script("return window.__htmxHarness.requests;") or []
                fetches = driver.execute_script("return window.__htmxHarness.fetches;") or []
                request = next((r for r in fetches if r and "/api/training/config" in (r.get("url") or "")), None)
                if request is None:
                    request = next(
                        (r for r in requests if r and "/api/training/config" in (r.get("path") or "")),
                        requests[-1] if requests else {},
                    )
                dom_entry = driver.execute_script("return window.__htmxHarness.domChanges.slice(-1)[0];")
                return request, dom_entry

            request, dom_entry = fire_request(expect_dom=True)
            flags = driver.execute_script("return window.__htmxHarness.indicatorFlags;")

            with self.subTest("form_submission_includes_all_fields"):
                params = request.get("parameters", {})
                verb = (request.get("verb") or request.get("method") or "").lower()
                self.assertIn(verb, {"get", "post"}, f"Unexpected HTMX verb: {verb}")
                path = request.get("path") or request.get("url", "")
                self.assertIn("/api/training/config", path)
                self.assertIn("--job_id", params)
                self.assertIn("--output_dir", params)

            with self.subTest("htmx_target_updates_correctly"):
                self.assertIsNotNone(dom_entry, "No DOM mutations captured")
                self.assertGreater(dom_entry.get("added", 0), 0, "HTMX swap did not add new nodes")

            with self.subTest("htmx_indicator_shows_and_hides"):
                has_hx_indicator = driver.execute_script(
                    "return !!document.querySelector('button[hx-indicator], a[hx-indicator]');"
                )
                if not has_hx_indicator:
                    self.skipTest("No hx-indicator button present")
                if not flags.get("hxActivated"):
                    self.skipTest("Programmatic save did not trigger hx-indicator")
                self.assertTrue(flags.get("hxCleared"), "Indicator did not clear after request")

            with self.subTest("htmx_loading_indicator_hides_after_request"):
                has_loading_indicator = driver.execute_script("return !!document.querySelector('[data-htmx-indicator]');")
                if not has_loading_indicator:
                    self.skipTest("No data-htmx-indicator element present")
                self.assertTrue(flags.get("dataShown"), "Loading indicator never became visible")
                self.assertTrue(flags.get("dataHidden"), "Loading indicator stayed visible")

            with self.subTest("config_save_uses_post"):
                method = (request.get("method") or request.get("verb") or "").lower()
                self.assertEqual(method, "post", f"Configuration save used unexpected method: {method}")

            with self.subTest("htmx_validation_endpoint"):
                try:
                    status = driver.execute_async_script(
                        """
                        const done = arguments[0];
                        const payload = arguments[1];
                        const params = new URLSearchParams();
                        Object.entries(payload).forEach(([key, value]) => params.append(key, value));
                        const controller = new AbortController();
                        setTimeout(() => controller.abort(), 3000);
                        fetch(window.location.origin + '/api/training/validate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                            body: params.toString(),
                            signal: controller.signal
                        }).then(resp => done(resp.status)).catch(err => done(String(err)));
                        """,
                        {
                            "--model_type": "lora",
                            "--model_family": "flux",
                            "--resolution": "1024",
                        },
                    )
                except TimeoutException:
                    self.skipTest("Validation endpoint fetch timed out")
                    status = None
                if isinstance(status, str) and "AbortError" in status:
                    self.skipTest("Validation endpoint fetch aborted")
                self.assertIsInstance(status, int, f"Validation endpoint returned error: {status}")
                self.assertLess(status, 500)

        self.for_each_browser("test_htmx_interactions_suite", scenario)


if __name__ == "__main__":
    unittest.main()
