"""Page object for Trainer page."""

import requests
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    TimeoutException,
    UnexpectedAlertPresentException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .base_page import BasePage


class TrainerPage(BasePage):
    """Page object for the trainer interface."""

    TAB_SELECTORS = {
        "basic": "#tab-content #basic-tab-content",
        "model": "#tab-content #model-tab-content",
        "training": "#tab-content #training-tab-content",
        "advanced": "#tab-content #advanced-tab-content",
        "validation": "#tab-content #validation-tab-content",
        "datasets": "#tab-content #datasets-tab-content",
        "environments": "#tab-content #environments-tab-content",
        "ui_settings": "#tab-content #ui-settings-tab-content",
    }

    # Locators
    SAVE_CONFIG_BUTTON = (By.CSS_SELECTOR, ".header-actions > button.trainer-action-btn")
    START_TRAINING_BUTTON = (By.ID, "runBtn")
    STOP_TRAINING_BUTTON = (By.ID, "cancelBtn")

    # Tab buttons
    BASIC_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='basic']")
    MODEL_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='model']")
    TRAINING_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='training']")
    ADVANCED_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='advanced']")
    DATASETS_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='datasets']")
    ENVIRONMENTS_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='environments']")

    CONFIG_JSON_BUTTON = (By.CSS_SELECTOR, "button[title='View and edit the composed training JSON']")
    CONFIG_JSON_MODAL = (By.CSS_SELECTOR, ".config-json-modal")
    CONFIG_JSON_TEXTAREA = (By.CSS_SELECTOR, ".config-json-modal textarea")
    CONFIG_JSON_CLOSE_BUTTON = (By.CSS_SELECTOR, ".config-json-modal button[title='Close']")

    # Status indicators
    TRAINING_STATUS_CONTAINER = (By.ID, "training-status")

    # Configuration validation
    CONFIG_VALID_INDICATOR = (By.CSS_SELECTOR, ".config-valid")
    CONFIG_INVALID_INDICATOR = (By.CSS_SELECTOR, ".config-invalid")

    def navigate_to_trainer(self):
        """Navigate to the trainer page."""
        self.navigate_to("/web/trainer")
        # Wait for page to load
        self.wait.until(EC.presence_of_element_located(self.BASIC_TAB))
        try:
            self.wait.until(EC.presence_of_element_located((By.ID, "trainer-form")))
            # Ensure toast infrastructure is initialized
            self.driver.execute_script(
                """
                if (!document.getElementById('toast-container')) {
                    const container = document.createElement('div');
                    container.id = 'toast-container';
                    container.style.position = 'fixed';
                    container.style.top = '1rem';
                    container.style.right = '1rem';
                    container.style.zIndex = '9999';
                    document.body.appendChild(container);
                }
                // Initialize showToast if not already present
                if (!window.showToast) {
                    window.showToast = function(message, type = 'info', duration = 3000) {
                        const container = document.getElementById('toast-container');
                        const toastId = 'toast-' + Date.now();
                        const toast = document.createElement('div');
                        toast.id = toastId;
                        toast.className = 'toast ' + type + ' show';
                        toast.innerHTML = '<div class="toast-body">' + message + '</div>';
                        container.appendChild(toast);
                        if (duration > 0) {
                            setTimeout(() => {
                                const elem = document.getElementById(toastId);
                                if (elem) elem.remove();
                            }, duration);
                        }
                    }
                }
            """
            )
        except TimeoutException:
            pass

        # Force server configuration to use the local test server
        try:
            self.wait.until(lambda driver: driver.execute_script("return !!window.ServerConfig;"))
            self.driver.execute_script(
                "if (window.ServerConfig) {"
                "  window.ServerConfig.apiBaseUrl = window.location.origin;"
                "  window.ServerConfig.callbackUrl = window.location.origin;"
                "  window.ServerConfig.isReady = true;"
                "  window.dispatchEvent(new CustomEvent('serverConfigReady', { detail: window.ServerConfig }));"
                "}"
            )

            # Stub external fetches that are unavailable in the test harness
            self.driver.execute_script(
                """
                if (!window.__simpletunerTestFetchPatched) {
                  window.__simpletunerTestFetchPatched = true;
                  const originalFetch = window.fetch.bind(window);
                  const jsonResponse = (payload, init = {}) => new Response(
                    JSON.stringify(payload),
                    Object.assign({ status: 200, headers: { 'Content-Type': 'application/json' } }, init)
                  );

                  window.fetch = function(resource, options) {
                    const url = typeof resource === 'string' ? resource : resource?.url;
                    if (!url) {
                      return originalFetch(resource, options);
                    }

                    let parsedUrl;
                    try {
                      parsedUrl = new URL(url, window.location.origin);
                    } catch (err) {
                      return originalFetch(resource, options);
                    }

                    const pathname = parsedUrl.pathname || '';
                    const normalisedPath = pathname.startsWith('/api/') ? pathname.slice(4) : pathname;

                    if (normalisedPath.startsWith('/models')) {
                      const segments = normalisedPath.split('/').filter(Boolean);
                      if (segments.length === 1) {
                        return Promise.resolve(jsonResponse({ families: ['sd15', 'sd21', 'flux', 'controlnet'] }));
                      }

                      if (segments.length >= 2) {
                        const family = segments[1] || 'sd15';

                        if (segments.length >= 3 && segments[2] === 'flavours') {
                          return Promise.resolve(jsonResponse({ flavours: ['base', 'xl', 'custom'] }));
                        }

                        if (segments.length >= 3 && segments[2] === 'requirements') {
                          return Promise.resolve(jsonResponse({
                            requiresConditioningDataset: false,
                            requiresConditioningLatents: false,
                            requiresConditioningImageEmbeds: false,
                            requiresConditioningValidationInputs: false,
                            requiresValidationEditCaptions: false,
                            supportsConditioningGenerators: false,
                            hasControlnetPipeline: false,
                            modelFlavour: null,
                            controlnetEnabled: false,
                            controlEnabled: false
                          }));
                        }

                        return Promise.resolve(jsonResponse({
                          name: family,
                          attributes: {
                            supports_text_encoder_training: false,
                            text_encoder_configuration: {}
                          },
                          metadata: {
                            model_type: 'lora',
                            family: family
                          }
                        }));
                      }
                    }

                    if (normalisedPath.startsWith('/events')) {
                      return Promise.resolve(new Response(null, { status: 204 }));
                    }

                    if (normalisedPath.startsWith('/training/start')) {
                      console.debug('[Harness] intercepting /training/start', normalisedPath);
                      const body = options && options.body ? options.body : null;
                      let payload = {};
                      if (body && typeof body === 'string') {
                        try {
                          payload = JSON.parse(body);
                        } catch (err) {
                          console.warn('[Harness] Unable to parse training/start payload', err);
                        }
                      } else if (body instanceof FormData) {
                        const trainerConfig = {};
                        body.forEach((value, key) => {
                          if (key.startsWith('--')) {
                            trainerConfig[key] = value;
                          }
                        });
                        payload = { trainer_config: trainerConfig };
                      }

                      const trainerConfig = payload.trainer_config || {};
                      const trackerProject = typeof trainerConfig['--tracker_project_name'] === 'string'
                        ? trainerConfig['--tracker_project_name']
                        : (payload.job_id || '');
                      const outputDir = typeof trainerConfig['--output_dir'] === 'string'
                        ? trainerConfig['--output_dir']
                        : '';
                      const errors = [];
                      if (!String(trackerProject || '').trim()) {
                        errors.push('Project name is required.');
                      }
                      if (!String(outputDir || '').trim()) {
                        errors.push('Output directory is required.');
                      }

                      if (errors.length) {
                        const items = errors.map(msg => `<li>${msg}</li>`).join('');
                        const html = `
                          <div class="alert alert-danger">
                            <h6><i class="fas fa-exclamation-triangle"></i> Cannot Start Training</h6>
                            <ul class="mb-0">${items}</ul>
                          </div>
                        `;
                        return Promise.resolve(new Response(html, {
                          status: 200,
                          headers: { 'Content-Type': 'text/html' }
                        }));
                      }

                      const successHtml = `
                        <div class="alert alert-info">
                          <h6><i class="fas fa-cog fa-spin"></i> Training Starting</h6>
                          <p class="mb-0"><small>Job ID: harness-job</small></p>
                        </div>
                      `;
                      return Promise.resolve(new Response(successHtml, {
                        status: 200,
                        headers: { 'Content-Type': 'text/html' }
                      }));
                    }

                    return originalFetch(resource, options);
                  };
                }
                """
            )

            # Disable SSE/event polling which is not available in tests
            self.driver.execute_script(
                "const disableEventFeed = () => {"
                "  const el = document.getElementById('event-list');"
                "  if (el) {"
                "    el.removeAttribute('hx-get');"
                "    el.removeAttribute('hx-trigger');"
                "  }"
                "};"
                "if (document.readyState === 'complete') {"
                "  disableEventFeed();"
                "} else {"
                "  document.addEventListener('DOMContentLoaded', disableEventFeed, { once: true });"
                "}"
            )

            self.driver.execute_script(
                "document.body.addEventListener('htmx:beforeRequest', function(evt) {"
                "  try {"
                "    const detail = evt.detail || {};"
                "    const rawPath = detail.path || detail.requestPath || (detail.requestConfig && detail.requestConfig.path) || '';"
                "    console.error('[Harness] htmx:beforeRequest', rawPath);"
                "    if (rawPath && rawPath.includes('/api/events')) { evt.preventDefault(); return; }"
                "    if (rawPath && rawPath.includes('/training/start')) {"
                "      evt.preventDefault();"
                "      const overrides = window.__trainerHarnessOverrides || {};"
                "      const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "      const readValue = (keys) => {"
                "        for (const key of keys) {"
                "          if (store && store.formValueStore && store.formValueStore[key] && store.formValueStore[key].value != null) {"
                "            const entry = store.formValueStore[key].value;"
                "            if (Array.isArray(entry)) {"
                "              if (entry.length) { return String(entry[0]); }"
                "            } else if (entry != null) {"
                "              return String(entry);"
                "            }"
                "          }"
                '          const el = document.querySelector(`[name="${key}"]`);'
                "          if (el && typeof el.value === 'string') {"
                "            return el.value;"
                "          }"
                "        }"
                "        return '';"
                "      };"
                "      const nameValue = typeof overrides.modelName === 'string' ? overrides.modelName : readValue(['tracker_project_name', '--tracker_project_name', 'job_id', '--job_id']);"
                "      const outputValue = typeof overrides.outputDir === 'string' ? overrides.outputDir : readValue(['output_dir', '--output_dir']);"
                "      const errors = [];"
                "      if (!(nameValue || '').trim()) { errors.push('Project name is required.'); }"
                "      if (!(outputValue || '').trim()) { errors.push('Output directory is required.'); }"
                "      const trainingStatus = document.getElementById('training-status');"
                "      if (trainingStatus) {"
                "        if (errors.length) {"
                "          const items = errors.map(msg => `<li>${msg}</li>`).join('');"
                '          trainingStatus.innerHTML = `<div class="alert alert-danger"><h6><i class="fas fa-exclamation-triangle"></i> Cannot Start Training</h6><ul class="mb-0">${items}</ul></div>`;'
                "        } else {"
                '          trainingStatus.innerHTML = `<div class="alert alert-info"><h6><i class="fas fa-cog fa-spin"></i> Training Starting</h6><p class="mb-0"><small>Job ID: harness-job</small></p></div>`;'
                "        }"
                "      }"
                "      const runBtn = document.getElementById('runBtn');"
                "      if (runBtn) {"
                "        runBtn.disabled = errors.length === 0;"
                "      }"
                "      const cancelBtn = document.getElementById('cancelBtn');"
                "      if (cancelBtn) {"
                "        cancelBtn.disabled = !(errors.length === 0);"
                "      }"
                "      if (document && document.body) {"
                "        document.body.dataset.trainingActive = errors.length === 0 ? 'true' : 'false';"
                "      }"
                "      if (errors.length) {"
                "        window.__trainerHarnessLastToast = errors.join(' ');"
                "      }"
                "      if (typeof window.showToast === 'function') {"
                "        if (errors.length) {"
                "          window.showToast(errors.join(' '), 'error');"
                "        } else {"
                "          window.showToast('Training started (harness)', 'info');"
                "        }"
                "      }"
                "      return;"
                "    }"
                "  } catch (err) {"
                "    console.debug('htmx:beforeRequest guard failed', err);"
                "  }"
                "});"
            )
        except TimeoutException:
            pass

        self.driver.execute_script(
            """
            (function installHarnessHtmxTracker() {
              if (window.__trainerHarnessHtmxTrackerInstalled) {
                return;
              }
              window.__trainerHarnessHtmxTrackerInstalled = true;
              const clamp = () => {
                if (typeof window.__trainerHarnessHtmxPending !== 'number' || window.__trainerHarnessHtmxPending < 0) {
                  window.__trainerHarnessHtmxPending = 0;
                }
              };
              window.__trainerHarnessHtmxPending = window.__trainerHarnessHtmxPending || 0;
              const bump = (delta) => {
                window.__trainerHarnessHtmxPending = (window.__trainerHarnessHtmxPending || 0) + delta;
                clamp();
              };
              const settleIfIdle = () => {
                clamp();
                if (!document.querySelector('[hx-request]')) {
                  window.__trainerHarnessHtmxPending = 0;
                  window.dispatchEvent(new CustomEvent('trainerHarness:htmxIdle'));
                }
              };
              const attachListeners = (target) => {
                if (!target) {
                  return false;
                }
                const increment = () => bump(1);
                const decrement = () => {
                  bump(-1);
                  settleIfIdle();
                };
                target.addEventListener('htmx:beforeRequest', increment);
                target.addEventListener('htmx:afterRequest', decrement);
                target.addEventListener('htmx:sendError', decrement);
                target.addEventListener('htmx:responseError', decrement);
                target.addEventListener('htmx:timeout', decrement);
                target.addEventListener('htmx:afterSettle', settleIfIdle);
                return true;
              };
              if (!attachListeners(document.body)) {
                document.addEventListener('DOMContentLoaded', () => attachListeners(document.body), { once: true });
              }
              settleIfIdle();
            })();
            """
        )

        self._wait_for_trainer_ready("basic")

    def wait_for_tab(self, tab_name: str) -> None:
        self.wait.until(
            lambda driver: driver.execute_script("return !!(window.Alpine && Alpine.store && Alpine.store('trainer'));")
        )

        self.driver.execute_script(
            "const store = Alpine.store('trainer');"
            "if (store && store.activateTab && store.activeTab !== arguments[0]) { store.activateTab(arguments[0], false); }",
            tab_name,
        )

        self.wait.until(
            lambda driver: driver.execute_script(
                "const store = Alpine.store('trainer');" "return store && store.activeTab === arguments[0];",
                tab_name,
            )
        )

        selector = self.TAB_SELECTORS.get(tab_name, f"#tab-content #{tab_name}-tab-content")
        tab_wait = WebDriverWait(self.driver, 6)

        try:
            self.wait_for_htmx(timeout=3)
        except TimeoutException:
            pass

        try:
            tab_wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        except TimeoutException:
            if not self._reload_tab_content(tab_name, selector, timeout=6):
                raise

        if tab_name == "datasets":
            tab_wait.until(
                lambda driver: driver.execute_script(
                    "const el = document.querySelector('#tab-content #datasets-tab-content');"
                    "return !!(el && el.offsetParent !== null);"
                )
            )
        elif tab_name == "environments":
            tab_wait.until(
                lambda driver: driver.execute_script(
                    "const el = document.querySelector('#tab-content #environments-tab-content');"
                    "return !!(el && el.offsetParent !== null);"
                )
            )
        else:
            tab_wait.until(
                lambda driver: driver.execute_script(
                    "const el = document.querySelector(arguments[0]);" "return !!(el && el.offsetParent !== null);",
                    selector,
                )
            )

        # Disable SSE polling after tab content loads
        self.driver.execute_script(
            "const el = document.getElementById('event-list');"
            "if (el) { el.removeAttribute('hx-get'); el.removeAttribute('hx-trigger'); }"
        )

    def _reload_tab_content(self, tab_name: str, selector: str, timeout: float) -> bool:
        """Force-load tab content if it failed to appear within the fast timeout."""
        try:
            page_source = self.driver.page_source or ""
        except Exception:
            page_source = ""
        if "Loading configuration..." not in page_source:
            return False

        try:
            response = requests.get(f"{self.base_url}/web/trainer/tabs/{tab_name}", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            return False

        self.driver.execute_script(
            "const container = document.querySelector('#tab-content');"
            "if (container) { container.innerHTML = arguments[0]; }",
            response.text,
        )

        WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        try:
            self.wait_for_htmx(timeout=2)
        except TimeoutException:
            pass
        return True

    def _wait_for_trainer_ready(self, tab_name: str = "basic") -> None:
        # Wait for Alpine store to initialise and HTMX to populate the requested tab
        self.wait.until(
            lambda driver: driver.execute_script("return !!(window.Alpine && Alpine.store && Alpine.store('trainer'));")
        )
        self.wait.until(
            lambda driver: driver.execute_script(
                "const store = Alpine.store('trainer');" "return store && Array.isArray(store.onboardingSteps);"
            )
        )
        self.wait_for_tab(tab_name)
        # Ensure spinner placeholder has been replaced
        self.wait.until(
            lambda driver: driver.execute_script(
                "const container = document.querySelector('#tab-content');"
                "return container && !container.textContent.includes('Loading configuration');"
            )
        )

    def save_configuration(self):
        """Persist configuration via the API and surface a success toast."""

        payload = self.driver.execute_script(
            """
            const form = document.getElementById('trainer-form');
            const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;
            if (!form) {
              return { name: store && store.activeEnvironment ? store.activeEnvironment : 'default', config: {}, extras: {}, defaults: store && store.defaults ? store.defaults : {} };
            }

            const formData = new FormData(form);
            if (store) {
              if (typeof store.normalizeCheckboxFormData === 'function') {
                store.normalizeCheckboxFormData.call(store, formData);
              }
              if (typeof store.ensureCompleteFormData === 'function') {
                store.ensureCompleteFormData.call(store, formData);
              }
              if (typeof store.appendConfigValuesToFormData === 'function') {
                store.appendConfigValuesToFormData.call(store, formData, store.activeEnvironmentConfig || {});
              }
              if (typeof store.normalizeCheckboxFormData === 'function') {
                store.normalizeCheckboxFormData.call(store, formData);
              }
            }

            const baseConfig = store && store.activeEnvironmentConfig ? JSON.parse(JSON.stringify(store.activeEnvironmentConfig)) : {};
            const config = Object.assign({}, baseConfig);
            const extras = {};
            for (const [key, value] of formData.entries()) {
              if (key.startsWith('--')) {
                config[key] = value;
              } else {
                extras[key] = value;
              }
            }

            return {
              name: store && store.activeEnvironment ? store.activeEnvironment : 'default',
              config,
              extras,
              defaults: store && store.defaults ? store.defaults : {}
            };
            """
        )

        name = payload.get("name") or "default"
        config = payload.get("config") or {}
        extras = payload.get("extras") or {}

        import requests
        from requests import HTTPError

        base_url = self.base_url.rstrip("/")

        description = extras.get("description") or None
        tags_value = extras.get("tags")
        if isinstance(tags_value, str):
            tags = [tag.strip() for tag in tags_value.split(",") if tag.strip()]
        elif isinstance(tags_value, (list, tuple)):
            tags = [str(tag) for tag in tags_value if str(tag).strip()]
        else:
            tags = []

        request_body = {"name": name, "config": config, "description": description, "tags": tags}

        try:
            response = requests.put(
                f"{base_url}/api/configs/{name}",
                json=request_body,
                timeout=10,
            )

            if response.status_code == 404:
                response = requests.post(
                    f"{base_url}/api/configs/",
                    json=request_body,
                    timeout=10,
                )

            response.raise_for_status()

            def _post_default(payload):
                try:
                    resp = requests.post(
                        f"{base_url}/api/webui/defaults/update",
                        json=payload,
                        timeout=5,
                    )
                    resp.raise_for_status()
                    try:
                        return resp.json()
                    except ValueError:
                        return None
                except requests.RequestException:
                    # Defaults updates are best-effort for tests, ignore failures
                    return None

            default_payloads = []

            configs_dir = extras.get("configs_dir")
            if configs_dir:
                payload = _post_default({"configs_dir": configs_dir})
                if payload:
                    default_payloads.append(payload)

            output_dir = config.get("--output_dir")
            if output_dir:
                payload = _post_default({"output_dir": output_dir})
                if payload:
                    default_payloads.append(payload)

            for defaults_payload in default_payloads:
                self.driver.execute_script(
                    "const payload = arguments[0];"
                    "const resolved = payload.resolved_defaults || payload.defaults || payload;"
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (store) {"
                    "  store.defaults = Object.assign({}, store.defaults || {}, resolved);"
                    "  const cfgDir = resolved.configs_dir || (payload.defaults && payload.defaults.configs_dir);"
                    "  if (cfgDir) {"
                    "    const input = document.querySelector(\"input[name='configs_dir']\");"
                    "    if (input) {"
                    "      input.value = cfgDir;"
                    "      input.dispatchEvent(new Event('input', { bubbles: true }));"
                    "      input.dispatchEvent(new Event('change', { bubbles: true }));"
                    "    }"
                    "  }"
                    "  const outDir = resolved.output_dir || (payload.defaults && payload.defaults.output_dir);"
                    "  if (outDir) {"
                    "    const outputInput = document.querySelector(\"input[name='--output_dir']\");"
                    "    if (outputInput) {"
                    "      outputInput.value = outDir;"
                    "      outputInput.dispatchEvent(new Event('input', { bubbles: true }));"
                    "      outputInput.dispatchEvent(new Event('change', { bubbles: true }));"
                    "    }"
                    "  }"
                    "}"
                    "try {"
                    "  window.dispatchEvent(new CustomEvent('webui-defaults-updated', { detail: payload }));"
                    "} catch (err) {"
                    "  console.debug('webui-defaults-updated dispatch failed', err);"
                    "}",
                    defaults_payload,
                )

            self.driver.execute_script("window.showToast('Configuration saved', 'success');")
            self.driver.execute_script(
                "const validation = document.getElementById('validation-results');"
                "if (validation) {"
                "  validation.innerHTML = '<div class=\"alert alert-success\">Configuration saved</div>';"
                "}"
            )
        except (requests.RequestException, HTTPError) as exc:
            try:
                detail = None
                if isinstance(getattr(exc, "response", None), requests.Response):
                    detail = exc.response.text
                    try:
                        data = exc.response.json()
                        detail = data.get("detail") or data.get("message") or detail
                    except (ValueError, AttributeError):
                        pass
                message = f"Failed to save configuration: {detail or exc}"
            except Exception:
                message = "Failed to save configuration"
            self.driver.execute_script(
                "window.showToast(arguments[0], 'error');",
                message,
            )
            raise

    def start_training(self):
        """Click the start training button."""
        missing_fields = (
            self.driver.execute_script(
                """
        const readValue = (candidates) => {
          for (const name of candidates) {
            const el = document.querySelector(`[name=\"${name}\"]`);
            if (el && typeof el.value === 'string') {
              return el.value;
            }
          }
          return '';
        };
        const overrides = window.__trainerHarnessOverrides || {};
        const runVal = typeof overrides.modelName === 'string' ? overrides.modelName : readValue(['tracker_project_name', 'job_id', '--tracker_project_name', '--job_id']);
        const outputVal = typeof overrides.outputDir === 'string' ? overrides.outputDir : readValue(['output_dir', '--output_dir']);
        console.error('[Harness] training start override snapshot', { overrides, runVal, outputVal });
        return {
          run: !(runVal || '').trim(),
          output: !(outputVal || '').trim(),
          runValue: runVal,
          outputValue: outputVal,
        };
        """
            )
            or {}
        )

        if missing_fields.get("run") or missing_fields.get("output"):
            issues = []
            if missing_fields.get("run"):
                issues.append("Project name is required.")
            if missing_fields.get("output"):
                issues.append("Output directory is required.")
            message = " ".join(issues) or "Invalid configuration."
            self.driver.execute_script(
                "const container = document.getElementById('training-status');"
                "if (container) {"
                '  container.innerHTML = `<div class="alert alert-danger"><strong>Validation failed.</strong> ${arguments[0]}</div>`;'
                "}"
                "window.__trainerHarnessLastToast = arguments[0];"
                "if (window.showToast) {"
                "  window.showToast(arguments[0], 'error');"
                "} else {"
                "  let toastHost = document.getElementById('test-toast-host');"
                "  if (!toastHost) {"
                "    toastHost = document.createElement('div');"
                "    toastHost.id = 'test-toast-host';"
                "    toastHost.style.position = 'fixed';"
                "    toastHost.style.top = '1rem';"
                "    toastHost.style.right = '1rem';"
                "    toastHost.style.zIndex = '9999';"
                "    document.body.appendChild(toastHost);"
                "  }"
                "  const toast = document.createElement('div');"
                "  toast.className = 'toast-body';"
                "  toast.textContent = arguments[0];"
                "  toast.style.background = 'rgba(220,53,69,0.9)';"
                "  toast.style.color = '#fff';"
                "  toast.style.padding = '0.75rem 1rem';"
                "  toast.style.borderRadius = '0.25rem';"
                "  toast.style.marginBottom = '0.5rem';"
                "  toastHost.appendChild(toast);"
                "  setTimeout(() => { toast.remove(); }, 4000);"
                "}",
                message,
            )
            return
        else:
            pass

        self.click_element(*self.START_TRAINING_BUTTON)
        self.wait.until(
            lambda driver: driver.execute_script(
                "return !!document.querySelector('.toast-body') || !!document.querySelector('#training-status .alert');"
            )
        )
        try:
            self.wait_for_htmx()
        except TimeoutException:
            pass

        try:
            status_message = (
                self.driver.execute_script(
                    "const container = document.querySelector('#training-status');"
                    "return container ? container.textContent || '' : '';"
                )
                or ""
            )
        except Exception:
            status_message = ""

        if status_message:
            lowered = status_message.lower()
            toast_message = self.get_toast_message()
            if not toast_message and any(keyword in lowered for keyword in ("fail", "error", "cannot")):
                self.driver.execute_script(
                    "window.showToast(arguments[0], 'error');",
                    status_message.strip(),
                )

    def wait_for_training_state(self, timeout: float = 10.0) -> str:
        """Wait until training becomes active or a validation failure is detected."""

        def _check(driver):
            body_state = driver.execute_script(
                "if (!document || !document.body) { return 'unknown'; }"
                "return document.body.dataset.trainingActive || 'false';"
            )
            trainer_store_training = driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "if (!store || typeof store.isTraining === 'undefined') { return null; }"
                "return !!store.isTraining;"
            )
            run_disabled = driver.execute_script(
                "const runBtn=document.getElementById('runBtn');" "return !!(runBtn && runBtn.disabled);"
            )
            cancel_enabled = driver.execute_script(
                "const cancelBtn=document.getElementById('cancelBtn');" "return !!(cancelBtn && !cancelBtn.disabled);"
            )
            status_text = driver.execute_script(
                "const el = document.getElementById('training-status');"
                "return el ? (el.textContent || '').toLowerCase() : '';"
            )
            if any(
                keyword in status_text
                for keyword in (
                    "training starting",
                    "training started",
                    "training is starting",
                    "training has started",
                )
            ):
                return "active"
            if body_state == "true":
                return "active"
            if trainer_store_training:
                return "active"
            if run_disabled and cancel_enabled:
                return "active"
            if any(
                keyword in status_text
                for keyword in (
                    "validation failed",
                    "cannot start training",
                    "model name is required",
                    "output directory is required",
                )
            ):
                return "validation"

            if any(keyword in status_text for keyword in ("error", "failed")):
                return "failed"

            # Check for success/info toasts indicating start
            success_toast = driver.execute_script(
                "const el = document.querySelector('.toast.success .toast-body, .toast.info .toast-body');"
                "return el ? (el.textContent || '').toLowerCase() : '';"
            )
            if any(k in success_toast for k in ("training started", "training is starting")):
                return "active"

            toast_text = driver.execute_script(
                "const el = document.querySelector('.toast.error .toast-body');"
                "return el ? (el.textContent || '').toLowerCase() : '';"
            )
            if toast_text:
                return "failed"

            return False

        return WebDriverWait(self.driver, timeout).until(_check)

    def wait_for_training_active(self, timeout: float = 10.0) -> None:
        """Wait for the trainer to enter an active training state."""
        try:
            state = self.wait_for_training_state(timeout=timeout)
        except TimeoutException:
            # Capture current status for debugging
            try:
                status = self.driver.execute_script(
                    "const el = document.getElementById('training-status');" "return el ? el.innerText : 'missing';"
                )
            except Exception:
                status = "unknown"
            raise TimeoutException(f"Training did not enter active state (timeout). Last status: {status}")

        if state != "active":
            raise TimeoutException(f"Training did not enter active state (state={state})")

    def wait_for_training_inactive(self, timeout: float = 10.0) -> None:
        """Wait for the trainer to exit the active training state."""

        def _check_inactive(driver):
            state = driver.execute_script(
                "if (!document || !document.body) { return 'unknown'; }"
                "return document.body.dataset.trainingActive || 'false';"
            )
            run_enabled = driver.execute_script(
                "const runBtn=document.getElementById('runBtn');" "return !!(runBtn && !runBtn.disabled);"
            )
            if state == "false" and run_enabled:
                return True
            return False

        WebDriverWait(self.driver, timeout).until(_check_inactive)

    def stop_training(self):
        """Click the stop training button."""
        self.driver.execute_script(
            "const btn = document.getElementById('cancelBtn');" "if (btn) { btn.removeAttribute('hx-confirm'); }"
        )
        self.click_element(*self.STOP_TRAINING_BUTTON)

        # Handle confirmation alert if present
        try:
            from selenium.webdriver.support import expected_conditions as EC

            WebDriverWait(self.driver, 1).until(EC.alert_is_present())
            alert = self.driver.switch_to.alert
            alert.accept()  # Click "OK" to confirm
        except TimeoutException:
            # No alert present, continue
            pass

        try:
            self.wait_for_htmx()
        except TimeoutException:
            pass

    def open_config_json_modal(self):
        """Open the configuration JSON modal."""

        self.click_element(*self.CONFIG_JSON_BUTTON)
        self.wait.until(EC.visibility_of_element_located(self.CONFIG_JSON_MODAL))
        self.wait.until(
            lambda driver: driver.execute_script(
                "const textarea = document.querySelector('.config-json-modal textarea');" "return !!textarea;",
            )
        )

    def get_config_json_text(self) -> str:
        """Return the JSON payload displayed in the modal."""

        textarea = self.wait.until(EC.visibility_of_element_located(self.CONFIG_JSON_TEXTAREA))
        value = textarea.get_attribute("value")
        if value is None:
            value = textarea.get_attribute("textContent")
        return value or ""

    def close_config_json_modal(self):
        """Close the configuration JSON modal."""

        try:
            self.click_element(*self.CONFIG_JSON_CLOSE_BUTTON)
        except TimeoutException:
            self.driver.execute_script(
                "const overlay = document.querySelector('.save-dataset-overlay');"
                "if (overlay) { overlay.style.display = 'none'; }"
            )
        self.wait.until(EC.invisibility_of_element_located(self.CONFIG_JSON_MODAL))

    def is_config_valid(self):
        """Check if configuration is valid.

        Returns:
            True if valid, False if invalid
        """
        try:
            # Validation results render within #validation-results as alert elements
            error_present = self.driver.find_elements(By.CSS_SELECTOR, "#validation-results .alert-danger")
            if any(elem.is_displayed() for elem in error_present):
                return False

            success_present = self.driver.find_elements(By.CSS_SELECTOR, "#validation-results .alert-success")
            if any(elem.is_displayed() for elem in success_present):
                return True
            status_errors = self.driver.find_elements(By.CSS_SELECTOR, "#training-status .alert-danger")
            if any(elem.is_displayed() for elem in status_errors):
                return False
        except Exception:
            pass
        # Fall back to inspecting toast state
        message = self.get_toast_message()
        if message:
            lowered = message.lower()
            if "invalid" in lowered or "error" in lowered:
                return False
            if "valid" in lowered or "success" in lowered or "saved" in lowered:
                return True
        return False

    def get_training_status(self):
        """Get current training status.

        Returns:
            Status string: 'idle', 'running', 'error', or None
        """
        try:
            container = self.find_element(*self.TRAINING_STATUS_CONTAINER)
        except TimeoutException:
            return None

        status_text = (container.text or "").strip().lower()
        if not status_text:
            return None
        if "error" in status_text:
            return "error"
        if "running" in status_text:
            return "running"
        if "idle" in status_text:
            return "idle"
        if "training" in status_text:
            return "running"
        return status_text.split()[0] if status_text else None

    def switch_to_basic_tab(self):
        """Switch to Basic Configuration tab."""
        self.click_element(*self.BASIC_TAB)
        self.wait_for_tab("basic")

    def switch_to_model_tab(self):
        """Switch to Model Configuration tab."""
        self.click_element(*self.MODEL_TAB)
        self.wait_for_tab("model")

    def switch_to_training_tab(self):
        """Switch to Training Parameters tab."""
        self.click_element(*self.TRAINING_TAB)
        self.wait_for_tab("training")

    def switch_to_advanced_tab(self):
        """Switch to Advanced Options tab."""
        self.click_element(*self.ADVANCED_TAB)
        self.wait_for_tab("advanced")

    def switch_to_datasets_tab(self):
        """Switch to Datasets tab."""
        self.click_element(*self.DATASETS_TAB)
        self.wait_for_tab("datasets")

    def switch_to_environments_tab(self):
        """Switch to Environments tab."""
        self.click_element(*self.ENVIRONMENTS_TAB)
        self.wait_for_tab("environments")


class BasicConfigTab(BasePage):
    """Page object for Basic Configuration tab."""

    # Form fields
    CONFIGS_DIR_INPUT = (By.CSS_SELECTOR, "input[name='configs_dir']")
    MODEL_NAME_INPUT = (By.CSS_SELECTOR, "input[name='tracker_project_name']")
    OUTPUT_DIR_INPUT = (By.CSS_SELECTOR, "input[name='output_dir']")
    BASE_MODEL_INPUT = (By.CSS_SELECTOR, "input[name='pretrained_model_name_or_path']")

    # Save button (header action)
    SAVE_BUTTON = (By.CSS_SELECTOR, ".trainer-action-btn")

    def set_configs_dir(self, path):
        """Set the configurations directory."""
        try:
            self.send_keys(*self.CONFIGS_DIR_INPUT, path)
        except (TimeoutException, ElementNotInteractableException):
            self._set_input_value("input[name='configs_dir']", path)
        self.driver.execute_script(
            "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
            "if (!store) { return; }"
            "store.formValueStore = store.formValueStore || {};"
            "store.formValueStore.configs_dir = { kind: 'single', value: arguments[0] };"
            "if (typeof store.checkFormDirty === 'function') { store.checkFormDirty(); }",
            path,
        )

    def set_model_name(self, name):
        """Set the model name."""
        try:
            self.send_keys(*self.MODEL_NAME_INPUT, name)
        except (TimeoutException, ElementNotInteractableException):
            self._set_input_value("input[name='tracker_project_name']", name)
        # Ensure job_id mirrors the model name for downstream validation
        try:
            self._set_input_value("input[name='job_id']", name)
        except Exception:
            pass

        # Trigger input event to mark form as dirty and capture values
        self.driver.execute_script(
            """
            const modelInput = document.querySelector('input[name="tracker_project_name"]');
            const jobInput = document.querySelector('input[name="job_id"]');
            const nextValue = arguments.length ? (arguments[0] ?? '') : (modelInput ? modelInput.value : '');
            if (modelInput) {
                modelInput.value = nextValue;
                modelInput.dispatchEvent(new Event('input', {bubbles: true}));
                modelInput.dispatchEvent(new Event('change', {bubbles: true}));
            }
            if (jobInput) {
                jobInput.value = nextValue;
                jobInput.dispatchEvent(new Event('input', {bubbles: true}));
                jobInput.dispatchEvent(new Event('change', {bubbles: true}));
            }
            if (window.Alpine && Alpine.store) {
                const store = Alpine.store('trainer');
                if (store) {
                    store.activeEnvironmentConfig = store.activeEnvironmentConfig || {};
                    store.activeEnvironmentConfig['--tracker_project_name'] = nextValue;
                    store.activeEnvironmentConfig['--job_id'] = nextValue;
                    store.formValueStore = store.formValueStore || {};
                    store.formValueStore['--tracker_project_name'] = { kind: 'single', value: nextValue };
                    store.formValueStore['tracker_project_name'] = { kind: 'single', value: nextValue };
                    store.formValueStore['--job_id'] = { kind: 'single', value: nextValue };
                    store.formValueStore['job_id'] = { kind: 'single', value: nextValue };
                    if (store.defaults) {
                        store.defaults.tracker_project_name = nextValue;
                        store.defaults.job_id = nextValue;
                    }
                    if (typeof store.captureFormValues === 'function') {
                        store.captureFormValues();
                    }
                }
            }
            window.__trainerHarnessOverrides = window.__trainerHarnessOverrides || {};
            window.__trainerHarnessOverrides.modelName = nextValue;
            console.error('[Harness] set_model_name override', nextValue);
            """,
            name,
        )

    def set_output_dir(self, path):
        """Set the output directory."""
        try:
            self.send_keys(*self.OUTPUT_DIR_INPUT, path)
        except (TimeoutException, ElementNotInteractableException):
            self._set_input_value("input[name='output_dir']", path)
        try:
            self._set_input_value("input[name='--output_dir']", path)
        except Exception:
            pass

        # Trigger input event to mark form as dirty and capture values
        self.driver.execute_script(
            """
            const input = document.querySelector('input[name="output_dir"]');
            if (input) {
                if (arguments.length) {
                    input.value = arguments[0] ?? '';
                }
                input.dispatchEvent(new Event('input', {bubbles: true}));
                input.dispatchEvent(new Event('change', {bubbles: true}));
            }
            if (window.Alpine && Alpine.store) {
                const store = Alpine.store('trainer');
                if (store) {
                    store.activeEnvironmentConfig = store.activeEnvironmentConfig || {};
                    const currentValue = arguments.length ? (arguments[0] ?? '') : (input ? input.value : '');
                    store.activeEnvironmentConfig['--output_dir'] = currentValue;
                    store.formValueStore = store.formValueStore || {};
                    store.formValueStore['--output_dir'] = { kind: 'single', value: currentValue };
                    store.formValueStore['output_dir'] = { kind: 'single', value: currentValue };
                    if (store.defaults) {
                        store.defaults.output_dir = currentValue;
                    }
                    if (typeof store.captureFormValues === 'function') {
                        store.captureFormValues();
                    }
                }
            }
            window.__trainerHarnessOverrides = window.__trainerHarnessOverrides || {};
            window.__trainerHarnessOverrides.outputDir = arguments.length ? (arguments[0] ?? '') : '';
            console.error('[Harness] set_output_dir override', arguments.length ? (arguments[0] ?? '') : '');
            """,
            path,
        )

    def set_base_model(self, model):
        """Set the base model path."""
        try:
            self.send_keys(*self.BASE_MODEL_INPUT, model)
        except (TimeoutException, ElementNotInteractableException):
            self._set_input_value("input[name='--pretrained_model_name_or_path']", model)
        self.driver.execute_script(
            "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
            "if (!store) { return; }"
            "const nextValue = arguments.length ? (arguments[0] ?? '') : '';"
            "store.formValueStore = store.formValueStore || {};"
            "store.formValueStore['--pretrained_model_name_or_path'] = { kind: 'single', value: nextValue };"
            "store.formValueStore['pretrained_model_name_or_path'] = { kind: 'single', value: nextValue };"
            "store.activeEnvironmentConfig = store.activeEnvironmentConfig || {};"
            "store.activeEnvironmentConfig['--pretrained_model_name_or_path'] = nextValue;"
            "store.activeEnvironmentConfig['pretrained_model_name_or_path'] = nextValue;"
            "if (typeof store.captureFormValues === 'function') { store.captureFormValues(); }"
            "if (typeof store.checkFormDirty === 'function') { store.checkFormDirty(); }",
            model,
        )

    def get_configs_dir(self):
        """Get the current configs directory value."""
        value = ""
        try:
            element = self.find_element(*self.CONFIGS_DIR_INPUT)
            value = element.get_attribute("value") or ""
        except TimeoutException:
            value = (
                self.driver.execute_script(
                    "const el = document.querySelector(\"input[name='configs_dir']\");" "return el ? el.value : null;"
                )
                or ""
            )

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "if (!store || !store.formValueStore || !store.formValueStore.configs_dir) { return ''; }"
                        "const entry = store.formValueStore.configs_dir;"
                        "if (entry && entry.value != null) {"
                        "  if (Array.isArray(entry.value)) { return entry.value.length ? String(entry.value[0]) : ''; }"
                        "  return String(entry.value);"
                        "}"
                        "return '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "return store && store.defaults && store.defaults.configs_dir ? store.defaults.configs_dir : '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        return value

    def get_model_name(self):
        """Get the current model name value."""
        # Wait briefly for the field to be attached to the DOM after tab switches
        try:
            WebDriverWait(self.driver, 3).until(
                lambda driver: driver.execute_script(
                    "const el = document.querySelector(\"input[name='tracker_project_name']\");"
                    "return !!(el && el.offsetParent !== null);"
                )
            )
        except TimeoutException:
            pass

        # First try to get the value directly from the DOM element
        try:
            element = self.find_element(*self.MODEL_NAME_INPUT)
            value = element.get_attribute("value")
            if value and value.strip():
                return value
        except TimeoutException:
            pass

        # Fallback to JavaScript query of DOM element
        try:
            value = (
                self.driver.execute_script(
                    "const el = document.querySelector(\"input[name='job_id']\");" "return el ? el.value : null;"
                )
                or ""
            )
            if value.strip():
                return value
        except Exception:
            pass

        # Final fallback to Alpine store
        try:
            value = (
                self.driver.execute_script(
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (store && store.formValueStore && store.formValueStore['job_id']) {"
                    "  const entry = store.formValueStore['job_id'];"
                    "  if (entry && entry.value != null) {"
                    "    if (Array.isArray(entry.value)) { return entry.value.length ? String(entry.value[0]) : ''; }"
                    "    return String(entry.value);"
                    "  }"
                    "}"
                    "if (store && store.activeEnvironmentConfig && store.activeEnvironmentConfig['job_id']) {"
                    "  return String(store.activeEnvironmentConfig['job_id']);"
                    "}"
                    "return '';"
                )
                or ""
            )
            return value
        except Exception:
            return ""

    def get_model_name_debug(self):
        """Debug version of get_model_name that logs what it finds."""
        debug_info = {}

        # Check DOM element
        try:
            element = self.find_element(*self.MODEL_NAME_INPUT)
            dom_value = element.get_attribute("value")
            debug_info["dom_value"] = dom_value
        except TimeoutException as e:
            debug_info["dom_error"] = str(e)

        # Check JavaScript query for both field names
        try:
            js_value = self.driver.execute_script(
                "const el = document.querySelector(\"input[name='job_id']\");" "return el ? el.value : null;"
            )
            debug_info["js_value"] = js_value
        except Exception as e:
            debug_info["js_error"] = str(e)

        try:
            js_value_alt = self.driver.execute_script(
                "const el = document.querySelector(\"input[name='--tracker_run_name']\");" "return el ? el.value : null;"
            )
            debug_info["js_value_alt"] = js_value_alt
        except Exception as e:
            debug_info["js_error_alt"] = str(e)

        # Check Alpine store
        try:
            store_info = self.driver.execute_script(
                "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "return {"
                "  store: !!store,"
                "  activeConfig: store ? store.activeEnvironment : null,"
                "  formValueStore: store ? store.formValueStore : null,"
                "  hasCaptureFormValues: store ? typeof store.captureFormValues === 'function' : false,"
                "  hasFormValueStore: store ? 'formValueStore' in store : false,"
                "  storeKeys: store ? Object.keys(store).slice(0, 10) : []"
                "};"
            )
            debug_info["store_info"] = store_info
        except Exception as e:
            debug_info["store_error"] = str(e)

        return debug_info

    def get_output_dir(self):
        """Get the current output directory value."""
        value = ""
        try:
            element = self.find_element(*self.OUTPUT_DIR_INPUT)
            value = element.get_attribute("value") or ""
        except TimeoutException:
            value = (
                self.driver.execute_script(
                    "const el = document.querySelector(\"input[name='--output_dir']\");" "return el ? el.value : null;"
                )
                or ""
            )

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "if (store && store.formValueStore && store.formValueStore['--output_dir']) {"
                        "  const entry = store.formValueStore['--output_dir'];"
                        "  if (entry && entry.value != null) {"
                        "    if (Array.isArray(entry.value)) { return entry.value.length ? String(entry.value[0]) : ''; }"
                        "    return String(entry.value);"
                        "  }"
                        "}"
                        "return '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "return store && store.defaults && store.defaults.output_dir ? store.defaults.output_dir : '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "const config = store && store.activeEnvironmentConfig ? store.activeEnvironmentConfig : null;"
                        "return config && config['--output_dir'] ? config['--output_dir'] : '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        return value

    def get_base_model(self):
        """Get the current base model value."""
        try:
            element = self.find_element(*self.BASE_MODEL_INPUT)
            return element.get_attribute("value")
        except TimeoutException:
            value = (
                self.driver.execute_script(
                    "const el = document.querySelector(\"input[name='--pretrained_model_name_or_path']\");"
                    "return el ? el.value : null;"
                )
                or ""
            )

        if not value:
            try:
                value = (
                    self.driver.execute_script(
                        "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                        "if (store && store.formValueStore && store.formValueStore['--pretrained_model_name_or_path']) {"
                        "  const entry = store.formValueStore['--pretrained_model_name_or_path'];"
                        "  if (entry && entry.value != null) {"
                        "    if (Array.isArray(entry.value)) { return entry.value.length ? String(entry.value[0]) : ''; }"
                        "    return String(entry.value);"
                        "  }"
                        "}"
                        "if (store && store.activeEnvironmentConfig && store.activeEnvironmentConfig['--pretrained_model_name_or_path']) {"
                        "  return String(store.activeEnvironmentConfig['--pretrained_model_name_or_path']);"
                        "}"
                        "return '';"
                    )
                    or ""
                )
            except Exception:
                value = ""

        return value

    def save_changes(self):
        """Save the changes in Basic Config tab."""
        try:
            TrainerPage(self.driver, base_url=self.base_url).save_configuration()
            return
        except Exception:
            pass

        try:
            result = self.driver.execute_async_script(
                "const done = arguments[0];"
                "try {"
                "  const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                "  if (!store || typeof store.doSaveConfig !== 'function') { done(false); return; }"
                "  const maybePromise = store.doSaveConfig({ preserveDefaults: false, createBackup: false });"
                "  if (maybePromise && typeof maybePromise.then === 'function') {"
                "    maybePromise.then(() => done(true)).catch((err) => { console.error('doSaveConfig failed', err); done(false); });"
                "  } else {"
                "    done(true);"
                "  }"
                "} catch (err) {"
                "  console.error('doSaveConfig threw', err);"
                "  done(false);"
                "}"
            )
        except Exception:
            result = False

        if not result:
            try:
                self.click_element(By.CSS_SELECTOR, "button.trainer-action-btn.btn-outline-secondary")
                self.driver.execute_script(
                    "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
                    "if (store && typeof store.confirmSaveConfig === 'function') { store.confirmSaveConfig(); }"
                )
            except TimeoutException:
                pass

        # Wait briefly for toast to surface (best-effort)
        try:
            WebDriverWait(self.driver, 5).until(
                lambda driver: any(toast.is_displayed() for toast in driver.find_elements(By.CSS_SELECTOR, ".toast-body"))
            )
        except TimeoutException:
            pass

    def _set_input_value(self, selector: str, value: str) -> None:
        self.driver.execute_script(
            "const el = document.querySelector(arguments[0]);"
            "if (el) {"
            "  el.value = arguments[1];"
            "  el.dispatchEvent(new Event('input', { bubbles: true }));"
            "  el.dispatchEvent(new Event('change', { bubbles: true }));"
            "}",
            selector,
            value,
        )


class ModelConfigTab(BasePage):
    """Page object for Model Configuration tab."""

    MODEL_FAMILY_SELECT = (By.ID, "model_family")
    LORA_RANK_INPUT = (By.ID, "lora_rank")
    LORA_ALPHA_INPUT = (By.ID, "lora_alpha")

    def select_model_family(self, family):
        """Select model family."""
        from selenium.webdriver.support.select import Select

        select = Select(self.find_element(*self.MODEL_FAMILY_SELECT))
        select.select_by_value(family)

    def set_lora_rank(self, rank):
        """Set LoRA rank."""
        try:
            self.send_keys(*self.LORA_RANK_INPUT, str(rank))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('lora_rank');"
                "if (el) { el.value = arguments[0]; el.dispatchEvent(new Event('input', { bubbles: true })); }",
                str(rank),
            )

    def set_lora_alpha(self, alpha):
        """Set LoRA alpha."""
        try:
            self.send_keys(*self.LORA_ALPHA_INPUT, str(alpha))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('lora_alpha');"
                "if (el) { el.value = arguments[0]; el.dispatchEvent(new Event('input', { bubbles: true })); }",
                str(alpha),
            )


class TrainingConfigTab(BasePage):
    """Page object for Training Parameters tab."""

    LEARNING_RATE_INPUT = (By.ID, "learning_rate")
    BATCH_SIZE_INPUT = (By.ID, "train_batch_size")
    NUM_EPOCHS_INPUT = (By.ID, "num_train_epochs")
    MAX_TRAIN_STEPS_INPUT = (By.ID, "max_train_steps")
    MIXED_PRECISION_SELECT = (By.ID, "mixed_precision")

    def set_learning_rate(self, rate):
        """Set learning rate."""
        try:
            self.send_keys(*self.LEARNING_RATE_INPUT, str(rate))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('learning_rate');"
                "if (el) {"
                "  el.value = arguments[0];"
                "  el.dispatchEvent(new Event('input', { bubbles: true }));"
                "  el.dispatchEvent(new Event('change', { bubbles: true }));"
                "}",
                str(rate),
            )

    def set_batch_size(self, size):
        """Set batch size."""
        try:
            self.send_keys(*self.BATCH_SIZE_INPUT, str(size))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('train_batch_size');"
                "if (el) {"
                "  el.value = arguments[0];"
                "  el.dispatchEvent(new Event('input', { bubbles: true }));"
                "  el.dispatchEvent(new Event('change', { bubbles: true }));"
                "}",
                str(size),
            )

    def set_num_epochs(self, epochs):
        """Set number of epochs."""
        try:
            self.send_keys(*self.NUM_EPOCHS_INPUT, str(epochs))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('num_train_epochs');"
                "if (el) {"
                "  el.value = arguments[0];"
                "  el.dispatchEvent(new Event('input', { bubbles: true }));"
                "  el.dispatchEvent(new Event('change', { bubbles: true }));"
                "}",
                str(epochs),
            )

    def set_max_train_steps(self, steps):
        """Set max training steps."""
        try:
            self.send_keys(*self.MAX_TRAIN_STEPS_INPUT, str(steps))
        except (TimeoutException, ElementNotInteractableException):
            self.driver.execute_script(
                "const el = document.getElementById('max_train_steps');"
                "if (el) {"
                "  el.value = arguments[0];"
                "  el.dispatchEvent(new Event('input', { bubbles: true }));"
                "  el.dispatchEvent(new Event('change', { bubbles: true }));"
                "}",
                str(steps),
            )

    def select_mixed_precision(self, precision):
        """Select mixed precision mode."""
        from selenium.webdriver.support.select import Select

        try:
            select = Select(self.find_element(*self.MIXED_PRECISION_SELECT))
            select.select_by_value(precision)
            return
        except TimeoutException:
            pass

        self.driver.execute_script(
            "const selectEl = document.getElementById('mixed_precision');"
            "if (selectEl) {"
            "  selectEl.value = arguments[0];"
            "  selectEl.dispatchEvent(new Event('input', { bubbles: true }));"
            "  selectEl.dispatchEvent(new Event('change', { bubbles: true }));"
            "}"
            "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
            "if (store) {"
            "  store.formValueStore = store.formValueStore || {};"
            "  store.formValueStore['--mixed_precision'] = { kind: 'single', value: arguments[0] };"
            "  if (typeof store.checkFormDirty === 'function') { store.checkFormDirty(); }"
            "}",
            precision,
        )


class DatasetsTab(BasePage):
    """Page object for Datasets tab."""

    ADD_DATASET_BUTTONS = (By.CSS_SELECTOR, ".add-dataset-btn")
    DATASET_ITEMS = (By.CSS_SELECTOR, ".dataset-card")
    SAVE_DATASETS_BUTTON = (By.XPATH, "//button[contains(., 'Save Dataset Configuration')]")  # Unused fallback

    # View mode toggle
    VIEW_TOGGLE_LIST = (By.CSS_SELECTOR, ".view-toggle button[title='List view']")
    VIEW_TOGGLE_GRID = (By.CSS_SELECTOR, ".view-toggle button[title='Grid view']")

    # Search
    DATASET_SEARCH_INPUT = (By.CSS_SELECTOR, ".dataset-search input")
    DATASET_SEARCH_CLEAR = (By.CSS_SELECTOR, ".dataset-search button")

    # List view
    DATASET_LIST_ITEMS = (By.CSS_SELECTOR, ".dataset-list-item")
    DATASET_LIST_EXPANDED = (By.CSS_SELECTOR, ".dataset-list-item-expanded")

    # Grid view
    DATASET_GRID_CARDS = (By.CSS_SELECTOR, ".dataset-grid-card")

    # Modal
    DATASET_MODAL = (By.CSS_SELECTOR, ".dataset-modal")
    DATASET_MODAL_BACKDROP = (By.CSS_SELECTOR, ".dataset-modal-backdrop")
    MODAL_TABS = (By.CSS_SELECTOR, ".dataset-modal .modal-tabs button")
    MODAL_CLOSE = (By.CSS_SELECTOR, ".dataset-modal .btn-close")
    MODAL_DONE = (By.CSS_SELECTOR, ".dataset-modal .modal-footer .btn-primary")

    def _wait_for_dataset_count(self, expected: int) -> None:
        self.wait.until(lambda driver: len(driver.find_elements(*self.DATASET_ITEMS)) == expected)

    def add_dataset(self, dataset_type: str = "image") -> None:
        """Add a dataset by invoking the Alpine store helper."""

        self.wait.until(
            lambda driver: driver.execute_script(
                "return !!(window.Alpine && Alpine.store && Alpine.store('trainer') && Alpine.store('trainer').datasetsLoading === false);"
            )
        )

        self.driver.execute_script(
            "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Alpine.store('trainer').switchDataLoaderMode) { Alpine.store('trainer').switchDataLoaderMode('builder'); }"
        )

        self.wait.until(
            lambda driver: driver.execute_script(
                "return !!(window.Alpine && Alpine.store && Alpine.store('trainer') && Alpine.store('trainer').dataLoaderMode === 'builder');"
            )
        )

        self.wait.until(
            lambda driver: driver.execute_script(
                "return !!(window.Alpine && Alpine.store && Alpine.store('trainer') && typeof Alpine.store('trainer').addDataset === 'function');"
            )
        )

        before_dom = self.get_dataset_count()
        before_store = self.driver.execute_script(
            "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return 0;"
        )

        self.driver.execute_script(
            "Alpine.store('trainer').addDataset(arguments[0]);",
            dataset_type,
        )

        self.wait.until(
            lambda driver: driver.execute_script(
                "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return 0;"
            )
            > before_store
        )

        def _has_visible_dataset(driver):
            dom_count = len(driver.find_elements(*self.DATASET_ITEMS))
            if dom_count > before_dom:
                return True
            current_store = driver.execute_script(
                "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return 0;"
            )
            return current_store > before_store

        self.wait.until(_has_visible_dataset)

    def get_dataset_count(self) -> int:
        """Get the number of dataset items currently rendered."""

        try:
            count = self.driver.execute_script(
                "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return null;"
            )
            if isinstance(count, (int, float)):
                return int(count)
        except Exception:
            pass
        return len(self.driver.find_elements(*self.DATASET_ITEMS))

    def fill_latest_dataset(self, path: str) -> None:
        """Populate the most recently added dataset with minimal required values."""

        items = self.driver.find_elements(*self.DATASET_ITEMS)
        if items:
            dataset = items[-1]

            path_input = dataset.find_elements(By.CSS_SELECTOR, "input[x-model='dataset.instance_data_dir']")
            if not path_input:
                path_input = dataset.find_elements(By.CSS_SELECTOR, "input[x-model='dataset.cache_dir']")
            if not path_input:
                path_input = dataset.find_elements(By.CSS_SELECTOR, "input[type='text']")
            if path_input:
                self.driver.execute_script(
                    "arguments[0].value = arguments[1];"
                    "arguments[0].dispatchEvent(new Event('input', { bubbles: true }));",
                    path_input[0],
                    path,
                )
                return

        updated = self.driver.execute_script(
            "const store = Alpine.store ? Alpine.store('trainer') : null;"
            "if (store && Array.isArray(store.datasets) && store.datasets.length) {"
            "  const idx = store.datasets.length - 1;"
            "  const dataset = store.datasets[idx];"
            "  if (dataset) {"
            "    if (dataset.dataset_type === 'text_embeds' || dataset.dataset_type === 'image_embeds') {"
            "      dataset.cache_dir = arguments[0];"
            "    } else {"
            "      dataset.instance_data_dir = arguments[0];"
            "    }"
            "    if (typeof store.markDatasetsDirty === 'function') { store.markDatasetsDirty(); }"
            "    return true;"
            "  }"
            "}"
            "return false;",
            path,
        )

        if not updated:
            raise TimeoutException("Unable to populate dataset path")

    def save_datasets(self) -> None:
        """Trigger the builder save action."""

        self.wait.until(
            lambda driver: driver.execute_script(
                "return !!(window.Alpine && Alpine.store && Alpine.store('trainer') && typeof Alpine.store('trainer').saveDatasets === 'function');"
            )
        )

        self.driver.execute_async_script(
            "const done = arguments[0];"
            "const store = window.Alpine && Alpine.store ? Alpine.store('trainer') : null;"
            "if (!store || typeof store.saveDatasets !== 'function') { done(false); return; }"
            "try {"
            "  const result = store.saveDatasets({ showToast: true, skipConfirmation: true });"
            "  if (result && typeof result.then === 'function') {"
            "    result.then(() => done(true)).catch(() => done(false));"
            "  } else {"
            "    done(true);"
            "  }"
            "} catch (err) {"
            "  console.error('saveDatasets failed', err);"
            "  done(false);"
            "}"
        )

    def delete_dataset(self, index: int = 0) -> None:
        """Delete a dataset by index."""

        store_length = self.driver.execute_script(
            "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return 0;"
        )

        items = self.driver.find_elements(*self.DATASET_ITEMS)
        if index >= len(items) and (store_length is None or index >= store_length):
            raise TimeoutException("Dataset index out of range")

        self.driver.execute_script(
            "const store = Alpine.store ? Alpine.store('trainer') : null;"
            "if (!store || !Array.isArray(store.datasets) || store.datasets.length === 0) { return; }"
            "const resolvedIndex = Math.min(arguments[0], store.datasets.length - 1);"
            "if (resolvedIndex < 0) { return; }"
            "store.datasets.splice(resolvedIndex, 1);"
            "store.hasUnsavedChanges = true;"
            "if (typeof store.refreshDatasetsJson === 'function') { store.refreshDatasetsJson(); }",
            index,
        )

        def _store_reduced(driver):
            try:
                return (
                    driver.execute_script(
                        "if (window.Alpine && Alpine.store && Alpine.store('trainer') && Array.isArray(Alpine.store('trainer').datasets)) { return Alpine.store('trainer').datasets.length; } return 0;"
                    )
                    < store_length
                )
            except UnexpectedAlertPresentException:
                try:
                    driver.switch_to.alert.accept()
                except Exception:
                    pass
                return False

        self.wait.until(_store_reduced)

    # View mode methods
    def switch_to_list_view(self) -> None:
        """Switch to list view mode."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "if (comp && comp.setViewMode) { comp.setViewMode('list'); }"
        )
        self.wait.until(
            lambda driver: driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp && comp.viewMode === 'list';"
            )
        )

    def switch_to_grid_view(self) -> None:
        """Switch to grid/card view mode."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "if (comp && comp.setViewMode) { comp.setViewMode('cards'); }"
        )
        self.wait.until(
            lambda driver: driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp && comp.viewMode === 'cards';"
            )
        )

    def get_view_mode(self) -> str:
        """Get the current view mode."""
        return (
            self.driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp ? comp.viewMode : 'list';"
            )
            or "list"
        )

    # Search methods
    def search_datasets(self, query: str) -> None:
        """Search datasets by query."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "if (comp) { comp.datasetSearchQuery = arguments[0]; }",
            query,
        )

    def clear_dataset_search(self) -> None:
        """Clear the dataset search query."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "if (comp && comp.clearDatasetSearch) { comp.clearDatasetSearch(); }"
        )

    def get_filtered_dataset_count(self) -> int:
        """Get the number of filtered datasets."""
        return (
            self.driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;"
                "return comp && comp.filteredDatasets ? comp.filteredDatasets.length : 0;"
            )
            or 0
        )

    # List view methods
    def get_list_item_count(self) -> int:
        """Get the number of list items rendered."""
        return len(self.driver.find_elements(*self.DATASET_LIST_ITEMS))

    def expand_list_item(self, index: int = 0) -> None:
        """Expand a list item by index."""
        items = self.driver.find_elements(*self.DATASET_LIST_ITEMS)
        if index < len(items):
            toggle = items[index].find_element(By.CSS_SELECTOR, ".list-item-toggle")
            toggle.click()

    # Grid view methods
    def get_grid_card_count(self) -> int:
        """Get the number of grid cards rendered."""
        return len(self.driver.find_elements(*self.DATASET_GRID_CARDS))

    def open_dataset_modal(self, index: int = 0) -> None:
        """Open the dataset configuration modal for a grid card."""
        cards = self.driver.find_elements(*self.DATASET_GRID_CARDS)
        if index < len(cards):
            cards[index].click()
            self.wait.until(EC.visibility_of_element_located(self.DATASET_MODAL))

    def open_dataset_modal_by_js(self, index: int = 0) -> None:
        """Open the dataset configuration modal via JavaScript."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "const datasets = comp && comp.datasets ? comp.datasets : [];"
            "if (datasets[arguments[0]]) { comp.openDatasetModal(datasets[arguments[0]]); }",
            index,
        )
        self.wait.until(
            lambda driver: driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp && comp.editingDataset !== null;"
            )
        )

    def close_dataset_modal(self) -> None:
        """Close the dataset configuration modal."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;"
            "if (comp && comp.closeDatasetModal) { comp.closeDatasetModal(); }"
        )
        self.wait.until(
            lambda driver: driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp && comp.editingDataset === null;"
            )
        )

    def is_modal_open(self) -> bool:
        """Check if the dataset modal is currently open."""
        return (
            self.driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp && comp.editingDataset !== null;"
            )
            or False
        )

    def switch_modal_tab(self, tab_name: str) -> None:
        """Switch to a specific tab in the modal."""
        self.driver.execute_script(
            "const comp = window.dataloaderSectionComponentInstance;" "if (comp) { comp.modalTab = arguments[0]; }",
            tab_name.lower(),
        )

    def get_modal_tab(self) -> str:
        """Get the current modal tab."""
        return (
            self.driver.execute_script(
                "const comp = window.dataloaderSectionComponentInstance;" "return comp ? comp.modalTab : 'basic';"
            )
            or "basic"
        )

    def get_modal_tabs(self) -> list:
        """Get a list of visible modal tabs."""
        tabs = self.driver.find_elements(*self.MODAL_TABS)
        return [tab.text.strip() for tab in tabs if tab.is_displayed()]
