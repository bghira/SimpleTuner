"""
Tests for HTMX behavior and interactions.

These tests ensure HTMX requests work correctly and DOM updates happen as expected.
"""

import time

import pytest
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class TestHTMXFormSubmissions:
    """Test HTMX form submission behavior."""

    @pytest.mark.e2e
    def test_htmx_form_submission_includes_all_fields(self, driver):
        """Test that hx-include properly includes all form fields."""
        driver.get("http://localhost:8001/web/trainer")

        # Fill in form fields
        driver.find_element(By.ID, "configs_dir").send_keys("/test/configs")
        driver.find_element(By.ID, "model_name").send_keys("test-model")
        driver.find_element(By.ID, "output_dir").send_keys("/test/output")

        # Intercept HTMX requests
        driver.execute_script(
            """
            window.htmxRequests = [];
            document.body.addEventListener('htmx:configRequest', function(evt) {
                window.htmxRequests.push({
                    path: evt.detail.path,
                    verb: evt.detail.verb,
                    parameters: evt.detail.parameters
                });
            });
        """
        )

        # Find save button and click it
        save_button = driver.find_element(By.CSS_SELECTOR, "button[hx-post*='/api/training/config']")
        save_button.click()

        # Wait a bit for request to be captured
        time.sleep(0.5)

        # Check captured request
        requests = driver.execute_script("return window.htmxRequests;")
        assert len(requests) > 0, "No HTMX requests captured"

        last_request = requests[-1]
        assert last_request["verb"] == "post"
        assert "/api/training/config" in last_request["path"]

        # Check that all fields were included
        params = last_request["parameters"]
        assert "configs_dir" in params
        assert "--job_id" in params  # model_name becomes --job_id
        assert "--output_dir" in params

    @pytest.mark.e2e
    def test_htmx_target_updates_correctly(self, driver):
        """Test that hx-target updates the correct DOM element."""
        driver.get("http://localhost:8001/web/trainer")

        # Set up mutation observer for DOM changes
        driver.execute_script(
            """
            window.domChanges = [];
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                window.domChanges.push({
                                    target: mutation.target.id || mutation.target.className,
                                    added: node.outerHTML
                                });
                            }
                        });
                    }
                });
            });
            observer.observe(document.body, { childList: true, subtree: true });
        """
        )

        # Click save button
        save_button = driver.find_element(By.CSS_SELECTOR, "button[hx-post*='/api/training/config']")
        save_button.click()

        # Wait for response
        time.sleep(1)

        # Check DOM changes
        changes = driver.execute_script("return window.domChanges;")

        # Should have updated something (like a success message)
        assert len(changes) > 0, "No DOM changes detected after HTMX request"

    @pytest.mark.e2e
    def test_htmx_indicator_shows_and_hides(self, driver):
        """Test that hx-indicator shows loading state during request."""
        driver.get("http://localhost:8001/web/trainer")

        # Find a button with hx-indicator
        buttons_with_indicator = driver.find_elements(By.CSS_SELECTOR, "button[hx-indicator]")

        if buttons_with_indicator:
            button = buttons_with_indicator[0]
            indicator_selector = button.get_attribute("hx-indicator")

            # Remove the # from selector
            indicator_id = indicator_selector.lstrip("#")

            # Check indicator is initially hidden
            indicator = driver.find_element(By.ID, indicator_id)
            assert not indicator.is_displayed(), "Indicator visible before request"

            # Set up observer for indicator visibility
            driver.execute_script(
                """
                window.indicatorShown = false;
                const indicator = document.getElementById(arguments[0]);
                if (indicator) {
                    const observer = new MutationObserver(function(mutations) {
                        if (indicator.classList.contains('htmx-request')) {
                            window.indicatorShown = true;
                        }
                    });
                    observer.observe(indicator, {
                        attributes: true,
                        attributeFilter: ['class']
                    });
                }
            """,
                indicator_id,
            )

            # Click button
            button.click()

            # Wait a bit
            time.sleep(0.5)

            # Check if indicator was shown
            indicator_shown = driver.execute_script("return window.indicatorShown;")
            assert indicator_shown, "Indicator was not shown during request"

    @pytest.mark.e2e
    def test_htmx_validation_endpoint(self, driver):
        """Test that validation endpoint works via HTMX."""
        driver.get("http://localhost:8001/web/trainer")

        # Set up response capture
        driver.execute_script(
            """
            window.htmxResponses = [];
            document.body.addEventListener('htmx:afterOnLoad', function(evt) {
                window.htmxResponses.push({
                    xhr: {
                        status: evt.detail.xhr.status,
                        responseText: evt.detail.xhr.responseText
                    }
                });
            });
        """
        )

        # Find validate button if exists
        validate_buttons = driver.find_elements(By.CSS_SELECTOR, "button[hx-post*='/api/training/validate']")

        if validate_buttons:
            validate_buttons[0].click()
            time.sleep(1)

            # Check response
            responses = driver.execute_script("return window.htmxResponses;")
            if responses:
                last_response = responses[-1]
                assert last_response["xhr"]["status"] == 200
                assert "alert" in last_response["xhr"]["responseText"]

    @pytest.mark.e2e
    def test_htmx_error_handling(self, driver):
        """Test HTMX error handling and user feedback."""
        driver.get("http://localhost:8001/web/trainer")

        # Set up error event listener
        driver.execute_script(
            """
            window.htmxErrors = [];
            document.body.addEventListener('htmx:responseError', function(evt) {
                window.htmxErrors.push({
                    status: evt.detail.xhr.status,
                    statusText: evt.detail.xhr.statusText
                });
            });
        """
        )

        # Simulate an error by sending invalid data
        # This would depend on your validation logic
        driver.find_element(By.ID, "model_name").clear()  # Clear required field

        # Try to submit
        submit_buttons = driver.find_elements(By.CSS_SELECTOR, "button[hx-post*='/api/training/start']")

        if submit_buttons:
            submit_buttons[0].click()
            time.sleep(1)

            # Check for validation errors displayed to user
            error_messages = driver.find_elements(By.CSS_SELECTOR, ".alert-danger")
            assert len(error_messages) > 0, "No error messages shown for invalid form"


class TestHTMXDynamicContent:
    """Test HTMX dynamic content loading."""

    @pytest.mark.e2e
    def test_htmx_tab_loading(self, driver):
        """Test that tabs load content dynamically via HTMX."""
        driver.get("http://localhost:8001/web/trainer")

        # Set up content load tracking
        driver.execute_script(
            """
            window.tabLoads = [];
            document.body.addEventListener('htmx:afterSwap', function(evt) {
                if (evt.detail.target.id && evt.detail.target.id.startsWith('tab-')) {
                    window.tabLoads.push(evt.detail.target.id);
                }
            });
        """
        )

        # Click through tabs
        tabs = [("Model Config", "tab-model"), ("Training", "tab-training"), ("Advanced", "tab-advanced")]

        for tab_text, expected_id in tabs:
            tab_link = driver.find_element(By.LINK_TEXT, tab_text)
            tab_link.click()
            time.sleep(0.5)

        # Check that tabs were loaded
        loaded_tabs = driver.execute_script("return window.tabLoads;")
        assert len(loaded_tabs) >= len(tabs) - 1, "Not all tabs loaded via HTMX"

    @pytest.mark.e2e
    def test_htmx_form_replacement(self, driver):
        """Test that HTMX can replace form content."""
        driver.get("http://localhost:8001/web/trainer")

        # Get initial form HTML
        initial_form = driver.find_element(By.ID, "trainer-form").get_attribute("innerHTML")

        # Trigger some HTMX action that replaces content
        # This depends on your specific implementation
        dataset_tab = driver.find_element(By.LINK_TEXT, "Datasets")
        dataset_tab.click()
        time.sleep(0.5)

        # Go back to basic tab
        basic_tab = driver.find_element(By.LINK_TEXT, "Basic Config")
        basic_tab.click()
        time.sleep(0.5)

        # Form should still be functional
        form = driver.find_element(By.ID, "trainer-form")
        assert form is not None, "Form disappeared after HTMX swaps"

    @pytest.mark.e2e
    def test_htmx_history_entries(self, driver):
        """Test that HTMX updates browser history correctly."""
        driver.get("http://localhost:8001/web/trainer")

        # Get initial history length
        initial_length = driver.execute_script("return window.history.length;")

        # Navigate through tabs with hx-push-url
        tabs_with_history = driver.find_elements(By.CSS_SELECTOR, "a[hx-push-url='true']")

        if tabs_with_history:
            for tab in tabs_with_history[:3]:
                tab.click()
                time.sleep(0.3)

            # Check history was updated
            final_length = driver.execute_script("return window.history.length;")
            assert final_length > initial_length, "HTMX didn't update browser history"


class TestHTMXEventHandling:
    """Test HTMX event handling and lifecycle."""

    @pytest.mark.e2e
    def test_htmx_event_order(self, driver):
        """Test that HTMX events fire in correct order."""
        driver.get("http://localhost:8001/web/trainer")

        # Set up comprehensive event tracking
        driver.execute_script(
            """
            window.htmxEvents = [];
            const events = [
                'htmx:configRequest',
                'htmx:beforeRequest',
                'htmx:beforeSend',
                'htmx:afterRequest',
                'htmx:beforeSwap',
                'htmx:afterSwap',
                'htmx:afterSettle'
            ];

            events.forEach(eventName => {
                document.body.addEventListener(eventName, function(evt) {
                    window.htmxEvents.push({
                        name: eventName,
                        timestamp: Date.now()
                    });
                });
            });
        """
        )

        # Trigger an HTMX request
        save_button = driver.find_element(By.CSS_SELECTOR, "button[hx-post*='/api/training/config']")
        save_button.click()

        # Wait for events to complete
        time.sleep(1)

        # Check event order
        events = driver.execute_script("return window.htmxEvents;")
        event_names = [e["name"] for e in events]

        # Verify events fired in correct order
        expected_order = ["htmx:configRequest", "htmx:beforeRequest", "htmx:afterRequest"]

        for i, expected_event in enumerate(expected_order):
            if expected_event in event_names:
                assert event_names.index(expected_event) >= i, f"Event {expected_event} fired out of order"

    @pytest.mark.e2e
    def test_htmx_request_headers(self, driver):
        """Test that HTMX adds proper headers to requests."""
        driver.get("http://localhost:8001/web/trainer")

        # Intercept and check request headers
        driver.execute_script(
            """
            window.capturedHeaders = null;
            document.body.addEventListener('htmx:configRequest', function(evt) {
                window.capturedHeaders = evt.detail.headers;
            });
        """
        )

        # Trigger request
        save_button = driver.find_element(By.CSS_SELECTOR, "button[hx-post*='/api/training/config']")
        save_button.click()
        time.sleep(0.5)

        # Check headers
        headers = driver.execute_script("return window.capturedHeaders;")
        assert headers is not None, "No headers captured"
        assert "HX-Request" in headers, "Missing HX-Request header"
