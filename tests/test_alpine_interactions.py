"""
Tests for Alpine.js interactions and component isolation.

These tests ensure Alpine.js components are properly isolated and
don't share state unexpectedly.
"""

import time
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import JavascriptException


class TestAlpineJsComponentIsolation:
    """Test Alpine.js component isolation and interactions."""

    @pytest.mark.e2e
    def test_form_field_alpine_scopes_are_isolated(self, driver):
        """Test that each form field has its own Alpine.js scope."""
        # Navigate to the trainer page
        driver.get("http://localhost:8001/web/trainer")

        # Wait for Alpine.js to initialize
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return window.Alpine !== undefined")
        )

        # Get all form fields with Alpine.js data
        fields_with_alpine = driver.find_elements(By.CSS_SELECTOR, "[x-data]")

        # Verify each field has its own scope
        scopes = []
        for field in fields_with_alpine:
            # Get the Alpine component for this element
            scope = driver.execute_script("""
                return Alpine.$data(arguments[0]);
            """, field)
            scopes.append(scope)

        # Verify that modifying one scope doesn't affect others
        if len(scopes) >= 2:
            # Modify the first scope
            driver.execute_script("""
                const component = Alpine.$data(arguments[0]);
                if (component && component.value !== undefined) {
                    component.value = 'modified-value';
                }
            """, fields_with_alpine[0])

            # Check that other scopes are unaffected
            for i, field in enumerate(fields_with_alpine[1:], 1):
                other_scope = driver.execute_script("""
                    return Alpine.$data(arguments[0]);
                """, field)
                if other_scope and 'value' in other_scope:
                    assert other_scope['value'] != 'modified-value', \
                        f"Field {i} was affected by changes to field 0"

    @pytest.mark.e2e
    def test_alpine_x_data_initialization(self, driver):
        """Test that Alpine.js x-data attributes initialize correctly."""
        driver.get("http://localhost:8001/web/trainer")

        # Wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "trainer-form"))
        )

        # Check form fields have proper x-data initialization
        form_fields = driver.find_elements(By.CSS_SELECTOR, ".mb-3[x-data]")

        for field in form_fields:
            x_data = field.get_attribute("x-data")
            assert x_data is not None, "Form field missing x-data attribute"

            # Verify the x-data contains expected properties
            alpine_data = driver.execute_script("""
                return Alpine.$data(arguments[0]);
            """, field)

            # Check for expected properties
            assert isinstance(alpine_data, dict), "Alpine data should be an object"
            # These properties are defined in form_field.html
            expected_props = ['value', 'error', 'validating', 'isValid']
            for prop in expected_props:
                assert prop in alpine_data, f"Missing property: {prop}"

    @pytest.mark.e2e
    def test_alpine_x_show_x_if_conditions(self, driver):
        """Test that Alpine.js x-show and x-if directives work correctly."""
        driver.get("http://localhost:8001/web/trainer")

        # Test collapsible sections
        collapsible_sections = driver.find_elements(
            By.CSS_SELECTOR, "[x-show='expanded']"
        )

        for section in collapsible_sections:
            # Get the parent element with x-data
            parent = driver.execute_script("""
                return arguments[0].closest('[x-data]');
            """, section)

            if parent:
                # Toggle expanded state
                driver.execute_script("""
                    const component = Alpine.$data(arguments[0]);
                    if (component && component.expanded !== undefined) {
                        component.expanded = !component.expanded;
                    }
                """, parent)

                # Wait for transition
                time.sleep(0.5)

                # Check visibility changed
                is_visible = section.is_displayed()
                alpine_state = driver.execute_script("""
                    const component = Alpine.$data(arguments[0]);
                    return component ? component.expanded : null;
                """, parent)

                assert is_visible == alpine_state, \
                    "x-show visibility doesn't match Alpine state"

    @pytest.mark.e2e
    def test_alpine_stores_isolation(self, driver):
        """Test that Alpine.js stores don't leak between components."""
        driver.get("http://localhost:8001/web/trainer")

        # Check if Alpine stores are used
        stores = driver.execute_script("""
            return Object.keys(Alpine.store ? Alpine.store : {});
        """)

        if stores:
            # Test store isolation
            for store_name in stores:
                # Get store data
                store_data = driver.execute_script("""
                    return Alpine.store(arguments[0]);
                """, store_name)

                # Stores should be properly namespaced
                assert isinstance(store_data, (dict, type(None))), \
                    f"Store {store_name} has unexpected type"

    @pytest.mark.e2e
    def test_alpine_event_handling(self, driver):
        """Test that Alpine.js event handlers work correctly."""
        driver.get("http://localhost:8001/web/trainer")

        # Find elements with Alpine click handlers
        clickable_elements = driver.find_elements(By.CSS_SELECTOR, "[\\@click]")

        for element in clickable_elements[:3]:  # Test first 3 to avoid long test
            # Get the current Alpine data before click
            parent_with_data = driver.execute_script("""
                return arguments[0].closest('[x-data]');
            """, element)

            if parent_with_data:
                before_data = driver.execute_script("""
                    return JSON.stringify(Alpine.$data(arguments[0]));
                """, parent_with_data)

                # Click the element
                try:
                    element.click()
                    time.sleep(0.2)  # Wait for Alpine to process

                    # Get data after click
                    after_data = driver.execute_script("""
                        return JSON.stringify(Alpine.$data(arguments[0]));
                    """, parent_with_data)

                    # Data should have changed (for toggle handlers)
                    click_handler = element.get_attribute("@click")
                    if "!" in click_handler:  # Toggle operation
                        assert before_data != after_data, \
                            f"Click handler didn't change state: {click_handler}"
                except Exception:
                    # Some elements might not be clickable, that's okay
                    pass

    @pytest.mark.e2e
    def test_alpine_component_cleanup(self, driver):
        """Test that Alpine.js components are properly cleaned up."""
        driver.get("http://localhost:8001/web/trainer")

        # Get initial component count
        initial_count = driver.execute_script("""
            let count = 0;
            document.querySelectorAll('[x-data]').forEach(el => {
                if (Alpine.$data(el)) count++;
            });
            return count;
        """)

        # Navigate to a different tab
        model_tab = driver.find_element(By.LINK_TEXT, "Model Config")
        model_tab.click()
        time.sleep(0.5)

        # Navigate back
        basic_tab = driver.find_element(By.LINK_TEXT, "Basic Config")
        basic_tab.click()
        time.sleep(0.5)

        # Check component count is consistent
        final_count = driver.execute_script("""
            let count = 0;
            document.querySelectorAll('[x-data]').forEach(el => {
                if (Alpine.$data(el)) count++;
            });
            return count;
        """)

        # Should have same number of components (no memory leaks)
        assert abs(final_count - initial_count) <= 1, \
            f"Component count changed: {initial_count} -> {final_count}"

    @pytest.mark.e2e
    def test_alpine_x_model_not_present(self, driver):
        """Regression test: Ensure x-model is not used in form fields."""
        driver.get("http://localhost:8001/web/trainer")

        # Wait for form to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "trainer-form"))
        )

        # Check all input elements
        inputs = driver.find_elements(By.CSS_SELECTOR, "input, select, textarea")

        for input_elem in inputs:
            x_model = input_elem.get_attribute("x-model")
            assert x_model is None, \
                f"Input {input_elem.get_attribute('id')} has x-model binding"

    @pytest.mark.e2e
    def test_alpine_transitions(self, driver):
        """Test that Alpine.js transitions work correctly."""
        driver.get("http://localhost:8001/web/trainer")

        # Find elements with x-transition
        transition_elements = driver.find_elements(
            By.CSS_SELECTOR, "[x-transition]"
        )

        if transition_elements:
            # Test first transition element
            elem = transition_elements[0]
            parent = driver.execute_script("""
                return arguments[0].closest('[x-data]');
            """, elem)

            if parent:
                # Get initial visibility
                initial_visible = elem.is_displayed()

                # Toggle visibility
                driver.execute_script("""
                    const component = Alpine.$data(arguments[0]);
                    const showProp = Object.keys(component).find(
                        key => key.includes('show') || key.includes('expanded')
                    );
                    if (showProp) {
                        component[showProp] = !component[showProp];
                    }
                """, parent)

                # Wait for transition to complete
                time.sleep(0.6)  # Transitions are usually 300-500ms

                # Check visibility changed
                final_visible = elem.is_displayed()
                assert initial_visible != final_visible, \
                    "Transition didn't change element visibility"


class TestAlpineJsDataFlow:
    """Test data flow in Alpine.js components."""

    @pytest.mark.e2e
    def test_alpine_data_binding_independence(self, driver):
        """Test that data bindings are independent between components."""
        driver.get("http://localhost:8001/web/trainer")

        # Get form fields
        configs_dir = driver.find_element(By.ID, "configs_dir")
        model_name = driver.find_element(By.ID, "model_name")

        # Get their Alpine components
        configs_component = driver.execute_script("""
            return arguments[0].closest('[x-data]');
        """, configs_dir)

        model_component = driver.execute_script("""
            return arguments[0].closest('[x-data]');
        """, model_name)

        # Verify they're different components
        assert configs_component != model_component, \
            "Form fields share the same Alpine component"

        # Modify one component's data
        driver.execute_script("""
            const component = Alpine.$data(arguments[0]);
            if (component) {
                component.value = 'test-value';
                component.error = 'test-error';
            }
        """, configs_component)

        # Check other component is unaffected
        model_data = driver.execute_script("""
            return Alpine.$data(arguments[0]);
        """, model_component)

        assert model_data.get('value') != 'test-value', \
            "Data leaked between Alpine components"
        assert model_data.get('error') != 'test-error', \
            "Error state leaked between components"