"""Base page class for Page Object Model."""

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


class BasePage:
    """Base page class with common functionality."""

    def __init__(self, driver, base_url=None):
        """Initialize base page.

        Args:
            driver: Selenium WebDriver instance
            base_url: Base URL for the application
        """
        self.driver = driver
        self.base_url = base_url or "http://localhost:8888"
        self.wait = WebDriverWait(driver, 10)

    def navigate_to(self, path=""):
        """Navigate to a specific path.

        Args:
            path: Path relative to base URL
        """
        url = f"{self.base_url}{path}" if path else self.base_url
        self.driver.get(url)

    def find_element(self, by, value):
        """Find element with wait.

        Args:
            by: Locator strategy (By.ID, By.CSS_SELECTOR, etc.)
            value: Locator value

        Returns:
            WebElement
        """
        return self.wait.until(EC.presence_of_element_located((by, value)))

    def find_elements(self, by, value):
        """Find multiple elements.

        Args:
            by: Locator strategy
            value: Locator value

        Returns:
            List of WebElements
        """
        return self.driver.find_elements(by, value)

    def click_element(self, by, value):
        """Click an element.

        Args:
            by: Locator strategy
            value: Locator value
        """
        element = self.wait.until(EC.element_to_be_clickable((by, value)))
        element.click()

    def send_keys(self, by, value, text, clear=True):
        """Send keys to an input element.

        Args:
            by: Locator strategy
            value: Locator value
            text: Text to send
            clear: Whether to clear the field first
        """
        element = self.find_element(by, value)
        if clear:
            element.clear()
        element.send_keys(text)

    def get_text(self, by, value):
        """Get text from an element.

        Args:
            by: Locator strategy
            value: Locator value

        Returns:
            Element text
        """
        return self.find_element(by, value).text

    def is_element_visible(self, by, value, timeout=10):
        """Check if element is visible.

        Args:
            by: Locator strategy
            value: Locator value
            timeout: Wait timeout

        Returns:
            True if visible, False otherwise
        """
        try:
            WebDriverWait(self.driver, timeout).until(EC.visibility_of_element_located((by, value)))
            return True
        except TimeoutException:
            return False

    def wait_for_element_to_disappear(self, by, value, timeout=10):
        """Wait for element to disappear.

        Args:
            by: Locator strategy
            value: Locator value
            timeout: Wait timeout
        """
        WebDriverWait(self.driver, timeout).until(EC.invisibility_of_element_located((by, value)))

    def scroll_to_element(self, element):
        """Scroll to element.

        Args:
            element: WebElement to scroll to
        """
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

    def wait_for_ajax(self, timeout=10):
        """Wait for AJAX requests to complete.

        Args:
            timeout: Maximum wait time
        """
        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script("return typeof jQuery !== 'undefined' ? jQuery.active == 0 : true")
        )

    def wait_for_htmx(self, timeout=10):
        """Wait for HTMX requests to complete.

        Args:
            timeout: Maximum wait time
        """
        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script("return typeof htmx !== 'undefined' ? htmx.ready : true")
        )

    def get_toast_message(self):
        """Get the current toast notification message.

        Returns:
            Toast message text or None if no toast visible
        """
        try:
            toast = self.find_element(By.CSS_SELECTOR, ".toast-body")
            return toast.text if toast.is_displayed() else None
        except:
            return None

    def dismiss_toast(self):
        """Dismiss the current toast notification."""
        try:
            close_button = self.find_element(By.CSS_SELECTOR, ".toast .btn-close")
            close_button.click()
            self.wait_for_element_to_disappear(By.CSS_SELECTOR, ".toast.show")
        except:
            pass

    def switch_to_tab(self, tab_name):
        """Switch to a specific tab.

        Args:
            tab_name: Name of the tab (basic, model, training, advanced, datasets, environments)
        """
        self.click_element(By.CSS_SELECTOR, f'[data-bs-target="#tab-{tab_name}"]')
        # Wait for tab content to be visible
        self.wait.until(EC.visibility_of_element_located((By.ID, f"tab-{tab_name}")))
