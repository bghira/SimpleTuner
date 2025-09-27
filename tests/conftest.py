"""Global pytest configuration and fixtures."""

import asyncio
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.wait import WebDriverWait

from simpletuner.simpletuner_sdk.server import ServerMode, create_app

# Test server configuration
TEST_HOST = "127.0.0.1"
TEST_PORT = 8888
TEST_BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


def run_test_server():
    """Run the test server in a separate process."""
    import uvicorn

    app = create_app(mode=ServerMode.TRAINER)
    uvicorn.run(app, host=TEST_HOST, port=TEST_PORT, log_level="error")


@pytest.fixture(scope="session")
def test_server():
    """Start the test server for the entire test session."""
    # Start server in separate process
    server_process = multiprocessing.Process(target=run_test_server)
    server_process.start()

    # Wait for server to be ready
    import requests

    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{TEST_BASE_URL}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        server_process.terminate()
        pytest.fail(f"Test server failed to start after {max_retries} retries")

    yield TEST_BASE_URL

    # Cleanup
    server_process.terminate()
    server_process.join()


@pytest.fixture
def chrome_options():
    """Configure Chrome options for testing."""
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # Disable images for faster tests
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    return options


@pytest.fixture
def firefox_options():
    """Configure Firefox options for testing."""
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--width=1920")
    options.add_argument("--height=1080")
    # Disable images for faster tests
    options.set_preference("permissions.default.image", 2)
    return options


@pytest.fixture(params=["chrome", "firefox"])
def driver(request, chrome_options, firefox_options, test_server) -> Generator[webdriver.Remote, None, None]:
    """Create WebDriver instance for testing."""
    if request.param == "chrome":
        driver = webdriver.Chrome(options=chrome_options)
    elif request.param == "firefox":
        driver = webdriver.Firefox(options=firefox_options)
    else:
        pytest.skip(f"Unsupported browser: {request.param}")

    # Set implicit wait
    driver.implicitly_wait(10)

    yield driver

    # Cleanup
    driver.quit()


@pytest.fixture
def wait(driver) -> WebDriverWait:
    """Create WebDriverWait instance with default timeout."""
    return WebDriverWait(driver, 10)


@pytest.fixture
def navigate_to_trainer(driver, test_server):
    """Helper fixture to navigate to trainer page."""

    def _navigate():
        driver.get(f"{test_server}/web/trainer")
        # Wait for page to load
        assert "SimpleTuner Training Studio" in driver.title

    return _navigate


@pytest.fixture
def mock_webui_state(tmp_path, monkeypatch):
    """Mock WebUI state directory for isolated testing."""
    # Create temporary state directory
    state_dir = tmp_path / ".simpletuner" / "webui"
    state_dir.mkdir(parents=True)

    # Monkeypatch HOME environment variable
    monkeypatch.setenv("HOME", str(tmp_path))

    return state_dir


@pytest.fixture
def mock_config_store(tmp_path, monkeypatch):
    """Mock config store directory for isolated testing."""
    # Create temporary config directory
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create a sample config
    sample_config = config_dir / "test-config.json"
    sample_config.write_text(
        """{
        "--job_id": "test-model",
        "--output_dir": "/tmp/output",
        "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
        "--model_family": "flux",
        "--learning_rate": "0.0001",
        "--train_batch_size": "1",
        "--num_train_epochs": "10"
    }"""
    )

    return config_dir


@pytest.fixture
def setup_test_environment(mock_webui_state, mock_config_store):
    """Set up complete test environment with mocked state and configs."""
    # Create initial WebUI state
    defaults_file = mock_webui_state / "defaults.json"
    defaults_file.write_text(
        f'{{"configs_dir": "{mock_config_store}", "output_dir": "/tmp/output", "active_config": "test-config"}}'
    )

    # Create onboarding state
    onboarding_file = mock_webui_state / "onboarding.json"
    onboarding_file.write_text('{"steps": {}}')

    return {"state_dir": mock_webui_state, "config_dir": mock_config_store}


# Browser-specific markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "chrome_only: run test only on Chrome")
    config.addinivalue_line("markers", "firefox_only: run test only on Firefox")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "e2e: end-to-end tests")


# Skip browser-specific tests
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle browser-specific markers."""
    for item in items:
        # Get browser from fixture request
        if "driver" in item.fixturenames:
            browser = item.callspec.params.get("driver", None) if hasattr(item, "callspec") else None

            # Skip chrome_only tests for non-Chrome browsers
            if item.get_closest_marker("chrome_only") and browser != "chrome":
                item.add_marker(pytest.mark.skip("Chrome only test"))

            # Skip firefox_only tests for non-Firefox browsers
            if item.get_closest_marker("firefox_only") and browser != "firefox":
                item.add_marker(pytest.mark.skip("Firefox only test"))


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Screenshot capture on failure
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture screenshot on test failure."""
    # Execute all other hooks to get the report object
    outcome = yield
    rep = outcome.get_result()

    # Only capture screenshot for failed tests with driver fixture
    if rep.when == "call" and rep.failed and "driver" in item.fixturenames:
        try:
            driver = item.funcargs["driver"]

            # Create screenshots directory
            screenshot_dir = Path("test-screenshots")
            screenshot_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_name = item.name.replace("[", "_").replace("]", "_")
            filename = f"{test_name}_{timestamp}.png"
            filepath = screenshot_dir / filename

            # Capture screenshot
            driver.save_screenshot(str(filepath))

            # Add screenshot info to test report
            extra = getattr(rep, "extra", [])
            extra.append({"name": "Screenshot", "path": str(filepath)})
            rep.extra = extra

            # Also capture page source for debugging
            page_source_file = screenshot_dir / f"{test_name}_{timestamp}.html"
            with open(page_source_file, "w", encoding="utf-8") as f:
                f.write(driver.page_source)

            # Log browser console errors
            console_log_file = screenshot_dir / f"{test_name}_{timestamp}_console.log"
            try:
                logs = driver.get_log("browser")
                with open(console_log_file, "w", encoding="utf-8") as f:
                    for entry in logs:
                        f.write(f"{entry['level']}: {entry['message']}\n")
            except Exception:
                # Not all browsers support console logs
                pass

            print(f"\nScreenshot saved: {filepath}")
            print(f"Page source saved: {page_source_file}")

        except Exception as e:
            print(f"\nFailed to capture screenshot: {e}")
