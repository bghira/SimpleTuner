"""Utilities for Selenium-based unittest suites."""

from __future__ import annotations

import atexit
import multiprocessing
import sys
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from simpletuner.simpletuner_sdk.server import ServerMode, create_app

# Prefer spawn to avoid CUDA fork issues; use fork only on macOS where spawn can break UI tests.
if hasattr(multiprocessing, "set_start_method"):
    target_method = "fork" if sys.platform == "darwin" else "spawn"
    try:
        multiprocessing.set_start_method(target_method, force=True)
    except RuntimeError:
        pass  # Already set

TEST_HOST = "127.0.0.1"


def _run_test_server(port: int, home_path: str, webui_config_path: str) -> None:
    import uvicorn

    # Ensure the subprocess uses the test's temporary HOME directory
    os.environ["HOME"] = home_path
    os.environ["SIMPLETUNER_WEB_UI_CONFIG"] = webui_config_path
    os.environ["TQDM_DISABLE"] = "1"
    # Prevent CUDA initialisation issues when running inside forked test workers.
    # Tests exercise the WebUI and do not need GPU access.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("DIFFUSERS_DISABLE_CUDA", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    app = create_app(mode=ServerMode.TRAINER)
    uvicorn.run(app, host=TEST_HOST, port=port, log_level="error")


class _TestServerManager:
    """Start a trainer server in a background process."""

    def __init__(self) -> None:
        self._process: Optional[multiprocessing.Process] = None
        self.port: Optional[int] = None
        self.home_path: Optional[str] = None

    @property
    def base_url(self) -> str:
        if not self.port:
            raise RuntimeError("Test server port unavailable")
        return f"http://{TEST_HOST}:{self.port}"

    def start(self, home_path: str) -> str:
        # If server is already running with a different home path, restart it
        if self._process and self._process.is_alive():
            if self.home_path == home_path:
                return self.base_url
            # Home path changed, need to restart the server
            self.stop()

        self.home_path = home_path
        webui_config_path = str(Path(home_path) / ".simpletuner" / "webui")
        self.port = self._find_free_port()
        self._process = multiprocessing.Process(target=_run_test_server, args=(self.port, home_path, webui_config_path))
        self._process.daemon = True
        self._process.start()
        timeout = os.environ.get("SELENIUM_SERVER_START_TIMEOUT")
        try:
            timeout_value = float(timeout) if timeout is not None else 45.0
        except (TypeError, ValueError):
            timeout_value = 45.0
        self._wait_for_server(self.port, timeout=timeout_value)
        return self.base_url

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._process = None
        self.port = None

    @staticmethod
    def _find_free_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((TEST_HOST, 0))
            return sock.getsockname()[1]

    def _wait_for_server(self, port: int, timeout: float = 45.0) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{TEST_HOST}:{port}/health", timeout=0.5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise RuntimeError("Test server failed to start within timeout")


_SERVER_MANAGER = _TestServerManager()
atexit.register(_SERVER_MANAGER.stop)

_GLOBAL_HOME_TMPDIR: Optional[tempfile.TemporaryDirectory] = None
_GLOBAL_HOME_PATH: Optional[Path] = None
_GLOBAL_PREVIOUS_HOME: Optional[str] = None


def _cleanup_global_home() -> None:
    global _GLOBAL_HOME_TMPDIR, _GLOBAL_HOME_PATH, _GLOBAL_PREVIOUS_HOME
    if _GLOBAL_PREVIOUS_HOME is not None:
        os.environ["HOME"] = _GLOBAL_PREVIOUS_HOME
    else:
        os.environ.pop("HOME", None)
    # Also clean up the WebUI config environment variable
    os.environ.pop("SIMPLETUNER_WEB_UI_CONFIG", None)
    os.environ.pop("TQDM_DISABLE", None)
    if _GLOBAL_HOME_TMPDIR is not None:
        _GLOBAL_HOME_TMPDIR.cleanup()
    _GLOBAL_HOME_TMPDIR = None
    _GLOBAL_HOME_PATH = None


def ensure_global_home() -> Path:
    """Provision a shared HOME directory for selenium-driven tests."""

    global _GLOBAL_HOME_TMPDIR, _GLOBAL_HOME_PATH, _GLOBAL_PREVIOUS_HOME
    if _GLOBAL_HOME_PATH is None:
        _GLOBAL_PREVIOUS_HOME = os.environ.get("HOME")
        tmpdir = tempfile.TemporaryDirectory()
        _GLOBAL_HOME_TMPDIR = tmpdir
        _GLOBAL_HOME_PATH = Path(tmpdir.name)
        configure_home(_GLOBAL_HOME_PATH)
        atexit.register(_cleanup_global_home)
    else:
        configure_home(_GLOBAL_HOME_PATH)
    return _GLOBAL_HOME_PATH


def ensure_test_server(home_path: Path) -> str:
    """Ensure the singleton test server is running."""
    return _SERVER_MANAGER.start(str(home_path))


def _chrome_options() -> ChromeOptions:
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    prefs = {"profile.managed_default_content_settings.images": 2}
    opts.add_experimental_option("prefs", prefs)
    return opts


def _firefox_options() -> FirefoxOptions:
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    opts.add_argument("--width=1920")
    opts.add_argument("--height=1080")
    opts.set_preference("permissions.default.image", 2)
    return opts


def create_driver(browser: str) -> webdriver.Remote:
    """Create a browser driver matching previous pytest fixtures."""
    if browser == "chrome":
        driver = webdriver.Chrome(options=_chrome_options())
    elif browser == "firefox":
        driver = webdriver.Firefox(options=_firefox_options())
    else:
        raise ValueError(f"Unsupported browser: {browser}")
    driver.implicitly_wait(10)
    return driver


def _write_console_log(driver, path: Path) -> None:
    try:
        logs = driver.get_log("browser")
    except Exception:
        return
    with path.open("w", encoding="utf-8") as handle:
        for entry in logs:
            handle.write(f"{entry.get('level')}: {entry.get('message')}\n")


def capture_debug_artifacts(driver: webdriver.Remote, label: str) -> None:
    """Capture screenshot, HTML, and console logs for a failing browser run."""
    screenshot_dir = Path("test-screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "_")
    screenshot_path = screenshot_dir / f"{safe_label}_{timestamp}.png"
    html_path = screenshot_dir / f"{safe_label}_{timestamp}.html"
    log_path = screenshot_dir / f"{safe_label}_{timestamp}_console.log"

    try:
        driver.save_screenshot(str(screenshot_path))
    except WebDriverException:
        pass

    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
    except WebDriverException:
        pass

    _write_console_log(driver, log_path)


@contextmanager
def browser_session(browser: str, label: str) -> Generator[webdriver.Remote, None, None]:
    """Context manager yielding a configured browser driver."""
    driver = create_driver(browser)
    try:
        yield driver
    except Exception:
        capture_debug_artifacts(driver, label)
        raise
    finally:
        driver.quit()


_BROWSER_SPEC = os.environ.get("SELENIUM_BROWSERS", "chrome")
_DEFAULT_BROWSERS = tuple(b.strip() for b in _BROWSER_SPEC.split(",") if b.strip()) or ("chrome",)


def iter_browsers(enabled: Optional[Iterable[str]] = None) -> Iterable[str]:
    """Yield the browsers we exercise in E2E suites."""
    return list(enabled or _DEFAULT_BROWSERS)


def configure_home(temp_path: Path) -> None:
    """Point HOME to the provided path for the duration of a test."""
    os.environ["HOME"] = str(temp_path)
    # Also set the WebUI config path to ensure test isolation
    os.environ["SIMPLETUNER_WEB_UI_CONFIG"] = str(temp_path / ".simpletuner" / "webui")
    # Disable TQDM progress bars during tests
    os.environ["TQDM_DISABLE"] = "1"


import unittest


class SeleniumTestCase(unittest.TestCase):
    """Base class that mirrors the old pytest Selenium fixtures."""

    BROWSERS: Iterable[str] = _DEFAULT_BROWSERS
    MAX_BROWSERS: int | None = None
    base_url: str = ""
    _driver_cache: Dict[str, webdriver.Remote]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.home_path = ensure_global_home()
        cls.base_url = ensure_test_server(cls.home_path)
        cls._driver_cache = {}

    @classmethod
    def tearDownClass(cls) -> None:
        for driver in list(getattr(cls, "_driver_cache", {}).values()):
            try:
                driver.quit()
            except Exception:
                pass
        if hasattr(cls, "_driver_cache"):
            cls._driver_cache.clear()

        # Clean up test-screenshots directory if empty
        screenshot_dir = Path("test-screenshots")
        if screenshot_dir.exists():
            try:
                # Only remove if empty
                screenshot_dir.rmdir()
            except OSError:
                # Directory not empty, leave it for debugging
                pass

        super().tearDownClass()

    def _ensure_browser(self, browser: str) -> webdriver.Remote:
        driver = self._driver_cache.get(browser)
        if driver is not None:
            try:
                driver.execute_script("return 1")
            except WebDriverException:
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = None
        if driver is None:
            driver = create_driver(browser)
            self._driver_cache[browser] = driver
        return driver

    @staticmethod
    def _reset_browser_state(driver: webdriver.Remote) -> None:
        try:
            driver.get("about:blank")
        except WebDriverException:
            pass
        try:
            driver.delete_all_cookies()
        except WebDriverException:
            pass
        try:
            driver.execute_script("window.sessionStorage.clear(); window.localStorage.clear();")
        except WebDriverException:
            pass

    def for_each_browser(self, label: str, callback) -> None:
        browsers = list(iter_browsers(self.BROWSERS))
        max_browsers = getattr(self, "MAX_BROWSERS", None)
        if isinstance(max_browsers, int) and max_browsers > 0:
            browsers = browsers[:max_browsers]
        for browser in browsers:
            with self.subTest(browser=browser):
                driver = self._ensure_browser(browser)
                self._reset_browser_state(driver)
                try:
                    callback(driver, browser)
                except Exception:
                    capture_debug_artifacts(driver, f"{self.__class__.__name__}.{label}.{browser}")
                    raise
