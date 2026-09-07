"""Browser coverage for the standalone training report."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

from simpletuner.helpers.training.local_metrics import MEDIA_FILENAME, render_static_report
from tests.selenium_support import _chrome_options


@unittest.skipUnless(os.environ.get("SIMPLETUNER_SELENIUM_TESTS") == "1", "Selenium tests disabled")
class TrainingReportNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = _chrome_options()
        options.add_experimental_option("prefs", {})
        cls.driver = webdriver.Chrome(options=options)
        cls.addClassCleanup(cls.driver.quit)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        output = Path(self.directory.name)
        records = []
        for step in (100, 200, 300):
            for index in range(12):
                if step == 200 and index == 1:
                    continue
                filename = f"{step}-{index}.png"
                Image.new("RGB", (300, 400), (step % 256, index * 20, 80)).save(output / filename)
                records.append(dict(step=step, index=index, label="Validation", type="image", path=filename))
        # An unrelated output must not stand in for a missing sample.
        records.append(dict(step=200, index=1, label="Other", type="image", path="200-0.png"))
        (output / MEDIA_FILENAME).write_text("\n".join(json.dumps(record) for record in records))
        self.driver.set_window_size(1000, 800)
        self.output = output
        self.driver.get(render_static_report(output).as_uri())

    def element(self, element_id):
        return self.driver.find_element(By.ID, element_id)

    def open_sample(self, index):
        self.driver.find_elements(By.CSS_SELECTOR, ".media-image-button")[index].click()

    def assert_image(self, step, index):
        image = self.driver.find_element(By.CSS_SELECTOR, "#media-lightbox-body img")
        self.assertTrue(image.get_attribute("src").endswith(f"/{step}-{index}.png"))
        WebDriverWait(self.driver, 5).until(
            lambda driver: driver.execute_script("return arguments[0].complete && arguments[0].naturalWidth > 0", image)
        )
        self.assertIn(f"step {step}", self.element("media-lightbox-caption").text)
        self.assertTrue(self.element("media-lightbox").is_displayed())

    def test_keyboard_switches_steps_without_switching_sample_or_scrolling(self):
        self.open_sample(1)
        before = self.driver.execute_script("return window.scrollY")
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_UP)
        self.assert_image(100, 1)
        self.assertEqual(before, self.driver.execute_script("return window.scrollY"))
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_UP)
        self.assert_image(100, 1)
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_DOWN)
        self.assert_image(300, 1)
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_RIGHT)
        self.assert_image(300, 2)
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_UP)
        self.assert_image(200, 2)
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_LEFT)
        self.assert_image(200, 0)
        self.driver.switch_to.active_element.send_keys(Keys.ESCAPE)
        self.assertFalse(self.element("media-lightbox").is_displayed())

    def test_mobile_step_buttons_keep_lightbox_open_and_gallery_in_sync(self):
        self.driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 390,
                "height": 844,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        self.addCleanup(self.driver.execute_cdp_cmd, "Emulation.clearDeviceMetricsOverride", {})
        self.open_sample(1)
        buttons = self.driver.find_elements(By.CSS_SELECTOR, "#media-lightbox-steps .media-step")
        self.assertEqual([button.text for button in buttons], ["Step 100", "Step 300"])
        bounds = self.driver.execute_script("return arguments[0].getBoundingClientRect().toJSON()", buttons[0])
        self.assertGreaterEqual(bounds["height"], 44)
        self.driver.execute_cdp_cmd(
            "Input.dispatchTouchEvent",
            {
                "type": "touchStart",
                "touchPoints": [{"x": bounds["x"] + 10, "y": bounds["y"] + 10}],
            },
        )
        self.driver.execute_cdp_cmd("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        self.assert_image(100, 1)
        self.assertLessEqual(self.element("media-lightbox-steps").rect["width"], 390)
        self.element("media-lightbox-size").click()
        self.driver.find_elements(By.CSS_SELECTOR, "#media-lightbox-steps .media-step")[1].click()
        self.assert_image(300, 1)
        self.assertEqual(self.element("media-lightbox-size").text, "Fit")
        self.element("media-lightbox-close").click()
        self.assertEqual(self.driver.find_element(By.CSS_SELECTOR, "#media-timeline .active").text, "Step 300")

    def test_gallery_step_navigation_stays_visible_when_scrolling(self):
        timeline = self.element("media-timeline")
        self.driver.execute_script("window.scrollTo(0, arguments[0].offsetTop + 400)", timeline)
        top = self.driver.execute_script("return arguments[0].getBoundingClientRect().top", timeline)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(top, 20)
        self.driver.find_elements(By.CSS_SELECTOR, "#media-timeline .media-step")[0].click()
        self.assertEqual(self.driver.find_element(By.CSS_SELECTOR, "#media-timeline .active").text, "Step 100")

    def test_single_step_keeps_sample_navigation_and_size_controls(self):
        media_path = self.output / MEDIA_FILENAME
        media_path.write_text(
            "\n".join(line for line in media_path.read_text().splitlines() if json.loads(line)["step"] == 300)
        )
        self.driver.get(render_static_report(self.output).as_uri())
        self.open_sample(0)
        self.assertFalse(self.element("media-lightbox-prev").is_enabled())
        for key in (Keys.ARROW_UP, Keys.ARROW_DOWN, Keys.ARROW_LEFT):
            self.driver.switch_to.active_element.send_keys(key)
            self.assert_image(300, 0)
        self.element("media-lightbox-next").click()
        self.assert_image(300, 1)
        self.element("media-lightbox-size").click()
        self.assertEqual(
            self.driver.find_element(By.CSS_SELECTOR, "#media-lightbox-body img").get_attribute("class"), "actual-size"
        )
        self.element("media-lightbox-close").click()
        self.open_sample(11)
        self.assertFalse(self.element("media-lightbox-next").is_enabled())
        self.driver.switch_to.active_element.send_keys(Keys.ARROW_RIGHT)
        self.assert_image(300, 11)

    def test_empty_report_has_no_navigation_or_lightbox(self):
        (self.output / MEDIA_FILENAME).write_text("")
        self.driver.get(render_static_report(self.output).as_uri())
        self.assertEqual(self.element("media-grid").text, "No validation media")
        self.assertFalse(self.driver.find_elements(By.CSS_SELECTOR, ".media-step"))
        self.assertFalse(self.element("media-lightbox").is_displayed())

    def test_mobile_timeline_reveals_current_step_in_long_runs(self):
        records = [
            dict(step=step, index=0, label="Validation", type="image", path="300-0.png") for step in range(100, 3100, 100)
        ]
        (self.output / MEDIA_FILENAME).write_text("\n".join(json.dumps(record) for record in records))
        self.driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 390,
                "height": 844,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        self.addCleanup(self.driver.execute_cdp_cmd, "Emulation.clearDeviceMetricsOverride", {})
        self.driver.get(render_static_report(self.output).as_uri())
        self.open_sample(0)
        for key, expected in ((None, "Step 3000"), (Keys.ARROW_UP, "Step 2900")):
            if key:
                self.driver.switch_to.active_element.send_keys(key)
            active = self.driver.find_element(By.CSS_SELECTOR, "#media-lightbox-steps .active")
            self.assertEqual(active.text, expected)
            bounds = self.driver.execute_script("return arguments[0].getBoundingClientRect().toJSON()", active)
            self.assertGreaterEqual(bounds["left"], 0)
            self.assertLessEqual(bounds["right"], 390)
