import unittest

from tests.test_webui_api import _WebUIBaseTestCase


class LycorisRoutesTestCase(_WebUIBaseTestCase, unittest.TestCase):
    def test_metadata_endpoint_returns_algorithms_and_presets(self) -> None:
        response = self.client.get("/api/lycoris/metadata")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("algorithms", payload)
        self.assertIn("presets", payload)
        self.assertIn("defaults", payload)
        self.assertTrue(any(algo["name"] == "lora" for algo in payload["algorithms"]))
        self.assertTrue(any(preset["name"] == "attn-only" for preset in payload["presets"]))


if __name__ == "__main__":
    unittest.main()
