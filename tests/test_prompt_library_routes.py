import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults
from tests.test_webui_api import _WebUIBaseTestCase


class PromptLibraryRoutesTestCase(_WebUIBaseTestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.configs_dir = Path(self.temp_dir / "configs")
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.libraries_dir = self.configs_dir / "validation_prompt_libraries"
        self.libraries_dir.mkdir(parents=True, exist_ok=True)

        defaults = WebUIDefaults(
            configs_dir=str(self.configs_dir),
            output_dir=str(self.temp_dir / "output"),
        )
        self.state_store.save_defaults(defaults)

        sample = {"intro": "A sample prompt"}
        sample_path = self.libraries_dir / "user_prompt_library-sample.json"
        sample_path.write_text(json.dumps(sample, indent=2))
        self.sample_data = sample

    def test_list_prompt_libraries(self) -> None:
        response = self.client.get("/api/prompt-libraries")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
        library = data["libraries"][0]
        self.assertEqual(library["filename"], "user_prompt_library-sample.json")
        self.assertEqual(library["relative_path"], "validation_prompt_libraries/user_prompt_library-sample.json")

    def test_get_prompt_library(self) -> None:
        response = self.client.get("/api/prompt-libraries/user_prompt_library-sample.json")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["entries"], self.sample_data)

    def test_save_prompt_library_and_rename(self) -> None:
        payload = {"entries": {"add": "new prompt"}}
        response = self.client.put("/api/prompt-libraries/user_prompt_library-new.json", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.libraries_dir / "user_prompt_library-new.json").exists())

        rename_payload = {
            "entries": {"renamed": "updated prompt"},
            "previous_filename": "user_prompt_library-new.json",
        }
        rename_resp = self.client.put("/api/prompt-libraries/user_prompt_library-renamed.json", json=rename_payload)
        self.assertEqual(rename_resp.status_code, 200)
        self.assertTrue((self.libraries_dir / "user_prompt_library-renamed.json").exists())
        self.assertFalse((self.libraries_dir / "user_prompt_library-new.json").exists())

    def test_invalid_filename_returns_error(self) -> None:
        payload = {"entries": {"bad": "value"}}
        response = self.client.put("/api/prompt-libraries/invalid name.json", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_save_prompt_library_with_adapter_strength(self) -> None:
        payload = {"entries": {"slider": {"prompt": "hello", "adapter_strength": 0.25}}}
        response = self.client.put("/api/prompt-libraries/user_prompt_library-slider.json", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["entries"]["slider"]["adapter_strength"], 0.25)

        get_resp = self.client.get("/api/prompt-libraries/user_prompt_library-slider.json")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["entries"]["slider"], {"prompt": "hello", "adapter_strength": 0.25})

    def test_save_prompt_library_with_bbox_entities(self) -> None:
        entities = [
            {"label": "cat", "bbox": [0.2, 0.3, 0.6, 0.8]},
            {"label": "table", "bbox": [0.0, 0.5, 1.0, 1.0]},
        ]
        payload = {
            "entries": {
                "cat_scene": {
                    "prompt": "a cat on a table",
                    "bbox_entities": entities,
                },
                "plain": "a simple prompt",
            }
        }
        response = self.client.put("/api/prompt-libraries/user_prompt_library-bbox.json", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["entries"]["cat_scene"]["prompt"], "a cat on a table")
        self.assertEqual(len(body["entries"]["cat_scene"]["bbox_entities"]), 2)
        self.assertEqual(body["entries"]["plain"], "a simple prompt")

        get_resp = self.client.get("/api/prompt-libraries/user_prompt_library-bbox.json")
        self.assertEqual(get_resp.status_code, 200)
        reloaded = get_resp.json()["entries"]
        self.assertEqual(reloaded["cat_scene"]["bbox_entities"][0]["label"], "cat")
        self.assertEqual(reloaded["plain"], "a simple prompt")

    def test_save_prompt_library_with_bbox_keyframes(self) -> None:
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]},
            {"frame": 20, "entities": [{"label": "cat", "bbox": [0.3, 0.4, 0.7, 0.8]}]},
        ]
        payload = {
            "entries": {
                "moving_cat": {
                    "prompt": "a cat walking",
                    "bbox_keyframes": keyframes,
                },
                "plain": "a simple prompt",
            }
        }
        response = self.client.put("/api/prompt-libraries/user_prompt_library-kf.json", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["entries"]["moving_cat"]["prompt"], "a cat walking")
        self.assertEqual(len(body["entries"]["moving_cat"]["bbox_keyframes"]), 2)
        self.assertEqual(body["entries"]["plain"], "a simple prompt")

        get_resp = self.client.get("/api/prompt-libraries/user_prompt_library-kf.json")
        self.assertEqual(get_resp.status_code, 200)
        reloaded = get_resp.json()["entries"]
        self.assertEqual(reloaded["moving_cat"]["bbox_keyframes"][0]["frame"], 0)
        self.assertEqual(reloaded["moving_cat"]["bbox_keyframes"][1]["entities"][0]["label"], "cat")


if __name__ == "__main__":
    unittest.main()
