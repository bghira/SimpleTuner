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


if __name__ == "__main__":
    unittest.main()
