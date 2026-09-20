import json
import unittest
from unittest.mock import patch

from simpletuner.helpers.prompts import PromptHandler


class _DummyBackend:
    def __init__(self, files):
        self.id = "test-backend"
        self._files = files

    def list_files(self, instance_data_dir=None, file_extensions=None):
        return self._files


class _DummyMetadataBackend:
    def __init__(self, captions):
        self._captions = captions

    def caption_cache_entry(self, image_path):
        return self._captions.get(image_path)


class PromptHandlerTests(unittest.TestCase):
    def test_instanceprompt_returns_entry_per_image(self):
        backend = _DummyBackend(["a.jpg", "b.jpg", "c.jpg"])
        with patch(
            "simpletuner.helpers.prompts.StateTracker.get_image_files",
            return_value=None,
        ):
            captions, missing, paths = PromptHandler.get_all_captions(
                instance_data_dir="",
                use_captions=False,
                prepend_instance_prompt=False,
                data_backend=backend,
                caption_strategy="instanceprompt",
                instance_prompt="minecraft",
                return_image_paths=True,
            )
        self.assertEqual(missing, [])
        self.assertEqual(captions, ["minecraft", "minecraft", "minecraft"])
        self.assertEqual(paths, ["a.jpg", "b.jpg", "c.jpg"])

    def test_webshart_get_all_captions_expands_structured_caption_variants(self):
        backend = _DummyBackend(["webshart://0/1/first.jpg", "webshart://0/2/second.jpg"])
        metadata_backend = _DummyMetadataBackend(
            {
                "webshart://0/1/first.jpg": ["first primary", "first alternate"],
                "webshart://0/2/second.jpg": [
                    {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]},
                    "second alternate",
                ],
            }
        )

        with (
            patch("simpletuner.helpers.prompts.StateTracker.get_data_backend_config", return_value={}),
            patch("simpletuner.helpers.prompts.StateTracker.get_image_files", return_value=None),
            patch(
                "simpletuner.helpers.prompts.StateTracker.get_data_backend",
                return_value={"metadata_backend": metadata_backend},
            ),
        ):
            captions, missing, paths = PromptHandler.get_all_captions(
                instance_data_dir="",
                use_captions=True,
                prepend_instance_prompt=False,
                data_backend=backend,
                caption_strategy="webshart",
                return_image_paths=True,
            )

        self.assertEqual(missing, [])
        self.assertEqual(captions[:2], ["first primary", "first alternate"])
        self.assertEqual(json.loads(captions[2]), {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]})
        self.assertEqual(captions[3], "second alternate")
        self.assertEqual(
            paths,
            [
                "webshart://0/1/first.jpg",
                "webshart://0/1/first.jpg",
                "webshart://0/2/second.jpg",
                "webshart://0/2/second.jpg",
            ],
        )

    def test_native_caption_object_is_one_json_prompt(self):
        caption = {"high_level_description": "a café", "style_description": {"medium": "photo"}}
        values = PromptHandler._normalize_caption_payload(caption)
        self.assertEqual(len(values), 1)
        self.assertEqual(json.loads(values[0]), caption)
        self.assertEqual(PromptHandler._restore_caption_payload_shape(caption, values), values[0])
        self.assertEqual(PromptHandler._caption_payload_values(caption), [caption])


if __name__ == "__main__":
    unittest.main()
