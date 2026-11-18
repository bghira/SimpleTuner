import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.prompts import PromptHandler


class _DummyMetadataBackend:
    def __init__(self, captions):
        self._captions = captions

    def caption_cache_entry(self, image_path):
        return self._captions.get(image_path)


def _runtime(caption_strategy, metadata_backend=None, **config):
    from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

    settings = dict(
        caption_strategy=caption_strategy,
        use_captions=caption_strategy != "instanceprompt",
        prepend_instance_prompt=False,
        instance_prompt=None,
        disable_multiline_split=False,
    )
    settings.update(config)
    return DatasetRuntime("test-backend", SimpleNamespace(id="test-backend"), metadata_backend, settings)


class PromptHandlerTests(unittest.TestCase):
    def test_instanceprompt_gives_each_sample_the_prompt(self):
        runtime = _runtime("instanceprompt", instance_prompt="minecraft")
        records = list(runtime.iter_caption_records(["a.jpg", "b.jpg", "c.jpg"]))
        self.assertEqual(records, [("minecraft", "a.jpg"), ("minecraft", "b.jpg"), ("minecraft", "c.jpg")])

    def test_webshart_structured_caption_variants_are_each_a_record(self):
        metadata_backend = _DummyMetadataBackend(
            {
                "webshart://0/1/first.jpg": ["first primary", "first alternate"],
                "webshart://0/2/second.jpg": [
                    {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]},
                    "second alternate",
                ],
            }
        )
        runtime = _runtime("webshart", metadata_backend)
        with patch(
            "simpletuner.helpers.prompts.StateTracker.get_data_backend",
            return_value={"metadata_backend": metadata_backend},
        ):
            records = list(runtime.iter_caption_records(["webshart://0/1/first.jpg", "webshart://0/2/second.jpg"]))

        captions = [caption for caption, _ in records]
        self.assertEqual(captions[:2], ["first primary", "first alternate"])
        self.assertEqual(json.loads(captions[2]), {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]})
        self.assertEqual(captions[3], "second alternate")
        self.assertEqual(
            [sample_ref for _, sample_ref in records],
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
