import json
import unittest
from contextlib import contextmanager

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.metadata.preparation import PreparationView
from simpletuner.helpers.prompts import PromptHandler


class _CaptionView:
    def __init__(self, refs, captions=None):
        self.refs = refs
        self.captions = captions or {}

    @contextmanager
    def open_preparation_view(self, *, worker, workers):
        yield PreparationView(iter(self.refs), self)

    def caption_cache_entry(self, image_path):
        return self.captions.get(image_path)


class PromptHandlerTests(unittest.TestCase):
    def test_caption_preparation_uses_only_accepted_refs(self):
        selected = "part.parquet/000000000001.jpg"
        metadata = _CaptionView([selected], {selected: "accepted", "part.parquet/000000000002.jpg": "excluded"})
        runtime = DatasetRuntime("source", None, metadata, {"caption_strategy": "webshart"})
        self.assertEqual(list(runtime.iter_accepted_captions()), ["accepted"])

    def test_instanceprompt_returns_entry_per_image(self):
        refs = ["a.jpg", "b.jpg", "c.jpg"]
        runtime = DatasetRuntime(
            "source", None, _CaptionView(refs), {"caption_strategy": "instanceprompt", "instance_prompt": "minecraft"}
        )
        self.assertEqual(list(runtime.iter_caption_records(iter(refs))), [("minecraft", ref) for ref in refs])

    def test_webshart_caption_traversal_expands_structured_caption_variants(self):
        first, second = "webshart://0/1/first.jpg", "webshart://0/2/second.jpg"
        structured = {"description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]}
        metadata = _CaptionView(
            [first, second], {first: ["first primary", "first alternate"], second: [structured, "second alternate"]}
        )
        runtime = DatasetRuntime("source", None, metadata, {"caption_strategy": "webshart"})
        prompts, refs = zip(*runtime.iter_caption_records(iter((first, second))), strict=True)
        self.assertEqual(prompts[:2], ("first primary", "first alternate"))
        self.assertEqual(json.loads(prompts[2]), structured)
        self.assertEqual(prompts[3], "second alternate")
        self.assertEqual(refs, (first, first, second, second))

    def test_native_caption_object_is_one_json_prompt(self):
        caption = {"high_level_description": "a café", "style_description": {"medium": "photo"}}
        values = PromptHandler._normalize_caption_payload(caption)
        self.assertEqual(len(values), 1)
        self.assertEqual(json.loads(values[0]), caption)
        self.assertEqual(PromptHandler._restore_caption_payload_shape(caption, values), values[0])
        self.assertEqual(PromptHandler._caption_payload_values(caption), [caption])


if __name__ == "__main__":
    unittest.main()
