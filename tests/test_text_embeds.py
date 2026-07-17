import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.models.common import TextEmbedCacheKey


class _DummyModel:
    def __init__(self, key_type):
        self._key_type = key_type

    def text_embed_cache_key(self):
        return self._key_type

    def pack_text_embeddings_for_cache(self, embeddings):
        return embeddings


class _DummyDataBackend:
    def __init__(self, backend_id="backend"):
        self.id = backend_id
        self.created_paths = []
        self.type = "local"

    def create_directory(self, path):
        self.created_paths.append(path)

    # The cache may call these helpers when interacting with backend IO.
    def exists(self, *_args, **_kwargs):
        return False

    def list_files(self, *_args, **_kwargs):
        return []

    def torch_save(self, *_args, **_kwargs):
        return None

    def torch_load(self, *_args, **_kwargs):
        raise FileNotFoundError

    def write(self, *_args, **_kwargs):
        return None

    def read(self, *_args, **_kwargs):
        return ""


class _DummyThread:
    def start(self):
        return None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _make_cache(key_type):
    backend = _DummyDataBackend()
    text_encoders = [SimpleNamespace(device="cpu")]
    tokenizers = [object()]

    @contextmanager
    def split_between_processes(records):
        yield records

    accelerator = SimpleNamespace(device="cpu", num_processes=1, split_between_processes=split_between_processes)
    model = _DummyModel(key_type)
    prompt_handler = None

    dummy_thread = _DummyThread()

    with patch("simpletuner.helpers.caching.text_embeds.Thread", return_value=dummy_thread):
        cache = TextEmbeddingCache(
            id="backend",
            data_backend=backend,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            accelerator=accelerator,
            cache_dir="/tmp/cache",
            model_type="test",
            prompt_handler=prompt_handler,
            model=model,
        )
    return cache


class TextEmbeddingCacheKeyTests(unittest.TestCase):
    def test_resolve_key_value_caption_fallback(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        record = {"prompt": "hello world"}
        resolved = cache._resolve_cache_key_value(record)
        self.assertEqual(resolved, "hello world")

    def test_resolve_key_value_prefers_record_key(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        record = {"prompt": "hello world", "key": "custom-key"}
        resolved = cache._resolve_cache_key_value(record)
        self.assertEqual(resolved, "custom-key")

    def test_resolve_key_value_raises_when_path_required(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        record = {"prompt": "hello world"}
        with self.assertRaises(ValueError):
            cache._resolve_cache_key_value(record)

    def test_normalize_prompts_infers_key_from_metadata(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        record = {
            "prompt": "hello world",
            "metadata": {
                "data_backend_id": "dataset-1",
                "dataset_relative_path": "path/to/sample.png",
            },
        }
        normalized = cache._normalize_prompt_records([record])
        self.assertEqual(normalized[0]["key"], "dataset-1:path/to/sample.png")

    def test_normalize_prompts_splits_multiline_caption_list(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        normalized = cache._normalize_prompt_records(
            [{"prompt": ["caption one", "caption two"], "metadata": {"image_path": "image.png"}}]
        )
        self.assertEqual([record["prompt"] for record in normalized], ["caption one", "caption two"])
        self.assertEqual([record["key"] for record in normalized], ["caption one", "caption two"])

    def test_batched_encode_saves_trimmed_per_caption_outputs(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        saved = []
        cache.save_to_cache = lambda filename, embeddings: saved.append((filename, embeddings))

        class BatchModel(_DummyModel):
            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.calls = []

            def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                self.calls.append((prompts, prompt_contexts))
                return {
                    "prompt_embeds": torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
                    "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
                    "add_text_embeds": torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5),
                }

        cache.model = BatchModel()
        records = [
            {"prompt": "short", "key": "short", "metadata": {"i": 1}},
            {"prompt": "longer", "key": "longer", "metadata": {"i": 2}},
        ]
        cache._encode_and_cache_prompt_batch(records, ["short.pt", "longer.pt"])

        self.assertEqual(cache.model.calls, [(["short", "longer"], [{"i": 1}, {"i": 2}])])
        self.assertEqual([filename for filename, _ in saved], ["short.pt", "longer.pt"])
        self.assertEqual(saved[0][1]["prompt_embeds"].shape, torch.Size([1, 2, 3]))
        self.assertEqual(saved[0][1]["attention_mask"].shape, torch.Size([1, 2]))
        self.assertEqual(saved[1][1]["prompt_embeds"].shape, torch.Size([1, 3, 3]))
        self.assertEqual(saved[1][1]["attention_mask"].shape, torch.Size([1, 3]))
        self.assertEqual(saved[0][1]["add_text_embeds"].shape, torch.Size([1, 5]))


if __name__ == "__main__":
    unittest.main()
