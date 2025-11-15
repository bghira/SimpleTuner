import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.models.common import TextEmbedCacheKey


class _DummyModel:
    def __init__(self, key_type):
        self._key_type = key_type

    def text_embed_cache_key(self):
        return self._key_type


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
    accelerator = SimpleNamespace(device="cpu", num_processes=1)
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


if __name__ == "__main__":
    unittest.main()
