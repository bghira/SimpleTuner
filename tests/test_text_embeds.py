import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.models.common import TextEmbedCacheKey


class _DummyModel:
    def __init__(self, key_type):
        self._key_type = key_type

    def text_embed_cache_key(self):
        return self._key_type

    def uses_text_embeddings_cache(self):
        return bool(getattr(self, "TEXT_ENCODER_CONFIGURATION", None))

    def pack_text_embeddings_for_cache(self, embeddings):
        return embeddings

    def requires_text_embed_image_context(self):
        return False


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


def _make_cache(key_type, **cache_kwargs):
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
            **cache_kwargs,
        )
    return cache


class TextEmbeddingCacheKeyTests(unittest.TestCase):
    def test_text_cache_disable_implies_ondemand(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_disable=True)

        self.assertTrue(cache.text_cache_disable)
        self.assertTrue(cache.text_cache_ondemand)

    def test_ondemand_cache_miss_encodes_without_writing_when_disabled(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_disable=True)

        class BatchModel(_DummyModel):
            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.encode_text_batch = MagicMock(return_value={"prompt_embeds": torch.ones(1, 2, 3)})

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache.model = BatchModel()
        cache.load_from_cache = MagicMock(side_effect=FileNotFoundError("missing"))

        output = cache.compute_prompt_embeddings_with_model(
            prompt_records=[{"prompt": "uncached prompt", "key": "uncached prompt", "metadata": {}}]
        )

        cache.model.encode_text_batch.assert_called_once()
        self.assertEqual(output["prompt_embeds"].shape, torch.Size([1, 2, 3]))
        self.assertEqual(cache.write_queue.qsize(), 0)

    def test_ondemand_cache_miss_rejects_missing_required_image_context(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_ondemand=True)

        class ImageContextModel(_DummyModel):
            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.encode_text_batch = MagicMock(return_value={"prompt_embeds": torch.ones(1, 2, 3)})

            def requires_text_embed_image_context(self):
                return True

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache.model = ImageContextModel()
        cache.load_from_cache = MagicMock(side_effect=FileNotFoundError("missing"))

        with self.assertRaisesRegex(ValueError, "requires image-context metadata") as context:
            cache.compute_prompt_embeddings_with_model(
                prompt_records=[{"prompt": "uncached prompt", "key": "uncached prompt", "metadata": {}}]
            )

        self.assertIn("text cache id: backend", str(context.exception))
        self.assertIn("data backend id: backend", str(context.exception))
        cache.model.encode_text_batch.assert_not_called()
        self.assertEqual(cache.write_queue.qsize(), 0)

    def test_ondemand_cache_miss_returns_unpacked_format(self):
        """On-demand encode round-trips through pack then unpack so the
        returned format matches what load_from_cache would return."""
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_ondemand=True)

        class PackingModel(_DummyModel):
            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.pack_input_dims = []

            def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                return {
                    "prompt_embeds": torch.arange(1 * 2 * 4, dtype=torch.float32).reshape(1, 2, 4),
                    "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                }

            def pack_text_embeddings_for_cache(self, embeddings):
                self.pack_input_dims.append(embeddings["prompt_embeds"].shape[-1])
                if embeddings["prompt_embeds"].shape[-1] == 2:
                    return embeddings
                packed = dict(embeddings)
                packed["prompt_embeds"] = embeddings["prompt_embeds"][..., :2].contiguous()
                return packed

            def unpack_text_embeddings_from_cache(self, embeddings):
                # Simulate restoring full width (mirrors load_from_cache path)
                unpacked = dict(embeddings)
                unpacked["prompt_embeds"] = torch.nn.functional.pad(
                    embeddings["prompt_embeds"], (0, 2)
                )
                return unpacked

        cache.model = PackingModel()
        cache.load_from_cache = MagicMock(side_effect=FileNotFoundError("missing"))
        cache.save_to_cache = MagicMock()

        output = cache.compute_prompt_embeddings_with_model(
            prompt_records=[{"prompt": "uncached prompt", "key": "uncached prompt", "metadata": {}}]
        )

        # Output should be in unpacked format (full width), same as cache-hit path
        self.assertEqual(output["prompt_embeds"].shape, torch.Size([1, 2, 4]))
        # save_to_cache receives unpacked data; it re-packs internally
        saved_embeddings = cache.save_to_cache.call_args.args[1]
        self.assertEqual(saved_embeddings["prompt_embeds"].shape, torch.Size([1, 2, 4]))
        self.assertEqual(cache.model.pack_input_dims, [4])

        def test_cache_miss_error_mentions_multi_caption_workaround(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)

        class CacheOnlyModel(_DummyModel):
            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache.model = CacheOnlyModel()
        cache.load_from_cache = MagicMock(side_effect=FileNotFoundError("missing"))

        with self.assertRaisesRegex(RuntimeError, "multi-caption") as context:
            cache.compute_prompt_embeddings_with_model(
                prompt_records=[{"prompt": "alternate caption", "key": "alternate caption", "metadata": {}}]
            )

        message = str(context.exception)
        self.assertIn("text_cache_ondemand: true", message)
        self.assertIn("clear the text embedding cache and rerun precache", message)

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

    def test_dataset_filename_hash_canonicalizes_path_normalized_webshart_uri(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)

        canonical = cache.create_hash("dataset-1:webshart://0/3/sample.mp4")
        path_normalized = cache.create_hash("dataset-1:webshart:/0/3/sample.mp4")

        self.assertEqual(path_normalized, canonical)

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

    def test_normalize_prompts_allows_multi_prompt_single_key(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        normalized = cache._normalize_prompt_records([{"prompt": ["caption one", "caption two"], "key": "shared-key"}])
        self.assertEqual([record["prompt"] for record in normalized], ["caption one", "caption two"])
        self.assertEqual([record["key"] for record in normalized], ["shared-key", "shared-key"])

    def test_normalize_prompts_accepts_multi_prompt_key_list(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        normalized = cache._normalize_prompt_records(
            [
                {
                    "prompt": ["caption one", "caption two"],
                    "key": ["dataset-1:path/to/sample.png:0", "dataset-1:path/to/sample.png:1"],
                    "metadata": {"image_path": "path/to/sample.png"},
                }
            ]
        )
        self.assertEqual([record["prompt"] for record in normalized], ["caption one", "caption two"])
        self.assertEqual(
            [record["key"] for record in normalized],
            ["dataset-1:path/to/sample.png:0", "dataset-1:path/to/sample.png:1"],
        )

    @patch("simpletuner.helpers.caching.text_embeds.StateTracker.get_text_cache_files", return_value={})
    def test_compute_embeddings_deduplicates_uncached_multi_prompt_key(self, _mock_cache_files):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        saved = []
        cache.save_to_cache = lambda filename, embeddings: saved.append((filename, embeddings))

        class BatchModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def __init__(self):
                super().__init__(TextEmbedCacheKey.DATASET_AND_FILENAME)
                self.calls = []

            def requires_text_embed_image_context(self):
                return False

            def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                self.calls.append((prompts, prompt_contexts))
                return {
                    "prompt_embeds": torch.ones(1, 2, 3),
                    "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                }

            def pack_text_embeddings_for_cache(self, embeddings):
                return embeddings

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache.model = BatchModel()

        cache.compute_embeddings_for_prompts(
            [
                {
                    "prompt": ["caption one", "caption two"],
                    "metadata": {
                        "data_backend_id": "dataset-1",
                        "dataset_relative_path": "path/to/sample.png",
                    },
                }
            ],
            return_concat=False,
            load_from_cache=False,
        )

        self.assertEqual(
            cache.model.calls,
            [(["caption one"], [{"data_backend_id": "dataset-1", "dataset_relative_path": "path/to/sample.png"}])],
        )
        self.assertEqual(len(saved), 1)

    @patch("simpletuner.helpers.caching.text_embeds.StateTracker.get_text_cache_files", return_value={})
    def test_compute_embeddings_does_not_resplit_rank_local_records(self, _mock_cache_files):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.split_prompt_records_between_processes = MagicMock(return_value=[])
        cache.save_to_cache = MagicMock()

        class BatchModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.encode_text_batch = MagicMock(
                    side_effect=lambda prompts, **_kwargs: {
                        "prompt_embeds": torch.ones(len(prompts), 2, 3),
                        "attention_mask": torch.ones(len(prompts), 2, dtype=torch.bool),
                    }
                )

        cache.model = BatchModel()

        cache.compute_embeddings_for_prompts(
            [
                {"prompt": "rank-local one", "key": "one"},
                {"prompt": "rank-local two", "key": "two"},
            ],
            return_concat=False,
            load_from_cache=False,
            split_between_processes=False,
        )

        cache.split_prompt_records_between_processes.assert_not_called()
        self.assertEqual(cache.model.encode_text_batch.call_count, 2)
        self.assertEqual(cache.save_to_cache.call_count, 2)

    @patch("simpletuner.helpers.caching.text_embeds.StateTracker.get_text_cache_files", return_value={})
    def test_validation_encode_marks_model_context(self, _mock_cache_files):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.save_to_cache = MagicMock()

        class ValidationAwareModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)
                self.validation_markers = []

            def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                self.validation_markers.append(getattr(self, "_current_prompt_is_validation", None))
                return {
                    "prompt_embeds": torch.ones(1, 2, 3),
                    "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                }

        cache.model = ValidationAwareModel()

        cache.compute_embeddings_for_prompts(
            [{"prompt": "a validation prompt", "key": "validation"}],
            is_validation=True,
            load_from_cache=False,
        )

        self.assertEqual(cache.model.validation_markers, [True])
        self.assertFalse(hasattr(cache.model, "_current_prompt_is_validation"))

    def test_multi_record_return_uses_model_collator_for_extra_tensors(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)

        class CollatingModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

            def collate_prompt_embeds(self, text_encoder_output):
                return {
                    "prompt_embeds": torch.cat([item["prompt_embeds"] for item in text_encoder_output], dim=0),
                    "text_token_tags": torch.cat([item["text_token_tags"] for item in text_encoder_output], dim=0),
                }

        cache.model = CollatingModel()
        cache.load_from_cache = MagicMock(
            side_effect=[
                {
                    "prompt_embeds": torch.full((1, 2, 3), 1.0),
                    "text_token_tags": torch.tensor([[1, 1]], dtype=torch.long),
                },
                {
                    "prompt_embeds": torch.full((1, 2, 3), 2.0),
                    "text_token_tags": torch.tensor([[2, 2]], dtype=torch.long),
                },
            ]
        )

        output = cache.compute_prompt_embeddings_with_model(
            prompt_records=[
                {"prompt": "first", "key": "first", "metadata": {}},
                {"prompt": "second", "key": "second", "metadata": {}},
            ],
        )

        self.assertEqual(output["prompt_embeds"].shape, torch.Size([2, 2, 3]))
        self.assertEqual(output["text_token_tags"].tolist(), [[1, 1], [2, 2]])

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

    def test_batched_encode_normalizes_single_sample_pooled_outputs(self):
        for pooled_is_batched in (False, True):
            with self.subTest(pooled_is_batched=pooled_is_batched):
                cache = _make_cache(TextEmbedCacheKey.CAPTION)
                saved = []
                cache.save_to_cache = lambda filename, embeddings: saved.append((filename, embeddings))

                class BatchModel(_DummyModel):
                    def __init__(self):
                        super().__init__(TextEmbedCacheKey.CAPTION)

                    def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                        pooled_prompt_embeds = torch.arange(5, dtype=torch.float32)
                        negative_pooled_prompt_embeds = torch.arange(5, 10, dtype=torch.float32)
                        if pooled_is_batched:
                            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
                            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.unsqueeze(0)
                        return {
                            "prompt_embeds": torch.ones(1, 2, 3),
                            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                            "pooled_prompt_embeds": pooled_prompt_embeds,
                            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                        }

                cache.model = BatchModel()
                records = [{"prompt": "short", "key": "short", "metadata": {}}]

                cache._encode_and_cache_prompt_batch(records, ["short.pt"])

                self.assertEqual(len(saved), 1)
                embeddings = saved[0][1]
                self.assertEqual(embeddings["pooled_prompt_embeds"].shape, torch.Size([5]))
                self.assertEqual(embeddings["negative_pooled_prompt_embeds"].shape, torch.Size([5]))
                torch.testing.assert_close(embeddings["pooled_prompt_embeds"], torch.arange(5, dtype=torch.float32))
                torch.testing.assert_close(
                    embeddings["negative_pooled_prompt_embeds"], torch.arange(5, 10, dtype=torch.float32)
                )


if __name__ == "__main__":
    unittest.main()
