import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.caching.artifact_listing import ArtifactListing
from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.training.state_tracker import StateTracker


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
    bulk_exists = BaseDataBackend.bulk_exists
    read_cached_tensors = BaseDataBackend.read_cached_tensors

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
    def test_precompute_consumes_bounded_batches_and_deduplicates_through_storage(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        stored = set()
        visited = []
        encoded = []
        cache.data_backend.exists = lambda path: path in stored

        def encode(records, **kwargs):
            self.assertFalse(kwargs["split_between_processes"])
            self.assertLessEqual(len(records), 2)
            self.assertLessEqual(len(visited) - sum(len(batch) for batch in encoded), 4)
            encoded.append([record["prompt"] for record in records])
            stored.update(cache.hash_prompt_with_path(record) for record in records)

        cache.compute_prompt_embeddings_with_model = encode

        def records():
            for prompt in ("one", "one", "two", "three", "one", "four"):
                visited.append(prompt)
                yield {"prompt": prompt, "key": prompt}

        progress = []
        with patch.object(cache.data_backend, "list_files", side_effect=AssertionError("storage listing")):
            self.assertEqual(cache.precompute_embeddings(records(), batch_size=2, progress_callback=progress.append), 6)
        self.assertEqual(encoded, [["one"], ["two", "three"], ["four"]])
        self.assertEqual(progress[-1], 6)

    def test_offline_index_reuses_listed_hits_and_probes_only_missing_keys(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        records = [{"prompt": prompt} for prompt in ("warm", "warm", "missing")]
        warm, _, missing = [cache.hash_prompt_with_path(record) for record in records]
        current = {warm}
        cache.data_backend.iter_file_keys = MagicMock(side_effect=lambda *_: iter(sorted(current)))
        cache.data_backend.exists = MagicMock(side_effect=lambda key: key in current)

        def encode(records, **kwargs):
            current.update(cache.hash_prompt_with_path(record) for record in records)

        cache.compute_prompt_embeddings_with_model = MagicMock(side_effect=encode)
        with TemporaryDirectory() as output_dir:
            listing = ArtifactListing(
                cache.data_backend, cache.cache_dir + "/", output_dir, SimpleNamespace(is_local_main_process=True)
            )
            cache.preparation_listing = listing
            self.assertEqual(cache.precompute_embeddings(records, batch_size=3), 3)
            self.assertEqual(current, {warm, missing})
            cache.compute_prompt_embeddings_with_model.assert_called_once()
            cache.data_backend.exists.assert_called_once_with(missing)
            cache.data_backend.iter_file_keys.assert_called_once_with(["pt"], cache.cache_dir + "/")

            cache.compute_prompt_embeddings_with_model.reset_mock()
            cache.data_backend.exists.reset_mock()
            self.assertEqual(cache.precompute_embeddings(records, batch_size=3), 3)
            cache.data_backend.exists.assert_called_once_with(missing)
            cache.compute_prompt_embeddings_with_model.assert_not_called()
            cache.data_backend.iter_file_keys.assert_called_once()

            cache.finish_preparation()
            self.assertFalse(listing.path.exists())

            current.remove(warm)
            cache.preparation_listing = ArtifactListing(
                cache.data_backend, cache.cache_dir + "/", output_dir, SimpleNamespace(is_local_main_process=True)
            )
            cache.data_backend.exists.reset_mock()
            self.assertEqual(cache.precompute_embeddings(records, batch_size=3), 3)
            cache.data_backend.exists.assert_called_once_with(warm)
            cache.compute_prompt_embeddings_with_model.assert_called_once()
            cache.close()

    def test_encode_only_skips_existing_artifacts_and_batches_duplicate_prompts(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_encode_only=True, text_encoder_batch_size=2)
        self.assertTrue(cache.text_cache_ondemand)
        self.assertIsNone(cache.batch_write_thread)
        cache.data_backend.exists = cache.data_backend.torch_load = cache.data_backend.torch_save = MagicMock(
            side_effect=AssertionError("encode-only mode accessed the artifact store")
        )
        cache.model.unpack_text_embeddings_from_cache = lambda value: value
        cache.model.encode_text_batch = MagicMock(return_value={"value": torch.tensor([1, 2])})
        with patch.object(cache.data_backend, "list_files", side_effect=AssertionError("cache inventory")):
            result = cache.read_training_batch(["one", "two", "one"])
        self.assertEqual([record["value"].item() for record in result], [1, 2, 1])
        cache.model.encode_text_batch.assert_called_once()
        self.assertEqual(cache.data_backend.created_paths, [])
        cache.close()

    def test_precomputed_batch_returns_cache_hits_and_surfaces_missing_entries(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.model.unpack_text_embeddings_from_cache = lambda value: value
        cache.model.encode_text_batch = MagicMock(side_effect=AssertionError("precomputed cache encoded a prompt"))
        cache.data_backend.exists = lambda key: key == cache.hash_prompt_with_path({"prompt": "cached"})
        cache.data_backend.torch_load = MagicMock(return_value={"value": torch.tensor(7)})

        result = cache.read_training_batch(["cached", "cached"])
        self.assertEqual([entry["value"].item() for entry in result], [7, 7])
        cache.data_backend.torch_load.assert_called_once()
        with self.assertRaisesRegex(FileNotFoundError, "Missing precomputed text embedding"):
            cache.read_training_batch(["missing"])
        cache.model.encode_text_batch.assert_not_called()

    def test_direct_prompt_batch_returns_cache_misses_on_the_same_device_as_hits(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_cache_ondemand=True)
        cache.accelerator.device = torch.device("meta")
        cache.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        cache.model.encode_text_batch = MagicMock(return_value={"prompt_embeds": torch.ones(1, 2)})
        cache.model.unpack_text_embeddings_from_cache = lambda value: value
        cache.model.collate_prompt_embeds = lambda rows: {"prompt_embeds": torch.cat([row["prompt_embeds"] for row in rows])}
        cache.load_from_cache = MagicMock(side_effect=[{"prompt_embeds": torch.zeros(1, 2)}, FileNotFoundError()])
        cache.save_to_cache = MagicMock()

        mixed = cache.compute_embeddings_for_prompts(["hit", "miss"])

        torch.testing.assert_close(mixed["prompt_embeds"], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        for encode_only, load_from_cache in ((False, False), (True, True)):
            with self.subTest(encode_only=encode_only, load_from_cache=load_from_cache):
                cache.text_cache_encode_only = encode_only
                direct = cache.compute_embeddings_for_prompts(["encode"], load_from_cache=load_from_cache)
                self.assertEqual(direct["prompt_embeds"].device, cache.accelerator.device)

    def test_online_mixed_hits_batch_misses_preserve_context_duplicates_and_read_only(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME, text_cache_disable=True, text_encoder_batch_size=2)
        contexts = [{"image_path": f"{index}.jpg"} for index in range(3)]
        records = [{"prompt": "caption", "key": str(index), "metadata": context} for index, context in enumerate(contexts)]
        cached = cache.hash_prompt_with_path(records[1])
        cache.data_backend.read_cached_tensors = MagicMock(
            side_effect=lambda keys: ({"value": torch.tensor([99])} if key == cached else None for key in keys)
        )
        cache.data_backend.exists = MagicMock(side_effect=AssertionError("bypassed artifact provider batch read"))
        cache.model.requires_text_embed_image_context = lambda: True
        cache.model.pack_text_embeddings_for_cache = MagicMock(side_effect=lambda value: value)
        cache.model.unpack_text_embeddings_from_cache = lambda value: value
        cache.model.encode_text_batch = MagicMock(return_value={"value": torch.tensor([2, 0])})
        cache._queue_packed = MagicMock(side_effect=AssertionError("writes disabled"))
        with patch.object(cache.data_backend, "list_files", side_effect=AssertionError("inventory")):
            result = cache.read_training_batch([records[2], records[1], records[0], records[2]])
        self.assertEqual([r["value"].item() for r in result], [2, 99, 0, 2])
        requested = next(iter(cache.data_backend.read_cached_tensors.call_args.args))
        self.assertEqual(list(requested), [cache.hash_prompt_with_path(records[index]) for index in (2, 1, 0)])
        cache.model.encode_text_batch.assert_called_once_with(
            ["caption", "caption"], is_negative_prompt=False, prompt_contexts=[contexts[2], contexts[0]]
        )
        self.assertEqual(cache.model.pack_text_embeddings_for_cache.call_count, 2)

    def test_precompute_uses_model_batching_and_completes_writes(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, text_encoder_batch_size=2)

        class BatchModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def encode_text_batch(self, prompts, **kwargs):
                return {
                    "prompt_embeds": torch.ones(len(prompts), 2, 3),
                    "attention_mask": torch.ones(len(prompts), 2).squeeze(0),
                }

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache.model = BatchModel(TextEmbedCacheKey.CAPTION)
        stored = {}
        cache.save_to_cache = lambda filename, embeds: stored.update({filename: embeds})
        cache.wait_for_pending_writes = MagicMock()
        records = ({"prompt": prompt, "key": prompt} for prompt in ("one", "two", "three"))
        self.assertEqual(cache.precompute_embeddings(records, batch_size=2), 3)
        self.assertEqual(len(stored), 3)
        self.assertTrue(all(value["prompt_embeds"].shape == (1, 2, 3) for value in stored.values()))
        cached = cache.hash_prompt_with_path({"prompt": "three"})
        self.assertEqual(stored[cached]["attention_mask"].shape, (2,))
        cache.wait_for_pending_writes.assert_called()

        cache.text_cache_ondemand = cache.text_cache_disable = True
        cache.data_backend.read_cached_tensors = lambda keys: (stored.get(key) for key in keys)
        online = cache.read_training_batch(["unknown", "three", "unknown"])
        self.assertEqual([entry["attention_mask"].shape for entry in online], [(2,), (2,), (2,)])
        self.assertTrue(torch.equal(online[0]["prompt_embeds"], online[1]["prompt_embeds"]))

    def test_precompute_cancellation_stops_source_consumption(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        consumed = []

        def records():
            for index in range(100):
                consumed.append(index)
                yield {"prompt": str(index), "key": str(index)}

        def cancel(_current):
            raise InterruptedError("cancelled")

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            cache.precompute_embeddings(records(), batch_size=3, progress_callback=cancel)
        self.assertEqual(consumed, [])

    def test_prompt_calls_do_not_retain_records_or_reuse_them_for_empty_input(self):
        import gc
        import weakref

        class Marker:
            pass

        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        calls = []
        cache.compute_prompt_embeddings_with_model = lambda records, **kwargs: calls.append(len(records))
        marker = Marker()
        reference = weakref.ref(marker)
        cache.compute_embeddings_for_prompts([{"prompt": "one", "metadata": {"marker": marker}}], return_concat=False)
        cache.compute_embeddings_for_prompts([], return_concat=False)
        del marker
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(calls, [1])

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
                unpacked["prompt_embeds"] = torch.nn.functional.pad(embeddings["prompt_embeds"], (0, 2))
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

    def test_path_keyed_hash_distinguishes_prompt_variants(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        first = cache.hash_prompt_with_path({"prompt": "caption one", "key": "dataset-1:path/to/sample.png"})
        second = cache.hash_prompt_with_path({"prompt": "caption two", "key": "dataset-1:path/to/sample.png"})

        self.assertNotEqual(first, second)

    def test_caption_keyed_hash_does_not_add_prompt_component(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        first = cache.hash_prompt_with_path({"prompt": "caption one", "key": "shared-key"})
        second = cache.hash_prompt_with_path({"prompt": "caption two", "key": "shared-key"})

        self.assertEqual(first, second)

    def test_filtered_embeddings_use_the_effective_caption_for_offline_and_online_access(self):
        with TemporaryDirectory() as directory:
            filter_file = Path(directory) / "filters.txt"
            filter_file.write_text("s/blue/red\nmarker \n")
            offline = _make_cache(TextEmbedCacheKey.CAPTION, caption_filter_list=str(filter_file))
            filter_file.write_text("s/blue/green\nmarker \n")
            online = _make_cache(TextEmbedCacheKey.CAPTION, caption_filter_list=str(filter_file), text_cache_disable=True)

        plain = _make_cache(TextEmbedCacheKey.CAPTION)
        prompt = {"prompt": "blue bird"}
        duplicate = {"prompt": "marker red bird"}
        legacy_name = plain.hash_prompt_with_path(prompt)
        offline_name = offline.hash_prompt_with_path(prompt)
        self.assertEqual(offline_name, offline.hash_prompt_with_path(duplicate))
        self.assertNotEqual(offline_name, legacy_name)
        self.assertNotEqual(offline_name, online.hash_prompt_with_path(prompt))
        final_input_unchanged = {"prompt": "bird"}
        self.assertNotEqual(
            offline.hash_prompt_with_path(final_input_unchanged), plain.hash_prompt_with_path(final_input_unchanged)
        )

        stored = {legacy_name: {"value": torch.tensor(-1)}}
        offline.model.TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}
        offline.model.encode_text_batch = MagicMock(return_value={"value": torch.tensor(1)})
        offline.model.unpack_text_embeddings_from_cache = lambda value: value
        offline.data_backend.exists = lambda key: key in stored
        offline.data_backend.torch_load = stored.__getitem__
        offline.save_to_cache = lambda key, value: stored.update({key: value})

        online.model.unpack_text_embeddings_from_cache = lambda value: value
        online.model.encode_text_batch = MagicMock(return_value={"value": torch.tensor(2)})
        online.data_backend.read_cached_tensors = lambda keys: (stored.get(key) for key in keys)

        with patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("looked up filter globally")):
            self.assertEqual(offline.precompute_embeddings([prompt, duplicate]), 2)
            self.assertEqual(offline.read_training_batch([prompt, duplicate])[1]["value"].item(), 1)
            self.assertEqual(online.read_training_batch([prompt])[0]["value"].item(), 2)
        self.assertIn(offline_name, stored)
        offline.model.encode_text_batch.assert_called_once_with(["red bird"], is_negative_prompt=False, prompt_contexts=[{}])
        online.model.encode_text_batch.assert_called_once_with(
            ["green bird"], is_negative_prompt=False, prompt_contexts=[{}]
        )

    def test_filtered_path_keys_keep_context_and_special_prompt_identity(self):
        filtered = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME, caption_filter_list=["s/^/prefix "])
        plain = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        filtered.model.text_embed_cache_context_key = MagicMock(return_value="context")
        record = {"prompt": "bird", "key": "dataset:sample.jpg", "metadata": {"width": 512}}
        filename, prompt = filtered._embedding_input(record)
        self.assertEqual(prompt, "prefix bird")
        self.assertNotEqual(filename, filtered.hash_prompt_with_path({**record, "key": "dataset:other.jpg"}))
        filtered.model.text_embed_cache_context_key.assert_any_call(record["metadata"], prompt="prefix bird")

        plain.model.text_embed_cache_context_key = lambda metadata, prompt: "context"
        dropout = {"prompt": "", "key": "__caption_dropout__"}
        self.assertEqual(filtered.hash_prompt_with_path(dropout), plain.hash_prompt_with_path(dropout))
        negative = {"prompt": "bird", "key": "__validation_negative__bird"}
        self.assertEqual(
            filtered._embedding_input(negative, is_negative_prompt=True), (plain.hash_prompt_with_path(negative), "bird")
        )

    def test_path_keyed_empty_prompt_preserves_sentinel_hash(self):
        cache = _make_cache(TextEmbedCacheKey.DATASET_AND_FILENAME)
        record = {"prompt": "", "key": "__caption_dropout__"}

        self.assertEqual(
            cache.hash_prompt_with_path(record),
            os.path.join(cache.cache_dir, cache.create_hash("__caption_dropout__") + ".pt"),
        )

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

    def test_compute_embeddings_caches_each_uncached_path_keyed_prompt(self):
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
            [
                (["caption one"], [{"data_backend_id": "dataset-1", "dataset_relative_path": "path/to/sample.png"}]),
                (["caption two"], [{"data_backend_id": "dataset-1", "dataset_relative_path": "path/to/sample.png"}]),
            ],
        )
        self.assertEqual(len(saved), 2)

    def test_compute_embeddings_does_not_resplit_rank_local_records(self):
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

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

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

    def test_compute_embeddings_waits_for_inflight_cache_write(self):
        write_started = Event()
        release_write = Event()
        backend = _DummyDataBackend()

        def blocking_save(*_args, **_kwargs):
            write_started.set()
            release_write.wait(timeout=5)

        backend.torch_save = blocking_save

        @contextmanager
        def split_between_processes(records):
            yield records

        accelerator = SimpleNamespace(device="cpu", num_processes=1, split_between_processes=split_between_processes)

        class BatchModel(_DummyModel):
            TEXT_ENCODER_CONFIGURATION = {"text_encoder": {}}

            def __init__(self):
                super().__init__(TextEmbedCacheKey.CAPTION)

            def encode_text_batch(self, prompts, is_negative_prompt=False, prompt_contexts=None):
                return {
                    "prompt_embeds": torch.ones(len(prompts), 2, 3),
                    "attention_mask": torch.ones(len(prompts), 2, dtype=torch.bool),
                }

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

        cache = TextEmbeddingCache(
            id="backend",
            data_backend=backend,
            text_encoders=[SimpleNamespace(device="cpu")],
            tokenizers=[object()],
            accelerator=accelerator,
            cache_dir="/tmp/cache",
            model_type="test",
            prompt_handler=None,
            model=BatchModel(),
        )
        errors = []

        def compute_embeddings():
            try:
                cache.compute_embeddings_for_prompts(
                    [{"prompt": "tiny dataset prompt", "key": "tiny"}],
                    return_concat=False,
                    load_from_cache=False,
                    split_between_processes=False,
                )
            except Exception as error:
                errors.append(error)

        compute_thread = Thread(target=compute_embeddings)
        try:
            compute_thread.start()
            self.assertTrue(write_started.wait(timeout=2), "background cache write did not start")
            compute_thread.join(timeout=0.2)
            self.assertTrue(compute_thread.is_alive(), "precompute returned before the cache write completed")

            release_write.set()
            compute_thread.join(timeout=2)
            self.assertFalse(compute_thread.is_alive(), "precompute did not return after the cache write completed")
            self.assertEqual(errors, [])
        finally:
            release_write.set()
            compute_thread.join(timeout=2)
            cache.process_write_batches = False
            cache.batch_write_thread.join(timeout=2)

    def test_tensor_byte_budget_includes_inflight_storage_and_an_oversized_view(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION, write_budget_bytes=64)
        started, release = Event(), Event()

        def save(data, path):
            started.set()
            release.wait(timeout=5)

        cache.data_backend.torch_save = save
        # One scalar still pins 128 bytes of tensor storage.
        cache._queue_packed("one", {"value": torch.zeros(32)[:1]})
        self.assertTrue(started.wait(timeout=2))
        queued = Thread(target=cache._queue_packed, args=("two", {"value": torch.zeros(16)}))
        try:
            queued.start()
            queued.join(timeout=0.1)
            self.assertTrue(queued.is_alive())
            self.assertEqual(cache.peak_write_bytes, 128)
        finally:
            release.set()
            queued.join(timeout=2)
            cache.close()
        self.assertEqual(cache.pending_write_bytes, 0)
        self.assertEqual(cache.peak_write_bytes, 128)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            cache._queue_packed("late", {})

    def test_wait_for_pending_writes_reports_background_failure(self):
        cache = _make_cache(TextEmbedCacheKey.CAPTION)
        cache.process_write_batch = MagicMock(side_effect=OSError("disk write failed"))
        cache.process_write_batches = False
        cache.write_queue.put(({}, "failed.pt", 0))

        with self.assertLogs("TextEmbeddingCache", level="ERROR"):
            cache.batch_write_embeddings()

        with self.assertRaisesRegex(RuntimeError, "text embedding cache writes failed") as context:
            cache.wait_for_pending_writes()
        self.assertIsInstance(context.exception.__cause__, OSError)

    def test_validation_encode_marks_model_context(self):
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

            def unpack_text_embeddings_from_cache(self, embeddings):
                return embeddings

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
