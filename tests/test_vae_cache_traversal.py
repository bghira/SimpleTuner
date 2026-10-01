import copy
import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend


class TestCacheGroups(unittest.TestCase):
    def test_default_preserves_bucket_and_file_order(self):
        buckets = {"wide": ["b", "a"], "empty": [], "square": ["c"]}
        self.assertEqual(list(BaseDataBackend.iter_cache_groups(None, buckets)), list(buckets.items()))

    def test_webshart_groups_by_numeric_shard_then_supplied_bucket_order(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        buckets = {
            "wide": ["webshart://10/1/b.jpg", "webshart://2/8/a.jpg", "webshart://2/3/c.jpg"],
            "square": ["webshart://10/2/d.jpg", "webshart://2/4/e.jpg"],
            "empty": [],
        }
        original = copy.deepcopy(buckets)
        self.assertEqual(
            list(backend.iter_cache_groups(buckets)),
            [
                ("wide", ["webshart://2/8/a.jpg", "webshart://2/3/c.jpg"]),
                ("square", ["webshart://2/4/e.jpg"]),
                ("wide", ["webshart://10/1/b.jpg"]),
                ("square", ["webshart://10/2/d.jpg"]),
            ],
        )
        self.assertEqual(buckets, original)
        self.assertEqual(list(backend.iter_cache_groups({})), [])
        with self.assertRaises(ValueError):
            list(backend.iter_cache_groups({"wide": ["invalid-sample"]}))


class TestVaeCacheTraversal(unittest.TestCase):
    @patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_ASPECTS": "false"})
    def test_shard_traversal_filters_before_grouping_and_drains_homogeneous_batches(self):
        self._run_pipeline(webshart=True)

    @patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_ASPECTS": "false"})
    def test_default_traversal_retains_bucket_order(self):
        self._run_pipeline(webshart=False)

    def _run_pipeline(self, webshart):
        wide0, wide1 = "webshart://0/0/wide.jpg", "webshart://1/0/wide.jpg"
        square0, square1 = "webshart://0/1/square.jpg", "webshart://1/1/square.jpg"
        restored = "webshart://0/2/restored.jpg"
        buckets = {
            "wide": [wide1, "cached", wide0, "other-rank"],
            "square": [square1, square0],
        }
        original = copy.deepcopy(buckets)
        cache = VAECache.__new__(VAECache)
        cache.id = "cache-traversal"
        cache.debug_log = Mock()
        cache._list_cached_images = Mock(return_value={"cached"})
        cache.metadata_backend = Mock()
        cache.metadata_backend.read_cache.return_value = buckets
        cache.metadata_backend.get_metadata_attribute_by_filepath.return_value = "wide"
        cache.local_unprocessed_files = [wide0, wide1, square0, square1, "cached", restored]
        cache.generate_vae_cache_filename = lambda path: (path + ".pt", path)
        cache._image_filename_from_vaecache_filename = lambda path: path
        cache.already_cached = Mock(return_value=False)
        cache.webhook_handler = None
        cache.max_workers = 2
        cache._vae_cache_written_lock = threading.Lock()
        cache._vae_cache_written_count = 0
        cache.accelerator = SimpleNamespace(is_local_main_process=False, wait_for_everyone=Mock())
        cache._close_nsfw_classifier_store = Mock()
        cache._finalize_deferred_metadata_filters = Mock()
        cache._write_nsfw_scan_report = Mock()
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        group_input = []

        def groups(files):
            group_input.append(copy.deepcopy(files))
            if webshart:
                yield from backend.iter_cache_groups(files)
            else:
                yield from BaseDataBackend.iter_cache_groups(None, files)

        cache.image_data_backend = SimpleNamespace(iter_cache_groups=groups)
        scanned = []

        def scan(files, bucket):
            scanned.extend(files)
            return files

        cache._filter_nsfw_relevant_files = scan
        for name in ("read_queue", "process_queue", "vae_input_queue", "write_queue"):
            setattr(cache, name, Queue())
        for name in ("read_batch_size", "process_queue_size", "vae_batch_size", "write_batch_size"):
            setattr(cache, name, 2)
        read_order, encoded_batches, written = [], [], []

        def transfer(source, destination, record=None):
            batch = [source.get() for _ in range(source.qsize())]
            if record is not None:
                record(batch)
            for item in batch:
                destination.put(item)

        cache.read_images_in_batch = lambda: transfer(
            cache.read_queue, cache.process_queue, lambda batch: read_order.extend(path for path, _ in batch)
        )
        cache._process_images_in_batch = lambda: transfer(cache.process_queue, cache.vae_input_queue)
        cache._encode_images_in_batch = lambda: transfer(cache.vae_input_queue, cache.write_queue, encoded_batches.append)

        def write():
            batch = [cache.write_queue.get() for _ in range(cache.write_queue.qsize())]
            written.extend(path for path, _ in batch)
            with cache._vae_cache_written_lock:
                cache._vae_cache_written_count += len(batch)

        cache._write_latents_in_batch = write
        progress = Mock()
        cache.process_buckets(progress_callback=progress)

        expected = [wide0, restored, square0, wide1, square1] if webshart else [wide1, wide0, restored, square1, square0]
        self.assertEqual(read_order, expected)
        self.assertEqual(scanned, expected)
        self.assertCountEqual(written, expected)
        self.assertEqual(cache._vae_cache_written_count, len(expected))
        self.assertEqual(group_input, [{"wide": [wide1, wide0, restored], "square": [square1, square0]}])
        self.assertEqual(buckets, original)
        self.assertTrue(encoded_batches)
        for batch in encoded_batches:
            self.assertEqual(len({bucket for _, bucket in batch}), 1)
            if webshart:
                self.assertEqual(len({backend.parse_sample_id(path).shard_idx for path, _ in batch}), 1)
        for name in ("read_queue", "process_queue", "vae_input_queue", "write_queue"):
            self.assertTrue(getattr(cache, name).empty())
        # A one-shard cache downloads once per shard instead of revisiting both for each aspect.
        shards = [backend.parse_sample_id(path).shard_idx for path in read_order]
        misses = sum(index == 0 or shard != shards[index - 1] for index, shard in enumerate(shards))
        self.assertEqual(misses, 2 if webshart else 4)
        self.assertEqual(progress.call_count, len(expected))
        cache.accelerator.wait_for_everyone.assert_called_once()
