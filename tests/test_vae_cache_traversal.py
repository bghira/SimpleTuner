import copy
import os
import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend


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

    def test_cache_scan_streams_keys_from_the_bound_artifact_store(self):
        cache = VAECache.__new__(VAECache)
        cache.cache_dir = "latents/"
        cache.read_batch_size = 2
        visited = []

        def entries(*args):
            for key in ("latents/a.pt", "latents/b.pt", "latents/c.pt"):
                visited.append(key)
                yield key

        cache.cache_data_backend = Mock()
        cache.cache_data_backend.iter_file_keys.side_effect = entries
        cache._read_from_storage_concurrently = Mock(side_effect=lambda keys, hide_errors: zip(keys, keys))
        results = cache.scan_cache_contents()
        self.assertEqual(next(results), ("latents/a.pt", "latents/a.pt"))
        self.assertEqual(visited, ["latents/a.pt", "latents/b.pt"])
        self.assertEqual(len(list(results)), 2)
        cache.cache_data_backend.iter_file_keys.assert_called_once_with(["pt"], "latents/")
        self.assertEqual(cache._read_from_storage_concurrently.call_count, 2)

    def test_native_cache_scan_checks_only_keys_from_the_prepared_view(self):
        metadata = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        metadata.native_policy = "global"
        metadata.vae_output_scaling_factor = 8
        metadata.iter_preparation_batches = Mock(return_value=iter([("wide", ("ok", "bad")), ("wide", ("nan", "missing"))]))
        metadata.scan_for_metadata = Mock(side_effect=AssertionError("read source catalogue"))
        source = Mock()
        dataset = DatasetRuntime("prepared", source, metadata, {})
        dataset.read_samples = Mock(side_effect=lambda refs: [{"target_size": (64, 32)} for _ in refs])
        cache = dataset.vae_cache = VAECache.__new__(VAECache)
        cache.metadata_backend = metadata
        cache.model = SimpleNamespace(vae_output_scaling_factor=4)
        cache.cache_data_backend = Mock()
        cache.read_batch_size = 2
        cache.cache_path_for_image = lambda ref: f"vae/{ref}.pt"
        contents = {
            "vae/ok.pt": {"latents": torch.zeros(4, 8, 16)},
            "vae/bad.pt": torch.zeros(4, 8, 32),
            "vae/nan.pt": torch.full((4, 8, 16), float("nan")),
            "vae/missing.pt": None,
        }
        cache._read_from_storage_concurrently = lambda keys, hide_errors: ((key, contents[key]) for key in keys)

        dataset.scan_cache_and_metadata("recreate")
        metadata.iter_preparation_batches.assert_called_once_with(0, 1, 2)
        self.assertEqual(
            [item.args[0] for item in cache.cache_data_backend.delete.call_args_list], ["vae/bad.pt", "vae/nan.pt"]
        )
        self.assertEqual([item.args[0] for item in dataset.read_samples.call_args_list], [["ok", "bad"], ["nan"]])
        cache.cache_data_backend.iter_file_keys.assert_not_called()
        source.delete.assert_not_called()
        with self.assertRaisesRegex(ValueError, "require.*recreate"):
            dataset.scan_cache_and_metadata("sync")

    def test_local_artifact_listing_streams_directories(self):
        storage = LocalDataBackend(accelerator=None, id="artifacts")

        def directories(*args, **kwargs):
            yield "latents", ["subdir"], ["a.pt", "notes.txt"]
            self.assertTrue(first_read)
            yield "latents/subdir", [], ["b.PT"]

        first_read = []
        with patch("simpletuner.helpers.data_backend.local.os.walk", side_effect=directories):
            keys = storage.iter_file_keys(["pt"], "latents")
            first_read.append(next(keys))
            self.assertEqual(first_read, [os.path.abspath("latents/a.pt")])
            self.assertEqual(list(keys), [os.path.abspath("latents/subdir/b.PT")])

    def test_cache_recreation_deletes_from_artifact_storage(self):
        source = Mock()
        store = Mock()
        store.type = "aws"
        metadata = SimpleNamespace(data_backend=source, save_cache=Mock())
        metadata.is_cache_inconsistent = lambda *args: MetadataBackend.is_cache_inconsistent(metadata, *args)
        cache = VAECache.__new__(VAECache)
        cache.cache_data_backend = store
        cache.generate_vae_cache_filename = lambda ref: ("vae/known.pt", "known.pt")
        cache.build_vae_cache_filename_map(["images/sample.jpg"])
        self.assertEqual(cache._image_filename_from_vaecache_filename("vae/known.pt"), "images/sample.jpg")
        invalid = torch.tensor([float("nan")])
        cache.scan_cache_contents = lambda: iter([("vae/unknown.pt", invalid), ("vae/known.pt", invalid)])
        with patch(
            "simpletuner.helpers.metadata.backends.base.StateTracker.get_args",
            return_value=SimpleNamespace(model_type="lora"),
        ):
            MetadataBackend.handle_vae_cache_inconsistencies(metadata, cache, "recreate")
        store.delete.assert_called_once_with("vae/known.pt")
        source.delete.assert_not_called()

    def test_accepted_partition_drives_cold_mixed_and_warm_latents(self):
        dataset = Mock()
        dataset.describe.return_value.accepted_samples = 3
        dataset.iter_preparation_batches.return_value = [("wide", ("a", "b")), ("square", ("c",))]
        dataset.read_samples.side_effect = lambda refs, fields: [{"image": ref} for ref in refs]
        stored = {"b"}
        cache = VAECache.__new__(VAECache)
        cache.vae_batch_size = 2
        cache.cache_path_for_image = lambda ref: "cache-" + ref
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.bulk_exists.side_effect = lambda keys: [
            key.removeprefix("cache-") in stored for key in keys
        ]
        cache.init_vae = Mock()
        cache.nsfw_check_enabled = False
        cache._process_images_in_batch = lambda refs, images, disable_queue: [
            (image, ref) for ref, image in zip(refs, images)
        ]
        cache._encode_images_in_batch = lambda pixels, disable_queue: [("cache-" + ref, ref, image) for image, ref in pixels]
        cache._write_latents_in_batch = lambda latents: stored.update(ref for _, ref, _ in latents)
        cache.prepare_dataset(dataset, worker=1, workers=2)
        self.assertEqual(stored, {"a", "b", "c"})
        dataset.iter_preparation_batches.assert_called_with(worker=1, workers=2, batch_size=2)
        cache.prepare_dataset(dataset, worker=1, workers=2)
        cache.init_vae.assert_called_once()
        self.assertEqual(dataset.read_samples.call_count, 2)

    def test_vae_inventory_is_only_needed_for_legacy_filtering_and_maintenance(self):
        cache = VAECache.__new__(VAECache)
        cache.metadata_backend = SimpleNamespace(native_policy=None)
        cache.vae_cache_ondemand = False
        cache.nsfw_check_enabled = False
        cache.delete_problematic_images = False
        self.assertFalse(cache.needs_legacy_cache_inventory({}))
        self.assertTrue(cache.needs_legacy_cache_inventory({"crop": True, "crop_aspect": "random"}))
        self.assertTrue(cache.needs_legacy_cache_inventory({"scan_for_errors": True}))
        cache.vae_cache_ondemand = True
        self.assertTrue(cache.needs_legacy_cache_inventory({"crop": True, "crop_aspect": "random"}))
        cache.vae_cache_ondemand = False
        cache.nsfw_check_enabled = True
        self.assertTrue(cache.needs_legacy_cache_inventory({}))
        cache.metadata_backend.native_policy = "global"
        self.assertFalse(cache.needs_legacy_cache_inventory({"scan_for_errors": True}))

    def test_legacy_preparation_preserves_partial_reads_and_accepts_audio_payloads(self):
        dataset = Mock(uses_native_schedule=False)
        dataset.describe.return_value.accepted_samples = 2
        dataset.iter_preparation_batches.return_value = [("audio", ("available.wav", "missing.wav"))]
        cache = VAECache.__new__(VAECache)
        cache.vae_batch_size = 2
        cache.nsfw_check_enabled = False
        cache.cache_path_for_image = lambda ref: "cache-" + ref
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.bulk_exists.side_effect = lambda keys: [False for _ in keys]
        cache._read_from_storage_concurrently = Mock(return_value=iter([("available.wav", "waveform")]))
        cache.init_vae = Mock()
        cache._process_images_in_batch = Mock(return_value=[("pixels", "available.wav")])
        cache._encode_images_in_batch = Mock(return_value=[("artifact", "available.wav", "latent")])
        cache._write_latents_in_batch = Mock()

        self.assertEqual(cache.prepare_dataset(dataset), 2)
        cache._process_images_in_batch.assert_called_once_with(["available.wav"], ["waveform"], disable_queue=True)
        cache._write_latents_in_batch.assert_called_once_with([("artifact", "available.wav", "latent")])
        dataset.read_samples.assert_not_called()

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
        cache.cache_path_for_image = lambda path: path + ".pt"
        cache._image_filename_from_vaecache_filename = lambda key: key.removesuffix(".pt")
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
