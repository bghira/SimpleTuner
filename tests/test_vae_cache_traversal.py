import copy
import json
import os
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

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

    def test_ondemand_cache_reads_keep_duplicate_payloads_and_report_missing_refs(self):
        cache = VAECache.__new__(VAECache)
        cache.id = "ondemand"
        cache.vae_cache_ondemand = True
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.read_cached_tensors.side_effect = lambda keys: [None for _ in keys]
        cache.generate_vae_cache_filename = lambda ref, *, view=None: (f"vae/{ref}.pt", "unused")
        cache._read_from_storage_concurrently = Mock(
            side_effect=[iter([("b", 2), ("a", 1), ("a", 3)]), iter([("b", 2), ("a", 1)])]
        )
        cache._process_images_in_batch = Mock(return_value=[])
        cache._encode_images_in_batch = Mock(
            return_value=[("vae/b.pt", "b", 20), ("vae/a.pt", "a", 10), ("vae/a.pt", "a", 30)]
        )
        cache._write_latents_in_batch = Mock(side_effect=lambda entries: [value for _, _, value in entries])

        self.assertEqual(cache.encode_images([None, None, None], ["a", "b", "a"]), [10, 20, 30])
        cache._process_images_in_batch.assert_called_once_with(["a", "b", "a"], [1, 2, 3], disable_queue=True, view=None)
        with self.assertRaisesRegex(OSError, "Dataset ondemand could not read the requested image a"):
            cache.encode_images([None, None, None], ["a", "b", "a"])
        self.assertEqual(cache._process_images_in_batch.call_count, 1)

    def test_ondemand_cache_merges_a_hit_with_new_latents_in_request_order(self):
        cache = VAECache.__new__(VAECache)
        cache.id = "ondemand"
        cache.vae_cache_ondemand = True
        cache.vae_cache_disable = True
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.read_cached_tensors.side_effect = lambda keys: [
            torch.tensor(42) if key == "vae/warm.pt" else None for key in keys
        ]
        cache.generate_vae_cache_filename = lambda ref, *, view=None: (f"vae/{ref}.pt", ref)
        cache._read_from_storage_concurrently = Mock(return_value=iter([("last", "pixels"), ("first", "pixels")]))
        cache._process_images_in_batch = Mock(return_value=[])
        cache._encode_images_in_batch = Mock(
            return_value=[("vae/last.pt", "last", torch.tensor(3)), ("vae/first.pt", "first", torch.tensor(1))]
        )

        actual = cache.encode_images([None] * 3, ["first", "warm", "last"])
        self.assertEqual([value.item() for value in actual], [1, 42, 3])
        cache.cache_data_backend.read_cached_tensors.assert_called_once_with(["vae/first.pt", "vae/warm.pt", "vae/last.pt"])
        cache.cache_data_backend.write_batch.assert_not_called()
        cache._read_from_storage_concurrently.assert_called_once_with(["first", "last"], hide_errors=True)

    def test_latent_cache_samples_by_artifact_key_without_advancing_training_rng(self):
        cache = VAECache.__new__(VAECache)
        cache.cache_dir = "latents"
        params = torch.zeros(2, 8, 4, 4)
        rng = torch.get_rng_state().clone()
        batched = cache._sample_latent_distribution(DiagonalGaussianDistribution(params), ["latents/a.pt", "latents/b.pt"])
        cache.cache_dir = "another-store"
        single = cache._sample_latent_distribution(DiagonalGaussianDistribution(params[:1]), ["another-store/b.pt"])
        torch.testing.assert_close(batched[1:2], single, rtol=0, atol=0)
        cache._epoch_cache_dir = "another-store/epoch-latents-v1/run"
        scoped = cache._sample_latent_distribution(
            DiagonalGaussianDistribution(params[:1]), [cache._epoch_cache_dir + "/bc/de/b.pt"]
        )
        torch.testing.assert_close(single, scoped, rtol=0, atol=0)
        self.assertFalse(torch.equal(batched[0], batched[1]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_legacy_artifact_inventory_round_trips_and_deletes_storage_keys(self):
        with TemporaryDirectory() as directory:
            cache = VAECache.__new__(VAECache)
            cache.id = "artifacts"
            cache.cache_dir = "latents"
            cache.cache_data_backend = Mock()
            cache.cache_data_backend.iter_file_keys.return_value = iter(["latents/old.pt", "latents/new.pt"])
            cache._artifact_file_keys = None
            cache._artifact_inventory = Path(directory) / "all_vae_cache_files_artifacts.json"
            cache.max_workers = 1
            cache.write_batch_size = 1
            self.assertEqual(cache._read_artifact_files(), {})
            self.assertEqual(cache._refresh_artifact_files(), {"latents/old.pt": False, "latents/new.pt": False})
            cache._artifact_file_keys = None
            self.assertEqual(cache._read_artifact_files(), json.loads(cache._artifact_inventory.read_text()))
            cache.clear_cache()
            self.assertEqual(
                [call.args[0] for call in cache.cache_data_backend.delete.call_args_list],
                ["latents/old.pt", "latents/new.pt"],
            )
            self.assertEqual(json.loads(cache._artifact_inventory.read_text()), {})

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
        metadata._published = {"view_revision": "prepared-v1"}
        metadata.iter_preparation_batches = Mock(return_value=iter([("wide", ("ok", "bad")), ("wide", ("nan", "missing"))]))
        metadata.scan_for_metadata = Mock(side_effect=AssertionError("read source catalogue"))
        source = Mock()
        dataset = DatasetRuntime("prepared", source, metadata, {})
        dataset.read_samples = Mock(side_effect=lambda refs, *, view: [{"target_size": (64, 32)} for _ in refs])
        cache = dataset.vae_cache = VAECache.__new__(VAECache)
        cache.metadata_backend = metadata
        cache.model = SimpleNamespace(vae_output_scaling_factor=4)
        cache.cache_data_backend = Mock()
        cache.read_batch_size = 2
        cache.cache_path_for_image = lambda ref, *, view=None: f"vae/{ref}.pt"
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
        self.assertTrue(
            all(item.kwargs["view"] == dataset.published_read_view() for item in dataset.read_samples.call_args_list)
        )
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
        metadata = SimpleNamespace(data_backend=source, save_cache=Mock(), runtime_args=SimpleNamespace(model_type="lora"))
        metadata.is_cache_inconsistent = lambda *args: MetadataBackend.is_cache_inconsistent(metadata, *args)
        cache = VAECache.__new__(VAECache)
        cache.cache_data_backend = store
        cache.generate_vae_cache_filename = lambda ref, *, view=None: ("vae/known.pt", "known.pt")
        cache.build_vae_cache_filename_map(["images/sample.jpg"])
        self.assertEqual(cache._image_filename_from_vaecache_filename("vae/known.pt"), "images/sample.jpg")
        invalid = torch.tensor([float("nan")])
        cache.scan_cache_contents = lambda: iter([("vae/unknown.pt", invalid), ("vae/known.pt", invalid)])
        MetadataBackend.handle_vae_cache_inconsistencies(metadata, cache, "recreate")
        store.delete.assert_called_once_with("vae/known.pt")
        source.delete.assert_not_called()

    def test_epoch_rebuild_deletes_only_this_workers_prepared_artifact_keys(self):
        cache = VAECache.__new__(VAECache)
        cache.write_batch_size = 2
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.bulk_exists.return_value = (True, False)
        cache.cache_path_for_image = lambda ref: f"cache/{ref}.pt"
        cache.accelerator = Mock()
        cache.vae_cache_ondemand = False
        cache.prepare_dataset = Mock()
        dataset = Mock()
        dataset.iter_preparation_batches.return_value = iter([("square", ("present", "missing"))])

        cache.rebuild_prepared_cache(dataset, worker=3, workers=8)

        dataset.iter_preparation_batches.assert_called_once_with(worker=3, workers=8, batch_size=2)
        cache.cache_data_backend.delete.assert_called_once_with("cache/present.pt")
        cache.cache_data_backend.iter_file_keys.assert_not_called()
        cache.prepare_dataset.assert_called_once_with(dataset, worker=3, workers=8)
        self.assertEqual(cache.accelerator.wait_for_everyone.call_count, 2)

    def test_epoch_latent_expiry_uses_scoped_s3_prefix_on_global_leader(self):
        cache = VAECache.__new__(VAECache)
        cache.id = "dataset"
        cache._epoch_cache_dir = "latents/epoch-latents-v1/run"
        store = cache.cache_data_backend = Mock(type="aws")
        cache.accelerator = Mock(is_main_process=False, is_local_main_process=True)
        prefix = cache._epoch_cache_dir + "/"
        key = prefix + "ab/cd/one.pt"
        deleted = []
        store.delete_batch.side_effect = deleted.extend

        cache.expire_epoch_latents()
        store.iter_file_keys.assert_not_called()

        cache.accelerator.is_main_process = True
        store.iter_file_keys.return_value = iter([key])
        cache.expire_epoch_latents()
        store.iter_file_keys.assert_called_once_with(["pt"], prefix)
        self.assertEqual(deleted, [key])
        self.assertEqual(cache.accelerator.wait_for_everyone.call_count, 2)

        deleted.clear()
        store.iter_file_keys.return_value = iter(["latents/epoch-latents-v1/run-other/one.pt"])
        with self.assertRaisesRegex(RuntimeError, "outside its prefix"):
            cache.expire_epoch_latents()
        self.assertEqual(deleted, [])

    def test_accepted_partition_drives_cold_mixed_and_warm_latents(self):
        dataset = Mock(requires_complete_preparation=True, uses_native_schedule=False)
        dataset.describe.return_value.accepted_samples = 3
        dataset.iter_preparation_batches.return_value = [("wide", ("a", "b")), ("square", ("c",))]
        dataset.read_sample_images.side_effect = lambda refs, **_: list(refs)
        stored = {"b"}
        cache = VAECache.__new__(VAECache)
        cache.vae_batch_size = 2
        cache.cache_path_for_image = lambda ref, *, view=None: "cache-" + ref
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.bulk_exists.side_effect = lambda keys: [
            key.removeprefix("cache-") in stored for key in keys
        ]
        cache.init_vae = Mock()
        cache.nsfw_check_enabled = False
        cache.delete_problematic_images = False
        cache._process_images_in_batch = lambda refs, images, disable_queue, *, view=None: [
            (image, ref) for ref, image in zip(refs, images)
        ]
        cache._encode_images_in_batch = lambda pixels, disable_queue, *, view=None: [
            ("cache-" + ref, ref, image) for image, ref in pixels
        ]
        cache._write_latents_in_batch = lambda latents: stored.update(ref for _, ref, _ in latents)
        cache.prepare_dataset(dataset, worker=1, workers=2)
        self.assertEqual(stored, {"a", "b", "c"})
        dataset.iter_preparation_batches.assert_called_with(worker=1, workers=2, batch_size=2)
        cache.prepare_dataset(dataset, worker=1, workers=2)
        cache.init_vae.assert_called_once()
        self.assertEqual(dataset.read_sample_images.call_count, 2)

        stored.remove("a")
        cache.delete_problematic_images = True
        ledger = Mock()
        dataset.metadata_backend.collect_exclusions.return_value = nullcontext(ledger)
        dataset.read_sample_images.side_effect = lambda refs, *, allow_missing: [None] if allow_missing else list(refs)
        source = dataset.data_backend
        cache.prepare_dataset(dataset, worker=1, workers=2)
        ledger.add.assert_called_once_with("a", "wide", "decode")
        self.assertEqual(stored, {"b", "c"})
        source.delete.assert_not_called()

    def test_aws_latent_preparation_lists_warm_prefix_and_probes_only_misses(self):
        from botocore.exceptions import ClientError

        from simpletuner.helpers.data_backend.aws import S3DataBackend

        dataset = Mock(requires_complete_preparation=False, uses_native_schedule=False)
        dataset.describe.return_value.accepted_samples = 3
        dataset.iter_preparation_batches.return_value = [("wide", ("a", "b")), ("square", ("c",))]
        with TemporaryDirectory() as directory, patch("simpletuner.helpers.data_backend.aws.boto3.client"):
            store = S3DataBackend("latents", "artifacts", None)
            stored = {"latents/a.pt", "latents/b.pt", "latents/c.pt"}
            paginate = store.client.get_paginator.return_value.paginate
            paginate.side_effect = lambda **kwargs: ({"Contents": [{"Key": key}]} for key in sorted(stored))

            def head(*, Bucket, Key):
                if Key not in stored:
                    raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
                return {}

            store.client.exceptions.NoSuchKey = type("NoSuchKey", (ClientError,), {})
            store.client.head_object.side_effect = head
            cache = VAECache.__new__(VAECache)
            cache.vae_batch_size = 2
            cache.cache_dir = "latents"
            cache._artifact_inventory = Path(directory) / "legacy-inventory.json"
            cache.accelerator = SimpleNamespace(is_local_main_process=True)
            cache.cache_data_backend = store
            cache.cache_path_for_image = lambda ref, *, view=None: f"latents/{ref}.pt"
            cache.init_vae = Mock()
            cache.nsfw_check_enabled = False
            cache.delete_problematic_images = False
            cache._read_from_storage_concurrently = Mock(side_effect=lambda refs, hide_errors: zip(refs, refs))
            cache._process_images_in_batch = lambda refs, images, disable_queue, view: list(zip(images, refs))
            cache._encode_images_in_batch = lambda pixels, disable_queue, view: [
                (f"latents/{ref}.pt", ref, image) for image, ref in pixels
            ]
            cache._write_latents_in_batch = Mock(side_effect=lambda batch: stored.update(key for key, _, _ in batch))

            self.assertEqual(cache.prepare_dataset(dataset), 3)
            store.client.head_object.assert_not_called()
            store.client.get_object.assert_not_called()
            cache._write_latents_in_batch.assert_not_called()
            cache.init_vae.assert_not_called()
            paginate.assert_called_once_with(Bucket="artifacts", Prefix="latents/", MaxKeys=1000)

            stored.remove("latents/b.pt")
            self.assertEqual(cache.prepare_dataset(dataset), 3)
            self.assertEqual(paginate.call_count, 2)
            store.client.head_object.assert_called_once_with(Bucket="artifacts", Key="latents/b.pt")
            cache._read_from_storage_concurrently.assert_called_once_with(["b"], hide_errors=True)
            cache._write_latents_in_batch.assert_called_once_with([("latents/b.pt", "b", "b")])
            self.assertEqual(list((Path(directory) / ".artifact-listings").iterdir()), [])

    def test_vae_inventory_is_only_needed_for_legacy_filtering_and_maintenance(self):
        cache = VAECache.__new__(VAECache)
        cache.metadata_backend = SimpleNamespace(requires_complete_preparation=False)
        cache.vae_cache_ondemand = False
        cache.nsfw_check_enabled = False
        cache.delete_problematic_images = False
        self.assertFalse(cache.needs_legacy_cache_inventory({}))
        self.assertFalse(cache.needs_legacy_cache_inventory({"crop": True, "crop_aspect": "random"}))
        self.assertTrue(cache.needs_legacy_cache_inventory({"scan_for_errors": True}))
        cache.vae_cache_ondemand = True
        self.assertTrue(cache.needs_legacy_cache_inventory({"vae_cache_clear_each_epoch": True}))
        cache.vae_cache_ondemand = False
        cache.nsfw_check_enabled = True
        self.assertTrue(cache.needs_legacy_cache_inventory({}))
        cache.metadata_backend.requires_complete_preparation = True
        self.assertFalse(cache.needs_legacy_cache_inventory({"scan_for_errors": True}))

    def test_legacy_preparation_preserves_partial_reads_and_accepts_audio_payloads(self):
        dataset = Mock(requires_complete_preparation=False)
        dataset.describe.return_value.accepted_samples = 2
        dataset.iter_preparation_batches.return_value = [("audio", ("available.wav", "missing.wav"))]
        cache = VAECache.__new__(VAECache)
        cache.vae_batch_size = 2
        cache.nsfw_check_enabled = False
        cache.cache_path_for_image = lambda ref, *, view=None: "cache-" + ref
        cache.cache_data_backend = Mock()
        cache.cache_data_backend.bulk_exists.side_effect = lambda keys: [False for _ in keys]
        cache._read_from_storage_concurrently = Mock(return_value=iter([("available.wav", "waveform")]))
        cache.init_vae = Mock()
        cache._process_images_in_batch = Mock(return_value=[("pixels", "available.wav")])
        cache._encode_images_in_batch = Mock(return_value=[("artifact", "available.wav", "latent")])
        cache._write_latents_in_batch = Mock()

        self.assertEqual(cache.prepare_dataset(dataset), 2)
        cache._process_images_in_batch.assert_called_once_with(
            ["available.wav"], ["waveform"], disable_queue=True, view=None
        )
        cache._write_latents_in_batch.assert_called_once_with([("artifact", "available.wav", "latent")])
        dataset.read_sample_images.assert_not_called()

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
        cache.generate_vae_cache_filename = lambda path, *, view=None: (path + ".pt", path)
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
