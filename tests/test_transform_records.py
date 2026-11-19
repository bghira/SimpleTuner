import json
import multiprocessing
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend
from simpletuner.helpers.metadata.transform_records import DECIDED, PythonTransformCache, encode_transform
from simpletuner.helpers.metadata.utils.duplicator import DatasetDuplicator
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import DatasetViewerService
from tests.unittest_support import APITestCase


def _write_in_another_process(directory, recipe, partition, records):
    cache = PythonTransformCache(directory, recipe)
    cache.enqueue(partition, records)
    cache.close()


class PythonTransformCacheTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)

    def open(self, recipe="recipe", **options):
        cache = PythonTransformCache(self.directory, recipe, **options)
        self.addCleanup(cache.close)
        return cache

    def test_reads_keep_the_order_and_repeats_asked_for_and_miss_as_none(self):
        cache = self.open()
        cache.enqueue("images", [(7, b"seven"), ("a/b.png", b"path")])
        cache.flush()
        requests = [("images", "a/b.png"), ("images", 8), ("images", 7), ("other", 7), ("images", "a/b.png")]
        self.assertEqual(cache.read_batch(requests), [b"path", None, b"seven", None, b"path"])

    def test_the_first_committed_value_is_kept(self):
        cache = self.open()
        cache.enqueue("images", [(1, b"first")])
        cache.enqueue("images", [(1, b"second"), (2, b"two")])
        cache.flush()
        self.assertEqual(self.open().read_batch([("images", 1), ("images", 2)]), [b"first", b"two"])

    def test_recipes_are_kept_apart(self):
        cache = self.open("one")
        cache.enqueue("images", [(1, b"one")])
        cache.flush()
        self.assertEqual(self.open("two").read_batch([("images", 1)]), [None])
        self.assertNotEqual(cache.path("images"), self.open("two").path("images"))

    def test_another_process_reads_what_one_recorded(self):
        context = multiprocessing.get_context("spawn")
        writer = context.Process(
            target=_write_in_another_process, args=(str(self.directory), "recipe", "images", [(3, b"three")])
        )
        writer.start()
        writer.join(60)
        self.assertEqual(writer.exitcode, 0)
        self.assertEqual(self.open().read_batch([("images", 3)]), [b"three"])

    def test_a_full_queue_makes_the_caller_wait(self):
        cache = self.open(queue_batches=1)
        release = threading.Event()
        self.addCleanup(release.set)
        open_writer = cache._open_writer

        def blocked(partition):
            release.wait()
            return open_writer(partition)

        with patch.object(cache, "_open_writer", side_effect=blocked):
            cache.enqueue("images", [(1, b"1")])  # taken by the writer, which waits
            cache.enqueue("images", [(2, b"2")])  # fills the queue
            third = threading.Thread(target=cache.enqueue, args=("images", [(3, b"3")]))
            third.start()
            third.join(0.3)
            self.assertTrue(third.is_alive())
            release.set()
            third.join(10)
            self.assertFalse(third.is_alive())
            cache.flush()
        self.assertEqual(cache.read_batch([("images", row) for row in (1, 2, 3)]), [b"1", b"2", b"3"])

    def test_a_failed_write_is_reported_at_flush_and_refuses_more(self):
        cache = self.open()
        with patch.object(cache, "_open_writer", side_effect=PermissionError("read-only volume")):
            cache.enqueue("images", [(1, b"1")])
            with self.assertRaisesRegex(OSError, "transform cache write failed: read-only volume"):
                cache.flush()
        with self.assertRaisesRegex(OSError, "read-only volume"):
            cache.enqueue("images", [(2, b"2")])
        cache._failure = None

    def test_a_batch_over_the_byte_limit_is_refused_whole(self):
        cache = self.open(max_batch_bytes=100)
        with self.assertRaisesRegex(ValueError, "exceeds max_batch_bytes"):
            cache.enqueue("images", [(1, b"x" * 40), (2, b"x" * 40)])
        cache.enqueue("images", [(1, b"x" * 40)])
        cache.flush()
        self.assertEqual(cache.read_batch([("images", 1), ("images", 2)]), [b"x" * 40, None])

    def test_a_path_row_counts_against_the_byte_limit(self):
        cache = self.open(max_batch_bytes=100)
        cache.enqueue("p", [(1, b"x" * 60)])
        with self.assertRaisesRegex(ValueError, "exceeds max_batch_bytes"):
            cache.enqueue("p", [("d" * 10, b"x" * 60)])

    def test_replace_records_over_what_is_recorded(self):
        cache = self.open()
        cache.enqueue("images", [(1, b"decided"), (2, b"two")])
        cache.flush()
        cache.replace("images", [(1, b"edited")])
        cache.enqueue("images", [(1, b"decided again")])
        cache.flush()
        self.assertEqual(cache.read_batch([("images", 1), ("images", 2)]), [b"edited", b"two"])

    def test_a_batch_queued_before_replace_keeps_the_replaced_value(self):
        cache = self.open()
        release = threading.Event()
        self.addCleanup(release.set)
        open_writer = cache._open_writer

        taken = threading.Event()

        def writer_blocked(partition):
            if threading.current_thread() is cache._writer:
                taken.set()
                release.wait()
            return open_writer(partition)

        with patch.object(cache, "_open_writer", side_effect=writer_blocked):
            cache.enqueue("images", [(1, b"decided")])
            self.assertTrue(taken.wait(10))
            cache.replace("images", [(1, b"edited")])
            release.set()
            cache.flush()
        self.assertEqual(cache.read_batch([("images", 1)]), [b"edited"])

    def test_a_closed_cache_takes_nothing_more(self):
        cache = self.open()
        cache.enqueue("images", [(1, b"1")])
        cache.close()
        self.assertEqual(cache.read_batch([("images", 1)]), [b"1"])
        with self.assertRaisesRegex(ValueError, "closed"):
            cache.enqueue("images", [(2, b"2")])

    def test_a_file_without_its_table_yet_reads_as_nothing_recorded(self):
        cache = self.open()
        path = cache.path("images")
        path.parent.mkdir(parents=True)
        path.touch()
        self.assertEqual(cache.read_batch([("images", 1)]), [None])


def _geometry(aspect_ratio, crop=(0, 0)):
    width = int(64 * aspect_ratio)
    return {
        "aspect_ratio": aspect_ratio,
        "target_size": (width, 64),
        "intermediary_size": (width, 64),
        "crop_coordinates": crop,
    }


class _Accelerator:
    def __init__(self, process_index):
        self.process_index = process_index
        self.num_processes = 2
        self.is_main_process = self.is_local_main_process = process_index == 0


class _BucketStorage:
    """A data backend whose files are not on this machine's disk, as S3's are."""

    type = "aws"

    def __init__(self, backend_id):
        self.id = backend_id
        self.files: dict[str, bytes] = {}

    def exists(self, path):
        return str(path) in self.files

    def read(self, path):
        return self.files[str(path)]

    def write(self, path, data):
        self.files[str(path)] = data.encode("utf-8") if isinstance(data, str) else data

    def get_abs_path(self, path):
        return path


class DiscoveryTransformRecordTests(unittest.TestCase):
    """A Discovery dataset keeps decided geometry in the store and what it measured in its metadata file."""

    FILES = ["a.png", "b.png"]

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = {"resolution": 1.0, "resolution_type": "area", "crop": False}
        self.storage = None
        StateTracker.set_args(
            SimpleNamespace(
                aspect_bucket_worker_count=1,
                enable_multiprocessing=False,
                aspect_bucket_disable_rebuild=False,
                cache_dir=str(self.root / "cache"),
                disable_bucket_pruning=True,
                train_batch_size=1,
                gradient_accumulation_steps=1,
                ignore_missing_files=True,
                model_type="",
            )
        )
        StateTracker.set_model(None)
        StateTracker.set_epoch(1)
        self.addCleanup(StateTracker.set_epoch, 1)
        self.addCleanup(StateTracker.data_backends.pop, "images", None)
        self.addCleanup(StateTracker.all_image_files.pop, "images", None)

    def backend(self, process_index=0):
        StateTracker.register_data_backend({"id": "images"})
        StateTracker.set_data_backend_config("images", dict(self.config))
        StateTracker.all_image_files["images"] = dict.fromkeys(self.FILES, False)
        if self.storage is not None:
            data_backend, root = self.storage, "prefix"
        else:
            data_backend, root = LocalDataBackend(accelerator=None, id="images"), str(self.root / "images")
        backend = DiscoveryMetadataBackend(
            id="images",
            instance_data_dir=root,
            cache_file=f"{root}/aspect_ratio_bucket_indices",
            metadata_file=f"{root}/aspect_ratio_bucket_metadata",
            data_backend=data_backend,
            accelerator=_Accelerator(process_index),
            batch_size=1,
            resolution=1.0,
            resolution_type="area",
            cache_file_suffix="images",
        )
        self.addCleanup(backend._close_transforms)
        return backend

    def compute(self, backend, decide, ignore_existing_cache=False):
        decided = []

        def process(path, buckets, metadata_updates=None, **kwargs):
            metadata = {"original_size": (128, 64), **decide(path)}
            buckets.setdefault(str(metadata["aspect_ratio"]), []).append(path)
            metadata_updates[path] = metadata
            decided.append(path)
            return buckets

        with patch.object(backend, "_process_for_bucket", side_effect=process):
            backend.compute_aspect_ratio_bucket_indices(ignore_existing_cache=ignore_existing_cache)
        return decided

    @staticmethod
    def buckets(backend):
        return {key: sorted(map(str, paths)) for key, paths in backend.aspect_ratio_bucket_indices.items()}

    def geometry_of(self, backend, path):
        return {key: backend.image_metadata[path][key] for key in DECIDED}

    def test_the_metadata_file_holds_what_was_measured_and_the_store_what_was_decided(self):
        main = self.backend()
        self.compute(main, lambda path: _geometry(2.0, (0, 3)))
        saved = json.loads((self.root / "images" / "aspect_ratio_bucket_metadata_images.json").read_text())
        self.assertEqual(saved, {path: {"original_size": [128, 64]} for path in self.FILES})
        index = json.loads((self.root / "images" / "aspect_ratio_bucket_indices_images.json").read_text())
        self.assertEqual(index["transform_recipe"], "discovery")

        reader = self.backend(process_index=1)
        reader.load_image_metadata()
        self.assertEqual(self.geometry_of(reader, "a.png"), _geometry(2.0, (0, 3)))
        self.assertEqual(reader.image_metadata["a.png"]["original_size"], [128, 64])

    def test_a_bucket_index_is_written_after_the_geometry_it_lists_is_committed(self):
        main = self.backend()
        index_written, seen = threading.Event(), []
        open_writer, write = PythonTransformCache._open_writer, main.data_backend.write

        def held_writer(cache, partition):
            index_written.wait(0.5)
            return open_writer(cache, partition)

        def write_and_look(path, data):
            if str(path).endswith("aspect_ratio_bucket_indices_images.json"):
                store = PythonTransformCache(self.root / "images" / "transforms", "discovery")
                seen.append(store.read_batch([("images", "a.png")])[0])
                store.close()
                index_written.set()
            return write(path, data)

        with (
            patch.object(PythonTransformCache, "_open_writer", autospec=True, side_effect=held_writer),
            patch.object(main.data_backend, "write", side_effect=write_and_look),
        ):
            main.record_transforms({"a.png": _geometry(2.0)})
            main.save_cache()
        self.assertEqual(seen, [encode_transform(_geometry(2.0))])

    def test_a_rank_reloading_after_an_epoch_recrop_reads_the_new_epochs_geometry(self):
        self.config.update(crop=True, crop_aspect="random")
        main = self.backend()
        self.compute(main, lambda path: _geometry(2.0))
        reader = self.backend(process_index=1)
        reader.load_image_metadata()
        self.assertEqual(self.geometry_of(reader, "a.png"), _geometry(2.0))

        StateTracker.set_epoch(2)
        self.compute(main, lambda path: _geometry(0.5), ignore_existing_cache=True)
        reader.reload_cache(set_config=False)
        self.assertEqual(self.geometry_of(reader, "a.png"), _geometry(0.5))
        self.assertEqual(self.buckets(reader), {"0.5": self.FILES})

        first_epoch = PythonTransformCache(self.root / "images" / "transforms", "discovery-epoch-1")
        self.addCleanup(first_epoch.close)
        self.assertEqual(first_epoch.read_batch([("images", "a.png")]), [encode_transform(_geometry(2.0))])

    def test_a_resumed_run_reads_under_the_recipe_its_bucket_index_names(self):
        self.config.update(crop=True, crop_aspect="random")
        StateTracker.set_epoch(3)
        self.compute(self.backend(), lambda path: _geometry(0.5))
        StateTracker.set_epoch(1)
        resumed = self.backend()
        self.assertEqual(self.compute(resumed, lambda path: _geometry(2.0)), [])
        self.assertEqual(self.geometry_of(resumed, "a.png"), _geometry(0.5))

    def test_a_forced_rescan_decides_every_sample_again(self):
        main = self.backend()
        self.compute(main, lambda path: _geometry(2.0))
        self.compute(main, lambda path: _geometry(0.5), ignore_existing_cache=True)
        reader = self.backend(process_index=1)
        reader.load_image_metadata()
        self.assertEqual(self.geometry_of(reader, "b.png"), _geometry(0.5))

    def test_a_sample_takes_the_geometry_first_recorded_for_it(self):
        recorded = PythonTransformCache(self.root / "images" / "transforms", "discovery")
        recorded.enqueue("images", [("a.png", encode_transform(_geometry(2.0, (1, 1))))])
        recorded.close()
        main = self.backend()
        self.compute(main, lambda path: _geometry(2.0, (5, 5)))
        self.assertEqual(self.geometry_of(main, "a.png"), _geometry(2.0, (1, 1)))
        self.assertEqual(self.geometry_of(main, "b.png"), _geometry(2.0, (5, 5)))

    def test_a_bucket_index_that_names_no_recipe_is_bucketed_again(self):
        images = self.root / "images"
        images.mkdir()
        (images / "aspect_ratio_bucket_indices_images.json").write_text(
            json.dumps({"config": {}, "aspect_ratio_bucket_indices": {"2.0": self.FILES}})
        )
        (images / "aspect_ratio_bucket_metadata_images.json").write_text(
            json.dumps({path: {"original_size": [128, 64], **_geometry(2.0)} for path in self.FILES})
        )
        main = self.backend()
        self.assertEqual(sorted(self.compute(main, lambda path: _geometry(0.5))), self.FILES)
        self.assertEqual(self.buckets(main), {"0.5": self.FILES})
        self.assertEqual(self.geometry_of(main, "a.png"), _geometry(0.5))

    def test_a_store_not_on_local_disk_is_published_beside_the_metadata_file(self):
        self.storage = _BucketStorage("images")
        main = self.backend()
        self.compute(main, lambda path: _geometry(2.0, (0, 7)))
        published = [key for key in self.storage.files if key.startswith("prefix/transforms/")]
        self.assertEqual(len(published), 1)

        reader = self.backend(process_index=1)
        reader.load_image_metadata()
        self.assertEqual(self.geometry_of(reader, "b.png"), _geometry(2.0, (0, 7)))
        local_copy = reader._transform_store().path("images")
        self.assertTrue(local_copy.is_relative_to(self.root / "cache" / "transforms" / "images" / "process-1"))
        self.assertEqual(local_copy.read_bytes(), self.storage.files[published[0]])


class ParquetTransformRecordTests(unittest.TestCase):
    """A Parquet-described dataset keeps decided geometry in the store and what it measured in its metadata file."""

    SIZES = {"a.png": (128, 64), "b.png": (64, 128)}

    def setUp(self):
        import pandas as pd

        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        rows = [dict(filename=name, caption=name, width=w, height=h) for name, (w, h) in self.SIZES.items()]
        pd.DataFrame(rows).to_parquet(self.root / "captions.parquet")
        StateTracker.set_args(
            SimpleNamespace(
                aspect_bucket_worker_count=1,
                enable_multiprocessing=False,
                aspect_bucket_disable_rebuild=False,
                aspect_bucket_alignment=8,
                aspect_bucket_rounding=2,
                cache_dir=str(self.root / "cache"),
                disable_bucket_pruning=True,
                train_batch_size=1,
                gradient_accumulation_steps=1,
                ignore_missing_files=True,
                model_type="",
                resolution_type="pixel",
            )
        )
        StateTracker.set_model(None)
        StateTracker.set_epoch(1)
        self.addCleanup(StateTracker.set_epoch, 1)
        self.addCleanup(StateTracker.data_backends.pop, "images", None)
        self.addCleanup(StateTracker.all_image_files.pop, "images", None)

    def backend(self, process_index=0):
        from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend

        StateTracker.register_data_backend({"id": "images"})
        StateTracker.set_data_backend_config("images", {"resolution": 64, "resolution_type": "pixel", "crop": False})
        StateTracker.all_image_files["images"] = dict.fromkeys(str(self.images / name) for name in self.SIZES)
        backend = ParquetMetadataBackend(
            id="images",
            instance_data_dir=str(self.images),
            cache_file=str(self.images / "aspect_ratio_bucket_indices"),
            metadata_file=str(self.images / "aspect_ratio_bucket_metadata"),
            data_backend=LocalDataBackend(accelerator=None, id="images"),
            accelerator=_Accelerator(process_index),
            batch_size=1,
            resolution=64,
            resolution_type="pixel",
            parquet_config={
                "path": str(self.root / "captions.parquet"),
                "filename_column": "filename",
                "caption_column": "caption",
                "width_column": "width",
                "height_column": "height",
                "identifier_includes_extension": True,
            },
            cache_file_suffix="images",
        )
        self.addCleanup(backend._close_transforms)
        return backend

    def saved(self, kind):
        return json.loads((self.images / f"aspect_ratio_bucket_{kind}_images.json").read_text())

    def test_the_metadata_file_holds_what_was_measured_and_the_store_what_was_decided(self):
        main = self.backend()
        main.compute_aspect_ratio_bucket_indices()
        paths = sorted(str(self.images / name) for name in self.SIZES)
        self.assertEqual(sorted(self.saved("metadata")), paths)
        self.assertTrue(all(not set(DECIDED) & set(metadata) for metadata in self.saved("metadata").values()))
        self.assertEqual(self.saved("indices")["transform_recipe"], "parquet")

        reader = self.backend(process_index=1)
        reader.reload_cache()
        for path in paths:
            self.assertEqual(
                {key: reader.image_metadata[path][key] for key in DECIDED},
                {key: main.image_metadata[path][key] for key in DECIDED},
            )

    def test_a_sample_takes_the_geometry_first_recorded_for_it(self):
        path = str(self.images / "a.png")
        recorded = PythonTransformCache(self.images / "transforms", "parquet")
        recorded.enqueue("images", [(path, encode_transform(_geometry(2.0, (1, 1))))])
        recorded.close()
        main = self.backend()
        main.compute_aspect_ratio_bucket_indices()
        self.assertEqual({key: main.image_metadata[path][key] for key in DECIDED}, _geometry(2.0, (1, 1)))

    def test_a_bucket_index_that_names_no_recipe_is_bucketed_again(self):
        main = self.backend()
        main.compute_aspect_ratio_bucket_indices()
        buckets = main.aspect_ratio_bucket_indices
        index = self.saved("indices")
        del index["transform_recipe"]
        index["aspect_ratio_bucket_indices"] = {"9.9": [path for paths in buckets.values() for path in paths]}
        (self.images / "aspect_ratio_bucket_indices_images.json").write_text(json.dumps(index))
        legacy = self.backend()
        legacy.reload_cache()
        legacy.compute_aspect_ratio_bucket_indices()
        self.assertEqual(legacy.aspect_ratio_bucket_indices, buckets)
        self.assertEqual(self.saved("indices")["transform_recipe"], "parquet")


class _HubDataset:
    """A Hugging Face dataset in memory; its cache files are written where the real backend writes them."""

    type = "huggingface"
    image_column = "image"
    streaming = False
    cache_file_path = HuggingfaceDatasetsBackend.cache_file_path
    write = HuggingfaceDatasetsBackend.write
    read = HuggingfaceDatasetsBackend.read
    exists = HuggingfaceDatasetsBackend.exists

    def __init__(self, cache_dir, sizes):
        self.id = "hub"
        self.cache_dir = str(cache_dir)
        self.dataset = [
            {"width": width, "height": height, "caption": f"sample {index}"} for index, (width, height) in enumerate(sizes)
        ]

    def _get_index_from_path(self, path):
        return int(str(path).split(".")[0])

    def get_dataset_item(self, index):
        return self.dataset[index]

    def get_abs_path(self, path):
        return path


class _HubAccelerator(_Accelerator):
    def main_process_first(self):
        return nullcontext()

    def wait_for_everyone(self):
        pass


class HuggingfaceTransformRecordTests(unittest.TestCase):
    """A Hugging Face dataset keeps decided geometry in the store beside its metadata file."""

    PATHS = ["0.jpg", "1.jpg"]

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name) / "hub-cache"
        self.config = {"resolution": 64, "resolution_type": "pixel", "crop": False, "dataset_type": "image"}
        StateTracker.set_args(
            SimpleNamespace(
                aspect_bucket_disable_rebuild=False,
                aspect_bucket_alignment=8,
                aspect_bucket_rounding=2,
                cache_dir=str(Path(temp.name) / "cache"),
                disable_bucket_pruning=True,
                train_batch_size=1,
                model_type="",
                resolution=64,
                resolution_type="pixel",
            )
        )
        StateTracker.set_model(None)
        StateTracker.set_epoch(1)
        self.addCleanup(StateTracker.set_epoch, 1)
        self.addCleanup(StateTracker.data_backends.pop, "hub", None)

    def backend(self, process_index=0, metadata_update_interval=3600):
        StateTracker.register_data_backend({"id": "hub"})
        StateTracker.set_data_backend_config("hub", dict(self.config))
        backend = HuggingfaceMetadataBackend(
            id="hub",
            instance_data_dir="",
            cache_file="aspect_ratio_bucket_indices",
            metadata_file="aspect_ratio_bucket_metadata",
            data_backend=_HubDataset(self.cache, [(128, 64), (64, 128)]),
            accelerator=_HubAccelerator(process_index),
            batch_size=1,
            resolution=64,
            resolution_type="pixel",
            hf_config={"width_column": "width", "height_column": "height", "caption_column": "caption"},
            metadata_update_interval=metadata_update_interval,
            cache_file_suffix="hub",
        )
        self.addCleanup(backend._close_transforms)
        return backend

    def metadata_directory(self):
        return self.cache / "huggingface_metadata" / "hub"

    def geometry_of(self, backend, path):
        return {
            key: tuple(backend.image_metadata[path][key]) if key != "aspect_ratio" else backend.image_metadata[path][key]
            for key in DECIDED
        }

    def decide_by(self, backend, decide):
        def process(path, buckets, metadata_updates=None, **kwargs):
            metadata = {"original_size": (128, 64), **decide(path)}
            buckets.setdefault(str(metadata["aspect_ratio"]), []).append(path)
            metadata_updates[path] = metadata
            return buckets

        return patch.object(backend, "_process_for_bucket", side_effect=process)

    def test_the_metadata_file_holds_what_was_measured_and_the_store_what_was_decided(self):
        main = self.backend()
        main.compute_aspect_ratio_bucket_indices()
        saved = json.loads((self.metadata_directory() / "aspect_ratio_bucket_metadata_hub.json").read_text())
        self.assertEqual(saved, {"0.jpg": {"original_size": [128, 64]}, "1.jpg": {"original_size": [64, 128]}})
        index = json.loads((self.metadata_directory() / "aspect_ratio_bucket_indices_hub.json").read_text())
        self.assertEqual(index["transform_recipe"], "huggingface")
        self.assertTrue(any((self.metadata_directory() / "transforms").rglob("*.sqlite3")))

        reader = self.backend(process_index=1)
        self.assertEqual(reader.image_metadata["0.jpg"]["target_size"], tuple(main.image_metadata["0.jpg"]["target_size"]))
        self.assertEqual(reader.image_metadata["1.jpg"]["aspect_ratio"], 0.5)

    def test_a_rank_reloading_after_an_epoch_recrop_reads_the_new_epochs_geometry(self):
        self.config.update(crop=True, crop_aspect="random")
        main = self.backend()
        with self.decide_by(main, lambda path: _geometry(2.0)):
            main.compute_aspect_ratio_bucket_indices()
        reader = self.backend(process_index=1)
        self.assertEqual(self.geometry_of(reader, "0.jpg"), _geometry(2.0))

        StateTracker.set_epoch(2)
        with self.decide_by(main, lambda path: _geometry(0.5)):
            main.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
        reader.reload_cache(set_config=False)
        self.assertEqual(self.geometry_of(reader, "0.jpg"), _geometry(0.5))

    def test_a_bucket_index_that_names_no_recipe_is_bucketed_again(self):
        self.metadata_directory().mkdir(parents=True)
        (self.metadata_directory() / "aspect_ratio_bucket_indices_hub.json").write_text(
            json.dumps({"config": {}, "aspect_ratio_bucket_indices": {"2.0": self.PATHS}})
        )
        main = self.backend()
        with self.decide_by(main, lambda path: _geometry(0.5)) as process:
            main.compute_aspect_ratio_bucket_indices()
        self.assertEqual(sorted(call.args[0] for call in process.call_args_list), self.PATHS)
        self.assertEqual(self.geometry_of(main, "1.jpg"), _geometry(0.5))

    def test_a_sample_takes_the_geometry_first_recorded_for_it(self):
        recorded = PythonTransformCache(self.metadata_directory() / "transforms", "huggingface")
        recorded.enqueue("hub", [("0.jpg", encode_transform(_geometry(2.0, (1, 1))))])
        recorded.close()
        main = self.backend()
        with self.decide_by(main, lambda path: _geometry(2.0, (5, 5))):
            main.compute_aspect_ratio_bucket_indices()
        self.assertEqual(self.geometry_of(main, "0.jpg"), _geometry(2.0, (1, 1)))
        self.assertEqual(self.geometry_of(main, "1.jpg"), _geometry(2.0, (5, 5)))

    def test_samples_saved_in_flight_are_recorded(self):
        main = self.backend(metadata_update_interval=0)
        with self.decide_by(main, lambda path: _geometry(2.0, (0, 1))):
            main.compute_aspect_ratio_bucket_indices()
        reader = self.backend(process_index=1)
        self.assertEqual(sorted(reader.image_metadata), self.PATHS)
        self.assertEqual(self.geometry_of(reader, "0.jpg"), _geometry(2.0, (0, 1)))


class CopiedDatasetGeometryTests(unittest.TestCase):
    """A copied dataset mirrors the source's geometry on every rank and has none of its own."""

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.args = SimpleNamespace(
            enable_multiprocessing=False,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            model_type="",
            skip_file_discovery="",
            vae_cache_scan_behaviour="sync",
            aspect_bucket_worker_count=1,
            aspect_bucket_disable_rebuild=False,
            cache_dir=f"{self.root}/cache",
            output_dir=f"{self.root}/output",
            disable_bucket_pruning=True,
            ignore_missing_files=True,
            delete_problematic_images=False,
            delete_unwanted_images=False,
            minimum_image_size=None,
            resolution=1.0,
            resolution_type="area",
        )
        StateTracker.set_args(self.args)

    def copy(self, process_index=0):
        backends = {}
        for dataset_id, dataset_type in (("source", "image"), ("control", "conditioning")):
            StateTracker.register_data_backend({"id": dataset_id})
            self.addCleanup(StateTracker.data_backends.pop, dataset_id, None)
            StateTracker.set_data_backend_config(
                dataset_id, {"resolution": 1.0, "resolution_type": "area", "dataset_type": dataset_type}
            )
            directory = f"{self.root}/{dataset_id}"
            metadata = DiscoveryMetadataBackend(
                id=dataset_id,
                instance_data_dir=directory,
                cache_file=f"{directory}/aspect_ratio_bucket_indices",
                metadata_file=f"{directory}/aspect_ratio_bucket_metadata",
                data_backend=LocalDataBackend(accelerator=None, id=dataset_id),
                accelerator=_Accelerator(process_index),
                batch_size=1,
                resolution=1.0,
                resolution_type="area",
            )
            self.addCleanup(metadata._close_transforms)
            entry = StateTracker.get_data_backend(dataset_id)
            entry.update(metadata_backend=metadata, instance_data_dir=directory, dataset_type=dataset_type)
            backends[dataset_id] = entry
        source = backends["source"]["metadata_backend"]
        source.aspect_ratio_bucket_indices = {"2.0": [f"{self.root}/source/a.png"]}
        source.image_metadata = {f"{self.root}/source/a.png": {"original_size": [128, 64], **_geometry(2.0)}}
        source.image_metadata_loaded = True
        DatasetDuplicator.copy_metadata(backends["source"], backends["control"])
        return backends["control"]

    def prepare(self, control, process_index):
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        accelerator = _Accelerator(process_index)
        accelerator.wait_for_everyone = lambda: None
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args, factory.accelerator = self.args, accelerator
        factory.model = SimpleNamespace(requires_conditioning_latents=lambda: False)
        backend = {"id": "control", "conditioning_type": "reference_strict", "scan_for_errors": True}
        with patch.object(control["metadata_backend"], "handle_vae_cache_inconsistencies"):
            factory._handle_error_scanning_and_metadata(backend, control, "reference_strict")

    def target_size(self, control):
        return control["metadata_backend"].get_metadata()[f"{self.root}/control/a.png"]["target_size"]

    def test_a_copied_dataset_keeps_the_source_geometry_without_reading_its_own_files(self):
        self.assertEqual(self.target_size(self.copy()), (128, 64))

    def test_the_other_ranks_keep_the_copied_geometry_after_preparation(self):
        control = self.copy(process_index=1)
        self.prepare(control, process_index=1)
        self.assertEqual(self.target_size(control), (128, 64))

    def test_an_error_scan_does_not_decide_a_copied_dataset_again(self):
        from PIL import Image

        control = self.copy()
        Path(f"{self.root}/control").mkdir(exist_ok=True)
        Image.new("RGB", (64, 32)).save(f"{self.root}/control/a.png")
        self.prepare(control, process_index=0)
        self.assertEqual(self.target_size(control), (128, 64))


class ViewerTransformRecordTests(unittest.TestCase):
    """The WebUI viewer reads and edits the geometry training recorded, under the recipe the bucket index names."""

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.images = Path(temp.name) / "images"
        self.images.mkdir()
        self.service = DatasetViewerService(project_root=Path(temp.name))
        self.config = {"id": "ds", "type": "local", "instance_data_dir": str(self.images), "maximum_aspect_ratio": 1.5}
        (self.images / "aspect_ratio_bucket_indices_ds.json").write_text(
            json.dumps({"aspect_ratio_bucket_indices": {"2.0": ["kept.png"]}, "transform_recipe": "discovery"})
        )
        self.measured = {"kept.png": {"original_size": [128, 64]}, "wide.png": {"original_size": [256, 64]}}
        (self.images / "aspect_ratio_bucket_metadata_ds.json").write_text(json.dumps(self.measured))
        store = self.store()
        store.enqueue("ds", [("kept.png", encode_transform(_geometry(2.0))), ("wide.png", encode_transform(_geometry(4.0)))])
        store.close()

    def store(self):
        store = PythonTransformCache(self.images / "transforms", "discovery")
        self.addCleanup(store.close)
        return store

    def recorded(self, path):
        from simpletuner.helpers.metadata.transform_records import decode_transform

        return decode_transform(self.store().read_batch([("ds", path)])[0])

    def metadata_file(self):
        return json.loads((self.images / "aspect_ratio_bucket_metadata_ds.json").read_text())

    def test_file_metadata_carries_the_recorded_geometry(self):
        meta = self.service.get_file_metadata(self.config, "kept.png")
        self.assertEqual(list(meta.target_size), [128, 64])
        self.assertEqual(meta.aspect_ratio, 2.0)
        self.assertEqual(list(meta.original_size), [128, 64])

    def test_a_crop_edit_replaces_the_recorded_crop_and_leaves_the_metadata_file(self):
        self.assertTrue(self.service.update_crop_coordinates(self.config, "kept.png", [4, 9]))
        self.assertEqual(self.recorded("kept.png"), _geometry(2.0, (4, 9)))
        self.assertEqual(self.metadata_file(), self.measured)
        self.assertEqual(list(self.service.get_file_metadata(self.config, "kept.png").crop_coordinates), [4, 9])

    def test_a_crop_edit_for_a_sample_with_no_recorded_geometry_fails(self):
        self.measured["new.png"] = {"original_size": [64, 64]}
        (self.images / "aspect_ratio_bucket_metadata_ds.json").write_text(json.dumps(self.measured))
        self.assertFalse(self.service.update_crop_coordinates(self.config, "new.png", [1, 1]))

    def test_a_rebuild_replaces_the_recorded_geometry(self):
        rebuilt = self.service.rebuild_file_metadata(
            {**self.config, "resolution": 32, "resolution_type": "pixel", "crop": False}, {}, "kept.png"
        )
        self.assertIsNotNone(rebuilt)
        recorded = self.recorded("kept.png")
        self.assertEqual(list(recorded["target_size"]), rebuilt["target_size"])
        self.assertNotEqual(recorded, _geometry(2.0))
        self.assertEqual(self.metadata_file(), self.measured)

    def test_filtered_samples_are_explained_by_their_recorded_aspect(self):
        report = self.service.get_filtered_files(self.config)
        self.assertEqual([(entry.path, entry.reason) for entry in report.files], [("wide.png", "aspect_out_of_range")])


class TestViewerRefusesWebshartGeometryEdits(APITestCase, unittest.TestCase):
    """A Webshart dataset's geometry follows from its settings and is kept in webshart's own cache."""

    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_crop_edits_and_rebuilds_answer_with_the_reason(self):
        routes = "simpletuner.simpletuner_sdk.server.routes.dataset_viewer"
        config = {"id": "shards", "type": "webshart", "dataset_type": "image"}
        requests = (
            ("patch", "/api/datasets/viewer/crop-coordinates", {"crop_coordinates": [0, 0]}),
            ("post", "/api/datasets/viewer/rebuild-metadata", {}),
        )
        with (
            patch(f"{routes}._get_dataset_config_by_id", return_value=config),
            patch(f"{routes}._get_viewer_service") as viewer,
        ):
            viewer.return_value.update_crop_coordinates.side_effect = AssertionError("no edit may be made")
            viewer.return_value.rebuild_file_metadata.side_effect = AssertionError("no edit may be made")
            for method, path, body in requests:
                with self.subTest(path=path):
                    response = getattr(self.client, method)(path, json={"dataset_id": "shards", "file_path": "a", **body})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("Webshart", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
