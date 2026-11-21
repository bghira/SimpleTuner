import contextlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetInfo, SplitDict, SplitInfo

from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend
from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler


class _DummyDataBackend:
    def __init__(self, item):
        self.item = item
        self.id = "test-backend"

    def _get_index_from_path(self, path):
        return 0

    def get_dataset_item(self, index):
        return self.item


class _MinimalHuggingfaceMetadataBackend(HuggingfaceMetadataBackend):
    """Collect just enough state to exercise _process_for_bucket without side effects."""

    def __init__(self, data_backend):
        self.data_backend = data_backend
        self.video_column = "video"
        self.minimum_num_frames = None
        self.maximum_num_frames = None
        self.minimum_image_size = None
        self.dataset_type = "video"
        self.bucket_report = None
        self.aspect_ratio_bucket_indices = {}
        self.quality_filter = None
        self.quality_column = "quality_assessment"
        self.bbox_column = None
        self.id = data_backend.id

    def _get_video_metadata_from_item(self, item):
        return item["metadata"]


class HuggingfaceMetadataBackendTests(unittest.TestCase):
    def test_null_caption_and_description_do_not_create_training_text(self):
        backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
        backend.dataset_type = "image"
        backend.caption_column = "caption"
        backend.fallback_caption_column = "alternate"
        backend.description_column = "description"

        self.assertIsNone(backend._extract_caption_from_item({"caption": float("nan"), "description": None}))
        self.assertEqual(
            backend._extract_caption_from_item({"caption": float("nan"), "alternate": " usable "}),
            "usable",
        )

    def test_delete_uses_the_artifact_write_path_and_protects_dataset_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            source = HuggingfaceDatasetsBackend(None, "images", "test", cache_dir=directory)
            for key in ("relative.pt", str(Path(directory) / "absolute.pt")):
                source.write(key, b"latent")
                self.assertTrue(source.exists(key))
                source.delete(key)
                self.assertFalse(source.exists(key))
                with self.assertRaises(FileNotFoundError):
                    source.delete(key)
            with self.assertRaises(NotImplementedError):
                source.delete("0.jpg")

    def test_cached_membership_stream_stays_pinned_when_the_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            source = HuggingfaceDatasetsBackend(None, "arrow", "test", cache_dir=directory)
            source.write("membership.json", '{"version":1}')
            with (
                patch.object(source, "read", side_effect=AssertionError("membership materialized")),
                source.open_read_stream("membership.json") as pinned,
            ):
                source.write("membership.json", '{"version":2}')
                self.assertEqual(pinned.read(), b'{"version":1}')
                with source.open_read_stream("membership.json") as latest:
                    self.assertEqual(latest.read(), b'{"version":2}')

    def test_caption_lookup_projects_columns_and_preserves_filtered_row_order(self):
        source = HuggingfaceDatasetsBackend(None, "test", "test")
        source.dataset = Dataset.from_list(
            [
                {"text": {"short": "first", "long": "second"}, "fallback": None, "quality": {"score": 1}, "image": b"a"},
                {"text": {"short": "rejected"}, "fallback": None, "quality": {"score": 0}, "image": b"b"},
                {"text": None, "fallback": "last", "quality": {"score": 1}, "image": b"c"},
            ]
        ).select([2, 0, 1])
        backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
        backend.data_backend = source
        backend.dataset_type = "image"
        backend.caption_column = ["text.short", "text.long"]
        backend.fallback_caption_column = "fallback"
        backend.description_column = "description"
        backend.quality_column = "quality"
        backend.quality_filter = {"score": 0.5}
        backend._caption_rows = backend._select_caption_columns(source.dataset)
        self.assertEqual(backend._caption_rows.column_names, ["text", "fallback", "quality"])
        self.assertEqual(backend.caption_cache_entry("0.jpg"), "last")
        self.assertEqual(backend.caption_cache_entry("1.jpg"), ["first", "second"])
        self.assertIsNone(backend.caption_cache_entry("2.jpg"))
        self.assertIsNone(backend.caption_cache_entry("3.jpg"))
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        runtime = DatasetRuntime(
            source.id,
            source,
            backend,
            {"caption_strategy": "huggingface", "prepend_instance_prompt": True, "instance_prompt": "style"},
        )
        with patch.object(backend, "caption_cache_entry", side_effect=AssertionError("unbatched caption read")):
            self.assertEqual(
                runtime.resolve_captions(["1.jpg", "0.jpg", "1.jpg"]),
                [["style first", "style second"], "style last", ["style first", "style second"]],
            )
            self.assertEqual(
                list(
                    runtime.iter_caption_records(
                        ["2.jpg", "0.jpg", "3.jpg"], batch_size=3, on_missing_caption=lambda ref: None
                    )
                ),
                [("style last", "0.jpg")],
            )
        with self.assertRaises(CaptionNotFoundError):
            PromptHandler.prepare_instance_prompt_from_huggingface("2.jpg", True, False, source, metadata_backend=backend)

    def test_accepted_preparation_assigns_filtered_rows_by_physical_arrow_shard(self):
        info = DatasetInfo(splits=SplitDict({"train": SplitInfo("train", num_examples=8, shard_lengths=[3, 2, 3])}))
        rows = Dataset.from_dict({"number": list(range(8))}, info=info, split="train")
        with (
            patch("datasets.load_dataset_builder", return_value=SimpleNamespace(info=info)),
            patch("datasets.load_dataset", return_value=rows),
        ):
            source = HuggingfaceDatasetsBackend(
                None, "arrow-shards", "test", filter_func=lambda item: item["number"] in (0, 2, 5, 7), num_proc=1
            )
            self.assertEqual(source.preparation_shards, (range(0, 2), range(2, 2), range(2, 4)))

        provider = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
        provider.data_backend = source
        provider.id = source.id
        provider.read_only = False
        provider.bucket_report = None
        provider.image_metadata = {}
        provider.aspect_ratio_bucket_indices = {"square": ["3.jpg", "1.jpg"], "wide": ["0.jpg", "2.jpg"]}
        with provider.open_preparation_view(worker=0, workers=3) as first:
            self.assertEqual(list(first), ["1.jpg", "0.jpg"])
        with provider.open_preparation_view(worker=1, workers=3) as empty:
            self.assertEqual(list(empty), [])
        with provider.open_preparation_view(worker=2, workers=3) as last:
            self.assertEqual(list(last), ["3.jpg", "2.jpg"])
        self.assertEqual(list(provider.iter_preparation_batches(2, 3, 4)), [("square", ("3.jpg",)), ("wide", ("2.jpg",))])

        with tempfile.TemporaryDirectory() as directory:
            from simpletuner.helpers.metadata.preparation import ShardedPreparationIndex

            provider.cache_file = Path(directory) / "buckets.json"
            source.write(provider.cache_file, {"aspect_ratio_bucket_indices": provider.aspect_ratio_bucket_indices})
            provider.read_only = True
            provider.aspect_ratio_bucket_indices = {}

            with provider.open_preparation_view(worker=2, workers=3) as cold:
                self.assertEqual(list(cold), ["3.jpg", "2.jpg"])
            index = provider._sharded_preparation_index()
            self.assertEqual(index.accepted_samples, 4)
            self.assertEqual({file.name for file in index.directory.glob("*.jsonl")}, {"0.jsonl", "2.jsonl"})

            with patch.object(ShardedPreparationIndex, "_publish", side_effect=AssertionError("warm index rebuild")):
                with provider.open_preparation_view(worker=0, workers=3) as warm:
                    self.assertEqual(list(warm), ["1.jpg", "0.jpg"])
                self.assertEqual(list(provider.iter_preparation_batches(1, 3, 2)), [])

            source.write(provider.cache_file, {"aspect_ratio_bucket_indices": {"square": ["3.jpg"]}})
            self.assertEqual(list(provider.iter_preparation_batches(2, 3, 2)), [("square", ("3.jpg",))])
            self.assertNotEqual(provider._sharded_preparation_index().directory, index.directory)

            source.write(
                provider.cache_file,
                {"source_revision": source.dataset._fingerprint, "aspect_ratio_bucket_indices": {"square": ["3.jpg"]}},
            )
            from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

            runtime = DatasetRuntime(source.id, source, provider, {})
            revision = runtime.occurrence_revision(None)
            source.dataset = Dataset.from_dict({"number": [9, 10, 11, 12]})
            self.assertNotEqual(runtime.occurrence_revision(None), revision)
            with self.assertRaisesRegex(ValueError, "different source revision"):
                list(provider.iter_preparation_batches(2, 3, 2))
            provider.metadata_semaphor = threading.Semaphore()
            provider.image_metadata = {"3.jpg": {"original_size": [8, 8]}}
            provider.reload_cache()
            provider.load_image_metadata()
            self.assertEqual(provider.aspect_ratio_bucket_indices, {})
            self.assertEqual(provider.image_metadata, {})

    def test_peer_keeps_copied_geometry_when_target_membership_has_an_old_revision(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.metadata.utils.duplicator import DatasetDuplicator

        accelerator = SimpleNamespace(
            num_processes=2,
            process_index=1,
            is_main_process=False,
            main_process_first=contextlib.nullcontext,
            wait_for_everyone=lambda: None,
        )
        args = SimpleNamespace(skip_file_discovery="", only_instance_prompt=False)
        with tempfile.TemporaryDirectory() as directory:

            def runtime(name, role):
                data = HuggingfaceDatasetsBackend(None, name, "fixture", cache_dir=directory)
                data.dataset = Dataset.from_dict({"text": ["current"]})
                config = {"dataset_type": role, "caption_strategy": "instanceprompt"}
                metadata = HuggingfaceMetadataBackend(
                    id=name,
                    instance_data_dir=directory,
                    cache_file=str(Path(directory) / (name + "-members")),
                    metadata_file=str(Path(directory) / (name + "-geometry")),
                    data_backend=data,
                    accelerator=accelerator,
                    batch_size=1,
                    resolution=64,
                    resolution_type="pixel",
                    hf_config={},
                    dataset_config=config,
                    runtime_args=args,
                )
                self.addCleanup(metadata.close)
                return DatasetRuntime(name, data, metadata, config)

            source = runtime("source", "image")
            target = runtime("target", "conditioning")
            target.data_backend.write(target.metadata_backend.cache_file, {"source_revision": "old"})
            row = {"target_size": [128, 64], "crop_coordinates": [8, 12]}
            source.metadata_backend.image_metadata = {"0.jpg": row}
            source.metadata_backend.image_metadata_loaded = True
            source.metadata_backend.aspect_ratio_bucket_indices = {"2.0": ["0.jpg"]}
            DatasetDuplicator.copy_metadata(source, target)
            self.assertEqual(target.metadata_backend.get_metadata()["0.jpg"], row)

            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.accelerator, factory.args = accelerator, args
            factory._handle_error_scanning_and_metadata(
                {"metadata_clone_source_id": source.id},
                {"id": target.id, "metadata_backend": target.metadata_backend},
                "reference_strict",
            )
            self.assertEqual(target.metadata_backend.get_metadata().get("0.jpg"), row)

    def test_preparation_scan_uses_the_same_sample_cap_as_discovery(self):
        rows = Dataset.from_dict({"width": [512] * 18, "height": [256] * 18})
        prepared = SimpleNamespace(
            aspect_ratio=2.0, target_size=(512, 256), crop_coordinates=(0, 0), intermediary_size=(512, 256)
        )
        accelerator = SimpleNamespace(main_process_first=contextlib.nullcontext, wait_for_everyone=lambda: None)
        source = HuggingfaceDatasetsBackend(accelerator, "test-cap", "test")
        source.dataset = rows
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("simpletuner.helpers.metadata.backends.huggingface.TrainingSample") as sample,
        ):
            root = Path(directory)
            sample.return_value.prepare.return_value = prepared
            provider = HuggingfaceMetadataBackend(
                id=source.id,
                instance_data_dir=directory,
                cache_file=str(root / "buckets"),
                metadata_file=str(root / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=512,
                resolution_type="pixel",
                hf_config={"width_column": "width", "height_column": "height"},
            )
            self.assertEqual(provider._limited_dataset_indices(), range(18))
            refs = [f"{index}.jpg" for index in range(18)]
            provider.max_num_samples = 3
            expected = provider._apply_max_num_samples_limit(refs)
            self.assertEqual(provider._discover_new_files(ignore_existing_cache=True), expected)

            provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(provider.read_page(limit=10), (expected, 3))
            self.assertEqual(set(provider.image_metadata), set(expected))
            provider.remove_images([expected[-1]])
            provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            provider.reload_cache()
            self.assertEqual(provider.read_page(limit=10), (expected[:-1], 2))
            provider.enqueue_prepared_metadata({expected[0]: {"target_size": [64, 32]}})
            provider.enqueue_prepared_metadata({expected[0]: {"target_size": [128, 64]}})
            provider.close()
            provider.load_image_metadata()
            self.assertEqual(provider.get_metadata_by_filepath(expected[0]), {"target_size": [128, 64]})

    def test_missing_caption_stays_out_of_warm_membership_and_epoch_recrop(self):
        accelerator = SimpleNamespace(main_process_first=contextlib.nullcontext, wait_for_everyone=lambda: None)
        source = HuggingfaceDatasetsBackend(accelerator, "captioned", "test")
        source.dataset = Dataset.from_dict(
            {"width": [512] * 3, "height": [256] * 3, "caption": ["first", "", ""], "fallback": ["", "", "third"]}
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("simpletuner.helpers.metadata.backends.huggingface.TrainingSample") as sample,
        ):
            sample.return_value.prepare.return_value = SimpleNamespace(
                aspect_ratio=2.0, target_size=(512, 256), crop_coordinates=(0, 0), intermediary_size=(512, 256)
            )
            settings = dict(
                id=source.id,
                instance_data_dir=directory,
                cache_file=str(Path(directory) / "buckets"),
                metadata_file=str(Path(directory) / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=512,
                resolution_type="pixel",
                hf_config={"width_column": "width", "height_column": "height", "fallback_caption_column": "fallback"},
                dataset_config={"caption_strategy": "huggingface"},
                runtime_args={"only_instance_prompt": True},
            )
            with_instance_prompt = HuggingfaceMetadataBackend(**settings)
            with_instance_prompt.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(with_instance_prompt.read_page(limit=10)[0], ["0.jpg", "1.jpg", "2.jpg"])
            with_instance_prompt.close()
            settings["runtime_args"]["only_instance_prompt"] = False
            provider = HuggingfaceMetadataBackend(**settings)
            self.addCleanup(provider.close)
            self.assertEqual(provider.read_page(limit=10)[0], ["0.jpg", "2.jpg"])
            provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            provider.reload_cache()
            self.assertEqual(provider.read_page(limit=10)[0], ["0.jpg", "2.jpg"])
            self.assertEqual(provider.read_caption_batch(["0.jpg", "2.jpg"]), ["first", "third"])
            # A fresh scan also rejects the empty row before it enters any cache.
            settings["cache_file"] += "-fresh"
            settings["metadata_file"] += "-fresh"
            fresh = HuggingfaceMetadataBackend(**settings)
            self.addCleanup(fresh.close)
            fresh.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(fresh.read_page(limit=10)[0], ["0.jpg", "2.jpg"])

    def test_arrow_transform_cache_reopens_and_reprocesses_by_shard(self):
        from simpletuner.helpers.metadata.backends.huggingface_transforms import HuggingFaceTransformStore

        accelerator = SimpleNamespace(main_process_first=contextlib.nullcontext, wait_for_everyone=lambda: None)
        source = HuggingfaceDatasetsBackend(accelerator, "arrow-geometry", "test")
        source.dataset = Dataset.from_dict({"width": [512] * 4, "height": [256] * 4})
        source._source_shard_lengths = (2, 0, 2)
        prepared = SimpleNamespace(
            aspect_ratio=2.0, target_size=(512, 256), crop_coordinates=(0, 0), intermediary_size=(512, 256)
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("simpletuner.helpers.metadata.backends.huggingface.TrainingSample") as sample,
        ):
            root = Path(directory)
            sample.return_value.prepare.return_value = prepared
            options = {
                "id": source.id,
                "instance_data_dir": directory,
                "cache_file": str(root / "buckets"),
                "metadata_file": str(root / "geometry"),
                "data_backend": source,
                "accelerator": accelerator,
                "batch_size": 1,
                "resolution": 512,
                "resolution_type": "pixel",
                "hf_config": {"width_column": "width", "height_column": "height"},
            }
            with contextlib.closing(HuggingfaceMetadataBackend(**options)) as provider:
                provider.compute_aspect_ratio_bucket_indices()
                self.assertIsInstance(provider.image_metadata, HuggingFaceTransformStore)
                original = provider.image_metadata.directory
                self.assertEqual({path.name for path in original.glob("*.sqlite3")}, {"0.sqlite3", "2.sqlite3"})
                self.assertFalse(list(root.glob("geometry*.json")))
                refs = ["3.jpg", "0.jpg", "3.jpg", "8.jpg"]
                first = provider.get_metadata_batch(refs)
                self.assertEqual(first[0], first[2])
                self.assertIsNone(first[-1])
                provider.enqueue_prepared_metadata({"0.jpg": {**first[1], "target_size": [128, 64]}})

            with (
                contextlib.closing(HuggingfaceMetadataBackend(**options)) as reopened,
                contextlib.closing(HuggingfaceMetadataBackend(**options)) as peer,
            ):
                self.assertEqual(reopened.image_metadata.directory, original)
                with peer.open_preparation_view(worker=0, workers=3) as view:
                    self.assertEqual(list(view), ["0.jpg", "1.jpg"])
                    self.assertEqual(view.metadata.get_metadata_by_filepath("0.jpg")["target_size"], [128, 64])
                    reopened.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
                    peer.reload_cache(set_config=False)
                    self.assertEqual(peer.get_metadata_batch(refs), first)
                    with peer.open_preparation_view(worker=0, workers=3) as refreshed:
                        self.assertEqual(list(refreshed), ["0.jpg", "1.jpg"])
                        self.assertEqual(refreshed.metadata.get_metadata_by_filepath("0.jpg"), first[1])
                    self.assertEqual(view.metadata.get_metadata_by_filepath("0.jpg")["target_size"], [128, 64])
                self.assertNotEqual(reopened.image_metadata.directory, original)
                self.assertEqual(reopened.get_metadata_batch(refs), first)
                current = reopened.image_metadata.directory
                with (
                    patch.object(reopened, "_process_for_bucket", side_effect=KeyboardInterrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    reopened.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)

            with contextlib.closing(HuggingfaceMetadataBackend(**options)) as after_cancel:
                self.assertEqual(after_cancel.image_metadata.directory, current)
                self.assertEqual(after_cancel.get_metadata_batch(refs), first)

    def test_metadata_reload_keeps_existing_entries_visible_until_read_completes(self):
        read_started = threading.Event()
        allow_read = threading.Event()

        class BlockingDataBackend:
            def exists(self, _path):
                return True

            def get_abs_path(self, path):
                return path

            def read(self, _path):
                read_started.set()
                if not allow_read.wait(timeout=5):
                    raise TimeoutError("metadata read was not released")
                return json.dumps({"1.wav": {"prompt": "new", "lyrics": "lyrics"}})

        backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
        backend.id = "audio-test"
        backend.metadata_file = Path("metadata.json")
        backend.data_backend = BlockingDataBackend()
        backend.metadata_semaphor = threading.Semaphore()
        backend.image_metadata = {"1.wav": {"prompt": "old", "lyrics": "lyrics"}}
        backend.image_metadata_loaded = True

        reload_thread = threading.Thread(target=backend.load_image_metadata)
        reload_thread.start()
        self.assertTrue(read_started.wait(timeout=5))
        self.assertEqual(backend.get_metadata_by_filepath("1.wav")["prompt"], "old")

        allow_read.set()
        reload_thread.join(timeout=5)
        self.assertFalse(reload_thread.is_alive())
        self.assertEqual(backend.get_metadata_by_filepath("1.wav")["prompt"], "new")

    def test_audio_caption_falls_back_to_text_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.flac"
            audio_path.touch()
            audio_path.with_suffix(".txt").write_text("riff-heavy caption", encoding="utf-8")
            media = SimpleNamespace(_hf_encoded={"path": str(audio_path)})

            backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
            backend.dataset_type = "audio"
            backend.caption_column = "caption"
            backend.fallback_caption_column = None
            backend.audio_caption_fields = ["prompt", "tags"]
            backend.description_column = "description"

            self.assertEqual(backend._extract_caption_from_item({"audio": media}), "riff-heavy caption")

    @patch("simpletuner.helpers.metadata.backends.huggingface.TrainingSample")
    def test_video_without_maximum_num_frames_is_not_flagged_as_too_many(self, mock_training_sample):
        prepared = SimpleNamespace(
            aspect_ratio=1.0,
            intermediary_size=(1, 1),
            crop_coordinates=(0, 0),
            target_size=(1, 1),
        )
        mock_training_sample.return_value.prepare.return_value = prepared

        item = {"video": object(), "metadata": {"original_size": (640, 480), "num_frames": 5}}
        backend = _MinimalHuggingfaceMetadataBackend(_DummyDataBackend(item))

        statistics = {"skipped": {"too_many_frames": 0}}
        aspect_ratio_buckets = backend._process_for_bucket("0.mp4", {}, statistics=statistics)

        self.assertIn("1.0", aspect_ratio_buckets)
        self.assertEqual(aspect_ratio_buckets["1.0"], ["0.mp4"])
        self.assertEqual(statistics["skipped"]["too_many_frames"], 0)


if __name__ == "__main__":
    unittest.main()
