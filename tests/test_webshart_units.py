"""A Webshart dataset read a shard at a time, on real Parquet and tar sources."""

import json
import tarfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from simpletuner.helpers.data_backend.config.validators import validate_webshart_shuffle_window
from simpletuner.helpers.data_backend.cursor import WindowedCursor
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
from simpletuner.helpers.metadata.decisions import UnitDecisions
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker

SHARDS = 3
# (width, height, caption) of what shard `s` holds, besides `s` more square images.
USABLE = [(64, 64, "square"), (96, 64, "wide"), (64, 96, "tall"), (128, 64, "wider")]
TOO_SMALL = (16, 16, "small")
UNCAPTIONED = (64, 64, None)


def encoded(size, image_format):
    data = BytesIO()
    Image.new("RGB", size, "red").save(data, format=image_format)
    return data.getvalue()


def shard_images(shard):
    return USABLE + [(64, 64, f"extra {index}") for index in range(shard)] + [TOO_SMALL, UNCAPTIONED]


def write_parquet_source(source: Path, shards: int = SHARDS) -> None:
    for shard in range(shards):
        rows = [dict(image=None, captions="dead", width=None, height=None, url_status="http_error", stored_bytes=None)]
        for width, height, caption in shard_images(shard):
            data = encoded((width, height), "PNG")
            rows.append(
                dict(
                    image={"bytes": data, "path": None},
                    captions=caption,
                    width=width,
                    height=height,
                    url_status="ok",
                    stored_bytes=len(data),
                )
            )
        pq.write_table(pa.Table.from_pylist(rows), source / f"part-{shard:03d}.parquet", row_group_size=4)


def write_tar_source(source: Path) -> None:
    for shard in range(SHARDS):
        archive_path = source / f"shard-{shard}.tar"
        sizes = {}
        with tarfile.open(archive_path, "w") as archive:
            for index, (width, height, caption) in enumerate(shard_images(shard)):
                members = {f"s{shard}_{index}.jpg": encoded((width, height), "JPEG")}
                sizes[f"s{shard}_{index}.jpg"] = (width, height)
                if caption is not None:
                    members[f"s{shard}_{index}.txt"] = caption.encode()
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, BytesIO(payload))
        files = {}
        with tarfile.open(archive_path) as archive:
            for info in archive.getmembers():
                files[info.name] = {"offset": info.offset_data, "length": info.size}
                if info.name in sizes:
                    width, height = sizes[info.name]
                    files[info.name].update(width=width, height=height, aspect=width / height)
        (source / f"shard-{shard}.json").write_text(
            json.dumps({"path": archive_path.name, "filesize": archive_path.stat().st_size, "files": files})
        )


def as_json(value):
    return json.loads(json.dumps(value))


def geometry(record):
    """A record without its crop position, which is drawn at random each time a sample is prepared."""
    return as_json({key: value for key, value in record.items() if key != "crop_coordinates"})


class WebshartSource:
    """Builds the same dataset with and without a window; subclasses choose the format."""

    format: str
    config = dict(caption_strategy="webshart", resolution=64, resolution_type="pixel", crop=False)

    def write_source(self, source: Path) -> None:
        raise NotImplementedError

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.write_source(self.source)
        self.dataset_config = dict(self.config)
        self.args = SimpleNamespace(
            disable_bucket_pruning=False,
            allow_dataset_oversubscription=False,
            ignore_missing_files=False,
            seed=42,
            aspect_bucket_disable_rebuild=False,
        )
        for patcher in (
            patch.object(StateTracker, "get_data_backend_config", side_effect=lambda *_, **__: self.dataset_config),
            patch.object(StateTracker, "get_args", return_value=self.args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "set_image_files"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def data_backend(self, name, shuffle_window=None, **options):
        return WebshartDataBackend(
            None,
            "ws",
            str(self.source),
            format=self.format,
            cache_dir=str(self.root / name),
            shuffle_window=shuffle_window,
            **options,
        )

    def metadata_backend(self, data_backend, name, **limits):
        return WebshartMetadataBackend(
            id=data_backend.id,
            instance_data_dir=data_backend.cache_dir,
            cache_file=str(self.root / f"{name}-buckets"),
            metadata_file=str(self.root / f"{name}-metadata"),
            data_backend=data_backend,
            accelerator=None,
            batch_size=2,
            resolution=64,
            resolution_type="pixel",
            minimum_image_size=32,
            **limits,
        )

    def windowed(self, name="windowed", window=2, **limits):
        return self.metadata_backend(self.data_backend(name, shuffle_window=window), name, **limits)

    def whole(self, name="whole", **limits):
        """The dataset as it is indexed without a window: every shard, up front."""
        backend = self.metadata_backend(self.data_backend(name), name, **limits)
        backend.compute_aspect_ratio_bucket_indices()
        return backend

    def cache_files(self):
        return {path: path.read_bytes() for path in self.root.rglob("*.json") if self.source not in path.parents}

    def opened(self, backend, unit, epoch=1):
        """A shard's pools and the record of each of its samples, with the shard closed again."""
        pools = backend.open_unit(unit, epoch=epoch)
        records = {ref: backend.get_metadata_by_filepath(ref) for refs in pools.values() for ref in refs}
        backend.close_unit(unit)
        return pools, records


class UnitTests(WebshartSource):
    def test_a_unit_is_a_shard(self):
        self.assertEqual(self.windowed().unit_count(), SHARDS)

    def test_opening_every_unit_finds_what_indexing_the_whole_dataset_finds(self):
        whole = self.whole()
        expected = {ref for refs in whole.aspect_ratio_bucket_indices.values() for ref in refs}
        self.assertEqual(len(expected), sum(len(USABLE) + shard for shard in range(SHARDS)))

        windowed = self.windowed()
        found = set()
        for unit in range(SHARDS):
            refs = {ref for pool in windowed.open_unit(unit, epoch=1).values() for ref in pool}
            self.assertEqual({windowed.data_backend.parse_sample_id(ref).shard_idx for ref in refs}, {unit})
            self.assertEqual(len(refs), len(USABLE) + unit)
            found |= refs
        self.assertEqual(found, expected)
        for ref in expected:
            self.assertEqual(geometry(windowed.get_metadata_by_filepath(ref)), geometry(whole.get_metadata_by_filepath(ref)))

    def test_a_pool_holds_the_samples_of_one_target_size(self):
        windowed = self.windowed()
        pools = windowed.open_unit(2, epoch=1)
        self.assertGreater(len(pools), 1)
        for pool, refs in pools.items():
            for ref in refs:
                width, height = windowed.get_metadata_by_filepath(ref)["target_size"]
                self.assertEqual(pool, f"{width}x{height}")

    def test_a_unit_is_described_only_while_it_is_open(self):
        windowed = self.windowed()
        first = [ref for pool in windowed.open_unit(0, epoch=1).values() for ref in pool]
        second = [ref for pool in windowed.open_unit(1, epoch=1).values() for ref in pool]

        windowed.close_unit(0)

        self.assertEqual([windowed.get_metadata_by_filepath(ref) for ref in first], [None] * len(first))
        self.assertNotIn(None, [windowed.get_metadata_by_filepath(ref) for ref in second])
        self.assertEqual(set(windowed.image_metadata), set(second))
        windowed.close_unit(1)
        self.assertEqual(windowed.image_metadata, {})

    def test_samples_outside_the_aspect_ratio_limits_are_left_out(self):
        limits = dict(minimum_aspect_ratio=0.9, maximum_aspect_ratio=1.6)
        whole = self.whole(**limits)
        expected = {ref for refs in whole.aspect_ratio_bucket_indices.values() for ref in refs}
        unlimited = {ref for refs in self.whole("unlimited").aspect_ratio_bucket_indices.values() for ref in refs}
        self.assertLess(len(expected), len(unlimited))

        windowed = self.windowed(**limits)
        found = {ref for unit in range(SHARDS) for pool in windowed.open_unit(unit, epoch=1).values() for ref in pool}

        self.assertEqual(found, expected)
        self.assertEqual(set(windowed.image_metadata), expected)

    def test_captions_and_pixels_of_an_open_unit_are_read_through_the_dataset(self):
        windowed = self.windowed()
        refs = sorted(ref for pool in windowed.open_unit(1, epoch=1).values() for ref in pool)
        runtime = DatasetRuntime(
            "ws",
            windowed.data_backend,
            windowed,
            {
                **self.config,
                "dataset_type": "image",
                "use_captions": True,
                "prepend_instance_prompt": False,
                "instance_prompt": None,
                "disable_multiline_split": False,
            },
        )

        records = runtime.read_samples(refs)

        self.assertEqual([record["image_path"] for record in records], refs)
        self.assertEqual(
            sorted(runtime.resolve_caption(ref) for ref in refs), sorted(caption for _, _, caption in shard_images(1)[:-2])
        )
        _, images = windowed.data_backend.read_image_batch(refs)
        self.assertEqual(
            sorted(image.size for image in images), sorted((width, height) for width, height, _ in shard_images(1)[:-2])
        )

    def test_nothing_describing_the_whole_dataset_is_read_or_written(self):
        # Files left by a run without a window, at the very paths the windowed dataset is given.
        self.whole("shared")
        before = self.cache_files()
        self.assertTrue(before)

        windowed = self.metadata_backend(self.data_backend("shared", shuffle_window=2), "shared")
        self.assertEqual(windowed.aspect_ratio_bucket_indices, {})
        self.assertEqual(windowed.image_metadata, {})

        opened = {ref for pool in windowed.open_unit(0, epoch=1).values() for ref in pool}
        windowed.save_cache()
        windowed.save_image_metadata()
        windowed.set_metadata_by_filepath(next(iter(opened)), windowed.image_metadata[next(iter(opened))])
        windowed.reload_cache()
        windowed.load_image_metadata()

        self.assertEqual(self.cache_files(), before)
        self.assertEqual(set(windowed.image_metadata), opened)
        self.assertEqual(windowed.aspect_ratio_bucket_indices, {})

    def test_indexing_the_whole_dataset_is_refused(self):
        windowed = self.windowed()
        with self.assertRaisesRegex(RuntimeError, "shuffle_window.*a shard at a time"):
            windowed.compute_aspect_ratio_bucket_indices()
        with self.assertRaisesRegex(RuntimeError, "shuffle_window.*a shard at a time"):
            windowed.refresh_buckets()

    def test_two_ranks_train_on_every_usable_sample_and_leave_nothing_open(self):
        expected = {ref for refs in self.whole().aspect_ratio_bucket_indices.values() for ref in refs}
        data_backend = self.data_backend("windowed", shuffle_window=2)
        trained = []
        for rank in range(2):
            metadata = self.metadata_backend(data_backend, f"rank{rank}")
            cursor = WindowedCursor(
                "ws",
                unit_count=metadata.unit_count(),
                open_unit=metadata.open_unit,
                close_unit=metadata.close_unit,
                window=2,
                batch_size=2,
                repeats=0,
                seed=7,
                dp_size=2,
                dp_rank=rank,
            )
            seen = set()
            while True:
                try:
                    pool, refs = cursor.next_batch()
                except MultiDatasetExhausted:
                    break
                sizes = {tuple(metadata.get_metadata_by_filepath(ref)["target_size"]) for ref in refs}
                self.assertEqual({f"{width}x{height}" for width, height in sizes}, {pool})
                seen.update(refs)
            trained.append(seen)
            self.assertEqual(metadata.image_metadata, {})

        self.assertEqual(trained[0] | trained[1], expected)
        self.assertEqual(trained[0] & trained[1], set())

    def test_a_shard_opened_again_gives_its_samples_the_size_and_crop_they_were_given(self):
        self.dataset_config = {**self.config, "crop": True, "crop_aspect": "square", "crop_style": "random"}
        windowed = self.windowed()
        first = self.opened(windowed, 2)
        self.assertEqual({record["target_size"] for record in first[1].values()}, {(64, 64)})
        self.assertGreater(len({record["crop_coordinates"] for record in first[1].values()}), 1)

        self.assertEqual(self.opened(windowed, 2), first)
        self.assertEqual(self.opened(windowed, 2, epoch=2), first)
        # Another process, or a resumed run.
        self.assertEqual(self.opened(self.windowed(), 2, epoch=5), first)

    def test_a_sample_left_out_when_its_shard_was_decided_is_not_decided_again(self):
        first = self.opened(self.windowed(), 1)
        self.assertEqual(len(first[1]), len(USABLE) + 1)

        again = self.windowed()
        with patch.object(again, "_prepare_bucket_entry", side_effect=AssertionError("decided again")):
            self.assertEqual(self.opened(again, 1), first)

    def test_a_dataset_that_recrops_each_epoch_keeps_a_shard_for_the_epoch_it_was_decided_in(self):
        self.dataset_config = {
            **self.config,
            "crop": True,
            "crop_aspect": "random",
            "crop_aspect_buckets": [0.5, 1.0, 2.0],
            "crop_style": "random",
        }
        windowed = self.windowed()
        with patch.object(windowed, "_decide_samples", wraps=windowed._decide_samples) as decide:
            first = self.opened(windowed, 2, epoch=1)
            self.assertEqual(self.opened(windowed, 2, epoch=1), first)
            self.assertEqual(decide.call_count, 1)

            second = self.opened(windowed, 2, epoch=2)
            self.assertEqual(decide.call_count, 2)
            self.assertEqual(windowed._unit_decisions().read(2)[0], 2)
            self.assertEqual(self.opened(windowed, 2, epoch=2), second)
            self.assertEqual(decide.call_count, 2)

            # Decided again for an epoch behind the recorded one too: a new run starts over at epoch 1.
            self.opened(windowed, 2, epoch=1)
            self.assertEqual(decide.call_count, 3)

            self.args.aspect_bucket_disable_rebuild = True
            self.opened(windowed, 2, epoch=7)
            self.assertEqual(decide.call_count, 3)
            self.assertEqual(windowed._unit_decisions().read(2)[0], 1)

    def test_decisions_are_kept_apart_by_the_settings_they_follow_from(self):
        first = self.windowed()
        self.opened(first, 0)
        directory = first._unit_decisions().directory
        self.assertEqual(directory.parent, Path(first.data_backend.cache_dir) / "decisions")
        self.assertEqual(
            sorted(path.relative_to(directory) for path in directory.rglob("*")), [Path("0"), Path("0/0.sqlite")]
        )

        def directory_with(config=None, args=None, **limits):
            self.dataset_config = {**self.config, **(config or {})}
            with patch.object(StateTracker, "get_args", return_value=SimpleNamespace(**vars(self.args), **(args or {}))):
                return self.windowed(**limits)._unit_decisions().directory

        self.assertEqual(directory_with(), directory)
        for key in ("train_batch_size", "repeats", "start_epoch", "start_step", "end_epoch", "end_step"):
            with self.subTest(key=key):
                self.assertEqual(directory_with({key: 3}), directory)
        for change in (
            dict(config={"resolution": 32}),
            dict(config={"crop": True}),
            dict(config={"caption_strategy": "instanceprompt"}),
            dict(args={"aspect_bucket_alignment": 16}),
            dict(args={"aspect_bucket_rounding": 3}),
            dict(minimum_aspect_ratio=0.9),
            dict(maximum_aspect_ratio=1.6),
        ):
            with self.subTest(change=change):
                self.assertNotEqual(directory_with(**change), directory)
        with patch.object(StateTracker, "get_model", return_value=SimpleNamespace(MAXIMUM_CANVAS_SIZE=4096)):
            self.assertNotEqual(directory_with(), directory)

        first.record_samples_per_unit(4.5)
        self.assertEqual(self.windowed().recorded_samples_per_unit(), 4.5)
        self.assertIsNone(self.windowed(minimum_aspect_ratio=0.9).recorded_samples_per_unit())

        other_settings = self.metadata_backend(first.data_backend, "other", minimum_aspect_ratio=0.9)
        with patch.object(other_settings, "_decide_samples", wraps=other_settings._decide_samples) as decide:
            self.opened(other_settings, 0)
        self.assertEqual(decide.call_count, 1)

    def test_the_size_of_a_shard_is_the_size_of_its_file(self):
        backend = self.data_backend("sizes")
        pattern = "*.parquet" if self.format == "parquet" else "*.tar"
        files = sorted(self.source.glob(pattern))
        self.assertEqual(
            [backend.dataset.get_shard_info(unit)["size"] for unit in range(SHARDS)], [path.stat().st_size for path in files]
        )

    def test_the_window_survives_a_trip_to_a_worker(self):
        backend = self.data_backend("windowed", shuffle_window=5)
        restored = WebshartDataBackend.from_instance_representation(backend.get_instance_representation())
        self.assertEqual(restored.shuffle_window, 5)
        plain = self.data_backend("plain")
        self.assertIsNone(
            WebshartDataBackend.from_instance_representation(plain.get_instance_representation()).shuffle_window
        )


class TestParquetUnits(UnitTests, unittest.TestCase):
    format = "parquet"

    def write_source(self, source):
        write_parquet_source(source)


class TestTarUnits(UnitTests, unittest.TestCase):
    format = "tar"

    def write_source(self, source):
        write_tar_source(source)


class TestWholeDatasetIndexing(WebshartSource, unittest.TestCase):
    """The shard loop that open_unit shares with indexing a dataset up front."""

    format = "parquet"

    def write_source(self, source):
        write_parquet_source(source)

    def test_the_sample_limit_stops_the_scan_at_the_shard_that_reaches_it(self):
        backend = self.metadata_backend(self.data_backend("limited"), "limited")
        backend.max_num_samples = 5
        backend.batch_size = 1  # a bucket of one sample is otherwise pruned once the scan ends
        listed = []
        entries_for_shard = backend._entries_for_shard

        def listing(shard_idx):
            listed.append(shard_idx)
            return entries_for_shard(shard_idx)

        with patch.object(backend, "_entries_for_shard", side_effect=listing):
            backend.compute_aspect_ratio_bucket_indices()

        refs = [ref for refs in backend.aspect_ratio_bucket_indices.values() for ref in refs]
        self.assertEqual(len(refs), 5)
        self.assertEqual(sorted(set(backend.image_metadata)), sorted(refs))
        # Shard 0 holds four usable samples, so the fifth comes from shard 1 and shard 2 is never listed.
        self.assertEqual(listed, [0, 1])
        self.assertEqual(backend.filtering_statistics["total_processed"], 5)

    def test_what_is_skipped_is_counted_by_reason(self):
        skipped = self.whole().filtering_statistics["skipped"]
        self.assertEqual(
            skipped, {"already_exists": 0, "metadata_missing": 0, "caption_missing": SHARDS, "too_small": SHARDS, "error": 0}
        )

    def test_a_second_scan_prepares_only_what_the_first_one_left_out(self):
        backend = self.whole()
        found = as_json(backend.aspect_ratio_bucket_indices)
        accepted = {ref for refs in found.values() for ref in refs}
        with patch.object(backend, "_prepare_bucket_entry", wraps=backend._prepare_bucket_entry) as prepare:
            backend.compute_aspect_ratio_bucket_indices()
        prepared = [call.args[2] for call in prepare.call_args_list]
        self.assertEqual(len(prepared), 2 * SHARDS)
        self.assertEqual(accepted.intersection(prepared), set())
        self.assertEqual(as_json(backend.aspect_ratio_bucket_indices), found)
        self.assertEqual(backend.filtering_statistics["skipped"]["already_exists"], len(accepted))

    def test_progress_is_reported_for_every_shard(self):
        backend = self.metadata_backend(self.data_backend("progress"), "progress")
        reports = []
        backend.compute_aspect_ratio_bucket_indices(progress_callback=lambda done, total: reports.append((done, total)))
        self.assertEqual(reports, [(1, SHARDS), (2, SHARDS), (3, SHARDS)])

    def test_progress_is_saved_while_the_scan_runs(self):
        backend = self.metadata_backend(self.data_backend("saving"), "saving")
        backend.metadata_update_interval = 0
        saves = []
        save_cache = backend.save_cache

        def saving(enforce_constraints=False):
            saves.append(sum(len(refs) for refs in backend.aspect_ratio_bucket_indices.values()))
            return save_cache(enforce_constraints=enforce_constraints)

        with patch.object(backend, "save_cache", side_effect=saving):
            backend.compute_aspect_ratio_bucket_indices()

        self.assertGreater(len(saves), 2)
        self.assertEqual(saves, sorted(saves))
        self.assertLess(saves[0], saves[-1])
        self.assertEqual(saves[-1], sum(len(USABLE) + shard for shard in range(SHARDS)))


class TestUnitDecisions(unittest.TestCase):
    DECISIONS = {
        3: dict(aspect_ratio=1.5, target_size=(96, 64), intermediary_size=(96, 96), crop_coordinates=(7, 0)),
        0: dict(aspect_ratio=1.0, target_size=(64, 64), intermediary_size=None, crop_coordinates=(0, 0)),
    }

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "decisions"
        self.store = UnitDecisions(self.directory)

    def files(self):
        return sorted(str(path.relative_to(self.directory)) for path in self.directory.rglob("*") if path.is_file())

    def test_a_unit_that_was_not_decided_has_nothing_recorded(self):
        self.assertIsNone(self.store.read(12))
        self.assertFalse(self.directory.exists())

    def test_what_is_written_for_a_unit_is_read_back(self):
        self.store.write(12, 4, self.DECISIONS)
        self.assertEqual(UnitDecisions(self.directory).read(12), (4, self.DECISIONS))
        self.assertIsNone(self.store.read(13))

    def test_a_unit_that_kept_no_sample_is_recorded_as_decided(self):
        self.store.write(12, 1, {})
        self.assertEqual(self.store.read(12), (1, {}))

    def test_writing_a_unit_again_replaces_its_rows_and_its_epoch(self):
        self.store.write(12, 1, self.DECISIONS)
        replacement = {5: dict(aspect_ratio=0.5, target_size=(32, 64), intermediary_size=(32, 64), crop_coordinates=(0, 0))}
        self.store.write(12, 2, replacement)
        self.assertEqual(self.store.read(12), (2, replacement))
        self.assertEqual(self.files(), ["0/12.sqlite"])

    def test_files_are_kept_a_thousand_units_to_a_directory(self):
        for unit in (999, 1000, 471954):
            self.store.write(unit, 1, {})
        self.assertEqual(self.files(), ["0/999.sqlite", "1/1000.sqlite", "471/471954.sqlite"])

    def test_the_measure_of_samples_per_unit_is_kept_beside_the_units(self):
        self.assertIsNone(self.store.read_samples_per_unit())
        self.store.write_samples_per_unit(812.25)
        self.assertEqual(UnitDecisions(self.directory).read_samples_per_unit(), 812.25)
        self.store.write_samples_per_unit(3.0)
        self.assertEqual(self.store.read_samples_per_unit(), 3.0)
        self.assertEqual(self.files(), ["samples_per_unit.json"])

    def test_a_write_that_fails_leaves_what_was_recorded(self):
        self.store.write(12, 1, self.DECISIONS)
        unstorable = {1: dict(self.DECISIONS[3], aspect_ratio=object())}
        with self.assertRaises(Exception):
            self.store.write(12, 2, unstorable)
        self.assertEqual(self.store.read(12), (1, self.DECISIONS))
        self.assertEqual(self.files(), ["0/12.sqlite"])

    def test_a_reader_sees_all_of_one_writing_or_all_of_another(self):
        def decisions(writer):
            return {
                sample: dict(aspect_ratio=1.0, target_size=(writer, sample), intermediary_size=None, crop_coordinates=(0, 0))
                for sample in range(200)
            }

        written = {writer: decisions(writer) for writer in (1, 2)}
        self.store.write(12, 1, written[1])

        def write(writer):
            for _ in range(25):
                self.store.write(12, writer, written[writer])

        def read(_):
            return [self.store.read(12) for _ in range(100)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            writers = [executor.submit(write, writer) for writer in (1, 2)]
            seen = [result for results in executor.map(read, range(2)) for result in results]
            for writer in writers:
                writer.result()

        self.assertEqual({epoch for epoch, _ in seen} - {1, 2}, set())
        for epoch, rows in seen:
            self.assertEqual(rows, written[epoch])
        self.assertEqual(self.files(), ["0/12.sqlite"])


class TestShuffleWindowSetting(unittest.TestCase):
    def test_a_window_is_a_whole_number_of_shards(self):
        for value in (None, 1, 340):
            validate_webshart_shuffle_window(value)
        for value in (0, -2, 2.5, "4", True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "whole number of shards, 1 or more"):
                validate_webshart_shuffle_window(value)

    def test_a_bad_window_is_refused_before_the_source_is_opened(self):
        with patch("webshart.discover_dataset", side_effect=AssertionError("source opened")):
            with self.assertRaisesRegex(ValueError, "webshart.shuffle_window"):
                WebshartDataBackend(None, "ws", "s3://bucket", format="parquet", shuffle_window=0)


class TestShardMetadataKeptForTheWindow(unittest.TestCase):
    SHARDS = 10

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        write_parquet_source(self.source, shards=self.SHARDS)

    def reads_on_a_second_pass(self, shuffle_window):
        import webshart

        with patch.object(webshart, "discover_dataset", wraps=webshart.discover_dataset) as discover:
            backend = WebshartDataBackend(
                None,
                "ws",
                str(self.source),
                format="parquet",
                cache_dir=str(self.root / f"cache-{shuffle_window}"),
                shuffle_window=shuffle_window,
            )
        for shard in range(self.SHARDS):
            backend.get_shard_metadata(shard)
        with patch.object(backend.loader, "get_metadata", wraps=backend.loader.get_metadata) as get_metadata:
            for shard in range(self.SHARDS):
                backend.get_shard_metadata(shard)
        return discover.call_args.kwargs["metadata_cache_size"], get_metadata.call_count

    def test_every_shard_of_the_window_stays_in_memory(self):
        self.assertEqual(self.reads_on_a_second_pass(self.SHARDS), (self.SHARDS, 0))

    def test_without_a_window_eight_shards_stay_in_memory(self):
        cache_size, reads = self.reads_on_a_second_pass(None)
        self.assertEqual(cache_size, 8)
        self.assertEqual(reads, self.SHARDS)

    def test_a_small_window_keeps_the_usual_eight(self):
        self.assertEqual(self.reads_on_a_second_pass(2)[0], 8)


class TestCaptureCatalogueSharedThroughTheMetadataCache(unittest.TestCase):
    """Processes sharing a metadata cache index a capture manifest once."""

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        write_parquet_source(self.source)
        lines = [
            json.dumps({"key": path.name, "size_bytes": path.stat().st_size, "num_rows": pq.read_metadata(path).num_rows})
            for path in sorted(self.source.glob("*.parquet"))
        ]
        (self.source / "files.jsonl").write_text("\n".join(lines) + "\n")
        (self.source / "dataset.json").write_text(
            json.dumps({"format": "parquet-image-capture", "version": 2, "files_manifest": "files.jsonl"})
        )

    def open(self, cache):
        return WebshartDataBackend(
            None,
            "ws",
            str(self.source),
            format="parquet",
            manifest="dataset.json",
            cache_dir=str(self.root / cache),
            metadata_cache_dir=str(self.root / "metadata"),
        )

    def test_the_second_process_reads_the_catalogue_the_first_indexed(self):
        import webshart.parquet

        with patch.object(
            webshart.parquet._Repository,
            "iter_capture_records",
            autospec=True,
            side_effect=webshart.parquet._Repository.iter_capture_records,
        ) as indexed:
            first = self.open("rank-0")
            second = self.open("rank-1")
        self.assertEqual(indexed.call_count, 1)
        self.assertEqual(len(list((self.root / "metadata").glob("capture-catalogue-*.sqlite3"))), 1)
        self.assertEqual(second.dataset.num_shards, SHARDS)
        self.assertEqual(
            second.list_samples_in_shard(SHARDS - 1),
            first.list_samples_in_shard(SHARDS - 1),
        )


class TestWindowFitsShardCache(unittest.TestCase):
    MIB = 2**20
    SIZES = [70 * MIB, 80 * MIB, 60 * MIB, 75 * MIB]

    def backend(self, *, window, shard_cache_gb, source="s3://bucket", free_gb=1000.0):
        metadata = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        metadata.id = "ws"
        metadata.shuffle_window = window
        metadata.data_backend = SimpleNamespace(
            source=source,
            shard_cache_gb=shard_cache_gb,
            shard_cache_dir=None if shard_cache_gb == 0 else "cache/shards",
            dataset=SimpleNamespace(get_shard_info=lambda unit: {"size": self.SIZES[unit]}),
        )
        patcher = patch(
            "simpletuner.helpers.metadata.backends.webshart.shutil.disk_usage",
            return_value=SimpleNamespace(free=free_gb * 2**30),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return metadata

    def test_a_window_the_cache_holds_for_every_rank_is_accepted(self):
        # 8 ranks x 4 shards x 80 MiB = 2.5 GiB
        self.backend(window=4, shard_cache_gb=2.5).check_window([0, 1], ranks=8)

    def test_the_largest_shard_of_the_ones_given_is_the_measure(self):
        metadata = self.backend(window=4, shard_cache_gb=2.4)
        metadata.check_window([0, 2, 3], ranks=8)  # 8 x 4 x 75 MiB = 2.34 GiB
        with self.assertRaises(ValueError):
            metadata.check_window([0, 1], ranks=8)

    def test_a_window_too_large_names_one_that_fits_and_the_cache_it_needs(self):
        with self.assertRaises(ValueError) as raised:
            self.backend(window=24, shard_cache_gb=5).check_window([0, 1], ranks=8)
        message = str(raised.exception)
        self.assertIn("(id=ws) webshart.shuffle_window=24", message)
        self.assertIn("each of 8 rank(s), about 15.0 GiB with shards of up to 80 MiB", message)
        self.assertIn("shard_cache_gb=5 cannot hold them", message)
        # 5 GiB / (8 x 80 MiB) = 8 shards; 8 x 24 x 80 MiB = 15 GiB
        self.assertIn("Set shuffle_window to 8 or less, or shard_cache_gb to at least 15.", message)
        self.assertNotIn("free", message)

    def test_a_cache_too_small_for_one_shard_a_rank_names_only_the_cache_it_needs(self):
        with self.assertRaises(ValueError) as raised:
            self.backend(window=2, shard_cache_gb=0.5).check_window([1], ranks=8)
        message = str(raised.exception)
        self.assertIn("Set shard_cache_gb to at least 2.", message)  # 8 x 2 x 80 MiB = 1.25 GiB
        self.assertNotIn("Set shuffle_window", message)

    def test_a_volume_without_room_for_that_cache_is_pointed_out(self):
        with self.assertRaises(ValueError) as raised:
            self.backend(window=24, shard_cache_gb=5, free_gb=9.5).check_window([0, 1], ranks=8)
        self.assertIn(
            "The volume holding cache/shards has 9.5 GiB free, which is not enough for that.", str(raised.exception)
        )

    def test_one_rank_needs_one_window(self):
        self.backend(window=4, shard_cache_gb=0.32).check_window([1], ranks=1)  # 4 x 80 MiB = 0.3125 GiB
        with self.assertRaises(ValueError):
            self.backend(window=4, shard_cache_gb=0.32).check_window([1], ranks=2)

    def test_a_dataset_that_caches_no_shards_takes_any_window(self):
        self.backend(window=1000, shard_cache_gb=0).check_window([0, 1], ranks=8)

    def test_a_local_dataset_takes_any_window(self):
        with TemporaryDirectory() as source:
            self.backend(window=1000, shard_cache_gb=0.01, source=source).check_window([0, 1], ranks=8)


if __name__ == "__main__":
    unittest.main()
