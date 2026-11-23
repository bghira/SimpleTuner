import json
import shutil
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from simpletuner.helpers.data_backend.config.image import ImageBackendConfig
from simpletuner.helpers.data_backend.filters import DatasetFilter
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend


class TestWebshartParquetBackend(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        self.source = source
        data = BytesIO()
        Image.new("RGB", (16, 8), "red").save(data, format="PNG")
        self.image = data.getvalue()
        self.rows = [
            dict(
                image={"bytes": self.image, "path": None},
                captions="first",
                width=16,
                height=8,
                url_status="ok",
                stored_bytes=len(self.image),
            ),
            dict(image=None, captions="dead", width=None, height=None, url_status="http_error", stored_bytes=None),
            dict(
                image={"bytes": self.image, "path": None},
                captions="third",
                width=16,
                height=8,
                url_status="ok",
                stored_bytes=len(self.image),
            ),
        ]
        pq.write_table(pa.Table.from_pylist(self.rows), source / "part.parquet", row_group_size=2)
        self.backend = WebshartDataBackend(
            None, "parquet", str(source), format="parquet", cache_dir=str(self.root / "cache")
        )

    def test_seeded_shards_keep_rank_rows_disjoint_and_resume_in_prefetch_order(self):
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("webshart")) < Version("0.9.3"):
            self.skipTest("Seeded shard order requires Webshart 0.9.3.")
        from webshart.parquet import _shuffled_shard_order

        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.preparation import TrainingPreparation
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        # Equal physical row counts, different accepted counts including one shard too small for two ranks.
        for shard in range(8):
            rows = [self.rows[0]] * (shard + 1) + [self.rows[1]] * (7 - shard)
            path = self.source / ("part.parquet" if shard == 0 else f"part-{shard}.parquet")
            pq.write_table(pa.Table.from_pylist(rows), path)
        self._publish_capture_manifest()
        config = dict(
            id="capture",
            type="webshart",
            source=str(self.source),
            resolution=64,
            resolution_type="pixel",
            webshart=dict(format="parquet", manifest="capture.json", sampling_policy="shard_local", shuffle_seed=0),
            caption_strategy="webshart",
            crop=False,
        )
        args = _CacheArgsNamespace(
            dict(output_dir=str(self.root), seed=7, max_train_steps=20, gradient_accumulation_steps=1)
        )
        online = TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)
        with closing(DatasetRuntime.from_config(config, args, _CacheAcceleratorStub(device="cpu"))) as runtime:
            self.assertTrue(runtime.prepare_for_training(online))
            metadata = runtime.metadata_backend
            for indexed in (False, True):
                if indexed:
                    runtime.prepare_view()
                for pad, row_order in ((False, "sequential"), (True, "shuffled")):
                    samples, orders, steps = [], [], []
                    for rank in range(2):
                        spec = replace(
                            metadata.epoch_spec(2, 2), dp_size=2, dp_rank=rank, repeats=1, pad=pad, sampling_order=row_order
                        )
                        self.assertEqual(spec.rank_shuffle_seed, 2 + rank)
                        order = list(_shuffled_shard_order(8, 2 + rank))
                        with closing(runtime.open_epoch(spec)) as cursor:
                            batches = list(iter(cursor.next_batch, None))
                        shard_of = runtime.data_backend.shard_of
                        seen = list(dict.fromkeys(shard_of(refs[0]) for _, refs in batches))
                        self.assertEqual(seen, [shard for shard in order if shard in seen])
                        self.assertGreater(len(seen), 3)
                        orders.append(seen)
                        samples.append({ref for _, refs in batches for ref in refs})
                        steps.append(len(batches))
                        if indexed:
                            self.assertEqual(runtime.describe(spec).batches, len(batches))
                        cut = len(batches) // 2
                        with closing(runtime.open_epoch(spec)) as cursor:
                            for _ in range(cut):
                                cursor.next_batch()
                            saved = cursor.state_dict()
                        with closing(runtime.open_epoch(replace(spec, shuffle_seed=9))) as changed:
                            with self.assertRaisesRegex(ValueError, "unchanged source"):
                                changed.load_state_dict(saved)
                        prefill = Mock()
                        prefill.shard_at.side_effect = order.__getitem__
                        with patch.object(runtime.data_backend, "sequential_prefill", return_value=prefill) as create:
                            with closing(runtime.open_epoch(spec)) as resumed:
                                create.assert_called_once_with(1, shuffle_seed=2 + rank)
                                resumed.load_state_dict(saved)
                                for expected in batches[cut:]:
                                    actual = resumed.next_batch()
                                    self.assertEqual(actual, expected)
                                    slot = prefill.advance.call_args.args[0]
                                    self.assertTrue(all(shard_of(ref) == order[slot] for ref in actual[1]))
                                with self.assertRaises(StopIteration):
                                    resumed.next_batch()
                            prefill.close.assert_called_once()
                    self.assertEqual(steps[0], steps[1])
                    self.assertNotEqual(orders[0], orders[1])
                    self.assertTrue(samples[0].isdisjoint(samples[1]))
                    if pad:
                        self.assertEqual(len(samples[0] | samples[1]), sum(range(2, 9)))

    def test_standalone_ranks_open_separate_shard_caches(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.preparation import TrainingPreparation
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        self._publish_capture_manifest()
        cache_root = self.root / "cache"
        config = {
            "id": "parquet",
            "type": "webshart",
            "source": str(self.source),
            "resolution": 64,
            "resolution_type": "pixel",
            "webshart": {
                "format": "parquet",
                "manifest": "capture.json",
                "sampling_policy": "shard_local",
                "cache_dir": str(cache_root),
                "shard_cache_dir": "{env:CACHE_ROOT}/shards/{global_rank}",
            },
        }
        args = _CacheArgsNamespace({"output_dir": str(self.root), "max_train_steps": 20, "seed": 19})
        online = TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)
        for rank in (0, 7):
            accelerator = _CacheAcceleratorStub(device="cpu")
            accelerator.is_local_main_process = rank == 0
            with (
                self.subTest(rank=rank),
                patch.dict("os.environ", {"CACHE_ROOT": str(cache_root), "RANK": str(rank)}),
                closing(DatasetRuntime.from_config(config, args, accelerator)) as runtime,
            ):
                shards = cache_root / "shards" / str(rank)
                self.assertEqual(Path(runtime.data_backend.shard_cache_dir), shards)
                self.assertTrue(shards.is_dir())
                self.assertEqual(Path(runtime.data_backend.metadata_cache_dir), cache_root / "metadata_cache")
                self.assertTrue(runtime.prepare_for_training(online))
                spec = replace(runtime.metadata_backend.epoch_spec(1, 1), sampling_order="sequential")
                with (
                    patch.object(runtime.data_backend, "sequential_prefill") as prefill,
                    closing(runtime.open_epoch(spec)) as epoch,
                ):
                    self.assertIs(epoch.prefill, prefill.return_value)

    def test_prefill_concurrency_uses_a_separate_reader(self):
        backend = self.backend
        backend.source = "s3://capture"
        foreground = Mock()
        backend.dataset = foreground
        background = Mock()
        with (
            patch.dict("os.environ", {"SIMPLETUNER_WEBSHART_PREFILL_DOWNLOADS": "16"}),
            patch.object(backend, "open_refresh_source", return_value=SimpleNamespace(dataset=background)),
        ):
            self.assertIs(backend.sequential_prefill(4), background.sequential_prefill.return_value)
        background.enable_shard_cache.assert_called_once_with(backend.shard_cache_dir, backend.shard_cache_gb, 16)
        background.sequential_prefill.assert_called_once_with(active_shards=4)
        self.assertEqual(backend.parallel_downloads, 4)
        foreground.enable_shard_cache.assert_not_called()

    def test_bundled_pixart_plan_prepares_all_three_uncropped_stages(self):
        from simpletuner.cli.examples import copy_example, list_examples
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.runtime.schedule import dataset_is_active
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace
        from simpletuner.simpletuner_sdk.server.services.example_configs_service import ExampleConfigsService

        name = "pixart/600M-512/default/progressive-webshart-full"
        self.assertIn(name, list_examples())
        imported = ExampleConfigsService().get_example(name)
        self.assertTrue(imported.dataloader_payload[-1]["text_cache_encode_only"])
        self.assertTrue(copy_example(name, str(self.root)))
        copied = self.root / name
        config = json.loads((copied / "config.json").read_text())
        self.assertEqual(config["data_backend_config"], "dataloader.json")
        self.assertEqual(config["data_backend_sampling"], "uniform")
        plan = json.loads((copied / config["data_backend_config"]).read_text())
        self.assertEqual(plan, imported.dataloader_payload)

        image = BytesIO()
        Image.new("RGB", (512, 512), "blue").save(image, format="PNG")
        for row in self.rows:
            if row["image"] is not None:
                row.update(image={"bytes": image.getvalue(), "path": None}, width=512, height=512)
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet")
        self._publish_capture_manifest()
        args = _CacheArgsNamespace({**config, "output_dir": str(self.root)})
        accelerator = _CacheAcceleratorStub(device="cpu")
        stages = plan[:-1]
        for index, (stage, expected) in enumerate(zip(stages, (128, 256, 512), strict=True)):
            with self.subTest(resolution=expected):
                options = stage["webshart"]
                self.assertEqual(options["sampling_policy"], "shard_local")
                stage["source"] = str(self.source)
                options.update(
                    manifest="capture.json",
                    cache_dir=str(self.root / "cache"),
                    metadata_cache_dir=str(self.root / "cache" / "metadata"),
                    shard_cache_dir=str(self.root / "cache" / "shards"),
                    shard_cache_gb=0,
                )
                with closing(DatasetRuntime.from_config(stage, args, accelerator)) as runtime:
                    runtime.prepare_view()
                    description = runtime.describe()
                    self.assertEqual((description.physical_rows, description.accepted_samples), (3, 2))
                    refs, count = runtime.read_page(limit=1)
                    self.assertEqual(count, 2)
                    (sample,) = runtime.read_samples(refs, view=runtime.published_read_view())
                    self.assertEqual(tuple(sample["target_size"]), (expected, expected))
                    self.assertFalse(runtime.config["crop"])
                    self.assertEqual(runtime.config["text_embeds"], plan[-1]["id"])
                active = [
                    candidate["id"]
                    for candidate in stages
                    if dataset_is_active(
                        candidate["id"], candidate, step_hint=index * 500, epoch_hint=1, update_state=False
                    )[0]
                ]
                self.assertEqual(active, [stage["id"]])

    def test_streaming_decode_exclusions_keep_ranks_and_resume_aligned(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.preparation import TrainingPreparation
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        # Two ranks, one unreadable local batch, then a whole unreadable window.
        good = self.rows[0]
        bad = {**good, "captions": "broken", "image": {"bytes": b"broken", "path": None}}
        self.rows = [bad, bad, good, {**good, "captions": "last"}]
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet")
        pq.write_table(pa.Table.from_pylist([bad] * 4), self.source / "z.parquet")
        pq.write_table(pa.Table.from_pylist([good] * 4), self.source / "zz.parquet")
        self._publish_capture_manifest()
        args = _CacheArgsNamespace(
            dict(
                output_dir=str(self.root),
                seed=19,
                gradient_accumulation_steps=1,
                max_train_steps=20,
                delete_problematic_images=True,
            )
        )
        config = dict(
            id="capture",
            type="webshart",
            source=str(self.source),
            webshart=dict(
                format="parquet", manifest="capture.json", cache_dir=str(self.root / "cache"), sampling_policy="shard_local"
            ),
            resolution=64,
            resolution_type="pixel",
            caption_strategy="webshart",
            crop=False,
        )
        online = TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)
        positions = []
        for rank in (0, 1, 0):
            reopened = bool(positions)
            with closing(DatasetRuntime.from_config(config, args, _CacheAcceleratorStub(device="cpu"))) as runtime:
                metadata = runtime.metadata_backend
                metadata.window_shards = 1
                self.assertTrue(metadata.delete_problematic_images)
                self.assertTrue(runtime.prepare_for_training(online))
                spec = replace(metadata.epoch_spec(1, 2), dp_size=2, dp_rank=rank)
                with closing(runtime.open_epoch(spec)) as cursor:
                    sampler = object.__new__(MultiAspectSampler)
                    sampler.id, sampler.dataset_type = "capture", "image"
                    sampler.dataset_runtime, sampler.epoch_provider = runtime, metadata
                    sampler._open_training_cursor = lambda: cursor
                    if positions and rank == 0:
                        cursor.load_state_dict(positions[0])
                    with (
                        patch.object(StateTracker, "get_args", return_value=SimpleNamespace(model_family="hdit_plus")),
                        patch.object(
                            runtime.data_backend, "read_image_batch", wraps=runtime.data_backend.read_image_batch
                        ) as read,
                    ):
                        batches = sampler._iterate_native()
                        if not positions or rank == 1:
                            first = next(batches)
                            self.assertEqual([record["instance_prompt_text"] for record in first], ["first", "last"])
                            self.assertTrue(all(record["decoded_training_image"].size == (16, 8) for record in first))
                            positions.append(first[0]["dataset_continuation"])
                        last = next(batches)
                        self.assertEqual(len(last), 2)
                        self.assertTrue(all(runtime.data_backend.shard_of(record["image_path"]) == 2 for record in last))
                        self.assertEqual(last[0]["dataset_continuation"]["position"], 3)
                        if reopened:
                            self.assertFalse(
                                any(
                                    runtime.data_backend.shard_of(ref) == 1
                                    for call in read.call_args_list
                                    for ref in call.args[0]
                                )
                            )
                with closing(sqlite3.connect(metadata.window_exclusions_path)) as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM exclusions WHERE reason='decode'").fetchone(), (6,))
                with patch.object(runtime.data_backend, "read_image_batch", side_effect=TimeoutError("shard fetch")):
                    window = metadata._open_training_partition(2)
                    bucket, refs = next(iter(window.buckets.items()))
                    with self.assertRaisesRegex(TimeoutError, "shard fetch"):
                        metadata.read_window_batch(2, bucket, refs)
        self.assertEqual(positions[0]["window_revision"], positions[1]["window_revision"])

    def test_seeded_decode_replacement_keeps_the_batch_on_its_data_rank(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.preparation import TrainingPreparation
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        good = self.rows[0]
        bad = {**good, "image": {"bytes": b"broken", "path": None}}
        self.rows = [bad, good, bad, good]
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet")
        pq.write_table(pa.Table.from_pylist([good] * 4), self.source / "z.parquet")
        self._publish_capture_manifest()
        config = dict(
            id="capture",
            type="webshart",
            source=str(self.source),
            resolution=64,
            resolution_type="pixel",
            caption_strategy="webshart",
            crop=False,
            webshart=dict(format="parquet", manifest="capture.json", sampling_policy="shard_local", shuffle_seed=0),
        )
        args = _CacheArgsNamespace(
            dict(
                output_dir=str(self.root),
                seed=19,
                max_train_steps=20,
                gradient_accumulation_steps=1,
                delete_problematic_images=True,
            )
        )
        samples = []
        for rank in range(2):
            accelerator = _CacheAcceleratorStub(device="cpu")
            accelerator.num_processes, accelerator.process_index = 2, rank
            with closing(DatasetRuntime.from_config(config, args, accelerator)) as runtime:
                self.assertTrue(
                    runtime.prepare_for_training(
                        TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)
                    )
                )
                metadata = runtime.metadata_backend
                spec = metadata.epoch_spec(1, 2)
                order = [rank, 1 - rank]
                with (
                    patch.object(runtime.data_backend, "seeded_shards", return_value=(order.__getitem__, None)),
                    closing(runtime.open_epoch(spec)) as cursor,
                    patch.object(StateTracker, "get_args", return_value=SimpleNamespace(model_family="hdit_plus")),
                ):
                    sampler = object.__new__(MultiAspectSampler)
                    sampler.id, sampler.dataset_type = "capture", "image"
                    sampler.dataset_runtime, sampler.epoch_provider = runtime, metadata
                    sampler._open_training_cursor = lambda: cursor
                    batches = sampler._iterate_native()
                    refs = []
                    for position in (1, 2):
                        batch = next(batches)
                        self.assertEqual(batch[0]["dataset_continuation"]["position"], position)
                        for row in batch:
                            ref = row["image_path"]
                            self.assertEqual(runtime.data_backend.shard_of(ref), batch[0]["dataset_read_view"].partition)
                            self.assertEqual(row["decoded_training_image"].size, (16, 8))
                            refs.append(ref)
                    samples.append(set(refs))
        self.assertTrue(samples[0].isdisjoint(samples[1]))
        self.assertEqual([len(s) for s in samples], [2, 4])

    def test_online_training_opens_one_shard_window_and_resumes_the_consumed_batch(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.preparation import DatasetReadView, TrainingPreparation
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        caption_rows = [{**row, "captions": [row["captions"], f"alternate {row['captions']}"]} for row in self.rows]
        table = pa.Table.from_pylist(caption_rows)
        for filename in ["part.parquet", *(f"part-{i}.parquet" for i in range(7))]:
            pq.write_table(table, self.source / filename, row_group_size=2)
        self._publish_capture_manifest()
        args = _CacheArgsNamespace(
            dict(
                output_dir=str(self.root),
                seed=19,
                max_train_steps=20,
                gradient_accumulation_steps=2,
                data_backend_sampling="uniform",
            )
        )
        config = dict(
            id="capture",
            type="webshart",
            source=str(self.source),
            webshart=dict(format="parquet", manifest="capture.json", sampling_policy="shard_local"),
            resolution=64,
            resolution_type="pixel",
            caption_strategy="webshart",
            crop=False,
        )
        online = TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)

        with (
            closing(DatasetRuntime.from_config(config, args, _CacheAcceleratorStub(device="cpu"))) as runtime,
            closing(DatasetRuntime.from_config(config, args, _CacheAcceleratorStub(device="cpu"))) as other_rank,
        ):
            metadata = runtime.metadata_backend
            spec = replace(metadata.epoch_spec(1, 1), dp_size=2)
            self.assertEqual(spec.sampling_order, "sequential")
            with (
                patch.object(metadata, "_scan_rows", side_effect=AssertionError("startup scanned rows")),
                patch.object(runtime.data_backend.dataset, "_file", side_effect=AssertionError("startup opened a shard")),
            ):
                for request in (
                    replace(online, online_text=False),
                    replace(online, ondemand_latents=False),
                    replace(online, requires_complete_view=True),
                ):
                    self.assertFalse(runtime.prepare_for_training(request))
                with patch.object(metadata.runtime_args, "data_backend_sampling", "auto-weighting"):
                    self.assertFalse(runtime.prepare_for_training(online))
                self.assertTrue(runtime.prepare_for_training(online))
                metadata.load_image_metadata()
                self.assertIsNone(runtime.describe().accepted_samples)
                self.assertIsNone(runtime.describe().batches)
                self.assertIsNone(runtime.training_batch_count())
                self.assertFalse(metadata.membership_path.exists())
                cursor = runtime.open_epoch(spec)
                initial_state = cursor.state_dict()
                self.assertEqual(initial_state["position"], 0)
            self.assertTrue(other_rank.prepare_for_training(online))

            project = runtime.data_backend.loader.iter_samples_in_shard
            with patch.object(runtime.data_backend.loader, "iter_samples_in_shard", wraps=project) as projected:
                first = cursor.next_batch()
                continuation = cursor.state_dict()
                first_partition = cursor.partition
                self.assertEqual(first_partition, 0)
                self.assertEqual(continuation["layout"]["sampling_order"], "sequential")
                self.assertLessEqual(projected.call_count, metadata.window_shards)
                self.assertGreater(projected.call_count, 0)
            self.assertFalse(metadata.membership_path.exists())

            with closing(other_rank.open_epoch(replace(spec, dp_rank=1))) as peer:
                peer_bucket, peer_refs = peer.next_batch()
                self.assertEqual(first[0], peer_bucket)
                self.assertTrue(set(first[1]).isdisjoint(peer_refs))
                self.assertEqual(peer.state_dict()["window_revision"], continuation["window_revision"])
            first_view = DatasetReadView(continuation["view"], spec.epoch, spec.seed, first_partition)
            (first_ref,) = first[1]
            with patch.object(runtime.data_backend, "read_caption_batch", side_effect=AssertionError("reread captions")):
                (sample,) = runtime.read_samples(first[1], fields=("metadata", "caption"), view=first_view)
                self.assertEqual(runtime.resolve_caption(first_ref), sample["caption"])
                captions = metadata.read_caption_batch([first_ref, first_ref])
                captions[0].append("changed")
                self.assertEqual(captions[1], sample["caption"])
                self.assertEqual(runtime.resolve_captions([first_ref]), [sample["caption"]])
            self.assertEqual(tuple(sample["target_size"]), (128, 64))
            self.assertIn(sample["caption"], (["first", "alternate first"], ["third", "alternate third"]))
            self.assertIn(sample["webshart"]["sample_idx"], (0, 2))
            batches_after_consume = [cursor.next_batch() for _ in range(4)]
            self.assertNotEqual(cursor.partition, first_partition)
            (next_ref,) = batches_after_consume[-1][1]
            with patch.object(
                runtime.data_backend, "read_caption_batch", wraps=runtime.data_backend.read_caption_batch
            ) as read:
                current, previous, duplicate = runtime.resolve_captions([next_ref, first_ref, next_ref])
                read.assert_called_once_with([first_ref])
                self.assertEqual(current, duplicate)
                self.assertEqual(previous, sample["caption"])

            with (
                closing(runtime.open_epoch(spec)) as resume,
                patch.object(metadata, "_prepare_metadata", side_effect=AssertionError("geometry was recomputed")),
            ):
                resume.load_state_dict(continuation)
                self.assertEqual([resume.next_batch() for _ in range(4)], batches_after_consume)
                resume.load_state_dict(initial_state)
                self.assertEqual(resume.next_batch(), first)
            with closing(runtime.open_epoch(spec)) as resume:
                with self.assertRaisesRegex(ValueError, "shard contents changed"):
                    resume.load_state_dict({**continuation, "window_revision": "changed"})
            with closing(runtime.open_epoch(replace(spec, sampling_order="shuffled"))) as shuffled_resume:
                with self.assertRaisesRegex(ValueError, "unchanged source"):
                    shuffled_resume.load_state_dict(continuation)
            cursor.close()

            runtime.prepare_view()
            self.assertEqual(runtime.describe().accepted_samples, 16)
            self.assertIsNone(metadata.caption_projection_strategy)
            self.assertFalse(runtime.prepare_for_training(online))

    def test_viewer_resolves_conditioning_against_each_parquet_catalogue(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace
        from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import (
            DatasetViewChanged,
            DatasetViewerService,
        )

        target_root = self.root / "reference"
        target_root.mkdir()
        shutil.copyfile(self.source / "part.parquet", target_root / "part.parquet")
        pq.write_table(pa.Table.from_pylist([self.rows[0]]), target_root / "ahead.parquet")
        source = {
            "id": "source",
            "type": "webshart",
            "source": str(self.source),
            "conditioning_data": ["target"],
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "crop": False,
            "webshart": {
                "format": "parquet",
                "sampling_policy": "shard_local",
                "cache_dir": str(self.root / "cache"),
                "metadata_cache_dir": str(self.root / "metadata"),
            },
        }
        target = {
            **source,
            "id": "target",
            "source": str(target_root),
            "dataset_type": "conditioning",
            "conditioning_type": "reference_strict",
            "conditioning_data": [],
        }
        args = _CacheArgsNamespace({"aspect_bucket_alignment": 8})
        accelerator = _CacheAcceleratorStub(device="cpu")
        publications = {}
        for config in (source, target):
            with closing(DatasetRuntime.from_config(config, args, accelerator)) as runtime:
                runtime.prepare_view()
                publications[config["id"]] = runtime.data_backend._cache_path(runtime.metadata_backend.cache_file)

        service = DatasetViewerService(project_root=self.root)
        with patch.object(service, "_find_cache_file", side_effect=lambda config, _: publications[config["id"]]):
            report = service.get_conditioning_pairs(source, target, limit=1)
            self.assertEqual(
                report["pairs"],
                [
                    {
                        "source_path": self.backend.sample_id(0, 0, "part.parquet/000000000000.jpg"),
                        "conditioning_path": self.backend.sample_id(1, 0, "part.parquet/000000000000.jpg"),
                    }
                ],
            )
            match = service.get_conditioning_file_match(
                source, [target], report["pairs"][0]["source_path"], view_revision=report["source_view_revision"]
            )
            self.assertEqual(match["file_path"], report["pairs"][0]["conditioning_path"])
            self.assertEqual(match["view_revision"], report["conditioning_view_revision"])
            self.assertIsNotNone(match["thumbnail"])
            orphans = service.get_orphaned_files(source, target)
            self.assertEqual(orphans.source_orphans, [])
            self.assertEqual(orphans.conditioning_orphans, [self.backend.sample_id(0, 0, "ahead.parquet/000000000000.jpg")])
            with self.assertRaises(DatasetViewChanged):
                service.get_conditioning_file_match(source, [target], report["pairs"][0]["source_path"], view_revision="old")

    def test_transform_cache_eviction_preserves_membership_and_recomputes_only_requested_rows(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend

        if not hasattr(self.backend.webshart, "TransformCache"):
            self.skipTest("Requires Webshart native transform cache support")
        config = {
            "webshart": {"sampling_policy": "shard_local"},
            "caption_strategy": "webshart",
            "resolution": 64,
            "resolution_type": "pixel",
            "crop": False,
        }
        args = SimpleNamespace(seed=5, aspect_bucket_alignment=8, aspect_bucket_rounding=2, disable_bucket_pruning=True)
        provider = WebshartMetadataBackend(
            id=self.backend.id,
            instance_data_dir=self.backend.cache_dir,
            cache_file=str(self.root / "buckets"),
            metadata_file=str(self.root / "metadata"),
            data_backend=self.backend,
            accelerator=SimpleNamespace(num_processes=1, process_index=0),
            batch_size=1,
            resolution=64,
            resolution_type="pixel",
            runtime_args=args,
            dataset_config=config,
        )
        self.addCleanup(provider.close)
        with patch.object(self.backend.webshart, "TransformCache", None):
            provider.compute_aspect_ratio_bucket_indices()
        refs, _ = provider.read_page(limit=10)
        self.assertEqual(len(refs), 2)
        legacy_geometry = provider.get_metadata_batch(refs)
        legacy_revision = provider.describe().view_revision
        with patch.object(provider, "_scan_rows", side_effect=AssertionError("implicit migration")):
            provider.compute_aspect_ratio_bucket_indices()
        self.assertEqual(provider.describe().view_revision, legacy_revision)
        provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
        self.assertEqual(provider.read_page(limit=10), (refs, 2))
        self.assertEqual(provider.get_metadata_batch(refs), legacy_geometry)
        missing = self.backend.sample_id(0, 1, "part.parquet/000000000001.jpg")
        revision = provider.describe().view_revision
        requests = [refs[1], missing, refs[1]]
        with patch.object(self.backend.dataset, "read_metadata_rows", side_effect=AssertionError("warm projection")):
            warm = provider.get_metadata_batch(requests)
        self.assertIsNone(warm[1])
        self.assertEqual(warm[0], warm[2])
        self.assertIsNotNone(warm[0])
        self.assertEqual(provider._geometry_lookup().get_metadata_batch(refs), [None, None])

        provider.close()
        shutil.rmtree(Path(self.backend.cache_dir) / ".webshart-transforms")
        provider.reload_cache()
        self.assertEqual(provider.describe().view_revision, revision)
        self.assertEqual(provider.read_page(limit=10), (refs, 2))
        project = self.backend.dataset.read_metadata_rows
        with (
            patch.object(self.backend.dataset, "read_metadata_rows", wraps=project) as read,
            patch.object(self.backend, "read_image_batch", side_effect=AssertionError("image payload read")),
        ):
            self.assertEqual(provider.get_metadata_batch(requests), warm)
            self.assertEqual(len(read.call_args_list), 1)
            self.assertEqual(read.call_args.args, (0, [self.backend.parse_sample_id(refs[1]).sample_idx]))
        provider.flush_prepared_metadata()
        with patch.object(self.backend.dataset, "read_metadata_rows", side_effect=AssertionError("second projection")):
            self.assertEqual(provider.get_metadata_batch(requests), warm)

        provider.close()
        (shard_cache,) = (Path(self.backend.cache_dir) / ".webshart-transforms").rglob("*.sqlite3")
        location = self.backend.parse_sample_id(refs[1])
        with closing(sqlite3.connect(shard_cache)) as db, db:
            (cached,) = db.execute("SELECT value FROM transforms WHERE row=?", (location.sample_idx,)).fetchone()
            stale = {**json.loads(cached), "crop_coordinates": [314, 159]}
            db.execute("UPDATE transforms SET value=? WHERE row=?", (json.dumps(stale).encode(), location.sample_idx))
        provider.reload_cache()
        self.assertEqual(provider.get_metadata_by_filepath(refs[1])["crop_coordinates"], [314, 159])
        with (
            patch.object(provider, "flush_prepared_metadata", side_effect=OSError("interrupted transform write")),
            self.assertRaisesRegex(OSError, "interrupted transform write"),
        ):
            provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
        provider.reload_cache()
        self.assertEqual(provider.describe().view_revision, revision)
        self.assertEqual(provider.get_metadata_by_filepath(refs[1])["crop_coordinates"], [314, 159])

        provider.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
        self.assertNotEqual(provider.describe().view_revision, revision)
        self.assertEqual(provider.read_page(limit=10), (refs, 2))
        self.assertEqual(provider.get_metadata_batch(requests), warm)
        with patch.object(provider, "_scan_rows", side_effect=AssertionError("invalidated refreshed view")):
            provider.compute_aspect_ratio_bucket_indices()

        provider.dataset_config.update(crop=True, crop_style="center", crop_aspect="square")
        provider.compute_aspect_ratio_bucket_indices()
        centered = provider._published
        provider.dataset_config["crop_style"] = "random"
        provider.compute_aspect_ratio_bucket_indices()
        randomized = provider._published
        self.assertEqual(centered["geometry_sha256"], randomized["geometry_sha256"])
        self.assertNotEqual(centered["view_revision"], randomized["view_revision"])

    def test_fixed_shape_epoch_recrop_projects_only_requested_rows(self):
        import torch

        from simpletuner.helpers.caching.vae import VAECache
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.preparation import DatasetReadView
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "webshart": {"sampling_policy": "global"},
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "crop": True,
            "crop_style": "random",
            "crop_aspect": "square",
            "vae_cache_clear_each_epoch": True,
        }
        args = SimpleNamespace(
            seed=5,
            model_family="sdxl",
            aspect_bucket_alignment=8,
            aspect_bucket_rounding=2,
            disable_bucket_pruning=True,
            gradient_accumulation_steps=1,
            max_train_steps=None,
        )
        accelerator = SimpleNamespace(
            num_processes=1, process_index=0, is_main_process=True, is_local_main_process=True, wait_for_everyone=Mock()
        )
        provider = WebshartMetadataBackend(
            id=self.backend.id,
            instance_data_dir=self.backend.cache_dir,
            cache_file=str(self.root / "buckets"),
            metadata_file=str(self.root / "metadata"),
            data_backend=self.backend,
            accelerator=accelerator,
            batch_size=1,
            resolution=8,
            resolution_type="pixel",
            runtime_args=args,
            dataset_config=config,
        )
        self.addCleanup(provider.close)
        runtime = DatasetRuntime(self.backend.id, self.backend, provider, config)
        runtime.sampler = Mock()
        runtime.prepare_view()
        (ref, other), count = runtime.read_page(limit=2)
        self.assertEqual(count, 2)
        self.assertTrue(runtime.recrops_lazily)
        revision = runtime.describe().view_revision
        (base,) = runtime.read_samples([ref])
        model = Mock(config=SimpleNamespace())
        base_vae_key = provider.image_artifact_key(ref, "vae", model)
        base_embed_key = provider.image_artifact_key(ref, "conditioning", model)
        store = LocalDataBackend(accelerator, "artifacts")
        cache_dir = str(self.root / "latents")

        def new_cache(run):
            return VAECache(
                runtime.id,
                model,
                vae=None,
                accelerator=accelerator,
                metadata_backend=provider,
                instance_data_dir=provider.instance_data_dir,
                image_data_backend=self.backend,
                cache_data_backend=store,
                cache_dir=cache_dir,
                cache_inventory_dir=str(self.root / run),
                dataset_config=runtime.config,
                vae_cache_ondemand=True,
            )

        cache = runtime.vae_cache = new_cache("training-run")
        prior_key = cache.cache_path_for_image(ref)
        foreign_key = new_cache("other-run").cache_path_for_image(ref)
        unscoped_key = str(Path(cache_dir) / f"{base_vae_key}.pt")
        tensor = torch.zeros(1)
        store.write_batch([prior_key, foreign_key, unscoped_key], [tensor] * 3)

        def encode(*, image_pixel_values, disable_queue, view):
            return [(cache.cache_path_for_image(path, view=view), path, value) for value, path, _, _ in image_pixel_values]

        with (
            patch.object(provider, "_scan_rows", side_effect=AssertionError("epoch scanned source rows")),
            patch.object(provider, "iter_preparation_batches", side_effect=AssertionError("epoch scanned membership")),
            patch.object(provider, "enqueue_prepared_metadata", side_effect=AssertionError("epoch rewrote base cache")),
            patch.object(self.backend.dataset, "read_metadata_rows", side_effect=AssertionError("warm projection")),
            patch.object(self.backend, "read_image_batch", side_effect=AssertionError("geometry read image bytes")),
            patch.object(provider._transforms, "read_batch", wraps=provider._transforms.read_batch) as read,
            patch.object(cache, "_encode_images_in_batch", side_effect=encode) as encoder,
        ):
            coordinates = []
            for epoch in range(2, 5):
                epoch_view = DatasetReadView(revision, epoch, args.seed)
                runtime.prepare_epoch(epoch, rebuild_random_aspect=True, gradient_accumulation_steps=1, apply_padding=False)
                runtime.start_epoch(epoch)
                read.assert_not_called()
                self.assertFalse(store.exists(prior_key))
                self.assertTrue(store.exists(foreign_key))
                self.assertTrue(store.exists(unscoped_key))
                prior_key = cache.cache_path_for_image(ref, view=epoch_view)
                self.assertFalse(store.exists(prior_key))
                latent = cache.encode_images([torch.tensor([epoch])], [ref], view=epoch_view)[0]
                repeated = cache.encode_images([torch.tensor([-1])], [ref], view=epoch_view)[0]
                torch.testing.assert_close(latent, repeated)
                encoder.assert_called_once()
                self.assertEqual(encoder.call_args.kwargs["view"], epoch_view)
                encoder.reset_mock()
                read.reset_mock()
                transformed = runtime.read_samples([ref, ref], view=epoch_view)
                read.assert_called_once_with([ref, ref])
                read.reset_mock()
                self.assertEqual(transformed[0], transformed[1])
                self.assertEqual(transformed[0]["target_size"], base["target_size"])
                self.assertEqual(transformed[0]["intermediary_size"], base["intermediary_size"])
                coordinates.append(transformed[0]["crop_coordinates"])
            self.assertTrue(any(offset != base["crop_coordinates"] for offset in coordinates))
            self.assertNotEqual(provider.image_artifact_key(ref, "vae", model, view=epoch_view), base_vae_key)
            self.assertEqual(provider.image_artifact_key(ref, "conditioning", model, view=epoch_view), base_embed_key)
            with provider.open_preparation_view() as view:
                self.assertEqual(view.metadata.get_metadata_by_filepath(ref)["crop_coordinates"], base["crop_coordinates"])
                with (
                    patch.object(provider, "_published", {**provider._published, "view_revision": "replacement"}),
                    patch.object(store, "read_cached_tensors", side_effect=AssertionError("stale view read the store")),
                ):
                    with self.assertRaisesRegex(ValueError, "view changed"):
                        view.metadata.get_metadata_by_filepath(ref)
                    with self.assertRaisesRegex(ValueError, "view changed"):
                        cache.encode_images([None], [ref], view=epoch_view)
            self.assertTrue(all(other not in call.args[0] for call in read.call_args_list))
            resumed_cache = new_cache("training-run")
            runtime.start_epoch(epoch + 1)
            self.assertEqual(resumed_cache.cache_path_for_image(ref, view=epoch_view), prior_key)
            self.assertNotEqual(resumed_cache.cache_path_for_image(ref), prior_key)
            torch.testing.assert_close(resumed_cache.encode_images([torch.tensor([-1])], [ref], view=epoch_view)[0], latent)
        self.assertEqual(runtime.describe().view_revision, revision)
        self.assertEqual(runtime.read_page(limit=2), ([ref, other], 2))

        with (
            closing(MultiAspectSampler(runtime, model=None, accelerator=accelerator, batch_size=1)) as training_sampler,
            patch.object(StateTracker, "get_args", return_value=args),
        ):
            runtime.sampler = training_sampler
            runtime.start_epoch(1)
            (first,) = next(iter(training_sampler))
            runtime.acknowledge_batch(first["dataset_continuation"])
            checkpoint = str(self.root / "epoch-checkpoint.json")
            runtime.save_checkpoint(checkpoint)

            runtime.prepare_epoch(2, rebuild_random_aspect=True, gradient_accumulation_steps=1, apply_padding=False)
            runtime.start_epoch(2)
            (expected,) = next(iter(training_sampler))
            runtime.restore_checkpoint(checkpoint, next_epoch=2)
            runtime.prepare_epoch(2, rebuild_random_aspect=True, gradient_accumulation_steps=1, apply_padding=False)
            runtime.start_epoch(2, resume_boundary=True)
            (resumed,) = next(iter(training_sampler))
            self.assertEqual(resumed["dataset_read_view"].epoch, 2)
            self.assertEqual(resumed, expected)
        runtime.sampler = Mock()

        image = Image.linear_gradient("L").rotate(90).resize((16, 8)).convert("RGB")
        epoch_with_changed_crop = next(
            epoch for epoch, offset in enumerate(coordinates, 2) if offset != base["crop_coordinates"]
        )
        runtime.start_epoch(epoch_with_changed_crop)
        changed_view = DatasetReadView(revision, epoch_with_changed_crop, args.seed)
        (updated,) = runtime.read_samples([ref], view=changed_view)
        with patch.object(StateTracker, "get_metadata_by_filepath", side_effect=AssertionError("global metadata lookup")):
            self.assertEqual(
                cache._gather_sample_metadata([ref], view=changed_view)[0]["metadata"]["crop_coordinates"],
                updated["crop_coordinates"],
            )
        cropped = TrainingSample(image, self.backend.id, updated, ref, data_backend_config=runtime.config).prepare()
        top, left = updated["crop_coordinates"]
        width, height = updated["target_size"]
        self.assertEqual(cropped.image.tobytes(), image.crop((left, top, left + width, top + height)).tobytes())

        sampler = MultiAspectSampler.__new__(MultiAspectSampler)
        sampler.dataset_runtime = runtime
        sampler.id = runtime.id
        sampler.model = None
        with closing(runtime.open_validation()) as cursor:
            sampler.validation_cursor = cursor
            cursor.next_refs(1)
            cursor.next_refs(1)
            position = cursor.state_dict()
            with (
                patch.object(
                    self.backend, "read_image_batch", side_effect=lambda refs: (refs, [image.copy() for _ in refs])
                ),
                patch.object(StateTracker, "get_model", return_value=None),
                patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("global dataset config")),
                patch.object(StateTracker, "get_args", return_value=args),
            ):
                (later_preview,) = sampler.retrieve_validation_set(1)
                runtime.start_epoch(1)
                cursor.load_state_dict(position)
                (first_preview,) = sampler.retrieve_validation_set(1)
            self.assertEqual(later_preview.ref, first_preview.ref)
            self.assertEqual(later_preview.prompt, first_preview.prompt)
            self.assertEqual(later_preview.image.tobytes(), first_preview.image.tobytes())
            self.assertEqual(later_preview.view.revision, position["view"])
            with (
                patch.object(provider, "_published", {**provider._published, "view_revision": "new-revision"}),
                patch.object(self.backend, "read_image_batch", side_effect=AssertionError("stale cursor fetched pixels")),
                self.assertRaisesRegex(ValueError, "view changed"),
            ):
                sampler.retrieve_validation_set(1)

        runtime.start_epoch(1)
        self.assertEqual(runtime.read_samples([ref]), [base])

        runtime.config["vae_cache_clear_each_epoch"] = False
        cache = runtime.vae_cache = new_cache("persistent-run")
        stable_key = cache.cache_path_for_image(ref)
        store.write(stable_key, tensor)
        runtime.prepare_epoch(2, rebuild_random_aspect=True, gradient_accumulation_steps=1, apply_padding=False)
        runtime.start_epoch(2)
        self.assertEqual(cache.cache_path_for_image(ref, view=changed_view), stable_key)
        self.assertEqual(runtime.read_samples([ref], view=changed_view), [base])
        with patch.object(cache, "_encode_images_in_batch", side_effect=AssertionError("re-encoded a stable latent")):
            torch.testing.assert_close(cache.encode_images([torch.ones(1)], [ref])[0], tensor)

    def test_prepared_geometry_reads_compact_and_existing_text_records(self):
        from simpletuner.helpers.metadata.backends.webshart_geometry import GeometryLookup, PreparedGeometry

        record = {
            "original_size": [1024, 1536],
            "crop_coordinates": [256, 512],
            "target_size": [512, 512],
            "webshart": {"shard_idx": 1, "sample_idx": 3, "filename": "data/0003.jpg"},
        }
        with closing(sqlite3.connect(":memory:")) as connection:
            geometry = PreparedGeometry(connection)
            geometry.add("512x512", "new", record)
            connection.execute("INSERT INTO geometry VALUES (?,?,?,?)", ("original", json.dumps(record), "512x512", 1))
            geometry.finish()
            lookup = GeometryLookup(connection)
            self.assertEqual(connection.execute("SELECT typeof(record) FROM geometry WHERE ref='new'").fetchone(), ("blob",))
            self.assertEqual(lookup.get_metadata_by_filepath("new"), record)
            self.assertEqual(
                lookup.get_metadata_batch(["original", "new", "new", "missing"]), [record, record, record, None]
            )

    def test_runtime_can_request_exact_source_count_without_null_image_rows(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        runtime = DatasetRuntime(self.backend.id, self.backend, None, {})
        self.assertEqual(runtime.count_source_rows(), 3)
        if not hasattr(self.backend.dataset, "count_rows"):
            with self.assertRaisesRegex(NotImplementedError, "installed Webshart"):
                runtime.count_source_rows(exclude_null_images=True)
            return
        self.assertEqual(runtime.count_source_rows(exclude_null_images=True), 2)
        self.assertEqual([row["sample_idx"] for row in self.backend.list_samples_in_shard(0)], [0, 2])

    def test_missing_jpeg_xl_decoder_is_reported_before_reading_payloads(self):
        ref = self.backend.sample_id(0, 0, "part.parquet/000000000000.jxl")
        with (
            patch("simpletuner.helpers.data_backend.webshart.Image.registered_extensions", return_value={}),
            patch.object(self.backend.loader, "load_samples", side_effect=AssertionError("opened payloads")),
        ):
            with self.assertRaisesRegex(ImportError, "pillow-jxl-plugin"):
                self.backend.read_image_batch([ref])
            with self.assertRaisesRegex(ImportError, "pillow-jxl-plugin"):
                self.backend.read_image(ref)

    def test_cache_reader_sees_only_complete_latent_writes(self):
        cache_ref = "latent.pt"
        self.backend.write(cache_ref, b"previous")
        write_started, release_write = Event(), Event()

        def save(_data, handle):
            handle.write(b"new ")
            handle.flush()
            write_started.set()
            if not release_write.wait(10):
                raise TimeoutError("test did not release latent writer")
            handle.write(b"latent")

        with patch("simpletuner.helpers.data_backend.webshart.torch.save", side_effect=save):
            with ThreadPoolExecutor(max_workers=1) as worker:
                write = worker.submit(self.backend.write, cache_ref, object())
                try:
                    self.assertTrue(write_started.wait(10))
                    self.assertEqual(self.backend.read(cache_ref), b"previous")
                finally:
                    release_write.set()
                write.result(timeout=10)
        self.assertEqual(self.backend.read(cache_ref), b"new latent")

    def test_prepared_view_uses_factory_bound_settings_without_tracker_lookups(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "metadata_backend": "webshart",
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = {"train_batch_size": 1, "disable_bucket_pruning": False, "aspect_bucket_alignment": 8, "seed": 123}
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("dataset settings lookup")),
            patch.object(StateTracker, "get_args", side_effect=AssertionError("run settings lookup")),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = create_metadata_provider(
                self.backend.id,
                config,
                self.backend,
                accelerator,
                args,
                instance_data_dir=self.backend.cache_dir,
                cache_root=self.backend.cache_dir,
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            self.assertEqual(metadata.describe().accepted_samples, 2)

    def test_standalone_scan_reuses_configured_training_geometry(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.data_backend.config import create_backend_config
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.models.hidream.model import HiDream
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import (
            DatasetScanService,
            _ScanAcceleratorStub,
            _ScanArgsNamespace,
        )

        config = {
            "id": "standalone",
            "type": "webshart",
            "metadata_backend": "webshart",
            "source": str(self.source),
            "webshart": {"format": "parquet", "sampling_policy": "global", "cache_dir": str(self.root / "scan")},
            "caption_strategy": "webshart",
            "resolution": 1536,
            "resolution_type": "pixel",
            "crop": True,
            "crop_style": "random",
        }
        run_settings = {
            "seed": 17,
            "aspect_bucket_alignment": 32,
            "aspect_bucket_rounding": 3,
            "model_family": "hidream",
            "train_batch_size": 2,
            "disable_bucket_pruning": False,
            "framerate": 30,
            "output_dir": str(self.root / "output"),
        }
        config["aspect_bucket_alignment"] = run_settings["aspect_bucket_alignment"]
        global_config = {f"--{key}": value for key, value in run_settings.items()}
        global_config["--aspect_bucket_alignment"] = 64
        with (
            patch.object(StateTracker, "set_args"),
            patch.object(StateTracker, "set_accelerator"),
            patch.object(StateTracker, "get_model", side_effect=AssertionError("global model lookup")),
            patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda path: path),
        ):
            DatasetScanService()._execute_scan(config, global_config, False, lambda *_: None)
            args = _ScanArgsNamespace(run_settings)
            accelerator = _ScanAcceleratorStub()
            with closing(DatasetRuntime.from_config(config, args, accelerator)) as runtime:
                refs, count = runtime.read_page()
                self.assertEqual(count, 2)
                before = runtime.read_samples(refs)
                for record in before:
                    width, height = record["target_size"]
                    self.assertLessEqual(width * height, HiDream.MAXIMUM_CANVAS_SIZE)
                revision = runtime.describe().view_revision
                self.assertEqual(
                    runtime.metadata_backend._published["transform_cache_recipe"]["canvas_limit"],
                    HiDream.MAXIMUM_CANVAS_SIZE,
                )

            scratch_backend = WebshartDataBackend(
                None, "standalone", str(self.source), format="parquet", cache_dir=config["webshart"]["cache_dir"]
            )
            resolved = create_backend_config(config, vars(args)).to_dict()["config"]
            provider = create_metadata_provider(
                config["id"],
                resolved,
                scratch_backend,
                accelerator,
                args,
                instance_data_dir=scratch_backend.cache_dir,
                cache_root=scratch_backend.cache_dir,
                model=HiDream.__new__(HiDream),
            )
            with (
                closing(provider),
                patch.object(WebshartMetadataBackend, "_scan_rows", side_effect=AssertionError("recomputed geometry")),
            ):
                provider.compute_aspect_ratio_bucket_indices()
                self.assertEqual(provider.describe().view_revision, revision)
                self.assertEqual(provider.get_metadata_batch(refs)[0]["target_size"], before[0]["target_size"])

    def test_seeded_remote_ranks_prefill_separate_caches_and_workers_reopen_the_same_path(self):
        import webshart

        root = self.root / "shards"
        with patch.object(webshart, "discover_dataset", return_value=self.backend.dataset):
            for rank in range(2):
                backend = WebshartDataBackend(
                    SimpleNamespace(num_processes=2, process_index=rank),
                    "capture",
                    "s3://capture",
                    format="parquet",
                    cache_dir=str(self.root / "cache"),
                    shard_cache_dir=str(root),
                    shuffle_seed=0,
                )
                self.assertEqual(Path(backend.shard_cache_dir), root / f"rank-{rank}")
                worker = WebshartDataBackend.from_instance_representation(backend.get_instance_representation())
                self.assertEqual(worker.shard_cache_dir, backend.shard_cache_dir)
                self.assertEqual(worker.shuffle_seed, 0)

    def test_remote_shard_cache_bounds_retained_readers(self):
        import webshart

        for source, limit, readers in ((str(self.source), 25, 8), ("s3://capture", 25, 1), ("s3://capture", 0, 8)):
            with (
                self.subTest(source=source, limit=limit),
                patch.object(webshart, "discover_dataset", return_value=self.backend.dataset) as discover,
            ):
                backend = WebshartDataBackend(
                    None, "capture", source, format="parquet", cache_dir=str(self.root / "cache"), shard_cache_gb=limit
                )
                self.assertEqual(discover.call_args.kwargs["metadata_cache_size"], readers)
                backend.open_refresh_source()
                self.assertEqual(discover.call_args.kwargs["metadata_cache_size"], readers)

    def test_capture_catalogue_uses_the_configured_metadata_cache(self):
        if not getattr(type(self.backend.dataset), "supports_catalogue_cache", False):
            self.skipTest("installed Webshart has no shared capture catalogue")
        self._publish_capture_manifest()
        metadata_cache = self.root / "shared-metadata"
        datasets = [
            WebshartDataBackend(
                None,
                name,
                str(self.source),
                format="parquet",
                manifest="capture.json",
                cache_dir=str(self.root / name),
                metadata_cache_dir=str(metadata_cache),
            )
            for name in ("global-schedule", "shard-local-schedule")
        ]
        self.assertEqual(datasets[0].dataset.fingerprint, datasets[1].dataset.fingerprint)
        self.assertEqual(len(list(metadata_cache.glob("capture-catalogue-*.sqlite3"))), 1)

    def test_standalone_runtime_binds_the_providers_default_caption_strategy(self):
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService, _CacheAcceleratorStub

        service = CacheJobService()
        args = service._build_args({"resolution": 8, "resolution_type": "pixel"})
        cache_dir = self.root / "standalone"
        config = {
            "id": "standalone",
            "type": "webshart",
            "source": str(self.source),
            "webshart": {"format": "parquet", "sampling_policy": "global", "cache_dir": str(cache_dir)},
            "preserve_data_backend_cache": True,
        }
        with patch.object(StateTracker, "set_data_backend_config", side_effect=AssertionError("registry configuration")):
            runtime = service._create_dataset_runtime(config, args, _CacheAcceleratorStub(device="cpu"))
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.config["caption_strategy"], "webshart")
        self.assertEqual(Path(runtime.config["instance_data_dir"]), cache_dir)
        self.assertIs(runtime.metadata_backend.active_dataset_config, runtime.config)

    def test_filtered_rows_keep_physical_identity_captions_and_pixels(self):
        entries = self.backend.list_samples_in_shard(0)
        self.assertEqual([e["sample_idx"] for e in entries], [0, 2])
        paths = self.backend.list_files()[0][2]
        self.assertEqual([self.backend.parse_sample_id(path).sample_idx for path in paths], [0, 2])
        self.assertEqual(self.backend.get_caption(paths[1]), "third")
        self.assertEqual(self.backend.read(paths[1]), self.image)
        keys, images = self.backend.read_image_batch([paths[1], paths[0], paths[1]])
        self.assertEqual(keys, [paths[1], paths[0], paths[1]])
        self.assertEqual([image.size for image in images], [(16, 8)] * 3)
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        filtered = self.backend.list_shard_sample_aspect_buckets(
            [0], dataset_filter=DatasetFilter({"path_include": "000000000002"})
        )
        self.assertEqual([e["sample_idx"] for e in filtered[0]["buckets"]["2.00"]], [2])

    def test_nan_caption_from_parquet_projection_is_missing(self):
        source = self.root / "float-captions"
        source.mkdir()
        pq.write_table(pa.Table.from_pylist([dict(self.rows[0], captions=float("nan"))]), source / "part.parquet")
        backend = WebshartDataBackend(
            None, "float-captions", str(source), format="parquet", cache_dir=str(self.root / "float-cache")
        )
        (path,) = backend.list_files()[0][2]
        self.assertIsNone(backend.get_caption(path))

    def test_projected_caption_batch_skips_shard_hydration_and_keeps_physical_order(self):
        if not self.backend.supports_caption_projection:
            self.skipTest("installed Webshart lacks projected row reads")
        paths = self.backend.list_files()[0][2]
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        cfg = {"caption_strategy": "webshart", "prepend_instance_prompt": True, "instance_prompt": "prefix"}
        runtime = DatasetRuntime(self.backend.id, self.backend, None, cfg)
        with (
            patch.object(self.backend.dataset, "_metadata", side_effect=AssertionError("whole shard")),
            patch.object(self.backend.dataset, "read_metadata_rows", wraps=self.backend.dataset.read_metadata_rows) as read,
        ):
            refs = [paths[1], paths[0], paths[1]]
            result = list(runtime.iter_caption_records(refs, batch_size=3))
            self.assertEqual(result, list(zip(["prefix third", "prefix first", "prefix third"], refs)))
            self.assertEqual(read.call_args.args, (0, [2, 0, 2]))
            self.assertEqual(read.call_count, 1)
            self.assertNotIn("image", read.call_args.kwargs["fields"])

    def test_roundtrip_preserves_format_and_omits_credentials(self):
        value = self.backend.get_instance_representation()
        self.assertEqual(value["format"], "parquet")
        self.assertNotIn("aws_access_key_id", value)
        self.assertNotIn("aws_secret_access_key", value)
        self.assertNotIn("aws_session_token", value)
        restored = self.backend.from_instance_representation(value)
        self.assertEqual(restored.get_caption(restored.list_files()[0][2][1]), "third")

    def test_aws_config_reaches_native_discovery_without_python_transfers(self):
        from simpletuner.helpers.data_backend.builders.webshart import WebshartBackendBuilder

        payload = dict(
            id="s3-images",
            type="webshart",
            source="s3://images",
            metadata_backend="webshart",
            caption_strategy="webshart",
            aws_region_name="us-east-1",
            aws_endpoint_url="https://storage.example",
            aws_access_key_id="example-access",
            aws_secret_access_key="example-secret",
            aws_session_token="example-session",
            aws_max_pool_connections=16,
            webshart=dict(format="parquet", manifest="combined/dataset.json", columns={"captions": "text"}),
        )
        config = ImageBackendConfig.from_dict(payload, {})
        roundtrip = config.to_dict()["config"]
        self.assertEqual(roundtrip["webshart"]["manifest"], "combined/dataset.json")
        self.assertEqual(roundtrip["aws_session_token"], "example-session")
        with patch("simpletuner.helpers.data_backend.builders.webshart.WebshartDataBackend") as cls:
            WebshartBackendBuilder(None, {})._create_backend(config)
            for key in (
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_endpoint_url",
                "aws_max_pool_connections",
            ):
                self.assertEqual(cls.call_args.kwargs[key], payload[key])
            self.assertEqual(cls.call_args.kwargs["format"], "parquet")

    def test_missing_and_corrupt_images_are_counted_as_decode_failures(self):
        self.rows[2]["image"]["bytes"] = b"broken image"
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        backend = WebshartDataBackend(
            None, "broken", str(self.source), format="parquet", cache_dir=str(self.root / "broken-cache")
        )
        keys, images = backend.read_image_batch(backend.list_files()[0][2])
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(images), 1)

    def test_indexed_pixel_training_excludes_unreadable_native_rows_when_requested(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.metadata.preparation import TrainingPreparation
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace

        self.rows[2]["image"]["bytes"] = b"broken image"
        shard = self.source / "part.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows), shard, row_group_size=2)
        original = shard.read_bytes()
        args = _CacheArgsNamespace(
            {
                "output_dir": str(self.root),
                "max_train_steps": 2,
                "data_backend_sampling": "uniform",
                "skip_file_discovery": "",
            }
        )
        accelerator = _CacheAcceleratorStub(device="cpu")
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = args
        factory.accelerator = accelerator
        config = {
            "id": "pixel",
            "type": "webshart",
            "source": str(self.source),
            "metadata_backend": "webshart",
            "caption_strategy": "webshart",
            "webshart": {"format": "parquet", "sampling_policy": "shard_local", "cache_dir": str(self.root / "pixel")},
            "resolution": 8,
            "resolution_type": "pixel",
            "crop": False,
            "delete_problematic_images": True,
        }
        prep = TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=True)
        with (
            closing(DatasetRuntime.from_config(config, args, accelerator)) as runtime,
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(factory, "_handle_config_versioning"),
        ):
            self.assertTrue(runtime.metadata_backend.delete_problematic_images)
            factory._handle_bucket_operations(
                config,
                {
                    "id": runtime.id,
                    "config": runtime.config,
                    "metadata_backend": runtime.metadata_backend,
                    "dataset_runtime": runtime,
                },
                None,
                preparation=prep,
            )
            refs, count = runtime.read_page()
            self.assertEqual(count, 1)
            self.assertEqual(runtime.resolve_captions(refs), ["first"])
            self.assertEqual(runtime.metadata_backend._published["exclusion_counts"], {"decode": 1})
            self.assertEqual(shard.read_bytes(), original)

    def test_metadata_bucketing_uses_dimensions_without_image_reads(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = dict(caption_strategy="webshart", resolution=64, resolution_type="pixel", crop=False)
        args = SimpleNamespace(
            disable_bucket_pruning=False, allow_dataset_oversubscription=False, ignore_missing_files=False, seed=42
        )
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=None,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            with patch.object(
                self.backend.loader, "load_sample", side_effect=AssertionError("metadata must not load image bytes")
            ):
                metadata.compute_aspect_ratio_bucket_indices()
            paths = [path for bucket in metadata.aspect_ratio_bucket_indices.values() for path in bucket]
            self.assertEqual(len(paths), 2)
            self.assertEqual(metadata.filtering_statistics["skipped"]["error"], 0)
            self.assertEqual({self.backend.parse_sample_id(path).sample_idx for path in paths}, {0, 2})
            from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

            runtime = DatasetRuntime(
                self.backend.id,
                self.backend,
                metadata,
                {
                    **config,
                    "dataset_type": "image",
                    "use_captions": True,
                    "prepend_instance_prompt": False,
                    "instance_prompt": None,
                    "disable_multiline_split": False,
                },
            )
            requests = [paths[1], paths[0], paths[1]]
            with (
                patch.object(StateTracker, "get_dataset", side_effect=AssertionError("registry lookup")),
                patch.object(self.backend.loader, "load_sample", side_effect=AssertionError("unexpected payload read")),
            ):
                records = runtime.read_samples(requests)
                self.assertEqual([record["image_path"] for record in records], requests)
                self.assertEqual([runtime.resolve_caption(ref) for ref in requests], ["third", "first", "third"])
                records[0]["webshart"]["sample_idx"] = 99
                self.assertEqual(records[2]["webshart"]["sample_idx"], 2)
                self.assertEqual(metadata.get_metadata_by_filepath(paths[1])["webshart"]["sample_idx"], 2)
            with (
                patch.object(self.backend.loader, "load_samples", wraps=self.backend.loader.load_samples) as batch_read,
                patch.object(metadata, "get_metadata_by_filepath", side_effect=AssertionError("geometry read")),
            ):
                payloads = runtime.read_samples(requests, fields={"image"})
                self.assertEqual([record["image"].size for record in payloads], [(16, 8)] * 3)
                self.assertEqual(batch_read.call_count, 1)
            reloaded = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=None,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            self.assertEqual(reloaded.aspect_ratio_bucket_indices, metadata.aspect_ratio_bucket_indices)

    def test_native_bucket_shapes_need_the_same_preparation_recipe(self):
        from ast import literal_eval

        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        native = self.backend.loader.list_shard_sample_aspect_buckets(
            [0], key="geometry-tuple", metadata_fields=("width", "height")
        )[0]["buckets"]
        native_shapes = {entry["sample_idx"]: literal_eval(shape) for shape, entries in native.items() for entry in entries}
        self.assertEqual(native_shapes, {0: (16, 8), 2: (16, 8)})

        scaled = self.backend.loader.list_shard_sample_aspect_buckets(
            [0],
            key="geometry-tuple",
            target_pixel_area=64 * 64,
            target_resolution_multiple=8,
            metadata_fields=("width", "height"),
        )[0]["buckets"]
        self.assertEqual([literal_eval(shape) for shape in scaled], [(88, 48)])

        args = SimpleNamespace(model_family="sdxl", aspect_bucket_alignment=8, aspect_bucket_rounding=2)
        accel = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        for resolution_type, resolution, crop, minimum_size, expected in (
            ("pixel", 8, False, None, native_shapes),
            ("pixel", 64, False, None, {0: (128, 64), 2: (128, 64)}),
            ("pixel", 64, True, None, {0: (64, 64), 2: (64, 64)}),
            ("pixel", 8, False, 9, {}),
            ("area", 64 * 64 / 1e6, False, None, {0: (88, 48), 2: (88, 48)}),
        ):
            config = {
                "caption_strategy": "webshart",
                "webshart": {"sampling_policy": "global"},
                "resolution": resolution,
                "resolution_type": resolution_type,
                "crop": crop,
                "crop_aspect": "square",
            }
            with (
                self.subTest(resolution=resolution, resolution_type=resolution_type, crop=crop, minimum_size=minimum_size),
                patch.object(StateTracker, "get_data_backend_config", return_value=config),
                patch.object(StateTracker, "get_args", return_value=args),
                patch.object(StateTracker, "get_model", return_value=None),
                patch.object(StateTracker, "aspect_resolution_map", {}),
                patch.object(StateTracker, "_save_to_disk"),
            ):
                metadata = WebshartMetadataBackend(
                    id=self.backend.id,
                    instance_data_dir=self.backend.cache_dir,
                    cache_file=str(self.root / "buckets"),
                    metadata_file=str(self.root / "metadata"),
                    data_backend=self.backend,
                    accelerator=accel,
                    batch_size=1,
                    resolution=resolution,
                    resolution_type=resolution_type,
                    minimum_image_size=minimum_size,
                )
                stats = {"total_processed": 0, "skipped": {"too_small": 0}}
                scanned = metadata._scan_rows((), stats, True, None)
                shapes = {self.backend.parse_sample_id(ref).sample_idx: record["target_size"] for _, ref, record in scanned}
                self.assertEqual(shapes, expected)

    def test_native_preparation_shape_is_independent_of_geometry_memo_and_scan_order(self):
        from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 1.0,
            "resolution_type": "area",
            "crop": False,
            "crop_style": None,
        }
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        args = SimpleNamespace(model_family="sdxl", aspect_bucket_alignment=8, aspect_bucket_rounding=2, seed=21)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=1.0,
                resolution_type="area",
            )
            aligned_8 = metadata._geometry_recipe(config)
            args.aspect_bucket_alignment = 64
            aligned_64 = metadata._geometry_recipe(config)
            config["crop"] = True
            config["crop_style"] = "random"
            config["crop_aspect"] = "random"
            config["crop_aspect_buckets"] = [0.75, 1.5]
            random_crop = metadata._geometry_recipe(config)

            with (
                patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("used live config")),
                patch.object(
                    StateTracker, "get_resolution_by_aspect", side_effect=AssertionError("used global aspect memo")
                ),
                patch.object(
                    StateTracker, "set_resolution_by_aspect", side_effect=AssertionError("wrote global aspect memo")
                ),
                patch.object(StateTracker, "get_webhook_handler", side_effect=AssertionError("geometry webhook")),
            ):
                for recipe, target in (
                    (aligned_64, (768, 1408)),
                    (aligned_8, (744, 1352)),
                    (aligned_64, (768, 1408)),
                ):
                    bucket, record = metadata._prepare_metadata("sample", {"original_size": (550, 1000)}, recipe)
                    self.assertEqual(bucket, f"{target[0]}x{target[1]}")
                    self.assertEqual(record["target_size"], target)
                    self.assertTrue(record["prepared_geometry"])

                def scan(refs):
                    return {
                        ref: metadata._prepare_metadata(ref, {"original_size": (1100, 2000)}, random_crop) for ref in refs
                    }

                first = scan(("a", "b", "c"))
                self.assertEqual(first, scan(("c", "b", "a")))

            _, record = first["a"]
            image = Image.new("RGB", (1100, 2000))
            sample = TrainingSample(image, self.backend.id, image_metadata=record)
            with patch.object(sample, "calculate_target_size", side_effect=AssertionError("recomputed prepared geometry")):
                self.assertEqual(sample.prepare().image.size, record["target_size"])

    def test_instance_prompt_eligibility_does_not_require_source_captions(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        self.rows[-1]["captions"] = None
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        source = WebshartDataBackend(
            accelerator, "instance-only", str(self.source), format="parquet", cache_dir=str(self.root / "instance-view")
        )
        args = SimpleNamespace(seed=12, disable_bucket_pruning=True)
        config = {
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "webshart": {"sampling_policy": "global"},
            "instance_prompt": "shared prompt",
        }
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "instance-buckets"),
                metadata_file=str(self.root / "instance-geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            with_captions = metadata.describe()
            self.assertEqual(with_captions.accepted_samples, 1)

            config["only_instance_prompt"] = True
            with patch.object(source, "caption_fields", side_effect=AssertionError("projected source captions")):
                metadata.compute_aspect_ratio_bucket_indices()
            instance_only = metadata.describe()
            self.assertEqual(instance_only.accepted_samples, 2)
            self.assertNotEqual(instance_only.view_revision, with_captions.view_revision)
            dataset = DatasetRuntime(source.id, source, metadata, {**config, **DatasetRuntime.caption_config(config, args)})
            refs, count = dataset.read_page(limit=2)
            self.assertEqual(count, 2)
            with patch.object(source, "read_caption_batch", side_effect=AssertionError("read source captions")):
                self.assertEqual(dataset.resolve_captions(refs), ["shared prompt", "shared prompt"])

    def test_native_publication_scans_each_canonical_group_separately(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "second.parquet", row_group_size=2)
        source = WebshartDataBackend(None, "groups", str(self.source), format="parquet", cache_dir=str(self.root / "groups"))
        source.prepared_partition_shards = 1
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            with patch.object(metadata, "_scan_rows", wraps=metadata._scan_rows) as scan:
                progress = []
                metadata.compute_aspect_ratio_bucket_indices(progress_callback=lambda *state: progress.append(state))
            self.assertEqual([call.kwargs["shard_indices"] for call in scan.call_args_list], [(0,), (1,)])
            self.assertEqual(progress, [(1, 2), (2, 2)])
            prepared, total = metadata.read_page(limit=10)
            self.assertEqual(total, 4)

            recipe = metadata._geometry_recipe(config)
            separate = {}
            for shard in (1, 0):
                records = metadata._scan_rows((), {"total_processed": 0}, True, None, recipe=recipe, shard_indices=(shard,))
                for bucket, ref, record in records:
                    self.assertEqual(bucket, "16x8")
                    self.assertEqual(tuple(metadata.get_metadata_by_filepath(ref)["target_size"]), record["target_size"])
                    separate[ref] = record
            self.assertEqual(set(separate), set(prepared))

    def _publish_capture_manifest(self):
        with (self.source / "files.jsonl").open("wb") as stream:
            manifest_digest = sha256()
            for shard in sorted(self.source.glob("*.parquet")):
                entry = {
                    "key": shard.name,
                    "num_rows": pq.read_metadata(shard).num_rows,
                    "size_bytes": shard.stat().st_size,
                    "sha256": sha256(shard.read_bytes()).hexdigest(),
                }
                line = json.dumps(entry).encode() + b"\n"
                manifest_digest.update(line)
                stream.write(line)
        (self.source / "capture.json").write_text(
            json.dumps(
                {
                    "format": "parquet-image-capture",
                    "version": 1,
                    "files_manifest": "files.jsonl",
                    "manifest_sha256": manifest_digest.hexdigest(),
                }
            )
        )

    def test_conditioning_and_embedding_requests_share_native_target_membership(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime, TextRequestBuilder
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(
            model_family="sdxl",
            disable_bucket_pruning=False,
            skip_file_discovery="",
            max_train_steps=1,
            allow_dataset_oversubscription=False,
            gradient_accumulation_steps=1,
            eval_dataset_id=None,
        )
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, is_local_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "conditioning-buckets"),
                metadata_file=str(self.root / "conditioning-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args = args
            factory.accelerator = accelerator
            config["conditioning_type"] = "reference_strict"
            with (
                patch.object(StateTracker, "get_dataset", side_effect=AssertionError("legacy source clone")),
                patch.object(factory, "_handle_config_versioning"),
            ):
                factory._handle_bucket_operations(
                    {**config, "id": self.backend.id},
                    {"id": self.backend.id, "config": config, "metadata_backend": metadata},
                    "reference_strict",
                )
            (accepted, excluded), count = metadata.read_page(limit=2)
            self.assertEqual(count, 2)
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            self.assertEqual(runtime.conditioning_ref(excluded), excluded)
            ref_from_reordered_source = self.backend.parse_sample_id(excluded)
            reordered = self.backend.sample_id(12, ref_from_reordered_source.sample_idx, ref_from_reordered_source.filename)
            self.assertEqual(runtime.conditioning_ref(reordered), excluded)
            self.assertIsNone(runtime.conditioning_ref(self.backend.sample_id(12, 0, "absent.parquet/000000000000.jpg")))

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                changes.add(excluded, next(iter(metadata.published_bucket_counts())), "decode")
                publication = self.backend._cache_path(metadata.cache_file)
                _publish(publication, metadata.describe().view_revision, db)
            metadata.reload_cache()
            self.assertEqual(runtime.conditioning_ref(accepted), accepted)
            self.assertIsNone(runtime.conditioning_ref(excluded))
            self.assertTrue(self.backend.exists(excluded))

            with patch.object(self.backend, "read_image", side_effect=AssertionError("excluded payload read")):
                self.assertIsNone(runtime.read_conditioning_sample(excluded))

            model = SimpleNamespace(text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION)
            source = DatasetRuntime("source", self.backend, metadata, config)
            read_metadata = Mock(side_effect=AssertionError("unexpected geometry read"))
            builder = TextRequestBuilder(model, source, conditioning_datasets=[runtime])
            records = list(builder.records([("kept", accepted), ("other", excluded)], read_metadata=read_metadata))
            self.assertEqual(records[0]["metadata"]["image_paths"], [accepted])
            self.assertNotIn("image_paths", records[1]["metadata"])

            other = SimpleNamespace(id="other", config={}, conditioning_ref=lambda _: None)
            builder = TextRequestBuilder(
                model,
                source,
                conditioning_datasets=[runtime, other],
                args=SimpleNamespace(conditioning_multidataset_sampling="random"),
            )
            records = list(builder.records([("missing every reference", excluded)], read_metadata=read_metadata))
            self.assertEqual([record["prompt"] for record in records], ["missing every reference"])
            self.assertNotIn("image_paths", records[0]["metadata"])

    def test_generated_conditioning_reuses_native_prepared_geometry_without_cloning(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.data_generation import DataGenerator
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.training.state_tracker import StateTracker

        native = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        root = self.root / "generated"
        derived = {
            **native,
            "auto_generated": True,
            "dataset_type": "conditioning",
            "metadata_backend": "discovery",
            "instance_data_dir": str(root),
            "conditioning_config": {"type": "sdr"},
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False, train_batch_size=1)
        accelerator = SimpleNamespace(
            num_processes=1, process_index=0, is_main_process=True, device="cpu", wait_for_everyone=lambda: None
        )
        with (
            patch.object(
                StateTracker, "get_data_backend_config", side_effect=lambda id: native if id == self.backend.id else derived
            ),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "derived-source-buckets"),
                metadata_file=str(self.root / "derived-source-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            source = DatasetRuntime(self.backend.id, self.backend, metadata, native)
            store = LocalDataBackend(accelerator, "derived")
            provider = create_metadata_provider(
                store.id,
                derived,
                store,
                accelerator,
                args,
                instance_data_dir=str(root),
                cache_root=str(root),
                aligned_source=source,
            )
            target = DatasetRuntime(store.id, store, provider, derived)
            refs, count = source.read_page(limit=2)
            self.assertEqual(count, 2)
            with (
                patch.object(self.backend, "list_files", side_effect=AssertionError("source list materialized")),
                patch.object(store, "list_files", side_effect=AssertionError("target list materialized")),
            ):
                DataGenerator(source, target, accelerator, num_workers=0).generate_dataset()
                with patch.object(self.backend, "read_image_batch", side_effect=AssertionError("warm source payload read")):
                    DataGenerator(source, target, accelerator, num_workers=0).generate_dataset()

            targets, target_count = target.read_page(limit=2)
            self.assertEqual(target_count, 2)
            self.assertEqual(targets, [target.conditioning_ref(ref) for ref in refs])
            self.assertTrue(all(Path(path).is_file() for path in targets))
            records = target.read_samples(targets)
            self.assertEqual([record["training_sample_path"] for record in records], refs)
            self.assertEqual(
                [record["target_size"] for record in records],
                [record["target_size"] for record in source.read_samples(refs)],
            )
            self.assertEqual(provider.aspect_ratio_bucket_indices, {})
            self.assertEqual(metadata.aspect_ratio_bucket_indices, {})
            self.assertEqual(provider.image_metadata, {})

            bucket = next(iter(metadata.published_bucket_counts()))
            with provider.collect_exclusions() as changes:
                changes.add(targets[0], bucket, "nsfw")
            self.addCleanup(provider.close)
            self.assertEqual(target.read_page(limit=2), ([targets[1]], 1))
            self.assertEqual(list(target.iter_preparation_batches(batch_size=2)), [(bucket, (targets[1],))])
            self.assertIsNone(target.conditioning_ref(refs[0]))
            self.assertEqual(source.read_page(limit=2), (refs, 2))
            self.assertTrue(Path(targets[0]).exists())
            provider.reload_cache()
            self.assertEqual(target.describe().accepted_samples, 1)
            with (
                patch.object(self.backend, "read_image_batch", side_effect=AssertionError("excluded source read")),
                patch.object(store, "exists", wraps=store.exists) as exists,
            ):
                DataGenerator(source, target, accelerator, num_workers=0).generate_dataset()
            exists.assert_called_once_with(targets[1])

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                changes.add(refs[0], bucket, "decode")
                _publish(self.backend._cache_path(metadata.cache_file), source.describe().view_revision, db)
            metadata.reload_cache()
            with self.assertRaisesRegex(ValueError, "Source membership changed"):
                target.read_page(limit=2)

    def test_configured_local_conditioning_uses_pinned_parquet_geometry(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        args = {"train_batch_size": 1, "disable_bucket_pruning": False, "aspect_bucket_alignment": 8}
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        native = {
            "dataset_type": "image",
            "metadata_backend": "webshart",
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
            "conditioning_data": ["reference"],
        }
        with patch.object(StateTracker, "get_model", return_value=None):
            geometry = create_metadata_provider(
                self.backend.id,
                native,
                self.backend,
                accelerator,
                args,
                instance_data_dir=self.backend.cache_dir,
                cache_root=str(self.root),
            )
            self.addCleanup(geometry.close)
            geometry.compute_aspect_ratio_bucket_indices()
        source = DatasetRuntime(self.backend.id, self.backend, geometry, native)
        local_root = self.root / "paired"
        local_files = [local_root / "part.parquet" / f"{index:012d}.png" for index in (0, 2)]
        local_files[0].parent.mkdir(parents=True)
        for path in local_files:
            path.write_bytes(self.image)

        factory = object.__new__(FactoryRegistry)
        factory.accelerator = accelerator
        factory.args = args
        factory.model = None
        store = LocalDataBackend(accelerator, "reference")
        config = {
            "id": store.id,
            "type": "local",
            "instance_data_dir": str(local_root),
            "dataset_type": "conditioning",
            "conditioning_type": "reference_strict",
            "conditioning_config": {"type": "i2v_first_frame"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        initialized = {"id": store.id, "data_backend": store, "instance_data_dir": str(local_root), "config": config}
        other = Mock(id="other", config=native, uses_native_schedule=True, data_backend=self.backend)
        other.describe.return_value = replace(source.describe(), view_revision="other-view")
        for linked in ({source.id: source, other.id: other}, {other.id: other, source.id: source}):
            with (
                patch.object(StateTracker, "get_datasets", return_value=linked),
                self.assertRaisesRegex(ValueError, "cannot share one prepared view"),
            ):
                factory._configure_metadata_backend(config, initialized.copy())
        with (
            patch.object(StateTracker, "get_datasets", return_value={source.id: source}),
            patch.object(StateTracker, "set_data_backend_config"),
            patch.object(factory, "_handle_config_versioning"),
            patch.object(store, "list_files", side_effect=AssertionError("unexpected conditioning inventory")),
        ):
            factory._configure_metadata_backend(config, initialized)
            factory._handle_bucket_operations(config, initialized, config["conditioning_type"])
        provider = initialized["metadata_backend"]
        self.addCleanup(provider.close)
        target = DatasetRuntime(store.id, store, provider, config)
        source_refs, _ = source.read_page(limit=2)
        refs = [str(path) for path in local_files]
        self.assertEqual(target.read_page(limit=2), (refs, 2))
        self.assertEqual([target.conditioning_ref(ref) for ref in source_refs], refs)
        with patch.object(geometry, "get_view_metadata", wraps=geometry.get_view_metadata) as scoped_read:
            records = target.read_samples([refs[1], refs[0], refs[1]], fields=("image", "metadata"))
        scoped_read.assert_called_once_with([source_refs[1], source_refs[0], source_refs[1]], source.published_read_view())
        self.assertEqual(
            [record["training_sample_path"] for record in records], [source_refs[1], source_refs[0], source_refs[1]]
        )
        bucket = next(iter(geometry.published_bucket_counts()))
        with provider.collect_exclusions() as changes:
            changes.add(refs[0], bucket, "decode")
        self.assertEqual(target.read_page(limit=2), ([refs[1]], 1))
        self.assertEqual(source.read_page(limit=2), (source_refs, 2))

    def test_capture_publication_reuses_only_unchanged_groups(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.metadata.backends.webshart_geometry import snapshot_path
        from simpletuner.helpers.training.state_tracker import StateTracker

        second = self.source / "second.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows), second, row_group_size=2)

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)

        def build(*, cap=None, refresh=False, unopened=(), batch_size=1, repeats=0, maximum_aspect_ratio=None):
            source = WebshartDataBackend(
                None,
                "capture",
                str(self.source),
                manifest="capture.json",
                format="parquet",
                cache_dir=str(self.root / "cache"),
            )
            source.prepared_partition_shards = 1
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=batch_size,
                resolution=config["resolution"],
                resolution_type="pixel",
                minimum_image_size=1,
                maximum_aspect_ratio=maximum_aspect_ratio,
                max_num_samples=cap,
                repeats=repeats,
            )
            self.addCleanup(metadata.close)
            original_open = source.dataset._file

            def open_shard(shard, **kwargs):
                self.assertNotIn(shard, unopened)
                return original_open(shard, **kwargs)

            with (
                patch.object(metadata, "_scan_rows", wraps=metadata._scan_rows) as scan,
                patch.object(source.dataset, "_file", side_effect=open_shard),
            ):
                metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=refresh)
            return metadata, [call.kwargs["shard_indices"] for call in scan.call_args_list]

        def first_partition(metadata):
            path = metadata.data_backend._cache_path(metadata.metadata_file)
            with closing(sqlite3.connect(snapshot_path(path, metadata._published["geometry_sha256"]))) as db:
                (digest,) = db.execute("SELECT digest FROM partitions WHERE id=0").fetchone()
            return snapshot_path(path, digest)

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            self._publish_capture_manifest()
            before, scanned = build()
            self.assertCountEqual(scanned, [(0,), (1,)])
            original_page = before.read_page(limit=10)
            original_revision = before.describe().view_revision
            self.assertEqual(original_page[1], 4)

            pruned, scanned = build(batch_size=5, unopened=(0, 1))
            self.assertEqual(scanned, [])
            self.assertEqual(pruned.read_page(limit=10), ([], 0))
            self.assertNotEqual(pruned.describe().view_revision, original_revision)
            self.assertTrue(first_partition(before).samefile(first_partition(pruned)))

            repeated, scanned = build(batch_size=5, repeats=1, unopened=(0, 1))
            self.assertEqual(scanned, [])
            self.assertEqual(repeated.read_page(limit=10), original_page)
            self.assertEqual(repeated.describe().view_revision, original_revision)

            args.disable_bucket_pruning = True
            unpruned, scanned = build(batch_size=5, unopened=(0, 1))
            self.assertEqual(scanned, [])
            self.assertEqual(unpruned.read_page(limit=10), original_page)
            args.disable_bucket_pruning = False

            aspect_filtered, scanned = build(maximum_aspect_ratio=1.1, unopened=(0, 1))
            self.assertEqual(scanned, [])
            self.assertEqual(aspect_filtered.read_page(limit=10), ([], 0))

            restored, scanned = build(unopened=(0, 1))
            self.assertEqual(scanned, [])
            self.assertEqual(restored.describe().view_revision, original_revision)

            updated_rows = [dict(row) for row in self.rows]
            updated_rows[-1]["captions"] = "   "
            pq.write_table(pa.Table.from_pylist(updated_rows), second, row_group_size=2)
            self._publish_capture_manifest()
            after, scanned = build(unopened=(0,))
            self.assertEqual(scanned, [(1,)])
            refs, total = after.read_page(limit=10)
            self.assertEqual(total, 3)
            self.assertEqual(after.filtering_statistics["total_processed"], 3)
            self.assertEqual(after.filtering_statistics["skipped"]["caption_missing"], 1)
            skipped = after.data_backend.sample_id(1, 2, "second.parquet/000000000002.jpg")
            self.assertIsNone(after.data_backend.get_caption(skipped))
            self.assertNotIn(skipped, refs)
            self.assertEqual(refs[:2], before.read_page(limit=10)[0][:2])
            self.assertEqual(after.get_metadata_by_filepath(refs[0]), before.get_metadata_by_filepath(refs[0]))
            self.assertNotEqual(after._published["source_revision"], before._published["source_revision"])
            self.assertEqual(after._published["physical_rows"], 6)
            self.assertTrue(first_partition(before).samefile(first_partition(after)))

            clean, scanned = build(refresh=True)
            self.assertCountEqual(scanned, [(0,), (1,)])
            self.assertEqual(clean._published["geometry_sha256"], after._published["geometry_sha256"])
            self.assertEqual(clean.filtering_statistics, after.filtering_statistics)
            self.assertTrue(first_partition(before).samefile(first_partition(clean)))

            config["resolution"] = 16
            resized, scanned = build()
            self.assertCountEqual(scanned, [(0,), (1,)])
            self.assertNotEqual(resized._published["bucket_counts"], before._published["bucket_counts"])

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                bucket = next(iter(resized.published_bucket_counts()))
                changes.add(refs[-1], bucket, "decode")
                publication = resized.data_backend._cache_path(resized.cache_file)
                _publish(publication, resized.describe().view_revision, db)

            config["resolution"] = 8
            filtered, scanned = build(unopened=(0,))
            self.assertEqual(scanned, [(1,)])
            self.assertEqual(filtered.read_page(limit=10), (refs[:2], 2))
            self.assertEqual(filtered._published["exclusion_counts"], {"decode": 1})

            capped, scanned = build(cap=1)
            self.assertEqual(scanned, [(0,)])
            self.assertEqual(capped.read_page(limit=10)[1], 1)

    def test_concurrent_views_share_cold_capture_group(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        self._publish_capture_manifest()
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        ready = Barrier(2, timeout=10)

        def make_metadata(view):
            source = WebshartDataBackend(
                None,
                f"concurrent-{view}",
                str(self.source),
                manifest="capture.json",
                format="parquet",
                cache_dir=str(self.root / "shared-cache"),
            )
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / f"{view}-buckets"),
                metadata_file=str(self.root / f"{view}-geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            return metadata

        def align_lookup(lookup):
            def resolve(*args):
                path = lookup(*args)
                self.assertIsNotNone(path)
                ready.wait()
                return path

            return resolve

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            first, second = (make_metadata(view) for view in ("first", "second"))
            with (
                patch.object(first, "_partition_cache_path", side_effect=align_lookup(first._partition_cache_path)),
                patch.object(second, "_partition_cache_path", side_effect=align_lookup(second._partition_cache_path)),
                patch.object(first, "_scan_rows", wraps=first._scan_rows) as first_scan,
                patch.object(second, "_scan_rows", wraps=second._scan_rows) as second_scan,
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                tasks = [pool.submit(view.compute_aspect_ratio_bucket_indices) for view in (first, second)]
                for task in tasks:
                    task.result(timeout=15)

            self.assertEqual(first_scan.call_count + second_scan.call_count, 1)
            self.assertEqual(first.read_page(limit=8), second.read_page(limit=8))
            self.assertEqual(first.describe().view_revision, second.describe().view_revision)

    def test_parallel_cold_preparation_resumes_after_a_worker_failure(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        for index in range(3):
            shard = self.source / f"additional-{index}.parquet"
            pq.write_table(pa.Table.from_pylist(self.rows), shard, row_group_size=2)
        self._publish_capture_manifest()
        source = WebshartDataBackend(
            None,
            "parallel",
            str(self.source),
            manifest="capture.json",
            format="parquet",
            parallel_downloads=2,
            cache_dir=str(self.root / "parallel"),
        )
        if not getattr(source.dataset, "supports_parallel_metadata_reads", False):
            self.skipTest("Requires a Webshart reader with synchronized capture catalogue access")
        source.prepared_partition_shards = 1
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "parallel" / "buckets"),
                metadata_file=str(self.root / "parallel" / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            first_started = Event()
            second_completed = Event()
            original_scan = metadata._scan_rows

            def fail_first_group(*args, shard_indices, **kwargs):
                if shard_indices == (0,):
                    first_started.set()
                    self.assertTrue(second_completed.wait(15), "second group did not run while the first was blocked")
                    raise OSError("first group could not read source")
                self.assertEqual(shard_indices, (1,), "scheduled past the worker window")
                self.assertTrue(first_started.wait(15))
                yield from original_scan(*args, shard_indices=shard_indices, **kwargs)
                second_completed.set()

            with (
                patch.object(metadata, "_scan_rows", side_effect=fail_first_group),
                self.assertRaisesRegex(OSError, "first group could not read source"),
            ):
                metadata.compute_aspect_ratio_bucket_indices()
            self.assertFalse(source.exists(metadata.cache_file))

            owner = get_ident()
            progress = []

            def report(current, total):
                self.assertEqual(get_ident(), owner)
                progress.append((current, total))

            with patch.object(metadata, "_scan_rows", wraps=original_scan) as scan:
                metadata.compute_aspect_ratio_bucket_indices(progress_callback=report)
            self.assertCountEqual([call.kwargs["shard_indices"] for call in scan.call_args_list], [(0,), (2,), (3,)])
            self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
            refs, total = metadata.read_page(limit=10)
            self.assertEqual(total, 8)
            self.assertEqual([source.parse_sample_id(ref).shard_idx for ref in refs], [0, 0, 1, 1, 2, 2, 3, 3])
            result = (replace(metadata.describe(), view_revision=None), metadata.filtering_statistics)
            geometry_digest = metadata._published["geometry_sha256"]

            source.parallel_downloads = 1
            metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual((replace(metadata.describe(), view_revision=None), metadata.filtering_statistics), result)
            self.assertEqual(metadata._published["geometry_sha256"], geometry_digest)
            self.assertEqual(metadata.read_page(limit=10), (refs, total))

            source.parallel_downloads = 2
            publication = source._cache_path(metadata.cache_file)
            inode = publication.stat().st_ino

            def cancel(current, total):
                self.assertEqual(get_ident(), owner)
                raise RuntimeError("scan cancelled")

            with self.assertRaisesRegex(RuntimeError, "scan cancelled"):
                metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True, progress_callback=cancel)
            self.assertEqual(publication.stat().st_ino, inode)
            self.assertEqual(metadata.read_page(limit=10), (refs, total))

    def test_caption_native_and_local_schedules_resume_one_prefetched_mixer(self):
        from accelerate import PartialState
        from torch.utils.data import DataLoader

        from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
        from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.data_backend.runtime.batch_fetcher import BatchFetcher
        from simpletuner.helpers.data_backend.runtime.dataloader_iterator import DatasetMixer
        from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.caption_collate import collate_caption_batch
        from simpletuner.helpers.training.state_tracker import StateTracker
        from tests.helpers.data_backend.test_caption_pipeline import DummyCaptionMetadataBackend

        accelerator = PartialState()
        config = {
            "webshart": {"sampling_policy": "shard_local"},
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "slider_strength": -1,
            "crop": True,
            "crop_style": "random",
            "crop_aspect": "square",
            "vae_cache_clear_each_epoch": True,
        }
        args = SimpleNamespace(
            model_family="sdxl",
            disable_bucket_pruning=False,
            gradient_accumulation_steps=1,
            max_train_steps=0,
            seed=9,
            print_sampler_statistics=False,
        )

        class SampleBatches:
            def __init__(self, sampler):
                self.sampler = sampler

            def __iter__(self):
                for (record,) in self.sampler:
                    yield {
                        key: record[key]
                        for key in (
                            "data_backend_id",
                            "image_path",
                            "dataset_continuation",
                            "caption_dropout_seed",
                            "crop_coordinates",
                        )
                        if key in record
                    }

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "epoch", 1),
            patch.object(StateTracker, "repeats", {}),
            patch.object(StateTracker, "exhausted_backends", []),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "mixed-buckets"),
                metadata_file=str(self.root / "mixed-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            captions = DummyCaptionMetadataBackend(count=1)

            def make_runtimes():
                images = DatasetRuntime(self.backend.id, self.backend, metadata, config)
                images.sampler = MultiAspectSampler(images, model=None, accelerator=accelerator, batch_size=1)
                self.addCleanup(images.sampler.close)
                images.training_dataloader = SampleBatches(images.sampler)
                text = DatasetRuntime("captions", None, captions, {"dataset_type": "caption", "slider_strength": 1})
                text.sampler = CaptionSampler("captions", captions, accelerator, 1, shuffle=False, track_consumed=True)
                text.training_dataloader = DataLoader(
                    CaptionDataset("captions", captions),
                    batch_size=1,
                    sampler=text.sampler,
                    collate_fn=collate_caption_batch,
                )

                local_storage = LocalDataBackend(accelerator, "local")
                local_metadata = DiscoveryMetadataBackend(
                    id="local",
                    instance_data_dir=str(self.root),
                    cache_file=str(self.root / "local-buckets"),
                    metadata_file=str(self.root / "local-geometry"),
                    data_backend=local_storage,
                    accelerator=accelerator,
                    batch_size=1,
                    resolution=8,
                    resolution_type="pixel",
                )
                ref = str(self.root / "local.jpg")
                local_metadata.aspect_ratio_bucket_indices = {"1.0": [ref]}
                local_metadata.image_metadata = {ref: {"original_size": (8, 8), "crop_coordinates": (0, 0)}}
                local = DatasetRuntime("local", local_storage, local_metadata, {"repeats": 1})
                local.sampler = MultiAspectSampler(
                    local, model=None, accelerator=accelerator, batch_size=1, track_consumed=True
                )
                self.addCleanup(local.sampler.close)
                local.training_dataloader = SampleBatches(local.sampler)
                return {dataset.id: dataset for dataset in (images, text, local)}

            def fetch(runtimes, saved=None):
                backends = dict(runtimes)
                for dataset in backends.values():
                    dataset.start_epoch(2)
                    self.assertTrue(dataset.supports_consumed_checkpoint)
                mixer = DatasetMixer(backends, epoch=2, step=0, saved=saved)
                fetcher = BatchFetcher(step=mixer.step, max_size=8, datasets=backends, mixer=mixer)
                fetcher.fetch_responses()
                return fetcher

            original = make_runtimes()
            fetcher = fetch(original)
            consumed = [fetcher.next_response(step) for step in (1, 2, 3)]
            self.assertEqual([batch["data_backend_id"] for batch in consumed], ["captions", self.backend.id, "local"])
            for batch in consumed:
                original[batch["data_backend_id"]].acknowledge_batch(batch["dataset_continuation"])
            checkpoint = str(self.root / "mixed-train-state.json")
            for dataset in original.values():
                dataset.save_checkpoint(checkpoint)
            expected = [fetcher.next_response(step) for step in (4, 5, 6)]
            self.assertCountEqual([batch["data_backend_id"] for batch in expected[:-1]], [self.backend.id, "local"])
            self.assertIs(expected[-1], False)

            resumed = make_runtimes()
            for dataset in resumed.values():
                dataset.restore_checkpoint(checkpoint)
            replay = fetch(resumed, saved=consumed[-1]["dataset_mixer_state"])
            self.assertEqual([replay.next_response(step) for step in (4, 5, 6)], expected)

    def test_native_scan_partition_sampler_and_consumed_checkpoint(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.preparation import DatasetReadView
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = dict(
            id=self.backend.id,
            type="webshart",
            webshart={"sampling_policy": "shard_local"},
            caption_strategy="webshart",
            resolution=64,
            resolution_type="pixel",
            crop=False,
        )
        args = SimpleNamespace(
            seed=32,
            model_family="sdxl",
            disable_bucket_pruning=False,
            gradient_accumulation_steps=1,
            max_train_steps=0,
            allow_dataset_oversubscription=False,
        )
        accel = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(self.backend.source_files, "publish", side_effect=AssertionError("corpus inventory")),
            patch.object(self.backend, "read", side_effect=AssertionError("whole geometry or membership read")),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accel,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            metadata.compute_aspect_ratio_bucket_indices()
            with (
                patch.object(metadata, "max_num_samples", 1),
                patch.object(metadata, "dataset_filter", DatasetFilter({"path_include": "000000000002"})),
                patch.object(metadata, "_all_shard_indices", return_value=range(2)),
                patch.object(
                    self.backend.loader, "iter_samples_in_shard", wraps=self.backend.loader.iter_samples_in_shard
                ) as project,
            ):
                scanned = list(metadata._scan_rows(set(), {"total_processed": 0}, True, None))
            self.assertEqual([self.backend.parse_sample_id(ref).sample_idx for _, ref, _ in scanned], [2])
            self.assertEqual([call.args[0] for call in project.call_args_list], [0])
            self.assertEqual(metadata.describe().accepted_samples, 2)
            with patch.object(metadata, "_scan_rows", side_effect=AssertionError("repeated scan")):
                metadata.compute_aspect_ratio_bucket_indices()
            metadata.split_buckets_between_processes()
            self.assertEqual(metadata.image_metadata, {})
            self.assertEqual(metadata.aspect_ratio_bucket_indices, {})
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            partitions = [
                [
                    ref
                    for _, batch in runtime.iter_preparation_batches(worker=worker, workers=3, batch_size=1)
                    for ref in batch
                ]
                for worker in range(3)
            ]
            self.assertEqual([len(p) for p in partitions], [1, 1, 0])
            self.assertEqual(len(set(partitions[0] + partitions[1])), 2)
            ref = partitions[0][0]
            other = partitions[1][0]
            from simpletuner.helpers.metadata.backends import webshart_geometry

            statements = []
            open_snapshot = webshart_geometry.read_snapshot

            def trace_snapshot(*args):
                db = open_snapshot(*args)
                db.set_trace_callback(statements.append)
                return db

            with patch.object(webshart_geometry, "read_snapshot", side_effect=trace_snapshot):
                rows = runtime.read_samples([other, ref, other], fields=("metadata",))
            self.assertEqual([row["image_path"] for row in rows], [other, ref, other])
            self.assertIsNot(rows[0]["webshart"], rows[2]["webshart"])
            self.assertEqual(sum("FROM geometry" in sql for sql in statements), 1)
            weights = self.root / "model" / "vae" / "weights.safetensors"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"first")
            encoder = SimpleNamespace(config=SimpleNamespace(vae_path=str(weights.parent.parent), revision="v1"))
            key = metadata.image_artifact_key(ref, "vae", encoder)
            with patch("simpletuner.helpers.metadata.backends.webshart.file_digest", side_effect=AssertionError("rehash")):
                self.assertEqual(metadata.image_artifact_key(ref, "vae", encoder), key)
            with patch.dict(metadata.dataset_config, vae_cache_clear_each_epoch=True):
                with patch.object(StateTracker, "epoch", 3):
                    self.assertEqual(metadata.image_artifact_key(ref, "vae", encoder), key)
            encoder.config.revision = "v2"
            self.assertNotEqual(metadata.image_artifact_key(ref, "vae", encoder), key)
            encoder.config.revision = "v1"
            weights.write_bytes(b"other")
            changed_weights = metadata.image_artifact_key(ref, "vae", encoder)
            self.assertNotEqual(changed_weights, key)
            with patch.dict(metadata.dataset_config, crop_style="corner"):
                self.assertNotEqual(metadata.image_artifact_key(ref, "vae", encoder), changed_weights)
            processor_file = self.root / "processor" / "preprocessor_config.json"
            processor_file.parent.mkdir()
            processor_file.write_text('{"size": 64}')
            encoder.config.image_encoder_pretrained_model_name_or_path = str(weights.parent)
            encoder.config.image_processor_pretrained_model_name_or_path = str(processor_file.parent)
            encoder.config.image_encoder_subfolder = ["vision"]
            original = metadata.image_artifact_key(ref, "conditioning", encoder)
            processor_file.write_text('{"size": 128}')
            metadata._artifact_models.clear()
            self.assertNotEqual(metadata.image_artifact_key(ref, "conditioning", encoder), original)
            sampler = MultiAspectSampler(runtime, model=None, accelerator=accel, batch_size=1)
            runtime.sampler = sampler
            self.addCleanup(metadata.close)
            self.addCleanup(sampler.close)
            self.assertEqual(len(sampler), 2)
            from simpletuner.helpers.metadata.backends.webshart_geometry import prepared_geometry_path
            from simpletuner.helpers.metadata.backends.webshart_membership import IndexedMembership

            publication = self.backend._cache_path(metadata.cache_file)
            original_index = prepared_geometry_path(publication, metadata._published)
            original_stat = original_index.stat()
            self.assertNotIn("aspect_ratio_bucket_indices", json.loads(publication.read_text()))
            with patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("startup scanned membership")):
                sampler._open_training_cursor()
            first = next(iter(sampler))[0]
            second = next(iter(sampler))[0]
            self.assertEqual(first["dataset_read_view"], DatasetReadView(runtime.describe().view_revision, 1, args.seed))
            self.assertNotEqual(first["image_path"], second["image_path"])
            runtime.acknowledge_batch(first["dataset_continuation"])
            name = str(self.root / "checkpoint.json")
            validation_first = sampler.validation_cursor.next_refs(1)
            sampler.save_state(name)
            validation_second = sampler.validation_cursor.next_refs(1)
            self.assertNotEqual(validation_first, validation_second)
            sampler.load_states(name)
            self.assertEqual(sampler.validation_cursor.next_refs(1), validation_second)
            again = next(iter(sampler))[0]
            self.assertEqual(again, second)
            import random

            training_state = sampler.training_cursor.state_dict()
            validation_state = sampler.validation_cursor.state_dict()
            with (
                patch.object(runtime, "resolve_caption", return_value=["alpha", "beta", "gamma"]),
                patch.object(sampler, "debug_log"),
            ):
                random_state = random.getstate()
                (first_preview,) = sampler.retrieve_validation_set(1)
                self.assertEqual(random.getstate(), random_state)
                sampler.validation_cursor.load_state_dict(validation_state)
                (same_preview,) = sampler.retrieve_validation_set(1)
            self.assertEqual(first_preview.ref, same_preview.ref)
            self.assertEqual(first_preview.prompt, same_preview.prompt)
            self.assertEqual(first_preview.image.tobytes(), same_preview.image.tobytes())
            self.assertEqual(first_preview.view.revision, validation_state["view"])
            self.assertEqual(sampler.training_cursor.state_dict(), training_state)
            sampler.reset_evaluation()
            self.assertEqual(next(iter(sampler))[0], first)
            sampler.reset_evaluation()
            metadata.reload_cache()
            with metadata.open_preparation_view(worker=1, workers=2) as view:
                refs = list(view)
                self.assertEqual(len(refs), 1)
                self.assertIn("target_size", view.metadata.get_metadata_by_filepath(refs[0]))

            bucket = next(iter(metadata.published_bucket_counts()))
            all_refs, total = runtime.read_page(bucket=bucket)
            self.assertEqual(total, 2)
            spec = metadata.epoch_spec(1, 1)
            self.assertEqual(runtime.describe(spec).scheduled_occurrences, 2)
            old_publication = metadata._published
            with (
                patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("filtering scanned membership")),
                metadata.collect_exclusions() as ledger,
            ):
                ledger.add(all_refs[1], bucket, "nsfw")
            with patch.object(metadata, "_published", old_publication):
                with self.assertRaisesRegex(ValueError, "membership changed"):
                    len(sampler)
                with self.assertRaisesRegex(ValueError, "membership changed"), metadata.open_preparation_view():
                    self.fail("opened preparation with stale source pins")
            self.assertEqual(len(sampler), 1)
            self.assertEqual(prepared_geometry_path(publication, metadata._published), original_index)
            self.assertEqual(original_index.stat().st_mtime_ns, original_stat.st_mtime_ns)
            self.assertNotEqual(metadata.describe().view_revision, old_publication["view_revision"])
            with patch.object(
                IndexedMembership, "iter_refs", side_effect=AssertionError("startup scanned filtered membership")
            ):
                with closing(metadata.open_epoch(spec)) as cursor:
                    self.assertEqual(cursor.batches, 1)
            with self.assertRaisesRegex(ValueError, "accepted Webshart view"):
                sampler.validation_cursor.load_state_dict(validation_state)
            sampler.validation_cursor.close()
            self.assertEqual(runtime.read_page(), ([all_refs[0]], 1))
            metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(runtime.read_page(), ([all_refs[0]], 1))
            with self.assertRaisesRegex(ValueError, "unchanged"):
                sampler.load_states(name)

            replacement = self.root / "replacement.parquet"
            original = self.source / "part.parquet"
            rows = [{**self.rows[0], "captions": "other"}, *self.rows[1:]]
            pq.write_table(pa.Table.from_pylist(rows), replacement, row_group_size=2)
            self.assertEqual(replacement.stat().st_size, original.stat().st_size)
            replacement.replace(original)
            metadata.reload_cache()
            for policy in ("global", "shard_local"):
                with closing(metadata.open_epoch(replace(spec, policy=policy, sampling_order="shuffled"))) as cursor:
                    _, refs = cursor.next_batch()
                    with self.assertRaisesRegex(ValueError, "object changed since the accepted dataset view"):
                        self.backend.read_caption_batch(refs)

            pinned_view = runtime.published_read_view()

            def interrupted(*_):
                self.assertEqual(runtime.published_read_view(), pinned_view)
                with self.assertRaisesRegex(ValueError, "object changed since the accepted dataset view"):
                    self.backend.read_caption_batch(all_refs[:1])
                raise OSError("rescan interrupted")

            with self.assertRaisesRegex(OSError, "rescan interrupted"):
                metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True, progress_callback=interrupted)
            self.assertEqual(runtime.published_read_view(), pinned_view)
            self.assertEqual(runtime.read_page(), ([all_refs[0]], 1))

            metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertNotEqual(runtime.published_read_view(), pinned_view)
            self.assertEqual(runtime.read_page(), (all_refs, 2))
            self.assertEqual(self.backend.read_caption_batch(all_refs), ["other", "third"])
            self.assertFalse(metadata._published.get("exclusion_revision"))

    def test_native_refresh_reuses_exclusions_only_for_unchanged_shard_content(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend

        original = self.source / "part.parquet"
        shutil.copyfile(original, self.source / "second.parquet")
        backend = WebshartDataBackend(
            None, "parquet", str(self.source), format="parquet", cache_dir=str(self.root / "cache")
        )
        metadata = WebshartMetadataBackend(
            id=backend.id,
            instance_data_dir=backend.cache_dir,
            cache_file=str(self.root / "buckets"),
            metadata_file=str(self.root / "metadata"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=64,
            resolution_type="pixel",
            dataset_config={
                "caption_strategy": "webshart",
                "webshart": {"sampling_policy": "shard_local"},
                "resolution": 64,
                "resolution_type": "pixel",
                "crop": False,
            },
            runtime_args=SimpleNamespace(seed=123, disable_bucket_pruning=False),
        )
        self.addCleanup(metadata.close)
        metadata.compute_aspect_ratio_bucket_indices()
        refs, count = metadata.read_page()
        self.assertEqual(count, 4)
        rejected = {backend.shard_of(ref): ref for ref in refs if backend.parse_sample_id(ref).sample_idx == 0}
        with metadata.collect_exclusions() as ledger:
            for ref in rejected.values():
                ledger.add(ref, next(iter(metadata.published_bucket_counts())), "decode")

        replacement = self.root / "replacement.parquet"
        pq.write_table(
            pa.Table.from_pylist([{**self.rows[0], "captions": "other"}, *self.rows[1:]]), replacement, row_group_size=2
        )
        replacement.replace(original)
        metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
        visible, count = metadata.read_page()
        self.assertEqual(count, 3)
        self.assertEqual(visible, [ref for ref in refs if ref != rejected[1]])
        self.assertEqual(metadata._published["exclusion_counts"], {"decode": 1})

    def test_text_preparation_after_real_rank_split_still_covers_all_accepted_rows(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService

        config = dict(id=self.backend.id, caption_strategy="webshart", resolution=64, resolution_type="pixel", crop=False)
        args = SimpleNamespace(seed=123, disable_bucket_pruning=False, allow_dataset_oversubscription=False)
        accelerator = SimpleNamespace(num_processes=2, process_index=1, is_main_process=False)
        model = SimpleNamespace(text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(self.backend.loader, "load_sample", side_effect=AssertionError("unexpected payload read")),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            if self.backend.supports_caption_projection:
                with patch.object(self.backend.dataset, "_metadata", side_effect=AssertionError("entire shard loaded")):
                    metadata.compute_aspect_ratio_bucket_indices()
                self.assertEqual(self.backend._shard_metadata_cache, {})
            else:
                metadata.compute_aspect_ratio_bucket_indices()
            metadata.split_buckets_between_processes(apply_padding=True)
            self.assertEqual(sum(map(len, metadata.aspect_ratio_bucket_indices.values())), 1)
            with patch.object(metadata, "read_cache", side_effect=AssertionError("read rank-local membership")):
                records = list(
                    CacheJobService._iter_prompt_records(
                        model, DatasetRuntime(self.backend.id, self.backend, metadata, config)
                    )
                )
            self.assertEqual([r["prompt"] for r in records], ["first", "third"])

            from simpletuner.helpers.data_backend.factory import FactoryRegistry

            args.output_dir = str(self.root)
            args.caption_dropout_probability = 0
            accelerator.device = "cpu"
            model.uses_text_embeddings_cache = lambda: True
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args, factory.accelerator, factory.model = args, accelerator, model
            factory.text_encoders = None
            factory._deferred_text_embed_datasets = []
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            with (
                patch.object(metadata, "read_cache", side_effect=AssertionError("rank-local inventory")),
                patch(
                    "simpletuner.helpers.metadata.backends.webshart.open_geometry", side_effect=AssertionError("geometry")
                ),
            ):
                cursor = runtime.open_validation()
                try:
                    selected = cursor.next_refs(1)
                    self.assertEqual(self.backend.parse_sample_id(selected[0]).sample_idx, 0)
                    self.assertEqual([self.backend.parse_sample_id(ref).sample_idx for ref in cursor.next_refs(5)], [2, 0])
                finally:
                    cursor.close()
            prepared = []
            cache = SimpleNamespace(
                text_cache_ondemand=False,
                read_batch_size=8,
                text_encoder_batch_size=2,
                precompute_embeddings=lambda stream, **kwargs: prepared.extend(stream),
            )
            runtime.text_embed_cache = cache
            bundle = dict(
                id=runtime.id,
                config=dict(config),
                instance_data_dir=None,
                data_backend=self.backend,
                metadata_backend=metadata,
                dataset_runtime=runtime,
                text_embed_cache=cache,
            )
            with (
                patch.object(self.backend.source_files, "read", side_effect=AssertionError("source inventory")),
                patch.object(StateTracker, "set_data_backend_config"),
            ):
                for deferred in (False, True):
                    prepared.clear()
                    bundle["config"]["conditioning_data"] = ["reference"] if deferred else []
                    model.requires_text_embed_image_context = lambda: deferred
                    for rank in range(2):
                        accelerator.process_index = rank
                        factory._process_text_embeddings(config, bundle, None)
                        if deferred:
                            self.assertEqual(factory._deferred_text_embed_datasets, [runtime])
                            factory._process_deferred_text_embeddings()
                            self.assertEqual(prepared[-1]["metadata"].pop("dataset_bindings"), {runtime.id: runtime})
                        self.assertEqual(prepared[-1], records[rank])
                    self.assertEqual(prepared, records)

    def test_catalog_changes_reject_old_sample_refs_and_worker_state(self):
        old_path = self.backend.list_files()[0][2][0]
        representation = self.backend.get_instance_representation()
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "aaa.parquet", row_group_size=2)
        changed = WebshartDataBackend(
            None, "parquet", str(self.source), format="parquet", cache_dir=str(self.root / "cache")
        )
        self.assertNotEqual(changed.dataset.fingerprint, self.backend.dataset.fingerprint)
        with self.assertRaisesRegex(ValueError, "identity"):
            changed.read(old_path)
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            self.backend.from_instance_representation(representation)

    def test_jxl_image_bytes_decode(self):
        try:
            import pillow_jxl
        except ImportError:
            self.skipTest("pillow-jxl-plugin is optional")
        data = BytesIO()
        Image.new("RGB", (16, 8), "blue").save(data, format="JXL", lossless=True)
        self.rows[0]["image"]["bytes"] = data.getvalue()
        self.rows[0]["stored_format"] = "jxl"
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        backend = WebshartDataBackend(
            None, "jxl", str(self.source), format="parquet", cache_dir=str(self.root / "jxl-cache")
        )
        image = backend.read_image(backend.list_files()[0][2][0])
        self.assertEqual(image.size, (16, 8))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
