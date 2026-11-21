import json
import sqlite3
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simpletuner.helpers.metadata.backends.webshart_epoch import EpochSpec, WebshartEpoch
from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
from simpletuner.helpers.metadata.backends.webshart_geometry import snapshot_path, stage_geometry
from simpletuner.helpers.metadata.backends.webshart_membership import (
    IndexedMembership,
    MembershipPages,
    publish_indexed_membership,
    shard_runs_path,
    write_membership,
)


def publish_geometry(path, buckets, shard_of, *, shards_per_partition=None):
    geometry = path.with_name("geometry.json")
    with stage_geometry(geometry, shards_per_partition=shards_per_partition) as (writer, staged):
        ordered = sorted(
            ((bucket, ref) for bucket, refs in buckets.items() for ref in refs),
            key=lambda item: shard_of(item[1]),
        )
        if shards_per_partition is None:
            for bucket, ref in ordered:
                writer.add(bucket, ref, {"ref": ref}, shard=shard_of(ref))
        else:
            for partition, rows in groupby(ordered, key=lambda row: shard_of(row[1]) // shards_per_partition):
                with writer.stage_partition(partition) as (group_writer, part):
                    for bucket, ref in rows:
                        group_writer.add(bucket, ref, {"ref": ref}, shard=shard_of(ref))
                writer.adopt_partition(partition, part)
        counts = writer.finish()
        writer.db.commit()
        digest = writer.digest.hexdigest()
        destination = snapshot_path(geometry, digest)
        if shards_per_partition is not None:
            writer.publish(geometry)
        staged.replace(destination)
    publish_indexed_membership(
        path,
        {
            "view_revision": digest,
            "source_revision": "capture",
            "geometry_sha256": digest,
            "geometry_name": geometry.name,
            "geometry_format": "sqlite-v1",
            "geometry_partition_shards": shards_per_partition,
            "bucket_counts": counts,
        },
    )
    return destination


class WebshartEpochTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "buckets.json"
        self.buckets = {
            "wide": [f"{shard}/{row}" for shard in range(12) for row in range(4)],
            "square": [f"{shard}/{row}" for shard in range(12) for row in range(4, 12)],
        }
        write_membership(self.path, self.buckets, {"source_revision": "source-1"})
        self.spec = EpochSpec("data", "global", 17, 2, 2, 2, 0, 1, 0, False, (2, 1))

    @staticmethod
    def shard(ref):
        return int(ref.split("/")[0])

    def open(self, spec):
        return WebshartEpoch(self.path.open("rb"), spec, self.shard)

    @staticmethod
    def drain(cursor):
        return list(iter(lambda: cursor.next_batch(), None))

    def test_same_coverage_distinct_rank_assignment_and_shard_locality(self):
        reference = Counter(ref for members in self.buckets.values() for ref in members)
        mixed = {}
        for policy in ("shard_local", "global"):
            data = []
            for rank in range(2):
                with closing(self.open(replace(self.spec, policy=policy, dp_rank=rank))) as cursor:
                    data.append(self.drain(cursor))
                    self.assertEqual(len(data[-1]), cursor.batches)
            self.assertEqual([b for b, _ in data[0]], [b for b, _ in data[1]])
            self.assertEqual(Counter(ref for batches in data for _, refs in batches for ref in refs), reference)
            mixed[policy] = sum(len({self.shard(ref) for ref in left[1] + right[1]}) > 1 for left, right in zip(*data))
        self.assertEqual(mixed["shard_local"], 0)
        self.assertGreater(mixed["global"], 12)

    def test_sequential_order_is_physical_and_resume_rejects_a_different_order(self):
        spec = replace(self.spec, policy="shard_local", sampling_order="sequential")
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                if indexed:
                    publish_geometry(self.path, self.buckets, self.shard)
                with closing(self.open(spec)) as cursor:
                    bucket, refs = cursor.next_batch()
                    self.assertEqual(refs, self.buckets[bucket][: spec.batch_size])
                    checkpoint = cursor.state_dict()
                    remaining = self.drain(cursor)
                    all_batches = [(bucket, refs), *remaining]
                    shards = [self.shard(ref) for _, batch in all_batches for ref in batch]
                    self.assertEqual(shards, sorted(shards))
                with closing(self.open(spec)) as cursor:
                    cursor.load_state_dict(checkpoint)
                    self.assertEqual(self.drain(cursor), remaining)
                with closing(self.open(replace(spec, sampling_order="shuffled"))) as cursor:
                    self.assertNotIn("sampling_order", cursor.state_dict()["layout"])
                    with self.assertRaisesRegex(ValueError, "unchanged source"):
                        cursor.load_state_dict(checkpoint)

    def test_cursor_resume_and_index_reuse(self):
        for policy in ("global", "shard_local"):
            spec = replace(self.spec, policy=policy)
            with closing(self.open(spec)) as first:
                first.next_batch()
                first.next_batch()
                state = json.loads(json.dumps(first.state_dict()))
                rest = self.drain(first)
            with (
                patch.object(self, "shard", side_effect=AssertionError("rebuilt shard index")),
                closing(self.open(spec)) as resumed,
            ):
                resumed.load_state_dict(state)
                self.assertEqual(self.drain(resumed), rest)
            with closing(self.open(replace(spec, batch_size=1))) as changed:
                with self.assertRaisesRegex(ValueError, "unchanged"):
                    changed.load_state_dict(state)

    def test_published_shard_ranges_reopen_without_membership_traversal(self):
        spec = replace(self.spec, policy="shard_local", dp_size=8, topology=(8, 1))
        original = {"wide": [f"4/{i}" for i in range(300)] + ["3/1", "4/300"], "empty": [], "square": ["1/0"]}
        write_membership(self.path, original, {"source_revision": "source-1"})
        baseline = []
        for rank in range(spec.dp_size):
            with closing(self.open(replace(spec, dp_rank=rank))) as cursor:
                baseline.append(self.drain(cursor))

        write_membership(self.path, original, {"source_revision": "source-1"}, shard_of=self.shard)
        with self.path.open("rb") as source:
            pages = MembershipPages.open(source)
            index = shard_runs_path(self.path, pages.header["view_revision"])
        self.assertTrue(index.is_file())

        for rank in range(spec.dp_size):
            with (
                patch.object(MembershipPages, "iter_refs", side_effect=AssertionError("scanned membership")),
                patch.object(self, "shard", side_effect=AssertionError("parsed a sample reference")),
            ):
                cursor = self.open(replace(spec, dp_rank=rank))
            with closing(cursor):
                self.assertEqual(
                    cursor.db.execute("SELECT shard,bucket,start,n FROM runs ORDER BY bucket,start").fetchall(),
                    [(4, 0, 0, 300), (3, 0, 300, 1), (4, 0, 301, 1), (1, 2, 0, 1)],
                )
                self.assertEqual(self.drain(cursor), baseline[rank])

        with self.assertRaisesRegex(ValueError, "invalid literal"):
            write_membership(self.path, {"wide": ["bad/1"]}, {"source_revision": "source-1"}, shard_of=self.shard)
        with self.path.open("rb") as source:
            self.assertEqual(MembershipPages.open(source).header["view_revision"], pages.header["view_revision"])

        index.unlink()
        with (
            patch.object(MembershipPages, "iter_refs", side_effect=AssertionError("fell back to a membership scan")),
            self.assertRaisesRegex(FileNotFoundError, "shard ranges"),
        ):
            self.open(spec)

    def test_indexed_shard_local_schedule_is_shared_and_pinned(self):
        spec = replace(self.spec, policy="shard_local", dp_size=8, topology=(8, 1), pad=True)
        expected = []
        for rank in range(spec.dp_size):
            with closing(self.open(replace(spec, dp_rank=rank))) as cursor:
                expected.append(self.drain(cursor))

        buckets = {bucket: [f"webshart://{ref}/image.jpg" for ref in refs] for bucket, refs in self.buckets.items()}
        publish_geometry(self.path, buckets, lambda ref: self.shard(ref.removeprefix("webshart://")), shards_per_partition=2)
        build = WebshartEpoch._order_shards

        def read_rank(rank):
            with closing(self.open(replace(spec, dp_rank=rank))) as cursor:
                schedule = Path(cursor.db.execute("PRAGMA database_list").fetchone()[2])
                batches = [
                    (bucket, [ref.split("//", 1)[1].rsplit("/", 1)[0] for ref in refs])
                    for bucket, refs in self.drain(cursor)
                ]
                return schedule, batches

        with (
            patch.object(WebshartEpoch, "_order_shards", autospec=True, side_effect=build) as builds,
            ThreadPoolExecutor(max_workers=spec.dp_size) as pool,
        ):
            ranks = list(pool.map(read_rank, range(spec.dp_size)))
        self.assertEqual(builds.call_count, 1)
        self.assertEqual(len({path for path, _ in ranks}), 1)
        self.assertEqual([batches for _, batches in ranks], expected)

        schedule = ranks[0][0]
        next_spec = replace(spec, epoch=spec.epoch + 1)
        with patch(
            "simpletuner.helpers.metadata.backends.webshart_epoch.FileLock",
            side_effect=AssertionError("warm schedule took a write lock"),
        ):
            pinned = self.open(spec)
        with closing(pinned):
            pinned.next_batch()
            state = pinned.state_dict()
            old_inode = schedule.stat().st_ino
            with (
                patch.object(WebshartEpoch, "_order_shards", side_effect=RuntimeError("interrupted schedule build")),
                self.assertRaisesRegex(RuntimeError, "interrupted schedule build"),
            ):
                self.open(next_spec)
            self.assertEqual(schedule.stat().st_ino, old_inode)
            with closing(self.open(next_spec)) as next_epoch:
                self.assertNotEqual(schedule.stat().st_ino, old_inode)
                self.assertEqual(next_epoch.batches, pinned.batches)
            remaining = self.drain(pinned)
        with closing(self.open(spec)) as resumed:
            resumed.load_state_dict(state)
            self.assertEqual(self.drain(resumed), remaining)

    def test_resume_validates_the_complete_fixed_layout_with_bounded_state(self):
        with closing(self.open(self.spec)) as cursor:
            start = cursor.state_dict()
            for _ in range(cursor.batches):
                cursor.next_batch()
            end = cursor.state_dict()
            self.assertEqual({**end, "position": 0}, start)
            self.assertEqual(len(json.dumps(start)), len(json.dumps(end)) - len(str(end["position"])) + 1)
        changes = (
            {"batch_size": 1},
            {"dp_size": 3},
            {"dp_rank": 1},
            {"accumulation": 3},
            {"repeats": 2},
            {"pad": True},
            {"policy": "shard_local"},
            {"topology": (2, 2)},
            {"seed": 3},
            {"occurrence_revision": "changed"},
            {"epoch": 3},
        )
        for change in changes:
            with self.subTest(change=change), closing(self.open(replace(self.spec, **change))) as cursor:
                with self.assertRaisesRegex(ValueError, "unchanged"):
                    cursor.load_state_dict(end)

    def test_sparse_buckets_repeats_accumulation_padding_and_publication_replacement(self):
        write_membership(self.path, {"a": ["4/1", "3/2", "4/3"], "b": []}, {"source_revision": "source-1"})
        for policy in ("global", "shard_local"):
            spec = replace(self.spec, policy=policy, repeats=1, accumulation=2, pad=True)
            outputs = []
            for rank in range(2):
                with closing(self.open(replace(spec, dp_rank=rank))) as cursor:
                    outputs.extend(ref for _, refs in self.drain(cursor) for ref in refs)
            self.assertEqual(len(outputs), 8)
            self.assertTrue(all(outputs.count(ref) >= 2 for ref in ("4/1", "3/2", "4/3")))
            with closing(self.open(replace(spec, pad=False))) as cursor:
                self.assertEqual(cursor.batches, 0)
        with closing(self.open(self.spec)) as first:
            state = first.state_dict()
        write_membership(self.path, {"a": ["1/1"]}, {"source_revision": "source-1"})
        with closing(self.open(self.spec)) as new:
            with self.assertRaisesRegex(ValueError, "unchanged"):
                new.load_state_dict(state)


def distributed_filter(rank, path, rendezvous, transport):
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from simpletuner.helpers.metadata.backends.webshart_exclusions import collect_exclusions
    from simpletuner.helpers.metadata.backends.webshart_membership import MembershipPages

    if transport == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(transport, rank=rank, world_size=2, init_method=rendezvous)
    try:

        def describe():
            with path.open("rb") as file:
                return MembershipPages.open(file).describe()

        backend = SimpleNamespace(
            cache_file=path,
            data_backend=SimpleNamespace(_cache_path=Path, shard_of=lambda ref: int(ref) // 100),
            accelerator=SimpleNamespace(is_local_main_process=rank == 0),
            describe=describe,
            reload_cache=lambda **kwargs: None,
        )
        with collect_exclusions(backend) as changes:
            for row in range(rank, 301, 2):
                changes.add(str(row), "wide", "nsfw")
        assert describe().accepted_samples == 1
        with path.open("rb") as file:
            assert MembershipPages.open(file).read_page("wide", 0, 2) == (["301"], 1)
        try:
            with collect_exclusions(backend) as changes:
                changes.add("301", "wide", "decode")
                if rank == 1:
                    raise OSError("decode failed")
        except (RuntimeError, OSError):
            pass
        else:
            raise AssertionError("failure on one rank did not abort the transaction")
        assert describe().accepted_samples == 1
    finally:
        dist.destroy_process_group()


class FilterPublicationTests(unittest.TestCase):
    def test_sparse_exclusions_preserve_index_and_pinned_readers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "membership.json"
            from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
            from simpletuner.helpers.metadata.backends.webshart_geometry import geometry_lookup

            def shard_of(ref):
                return WebshartDataBackend.parse_sample_id(ref).shard_idx

            wide = [f"webshart://{shard}/{row}/wide.jpg" for shard in range(3) for row in range(300)]
            square = ["webshart://0/301/square.jpg", "webshart://0/302/square.jpg"]
            original = publish_geometry(path, {"wide": wide, "square": square}, shard_of, shards_per_partition=1)
            original_stat = original.stat()
            rejected = {wide[0], wide[1], wide[298], wide[-1], *wide[300:600]}
            with path.open("rb") as old, closing(sqlite3.connect("")) as db:
                pinned = MembershipPages.open(old)
                ledger = Exclusions(db)
                for ref in rejected:
                    ledger.add(ref, "wide", "decode")
                for ref in square:
                    ledger.add(ref, "square", "nsfw")
                expected = pinned.header["view_revision"]
                with patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("scanned base refs")):
                    _publish(path, expected, db)
                    _publish(path, expected, db)
                self.assertEqual(pinned.read_page("wide", 298, 3), (wide[298:301], 900))
            self.assertEqual(original.stat().st_ino, original_stat.st_ino)
            self.assertEqual(original.stat().st_mtime_ns, original_stat.st_mtime_ns)

            kept = [ref for ref in wide if ref not in rejected]
            with path.open("rb") as current:
                view = MembershipPages.open(current)
                self.assertEqual(view.read_page(None, 294, 9), (kept[294:303], len(kept)))
                self.assertEqual(view.read_page("square", 0, 1), ([], 0))
                positions = [300, 0, len(kept) - 1, 300]
                selected = [kept[i] for i in positions]
                self.assertEqual(view.read_positions("wide", positions), selected)
                with closing(view.open_geometry()) as db:
                    self.assertEqual(db.execute("SELECT id FROM partitions ORDER BY id").fetchall(), [(0,), (1,), (2,)])
                    lookup = geometry_lookup(path, view.header, db)
                    self.assertEqual(lookup.get_metadata_batch(selected), [{"ref": ref} for ref in selected])
                all_workers = [ref for worker in range(3) for ref in view.partition(worker, 3)]
                self.assertEqual(all_workers, kept)
                revision = view.header["view_revision"]

            for policy in ("global", "shard_local"):
                spec = EpochSpec("data", policy, 7, 2, 2, 1, 0, 1, 0, False, (1, 1))
                with (
                    patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("scanned base refs")),
                    closing(WebshartEpoch(path.open("rb"), spec, shard_of)) as epoch,
                ):
                    produced = [ref for _, refs in WebshartEpochTests.drain(epoch) for ref in refs]
                    self.assertEqual(Counter(produced), Counter(kept))

            with closing(sqlite3.connect("")) as db:
                ledger = Exclusions(db)
                ledger.add(kept[0], "wide", "nsfw")
                ledger.add("wrong/0", "wide", "decode")
                with self.assertRaisesRegex(ValueError, "outside"):
                    _publish(path, revision, db)
                with path.open("rb") as current:
                    self.assertEqual(MembershipPages.open(current).header["view_revision"], revision)
                db.execute("DELETE FROM exclusions WHERE ref='wrong/0'")
                _publish(path, revision, db)
                with path.open("rb") as current:
                    view = MembershipPages.open(current)
                    self.assertEqual(view.read_page(None, 0, 3), (kept[1:4], len(kept) - 1))
                    self.assertEqual(view.header["exclusion_counts"], {"decode": len(rejected), "nsfw": 3})

    def test_routing_index_adopts_independently_built_shard_groups(self):
        from contextlib import ExitStack

        with TemporaryDirectory() as directory, ExitStack() as stack:
            target = Path(directory) / "metadata.json"
            prepared = {}
            for shard in (1, 0):
                writer, path = stack.enter_context(stage_geometry(target))
                writer.pin_shard(shard, f"pin-{shard}")
                writer.add("square", f"webshart://{shard}/0/image.jpg", {}, shard=shard)
                writer.finish()
                writer.db.commit()
                prepared[shard] = path

            adopted, _ = stack.enter_context(stage_geometry(target, shards_per_partition=1))
            with self.assertRaisesRegex(ValueError, "outside its partition"):
                adopted.adopt_partition(0, prepared[1])
            for shard in (0, 1):
                adopted.adopt_partition(shard, prepared[shard])
            self.assertEqual(adopted.finish(), {"square": 2})

            self.assertEqual(
                adopted.db.execute("SELECT * FROM bucket_partitions ORDER BY start").fetchall(),
                [("square", 0, 1, 0), ("square", 1, 1, 1)],
            )
            self.assertEqual(
                adopted.db.execute("SELECT * FROM runs ORDER BY shard").fetchall(),
                [(0, "square", 0, 1), (1, "square", 1, 1)],
            )
            self.assertEqual(
                adopted.db.execute("SELECT * FROM shards ORDER BY position").fetchall(), [(0, "pin-0"), (1, "pin-1")]
            )

    def test_unmodified_partition_files_are_reused_when_publishing_a_new_view(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "membership.json"
            from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

            def shard_of(ref):
                return WebshartDataBackend.parse_sample_id(ref).shard_idx

            buckets = {"wide": ["webshart://0/0/a.jpg", "webshart://4/0/b.jpg"]}
            first = publish_geometry(path, buckets, shard_of, shards_per_partition=4)
            with closing(sqlite3.connect(first)) as db:
                parts = dict(db.execute("SELECT id,digest FROM partitions"))
            retained = snapshot_path(path.with_name("geometry.json"), parts[0])
            inode = retained.stat().st_ino
            buckets["wide"][1] = "webshart://4/0/c.jpg"
            second = publish_geometry(path, buckets, shard_of, shards_per_partition=4)
            self.assertNotEqual(first, second)
            with closing(sqlite3.connect(second)) as db:
                replaced = dict(db.execute("SELECT id,digest FROM partitions"))
            self.assertEqual(replaced[0], parts[0])
            self.assertNotEqual(replaced[1], parts[1])
            self.assertEqual(retained.stat().st_ino, inode)

    def test_ranks_agree_after_success_and_rollback(self):
        import os

        import torch.distributed as dist
        from torch.multiprocessing import spawn

        if not dist.is_available():
            self.skipTest("PyTorch distributed unavailable")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "membership.json"
            publish_geometry(path, {"wide": list(map(str, range(302)))}, lambda ref: int(ref) // 100)
            spawn(
                distributed_filter,
                args=(path, (path.parent / "rendezvous").as_uri(), os.environ.get("WEBSHART_TEST_TRANSPORT", "gloo")),
                nprocs=2,
            )

    def test_pinned_readers_atomic_failure_and_sparse_page(self):
        from simpletuner.helpers.metadata.backends.webshart_exclusions import read_exclusion_page
        from simpletuner.helpers.metadata.backends.webshart_membership import MembershipPages

        with TemporaryDirectory() as directory:
            path = Path(directory) / "membership.json"
            write_membership(path, {"wide": ["a", "b"], "square": ["c", "d"]}, {"source_revision": "capture"})
            with path.open("rb") as source, closing(sqlite3.connect("")) as db:
                pinned = MembershipPages.open(source)
                ledger = Exclusions(db)
                ledger.add("a", "wide", "nsfw")
                ledger.add("a", "wide", "nsfw")
                _publish(path, pinned.header["view_revision"], db)
                _publish(path, pinned.header["view_revision"], db)
                self.assertEqual(pinned.read_page(None, 0, 4)[0], ["a", "b", "c", "d"])
                with path.open("rb") as new:
                    revision = MembershipPages.open(new).header["view_revision"]
                db.execute("DELETE FROM exclusions")
                ledger.add("x", "wide", "decode")
                with self.assertRaisesRegex(ValueError, "outside"):
                    _publish(path, revision, db)
                with path.open("rb") as new:
                    self.assertEqual(MembershipPages.open(new).header["view_revision"], revision)
                db.execute("DELETE FROM exclusions")
                ledger.add("c", "square", "decode")
                _publish(path, revision, db)
                with path.open("rb") as new:
                    view = MembershipPages.open(new)
                    self.assertEqual(view.read_page(None, 0, 4), (["b", "d"], 2))
                    self.assertEqual(
                        read_exclusion_page(path, view.header, None, 1, 1), ({"nsfw": 1, "decode": 1}, [("c", "decode")])
                    )
                    self.assertEqual(read_exclusion_page(path, view.header, "nsfw", 5, 0), ({"nsfw": 1}, [("a", "nsfw")]))
