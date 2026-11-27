import io
import json
import tarfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.metadata.preparation import TrainingPreparation
from simpletuner.simpletuner_sdk.server.services.cache_job_service import _CacheAcceleratorStub, _CacheArgsNamespace


class FormatSamplingTests(unittest.TestCase):
    def test_rare_aspect_carries_through_checkpoints_and_accumulation(self):
        from simpletuner.helpers.metadata.backends.webshart_epoch import EpochSpec, ShardLocalBatches

        def members(window):
            return str(window), {
                key: [f"{window}/{key}/{row}" for row in range(size)] for key, size in (("square", 256), ("wide", 160))
            }

        for accumulation, repeats, pad in ((1, 0, False), (2, 0, False), (2, 1, True)):
            by_rank = []
            for rank in range(8):
                spec = EpochSpec("fixture", "shard_local", 42, 1, 32, 8, rank, accumulation, repeats, pad, ())

                def open_cursor():
                    return ShardLocalBatches(
                        spec, "source", "view", 4, members, window_shards=4, prefill=None, shard_at=None
                    )

                with closing(open_cursor()) as cursor:
                    states, batches = [], []
                    for batch in iter(cursor.next_batch, None):
                        states.append(json.loads(json.dumps(cursor.state_dict())))
                        batches.append(batch)
                    self.assertEqual(states[-1], json.loads(json.dumps(cursor.state_dict())))
                for at, state in enumerate(states):
                    with closing(open_cursor()) as resumed:
                        resumed.load_state_dict(state)
                        self.assertEqual(list(iter(resumed.next_batch, None)), batches[at + 1 :])
                wide = [ref for key, refs in batches if key == "wide" for ref in refs]
                if not pad:
                    self.assertEqual(len(wide), 64)
                self.assertTrue(wide)
                by_rank.append(set(wide))
            if not pad:
                self.assertEqual(len(set.union(*by_rank)), sum(map(len, by_rank)))

    def test_training_cursors_share_rows_across_formats_and_preparation_modes(self):
        import webshart

        if not hasattr(webshart.DiscoveredDataset, "shard_locations"):
            self.skipTest("Requires Webshart TAR window support")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar = root / "tar"
            parquet = root / "parquet"
            tar.mkdir()
            parquet.mkdir()
            for shard in range(9):
                rows = []
                archive = tar / f"{shard:03d}.tar"
                with tarfile.open(archive, "w") as writer:
                    for i, width in enumerate((64, 64, 64, 128, 128)):
                        name = f"{i:04d}.png"
                        stream = io.BytesIO()
                        Image.new("RGB", (width, 64), (shard, i, 0)).save(stream, format="PNG")
                        data = stream.getvalue()
                        entry = tarfile.TarInfo(name)
                        entry.size = len(data)
                        writer.addfile(entry, io.BytesIO(data))
                        rows.append(dict(image=data, width=width, height=64, captions=f"caption {shard}-{i}"))
                with tarfile.open(archive) as reader:
                    files = {
                        e.name: dict(
                            offset=e.offset_data,
                            length=e.size,
                            width=r["width"],
                            height=64,
                            aspect=r["width"] / 64,
                            captions=[r["captions"]],
                        )
                        for e, r in zip(reader, rows, strict=True)
                    }
                archive.with_suffix(".json").write_text(json.dumps(dict(filesize=archive.stat().st_size, files=files)))
                pq.write_table(pa.Table.from_pylist(rows), parquet / f"{shard:03d}.parquet")
            for seed in (None, 17):
                expected = {}
                for fmt in ("tar", "parquet"):
                    args = _CacheArgsNamespace(
                        dict(output_dir=str(root), seed=7, max_train_steps=40, gradient_accumulation_steps=1)
                    )
                    config = dict(
                        id="paired-fixture",
                        type="webshart",
                        source=str(root / fmt),
                        resolution=64,
                        resolution_type="pixel",
                        crop=False,
                        caption_strategy="webshart",
                        cache_dir=str(root / f"cache-{fmt}-{seed}"),
                        webshart=dict(format=fmt, sampling_policy="shard_local", sampling_order="shuffled"),
                    )
                    if seed is not None:
                        config["webshart"]["shuffle_seed"] = seed
                    with closing(DatasetRuntime.from_config(config, args, _CacheAcceleratorStub(device="cpu"))) as rt:
                        self.assertTrue(
                            rt.prepare_for_training(
                                TrainingPreparation(online_text=True, ondemand_latents=True, requires_complete_view=False)
                            )
                        )
                        for indexed in (False, True):
                            if indexed:
                                rt.prepare_view()
                            by_rank = []
                            for rank in (0, 1):
                                spec = replace(rt.metadata_backend.epoch_spec(2, 3), dp_size=2, dp_rank=rank, pad=False)
                                with closing(rt.open_epoch(spec)) as cursor:
                                    first = cursor.next_batch()
                                    saved = json.loads(json.dumps(cursor.state_dict()))
                                    rest = list(iter(cursor.next_batch, None))
                                with closing(rt.open_epoch(spec)) as resumed:
                                    resumed.load_state_dict(saved)
                                    self.assertEqual(list(iter(resumed.next_batch, None)), rest)
                                batches = [
                                    (
                                        bucket,
                                        [
                                            (loc.shard_idx, loc.sample_idx)
                                            for loc in map(rt.data_backend.parse_sample_id, refs)
                                        ],
                                    )
                                    for bucket, refs in [first, *rest]
                                ]
                                if fmt == "tar" and not indexed:
                                    expected[rank] = batches
                                self.assertEqual(batches, expected[rank], (fmt, indexed, seed, rank))
                                by_rank.append({i for _, refs in batches for i in refs})
                            self.assertTrue(by_rank[0].isdisjoint(by_rank[1]))
                            self.assertEqual(len(expected[0]), len(expected[1]))
                            self.assertTrue(any(len({s for s, _ in refs}) > 1 for _, refs in expected[0]))
