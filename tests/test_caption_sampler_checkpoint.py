import json
import os
import tempfile
import unittest
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch.utils.data import DataLoader

from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.data_backend.runtime.dataloader_iterator import random_dataloader_iterator
from simpletuner.helpers.training.caption_collate import collate_caption_batch
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker
from tests.helpers.data_backend.test_caption_pipeline import DummyCaptionMetadataBackend, caption_sampler


class CaptionSamplerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.metadata = DummyCaptionMetadataBackend(5)
        self.accelerator = SimpleNamespace(num_processes=1, process_index=0)
        self.args = SimpleNamespace(
            seed=42,
            gradient_accumulation_steps=1,
            data_backend_sampling="uniform",
            allow_dataset_oversubscription=False,
            i_know_what_i_am_doing=False,
        )
        for key, value in [
            ("args", self.args),
            ("data_backends", {}),
            ("exhausted_backends", []),
            ("epoch", 1),
            ("global_step", 0),
        ]:
            patcher = patch.object(StateTracker, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def sampler(self, batch_size=2, repeats=0):
        StateTracker.set_data_backend_config("caption", {"repeats": repeats, "probability": 1})
        return caption_sampler(self.metadata, self.accelerator, batch_size)

    def loader(self, sampler):
        return DataLoader(
            CaptionDataset("caption", self.metadata),
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_caption_batch,
        )

    def test_real_random_iterator_advances_and_exhausts_once(self):
        sampler = self.sampler()
        loader = self.loader(sampler)
        backends = {"caption": loader}
        seen = [caption for step in range(3) for caption in random_dataloader_iterator(step, backends)["metadata_ids"]]
        self.assertEqual(sorted(seen[:5]), self.metadata.list_metadata_ids())
        self.assertEqual(seen[5], seen[0])
        self.assertFalse(random_dataloader_iterator(3, backends))
        self.assertEqual(backends, {})
        self.assertEqual(sampler.cursor.epoch, 2)
        self.assertEqual(sampler.cursor.progress(), (0, 5))
        fresh_epoch = random_dataloader_iterator(4, {"caption": loader})
        self.assertEqual(len(fresh_epoch["metadata_ids"]), 2)

    def test_mid_epoch_checkpoint_restores_exact_next_batch_on_each_rank(self):
        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                self.accelerator.num_processes = 2
                self.accelerator.process_index = rank
                sampler = self.sampler(batch_size=1, repeats=2)
                loader = self.loader(sampler)
                for step in range(3):
                    random_dataloader_iterator(step, {"caption": loader})
                state_path = str(Path(directory, f"training_state-rank{rank}.json"))
                sampler.save_state(state_path)
                expected = [random_dataloader_iterator(step, {"caption": loader})["metadata_ids"] for step in range(3, 6)]
                resumed = self.sampler(batch_size=1, repeats=2)
                resumed.load_states(state_path)
                resumed.log_state()
                resumed_loader = self.loader(resumed)
                actual = [
                    random_dataloader_iterator(step, {"caption": resumed_loader})["metadata_ids"] for step in range(3, 6)
                ]
                self.assertEqual(actual, expected)

    def test_checkpoint_at_epoch_end_restores_exhaustion_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = self.sampler()
            list(islice(sampler, len(sampler)))
            state_path = str(Path(directory, "training_state.json"))
            sampler.save_state(state_path)
            resumed = self.sampler()
            resumed.load_states(state_path)
            for instance in (sampler, resumed):
                with self.assertRaises(MultiDatasetExhausted):
                    next(iter(instance))
                self.assertEqual(instance.cursor.epoch, 2)
            self.assertEqual(next(iter(resumed)), next(iter(sampler)))

    def test_drop_unfinished_epoch_only_moves_a_started_epoch(self):
        sampler = self.sampler()
        sampler.drop_unfinished_epoch()
        self.assertEqual(sampler.cursor.epoch, 1)
        next(iter(sampler))
        sampler.drop_unfinished_epoch()
        self.assertEqual(sampler.cursor.epoch, 2)
        self.assertEqual(len(list(islice(sampler, 3))), 3)

    def test_repeats_are_not_reapplied_by_random_iterator(self):
        self.metadata = DummyCaptionMetadataBackend(2)
        sampler = self.sampler(batch_size=1, repeats=2)
        backends = {"caption": self.loader(sampler)}
        seen = [random_dataloader_iterator(step, backends)["metadata_ids"][0] for step in range(6)]
        self.assertEqual(sorted(seen), ["meta-0"] * 3 + ["meta-1"] * 3)
        self.assertFalse(random_dataloader_iterator(6, backends))
        self.assertEqual(sampler.cursor.epoch, 2)

    def test_captions_that_cannot_fill_a_batch_on_every_rank_need_oversubscription(self):
        self.metadata = DummyCaptionMetadataBackend(1)
        self.accelerator.num_processes = 3
        with self.assertRaisesRegex(ValueError, "zero usable batches"):
            len(self.sampler(batch_size=4))
        self.args.allow_dataset_oversubscription = True
        for rank in range(3):
            with self.subTest(rank=rank):
                self.accelerator.process_index = rank
                sampler = self.sampler(batch_size=4)
                self.assertEqual(len(sampler), 1)
                self.assertEqual(list(islice(sampler, 1)), [("meta-0",) * 4])

    def test_resume_rejects_a_changed_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            next(iter(sampler))
            sampler.save_state(state_path)
            with self.assertRaisesRegex(ValueError, "checkpoint batch_size=2 does not match current batch_size=1"):
                self.sampler(batch_size=1).load_states(state_path)
            with self.assertRaisesRegex(ValueError, "different dataset layout"):
                self.sampler(repeats=1).load_states(state_path)
            for attribute, value in [("num_processes", 2), ("process_index", 1)]:
                self.accelerator.num_processes = 2
                setattr(self.accelerator, attribute, value)
                with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "different dataset layout"):
                    self.sampler().load_states(state_path)

    def test_resume_with_a_changed_caption_list_starts_the_epoch_over(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            list(islice(sampler, len(sampler)))
            with self.assertRaises(MultiDatasetExhausted):
                next(iter(sampler))
            next(iter(sampler))
            sampler.save_state(state_path)
            self.metadata = DummyCaptionMetadataBackend(6)
            resumed = self.sampler()
            with patch("simpletuner.helpers.data_backend.cursor.logger") as logger:
                resumed.load_states(state_path)
            self.assertIn("5 samples then, 6 now", logger.warning.call_args.args[0])
            self.assertEqual(resumed.cursor.epoch, 2)
            self.assertEqual(resumed.cursor.progress(), (0, 6))

    def test_resume_with_a_changed_batch_size_needs_the_override(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            consumed = next(iter(sampler))
            sampler.save_state(state_path)
            self.args.i_know_what_i_am_doing = True
            resumed = self.sampler(batch_size=1)
            resumed.load_states(state_path)
            remaining = [caption for batch in islice(resumed, 3) for caption in batch]
            self.assertEqual(sorted(consumed + tuple(remaining)), self.metadata.list_metadata_ids())
            with self.assertRaises(MultiDatasetExhausted):
                next(iter(resumed))

    def test_missing_or_older_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            with self.assertRaisesRegex(ValueError, "checkpoint state is missing"):
                sampler.load_states(state_path)
            Path(sampler.state_manager.mangle_state_path(state_path)).write_text(
                json.dumps({"epoch": 0, "cursor": 2, "layout": {"batch_size": 2, "captions_sha256": "0" * 64}})
            )
            with self.assertRaisesRegex(ValueError, "older release"):
                sampler.load_states(state_path)

    def test_a_prefetched_caption_dataset_saves_the_batch_training_consumed(self):
        from simpletuner.helpers.data_backend.runtime import BatchFetcher

        self.addCleanup(StateTracker.set_consumed_position, None)
        reference = self.sampler()
        random_dataloader_iterator(0, {"caption": self.loader(reference)})

        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(**vars(self.args), dataloader_prefetch=True, train_batch_size=2)
        factory.accelerator = self.accelerator
        init_backend = {"id": "caption", "config": {}, "metadata_backend": self.metadata, "data_backend": None}
        factory._create_caption_dataloader({}, init_backend)
        fetcher = BatchFetcher(step=0, max_size=3, datasets={"caption": init_backend["train_dataloader"]})
        fetcher.fetch_responses()
        fetcher.next_response(1)
        self.assertNotEqual(init_backend["sampler"].cursor.state(), reference.cursor.state())

        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            init_backend["sampler"].save_state(state_path)
            resumed = self.sampler()
            resumed.load_states(state_path)
        self.assertEqual(resumed.cursor.state(), reference.cursor.state())

    def test_an_unshuffled_caption_dataset_is_rejected(self):
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(dataloader_prefetch=False)
        with self.assertRaisesRegex(ValueError, "remove shuffle=false"):
            factory._create_caption_dataloader({"shuffle": False}, {"id": "captions"})


if __name__ == "__main__":
    unittest.main()
