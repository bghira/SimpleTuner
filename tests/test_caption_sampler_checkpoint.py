import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch.utils.data import DataLoader

from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.data_backend.runtime.dataloader_iterator import random_dataloader_iterator
from simpletuner.helpers.metadata.captions import CaptionRecord
from simpletuner.helpers.training.caption_collate import collate_caption_batch
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker
from tests.helpers.data_backend.test_caption_pipeline import DummyCaptionMetadataBackend


class CaptionSamplerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.metadata = DummyCaptionMetadataBackend(5)
        self.accelerator = SimpleNamespace(num_processes=1, process_index=0, gradient_accumulation_steps=1)
        self.args = SimpleNamespace(seed=42, gradient_accumulation_steps=1, data_backend_sampling="uniform")
        for key, value in [
            ("args", self.args),
            ("data_backends", {}),
            ("exhausted_backends", []),
            ("repeats", {}),
            ("epoch", 1),
            ("global_step", 0),
        ]:
            patcher = patch.object(StateTracker, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        StateTracker.set_data_backend_config("caption", {"repeats": 0, "probability": 1})

    def sampler(self, **kwargs):
        return CaptionSampler(
            "caption",
            self.metadata,
            self.accelerator,
            kwargs.pop("batch_size", 2),
            shuffle=kwargs.pop("shuffle", False),
            seed=42,
            **kwargs,
        )

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
        outputs = [random_dataloader_iterator(step, backends)["metadata_ids"] for step in range(3)]
        self.assertEqual(outputs, [["meta-0", "meta-1"], ["meta-2", "meta-3"], ["meta-4", "meta-0"]])
        self.assertFalse(random_dataloader_iterator(3, backends))
        self.assertEqual(backends, {})
        self.assertEqual(sampler.epoch, 1)
        self.assertEqual(sampler._cursor, 0)
        fresh_epoch = random_dataloader_iterator(4, {"caption": loader})
        self.assertEqual(fresh_epoch["metadata_ids"], ["meta-0", "meta-1"])

    def test_mid_epoch_checkpoint_restores_exact_next_batch_on_each_rank(self):
        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                self.accelerator.num_processes = 2
                self.accelerator.process_index = rank
                sampler = self.sampler(batch_size=1, shuffle=True, repeats=2)
                sampler.set_epoch(3)
                loader = self.loader(sampler)
                for step in range(3):
                    random_dataloader_iterator(step, {"caption": loader})
                state_path = str(Path(directory, f"training_state-rank{rank}.json"))
                sampler.save_state(state_path)
                expected = [random_dataloader_iterator(step, {"caption": loader})["metadata_ids"] for step in range(3, 6)]
                resumed = self.sampler(batch_size=1, shuffle=True, repeats=2)
                resumed.load_states(state_path)
                resumed.set_epoch(3)
                resumed.log_state()
                resumed_loader = self.loader(resumed)
                actual = [
                    random_dataloader_iterator(step, {"caption": resumed_loader})["metadata_ids"] for step in range(3, 6)
                ]
                self.assertEqual(actual, expected)

    def test_checkpoint_at_epoch_end_restores_exhaustion_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = self.sampler()
            list(sampler)
            state_path = str(Path(directory, "training_state.json"))
            sampler.save_state(state_path)
            resumed = self.sampler()
            resumed.load_states(state_path)
            with self.assertRaises(MultiDatasetExhausted):
                next(iter(resumed))
            self.assertEqual(resumed.epoch, 1)
            self.assertEqual(next(iter(resumed)), ("meta-0", "meta-1"))

    def test_repeats_are_not_reapplied_by_random_iterator(self):
        self.metadata = DummyCaptionMetadataBackend(2)
        StateTracker.set_data_backend_config("caption", {"repeats": 2})
        sampler = self.sampler(batch_size=1, repeats=2)
        backends = {"caption": self.loader(sampler)}
        seen = [random_dataloader_iterator(step, backends)["metadata_ids"][0] for step in range(6)]
        self.assertEqual(seen, ["meta-0", "meta-1"] * 3)
        self.assertFalse(random_dataloader_iterator(6, backends))
        self.assertEqual(sampler.epoch, 1)

    def test_tiny_dataset_pads_full_batches_on_all_ranks(self):
        self.metadata = DummyCaptionMetadataBackend(1)
        for rank in range(3):
            with self.subTest(rank=rank):
                self.accelerator.num_processes = 3
                self.accelerator.process_index = rank
                sampler = self.sampler(batch_size=4)
                self.assertEqual(list(sampler), [("meta-0",) * 4])
                self.assertEqual(len(sampler), 1)

    def test_resume_rejects_changed_layout_and_caption_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            next(iter(sampler))
            sampler.save_state(state_path)
            for attribute, value in [("batch_size", 1), ("repeats", 1), ("shuffle", True), ("seed", 7)]:
                resumed = self.sampler()
                setattr(resumed, attribute, value)
                with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "unchanged dataset and topology"):
                    resumed.load_states(state_path)
            for attribute, value in [
                ("num_processes", 2),
                ("process_index", 1),
                ("gradient_accumulation_steps", 2),
                ("distributed_type", "FSDP"),
                ("parallelism_config", SimpleNamespace(cp_size=2)),
            ]:
                previous = getattr(self.accelerator, attribute, None)
                setattr(self.accelerator, attribute, value)
                with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "unchanged dataset and topology"):
                    self.sampler().load_states(state_path)
                if previous is None:
                    delattr(self.accelerator, attribute)
                else:
                    setattr(self.accelerator, attribute, previous)
            self.metadata._records["meta-0"].caption_text = "edited caption"
            with self.assertRaisesRegex(ValueError, "captions_sha256"):
                self.sampler().load_states(state_path)

    def test_missing_or_invalid_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory, "training_state.json"))
            sampler = self.sampler()
            with self.assertRaisesRegex(ValueError, "checkpoint state is missing"):
                sampler.load_states(state_path)
            sampler.save_state(state_path)
            path = Path(sampler.state_manager.mangle_state_path(state_path))
            state = json.loads(path.read_text())
            state["cursor"] = 1
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "invalid epoch or cursor"):
                sampler.load_states(state_path)

    def test_caption_prefetch_is_rejected_before_loader_creation(self):
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(dataloader_prefetch=True)
        with self.assertRaisesRegex(ValueError, "dataloader_prefetch=false"):
            factory._create_caption_dataloader({}, {})


if __name__ == "__main__":
    unittest.main()
