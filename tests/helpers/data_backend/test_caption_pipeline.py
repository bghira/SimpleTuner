import unittest
from itertools import islice
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
from simpletuner.helpers.data_backend.config.image import ImageBackendConfig
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.metadata.captions import CaptionRecord
from simpletuner.helpers.training.caption_collate import collate_caption_batch
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker


class DummyCaptionMetadataBackend:
    def __init__(self, count: int = 3):
        self._records = {
            f"meta-{index}": CaptionRecord(
                metadata_id=f"meta-{index}",
                caption_text=f"caption {index}",
                data_backend_id="caption",
            )
            for index in range(count)
        }

    @property
    def aspect_ratio_bucket_indices(self):
        return {"captions": list(self._records)}

    def list_metadata_ids(self):
        return list(self._records.keys())

    def get_record(self, metadata_id: str):
        return self._records.get(metadata_id)

    def __len__(self):
        return len(self._records)


def caption_sampler(metadata_backend, accelerator, batch_size):
    """A caption sampler over its own runtime, traversed as StateTracker's args and dataset config say."""
    runtime = DatasetRuntime("caption", None, metadata_backend, {"dataset_type": DatasetType.CAPTION})
    runtime.configure_cursor(accelerator=accelerator, batch_size=batch_size, shared_by_all_ranks=False)
    return CaptionSampler(runtime)


class CaptionDatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.backend = DummyCaptionMetadataBackend()
        self.accelerator = SimpleNamespace(num_processes=1, process_index=0)
        for key, value in [("args", SimpleNamespace(seed=42, gradient_accumulation_steps=1)), ("data_backends", {})]:
            patcher = patch.object(StateTracker, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_dataset_returns_payloads(self):
        dataset = CaptionDataset("caption", self.backend)
        batch = dataset[["meta-0", "meta-1"]]
        self.assertEqual(batch["data_backend_id"], "caption")
        self.assertEqual(batch["dataset_type"], DatasetType.CAPTION)
        self.assertEqual(len(batch["records"]), 2)
        captions = [record["caption_text"] for record in batch["records"]]
        self.assertEqual(captions, ["caption 0", "caption 1"])

    def test_sampler_batches_cover_all_records(self):
        sampler = caption_sampler(self.backend, self.accelerator, 2)
        self.assertEqual(len(sampler), 2)
        batches = list(islice(sampler, 2))
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        flattened = [item for batch in batches for item in batch]
        self.assertEqual(sorted(flattened[:3]), self.backend.list_metadata_ids())
        # The batch the captions cannot fill is completed from the head of the epoch's order.
        self.assertEqual(flattened[3], flattened[0])
        with self.assertRaises(MultiDatasetExhausted):
            next(iter(sampler))

    def test_sampler_respects_distributed_slicing(self):
        distributed_backend = DummyCaptionMetadataBackend(count=4)
        acc0 = SimpleNamespace(num_processes=2, process_index=0)
        acc1 = SimpleNamespace(num_processes=2, process_index=1)
        sampler_rank0 = caption_sampler(distributed_backend, acc0, 2)
        sampler_rank1 = caption_sampler(distributed_backend, acc1, 2)
        batches_rank0 = list(islice(sampler_rank0, 1))
        batches_rank1 = list(islice(sampler_rank1, 1))
        self.assertEqual(len(sampler_rank0), 1)
        self.assertEqual(len(sampler_rank1), 1)
        flat_rank0 = [item for batch in batches_rank0 for item in batch]
        flat_rank1 = [item for batch in batches_rank1 for item in batch]
        self.assertEqual(len(flat_rank0), 2)
        self.assertEqual(len(flat_rank1), 2)
        self.assertEqual(
            sorted(flat_rank0 + flat_rank1),
            sorted(distributed_backend.list_metadata_ids()),
        )

    def test_caption_collate_shapes_batch(self):
        dataset = CaptionDataset("caption", self.backend)
        example = dataset[["meta-0", "meta-1"]]
        payload = collate_caption_batch([example])
        self.assertEqual(payload["data_backend_id"], "caption")
        self.assertEqual(payload["dataset_type"], DatasetType.CAPTION)
        self.assertEqual(payload["captions"], ["caption 0", "caption 1"])
        self.assertEqual(payload["metadata_ids"], ["meta-0", "meta-1"])


class CaptionConfigValidationTests(unittest.TestCase):
    def test_csv_caption_backend_rejected(self):
        backend_dict = {
            "id": "captions",
            "type": "csv",
            "dataset_type": "caption",
            "instance_data_dir": "/tmp/captions",
        }
        args = {"resolution": 1024, "resolution_type": "pixel"}
        config = ImageBackendConfig.from_dict(backend_dict, args)
        with self.assertRaisesRegex(ValueError, "Caption datasets cannot use CSV backends"):
            config.validate(args)


if __name__ == "__main__":
    unittest.main()
