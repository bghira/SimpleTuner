import os
import unittest
from copy import deepcopy
from types import SimpleNamespace

from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.utils.duplicator import DatasetDuplicator
from simpletuner.helpers.training.state_tracker import StateTracker


class DummyAccelerator:
    """Lightweight stand-in for accelerate.Accelerator used to deterministically split data."""

    def __init__(self, num_processes: int, process_index: int):
        self.num_processes = num_processes
        self.process_index = process_index
        self.is_main_process = process_index == 0
        self.is_local_main_process = self.is_main_process

    class _SplitContext:
        def __init__(self, data, num_processes: int, process_index: int, apply_padding: bool):
            self._data = list(data)
            self._num_processes = max(num_processes, 1)
            self._process_index = process_index
            self._apply_padding = apply_padding

        def __enter__(self):
            import math

            if self._num_processes == 1:
                return self._data

            chunk_size = math.ceil(len(self._data) / self._num_processes)
            start = self._process_index * chunk_size
            end = start + chunk_size
            subset = self._data[start:end]

            if self._apply_padding and len(subset) < chunk_size and len(self._data) > 0:
                subset = subset + [None] * (chunk_size - len(subset))
            return subset

        def __exit__(self, exc_type, exc, tb):
            return False

    def split_between_processes(self, data, apply_padding: bool = False):
        return self._SplitContext(data, self.num_processes, self.process_index, apply_padding)


class InMemoryDataBackend:
    """Minimal data backend that keeps cache writes in memory for unit tests."""

    def __init__(self, backend_id: str):
        self.id = backend_id
        self._storage: dict[str, str] = {}

    def exists(self, path):
        return str(path) in self._storage

    def read(self, path):
        return self._storage[str(path)]

    def write(self, path, data):
        self._storage[str(path)] = data

    def read_image(self, path):  # pragma: no cover - not exercised in this test
        raise NotImplementedError("Image reads are not required for this unit test.")


class TestConditioningSplitAlignment(unittest.TestCase):
    def setUp(self):
        self._previous_shuffle = os.environ.get("SIMPLETUNER_SHUFFLE_BUCKETS")
        os.environ["SIMPLETUNER_SHUFFLE_BUCKETS"] = "0"

    def tearDown(self):
        if self._previous_shuffle is None:
            os.environ.pop("SIMPLETUNER_SHUFFLE_BUCKETS", None)
        else:
            os.environ["SIMPLETUNER_SHUFFLE_BUCKETS"] = self._previous_shuffle
        StateTracker.clear_data_backends()
        StateTracker.all_image_files = {}
        StateTracker.all_conditioning_image_embed_files = {}
        StateTracker.all_vae_cache_files = {}
        StateTracker.all_text_cache_files = {}

    def _init_state(self, *, num_processes: int, process_index: int) -> DummyAccelerator:
        accelerator = DummyAccelerator(num_processes=num_processes, process_index=process_index)
        args = SimpleNamespace(
            output_dir="/tmp",
            train_batch_size=1,
            gradient_accumulation_steps=1,
            disable_bucket_pruning=False,
            ignore_missing_files=True,
            enable_multiprocessing=False,
            delete_unwanted_images=False,
            model_type="",
        )
        StateTracker.set_args(args)
        StateTracker.set_accelerator(accelerator)
        return accelerator

    def _prepare_metadata_backends(
        self,
        *,
        accelerator: DummyAccelerator,
        base_buckets: dict[str, list[str]],
        source_id: str,
        source_dir: str,
        conditioning_id: str,
        conditioning_dir: str,
    ):
        StateTracker.register_data_backend({"id": source_id})
        StateTracker.set_data_backend_config(
            source_id,
            {
                "resolution_type": "area",
                "resolution": 1.0,
                "repeats": 0,
                "instance_data_dir": source_dir,
            },
        )
        train_backend = InMemoryDataBackend(source_id)
        training_metadata = DiscoveryMetadataBackend(
            id=source_id,
            instance_data_dir=source_dir,
            cache_file=f"{source_dir}/aspect_ratio_bucket_indices",
            metadata_file=f"{source_dir}/aspect_ratio_bucket_metadata",
            data_backend=train_backend,
            accelerator=accelerator,
            batch_size=1,
            resolution=1.0,
            resolution_type="area",
            repeats=0,
        )
        training_metadata.aspect_ratio_bucket_indices = deepcopy(base_buckets)
        source_entry = StateTracker.get_data_backend(source_id)
        source_entry.update(
            {
                "metadata_backend": training_metadata,
                "instance_data_dir": source_dir,
                "dataset_type": "image",
            }
        )
        training_metadata.split_buckets_between_processes()

        StateTracker.register_data_backend({"id": conditioning_id})
        StateTracker.set_data_backend_config(
            conditioning_id,
            {
                "resolution_type": "area",
                "resolution": 1.0,
                "repeats": 0,
                "instance_data_dir": conditioning_dir,
                "source_dataset_id": source_id,
            },
        )
        conditioning_backend = InMemoryDataBackend(conditioning_id)
        conditioning_metadata = DiscoveryMetadataBackend(
            id=conditioning_id,
            instance_data_dir=conditioning_dir,
            cache_file=f"{conditioning_dir}/aspect_ratio_bucket_indices",
            metadata_file=f"{conditioning_dir}/aspect_ratio_bucket_metadata",
            data_backend=conditioning_backend,
            accelerator=accelerator,
            batch_size=1,
            resolution=1.0,
            resolution_type="area",
            repeats=0,
        )
        conditioning_entry = StateTracker.get_data_backend(conditioning_id)
        conditioning_entry.update(
            {
                "metadata_backend": conditioning_metadata,
                "instance_data_dir": conditioning_dir,
                "dataset_type": "conditioning",
            }
        )

        DatasetDuplicator.copy_metadata(source_backend=source_entry, target_backend=conditioning_entry)
        conditioning_metadata.split_buckets_between_processes()
        return training_metadata, conditioning_metadata

    def test_conditioning_split_matches_training_split(self):
        base_images = {
            "1.0": [f"/datasets/train/img_{i}.png" for i in range(4)],
            "1.5": [f"/datasets/train/img_{i}.png" for i in range(4, 8)],
        }
        num_processes = 2

        for process_index in range(num_processes):
            with self.subTest(process=process_index):
                accelerator = self._init_state(num_processes=num_processes, process_index=process_index)
                train_meta, cond_meta = self._prepare_metadata_backends(
                    accelerator=accelerator,
                    base_buckets=base_images,
                    source_id="multigpu",
                    source_dir="/datasets/train",
                    conditioning_id="multigpu_control",
                    conditioning_dir="/datasets/train_control",
                )
                training_buckets = {bucket: list(paths) for bucket, paths in train_meta.aspect_ratio_bucket_indices.items()}
                conditioning_buckets = {
                    bucket: list(paths) for bucket, paths in cond_meta.aspect_ratio_bucket_indices.items()
                }

                self.assertEqual(set(training_buckets), set(conditioning_buckets))

                for bucket in training_buckets:
                    expected_paths = [
                        path.replace("/datasets/train", "/datasets/train_control", 1) for path in training_buckets[bucket]
                    ]
                    self.assertListEqual(
                        conditioning_buckets[bucket],
                        expected_paths,
                        msg=f"Bucket {bucket} mismatch on process {process_index}",
                    )


if __name__ == "__main__":  # pragma: no cover - convenience for local debugging
    unittest.main()
