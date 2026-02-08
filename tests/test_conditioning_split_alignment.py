import multiprocessing
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


class StaleReadDataBackend(InMemoryDataBackend):
    """Data backend that can simulate stale cache reads returning empty JSON."""

    def __init__(self, backend_id: str):
        super().__init__(backend_id)
        self.simulate_stale_read = False

    def read(self, path):
        if self.simulate_stale_read:
            return "{}"
        return super().read(path)


class SharedDataBackend:
    """Process-safe backend that shares storage across processes via a Manager dict."""

    def __init__(self, backend_id: str, storage, flags=None, flag_key: str | None = None):
        self.id = backend_id
        self._storage = storage
        self._flags = flags or {}
        self._flag_key = flag_key

    def exists(self, path):
        return str(path) in self._storage

    def read(self, path):
        key = str(path)
        if self._flag_key and self._flags.get(self._flag_key, False) and key in self._storage:
            return "{}"
        return self._storage[key]

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

    def _init_state(
        self,
        *,
        num_processes: int,
        process_index: int,
        train_batch_size: int = 1,
    ) -> DummyAccelerator:
        accelerator = DummyAccelerator(num_processes=num_processes, process_index=process_index)
        args = SimpleNamespace(
            output_dir="/tmp",
            train_batch_size=train_batch_size,
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
        source_config_overrides: dict | None = None,
        conditioning_config_overrides: dict | None = None,
        source_data_backend: InMemoryDataBackend | None = None,
        conditioning_data_backend: InMemoryDataBackend | None = None,
        before_copy_callback=None,
    ):
        source_config = {
            "resolution_type": "area",
            "resolution": 1.0,
            "repeats": 1,  # Increased from 0 to pass validation (16 images × 2 = 32 samples for effective_batch_size=32)
            "instance_data_dir": source_dir,
        }
        if source_config_overrides:
            source_config.update(source_config_overrides)

        StateTracker.register_data_backend({"id": source_id})
        StateTracker.set_data_backend_config(
            source_id,
            source_config,
        )
        train_backend = source_data_backend or InMemoryDataBackend(source_id)
        training_metadata = DiscoveryMetadataBackend(
            id=source_id,
            instance_data_dir=source_dir,
            cache_file=f"{source_dir}/aspect_ratio_bucket_indices",
            metadata_file=f"{source_dir}/aspect_ratio_bucket_metadata",
            data_backend=train_backend,
            accelerator=accelerator,
            batch_size=StateTracker.get_args().train_batch_size,
            resolution=source_config.get("resolution", 1.0),
            resolution_type=source_config.get("resolution_type", "area"),
            minimum_image_size=source_config.get("minimum_image_size"),
            repeats=int(source_config.get("repeats", 0) or 0),
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

        conditioning_config = {
            "resolution_type": "area",
            "resolution": 1.0,
            "repeats": 1,  # Increased from 0 to pass validation
            "instance_data_dir": conditioning_dir,
            "source_dataset_id": source_id,
        }
        if conditioning_config_overrides:
            conditioning_config.update(conditioning_config_overrides)

        StateTracker.register_data_backend({"id": conditioning_id})
        StateTracker.set_data_backend_config(
            conditioning_id,
            conditioning_config,
        )
        conditioning_backend = conditioning_data_backend or InMemoryDataBackend(conditioning_id)
        conditioning_metadata = DiscoveryMetadataBackend(
            id=conditioning_id,
            instance_data_dir=conditioning_dir,
            cache_file=f"{conditioning_dir}/aspect_ratio_bucket_indices",
            metadata_file=f"{conditioning_dir}/aspect_ratio_bucket_metadata",
            data_backend=conditioning_backend,
            accelerator=accelerator,
            batch_size=StateTracker.get_args().train_batch_size,
            resolution=conditioning_config.get("resolution", 1.0),
            resolution_type=conditioning_config.get("resolution_type", "area"),
            minimum_image_size=conditioning_config.get("minimum_image_size"),
            repeats=int(conditioning_config.get("repeats", 0) or 0),
        )
        conditioning_entry = StateTracker.get_data_backend(conditioning_id)
        conditioning_entry.update(
            {
                "metadata_backend": conditioning_metadata,
                "instance_data_dir": conditioning_dir,
                "dataset_type": "conditioning",
            }
        )

        if before_copy_callback:
            before_copy_callback(
                training_metadata,
                conditioning_metadata,
                train_backend,
                conditioning_backend,
                source_entry,
                conditioning_entry,
            )

        DatasetDuplicator.copy_metadata(source_entry, conditioning_entry)
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

                self.assertTrue(
                    getattr(cond_meta, "read_only", False),
                    msg="Conditioning metadata should be marked read-only after duplication.",
                )
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

                cond_meta_clone = DiscoveryMetadataBackend(
                    id=f"{cond_meta.id}_clone",
                    instance_data_dir=cond_meta.instance_data_dir,
                    cache_file=f"{cond_meta.instance_data_dir}/aspect_ratio_bucket_indices_clone",
                    metadata_file=f"{cond_meta.instance_data_dir}/aspect_ratio_bucket_metadata_clone",
                    data_backend=InMemoryDataBackend(f"{cond_meta.id}_clone"),
                    accelerator=accelerator,
                    batch_size=1,
                    resolution=1.0,
                    resolution_type="area",
                    repeats=0,
                )
                cond_meta_clone.aspect_ratio_bucket_indices = deepcopy(cond_meta.aspect_ratio_bucket_indices)
                cond_meta_clone.split_buckets_between_processes()
                mutated = {bucket: list(paths) for bucket, paths in cond_meta_clone.aspect_ratio_bucket_indices.items()}
                self.assertNotEqual(
                    conditioning_buckets,
                    mutated,
                    msg="Double splitting should alter conditioning buckets — guard against this.",
                )

    def test_reference_strict_conditioning_inherits_sampling_parameters(self):
        base_images = {"1.0": [f"/datasets/train/img_{i}.png" for i in range(26)]}
        accelerator = self._init_state(num_processes=4, process_index=0, train_batch_size=8)
        source_id = "inherit_train"
        conditioning_id = "inherit_control"
        source_config_overrides = {
            "resolution": 1.048576,
            "resolution_type": "area",
            "minimum_image_size": 0.25,
            "maximum_image_size": 4.25,
            "target_downsample_size": 1.75,
            "repeats": 80,
            "crop": True,
            "crop_aspect": "random",
            "crop_aspect_buckets": [0.5, 1.0, 1.5],
            "crop_style": "center",
        }
        conditioning_config_overrides = {
            "source_dataset_id": source_id,
            "conditioning_type": "reference_strict",
            "repeats": 0,
            "minimum_image_size": 0.0,
            "maximum_image_size": None,
            "target_downsample_size": None,
            "crop": False,
            "crop_aspect": "square",
            "crop_aspect_buckets": [],
            "crop_style": "random",
        }

        train_meta, cond_meta = self._prepare_metadata_backends(
            accelerator=accelerator,
            base_buckets=base_images,
            source_id=source_id,
            source_dir="/datasets/inherit_train",
            conditioning_id=conditioning_id,
            conditioning_dir="/datasets/inherit_control",
            source_config_overrides=source_config_overrides,
            conditioning_config_overrides=conditioning_config_overrides,
        )

        conditioning_config = StateTracker.get_data_backend_config(conditioning_id)

        self.assertEqual(conditioning_config["repeats"], source_config_overrides["repeats"])
        self.assertEqual(cond_meta.repeats, source_config_overrides["repeats"])
        self.assertAlmostEqual(conditioning_config["resolution"], source_config_overrides["resolution"])
        self.assertEqual(conditioning_config["resolution_type"], source_config_overrides["resolution_type"])
        self.assertEqual(conditioning_config["minimum_image_size"], source_config_overrides["minimum_image_size"])
        self.assertEqual(conditioning_config["maximum_image_size"], source_config_overrides["maximum_image_size"])
        self.assertEqual(
            conditioning_config["target_downsample_size"],
            source_config_overrides["target_downsample_size"],
        )
        self.assertTrue(conditioning_config["crop"])
        self.assertEqual(conditioning_config["crop_aspect"], source_config_overrides["crop_aspect"])
        self.assertEqual(conditioning_config["crop_style"], source_config_overrides["crop_style"])
        self.assertEqual(conditioning_config["crop_aspect_buckets"], source_config_overrides["crop_aspect_buckets"])
        self.assertGreater(len(cond_meta), 0, "Conditioning metadata should provide batches after duplication.")

    def test_reference_strict_duplication_survives_stale_cache_reload(self):
        base_images = {"1.0": [f"/datasets/train/img_{i}.png" for i in range(16)]}
        accelerator = self._init_state(num_processes=4, process_index=1, train_batch_size=8)
        source_id = "stale_train"
        conditioning_id = "stale_control"
        stale_backend = StaleReadDataBackend(source_id)

        def before_copy(train_meta, cond_meta, train_backend, cond_backend, source_entry, conditioning_entry):
            split_view = {bucket: list(paths) for bucket, paths in train_meta.aspect_ratio_bucket_indices.items()}
            original_read_only = getattr(train_meta, "read_only", False)
            train_meta.read_only = False
            train_meta.aspect_ratio_bucket_indices = deepcopy(base_images)
            train_meta.save_cache()
            train_meta.aspect_ratio_bucket_indices = split_view
            train_meta.read_only = original_read_only
            stale_backend.simulate_stale_read = True

        _, cond_meta = self._prepare_metadata_backends(
            accelerator=accelerator,
            base_buckets=base_images,
            source_id=source_id,
            source_dir="/datasets/stale_train",
            conditioning_id=conditioning_id,
            conditioning_dir="/datasets/stale_control",
            source_data_backend=stale_backend,
            before_copy_callback=before_copy,
        )

        total_conditioning_images = sum(len(paths) for paths in cond_meta.aspect_ratio_bucket_indices.values())
        self.assertGreater(
            total_conditioning_images,
            0,
            "Conditioning duplication should retain samples even when cache reload returns empty data.",
        )

    def test_reference_strict_duplication_multi_process_reload(self):
        manager = multiprocessing.Manager()
        try:
            shared_storage = manager.dict()
            conditioning_storage = manager.dict()
            flags = manager.dict()
            cache_ready = manager.Event()
            result_queue = manager.Queue()

            base_images = {"1.0": [f"/datasets/shared/img_{i}.png" for i in range(16)]}
            num_processes = 2

            processes = [
                multiprocessing.Process(
                    target=_run_reference_strict_duplication_worker,
                    args=(
                        index,
                        num_processes,
                        base_images,
                        shared_storage,
                        conditioning_storage,
                        flags,
                        cache_ready,
                        result_queue,
                    ),
                )
                for index in range(num_processes)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            self.assertEqual(len(results), num_processes)
            for rank, train_total, cond_total in results:
                self.assertGreater(train_total, 0, f"Process {rank} should retain training buckets.")
                self.assertGreater(cond_total, 0, f"Process {rank} should retain conditioning buckets.")
        finally:
            manager.shutdown()


if __name__ == "__main__":  # pragma: no cover - convenience for local debugging
    unittest.main()


def _run_reference_strict_duplication_worker(
    process_index,
    num_processes,
    base_images,
    shared_storage,
    conditioning_storage,
    flags,
    cache_ready,
    result_queue,
):
    test_case = TestConditioningSplitAlignment()
    test_case.setUp()
    try:
        accelerator = test_case._init_state(
            num_processes=num_processes,
            process_index=process_index,
            train_batch_size=8,
        )
        flag_key = f"simulate_stale_read_{process_index}"
        flags[flag_key] = False
        source_backend = SharedDataBackend("shared_source", shared_storage, flags, flag_key)
        conditioning_backend = SharedDataBackend(f"shared_control_{process_index}", conditioning_storage)

        def before_copy(
            train_meta,
            cond_meta,
            train_backend,
            cond_backend,
            source_entry,
            conditioning_entry,
        ):
            if process_index == 0:
                split_view = {bucket: list(paths) for bucket, paths in train_meta.aspect_ratio_bucket_indices.items()}
                original_read_only = getattr(train_meta, "read_only", False)
                train_meta.read_only = False
                train_meta.aspect_ratio_bucket_indices = deepcopy(base_images)
                train_meta.save_cache()
                train_meta.aspect_ratio_bucket_indices = split_view
                train_meta.read_only = original_read_only
                cache_ready.set()
            else:
                cache_ready.wait()
                flags[flag_key] = True

        train_meta, cond_meta = test_case._prepare_metadata_backends(
            accelerator=accelerator,
            base_buckets=base_images,
            source_id="shared_source",
            source_dir="/datasets/shared_source",
            conditioning_id=f"shared_control_{process_index}",
            conditioning_dir=f"/datasets/shared_control_{process_index}",
            source_data_backend=source_backend,
            conditioning_data_backend=conditioning_backend,
            before_copy_callback=before_copy,
        )

        train_total = sum(len(paths) for paths in train_meta.aspect_ratio_bucket_indices.values())
        cond_total = sum(len(paths) for paths in cond_meta.aspect_ratio_bucket_indices.values())
        result_queue.put((process_index, train_total, cond_total))
    finally:
        test_case.tearDown()
