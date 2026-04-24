"""Tests for dataset scan service stubs and their compatibility with
the metadata backend infrastructure.

These tests verify that _ScanAcceleratorStub and _ScanArgsNamespace
provide all attributes/methods that metadata backends access at runtime,
preventing AttributeError crashes during standalone scans.
"""

import unittest

from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import (
    DatasetScanService,
    ScanStatus,
    _ScanAcceleratorStub,
    _ScanArgsNamespace,
)


class TestScanAcceleratorStub(unittest.TestCase):
    """Verify the accelerator stub exposes every attribute/method that
    metadata backends access on the real Accelerator object."""

    def setUp(self):
        self.stub = _ScanAcceleratorStub()

    # --- Attribute access (base.py lines 69-70, 85, 89) ---

    def test_num_processes(self):
        self.assertEqual(self.stub.num_processes, 1)

    def test_process_index(self):
        self.assertEqual(self.stub.process_index, 0)

    def test_is_main_process(self):
        self.assertTrue(self.stub.is_main_process)

    def test_is_local_main_process(self):
        self.assertTrue(self.stub.is_local_main_process)

    def test_device(self):
        self.assertEqual(self.stub.device, "cpu")

    def test_data_parallel_rank(self):
        self.assertEqual(self.stub.data_parallel_rank, 0)

    def test_data_parallel_shard_rank(self):
        self.assertEqual(self.stub.data_parallel_shard_rank, 0)

    # --- Methods (huggingface.py lines 145-151, base.py line 870) ---

    def test_wait_for_everyone_is_noop(self):
        self.stub.wait_for_everyone()  # should not raise

    def test_main_process_first_is_context_manager(self):
        with self.stub.main_process_first():
            pass  # should not raise

    def test_split_between_processes_returns_input(self):
        data = [1, 2, 3, 4, 5]
        with self.stub.split_between_processes(data) as result:
            self.assertIs(result, data)

    def test_split_between_processes_with_padding(self):
        data = ["a", "b"]
        with self.stub.split_between_processes(data, apply_padding=True) as result:
            self.assertIs(result, data)


class TestScanArgsNamespace(unittest.TestCase):
    """Verify the args namespace exposes every attribute that metadata
    backends access via StateTracker.get_args().ATTR during scanning.

    Attributes accessed in:
      - base.py: aspect_bucket_worker_count, enable_multiprocessing,
        disable_bucket_pruning, ignore_missing_files, model_type,
        allow_dataset_oversubscription, data_backend_sampling,
        gradient_accumulation_steps, framerate, controlnet
      - discovery.py: (inherits from base)
      - state_tracker.py: output_dir
    """

    def setUp(self):
        self.args = _ScanArgsNamespace()

    # --- Attributes accessed by base.py compute_aspect_ratio_bucket_indices ---

    def test_aspect_bucket_worker_count(self):
        self.assertIsInstance(self.args.aspect_bucket_worker_count, int)
        self.assertGreater(self.args.aspect_bucket_worker_count, 0)

    def test_enable_multiprocessing(self):
        self.assertFalse(self.args.enable_multiprocessing)

    def test_model_type(self):
        self.assertIsNotNone(self.args.model_type)

    # --- Attributes accessed by _enforce_min_bucket_size / _prune_small_buckets ---

    def test_disable_bucket_pruning(self):
        self.assertTrue(self.args.disable_bucket_pruning)

    # --- Attributes accessed by _check_for_missing_files ---

    def test_ignore_missing_files(self):
        self.assertIsInstance(self.args.ignore_missing_files, bool)

    # --- Attributes accessed by StateTracker._load_from_disk ---

    def test_output_dir(self):
        self.assertIsNotNone(self.args.output_dir)
        self.assertIsInstance(self.args.output_dir, str)

    # --- Attributes accessed by bucket splitting / dataloader ---

    def test_controlnet(self):
        self.assertIsInstance(self.args.controlnet, bool)

    def test_allow_dataset_oversubscription(self):
        self.assertIsInstance(self.args.allow_dataset_oversubscription, bool)

    def test_data_backend_sampling(self):
        self.assertIsNotNone(self.args.data_backend_sampling)

    def test_framerate(self):
        self.assertIsInstance(self.args.framerate, int)

    def test_gradient_accumulation_steps(self):
        self.assertIsInstance(self.args.gradient_accumulation_steps, int)
        self.assertGreater(self.args.gradient_accumulation_steps, 0)

    # --- Attributes from metadata backend __init__ ---

    def test_resolution(self):
        self.assertIsNotNone(self.args.resolution)

    def test_resolution_type(self):
        self.assertIn(self.args.resolution_type, ("pixel", "area"))

    def test_minimum_image_size(self):
        self.assertIsNotNone(self.args.minimum_image_size)

    def test_train_batch_size(self):
        self.assertIsInstance(self.args.train_batch_size, int)
        self.assertGreater(self.args.train_batch_size, 0)

    def test_delete_problematic_images(self):
        self.assertIsInstance(self.args.delete_problematic_images, bool)

    def test_compress_disk_cache(self):
        self.assertIsInstance(self.args.compress_disk_cache, bool)

    # --- Override mechanism ---

    def test_overrides_applied(self):
        args = _ScanArgsNamespace({"resolution": 512, "model_type": "lora"})
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.model_type, "lora")

    def test_overrides_do_not_remove_defaults(self):
        args = _ScanArgsNamespace({"resolution": 768})
        self.assertTrue(hasattr(args, "enable_multiprocessing"))
        self.assertTrue(hasattr(args, "output_dir"))

    # --- dict-like interface used by builder code ---

    def test_contains_existing_key(self):
        self.assertIn("resolution", self.args)

    def test_contains_missing_key(self):
        self.assertNotIn("nonexistent_attr_xyz", self.args)

    def test_get_existing_key(self):
        self.assertEqual(self.args.get("resolution"), 1024)

    def test_get_missing_key_with_default(self):
        self.assertEqual(self.args.get("nonexistent", "fallback"), "fallback")


class TestScanServiceConcurrency(unittest.TestCase):
    """Verify the scan service enforces single-scan concurrency."""

    def setUp(self):
        self.service = DatasetScanService(broadcast_fn=None)
        # Reset class-level state
        DatasetScanService._active_job = None
        DatasetScanService._active_queue = None
        DatasetScanService._scan_thread = None

    def tearDown(self):
        DatasetScanService._active_job = None
        DatasetScanService._active_queue = None
        DatasetScanService._scan_thread = None

    def test_get_active_status_returns_none_when_idle(self):
        result = self.service.get_active_status()
        self.assertIsNone(result)

    def test_cancel_scan_returns_false_when_idle(self):
        self.assertFalse(self.service.cancel_scan())

    def test_get_scan_status_returns_none_for_unknown_job(self):
        self.assertIsNone(self.service.get_scan_status("nonexistent"))

    def test_get_queue_status_returns_none_for_unknown_queue(self):
        self.assertIsNone(self.service.get_queue_status("nonexistent"))


class TestScanServiceActiveStatus(unittest.TestCase):
    """Verify get_active_status returns correct state."""

    def setUp(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import ScanJob

        self.service = DatasetScanService(broadcast_fn=None)
        DatasetScanService._active_queue = None

        self.job = ScanJob(job_id="test-123", dataset_id="my-dataset")
        self.job.status = ScanStatus.RUNNING
        self.job.current = 50
        self.job.total = 100
        DatasetScanService._active_job = self.job

    def tearDown(self):
        DatasetScanService._active_job = None
        DatasetScanService._active_queue = None
        DatasetScanService._scan_thread = None

    def test_returns_status_when_running(self):
        result = self.service.get_active_status()
        self.assertIsNotNone(result)
        self.assertEqual(result["dataset_id"], "my-dataset")
        self.assertEqual(result["current"], 50)
        self.assertEqual(result["total"], 100)
        self.assertEqual(result["status"], "running")

    def test_returns_none_when_completed(self):
        self.job.status = ScanStatus.COMPLETED
        result = self.service.get_active_status()
        self.assertIsNone(result)

    def test_returns_none_when_failed(self):
        self.job.status = ScanStatus.FAILED
        result = self.service.get_active_status()
        self.assertIsNone(result)


class TestPixelAreaConversion(unittest.TestCase):
    """Verify pixel_area -> area conversion mirrors factory behavior."""

    def test_converts_pixel_area_to_area(self):
        config = {
            "id": "test",
            "resolution_type": "pixel_area",
            "resolution": 1024,
            "maximum_image_size": 1024,
            "target_downsample_size": 1024,
            "minimum_image_size": 256,
        }
        args = _ScanArgsNamespace()
        result = DatasetScanService._convert_pixel_area(config, args)

        self.assertEqual(result["resolution_type"], "area")
        self.assertAlmostEqual(result["resolution"], 1.048576)
        self.assertAlmostEqual(result["maximum_image_size"], 1.048576)
        self.assertAlmostEqual(result["target_downsample_size"], 1.048576)
        self.assertAlmostEqual(result["minimum_image_size"], 0.065536)

    def test_leaves_pixel_type_unchanged(self):
        config = {
            "id": "test",
            "resolution_type": "pixel",
            "resolution": 1024,
        }
        args = _ScanArgsNamespace()
        result = DatasetScanService._convert_pixel_area(config, args)

        self.assertEqual(result["resolution_type"], "pixel")
        self.assertEqual(result["resolution"], 1024)

    def test_leaves_area_type_unchanged(self):
        config = {
            "id": "test",
            "resolution_type": "area",
            "resolution": 1.0,
        }
        args = _ScanArgsNamespace()
        result = DatasetScanService._convert_pixel_area(config, args)

        self.assertEqual(result["resolution_type"], "area")
        self.assertEqual(result["resolution"], 1.0)

    def test_does_not_mutate_original(self):
        config = {
            "id": "test",
            "resolution_type": "pixel_area",
            "resolution": 512,
        }
        args = _ScanArgsNamespace()
        result = DatasetScanService._convert_pixel_area(config, args)

        self.assertEqual(config["resolution_type"], "pixel_area")
        self.assertEqual(config["resolution"], 512)
        self.assertEqual(result["resolution_type"], "area")

    def test_skips_zero_size_fields(self):
        config = {
            "id": "test",
            "resolution_type": "pixel_area",
            "resolution": 1024,
            "minimum_image_size": 0,
        }
        args = _ScanArgsNamespace()
        result = DatasetScanService._convert_pixel_area(config, args)

        self.assertEqual(result["minimum_image_size"], 0)


if __name__ == "__main__":
    unittest.main()
