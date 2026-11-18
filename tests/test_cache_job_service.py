"""Tests for cache_job_service: capabilities, job management, mutual exclusion."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.simpletuner_sdk.server.services.cache_job_service import (
    CacheJob,
    CacheJobService,
    CacheJobStatus,
    CacheType,
    _CacheAcceleratorStub,
    _CacheArgsNamespace,
)


class TestCacheCapabilities(unittest.TestCase):
    def test_flux_has_text_and_vae(self):
        caps = CacheJobService.get_capabilities("flux", [], {})
        self.assertTrue(caps["text_embeds"])
        self.assertTrue(caps["vae"])
        self.assertEqual(caps["conditioning_types"], [])

    def test_heartmula_no_text_embeds(self):
        caps = CacheJobService.get_capabilities("heartmula", [], {})
        self.assertFalse(caps["text_embeds"])
        self.assertTrue(caps["vae"])

    def test_omnigen_no_text_embeds(self):
        caps = CacheJobService.get_capabilities("omnigen", [], {})
        self.assertFalse(caps["text_embeds"])

    def test_empty_model_family_no_text_embeds(self):
        caps = CacheJobService.get_capabilities("", [], {})
        self.assertFalse(caps["text_embeds"])

    def test_conditioning_data_detected(self):
        datasets = [{"id": "ds1", "conditioning_data": ["cond1"]}]
        caps = CacheJobService.get_capabilities("flux", datasets, {})
        self.assertIn("image_embeds", caps["conditioning_types"])

    def test_conditioning_generators_detected(self):
        datasets = [{"id": "ds1", "conditioning": [{"type": "canny"}, {"type": "depth"}]}]
        caps = CacheJobService.get_capabilities("sdxl", datasets, {})
        self.assertIn("canny", caps["conditioning_types"])
        self.assertIn("depth", caps["conditioning_types"])

    def test_conditioning_generators_dict_form(self):
        datasets = [{"id": "ds1", "conditioning": {"type": "canny"}}]
        caps = CacheJobService.get_capabilities("sdxl", datasets, {})
        self.assertIn("canny", caps["conditioning_types"])

    def test_grounding_from_global_config(self):
        caps = CacheJobService.get_capabilities("sdxl", [], {"max_grounding_entities": 4})
        self.assertIn("grounding", caps["conditioning_types"])

    def test_no_grounding_when_zero(self):
        caps = CacheJobService.get_capabilities("sdxl", [], {"max_grounding_entities": 0})
        self.assertNotIn("grounding", caps["conditioning_types"])

    def test_no_duplicate_conditioning_types(self):
        datasets = [
            {"id": "ds1", "conditioning_data": ["x"]},
            {"id": "ds2", "conditioning_data": ["y"]},
        ]
        caps = CacheJobService.get_capabilities("flux", datasets, {})
        self.assertEqual(caps["conditioning_types"].count("image_embeds"), 1)


class TestCacheArgsNamespace(unittest.TestCase):
    def test_config_values_set(self):
        args = _CacheArgsNamespace({"resolution": 512, "model_family": "flux"})
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.model_family, "flux")

    def test_defaults_preserved(self):
        args = _CacheArgsNamespace({})
        self.assertEqual(args.train_batch_size, 1)
        self.assertTrue(args.hash_filenames)
        self.assertEqual(args.resolution, 1024)

    def test_config_overrides_defaults(self):
        args = _CacheArgsNamespace({"resolution": 768})
        self.assertEqual(args.resolution, 768)

    def test_cli_style_keys_stripped(self):
        args = _CacheArgsNamespace({"--cache_dir_text": "/tmp/text", "--model_family": "flux"})
        self.assertEqual(args.cache_dir_text, "/tmp/text")
        self.assertEqual(args.model_family, "flux")
        self.assertIn("cache_dir_text", args)
        self.assertIn("model_family", args)

    def test_contains(self):
        args = _CacheArgsNamespace({"foo": "bar"})
        self.assertIn("foo", args)
        self.assertIn("resolution", args)  # default key
        self.assertNotIn("nonexistent_key_xyz", args)

    def test_getattr_returns_none_for_unknown(self):
        args = _CacheArgsNamespace({})
        self.assertIsNone(args.pretrained_model_name_or_path)
        self.assertIsNone(args.some_totally_unknown_attribute)

    def test_getattr_raises_for_private(self):
        args = _CacheArgsNamespace({})
        with self.assertRaises(AttributeError):
            _ = args._private_thing

    def test_get(self):
        args = _CacheArgsNamespace({"foo": "bar"})
        self.assertEqual(args.get("foo"), "bar")
        self.assertEqual(args.get("nonexistent", 42), 42)


class TestCacheAcceleratorStub(unittest.TestCase):
    def test_properties(self):
        stub = _CacheAcceleratorStub()
        self.assertEqual(stub.num_processes, 1)
        self.assertTrue(stub.is_main_process)
        self.assertTrue(stub.is_local_main_process)
        self.assertIsNotNone(stub.device)

    def test_context_managers(self):
        stub = _CacheAcceleratorStub()
        with stub.main_process_first():
            pass
        with stub.split_between_processes([1, 2, 3]) as data:
            self.assertEqual(data, [1, 2, 3])

    def test_unwrap_model(self):
        stub = _CacheAcceleratorStub()
        obj = object()
        self.assertIs(stub.unwrap_model(obj), obj)


def waiting_cache_worker(connection, cancelled, job, config, global_config, plan):
    from simpletuner.helpers.training.state_tracker import StateTracker

    StateTracker.set_args(global_config)
    job.status = CacheJobStatus.RUNNING
    job.stage = "waiting"
    with connection:
        connection.send(CacheJobService._job_to_dict(job))
        if not cancelled.wait(30):
            raise TimeoutError("Did not receive cancellation")
        job.status = CacheJobStatus.CANCELLED
        connection.send(CacheJobService._job_to_dict(job))


class TestCacheJobManagement(unittest.TestCase):
    def setUp(self):
        self.service = CacheJobService(broadcast_fn=MagicMock())
        # Reset class-level state
        CacheJobService._active_job = None
        CacheJobService._thread = None

    def test_get_active_status_when_idle(self):
        self.assertIsNone(self.service.get_active_status())

    def test_spawned_worker_has_isolated_state_and_receives_cancellation(self):
        import threading

        from simpletuner.helpers.training.state_tracker import StateTracker

        ready = threading.Event()
        before = StateTracker.get_args()
        self.service._broadcast_fn = lambda data, event_type: ready.set() if data["stage"] == "waiting" else None
        with (
            patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.get_scan_service") as scan,
            patch.object(self.service, "_check_training_running", return_value=False),
            patch(
                "simpletuner.simpletuner_sdk.server.services.cache_job_service._execute_cache_process", waiting_cache_worker
            ),
        ):
            scan.return_value.get_active_status.return_value = None
            job = self.service.start_cache_job("images", "vae", {"id": "images"}, {"tenant": "worker"})
            try:
                self.assertTrue(ready.wait(30))
                self.assertIs(StateTracker.get_args(), before)
            finally:
                self.service.cancel()
                self.service._thread.join(30)
            self.assertFalse(self.service._thread.is_alive())
            self.assertIsNone(self.service.get_active_status())
            self.assertEqual(self.service.get_job_status(job)["status"], "cancelled")

    def test_spawned_worker_propagates_terminal_error(self):
        with (
            patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.get_scan_service") as scan,
            patch.object(self.service, "_check_training_running", return_value=False),
        ):
            scan.return_value.get_active_status.return_value = None
            job = self.service.start_cache_job("images", "invalid", {"id": "images"}, {})
            self.service._thread.join(30)
            self.assertFalse(self.service._thread.is_alive())
            self.assertIn("Unknown cache type", self.service.get_job_status(job)["error"])
            self.assertEqual(self.service.get_job_status(job)["status"], "failed")

    def test_cancel_when_idle(self):
        self.assertFalse(self.service.cancel())

    def test_job_to_dict(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
            stage="Processing",
            current=10,
            total=100,
        )
        d = CacheJobService._job_to_dict(job)
        self.assertEqual(d["job_id"], "abc")
        self.assertEqual(d["dataset_id"], "ds1")
        self.assertEqual(d["cache_type"], "vae")
        self.assertEqual(d["status"], "running")
        self.assertEqual(d["stage"], "Processing")
        self.assertEqual(d["current"], 10)
        self.assertEqual(d["total"], 100)
        self.assertIsNone(d["error"])

    def test_mutual_exclusion_with_scan(self):
        """start_cache_job should reject when scan is active."""
        with patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.get_scan_service") as mock_scan:
            mock_scan.return_value.get_active_status.return_value = {
                "active": True,
                "dataset_id": "x",
            }
            with self.assertRaises(RuntimeError) as ctx:
                self.service.start_cache_job("ds1", "vae", {"id": "ds1"}, {})
            self.assertIn("scan", str(ctx.exception).lower())

    def test_cancel_active_job(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
        )
        CacheJobService._active_job = job
        self.assertTrue(self.service.cancel())
        self.assertEqual(job.status, CacheJobStatus.CANCELLED)
        self.assertIsNotNone(job.finished_at)

    def test_cancel_broadcasts_cancelled_status(self):
        mock_fn = MagicMock()
        service = CacheJobService(broadcast_fn=mock_fn)
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
        )
        CacheJobService._active_job = job
        service.cancel()
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        self.assertEqual(call_kwargs.kwargs["event_type"], "dataset_cache")
        self.assertEqual(call_kwargs.kwargs["data"]["status"], "cancelled")

    def test_get_active_status_when_running(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
        )
        CacheJobService._active_job = job
        status = self.service.get_active_status()
        self.assertIsNotNone(status)
        self.assertEqual(status["job_id"], "abc")
        self.assertEqual(status["status"], "running")

    def test_get_active_status_after_completion(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.COMPLETED,
        )
        CacheJobService._active_job = job
        self.assertIsNone(self.service.get_active_status())

    def test_get_job_status(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="text_embeds",
            status=CacheJobStatus.LOADING_MODEL,
        )
        CacheJobService._active_job = job
        status = self.service.get_job_status("abc")
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "loading_model")

    def test_get_job_status_wrong_id(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
        )
        CacheJobService._active_job = job
        self.assertIsNone(self.service.get_job_status("xyz"))

    def test_reject_concurrent_jobs(self):
        job = CacheJob(
            job_id="abc",
            dataset_id="ds1",
            cache_type="vae",
            status=CacheJobStatus.RUNNING,
        )
        CacheJobService._active_job = job

        with (
            patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.get_scan_service") as mock_scan,
            patch.object(CacheJobService, "_check_training_running", return_value=False),
        ):
            mock_scan.return_value.get_active_status.return_value = None
            with self.assertRaises(RuntimeError) as ctx:
                self.service.start_cache_job("ds2", "text_embeds", {"id": "ds2"}, {})
            self.assertIn("already in progress", str(ctx.exception))

    def test_mutual_exclusion_with_training(self):
        """start_cache_job should reject when training is active."""
        with (
            patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.get_scan_service") as mock_scan,
            patch.object(CacheJobService, "_check_training_running", return_value=True),
        ):
            mock_scan.return_value.get_active_status.return_value = None
            with self.assertRaises(RuntimeError) as ctx:
                self.service.start_cache_job("ds1", "vae", {"id": "ds1"}, {})
            self.assertIn("training", str(ctx.exception).lower())


class TestStateTrackerSetup(unittest.TestCase):
    """Verify that cache execute methods set StateTracker globals correctly."""

    def test_accelerator_stub_has_is_local_main_process(self):
        """Standalone providers use the local-main-process flag during setup."""
        stub = _CacheAcceleratorStub()
        self.assertTrue(stub.is_local_main_process)

    def test_execute_sets_and_restores_accelerator(self):
        """_execute_text_embed_cache must set StateTracker.accelerator and restore it."""
        from simpletuner.helpers.training.state_tracker import StateTracker

        original = StateTracker.get_accelerator()
        StateTracker.set_accelerator(None)
        try:
            service = CacheJobService(broadcast_fn=MagicMock())
            job = CacheJob(
                job_id="test",
                dataset_id="ds1",
                cache_type="text_embeds",
                status=CacheJobStatus.PENDING,
            )

            # Patch _load_model to capture the accelerator state and bail out
            captured = {}

            def capture_and_bail(self_svc, g_cfg, args, accelerator):
                captured["accelerator"] = StateTracker.get_accelerator()
                raise RuntimeError("short-circuit after accelerator set")

            with patch.object(CacheJobService, "_load_model", capture_and_bail):
                try:
                    service._execute_text_embed_cache(job, {"id": "ds1"}, {})
                except RuntimeError:
                    pass

            # Accelerator was set to the stub inside execute
            self.assertIsNotNone(captured.get("accelerator"))
            self.assertTrue(captured["accelerator"].is_local_main_process)
            # Restored to None afterwards
            self.assertIsNone(StateTracker.get_accelerator())
        finally:
            StateTracker.set_accelerator(original)


class _LegacyPreparationMetadata(SimpleNamespace):
    _preparation_source = None
    close = MetadataBackend.close
    _open_accepted_groups = MetadataBackend._open_accepted_groups
    _preparation_ownership = MetadataBackend._preparation_ownership
    open_preparation_view = MetadataBackend.open_preparation_view


class TestImageArtifactPreparation(unittest.TestCase):
    def test_vae_and_conditioning_jobs_use_the_configured_artifact_store(self):
        service = CacheJobService()
        source = Mock(type="webshart")
        store = Mock(type="aws")
        global_config = {"model_family": "sdxl"}
        for role, kind, cache_dir_key, implementation in (
            ("image_embeds", "vae", "cache_dir_vae", "simpletuner.helpers.caching.vae.VAECache"),
            (
                "conditioning_image_embeds",
                "conditioning",
                "cache_dir_conditioning_image_embeds",
                "simpletuner.helpers.caching.image_embed.ImageEmbedCache",
            ),
        ):
            with self.subTest(role=role):
                dataset = {"id": "source", "type": "webshart", role: "artifacts", cache_dir_key: "cache/images"}
                entry = {
                    "id": "artifacts",
                    "type": "aws",
                    "dataset_type": role,
                    "aws_bucket_name": "artifacts",
                    "aws_endpoint_url": "https://storage.example",
                    "aws_access_key_id": "example",
                    "aws_secret_access_key": "example",
                }
                metadata = Mock()
                runtime = service._text_dataset({**dataset, "instance_data_dir": "source"}, source, metadata)
                job = CacheJob("job", "source", kind)
                with (
                    patch.object(service, "_load_model", return_value=Mock()),
                    patch.object(service, "_create_dataset_runtime", return_value=runtime),
                    patch.object(service, "_prepare_image_artifacts"),
                    patch("simpletuner.helpers.data_backend.builders.aws.S3DataBackend", return_value=store) as s3,
                    patch(implementation) as cache,
                ):
                    service._run_cache_job(job, dataset, global_config, [dataset, entry])

                self.assertEqual(job.status, CacheJobStatus.COMPLETED, job.error)
                self.assertEqual(s3.call_args.kwargs["bucket_name"], "artifacts")
                self.assertIs(cache.call_args.kwargs["image_data_backend"], source)
                self.assertIs(cache.call_args.kwargs["cache_data_backend"], store)
                self.assertEqual(cache.call_args.kwargs["cache_dir"], "cache/images")
                metadata.close.assert_called_once_with()


class TestStreamingTextPreparation(unittest.TestCase):

    def test_artifact_store_is_independent_of_source(self):
        from pathlib import Path

        service = CacheJobService()
        source = {"id": "source", "type": "webshart", "text_embeds": "embeddings"}
        store = {"id": "embeddings", "type": "local", "dataset_type": "text_embeds"}
        with TemporaryDirectory() as directory:
            store["cache_dir"] = directory
            args = _CacheArgsNamespace({})
            accelerator = _CacheAcceleratorStub()
            default = dict(store, id="default-store", default=True)
            config, backend, prefix = service._text_embed_store(source, [source, default, store], {}, args, accelerator)
            self.assertEqual((backend.type, backend.id, Path(prefix)), ("local", store["id"], Path(directory)))
            aws = dict(
                store,
                type="aws",
                aws_bucket_name="embeddings",
                aws_data_prefix="cache/text",
                aws_endpoint_url="https://storage.example",
                aws_access_key_id="example",
                aws_secret_access_key="example",
            )
            with patch("simpletuner.helpers.data_backend.builders.aws.S3DataBackend") as create:
                config, backend, prefix = service._text_embed_store(source, [source, default, aws], {}, args, accelerator)
            self.assertEqual(prefix, "cache/text")
            self.assertIs(backend, create.return_value)
            self.assertEqual(create.call_args.kwargs["bucket_name"], "embeddings")
            with self.assertRaisesRegex(ValueError, "not found"):
                service._text_embed_store(source, [source, default], {}, args, accelerator)
            with self.assertRaisesRegex(ValueError, "writes disabled"):
                service._text_embed_store(source, [source, dict(store, text_cache_disable=True)], {}, args, accelerator)

    def test_job_streams_accepted_samples_without_inventory_discovery(self):
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        service = CacheJobService(broadcast_fn=MagicMock())
        metadata = _LegacyPreparationMetadata(
            read_only=False,
            read_cache=lambda: {"1.0": ["accepted.jpg", "also-accepted.jpg"]},
            get_metadata_by_filepath=lambda ref: {},
        )
        inventory = MagicMock()
        inventory.read.side_effect = AssertionError("source file inventory")
        data_backend = SimpleNamespace(id="stream-test", source_files=inventory)
        model = SimpleNamespace(
            text_encoders=[],
            tokenizers=[],
            load_text_encoder=MagicMock(),
            load_text_tokenizer=MagicMock(),
            text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION,
        )
        captured = []

        def consume(records, *, batch_size, progress_callback):
            self.assertEqual(batch_size, 2)
            self.assertNotIsInstance(records, list)
            progress_callback(0)
            for index, record in enumerate(records, 1):
                captured.append(record)
                progress_callback(index)
            return len(captured)

        cache = SimpleNamespace(
            read_batch_size=2,
            text_encoder_batch_size=1,
            encode_dropout_caption=MagicMock(),
            precompute_embeddings=consume,
            close=MagicMock(),
        )
        job = CacheJob(job_id="stream-job", dataset_id="stream-test", cache_type="text_embeds")
        with TemporaryDirectory() as cache_dir:
            config = {"id": "stream-test", "caption_strategy": "filename", "cache_dir_text": cache_dir}
            runtime = service._text_dataset(config, data_backend, metadata)
            with (
                patch.object(service, "_load_model", return_value=model),
                patch.object(service, "_create_dataset_runtime", return_value=runtime),
                patch.object(service, "_cleanup_gpu"),
                patch("simpletuner.helpers.caching.text_embeds.TextEmbeddingCache", return_value=cache),
                patch("simpletuner.helpers.data_backend.factory.move_text_encoders"),
            ):
                service._execute_text_embed_cache(job, config, {"model_family": "test"})
        self.assertEqual([record["prompt"] for record in captured], ["accepted", "also-accepted"])
        self.assertEqual((job.current, job.total), (2, 2))
        cache.close.assert_called_once_with()

    def test_standalone_preparation_rejects_rank_local_training_buckets(self):
        records = CacheJobService._iter_prompt_records(
            None,
            CacheJobService._text_dataset(
                {"id": "dataset"},
                None,
                _LegacyPreparationMetadata(
                    read_only=True,
                    cache_file="missing.json",
                    data_backend=Mock(open_read_stream=Mock(side_effect=FileNotFoundError)),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "publish accepted membership"):
            next(records)

    def test_standalone_preparation_requires_an_accepted_sample_view(self):
        records = CacheJobService._iter_prompt_records(
            None,
            CacheJobService._text_dataset(
                {"id": "dataset"}, None, _LegacyPreparationMetadata(read_only=False, read_cache=lambda: {})
            ),
        )
        with self.assertRaisesRegex(ValueError, "scan the dataset"):
            next(records)


class TestCacheJobBroadcast(unittest.TestCase):
    def test_broadcast_called(self):
        mock_fn = MagicMock()
        service = CacheJobService(broadcast_fn=mock_fn)
        service._broadcast("dataset_cache", {"status": "running"})
        mock_fn.assert_called_once_with(data={"status": "running"}, event_type="dataset_cache")

    def test_broadcast_noop_without_fn(self):
        service = CacheJobService(broadcast_fn=None)
        service._broadcast("dataset_cache", {"status": "running"})


if __name__ == "__main__":
    unittest.main()
