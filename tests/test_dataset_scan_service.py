import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from PIL import Image

from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService


class TestClearConditioningCache(unittest.TestCase):
    """Verify clear_conditioning_cache only removes auto-generated dirs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetScanService()

        # Set up a cache root with conditioning_data dirs
        self.cache_root = Path(self.tmpdir) / "cache"
        self.cond_root = self.cache_root / "conditioning_data"

        # Auto-generated conditioning dir (should be cleared)
        self.auto_dir = self.cond_root / "myds_conditioning_canny"
        self.auto_dir.mkdir(parents=True)
        (self.auto_dir / "img001.png").write_bytes(b"fake")

        # Manually declared conditioning dir (should NOT be cleared)
        self.manual_dir = self.cond_root / "my_manual_cond"
        self.manual_dir.mkdir(parents=True)
        (self.manual_dir / "img001.png").write_bytes(b"fake")

        # VAE dir so _derive works
        self.vae_dir = self.cache_root / "vae" / "sdxl" / "myds"
        self.vae_dir.mkdir(parents=True)

    def _dataset_config(self):
        return {
            "id": "myds",
            "type": "local",
            "instance_data_dir": str(self.tmpdir),
            "cache_dir_vae": str(self.vae_dir),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
            "conditioning_data": ["my_manual_cond"],
        }

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_clears_auto_generated_only(self, _mock):
        cleared = self.service.clear_conditioning_cache(self._dataset_config(), {})
        self.assertEqual(len(cleared), 1)
        self.assertIn("myds_conditioning_canny", cleared[0])
        self.assertFalse(self.auto_dir.exists())
        self.assertTrue(self.manual_dir.exists())

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_no_generators_returns_empty(self, _mock):
        config = self._dataset_config()
        del config["conditioning"]
        cleared = self.service.clear_conditioning_cache(config, {})
        self.assertEqual(cleared, [])

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_nonexistent_dir_skipped(self, _mock):
        # Remove the auto-generated dir so it doesn't exist
        import shutil

        shutil.rmtree(self.auto_dir)
        cleared = self.service.clear_conditioning_cache(self._dataset_config(), {})
        self.assertEqual(cleared, [])


class TestDatasetScanServiceOutputDir(unittest.TestCase):
    def test_relative_scan_output_dir_is_resolved_under_webui_output_root(self):
        with patch.object(
            DatasetScanService,
            "_get_resolved_webui_output_dir",
            return_value="/srv/simpletuner/output",
        ):
            resolved = DatasetScanService._resolve_scan_output_dir("output/scan")

        self.assertEqual(resolved, "/srv/simpletuner/output/scan")

    def test_absolute_scan_output_dir_is_used_as_is(self):
        with patch.object(
            DatasetScanService,
            "_get_resolved_webui_output_dir",
            return_value="/srv/simpletuner/output",
        ):
            resolved = DatasetScanService._resolve_scan_output_dir("/var/tmp/simpletuner-scan")

        self.assertEqual(resolved, "/var/tmp/simpletuner-scan")


def wait_for_scan_cancel(connection, cancelled, *_):
    with connection:
        connection.send({"current": 1, "total": 2})
        if not cancelled.wait(timeout=20):
            raise RuntimeError("The scan did not receive cancellation.")


def reject_scan(connection, *_):
    with connection:
        connection.send({"error": "replacement scan failed"})


class TestDatasetScanProcess(unittest.TestCase):
    def test_scan_uses_a_scoped_runtime_without_setting_dataset_registry_config(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.training.state_tracker import StateTracker

        runtime = Mock()
        progress = Mock()
        with (
            patch.object(DatasetRuntime, "from_config", return_value=runtime),
            patch.object(StateTracker, "set_args"),
            patch.object(StateTracker, "set_accelerator"),
            patch.object(StateTracker, "set_data_backend_config", side_effect=AssertionError("global dataset settings")),
        ):
            DatasetScanService()._execute_scan({"id": "dataset", "type": "local"}, {}, True, progress)
        runtime.prepare_view.assert_called_once_with(force_rescan=True, progress_callback=progress)
        runtime.close.assert_called_once_with()

    def test_scan_preserves_live_registration_of_same_id(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        service = DatasetScanService()
        runtime = object()
        training_args = StateTracker.get_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (128, 128)).save(root / "image.png")
            vae_dir = root / "latents"
            vae_dir.mkdir()
            latent = vae_dir / ("0" * 64 + ".pt")
            latent.touch()
            other_file = vae_dir / "existing.pt"
            other_file.touch()
            config = {
                "id": "example",
                "type": "local",
                "instance_data_dir": str(root),
                "crop": False,
                "resolution": 128,
                "resolution_type": "pixel",
                "cache_dir_vae": str(vae_dir),
            }
            with (
                patch.object(StateTracker, "datasets", {"example": runtime}),
                patch.object(StateTracker, "_pending_dataset_configs", {}),
            ):
                job_id = service.scan_dataset(
                    "example", config, {"output_dir": str(root / "output")}, force_rescan=True, clear_vae_cache=True
                )
                service._scan_thread.join(timeout=45)
                self.assertFalse(service._scan_thread.is_alive())
                result = service.get_scan_status(job_id)
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(result["current"], 1)
                self.assertIs(StateTracker.get_dataset("example"), runtime)
                self.assertEqual(StateTracker._pending_dataset_configs, {})
                self.assertIs(StateTracker.get_args(), training_args)
                self.assertTrue((root / "output" / "all_image_files_example.json").is_file())
                self.assertFalse(latent.exists())
                self.assertTrue(other_file.is_file())

    def test_scan_binds_embedded_audio_to_video_from_saved_plan(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        service = DatasetScanService()
        video = {"id": "video", "type": "local", "dataset_type": "video"}
        audio = {
            "id": "audio",
            "type": "local",
            "dataset_type": "audio",
            "audio": {"source_from_video": True, "source_dataset_id": "video"},
        }
        audio_runtime, video_runtime = Mock(), Mock()
        audio_runtime.prepare_view.side_effect = lambda **_: self.assertIs(audio_runtime.source_dataset, video_runtime)
        with patch.object(DatasetRuntime, "from_config", side_effect=[audio_runtime, video_runtime]) as open_dataset:
            service._execute_scan(audio, {}, False, Mock(), dataset_plan=[audio, video])
        self.assertEqual(open_dataset.call_args.args[0], video)
        self.assertEqual(open_dataset.call_args.kwargs, {"preparation_only": True})
        audio_runtime.close.assert_called_once_with()
        video_runtime.close.assert_called_once_with()

    def test_latent_clear_checks_configured_store_and_source(self):
        service = DatasetScanService()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent = root / ("0" * 64 + ".pt")
            latent.touch()
            config = {"id": "video", "instance_data_dir": str(root), "cache_dir_vae": str(root)}
            with self.assertRaisesRegex(ValueError, "overlaps a dataset source"):
                service.clear_vae_cache(config, {})
            config["image_embeds"] = "remote"
            plan = [config, {"id": "remote", "type": "aws", "dataset_type": "image_embeds"}]
            with self.assertRaisesRegex(ValueError, "Local artifact store"):
                service.scan_dataset("video", config, force_rescan=True, clear_vae_cache=True, dataset_plan=plan)
            self.assertTrue(latent.exists())

    def test_failed_scan_does_not_clear_existing_artifacts(self):
        service = DatasetScanService()
        with (
            patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service._execute_scan_process", reject_scan),
            patch.object(service, "clear_vae_cache") as clear_vae,
            patch.object(service, "clear_conditioning_cache") as clear_conditioning,
        ):
            job_id = service.scan_dataset(
                "scan", {"id": "scan"}, force_rescan=True, clear_vae_cache=True, clear_conditioning_cache=True
            )
            service._scan_thread.join(timeout=25)
            self.assertFalse(service._scan_thread.is_alive())
            self.assertEqual(service.get_scan_status(job_id)["status"], "failed")
            clear_vae.assert_not_called()
            clear_conditioning.assert_not_called()

    def test_cancellation_reaches_the_running_process(self):
        progress = Event()
        events = []

        def broadcast(*, event_type, data):
            events.append(data)
            if data.get("current") == 1:
                progress.set()

        service = DatasetScanService(broadcast_fn=broadcast)
        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.dataset_scan_service._execute_scan_process",
                wait_for_scan_cancel,
            ),
            patch.object(service, "clear_vae_cache") as clear_vae,
        ):
            job_id = service.scan_dataset(
                "scan", {"id": "scan"}, force_rescan=True, clear_vae_cache=True, plan_scope="current"
            )
            try:
                self.assertTrue(progress.wait(timeout=20))
                self.assertFalse(service.cancel_scan(job_id=job_id, plan_scope="previous"))
                self.assertFalse(service.cancel_scan(job_id="stale", plan_scope="current"))
                self.assertTrue(service.cancel_scan(job_id=job_id, plan_scope="current"))
            finally:
                service._scan_thread.join(timeout=25)
            self.assertFalse(service._scan_thread.is_alive())
            clear_vae.assert_not_called()
        self.assertEqual(service.get_scan_status(job_id)["status"], "cancelled")
        self.assertEqual([event["status"] for event in events[-2:]], ["cancelling", "cancelled"])
        self.assertTrue(all(event["plan_scope"] == "current" and event["job_id"] == job_id for event in events))

    def test_queue_publishes_terminal_status_even_after_failed_jobs(self):
        events = []
        service = DatasetScanService(broadcast_fn=lambda **event: events.append(event))
        with patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service._execute_scan_process", reject_scan):
            queue_id = service.scan_all([{"id": "first"}, {"id": "second"}], plan_scope="current")
            service._scan_thread.join(timeout=30)
        self.assertFalse(service._scan_thread.is_alive())
        status = service.get_queue_status(queue_id)
        self.assertEqual(status["plan_scope"], "current")
        self.assertFalse(status["active"])
        self.assertEqual(status["completed"], 0)
        self.assertEqual([job["queue_index"] for job in status["datasets"].values()], [0, 1])
        self.assertEqual([job["status"] for job in status["datasets"].values()], ["failed", "failed"])
        self.assertEqual(events[-1], {"event_type": "dataset_scan_queue", "data": status})


if __name__ == "__main__":
    unittest.main()
