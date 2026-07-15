import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service import (
    CheckpointInferenceService,
    CheckpointInferenceServiceError,
)


class TestCheckpointInferenceService(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_dir = Path(self.temporary_directory.name)
        self.config = {
            "--output_dir": str(self.output_dir),
            "--validation_prompt": "configured prompt",
            "--user_prompt_library": "user_prompt_library-portraits.json",
        }
        self.service = CheckpointInferenceService()
        self.service._load_environment = Mock(return_value=(self.config, self.output_dir))

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.process_keeper")
    def test_active_session_prunes_stale_records(self, process_keeper: MagicMock) -> None:
        active = {"session_id": "active", "job_id": "active-job", "environment": "active-env"}
        stale = {"session_id": "stale", "job_id": "stale-job", "environment": "stale-env"}
        self.service._sessions = {"active": active, "stale": stale}
        self.service._environment_sessions = {"active-env": "active", "stale-env": "stale"}
        process_keeper.get_process_status.side_effect = lambda job_id: "running" if job_id == "active-job" else "completed"

        result = self.service.active_session()

        self.assertEqual(result, active)
        self.assertEqual(self.service._sessions, {"active": active})
        self.assertEqual(self.service._environment_sessions, {"active-env": "active"})

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.PromptLibraryService")
    @patch(
        "simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.built_in_prompts",
        {"landscape": "built-in prompt"},
    )
    def test_resolve_prompts_combines_selected_sources(self, prompt_library_service: MagicMock) -> None:
        prompt_library_service.return_value.read_library.return_value = {
            "entries": {
                "portrait": {
                    "prompt": "library prompt",
                    "adapter_strength": 0.7,
                }
            }
        }

        result = self.service.resolve_prompts(
            config=self.config,
            use_configured_prompt=True,
            use_builtin_library=True,
            user_library_filename="user_prompt_library-portraits.json",
            custom_prompts=["custom prompt", "  "],
        )

        self.assertEqual([entry["source"] for entry in result], ["configured", "builtin", "user-library", "custom"])
        self.assertEqual(result[2]["adapter_strength"], 0.7)
        prompt_library_service.return_value.read_library.assert_called_once_with("user_prompt_library-portraits.json")

    def test_start_rejects_keep_loaded_for_multiple_checkpoints(self) -> None:
        with self.assertRaises(CheckpointInferenceServiceError) as context:
            self.service.start(
                environment="test",
                checkpoint_names=["checkpoint-100", "checkpoint-200"],
                use_configured_prompt=True,
                use_builtin_library=False,
                user_library_filename=None,
                custom_prompts=[],
                filename_style="descriptive",
                keep_loaded=True,
                streaming_preview=False,
                idle_timeout_minutes=15,
                settings={},
            )

        self.assertIn("single checkpoint", context.exception.message)

    def test_resolve_prompts_reads_configured_absolute_library(self) -> None:
        library_path = self.output_dir / "legacy_prompt_library.json"
        library_path.write_text(json.dumps({"legacy": "legacy prompt"}), encoding="utf-8")
        config = {**self.config, "--user_prompt_library": str(library_path)}

        result = self.service.resolve_prompts(
            config=config,
            use_configured_prompt=False,
            use_builtin_library=False,
            user_library_filename=library_path.name,
            custom_prompts=[],
        )

        self.assertEqual(result[0]["prompt"], "legacy prompt")
        self.assertEqual(result[0]["source"], "user-library")

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.PromptLibraryService")
    def test_prompt_sources_include_active_environment_inference_defaults(self, prompt_library_service: MagicMock) -> None:
        prompt_library_service.return_value.list_libraries.return_value = []
        self.config.update(
            {
                "validation_num_inference_steps": 24,
                "--validation_guidance": 5.5,
                "--validation_resolution": "512,768x512",
                "--validation_multigpu": "batch-parallel",
                "context_parallel_size": "2",
                "--user_prompt_library": None,
            }
        )

        result = self.service.prompt_sources("test")

        self.assertEqual(
            result["inference_defaults"],
            {
                "num_inference_steps": 24,
                "guidance_scale": 5.5,
                "validation_resolution": "512,768x512",
            },
        )
        self.assertEqual(result["unsupported_multigpu_modes"], ["batch-parallel", "context-parallel"])

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.PromptLibraryService")
    def test_prompt_sources_use_registered_defaults_when_environment_omits_values(
        self, prompt_library_service: MagicMock
    ) -> None:
        prompt_library_service.return_value.list_libraries.return_value = []
        self.config["--user_prompt_library"] = None

        result = self.service.prompt_sources("test")

        self.assertEqual(
            result["inference_defaults"],
            {"num_inference_steps": 30, "guidance_scale": 7.5, "validation_resolution": "256"},
        )
        self.assertEqual(result["unsupported_multigpu_modes"], [])

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.process_keeper")
    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.CheckpointManager")
    def test_start_submits_valid_checkpoint_session(
        self,
        checkpoint_manager: MagicMock,
        process_keeper: MagicMock,
    ) -> None:
        checkpoint_manager.return_value.validate_checkpoint.return_value = (True, "valid")
        process_keeper.get_process_status.return_value = "not_found"

        result = self.service.start(
            environment="test",
            checkpoint_names=["checkpoint-100"],
            use_configured_prompt=True,
            use_builtin_library=False,
            user_library_filename=None,
            custom_prompts=["custom prompt"],
            filename_style="content-hash",
            keep_loaded=True,
            streaming_preview=True,
            idle_timeout_minutes=7,
            settings={"seed": 123, "validation_resolution": "512,768x512"},
        )

        self.assertEqual(result["status"], "loading")
        self.assertEqual(result["prompt_count"], 2)
        self.assertEqual(result["generation_count"], 4)
        payload = process_keeper.submit_job.call_args.args[2]
        self.assertEqual(payload["checkpoint_names"], ["checkpoint-100"])
        self.assertEqual(payload["filename_style"], "content-hash")
        self.assertTrue(payload["streaming_preview"])
        self.assertEqual(payload["settings"]["validation_resolution"], "512,768x512")
        self.assertEqual(payload["idle_timeout_seconds"], 420)
        self.assertTrue((self.output_dir / "inference" / result["session_id"]).is_dir())

    def test_start_rejects_invalid_validation_resolution(self) -> None:
        with self.assertRaisesRegex(CheckpointInferenceServiceError, "Invalid validation resolution"):
            self.service.start(
                environment="test",
                checkpoint_names=["checkpoint-100"],
                use_configured_prompt=True,
                use_builtin_library=False,
                user_library_filename=None,
                custom_prompts=[],
                filename_style="descriptive",
                keep_loaded=False,
                streaming_preview=False,
                idle_timeout_minutes=15,
                settings={"validation_resolution": "512,wide"},
            )

    def test_history_builds_environment_scoped_media_urls(self) -> None:
        sidecar = self.output_dir / "inference" / "session-one" / "checkpoint-100" / "output.png.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.with_suffix("").write_bytes(b"generated output")
        sidecar.write_text(
            json.dumps(
                {
                    "session_id": "session-one",
                    "checkpoint": "checkpoint-100",
                    "filename": "output.png",
                    "prompt": "a prompt",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        result = self.service.history("environment with spaces", page=1, page_size=24)

        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["items"][0]["media_url"],
            "/api/checkpoints/inference/media/session-one/checkpoint-100/output.png?environment=environment%20with%20spaces",
        )
        self.assertEqual(result["items"][0]["media_path"], "session-one/checkpoint-100/output.png")

    def test_history_ignores_orphaned_sidecars(self) -> None:
        sidecar = self.output_dir / "inference" / "session-one" / "checkpoint-100" / "missing.png.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({"created_at": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")

        result = self.service.history("test", page=1, page_size=24)

        self.assertEqual(result["items"], [])
        self.assertEqual(result["total"], 0)

    def test_delete_history_removes_selected_media_and_sidecars(self) -> None:
        session_dir = self.output_dir / "inference" / "session-one"
        (session_dir / "session.json").parent.mkdir(parents=True)
        (session_dir / "session.json").write_text('{"status": "completed"}', encoding="utf-8")
        media_paths = []
        for filename in ("first.png", "second.mp4"):
            media = session_dir / "checkpoint-100" / filename
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(b"generated output")
            media.with_suffix(f"{media.suffix}.json").write_text(json.dumps({"filename": filename}), encoding="utf-8")
            media_paths.append(media.relative_to(self.output_dir / "inference").as_posix())

        result = self.service.delete_history("test", media_paths)

        self.assertEqual(result["deleted_count"], 2)
        for media_path in media_paths:
            media = self.output_dir / "inference" / media_path
            self.assertFalse(media.exists())
            self.assertFalse(media.with_suffix(f"{media.suffix}.json").exists())
        self.assertTrue((session_dir / "session.json").is_file())

    def test_delete_history_validates_entire_batch_before_deleting(self) -> None:
        media = self.output_dir / "inference" / "session-one" / "checkpoint-100" / "first.png"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"generated output")
        sidecar = media.with_suffix(".png.json")
        sidecar.write_text('{"filename": "first.png"}', encoding="utf-8")

        with self.assertRaises(CheckpointInferenceServiceError) as context:
            self.service.delete_history(
                "test",
                ["session-one/checkpoint-100/first.png", "../outside.png"],
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertTrue(media.is_file())
        self.assertTrue(sidecar.is_file())

    def test_media_path_rejects_traversal(self) -> None:
        outside_file = self.output_dir / "outside.png"
        outside_file.write_bytes(b"png")

        with self.assertRaises(CheckpointInferenceServiceError) as context:
            self.service.media_path("test", "../outside.png")

        self.assertEqual(context.exception.status_code, 404)

    def test_media_path_allows_generated_outputs_and_streaming_preview(self) -> None:
        generated = self.output_dir / "inference" / "session-one" / "checkpoint-100" / "output.png"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(b"png")
        generated.with_suffix(".png.json").write_text("{}", encoding="utf-8")
        preview = generated.parents[1] / "preview.png"
        preview.write_bytes(b"png")
        preview.with_suffix(".json").write_text("{}", encoding="utf-8")

        self.assertEqual(
            self.service.media_path("test", "session-one/checkpoint-100/output.png"),
            generated.resolve(),
        )
        self.assertEqual(self.service.media_path("test", "session-one/preview.png"), preview.resolve())

    def test_media_path_rejects_internal_and_untracked_files(self) -> None:
        session_dir = self.output_dir / "inference" / "session-one"
        cache_file = session_dir / ".prompt_cache" / "embedding.png"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(b"internal")
        cache_file.with_suffix(".png.json").write_text("{}", encoding="utf-8")
        (session_dir / "session.json").write_text("{}", encoding="utf-8")
        untracked = session_dir / "checkpoint-100" / "untracked.png"
        untracked.parent.mkdir()
        untracked.write_bytes(b"png")

        for media_path in (
            "session-one/session.json",
            "session-one/.prompt_cache/embedding.png",
            "session-one/checkpoint-100/untracked.png",
        ):
            with self.subTest(media_path=media_path):
                with self.assertRaises(CheckpointInferenceServiceError) as context:
                    self.service.media_path("test", media_path)
                self.assertEqual(context.exception.status_code, 404)

    def test_status_includes_streaming_preview_and_latest_completed_output(self) -> None:
        session_dir = self.output_dir / "inference" / "session-one"
        output = session_dir / "checkpoint-100" / "output.png"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"completed output")
        output.with_suffix(".png.json").write_text(
            json.dumps(
                {
                    "checkpoint": "checkpoint-100",
                    "filename": "output.png",
                    "prompt": "completed prompt",
                    "media_type": "image",
                    "created_at": "2026-01-01T00:00:02+00:00",
                }
            ),
            encoding="utf-8",
        )
        (session_dir / "preview.png").write_bytes(b"preview")
        (session_dir / "preview.json").write_text(
            json.dumps(
                {
                    "checkpoint": "checkpoint-100",
                    "prompt": "preview prompt",
                    "step_label": "2/20",
                    "media_type": "image",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                }
            ),
            encoding="utf-8",
        )
        (session_dir / "session.json").write_text(
            json.dumps({"session_id": "session-one", "status": "completed", "streaming_preview": True}),
            encoding="utf-8",
        )

        result = self.service.status("test environment", "session-one")

        self.assertEqual(result["latest_output"]["prompt"], "completed prompt")
        self.assertFalse(result["latest_output"]["streaming"])
        self.assertEqual(result["preview"]["step_label"], "2/20")
        self.assertTrue(result["preview"]["streaming"])
        self.assertIn("environment=test%20environment", result["preview"]["media_url"])
        self.assertIn("&v=", result["preview"]["media_url"])

    def test_status_discards_completed_session_record(self) -> None:
        session_id = "session-one"
        session_dir = self.output_dir / "inference" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps({"session_id": session_id, "status": "completed"}),
            encoding="utf-8",
        )
        self.service._sessions[session_id] = {
            "session_id": session_id,
            "job_id": "infer-one",
            "environment": "test",
        }
        self.service._environment_sessions["test"] = session_id

        self.service.status("test", session_id)

        self.assertNotIn(session_id, self.service._sessions)
        self.assertNotIn("test", self.service._environment_sessions)

    @patch("simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service.process_keeper")
    def test_generate_queues_prompt_for_loaded_worker(self, process_keeper: MagicMock) -> None:
        session_id = "session-one"
        session_dir = self.output_dir / "inference" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps({"session_id": session_id, "status": "loaded", "updated_at": "old"}),
            encoding="utf-8",
        )
        self.service._sessions[session_id] = {
            "session_id": session_id,
            "job_id": "infer-one",
            "environment": "test",
        }
        process_keeper.get_process_status.return_value = "running"

        result = self.service.generate(
            environment="test",
            session_id=session_id,
            custom_prompts=["another prompt"],
            filename_style="prompt",
            settings={"seed": 3},
        )

        self.assertEqual(result["status"], "queued")
        process_keeper.send_process_command.assert_called_once()
        state = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "queued")


if __name__ == "__main__":
    unittest.main()
