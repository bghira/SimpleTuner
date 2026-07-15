import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.inference import CheckpointInferenceRuntime, run_checkpoint_inference


class _WorkerConfig:
    def __init__(self, payload, commands=None):
        self.__dict__.update(payload)
        self._commands = list(commands or [])

    def should_abort(self):
        return False

    def consume_process_command(self, timeout=0.1):
        return self._commands.pop(0) if self._commands else {"command": "inference_unload"}


class TestCheckpointInferenceWorker(unittest.TestCase):
    def test_flush_embed_cache_times_out_instead_of_blocking(self) -> None:
        runtime = object.__new__(CheckpointInferenceRuntime)
        thread = MagicMock()
        thread.is_alive.return_value = True
        runtime.embed_cache = SimpleNamespace(process_write_batches=True, batch_write_thread=thread)

        with patch("simpletuner.inference.logger.warning") as warning:
            with self.assertRaisesRegex(RuntimeError, "Timed out while flushing"):
                runtime._flush_embed_cache()

        thread.join.assert_called_once_with(timeout=CheckpointInferenceRuntime.EMBED_CACHE_FLUSH_TIMEOUT_SECONDS)
        warning.assert_called_once_with(
            "Timed out after %s seconds while flushing the inference prompt cache.",
            CheckpointInferenceRuntime.EMBED_CACHE_FLUSH_TIMEOUT_SECONDS,
        )

    def test_apply_settings_updates_validation_resolutions(self) -> None:
        runtime = object.__new__(CheckpointInferenceRuntime)
        runtime.trainer = SimpleNamespace(
            config=SimpleNamespace(
                validation_seed=42,
                validation_num_inference_steps=30,
                validation_guidance=7.5,
                validation_resolution="256",
                model_flavour="",
            ),
            validation=SimpleNamespace(validation_resolutions=[(256, 256)]),
        )

        seed = runtime._apply_settings(
            {
                "seed": 7,
                "num_inference_steps": 20,
                "guidance_scale": 4.5,
                "validation_resolution": "512,768x512",
            }
        )

        self.assertEqual(seed, 7)
        self.assertEqual(runtime.trainer.config.validation_resolution, "512,768x512")
        self.assertEqual(runtime.trainer.validation.validation_resolutions, [(512, 512), (768, 512)])

    def test_streaming_preview_writes_current_frame_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = object.__new__(CheckpointInferenceRuntime)
            runtime.checkpoint_name = "checkpoint-100"
            runtime.session_dir = Path(temporary_directory) / "session-one"
            image = Image.new("RGB", (8, 8), color=(10, 20, 30))

            runtime._write_preview(
                structured_data={
                    "body": "preview prompt",
                    "data": {"prompt": "preview prompt", "step": 2, "step_label": "2/20"},
                },
                images=[image],
                videos=None,
            )

            self.assertTrue((runtime.session_dir / "preview.png").is_file())
            metadata = json.loads((runtime.session_dir / "preview.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["checkpoint"], "checkpoint-100")
            self.assertEqual(metadata["step_label"], "2/20")

    def test_content_hash_filename_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = object.__new__(CheckpointInferenceRuntime)
            runtime.checkpoint_name = "checkpoint-100"
            runtime.session_dir = Path(temporary_directory) / "session-one"
            runtime.trainer = SimpleNamespace(config=SimpleNamespace(framerate=16))
            image = Image.new("RGB", (8, 8), color=(10, 20, 30))

            metadata = runtime._save_media(
                image,
                prompt="a test prompt",
                shortname="custom_001",
                seed=42,
                style="content-hash",
                index=0,
                settings={"seed": 42},
            )

            output_path = runtime.session_dir / "checkpoint-100" / metadata["filename"]
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.stem, hashlib.sha256(output_path.read_bytes()).hexdigest())
            sidecar = output_path.with_suffix(".png.json")
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["prompt"], "a test prompt")

    @patch("simpletuner.inference.CheckpointInferenceRuntime")
    def test_batch_worker_unloads_between_checkpoints(self, runtime_class: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory) / "session-one"
            session_dir.mkdir()
            runtimes = [MagicMock(), MagicMock()]
            runtime_class.side_effect = runtimes
            for runtime in runtimes:
                runtime.generate.side_effect = lambda entries, progress_callback, **kwargs: [
                    progress_callback(len(entries), len(entries), entries[-1]["prompt"])
                ]
            config = _WorkerConfig(
                {
                    "session_dir": str(session_dir),
                    "job_id": "infer-one",
                    "environment": "test",
                    "checkpoint_names": ["checkpoint-100", "checkpoint-200"],
                    "prompt_entries": [{"shortname": "one", "prompt": "prompt one"}],
                    "filename_style": "descriptive",
                    "settings": {},
                    "keep_loaded": False,
                    "trainer_config": {},
                }
            )

            result = run_checkpoint_inference(config)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_prompts"], 2)
            self.assertEqual(runtime_class.call_count, 2)
            runtimes[0].close.assert_called_once()
            runtimes[1].close.assert_called_once()

    @patch("simpletuner.inference.CheckpointInferenceRuntime")
    def test_persistent_worker_processes_custom_prompt_before_unload(self, runtime_class: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory) / "session-one"
            session_dir.mkdir()
            runtime = runtime_class.return_value
            runtime.generate.side_effect = lambda entries, progress_callback, **kwargs: [
                progress_callback(len(entries), len(entries), entries[-1]["prompt"])
            ]
            config = _WorkerConfig(
                {
                    "session_dir": str(session_dir),
                    "job_id": "infer-one",
                    "environment": "test",
                    "checkpoint_names": ["checkpoint-100"],
                    "prompt_entries": [{"shortname": "initial", "prompt": "initial prompt"}],
                    "filename_style": "descriptive",
                    "settings": {},
                    "keep_loaded": True,
                    "streaming_preview": True,
                    "trainer_config": {},
                },
                commands=[
                    {
                        "command": "inference_generate",
                        "data": {
                            "prompt_entries": [{"shortname": "custom", "prompt": "custom prompt"}],
                            "filename_style": "prompt",
                            "settings": {"seed": 7},
                        },
                    },
                    {"command": "inference_unload"},
                ],
            )

            result = run_checkpoint_inference(config)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_prompts"], 2)
            self.assertEqual(runtime.generate.call_count, 2)
            self.assertEqual(runtime.generate.call_args_list[1].kwargs["filename_style"], "prompt")
            self.assertTrue(runtime_class.call_args.kwargs["streaming_preview"])
            runtime.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
