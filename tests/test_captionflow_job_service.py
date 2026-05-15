from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.captionflow_job_service import (
    HUGGINGFACE_SOURCE,
    _build_orchestrator_config,
    _discover_python_nvidia_library_dirs,
    _prepare_captionflow_import_shims,
    _prepare_captionflow_subprocess_env,
    _prepare_cuda_library_shims,
    _process_failure_message,
    _run_async,
    _stop_orchestrator_before_export,
    _wait_for_captionflow_storage_contents,
    _wait_for_orchestrator_ready,
    build_captionflow_runtime_config,
    resolve_captioning_dataset_path,
    resolve_captioning_dataset_source,
)
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType
from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import fetch_job_logs


class CaptionFlowJobServiceTestCase(unittest.TestCase):
    def test_build_runtime_config_uses_dataset_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "images"
            dataset_dir.mkdir()

            config = build_captionflow_runtime_config(
                job_id="job12345",
                dataset_id="images",
                dataset_config={
                    "id": "images",
                    "dataset_type": "image",
                    "instance_data_dir": str(dataset_dir),
                },
                global_config={"--output_dir": str(root / "output")},
                request_config={
                    "model": "test/model",
                    "prompt": "describe",
                    "worker_count": 2,
                    "batch_size": 4,
                    "chunk_size": 128,
                },
            )

        self.assertEqual(config["dataset_path"], str(dataset_dir.resolve()))
        self.assertEqual(config["dataset_export_dir"], str(dataset_dir.resolve()))
        self.assertEqual(config["workspace_dir"], str(root / "output" / "captionflow" / "job12345"))
        self.assertEqual(config["worker_count"], 2)
        self.assertEqual(config["batch_size"], 4)
        self.assertEqual(config["model"], "test/model")

    def test_resolve_dataset_path_rejects_embed_dataset(self) -> None:
        with self.assertRaises(ValueError):
            resolve_captioning_dataset_path(
                {
                    "id": "text",
                    "dataset_type": "text_embeds",
                    "instance_data_dir": "/tmp",
                }
            )

    def test_huggingface_dataset_maps_to_captionflow_source(self) -> None:
        source = resolve_captioning_dataset_source(
            {
                "id": "hf-images",
                "type": "huggingface",
                "dataset_type": "image",
                "dataset_name": "org/dataset",
                "split": "validation",
                "image_column": "image",
                "huggingface": {"dataset_config": "default"},
            }
        )

        self.assertEqual(source["source_type"], HUGGINGFACE_SOURCE)
        self.assertEqual(source["dataset_path"], "org/dataset")
        self.assertEqual(source["dataset_split"], "validation")
        self.assertEqual(source["dataset_config"], "default")
        self.assertEqual(source["dataset_image_column"], "image")
        self.assertEqual(source["dataset_url_column"], "")

    def test_huggingface_runtime_config_has_no_textfile_export_dir(self) -> None:
        config = build_captionflow_runtime_config(
            job_id="hf12345",
            dataset_id="hf-images",
            dataset_config={
                "id": "hf-images",
                "type": "huggingface",
                "dataset_type": "image",
                "dataset_name": "org/dataset",
            },
            global_config={"--output_dir": "/tmp/output"},
            request_config={},
        )

        self.assertIsNone(config["dataset_export_dir"])
        self.assertFalse(config["export_textfiles"])
        self.assertTrue(config["export_jsonl"])

    def test_huggingface_url_image_column_maps_to_url_column(self) -> None:
        source = resolve_captioning_dataset_source(
            {
                "id": "hf-images",
                "type": "huggingface",
                "dataset_type": "image",
                "dataset_name": "org/dataset",
                "image_column": "image_url",
            }
        )

        self.assertEqual(source["dataset_image_column"], "image_url")
        self.assertEqual(source["dataset_url_column"], "image_url")

    def test_orchestrator_config_targets_local_filesystem_processor(self) -> None:
        config = SimpleNamespace(
            workspace_dir="/tmp/simpletuner-captionflow",
            source_type="local_filesystem",
            chunk_size=64,
            dataset_path="/data/images",
            dataset_name="images",
            model="test/model",
            gpu_memory_utilization=0.8,
            batch_size=2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
            prompt="describe",
        )

        result = _build_orchestrator_config(config, 12345, 12346, "token")
        orchestrator = result["orchestrator"]
        self.assertEqual(orchestrator["dataset"]["processor_type"], "local_filesystem")
        self.assertEqual(orchestrator["dataset"]["http_port"], 12346)
        self.assertEqual(orchestrator["vllm"]["model"], "test/model")
        self.assertEqual(orchestrator["auth"]["worker_tokens"][0]["token"], "token")

    def test_orchestrator_config_targets_huggingface_processor(self) -> None:
        config = SimpleNamespace(
            workspace_dir="/tmp/simpletuner-captionflow",
            source_type="huggingface_datasets",
            chunk_size=64,
            dataset_path="org/dataset",
            dataset_name="hf-images",
            dataset_config="default",
            dataset_split="validation",
            dataset_image_column="image",
            dataset_url_column="",
            model="test/model",
            gpu_memory_utilization=0.8,
            batch_size=2,
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
            prompt="describe",
        )

        result = _build_orchestrator_config(config, 12345, 12346, "token")
        dataset = result["orchestrator"]["dataset"]
        self.assertEqual(dataset["processor_type"], "huggingface_datasets")
        self.assertEqual(dataset["dataset_path"], "org/dataset")
        self.assertEqual(dataset["dataset_config"], "default")
        self.assertEqual(dataset["dataset_split"], "validation")
        self.assertEqual(dataset["dataset_url_column"], "")

    def test_raw_orchestrator_config_preserves_stages_and_overrides_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "images"
            dataset_dir.mkdir()
            raw_config = """
orchestrator:
  host: 0.0.0.0
  port: 8765
  ssl:
    cert: /tmp/example.crt
    key: /tmp/example.key
  chunk_size: 1000
  dataset:
    type: huggingface
    processor_type: huggingface_webdataset
    dataset_path: ignored/source
  vllm:
    model: Qwen/Qwen2.5-VL-3B-Instruct
    stages:
      - name: base_caption
        prompts:
          - describe this image in detail
        output_field: caption
      - name: caption_shortening
        prompts:
          - "Please condense this elaborate caption: {caption}"
        output_field: condensed_caption
        requires: ["base_caption"]
  storage:
    data_dir: /tmp/raw-data
    checkpoint_dir: /tmp/raw-checkpoints
    caption_buffer_size: 50
  auth:
    worker_tokens:
      - token: raw-token
        name: raw worker
"""
            runtime_config = build_captionflow_runtime_config(
                job_id="raw12345",
                dataset_id="images",
                dataset_config={
                    "id": "images",
                    "dataset_type": "image",
                    "instance_data_dir": str(dataset_dir),
                },
                global_config={"--output_dir": str(root / "output")},
                request_config={
                    "worker_count": 1,
                    "raw_config": raw_config,
                },
            )

        result = _build_orchestrator_config(SimpleNamespace(**runtime_config), 12345, 12346, "worker-token")
        orchestrator = result["orchestrator"]

        self.assertEqual(orchestrator["host"], "127.0.0.1")
        self.assertEqual(orchestrator["port"], 12345)
        self.assertNotIn("ssl", orchestrator)
        self.assertEqual(orchestrator["dataset"]["processor_type"], "local_filesystem")
        self.assertEqual(orchestrator["dataset"]["dataset_path"], str(dataset_dir.resolve()))
        self.assertEqual(orchestrator["dataset"]["http_port"], 12346)
        self.assertEqual(orchestrator["chunk_size"], 1000)
        self.assertEqual(
            orchestrator["storage"]["data_dir"],
            str(root / "output" / "captionflow" / "raw12345" / "caption_data"),
        )
        self.assertEqual(
            orchestrator["storage"]["checkpoint_dir"],
            str(root / "output" / "captionflow" / "raw12345" / "checkpoints"),
        )
        self.assertEqual(orchestrator["storage"]["caption_buffer_size"], 50)
        self.assertEqual(orchestrator["auth"]["worker_tokens"][0]["token"], "worker-token")
        self.assertEqual(orchestrator["vllm"]["stages"][1]["requires"], ["base_caption"])
        self.assertEqual(orchestrator["vllm"]["stages"][1]["output_field"], "condensed_caption")

    def test_captionflow_import_shim_provides_cv2_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shim_dir = _prepare_captionflow_import_shims(Path(tmpdir))
            shim_file = shim_dir / "cv2.py"

            self.assertTrue(shim_file.exists())
            self.assertIn("imdecode", shim_file.read_text(encoding="utf-8"))

    def test_captionflow_subprocess_env_adds_nvidia_library_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_root = root / "site-packages"
            cuda_lib = site_root / "nvidia" / "cu13" / "lib"
            cuda_lib.mkdir(parents=True)
            (cuda_lib / "libcudart.so.13").write_text("", encoding="utf-8")

            library_dirs = _discover_python_nvidia_library_dirs([site_root])
            shim_dir = _prepare_cuda_library_shims(root / "workspace", library_dirs)

            self.assertEqual(library_dirs, [cuda_lib])
            self.assertIsNotNone(shim_dir)
            self.assertTrue((shim_dir / "libcudart.so").is_symlink())

    def test_captionflow_subprocess_env_includes_cuda_shim_and_ld_library_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _prepare_captionflow_subprocess_env(Path(tmpdir))

            self.assertIn("CAPTIONFLOW_LOG_DIR", env)
            self.assertIn("import_shims", env.get("PYTHONPATH", ""))
            if Path(".venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13").exists():
                self.assertIn("cuda_libs", env.get("LD_LIBRARY_PATH", ""))
                self.assertIn("site-packages/nvidia/cu13/lib", env.get("LD_LIBRARY_PATH", ""))

    def test_process_failure_message_includes_last_error_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "worker.log"
            log_path.write_text(
                "\n".join(
                    [
                        "starting worker",
                        "Traceback (most recent call last):",
                        "  File example.py, line 1, in <module>",
                        "ImportError: libcudart.so.12: cannot open shared object file: No such file or directory",
                    ]
                ),
                encoding="utf-8",
            )

            message = _process_failure_message("worker", 1, log_path)

        self.assertIn("CaptionFlow worker exited with status 1", message)
        self.assertIn("ImportError: libcudart.so.12", message)
        self.assertIn("worker.log", message)

    def test_wait_for_orchestrator_ready_timeout_includes_log_tail(self) -> None:
        class RunningProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "orchestrator.log"
            log_path.write_text("Loading HuggingFace dataset shards\n", encoding="utf-8")
            with self.assertRaisesRegex(TimeoutError, "Loading HuggingFace dataset shards"):
                _wait_for_orchestrator_ready(
                    RunningProcess(),
                    log_path,
                    "127.0.0.1",
                    1,
                    0.01,
                )

    def test_wait_for_orchestrator_ready_uses_captionflow_ready_log_line(self) -> None:
        class RunningProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "orchestrator.log"
            log_path.write_text("INFO     Orchestrator ready for connections\n", encoding="utf-8")

            _wait_for_orchestrator_ready(
                RunningProcess(),
                log_path,
                "127.0.0.1",
                1,
                1,
            )

    def test_wait_for_captionflow_storage_contents_retries_until_columns_exist(self) -> None:
        empty = SimpleNamespace(rows=[], columns=[], output_fields=[], metadata={"message": "No data available"})
        ready = SimpleNamespace(rows=[{"captions": ["ok"]}], columns=["job_id", "captions"], output_fields=["captions"])
        calls = [empty, ready]

        async def fake_load(_storage_dir):
            return calls.pop(0)

        with patch(
            "simpletuner.simpletuner_sdk.server.services.captionflow_job_service._load_captionflow_storage_contents",
            fake_load,
        ):
            contents = _wait_for_captionflow_storage_contents(Path("/tmp/caption-data"), timeout=2)

        self.assertEqual(contents.columns, ["job_id", "captions"])

    def test_wait_for_captionflow_storage_contents_reports_empty_storage(self) -> None:
        empty = SimpleNamespace(rows=[], columns=[], output_fields=[], metadata={"message": "No data available"})

        async def fake_load(_storage_dir):
            return empty

        with patch(
            "simpletuner.simpletuner_sdk.server.services.captionflow_job_service._load_captionflow_storage_contents",
            fake_load,
        ):
            with self.assertRaisesRegex(RuntimeError, "No data available"):
                _wait_for_captionflow_storage_contents(Path("/tmp/caption-data"), timeout=0.01)

    def test_stop_orchestrator_before_export_sends_interrupt_for_checkpoint(self) -> None:
        class RunningProcess:
            returncode = 0

            def __init__(self):
                self.signals = []

            def poll(self):
                return None

            def send_signal(self, value):
                self.signals.append(value)

            def wait(self, timeout):
                return self.returncode

        process = RunningProcess()
        _stop_orchestrator_before_export(process, Path("/tmp/orchestrator.log"), timeout=1)

        self.assertEqual(process.signals, [2])

    def test_captionflow_job_logs_read_orchestrator_and_worker_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            logs_dir = workspace / "logs"
            logs_dir.mkdir()
            (logs_dir / "orchestrator.log").write_text("orchestrator ready\n", encoding="utf-8")
            (logs_dir / "worker-1.log").write_text("worker captioned batch 1\n", encoding="utf-8")
            job = SimpleNamespace(
                job_id="caption123",
                job_type=JobType.LOCAL,
                provider="captionflow",
                metadata={"handler": "captionflow"},
                output_url=str(workspace),
            )

            logs = _run_async(fetch_job_logs(job))

        self.assertIn("logs/orchestrator.log", logs)
        self.assertIn("orchestrator ready", logs)
        self.assertIn("logs/worker-1.log", logs)
        self.assertIn("worker captioned batch 1", logs)


if __name__ == "__main__":
    unittest.main()
