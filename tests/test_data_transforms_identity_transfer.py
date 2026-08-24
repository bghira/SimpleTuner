import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from simpletuner.helpers.data_transforms import process_data_transforms
from simpletuner.helpers.data_transforms.identity_transfer import (
    VOICE_TRANSFORM_FORMAT,
    VOICE_TRANSFORM_FORMAT_VERSION,
    IdentityTransferTransform,
    RVCTransformLogger,
)


class TestIdentityTransferTransform(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.source_dir = Path(self.temp_dir) / "source"
        self.source_dir.mkdir()
        (self.source_dir / "sample.flac").write_bytes(b"not real audio")
        self.args = SimpleNamespace(output_dir=str(Path(self.temp_dir) / "output"))
        self.source_backend = {
            "id": "artist-source",
            "type": "local",
            "dataset_type": "audio",
            "instance_data_dir": str(self.source_dir),
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "audio": {"sample_rate": 44100, "channels": 2},
        }

    def _transform(self, config=None, accelerator=None):
        return IdentityTransferTransform(
            global_config=self.args,
            source_backend_config=self.source_backend,
            transform_config=config or {"task": "identity_transfer"},
            accelerator=accelerator,
        )

    def test_process_data_transforms_reuses_generated_cache_as_primary_audio_dataset(self):
        transform = self._transform()
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        fingerprint = transform._fingerprint(normalised)
        generated_dir = Path(normalised["target"]["instance_data_dir"])
        generated_dir.mkdir(parents=True)
        (generated_dir / ".simpletuner_identity_transfer.json").write_text(
            json.dumps(
                {
                    "format": VOICE_TRANSFORM_FORMAT,
                    "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
                    "task": "identity_transfer",
                    "method": "rvc",
                    "fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )

        config = process_data_transforms(
            global_config=self.args,
            data_backend_config=[
                {
                    **self.source_backend,
                    "data_transforms": [{"task": "identity_transfer"}],
                }
            ],
        )

        generated = config[1]
        self.assertEqual(generated["id"], "artist-source_identity_transfer")
        self.assertEqual(generated["dataset_type"], "audio")
        self.assertEqual(generated["generated_by"], "data_transforms")
        self.assertEqual(generated["source_dataset_id"], "artist-source")
        self.assertNotIn("auto_generated", generated)
        self.assertNotIn("metadata_clone_source_id", generated)
        self.assertEqual(generated["audio"]["sample_rate"], 44100)

    def test_identity_transfer_rejects_non_audio_source(self):
        image_source = {**self.source_backend, "dataset_type": "image"}
        with self.assertRaises(ValueError) as context:
            process_data_transforms(
                global_config=self.args,
                data_backend_config=[
                    {
                        **image_source,
                        "data_transforms": [{"task": "identity_transfer"}],
                    }
                ],
            )

        self.assertIn("only supports source dataset_type", str(context.exception))

    def test_rank_shard_splits_inputs_by_accelerator_rank(self):
        accelerator = MagicMock()
        accelerator.process_index = 1
        accelerator.num_processes = 3
        transform = self._transform(accelerator=accelerator)

        self.assertEqual(transform._rank_shard(["a", "b", "c", "d", "e", "f", "g"]), ["b", "e"])

    def test_local_voice_model_artifact_reuses_matching_manifest(self):
        transform = self._transform({"task": "identity_transfer", "id": "voice-transfer"})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        fingerprint = transform._fingerprint(normalised)
        cache_dir = Path(normalised["model"]["cache_dir"])
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.pth").write_bytes(b"model")
        (cache_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": VOICE_TRANSFORM_FORMAT,
                    "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
                    "task": "identity_transfer",
                    "method": "rvc",
                    "fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )

        artifact = transform._resolve_voice_model(
            normalised,
            fingerprint,
            cache_dir,
            RVCTransformLogger(str(Path(self.temp_dir) / "output")),
        )

        self.assertEqual(artifact.model_path, cache_dir / "model.pth")

    def test_missing_voice_model_requires_explicit_training(self):
        transform = self._transform({"task": "identity_transfer", "model": {"train_if_missing": False}})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})

        with self.assertRaises(ValueError) as context:
            transform._resolve_voice_model(
                normalised,
                transform._fingerprint(normalised),
                Path(normalised["model"]["cache_dir"]),
                RVCTransformLogger(str(Path(self.temp_dir) / "output")),
            )

        self.assertIn("train_if_missing is false", str(context.exception))

    def test_logger_writes_local_json_files(self):
        run_logger = RVCTransformLogger(str(Path(self.temp_dir) / "output"))

        run_logger.event("voice-transfer", "voice_model_reused", source="local")
        run_logger.summary("voice-transfer", status="reused_generated_cache")

        self.assertTrue((Path(self.temp_dir) / "output" / "logs" / "rvc" / "training_stats.jsonl").exists())
        summary = json.loads((Path(self.temp_dir) / "output" / "logs" / "rvc" / "summary.json").read_text())
        self.assertEqual(summary["status"], "reused_generated_cache")


if __name__ == "__main__":
    unittest.main()
