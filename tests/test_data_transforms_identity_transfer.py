import json
import math
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.data_transforms import process_data_transforms
from simpletuner.helpers.data_transforms.identity_transfer import (
    VOICE_TRANSFORM_FORMAT,
    VOICE_TRANSFORM_FORMAT_VERSION,
    IdentityTransferTransform,
    RVCTransformLogger,
    VoiceModelArtifact,
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

    def _write_wav(self, path: Path, frequency: float = 220.0, sample_rate: int = 16000, seconds: float = 0.25):
        sample_count = int(sample_rate * seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for idx in range(sample_count):
                value = int(math.sin(2.0 * math.pi * frequency * idx / sample_rate) * 12000)
                frames.extend(value.to_bytes(2, byteorder="little", signed=True))
            handle.writeframes(bytes(frames))

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

        self.assertFalse(normalised["model"]["push_to_hub"])
        self.assertFalse(normalised["model"]["public"])
        self.assertEqual(normalised["model"]["separation_method"], "demucs")

        fingerprint = transform._voice_model_fingerprint(normalised)
        cache_dir = Path(normalised["model"]["cache_dir"])
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"model")
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

        self.assertEqual(artifact.model_path, cache_dir / "model.safetensors")

    def test_identity_stem_debug_dir_does_not_change_voice_model_fingerprint(self):
        transform = self._transform({"task": "identity_transfer", "id": "voice-transfer"})
        base = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        with_debug = json.loads(json.dumps(base))
        with_debug["model"]["identity_stem_debug_dir"] = "debug-stems"

        self.assertEqual(
            transform._voice_model_fingerprint(base),
            transform._voice_model_fingerprint(with_debug),
        )

    def test_local_voice_model_artifact_loads_legacy_pth_manifest(self):
        transform = self._transform({"task": "identity_transfer", "id": "voice-transfer"})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        fingerprint = transform._voice_model_fingerprint(normalised)
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

    def test_local_voice_model_artifact_reuses_hub_rvc_layout(self):
        transform = self._transform({"task": "identity_transfer", "id": "voice-transfer"})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        fingerprint = transform._voice_model_fingerprint(normalised)
        cache_dir = Path(normalised["model"]["cache_dir"])
        voice_dir = cache_dir / "voice_transform"
        voice_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text('{"model_name": "Test Voice"}', encoding="utf-8")
        (voice_dir / "model.safetensors").write_bytes(b"model")
        (voice_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": VOICE_TRANSFORM_FORMAT,
                    "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
                    "task": "identity_transfer",
                    "method": "rvc",
                    "fingerprint": fingerprint,
                    "model_name": "Test Voice",
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

        self.assertEqual(artifact.cache_dir, voice_dir)
        self.assertEqual(artifact.model_path, voice_dir / "model.safetensors")

    def test_missing_voice_model_requires_explicit_training(self):
        transform = self._transform({"task": "identity_transfer", "model": {"train_if_missing": False}})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})

        with self.assertRaises(ValueError) as context:
            transform._resolve_voice_model(
                normalised,
                transform._voice_model_fingerprint(normalised),
                Path(normalised["model"]["cache_dir"]),
                RVCTransformLogger(str(Path(self.temp_dir) / "output")),
            )

        self.assertIn("train_if_missing is false", str(context.exception))

    def test_push_to_hub_passes_public_flag_to_hub_cache(self):
        transform = self._transform(
            {
                "task": "identity_transfer",
                "id": "voice-transfer",
                "model": {
                    "identity_data_dir": str(Path(self.temp_dir) / "identity"),
                    "hub_model_id": "org/target-voice-rvc",
                    "reuse_from_hub": False,
                    "push_to_hub": True,
                    "public": True,
                },
            }
        )
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})
        fingerprint = transform._voice_model_fingerprint(normalised)
        cache_dir = Path(normalised["model"]["cache_dir"])

        def fake_train(_self, source_backend_config, transform_config, cache_dir, fingerprint, manifest_base, **_kwargs):
            cache_dir.mkdir(parents=True)
            model_path = cache_dir / "model.pth"
            manifest_path = cache_dir / "manifest.json"
            model_path.write_bytes(b"rvc")
            manifest = {**manifest_base, "fingerprint": fingerprint, "voice_model": {"kind": "simpletuner-rvc-v2-f0"}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return VoiceModelArtifact(cache_dir, manifest_path, model_path, None, manifest)

        with (
            patch("simpletuner.helpers.data_transforms.identity_transfer.RVCTrainer.train", new=fake_train),
            patch("simpletuner.helpers.data_transforms.identity_transfer.HubVoiceModelCache") as hub_cache,
        ):
            transform._resolve_voice_model(
                normalised,
                fingerprint,
                cache_dir,
                RVCTransformLogger(str(Path(self.temp_dir) / "output")),
            )

        hub_cache.assert_called_once_with("org/target-voice-rvc", token=None, public=True)
        hub_cache.return_value.upload.assert_called_once()

    def test_voice_model_fingerprint_is_independent_of_conversion_source(self):
        transform = self._transform(
            {
                "task": "identity_transfer",
                "id": "voice-transfer",
                "model": {"identity_data_dir": str(Path(self.temp_dir) / "identity")},
                "target": {"instance_data_dir": str(Path(self.temp_dir) / "generated-a")},
            }
        )
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})

        other_source = {
            **self.source_backend,
            "id": "other-source",
            "instance_data_dir": str(Path(self.temp_dir) / "other-source"),
        }
        other_transform = IdentityTransferTransform(
            global_config=self.args,
            source_backend_config=other_source,
            transform_config={
                "task": "identity_transfer",
                "id": "voice-transfer-other",
                "model": {"identity_data_dir": str(Path(self.temp_dir) / "identity")},
                "target": {"instance_data_dir": str(Path(self.temp_dir) / "generated-b")},
            },
        )
        other_normalised = other_transform._normalise_transform_config(existing_backend_ids={"other-source"})

        self.assertEqual(
            transform._voice_model_fingerprint(normalised),
            other_transform._voice_model_fingerprint(other_normalised),
        )
        self.assertNotEqual(
            transform._generated_fingerprint(normalised),
            other_transform._generated_fingerprint(other_normalised),
        )

    def test_process_data_transforms_trains_and_converts_with_rvc_artifact(self):
        source_dir = Path(self.temp_dir) / "convert-source"
        identity_dir = Path(self.temp_dir) / "identity"
        generated_dir = Path(self.temp_dir) / "generated"
        self._write_wav(source_dir / "source.wav", frequency=180.0)
        self._write_wav(identity_dir / "voice.wav", frequency=320.0)
        (source_dir / "source.txt").write_text("rock vocal test", encoding="utf-8")

        def fake_train(_self, source_backend_config, transform_config, cache_dir, fingerprint, manifest_base, **_kwargs):
            self.assertEqual(transform_config["model"]["separation_method"], "demucs")
            cache_dir.mkdir(parents=True)
            model_path = cache_dir / "model.pth"
            manifest_path = cache_dir / "manifest.json"
            model_path.write_bytes(b"rvc")
            manifest = {**manifest_base, "fingerprint": fingerprint, "voice_model": {"kind": "simpletuner-rvc-v2-f0"}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return VoiceModelArtifact(cache_dir, manifest_path, model_path, None, manifest)

        def fake_convert(_self, source_backend_config, target_backend_config, *_args, **_kwargs):
            output_dir = Path(target_backend_config["instance_data_dir"])
            output_dir.mkdir(parents=True)
            shutil.copy2(Path(source_backend_config["instance_data_dir"]) / "source.wav", output_dir / "source.wav")
            shutil.copy2(Path(source_backend_config["instance_data_dir"]) / "source.txt", output_dir / "source.txt")

        with (
            patch("simpletuner.helpers.data_transforms.identity_transfer.RVCTrainer.train", new=fake_train),
            patch("simpletuner.helpers.data_transforms.identity_transfer.RVCConverter.convert", new=fake_convert),
        ):
            config = process_data_transforms(
                global_config=self.args,
                data_backend_config=[
                    {
                        "id": "convert-source",
                        "type": "local",
                        "dataset_type": "audio",
                        "instance_data_dir": str(source_dir),
                        "metadata_backend": "discovery",
                        "caption_strategy": "textfile",
                        "audio": {"sample_rate": 48000, "channels": 1},
                        "data_transforms": [
                            {
                                "task": "identity_transfer",
                                "id": "voice-transfer",
                                "model": {
                                    "identity_data_dir": str(identity_dir),
                                    "sample_rate": 48000,
                                    "identity_audio_mode": "vocal_only",
                                    "training_steps": 4,
                                    "batch_size": 1,
                                    "learning_rate": 1e-4,
                                    "device": "cpu",
                                },
                                "conversion": {
                                    "audio_mode": "vocal_only",
                                    "timbre_strength": 0.5,
                                    "device": "cpu",
                                },
                                "target": {"instance_data_dir": str(generated_dir)},
                            }
                        ],
                    }
                ],
            )

        generated = config[1]
        self.assertEqual(generated["id"], "voice-transfer")
        self.assertTrue((generated_dir / "source.wav").exists())
        self.assertEqual((generated_dir / "source.txt").read_text(encoding="utf-8"), "rock vocal test")
        model_path = Path(self.args.output_dir) / "cache" / "data_transforms" / "voice-transfer" / "rvc_model" / "model.pth"
        self.assertTrue(model_path.exists())

    def test_training_requires_identity_data_dir(self):
        transform = self._transform({"task": "identity_transfer"})
        normalised = transform._normalise_transform_config(existing_backend_ids={"artist-source"})

        with self.assertRaises(ValueError) as context:
            transform._resolve_voice_model(
                normalised,
                transform._fingerprint(normalised),
                Path(normalised["model"]["cache_dir"]),
                RVCTransformLogger(str(Path(self.temp_dir) / "output")),
            )

        self.assertIn("identity_data_dir is required", str(context.exception))

    def test_logger_writes_local_json_files(self):
        run_logger = RVCTransformLogger(str(Path(self.temp_dir) / "output"))

        run_logger.event("voice-transfer", "voice_model_reused", source="local")
        run_logger.summary("voice-transfer", status="reused_generated_cache")

        self.assertTrue((Path(self.temp_dir) / "output" / "logs" / "rvc" / "training_stats.jsonl").exists())
        summary = json.loads((Path(self.temp_dir) / "output" / "logs" / "rvc" / "summary.json").read_text())
        self.assertEqual(summary["status"], "reused_generated_cache")


if __name__ == "__main__":
    unittest.main()
