import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file

from simpletuner.helpers.data_transforms import identity_transfer  # noqa: F401
from simpletuner.helpers.data_transforms.base import process_data_transforms
from simpletuner.helpers.rvc.simple import RVCRecord, SimpleRVCConverter, SimpleRVCTrainer, _load_model_payload


class RVCIndexSmokeTests(unittest.TestCase):
    def _record(self, frames: int, dims: int = 768) -> RVCRecord:
        return RVCRecord(
            phone=torch.randn(frames, dims),
            pitch=torch.full((frames,), 128, dtype=torch.long),
            pitchf=torch.full((frames,), 220.0, dtype=torch.float32),
            spec=torch.randn(1025, frames),
            wave=torch.randn(1, frames * 320),
        )

    def test_build_index_writes_flat_index_and_feature_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            records = [self._record(4), self._record(3)]

            index_path = SimpleRVCTrainer()._build_index(records, cache_dir, {})

            self.assertIsNotNone(index_path)
            self.assertTrue(index_path.exists())
            self.assertTrue((cache_dir / "features.safetensors").exists())

            import faiss

            index = faiss.read_index(str(index_path))
            self.assertEqual(index.ntotal, 7)
            self.assertEqual(index.d, 768)

    def test_rvc_model_loader_supports_safetensors_and_legacy_pth(self):
        state_dict = {"emb_g.weight": torch.randn(1, 2)}
        metadata = {
            "kind": "simpletuner-rvc-v2-f0",
            "version": "v2",
            "f0": "true",
            "sample_rate": "48000",
            "config_json": "{}",
            "training_json": '{"steps": 1.0}',
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            safetensors_path = root / "model.safetensors"
            pth_path = root / "model.pth"
            save_safetensors_file(state_dict, str(safetensors_path), metadata=metadata)
            torch.save(
                {
                    "kind": "simpletuner-rvc-v2-f0",
                    "version": "v2",
                    "f0": True,
                    "sample_rate": 48000,
                    "config": {},
                    "generator_state_dict": state_dict,
                    "training": {"steps": 1.0},
                },
                pth_path,
            )

            safetensors_payload = _load_model_payload(safetensors_path)
            pth_payload = _load_model_payload(pth_path)

        self.assertEqual(safetensors_payload["kind"], "simpletuner-rvc-v2-f0")
        self.assertEqual(pth_payload["kind"], "simpletuner-rvc-v2-f0")
        self.assertIn("emb_g.weight", safetensors_payload["generator_state_dict"])
        self.assertIn("emb_g.weight", pth_payload["generator_state_dict"])

    def test_torch_retrieve_uses_nearest_feature_vectors(self):
        converter = SimpleRVCConverter()
        index_vectors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        query = np.array([[0.95, 0.05]], dtype=np.float32)

        retrieved = converter._torch_retrieve(query, index_vectors, torch.device("cpu"))

        self.assertEqual(retrieved.shape, query.shape)
        self.assertGreater(retrieved[0, 0], retrieved[0, 1])


@unittest.skipUnless(
    os.environ.get("SIMPLETUNER_RUN_RVC_SMOKE") == "1",
    "set SIMPLETUNER_RUN_RVC_SMOKE=1 to run the real RVC training/conversion smoke test",
)
class RVCIdentityTransferEndToEndSmokeTest(unittest.TestCase):
    def test_identity_transfer_trains_rvc_and_writes_audio(self):
        source_env = os.environ.get("SIMPLETUNER_RVC_SMOKE_SOURCE")
        identity_env = os.environ.get("SIMPLETUNER_RVC_SMOKE_IDENTITY")
        if not source_env or not identity_env:
            self.skipTest("SIMPLETUNER_RVC_SMOKE_SOURCE and SIMPLETUNER_RVC_SMOKE_IDENTITY are required")

        source_dir = Path(source_env).expanduser()
        identity_dir = Path(identity_env).expanduser()
        self.assertTrue(source_dir.exists())
        self.assertTrue(identity_dir.exists())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            generated_dir = temp_root / "generated"
            output_dir = temp_root / "output"
            config = [
                {
                    "id": "rvc-smoke-source",
                    "type": "local",
                    "dataset_type": "audio",
                    "metadata_backend": "discovery",
                    "caption_strategy": "textfile",
                    "instance_data_dir": str(source_dir),
                    "audio": {"sample_rate": 48000, "channels": 2, "audio_only": True},
                    "data_transforms": [
                        {
                            "id": "rvc-smoke-generated",
                            "task": "identity_transfer",
                            "method": "rvc",
                            "model": {
                                "identity_data_dir": str(identity_dir),
                                "cache_dir": str(output_dir / "cache" / "rvc_model"),
                                "train_if_missing": True,
                                "force_retrain": True,
                                "sample_rate": 48000,
                                "identity_audio_mode": os.environ.get("SIMPLETUNER_RVC_SMOKE_IDENTITY_MODE", "separate"),
                                "training_steps": int(os.environ.get("SIMPLETUNER_RVC_SMOKE_STEPS", "1")),
                                "batch_size": 1,
                                "learning_rate": 1e-4,
                                "max_seconds_per_file": float(os.environ.get("SIMPLETUNER_RVC_SMOKE_MAX_SECONDS", "5.0")),
                                "build_index": True,
                                "flat_index_threshold": 1000000,
                                "device": os.environ.get("SIMPLETUNER_RVC_SMOKE_DEVICE", "cpu"),
                                "demucs_device": os.environ.get("SIMPLETUNER_RVC_SMOKE_DEMUCS_DEVICE", "cpu"),
                            },
                            "conversion": {
                                "audio_mode": os.environ.get("SIMPLETUNER_RVC_SMOKE_AUDIO_MODE", "separate_convert_remix"),
                                "separation_method": "demucs",
                                "demucs_device": os.environ.get("SIMPLETUNER_RVC_SMOKE_DEMUCS_DEVICE", "cpu"),
                                "timbre_strength": 1.0,
                                "retrieval_strength": 0.75,
                                "torch_retrieval": True,
                                "device": os.environ.get("SIMPLETUNER_RVC_SMOKE_DEVICE", "cpu"),
                            },
                            "target": {
                                "id": "rvc-smoke-generated",
                                "type": "local",
                                "dataset_type": "audio",
                                "metadata_backend": "discovery",
                                "caption_strategy": "textfile",
                                "instance_data_dir": str(generated_dir),
                                "audio": {"sample_rate": 48000, "channels": 2, "audio_only": True},
                            },
                        }
                    ],
                }
            ]

            result = process_data_transforms(SimpleNamespace(output_dir=str(output_dir)), config)
            generated_audio = sorted(generated_dir.glob("*.wav"))

            self.assertEqual(len(result), 2)
            self.assertGreaterEqual(len(generated_audio), 1)
            self.assertTrue((generated_dir / ".simpletuner_identity_transfer.json").exists())
            self.assertTrue((output_dir / "cache" / "rvc_model" / "model.safetensors").exists())
            self.assertTrue((output_dir / "cache" / "rvc_model" / "manifest.json").exists())
            self.assertTrue((output_dir / "cache" / "rvc_model" / "index.index").exists())
            self.assertTrue((output_dir / "cache" / "rvc_model" / "features.safetensors").exists())

            for path in generated_audio:
                self.assertGreater(path.stat().st_size, 1024)
                for suffix in (".txt", ".lyrics"):
                    source_sidecar = source_dir / path.with_suffix(suffix).name
                    if source_sidecar.exists():
                        self.assertTrue(path.with_suffix(suffix).exists())

            if os.environ.get("SIMPLETUNER_RVC_SMOKE_KEEP_OUTPUT"):
                keep_dir = Path(os.environ["SIMPLETUNER_RVC_SMOKE_KEEP_OUTPUT"]).expanduser()
                shutil.copytree(temp_root, keep_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    unittest.main()
