import json
import math
import os
import struct
import tempfile
import unittest
import wave
from io import BytesIO
from typing import Optional, Tuple

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


def _write_test_wav(path: str, sample_rate: int = 8000, num_samples: int = 1600) -> Tuple[int, int]:
    """Write a simple sine wave to the provided path."""
    tone_frequency = 440.0
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = [int(32767 * math.sin(2 * math.pi * tone_frequency * (i / sample_rate))) for i in range(num_samples)]
        frame_bytes = b"".join(struct.pack("<h", frame) for frame in frames)
        wav_file.writeframes(frame_bytes)
    return sample_rate, num_samples


class FilesystemDataBackend(BaseDataBackend):
    """Lightweight backend used for integration tests that works directly on the local filesystem."""

    def __init__(self, root_dir: str, dataset_id: str) -> None:
        self.root_dir = root_dir
        self.id = dataset_id
        self.dataset_type = DatasetType.AUDIO

    def _resolve(self, identifier: str) -> str:
        if os.path.isabs(identifier):
            return identifier
        return os.path.join(self.root_dir, identifier)

    def read(self, identifier, as_byteIO: bool = False):
        resolved = self._resolve(identifier)
        with open(resolved, "rb") as file:
            data = file.read()
        if as_byteIO:
            return BytesIO(data)
        return data

    def write(self, identifier, data):
        resolved = self._resolve(identifier)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        mode = "wb"
        payload = data.encode("utf-8") if isinstance(data, str) else data
        with open(resolved, mode) as file:
            file.write(payload)

    def delete(self, identifier):
        resolved = self._resolve(identifier)
        if os.path.exists(resolved):
            os.remove(resolved)

    def exists(self, identifier):
        return os.path.exists(self._resolve(identifier))

    def open_file(self, identifier, mode):
        return open(self._resolve(identifier), mode)

    def list_files(self, file_extensions=None, instance_data_dir=None):
        instance_dir = instance_data_dir or self.root_dir
        matched = []
        for root, _, files in os.walk(instance_dir):
            filtered = [
                os.path.join(root, f)
                for f in files
                if not file_extensions or any(f.lower().endswith(ext) for ext in file_extensions)
            ]
            if filtered:
                matched.append((root, [], filtered))
        return matched

    def get_abs_path(self, sample_path: str = None):
        target = sample_path or self.root_dir
        return os.path.abspath(self._resolve(target))

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        raise NotImplementedError("Image reading not supported in filesystem backend tests.")

    def read_image_batch(self, filepaths: list, delete_problematic_images: bool = False):
        raise NotImplementedError("Image reading not supported in filesystem backend tests.")

    def create_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)

    def torch_load(self, filename):
        raise NotImplementedError("Torch serialization not required for filesystem backend tests.")

    def torch_save(self, data, filename):
        raise NotImplementedError("Torch serialization not required for filesystem backend tests.")

    def write_batch(self, identifiers, files):
        for identifier, data in zip(identifiers, files):
            self.write(identifier, data)

    def get_instance_representation(self) -> dict:
        return {"backend_type": "filesystem", "root": self.root_dir, "id": self.id}

    @staticmethod
    def from_instance_representation(representation: dict) -> "FilesystemDataBackend":
        return FilesystemDataBackend(representation["root"], representation["id"])


class TestAudioMetadataBackends(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        StateTracker.clear_data_backends()

    def tearDown(self) -> None:
        self.tempdir.cleanup()
        StateTracker.clear_data_backends()

    def _build_backend(self, dataset_id: str) -> FilesystemDataBackend:
        return FilesystemDataBackend(self.tempdir.name, dataset_id)

    def _register_dataset_config(self, dataset_id: str, audio_config: Optional[dict] = None) -> None:
        default_audio = {"bucket_strategy": "duration", "duration_interval": 15.0}
        merged_audio = default_audio if audio_config is None else {**default_audio, **audio_config}
        StateTracker.set_data_backend_config(
            dataset_id,
            {
                "dataset_type": DatasetType.AUDIO.value,
                "audio": merged_audio,
            },
        )

    def _parquet_backend(
        self,
        dataset_id: str,
        data_backend: FilesystemDataBackend,
        manifest_path: str,
        include_audio_columns: bool = True,
    ) -> ParquetMetadataBackend:
        parquet_config = {
            "path": manifest_path,
            "filename_column": "filepath",
            "identifier_includes_extension": True,
            "audio_sample_rate_column": "sample_rate",
            "audio_num_samples_column": "num_samples",
            "audio_duration_column": "duration",
            "audio_channels_column": "channels",
        }
        if not include_audio_columns:
            parquet_config.pop("audio_sample_rate_column", None)
            parquet_config.pop("audio_num_samples_column", None)
            parquet_config.pop("audio_duration_column", None)
            parquet_config.pop("audio_channels_column", None)
        return ParquetMetadataBackend(
            id=dataset_id,
            instance_data_dir=self.tempdir.name,
            cache_file=os.path.join(self.tempdir.name, "cache_parquet"),
            metadata_file=os.path.join(self.tempdir.name, "metadata_parquet"),
            data_backend=data_backend,
            accelerator=None,
            batch_size=1,
            resolution=1.0,
            resolution_type="pixel",
            parquet_config=parquet_config,
            delete_problematic_images=False,
            delete_unwanted_images=False,
            metadata_update_interval=1,
            minimum_image_size=None,
            minimum_aspect_ratio=None,
            maximum_aspect_ratio=None,
            num_frames=None,
            minimum_num_frames=None,
            maximum_num_frames=None,
            cache_file_suffix=None,
            repeats=0,
        )

    def test_parquet_audio_bucket_uses_manifest_metadata(self):
        dataset_id = "audio_parquet_manifest"
        audio_path = os.path.join(self.tempdir.name, "manifest.wav")
        sample_rate, num_samples = _write_test_wav(audio_path)
        duration_seconds = round(num_samples / sample_rate, 2)

        manifest_path = os.path.join(self.tempdir.name, "manifest.jsonl")
        record = {
            "filepath": os.path.basename(audio_path),
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": duration_seconds,
            "channels": 2,
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(record) + "\n")

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id)
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path)

        metadata_updates = {}
        stats = {}
        bucket_indices = parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics=stats,
        )

        self.assertIn(audio_path, sum(bucket_indices.values(), []))
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["sample_rate"], sample_rate)
        self.assertEqual(audio_metadata["num_samples"], num_samples)
        self.assertEqual(audio_metadata["duration_seconds"], duration_seconds)
        self.assertEqual(audio_metadata["num_channels"], 2)
        self.assertEqual(audio_metadata["truncation_mode"], "beginning")

    def test_parquet_audio_bucket_reads_from_audio_when_manifest_missing_values(self):
        dataset_id = "audio_parquet_fallback"
        audio_path = os.path.join(self.tempdir.name, "fallback.wav")
        sample_rate, num_samples = _write_test_wav(audio_path)

        manifest_path = os.path.join(self.tempdir.name, "fallback.jsonl")
        record = {"filepath": os.path.basename(audio_path)}
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(record) + "\n")

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id)
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path, include_audio_columns=False)

        metadata_updates = {}
        stats = {}
        parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics=stats,
        )

        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["sample_rate"], sample_rate)
        self.assertEqual(audio_metadata["num_samples"], num_samples)
        self.assertAlmostEqual(
            audio_metadata["duration_seconds"],
            num_samples / sample_rate,
            places=4,
        )
        self.assertEqual(audio_metadata["num_channels"], 1)
        self.assertEqual(audio_metadata["truncation_mode"], "beginning")

    def test_parquet_audio_bucket_enforces_max_duration(self):
        dataset_id = "audio_parquet_max_duration"
        audio_path = os.path.join(self.tempdir.name, "too_long.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=100, num_samples=10000)  # 100s clip
        duration_seconds = num_samples / sample_rate

        manifest_path = os.path.join(self.tempdir.name, "too_long.jsonl")
        record = {
            "filepath": os.path.basename(audio_path),
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": duration_seconds,
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(record) + "\n")

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"max_duration_seconds": 30})
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path)

        stats = {}
        metadata_updates = {}
        bucket_indices = parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics=stats,
        )

        self.assertEqual(bucket_indices, {})
        self.assertEqual(metadata_updates, {})
        self.assertEqual(stats.get("skipped", {}).get("too_long"), 1)

    def test_parquet_audio_bucket_honours_truncation_mode(self):
        dataset_id = "audio_parquet_truncation_mode"
        audio_path = os.path.join(self.tempdir.name, "trunc.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=16000, num_samples=16000)
        duration_seconds = num_samples / sample_rate

        manifest_path = os.path.join(self.tempdir.name, "trunc.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(
                json.dumps(
                    {
                        "filepath": os.path.basename(audio_path),
                        "sample_rate": sample_rate,
                        "num_samples": num_samples,
                        "duration": duration_seconds,
                    }
                )
                + "\n"
            )

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"truncation_mode": "random", "duration_interval": 1.0})
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path)

        metadata_updates = {}
        parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics={},
        )
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["truncation_mode"], "random")

    def test_parquet_audio_bucket_quantizes_duration_interval(self):
        dataset_id = "audio_parquet_duration_bucket"
        audio_path = os.path.join(self.tempdir.name, "duration.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=100, num_samples=7700)
        duration_seconds = num_samples / sample_rate
        manifest_path = os.path.join(self.tempdir.name, "duration.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(
                json.dumps(
                    {
                        "filepath": os.path.basename(audio_path),
                        "sample_rate": sample_rate,
                        "num_samples": num_samples,
                        "duration": duration_seconds,
                        "channels": 1,
                    }
                )
                + "\n"
            )

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"duration_interval": 15.0})
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path)

        metadata_updates = {}
        stats = {}
        bucket_indices = parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics=stats,
        )
        self.assertIn("75s", bucket_indices)
        self.assertIn(audio_path, bucket_indices["75s"])
        audio_metadata = metadata_updates[audio_path]
        self.assertAlmostEqual(audio_metadata["original_duration_seconds"], duration_seconds, places=6)
        self.assertEqual(audio_metadata["duration_seconds"], 75.0)
        self.assertEqual(audio_metadata["bucket_duration_seconds"], 75.0)
        self.assertEqual(audio_metadata["truncation_mode"], "beginning")

    def test_discovery_audio_bucket_respects_duration_interval(self):
        dataset_id = "audio_discovery_bucket"
        audio_path = os.path.join(self.tempdir.name, "discovery.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=50, num_samples=3250)

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"duration_interval": 20.0})
        discovery_backend = DiscoveryMetadataBackend(
            id=dataset_id,
            instance_data_dir=self.tempdir.name,
            cache_file=os.path.join(self.tempdir.name, "cache_discovery"),
            metadata_file=os.path.join(self.tempdir.name, "metadata_discovery"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=1.0,
            resolution_type="pixel",
        )

        metadata_updates = {}
        bucket_indices = discovery_backend._process_audio_sample(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics={},
        )
        self.assertIn("60s", bucket_indices)
        self.assertIn(audio_path, bucket_indices["60s"])
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["sample_rate"], sample_rate)
        self.assertEqual(audio_metadata["num_samples"], num_samples)
        self.assertAlmostEqual(audio_metadata["original_duration_seconds"], num_samples / sample_rate, places=4)
        self.assertEqual(audio_metadata["duration_seconds"], 60.0)
        self.assertEqual(audio_metadata["bucket_duration_seconds"], 60.0)
        self.assertEqual(audio_metadata["num_channels"], 1)
        self.assertEqual(audio_metadata["truncation_mode"], "beginning")

    def test_discovery_audio_bucket_enforces_max_duration(self):
        dataset_id = "audio_discovery_max_duration"
        audio_path = os.path.join(self.tempdir.name, "discovery_long.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=50, num_samples=5000)  # 100s clip

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"max_duration_seconds": 20})
        discovery_backend = DiscoveryMetadataBackend(
            id=dataset_id,
            instance_data_dir=self.tempdir.name,
            cache_file=os.path.join(self.tempdir.name, "cache_discovery"),
            metadata_file=os.path.join(self.tempdir.name, "metadata_discovery"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=1.0,
            resolution_type="pixel",
        )

        metadata_updates = {}
        stats = {}
        bucket_indices = discovery_backend._process_audio_sample(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics=stats,
        )

        self.assertEqual(bucket_indices, {})
        self.assertEqual(metadata_updates, {})
        self.assertEqual(stats.get("skipped", {}).get("too_long"), 1)

    def test_discovery_audio_bucket_honours_truncation_mode(self):
        dataset_id = "audio_discovery_truncation_mode"
        audio_path = os.path.join(self.tempdir.name, "disc_trunc.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=16000, num_samples=8000)

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id, audio_config={"truncation_mode": "end", "duration_interval": 2.0})
        discovery_backend = DiscoveryMetadataBackend(
            id=dataset_id,
            instance_data_dir=self.tempdir.name,
            cache_file=os.path.join(self.tempdir.name, "cache_discovery"),
            metadata_file=os.path.join(self.tempdir.name, "metadata_discovery"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=1.0,
            resolution_type="pixel",
        )

        metadata_updates = {}
        discovery_backend._process_audio_sample(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics={},
        )
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["truncation_mode"], "end")

    def test_discovery_audio_metadata_includes_image_fields(self):
        dataset_id = "audio_discovery_placeholders"
        audio_path = os.path.join(self.tempdir.name, "placeholder.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=8000, num_samples=4000)

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id)
        discovery_backend = DiscoveryMetadataBackend(
            id=dataset_id,
            instance_data_dir=self.tempdir.name,
            cache_file=os.path.join(self.tempdir.name, "cache_discovery"),
            metadata_file=os.path.join(self.tempdir.name, "metadata_discovery"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=1.0,
            resolution_type="pixel",
        )

        metadata_updates = {}
        discovery_backend._process_audio_sample(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics={},
        )
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["image_path"], audio_path)
        self.assertEqual(audio_metadata["audio_path"], audio_path)
        self.assertEqual(audio_metadata["crop_coordinates"], (0, 0))
        self.assertEqual(audio_metadata["aspect_ratio"], 1.0)
        self.assertEqual(audio_metadata["original_size"], (num_samples, 1))
        self.assertEqual(audio_metadata["intermediary_size"], (num_samples, 1))
        self.assertEqual(audio_metadata["target_size"], (num_samples, 1))

    def test_parquet_audio_metadata_includes_image_fields(self):
        dataset_id = "audio_parquet_placeholders"
        audio_path = os.path.join(self.tempdir.name, "placeholder_parquet.wav")
        sample_rate, num_samples = _write_test_wav(audio_path, sample_rate=16000, num_samples=3200)
        duration_seconds = round(num_samples / sample_rate, 2)

        manifest_path = os.path.join(self.tempdir.name, "parquet_placeholders.jsonl")
        record = {
            "filepath": os.path.basename(audio_path),
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration": duration_seconds,
            "channels": 2,
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(record) + "\n")

        backend = self._build_backend(dataset_id)
        self._register_dataset_config(dataset_id)
        parquet_backend = self._parquet_backend(dataset_id, backend, manifest_path)

        metadata_updates = {}
        parquet_backend._process_for_bucket(
            image_path_str=audio_path,
            aspect_ratio_bucket_indices={},
            metadata_updates=metadata_updates,
            statistics={},
        )
        audio_metadata = metadata_updates[audio_path]
        self.assertEqual(audio_metadata["image_path"], audio_path)
        self.assertEqual(audio_metadata["audio_path"], audio_path)
        self.assertEqual(audio_metadata["crop_coordinates"], (0, 0))
        self.assertEqual(audio_metadata["aspect_ratio"], 1.0)
        self.assertEqual(audio_metadata["original_size"], (num_samples, 2))
        self.assertEqual(audio_metadata["intermediary_size"], (num_samples, 2))
        self.assertEqual(audio_metadata["target_size"], (num_samples, 2))
