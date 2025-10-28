import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class InMemoryDataBackend:
    def __init__(self):
        self.id = "caption"
        self._storage = {}

    def _key(self, identifier):
        return str(identifier)

    def read(self, identifier):
        key = self._key(identifier)
        if key not in self._storage:
            raise FileNotFoundError(identifier)
        return self._storage[key]

    def write(self, identifier, data):
        identifier = self._key(identifier)
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._storage[identifier] = data

    def exists(self, identifier):
        return self._key(identifier) in self._storage

    def list_files(self, instance_data_dir, file_extensions):
        results = []
        for path in self._storage:
            if not path.startswith(instance_data_dir):
                continue
            suffix = Path(path).suffix.lstrip(".").lower()
            if file_extensions and suffix not in file_extensions:
                continue
            parent = str(Path(path).parent)
            results.append((parent, [], [path]))
        return results

    def get_abs_path(self, sample_path):
        return sample_path


class DummyHFBackend(InMemoryDataBackend):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset


class TestCaptionMetadataBackend(unittest.TestCase):
    def setUp(self):
        StateTracker.set_args(SimpleNamespace(output_dir="/tmp"))
        StateTracker.data_backends = {}
        StateTracker.all_caption_files = {}
        self.backend = InMemoryDataBackend()
        self.accelerator = SimpleNamespace(is_local_main_process=True)
        self.instance_dir = "/datasets/captions"
        self.cache_file = "/cache/captions.cache"
        self.metadata_file = "/cache/captions.meta"

    def _create_backend(self):
        return CaptionMetadataBackend(
            id="caption",
            instance_data_dir=self.instance_dir,
            cache_file=self.cache_file,
            metadata_file=self.metadata_file,
            data_backend=self.backend,
            accelerator=self.accelerator,
            batch_size=1,
        )

    def test_ingest_from_various_formats(self):
        self.backend.write(f"{self.instance_dir}/basic.txt", "first caption\n\nsecond")
        json_payload = json.dumps({"captions": ["json one", "json two"]})
        self.backend.write(f"{self.instance_dir}/data.json", json_payload)
        jsonl_payload = "\n".join([json.dumps({"caption": "line a"}), '"line b"'])
        self.backend.write(f"{self.instance_dir}/mixed.jsonl", jsonl_payload)

        caption_backend = self._create_backend()
        file_cache = {
            f"{self.instance_dir}/basic.txt": False,
            f"{self.instance_dir}/data.json": False,
            f"{self.instance_dir}/mixed.jsonl": False,
        }

        created = caption_backend.ingest_from_file_cache(file_cache)
        self.assertEqual(created, 6)
        self.assertEqual(len(list(caption_backend.iter_records())), 6)

        payload = self.backend.read(str(caption_backend.metadata_file))
        stored = json.loads(payload.decode("utf-8"))
        self.assertEqual(len(stored), 6)
        self.assertTrue(all(entry["caption_text"] for entry in stored))

    def test_metadata_round_trip(self):
        self.backend.write(f"{self.instance_dir}/only.txt", "one caption")
        caption_backend = self._create_backend()
        caption_backend.ingest_from_file_cache({f"{self.instance_dir}/only.txt": False})

        reloaded_backend = self._create_backend()
        reloaded_backend.load_image_metadata()
        records = list(reloaded_backend.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].caption_text, "one caption")
        self.assertEqual(records[0].data_backend_id, "caption")

    def test_ingest_from_parquet_manifest(self):
        manifest_path = "/manifests/captions.jsonl"
        rows = [
            {"filename": "alpha.jpg", "caption": "First entry"},
            {"filename": "beta.png", "caption": ["Second", "Alt second"]},
        ]
        payload = "\n".join(json.dumps(row) for row in rows)
        self.backend.write(manifest_path, payload)

        caption_backend = CaptionMetadataBackend(
            id="caption",
            instance_data_dir=self.instance_dir,
            cache_file=self.cache_file,
            metadata_file=self.metadata_file,
            data_backend=self.backend,
            accelerator=self.accelerator,
            batch_size=1,
            caption_ingest_strategy="parquet",
            parquet_config={
                "path": manifest_path,
                "filename_column": "filename",
                "caption_column": "caption",
                "identifier_includes_extension": False,
                "default_extension": "txt",
            },
        )

        created = caption_backend.ingest_from_parquet_config()
        self.assertEqual(created, 3)
        records = list(caption_backend.iter_records())
        self.assertEqual(len(records), 3)
        texts = sorted(record.caption_text for record in records)
        self.assertEqual(texts, ["Alt second", "First entry", "Second"])

    def test_ingest_from_huggingface_dataset(self):
        dataset = [
            {"prompt": "  Example 1  ", "quality": {"score": 0.9}},
            {"prompt": ["Example 2a", "Example 2b"], "quality": {"score": 0.4}},
            {"description": "fallback caption", "quality": {"score": 0.8}},
        ]
        hf_backend = DummyHFBackend(dataset)
        hf_backend.id = "caption"

        caption_backend = CaptionMetadataBackend(
            id="caption",
            instance_data_dir=self.instance_dir,
            cache_file=self.cache_file,
            metadata_file=self.metadata_file,
            data_backend=hf_backend,
            accelerator=self.accelerator,
            batch_size=1,
            caption_ingest_strategy="huggingface",
            hf_config={
                "caption_column": "prompt",
                "quality_column": "quality",
                "description_column": "description",
                "repo_id": "hf/test-dataset",
            },
            quality_filter={"score": 0.5},
        )

        created = caption_backend.ingest_from_huggingface_dataset()
        self.assertEqual(created, 2)
        payload = list(caption_backend.iter_records())
        captions = sorted(record.caption_text for record in payload)
        self.assertEqual(captions, ["Example 1", "fallback caption"])


if __name__ == "__main__":
    unittest.main()
