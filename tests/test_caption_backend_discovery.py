import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.data_backend.builders.local import LocalBackendBuilder
from simpletuner.helpers.data_backend.config import create_backend_config
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.training.state_tracker import StateTracker


class CaptionDiscoveryRestartTests(unittest.TestCase):
    def test_builder_preserves_extensions_and_restart_excludes_own_json(self):
        for extensions in (None, ["jsonl"]):
            with self.subTest(extensions=extensions), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "captions"
                source.mkdir()
                (source / "prompts.jsonl").write_text('"a fox"\n"a landscape"\n', encoding="utf-8")
                (source / "extra.json").write_text(json.dumps({"caption": "a lake"}), encoding="utf-8")
                args = SimpleNamespace(
                    output_dir=str(root / "output"),
                    train_batch_size=1,
                    resolution=512,
                    resolution_type="pixel",
                    metadata_update_interval=3600,
                )
                accelerator = SimpleNamespace(is_local_main_process=True, is_main_process=True, num_processes=1)
                declaration = {
                    "id": "captions",
                    "type": "local",
                    "dataset_type": "caption",
                    "instance_data_dir": str(source),
                    "preserve_data_backend_cache": True,
                }
                if extensions is not None:
                    declaration["caption_file_extensions"] = extensions
                config = create_backend_config(declaration, vars(args))
                config.validate(vars(args))
                factory = FactoryRegistry.__new__(FactoryRegistry)
                factory.args = args
                expected = ["a fox", "a landscape"] if extensions else ["a lake", "a fox", "a landscape"]

                with (
                    patch.object(StateTracker, "all_caption_files", {}),
                    patch.object(StateTracker, "data_backends", {}),
                    patch.object(StateTracker, "args", args),
                ):
                    StateTracker.set_data_backend_config("captions", config.to_dict()["config"])
                    builder = LocalBackendBuilder(accelerator, vars(args))
                    first = builder.build_with_metadata(config, vars(args))
                    metadata = first["metadata_backend"]
                    if extensions is not None:
                        self.assertEqual(metadata.caption_extensions, ("jsonl",))
                    discovered = factory._discover_caption_files(first, metadata)
                    self.assertEqual(metadata.ingest_from_file_cache(discovered), len(expected))
                    self.assertEqual([record.caption_text for record in metadata.iter_records()], expected)
                    self.assertTrue(metadata.cache_file.exists())
                    self.assertTrue(metadata.metadata_file.exists())

                    # A fresh process discovers the JSON caches that now live beside the source captions.
                    StateTracker.all_caption_files["captions"] = {}
                    second = builder.build_with_metadata(config, vars(args))
                    metadata = second["metadata_backend"]
                    discovered = factory._discover_caption_files(second, metadata)
                    self.assertNotIn(str(metadata.cache_file), discovered)
                    self.assertNotIn(str(metadata.metadata_file), discovered)
                    self.assertEqual(metadata.ingest_from_file_cache(discovered), len(expected))
                    self.assertEqual([record.caption_text for record in metadata.iter_records()], expected)

                    # Old persisted listings are filtered too, and direct ingestion has the same source guard.
                    polluted = {**discovered, str(metadata.cache_file): False, str(metadata.metadata_file): False}
                    StateTracker.all_caption_files["captions"] = polluted
                    filtered = factory._discover_caption_files(second, metadata)
                    self.assertEqual(set(filtered), set(discovered))
                    self.assertEqual(metadata.ingest_from_file_cache(polluted), len(expected))
                    self.assertEqual([record.caption_text for record in metadata.iter_records()], expected)


if __name__ == "__main__":
    unittest.main()
