import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training import dynamo_cache


def _config(path, hub_repo_id=None, **overrides):
    values = dict(
        dynamo_cache_export=str(path) if path is not None else None,
        dynamo_hub_repo_id=hub_repo_id,
        dynamo_backend="inductor",
        model_family="test",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _cache_info(**artifacts):
    return SimpleNamespace(artifacts=artifacts)


class DynamoCacheManagerTests(unittest.TestCase):
    def test_hub_repo_requires_cache_path(self):
        with self.assertRaisesRegex(ValueError, "requires --dynamo_cache_export"):
            dynamo_cache.DynamoCacheManager(_config(None, "owner/cache"))

    def test_directory_path_uses_stable_generated_filename_locally_and_on_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_directory = Path(directory) / "compiler-caches"
            manager = dynamo_cache.DynamoCacheManager(_config(f"{cache_directory}/", "owner/cache"))

            self.assertEqual(manager.local_path.parent, cache_directory)
            self.assertEqual(manager.local_path.name, manager.generated_filename)
            self.assertTrue(manager.generated_filename.startswith("simpletuner-dynamo-test-default-"))
            self.assertTrue(manager.generated_filename.endswith(".ptcache"))
            self.assertEqual(manager.hub_path, manager.generated_filename)

        relative_manager = dynamo_cache.DynamoCacheManager(_config("compiler-caches", "owner/cache"))
        self.assertEqual(
            relative_manager.hub_path,
            f"compiler-caches/{relative_manager.generated_filename}",
        )

    def test_generated_filename_changes_with_graph_relevant_config(self):
        first = dynamo_cache.DynamoCacheManager(
            _config("compiler-caches", resolution=512, gradient_checkpointing_segment_stride=4)
        )
        second = dynamo_cache.DynamoCacheManager(
            _config("compiler-caches", resolution=768, gradient_checkpointing_segment_stride=4)
        )
        third = dynamo_cache.DynamoCacheManager(
            _config("compiler-caches", resolution=512, gradient_checkpointing_segment_stride=8)
        )
        fourth = dynamo_cache.DynamoCacheManager(
            _config(
                "compiler-caches",
                resolution=512,
                gradient_checkpointing_segment_stride=4,
                dynamo_wrapper="python",
            )
        )

        self.assertNotEqual(first.generated_filename, second.generated_filename)
        self.assertNotEqual(first.generated_filename, third.generated_filename)
        self.assertNotEqual(first.generated_filename, fourth.generated_filename)

    def test_loads_compatible_local_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.ptcache"
            payload = b"compiled-cache"
            cache_path.write_bytes(payload)
            manager = dynamo_cache.DynamoCacheManager(_config(cache_path))
            Path(f"{cache_path}.manifest.json").write_text(
                json.dumps(
                    {
                        "runtime": manager.runtime_signature,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                dynamo_cache.torch.compiler,
                "load_cache_artifacts",
                return_value=_cache_info(aot_autograd=["aot-1"], inductor=["inductor-1"]),
            ) as load_cache:
                loaded = manager.load()

            self.assertTrue(loaded)
            load_cache.assert_called_once_with(payload)
            self.assertEqual(manager.known_keys["aot_autograd"], {"aot-1"})

    def test_rejects_incompatible_runtime_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.ptcache"
            cache_path.write_bytes(b"compiled-cache")
            Path(f"{cache_path}.manifest.json").write_text(
                json.dumps({"runtime": {"torch": "different"}}),
                encoding="utf-8",
            )
            manager = dynamo_cache.DynamoCacheManager(_config(cache_path))

            with patch.object(dynamo_cache.torch.compiler, "load_cache_artifacts") as load_cache:
                loaded = manager.load()

            self.assertFalse(loaded)
            load_cache.assert_not_called()

    def test_exports_only_when_artifact_keys_grow(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.ptcache"
            manager = dynamo_cache.DynamoCacheManager(_config(cache_path))
            cache_info = _cache_info(aot_autograd=["aot-1"], inductor=["inductor-1"])

            with patch.object(
                dynamo_cache.torch.compiler,
                "save_cache_artifacts",
                return_value=(b"first-cache", cache_info),
            ):
                self.assertTrue(manager.export("first step"))

            self.assertEqual(dynamo_cache._unpack_segments(cache_path.read_bytes()), [b"first-cache"])
            manifest = json.loads(Path(f"{cache_path}.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_counts"], {"aot_autograd": 1, "inductor": 1})
            self.assertEqual(manifest["segment_count"], 1)

            with (
                patch.object(
                    dynamo_cache.torch.compiler,
                    "save_cache_artifacts",
                    return_value=(b"different-serialization", cache_info),
                ),
                patch.object(dynamo_cache, "_atomic_write") as atomic_write,
            ):
                self.assertFalse(manager.export("training completion"))
            atomic_write.assert_not_called()

    def test_new_process_segment_preserves_loaded_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.ptcache"
            cache_path.write_bytes(dynamo_cache._pack_segments([b"loaded-segment"]))
            manager = dynamo_cache.DynamoCacheManager(_config(cache_path))

            with patch.object(
                dynamo_cache.torch.compiler,
                "load_cache_artifacts",
                return_value=_cache_info(inductor=["loaded-key"]),
            ):
                self.assertTrue(manager.load())

            with patch.object(
                dynamo_cache.torch.compiler,
                "save_cache_artifacts",
                return_value=(b"current-process-v1", _cache_info(inductor=["new-key-1"])),
            ):
                self.assertTrue(manager.export("first step"))
            self.assertEqual(
                dynamo_cache._unpack_segments(cache_path.read_bytes()),
                [b"loaded-segment", b"current-process-v1"],
            )

            with patch.object(
                dynamo_cache.torch.compiler,
                "save_cache_artifacts",
                return_value=(
                    b"current-process-v2",
                    _cache_info(inductor=["new-key-1", "new-key-2"]),
                ),
            ):
                self.assertTrue(manager.export("training completion"))
            self.assertEqual(
                dynamo_cache._unpack_segments(cache_path.read_bytes()),
                [b"loaded-segment", b"current-process-v2"],
            )
            manifest = json.loads(Path(f"{cache_path}.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["segment_count"], 2)

    def test_rejects_future_manifest_schema_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.ptcache"
            cache_path.write_bytes(b"compiled-cache")
            Path(f"{cache_path}.manifest.json").write_text(
                json.dumps({"schema_version": dynamo_cache._SCHEMA_VERSION + 1}),
                encoding="utf-8",
            )
            manager = dynamo_cache.DynamoCacheManager(_config(cache_path))

            with patch.object(dynamo_cache.torch.compiler, "load_cache_artifacts") as load_cache:
                loaded = manager.load()

            self.assertFalse(loaded)
            load_cache.assert_not_called()

    def test_loads_hub_cache_before_local_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "compiler" / "cache.ptcache"
            remote_blob = Path(directory) / "remote-cache.ptcache"
            remote_manifest = Path(directory) / "remote-cache.manifest.json"
            payload = b"hub-cache"
            remote_blob.write_bytes(payload)
            manager = dynamo_cache.DynamoCacheManager(_config(local_path, "owner/cache"))
            remote_manifest.write_text(
                json.dumps(
                    {
                        "runtime": manager.runtime_signature,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            api = MagicMock()
            api.file_exists.return_value = True

            def _download(*, filename, **kwargs):
                if filename.endswith(".manifest.json"):
                    return str(remote_manifest)
                return str(remote_blob)

            with (
                patch.object(dynamo_cache, "HfApi", return_value=api),
                patch.object(dynamo_cache, "hf_hub_download", side_effect=_download),
                patch.object(
                    dynamo_cache.torch.compiler,
                    "load_cache_artifacts",
                    return_value=_cache_info(inductor=["hub-key"]),
                ),
            ):
                self.assertTrue(manager.load())

            self.assertEqual(manager.known_keys["inductor"], {"hub-key"})

    def test_uploads_blob_and_manifest_in_one_commit(self):
        manager = dynamo_cache.DynamoCacheManager(_config("cache/ltx.ptcache", "owner/cache"))
        api = MagicMock()
        with (
            patch.object(dynamo_cache, "HfApi", return_value=api),
            patch.object(manager, "_hub_token", return_value=None),
        ):
            manager._upload(b"blob", b"manifest", "first step")

        api.create_repo.assert_called_once_with(
            repo_id="owner/cache",
            token=None,
            exist_ok=True,
            private=True,
        )
        operations = api.create_commit.call_args.kwargs["operations"]
        self.assertEqual(
            [operation.path_in_repo for operation in operations],
            [
                "cache/ltx.ptcache",
                "cache/ltx.ptcache.manifest.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
