"""Tests for Linux tmpfs and macOS RAM-disk cache storage."""

import os
import plistlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.data_backend.memory import MemoryDataBackend


class TestMemoryDataBackend(unittest.TestCase):
    def setUp(self):
        self.source_temp = tempfile.TemporaryDirectory()
        self.mount_temp = tempfile.TemporaryDirectory()
        self.source = Path(self.source_temp.name)
        self.mount = Path(self.mount_temp.name)

    def tearDown(self):
        self.source_temp.cleanup()
        self.mount_temp.cleanup()

    def _backend(self, *, sudo=False, size=None):
        mount_state = iter([False, True, True])
        with (
            patch("simpletuner.helpers.data_backend.memory.platform.system", return_value="Linux"),
            patch("simpletuner.helpers.data_backend.memory.os.geteuid", return_value=0),
            patch("simpletuner.helpers.data_backend.memory.os.path.ismount", side_effect=lambda _path: next(mount_state)),
            patch(
                "simpletuner.helpers.data_backend.memory.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
            ) as run,
        ):
            backend = MemoryDataBackend(
                accelerator=None,
                id="memory-test",
                source_path=str(self.source),
                mount_path=str(self.mount),
                filesystem_size=size,
                filesystem_sudo=sudo,
            )
        return backend, run

    def test_mounts_tmpfs_and_preloads_source(self):
        (self.source / "nested").mkdir()
        (self.source / "nested" / "embedding.pt").write_text("sample", encoding="utf-8")

        backend, run = self._backend(size="64G")

        self.assertIsInstance(backend, MemoryDataBackend)
        self.assertEqual(backend.type, "memory")
        self.assertEqual((self.mount / "nested" / "embedding.pt").read_text(encoding="utf-8"), "sample")
        self.assertEqual(
            run.call_args.args[0],
            ["mount", "-t", "tmpfs", "-o", "size=64G", "tmpfs", str(self.mount)],
        )
        backend._closed = True

    def test_inherits_local_read_write_and_torch_operations(self):
        backend, _run = self._backend()
        byte_path = self.mount / "bytes" / "value.bin"
        tensor_path = self.mount / "tensors" / "value.pt"

        backend.write(str(byte_path), b"value")
        backend.create_directory(str(tensor_path.parent))
        backend.torch_save(torch.tensor([1, 2, 3]), str(tensor_path))

        self.assertEqual(backend.read(str(byte_path)), b"value")
        torch.testing.assert_close(backend.torch_load(str(tensor_path)), torch.tensor([1, 2, 3]))
        backend._closed = True

    def test_uses_noninteractive_sudo_when_enabled(self):
        backend = MemoryDataBackend.__new__(MemoryDataBackend)
        backend.filesystem_sudo = True
        with (
            patch("simpletuner.helpers.data_backend.memory.os.geteuid", return_value=1000),
            patch("simpletuner.helpers.data_backend.memory.shutil.which", return_value="/usr/bin/sudo"),
        ):
            command = backend._privileged_command(["mount", "tmpfs"])
        self.assertEqual(command, ["sudo", "-n", "mount", "tmpfs"])

    def test_rejects_sudo_when_command_is_unavailable(self):
        backend = MemoryDataBackend.__new__(MemoryDataBackend)
        backend.filesystem_sudo = True
        with (
            patch("simpletuner.helpers.data_backend.memory.os.geteuid", return_value=1000),
            patch("simpletuner.helpers.data_backend.memory.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "requires sudo to be installed"),
        ):
            backend._privileged_command(["mount", "tmpfs"])

    def test_rejects_overlapping_source_and_mount_paths(self):
        overlapping_paths = [
            (self.source, self.source),
            (self.source, self.source / "tmpfs"),
            (self.mount / "cache", self.mount),
        ]
        for source, mount in overlapping_paths:
            with self.subTest(source=source, mount=mount), self.assertRaisesRegex(ValueError, "must not overlap"):
                MemoryDataBackend(
                    accelerator=None,
                    id="memory-test",
                    source_path=str(source),
                    mount_path=str(mount),
                )

    def test_rejects_unprivileged_mount_without_sudo(self):
        backend = MemoryDataBackend.__new__(MemoryDataBackend)
        backend.filesystem_sudo = False
        with (
            patch("simpletuner.helpers.data_backend.memory.os.geteuid", return_value=1000),
            self.assertRaisesRegex(PermissionError, "memory_filesystem_sudo=true"),
        ):
            backend._privileged_command(["mount", "tmpfs"])

    def test_rejects_unsupported_platform(self):
        backend = MemoryDataBackend.__new__(MemoryDataBackend)
        backend.mount_path = str(self.mount)
        backend.filesystem_platform = "Windows"
        with self.assertRaisesRegex(RuntimeError, "Linux tmpfs or macOS RAM disk"):
            backend._mount()

    def test_macos_sector_count(self):
        self.assertEqual(MemoryDataBackend._macos_sector_count("4G"), 8_388_608)
        self.assertEqual(MemoryDataBackend._macos_sector_count("1.5GiB"), 3_145_728)
        self.assertEqual(MemoryDataBackend._macos_sector_count("512"), 1)

    def test_macos_sector_count_requires_byte_size(self):
        for value in (None, "80%", "invalid", "0G"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MemoryDataBackend._macos_sector_count(value)

    def test_mounts_and_formats_macos_ram_disk(self):
        (self.source / "embedding.pt").write_text("sample", encoding="utf-8")
        command_results = [
            SimpleNamespace(returncode=0, stderr="", stdout="/dev/disk9\n"),
            SimpleNamespace(returncode=0, stderr="", stdout="Finished erase"),
            SimpleNamespace(
                returncode=0,
                stderr="",
                stdout=plistlib.dumps({"DeviceNode": "/dev/disk10s1"}).decode("utf-8"),
            ),
            SimpleNamespace(returncode=0, stderr="", stdout="Unmounted"),
            SimpleNamespace(returncode=0, stderr="", stdout="Mounted"),
        ]
        mount_state = iter([False, True, True])
        with (
            patch("simpletuner.helpers.data_backend.memory.platform.system", return_value="Darwin"),
            patch("simpletuner.helpers.data_backend.memory.os.getpid", return_value=1234),
            patch("simpletuner.helpers.data_backend.memory.os.path.ismount", side_effect=lambda _path: next(mount_state)),
            patch("simpletuner.helpers.data_backend.memory.subprocess.run", side_effect=command_results) as run,
        ):
            backend = MemoryDataBackend(
                accelerator=None,
                id="memory-test",
                source_path=str(self.source),
                mount_path=str(self.mount),
                filesystem_size="4G",
            )

        self.assertEqual(backend._device_path, "/dev/disk9")
        self.assertEqual(backend._volume_device_path, "/dev/disk10s1")
        self.assertEqual((self.mount / "embedding.pt").read_text(encoding="utf-8"), "sample")
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["hdiutil", "attach", "-nomount", "ram://8388608"],
                ["diskutil", "eraseVolume", "APFS", "SimpleTuner-memory-test-1234", "/dev/disk9"],
                ["diskutil", "info", "-plist", "/Volumes/SimpleTuner-memory-test-1234"],
                ["diskutil", "unmount", "/dev/disk10s1"],
                ["diskutil", "mount", "-mountPoint", str(self.mount), "/dev/disk10s1"],
            ],
        )
        backend._closed = True

    def test_close_unmounts_owned_filesystem(self):
        backend, _run = self._backend()
        with (
            patch("simpletuner.helpers.data_backend.memory.os.geteuid", return_value=0),
            patch(
                "simpletuner.helpers.data_backend.memory.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
            ) as run,
        ):
            backend.close()

        run.assert_called_once_with(
            ["umount", str(self.mount)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_close_detaches_owned_macos_ram_disk(self):
        backend = MemoryDataBackend.__new__(MemoryDataBackend)
        backend._closed = False
        backend._owns_mount = True
        backend.filesystem_platform = "Darwin"
        backend._device_path = "/dev/disk9"
        backend.mount_path = str(self.mount)

        with patch(
            "simpletuner.helpers.data_backend.memory.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
        ) as run:
            backend.close()

        run.assert_called_once_with(
            ["hdiutil", "detach", "/dev/disk9"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertTrue(backend._closed)

    def test_instance_representation_attaches_to_existing_mount(self):
        representation = {
            "backend_type": "memory",
            "id": "memory-test",
            "source_path": str(self.source),
            "mount_path": str(self.mount),
            "filesystem_size": "32G",
            "filesystem_sudo": False,
            "compress_cache": True,
        }
        with patch("simpletuner.helpers.data_backend.memory.os.path.ismount", return_value=True):
            backend = MemoryDataBackend.from_instance_representation(representation)

        self.assertEqual(backend.mount_path, os.path.abspath(self.mount))
        self.assertEqual(backend.filesystem_size, "32G")
        self.assertTrue(backend.compress_cache)


if __name__ == "__main__":
    unittest.main()
