import atexit
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Optional

from simpletuner.helpers.data_backend.local import LocalDataBackend

logger = logging.getLogger("MemoryDataBackend")


class MemoryDataBackend(LocalDataBackend):
    """Cache storage backend backed by tmpfs or a macOS RAM disk."""

    def __init__(
        self,
        accelerator,
        id: str,
        source_path: Optional[str] = None,
        mount_path: Optional[str] = None,
        filesystem_size: Optional[str] = None,
        filesystem_sudo: bool = False,
        compress_cache: bool = False,
        attach_existing: bool = False,
    ):
        super().__init__(accelerator=accelerator, id=id, compress_cache=compress_cache)
        self.type = "memory"
        self.source_path = os.path.abspath(source_path) if source_path else None
        self.mount_path = os.path.abspath(mount_path or os.path.join(tempfile.gettempdir(), "simpletuner-memory", self.id))
        self.filesystem_size = str(filesystem_size) if filesystem_size is not None else None
        self.filesystem_sudo = filesystem_sudo
        self.filesystem_platform = platform.system()
        self._device_path: Optional[str] = None
        self._volume_device_path: Optional[str] = None
        self._owns_mount = False
        self._closed = False

        if self.source_path:
            source = Path(self.source_path)
            mount = Path(self.mount_path)
            if source == mount or source in mount.parents or mount in source.parents:
                raise ValueError("memory_filesystem_path and the source cache directory must not overlap.")

        if attach_existing:
            if not os.path.ismount(self.mount_path):
                raise RuntimeError(f"Memory backend mount is not available at {self.mount_path}.")
            return

        is_local_main_process = accelerator is None or getattr(accelerator, "is_local_main_process", True)
        if is_local_main_process:
            self._mount()
            try:
                self._preload_source()
            except Exception:
                self.close()
                raise

        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        if not os.path.ismount(self.mount_path):
            raise RuntimeError(f"Memory backend filesystem is not available at {self.mount_path}.")

    def _privileged_command(self, command: list[str]) -> list[str]:
        if os.geteuid() == 0:
            return command
        if self.filesystem_sudo:
            if shutil.which("sudo") is None:
                raise RuntimeError("memory_filesystem_sudo=true requires sudo to be installed and available on PATH.")
            return ["sudo", "-n", *command]
        raise PermissionError(
            "Mounting a memory backend requires root privileges. Run as root or set "
            "memory_filesystem_sudo=true for non-interactive sudo."
        )

    def _prepare_mount_directory(self) -> None:
        if os.path.ismount(self.mount_path):
            raise RuntimeError(f"Memory backend mount path is already mounted: {self.mount_path}")

        mount_directory = Path(self.mount_path)
        mount_directory.mkdir(parents=True, exist_ok=True)
        if any(mount_directory.iterdir()):
            raise RuntimeError(f"Memory backend mount path must be empty: {self.mount_path}")

    @staticmethod
    def _run_command(command: list[str], action: str) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{action} requires the '{command[0]}' command to be available on PATH.") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"{action} failed: {detail}")
        return result

    @staticmethod
    def _macos_sector_count(filesystem_size: Optional[str]) -> int:
        if not filesystem_size:
            raise ValueError("memory_filesystem_size is required for macOS RAM disks.")

        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTPE]?)(?:i?B)?\s*", filesystem_size, re.IGNORECASE)
        if not match:
            raise ValueError("memory_filesystem_size must be a byte size such as '4096M' or '4G' for macOS RAM disks.")

        amount = Decimal(match.group(1))
        multipliers = {
            "": 1,
            "K": 1024,
            "M": 1024**2,
            "G": 1024**3,
            "T": 1024**4,
            "P": 1024**5,
            "E": 1024**6,
        }
        byte_count = int(amount * multipliers[match.group(2).upper()])
        if byte_count <= 0:
            raise ValueError("memory_filesystem_size must be greater than zero.")
        return (byte_count + 511) // 512

    def _mount_linux(self) -> None:
        options = ["size=" + self.filesystem_size] if self.filesystem_size else []
        command = ["mount", "-t", "tmpfs"]
        if options:
            command.extend(["-o", ",".join(options)])
        command.extend(["tmpfs", self.mount_path])

        self._run_command(self._privileged_command(command), f"Mounting tmpfs at {self.mount_path}")

    def _mount_macos(self) -> None:
        sectors = self._macos_sector_count(self.filesystem_size)
        attach_result = self._run_command(
            ["hdiutil", "attach", "-nomount", f"ram://{sectors}"],
            "Creating macOS RAM disk",
        )
        self._device_path = next(
            (line.split()[0] for line in attach_result.stdout.splitlines() if line.strip().startswith("/dev/")),
            None,
        )
        if not self._device_path:
            raise RuntimeError("Creating macOS RAM disk did not return a device identifier.")

        volume_id = re.sub(r"[^A-Za-z0-9_-]+", "-", self.id).strip("-")[:32] or "cache"
        volume_label = f"SimpleTuner-{volume_id}-{os.getpid()}"
        try:
            self._run_command(
                ["diskutil", "eraseVolume", "APFS", volume_label, self._device_path],
                f"Formatting macOS RAM disk {self._device_path}",
            )
            info_result = self._run_command(
                ["diskutil", "info", "-plist", f"/Volumes/{volume_label}"],
                f"Inspecting macOS RAM disk {self._device_path}",
            )
            try:
                volume_info = plistlib.loads(info_result.stdout.encode("utf-8"))
            except plistlib.InvalidFileException as exc:
                raise RuntimeError("macOS RAM disk information did not contain a valid volume device.") from exc
            volume_device_path = volume_info.get("DeviceNode")
            if not isinstance(volume_device_path, str) or not volume_device_path.startswith("/dev/"):
                raise RuntimeError("macOS RAM disk information did not contain a valid volume device.")
            self._volume_device_path = volume_device_path
            self._run_command(
                ["diskutil", "unmount", self._volume_device_path],
                f"Unmounting the initial macOS RAM disk volume {self._volume_device_path}",
            )
            self._run_command(
                ["diskutil", "mount", "-mountPoint", self.mount_path, self._volume_device_path],
                f"Mounting macOS RAM disk at {self.mount_path}",
            )
        except Exception:
            subprocess.run(
                ["hdiutil", "detach", self._device_path],
                check=False,
                capture_output=True,
                text=True,
            )
            self._device_path = None
            self._volume_device_path = None
            raise

    def _mount(self) -> None:
        if self.filesystem_platform not in {"Linux", "Darwin"}:
            raise RuntimeError("MemoryDataBackend requires Linux tmpfs or macOS RAM disk support.")

        self._prepare_mount_directory()
        if self.filesystem_platform == "Linux":
            self._mount_linux()
        else:
            self._mount_macos()

        if not os.path.ismount(self.mount_path):
            if self.filesystem_platform == "Darwin" and self._device_path:
                subprocess.run(
                    ["hdiutil", "detach", self._device_path],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._device_path = None
                self._volume_device_path = None
            raise RuntimeError(f"Mount command completed but {self.mount_path} is not a mount point.")

        self._owns_mount = True
        atexit.register(self.close)

    def _preload_source(self) -> None:
        if not self.source_path or not os.path.exists(self.source_path):
            return
        if not os.path.isdir(self.source_path):
            raise ValueError(f"Memory backend source path must be a directory: {self.source_path}")

        shutil.copytree(self.source_path, self.mount_path, dirs_exist_ok=True)

    def close(self) -> None:
        if self._closed or not self._owns_mount:
            return
        if self.filesystem_platform == "Darwin":
            command = ["hdiutil", "detach", self._device_path]
        else:
            command = self._privileged_command(["umount", self.mount_path])
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown unmount error"
            logger.warning("Failed to release memory backend at %s: %s", self.mount_path, detail)
            return
        self._closed = True
        self._device_path = None
        self._volume_device_path = None

    def get_instance_representation(self) -> dict:
        return {
            "backend_type": "memory",
            "id": self.id,
            "source_path": self.source_path,
            "mount_path": self.mount_path,
            "filesystem_size": self.filesystem_size,
            "filesystem_sudo": self.filesystem_sudo,
            "compress_cache": self.compress_cache,
        }

    @staticmethod
    def from_instance_representation(representation: dict) -> "MemoryDataBackend":
        if representation.get("backend_type") != "memory":
            raise ValueError(f"Expected backend_type 'memory', got {representation.get('backend_type')}")
        return MemoryDataBackend(
            accelerator=None,
            id=representation["id"],
            source_path=representation.get("source_path"),
            mount_path=representation["mount_path"],
            filesystem_size=representation.get("filesystem_size"),
            filesystem_sudo=representation.get("filesystem_sudo", False),
            compress_cache=representation.get("compress_cache", False),
            attach_existing=True,
        )
