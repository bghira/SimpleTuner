from helpers.data_backend.base import BaseDataBackend
from pathlib import Path
import os


class LocalDataBackend(BaseDataBackend):
    def read(self, filepath):
        """Read and return the content of the file."""
        with open(filepath, "rb") as file:
            return file.read()

    def write(self, filepath, data):
        """Write data to the specified file."""
        # Convert data to Bytes:    
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(filepath, "wb") as file:
            file.write(data)

    def delete(self, filepath):
        """Delete the specified file."""
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            raise FileNotFoundError(f"{filepath} not found.")

    def exists(self, filepath):
        """Check if the file exists."""
        return os.path.exists(filepath)
    
    def open_file(self, filepath, mode):
        """Open the file in the specified mode."""
        return open(filepath, mode)
    
    def list_files(self, str_pattern: str, instance_data_root: str):
        """
        List all files matching the pattern.
        Creates Path objects of each file found.
        """
        if instance_data_root is None:
            raise ValueError("instance_data_root must be specified.")
        def _rglob_follow_symlinks(path: Path, pattern: str):
            for p in path.glob(pattern):
                yield p
            for p in path.iterdir():
                if p.is_dir() and not p.is_symlink():
                    yield from _rglob_follow_symlinks(p, pattern)
                elif p.is_symlink():
                    real_path = Path(os.readlink(p))
                    if real_path.is_dir():
                        yield from _rglob_follow_symlinks(real_path, pattern)

        return list(_rglob_follow_symlinks(Path(instance_data_root), str_pattern))
