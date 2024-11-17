from helpers.data_backend.base import BaseDataBackend
from helpers.image_manipulation.load import load_image
from pathlib import Path
from io import BytesIO
import os
import logging
import torch
from typing import Any
from regex import regex
import fcntl
import tempfile
import shutil
from helpers.training.multi_process import _get_rank

logger = logging.getLogger("LocalDataBackend")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class LocalDataBackend(BaseDataBackend):
    def __init__(self, accelerator, id: str, compress_cache: bool = False):
        self.accelerator = accelerator
        self.id = id
        self.type = "local"
        self.compress_cache = compress_cache

    def read(self, filepath, as_byteIO: bool = False):
        """Read and return the content of the file."""
        with open(filepath, "rb") as file:
            # Acquire a shared lock
            fcntl.flock(file, fcntl.LOCK_SH)
            try:
                data = file.read()
                if not as_byteIO:
                    return data
                return BytesIO(data)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)

    def write(self, filepath: str, data: Any) -> None:
        """Write the provided data to the specified filepath."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_dir = os.path.dirname(filepath)
        temp_file_path = os.path.join(temp_dir, f".{os.path.basename(filepath)}.tmp{_get_rank()}")

        # Open the temporary file for writing
        with open(temp_file_path, "wb") as temp_file:
            # Acquire an exclusive lock on the temporary file
            fcntl.flock(temp_file, fcntl.LOCK_EX)
            try:
                # Write data to the temporary file
                if isinstance(data, torch.Tensor):
                    # Use the torch_save method, passing the temp file
                    self.torch_save(data, temp_file)
                    os.rename(temp_file_path, filepath)
                    return  # torch_save handles closing the file
                elif isinstance(data, str):
                    data = data.encode("utf-8")
                else:
                    logger.debug(
                        f"Received an unknown data type to write to disk. Doing our best: {type(data)}"
                    )
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            finally:
                # Release the lock
                fcntl.flock(temp_file, fcntl.LOCK_UN)

        # Atomically replace the target file with the temporary file
        os.rename(temp_file_path, filepath)


    def delete(self, filepath):
        """Delete the specified file."""
        if os.path.exists(filepath):
            logger.debug(f"Deleting file: {filepath}")
            os.remove(filepath)
        else:
            raise FileNotFoundError(f"{filepath} not found.")
        # Check if file exists:
        if self.exists(filepath):
            raise Exception(f"Failed to delete {filepath}")

    def exists(self, filepath):
        """Check if the file exists."""
        return os.path.exists(filepath)

    def open_file(self, filepath, mode):
        """Open the file in the specified mode."""
        return open(filepath, mode)

    def list_files(self, file_extensions: list, instance_data_dir: str):
        """
        List all files matching the given file extensions.
        Creates Path objects of each file found.
        """
        logger.debug(
            f"LocalDataBackend.list_files: file_extensions={file_extensions}, instance_data_dir={instance_data_dir}"
        )
        if instance_data_dir is None:
            raise ValueError("instance_data_dir must be specified.")

        def _rglob_follow_symlinks(path: Path, extensions: list):
            # Skip Spotlight and Jupyter directories
            forbidden_directories = [
                ".Spotlight-V100",
                ".Trashes",
                ".fseventsd",
                ".TemporaryItems",
                ".zfs",
                ".ipynb_checkpoints",
            ]
            if path.name in forbidden_directories:
                return

            # If no extensions are provided, list all files
            if not extensions:
                for p in path.rglob("*"):
                    if p.is_file():
                        yield p
            else:
                for ext in extensions:
                    for p in path.rglob(ext):
                        yield p

            for p in path.iterdir():
                if p.is_dir() and not p.is_symlink():
                    yield from _rglob_follow_symlinks(p, extensions)
                elif p.is_symlink():
                    real_path = Path(os.readlink(p))
                    if real_path.is_dir():
                        yield from _rglob_follow_symlinks(real_path, extensions)

        # If file_extensions is None, list all files
        extensions = (
            [f"*.{ext.lower()}" for ext in file_extensions] if file_extensions else None
        )

        paths = list(_rglob_follow_symlinks(Path(instance_data_dir), extensions))

        # Group files by their parent directory
        path_dict = {}
        for path in paths:
            parent = str(path.parent)
            if parent not in path_dict:
                path_dict[parent] = []
            path_dict[parent].append(str(path.absolute()))

        results = [(subdir, [], files) for subdir, files in path_dict.items()]
        return results

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        # Remove embedded null byte:
        filepath = filepath.replace("\x00", "")
        try:
            image = load_image(filepath)
            return image
        except Exception as e:
            import traceback

            logger.error(
                f"Encountered error opening image {filepath}: {e}, traceback: {traceback.format_exc()}"
            )
            if delete_problematic_images:
                logger.error(
                    "Deleting image, because --delete_problematic_images is provided."
                )
                self.delete(filepath)
            else:
                exit(1)
                raise e

    def read_image_batch(
        self, filepaths: list, delete_problematic_images: bool = False
    ) -> list:
        """Read a batch of images from the specified filepaths."""
        if type(filepaths) != list:
            raise ValueError(
                f"read_image_batch must be given a list of image filepaths. we received: {filepaths}"
            )
        output_images = []
        available_keys = []
        for filepath in filepaths:
            try:
                image_data = self.read_image(filepath, delete_problematic_images)
                if image_data is None:
                    logger.warning(f"Unable to load image '{filepath}', skipping.")
                    continue
                output_images.append(image_data)
                available_keys.append(filepath)
            except Exception as e:
                if delete_problematic_images:
                    logger.error(
                        f"Deleting image '{filepath}', because --delete_problematic_images is provided. Error: {e}"
                    )
                else:
                    logger.warning(
                        f"A problematic image {filepath} is detected, but we are not allowed to remove it, because --delete_problematic_image is not provided."
                        f" Please correct this manually. Error: {e}"
                    )
        return (available_keys, output_images)

    def create_directory(self, directory_path):
        if os.path.exists(directory_path):
            return
        logger.debug(f"Creating directory: {directory_path}")
        os.makedirs(directory_path, exist_ok=True)

    def torch_load(self, filename):
        """
        Load a torch tensor from a file.
        """
        if not self.exists(filename):
            raise FileNotFoundError(f"{filename} not found.")

        stored_tensor = self.read(filename, as_byteIO=True)

        if self.compress_cache:
            try:
                stored_tensor = self._decompress_torch(stored_tensor)
            except Exception as e:
                pass

        if hasattr(stored_tensor, "seek"):
            stored_tensor.seek(0)
        try:
            loaded_tensor = torch.load(stored_tensor, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load corrupt torch file '{filename}': {e}")
            if "invalid load key" in str(e):
                self.delete(filename)
            raise e
        return loaded_tensor

    def torch_save(self, data, original_location):
        """
        Save a torch tensor to a file.
        """
        if isinstance(original_location, str):
            filepath = original_location
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            temp_dir = os.path.dirname(filepath)
            temp_file_path = os.path.join(temp_dir, f".{os.path.basename(filepath)}.tmp{_get_rank()}")

            with open(temp_file_path, "wb") as temp_file:
                # Acquire an exclusive lock on the temporary file
                fcntl.flock(temp_file, fcntl.LOCK_EX)
                try:
                    if self.compress_cache:
                        compressed_data = self._compress_torch(data)
                        temp_file.write(compressed_data)
                    else:
                        torch.save(data, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                finally:
                    # Release the lock
                    fcntl.flock(temp_file, fcntl.LOCK_UN)
            # Atomically replace the target file with the temporary file
            os.rename(temp_file_path, filepath)
        else:
            # Handle the case where original_location is a file object
            temp_file = original_location
            # Acquire an exclusive lock on the file object
            fcntl.flock(temp_file, fcntl.LOCK_EX)
            try:
                if self.compress_cache:
                    compressed_data = self._compress_torch(data)
                    temp_file.write(compressed_data)
                else:
                    torch.save(data, temp_file)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            finally:
                # Release the lock
                fcntl.flock(temp_file, fcntl.LOCK_UN)

    def write_batch(self, filepaths: list, data_list: list) -> None:
        """Write a batch of data to the specified filepaths."""
        for filepath, data in zip(filepaths, data_list):
            self.write(filepath, data)
