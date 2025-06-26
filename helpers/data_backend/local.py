from helpers.data_backend.base import BaseDataBackend
from helpers.image_manipulation.load import load_image, load_video
from helpers.training import video_file_extensions, image_file_extensions
from pathlib import Path
from io import BytesIO
import os
import logging
import torch
from typing import Any, List, Tuple, Union
from atomicwrites import atomic_write

logger = logging.getLogger("LocalDataBackend")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class LocalDataBackend(BaseDataBackend):
    def __init__(self, accelerator, id: str, compress_cache: bool = False):
        self.accelerator = accelerator
        self.id = id
        self.type = "local"
        self.compress_cache = compress_cache

    def read(self, filepath: str, as_byteIO: bool = False) -> Any:
        """Read and return the content of the file."""
        try:
            with open(filepath, "rb") as file:
                data = file.read()
                if not as_byteIO:
                    return data
                return BytesIO(data)
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise

    def write(self, filepath: str, data: Any) -> None:
        """Write the provided data to the specified filepath atomically."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mode = "wb"

        try:
            with atomic_write(
                filepath, mode=mode, overwrite=True, encoding=None
            ) as temp_file:
                if isinstance(data, Union[dict, torch.Tensor]):
                    self.torch_save(data, temp_file)
                elif isinstance(data, str):
                    temp_file.write(data.encode("utf-8"))
                elif isinstance(data, bytes):
                    temp_file.write(data)
                else:
                    logger.debug(
                        f"Received an unknown data type to write to disk. Attempting to write as bytes: {type(data)}"
                    )
                    temp_file.write(data)
        except Exception as e:
            logger.error(f"Failed to write data to {filepath}: {e}")
            raise

    def delete(self, filepath: str) -> None:
        """Delete the specified file."""
        try:
            if os.path.exists(filepath):
                logger.debug(f"Deleting file: {filepath}")
                os.remove(filepath)
                logger.info(f"Successfully deleted file: {filepath}")
            else:
                raise FileNotFoundError(f"{filepath} not found.")
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {e}")
            raise

        # Verify deletion
        if self.exists(filepath):
            error_msg = f"Failed to delete {filepath}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def exists(self, filepath: str) -> bool:
        """Check if the file exists."""
        return os.path.exists(filepath)

    def open_file(self, filepath: str, mode: str):
        """Open the file in the specified mode."""
        try:
            return open(filepath, mode)
        except Exception as e:
            logger.error(f"Error opening file {filepath} with mode {mode}: {e}")
            raise

    def list_files(
        self, file_extensions: List[str], instance_data_dir: str
    ) -> List[Tuple[str, List, List[str]]]:
        """
        List all files matching the given file extensions.
        Creates Path objects of each file found.
        """
        logger.debug(
            f"LocalDataBackend.list_files: file_extensions={file_extensions}, instance_data_dir={instance_data_dir}"
        )
        if not instance_data_dir:
            raise ValueError("instance_data_dir must be specified.")

        def _rglob_follow_symlinks(path: Path, extensions: List[str]):
            # Skip Spotlight and Jupyter directories
            forbidden_directories = {
                ".Spotlight-V100",
                ".Trashes",
                ".fseventsd",
                ".TemporaryItems",
                ".zfs",
                ".ipynb_checkpoints",
            }
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
                        if p.is_file():
                            yield p

            for p in path.iterdir():
                if p.is_dir() and not p.is_symlink():
                    yield from _rglob_follow_symlinks(p, extensions)
                elif p.is_symlink():
                    try:
                        real_path = p.resolve()
                        if real_path.is_dir():
                            yield from _rglob_follow_symlinks(real_path, extensions)
                    except Exception as e:
                        logger.warning(f"Broken symlink encountered: {p} - {e}")

        # Prepare the extensions for globbing
        extensions = (
            [f"*.{ext.lower()}" for ext in file_extensions] if file_extensions else None
        )

        paths = list(_rglob_follow_symlinks(Path(instance_data_dir), extensions))

        # Group files by their parent directory
        path_dict = {}
        for path in paths:
            parent = str(path.parent)
            path_dict.setdefault(parent, []).append(str(path.absolute()))

        results = [(subdir, [], files) for subdir, files in path_dict.items()]
        return results

    def get_abs_path(self, sample_path: str) -> str:
        """
        Given a relative path of a sample, return the absolute path.
        If sample_path is None, return the current working directory.
        """
        if sample_path is None:
            sample_path = os.getcwd()
        abs_path = os.path.abspath(sample_path)

        return abs_path

    def read_image(self, filepath: str, delete_problematic_images: bool = False) -> Any:
        """Read an image from the specified filepath."""
        filepath = filepath.replace("\x00", "")
        file_extension = os.path.splitext(filepath)[1].lower().strip(".")
        file_loader = load_image
        if file_extension in video_file_extensions:
            file_loader = load_video
        try:
            image = file_loader(filepath)
            return image
        except Exception as e:
            logger.error(
                f"Encountered error opening image {filepath}: {e}", exc_info=True
            )
            if delete_problematic_images:
                try:
                    logger.error(
                        "Deleting image, because --delete_problematic_images is provided."
                    )
                    self.delete(filepath)
                except Exception as del_e:
                    logger.error(
                        f"Failed to delete problematic image {filepath}: {del_e}"
                    )
            else:
                raise e

    def read_image_batch(
        self, filepaths: List[str], delete_problematic_images: bool = False
    ) -> Tuple[List[str], List[Any]]:
        """Read a batch of images from the specified filepaths."""
        if not isinstance(filepaths, list):
            raise ValueError(
                f"read_image_batch must be given a list of image filepaths. Received type: {type(filepaths)}"
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
                    try:
                        self.delete(filepath)
                    except Exception as del_e:
                        logger.error(
                            f"Failed to delete problematic image {filepath}: {del_e}"
                        )
                else:
                    logger.warning(
                        f"A problematic image {filepath} is detected, but we are not allowed to remove it, because --delete_problematic_images is not provided."
                        f" Please correct this manually. Error: {e}"
                    )
        return available_keys, output_images

    def create_directory(self, directory_path: str) -> None:
        """Create a directory if it does not exist."""
        if os.path.exists(directory_path):
            return
        try:
            logger.debug(f"Creating directory: {directory_path}")
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Directory created: {directory_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise

    def torch_load(self, filename: str) -> torch.Tensor:
        """
        Load a torch tensor from a file.
        """
        if not self.exists(filename):
            error_msg = f"{filename} not found."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with self.read(filename, as_byteIO=True) as stored_tensor:
                if self.compress_cache:
                    stored_tensor = self._decompress_torch(stored_tensor)
                stored_tensor.seek(0)
                loaded_tensor = torch.load(stored_tensor, map_location="cpu")
            return loaded_tensor
        except Exception as e:
            logger.error(f"Failed to load torch file '{filename}': {e}", exc_info=True)
            if "invalid load key" in str(e):
                try:
                    self.delete(filename)
                    logger.info(f"Deleted corrupt torch file: {filename}")
                except Exception as del_e:
                    logger.error(
                        f"Failed to delete corrupt torch file {filename}: {del_e}"
                    )
            raise e

    def torch_save(self, data: torch.Tensor, original_location: Any) -> None:
        """
        Save a torch tensor to a file object or filepath.
        """
        try:
            if isinstance(original_location, str):
                # original_location is a filepath
                with atomic_write(
                    original_location, mode="wb", overwrite=True, encoding=None
                ) as temp_file:
                    if self.compress_cache:
                        compressed_data = self._compress_torch(data)
                        temp_file.write(compressed_data)
                    else:
                        torch.save(data, temp_file)
            else:
                # original_location is a file-like object
                if self.compress_cache:
                    compressed_data = self._compress_torch(data)
                    original_location.write(compressed_data)
                else:
                    torch.save(data, original_location)
                original_location.flush()
                os.fsync(original_location.fileno())
        except Exception as e:
            logger.error(f"Failed to save torch tensor: {e}", exc_info=True)
            raise

    def write_batch(self, filepaths: List[str], data_list: List[Any]) -> None:
        """Write a batch of data to the specified filepaths atomically."""
        if len(filepaths) != len(data_list):
            error_msg = "filepaths and data_list must have the same length."
            logger.error(error_msg)
            raise ValueError(error_msg)

        for filepath, data in zip(filepaths, data_list):
            try:
                self.write(filepath, data)
                logger.debug(f"Successfully wrote to {filepath}")
            except Exception as e:
                logger.error(f"Failed to write to {filepath}: {e}")
                raise
