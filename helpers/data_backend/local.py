from helpers.data_backend.base import BaseDataBackend
from helpers.image_manipulation.load import load_image
from pathlib import Path
from io import BytesIO
import os, logging, torch, gzip
from typing import Any

logger = logging.getLogger("LocalDataBackend")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class LocalDataBackend(BaseDataBackend):
    def __init__(self, accelerator, id: str, compress_cache: bool = False):
        self.accelerator = accelerator
        self.id = id
        self.compress_cache = compress_cache

    def read(self, filepath, as_byteIO: bool = False):
        """Read and return the content of the file."""
        # Openfilepath as BytesIO:
        with open(filepath, "rb") as file:
            data = file.read()
        if not as_byteIO:
            return data
        return BytesIO(data)

    def write(self, filepath: str, data: Any) -> None:
        """Write the provided data to the specified filepath."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            # Check if data is a Tensor, and if so, save it appropriately
            if isinstance(data, torch.Tensor):
                # logger.debug(f"Writing a torch file to disk.")
                return self.torch_save(data, file)
            elif isinstance(data, str):
                # logger.debug(f"Writing a string to disk as {filepath}: {data}")
                data = data.encode("utf-8")
            else:
                logger.debug(
                    f"Received an unknown data type to write to disk. Doing our best: {type(data)}"
                )
            file.write(data)

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

    def list_files(self, str_pattern: str, instance_data_root: str):
        """
        List all files matching the pattern.
        Creates Path objects of each file found.
        """
        logger.debug(
            f"LocalDataBackend.list_files: str_pattern={str_pattern}, instance_data_root={instance_data_root}"
        )
        if instance_data_root is None:
            raise ValueError("instance_data_root must be specified.")

        def _rglob_follow_symlinks(path: Path, pattern: str):
            # Skip Spotlight directories
            forbidden_directories = [
                ".Spotlight-V100",
                ".Trashes",
                ".fseventsd",
                ".TemporaryItems",
                ".zfs",
            ]
            # Add Jupyter directories
            forbidden_directories += [".ipynb_checkpoints"]
            if path.name in forbidden_directories:
                return

            for p in path.glob(pattern):
                yield p
            for p in path.iterdir():
                if p.is_dir() and not p.is_symlink():
                    yield from _rglob_follow_symlinks(p, pattern)
                elif p.is_symlink():
                    real_path = Path(os.readlink(p))
                    if real_path.is_dir():
                        yield from _rglob_follow_symlinks(real_path, pattern)

        paths = list(_rglob_follow_symlinks(Path(instance_data_root), str_pattern))

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
                    f"Deleting image, because --delete_problematic_images is provided."
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
                logger.error(
                    f"Failed to decompress torch file, falling back to passthrough: {e}"
                )
        if hasattr(stored_tensor, "seek"):
            stored_tensor.seek(0)
        try:
            loaded_tensor = torch.load(stored_tensor, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load torch file '{filename}': {e}")
            raise e
        return loaded_tensor

    def torch_save(self, data, original_location):
        """
        Save a torch tensor to a file.
        """
        if isinstance(original_location, str):
            location = self.open_file(original_location, "wb")
        else:
            location = original_location

        if self.compress_cache:
            compressed_data = self._compress_torch(data)
            location.write(compressed_data)
        else:
            torch.save(data, location)
        location.close()

    def write_batch(self, filepaths: list, data_list: list) -> None:
        """Write a batch of data to the specified filepaths."""
        for filepath, data in zip(filepaths, data_list):
            self.write(filepath, data)
