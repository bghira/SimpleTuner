import hashlib
import logging
import torch
from pathlib import Path
from io import BytesIO
import os
from typing import Any, Union, Optional, BinaryIO
from PIL import Image
import numpy as np

from helpers.data_backend.base import BaseDataBackend
from helpers.image_manipulation.load import load_image
from helpers.training.multi_process import should_log

logger = logging.getLogger("HuggingfaceDatasetsBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class HuggingfaceDatasetsBackend(BaseDataBackend):
    def __init__(
        self,
        accelerator,
        id: str,
        dataset_name: str,
        split: str = "train",
        revision: str = None,
        image_column: str = "image",
        cache_dir: Optional[str] = None,
        compress_cache: bool = False,
        streaming: bool = False,
        filter_func: Optional[callable] = None,
        num_proc: int = 16,
    ):
        """
        Initialize the Hugging Face datasets backend.

        Args:
            accelerator: The accelerator instance
            id: Unique identifier for this backend
            dataset_name: Name of the HF dataset (e.g., 'Yuanshi/Subjects200K')
            split: Dataset split to use (default: 'train')
            revision: Dataset revision/version
            image_column: Column name containing images
            cache_dir: Local cache directory for HF datasets
            compress_cache: Whether to compress cached data
            streaming: Whether to use streaming mode
            filter_func: Optional function to filter dataset items
            num_proc: Number of processes for filtering
        """
        self.id = id
        self.type = "huggingface"
        self.accelerator = accelerator
        self.dataset_name = dataset_name
        self.split = split
        self.revision = revision
        self.image_column = image_column
        self.cache_dir = cache_dir
        self.compress_cache = compress_cache
        self.streaming = streaming
        self.filter_func = filter_func
        self.num_proc = num_proc

        # Virtual file system mapping: index -> virtual path
        self._path_to_index = {}
        self._index_to_path = {}

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the Hugging Face dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        logger.info(f"Loading dataset {self.dataset_name} (split: {self.split})")

        self.dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            revision=self.revision,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )

        # Apply filter if provided
        if self.filter_func and not self.streaming:
            logger.info("Applying filter to dataset...")
            self.dataset = self.dataset.filter(
                self.filter_func,
                num_proc=self.num_proc,
            )
            logger.info(f"Dataset filtered to {len(self.dataset)} items")

        # Build virtual path mapping
        self._build_path_mapping()

    def _build_path_mapping(self):
        """Build mapping between indices and virtual file paths."""
        if not self.streaming:
            dataset_len = len(self.dataset)
            for idx in range(dataset_len):
                # Create a virtual path like "0.jpg", "1.jpg", etc.
                virtual_path = f"{idx}.jpg"
                self._path_to_index[virtual_path] = idx
                self._index_to_path[idx] = virtual_path
        else:
            logger.warning(
                "Streaming mode enabled - path mapping will be built on demand"
            )

    def _get_index_from_path(self, filepath: str) -> Optional[int]:
        """Extract index from virtual file path."""
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Handle absolute paths by taking just the filename
        basename = os.path.basename(filepath)

        # Try direct lookup first
        if basename in self._path_to_index:
            return self._path_to_index[basename]

        # Try to extract index from filename (e.g., "123.jpg" -> 123)
        try:
            index = int(os.path.splitext(basename)[0])
            return index
        except ValueError:
            logger.error(f"Could not extract index from path: {filepath}")
            return None

    def read(self, location, as_byteIO: bool = False):
        """Read and return the content of the file (image from dataset or cache file)."""
        if isinstance(location, Path):
            location = str(location)

        # Handle cache files
        if location.endswith(".json"):
            cache_dir = Path("cache") / "huggingface_metadata" / self.id
            cache_path = cache_dir / Path(location).name

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    data = f.read()
                if as_byteIO:
                    return BytesIO(data)
                return data
            else:
                logger.error(f"Cache file not found: {cache_path}")
                return None

        # Handle virtual dataset files
        index = self._get_index_from_path(location)
        if index is None:
            logger.error(f"Invalid path: {location}")
            return None

        try:
            # Get the item from dataset
            item = self.dataset[index]
            image = item.get(self.image_column)

            if image is None:
                logger.error(
                    f"No image found in column '{self.image_column}' for index {index}"
                )
                return None

            # Handle different image types
            if isinstance(image, Image.Image):
                # PIL Image - convert to bytes
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(image, np.ndarray):
                # NumPy array - convert to PIL then bytes
                pil_image = Image.fromarray(image)
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(image, bytes):
                # Already bytes
                data = image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None

            if as_byteIO:
                return BytesIO(data)
            return data

        except Exception as e:
            logger.error(f"Error reading from dataset at index {index}: {e}")
            return None

    def write(self, filepath: Union[str, Path], data: Any) -> None:
        """
        Write operation - only supported for cache files (JSON).
        """
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Only allow writing cache files
        if filepath.endswith(".json"):
            # For cache files, we'll write to a local directory
            cache_dir = Path("cache") / "huggingface_metadata" / self.id
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_path = cache_dir / Path(filepath).name

            if isinstance(data, (dict, list)):
                import json

                data = json.dumps(data)
            elif isinstance(data, str):
                pass  # Already a string
            else:
                data = str(data)

            with open(cache_path, "w") as f:
                f.write(data)

            logger.debug(f"Wrote cache file to {cache_path}")
        else:
            logger.warning("Write operations are only supported for JSON cache files")
            raise NotImplementedError(
                "Hugging Face datasets are read-only except for cache files"
            )

    def delete(self, filepath):
        """Delete operation - not supported for HF datasets."""
        logger.warning("Delete operations are not supported for Hugging Face datasets")
        raise NotImplementedError("Hugging Face datasets are read-only")

    def exists(self, filepath):
        """Check if the virtual file exists (i.e., valid index)."""
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Check for cache files first
        if filepath.endswith(".json"):
            cache_dir = Path("cache") / "huggingface_metadata" / self.id
            cache_path = cache_dir / Path(filepath).name
            return cache_path.exists()

        # For virtual files, check index
        index = self._get_index_from_path(filepath)
        if index is None:
            return False

        if not self.streaming:
            return 0 <= index < len(self.dataset)
        else:
            # For streaming, we can't easily check bounds
            return True

    def open_file(self, filepath, mode):
        """Open the file in the specified mode."""
        if "w" in mode:
            raise NotImplementedError("Write operations are not supported")
        return BytesIO(self.read(filepath))

    def list_files(
        self, file_extensions: list = None, instance_data_dir: str = None
    ) -> list:
        """
        List all virtual files in the dataset.
        Returns format compatible with os.walk: [(root, dirs, files), ...]
        """
        if self.streaming:
            logger.warning("Cannot list files in streaming mode")
            return []

        # For HF datasets, we use a flat structure (no subdirectories)
        files = []

        for idx in range(len(self.dataset)):
            virtual_path = self._index_to_path.get(idx, f"{idx}.jpg")

            # Check file extension if filter provided
            if file_extensions:
                ext = os.path.splitext(virtual_path)[1].lower().strip(".")
                if ext not in file_extensions:
                    continue

            files.append(virtual_path)

        # Return in os.walk format: [(directory, subdirs, files)]
        return [("", [], files)]

    def get_abs_path(self, sample_path: str) -> str:
        """
        Given a relative path, return the absolute path.
        For HF datasets, we just return the virtual path.
        """
        if self.exists(sample_path):
            return sample_path
        return None

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        """Read an image from the dataset."""
        try:
            image_data = self.read(filepath, as_byteIO=True)
            if image_data is None:
                return None
            image = load_image(image_data)
            return image
        except Exception as e:
            logger.error(f"Error opening image {filepath}: {e}")
            if delete_problematic_images:
                logger.warning(
                    "Cannot delete from HF dataset - skipping problematic image"
                )
            return None

    def read_image_batch(
        self, filepaths: list, delete_problematic_images: bool = False
    ) -> list:
        """Read a batch of images from the dataset."""
        output_images = []
        available_keys = []

        for filepath in filepaths:
            image = self.read_image(filepath, delete_problematic_images)
            if image is not None:
                output_images.append(image)
                available_keys.append(filepath)
            else:
                logger.warning(f"Unable to load image '{filepath}', skipping.")

        return (available_keys, output_images)

    def create_directory(self, directory_path):
        """No-op for HF datasets."""
        pass

    def torch_load(self, filename):
        """Load a torch tensor - not typically used with HF datasets."""
        raise NotImplementedError("Torch load not supported for HF datasets")

    def torch_save(self, data, location: Union[str, Path, BytesIO]):
        """Save a torch tensor - not supported for HF datasets."""
        raise NotImplementedError("Torch save not supported for HF datasets")

    def write_batch(self, filepaths: list, data_list: list) -> None:
        """Write batch - not supported for HF datasets."""
        raise NotImplementedError("Write operations not supported for HF datasets")

    def save_state(self):
        """No state to save for HF datasets."""
        pass

    def get_dataset_item(self, index: int):
        """Get the full dataset item at the given index."""
        if not self.streaming and (index < 0 or index >= len(self.dataset)):
            return None
        return self.dataset[index]

    def __len__(self):
        """Return the number of items in the dataset."""
        if self.streaming:
            logger.warning("Cannot get length of streaming dataset")
            return 0
        return len(self.dataset)
