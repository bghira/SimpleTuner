import hashlib
import logging
import torch
from pathlib import Path
from io import BytesIO
import os
from typing import Any, Union, Optional, BinaryIO
from PIL import Image
import numpy as np

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

from torchvision.io.video_reader import VideoReader

logger = logging.getLogger("HuggingfaceDatasetsBackend")


class HuggingfaceDatasetsBackend(BaseDataBackend):
    def __init__(
        self,
        accelerator,
        id: str,
        dataset_name: str,
        split: str = "train",
        revision: str = None,
        image_column: str = "image",
        video_column: str = "video",
        cache_dir: Optional[str] = None,
        compress_cache: bool = False,
        streaming: bool = False,
        filter_func: Optional[callable] = None,
        num_proc: int = 16,
        composite_config: dict = {},
        dataset_type: str = "image",
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
        self.dataset_type = dataset_type
        self.file_extension = "jpg" if dataset_type == "image" else "mp4"
        self.split = split
        self.revision = revision
        self.image_column = image_column
        self.video_column = video_column
        self.cache_dir = cache_dir
        self.compress_cache = compress_cache
        self.streaming = streaming
        self.filter_func = filter_func
        self.num_proc = num_proc
        self.composite_config = composite_config

        # Virtual file system mapping: index -> virtual path
        self._path_to_index = {}
        self._index_to_path = {}
        if should_log():
            logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
        else:
            logger.setLevel("ERROR")

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the Hugging Face dataset."""
        try:
            from datasets import load_dataset, load_dataset_builder
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # First, inspect the dataset structure
        logger.info(f"Inspecting dataset {self.dataset_name}")
        builder = load_dataset_builder(self.dataset_name, cache_dir=self.cache_dir)
        logger.info(f"Dataset info: {builder.info}")
        logger.info(f"Available splits: {list(builder.info.splits.keys())}")

        logger.info(f"Loading dataset {self.dataset_name} (split: {self.split})")

        # Load with explicit parameters to ensure all data is loaded
        self.dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            revision=self.revision,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
            download_mode="reuse_dataset_if_exists",  # or "force_redownload" if needed
        )

        # Log the initial size
        if not self.streaming:
            logger.info(f"Loaded dataset with {len(self.dataset)} items")

            # Check if this looks like a subset
            if len(self.dataset) == 8200:
                logger.warning(
                    "Dataset has exactly 8200 items - this might be a single shard!"
                )

        # Apply filter if provided
        if self.filter_func and not self.streaming:
            logger.info("Applying filter to dataset...")
            original_size = len(self.dataset)
            self.dataset = self.dataset.filter(
                self.filter_func,
                num_proc=self.num_proc,
            )
            logger.info(
                f"Dataset filtered from {original_size} to {len(self.dataset)} items"
            )

        # Build virtual path mapping
        self._build_path_mapping()

    def _build_path_mapping(self):
        """Build mapping between indices and virtual file paths."""
        if not self.streaming:
            dataset_len = len(self.dataset)
            for idx in range(dataset_len):
                # Create virtual paths based on composite configuration
                virtual_path = f"{idx}.{self.file_extension}"

                self._path_to_index[virtual_path] = idx
                self._index_to_path[idx] = virtual_path

                # Also map the simple format for backwards compatibility
                simple_path = f"{idx}.{self.file_extension}"
                if simple_path != virtual_path:
                    self._path_to_index[simple_path] = idx
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

        # Handle simple format (e.g., "123.{self.file_extension}" -> 123)
        try:
            index = int(os.path.splitext(basename)[0])
            return index
        except ValueError:
            logger.error(f"Could not extract index from path: {filepath}")
            raise

    def get_instance_representation(self) -> dict:
        """Get a serializable representation of this backend instance."""
        return {
            "backend_type": "huggingface",
            "id": self.id,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "revision": self.revision,
            "image_column": self.image_column,
            "cache_dir": self.cache_dir,
            "compress_cache": self.compress_cache,
            "streaming": self.streaming,
            # Note: filter_func is not serializable
            "filter_func_name": self.filter_func.__name__ if self.filter_func else None,
            "num_proc": self.num_proc,
            "composite_config": self.composite_config,
        }

    @staticmethod
    def from_instance_representation(
        representation: dict,
    ) -> "HuggingfaceDatasetsBackend":
        """Create a new HuggingfaceDatasetsBackend instance from a serialized representation."""
        if representation.get("backend_type") != "huggingface":
            raise ValueError(
                f"Expected backend_type 'huggingface', got {representation.get('backend_type')}"
            )

        # Note: filter_func cannot be serialized/deserialized automatically
        # If needed, you'd have to implement a registry of filter functions
        filter_func = None
        if representation.get("filter_func_name"):
            import logging

            logger = logging.getLogger("HuggingfaceDatasetsBackend")
            logger.warning(
                f"Filter function '{representation['filter_func_name']}' cannot be automatically restored in subprocess"
            )

        return HuggingfaceDatasetsBackend(
            accelerator=None,  # Will be set by subprocess if needed
            id=representation["id"],
            dataset_name=representation["dataset_name"],
            split=representation.get("split", "train"),
            revision=representation.get("revision"),
            image_column=representation.get("image_column", "image"),
            cache_dir=representation.get("cache_dir"),
            compress_cache=representation.get("compress_cache", False),
            streaming=representation.get("streaming", False),
            filter_func=filter_func,  # Would need special handling
            num_proc=representation.get("num_proc", 16),
            composite_config=representation.get("composite_config", {}),
        )

    def read(self, location, as_byteIO: bool = False):
        """Read and return the content of the file (image from dataset or cache file)."""
        if isinstance(location, Path):
            location = str(location)

        # Handle cache files (.json, .pt, etc.) - these should be read from local filesystem
        cache_extensions = [".json", ".pt", ".msgpack", ".safetensors"]
        if any(location.endswith(ext) for ext in cache_extensions):
            # Try location as-is first (in case it's already a full path)
            location_path = Path(location)
            if location_path.exists():
                with open(location_path, "rb") as f:
                    data = f.read()
                if as_byteIO:
                    return BytesIO(data)
                return data

            # Otherwise, check standard cache directories based on file type
            if hasattr(self, "cache_dir") and self.cache_dir:
                filename = Path(location).name

                # Determine cache directory based on file type (matching write logic)
                if location.endswith(".json"):
                    cache_path = (
                        Path(self.cache_dir)
                        / "huggingface_metadata"
                        / self.id
                        / filename
                    )
                elif any(location.endswith(ext) for ext in [".pt", ".safetensors"]):
                    cache_path = Path(self.cache_dir) / "vae" / self.id / filename
                else:
                    cache_path = Path(self.cache_dir) / "cache" / self.id / filename

                if cache_path.exists():
                    logger.debug(f"Reading from location: {cache_path}")
                    with open(cache_path, "rb") as f:
                        data = f.read()
                    if as_byteIO:
                        return BytesIO(data)
                    return data
                else:
                    logger.warning(f"Could not read from location: {cache_path}")

            logger.error(f"Cache file not found: {location}")
            return None

        # Handle virtual dataset files (the existing logic)
        index = self._get_index_from_path(location)
        if index is None:
            logger.error(f"Invalid path: {location}")
            return None

        try:
            # Get the item from dataset
            item = self.dataset[index]
            sample = item.get(
                self.video_column if self.dataset_type == "video" else self.image_column
            )

            if sample is None:
                logger.error(
                    f"No {self.dataset_type} found in column '{self.video_column if self.dataset_type == 'video' else self.image_column}' for index {index}"
                )
                return None

            # Handle composite images if configured
            if (
                hasattr(self, "composite_config")
                and isinstance(self.composite_config, dict)
                and self.composite_config.get("enabled")
            ):
                image_count = self.composite_config.get("image_count", 1)
                select_index = self.composite_config.get("select_index", 0)

                if isinstance(sample, Image.Image):
                    width, height = sample.size
                    slice_width = width // image_count

                    # Calculate crop box for the selected index
                    left = select_index * slice_width
                    right = (select_index + 1) * slice_width

                    # Crop the image
                    sample = sample.crop((left, 0, right, height))

            # Handle different image types
            if isinstance(sample, Image.Image):
                # PIL Image - convert to bytes
                buffer = BytesIO()
                sample.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(sample, np.ndarray):
                # NumPy array - convert to PIL then bytes
                pil_image = Image.fromarray(sample)
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(sample, bytes):
                # Already bytes
                data = sample
            elif isinstance(sample, VideoReader):
                # VideoReader - encode all frames into a video file in memory
                import cv2

                frames = []
                for frame in sample:
                    # frame['data'] is a torch tensor (T, H, W, C)
                    frame_np = frame["data"].numpy()
                    # Convert from RGB to BGR for OpenCV
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    frames.append(frame_np)

                if not frames:
                    logger.error("VideoReader contains no frames")
                    return None

                height, width, channels = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                temp_video = BytesIO()
                # OpenCV cannot write directly to BytesIO, so use a temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpfile:
                    out = cv2.VideoWriter(
                        tmpfile.name,
                        fourcc,
                        sample.get_metadata()["video"]["fps"][0],
                        (width, height),
                    )
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    tmpfile.seek(0)
                    data = tmpfile.read()
                # Now data contains the video bytes
            else:
                logger.error(f"Unsupported image type: {type(sample)}")
                return None

            if as_byteIO:
                return BytesIO(data)
            return data

        except Exception as e:
            logger.error(f"Error reading from dataset at index {index}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def write(self, filepath: Union[str, Path], data: Any) -> None:
        """
        Write operation - supported for cache files.
        """
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Allow writing cache files
        cache_extensions = [".json", ".pt", ".msgpack", ".safetensors"]
        if any(filepath.endswith(ext) for ext in cache_extensions):
            # Determine cache directory based on file path
            filepath_path = Path(filepath)

            # If it's already an absolute path with a directory, use it
            if filepath_path.is_absolute() and filepath_path.parent.exists():
                cache_path = filepath_path
            else:
                # Otherwise, create a cache directory structure
                if not hasattr(self, "cache_dir") or not self.cache_dir:
                    raise ValueError(
                        f"Cannot write cache file {filepath} - no cache_dir configured for HuggingFace backend"
                    )

                # Determine subdirectory based on file type
                if filepath.endswith(".json"):
                    cache_subdir = (
                        Path(self.cache_dir) / "huggingface_metadata" / self.id
                    )
                elif any(filepath.endswith(ext) for ext in [".pt", ".safetensors"]):
                    # VAE cache files
                    cache_subdir = Path(self.cache_dir) / "vae" / self.id
                else:
                    # Other cache files
                    cache_subdir = Path(self.cache_dir) / "cache" / self.id

                cache_subdir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_subdir / Path(filepath).name

            # Write the file
            if filepath.endswith(".json"):
                if isinstance(data, (dict, list)):
                    import json

                    data = json.dumps(data)
                elif not isinstance(data, str):
                    data = str(data)

                with open(cache_path, "w") as f:
                    f.write(data)
            else:
                # Binary data
                with open(cache_path, "wb") as f:
                    if isinstance(data, bytes):
                        f.write(data)
                    elif hasattr(data, "save"):
                        # For torch tensors or similar objects with save method
                        data.save(f)
                    else:
                        # Fallback to torch.save
                        import torch

                        torch.save(data, f)

            logger.debug(f"Wrote cache file to {cache_path}")
        else:
            logger.warning(
                f"Write operations are only supported for cache files, not {filepath}"
            )
            raise NotImplementedError(
                "Hugging Face datasets are read-only except for cache files"
            )

    def exists(self, filepath):
        """Check if the file exists (cache file or valid dataset index)."""
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Check for cache files first
        cache_extensions = [".json", ".pt", ".msgpack", ".safetensors"]
        if any(filepath.endswith(ext) for ext in cache_extensions):
            # Check if it's already a full path
            filepath_path = Path(filepath)
            if filepath_path.exists():
                return True

            # Check standard cache directories based on file type
            if hasattr(self, "cache_dir") and self.cache_dir:
                filename = Path(filepath).name

                # Determine cache directory based on file type (matching write logic)
                if filepath.endswith(".json"):
                    cache_path = (
                        Path(self.cache_dir)
                        / "huggingface_metadata"
                        / self.id
                        / filename
                    )
                elif any(filepath.endswith(ext) for ext in [".pt", ".safetensors"]):
                    cache_path = Path(self.cache_dir) / "vae" / self.id / filename
                else:
                    cache_path = Path(self.cache_dir) / "cache" / self.id / filename

                if cache_path.exists():
                    logger.debug(f"Cache file exists: {cache_path}")
                    return True

            return False

        # For virtual files, check index
        index = self._get_index_from_path(filepath)
        if index is None:
            return False

        if not self.streaming:
            return 0 <= index < len(self.dataset)
        else:
            # For streaming, we can't easily check bounds
            return True

    def delete(self, filepath):
        """Delete operation - not supported for HF datasets."""
        logger.warning("Delete operations are not supported for Hugging Face datasets")
        raise NotImplementedError("Hugging Face datasets are read-only")

    def open_file(self, filepath, mode):
        """Open the file in the specified mode."""
        if "w" in mode:
            raise NotImplementedError("Write operations are not supported")
        return BytesIO(self.read(filepath))

    def list_files(
        self, file_extensions: list = None, instance_data_dir: str = None
    ) -> list:
        """
        List all virtual files in the dataset or files in a real directory if instance_data_dir is provided.
        Returns format compatible with os.walk: [(root, dirs, files), ...]
        """
        if instance_data_dir:
            # List files from the real directory
            instance_data_dir = str(instance_data_dir)
            logger.debug(f"Listing files via {instance_data_dir=}")
            if not os.path.exists(instance_data_dir):
                logger.warning(f"Directory does not exist: {instance_data_dir}")
                return []
            result = []
            logger.debug(f"Running os.walk")
            for root, dirs, files in os.walk(instance_data_dir):
                filtered_files = []
                for f in files:
                    if file_extensions:
                        ext = os.path.splitext(f)[1].lower().strip(".")
                        if ext not in file_extensions:
                            logger.debug(
                                f"Skipping {ext=} cuz not in {file_extensions}"
                            )
                            continue
                    filtered_files.append(os.path.join(root, f))
                result.append((root, dirs, filtered_files))
            logger.debug("Returning results")
            return result

        if self.streaming:
            logger.warning("Cannot list files in streaming mode")
            return []

        # For HF datasets, we use a flat structure (no subdirectories)
        files = []
        for idx in range(len(self.dataset)):
            virtual_path = self._index_to_path.get(idx, f"{idx}.{self.file_extension}")
            if file_extensions:
                ext = os.path.splitext(virtual_path)[1].lower().strip(".")
                if ext not in file_extensions:
                    continue
            files.append(virtual_path)
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
            loader = load_image
            if self.dataset_type == "video":
                loader = load_video
            image = loader(image_data)
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

    def save_state(self):
        """No state to save for HF datasets."""
        pass

    def get_dataset_item(self, index: int):
        """Get the full dataset item at the given index."""
        if not self.streaming and (index < 0 or index >= len(self.dataset)):
            logger.warning(
                f"Retrieving {index=} for {len(self.dataset)=}, returning None"
            )
            return None
        return self.dataset[index]

    def __len__(self):
        """Return the number of items in the dataset."""
        if self.streaming:
            logger.warning("Cannot get length of streaming dataset")
            return 0
        return len(self.dataset)

    def write_batch(self, filepaths: list, data_list: list) -> None:
        """Write batch - supported for cache files."""
        if len(filepaths) != len(data_list):
            raise ValueError("Number of filepaths must match number of data items")

        for filepath, data in zip(filepaths, data_list):
            self.write(filepath, data)

    def torch_load(self, filename):
        """Load a torch tensor from cache."""
        if isinstance(filename, Path):
            filename = str(filename)

        # Try to read the file
        data = self.read(filename, as_byteIO=True)
        if data is None:
            raise FileNotFoundError(f"Could not find file: {filename}")

        import torch

        return torch.load(data)

    def torch_save(self, data, location: Union[str, Path, BytesIO]):
        """Save a torch tensor to cache."""
        if isinstance(location, BytesIO):
            import torch

            torch.save(data, location)
        else:
            self.write(location, data)

    def create_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)
