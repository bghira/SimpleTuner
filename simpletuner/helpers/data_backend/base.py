import gzip
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import closing
from io import BytesIO

import torch


def ordered_sample_payloads(dataset_id, refs, returned):
    """Associate bounded provider results with the requested occurrences."""
    payloads = defaultdict(deque)
    for ref, payload in returned:
        payloads[ref].append(payload)
    for ref in refs:
        payload = payloads[ref].popleft() if payloads[ref] else None
        if payload is None:
            raise OSError(f"Dataset {dataset_id} could not read the requested image {ref}")
        yield payload


class BaseDataBackend(ABC):
    # A sharded provider can choose a canonical number of source shards per prepared partition.
    prepared_partition_shards = None

    def count_source_rows(self, *, exclude_null_images=False):
        """Return the exact source row count when the provider supports it."""
        raise NotImplementedError("This source does not expose exact row counts.")

    def iter_cache_groups(self, bucket_files: dict):
        """Yield (bucket, files) groups in input order; callers drain queues between groups."""
        yield from bucket_files.items()

    @abstractmethod
    def read(self, identifier):
        """
        Read data based on the identifier.
        """
        pass

    @abstractmethod
    def write(self, identifier, data):
        """
        Write data to the specified identifier.
        """
        pass

    @abstractmethod
    def delete(self, identifier):
        """
        Delete data associated with the identifier.
        """
        pass

    @abstractmethod
    def exists(self, identifier):
        """
        Check if the identifier exists.
        """
        pass

    def bulk_exists(self, identifiers):
        """Return presence for each requested key in the same order."""
        return [self.exists(identifier) for identifier in identifiers]

    @abstractmethod
    def open_file(self, identifier, mode):
        """
        Open the identifier (file or object) in the specified mode.
        """
        pass

    def open_read_stream(self, identifier):
        """Open a binary reader for persisted metadata or artifact manifests."""
        return closing(self.open_file(identifier, "rb"))

    @abstractmethod
    def list_files(self, file_extensions: list, instance_data_dir: str = None) -> tuple:
        """
        List all files matching the pattern.
        """
        pass

    def iter_file_keys(self, file_extensions, instance_data_dir=None):
        """Yield matching keys. Backends can replace the legacy grouped listing."""
        for _, _, keys in self.list_files(file_extensions, instance_data_dir):
            yield from keys

    @abstractmethod
    def get_abs_path(self, sample_path: str = None) -> tuple:
        """
        Given a relative path of a sample, return the absolute path.
        """
        pass

    @abstractmethod
    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        """
        Read an image from the backend and return a PIL Image.
        """
        pass

    @abstractmethod
    def read_image_batch(self, filepaths: str, delete_problematic_images: bool = False):
        """
        Read a batch of images from the backend and return a list of PIL Images.
        """
        pass

    @abstractmethod
    def create_directory(self, directory_path):
        """
        Creates a directory in the backend.
        """
        pass

    @abstractmethod
    def torch_load(self, filename):
        """
        Reads content from the backend and loads it with torch.
        """
        pass

    def read_cached_tensors(self, keys):
        """Return cached tensors in request order, with None for missing keys."""
        for key in keys:
            try:
                yield self.torch_load(key) if self.exists(key) else None
            except FileNotFoundError:
                yield None

    @abstractmethod
    def torch_save(self, data, filename):
        """
        Saves the data using torch to the backend.
        """
        pass

    @abstractmethod
    def write_batch(self, identifiers, files):
        """
        Write a batch of files to the specified identifiers.
        """
        pass

    @abstractmethod
    def get_instance_representation(self) -> dict:
        """
        Get a serializable representation of this backend instance.
        This should include all necessary information to recreate the backend
        in another process.

        Returns:
            dict: Serializable dictionary containing backend configuration
        """
        pass

    @staticmethod
    @abstractmethod
    def from_instance_representation(representation: dict) -> "BaseDataBackend":
        """
        Create a new backend instance from a serialized representation.

        Args:
            representation: Dictionary created by get_instance_representation()

        Returns:
            BaseDataBackend: New instance of the appropriate backend
        """
        pass

    def _decompress_torch(self, gzip_data: BytesIO):
        """
        We've read the gzip from disk. Just decompress it.
        """
        # bytes object might not have seek. workaround:
        if not hasattr(gzip_data, "seek"):
            gzip_data = BytesIO(gzip_data)
        gzip_data.seek(0)
        with gzip.GzipFile(fileobj=gzip_data, mode="rb") as file:
            try:
                decompressed_data = file.read()
            except Exception as e:
                # Handle decompression errors
                gzip_data.seek(0)
                return gzip_data
        return BytesIO(decompressed_data)

    def _compress_torch(self, data):
        """
        Compress the torch data before writing it to disk.
        """
        output_data_container = BytesIO()
        torch.save(data, output_data_container)
        output_data_container.seek(0)
        with BytesIO() as compressed_output:
            with gzip.GzipFile(fileobj=compressed_output, mode="wb") as file:
                file.write(output_data_container.getvalue())
            return compressed_output.getvalue()
