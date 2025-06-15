from abc import ABC, abstractmethod
from io import BytesIO
import gzip
import torch


class BaseDataBackend(ABC):
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

    @abstractmethod
    def open_file(self, identifier, mode):
        """
        Open the identifier (file or object) in the specified mode.
        """
        pass

    @abstractmethod
    def list_files(self, file_extensions: list, instance_data_dir: str = None) -> tuple:
        """
        List all files matching the pattern.
        """
        pass

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

    def _decompress_torch(self, gzip_data: BytesIO):
        """
        We've read the gzip from disk. Just decompress it.
        """
        # bytes object might not have seek. workaround:
        if not hasattr(gzip_data, "seek"):
            gzip_data = BytesIO(gzip_data)
        gzip_data.seek(0)
        with gzip.GzipFile(fileobj=gzip_data, mode="rb") as file:
            decompressed_data = file.read()
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
