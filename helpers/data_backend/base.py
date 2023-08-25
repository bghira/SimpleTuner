from abc import ABC, abstractmethod


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
    def list_files(self, pattern: str, instance_data_root: str = None):
        """
        List all files matching the pattern.
        """
        pass