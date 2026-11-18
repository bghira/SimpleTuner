from simpletuner.helpers.data_backend.source_inventory import SourceFileInventory

# Mock data_backend for unit testing


class MockDataBackend:
    def __init__(self):
        self.source_files = SourceFileInventory()

    @staticmethod
    def read(image_path_str):
        # Dummy read method for testing
        return b"fake_image_data"
