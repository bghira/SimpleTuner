# Mock data_backend for unit testing


class MockDataBackend:
    @staticmethod
    def read(image_path_str):
        # Dummy read method for testing
        return b"fake_image_data"
