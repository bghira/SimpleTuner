import unittest
from unittest.mock import Mock

from simpletuner.helpers.data_backend.factory import DataBackendFactory


class LegacyAspectRatioBucketCoercionTests(unittest.TestCase):
    def setUp(self):
        self.args = Mock()
        self.args.skip_file_discovery = []
        self.args.train_batch_size = 1
        self.args.metadata_update_interval = 1
        self.args.delete_problematic_images = False
        self.args.delete_unwanted_images = False

        self.factory = DataBackendFactory(args=self.args, accelerator=Mock())

    def test_convert_legacy_bucket_keys(self):
        metadata_backend = Mock()
        metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["foo"],
            1.5: ["bar"],
        }
        init_backend = {
            "id": "test",
            "metadata_backend": metadata_backend,
            "config": {},
        }
        backend_config = {
            "id": "dataset",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": "/tmp",
        }
        with unittest.mock.patch(
            "simpletuner.helpers.data_backend.factory.get_metadata_backend",
            return_value=metadata_backend,
        ):
            self.factory._initialise_backend(backend_config, init_backend)
        coerced = metadata_backend.aspect_ratio_bucket_indices
        self.assertIn(1.0, coerced)
        self.assertIn(1.5, coerced)
        self.assertEqual(coerced[1.0], ["foo"])
