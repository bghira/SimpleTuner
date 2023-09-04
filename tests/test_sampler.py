import unittest
from unittest.mock import Mock, patch
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.state import BucketStateManager
from tests.helpers.data import MockDataBackend

class TestMultiAspectSampler(unittest.TestCase):

    def setUp(self):
        self.bucket_manager = Mock(spec=BucketManager)
        self.bucket_manager.aspect_ratio_bucket_indices = {'1.0': ['image1', 'image2']}
        self.data_backend = MockDataBackend()
        self.batch_size = 2
        self.seen_images_path = "/some/fake/seen_images.json"
        self.state_path = "/some/fake/state.json"

        self.sampler = MultiAspectSampler(
            bucket_manager=self.bucket_manager,
            data_backend=self.data_backend,
            batch_size=self.batch_size,
            seen_images_path=self.seen_images_path,
            state_path=self.state_path,
        )
        
        self.sampler.state_manager = Mock(spec=BucketStateManager)
        self.sampler.state_manager.load_state.return_value = {}

    def test_len(self):
        self.assertEqual(len(self.sampler), 2)
        
    def test_save_state(self):
        with patch.object(self.sampler.state_manager, 'save_state') as mock_save_state:
            self.sampler.save_state()
        mock_save_state.assert_called_once()

    def test_load_buckets(self):
        buckets = self.sampler.load_buckets()
        self.assertEqual(buckets, ['1.0'])

    def test_change_bucket(self):
        self.sampler.buckets = ['1.0', '1.5']
        self.sampler.exhausted_buckets = ['1.0']
        self.sampler.change_bucket()
        self.assertEqual(self.sampler.current_bucket, 1)  # Should now point to '1.5'

    def test_move_to_exhausted(self):
        self.sampler.current_bucket = 0  # Pointing to '1.0'
        self.sampler.buckets = ['1.0']
        self.sampler.move_to_exhausted()
        self.assertEqual(self.sampler.exhausted_buckets, ['1.0'])
        self.assertEqual(self.sampler.buckets, [])

    # Add more test cases here for edge cases and functionality

if __name__ == "__main__":
    unittest.main()
