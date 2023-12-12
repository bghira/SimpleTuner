# A class to act as a wrapper for multiple MultiAspectSampler objects, feeding samples from them in proportion.
from helpers.multiaspect.bucket import BucketManager
from helpers.data_backend.base import BaseDataBackend
from helpers.multiaspect.sampler import MultiAspectSampler


class MultiSampler:
    def __init__(
        self,
        bucket_manager: BucketManager,
        data_backend: BaseDataBackend,
        accelerator,
        args: dict,
    ):
        self.batch_size = args.train_batch_size
        self.seen_images_path = args.seen_state_path
        self.state_path = args.state_path
        self.debug_aspect_buckets = args.debug_aspect_buckets
        self.delete_unwanted_images = args.delete_unwanted_images
        self.resolution = args.resolution
        self.resolution_type = args.resolution_type
        self.args = args

    def configure(self):
        if self.args.data_backend is None:
            raise ValueError("Must provide a data backend via --data_backend")
        if self.args.data_backend != "multi":
            # Return a basic MultiAspectSampler for the single data backend:
            self.sampler = self.get_single_sampler()
            return
        # Configure a multi-aspect sampler:

    def get_single_sampler(self) -> list:
        """
        Get a single MultiAspectSampler object.
        """
        return [
            MultiAspectSampler(
                batch_size=self.batch_size,
                seen_images_path=self.seen_images_path,
                state_path=self.state_path,
                debug_aspect_buckets=self.debug_aspect_buckets,
                delete_unwanted_images=self.delete_unwanted_images,
                resolution=self.resolution,
                resolution_type=self.resolution_type,
            )
        ]
