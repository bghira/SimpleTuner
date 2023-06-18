import torch, logging
from .state_tracker import StateTracker
class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size=15):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())
        self.batch_size = batch_size
        self.current_bucket = 0

    def __iter__(self):
        while True:
            # Choose the bucket to yield from
            bucket = self.buckets[self.current_bucket]
            # If the bucket has enough samples for a full batch, yield from it
            if len(self.aspect_ratio_bucket_indices[bucket]) >= self.batch_size:
                logging.info(f'Yielding a batch for bucket {bucket}.')
                for _ in range(self.batch_size):
                    if StateTracker.status_training():
                        yield self.aspect_ratio_bucket_indices[bucket].pop()
                    else:
                        # Yield a dummy image if we're not started yet.
                        yield self.aspect_ratio_bucket_indices[bucket][0]
                # Move on to the next bucket after yielding a batch
                self.current_bucket = (self.current_bucket + 1) % len(self.buckets)
            else:
                logging.info(f'Bucket {bucket} is empty or doesn\'t have enough samples for a full batch. Moving to the next bucket.')
                self.current_bucket = (self.current_bucket + 1) % len(self.buckets)
                # If all buckets are empty or don't have enough samples for a full batch, break the loop
                if all(len(self.aspect_ratio_bucket_indices[bucket]) < self.batch_size for bucket in self.buckets):
                    break

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())
