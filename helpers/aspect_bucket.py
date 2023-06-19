import torch, logging
from .state_tracker import StateTracker

class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size=15):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())
        self.exhausted_buckets = []
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
                # Move this bucket to the exhausted list and remove from active buckets
                logging.info(f'Bucket {bucket} is empty or doesn\'t have enough samples for a full batch. Moving to the next bucket.')
                self.exhausted_buckets.append(self.buckets.pop(self.current_bucket))
                # If all buckets are empty or don't have enough samples for a full batch, break the loop
                if not self.buckets:
                    logging.info(f'All buckets are exhausted. Exiting...')
                    break
                else:
                    # Calculate next bucket index
                    self.current_bucket %= len(self.buckets)

            # Log the state of buckets
            self.log_buckets()

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())

    def log_buckets(self):
        # Log active and exhausted buckets in human-readable format
        active_buckets_str = ', '.join(f'{1 / b:.1f}:1' for b in self.buckets)
        exhausted_buckets_str = ', '.join(f'{1 / b:.1f}:1' for b in self.exhausted_buckets)
        logging.debug(f'Active Buckets: {active_buckets_str if active_buckets_str else "None"}')
        logging.debug(f'Exhausted Buckets: {exhausted_buckets_str if exhausted_buckets_str else "None"}')
