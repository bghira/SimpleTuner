import torch, logging, random
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
            logging.debug('Querying bucket for item.')
            if len(self.aspect_ratio_bucket_indices[bucket]) >= self.batch_size:
                logging.info(f'Yielding a batch for bucket {bucket}.')
                for _ in range(self.batch_size):
                    yield random.choice(self.aspect_ratio_bucket_indices[bucket])
                # Move on to the next bucket after yielding a batch
                self.current_bucket = (self.current_bucket + 1) % len(self.buckets)
                # Log the state of buckets
                self.log_buckets()
            else:
                # If we're in training mode, move this bucket to the exhausted list and remove from active buckets
                if StateTracker.status_training():
                    logging.info(f'Bucket {bucket} is empty or doesn\'t have enough samples for a full batch. Moving to the next bucket.')
                    self.exhausted_buckets.append(self.buckets[self.current_bucket])
                    # Log the state of buckets
                    self.log_buckets()
                # If all buckets are empty or don't have enough samples for a full batch, break the loop
                if not self.buckets:
                    logging.info(f'All buckets are exhausted. Exiting...')
                    break
                else:
                    # Calculate next bucket index
                    self.current_bucket %= len(self.buckets)
    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())

    def log_buckets(self):
        def convert_to_human_readable(ratio_float):
            ratio_float = round(ratio_float, 2)
            if ratio_float == 1.0:
                return "1:1"
            if ratio_float == 0.5:
                return "1:2"
            if ratio_float == 0.67:
                return "2:3"
            if ratio_float == 0.75:
                return "3:4"
            if ratio_float == 0.8:
                return "4:5"
            if ratio_float == 1.33:
                return "4:3"
            if ratio_float == 1.5:
                return "3:2"
            if ratio_float == 1.78:
                return "16:9"
            if ratio_float == 2.0:
                return "2:1"
            return str(ratio_float)
        # Log active and exhausted buckets in human-readable format
        logging.debug(f'Active Buckets: {", ".join(convert_to_human_readable(float(b)) for b in self.buckets)}')
        logging.debug(f'Exhausted Buckets: {", ".join(convert_to_human_readable(float(b)) for b in self.exhausted_buckets)}')
