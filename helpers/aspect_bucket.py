import torch, logging, random, time
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
            bucket = self.buckets[self.current_bucket]
            if len(self.aspect_ratio_bucket_indices[bucket]) >= self.batch_size:
                yield from random.choices(
                    self.aspect_ratio_bucket_indices[bucket], k=self.batch_size
                )
                self.log_state()
            else:
                if StateTracker.status_training():
                    self.move_to_exhausted()
                if not self.buckets:
                    logging.info(f"All buckets are exhausted. Exiting...")
                    break
                else:
                    self.change_bucket()
            self.current_bucket = (self.current_bucket + 1) % len(self.buckets)

    def __len__(self):
        return sum(
            len(indices) for indices in self.aspect_ratio_bucket_indices.values()
        )

    def change_bucket(self):
        self.current_bucket %= len(self.buckets)
        logging.info(f"Changing bucket to {self.buckets[self.current_bucket]}.")

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        logging.info(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Moving to the next bucket."
        )
        self.log_state()

    def log_state(self):
        logging.debug(
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b)) for b in self.buckets)}'
        )
        logging.debug(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b)) for b in self.exhausted_buckets)}'
        )

    @staticmethod
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
