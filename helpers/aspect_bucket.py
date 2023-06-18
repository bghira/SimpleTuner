import torch, time

class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size=15):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())
        self.batch_size = batch_size
        self.current_bucket = 0
        self.current_count = 0

    def __iter__(self):
        while True:
            # Choose the bucket to yield from
            bucket = self.buckets[self.current_bucket]
            # Check if there are enough indices left in the bucket for a batch
            if len(self.aspect_ratio_bucket_indices[bucket]) >= self.batch_size:
                print(f'Yielding a sample for bucket {bucket}.')
                yield self.aspect_ratio_bucket_indices[bucket].pop()
                self.current_count += 1
                # If we've reached the batch size, move to the next bucket
                if self.current_count >= self.batch_size:
                    self.current_bucket = (self.current_bucket + 1) % len(self.buckets)
                    self.current_count = 0
            else:
                print(f'Bucket {bucket} is empty. Moving to the next bucket.')
                self.current_bucket = (self.current_bucket + 1) % len(self.buckets)

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())
