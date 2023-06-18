import torch

class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size = 15):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())
        self.batch_size = batch_size

    def __iter__(self):
        batch_size_per_bucket = self.batch_size // len(self.buckets)
        while True:
            for bucket in self.buckets:
                if len(self.aspect_ratio_bucket_indices[bucket]) >= batch_size_per_bucket:
                    print(f'Yielding a batch for bucket {bucket}.')
                    for _ in range(batch_size_per_bucket):
                        yield self.aspect_ratio_bucket_indices[bucket].pop()

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())
