import torch

class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())

    def __iter__(self):
        while True:
            for bucket in self.buckets:
                if self.aspect_ratio_bucket_indices[bucket]:
                    print(f'Yielding an index for bucket {bucket}.')
                    yield self.aspect_ratio_bucket_indices[bucket].pop()

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())
