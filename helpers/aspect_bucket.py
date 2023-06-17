import torch

class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(self, aspect_ratio_bucket_indices, batch_size):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(aspect_ratio_bucket_indices.keys())
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            for bucket in self.buckets:
                batch_indices = []
                for _ in range(self.batch_size):
                    if self.aspect_ratio_bucket_indices[bucket]:
                        batch_indices.append(self.aspect_ratio_bucket_indices[bucket].pop())
                    else:
                        break
                if len(batch_indices) == self.batch_size:  # Only yield complete batches
                    print(f'Yielding batch for bucket {bucket}')
                    yield batch_indices
                else:
                    print(f'Incomplete bucket for aspect discarded: {bucket}')

    def __len__(self):
        return sum(len(indices) for indices in self.aspect_ratio_bucket_indices.values())
