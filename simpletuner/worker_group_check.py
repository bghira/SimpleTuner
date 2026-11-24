"""GPU collective used by the Hardware tab's worker group check."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def verify_rank_sum(device):
    size = dist.get_world_size()
    value = torch.tensor(dist.get_rank() + 1, device=device, dtype=torch.int64)
    dist.all_reduce(value)
    if value.item() != size * (size + 1) // 2:
        raise RuntimeError("NCCL all-reduce returned an unexpected rank sum")
    print(f"Rank {dist.get_rank()} of {size} passed {dist.get_backend()} all-reduce", flush=True)


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=30), device_id=device)
    try:
        verify_rank_sum(device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
