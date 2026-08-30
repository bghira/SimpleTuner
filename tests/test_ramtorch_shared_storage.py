import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from simpletuner.helpers.ramtorch.utils import attach_shared_ramtorch_parameters


class _QuantizedBuffer(torch.Tensor):
    @staticmethod
    def __new__(cls, weight: torch.Tensor, scale: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            weight.shape,
            strides=weight.stride(),
            storage_offset=weight.storage_offset(),
            dtype=torch.float32,
            device=weight.device,
        )

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        self.weight = weight
        self.scale = scale

    def __tensor_flatten__(self):
        return ["weight", "scale"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, _metadata, outer_size=None, outer_stride=None):
        del outer_size, outer_stride
        return cls(inner_tensors["weight"], inner_tensors["scale"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(f"{cls.__name__} does not implement {func}")


class _QuantizedBufferModel(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        payload = _QuantizedBuffer(
            torch.full((4, 4), float(rank), dtype=torch.float32),
            torch.full((4, 1), float(rank), dtype=torch.float32),
        )
        payload.is_ramtorch = True
        self.register_buffer("quantized_weight", payload)


def _distributed_shared_buffer_worker(rank: int, world_size: int, rendezvous_path: str, use_cuda: bool) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _QuantizedBufferModel(rank)
        attached = attach_shared_ramtorch_parameters(model)
        if attached != 2:
            raise AssertionError(f"expected two shared payload storages, got {attached}")

        if rank == 0:
            model.quantized_weight.weight.fill_(7)
            model.quantized_weight.scale.fill_(3)
        dist.barrier()

        torch.testing.assert_close(model.quantized_weight.weight, torch.full((4, 4), 7.0))
        torch.testing.assert_close(model.quantized_weight.scale, torch.full((4, 1), 3.0))
        if not model.quantized_weight.weight.untyped_storage().is_shared():
            raise AssertionError("quantized weight storage is not shared")
        if not model.quantized_weight.scale.untyped_storage().is_shared():
            raise AssertionError("quantized scale storage is not shared")

        if use_cuda:
            device = torch.device("cuda", 0)
            torch.testing.assert_close(
                model.quantized_weight.weight.to(device, non_blocking=True).cpu(),
                torch.full((4, 4), 7.0),
            )
            torch.testing.assert_close(
                model.quantized_weight.scale.to(device, non_blocking=True).cpu(),
                torch.full((4, 1), 3.0),
            )
    finally:
        dist.destroy_process_group()


class RamTorchSharedStorageTests(unittest.TestCase):
    def test_tensor_subclass_payloads_share_storage_across_independent_ranks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_path = os.path.join(temp_dir, "gloo-store")
            mp.spawn(
                _distributed_shared_buffer_worker,
                args=(2, rendezvous_path, torch.cuda.is_available()),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
