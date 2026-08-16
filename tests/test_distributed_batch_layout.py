import os
import time
import traceback
import unittest
from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from simpletuner.helpers.data_backend.runtime.context_parallel_sync import (
    gather_sample_weighted_scalar,
    gather_variable_batch_tensor,
    resolve_distributed_batch_layout,
)


class _GatherAccelerator:
    def __init__(self, *, world_size, process_index, gathered_values):
        self.num_processes = world_size
        self.process_index = process_index
        self.device = torch.device("cpu")
        self._gathered_values = list(gathered_values)
        self.gather_inputs = []

    def gather(self, tensor):
        self.gather_inputs.append(tensor.detach().clone())
        return self._gathered_values.pop(0).to(dtype=tensor.dtype, device=tensor.device)


class _GlooAccelerator:
    def __init__(self, rank, world_size):
        self.num_processes = world_size
        self.process_index = rank
        self.device = torch.device("cpu")
        self.gather_shapes = []

    def gather(self, tensor):
        self.gather_shapes.append(tuple(tensor.shape))
        gathered = [torch.empty_like(tensor) for _ in range(self.num_processes)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)


def _run_unequal_batch_gloo_worker(rank, world_size, init_method, result_queue):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10),
        )
        accelerator = _GlooAccelerator(rank, world_size)
        local_batch_size = 1 if rank == 0 else 3
        local_loss = torch.tensor(2.0 if rank == 0 else 4.0)
        local_values = torch.tensor([10.0]) if rank == 0 else torch.tensor([20.0, 30.0, 40.0])

        weighted_loss = gather_sample_weighted_scalar(local_loss, local_batch_size, accelerator)
        gathered_values = gather_variable_batch_tensor(local_values, accelerator)
        result_queue.put(
            (
                "ok",
                rank,
                float(weighted_loss),
                gathered_values.tolist(),
                accelerator.gather_shapes,
            )
        )
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedBatchLayoutTests(unittest.TestCase):
    def test_real_gloo_collectives_support_rank_varying_batch_sizes(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("torch.distributed with Gloo is unavailable")

        context = mp.get_context("spawn")
        result_queue = context.Queue()
        with TemporaryDirectory() as temp_dir:
            init_method = f"file://{os.path.join(temp_dir, 'gloo-rendezvous')}"
            processes = [
                context.Process(
                    target=_run_unequal_batch_gloo_worker,
                    args=(rank, 2, init_method, result_queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()

            deadline = time.monotonic() + 30
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            timed_out = [process for process in processes if process.is_alive()]
            for process in timed_out:
                process.terminate()
            for process in timed_out:
                process.join(5)

        self.assertFalse(timed_out, "Gloo regression workers exceeded the 30-second timeout")
        self.assertEqual([process.exitcode for process in processes], [0, 0])
        results = sorted((result_queue.get(timeout=3) for _ in processes), key=lambda item: item[1])
        result_queue.close()
        result_queue.join_thread()

        self.assertEqual([result[0] for result in results], ["ok", "ok"])
        for _status, _rank, weighted_loss, gathered_values, gather_shapes in results:
            self.assertEqual(weighted_loss, 3.5)
            self.assertEqual(gathered_values, [10.0, 20.0, 30.0, 40.0])
            self.assertEqual(gather_shapes, [(2,), (1,), (3,)])

    def test_rank_varying_batch_sizes_have_global_count_and_prefix_offset(self):
        accelerator = _GatherAccelerator(
            world_size=3,
            process_index=1,
            gathered_values=[torch.tensor([1, 3, 2])],
        )

        layout = resolve_distributed_batch_layout(accelerator, local_batch_size=3)

        self.assertEqual(layout.global_batch_size, 6)
        self.assertEqual(layout.local_batch_offset, 1)
        self.assertEqual(layout.data_replica_batch_sizes, (1, 3, 2))
        self.assertEqual(tuple(accelerator.gather_inputs[0].shape), (1,))

    def test_context_parallel_layout_counts_each_model_replica_once(self):
        accelerator = _GatherAccelerator(
            world_size=4,
            process_index=2,
            gathered_values=[torch.tensor([1, 1, 3, 3])],
        )
        accelerator.parallelism_config = SimpleNamespace(
            cp_size=2,
            cp_enabled=True,
            dp_replicate_size=2,
            dp_shard_size=1,
        )
        accelerator.torch_device_mesh = MagicMock()
        accelerator.torch_device_mesh.get_group.return_value = MagicMock()
        accelerator.torch_device_mesh.get_local_rank.return_value = 0

        layout = resolve_distributed_batch_layout(accelerator, local_batch_size=3)

        self.assertEqual(layout.global_batch_size, 4)
        self.assertEqual(layout.local_batch_offset, 1)
        self.assertEqual(layout.data_rank, 1)
        self.assertEqual(layout.model_replica_size, 2)
        self.assertEqual(layout.data_replica_batch_sizes, (1, 3))

    def test_variable_batch_tensor_gather_pads_and_trims_rank_values(self):
        accelerator = _GatherAccelerator(
            world_size=2,
            process_index=1,
            gathered_values=[
                torch.tensor([1, 3]),
                torch.tensor([10.0, 0.0, 0.0, 20.0, 30.0, 40.0]),
            ],
        )
        local_values = torch.tensor([20.0, 30.0, 40.0])

        gathered = gather_variable_batch_tensor(local_values, accelerator)

        self.assertTrue(torch.equal(gathered, torch.tensor([10.0, 20.0, 30.0, 40.0])))
        self.assertEqual(tuple(accelerator.gather_inputs[0].shape), (1,))
        self.assertEqual(tuple(accelerator.gather_inputs[1].shape), (3,))

    def test_variable_batch_tensor_gather_preserves_context_parallel_rank_values(self):
        accelerator = _GatherAccelerator(
            world_size=4,
            process_index=2,
            gathered_values=[
                torch.tensor([1, 1, 2, 2]),
                torch.tensor([10.0, 0.0, 11.0, 0.0, 20.0, 30.0, 21.0, 31.0]),
            ],
        )
        accelerator.parallelism_config = SimpleNamespace(
            cp_size=2,
            cp_enabled=True,
            dp_replicate_size=2,
            dp_shard_size=1,
        )
        accelerator.torch_device_mesh = MagicMock()
        accelerator.torch_device_mesh.get_group.return_value = MagicMock()
        accelerator.torch_device_mesh.get_local_rank.return_value = 0

        gathered = gather_variable_batch_tensor(torch.tensor([20.0, 30.0]), accelerator)

        self.assertTrue(torch.equal(gathered, torch.tensor([10.0, 11.0, 20.0, 30.0, 21.0, 31.0])))
        self.assertEqual(tuple(accelerator.gather_inputs[0].shape), (1,))
        self.assertEqual(tuple(accelerator.gather_inputs[1].shape), (2,))

    def test_variable_batch_tensor_gather_preserves_single_replica_cp_ranks(self):
        accelerator = _GatherAccelerator(
            world_size=2,
            process_index=1,
            gathered_values=[
                torch.tensor([2, 2]),
                torch.tensor([10.0, 20.0, 11.0, 21.0]),
            ],
        )
        accelerator.parallelism_config = SimpleNamespace(
            cp_size=2,
            cp_enabled=True,
            dp_replicate_size=1,
            dp_shard_size=1,
        )
        accelerator.torch_device_mesh = MagicMock()
        accelerator.torch_device_mesh.get_group.return_value = MagicMock()
        accelerator.torch_device_mesh.get_local_rank.return_value = 1

        gathered = gather_variable_batch_tensor(torch.tensor([11.0, 21.0]), accelerator)

        self.assertTrue(torch.equal(gathered, torch.tensor([10.0, 20.0, 11.0, 21.0])))

    def test_sample_weighted_loss_uses_fixed_two_value_contribution(self):
        accelerator = _GatherAccelerator(
            world_size=2,
            process_index=1,
            gathered_values=[torch.tensor([2.0, 1.0, 12.0, 3.0])],
        )

        result = gather_sample_weighted_scalar(torch.tensor(4.0), local_batch_size=3, accelerator=accelerator)

        self.assertEqual(float(result), 3.5)
        self.assertTrue(torch.equal(accelerator.gather_inputs[0], torch.tensor([12.0, 3.0])))
        self.assertEqual(tuple(accelerator.gather_inputs[0].shape), (2,))


if __name__ == "__main__":
    unittest.main()
