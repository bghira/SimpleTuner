import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.context_parallel import (
    ContextParallelTopology,
    build_context_parallel_topology,
    configure_cp_only_accelerator,
    normalize_context_parallel_size,
    normalize_context_parallel_strategy,
)


class ContextParallelTopologyTests(unittest.TestCase):
    def test_cp_only_topology_uses_replicated_data_parallel_axis(self):
        topology = build_context_parallel_topology(
            SimpleNamespace(fsdp_enable=False),
            world_size=8,
            cp_size=2,
            strategy="allgather",
        )

        self.assertEqual(topology.cp_size, 2)
        self.assertEqual(topology.dp_replicate_size, 4)
        self.assertEqual(topology.dp_shard_size, 1)

    def test_cp_only_two_gpus_has_one_data_parallel_batch(self):
        topology = build_context_parallel_topology(
            SimpleNamespace(fsdp_enable=False),
            world_size=2,
            cp_size=2,
            strategy="allgather",
        )

        self.assertEqual(topology.dp_replicate_size, 1)
        self.assertEqual(topology.dp_shard_size, 1)

    def test_fsdp2_topology_keeps_existing_shard_axis(self):
        topology = build_context_parallel_topology(
            SimpleNamespace(fsdp_enable=True, fsdp_version=2),
            world_size=8,
            cp_size=2,
            strategy="allgather",
        )

        self.assertEqual(topology.dp_replicate_size, 1)
        self.assertEqual(topology.dp_shard_size, 4)

    def test_rejects_non_divisible_world_size(self):
        with self.assertRaisesRegex(ValueError, "evenly divide"):
            build_context_parallel_topology(
                SimpleNamespace(fsdp_enable=False),
                world_size=7,
                cp_size=2,
                strategy="allgather",
            )

    def test_normalizes_context_parallel_options(self):
        self.assertIsNone(normalize_context_parallel_size(None))
        self.assertEqual(normalize_context_parallel_size("2"), 2)
        self.assertEqual(normalize_context_parallel_strategy("ALLGATHER"), "allgather")
        self.assertEqual(normalize_context_parallel_strategy(None), "allgather")
        with self.assertRaises(ValueError):
            normalize_context_parallel_size("nope")
        with self.assertRaises(ValueError):
            normalize_context_parallel_strategy("ring")


class ContextParallelAcceleratorAttachTests(unittest.TestCase):
    @patch("simpletuner.helpers.training.context_parallel.torch.distributed.is_initialized", return_value=True)
    @patch("simpletuner.helpers.training.context_parallel.torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.device_mesh.init_device_mesh")
    def test_configure_cp_only_attaches_parallelism_config_and_mesh(
        self,
        mock_init_device_mesh,
        _mock_dist_available,
        _mock_dist_initialized,
    ):
        mesh = MagicMock()
        mock_init_device_mesh.return_value = mesh
        accelerator = SimpleNamespace(device=torch.device("cpu"), state=SimpleNamespace())

        configure_cp_only_accelerator(
            accelerator,
            ContextParallelTopology(cp_size=2, dp_replicate_size=4, dp_shard_size=1, strategy="allgather"),
        )

        mock_init_device_mesh.assert_called_once_with(
            device_type="cpu",
            mesh_shape=(4, 2, 1),
            mesh_dim_names=("dp_replicate", "ring", "ulysses"),
        )
        self.assertIs(accelerator.state.device_mesh, mesh)
        self.assertEqual(accelerator.state.parallelism_config.cp_size, 2)
        self.assertEqual(accelerator.state.parallelism_config.dp_replicate_size, 4)
        self.assertEqual(accelerator.state.parallelism_config.dp_shard_size, 1)
        self.assertTrue(accelerator.state.parallelism_config.cp_enabled)
        self.assertFalse(accelerator.state.parallelism_config.tp_enabled)
        self.assertFalse(accelerator.state.parallelism_config.dp_shard_enabled)

    @patch("simpletuner.helpers.training.context_parallel.torch.distributed.is_initialized", return_value=True)
    @patch("simpletuner.helpers.training.context_parallel.torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.device_mesh.init_device_mesh")
    def test_configure_cp_only_attaches_ulysses_mesh_for_alltoall(
        self,
        mock_init_device_mesh,
        _mock_dist_available,
        _mock_dist_initialized,
    ):
        mesh = MagicMock()
        mock_init_device_mesh.return_value = mesh
        accelerator = SimpleNamespace(device=torch.device("cpu"), state=SimpleNamespace())

        configure_cp_only_accelerator(
            accelerator,
            ContextParallelTopology(cp_size=2, dp_replicate_size=4, dp_shard_size=1, strategy="alltoall"),
        )

        mock_init_device_mesh.assert_called_once_with(
            device_type="cpu",
            mesh_shape=(4, 1, 2),
            mesh_dim_names=("dp_replicate", "ring", "ulysses"),
        )
        self.assertIs(accelerator.state.device_mesh, mesh)


class ContextParallelSyncMeshTests(unittest.TestCase):
    def test_get_cp_info_flattens_diffusers_ring_ulysses_mesh(self):
        from simpletuner.helpers.data_backend.runtime.context_parallel_sync import get_cp_info

        parallelism_config = SimpleNamespace(cp_size=2, cp_enabled=True)
        cp_mesh = MagicMock()
        cp_group = MagicMock()
        cp_mesh.get_group.return_value = cp_group
        cp_mesh.get_local_rank.return_value = 1
        ring_ulysses_mesh = MagicMock()
        ring_ulysses_mesh._flatten.return_value = cp_mesh

        device_mesh = MagicMock()
        device_mesh.mesh_dim_names = ("dp_replicate", "ring", "ulysses")
        device_mesh.get_group.side_effect = KeyError("cp")
        device_mesh.__getitem__.return_value = ring_ulysses_mesh

        accelerator = SimpleNamespace(parallelism_config=parallelism_config, torch_device_mesh=device_mesh)

        cp_enabled, group, rank, size = get_cp_info(accelerator)

        self.assertTrue(cp_enabled)
        self.assertIs(group, cp_group)
        self.assertEqual(rank, 1)
        self.assertEqual(size, 2)
        device_mesh.__getitem__.assert_called_once_with(("ring", "ulysses"))


if __name__ == "__main__":
    unittest.main()
