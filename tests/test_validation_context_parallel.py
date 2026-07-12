import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.training.validation import Validation, _ValidationWorkItem


def _validation_for_rank(rank: int, *, world_size: int = 8):
    validation = Validation.__new__(Validation)
    validation.accelerator = SimpleNamespace(
        num_processes=world_size,
        process_index=rank,
        is_main_process=rank == 0,
    )
    validation.config = SimpleNamespace(validation_multigpu="batch-parallel")
    return validation


def _work_items(count: int):
    return [
        _ValidationWorkItem(
            index=idx,
            shortname=f"item_{idx}",
            prompt=f"prompt {idx}",
            conditioning=None,
            adapter_strength=None,
        )
        for idx in range(count)
    ]


class ValidationContextParallelTests(unittest.TestCase):
    @contextmanager
    def _patch_cp(self, *, data_rank: int, data_local_rank: int, data_parallel_size: int = 4):
        with (
            patch("simpletuner.helpers.training.validation.get_cp_info", return_value=(True, object(), 0, 2)),
            patch(
                "simpletuner.helpers.training.validation.get_model_replica_data_info",
                return_value=(True, data_rank, data_local_rank, 2, data_parallel_size),
            ),
        ):
            yield

    def test_context_parallel_splits_prompts_by_model_replica(self):
        items = _work_items(5)
        validation = _validation_for_rank(2)
        with self._patch_cp(data_rank=1, data_local_rank=0):
            local_items, use_distributed, worker_count = validation._split_validation_work_items(items)

        self.assertTrue(use_distributed)
        self.assertEqual(worker_count, 4)
        self.assertEqual([item.index for item in local_items], [1])

    def test_context_parallel_replicates_prompt_within_cp_group(self):
        items = _work_items(3)
        leader = _validation_for_rank(0)
        peer = _validation_for_rank(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            leader_items, _, _ = leader._split_validation_work_items(items)
        with self._patch_cp(data_rank=0, data_local_rank=1):
            peer_items, _, _ = peer._split_validation_work_items(items)

        self.assertEqual([item.index for item in leader_items], [0])
        self.assertEqual([item.index for item in peer_items], [0])

    def test_context_parallel_only_leader_publishes_payloads(self):
        leader = _validation_for_rank(0)
        peer = _validation_for_rank(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            self.assertTrue(leader._should_publish_validation_payloads())
        with self._patch_cp(data_rank=0, data_local_rank=1):
            self.assertFalse(peer._should_publish_validation_payloads())

    def test_context_parallel_keeps_single_prompt_distributed(self):
        validation = _validation_for_rank(0)
        items = _work_items(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            local_items, use_distributed, worker_count = validation._split_validation_work_items(items)

        self.assertTrue(use_distributed)
        self.assertEqual(worker_count, 4)
        self.assertEqual([item.index for item in local_items], [0])


if __name__ == "__main__":
    unittest.main()
