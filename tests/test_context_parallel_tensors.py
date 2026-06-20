import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.context_parallel_tensors import (
    prepare_cp_attention_mask,
    shard_cp_tensor,
    unshard_cp_tensor,
)


class ContextParallelTensorHelpersTests(unittest.TestCase):
    def test_shard_helpers_are_noops_without_context_parallel(self):
        tensor = torch.randn(2, 4, 3)
        parallel_config = SimpleNamespace(context_parallel_config=None)

        self.assertIs(shard_cp_tensor(tensor, parallel_config), tensor)
        self.assertIs(unshard_cp_tensor(tensor, parallel_config), tensor)

    def test_attention_mask_expands_without_context_parallel(self):
        mask = torch.tensor([[True, False, True]])

        prepared = prepare_cp_attention_mask(mask, 3, None, model_name="Test")

        self.assertEqual(prepared.shape, (1, 1, 1, 3))
        self.assertTrue(torch.equal(prepared[:, 0, 0], mask))

    def test_attention_mask_crops_to_full_ulysses_key_length(self):
        mask = torch.arange(10).view(1, 10)
        parallel_config = SimpleNamespace(context_parallel_config=SimpleNamespace(ulysses_degree=2))

        right = prepare_cp_attention_mask(mask, 3, parallel_config, model_name="Test", crop="right")
        left = prepare_cp_attention_mask(mask, 3, parallel_config, model_name="Test", crop="left")

        self.assertTrue(torch.equal(right.flatten(), torch.tensor([4, 5, 6, 7, 8, 9])))
        self.assertTrue(torch.equal(left.flatten(), torch.tensor([0, 1, 2, 3, 4, 5])))

    def test_attention_mask_rejects_short_ulysses_mask(self):
        mask = torch.ones(1, 5, dtype=torch.bool)
        parallel_config = SimpleNamespace(context_parallel_config=SimpleNamespace(ulysses_degree=2))

        with self.assertRaisesRegex(ValueError, "shorter than attention key length"):
            prepare_cp_attention_mask(mask, 3, parallel_config, model_name="Test")


if __name__ == "__main__":
    unittest.main()
