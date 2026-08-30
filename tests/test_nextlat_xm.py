import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from simpletuner.helpers.training.explorative_modeling import (
    ExplorativeModelingConfig,
    blockwise_cross_entropy,
    repeat_batch_for_candidates,
    reshape_candidate_batch,
    select_min_candidate_loss,
    select_winning_candidates,
)
from simpletuner.helpers.training.nextlat import (
    NextLatRegularizer,
    infer_nextlat_block_count,
    infer_nextlat_hidden_size,
    nextlat_enabled_from_config,
)
from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry


class DummyAccelerator:
    device = torch.device("cpu")


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Linear(4, 4) for _ in range(3)]))


class TransformerBlocksOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])


class AttentionInnerDimDiffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_attention_heads=3, attention_head_dim=2)
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])


class NextLatXmTests(unittest.TestCase):
    def test_field_registry_exposes_nextlat_and_xm_options(self):
        registry = FieldRegistry()
        self.assertEqual(registry.get_field("xm_enabled").arg_name, "--xm_enabled")
        self.assertEqual(registry.get_field("nextlat_enabled").arg_name, "--nextlat_enabled")
        self.assertEqual(registry.get_field("xm_candidate_count").default_value, 1)
        self.assertEqual(registry.get_field("nextlat_block_index").default_value, -1)

    def test_xm_config_rejects_enabled_single_candidate(self):
        config = SimpleNamespace(
            xm_enabled=True,
            xm_candidate_count=1,
            xm_training_target="route",
            xm_selection_scope="sample",
            xm_block_size=0,
        )
        with self.assertRaisesRegex(ValueError, "at least 2"):
            ExplorativeModelingConfig.from_config(config)

    def test_xm_config_uses_defaults_for_missing_mock_attributes(self):
        config = MagicMock()
        xm_config = ExplorativeModelingConfig.from_config(config)
        self.assertFalse(xm_config.enabled)
        self.assertEqual(xm_config.candidate_count, 1)
        self.assertEqual(xm_config.training_target, "noise")
        self.assertEqual(xm_config.selection_scope, "sample")
        self.assertFalse(nextlat_enabled_from_config(config))

    def test_candidate_selection_uses_min_per_sample(self):
        candidate_losses = torch.tensor([[0.8, 0.2, 0.5], [0.1, 0.4, 0.3]])
        loss, winners = select_min_candidate_loss(candidate_losses)
        self.assertTrue(torch.equal(winners, torch.tensor([1, 0, 1])))
        self.assertAlmostEqual(loss.item(), (0.1 + 0.2 + 0.3) / 3)

    def test_repeat_batch_for_candidates_uses_candidate_major_order(self):
        values = torch.tensor([[1, 10], [2, 20], [3, 30]])
        expanded = repeat_batch_for_candidates(values, candidate_count=2)
        expected = torch.tensor([[1, 10], [2, 20], [3, 30], [1, 10], [2, 20], [3, 30]])
        self.assertTrue(torch.equal(expanded, expected))
        self.assertTrue(torch.equal(reshape_candidate_batch(expanded, 2)[1], values))

    def test_select_winning_candidates_restores_original_batch(self):
        values = torch.arange(2 * 3 * 4).reshape(6, 4)
        winners = torch.tensor([1, 0, 1])
        selected = select_winning_candidates(values, winners, candidate_count=2)
        candidate_view = reshape_candidate_batch(values, 2)
        expected = torch.stack([candidate_view[1, 0], candidate_view[0, 1], candidate_view[1, 2]])
        self.assertTrue(torch.equal(selected, expected))

    def test_blockwise_cross_entropy_returns_sample_losses(self):
        logits = torch.tensor(
            [
                [[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]],
                [[0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [5.0, 0.0]],
            ]
        )
        targets = torch.tensor([[0, 1, 0, 1], [1, 0, -100, -100]])
        losses = blockwise_cross_entropy(logits, targets, block_size=2)
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertLess(losses.max().item(), 0.01)

    def test_nextlat_regularizer_computes_loss_from_captured_layer(self):
        config = SimpleNamespace(
            nextlat_enabled=True,
            nextlat_weight=0.5,
            nextlat_block_index=-1,
            nextlat_state_loss="smooth_l1",
            nextlat_kl_weight=0.0,
            weight_dtype=torch.float32,
        )
        model = DummyTransformer()
        self.assertEqual(infer_nextlat_hidden_size(model), 4)
        self.assertEqual(infer_nextlat_block_count(model), 3)
        regularizer = NextLatRegularizer(config, DummyAccelerator(), hidden_size=4, block_count=3)
        regularizer.attach_to_model(model, dtype=torch.float32)
        hidden = torch.randn(2, 5, 4)
        loss, logs = regularizer.compute_loss({"layer_2": hidden}, {})
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("nextlat_loss", logs)
        self.assertIn("nextlat_state_loss", logs)

    def test_nextlat_block_count_accepts_transformer_blocks_without_single_blocks(self):
        self.assertEqual(infer_nextlat_block_count(TransformerBlocksOnly()), 2)

    def test_nextlat_hidden_size_prefers_token_width_over_attention_inner_dim(self):
        self.assertEqual(infer_nextlat_hidden_size(AttentionInnerDimDiffers()), 4)

    def test_nextlat_predictor_accepts_bfloat16_hidden_states_with_float_parameters(self):
        config = SimpleNamespace(
            nextlat_enabled=True,
            nextlat_weight=0.5,
            nextlat_block_index=-1,
            nextlat_state_loss="smooth_l1",
            nextlat_kl_weight=0.0,
        )
        regularizer = NextLatRegularizer(config, DummyAccelerator(), hidden_size=4, block_count=1)
        regularizer.attach_to_model(nn.Module(), dtype=torch.float32)
        hidden = torch.randn(2, 5, 4, dtype=torch.bfloat16)

        loss, _ = regularizer.compute_loss({"layer_0": hidden}, {})

        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
