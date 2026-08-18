import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.minimax_music3.eval_control_lora import (
    checkpoint_args,
    classifier_free_guidance_logits,
    exclude_semantic_target,
    nearest_reference_codes,
    rollout_metrics,
    semantic_target_ranks,
    shifted_reference_sample,
    substituted_reference_sample,
    summarize_target_ranks,
    summarize_teacher_forced_pair,
)
from scripts.minimax_music3.eval_prefix_distillation import parse_guidance_scales, sustained_relock_frame
from scripts.minimax_music3.eval_reference_control import (
    is_validation_clip,
    mapped_reference_positions,
    parse_probabilities,
    select_pair_path,
    sustained_relock_latency,
)
from scripts.minimax_music3.train_control_lora import (
    ControlLoRATrainingModel,
    aligned_reference_hint,
    build_control_datasets,
    collate_control_samples,
    generated_warmup_codes,
    initialize_distributed,
    load_clip_ids_csv,
    mismatched_reference_margin_loss,
    reference_delta_control_hidden_states,
    replace_with_mismatched_references,
    semantic_cross_entropy,
    sequential_corrupted_feedback,
    target_control_teacher_hidden_states,
    use_mismatched_reference_step,
)
from scripts.minimax_music3.train_prefix_distillation import (
    base_model_teacher,
    frame_loss_weights,
    prefix_warmup_codes,
    reference_prefix_embeddings,
    special_token_id,
    teacher_targets,
    weighted_frame_mean,
    weighted_head_mean,
)
from scripts.minimax_music3.train_reference_control import (
    CachedStylePairDataset,
    load_initial_weights,
    splice_sampled_feedback,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (
    ControlLoRAConfig,
    ControlResidualProjector,
    MiniMaxMusic3ControlLoRAAdapter,
    MiniMaxMusic3ReferenceControlAdapter,
    ReferenceConditionedDecoderLayer,
    ReferenceControlConfig,
    aligned_reference_mask,
    create_qwen_lokr_adapter,
    prefix_adapter_checkpoint_filename,
    quantize_qwen_linears,
)


def tiny_qwen() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )


class ReferenceControlTest(unittest.TestCase):
    def test_control_residual_query_gate_preserves_old_path_at_initialization(self):
        projector = ControlResidualProjector(hidden_size=4, rank=2)
        with torch.no_grad():
            projector.down.weight.fill_(0.25)
            projector.up.weight.fill_(0.5)
        control = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        first_query = torch.zeros_like(control)
        second_query = torch.ones_like(control)
        expected = projector.up(torch.nn.functional.silu(projector.down(projector.norm(control))))
        torch.testing.assert_close(projector(control, first_query), expected)
        torch.testing.assert_close(projector(control, second_query), expected)

        with torch.no_grad():
            projector.query_down.weight[0, 0] = 1.0
        self.assertFalse(torch.equal(projector(control, first_query), projector(control, control)))

    def test_reference_delta_control_states_are_zero_for_null_reference(self):
        model = tiny_qwen().eval()
        text_embeddings = torch.randn(1, 2, 64)
        reference_embeddings = torch.randn(1, 3, 64)
        positions = torch.arange(3, dtype=torch.float32).unsqueeze(0)
        controls = reference_delta_control_hidden_states(
            model,
            text_embeddings,
            reference_embeddings,
            positions,
            positions,
            null_reference=True,
            text_attention_mask=torch.ones(1, 2, dtype=torch.long),
            reference_attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        self.assertEqual(len(controls), len(model.model.layers))
        for control_state in controls:
            self.assertLess(control_state.abs().max().item(), 1e-6)

    def test_target_control_teacher_restores_lokr_and_training_mode(self):
        class Lokr:
            def __init__(self):
                self.values = []

            def set_multiplier(self, value):
                self.values.append(value)

        model = tiny_qwen().train()
        lokr = Lokr()
        positions = torch.arange(3, dtype=torch.float32).unsqueeze(0)
        with patch(
            "scripts.minimax_music3.train_control_lora.embed_rvq_frames",
            return_value=torch.randn(1, 3, 64),
        ):
            controls = target_control_teacher_hidden_states(
                model,
                None,
                lokr,
                torch.randn(1, 2, 64),
                torch.zeros(1, 3, 8, dtype=torch.long),
                positions,
                torch.ones(1, 2, dtype=torch.long),
            )
        self.assertEqual(len(controls), len(model.model.layers))
        self.assertTrue(model.training)
        self.assertEqual(lokr.values, [0.0, 1.0])

    def test_shifted_reference_sample_preserves_target_and_rotates_reference(self):
        sample = {
            "target_codes": torch.tensor([[10], [11], [12]]),
            "reference_codes": torch.tensor([[0], [1], [2]]),
            "query_positions": torch.tensor([0.0, 1.0, 2.0]),
            "key_positions": torch.tensor([0.0, 1.0, 2.0]),
        }
        shifted = shifted_reference_sample(sample, 1)
        self.assertIs(shifted["target_codes"], sample["target_codes"])
        torch.testing.assert_close(shifted["reference_codes"], torch.tensor([[2], [0], [1]]))
        torch.testing.assert_close(shifted["feedback_warmup_codes"], torch.tensor([[0]]))
        torch.testing.assert_close(sample["reference_codes"], torch.tensor([[0], [1], [2]]))

    def test_substituted_reference_sample_maps_query_to_replacement_timeline(self):
        sample = {
            "target_codes": torch.tensor([[10], [11], [12]]),
            "reference_codes": torch.tensor([[0], [1], [2]]),
            "query_positions": torch.tensor([2.0, 3.0, 4.0]),
            "key_positions": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        replacement = {
            "reference_codes": torch.tensor([[20], [21], [22], [23], [24]]),
            "key_positions": torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
        }
        substituted = substituted_reference_sample(sample, replacement)
        self.assertIs(substituted["target_codes"], sample["target_codes"])
        torch.testing.assert_close(substituted["reference_codes"], replacement["reference_codes"])
        torch.testing.assert_close(substituted["query_positions"], torch.tensor([10.0, 12.0, 14.0]))
        torch.testing.assert_close(substituted["feedback_warmup_codes"], torch.tensor([[1]]))

    def test_teacher_forced_pair_summary_separates_fixes_and_regressions(self):
        summary = summarize_teacher_forced_pair(
            reference_ce=torch.tensor([0.5, 2.0, 0.25, 3.0]),
            null_ce=torch.tensor([1.0, 1.0, 2.0, 2.0]),
            reference_correct=torch.tensor([True, False, True, False]),
            null_correct=torch.tensor([False, True, False, False]),
            semantic_transition=torch.tensor([False, True, True, False]),
        )
        self.assertEqual(summary["reference_fixes"], 2)
        self.assertEqual(summary["reference_regressions"], 1)
        self.assertAlmostEqual(summary["conditional_fix_rate"], 2 / 3)
        self.assertAlmostEqual(summary["conditional_regression_rate"], 1.0)
        self.assertAlmostEqual(summary["net_available_headroom_gain"], 1 / 3)
        self.assertAlmostEqual(summary["ce_gain_mean"], 0.0625)
        self.assertAlmostEqual(summary["semantic_transition"]["ce_gain_mean"], 0.375)

    def test_teacher_forced_pair_summary_marks_undefined_conditional_rates(self):
        summary = summarize_teacher_forced_pair(
            reference_ce=torch.tensor([0.5, 0.25]),
            null_ce=torch.tensor([1.0, 0.5]),
            reference_correct=torch.tensor([True, True]),
            null_correct=torch.tensor([True, True]),
            semantic_transition=torch.tensor([False, False]),
        )
        self.assertIsNone(summary["conditional_fix_rate"])
        self.assertIsNone(summary["net_available_headroom_gain"])
        self.assertEqual(summary["semantic_transition"]["frames"], 0)

    def test_rollout_metrics_excludes_teacher_prefix(self):
        trace = SimpleNamespace(
            generated_codes=torch.tensor([[1, 1], [2, 2], [9, 3], [4, 4], [5, 0]]),
            target_log_probs=torch.tensor([-1.0, -1.0, -3.0, -0.5, -2.0]),
            target_top1=torch.tensor([True, True, False, True, False]),
            target_ranks=torch.tensor([1, 1, 10, 1, 6]),
            seconds=1.0,
            evaluation_start=2,
        )
        target = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        metrics = rollout_metrics(trace, target)
        self.assertEqual(metrics["teacher_prefix_frames"], 2)
        self.assertEqual(metrics["evaluated_frames"], 3)
        self.assertEqual(metrics["first_semantic_miss"], 2)
        self.assertEqual(metrics["semantic_match_frames"], [3, 4])
        self.assertAlmostEqual(metrics["semantic_target_top1"], 2 / 3)
        self.assertAlmostEqual(metrics["acoustic_target_top1"], 2 / 3)
        self.assertAlmostEqual(metrics["target_semantic_top5_mean"], 1 / 3)
        self.assertAlmostEqual(metrics["target_semantic_top50_mean"], 1.0)

    def test_semantic_target_rank_metrics(self):
        logits = torch.tensor([[[1.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0]]])
        targets = torch.tensor([[2, 3]])
        ranks = semantic_target_ranks(logits, targets)
        torch.testing.assert_close(ranks, torch.tensor([[2, 4]]))
        summary = summarize_target_ranks(ranks.flatten(), torch.tensor([1, 100]))
        self.assertEqual(summary["reference_top5"], 1.0)
        self.assertEqual(summary["null_top50"], 0.5)

    def test_classifier_free_guidance_uses_uncontrolled_branch_as_baseline(self):
        conditioned = torch.tensor([[4.0, 2.0]])
        unconditioned = torch.tensor([[1.0, 3.0]])
        torch.testing.assert_close(
            classifier_free_guidance_logits(conditioned, unconditioned, 1.5),
            torch.tensor([[5.5, 1.5]]),
        )
        torch.testing.assert_close(
            classifier_free_guidance_logits(conditioned, unconditioned, 0.0),
            unconditioned,
        )

    def test_exclude_semantic_target_preserves_other_logits(self):
        logits = torch.tensor([[1.0, 4.0, 3.0], [5.0, 6.0, 7.0]])
        perturbed = exclude_semantic_target(logits, torch.tensor([1, 2]))
        self.assertTrue(torch.isneginf(perturbed[0, 1]))
        self.assertTrue(torch.isneginf(perturbed[1, 2]))
        torch.testing.assert_close(perturbed[0, [0, 2]], logits[0, [0, 2]])
        torch.testing.assert_close(perturbed[1, [0, 1]], logits[1, [0, 1]])
        torch.testing.assert_close(logits, torch.tensor([[1.0, 4.0, 3.0], [5.0, 6.0, 7.0]]))

    def test_nearest_reference_codes_maps_query_timeline(self):
        reference = torch.tensor([[[10], [20], [30]]])
        actual = nearest_reference_codes(
            reference,
            torch.tensor([[0.1, 1.8, 3.9]]),
            torch.tensor([[0.0, 2.0, 4.0]]),
        )
        torch.testing.assert_close(actual, torch.tensor([[[10], [20], [30]]]))

    def test_partial_torchrun_environment_is_rejected(self):
        with patch.dict("os.environ", {"RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "RANK, LOCAL_RANK, and WORLD_SIZE"):
                initialize_distributed("cpu")

    def test_control_lora_training_wrapper_owns_indirect_adapter_parameters(self):
        model = tiny_qwen()
        model.requires_grad_(False)
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        depth_decoder = nn.Linear(2, 2)
        depth_decoder.requires_grad_(False)
        lokr_network = nn.Linear(2, 2, bias=False)
        wrapper = ControlLoRATrainingModel(model, depth_decoder, adapter, lokr_network)
        wrapper_parameter_ids = {id(parameter) for parameter in wrapper.parameters()}
        self.assertTrue({id(parameter) for parameter in adapter.parameters()} <= wrapper_parameter_ids)
        self.assertTrue({id(parameter) for parameter in lokr_network.parameters()} <= wrapper_parameter_ids)

    def test_aligned_reference_hint_interpolates_query_timeline(self):
        reference = torch.tensor([[[0.0], [10.0], [20.0]]])
        query = torch.tensor([[0.0, 0.5, 1.5, 2.0]])
        keys = torch.tensor([[0.0, 1.0, 2.0]])
        expected = torch.tensor([[[0.0], [5.0], [15.0], [20.0]]])
        torch.testing.assert_close(aligned_reference_hint(reference, query, keys), expected)

    def test_control_lora_is_exactly_zero_at_initialization(self):
        torch.manual_seed(0)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, model.config.hidden_size)
        expected = model.model(inputs_embeds=inputs, output_hidden_states=True).last_hidden_state
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        control = model.model(inputs_embeds=inputs + 0.1, output_hidden_states=True).hidden_states[1:]
        actual = model.model(
            inputs_embeds=inputs,
            control_hidden_states=control,
            control_query_start=3,
        ).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_control_lora_changes_only_controlled_positions(self):
        torch.manual_seed(0)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, model.config.hidden_size)
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        control = model.model(inputs_embeds=inputs + 0.1, output_hidden_states=True).hidden_states[1:]
        with torch.no_grad():
            adapter.residuals[0].up.weight.fill_(0.01)
        baseline = model.model(inputs_embeds=inputs).last_hidden_state
        controlled = model.model(
            inputs_embeds=inputs,
            control_hidden_states=control,
            control_query_start=3,
        ).last_hidden_state
        torch.testing.assert_close(controlled[:, :3], baseline[:, :3], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(controlled[:, 3:], baseline[:, 3:]))

    def test_control_lora_accepts_target_sized_contextual_deltas(self):
        torch.manual_seed(0)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, model.config.hidden_size)
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        with torch.no_grad():
            adapter.residuals[0].up.weight.fill_(0.01)
        control = tuple(torch.randn(1, 4, 64) for _ in model.model.layers)
        baseline = model.model(inputs_embeds=inputs).last_hidden_state
        controlled = model.model(
            inputs_embeds=inputs,
            control_hidden_states=control,
            control_query_start=3,
        ).last_hidden_state
        torch.testing.assert_close(controlled[:, :3], baseline[:, :3], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(controlled[:, 3:], baseline[:, 3:]))

    def test_control_lora_zero_scale_disables_contextual_deltas(self):
        torch.manual_seed(0)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, model.config.hidden_size)
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        with torch.no_grad():
            adapter.residuals[0].up.weight.fill_(0.01)
        control = tuple(torch.randn(1, 4, 64) for _ in model.model.layers)
        baseline = model.model(inputs_embeds=inputs).last_hidden_state
        controlled = model.model(
            inputs_embeds=inputs,
            control_hidden_states=control,
            control_query_start=3,
            control_scale=0.0,
        ).last_hidden_state
        torch.testing.assert_close(controlled, baseline, rtol=0.0, atol=0.0)

    def test_control_lora_zero_multiplier_disables_residuals(self):
        torch.manual_seed(0)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, model.config.hidden_size)
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        control = model.model(inputs_embeds=inputs + 0.1, output_hidden_states=True).hidden_states[1:]
        with torch.no_grad():
            adapter.residuals[0].up.weight.fill_(0.01)
        adapter.set_multiplier(0.0)
        baseline = model.model(inputs_embeds=inputs).last_hidden_state
        controlled = model.model(
            inputs_embeds=inputs,
            control_hidden_states=control,
            control_query_start=3,
        ).last_hidden_state
        torch.testing.assert_close(controlled, baseline, rtol=0.0, atol=0.0)

    def test_lokr_targets_attention_and_ffn_projections(self):
        model = tiny_qwen()
        network = create_qwen_lokr_adapter(model, rank=4, alpha=4.0)
        self.assertEqual(len(network.loras), len(model.model.layers) * 7)
        names = {module.lora_name for module in network.loras}
        self.assertTrue(any("self_attn_q_proj" in name for name in names))
        self.assertTrue(any("mlp_down_proj" in name for name in names))

    def test_lokr_can_force_bypass_mode_for_quantized_base(self):
        model = tiny_qwen()
        network = create_qwen_lokr_adapter(model, rank=4, alpha=4.0, bypass_mode=True)
        self.assertTrue(all(module.bypass_mode for module in network.loras))

    def test_qwen_quantization_requires_frozen_parameters(self):
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            quantize_qwen_linears(tiny_qwen(), "int8-weight-only")

    def test_prefix_adapter_checkpoint_filenames_are_distinct(self):
        self.assertEqual(prefix_adapter_checkpoint_filename("oftv2"), "qwen_prefix_oftv2.safetensors")
        self.assertEqual(prefix_adapter_checkpoint_filename("lokr"), "qwen_prefix_lokr.safetensors")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            prefix_adapter_checkpoint_filename("lora")

    def test_reference_guidance_scale_parser(self):
        self.assertEqual(parse_guidance_scales("1, 2.5,4"), (1.0, 2.5, 4.0))
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "non-negative"):
            parse_guidance_scales("1,-1")

    def test_reference_first_warmup_uses_conditioning_code(self):
        codes = torch.arange(48).reshape(1, 6, 8)
        actual = prefix_warmup_codes(None, None, None, None, codes, "reference-first")
        torch.testing.assert_close(actual, codes[:, :1])

    def test_generated_control_warmup_uses_frozen_text_branch(self):
        class Body(nn.Module):
            def forward(self, *, inputs_embeds, attention_mask, use_cache):
                self.attention_mask = attention_mask
                return SimpleNamespace(last_hidden_state=inputs_embeds + 1)

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Body()

        class Lokr:
            def __init__(self):
                self.multipliers = []

            def set_multiplier(self, value):
                self.multipliers.append(value)

        language_model = LanguageModel()
        language_model.train()
        lokr = Lokr()
        text_embeddings = torch.zeros(2, 3, 4)
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        expected = torch.full((2, 1, 8), 7, dtype=torch.long)
        with patch(
            "scripts.minimax_music3.train_control_lora.sample_codes_from_hidden",
            return_value=expected,
        ) as sample_codes:
            actual = generated_warmup_codes(language_model, object(), lokr, text_embeddings, attention_mask)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(language_model.model.attention_mask, attention_mask)
        torch.testing.assert_close(sample_codes.call_args.args[2], torch.ones(2, 1, 4))
        self.assertEqual(sample_codes.call_args.kwargs["top_k"], 1)
        self.assertEqual(lokr.multipliers, [0.0, 1.0])
        self.assertTrue(language_model.training)

    def test_sequential_corruption_uses_generated_history_after_clean_context(self):
        class Lokr:
            def __init__(self):
                self.multipliers = []

            def set_multiplier(self, value):
                self.multipliers.append(value)

        model = tiny_qwen().train()
        adapter = MiniMaxMusic3ControlLoRAAdapter(
            ControlLoRAConfig(hidden_size=64, residual_rank=4, layer_indices=(0, 1, 2, 3))
        )
        adapter.install(model)
        depth_decoder = SimpleNamespace(config=SimpleNamespace(num_codebooks=8))
        lokr = Lokr()
        clean_feedback = torch.tensor(
            [
                [[1] * 8, [2] * 8, [3] * 8, [4] * 8],
                [[5] * 8, [6] * 8, [7] * 8, [8] * 8],
            ]
        )
        sampled = [torch.full((2, 1, 8), value) for value in (10, 20, 30)]
        positions = torch.arange(4, dtype=torch.float32).expand(2, -1)

        def fake_embeddings(_language_model, _depth_decoder, codes):
            return codes[..., :1].float().expand(-1, -1, 64)

        with (
            patch("scripts.minimax_music3.train_control_lora.embed_rvq_frames", side_effect=fake_embeddings),
            patch("scripts.minimax_music3.train_control_lora.sample_codes_from_hidden", side_effect=sampled),
        ):
            feedback, fraction = sequential_corrupted_feedback(
                model,
                depth_decoder,
                lokr,
                torch.randn(2, 3, 64),
                clean_feedback,
                clean_feedback,
                positions,
                positions,
                hint_scale=1.0,
                null_reference=False,
                text_attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
                reference_attention_mask=torch.ones(2, 4, dtype=torch.long),
                loss_start=1,
                corruption_rate=1.0,
                sampling_top_k=1,
            )

        expected = clean_feedback.clone()
        expected[:, 2] = 20
        expected[:, 3] = 30
        torch.testing.assert_close(feedback, expected)
        self.assertEqual(fraction, 1.0)
        self.assertEqual(lokr.multipliers, [1.0, 0.0, 1.0])
        self.assertTrue(model.training)

    def test_legacy_control_checkpoint_keeps_reference_first_warmup(self):
        config = {
            "lokr_rank": 4,
            "lokr_alpha": 4.0,
            "hint_scale": 1.0,
            "reference_dropout": 0.1,
            "feedback_corruption_rate": 0.5,
            "feedback_sampling_top_k": 1,
            "semantic_loss_weight": 16.0,
        }
        self.assertEqual(checkpoint_args(config).feedback_warmup_mode, "reference-first")

    def test_semantic_head_weight_changes_code_loss_average(self):
        semantic = torch.tensor(2.0)
        depth = [torch.tensor(1.0), torch.tensor(1.0)]
        self.assertEqual(weighted_head_mean(semantic, depth, 2.0).item(), 1.5)

    def test_initial_semantic_frame_weight_targets_cold_start(self):
        logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
        targets = torch.tensor([[0, 0]])
        unweighted = semantic_cross_entropy(logits, targets, 1.0)
        weighted = semantic_cross_entropy(logits, targets, 3.0)
        self.assertGreater(weighted.item(), unweighted.item())
        with self.assertRaisesRegex(ValueError, "at least 1"):
            semantic_cross_entropy(logits, targets, 0.5)

    def test_prefix_sustained_relock_requires_consecutive_matches_after_miss(self):
        matches = torch.tensor([True, False, True, False, True, True, True])
        self.assertEqual(sustained_relock_frame(matches, 2), 4)
        self.assertIsNone(sustained_relock_frame(matches, 4))

    def test_teacher_targets_use_target_audio_prefix(self):
        class Tokenizer:
            def encode(self, token, add_special_tokens=False):
                return {"<|audio_start|>": [10], "<|audio_end|>": [11]}[token]

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(embed_tokens=nn.Embedding(170_000, 4))

        class Oft:
            def set_multiplier(self, value):
                pass

        language_model = LanguageModel()
        depth_decoder = SimpleNamespace(
            config=SimpleNamespace(num_codebooks=8, audio_vocab_size=16),
            audio_embeddings=nn.Embedding(7 * 16, 4),
        )
        target_codes = torch.zeros(1, 3, 8, dtype=torch.long)
        expected_prefix = reference_prefix_embeddings(
            language_model,
            depth_decoder,
            Tokenizer(),
            target_codes,
            null_reference=False,
        )
        with (
            patch(
                "scripts.minimax_music3.train_prefix_distillation.greedy_prefix_warmup_codes",
                return_value=torch.zeros(1, 1, 8, dtype=torch.long),
            ),
            patch(
                "scripts.minimax_music3.train_prefix_distillation.conditioned_hidden_states",
                return_value=torch.zeros(1, 3, 4),
            ) as conditioned,
            patch(
                "scripts.minimax_music3.train_prefix_distillation.semantic_logits",
                return_value=torch.zeros(1, 3, 16),
            ),
            patch(
                "scripts.minimax_music3.train_prefix_distillation.depth_outputs",
                return_value=([], torch.zeros(1, 3, 4)),
            ),
        ):
            teacher_targets(
                language_model,
                depth_decoder,
                Oft(),
                Tokenizer(),
                torch.ones(1, 2, dtype=torch.long),
                target_codes,
            )
        torch.testing.assert_close(conditioned.call_args.args[2], expected_prefix)

    def test_prefix_anchor_weights_change_frame_average(self):
        weights = frame_loss_weights(4, 2, 3.0, device=torch.device("cpu"))
        torch.testing.assert_close(weights, torch.tensor([3.0, 3.0, 1.0, 1.0]))
        values = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        self.assertEqual(weighted_frame_mean(values, weights).item(), 0.75)

    def test_reference_prefix_preserves_delimiters_when_reference_is_null(self):
        class Tokenizer:
            def encode(self, token, add_special_tokens=False):
                if add_special_tokens:
                    raise AssertionError
                return {"<|audio_start|>": [10], "<|audio_end|>": [11]}[token]

        class Embed(nn.Module):
            def forward(self, token_ids):
                return token_ids.float().unsqueeze(-1).repeat_interleave(4, dim=-1)

        language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=Embed()))
        depth_decoder = SimpleNamespace(
            config=SimpleNamespace(num_codebooks=8, audio_vocab_size=16),
            audio_embeddings=nn.Embedding(7 * 16, 4),
        )
        reference_codes = torch.zeros(1, 3, 8, dtype=torch.long)
        prefix = reference_prefix_embeddings(
            language_model,
            depth_decoder,
            Tokenizer(),
            reference_codes,
            null_reference=True,
        )
        self.assertEqual(prefix.shape, (1, 5, 4))
        torch.testing.assert_close(prefix[:, 0], torch.full((1, 4), 10.0))
        torch.testing.assert_close(prefix[:, 1:-1], torch.zeros(1, 3, 4))
        torch.testing.assert_close(prefix[:, -1], torch.full((1, 4), 11.0))

    def test_teacher_context_restores_oft_multiplier_and_training_mode(self):
        class Oft:
            def __init__(self):
                self.values = []

            def set_multiplier(self, value):
                self.values.append(value)

        model = nn.Linear(2, 2).train()
        oft = Oft()
        with base_model_teacher(model, oft):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(model.training)
        self.assertEqual(oft.values, [0.0, 1.0])

    def test_special_token_must_be_atomic(self):
        tokenizer = SimpleNamespace(encode=lambda token, add_special_tokens=False: [1, 2])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            special_token_id(tokenizer, "<|audio_start|>")

    def test_load_initial_weights_requires_matching_topology(self):
        config = ReferenceControlConfig(
            hidden_size=64,
            control_dim=32,
            num_heads=4,
            layer_indices=(1, 3),
        )
        source = MiniMaxMusic3ReferenceControlAdapter(config)
        target = MiniMaxMusic3ReferenceControlAdapter(config)
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / "reference_control.json").write_text(
                json.dumps(config.to_dict()),
                encoding="utf-8",
            )
            save_file(source.state_dict(), checkpoint_dir / "reference_control.safetensors")
            load_initial_weights(checkpoint_dir, target, None)
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
                torch.testing.assert_close(target_parameter, source_parameter)

            mismatched = MiniMaxMusic3ReferenceControlAdapter(
                ReferenceControlConfig(
                    hidden_size=64,
                    control_dim=16,
                    num_heads=4,
                    layer_indices=(1, 3),
                )
            )
            with self.assertRaisesRegex(ValueError, "topology"):
                load_initial_weights(checkpoint_dir, mismatched, None)

    def test_aligned_reference_mask(self):
        query = torch.tensor([[2.0, 5.0]])
        keys = torch.tensor([[0.0, 2.0, 4.0, 6.0]])
        expected = torch.tensor([[[True, True, True, False], [False, False, True, True]]])
        torch.testing.assert_close(aligned_reference_mask(query, keys, window_frames=2), expected)

    def test_adapter_preserves_output_without_reference(self):
        torch.manual_seed(1)
        model = tiny_qwen().eval()
        inputs = torch.randn(1, 7, 64)
        expected = model.model(inputs_embeds=inputs).last_hidden_state
        adapter = MiniMaxMusic3ReferenceControlAdapter(
            ReferenceControlConfig(
                hidden_size=64,
                control_dim=32,
                num_heads=4,
                layer_indices=(1, 3),
                gate_init=0.0,
            )
        )
        adapter.install(model)
        actual = model.model(inputs_embeds=inputs).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        adapter.uninstall(model)
        self.assertNotIsInstance(model.model.layers[1], ReferenceConditionedDecoderLayer)

    def test_checkpointed_backward_reaches_control_modules(self):
        torch.manual_seed(2)
        model = tiny_qwen().train()
        model.requires_grad_(False)
        model.gradient_checkpointing_enable()
        adapter = MiniMaxMusic3ReferenceControlAdapter(
            ReferenceControlConfig(
                hidden_size=64,
                control_dim=32,
                num_heads=4,
                layer_indices=(1, 3),
                window_frames=2,
            )
        )
        adapter.install(model)
        reference = adapter.encode_reference(torch.randn(1, 7, 64))
        output = model.model(
            inputs_embeds=torch.randn(1, 9, 64),
            use_cache=False,
            reference_memory=reference,
            reference_query_positions=torch.arange(5).unsqueeze(0),
            reference_key_positions=torch.arange(7).unsqueeze(0),
            reference_query_start=4,
        ).last_hidden_state
        output[:, 4:].square().mean().backward()
        self.assertIsNotNone(adapter.memory_encoder.proj.weight.grad)
        for control in adapter.controls:
            self.assertIsNotNone(control.to_q.weight.grad)
            self.assertIsNotNone(control.gate.grad)

    def test_mapped_reference_positions_use_full_pair_timelines(self):
        positions = mapped_reference_positions(3, target_frame_count=5, reference_frame_count=9)
        torch.testing.assert_close(positions, torch.tensor([0.0, 2.0, 4.0]))

    def test_select_pair_path_uses_training_validation_hash(self):
        with TemporaryDirectory() as directory:
            shard = Path(directory) / "shard-00000"
            shard.mkdir()
            validation_id = next(f"clip-{index}" for index in range(100) if is_validation_clip(f"clip-{index}", 0.1))
            path = shard / f"{validation_id}.safetensors"
            path.touch()
            self.assertEqual(select_pair_path(Path(directory), None, 0.1), path)

    def test_overfit_dataset_uses_requested_fixed_crop(self):
        with TemporaryDirectory() as directory:
            shard = Path(directory) / "shard-00000"
            shard.mkdir()
            path = shard / "overfit-clip.safetensors"
            codes = torch.arange(80).reshape(10, 8).to(torch.int16)
            save_file(
                {"target_codes": codes, "reference_codes": codes.clone()},
                path,
                metadata={"clip_id": path.stem, "prompt": "metal", "lyrics": "lyrics"},
            )
            dataset = CachedStylePairDataset(
                Path(directory),
                crop_frames=4,
                reference_context_frames=2,
                clip_id=path.stem,
                fixed_crop_start=3,
            )
            sample = dataset[0]
            torch.testing.assert_close(sample["target_codes"], codes[3:7].long())
            torch.testing.assert_close(sample["query_positions"], torch.arange(3, 7).float())
            self.assertEqual(sample["loss_start"], 0)

    def test_dataset_selects_requested_clip_ids_in_csv_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard-00000"
            shard.mkdir()
            codes = torch.arange(48).reshape(6, 8).to(torch.int16)
            for clip_id in ("clip-a", "clip-b"):
                save_file(
                    {"target_codes": codes, "reference_codes": codes.clone()},
                    shard / f"{clip_id}.safetensors",
                    metadata={"clip_id": clip_id, "prompt": "metal", "lyrics": "lyrics"},
                )
            csv_path = root / "clips.csv"
            csv_path.write_text("clip_id\nclip-b\nclip-a\n", encoding="utf-8")
            clip_ids = load_clip_ids_csv(csv_path)
            dataset = CachedStylePairDataset(
                root,
                crop_frames=4,
                reference_context_frames=1,
                clip_ids=clip_ids,
                fixed_crop_start=0,
            )
            self.assertEqual([path.stem for path in dataset.paths], ["clip-b", "clip-a"])

    def test_control_datasets_use_hash_validation_split_without_overfit_selection(self):
        args = SimpleNamespace(
            overfit_ids_csv=None,
            overfit_clip_id=None,
            random_crops=True,
            overfit_start_frame=0,
            cache_dir=Path("cache"),
            crop_frames=20,
        )
        training = object()
        validation = object()
        with patch(
            "scripts.minimax_music3.train_control_lora.CachedStylePairDataset",
            side_effect=(training, validation),
        ) as dataset:
            actual_training, actual_validation, clip_ids = build_control_datasets(args)
        self.assertIs(actual_training, training)
        self.assertIs(actual_validation, validation)
        self.assertEqual(clip_ids, ())
        self.assertEqual(dataset.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(dataset.call_args_list[1].kwargs["split"], "validation")
        self.assertIsNone(dataset.call_args_list[0].kwargs["fixed_crop_start"])

    def test_control_datasets_reuse_training_data_for_single_clip_overfit(self):
        args = SimpleNamespace(
            overfit_ids_csv=None,
            overfit_clip_id="clip-a",
            random_crops=False,
            overfit_start_frame=3,
            cache_dir=Path("cache"),
            crop_frames=20,
        )
        training = object()
        with patch(
            "scripts.minimax_music3.train_control_lora.CachedStylePairDataset",
            return_value=training,
        ) as dataset:
            actual_training, actual_validation, clip_ids = build_control_datasets(args)
        self.assertIs(actual_training, training)
        self.assertIs(actual_validation, training)
        self.assertEqual(clip_ids, ("clip-a",))
        self.assertEqual(dataset.call_count, 1)
        self.assertEqual(dataset.call_args.kwargs["fixed_crop_start"], 3)

    def test_control_lora_collate_pads_reference_timeline(self):
        samples = []
        for index, reference_frames in enumerate((4, 6)):
            samples.append(
                {
                    "clip_id": f"clip-{index}",
                    "metadata": {"clip_id": f"clip-{index}"},
                    "prompt": "prompt",
                    "lyrics": "lyrics",
                    "target_codes": torch.zeros(3, 8, dtype=torch.long),
                    "loss_start": 0,
                    "reference_codes": torch.full((reference_frames, 8), index, dtype=torch.long),
                    "query_positions": torch.arange(3, dtype=torch.float32),
                    "key_positions": torch.arange(reference_frames, dtype=torch.float32),
                }
            )
        batch = collate_control_samples(samples)
        self.assertEqual(batch["target_codes"].shape, (2, 3, 8))
        self.assertEqual(batch["reference_codes"].shape, (2, 6, 8))
        torch.testing.assert_close(
            batch["reference_attention_mask"],
            torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]),
        )
        torch.testing.assert_close(batch["key_positions"][0], torch.tensor([0, 1, 2, 3, 3, 3]).float())

    def test_mismatched_reference_preserves_warmup_and_remaps_timeline(self):
        def make_sample(clip_id, reference_values, query_positions, key_positions):
            return {
                "clip_id": clip_id,
                "metadata": {"clip_id": clip_id},
                "prompt": "prompt",
                "lyrics": "lyrics",
                "target_codes": torch.zeros(3, 8, dtype=torch.long),
                "loss_start": 0,
                "reference_codes": torch.tensor(reference_values, dtype=torch.long).unsqueeze(1).expand(-1, 8),
                "query_positions": torch.tensor(query_positions, dtype=torch.float32),
                "key_positions": torch.tensor(key_positions, dtype=torch.float32),
            }

        target = collate_control_samples([make_sample("target", [10, 11, 12], [5, 6, 7], [4, 6, 8])])
        replacement = collate_control_samples([make_sample("replacement", [20, 21, 22, 23], [20, 21, 22], [19, 20, 21, 22])])
        mismatched = replace_with_mismatched_references(target, replacement)
        torch.testing.assert_close(mismatched["reference_codes"], replacement["reference_codes"])
        torch.testing.assert_close(mismatched["matched_reference_codes"], target["reference_codes"])
        torch.testing.assert_close(
            mismatched["key_positions"], torch.tensor([[5.0, 5.6667, 6.3333, 7.0]]), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(mismatched["feedback_warmup_codes"], torch.full((1, 1, 8), 10))

    def test_mismatched_reference_schedule_is_reproducible(self):
        schedule = [use_mismatched_reference_step(step, 0.5) for step in range(20)]
        self.assertEqual(schedule, [False, True] * 10)
        self.assertEqual(sum(use_mismatched_reference_step(step, 0.25) for step in range(20)), 5)
        self.assertFalse(use_mismatched_reference_step(0, 0.0))

    def test_mismatched_reference_margin_pushes_wrong_target_logit_down(self):
        matched = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
        mismatched = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
        loss, _, _, active = mismatched_reference_margin_loss(matched, mismatched, torch.tensor([[0]]), 0.5)
        loss.backward()
        self.assertEqual(active.item(), 0.5)
        self.assertLess(matched.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(mismatched.grad[0, 0, 0].item(), 0.0)

    def test_dataset_prepends_true_context_without_expanding_loss_crop(self):
        with TemporaryDirectory() as directory:
            shard = Path(directory) / "shard-00000"
            shard.mkdir()
            path = shard / "context-clip.safetensors"
            codes = torch.arange(96).reshape(12, 8).to(torch.int16)
            save_file(
                {"target_codes": codes, "reference_codes": codes.clone()},
                path,
                metadata={"clip_id": path.stem, "prompt": "metal", "lyrics": "lyrics"},
            )
            dataset = CachedStylePairDataset(
                Path(directory),
                crop_frames=4,
                reference_context_frames=2,
                clip_id=path.stem,
                fixed_crop_start=5,
                target_context_frames=3,
            )
            sample = dataset[0]
            torch.testing.assert_close(sample["target_codes"], codes[2:9].long())
            torch.testing.assert_close(sample["query_positions"], torch.arange(2, 9).float())
            self.assertEqual(sample["loss_start"], 3)

    def test_sampled_feedback_preserves_context_and_first_supervised_predecessor(self):
        clean = torch.arange(48).reshape(1, 6, 8)
        sampled = clean + 1000
        feedback, fraction = splice_sampled_feedback(
            clean,
            sampled,
            loss_start=2,
            corruption_rate=1.0,
        )
        torch.testing.assert_close(feedback[:, :3], clean[:, :3])
        torch.testing.assert_close(feedback[:, 3:], sampled[:, 2:5])
        self.assertEqual(fraction, 1.0)

    def test_sustained_relock_latency_ignores_single_frame_reacquisition(self):
        top1 = torch.tensor([True, True, False, True, False, True, True, True, False])
        self.assertEqual(sustained_relock_latency(top1, perturbation_frame=2, consecutive_frames=3), 3)
        self.assertIsNone(sustained_relock_latency(top1, perturbation_frame=6, consecutive_frames=3))

    def test_feedback_probability_parser(self):
        self.assertEqual(parse_probabilities("0,0.5,1"), (0.0, 0.5, 1.0))
        with self.assertRaisesRegex(ValueError, "values in"):
            parse_probabilities("1.1")


if __name__ == "__main__":
    unittest.main()
