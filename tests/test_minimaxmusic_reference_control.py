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

from scripts.minimax_music3.eval_prefix_distillation import parse_guidance_scales, sustained_relock_frame
from scripts.minimax_music3.eval_reference_control import (
    is_validation_clip,
    mapped_reference_positions,
    parse_probabilities,
    select_pair_path,
    sustained_relock_latency,
)
from scripts.minimax_music3.train_control_lora import aligned_reference_hint, collate_control_samples, load_clip_ids_csv
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
    MiniMaxMusic3ControlLoRAAdapter,
    MiniMaxMusic3ReferenceControlAdapter,
    ReferenceConditionedDecoderLayer,
    ReferenceControlConfig,
    aligned_reference_mask,
    create_qwen_lokr_adapter,
    prefix_adapter_checkpoint_filename,
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

    def test_semantic_head_weight_changes_code_loss_average(self):
        semantic = torch.tensor(2.0)
        depth = [torch.tensor(1.0), torch.tensor(1.0)]
        self.assertEqual(weighted_head_mean(semantic, depth, 2.0).item(), 1.5)

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
        torch.testing.assert_close(batch["key_positions"][0], torch.tensor([0, 1, 2, 3, 3, 3]).float())

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
