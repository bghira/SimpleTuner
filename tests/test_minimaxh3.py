import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from accelerate.utils.operations import convert_to_fp32
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from simpletuner.helpers.acceleration import AccelerationBackend
from simpletuner.helpers.models.common import PipelineTypes, TextEmbedCacheKey
from simpletuner.helpers.models.ideogram.quantized_loading import Fp8Linear
from simpletuner.helpers.models.minimaxh3.activations import MiniMaxH3FeedForward
from simpletuner.helpers.models.minimaxh3.autoencoder import AutoencoderKLMiniMaxH3
from simpletuner.helpers.models.minimaxh3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
from simpletuner.helpers.models.minimaxh3.before_encoder import MiniMaxH3SetupStep
from simpletuner.helpers.models.minimaxh3.denoise import _denoiser_inputs, _predict_guided_velocity
from simpletuner.helpers.models.minimaxh3.encoders import MiniMaxH3TextEncoderStep
from simpletuner.helpers.models.minimaxh3.model import MiniMaxH3
from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
    MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY,
    MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY,
    MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY,
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
    _convert_minimax_h3_comfy_lora_to_diffusers,
    _convert_minimax_h3_diffusers_lora_to_comfyui,
    _convert_minimax_h3_diffusers_swiglu_lora_layout,
)
from simpletuner.helpers.models.minimaxh3.packing import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    build_packed_sequence,
    build_row_timestep_intervals,
    build_row_timesteps,
    video_latent_num_frames,
)
from simpletuner.helpers.models.minimaxh3.packing_ref2va import build_ref2va_presentation
from simpletuner.helpers.models.minimaxh3.sparse_attention import (
    MiniMaxH3SparseAttentionConfig,
    MiniMaxH3SparseAttentionLayout,
    _build_reordered_layout,
    _reorder_qkv,
    _restore_output,
    minimax_h3_sparse_attention,
    parse_h3_sparse_block_shape,
)
from simpletuner.helpers.models.minimaxh3.transformer import (
    H3_REFERENCE_MODE,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
    _convert_minimax_h3_native_swiglu_scale_to_diffusers,
    _convert_minimax_h3_native_swiglu_to_diffusers,
    _gather_h3_context_parallel_output,
    _pad_h3_context_parallel_layout,
    resolve_h3_reference_mode,
)
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import _validation_negative_prompt_record, prepare_validation_prompt_list


def tiny_h3_transformer(num_layers: int = 2, **kwargs) -> MiniMaxH3Transformer3DModel:
    config = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "hidden_size": 16,
        "num_layers": num_layers,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "in_channels": 2,
        "audio_in_channels": 3,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "freq_dim": 8,
        "time_embed_hidden_dim": 16,
        "time_embed_dim": 16,
        "rope_freq_dim": 1,
    }
    config.update(kwargs)
    return MiniMaxH3Transformer3DModel(**config)


def tiny_h3_vae(**kwargs) -> AutoencoderKLMiniMaxH3:
    config = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 2,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "spatial_downsample_factors": (2,),
        "temporal_downsample_factors": (1,),
        "norm_num_groups": 1,
        "decoder_num_layers": 1,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_num_register_tokens": 2,
        "decoder_ffn_mult": 2,
        "latents_mean": (0.25, -0.5),
        "latents_std": (1.25, 2.0),
    }
    config.update(kwargs)
    return AutoencoderKLMiniMaxH3(**config)


class H3ContextParallelOutputGatherTests(unittest.TestCase):
    def test_uses_legacy_all_gather_and_selects_local_gradient_shard(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).requires_grad_()
        expected_group = object()
        mesh = Mock()
        mesh.get_group.return_value = expected_group
        context_config = SimpleNamespace(_flattened_mesh=mesh)

        def gather(shards, local_tensor, group=None):
            self.assertIs(group, expected_group)
            shards[0].copy_(local_tensor)
            shards[1].copy_(local_tensor + 10)

        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.all_gather", side_effect=gather) as all_gather,
        ):
            gathered = _gather_h3_context_parallel_output(tensor, context_config)
            gathered.sum().backward()

        all_gather.assert_called_once()
        self.assertTrue(torch.equal(gathered[:, :2], tensor.detach()))
        self.assertTrue(torch.equal(gathered[:, 2:], tensor.detach() + 10))
        self.assertTrue(torch.equal(tensor.grad, torch.ones_like(tensor)))


def tiny_inputs(batch_size: int = 1):
    text_tags = torch.full((5,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    layout = build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )
    timestep, timestep_indices = build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=0.999,
    )
    return {
        "hidden_states": torch.randn(batch_size, 2, 8),
        "audio_hidden_states": torch.randn(batch_size, 4, 3),
        "encoder_hidden_states": torch.randn(batch_size, 5, 6),
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": layout.token_tags,
        "position_ids": layout.position_ids,
        "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }


class TestMiniMaxH3RotaryPosEmbed(unittest.TestCase):
    def test_moves_inv_freq_to_position_device(self):
        rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=2)
        position_ids = torch.empty((4, 3), dtype=torch.long, device="meta")

        cos, sin = rope(position_ids)

        self.assertEqual(cos.device.type, "meta")
        self.assertEqual(sin.device.type, "meta")
        self.assertEqual(cos.shape, (4, 12))
        self.assertEqual(sin.shape, (4, 12))

    def test_supports_per_sample_position_ids(self):
        rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=2)
        position_ids = torch.arange(24, dtype=torch.float32).view(2, 4, 3)

        cos, sin = rope(position_ids)

        self.assertEqual(cos.shape, (2, 4, 12))
        self.assertEqual(sin.shape, (2, 4, 12))
        expected_cos, expected_sin = rope(position_ids[1])
        self.assertTrue(torch.equal(cos[1], expected_cos))
        self.assertTrue(torch.equal(sin[1], expected_sin))


class TestMiniMaxH3ContextParallelLayout(unittest.TestCase):
    def test_pads_odd_layout_to_context_parallel_degree(self):
        position_ids = torch.arange(15, dtype=torch.long).view(5, 3)
        token_tags = torch.tensor([1, 1, 0, 0, 2], dtype=torch.long)
        timestep_indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

        padded_positions, padded_tags, padded_timesteps = _pad_h3_context_parallel_layout(
            position_ids,
            token_tags,
            timestep_indices,
            degree=2,
        )

        self.assertEqual(padded_positions.shape, (6, 3))
        self.assertTrue(torch.equal(padded_positions[:5], position_ids))
        self.assertEqual(padded_positions[-1].tolist(), [0, 0, 0])
        self.assertEqual(padded_tags.tolist(), [1, 1, 0, 0, 2, -1])
        self.assertEqual(padded_timesteps.tolist(), [0, 0, 1, 1, 2, 0])

    def test_leaves_divisible_layout_storage_unchanged(self):
        position_ids = torch.zeros((6, 3), dtype=torch.long)
        token_tags = torch.zeros(6, dtype=torch.long)
        timestep_indices = torch.zeros(6, dtype=torch.long)

        result = _pad_h3_context_parallel_layout(position_ids, token_tags, timestep_indices, degree=2)

        self.assertIs(result[0], position_ids)
        self.assertIs(result[1], token_tags)
        self.assertIs(result[2], timestep_indices)

    def test_rejects_non_positive_degree(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _pad_h3_context_parallel_layout(
                torch.zeros((1, 3), dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                degree=0,
            )

    def test_padding_rows_do_not_change_live_outputs(self):
        model = tiny_h3_transformer(num_layers=1).eval()
        inputs = tiny_inputs()
        padded_inputs = dict(inputs)
        (
            padded_inputs["position_ids"],
            padded_inputs["token_tags"],
            padded_inputs["timestep_indices"],
        ) = _pad_h3_context_parallel_layout(
            inputs["position_ids"],
            inputs["token_tags"],
            inputs["timestep_indices"],
            degree=inputs["position_ids"].shape[0] + 1,
        )

        with torch.no_grad():
            base = model(**inputs)
            padded = model(**padded_inputs)

        self.assertTrue(torch.allclose(base.sample, padded.sample, atol=1e-6))
        self.assertTrue(torch.allclose(base.audio_sample, padded.audio_sample, atol=1e-6))


class TestMiniMaxH3SparseAttention(unittest.TestCase):
    def test_denoiser_declares_sparse_lattice_geometry(self):
        input_names = {input_param.name for input_param in _denoiser_inputs()}

        self.assertTrue({"num_latent_frames", "latent_height", "latent_width"}.issubset(input_names))

    def test_parses_3d_128_token_block_shapes(self):
        self.assertEqual(parse_h3_sparse_block_shape("1x8x16"), (1, 8, 16))
        self.assertEqual(parse_h3_sparse_block_shape([2, 8, 8]), (2, 8, 8))

        with self.assertRaisesRegex(ValueError, "exactly 128"):
            parse_h3_sparse_block_shape("1,8,8")

    def test_reorder_round_trip_preserves_non_divisible_lattice(self):
        sparse_layout = MiniMaxH3SparseAttentionLayout(target_start=7, target_shape=(2, 9, 17))
        reordered_layout = _build_reordered_layout(sparse_layout, (1, 8, 16), torch.device("cpu"))
        value = torch.arange(2 * 3 * (7 + 2 * 9 * 17) * 4, dtype=torch.float32).reshape(2, 3, 7 + 2 * 9 * 17, 4)

        restored = _restore_output(_reorder_qkv(value, reordered_layout), reordered_layout)

        self.assertTrue(torch.equal(restored, value))

    def test_reorder_round_trip_preserves_context_parallel_tail_padding(self):
        sparse_layout = MiniMaxH3SparseAttentionLayout(
            target_start=7,
            target_shape=(2, 9, 17),
            trailing_padding=5,
        )
        reordered_layout = _build_reordered_layout(sparse_layout, (1, 8, 16), torch.device("cpu"))
        sequence_length = 7 + 2 * 9 * 17 + 5
        value = torch.arange(2 * 3 * sequence_length * 4, dtype=torch.float32).reshape(2, 3, sequence_length, 4)

        restored = _restore_output(_reorder_qkv(value, reordered_layout), reordered_layout)

        self.assertTrue(torch.equal(restored, value))

    def test_configuration_propagates_to_transformer_layers(self):
        model = tiny_h3_transformer(num_layers=3)
        config = model.configure_h3_sparse_attention(
            mode="moba",
            block_shape="4,4,8",
            video_kv_fraction=0.25,
            share_across_heads=True,
            start_layer=1,
        )

        self.assertEqual(config, MiniMaxH3SparseAttentionConfig("moba3d", (4, 4, 8), 0.25, True, 1))
        self.assertTrue(all(block.attn._h3_sparse_attention_config is config for block in model.transformer_blocks))
        self.assertEqual(
            [block.attn._h3_sparse_layer_index for block in model.transformer_blocks],
            [0, 1, 2],
        )

    def test_sparse_forward_requires_target_video_shape(self):
        model = tiny_h3_transformer(num_layers=1)
        model.configure_h3_sparse_attention(mode="moba3d")

        with self.assertRaisesRegex(ValueError, "video_hidden_shape"):
            model(**tiny_inputs())

    def test_context_parallel_replaces_an_incompatible_dense_backend(self):
        model = tiny_h3_transformer(num_layers=1)
        model.transformer_blocks[0].attn.processor._attention_backend = "_native_efficient"

        with (
            patch.object(model, "set_attention_backend") as set_backend,
            patch(
                "diffusers.models.modeling_utils.ModelMixin.enable_parallelism",
                return_value="enabled",
            ) as enable_parallelism,
        ):
            result = model.enable_parallelism(config=SimpleNamespace(ring_degree=1, ulysses_degree=2))

        self.assertEqual(result, "enabled")
        set_backend.assert_called_once_with("native")
        enable_parallelism.assert_called_once()

    def test_context_parallel_replaces_mask_incompatible_flash_backend(self):
        model = tiny_h3_transformer(num_layers=1)
        model.transformer_blocks[0].attn.processor._attention_backend = "_native_flash"

        with (
            patch.object(model, "set_attention_backend") as set_backend,
            patch(
                "diffusers.models.modeling_utils.ModelMixin.enable_parallelism",
                return_value="enabled",
            ),
        ):
            model.enable_parallelism(config=SimpleNamespace(ring_degree=1, ulysses_degree=2))

        set_backend.assert_called_once_with("native")

    def test_context_parallel_preserves_mask_capable_backend(self):
        for backend in ("native", "_native_cudnn"):
            with self.subTest(backend=backend):
                model = tiny_h3_transformer(num_layers=1)
                model.transformer_blocks[0].attn.processor._attention_backend = backend

                with (
                    patch.object(model, "set_attention_backend") as set_backend,
                    patch("diffusers.models.modeling_utils.ModelMixin.enable_parallelism"),
                ):
                    model.enable_parallelism(config=SimpleNamespace(ring_degree=1, ulysses_degree=2))

                set_backend.assert_not_called()

    def test_context_parallel_rejects_ring_strategy(self):
        model = tiny_h3_transformer(num_layers=1)

        with (
            patch("diffusers.models.modeling_utils.ModelMixin.enable_parallelism"),
            self.assertRaisesRegex(ValueError, 'context_parallel_strategy="alltoall"'),
        ):
            model.enable_parallelism(config=SimpleNamespace(ring_degree=2, ulysses_degree=1))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA FlexAttention")
    def test_full_video_budget_matches_dense_forward_and_backward(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        layout = MiniMaxH3SparseAttentionLayout(target_start=35, target_shape=(2, 9, 17))
        config = MiniMaxH3SparseAttentionConfig(
            mode="moba3d",
            block_shape=(1, 8, 16),
            video_kv_fraction=1.0,
        )
        tensors = [
            torch.randn(
                1,
                35 + 2 * 9 * 17,
                2,
                64,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        sparse = minimax_h3_sparse_attention(*tensors, layout=layout, config=config)
        dense = torch.nn.functional.scaled_dot_product_attention(
            *(tensor.permute(0, 2, 1, 3) for tensor in tensors)
        ).permute(0, 2, 1, 3)

        self.assertTrue(torch.allclose(sparse, dense, atol=3e-2, rtol=3e-2))
        upstream = torch.randn_like(sparse)
        sparse_grads = torch.autograd.grad(sparse, tensors, upstream, retain_graph=True)
        dense_grads = torch.autograd.grad(dense, tensors, upstream)
        for sparse_grad, dense_grad in zip(sparse_grads, dense_grads):
            self.assertTrue(torch.allclose(sparse_grad, dense_grad, atol=4e-2, rtol=4e-2))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA FlexAttention")
    def test_sparse_video_route_compiles_and_backpropagates(self):
        torch.manual_seed(11)
        device = torch.device("cuda")
        layout = MiniMaxH3SparseAttentionLayout(target_start=128, target_shape=(2, 8, 16))
        config = MiniMaxH3SparseAttentionConfig(
            mode="moba3d",
            block_shape=(1, 8, 16),
            video_kv_fraction=0.5,
        )

        def sparse_forward(query, key, value):
            return minimax_h3_sparse_attention(query, key, value, layout=layout, config=config)

        compiled_forward = torch.compile(sparse_forward, fullgraph=False, dynamic=False)
        tensors = [torch.randn(1, 384, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        output = compiled_forward(*tensors)
        output.float().square().mean().backward()

        self.assertEqual(output.shape, tensors[0].shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in tensors))

    def test_row_timestep_intervals_keep_video_and_audio_endpoints_distinct(self):
        text_tags = torch.full((2,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        layout = build_packed_sequence(
            text_token_tags=text_tags,
            num_latent_frames=1,
            latent_height=2,
            latent_width=2,
            num_audio_latents=1,
            patch_size=(1, 2, 2),
        )

        timestep, r_timestep, timestep_indices = build_row_timestep_intervals(
            layout,
            video_timestep=0.0,
            audio_timestep=0.0,
            condition_video_timestep=0.999,
            condition_audio_timestep=0.999,
            video_r_timestep=0.1,
            audio_r_timestep=0.25,
        )

        row_timesteps = timestep[timestep_indices]
        row_r_timesteps = r_timestep[timestep_indices]
        self.assertTrue(torch.all(row_r_timesteps[layout.video_indices] == 0.1))
        self.assertTrue(torch.all(row_r_timesteps[layout.audio_indices] == 0.25))
        self.assertTrue(torch.all(row_timesteps == 0.0))


def tiny_inputs_with_reference(batch_size: int = 1):
    text_tags = torch.full((5,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    layout = build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first",),
    )
    timestep, timestep_indices = build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=0.999,
    )
    return (
        {
            "hidden_states": torch.randn(batch_size, layout.video_indices.shape[0], 8),
            "audio_hidden_states": torch.randn(batch_size, layout.audio_indices.shape[0], 3),
            "encoder_hidden_states": torch.randn(batch_size, text_tags.shape[0], 6),
            "timestep": timestep,
            "timestep_indices": timestep_indices,
            "token_tags": layout.token_tags,
            "position_ids": layout.position_ids,
            "video_indices": layout.video_indices,
            "audio_indices": layout.audio_indices,
            "text_indices": layout.text_indices,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_condition_audio_rows": layout.num_condition_audio_rows,
        },
        layout,
    )


def tiny_block_state_for_guidance():
    positive_tags = torch.full((5,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    negative_tags = torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    positive_layout = build_packed_sequence(
        positive_tags,
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )
    negative_layout = build_packed_sequence(
        negative_tags,
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )
    positive_plan = [
        build_row_timesteps(positive_layout, 0.25, 0.5, 0.999, 1.0),
    ]
    negative_plan = [
        build_row_timesteps(negative_layout, 0.25, 0.5, 0.999, 1.0),
    ]
    return SimpleNamespace(
        latents=torch.zeros(positive_layout.video_indices.shape[0], 8),
        audio_latents=torch.zeros(positive_layout.audio_indices.shape[0], 3),
        prompt_embeds=torch.full((1, positive_tags.shape[0], 6), 2.0),
        negative_prompt_embeds=torch.full((1, negative_tags.shape[0], 6), 1.0),
        row_timestep_plan=positive_plan,
        negative_row_timestep_plan=negative_plan,
        token_tags=positive_layout.token_tags,
        negative_token_tags=negative_layout.token_tags,
        position_ids=positive_layout.position_ids,
        negative_position_ids=negative_layout.position_ids,
        video_indices=positive_layout.video_indices,
        negative_video_indices=negative_layout.video_indices,
        audio_indices=positive_layout.audio_indices,
        negative_audio_indices=negative_layout.audio_indices,
        text_indices=positive_layout.text_indices,
        negative_text_indices=negative_layout.text_indices,
        attention_kwargs=None,
        guidance_scale=3.0,
        use_cfg_zero_star=False,
        guidance_rescale=None,
        skip_guidance_layers=[1],
        skip_layer_guidance_scale=2.8,
        skip_layer_guidance_start=-1.0,
        skip_layer_guidance_stop=2.0,
        no_cfg_until_timestep=0,
        cfg_end_timestep=None,
    )


def comfy_quant_metadata_tensor(metadata: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(metadata).encode("utf-8")), dtype=torch.uint8)


def comfy_head_interleaved_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    inner_dim = q.shape[0]
    heads = inner_dim // head_dim
    return (
        torch.stack(
            [
                q.reshape(heads, head_dim, *q.shape[1:]),
                k.reshape(heads, head_dim, *k.shape[1:]),
                v.reshape(heads, head_dim, *v.shape[1:]),
            ],
            dim=1,
        )
        .reshape(inner_dim * 3, *q.shape[1:])
        .contiguous()
    )


class FakeH3Transformer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        skip_layers=None,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        minimax_h3_reference_mode="vanilla",
        return_dict=False,
    ):
        self.calls.append(
            {
                "text_rows": text_indices.shape[0],
                "sequence_rows": position_ids.shape[0],
                "skip_layers": skip_layers,
                "num_condition_video_rows": num_condition_video_rows,
                "num_condition_audio_rows": num_condition_audio_rows,
                "minimax_h3_reference_mode": minimax_h3_reference_mode,
            }
        )
        value = float(encoder_hidden_states.mean())
        if skip_layers:
            value -= 0.25
        video = torch.full((1, video_indices.shape[0], hidden_states.shape[-1]), value)
        audio = torch.full((1, audio_indices.shape[0], audio_hidden_states.shape[-1]), value + 10.0)
        return video, audio


class FakeLoraTarget:
    def __init__(self):
        self.calls = []

    def load_lora_adapter(self, state_dict, **kwargs):
        self.calls.append((state_dict, kwargs))


class FakeVAE:
    device = torch.device("cpu")

    def __init__(self):
        self.enable_tiling_calls = []
        self.disable_tiling_calls = 0
        self.temporal_chunking_enabled = False
        self.slicing_enabled = False

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def enable_tiling(self, **kwargs):
        self.enable_tiling_calls.append(kwargs)

    def disable_tiling(self):
        self.disable_tiling_calls += 1

    def enable_slicing(self):
        self.slicing_enabled = True

    def disable_slicing(self):
        self.slicing_enabled = False

    def enable_temporal_chunking(self):
        self.temporal_chunking_enabled = True


class FakeH3Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + index for index, _ in enumerate(str(text).split())]}

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": 10,
            "<|image_pad|>": 11,
            "<|vision_end|>": 12,
        }[token]


class FakeH3Processor:
    class ImageProcessor:
        merge_size = 1

        def __init__(self):
            self.calls = []

        def __call__(self, images, return_tensors):
            self.calls.append((images, return_tensors))
            return {
                "pixel_values": torch.ones(len(images), 2),
                "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
            }

    def __init__(self):
        self.image_processor = self.ImageProcessor()

    def create_mm_token_type_ids(self, token_batches):
        return [[0] * len(token_ids) for token_ids in token_batches]


class FakeH3TextEncoder:
    dtype = torch.float32
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=MINIMAX_H3_TEXT_ENCODER_LAYER + 1))

    def __init__(self):
        self.model = self
        self.last_input_ids = None
        self.last_attention_mask = None

    def __call__(self, **kwargs):
        self.last_input_ids = kwargs["input_ids"].detach().cpu()
        self.last_attention_mask = kwargs["attention_mask"].detach().cpu()
        batch_size, sequence_length = self.last_input_ids.shape
        hidden_states = [None] * (MINIMAX_H3_TEXT_ENCODER_LAYER + 1)
        hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER] = torch.ones(batch_size, sequence_length, 2)
        return SimpleNamespace(hidden_states=hidden_states)


class FakeMusubiManager:
    def __init__(self, managed_block_idx=1):
        self.managed_block_idx = managed_block_idx
        self.calls = []

    def activate(self, blocks, compute_device, grad_enabled):
        self.blocks = list(blocks)
        self.calls.append(("activate", len(self.blocks), str(compute_device), grad_enabled))
        return True

    def is_managed_block(self, block_idx):
        return block_idx == self.managed_block_idx

    def stream_in(self, block, compute_device):
        self.calls.append(("stream_in", id(block), str(compute_device)))

    def stream_out(self, block):
        self.calls.append(("stream_out", id(block)))


class FakePosterior:
    def mode(self):
        return torch.ones(1, 24, 2, 2, 2)

    def sample(self):
        raise AssertionError("MiniMax-H3 video cache should use posterior mode.")


class FakeKeyframePosterior:
    def mode(self):
        return torch.ones(1, 24, 1, 2, 2)

    def sample(self, generator=None):
        raise AssertionError("MiniMax-H3 first-frame conditioning cache should use posterior mode.")


class MiniMaxH3Tests(unittest.TestCase):
    def test_registry_metadata_resolves(self):
        model_cls = ModelRegistry.get("minimaxh3")
        self.assertEqual(model_cls.NAME, "MiniMax H3")
        self.assertIn("convrot-int8", model_cls.get_flavour_choices())
        resolved = model_cls.get_real_class() if hasattr(model_cls, "get_real_class") else model_cls
        self.assertIs(resolved, MiniMaxH3)

    def test_model_config_path_keeps_local_components_with_single_file_transformer(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.config = SimpleNamespace(
            model_family="minimaxh3",
            pretrained_model_name_or_path="/src/model",
            pretrained_transformer_model_name_or_path="/src/weights/transformer.safetensors",
        )

        self.assertEqual(model._model_config_path(), "/src/model")

    def test_image_mode_keeps_single_frame_geometry(self):
        self.assertEqual(MiniMaxH3.adjust_video_frames(1), 1)
        self.assertEqual(align_num_frames(1), 1)
        self.assertEqual(video_latent_num_frames(1), 1)
        self.assertEqual(MiniMaxH3._video_frames_from_latent_frames(1), 1)
        MiniMaxH3SetupStep._check_inputs(SimpleNamespace(height=1024, width=1024, num_frames=1))

    def test_video_mode_still_aligns_to_h3_chunk_grid(self):
        self.assertEqual(MiniMaxH3.adjust_video_frames(6), 22)
        self.assertEqual(align_num_frames(6), 22)
        self.assertEqual(video_latent_num_frames(22), 7)
        self.assertEqual(MiniMaxH3._video_frames_from_latent_frames(7), 22)

    def test_video_vae_encode_uses_single_clip_for_image_mode(self):
        vae = AutoencoderKLMiniMaxH3.__new__(AutoencoderKLMiniMaxH3)
        vae.use_slicing = False
        vae._encode_clip = Mock(return_value=torch.zeros(1, 4, 1, 2, 2))
        vae._encode = Mock()

        posterior = vae.encode(torch.zeros(1, 3, 1, 16, 16), return_dict=False)[0]

        vae._encode_clip.assert_called_once()
        vae._encode.assert_not_called()
        self.assertEqual(posterior.mode().shape, (1, 2, 1, 2, 2))

    def test_video_vae_decode_uses_single_clip_tail_for_image_mode(self):
        vae = AutoencoderKLMiniMaxH3.__new__(AutoencoderKLMiniMaxH3)
        vae.use_slicing = False
        decoded_clip = torch.arange(24, dtype=torch.float32).view(1, 3, 8, 1, 1)
        vae._decode_clip = Mock(return_value=decoded_clip)

        decoded = vae.decode(torch.zeros(1, 2, 1, 1, 1), return_dict=False)[0]

        vae._decode_clip.assert_called_once()
        self.assertEqual(vae._decode_clip.call_args.args[0].shape[2], 1)
        self.assertTrue(torch.equal(decoded, decoded_clip[:, :, -1:]))

    def test_video_vae_decoder_defaults_to_diffusers_hidden_first_order(self):
        vae = tiny_h3_vae()
        self.assertFalse(vae.config.decoder_swiglu_gate_first)
        self.assertFalse(vae.decoder.transformer_blocks[0].ff.net[0].gate_first)

    def test_i2v_like_hook_enables_first_frame_conditioning_cache(self):
        model = SimpleNamespace(config=SimpleNamespace(model_flavour="convrot-int8"))
        self.assertTrue(MiniMaxH3._is_i2v_like_flavour(model))

    def test_text_embed_cache_uses_dataset_filename_for_image_context(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)
        self.assertTrue(model.requires_text_embed_image_context())
        self.assertFalse(model.should_precompute_dropout_caption())
        self.assertFalse(model.use_text_cache_dropout_sentinel())
        self.assertTrue(model.uses_image_context_dropout_caption_cache())

    def test_text_embed_dropout_key_is_per_sample(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        self.assertEqual(
            model.text_embed_cache_key_value(prompt="caption", default_key="dataset:11.mp4", metadata={}),
            "dataset:11.mp4",
        )
        self.assertEqual(
            model.text_embed_cache_key_value(prompt="", default_key="dataset:11.mp4", metadata={}),
            "dataset:11.mp4:__caption_dropout__",
        )

    def test_text_embed_metadata_maps_first_frame_conditioning_to_png(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        conditioning_datasets = [
            {
                "id": "video_conditioning",
                "config": {
                    "instance_data_dir": "/cache/conditioning",
                    "conditioning_config": {"type": "i2v_first_frame"},
                },
            }
        ]
        with patch(
            "simpletuner.helpers.models.minimaxh3.model.StateTracker.get_conditioning_datasets",
            return_value=conditioning_datasets,
        ):
            metadata = model.text_embed_cache_metadata_for_filepath(
                init_backend={},
                image_path="11.mp4",
                prompt="caption",
                data_backend_id="video_backend",
                dataset_relative_path="nested/11.mp4",
            )

        self.assertEqual(metadata["image_paths"], ["/cache/conditioning/nested/11.png"])
        self.assertEqual(metadata["data_backend_ids"], ["video_conditioning"])
        self.assertEqual(metadata["image_path"], "/cache/conditioning/nested/11.png")
        self.assertEqual(metadata["data_backend_id"], "video_conditioning")

    def test_encode_prompts_passes_image_context_to_h3_conditioner(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model._current_prompt_contexts = [{"conditioning_pixel_values": Image.new("RGB", (8, 8), "white")}]
        model._text_encoder_components = lambda: SimpleNamespace(transformer=SimpleNamespace(dtype=torch.float32))

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.zeros(1, 3, 4),
                torch.tensor([1, 0, 1], dtype=torch.long),
            ),
        ) as encode_prompt:
            encoded = model._encode_prompts(["caption"])

        self.assertEqual(encoded["prompt_embeds"].shape, (1, 3, 4))
        self.assertEqual(encoded["text_token_tags"].tolist(), [[1, 0, 1]])
        call_kwargs = encode_prompt.call_args.kwargs
        self.assertEqual(call_kwargs["device"], torch.device("cpu"))
        self.assertEqual(call_kwargs["dtype"], torch.float32)
        self.assertEqual(len(call_kwargs["images"]), 1)
        self.assertIsInstance(call_kwargs["images"][0], Image.Image)

    def test_validation_prompt_can_encode_as_t2va_without_image_context(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model._current_prompt_contexts = [{}]
        model._current_prompt_is_validation = True
        model._text_encoder_components = lambda: SimpleNamespace(transformer=SimpleNamespace(dtype=torch.float32))

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.zeros(1, 3, 4),
                torch.tensor([1, 1, 1], dtype=torch.long),
            ),
        ) as encode_prompt:
            encoded = model._encode_prompts(["caption"])

        self.assertEqual(encoded["prompt_embeds"].shape, (1, 3, 4))
        self.assertIsNone(encode_prompt.call_args.kwargs["images"])

    def test_t2v_training_prompt_ignores_source_video_as_image_context(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model._current_prompt_contexts = [
            {
                "image_path": "/dataset/target.mp4",
                "data_backend_id": "openvid",
            }
        ]
        model._text_encoder_components = lambda: SimpleNamespace(transformer=SimpleNamespace(dtype=torch.float32))

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.zeros(1, 3, 4),
                torch.tensor([1, 1, 1], dtype=torch.long),
            ),
        ) as encode_prompt:
            model._encode_prompts(["caption"])

        self.assertIsNone(encode_prompt.call_args.kwargs["images"])

    def test_declared_reference_context_still_requires_resolvable_image(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model._current_prompt_contexts = [{"image_paths": ["/missing/reference.png"]}]
        model._text_encoder_components = lambda: SimpleNamespace(transformer=SimpleNamespace(dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "Failed to resolve MiniMax-H3 text conditioning image"):
            model._encode_prompts(["caption"])

    def test_encode_prompts_batches_multiple_h3_conditioner_requests(self):
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model._current_prompt_contexts = [
            {"conditioning_pixel_values": Image.new("RGB", (8, 8), "white")},
            {"conditioning_pixel_values": Image.new("RGB", (8, 8), "black")},
        ]
        model._text_encoder_components = lambda: SimpleNamespace(transformer=SimpleNamespace(dtype=torch.float32))
        batch_outputs = [
            (torch.zeros(1, 2, 4), torch.tensor([1, 0], dtype=torch.long)),
            (torch.ones(1, 3, 4), torch.tensor([1, 0, 1], dtype=torch.long)),
        ]

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt_batch",
            return_value=batch_outputs,
        ) as encode_prompt_batch:
            encoded = model._encode_prompts(["short", "a longer caption"])

        self.assertEqual(encoded["prompt_embeds"].shape, (2, 3, 4))
        self.assertEqual(encoded["text_token_tags"].tolist(), [[1, 0, -1], [1, 0, 1]])
        self.assertEqual(encode_prompt_batch.call_count, 1)
        self.assertEqual(encode_prompt_batch.call_args.args[1], ["short", "a longer caption"])
        self.assertEqual(len(encode_prompt_batch.call_args.kwargs["image_batches"]), 2)

    def test_text_encoder_block_accepts_precomputed_validation_embeds(self):
        step = MiniMaxH3TextEncoderStep()
        state = object()
        components = object()
        block_state = SimpleNamespace(
            prompt=None,
            prompt_embeds=torch.zeros(1, 3, 4),
            text_token_tags=torch.ones(3, dtype=torch.long),
        )
        step.get_block_state = Mock(return_value=block_state)
        step.set_block_state = Mock()

        result_components, result_state = step(components, state)

        self.assertIs(result_components, components)
        self.assertIs(result_state, state)
        step.set_block_state.assert_called_once_with(state, block_state)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_packing_moves_cpu_text_tags_to_default_cuda_device(self):
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            layout = build_packed_sequence(
                text_token_tags=torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long, device="cpu"),
                num_latent_frames=1,
                latent_height=1,
                latent_width=1,
                num_audio_latents=1,
                patch_size=(1, 1, 1),
            )
        finally:
            torch.set_default_device(previous_device)

        self.assertEqual(layout.token_tags.device.type, "cuda")
        self.assertTrue(
            torch.equal(
                layout.token_tags[layout.text_indices].cpu(),
                torch.full((3,), MINIMAX_H3_TEXT_TAG),
            )
        )

    def test_set_timesteps_allows_video_only_pipeline_without_audio_scheduler(self):
        step = MiniMaxH3SetTimestepsStep()
        timesteps = torch.tensor([1.0, 0.0])
        scheduler = SimpleNamespace(
            timesteps=None,
            set_timesteps=lambda num_steps, device: setattr(
                scheduler,
                "timesteps",
                timesteps.to(device),
            ),
        )
        components = SimpleNamespace(
            _execution_device=torch.device("cpu"),
            scheduler=scheduler,
            audio_scheduler=None,
        )
        layout = build_packed_sequence(
            text_token_tags=torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            num_latent_frames=1,
            latent_height=1,
            latent_width=1,
            num_audio_latents=0,
            patch_size=(1, 1, 1),
        )
        block_state = SimpleNamespace(num_inference_steps=2, layout=layout, negative_layout=None)
        state = object()
        step.get_block_state = Mock(return_value=block_state)
        step.set_block_state = Mock()

        result_components, result_state = step(components, state)

        self.assertIs(result_components, components)
        self.assertIs(result_state, state)
        self.assertTrue(torch.equal(block_state.audio_timesteps, block_state.timesteps))
        self.assertEqual(len(block_state.row_timestep_plan), 2)
        self.assertIsNone(block_state.negative_row_timestep_plan)
        step.set_block_state.assert_called_once_with(state, block_state)

    def test_h3_feedforward_uses_diffusers_swiglu_hidden_first_order(self):
        feed_forward = MiniMaxH3FeedForward(2, inner_dim=2, bias=False)
        with torch.no_grad():
            feed_forward.net[0].proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 0.0],
                        [0.0, 3.0],
                    ]
                )
            )
            feed_forward.net[2].weight.copy_(torch.eye(2))

        hidden_states = torch.tensor([[0.5, -1.0]])
        result = feed_forward(hidden_states)
        expected = torch.tensor([[0.5, -1.0]]) * torch.nn.functional.silu(torch.tensor([[1.0, -3.0]]))

        self.assertTrue(torch.allclose(result, expected))
        self.assertIn("net.0.proj.weight", feed_forward.state_dict())
        self.assertIn("net.2.weight", feed_forward.state_dict())

    def test_h3_feedforward_can_use_comfy_swiglu_gate_first_order(self):
        feed_forward = MiniMaxH3FeedForward(2, inner_dim=2, bias=False, gate_first=True)
        with torch.no_grad():
            feed_forward.net[0].proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 0.0],
                        [0.0, 3.0],
                    ]
                )
            )
            feed_forward.net[2].weight.copy_(torch.eye(2))

        hidden_states = torch.tensor([[0.5, -1.0]])
        result = feed_forward(hidden_states)
        expected = torch.nn.functional.silu(torch.tensor([[0.5, -1.0]])) * torch.tensor([[1.0, -3.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_native_swiglu_conversion_preserves_forward_and_backward(self):
        torch.manual_seed(23)
        hidden_states = torch.randn(2, 3, requires_grad=True)
        native_weight = torch.randn(8, 3, requires_grad=True)
        down_weight = torch.randn(3, 4, requires_grad=True)

        native_gate, native_value = torch.nn.functional.linear(hidden_states, native_weight).chunk(2, dim=-1)
        native_output = torch.nn.functional.linear(torch.nn.functional.silu(native_gate) * native_value, down_weight)
        native_grads = torch.autograd.grad(native_output.square().sum(), (hidden_states, native_weight, down_weight))

        converted_hidden = hidden_states.detach().clone().requires_grad_()
        converted_weight = _convert_minimax_h3_native_swiglu_to_diffusers(
            "blocks.0.mlp.fc1.weight",
            native_weight.detach(),
        ).requires_grad_()
        converted_down = down_weight.detach().clone().requires_grad_()
        converted_value, converted_gate = torch.nn.functional.linear(converted_hidden, converted_weight).chunk(2, dim=-1)
        converted_output = torch.nn.functional.linear(
            converted_value * torch.nn.functional.silu(converted_gate),
            converted_down,
        )
        converted_grads = torch.autograd.grad(
            converted_output.square().sum(),
            (converted_hidden, converted_weight, converted_down),
        )

        self.assertTrue(torch.equal(native_output, converted_output))
        self.assertTrue(torch.allclose(native_grads[0], converted_grads[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                native_grads[1],
                _convert_minimax_h3_native_swiglu_to_diffusers(
                    "blocks.0.mlp.fc1.weight",
                    converted_grads[1],
                ),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(native_grads[2], converted_grads[2], atol=1e-6, rtol=1e-6))

    def test_native_swiglu_conversion_reorders_quantization_scales(self):
        scale = torch.arange(8, dtype=torch.float32).view(8, 1)

        converted = _convert_minimax_h3_native_swiglu_scale_to_diffusers(
            "blocks.0.mlp.fc1.weight",
            scale,
        )

        self.assertTrue(torch.equal(converted, torch.cat((scale[4:], scale[:4]), dim=0)))
        scalar = torch.tensor(0.125)
        self.assertIs(
            _convert_minimax_h3_native_swiglu_scale_to_diffusers("blocks.0.mlp.fc1.weight", scalar),
            scalar,
        )

    def test_transformer_propagates_comfy_swiglu_gate_first_config(self):
        model = tiny_h3_transformer(swiglu_gate_first=True)

        self.assertTrue(model.config.swiglu_gate_first)
        self.assertTrue(model.token_refiner.refiner_blocks[0].ff.net[0].gate_first)
        self.assertTrue(model.transformer_blocks[0].ff.net[0].gate_first)

    def test_transformer_forward_and_ffn_checkpoint(self):
        model = tiny_h3_transformer()
        inputs = tiny_inputs()
        output = model(**inputs)
        self.assertEqual(output.sample.shape, (1, 2, 8))
        self.assertEqual(output.audio_sample.shape, (1, 4, 3))

        model.train()
        model.gradient_checkpointing = True
        model.set_gradient_checkpointing_backend("torch-ffn")
        inputs = tiny_inputs()
        for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
            inputs[key].requires_grad_(True)
        output = model(**inputs)
        loss = output.sample.square().mean() + output.audio_sample.square().mean()
        loss.backward()
        self.assertIsNotNone(inputs["hidden_states"].grad)

    def test_minimax_h3_reference_mode_config_field_is_parseable(self):
        from simpletuner.helpers.configuration.cmd_args import get_argument_parser

        parser = get_argument_parser()
        args = parser.parse_args(
            [
                "--model_family",
                "minimaxh3",
                "--output_dir",
                "/tmp/simpletuner-test",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--data_backend_config",
                "/tmp/backend.json",
                "--minimax_h3_reference_mode",
                "cached_kv",
                "--minimax_h3_target_mode",
                "av",
                "--minimax_h3_sparse_attention",
                "moba3d",
                "--minimax_h3_sparse_block_shape",
                "2,8,8",
                "--minimax_h3_sparse_video_kv_fraction",
                "0.25",
                "--minimax_h3_sparse_share_heads",
                "--minimax_h3_sparse_start_layer",
                "2",
                "--audio_flow_schedule_shift",
                "3.0",
            ]
        )

        self.assertEqual(args.minimax_h3_reference_mode, "cached_kv")
        self.assertEqual(args.minimax_h3_target_mode, "av")
        self.assertEqual(args.minimax_h3_sparse_attention, "moba3d")
        self.assertEqual(args.minimax_h3_sparse_block_shape, "2,8,8")
        self.assertEqual(args.minimax_h3_sparse_video_kv_fraction, 0.25)
        self.assertTrue(args.minimax_h3_sparse_share_heads)
        self.assertEqual(args.minimax_h3_sparse_start_layer, 2)
        self.assertEqual(args.audio_flow_schedule_shift, 3.0)
        self.assertIs(resolve_h3_reference_mode("cached-kv"), H3_REFERENCE_MODE.CachedKV)

    def test_minimax_h3_target_mode_uses_video_only_by_default(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.StateTracker.get_data_backend_config",
            return_value={},
        ):
            self.assertFalse(wrapper.uses_audio_latents_for_data_backend("video"))

    def test_minimax_h3_target_mode_can_enable_audio_from_source_backend(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")
        configs = {
            "video": {"dataset_type": "video", "h3_target_mode": "av"},
            "audio": {"dataset_type": "audio", "source_dataset_id": "video"},
        }

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.StateTracker.get_data_backend_config",
            side_effect=lambda backend_id: configs.get(backend_id, {}),
        ):
            self.assertTrue(wrapper.uses_audio_latents_for_data_backend("audio"))

    def test_minimax_h3_validation_auto_target_mode_uses_detected_audio_data(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")
        wrapper.configure_data_signals(has_audio=True)

        pipeline_kwargs = wrapper.update_pipeline_call_kwargs({})

        self.assertEqual(pipeline_kwargs["minimax_h3_target_mode"], "av")

    def test_minimax_h3_validation_auto_target_mode_uses_video_without_audio_data(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")
        wrapper.configure_data_signals(has_video=True)

        pipeline_kwargs = wrapper.update_pipeline_call_kwargs({})

        self.assertEqual(pipeline_kwargs["minimax_h3_target_mode"], "video")

    def test_minimax_h3_validation_preserves_explicit_pipeline_target_mode(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")
        wrapper.configure_data_signals(has_audio=True)

        pipeline_kwargs = wrapper.update_pipeline_call_kwargs({"minimax_h3_target_mode": "video"})

        self.assertEqual(pipeline_kwargs["minimax_h3_target_mode"], "video")

    def test_minimax_h3_validation_uses_audio_vae_sample_rate(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.audio_vae = SimpleNamespace(config=SimpleNamespace(sampling_rate=48000))

        self.assertEqual(wrapper.validation_audio_sample_rate(), 48000)

    def test_minimax_h3_validation_forwards_configured_frame_count(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            minimax_h3_target_mode="video",
            validation_num_video_frames=345,
        )

        pipeline_kwargs = wrapper.update_pipeline_call_kwargs({})

        self.assertEqual(pipeline_kwargs["num_frames"], 345)

    def test_transformer_cached_reference_mode_reuses_static_kv(self):
        model = tiny_h3_transformer(num_layers=1)
        model.eval()
        inputs, layout = tiny_inputs_with_reference()

        with torch.no_grad():
            first = model(**inputs, minimax_h3_reference_mode=H3_REFERENCE_MODE.CachedKV)
            second = model(**inputs, minimax_h3_reference_mode="cached_kv")

        stats = model.get_h3_reference_kv_stats()
        self.assertEqual(first.sample.shape, (1, layout.video_indices.shape[0], 8))
        self.assertEqual(first.audio_sample.shape, (1, layout.audio_indices.shape[0], 3))
        self.assertTrue(torch.allclose(first.sample, second.sample, atol=1e-6))
        self.assertTrue(torch.allclose(first.audio_sample, second.audio_sample, atol=1e-6))
        self.assertGreater(stats.get("hits", 0), 0)
        self.assertGreater(stats.get("post_hits", 0), 0)

    def test_transformer_musubi_streams_managed_block_back_out(self):
        model = tiny_h3_transformer(num_layers=3)
        fake_manager = FakeMusubiManager(managed_block_idx=1)
        model._musubi_block_swap = fake_manager

        output = model(**tiny_inputs())

        self.assertEqual(output.sample.shape, (1, 2, 8))
        self.assertEqual(
            fake_manager.calls,
            [
                ("activate", 3, "cpu", True),
                ("stream_in", id(model.transformer_blocks[1]), "cpu"),
                ("stream_out", id(model.transformer_blocks[1])),
            ],
        )

    def test_acceleration_presets_include_h3_ramtorch_and_musubi_block_swap(self):
        presets = MiniMaxH3.get_acceleration_presets()
        ramtorch = next(preset for preset in presets if preset.backend is AccelerationBackend.RAMTORCH)
        musubi = {preset.level: preset for preset in presets if preset.backend is AccelerationBackend.MUSUBI_BLOCK_SWAP}

        self.assertTrue(ramtorch.config["ramtorch"])
        self.assertFalse(ramtorch.config["ramtorch_disable_extensions"])
        self.assertTrue(ramtorch.config["ramtorch_text_encoder"])
        self.assertIn("transformer_blocks.0.*", ramtorch.config["ramtorch_target_modules"])
        self.assertIn("transformer_blocks.24.*", ramtorch.config["ramtorch_target_modules"])
        self.assertEqual(musubi["light"].config["musubi_blocks_to_swap"], 12)
        self.assertEqual(musubi["balanced"].config["musubi_blocks_to_swap"], 25)
        self.assertEqual(musubi["aggressive"].config["musubi_blocks_to_swap"], 37)

    def test_h3_low_vram_examples_enable_text_encoder_ramtorch(self):
        examples_root = Path(__file__).resolve().parents[1] / "simpletuner" / "examples"
        with (examples_root / "minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch" / "config.json").open() as handle:
            config_24g = json.load(handle)
        with (examples_root / "minimaxh3-fl2va-convrot-int8-32g.peft-lora" / "config.json").open() as handle:
            config_32g = json.load(handle)

        self.assertTrue(config_24g["ramtorch"])
        self.assertFalse(config_24g["ramtorch_disable_extensions"])
        self.assertTrue(config_24g["ramtorch_text_encoder"])
        self.assertEqual(config_24g["ramtorch_transformer_percent"], 100)

        self.assertTrue(config_32g["ramtorch"])
        self.assertFalse(config_32g["ramtorch_disable_extensions"])
        self.assertTrue(config_32g["ramtorch_text_encoder"])
        self.assertEqual(config_32g["ramtorch_transformer_percent"], 0)

    def test_transformer_output_supports_accelerate_fp32_conversion(self):
        output = MiniMaxH3TransformerOutput(
            sample=torch.ones(1, 2, 8, dtype=torch.bfloat16),
            audio_sample=torch.ones(1, 4, 3, dtype=torch.bfloat16),
        )

        converted = convert_to_fp32(output)

        self.assertEqual(converted.sample.dtype, torch.float32)
        self.assertEqual(converted.audio_sample.dtype, torch.float32)

    def test_transformer_hidden_state_capture(self):
        model = tiny_h3_transformer()
        inputs = tiny_inputs()
        hidden_states_buffer = {}
        output = model(
            **inputs,
            hidden_states_buffer=hidden_states_buffer,
            output_hidden_states=True,
            hidden_state_layer=1,
            video_hidden_shape=(2, 1, 1),
        )
        self.assertEqual(output.crepa_hidden_states.shape, (1, 2, 1, 16))
        self.assertEqual(hidden_states_buffer["layer_0"].shape, (1, 2, 1, 16))
        self.assertEqual(hidden_states_buffer["layer_1"].shape, (1, 2, 1, 16))

    def test_context_parallel_rejects_hidden_state_capture(self):
        model = tiny_h3_transformer()
        model._parallel_config = SimpleNamespace(context_parallel_config=object())
        inputs = tiny_inputs()
        with self.assertRaisesRegex(ValueError, "context_parallel_size"):
            model(
                **inputs,
                hidden_states_buffer={},
                output_hidden_states=True,
                video_hidden_shape=(2, 1, 1),
            )

    def test_transformer_supports_twinflow_and_flowmap_time_conditioning(self):
        inputs = tiny_inputs()
        model = tiny_h3_transformer()
        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            model(**inputs, timestep_sign=torch.tensor([-1.0]))

        model = tiny_h3_transformer(enable_time_sign_embed=True)
        base = model(**inputs).sample
        model.time_sign_embed.weight.data[1].fill_(0.05)
        signed = model(**inputs, timestep_sign=torch.tensor([-1.0])).sample
        self.assertFalse(torch.allclose(base, signed))

        model = tiny_h3_transformer()
        base = model(**inputs).sample
        model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        same_delta = model(**inputs, r_timestep=inputs["timestep"]).sample
        self.assertTrue(torch.allclose(base, same_delta, atol=1e-6))

    def test_transformer_supports_adaln_curve_table(self):
        model = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        self.assertIsNone(model.time_embedder)
        model.adaln_t_table.copy_(torch.arange(15, dtype=torch.float32).view(5, 3))

        temb = model._time_embedding(torch.tensor([0.0, 0.25, 1.0]))

        self.assertTrue(torch.equal(temb[0], model.adaln_t_table[0]))
        self.assertTrue(torch.equal(temb[1], model.adaln_t_table[1]))
        self.assertTrue(torch.equal(temb[2], model.adaln_t_table[-1]))
        with self.assertRaisesRegex(ValueError, "TwinFlow"):
            model._time_embedding(torch.tensor([0.5]), timestep_sign=torch.tensor([-1.0]))

    def test_transformer_adaln_curve_uses_positive_dataward_flowmap_interval(self):
        model = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        model.adaln_t_table.copy_(torch.arange(15, dtype=torch.float32).view(5, 3))
        model.enable_flowmap_time_conditioning(gate_value=1.0, deltatime_type="t-r")

        temb = model._time_embedding(torch.tensor([0.25]), r_timestep=torch.tensor([0.75]))

        self.assertTrue(torch.equal(temb[0], model.adaln_t_table[2]))

    def test_transformer_adaln_curve_uses_independent_trainable_delta_table(self):
        model = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        model.adaln_t_table.copy_(torch.arange(15, dtype=torch.float32).view(5, 3))
        model.enable_flowmap_time_conditioning(gate_value=1.0, deltatime_type="r")

        self.assertIsNotNone(model.delta_adaln_embedder)
        self.assertNotEqual(model.delta_adaln_embedder.weight.data_ptr(), model.adaln_t_table.data_ptr())
        with torch.no_grad():
            model.delta_adaln_embedder.weight.add_(100.0)

        base = model._time_embedding(torch.tensor([0.25]))
        flowmap = model._time_embedding(torch.tensor([0.25]), r_timestep=torch.tensor([0.5]))

        self.assertTrue(torch.equal(base[0], model.adaln_t_table[1]))
        self.assertTrue(torch.equal(flowmap[0], model.adaln_t_table[2] + 100.0))

    def test_transformer_adaln_delta_table_updates_large_interval_output(self):
        torch.manual_seed(31)
        model = tiny_h3_transformer(num_layers=1, time_embed_dim=16, adaln_curve_grid=5).train()
        with torch.no_grad():
            model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
        model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        model.requires_grad_(False)
        model.delta_adaln_embedder.weight.requires_grad_(True)
        optimizer = torch.optim.SGD([model.delta_adaln_embedder.weight], lr=0.1)
        inputs = tiny_inputs()
        inputs["timestep"] = torch.full_like(inputs["timestep"], 0.1)
        r_timestep = torch.full_like(inputs["timestep"], 0.9)

        base_before = model(**inputs).sample.detach().clone()
        flow_before = model(**inputs, r_timestep=r_timestep).sample.detach().clone()
        loss = model(**inputs, r_timestep=r_timestep).sample.square().mean()
        loss.backward()

        self.assertIsNotNone(model.delta_adaln_embedder.weight.grad)
        self.assertGreater(float(model.delta_adaln_embedder.weight.grad.norm()), 0.0)
        optimizer.step()

        base_after = model(**inputs).sample.detach()
        flow_after = model(**inputs, r_timestep=r_timestep).sample.detach()
        self.assertTrue(torch.equal(base_before, base_after))
        self.assertFalse(torch.equal(flow_before, flow_after))

    def test_anyflow_lora_targets_ffn_and_available_time_embedders(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            distillation_method="anyflow",
            distillation_config={"anyflow": {}},
            lora_type="standard",
            peft_lora_target_modules=None,
            slider_lora_target=False,
            controlnet=False,
        )
        wrapper.model = tiny_h3_transformer()
        wrapper.model.enable_flowmap_time_conditioning()
        wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)

        targets = wrapper.get_lora_target_layers()

        self.assertIn("ff.net.0.proj", targets)
        self.assertIn("ff.net.2", targets)
        self.assertIn("time_embedder.linear_1", targets)
        self.assertIn("delta_time_embedder.linear_1", targets)
        self.assertIsNone(wrapper.get_lora_save_layers())

    def test_anyflow_lora_targets_can_freeze_time_embedders(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            distillation_method="anyflow",
            distillation_config={
                "anyflow": {
                    "train_time_embedder": False,
                    "train_delta_embedder": False,
                }
            },
            lora_type="standard",
            peft_lora_target_modules=None,
            slider_lora_target=False,
            controlnet=False,
        )
        wrapper.model = tiny_h3_transformer()
        wrapper.model.enable_flowmap_time_conditioning()
        wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)

        targets = wrapper.get_lora_target_layers()

        self.assertIn("ff.net.0.proj", targets)
        self.assertIn("ff.net.2", targets)
        self.assertNotIn("time_embedder.linear_1", targets)
        self.assertNotIn("delta_time_embedder.linear_1", targets)
        self.assertIsNone(wrapper.get_lora_save_layers())

    def test_h3_anyflow_guidance_defaults_follow_guidance_distilled_base(self):
        for method, distillation_config in (
            ("anyflow", {"anyflow": {}}),
            (
                "h3_drift",
                {
                    "h3_drift": {
                        "inner_distillation_method": "anyflow",
                        "inner_distillation_config": {},
                    }
                },
            ),
        ):
            with self.subTest(method=method):
                wrapper = MiniMaxH3.__new__(MiniMaxH3)
                wrapper.config = SimpleNamespace(
                    distillation_method=method,
                    distillation_config=distillation_config,
                    framerate=MINIMAX_H3_FPS,
                    flow_schedule_shift=12.0,
                    audio_flow_schedule_shift=3.0,
                    vae_enable_tiling=True,
                    vae_enable_temporal_roll=True,
                )

                wrapper.check_user_config()

                anyflow_config = wrapper._anyflow_distillation_config()
                self.assertEqual(anyflow_config["fuse_guidance_scale"], 1.0)
                self.assertEqual(anyflow_config["real_score_guidance_scale"], 0.0)

    def test_h3_anyflow_guidance_defaults_preserve_explicit_values(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            distillation_method="anyflow",
            distillation_config={
                "anyflow": {
                    "fuse_guidance_scale": 2.0,
                    "real_score_guidance_scale": 0.5,
                }
            },
            framerate=MINIMAX_H3_FPS,
            flow_schedule_shift=12.0,
            audio_flow_schedule_shift=3.0,
            vae_enable_tiling=True,
            vae_enable_temporal_roll=True,
        )

        wrapper.check_user_config()

        anyflow_config = wrapper._anyflow_distillation_config()
        self.assertEqual(anyflow_config["fuse_guidance_scale"], 2.0)
        self.assertEqual(anyflow_config["real_score_guidance_scale"], 0.5)

    def test_anyflow_curve_checkpoint_saves_delta_table_with_adapter(self):
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            distillation_method="anyflow",
            distillation_config={"anyflow": {}},
            lora_type="standard",
            peft_lora_target_modules=None,
            slider_lora_target=False,
            controlnet=False,
        )
        wrapper.model = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        wrapper.model.enable_flowmap_time_conditioning()
        wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)
        wrapper.model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q"],
                modules_to_save=wrapper.get_lora_save_layers(),
            )
        )

        state_dict = get_peft_model_state_dict(wrapper.model)

        self.assertIn("delta_adaln_embedder.weight", state_dict)
        self.assertTrue(wrapper.model.delta_adaln_embedder.modules_to_save.default.weight.requires_grad)
        self.assertFalse(wrapper.model.delta_adaln_embedder.original_module.weight.requires_grad)
        wrapper._assert_anyflow_endpoint_parameters_trainable()

    def test_anyflow_endpoint_assertion_rejects_missing_trainable_delta(self):
        from peft import LoraConfig

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            distillation_method="anyflow",
            distillation_config={"anyflow": {"train_delta_embedder": True}},
        )
        wrapper.model = tiny_h3_transformer(time_embed_dim=16)
        wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)
        wrapper.model.enable_flowmap_time_conditioning()
        wrapper.model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]))

        with self.assertRaisesRegex(RuntimeError, "no trainable delta timestep parameters"):
            wrapper._assert_anyflow_endpoint_parameters_trainable()

    def test_flow_matching_timesteps_use_h3_dataward_convention(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        sigmas = torch.tensor([0.0, 0.25, 1.0])

        timesteps = wrapper.flow_matching_timesteps_from_sigmas(
            sigmas,
            reference_timesteps=torch.tensor([0.0, 250.0, 1000.0]),
        )

        self.assertTrue(torch.equal(timesteps, torch.tensor([1.0, 0.75, 0.0])))

    def test_transformer_adaln_curve_table_casts_lerp_weight_to_table_dtype(self):
        model = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        model.adaln_t_table.data = torch.arange(15, dtype=torch.bfloat16).view(5, 3)

        temb = model._time_embedding(torch.tensor([0.125], dtype=torch.bfloat16))

        self.assertEqual(temb.dtype, torch.bfloat16)

    def test_model_predict_packs_and_unpacks(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer(enable_time_sign_embed=True)
        wrapper.model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32, twinflow_enabled=True)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model

        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 2, 2, 2),
            "audio_noisy_latents": torch.randn(1, 2, 3, 2),
            "encoder_hidden_states": torch.randn(1, 5, 6),
            "timesteps": torch.tensor([0.25]),
            "audio_timesteps": torch.tensor([0.5]),
            "text_token_tags": torch.full((1, 5), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            "twinflow_time_sign": torch.tensor([-1.0]),
            "flowmap_r_timesteps": torch.tensor([0.25]),
            "minimax_h3_target_mode": "av",
        }
        output = wrapper.model_predict(prepared_batch)
        self.assertEqual(output["model_prediction"].shape, (1, 2, 2, 2, 2))
        self.assertEqual(output["audio_prediction"].shape, (1, 2, 3, 2))

    def test_model_predict_batches_variable_text_lengths_and_timesteps_exactly(self):
        torch.manual_seed(17)
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer(num_layers=1).eval()
        wrapper.model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model

        noisy_latents = torch.randn(2, 2, 2, 2, 2)
        encoder_hidden_states = torch.randn(2, 5, 6)
        encoder_hidden_states[0, 3:] = 0
        text_token_tags = torch.tensor(
            [
                [MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG, -1, -1],
                [MINIMAX_H3_TEXT_TAG] * 5,
            ],
            dtype=torch.long,
        )
        timesteps = torch.tensor([0.25, 0.75])
        r_timesteps = torch.tensor([0.5, 0.9])
        prepared_batch = {
            "noisy_latents": noisy_latents,
            "encoder_hidden_states": encoder_hidden_states,
            "timesteps": timesteps,
            "text_token_tags": text_token_tags,
            "flowmap_r_timesteps": r_timesteps,
            "minimax_h3_target_mode": "video",
        }

        with torch.no_grad():
            batched = wrapper.model_predict(prepared_batch)["model_prediction"]
            individual = []
            for batch_index, text_length in enumerate((3, 5)):
                single_batch = {
                    "noisy_latents": noisy_latents[batch_index : batch_index + 1],
                    "encoder_hidden_states": encoder_hidden_states[batch_index : batch_index + 1, :text_length],
                    "timesteps": timesteps[batch_index : batch_index + 1],
                    "text_token_tags": text_token_tags[batch_index : batch_index + 1, :text_length],
                    "flowmap_r_timesteps": r_timesteps[batch_index : batch_index + 1],
                    "minimax_h3_target_mode": "video",
                }
                individual.append(wrapper.model_predict(single_batch)["model_prediction"])

        self.assertTrue(torch.allclose(batched, torch.cat(individual), atol=1e-5, rtol=1e-5))

    def test_model_predict_video_target_mode_omits_audio_rows(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer()
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model

        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 5, 6),
            "timesteps": torch.tensor([0.25]),
            "text_token_tags": torch.full((1, 5), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            "minimax_h3_target_mode": "video",
        }

        output = wrapper.model_predict(prepared_batch)

        self.assertEqual(output["model_prediction"].shape, (1, 2, 2, 2, 2))
        self.assertIsNone(output["audio_prediction"])

    def test_model_predict_image_latents_ignore_audio_rows_even_when_av_requested(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer()
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model

        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 1, 2, 2),
            "audio_noisy_latents": torch.randn(1, 2, 3, 2),
            "encoder_hidden_states": torch.randn(1, 5, 6),
            "timesteps": torch.tensor([0.25]),
            "audio_timesteps": torch.tensor([0.5]),
            "text_token_tags": torch.full((1, 5), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            "minimax_h3_target_mode": "av",
        }

        output = wrapper.model_predict(prepared_batch)

        self.assertEqual(output["model_prediction"].shape, (1, 2, 1, 2, 2))
        self.assertIsNone(output["audio_prediction"])

    def test_prepare_batch_conditions_maps_audio_to_audio_flow_shift(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(
            weight_dtype=torch.float32,
            input_perturbation=0,
            input_perturbation_steps=None,
            flow_schedule_shift=12.0,
            audio_flow_schedule_shift=3.0,
            minimax_h3_target_mode="av",
        )
        wrapper._warned_missing_audio = False
        wrapper._warned_audio_disabled = False
        wrapper._audio_latent_channels = lambda: 3
        wrapper._expected_audio_latents = lambda latents: 2

        batch = {
            "latents": torch.zeros(1, 2, 2, 2, 2),
            "sigmas": torch.tensor([0.5]),
        }

        result = wrapper.prepare_batch_conditions(batch, state={"global_step": 0})

        self.assertTrue(torch.allclose(result["audio_sigmas"], torch.full((1, 1, 1, 1), 0.2)))
        self.assertTrue(torch.allclose(result["audio_timesteps"], torch.tensor([0.8])))
        self.assertTrue(torch.equal(result["audio_latent_mask"], torch.zeros(1)))

    def test_prepare_batch_conditions_av_target_mode_drops_cached_audio_for_image_latents(
        self,
    ):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(
            weight_dtype=torch.float32,
            minimax_h3_target_mode="av",
        )
        wrapper._warned_audio_disabled = False
        wrapper._warned_image_audio_disabled = False

        batch = {
            "latents": torch.zeros(1, 2, 1, 2, 2),
            "audio_latent_batch": torch.ones(1, 2, 3, 2),
            "audio_latent_mask": torch.ones(1),
        }

        result = wrapper.prepare_batch_conditions(batch, state={"global_step": 0})

        self.assertEqual(result["minimax_h3_target_mode"], "video")
        for key in (
            "audio_latent_batch",
            "audio_latents",
            "audio_latent_mask",
            "audio_noise",
            "audio_sigmas",
            "audio_timesteps",
            "audio_noisy_latents",
        ):
            self.assertNotIn(key, result)

    def test_prepare_batch_conditions_video_target_mode_drops_cached_audio(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(
            weight_dtype=torch.float32,
            minimax_h3_target_mode="video",
        )
        wrapper._warned_audio_disabled = False

        batch = {
            "latents": torch.zeros(1, 2, 2, 2, 2),
            "audio_latent_batch": torch.ones(1, 2, 3, 2),
            "audio_latent_mask": torch.ones(1),
        }

        result = wrapper.prepare_batch_conditions(batch, state={"global_step": 0})

        self.assertEqual(result["minimax_h3_target_mode"], "video")
        for key in (
            "audio_latent_batch",
            "audio_latents",
            "audio_latent_mask",
            "audio_noise",
            "audio_sigmas",
            "audio_timesteps",
            "audio_noisy_latents",
        ):
            self.assertNotIn(key, result)

    def test_prepare_batch_conditions_pins_keyframe_noise_for_repeated_predictions(
        self,
    ):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32, minimax_h3_target_mode="video")
        wrapper._warned_audio_disabled = False
        conditioning_latents = torch.zeros(1, 2, 1, 2, 2)

        result = wrapper.prepare_batch_conditions(
            {
                "latents": torch.zeros(1, 2, 2, 2, 2),
                "conditioning_latents": conditioning_latents,
            },
            state={"global_step": 0},
        )

        self.assertEqual(result["h3_conditioning_noise"].shape, conditioning_latents.shape)
        self.assertEqual(result["h3_conditioning_noise"].dtype, conditioning_latents.dtype)

    def test_loss_uses_explicit_audio_target_for_regularisation_batches(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            audio_loss_weight=1.0,
            loss_type="l2",
            scheduled_sampling_reflexflow=False,
        )
        wrapper.diff2flow_bridge = None
        prepared_batch = {
            "latents": torch.zeros(1, 2, 1, 1, 1),
            "noise": torch.zeros(1, 2, 1, 1, 1),
            "target": torch.zeros(1, 2, 1, 1, 1),
            "timesteps": torch.tensor([0.5]),
            "audio_latents": torch.zeros(1, 2, 3, 2),
            "audio_noise": torch.full((1, 2, 3, 2), 10.0),
            "audio_target": torch.full((1, 2, 3, 2), 3.0),
            "audio_latent_mask": torch.ones(1),
        }
        model_output = {
            "model_prediction": torch.zeros(1, 2, 1, 1, 1),
            "audio_prediction": torch.ones(1, 2, 3, 2),
        }

        loss, logs = wrapper.loss_with_logs(prepared_batch, model_output)

        self.assertTrue(torch.allclose(loss, torch.tensor(4.0)))
        self.assertAlmostEqual(logs["video_loss"], 0.0)
        self.assertAlmostEqual(logs["audio_loss"], 4.0)

    def test_load_vae_uses_diffusers_component_subfolder(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.vae = None
        vae = FakeVAE()
        wrapper.config = SimpleNamespace(
            pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
            pretrained_vae_model_name_or_path=None,
            revision=None,
            variant=None,
            vae_dtype=None,
            vae_enable_tiling=False,
            vae_enable_slicing=False,
            vae_enable_temporal_roll=False,
            weight_dtype=torch.float32,
        )
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.post_vae_load_setup = Mock()
        wrapper._load_audio_vae = Mock()

        with patch.object(MiniMaxH3.AUTOENCODER_CLASS, "from_pretrained", return_value=vae) as from_pretrained:
            wrapper.load_vae(move_to_device=False)

        _, kwargs = from_pretrained.call_args
        self.assertEqual(kwargs["subfolder"], "vae")
        self.assertTrue(wrapper.config.vae_enable_tiling)
        self.assertTrue(wrapper.config.vae_enable_temporal_roll)
        self.assertEqual(
            vae.enable_tiling_calls,
            [
                {
                    "tile_sample_min_height": 256,
                    "tile_sample_min_width": 256,
                    "tile_sample_min_overlap_height": 64,
                    "tile_sample_min_overlap_width": 64,
                }
            ],
        )
        self.assertEqual(vae.disable_tiling_calls, 0)
        self.assertTrue(vae.temporal_chunking_enabled)
        wrapper._load_audio_vae.assert_called_once_with(move_to_device=False)

    def test_load_vae_uses_single_file_loader_for_explicit_vae_checkpoint(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.vae = None
        vae = FakeVAE()
        wrapper.config = SimpleNamespace(
            pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
            pretrained_vae_model_name_or_path="/tmp/minimax_h3_video_vae_int8_convrot.safetensors",
            revision=None,
            variant=None,
            vae_dtype=None,
            vae_enable_tiling=False,
            vae_enable_slicing=False,
            vae_enable_temporal_roll=False,
            weight_dtype=torch.float32,
        )
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.post_vae_load_setup = Mock()
        wrapper._load_audio_vae = Mock()

        with patch.object(MiniMaxH3.AUTOENCODER_CLASS, "from_single_file", return_value=vae) as from_single_file:
            wrapper.load_vae(move_to_device=False)

        from_single_file.assert_called_once_with(
            "/tmp/minimax_h3_video_vae_int8_convrot.safetensors",
            torch_dtype=torch.float32,
            revision=None,
        )
        self.assertEqual(
            vae.enable_tiling_calls[0],
            {
                "tile_sample_min_height": 256,
                "tile_sample_min_width": 256,
                "tile_sample_min_overlap_height": 64,
                "tile_sample_min_overlap_width": 64,
            },
        )
        self.assertTrue(vae.temporal_chunking_enabled)
        wrapper._load_audio_vae.assert_called_once_with(move_to_device=False)

    def test_check_user_config_forces_h3_video_vae_reference_settings(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            model_family="minimaxh3",
            model_type="lora",
            model_flavour="fl2va",
            pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
            pretrained_transformer_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            vae_enable_tiling=False,
            vae_enable_temporal_roll=False,
            flow_schedule_shift=3.0,
            audio_flow_schedule_shift=None,
            framerate=None,
            validation_seed=None,
            seed=42,
        )
        wrapper.accelerator = SimpleNamespace()

        wrapper.check_user_config()

        self.assertTrue(wrapper.config.vae_enable_tiling)
        self.assertTrue(wrapper.config.vae_enable_temporal_roll)
        self.assertEqual(wrapper.config.flow_schedule_shift, 12.0)
        self.assertEqual(wrapper.config.audio_flow_schedule_shift, 3.0)
        self.assertEqual(wrapper.config.framerate, MINIMAX_H3_FPS)

    def test_convrot_int8_flavour_defaults_to_convrot_vae_checkpoint(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            model_flavour="convrot-int8",
            pretrained_model_name_or_path=None,
            pretrained_transformer_model_name_or_path=None,
            pretrained_transformer_subfolder="transformer",
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=None,
            audio_flow_schedule_shift=None,
        )

        wrapper.setup_model_flavour()

        self.assertIn(
            "MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors",
            wrapper.config.pretrained_transformer_model_name_or_path,
        )
        self.assertIn(
            "minimax_h3_video_vae_int8_convrot.safetensors",
            wrapper.config.pretrained_vae_model_name_or_path,
        )
        self.assertIn("minimax_h3_video_vae_int8_convrot.safetensors", wrapper.config.vae_path)

    def test_get_pipeline_registers_modular_components_after_init(self):
        class RecordingPipeline:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs
                self.updated_components = {}

            def update_components(self, **kwargs):
                self.updated_components.update(kwargs)

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(model_flavour=None, flow_schedule_shift=12.0, audio_flow_schedule_shift=3.0)
        wrapper.pipelines = {}
        wrapper.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: RecordingPipeline}
        wrapper.vae = object()
        wrapper.audio_vae = object()
        wrapper.text_encoders = [object()]
        wrapper.tokenizers = [object()]
        wrapper.model = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))
        wrapper.unwrap_model = Mock(side_effect=lambda model: model)
        wrapper._load_processor_for_pipeline = Mock(return_value=object())
        wrapper._model_config_path = Mock(return_value="/tmp/minimax-h3")
        wrapper.get_vae = Mock(return_value=wrapper.vae)
        wrapper._load_audio_vae = Mock()

        pipeline = wrapper.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

        wrapper.get_vae.assert_called_once_with()
        wrapper._load_audio_vae.assert_called_once_with(move_to_device=True)
        self.assertEqual(set(pipeline.init_kwargs), {"blocks", "pretrained_model_name_or_path"})
        self.assertIs(pipeline.updated_components["vae"], wrapper.vae)
        self.assertIs(pipeline.updated_components["audio_vae"], wrapper.audio_vae)
        self.assertIs(pipeline.updated_components["text_encoder"], wrapper.text_encoders[0])
        self.assertIs(pipeline.updated_components["tokenizer"], wrapper.tokenizers[0])
        self.assertIs(pipeline.updated_components["transformer"], wrapper.model)
        self.assertIn("scheduler", pipeline.updated_components)
        self.assertIn("audio_scheduler", pipeline.updated_components)

    def test_get_pipeline_refreshes_cached_validation_decoders(self):
        class RecordingPipeline:
            def __init__(self):
                self.updated_components = {}

            def update_components(self, **kwargs):
                self.updated_components.update(kwargs)

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(model_flavour=None)
        pipeline = RecordingPipeline()
        wrapper.pipelines = {PipelineTypes.TEXT2IMG: pipeline}
        wrapper.vae = object()
        wrapper.audio_vae = object()
        wrapper.model = object()
        wrapper.get_vae = Mock(return_value=wrapper.vae)
        wrapper._load_audio_vae = Mock()
        wrapper.unwrap_model = Mock(return_value=wrapper.model)

        result = wrapper.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

        self.assertIs(result, pipeline)
        self.assertIs(pipeline.updated_components["transformer"], wrapper.model)
        self.assertIs(pipeline.updated_components["vae"], wrapper.vae)
        self.assertIs(pipeline.updated_components["audio_vae"], wrapper.audio_vae)

    def test_unload_vae_detaches_cached_validation_decoders(self):
        class RecordingPipeline:
            def __init__(self):
                self.updated_components = {}

            def update_components(self, **kwargs):
                self.updated_components.update(kwargs)

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        pipeline = RecordingPipeline()
        wrapper.pipelines = {PipelineTypes.TEXT2IMG: pipeline}
        wrapper.vae = FakeVAE()
        wrapper.audio_vae = FakeVAE()

        wrapper.unload_vae()

        self.assertIsNone(wrapper.vae)
        self.assertIsNone(wrapper.audio_vae)
        self.assertIsNone(pipeline.updated_components["vae"])
        self.assertIsNone(pipeline.updated_components["audio_vae"])

    def test_check_user_config_defaults_to_h3_fps(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(framerate=None)

        wrapper.check_user_config()

        self.assertEqual(wrapper.config.framerate, MINIMAX_H3_FPS)

    def test_pre_vae_encode_transform_maps_common_pixel_range_to_h3_image_stats(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        sample = torch.zeros(1, 3, 1, 1, 3)
        sample[..., 0] = -1.0
        sample[..., 1] = 0.0
        sample[..., 2] = 1.0

        result = wrapper.pre_vae_encode_transform_sample(sample)

        mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(1, 3, 1, 1, 1)
        std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(1, 3, 1, 1, 1)
        expected_pixels = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 1, 3).expand(1, 3, 1, 1, 3)
        expected = (expected_pixels - mean) / std
        self.assertTrue(torch.allclose(result, expected))

    def test_video_vae_cache_uses_posterior_mode(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        vae = AutoencoderKLMiniMaxH3.__new__(AutoencoderKLMiniMaxH3)
        vae.encode = Mock(return_value=SimpleNamespace(latent_dist=FakePosterior()))

        latents = wrapper.encode_cache_batch(vae, torch.zeros(1, 3, 5, 16, 16))

        self.assertTrue(torch.equal(latents, torch.ones(1, 24, 2, 2, 2)))
        vae.encode.assert_called_once()
        self.assertTrue(vae.encode.call_args.kwargs["return_dict"])

    def test_i2v_first_frame_vae_cache_uses_spatial_keyframe_encode_and_posterior_mode(
        self,
    ):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        vae = AutoencoderKLMiniMaxH3.__new__(AutoencoderKLMiniMaxH3)
        vae.encode = Mock()
        vae._encode_clip = Mock(return_value=torch.zeros(1, 48, 1, 2, 2))
        metadata_entries = [
            {
                "filepath": "/conditioning/11.png",
                "metadata": {
                    "image_path": "/conditioning/11.png",
                    "training_sample_path": "11.mp4",
                },
            }
        ]

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.DiagonalGaussianDistribution",
            return_value=FakeKeyframePosterior(),
        ):
            latents = wrapper.encode_cache_batch(
                vae,
                torch.zeros(1, 3, 1, 16, 16),
                metadata_entries=metadata_entries,
            )

        self.assertEqual(latents.shape, (1, 24, 1, 2, 2))
        self.assertTrue(torch.equal(latents, torch.ones_like(latents)))
        vae._encode_clip.assert_called_once()
        vae.encode.assert_not_called()

    def test_encode_prompts_uses_nonempty_null_prompt(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper._current_prompt_contexts = [{"conditioning_pixel_values": Image.new("RGB", (8, 8), "white")}]
        wrapper._text_encoder_components = Mock(return_value=object())

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.ones(1, 1, 2),
                torch.tensor([MINIMAX_H3_TEXT_TAG], dtype=torch.long),
            ),
        ) as encode_prompt:
            result = wrapper._encode_prompts([""])

        self.assertEqual(encode_prompt.call_args.args[1], " ")
        self.assertFalse(encode_prompt.call_args.kwargs["null_instruction"])
        self.assertEqual(encode_prompt.call_args.kwargs["max_length"], 512)
        self.assertEqual(len(encode_prompt.call_args.kwargs["images"]), 1)
        self.assertEqual(result["prompt_embeds"].shape, (1, 1, 2))
        self.assertEqual(result["text_token_tags"].shape, (1, 1))

    def test_encode_prompts_honors_h3_tokenizer_max_length_override(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32, tokenizer_max_length=128)
        wrapper._current_prompt_contexts = [{"conditioning_pixel_values": Image.new("RGB", (8, 8), "white")}]
        wrapper._text_encoder_components = Mock(return_value=object())

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.ones(1, 1, 2),
                torch.tensor([MINIMAX_H3_TEXT_TAG], dtype=torch.long),
            ),
        ) as encode_prompt:
            wrapper._encode_prompts(["visible prompt"])

        self.assertEqual(encode_prompt.call_args.kwargs["max_length"], 128)

    def test_text_encoder_null_instruction_preserves_prompt_length_with_null_ids(self):
        text_encoder = FakeH3TextEncoder()
        components = SimpleNamespace(
            text_encoder=text_encoder,
            tokenizer=FakeH3Tokenizer(),
            processor=FakeH3Processor(),
            transformer=SimpleNamespace(dtype=torch.float32),
            _execution_device=torch.device("cpu"),
        )

        prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
            components,
            "alpha beta gamma",
            device=torch.device("cpu"),
            dtype=torch.float32,
            null_instruction=True,
        )

        self.assertTrue(torch.equal(text_encoder.last_input_ids, torch.zeros(1, 3, dtype=torch.long)))
        self.assertEqual(prompt_embeds.shape, (1, 3, 2))
        self.assertTrue(torch.equal(text_token_tags, torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)))

    def test_text_encoder_caps_caption_tokens_to_512_by_default(self):
        text_encoder = FakeH3TextEncoder()
        components = SimpleNamespace(
            text_encoder=text_encoder,
            tokenizer=FakeH3Tokenizer(),
            processor=FakeH3Processor(),
            transformer=SimpleNamespace(dtype=torch.float32),
            _execution_device=torch.device("cpu"),
        )
        prompt = " ".join(f"token{i}" for i in range(600))

        prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
            components,
            prompt,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(prompt_embeds.shape, (1, 512, 2))
        self.assertEqual(text_encoder.last_input_ids.shape, (1, 512))
        self.assertEqual(text_token_tags.shape, (512,))

    def test_text_encoder_batch_pads_qwen_input_and_returns_true_lengths(self):
        text_encoder = FakeH3TextEncoder()
        components = SimpleNamespace(
            text_encoder=text_encoder,
            tokenizer=FakeH3Tokenizer(),
            processor=FakeH3Processor(),
            transformer=SimpleNamespace(dtype=torch.float32),
            _execution_device=torch.device("cpu"),
        )

        encoded = MiniMaxH3TextEncoderStep.encode_prompt_batch(
            components,
            ["alpha", "alpha beta gamma"],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(text_encoder.last_input_ids.shape, (2, 3))
        self.assertTrue(
            torch.equal(
                text_encoder.last_attention_mask,
                torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.long),
            )
        )
        self.assertEqual(encoded[0][0].shape, (1, 1, 2))
        self.assertEqual(encoded[0][1].shape, (1,))
        self.assertEqual(encoded[1][0].shape, (1, 3, 2))
        self.assertEqual(encoded[1][1].shape, (3,))

    def test_text_encoder_batch_preprocesses_all_request_images_together(self):
        text_encoder = FakeH3TextEncoder()
        processor = FakeH3Processor()
        components = SimpleNamespace(
            text_encoder=text_encoder,
            tokenizer=FakeH3Tokenizer(),
            processor=processor,
            transformer=SimpleNamespace(dtype=torch.float32),
            _execution_device=torch.device("cpu"),
        )
        images = [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "black")]

        encoded = MiniMaxH3TextEncoderStep.encode_prompt_batch(
            components,
            ["first", "second"],
            image_batches=[[images[0]], [images[1]]],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(len(processor.image_processor.calls), 1)
        self.assertEqual(processor.image_processor.calls[0][0], images)
        self.assertEqual(len(encoded), 2)

    def test_text_encoder_max_length_zero_disables_caption_cap(self):
        text_encoder = FakeH3TextEncoder()
        components = SimpleNamespace(
            text_encoder=text_encoder,
            tokenizer=FakeH3Tokenizer(),
            processor=FakeH3Processor(),
            transformer=SimpleNamespace(dtype=torch.float32),
            _execution_device=torch.device("cpu"),
        )
        prompt = " ".join(f"token{i}" for i in range(600))

        prompt_embeds, _ = MiniMaxH3TextEncoderStep.encode_prompt(
            components,
            prompt,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_length=0,
        )

        self.assertEqual(prompt_embeds.shape, (1, 600, 2))

    def test_ref2va_prompt_cap_preserves_reference_rows(self):
        reference = SimpleNamespace(kind="image", has_audio=False)

        token_ids, token_tags = build_ref2va_presentation(
            FakeH3Tokenizer(),
            "alpha beta gamma",
            [reference],
            [2],
            [],
            max_prompt_length=1,
        )

        self.assertEqual(token_ids, [100, 101, 10, 11, 11, 12, 100])
        self.assertEqual(
            token_tags,
            [
                MINIMAX_H3_TEXT_TAG,
                MINIMAX_H3_TEXT_TAG,
                MINIMAX_H3_VIDEO_TAG,
                MINIMAX_H3_VIDEO_TAG,
                MINIMAX_H3_VIDEO_TAG,
                MINIMAX_H3_VIDEO_TAG,
                MINIMAX_H3_TEXT_TAG,
            ],
        )

    def test_ref2va_null_prompt_replaces_only_final_prompt_rows(self):
        token_ids, token_tags = build_ref2va_presentation(
            FakeH3Tokenizer(),
            "alpha beta",
            [],
            [],
            [],
            null_prompt_token_id=0,
        )

        self.assertEqual(token_ids, [0, 0])
        self.assertEqual(token_tags, [MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG])

    def test_negative_null_prompt_uses_positive_prompt_context(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper._current_prompt_contexts = [
            {
                "conditioning_pixel_values": Image.new("RGB", (8, 8), "white"),
                "positive_prompt": "wide lens action",
            }
        ]
        wrapper._text_encoder_components = Mock(return_value=object())

        with patch(
            "simpletuner.helpers.models.minimaxh3.model.MiniMaxH3TextEncoderStep.encode_prompt",
            return_value=(
                torch.ones(1, 3, 2),
                torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            ),
        ) as encode_prompt:
            result = wrapper._encode_prompts([""], is_negative_prompt=True)

        self.assertEqual(encode_prompt.call_args.args[1], "wide lens action")
        self.assertTrue(encode_prompt.call_args.kwargs["null_instruction"])
        self.assertEqual(result["prompt_embeds"].shape, (1, 3, 2))

    def test_guided_velocity_uses_negative_layout_and_skip_layers(self):
        transformer = FakeH3Transformer()
        block_state = tiny_block_state_for_guidance()

        video, audio = _predict_guided_velocity(transformer, block_state, i=0, num_steps=1)

        self.assertTrue(torch.allclose(video, torch.full_like(video, 4.7)))
        self.assertTrue(torch.allclose(audio, torch.full_like(audio, 14.7)))
        self.assertEqual([call["text_rows"] for call in transformer.calls], [5, 3, 5])
        self.assertEqual(
            transformer.calls[1]["sequence_rows"],
            block_state.negative_position_ids.shape[0],
        )
        self.assertEqual(transformer.calls[2]["skip_layers"], [1])

    def test_prepare_layout_persists_negative_branch_outputs(self):
        output_names = {output.name for output in MiniMaxH3PrepareLayoutStep().intermediate_outputs}

        self.assertTrue(
            {
                "negative_layout",
                "negative_position_ids",
                "negative_token_tags",
                "negative_video_indices",
                "negative_audio_indices",
                "negative_text_indices",
            }.issubset(output_names)
        )

    def test_prepare_layout_sets_absent_negative_branch_outputs_to_none(self):
        step = MiniMaxH3PrepareLayoutStep()
        block_state = SimpleNamespace(
            text_token_tags=torch.full((3,), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            negative_text_token_tags=None,
            num_latent_frames=1,
            latent_height=2,
            latent_width=2,
            num_audio_latents=0,
            keyframe_anchors=(),
        )
        step.get_block_state = Mock(return_value=block_state)
        step.set_block_state = Mock()
        components = SimpleNamespace(patch_size=(1, 2, 2), _execution_device=torch.device("cpu"))

        step(components, object())

        for name in (
            "negative_layout",
            "negative_position_ids",
            "negative_token_tags",
            "negative_video_indices",
            "negative_audio_indices",
            "negative_text_indices",
        ):
            self.assertIsNone(getattr(block_state, name))

    def test_guided_velocity_supports_deguidance(self):
        transformer = FakeH3Transformer()
        block_state = tiny_block_state_for_guidance()
        block_state.guidance_scale = 1.0 / 3.0
        block_state.skip_guidance_layers = []

        video, audio = _predict_guided_velocity(transformer, block_state, i=0, num_steps=1)

        self.assertTrue(torch.allclose(video, torch.full_like(video, 4.0 / 3.0)))
        self.assertTrue(torch.allclose(audio, torch.full_like(audio, 34.0 / 3.0)))
        self.assertEqual([call["text_rows"] for call in transformer.calls], [5, 3])

    def test_convert_negative_text_embed_for_pipeline_enables_real_cfg(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=4.0, validation_no_cfg_until_timestep=2)
        result = wrapper.convert_negative_text_embed_for_pipeline(
            {
                "prompt_embeds": torch.ones(1, 3, 6),
                "text_token_tags": torch.full((1, 3), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            }
        )
        self.assertEqual(result["guidance_scale_real"], 4.0)
        self.assertEqual(result["no_cfg_until_timestep"], 2)
        self.assertEqual(result["negative_text_token_tags"].shape, (1, 3))

    def test_validation_negative_prompt_is_real_cfg_only(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=1.0)

        self.assertFalse(wrapper.uses_validation_negative_prompt())
        self.assertFalse(wrapper.should_precompute_validation_negative_prompt())
        self.assertTrue(wrapper.validation_negative_prompt_requires_prompt_context())

        wrapper.config.validation_guidance_real = 2.0
        self.assertTrue(wrapper.uses_validation_negative_prompt())
        self.assertFalse(wrapper.should_precompute_validation_negative_prompt())

        wrapper.config.validation_guidance_real = 1.0 / 3.0
        self.assertTrue(wrapper.uses_validation_negative_prompt())

    def test_validation_negative_prompt_record_uses_image_context_key(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=2.0)
        args = SimpleNamespace(model_family="minimax_h3")
        image = Image.new("RGB", (8, 8), "white")

        record = _validation_negative_prompt_record(
            args,
            wrapper,
            "bad blur",
            "sample-a",
            image,
            positive_prompt="visible prompt",
        )

        prompt_hash = hashlib.md5("bad blur".encode("utf-8")).hexdigest()
        self.assertEqual(record["prompt"], "bad blur")
        self.assertEqual(record["key"], f"sample-a:__validation_negative__{prompt_hash}")
        self.assertIn("conditioning_pixel_values", record["metadata"])
        self.assertEqual(tuple(record["metadata"]["conditioning_pixel_values"].shape), (3, 8, 8))
        self.assertEqual(record["metadata"]["positive_prompt"], "visible prompt")

    def test_validation_negative_prompt_record_requires_context(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=2.0)
        args = SimpleNamespace(model_family="minimax_h3")

        with self.assertRaisesRegex(ValueError, "requires prompt or image context"):
            _validation_negative_prompt_record(args, wrapper, "bad blur", "sample-a", None)

    def test_validation_negative_prompt_record_accepts_t2v_prompt_context(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=2.0)
        args = SimpleNamespace(model_family="minimax_h3")

        record = _validation_negative_prompt_record(
            args,
            wrapper,
            "",
            "sample-a",
            None,
            positive_prompt="visible prompt",
        )

        self.assertEqual(record["metadata"], {"positive_prompt": "visible prompt"})

    def test_validation_negative_prompt_is_precomputed_for_h3(self):
        class DummyEmbedCache:
            model_type = "minimax_h3"
            text_cache_ondemand = False

            def __init__(self):
                self.compute_embeddings_for_prompts = Mock()
                self.encode_validation_negative_prompt = Mock()

        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(validation_guidance_real=2.0)
        wrapper.log_model_devices = Mock()
        args = SimpleNamespace(
            model_family="minimax_h3",
            model_flavour="base",
            controlnet=False,
            control=False,
            validation_using_datasets=False,
            validation_prompt_library=False,
            user_prompt_library=None,
            validation_prompt="visible prompt",
            validation_negative_prompt="",
            validation_disable_unconditional=True,
            data_backend_config="config.json",
        )
        embed_cache = DummyEmbedCache()

        with (
            patch(
                "simpletuner.helpers.training.validation.StateTracker.get_args",
                return_value=args,
            ),
            patch(
                "simpletuner.helpers.training.validation.StateTracker.get_validation_sample_images",
                return_value=None,
            ),
        ):
            metadata = prepare_validation_prompt_list(args, embed_cache, wrapper)

        self.assertEqual(
            [entry.prompt for entry in metadata["validation_prompts"]],
            ["visible prompt"],
        )
        wrapper.log_model_devices.assert_called_once_with()
        embed_cache.encode_validation_negative_prompt.assert_not_called()
        negative_calls = [
            call
            for call in embed_cache.compute_embeddings_for_prompts.call_args_list
            if call.kwargs.get("is_negative_prompt")
        ]
        self.assertEqual(len(negative_calls), 1)
        negative_record = negative_calls[0].args[0][0]
        self.assertEqual(negative_record["prompt"], "")
        self.assertEqual(negative_record["metadata"], {"positive_prompt": "visible prompt"})
        self.assertFalse(negative_calls[0].kwargs["load_from_cache"])

    def test_comfy_lora_conversion_maps_h3_names_and_splits_qkv(self):
        qkv_down = torch.randn(4, 16)
        qkv_up = torch.randn(48, 4)
        state_dict = {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": qkv_down,
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": qkv_up,
            "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(4.0),
            "diffusion_model.video_patch_proj.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.video_patch_proj.lora_B.weight": torch.randn(16, 4),
        }

        converted, network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            state_dict,
            target_prefix="transformer",
        )

        q_key = "transformer.transformer_blocks.0.attn.to_q.lora.down.weight"
        k_key = "transformer.transformer_blocks.0.attn.to_k.lora.down.weight"
        v_key = "transformer.transformer_blocks.0.attn.to_v.lora.down.weight"
        self.assertTrue(torch.equal(converted[q_key], qkv_down))
        self.assertTrue(torch.equal(converted[k_key], qkv_down))
        self.assertTrue(torch.equal(converted[v_key], qkv_down))
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.attn.to_q.lora.up.weight"],
                qkv_up[:16],
            )
        )
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.attn.to_k.lora.up.weight"],
                qkv_up[16:32],
            )
        )
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.attn.to_v.lora.up.weight"],
                qkv_up[32:],
            )
        )
        self.assertIn("transformer.proj_in.lora.down.weight", converted)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_q.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_k.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_v.alpha"], 4.0)

    def test_comfy_lora_export_fuses_independent_qkv_without_changing_delta(self):
        state_dict = {}
        expected_deltas = []
        for index, projection in enumerate(("to_q", "to_k", "to_v"), start=1):
            down = torch.full((2, 3), float(index))
            up = torch.full((4, 2), float(index + 3))
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            state_dict[f"{prefix}.lora_A.weight"] = down
            state_dict[f"{prefix}.lora_B.weight"] = up
            expected_deltas.append(up @ down)

        converted = _convert_minimax_h3_diffusers_lora_to_comfyui(state_dict)

        prefix = "diffusion_model.blocks.0.attn.qkv_proj"
        fused_down = converted[f"{prefix}.lora_A.weight"]
        fused_up = converted[f"{prefix}.lora_B.weight"]
        self.assertEqual(tuple(fused_down.shape), (6, 3))
        self.assertEqual(tuple(fused_up.shape), (12, 6))
        self.assertEqual(converted[f"{prefix}.alpha"].item(), 6.0)
        self.assertTrue(torch.equal(fused_up @ fused_down, torch.cat(expected_deltas, dim=0)))

    def test_comfy_lora_round_trip_restores_independent_qkv_ranks(self):
        state_dict = {}
        for index, projection in enumerate(("to_q", "to_k", "to_v"), start=1):
            down = torch.full((2, 3), float(index))
            up = torch.full((4, 2), float(index + 3))
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            state_dict[f"{prefix}.lora_A.weight"] = down
            state_dict[f"{prefix}.lora_B.weight"] = up

        comfy = _convert_minimax_h3_diffusers_lora_to_comfyui(state_dict)
        restored, alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            comfy,
            target_prefix="transformer",
        )

        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            self.assertTrue(
                torch.equal(
                    restored[f"{prefix}.lora.down.weight"],
                    state_dict[f"{prefix}.lora_A.weight"],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored[f"{prefix}.lora.up.weight"],
                    state_dict[f"{prefix}.lora_B.weight"],
                )
            )
            self.assertEqual(alphas[f"{prefix}.alpha"], 2.0)

    def test_comfy_lora_export_keeps_shared_qkv_rank_and_maps_mlp(self):
        shared_down = torch.randn(2, 3)
        state_dict = {}
        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            state_dict[f"{prefix}.lora.down.weight"] = shared_down.clone()
            state_dict[f"{prefix}.lora.up.weight"] = torch.randn(4, 2)
        hidden = torch.full((2, 2), 2.0)
        gate = torch.full((2, 2), 1.0)
        mlp_prefix = "transformer.transformer_blocks.0.ff.net.0.proj"
        state_dict[f"{mlp_prefix}.lora.down.weight"] = torch.randn(2, 3)
        state_dict[f"{mlp_prefix}.lora.up.weight"] = torch.cat((hidden, gate), dim=0)

        converted = _convert_minimax_h3_diffusers_lora_to_comfyui(state_dict)

        qkv_prefix = "diffusion_model.blocks.0.attn.qkv_proj"
        self.assertEqual(tuple(converted[f"{qkv_prefix}.lora_A.weight"].shape), (2, 3))
        self.assertEqual(tuple(converted[f"{qkv_prefix}.lora_B.weight"].shape), (12, 2))
        mlp_up = converted["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
        self.assertTrue(torch.equal(mlp_up, torch.cat((gate, hidden), dim=0)))

    def test_comfy_lora_export_maps_direct_fused_qkv_projection(self):
        prefix = "transformer.transformer_blocks.0.attn.to_qkv"
        down = torch.randn(4, 8)
        up = torch.randn(24, 4)

        converted = _convert_minimax_h3_diffusers_lora_to_comfyui(
            {
                f"{prefix}.lora_A.weight": down,
                f"{prefix}.lora_B.weight": up,
            }
        )

        native_prefix = "diffusion_model.blocks.0.attn.qkv_proj"
        self.assertTrue(torch.equal(converted[f"{native_prefix}.lora_A.weight"], down))
        self.assertTrue(torch.equal(converted[f"{native_prefix}.lora_B.weight"], up))
        self.assertEqual(converted[f"{native_prefix}.alpha"].item(), 4.0)

    def test_model_comfy_lora_save_uses_native_h3_exporter(self):
        model = object.__new__(MiniMaxH3)
        model.config = SimpleNamespace(
            controlnet=False,
            lora_format="comfyui",
            model_family="minimaxh3",
        )
        pipeline_class = model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(pipeline_class, "save_lora_weights") as save_lora_weights:
            model.save_lora_weights(
                tmpdir,
                transformer_lora_layers={"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3)},
            )
            save_function = save_lora_weights.call_args.kwargs["save_function"]
            output_path = f"{tmpdir}/h3-comfy.safetensors"
            shared_down = torch.randn(2, 3)
            weights = {}
            for projection in ("to_q", "to_k", "to_v"):
                prefix = f"transformer.transformer_blocks.0.attn.{projection}"
                weights[f"{prefix}.lora_A.weight"] = shared_down.clone()
                weights[f"{prefix}.lora_B.weight"] = torch.randn(4, 2)
            save_function(weights, output_path)
            saved = load_file(output_path)
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                adapter_metadata = json.loads(handle.metadata()["lora_adapter_metadata"])

        self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", saved)
        self.assertNotIn("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight", saved)
        self.assertTrue(adapter_metadata[MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY])

    def test_model_diffusers_lora_save_marks_hidden_first_swiglu_layout(self):
        model = object.__new__(MiniMaxH3)
        model.config = SimpleNamespace(
            controlnet=False,
            lora_format="diffusers",
            model_family="minimaxh3",
        )
        model.model = tiny_h3_transformer(swiglu_gate_first=False)
        model.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)
        pipeline_class = model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(pipeline_class, "save_lora_weights") as save_lora_weights:
            model.save_lora_weights(
                tmpdir,
                transformer_lora_layers={"transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.ones(2, 3)},
                transformer_lora_adapter_metadata={"adapter": "h3"},
            )

        adapter_metadata = save_lora_weights.call_args.kwargs["transformer_lora_adapter_metadata"]
        self.assertEqual(adapter_metadata["adapter"], "h3")
        self.assertFalse(adapter_metadata[MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY])

    def test_comfy_anyflow_table_adapter_round_trip_preserves_delta_and_flowmap_metadata(self):
        from diffusers.training_utils import _collate_lora_metadata
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        source = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        source.enable_flowmap_time_conditioning(gate_value=0.4, deltatime_type="r")
        source.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q", "to_k", "to_v"],
                modules_to_save=["delta_adaln_embedder"],
            )
        )
        with torch.no_grad():
            source.delta_adaln_embedder.modules_to_save.default.weight.add_(7.0)
        expected_delta = source.delta_adaln_embedder.modules_to_save.default.weight.detach().clone()

        wrapper = object.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            controlnet=False,
            lora_format="comfyui",
            model_family="minimaxh3",
            model_flavour="convrot-int8",
        )
        wrapper.model = source
        wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper.save_lora_weights(
                tmpdir,
                transformer_lora_layers=get_peft_model_state_dict(source),
                **_collate_lora_metadata({"transformer": source}),
            )
            output_path = f"{tmpdir}/pytorch_lora_weights.safetensors"
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                saved_keys = list(handle.keys())
                adapter_metadata = json.loads(handle.metadata()["lora_adapter_metadata"])

            target = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
            pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
            pipe._component_specs = {"transformer": None}
            pipe.transformer = target
            pipe.load_lora_weights(tmpdir, adapter_name="roundtrip")

        self.assertIn("diffusion_model.delta_adaln_embedder.weight", saved_keys)
        self.assertEqual(adapter_metadata["modules_to_save"], ["delta_adaln_embedder"])
        self.assertAlmostEqual(adapter_metadata[MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY], 0.4)
        self.assertEqual(adapter_metadata[MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY], "r")
        self.assertEqual(target.peft_config["roundtrip"].modules_to_save, ["delta_adaln_embedder"])
        self.assertEqual(target.flowmap_deltatime_type, "r")
        self.assertAlmostEqual(float(target.flowmap_delta_emb_gate.item()), 0.4)
        self.assertTrue(
            torch.equal(
                target.delta_adaln_embedder.modules_to_save.roundtrip.weight,
                expected_delta,
            )
        )

    def test_lora_loader_recovers_legacy_table_sidecar_without_metadata(self):
        source = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        source.enable_flowmap_time_conditioning()
        legacy_state = {
            "diffusion_model.delta_adaln_embedder.weight": torch.full((5, 3), 6.0),
            "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": torch.randn(2, 32),
            "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": torch.randn(16, 2),
            "diffusion_model.blocks.0.mlp.fc2.alpha": torch.tensor(2.0),
        }

        target = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe._component_specs = {"transformer": None}
        pipe.transformer = target
        pipe.lora_state_dict = lambda _path, **_kwargs: legacy_state

        pipe.load_lora_weights("unused", adapter_name="legacy")

        self.assertEqual(target.peft_config["legacy"].modules_to_save, ["delta_adaln_embedder"])
        self.assertTrue(
            torch.equal(
                target.delta_adaln_embedder.modules_to_save.legacy.weight,
                torch.full((5, 3), 6.0),
            )
        )

    def test_lora_loader_enables_frozen_flowmap_delta_from_metadata(self):
        state_dict = {
            "transformer.transformer_blocks.0.ff.net.2.lora_A.weight": torch.randn(2, 32),
            "transformer.transformer_blocks.0.ff.net.2.lora_B.weight": torch.randn(16, 2),
        }
        metadata = {
            "transformer.r": 2,
            "transformer.lora_alpha": 2,
            "transformer.target_modules": ["transformer_blocks.0.ff.net.2"],
            "transformer.rank_pattern": {},
            "transformer.alpha_pattern": {},
            f"transformer.{MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY}": 0.2,
            f"transformer.{MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY}": "r",
        }
        target = tiny_h3_transformer(time_embed_dim=3, adaln_curve_grid=5)
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe._component_specs = {"transformer": None}
        pipe.transformer = target
        pipe.lora_state_dict = lambda _path, **_kwargs: (state_dict, metadata)

        pipe.load_lora_weights("unused", adapter_name="frozen-delta")

        self.assertIsNotNone(target.delta_adaln_embedder)
        self.assertEqual(target.flowmap_deltatime_type, "r")
        self.assertAlmostEqual(float(target.flowmap_delta_emb_gate.item()), 0.2)

    def test_training_resume_imports_native_comfy_lora_keys(self):
        wrapper = object.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            controlnet=False,
            lora_format="comfyui",
            model_flavour="fl2va",
        )
        wrapper.model = tiny_h3_transformer(swiglu_gate_first=False)
        wrapper.controlnet = None
        wrapper.text_encoders = []
        wrapper.accelerator = SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper=True: model,
        )
        source_qkv = {}
        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            source_qkv[f"{prefix}.lora_A.weight"] = torch.randn(2, 8)
            source_qkv[f"{prefix}.lora_B.weight"] = torch.randn(8, 2)
        native_state_dict = _convert_minimax_h3_diffusers_lora_to_comfyui(source_qkv)
        native_state_dict.update(
            {
                "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": torch.randn(4, 8),
                "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": torch.cat(
                    (torch.full((4, 4), 1.0), torch.full((4, 4), 2.0)),
                    dim=0,
                ),
                "diffusion_model.time_embedder.proj_in.lora_A.weight": torch.randn(4, 8),
                "diffusion_model.time_embedder.proj_in.lora_B.weight": torch.randn(16, 4),
            }
        )
        pipeline_class = wrapper.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        with (
            patch.object(pipeline_class, "lora_state_dict", return_value=native_state_dict),
            patch("peft.utils.set_peft_model_state_dict") as set_peft_state,
        ):
            set_peft_state.return_value = SimpleNamespace(unexpected_keys=[])
            wrapper.load_lora_weights([wrapper.model], "unused")

        loaded = set_peft_state.call_args.args[1]
        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer_blocks.0.attn.{projection}"
            source_prefix = f"transformer.{prefix}"
            self.assertTrue(
                torch.equal(
                    loaded[f"{prefix}.lora_A.weight"],
                    source_qkv[f"{source_prefix}.lora_A.weight"],
                )
            )
            self.assertTrue(
                torch.equal(
                    loaded[f"{prefix}.lora_B.weight"],
                    source_qkv[f"{source_prefix}.lora_B.weight"],
                )
            )
        self.assertIn("transformer_blocks.0.ff.net.0.proj.lora_A.weight", loaded)
        self.assertIn("transformer_blocks.0.ff.net.0.proj.lora_B.weight", loaded)
        self.assertIn("time_embedder.linear_1.lora_A.weight", loaded)
        self.assertIn("time_embedder.linear_1.lora_B.weight", loaded)
        self.assertNotIn("blocks.0.attn.qkv_proj.lora_A.weight", loaded)

    def test_training_resume_uses_metadata_for_ambiguous_diffusers_swiglu_layout(self):
        wrapper = object.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            controlnet=False,
            lora_format="diffusers",
            model_flavour="fl2va",
            train_text_encoder=False,
        )
        wrapper.model = tiny_h3_transformer(swiglu_gate_first=False)
        wrapper.controlnet = None
        wrapper.text_encoders = []
        wrapper.accelerator = SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper=True: model,
        )
        gate = torch.full((4, 4), 1.0)
        hidden = torch.full((4, 4), 2.0)
        prefix = "transformer.transformer_blocks.0.ff.net.0.proj"
        source_state_dict = {
            f"{prefix}.lora_A.weight": torch.randn(4, 8),
            f"{prefix}.lora_B.weight": torch.cat((gate, hidden), dim=0),
        }
        pipeline_class = wrapper.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        with (
            patch.object(
                pipeline_class,
                "lora_state_dict",
                return_value=(source_state_dict, {"transformer.swiglu_gate_first": True}),
            ) as lora_state_dict,
            patch("peft.utils.set_peft_model_state_dict") as set_peft_state,
        ):
            set_peft_state.return_value = SimpleNamespace(unexpected_keys=[])
            wrapper.load_lora_weights([wrapper.model], "unused")

        lora_state_dict.assert_called_once_with("unused", return_lora_metadata=True)
        loaded = set_peft_state.call_args.args[1]
        self.assertTrue(
            torch.equal(
                loaded["transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
                torch.cat((hidden, gate), dim=0),
            )
        )

    def test_training_resume_fails_when_peft_rejects_every_lora_key(self):
        wrapper = object.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            controlnet=False,
            lora_format="comfyui",
            model_flavour="fl2va",
        )
        wrapper.model = tiny_h3_transformer(swiglu_gate_first=False)
        wrapper.controlnet = None
        wrapper.text_encoders = []
        wrapper.accelerator = SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper=True: model,
        )
        native_state_dict = {
            "diffusion_model.time_embedder.proj_in.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.time_embedder.proj_in.lora_B.weight": torch.randn(16, 4),
        }
        pipeline_class = wrapper.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        def reject_all(_model, state_dict, adapter_name):
            del adapter_name
            return SimpleNamespace(unexpected_keys=list(state_dict))

        with (
            patch.object(pipeline_class, "lora_state_dict", return_value=native_state_dict),
            patch("peft.utils.set_peft_model_state_dict", side_effect=reject_all),
            self.assertRaisesRegex(ValueError, "rejected every denoiser tensor"),
        ):
            wrapper.load_lora_weights([wrapper.model], "unused")

    def test_comfy_lora_conversion_swaps_swiglu_rows_for_hidden_first_target(self):
        fc1_down = torch.randn(4, 16)
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        state_dict = {
            "blocks.0.mlp.fc1.lora_A.weight": fc1_down,
            "blocks.0.mlp.fc1.lora_B.weight": torch.cat([gate, hidden], dim=0),
        }

        converted, _network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            state_dict,
            target_prefix="transformer",
            target_swiglu_gate_first=False,
        )

        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff.net.0.proj.lora.down.weight"],
                fc1_down,
            )
        )
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff.net.0.proj.lora.up.weight"],
                torch.cat([hidden, gate], dim=0),
            )
        )

    def test_comfy_lora_swiglu_conversion_preserves_adapter_forward(self):
        torch.manual_seed(29)
        hidden_states = torch.randn(2, 3)
        native_base = torch.randn(8, 3)
        native_down = torch.randn(2, 3)
        native_up = torch.randn(8, 2)

        native_projection = torch.nn.functional.linear(
            hidden_states,
            native_base + native_up @ native_down,
        )
        native_gate, native_value = native_projection.chunk(2, dim=-1)
        native_output = torch.nn.functional.silu(native_gate) * native_value

        converted, _network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            {
                "blocks.0.mlp.fc1.lora_A.weight": native_down,
                "blocks.0.mlp.fc1.lora_B.weight": native_up,
            },
            target_prefix="transformer",
        )
        canonical_base = _convert_minimax_h3_native_swiglu_to_diffusers(
            "blocks.0.mlp.fc1.weight",
            native_base,
        )
        canonical_down = converted["transformer.transformer_blocks.0.ff.net.0.proj.lora.down.weight"]
        canonical_up = converted["transformer.transformer_blocks.0.ff.net.0.proj.lora.up.weight"]
        canonical_projection = torch.nn.functional.linear(
            hidden_states,
            canonical_base + canonical_up @ canonical_down,
        )
        canonical_value, canonical_gate = canonical_projection.chunk(2, dim=-1)
        canonical_output = canonical_value * torch.nn.functional.silu(canonical_gate)

        self.assertTrue(torch.equal(canonical_output, native_output))

    def test_comfy_lora_conversion_preserves_swiglu_rows_for_gate_first_target(self):
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        fc1_up = torch.cat([gate, hidden], dim=0)
        state_dict = {
            "blocks.0.mlp.fc1.lora_A.weight": torch.randn(4, 16),
            "blocks.0.mlp.fc1.lora_B.weight": fc1_up,
        }

        converted, _network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            state_dict,
            target_prefix="transformer",
            target_swiglu_gate_first=True,
        )

        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff.net.0.proj.lora.up.weight"],
                fc1_up,
            )
        )

    def test_diffusers_lora_layout_conversion_swaps_only_swiglu_up_rows(self):
        down = torch.randn(2, 4)
        gate = torch.full((3, 2), 1.0)
        hidden = torch.full((3, 2), 2.0)
        attention_up = torch.randn(4, 2)
        state_dict = {
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": down,
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.cat((gate, hidden), dim=0),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": attention_up,
        }

        converted = _convert_minimax_h3_diffusers_swiglu_lora_layout(
            state_dict,
            source_gate_first=True,
            target_gate_first=False,
        )

        self.assertIs(converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight"], down)
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
                torch.cat((hidden, gate), dim=0),
            )
        )
        self.assertIs(converted["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"], attention_up)

    def test_lora_loader_detects_bare_h3_native_layout(self):
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.transformer.config = SimpleNamespace(swiglu_gate_first=False)
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "blocks.0.mlp.fc1.lora_A.weight": torch.randn(4, 16),
            "blocks.0.mlp.fc1.lora_B.weight": torch.cat([gate, hidden], dim=0),
        }

        pipe.load_lora_weights("unused", adapter_name="h3")

        loaded_state_dict, kwargs = pipe.transformer.calls[0]
        self.assertIn(
            "transformer.transformer_blocks.0.ff.net.0.proj.lora.down.weight",
            loaded_state_dict,
        )
        self.assertTrue(
            torch.equal(
                loaded_state_dict["transformer.transformer_blocks.0.ff.net.0.proj.lora.up.weight"],
                torch.cat([hidden, gate], dim=0),
            )
        )
        self.assertEqual(kwargs["adapter_name"], "h3")

    def test_lora_loader_uses_metadata_for_ambiguous_diffusers_swiglu_layout(self):
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.transformer.config = SimpleNamespace(swiglu_gate_first=False)
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        state_dict = {
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.randn(4, 16),
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.cat((gate, hidden), dim=0),
        }
        pipe.lora_state_dict = lambda _path, **_kwargs: (
            state_dict,
            {"transformer.swiglu_gate_first": True},
        )

        pipe.load_lora_weights("unused", adapter_name="h3")

        loaded_state_dict, _kwargs = pipe.transformer.calls[0]
        self.assertTrue(
            torch.equal(
                loaded_state_dict["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
                torch.cat((hidden, gate), dim=0),
            )
        )

    def test_lora_loader_leaves_unmarked_diffusers_swiglu_layout_unchanged(self):
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.transformer.config = SimpleNamespace(swiglu_gate_first=False)
        up = torch.randn(4, 2)
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.randn(2, 4),
            "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": up,
        }

        pipe.load_lora_weights("unused", adapter_name="h3")

        loaded_state_dict, _kwargs = pipe.transformer.calls[0]
        self.assertIs(
            loaded_state_dict["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
            up,
        )

    def test_init_lora_prepares_bare_h3_native_layout_and_targets(self):
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            init_lora_path = f"{tmpdir}/h3-native.safetensors"
            save_file(
                {
                    "blocks.0.mlp.fc1.lora_A.weight": torch.randn(4, 16),
                    "blocks.0.mlp.fc1.lora_B.weight": torch.cat([gate, hidden], dim=0),
                    "blocks.0.mlp.fc1.alpha": torch.tensor(2.0),
                },
                init_lora_path,
            )
            wrapper = MiniMaxH3.__new__(MiniMaxH3)
            wrapper.config = SimpleNamespace(
                init_lora=init_lora_path,
                lora_format=None,
                lora_type="standard",
                peft_lora_target_modules=None,
                model_flavour="fl2va",
                controlnet=False,
            )
            wrapper.model = tiny_h3_transformer(swiglu_gate_first=False)
            wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)
            wrapper.controlnet = None

            prepared = wrapper._load_init_lora_state_dict()
            targets = wrapper.get_lora_target_layers()

        self.assertTrue(
            torch.equal(
                prepared["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
                torch.cat([hidden, gate], dim=0),
            )
        )
        self.assertIn("transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight", prepared)
        self.assertEqual(prepared["transformer.transformer_blocks.0.ff.net.0.proj.alpha"].item(), 2.0)
        self.assertEqual(targets, ["transformer_blocks.0.ff.net.0.proj"])

    def test_init_lora_uses_metadata_for_ambiguous_diffusers_swiglu_layout(self):
        gate = torch.full((2, 4), 1.0)
        hidden = torch.full((2, 4), 2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            init_lora_path = f"{tmpdir}/h3-peft.safetensors"
            save_file(
                {
                    "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.randn(4, 16),
                    "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.cat((gate, hidden), dim=0),
                },
                init_lora_path,
                metadata={
                    "lora_adapter_metadata": json.dumps({"transformer.swiglu_gate_first": True}),
                },
            )
            wrapper = MiniMaxH3.__new__(MiniMaxH3)
            wrapper.config = SimpleNamespace(
                init_lora=init_lora_path,
                lora_format=None,
                model_flavour="fl2va",
            )
            wrapper.model = tiny_h3_transformer(swiglu_gate_first=False)
            wrapper.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)

            prepared = wrapper._load_init_lora_state_dict()

        self.assertTrue(
            torch.equal(
                prepared["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"],
                torch.cat((hidden, gate), dim=0),
            )
        )

    def test_ref2va_lora_loader_retargets_transformer_prefix(self):
        pipe = MiniMaxH3Ref2VAModularPipeline.__new__(MiniMaxH3Ref2VAModularPipeline)
        pipe.transformer_ref = FakeLoraTarget()
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "transformer.proj_in.lora.down.weight": torch.randn(4, 8),
            "transformer.proj_in.lora.up.weight": torch.randn(16, 4),
        }

        pipe.load_lora_weights("unused", adapter_name="h3")

        loaded_state_dict, kwargs = pipe.transformer_ref.calls[0]
        self.assertIn("transformer_ref.proj_in.lora.down.weight", loaded_state_dict)
        self.assertIn("transformer_ref.proj_in.lora.up.weight", loaded_state_dict)
        self.assertEqual(kwargs["adapter_name"], "h3")
        self.assertEqual(kwargs["prefix"], "transformer_ref")

    def test_lora_loader_synthesizes_rank_alphas_for_mixed_rank_without_alpha(self):
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "transformer.transformer_blocks.0.attn.to_q.lora.down.weight": torch.randn(64, 8),
            "transformer.transformer_blocks.0.attn.to_q.lora.up.weight": torch.randn(16, 64),
            "transformer.proj_in.lora.down.weight": torch.randn(16, 8),
            "transformer.proj_in.lora.up.weight": torch.randn(16, 16),
        }

        pipe.load_lora_weights("unused", adapter_name="h3")

        _, kwargs = pipe.transformer.calls[0]
        self.assertEqual(
            kwargs["network_alphas"]["transformer.transformer_blocks.0.attn.to_q.alpha"],
            64.0,
        )
        self.assertEqual(kwargs["network_alphas"]["transformer.proj_in.alpha"], 16.0)

    def test_lora_loader_preserves_explicit_state_dict_alphas(self):
        pipe = MiniMaxH3ModularPipeline.__new__(MiniMaxH3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "transformer.transformer_blocks.0.attn.to_q.lora.down.weight": torch.randn(64, 8),
            "transformer.transformer_blocks.0.attn.to_q.lora.up.weight": torch.randn(16, 64),
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(32.0),
            "transformer.proj_in.lora.down.weight": torch.randn(16, 8),
            "transformer.proj_in.lora.up.weight": torch.randn(16, 16),
            "transformer.proj_in.alpha": torch.tensor(8.0),
        }

        pipe.load_lora_weights("unused", adapter_name="h3")

        loaded_state_dict, kwargs = pipe.transformer.calls[0]
        self.assertNotIn("transformer.transformer_blocks.0.attn.to_q.alpha", loaded_state_dict)
        self.assertNotIn("transformer.proj_in.alpha", loaded_state_dict)
        self.assertEqual(
            kwargs["network_alphas"]["transformer.transformer_blocks.0.attn.to_q.alpha"],
            32.0,
        )
        self.assertEqual(kwargs["network_alphas"]["transformer.proj_in.alpha"], 8.0)

    def test_lora_saver_packs_transformer_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(MiniMaxH3ModularPipeline, "write_lora_layers") as write_lora_layers:
                MiniMaxH3ModularPipeline.save_lora_weights(
                    save_directory=tmpdir,
                    transformer_lora_layers={"proj_in.lora.down.weight": torch.ones(1, 1)},
                    transformer_lora_adapter_metadata={"adapter": "h3"},
                )

        kwargs = write_lora_layers.call_args.kwargs
        self.assertIn("transformer.proj_in.lora.down.weight", kwargs["state_dict"])
        self.assertEqual(kwargs["lora_adapter_metadata"], {"transformer.adapter": "h3"})

    def test_video_vae_single_file_loader_infers_tiny_config(self):
        model = tiny_h3_vae()
        state_dict = dict(model.state_dict())
        state_dict["latents_mean"] = torch.tensor(model.config.latents_mean, dtype=torch.float32)
        state_dict["latents_std"] = torch.tensor(model.config.latents_std, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-vae.safetensors"
            save_file(state_dict, path)
            loaded = AutoencoderKLMiniMaxH3.from_single_file(path, torch_dtype=torch.float32)

        self.assertEqual(loaded.config.latent_channels, 2)
        self.assertEqual(tuple(loaded.config.block_out_channels), (4,))
        self.assertEqual(tuple(loaded.config.spatial_downsample_factors), (2,))
        self.assertEqual(tuple(loaded.config.temporal_downsample_factors), (1,))
        self.assertEqual(tuple(loaded.config.latents_mean), tuple(model.config.latents_mean))
        self.assertEqual(tuple(loaded.config.latents_std), tuple(model.config.latents_std))
        self.assertFalse(loaded.decoder.rope.inv_freq.is_meta)
        self.assertFalse(loaded.decoder.transformer_blocks[0].ff.net[0].gate_first)

    def test_video_vae_single_file_loader_preserves_raw_swiglu_order(self):
        model = tiny_h3_vae(decoder_swiglu_gate_first=True)
        state_dict = dict(model.state_dict())
        for suffix in ("weight", "bias"):
            diffusers_key = f"decoder.transformer_blocks.0.ff.net.0.proj.{suffix}"
            raw_key = f"decoder.transformer_blocks.0.ff.w1.{suffix}"
            state_dict[raw_key] = state_dict.pop(diffusers_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-vae-raw.safetensors"
            save_file(state_dict, path)
            loaded = AutoencoderKLMiniMaxH3.from_single_file(path, torch_dtype=torch.float32)

        self.assertTrue(loaded.config.decoder_swiglu_gate_first)
        self.assertTrue(loaded.decoder.transformer_blocks[0].ff.net[0].gate_first)

    def test_video_vae_single_file_loader_splits_comfy_head_interleaved_qkv(self):
        model = tiny_h3_vae()
        state_dict = dict(model.state_dict())
        head_dim = model.config.decoder_attention_head_dim
        q_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_q.weight")
        k_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_k.weight")
        v_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_v.weight")
        q_bias = state_dict.pop("decoder.transformer_blocks.0.attn.to_q.bias")
        k_bias = state_dict.pop("decoder.transformer_blocks.0.attn.to_k.bias")
        v_bias = state_dict.pop("decoder.transformer_blocks.0.attn.to_v.bias")
        q_weight = torch.arange(q_weight.numel(), dtype=torch.float32).reshape_as(q_weight)
        k_weight = torch.arange(k_weight.numel(), dtype=torch.float32).reshape_as(k_weight) + 1_000.0
        v_weight = torch.arange(v_weight.numel(), dtype=torch.float32).reshape_as(v_weight) + 2_000.0
        q_bias = torch.arange(q_bias.numel(), dtype=torch.float32).reshape_as(q_bias)
        k_bias = torch.arange(k_bias.numel(), dtype=torch.float32).reshape_as(k_bias) + 100.0
        v_bias = torch.arange(v_bias.numel(), dtype=torch.float32).reshape_as(v_bias) + 200.0
        state_dict["decoder.transformer_blocks.0.attn.to_qkv.weight"] = comfy_head_interleaved_qkv(
            q_weight, k_weight, v_weight, head_dim
        )
        state_dict["decoder.transformer_blocks.0.attn.to_qkv.bias"] = comfy_head_interleaved_qkv(
            q_bias, k_bias, v_bias, head_dim
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-vae-comfy-qkv.safetensors"
            save_file(state_dict, path)
            loaded = AutoencoderKLMiniMaxH3.from_single_file(
                path,
                torch_dtype=torch.float32,
                decoder_num_attention_heads=model.config.decoder_num_attention_heads,
                decoder_attention_head_dim=head_dim,
            )

        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_q.weight, q_weight))
        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_k.weight, k_weight))
        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_v.weight, v_weight))
        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_q.bias, q_bias))
        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_k.bias, k_bias))
        self.assertTrue(torch.equal(loaded.decoder.transformer_blocks[0].attn.to_v.bias, v_bias))

    def test_video_vae_single_file_loader_accepts_comfy_convrot_to_out(self):
        model = tiny_h3_vae()
        state_dict = dict(model.state_dict())
        target_key = "decoder.transformer_blocks.0.attn.to_out.0.weight"
        target_weight = state_dict.pop(target_key)
        source_key = "decoder.transformer_blocks.0.attn.to_out.weight"
        state_dict[source_key] = torch.zeros(target_weight.shape, dtype=torch.int8)
        state_dict[f"{source_key}_scale"] = torch.ones(target_weight.shape[0], 1, dtype=torch.float32)
        state_dict["decoder.transformer_blocks.0.attn.to_out.comfy_quant"] = comfy_quant_metadata_tensor(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-vae-convrot.safetensors"
            save_file(state_dict, path)
            with patch("simpletuner.helpers.models.z_image.quantized_loading._wrap_convrot_linear") as wrap_convrot:
                loaded = AutoencoderKLMiniMaxH3.from_single_file(path, torch_dtype=torch.float32)

        wrap_convrot.assert_called_once()
        self.assertEqual(wrap_convrot.call_args.args[1], "decoder.transformer_blocks.0.attn.to_out.0")
        self.assertEqual(wrap_convrot.call_args.kwargs["hadamard_group_size"], 256)
        self.assertEqual(wrap_convrot.call_args.kwargs["result_dtype"], torch.float32)
        self.assertEqual(loaded.quantization_method, "minimax_h3_vae_comfy_convrot_sdnq")

    def test_video_vae_single_file_loader_splits_comfy_convrot_qkv(self):
        model = tiny_h3_vae()
        state_dict = dict(model.state_dict())
        q_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_q.weight")
        k_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_k.weight")
        v_weight = state_dict.pop("decoder.transformer_blocks.0.attn.to_v.weight")
        q_weight = torch.full(q_weight.shape, 1, dtype=torch.int8)
        k_weight = torch.full(k_weight.shape, 2, dtype=torch.int8)
        v_weight = torch.full(v_weight.shape, 3, dtype=torch.int8)
        q_scale = torch.full((q_weight.shape[0], 1), 1.0, dtype=torch.float32)
        k_scale = torch.full((k_weight.shape[0], 1), 2.0, dtype=torch.float32)
        v_scale = torch.full((v_weight.shape[0], 1), 3.0, dtype=torch.float32)
        source_key = "decoder.transformer_blocks.0.attn.to_qkv.weight"
        head_dim = model.config.decoder_attention_head_dim
        fused_weight = comfy_head_interleaved_qkv(q_weight, k_weight, v_weight, head_dim)
        state_dict[source_key] = fused_weight
        state_dict[f"{source_key}_scale"] = comfy_head_interleaved_qkv(q_scale, k_scale, v_scale, head_dim)
        state_dict["decoder.transformer_blocks.0.attn.to_qkv.comfy_quant"] = comfy_quant_metadata_tensor(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 128}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-vae-convrot-qkv.safetensors"
            save_file(state_dict, path)
            with patch("simpletuner.helpers.models.z_image.quantized_loading._wrap_convrot_linear") as wrap_convrot:
                AutoencoderKLMiniMaxH3.from_single_file(
                    path,
                    torch_dtype=torch.bfloat16,
                    decoder_num_attention_heads=model.config.decoder_num_attention_heads,
                    decoder_attention_head_dim=head_dim,
                )

        self.assertEqual(wrap_convrot.call_count, 3)
        self.assertEqual(
            [call.args[1] for call in wrap_convrot.call_args_list],
            [
                "decoder.transformer_blocks.0.attn.to_q",
                "decoder.transformer_blocks.0.attn.to_k",
                "decoder.transformer_blocks.0.attn.to_v",
            ],
        )
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[0].args[2], q_weight))
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[1].args[2], k_weight))
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[2].args[2], v_weight))
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[0].args[3], q_scale))
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[1].args[3], k_scale))
        self.assertTrue(torch.equal(wrap_convrot.call_args_list[2].args[3], v_scale))
        self.assertTrue(all(call.kwargs["hadamard_group_size"] == 128 for call in wrap_convrot.call_args_list))

    def test_single_file_diffusers_loader_infers_tiny_config(self):
        model = tiny_h3_transformer(num_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3.safetensors"
            save_file(model.state_dict(), path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)
        self.assertEqual(loaded.config.hidden_size, 16)
        self.assertEqual(loaded.config.num_layers, 1)
        self.assertEqual(tuple(loaded.config.patch_size), (1, 2, 2))
        self.assertFalse(loaded.config.swiglu_gate_first)
        self.assertFalse(loaded.rope.inv_freq.is_meta)

    def test_single_file_native_loader_normalizes_swiglu_to_diffusers_order(self):
        model = tiny_h3_transformer(num_layers=1)
        native_state_dict = {}
        for key, value in model.state_dict().items():
            native_key = key
            for source, target in (
                ("audio_proj_in.", "audio_patch_proj."),
                ("proj_in.", "video_patch_proj."),
                ("context_embedder.", "condition_proj."),
                ("time_embedder.linear_1.", "time_embedder.proj_in."),
                ("time_embedder.linear_2.", "time_embedder.proj_out."),
                ("norm_out.norm.", "final_layer.norm."),
                ("norm_out.linear.", "final_layer.adaln_proj.linear."),
                ("audio_proj_out.", "final_layer.audio_out."),
                ("proj_out.", "final_layer.video_out."),
            ):
                if native_key.startswith(source):
                    native_key = native_key.replace(source, target, 1)
                    break
            native_key = native_key.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
            native_key = native_key.replace("transformer_blocks.", "blocks.", 1)
            native_key = native_key.replace(".attn.norm_q.", ".attn.q_norm.")
            native_key = native_key.replace(".attn.norm_k.", ".attn.k_norm.")
            native_key = native_key.replace(".attn.to_out.0.", ".attn.out_proj.")
            native_key = native_key.replace(".ff.net.0.proj.", ".mlp.fc1.")
            native_key = native_key.replace(".ff.net.2.", ".mlp.fc2.")
            if native_key.endswith(".mlp.fc1.weight"):
                value, gate = value.chunk(2, dim=0)
                value = torch.cat((gate, value), dim=0).contiguous()
            native_state_dict[native_key] = value

        for prefix in ("blocks.0", "token_refiner.blocks.0"):
            q = native_state_dict.pop(f"{prefix}.attn.to_q.weight")
            k = native_state_dict.pop(f"{prefix}.attn.to_k.weight")
            v = native_state_dict.pop(f"{prefix}.attn.to_v.weight")
            native_state_dict[f"{prefix}.attn.qkv_proj.weight"] = torch.cat((q, k, v), dim=0)
        native_state_dict["rope.inv_freq"] = model.rope.inv_freq

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-native.safetensors"
            save_file(native_state_dict, path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)

        self.assertFalse(loaded.config.swiglu_gate_first)
        self.assertEqual(type(loaded.transformer_blocks[0].ff).__name__, "FeedForward")
        self.assertTrue(
            torch.equal(
                loaded.transformer_blocks[0].ff.net[0].proj.weight,
                model.transformer_blocks[0].ff.net[0].proj.weight,
            )
        )
        inputs = tiny_inputs()
        model.eval()
        loaded.eval()
        expected = model(**inputs)
        actual = loaded(**inputs)
        self.assertTrue(torch.equal(actual.sample, expected.sample))
        self.assertTrue(torch.equal(actual.audio_sample, expected.audio_sample))

    def test_single_file_loader_accepts_trailing_safetensors_bytes(self):
        model = tiny_h3_transformer(num_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-trailing.safetensors"
            save_file(model.state_dict(), path)
            with open(path, "ab") as handle:
                handle.write(b"trailing")
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)
        self.assertEqual(loaded.config.hidden_size, 16)
        self.assertEqual(loaded.config.num_layers, 1)

    def test_single_file_loader_supports_adaln_curve_table(self):
        model = tiny_h3_transformer(num_layers=1, time_embed_dim=3, adaln_curve_grid=5)
        model.adaln_t_table.copy_(torch.arange(15, dtype=torch.float32).view(5, 3))
        rope_inv_freq = model.rope.inv_freq + 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-curve.safetensors"
            state_dict = dict(model.state_dict())
            state_dict["rope.inv_freq"] = rope_inv_freq
            for key in (
                "transformer_blocks.0.adaln_proj.linear.weight",
                "transformer_blocks.0.adaln_proj.linear.bias",
                "norm_out.linear.weight",
                "norm_out.linear.bias",
            ):
                state_dict[key] = state_dict[key].to(torch.float16)
            save_file(state_dict, path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.bfloat16)
        self.assertEqual(loaded.config.adaln_curve_grid, 5)
        self.assertIsNone(loaded.time_embedder)
        self.assertEqual(loaded.transformer_blocks[0].adaln_proj.linear.in_features, 3)
        self.assertTrue(torch.equal(loaded.adaln_t_table, model.adaln_t_table))
        self.assertTrue(torch.equal(loaded.rope.inv_freq, rope_inv_freq))
        self.assertEqual(loaded.transformer_blocks[0].adaln_proj.linear.weight.dtype, torch.float32)
        self.assertEqual(loaded.transformer_blocks[0].adaln_proj.linear.bias.dtype, torch.float32)
        self.assertEqual(loaded.norm_out.linear.weight.dtype, torch.float32)
        self.assertEqual(loaded.norm_out.linear.bias.dtype, torch.float32)
        self.assertEqual(loaded.context_embedder.weight.dtype, torch.bfloat16)

        normed = loaded.norm_out(
            torch.randn(1, 2, 16, dtype=torch.bfloat16),
            torch.randn(1, 3, dtype=torch.float32),
            torch.tensor([0, 0], dtype=torch.long),
        )
        self.assertEqual(normed.dtype, torch.bfloat16)

    def test_single_file_loader_accepts_abiray_convrot_metadata(self):
        model = tiny_h3_transformer(num_layers=1)
        state_dict = dict(model.state_dict())
        target_key = "transformer_blocks.0.attn.to_out.0.weight"
        target_weight = state_dict.pop(target_key)
        source_key = "blocks.0.attn.out_proj.weight"
        state_dict[source_key] = torch.zeros(target_weight.shape, dtype=torch.int8)
        state_dict[f"{source_key}_scale"] = torch.ones(target_weight.shape[0], 1, dtype=torch.float32)
        state_dict["blocks.0.attn.out_proj.comfy_quant"] = comfy_quant_metadata_tensor(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-convrot.safetensors"
            save_file(state_dict, path)
            with patch("simpletuner.helpers.models.z_image.quantized_loading._wrap_convrot_linear") as wrap_convrot:
                MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)

        wrap_convrot.assert_called_once()
        self.assertEqual(wrap_convrot.call_args.args[1], "transformer_blocks.0.attn.to_out.0")
        self.assertEqual(wrap_convrot.call_args.kwargs["hadamard_group_size"], 256)

    def test_single_file_loader_accepts_comfy_fp8_scale_metadata(self):
        model = tiny_h3_transformer(num_layers=1)
        state_dict = dict(model.state_dict())
        target_key = "transformer_blocks.0.attn.to_out.0.weight"
        target_weight = state_dict.pop(target_key)
        source_key = "blocks.0.attn.out_proj.weight"
        weight_scale = torch.tensor(0.125, dtype=torch.float32)
        state_dict[source_key] = (target_weight.to(torch.float32) / weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        state_dict[f"{source_key}_scale"] = weight_scale
        state_dict["blocks.0.attn.out_proj.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
        state_dict["blocks.0.attn.out_proj.comfy_quant"] = comfy_quant_metadata_tensor({"format": "float8_e4m3fn"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-fp8.safetensors"
            save_file(state_dict, path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.bfloat16)

        layer = loaded.transformer_blocks[0].attn.to_out[0]
        self.assertIsInstance(layer, Fp8Linear)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight_scale.shape), (target_weight.shape[0],))
        self.assertTrue(torch.equal(layer.weight_scale, weight_scale.expand(target_weight.shape[0])))
        self.assertEqual(loaded.quantization_method, "minimax_h3_comfy_fp8")

    def test_single_file_loader_splits_comfy_fp8_qkv_weight_and_scale(self):
        model = tiny_h3_transformer(num_layers=1)
        state_dict = dict(model.state_dict())
        q_weight = state_dict["transformer_blocks.0.attn.to_q.weight"]
        k_weight = state_dict["transformer_blocks.0.attn.to_k.weight"]
        v_weight = state_dict["transformer_blocks.0.attn.to_v.weight"]
        source_key = "blocks.0.attn.qkv_proj.weight"
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        weight_scale = torch.tensor(0.125, dtype=torch.float32)
        state_dict[source_key] = (fused_weight.to(torch.float32) / weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        state_dict[f"{source_key}_scale"] = weight_scale
        state_dict["blocks.0.attn.qkv_proj.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
        state_dict["blocks.0.attn.qkv_proj.comfy_quant"] = comfy_quant_metadata_tensor({"format": "float8_e4m3fn"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3-fp8-qkv.safetensors"
            save_file(state_dict, path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.bfloat16)

        for layer, expected_weight in (
            (loaded.transformer_blocks[0].attn.to_q, q_weight),
            (loaded.transformer_blocks[0].attn.to_k, k_weight),
            (loaded.transformer_blocks[0].attn.to_v, v_weight),
        ):
            self.assertIsInstance(layer, Fp8Linear)
            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(layer.weight.shape), tuple(expected_weight.shape))
            self.assertTrue(torch.equal(layer.weight_scale, weight_scale.expand(expected_weight.shape[0])))

    def test_collate_prompt_embeds_pads_tags(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        result = wrapper.collate_prompt_embeds(
            [
                {
                    "prompt_embeds": torch.ones(1, 2, 4),
                    "text_token_tags": torch.tensor([[MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG]]),
                },
                {
                    "prompt_embeds": torch.ones(1, 3, 4),
                    "text_token_tags": torch.tensor([MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_TEXT_TAG]),
                },
            ]
        )
        self.assertEqual(result["prompt_embeds"].shape, (2, 3, 4))
        self.assertEqual(result["text_token_tags"][0, -1].item(), -1)

    def test_slice_text_embedding_for_cache_removes_batch_padding(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        output = {
            "prompt_embeds": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            "text_token_tags": torch.tensor([[1, 0, -1], [1, 0, 1]], dtype=torch.long),
        }

        first = wrapper.slice_text_embedding_for_cache(output, batch_index=0, batch_size=2)
        second = wrapper.slice_text_embedding_for_cache(output, batch_index=1, batch_size=2)

        self.assertEqual(first["prompt_embeds"].shape, (1, 2, 4))
        self.assertEqual(first["text_token_tags"].tolist(), [[1, 0]])
        self.assertEqual(second["prompt_embeds"].shape, (1, 3, 4))
        self.assertEqual(second["text_token_tags"].tolist(), [[1, 0, 1]])

    def test_supports_fake_video_stream_with_audio_lora_target(self):
        self.assertTrue(MiniMaxH3.SUPPORTS_FAKE_VIDEO_STREAM)
        self.assertTrue(MiniMaxH3.supports_audio_only_training())
        self.assertEqual(MiniMaxH3.DEFAULT_AUDIO_CHANNELS, 2)
        self.assertIn("audio_proj_in", MiniMaxH3.AUDIO_LORA_TARGET)
        self.assertIn("audio_proj_out", MiniMaxH3.AUDIO_LORA_TARGET)

    def test_audio_only_data_uses_audio_lora_target(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            lora_type="standard",
            peft_lora_target_modules=None,
            controlnet=False,
            slider_lora_target=False,
        )
        wrapper.configure_data_signals(has_audio=True)

        self.assertEqual(wrapper.get_lora_target_layers(), MiniMaxH3.AUDIO_LORA_TARGET)

    def test_manual_lora_target_overrides_audio_default(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(
            lora_type="standard",
            peft_lora_target_modules=["custom.audio.layer"],
            controlnet=False,
            slider_lora_target=False,
        )
        wrapper.configure_data_signals(has_audio=True)

        self.assertEqual(wrapper.get_lora_target_layers(), ["custom.audio.layer"])

    def test_audio_backend_selects_av_target_mode_by_default(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(minimax_h3_target_mode="auto")

        with patch.object(StateTracker, "get_data_backend_config", return_value={"dataset_type": "audio"}):
            self.assertEqual(wrapper._h3_target_mode_for_data_backend("music"), "av")
            self.assertTrue(wrapper.uses_audio_latents_for_data_backend("music"))

    def test_audio_dataset_uses_audio_vae(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.audio_vae = object()
        wrapper._load_audio_vae = Mock()

        result = wrapper.get_vae_for_dataset_type("audio")

        wrapper._load_audio_vae.assert_called_once_with(move_to_device=True)
        self.assertIs(result, wrapper.audio_vae)

    def test_fake_video_stream_has_one_spatial_token_per_frame(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer(num_layers=1)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model
        audio_length = 10 * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
        batch = {"audio_latent_batch": torch.zeros(1, 2, 3, audio_length)}

        latents = wrapper._build_fake_video_latents(batch, torch.device("cpu"), torch.float32)

        self.assertEqual(latents.shape, (1, 2, 72, 2, 2))
        packed = latents.shape[2] * (latents.shape[3] // 2) * (latents.shape[4] // 2)
        self.assertEqual(packed, latents.shape[2])

    def test_fake_video_stream_runs_joint_h3_forward(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.model = tiny_h3_transformer(num_layers=1)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(weight_dtype=torch.float32)
        wrapper.LATENT_CHANNEL_COUNT = 2
        wrapper.unwrap_model = lambda model=None: model
        audio_noisy = torch.randn(1, 2, 3, 2)
        fake_video = wrapper._build_fake_video_latents(
            {"audio_latent_batch": audio_noisy}, torch.device("cpu"), torch.float32
        )

        output = wrapper.model_predict(
            {
                "noisy_latents": fake_video,
                "audio_noisy_latents": audio_noisy,
                "encoder_hidden_states": torch.randn(1, 5, 6),
                "timesteps": torch.tensor([0.25]),
                "audio_timesteps": torch.tensor([0.5]),
                "text_token_tags": torch.full((1, 5), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
                "minimax_h3_target_mode": "av",
            }
        )

        self.assertEqual(output["model_prediction"].shape, fake_video.shape)
        self.assertEqual(output["audio_prediction"].shape, audio_noisy.shape)

    def test_audio_only_loss_ignores_fake_video_prediction(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.config = SimpleNamespace(audio_loss_weight=1.0)
        video_prediction = torch.full((1, 2, 2, 2, 2), 100.0, requires_grad=True)
        audio_prediction = torch.ones(1, 2, 3, 2, requires_grad=True)

        loss, video_loss, audio_loss, _ = wrapper._compute_av_loss(
            {
                "video_latent_mask": torch.zeros(1),
                "audio_latent_mask": torch.ones(1),
                "audio_target": torch.zeros_like(audio_prediction),
            },
            {"model_prediction": video_prediction, "audio_prediction": audio_prediction},
        )
        loss.backward()

        self.assertEqual(video_loss.item(), 0.0)
        self.assertEqual(audio_loss.item(), 1.0)
        self.assertEqual(loss.item(), 1.0)
        self.assertTrue(torch.equal(video_prediction.grad, torch.zeros_like(video_prediction)))
        self.assertTrue(bool(torch.all(audio_prediction.grad > 0)))


if __name__ == "__main__":
    unittest.main()
