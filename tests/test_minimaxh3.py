import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from accelerate.utils.operations import convert_to_fp32
from PIL import Image
from safetensors.torch import save_file

from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.models.ideogram.quantized_loading import Fp8Linear
from simpletuner.helpers.models.minimaxh3.activations import MiniMaxH3FeedForward
from simpletuner.helpers.models.minimaxh3.autoencoder import AutoencoderKLMiniMaxH3
from simpletuner.helpers.models.minimaxh3.denoise import _predict_guided_velocity
from simpletuner.helpers.models.minimaxh3.model import MiniMaxH3
from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
    _convert_minimax_h3_comfy_lora_to_diffusers,
)
from simpletuner.helpers.models.minimaxh3.packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_TAG,
    build_packed_sequence,
    build_row_timesteps,
)
from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Transformer3DModel, MiniMaxH3TransformerOutput
from simpletuner.helpers.models.registry import ModelRegistry


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
        return_dict=False,
    ):
        self.calls.append(
            {
                "text_rows": text_indices.shape[0],
                "sequence_rows": position_ids.shape[0],
                "skip_layers": skip_layers,
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

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


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
            return_value=(torch.zeros(1, 3, 4), torch.tensor([1, 0, 1], dtype=torch.long)),
        ) as encode_prompt:
            encoded = model._encode_prompts(["caption"])

        self.assertEqual(encoded["prompt_embeds"].shape, (1, 3, 4))
        self.assertEqual(encoded["text_token_tags"].tolist(), [[1, 0, 1]])
        call_kwargs = encode_prompt.call_args.kwargs
        self.assertEqual(call_kwargs["device"], torch.device("cpu"))
        self.assertEqual(call_kwargs["dtype"], torch.float32)
        self.assertEqual(len(call_kwargs["images"]), 1)
        self.assertIsInstance(call_kwargs["images"][0], Image.Image)

    def test_h3_feedforward_uses_comfy_swiglu_gate_first_order(self):
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
        expected = torch.nn.functional.silu(torch.tensor([[0.5, -1.0]])) * torch.tensor([[1.0, -3.0]])

        self.assertTrue(torch.allclose(result, expected))
        self.assertIn("net.0.proj.weight", feed_forward.state_dict())
        self.assertIn("net.2.weight", feed_forward.state_dict())

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
        }
        output = wrapper.model_predict(prepared_batch)
        self.assertEqual(output["model_prediction"].shape, (1, 2, 2, 2, 2))
        self.assertEqual(output["audio_prediction"].shape, (1, 2, 3, 2))

    def test_prepare_batch_conditions_maps_audio_to_audio_flow_shift(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
        wrapper.config = SimpleNamespace(
            weight_dtype=torch.float32,
            input_perturbation=0,
            input_perturbation_steps=None,
            flow_schedule_shift=12.0,
            audio_flow_schedule_shift=3.0,
        )
        wrapper._warned_missing_audio = False
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

    def test_load_vae_uses_diffusers_component_subfolder(self):
        wrapper = MiniMaxH3.__new__(MiniMaxH3)
        wrapper.vae = None
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

        with patch.object(MiniMaxH3.AUTOENCODER_CLASS, "from_pretrained", return_value=FakeVAE()) as from_pretrained:
            wrapper.load_vae(move_to_device=False)

        _, kwargs = from_pretrained.call_args
        self.assertEqual(kwargs["subfolder"], "vae")
        wrapper._load_audio_vae.assert_called_once_with(move_to_device=False)

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

    def test_i2v_first_frame_vae_cache_uses_spatial_keyframe_encode_and_posterior_mode(self):
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
        self.assertEqual(len(encode_prompt.call_args.kwargs["images"]), 1)
        self.assertEqual(result["prompt_embeds"].shape, (1, 1, 2))
        self.assertEqual(result["text_token_tags"].shape, (1, 1))

    def test_guided_velocity_uses_negative_layout_and_skip_layers(self):
        transformer = FakeH3Transformer()
        block_state = tiny_block_state_for_guidance()

        video, audio = _predict_guided_velocity(transformer, block_state, i=0, num_steps=1)

        self.assertTrue(torch.allclose(video, torch.full_like(video, 4.7)))
        self.assertTrue(torch.allclose(audio, torch.full_like(audio, 14.7)))
        self.assertEqual([call["text_rows"] for call in transformer.calls], [5, 3, 5])
        self.assertEqual(transformer.calls[1]["sequence_rows"], block_state.negative_position_ids.shape[0])
        self.assertEqual(transformer.calls[2]["skip_layers"], [1])

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
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_q.lora.up.weight"], qkv_up[:16]))
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_k.lora.up.weight"], qkv_up[16:32]))
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_v.lora.up.weight"], qkv_up[32:]))
        self.assertIn("transformer.proj_in.lora.down.weight", converted)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_q.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_k.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_v.alpha"], 4.0)

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

    def test_single_file_diffusers_loader_infers_tiny_config(self):
        model = tiny_h3_transformer(num_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny-h3.safetensors"
            save_file(model.state_dict(), path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)
        self.assertEqual(loaded.config.hidden_size, 16)
        self.assertEqual(loaded.config.num_layers, 1)
        self.assertEqual(tuple(loaded.config.patch_size), (1, 2, 2))
        self.assertFalse(loaded.rope.inv_freq.is_meta)

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
            save_file(state_dict, path)
            loaded = MiniMaxH3Transformer3DModel.from_single_file(path, torch_dtype=torch.float32)
        self.assertEqual(loaded.config.adaln_curve_grid, 5)
        self.assertIsNone(loaded.time_embedder)
        self.assertEqual(loaded.transformer_blocks[0].adaln_proj.linear.in_features, 3)
        self.assertTrue(torch.equal(loaded.adaln_t_table, model.adaln_t_table))
        self.assertTrue(torch.equal(loaded.rope.inv_freq, rope_inv_freq))

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


if __name__ == "__main__":
    unittest.main()
