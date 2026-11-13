import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from simpletuner.helpers.models.minimaxh3.autoencoder_image import AutoencoderKLMiniMaxH3Image
from simpletuner.helpers.models.minimaxh3.image import MINIMAX_H3_IMAGE_REPO, MINIMAX_H3_IMAGE_SIGMAS
from simpletuner.helpers.models.minimaxh3.model import MiniMaxH3
from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Transformer3DModel
from tests.test_minimaxh3 import tiny_h3_transformer


def image_wrapper(vae_path=None):
    wrapper = MiniMaxH3.__new__(MiniMaxH3)
    wrapper.config = SimpleNamespace(
        model_flavour="h3_image",
        pretrained_model_name_or_path=None,
        pretrained_transformer_model_name_or_path=None,
        pretrained_transformer_subfolder=None,
        pretrained_vae_model_name_or_path=vae_path,
        vae_path=vae_path,
        flow_schedule_shift=None,
        audio_flow_schedule_shift=None,
    )
    wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"))
    wrapper.setup_model_flavour()
    return wrapper


class H3ImageTests(unittest.TestCase):
    def test_image_layout_advances_each_reference_clock(self):
        from simpletuner.helpers.models.minimaxh3.image_blocks import build_image_layout

        layout = build_image_layout(torch.tensor([1, 0, 1]), [(4, 8), (8, 4)], 8, 8, (1, 2, 2))
        self.assertEqual(layout.num_condition_video_rows, 16)
        torch.testing.assert_close(layout.position_ids[3:11, 0], torch.full((8,), 3.0, dtype=torch.float64))
        torch.testing.assert_close(layout.position_ids[11:19, 0], torch.full((8,), 4.0, dtype=torch.float64))
        torch.testing.assert_close(layout.position_ids[19:, 0], torch.full((16,), 5.0, dtype=torch.float64))
        self.assertEqual(layout.token_tags[1], 0)

    def test_image_scheduler_uses_all_six_heads_and_exact_grid(self):
        from simpletuner.helpers.models.minimaxh3.image import MiniMaxH3ImageScheduler

        scheduler = MiniMaxH3ImageScheduler()
        scheduler.set_timesteps(6)
        self.assertEqual(len(scheduler.timesteps), 6)
        timesteps = torch.tensor([1 - sigma for sigma in MINIMAX_H3_IMAGE_SIGMAS[:-1]])
        torch.testing.assert_close(scheduler.sigmas, torch.cat([1 - timesteps, torch.zeros(1)]), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "six"):
            scheduler.set_timesteps(30)

    def test_image_blocks_generate_a_still_image_with_cached_text(self):
        from diffusers.video_processor import VideoProcessor
        from PIL import Image

        from simpletuner.helpers.models.minimaxh3.image import MiniMaxH3ImageScheduler
        from simpletuner.helpers.models.minimaxh3.image_blocks import MiniMaxH3ImageBlocks
        from simpletuner.helpers.models.minimaxh3.pipeline import MiniMaxH3Pipeline

        model = tiny_h3_transformer(num_layers=1, audio_enabled=False, num_pdd_heads=6)
        vae = AutoencoderKLMiniMaxH3Image(
            latent_channels=2,
            block_out_channels=(4,),
            layers_per_block=1,
            spatial_downsample_factors=(2,),
            temporal_downsample_factors=(1,),
            norm_num_groups=1,
            decoder_num_layers=1,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
        )
        pipeline = MiniMaxH3Pipeline(blocks=MiniMaxH3ImageBlocks())
        pipeline.update_components(
            transformer=model,
            vae=vae,
            scheduler=MiniMaxH3ImageScheduler(),
            video_processor=VideoProcessor(do_resize=False, do_normalize=False, vae_scale_factor=2, vae_latent_channels=2),
        )
        for images in (None, [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))]):
            with self.subTest(references=0 if images is None else len(images)):
                with patch(
                    "simpletuner.helpers.models.minimaxh3.image_blocks.resize_reference", side_effect=lambda image, *_: image
                ):
                    state = pipeline(
                        prompt_embeds=torch.randn(1, 2, 6),
                        text_token_tags=torch.ones(2, dtype=torch.long),
                        image=images,
                        height=32,
                        width=32,
                        num_inference_steps=6,
                        output_type="pil",
                    )
                self.assertIsInstance(state.get("videos")[0], Image.Image)
                self.assertEqual(state.get("videos")[0].size, (32, 32))

    def test_prompt_expander_option_is_parseable(self):
        from simpletuner.helpers.configuration.cmd_args import get_argument_parser

        args = get_argument_parser().parse_args(
            [
                "--model_family",
                "minimaxh3",
                "--model_flavour",
                "h3_image",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--output_dir",
                "output/test",
                "--data_backend_config",
                "data.json",
                "--minimax_h3_prompt_expander_url",
                "http://localhost:30000",
            ]
        )
        self.assertEqual(args.minimax_h3_prompt_expander_url, "http://localhost:30000")

    def test_prompt_expander_requests_preserve_edit_reference_order(self):
        import json

        from PIL import Image

        from simpletuner.helpers.models.minimaxh3.prompt_expander import MiniMaxH3ImagePromptExpander
        from simpletuner.helpers.models.prompt_upsampling import PromptUpsamplingRequest

        expander = MiniMaxH3ImagePromptExpander.__new__(MiniMaxH3ImagePromptExpander)
        expander.prompts = {
            "t2i/system_prompt.txt": "system",
            "t2i/task_prompt.txt": "Input: {}",
            "edit/task_prompt.txt": "edit",
            "t2i/request_params.json": json.dumps({"max_tokens": 4096, "response_format": {"json_schema": {"schema": {}}}}),
        }
        request = PromptUpsamplingRequest("a red kite", 2048, 4096)
        text = expander.build_request(request)
        self.assertIn("4096x2048", text["messages"][1]["content"])
        self.assertEqual(text["model"], "MiniMaxAI/MiniMax-H3-Image-PE")
        references = (Image.new("RGB", (32, 16)), Image.new("RGB", (16, 32)))
        edit = expander.build_request(PromptUpsamplingRequest("replace the kite", 2048, 4096, references))
        content = edit["messages"][0]["content"]
        self.assertIn("<Picture 1> [aspect_ratio 2:1]", content[2]["text"])
        self.assertIn("<Picture 2> [aspect_ratio 1:2]", content[4]["text"])
        self.assertTrue(content[3]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(
            edit["response_format"]["json_schema"]["schema"]["required"], ["expanded_caption", "target_aspect_ratio"]
        )

    def test_image_flavour_preserves_custom_vae(self):
        for custom_path in (None, "custom/image-vae", "custom/image-vae.safetensors"):
            with self.subTest(custom_path=custom_path):
                wrapper = image_wrapper(custom_path)
                self.assertIs(wrapper.AUTOENCODER_CLASS, AutoencoderKLMiniMaxH3Image)
                self.assertEqual(wrapper.MODEL_SUBFOLDER, "transformer")
                self.assertEqual(wrapper.config.pretrained_model_name_or_path, MINIMAX_H3_IMAGE_REPO)
                self.assertEqual(wrapper.config.pretrained_transformer_model_name_or_path, MINIMAX_H3_IMAGE_REPO)
                self.assertEqual(wrapper.config.pretrained_vae_model_name_or_path, custom_path or MINIMAX_H3_IMAGE_REPO)
                self.assertEqual(wrapper.config.vae_path, custom_path or MINIMAX_H3_IMAGE_REPO)
                self.assertEqual(wrapper.config.validation_num_video_frames, 1)
                self.assertFalse(wrapper.supports_audio_inputs())
                self.assertFalse(wrapper.uses_audio_latents())
                wrapper._load_audio_vae()
                self.assertEqual(wrapper.config.pretrained_transformer_subfolder, "transformer_diffusers")

    def test_pdd_training_samples_all_released_heads_at_matching_sigmas(self):
        wrapper = image_wrapper()
        for index in range(6):
            with patch("torch.randint", return_value=torch.tensor(index)):
                batch = {"latents": torch.zeros(2, 24, 1, 2, 2)}
                sigmas, timesteps = wrapper.sample_flow_sigmas(batch, {})
            self.assertEqual(batch["h3_pdd_step_index"], index)
            torch.testing.assert_close(sigmas, torch.full((2,), MINIMAX_H3_IMAGE_SIGMAS[index]))
            torch.testing.assert_close(timesteps, 1 - sigmas)

    def test_image_transformer_checkpoint_roundtrip_without_audio(self):
        model = tiny_h3_transformer(num_layers=1, audio_enabled=False, num_pdd_heads=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.safetensors"
            save_file(model.state_dict(), path)
            restored = MiniMaxH3Transformer3DModel.from_single_file(str(path))
        self.assertFalse(restored.config.audio_enabled)
        self.assertEqual(restored.config.num_pdd_heads, 6)
        self.assertIsNone(restored.audio_proj_in)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])

    def test_native_comfy_checkpoint_unfuses_image_heads(self):
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import _map_minimax_h3_diffusers_lora_module_to_native

        model = tiny_h3_transformer(num_layers=1, audio_enabled=False, num_pdd_heads=6, adaln_curve_grid=5, time_embed_dim=3)
        native = {}
        for key, value in model.state_dict().items():
            if key.startswith("pdd_proj_out."):
                continue
            module, suffix = key.rsplit(".", 1) if "." in key else (key, None)
            name = _map_minimax_h3_diffusers_lora_module_to_native(module)
            name = f"{name}.{suffix}" if suffix else name
            if name.endswith(".mlp.fc1.weight"):
                hidden, gate = value.chunk(2, dim=0)
                value = torch.cat([gate, hidden])
            native[name] = value.contiguous()
        for prefix in ("blocks.0", "token_refiner.blocks.0"):
            native[f"{prefix}.attn.qkv_proj.weight"] = torch.cat(
                [native.pop(f"{prefix}.attn.to_{projection}.weight") for projection in ("q", "k", "v")]
            )
        for suffix in ("weight", "bias"):
            native[f"final_layer.video_out.{suffix}"] = torch.cat(
                [model.state_dict()[f"pdd_proj_out.{index}.{suffix}"] for index in range(6)]
            ).contiguous()
        native["rope.inv_freq"] = model.rope.inv_freq
        native["blocks.0.attn.comfy_attention.config"] = torch.tensor(
            list(b'[{"attention": "comfy_kitchen_int8"}]'), dtype=torch.uint8
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native-image.safetensors"
            save_file(native, path)
            restored = MiniMaxH3Transformer3DModel.from_single_file(str(path))
        self.assertFalse(restored.config.audio_enabled)
        self.assertEqual(restored.config.num_pdd_heads, 6)
        self.assertEqual(restored.config.adaln_curve_grid, 5)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_image_comfyui_adapter_preserves_nonzero_effective_updates(self):
        from peft import LoraConfig, get_peft_model_state_dict

        from scripts.verify_minimax_h3_image import verify_comfyui_adapter

        model = tiny_h3_transformer(num_layers=1, audio_enabled=False, num_pdd_heads=6)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.1)
        comfy = verify_comfyui_adapter(get_peft_model_state_dict(model))
        self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", comfy)

    def test_pdd_head_selection_and_trunk_gradients(self):
        model = tiny_h3_transformer(num_layers=1, audio_enabled=False, num_pdd_heads=6)
        inputs = dict(
            hidden_states=torch.randn(1, 1, 8),
            audio_hidden_states=torch.empty(1, 0, 3),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.tensor([0.0, 1.0]),
            timestep_indices=torch.tensor([1, 1, 0]),
            token_tags=torch.tensor([1, 1, 0]),
            position_ids=torch.zeros(3, 3),
            video_indices=torch.tensor([2]),
            audio_indices=torch.empty(0, dtype=torch.long),
            text_indices=torch.tensor([0, 1]),
        )
        with self.assertRaisesRegex(ValueError, "pdd_step_index"):
            model(**inputs)
        for index in range(6):
            model.zero_grad(set_to_none=True)
            output = model(**inputs, pdd_step_index=index)
            self.assertEqual(output.sample.shape, (1, 1, 8))
            self.assertEqual(output.audio_sample.shape, (1, 0, 3))
            output.sample.square().mean().backward()
            self.assertIsNotNone(model.transformer_blocks[0].attn.to_q.weight.grad)
            self.assertIsNotNone(model.pdd_proj_out[index].weight.grad)
            self.assertTrue(torch.isfinite(model.proj_in.weight.grad).all())

    def test_image_vae_roundtrip_accepts_image_and_single_frame_video(self):
        vae = AutoencoderKLMiniMaxH3Image(
            latent_channels=2,
            block_out_channels=(4,),
            layers_per_block=1,
            spatial_downsample_factors=(2,),
            temporal_downsample_factors=(1,),
            norm_num_groups=1,
            decoder_num_layers=1,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=2,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
        ).eval()
        pixels = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            image_latents = vae.encode(pixels).latent_dist.mode()
            video_latents = vae.encode(pixels.unsqueeze(2)).latent_dist.mode()
            torch.testing.assert_close(image_latents, video_latents[:, :, 0])
            decoded = vae.decode(image_latents).sample
            video_decoded = vae.decode(video_latents).sample
            self.assertEqual(decoded.shape, pixels.shape)
            torch.testing.assert_close(decoded, video_decoded[:, :, 0])
        with self.assertRaisesRegex(ValueError, "exactly one frame"):
            vae.encode(pixels.unsqueeze(2).expand(-1, -1, 2, -1, -1))


if __name__ == "__main__":
    unittest.main()
