import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.kandinsky5_image.model import Kandinsky5Image
from simpletuner.helpers.models.kandinsky5_image.pipeline_kandinsky5_i2i import Kandinsky5I2IPipeline
from simpletuner.helpers.models.kandinsky5_image.pipeline_kandinsky5_t2i import Kandinsky5T2IPipeline
from simpletuner.helpers.models.kandinsky5_video.model import Kandinsky5Video
from simpletuner.helpers.models.kandinsky5_video.pipeline_kandinsky5_i2v import Kandinsky5I2VPipeline
from simpletuner.helpers.models.kandinsky5_video.pipeline_kandinsky5_t2v import Kandinsky5T2VPipeline
from simpletuner.helpers.models.kandinsky5_video.transformer_kandinsky5 import Kandinsky5Transformer3DModel


def make_tiny_kandinsky_transformer(visual_cond: bool) -> Kandinsky5Transformer3DModel:
    """
    Instantiate a real Kandinsky transformer with reduced dimensions for fast tests.
    """
    return Kandinsky5Transformer3DModel(
        in_visual_dim=16,
        in_text_dim=8,
        in_text_dim2=4,
        time_dim=2,
        out_visual_dim=16,
        patch_size=(1, 2, 2),
        model_dim=12,
        ff_dim=24,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(2, 2, 2),
        visual_cond=visual_cond,
    )


class DummyImageVAE:
    def __init__(self, scaling_factor: int = 1, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self.config = SimpleNamespace(scaling_factor=scaling_factor)

    class LatentDist:
        def __init__(self, value: torch.Tensor):
            self.value = value

        def sample(self):
            return self.value

    def encode(self, x: torch.Tensor):
        latent = torch.ones(x.shape[0], 16, 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
        return SimpleNamespace(latent_dist=self.LatentDist(latent))


class DummyVideoVAE:
    def __init__(
        self,
        scaling_factor: float = 1.0,
        temporal_compression_ratio: int = 1,
        spatial_compression_ratio: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        self.config = SimpleNamespace(
            scaling_factor=scaling_factor,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )

    class LatentDist:
        def __init__(self, value: torch.Tensor):
            self.value = value

        def sample(self):
            return self.value

    def encode(self, x: torch.Tensor):
        bsz, _, frames, height, width = x.shape
        latent = torch.ones(
            bsz,
            16,
            frames,
            height // self.config.spatial_compression_ratio,
            width // self.config.spatial_compression_ratio,
            device=x.device,
            dtype=x.dtype,
        )
        return SimpleNamespace(latent_dist=self.LatentDist(latent))


class DummyScheduler:
    def __init__(self):
        self.order = 1
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.tensor([float(num_inference_steps)], device=device)

    def step(self, pred, t, latents, return_dict=False):
        return (latents,)


def _make_flow_scheduler():
    scheduler = MagicMock()
    scheduler.config = SimpleNamespace(prediction_type="flow_matching")
    scheduler.timesteps = torch.tensor([0.0])
    scheduler.set_timesteps = MagicMock()
    return scheduler


class DummyPromptPipeline:
    def __init__(self):
        self.calls = []

    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt=None,
        num_videos_per_prompt=None,
        max_sequence_length=None,
        device=None,
        dtype=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "num_images_per_prompt": num_images_per_prompt,
                "num_videos_per_prompt": num_videos_per_prompt,
                "max_sequence_length": max_sequence_length,
                "device": device,
                "dtype": dtype,
            }
        )
        batch = len(prompt)
        prompt_embeds_qwen = torch.zeros(batch, 2, 8, device=device, dtype=dtype or torch.float32)
        prompt_embeds_clip = torch.ones(batch, 4, device=device, dtype=dtype or torch.float32)
        prompt_cu_seqlens = torch.arange(0, batch + 1, device=device, dtype=torch.int32)
        return prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens


class Kandinsky5ImageModelTests(unittest.TestCase):
    def setUp(self):
        self.scheduler_patch = patch(
            "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
            side_effect=lambda *args, **kwargs: _make_flow_scheduler(),
        )
        self.scheduler_patch.start()
        self.addCleanup(self.scheduler_patch.stop)

        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            model_family="kandinsky5-image",
            model_flavour=None,
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=1.0,
            rescale_betas_zero_snr=False,
            training_scheduler_timestep_spacing=None,
            prediction_type=None,
            weight_dtype=torch.float32,
            controlnet=False,
            control=False,
            tokenizer_max_length=None,
        )
        self.model = Kandinsky5Image(self.config, self.accelerator)
        self.model.unwrap_model = lambda model=None: model or self.model
        self.model.model = make_tiny_kandinsky_transformer(visual_cond=False)

    def test_convert_text_embed_for_pipeline_generates_cu_seqlens(self):
        embeds = torch.randn(2, 3, 4)
        pooled = torch.randn(2, 4)
        attention_mask = torch.ones(2, 3, dtype=torch.int32)
        text_embedding = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled,
            "attention_masks": attention_mask,
        }

        converted = self.model.convert_text_embed_for_pipeline(text_embedding, prompt="ok")
        expected_cu = torch.tensor([0, 3, 6], dtype=torch.int32)

        self.assertTrue(torch.equal(converted["prompt_embeds_qwen"], embeds))
        self.assertTrue(torch.equal(converted["prompt_embeds_clip"], pooled))
        self.assertTrue(torch.equal(converted["prompt_cu_seqlens"], expected_cu))

    def test_check_user_config_sets_tokenizer_limit(self):
        # default None -> set to 256
        self.model.check_user_config()
        self.assertEqual(self.model.config.tokenizer_max_length, 256)

        # values above limit are clamped
        self.model.config.tokenizer_max_length = 300
        self.model.check_user_config()
        self.assertEqual(self.model.config.tokenizer_max_length, 256)

    def test_model_predict_injects_visual_conditioning(self):
        self.model.model = make_tiny_kandinsky_transformer(visual_cond=True)
        noisy_latents = torch.randn(1, 16, 2, 2)
        cond_latents = torch.ones(1, 16, 2, 2)
        prepared_batch = {
            "noisy_latents": noisy_latents,
            "conditioning_latents": cond_latents,
            "encoder_hidden_states": torch.randn(1, 2, 8),
            "timesteps": torch.tensor([0.5]),
            "added_cond_kwargs": {"text_embeds": torch.randn(1, 4)},
        }

        output = self.model.model_predict(prepared_batch)["model_prediction"]

        self.assertEqual(output.shape, (1, 16, 2, 2))

    def test_pack_and_unpack_text_embeddings_preserve_pooled_embeds(self):
        self.model.config.tokenizer_max_length = 6
        prompt_embeds = torch.arange(1 * 6 * 2, dtype=torch.float32).view(1, 6, 2)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int32)
        pooled = torch.tensor([[9.0, 8.0, 7.0, 6.0]], dtype=torch.float32)
        embeddings = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "attention_masks": attention_mask,
        }

        packed = self.model.pack_text_embeddings_for_cache(embeddings)
        self.assertEqual(packed["prompt_embeds"].shape, torch.Size([1, 4, 2]))
        self.assertEqual(packed["attention_masks"].shape, torch.Size([1, 4]))
        self.assertIn("_pad_slices", packed)
        self.assertTrue(torch.equal(packed["_pad_slices"]["prompt_embeds"], prompt_embeds[:, 3:4]))
        self.assertTrue(torch.equal(packed["pooled_prompt_embeds"], pooled))

        unpacked = self.model.unpack_text_embeddings_from_cache(packed)
        self.assertEqual(unpacked["prompt_embeds"].shape, torch.Size([1, 6, 2]))
        self.assertTrue(torch.equal(unpacked["prompt_embeds"][:, :4], prompt_embeds[:, :4]))
        pad_token = prompt_embeds[:, 3:4].expand(-1, 2, -1)
        self.assertTrue(torch.equal(unpacked["prompt_embeds"][:, 4:], pad_token))
        self.assertTrue(torch.equal(unpacked["attention_masks"], torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int32)))
        self.assertTrue(torch.equal(unpacked["pooled_prompt_embeds"], pooled))

    def test_unpack_text_embeddings_squeezes_pooled_prompt_embeds(self):
        self.model.config.tokenizer_max_length = 4
        pooled = torch.ones(1, 1, 3, dtype=torch.float32)
        embeddings = {
            "pooled_prompt_embeds": pooled,
            "prompt_embeds": torch.zeros(1, 4, 2),
            "attention_masks": torch.ones(1, 4, dtype=torch.int32),
        }

        unpacked = self.model.unpack_text_embeddings_from_cache(embeddings)
        self.assertEqual(unpacked["pooled_prompt_embeds"].shape, torch.Size([1, 3]))
        self.assertTrue(torch.equal(unpacked["pooled_prompt_embeds"], pooled[:, 0, :]))

    def test_model_predict_without_visual_conditioning_preserves_channels(self):
        self.model.model = make_tiny_kandinsky_transformer(visual_cond=False)
        noisy_latents = torch.randn(1, 16, 2, 2)
        prepared_batch = {
            "noisy_latents": noisy_latents,
            "encoder_hidden_states": torch.randn(1, 2, 8),
            "timesteps": torch.tensor([0.0]),
            "added_cond_kwargs": {"text_embeds": torch.randn(1, 4)},
        }

        output = self.model.model_predict(prepared_batch)["model_prediction"]

        self.assertEqual(output.shape, (1, 16, 2, 2))

    def test_i2i_prepare_latents_sets_condition_and_mask(self):
        pipeline = Kandinsky5I2IPipeline(
            transformer=make_tiny_kandinsky_transformer(visual_cond=True),
            vae=DummyImageVAE(),
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            scheduler=DummyScheduler(),
        )
        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=4,
            width=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
            image=torch.ones(3, 8, 8),
        )

        self.assertEqual(latents.shape, (1, 1, 4, 4, 33))
        self.assertTrue(torch.allclose(latents[:, 0, :, :, 16:32], torch.ones_like(latents[:, 0, :, :, 16:32])))
        self.assertTrue(torch.all(latents[:, 0, :, :, 32:] == 1.0))

    def test_t2i_pipeline_call_runs_with_embeddings(self):
        transformer = make_tiny_kandinsky_transformer(visual_cond=False)
        pipeline = Kandinsky5T2IPipeline(
            transformer=transformer,
            vae=DummyImageVAE(),
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            scheduler=DummyScheduler(),
        )
        object.__setattr__(pipeline, "interrupt", False)
        pipeline._interrupt = False

        prompt_embeds_qwen = torch.randn(1, 2, 8)
        prompt_embeds_clip = torch.randn(1, 4)
        prompt_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

        output = pipeline(
            prompt=None,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            height=16,
            width=16,
            num_inference_steps=1,
            guidance_scale=1.0,
            output_type="latent",
            return_dict=True,
        )

        self.assertEqual(output.frames.shape, (1, 1, 16, 16, 16))

    def test_encode_prompts_passes_expected_args_to_pipeline(self):
        pipeline = DummyPromptPipeline()
        self.model.pipelines = {PipelineTypes.TEXT2IMG: pipeline}
        self.model.config.tokenizer_max_length = 8

        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens, attention_mask = self.model._encode_prompts(
            ["a prompt", "b prompt"]
        )

        self.assertEqual(len(pipeline.calls), 1)
        call = pipeline.calls[0]
        self.assertEqual(call["num_images_per_prompt"], 1)
        self.assertEqual(call["max_sequence_length"], 8)
        self.assertEqual(call["device"], self.accelerator.device)
        self.assertTrue(torch.equal(prompt_cu_seqlens, torch.tensor([0, 1, 2], dtype=torch.int32)))
        self.assertEqual(attention_mask.shape, torch.Size([2, prompt_embeds_qwen.shape[1]]))
        self.assertEqual(prompt_embeds_clip.shape, torch.Size([2, 4]))


class Kandinsky5VideoModelTests(unittest.TestCase):
    def setUp(self):
        self.scheduler_patch = patch(
            "diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",
            side_effect=lambda *args, **kwargs: _make_flow_scheduler(),
        )
        self.scheduler_patch.start()
        self.addCleanup(self.scheduler_patch.stop)

        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            model_family="kandinsky5-video",
            model_flavour=None,
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=1.0,
            rescale_betas_zero_snr=False,
            training_scheduler_timestep_spacing=None,
            prediction_type=None,
            weight_dtype=torch.float32,
            controlnet=False,
            control=False,
            tokenizer_max_length=None,
            framerate=None,
        )
        self.model = Kandinsky5Video(self.config, self.accelerator)
        self.model.unwrap_model = lambda model=None: model or self.model
        self.model.model = make_tiny_kandinsky_transformer(visual_cond=False)

    def test_convert_text_embed_for_pipeline_video_builds_cu_seqlens(self):
        embeds = torch.randn(1, 2, 6)
        pooled = torch.randn(1, 4)
        attention_mask = torch.tensor([[1, 0]], dtype=torch.int32)
        text_embedding = {
            "prompt_embeds": embeds,
            "pooled_prompt_embeds": pooled,
            "attention_masks": attention_mask,
        }

        converted = self.model.convert_text_embed_for_pipeline(text_embedding, prompt="ok")
        expected_cu = torch.tensor([0, 1], dtype=torch.int32)

        self.assertTrue(torch.equal(converted["prompt_embeds_qwen"], embeds))
        self.assertTrue(torch.equal(converted["prompt_embeds_clip"], pooled))
        self.assertTrue(torch.equal(converted["prompt_cu_seqlens"], expected_cu))

        negative = self.model.convert_negative_text_embed_for_pipeline(text_embedding, prompt="ok")
        self.assertTrue(torch.equal(negative["negative_prompt_cu_seqlens"], expected_cu))

    def test_check_user_config_sets_default_framerate_and_tokenizer_cap(self):
        self.model.check_user_config()
        self.assertEqual(self.model.config.framerate, 24)
        self.assertEqual(self.model.config.tokenizer_max_length, 256)

        self.model.config.tokenizer_max_length = 999
        self.model.config.framerate = None
        self.model.check_user_config()
        self.assertEqual(self.model.config.tokenizer_max_length, 256)
        self.assertEqual(self.model.config.framerate, 24)

    def test_model_predict_video_visual_cond_expands_channels(self):
        self.model.model = make_tiny_kandinsky_transformer(visual_cond=True)
        noisy_latents = torch.randn(1, 16, 2, 2, 2)
        cond_latents = torch.ones(1, 16, 2, 2)
        prepared_batch = {
            "noisy_latents": noisy_latents,
            "conditioning_latents": cond_latents,
            "encoder_hidden_states": torch.randn(1, 2, 8),
            "timesteps": torch.tensor([0.1]),
            "added_cond_kwargs": {"text_embeds": torch.randn(1, 4)},
        }

        output = self.model.model_predict(prepared_batch)["model_prediction"]

        self.assertEqual(output.shape, (1, 16, 2, 2, 2))

    def test_i2v_prepare_latents_sets_first_frame_conditioning(self):
        pipeline = Kandinsky5I2VPipeline(
            transformer=make_tiny_kandinsky_transformer(visual_cond=True),
            vae=DummyVideoVAE(temporal_compression_ratio=1, spatial_compression_ratio=1),
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            scheduler=DummyScheduler(),
        )

        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=8,
            width=8,
            num_frames=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
            image=torch.ones(3, 8, 8),
        )

        self.assertEqual(latents.shape, (1, 3, 8, 8, 33))
        self.assertTrue(torch.allclose(latents[:, 0, :, :, 16:32], torch.ones_like(latents[:, 0, :, :, 16:32])))
        self.assertTrue(torch.all(latents[:, 0, :, :, 32:] == 1.0))
        self.assertTrue(torch.all(latents[:, 1:, :, :, 32:] == 0.0))

    def test_t2v_pipeline_call_runs_with_embeddings(self):
        transformer = make_tiny_kandinsky_transformer(visual_cond=False)
        pipeline = Kandinsky5T2VPipeline(
            transformer=transformer,
            vae=DummyVideoVAE(temporal_compression_ratio=1, spatial_compression_ratio=1),
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            scheduler=DummyScheduler(),
        )
        pipeline._interrupt = False

        prompt_embeds_qwen = torch.randn(1, 2, 8)
        prompt_embeds_clip = torch.randn(1, 4)
        prompt_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

        output = pipeline(
            prompt=None,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            height=16,
            width=16,
            num_frames=3,
            num_inference_steps=1,
            guidance_scale=1.0,
            output_type="latent",
            return_dict=True,
        )

        self.assertEqual(output.frames.shape, (1, 4, 16, 16, 16))

    def test_encode_prompts_video_uses_pipeline_and_builds_mask(self):
        pipeline = DummyPromptPipeline()
        self.model.pipelines = {PipelineTypes.TEXT2IMG: pipeline}
        self.model.config.tokenizer_max_length = 6

        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens, attention_mask = self.model._encode_prompts(["clip"])

        self.assertEqual(len(pipeline.calls), 1)
        call = pipeline.calls[0]
        self.assertEqual(call["num_videos_per_prompt"], 1)
        self.assertEqual(call["max_sequence_length"], 6)
        self.assertEqual(call["device"], self.accelerator.device)
        self.assertTrue(torch.equal(prompt_cu_seqlens, torch.tensor([0, 1], dtype=torch.int32)))
        self.assertEqual(attention_mask.shape, torch.Size([1, prompt_embeds_qwen.shape[1]]))
        self.assertEqual(prompt_embeds_clip.shape, torch.Size([1, 4]))

    def test_pack_and_unpack_text_embeddings_preserve_pooled_embeds_video(self):
        self.model.config.tokenizer_max_length = 6
        prompt_embeds = torch.arange(1 * 6 * 2, dtype=torch.float32).view(1, 6, 2)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int32)
        pooled = torch.tensor([[5.0, 4.0, 3.0, 2.0]], dtype=torch.float32)
        embeddings = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "attention_masks": attention_mask,
        }

        packed = self.model.pack_text_embeddings_for_cache(embeddings)
        self.assertEqual(packed["prompt_embeds"].shape, torch.Size([1, 4, 2]))
        self.assertEqual(packed["attention_masks"].shape, torch.Size([1, 4]))
        self.assertIn("_pad_slices", packed)
        self.assertTrue(torch.equal(packed["_pad_slices"]["prompt_embeds"], prompt_embeds[:, 3:4]))
        self.assertTrue(torch.equal(packed["pooled_prompt_embeds"], pooled))

        unpacked = self.model.unpack_text_embeddings_from_cache(packed)
        self.assertEqual(unpacked["prompt_embeds"].shape, torch.Size([1, 6, 2]))
        self.assertTrue(torch.equal(unpacked["prompt_embeds"][:, :4], prompt_embeds[:, :4]))
        pad_token = prompt_embeds[:, 3:4].expand(-1, 2, -1)
        self.assertTrue(torch.equal(unpacked["prompt_embeds"][:, 4:], pad_token))
        self.assertTrue(torch.equal(unpacked["attention_masks"], torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int32)))
        self.assertTrue(torch.equal(unpacked["pooled_prompt_embeds"], pooled))

    def test_unpack_text_embeddings_squeezes_pooled_prompt_embeds_video(self):
        self.model.config.tokenizer_max_length = 4
        pooled = torch.ones(1, 1, 3, dtype=torch.float32)
        embeddings = {
            "pooled_prompt_embeds": pooled,
            "prompt_embeds": torch.zeros(1, 4, 2),
            "attention_masks": torch.ones(1, 4, dtype=torch.int32),
        }

        unpacked = self.model.unpack_text_embeddings_from_cache(embeddings)
        self.assertEqual(unpacked["pooled_prompt_embeds"].shape, torch.Size([1, 3]))
        self.assertTrue(torch.equal(unpacked["pooled_prompt_embeds"], pooled[:, 0, :]))


if __name__ == "__main__":
    unittest.main()
