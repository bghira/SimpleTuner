# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
from typing import Callable, Dict, List, Optional, Union

import regex as re
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from simpletuner.helpers.models.kandinsky_lora_loader import KandinskyLoraLoaderMixin

from .pipeline_output import KandinskyPipelineOutput
from .transformer_kandinsky5 import Kandinsky5Transformer3DModel

logger = logging.get_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


if is_ftfy_available():
    import ftfy


def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class Kandinsky5T2IPipeline(DiffusionPipeline, KandinskyLoraLoaderMixin):
    """
    Text-to-image pipeline for Kandinsky 5 (Diffusers port). Minimal version for training/inference in SimpleTuner.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_qwen",
        "prompt_embeds_clip",
        "negative_prompt_embeds_qwen",
        "negative_prompt_embeds_clip",
    ]

    def __init__(
        self,
        transformer: Kandinsky5Transformer3DModel,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2VLProcessor,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        self.prompt_template = "\n".join(
            [
                "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ]
        )
        self.prompt_template_encode_start_idx = 41
        self.image_processor = None
        if getattr(self, "vae", None) is not None:
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def vae_scale_factor_spatial(self):
        vae = getattr(self, "vae", None)
        if vae is None or getattr(vae, "config", None) is None:
            return 1.0
        return getattr(vae.config, "scaling_factor", 1.0)

    @property
    def guidance_scale(self):
        return getattr(self, "_guidance_scale", 1.0)

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1.0

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text encoder/tokenizer must be loaded to encode prompts.")
        dtype = dtype or self.text_encoder.dtype

        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = [prompt_clean(p) for p in prompt]
        batch_size = len(prompt)

        # Qwen
        prompt_embeds_qwen, prompt_cu_seqlens = self._encode_prompt_qwen(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
        )
        # CLIP
        prompt_embeds_clip = self._encode_prompt_clip(prompt=prompt, device=device, dtype=dtype)

        # Repeat for num_images_per_prompt
        prompt_embeds_qwen = prompt_embeds_qwen.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_qwen = prompt_embeds_qwen.view(batch_size * num_images_per_prompt, -1, prompt_embeds_qwen.shape[-1])
        prompt_embeds_clip = prompt_embeds_clip.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_clip = prompt_embeds_clip.view(batch_size * num_images_per_prompt, -1)

        original_lengths = prompt_cu_seqlens.diff()
        repeated_lengths = original_lengths.repeat_interleave(num_images_per_prompt)
        repeated_cu_seqlens = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), repeated_lengths.cumsum(0)])

        return prompt_embeds_qwen, prompt_embeds_clip, repeated_cu_seqlens

    def _encode_prompt_qwen(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        full_texts = [self.prompt_template.format(p) for p in prompt]

        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        embeds = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )[
            "hidden_states"
        ][-1][:, self.prompt_template_encode_start_idx :]

        attention_mask = inputs["attention_mask"][:, self.prompt_template_encode_start_idx :]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32)

        return embeds.to(dtype), cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        if self.text_encoder_2 is None or self.tokenizer_2 is None:
            raise ValueError("Secondary text encoder/tokenizer must be loaded to encode prompts.")
        dtype = dtype or self.text_encoder_2.dtype

        inputs = self.tokenizer_2(
            prompt,
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        pooled_embed = self.text_encoder_2(**inputs)["pooler_output"]
        return pooled_embed.to(dtype)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 1024,
        width: int = 1024,
        num_frames: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,  # accepted for I2I overrides
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            (num_frames - 1) // 1 + 1,  # keep same interface as video
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            num_channels_latents,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds_qwen=None,
        prompt_embeds_clip=None,
        negative_prompt_embeds_qwen=None,
        negative_prompt_embeds_clip=None,
        prompt_cu_seqlens=None,
        negative_prompt_cu_seqlens=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt_embeds_qwen is not None or prompt_embeds_clip is not None or prompt_cu_seqlens is not None:
            if prompt_embeds_qwen is None or prompt_embeds_clip is None or prompt_cu_seqlens is None:
                raise ValueError("If any prompt embed is provided, all prompt and cu_seqlens must be provided.")
        if (
            negative_prompt_embeds_qwen is not None
            or negative_prompt_embeds_clip is not None
            or negative_prompt_cu_seqlens is not None
        ):
            if (
                negative_prompt_embeds_qwen is None
                or negative_prompt_embeds_clip is None
                or negative_prompt_cu_seqlens is None
            ):
                raise ValueError("If any negative prompt embed is provided, all negative prompt embeds must be provided.")

        if prompt is None and prompt_embeds_qwen is None:
            raise ValueError("Provide either `prompt` or prompt embeddings.")

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    @replace_example_docstring("")
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds_qwen: Optional[torch.Tensor] = None,
        prompt_embeds_clip: Optional[torch.Tensor] = None,
        negative_prompt_embeds_qwen: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clip: Optional[torch.Tensor] = None,
        prompt_cu_seqlens: Optional[torch.Tensor] = None,
        negative_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        """
        Run Kandinsky 5 text-to-image generation.

        Examples:
            This placeholder is replaced at import time.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            negative_prompt_embeds_qwen=negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip=negative_prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_qwen.shape[0]

        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device
        dtype = self.transformer.dtype

        if prompt_embeds_qwen is None:
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = self.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt) if prompt is not None else [negative_prompt]
            elif len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have same length as `prompt`. Got {len(negative_prompt)} vs {len(prompt)}."
                )

            if negative_prompt_embeds_qwen is None:
                negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_prompt_cu_seqlens = self.encode_prompt(
                    prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_visual_dim
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames=1,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
            image=kwargs.get("image"),
        )

        text_rope_pos = torch.arange(prompt_cu_seqlens.diff().max().item(), device=device)
        negative_text_rope_pos = (
            torch.arange(negative_prompt_cu_seqlens.diff().max().item(), device=device)
            if negative_prompt_cu_seqlens is not None
            else None
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.unsqueeze(0).repeat(batch_size * num_images_per_prompt)
                visual_rope_pos = [
                    torch.arange(1, device=device),
                    torch.arange(height // self.vae_scale_factor_spatial // 2, device=device),
                    torch.arange(width // self.vae_scale_factor_spatial // 2, device=device),
                ]

                pred_velocity = self.transformer(
                    hidden_states=latents.to(dtype),
                    encoder_hidden_states=prompt_embeds_qwen.to(dtype),
                    pooled_projections=prompt_embeds_clip.to(dtype),
                    timestep=timestep.to(dtype),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=(1, 2, 2),
                    sparse_params=None,
                    return_dict=True,
                ).sample

                if self.do_classifier_free_guidance and negative_prompt_embeds_qwen is not None:
                    uncond_pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype),
                        encoder_hidden_states=negative_prompt_embeds_qwen.to(dtype),
                        pooled_projections=negative_prompt_embeds_clip.to(dtype),
                        timestep=timestep.to(dtype),
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=negative_text_rope_pos,
                        scale_factor=(1, 2, 2),
                        sparse_params=None,
                        return_dict=True,
                    ).sample

                    pred_velocity = uncond_pred_velocity + guidance_scale * (pred_velocity - uncond_pred_velocity)

                latents[:, :, :, :, :num_channels_latents] = self.scheduler.step(
                    pred_velocity, t, latents[:, :, :, :, :num_channels_latents], return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_qwen = callback_outputs.pop("prompt_embeds_qwen", prompt_embeds_qwen)
                    prompt_embeds_clip = callback_outputs.pop("prompt_embeds_clip", prompt_embeds_clip)
                    negative_prompt_embeds_qwen = callback_outputs.pop(
                        "negative_prompt_embeds_qwen", negative_prompt_embeds_qwen
                    )
                    negative_prompt_embeds_clip = callback_outputs.pop(
                        "negative_prompt_embeds_clip", negative_prompt_embeds_clip
                    )

                if i == len(timesteps) - 1 or ((i + 1) > len(timesteps) - num_inference_steps * self.scheduler.order):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = latents[:, :, :, :, :num_channels_latents]

        if output_type != "latent":
            if self.vae is None:
                raise ValueError("VAE is not loaded; set output_type='latent' or load the VAE to decode images.")
            latents = latents.to(self.vae.dtype)
            images = latents.reshape(
                batch_size,
                num_images_per_prompt,
                1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                num_channels_latents,
            )
            images = images.permute(0, 1, 5, 2, 3, 4)
            images = images.reshape(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            images = images / getattr(self.vae.config, "scaling_factor", 1.0)
            images = self.vae.decode(images).sample
            if self.image_processor is None:
                images = images
            else:
                images = self.image_processor.postprocess_image(images, output_type=output_type)
        else:
            images = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return KandinskyPipelineOutput(frames=images)
