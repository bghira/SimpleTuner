from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from simpletuner.helpers.models.longcat_video import optimized_scale, pack_video_latents, unpack_video_latents
from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEFAULT_VAE_SCALE_FACTOR_TEMPORAL = 4
DEFAULT_VAE_SCALE_FACTOR_SPATIAL = 8
DEFAULT_VAE_Z_DIM = 16
DEFAULT_VAE_LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
DEFAULT_VAE_LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
]


@dataclass
class LongCatVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, List["np.ndarray"]]  # type: ignore[name-defined]


def _retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class LongCatVideoPipeline(DiffusionPipeline):
    """
    Pipeline for LongCat-Video text-to-video and image-to-video generation.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "latents_packed", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: Optional[AutoencoderKLWan],
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer: LongCatVideoTransformer3DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        (
            latents_mean_config,
            latents_std_config,
            z_dim,
            vae_scale_factor_temporal,
            vae_scale_factor_spatial,
        ) = self._resolve_vae_config(vae)

        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.max_tokenizer_len = 512

        latents_mean = torch.tensor(latents_mean_config, dtype=torch.float32).view(1, z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(latents_std_config, dtype=torch.float32).view(1, z_dim, 1, 1, 1)
        latents_mean.requires_grad = False
        latents_std.requires_grad = False
        self.latents_mean = latents_mean
        self.latents_std = latents_std

    def _resolve_vae_config(self, vae: Optional[AutoencoderKLWan]) -> Tuple[List[float], List[float], int, int, int]:
        vae_config = getattr(vae, "config", None)

        latents_mean = getattr(vae_config, "latents_mean", None) or DEFAULT_VAE_LATENTS_MEAN
        latents_std = getattr(vae_config, "latents_std", None) or DEFAULT_VAE_LATENTS_STD
        z_dim = getattr(vae_config, "z_dim", None) or DEFAULT_VAE_Z_DIM
        vae_scale_factor_temporal = getattr(vae_config, "scale_factor_temporal", DEFAULT_VAE_SCALE_FACTOR_TEMPORAL)
        vae_scale_factor_spatial = getattr(vae_config, "scale_factor_spatial", DEFAULT_VAE_SCALE_FACTOR_SPATIAL)

        if len(latents_mean) != z_dim or len(latents_std) != z_dim:
            raise ValueError(
                f"VAE latent statistics length ({len(latents_mean)}, {len(latents_std)}) " f"does not match z_dim ({z_dim})."
            )

        return latents_mean, latents_std, z_dim, vae_scale_factor_temporal, vae_scale_factor_spatial

    def _require_vae(self) -> AutoencoderKLWan:
        if self.vae is None:
            raise ValueError("VAE is not loaded; load a VAE before encoding or decoding latents.")
        return self.vae

    def _get_attention_mask(self, attention_mask: torch.Tensor, target_batch: int, num_videos_per_prompt: int):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt, 1, 1)
        attention_mask = attention_mask.view(target_batch, 1, 1, -1)
        return attention_mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return getattr(self, "_num_timesteps", 0)

    @property
    def current_timestep(self):
        return getattr(self, "_current_timestep", None)

    @property
    def interrupt(self):
        return getattr(self, "_interrupt", False)

    def _pad_prompt_embeds(self, prompt_embeds: torch.Tensor):
        target_dim = getattr(self.transformer.config, "caption_channels", prompt_embeds.shape[-1])
        if prompt_embeds.shape[-1] == target_dim:
            return prompt_embeds
        if prompt_embeds.shape[-1] < target_dim:
            pad = target_dim - prompt_embeds.shape[-1]
            prompt_embeds = F.pad(prompt_embeds, (0, pad))
        else:
            prompt_embeds = prompt_embeds[..., :target_dim]
        return prompt_embeds

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Union[str, List[str]] = None,
        max_sequence_length: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        max_length = max_sequence_length or self.max_tokenizer_len

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)

        target_dtype = dtype or self.text_encoder.dtype
        if device.type == "mps" and target_dtype == torch.float64:
            target_dtype = torch.float32

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-1].to(device=device, dtype=target_dtype)
        prompt_embeds = self._pad_prompt_embeds(prompt_embeds[:, :max_length, :])

        prompt_embeds = prompt_embeds.unsqueeze(1)
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, max_length, -1)
        prompt_attention_mask = self._get_attention_mask(attention_mask, prompt_embeds.shape[0], num_videos_per_prompt).to(
            device
        )

        if not do_classifier_free_guidance:
            return prompt_embeds, prompt_attention_mask, None, None

        neg_prompt = negative_prompt or ""
        neg_prompt = batch_size * [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt
        negative_tokenized = self.tokenizer(
            neg_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        negative_input_ids = negative_tokenized.input_ids.to(device)
        negative_attention_mask = negative_tokenized.attention_mask.to(device)

        negative_outputs = self.text_encoder(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_outputs.hidden_states[-1].to(device=device, dtype=target_dtype)
        negative_prompt_embeds = self._pad_prompt_embeds(negative_prompt_embeds[:, :max_length, :])
        negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, 1, max_length, -1)
        negative_attention_mask = self._get_attention_mask(
            negative_attention_mask, negative_prompt_embeds.shape[0], num_videos_per_prompt
        ).to(device)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_attention_mask

    def _normalize_latents(self, latents: torch.Tensor, device: torch.device, dtype: torch.dtype):
        latents_mean = self.latents_mean.to(device=device, dtype=dtype)
        latents_std = self.latents_std.to(device=device, dtype=dtype)
        return pack_video_latents(latents, latents_mean, latents_std)

    def _denormalize_latents(self, latents: torch.Tensor, device: torch.device, dtype: torch.dtype):
        latents_mean = self.latents_mean.to(device=device, dtype=dtype)
        latents_std = self.latents_std.to(device=device, dtype=dtype)
        return unpack_video_latents(latents, latents_mean, latents_std)

    def _pack_latents(self, latents: torch.Tensor):
        return self._normalize_latents(latents, device=latents.device, dtype=latents.dtype)

    def _unpack_latents(self, latents: torch.Tensor):
        return self._denormalize_latents(latents, device=latents.device, dtype=latents.dtype)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_cond_frames: int = 0,
        num_cond_frames_added: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        if (image is not None) and (video is not None):
            raise ValueError("Provide only one of `image` or `video` for conditioning.")

        if latents is None:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        cond_count = 0
        if image is not None or video is not None:
            condition_data = image if image is not None else video
            is_image = image is not None
            cond_latents = []
            for i in range(batch_size):
                gen = generator[i] if isinstance(generator, list) else generator
                if is_image:
                    encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                else:
                    encoded_input = condition_data[i][:, -(num_cond_frames - num_cond_frames_added) :].unsqueeze(0)
                if num_cond_frames_added > 0:
                    pad_front = encoded_input[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                    encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                vae = self._require_vae()
                latent = _retrieve_latents(vae.encode(encoded_input), gen)
                cond_latents.append(latent)

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
            cond_latents = self._normalize_latents(cond_latents, device=device, dtype=dtype)

            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
            latents[:, :, :num_cond_latents] = cond_latents
            cond_count = num_cond_latents

        return latents, cond_count

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ("latents",),
        max_sequence_length: Optional[int] = None,
    ):
        if prompt is None:
            raise ValueError("`prompt` must not be None.")

        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if height % scale_factor_spatial != 0 or width % scale_factor_spatial != 0:
            raise ValueError(
                f"`height and width` have to be divisible by {scale_factor_spatial} but are {height} and {width}."
            )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                "`num_frames - 1` has to be divisible by %s. Rounding to the nearest valid value.",
                self.vae_scale_factor_temporal,
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else torch.float32
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
            device=device,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                (
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                    _,
                    _,
                ) = self.encode_prompt(
                    prompt=[""] * batch_size,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    dtype=dtype,
                    device=device,
                )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0).to(device)
        else:
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_mask = prompt_attention_mask.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_channels

        image_tensor = None
        if image is not None:
            image_tensor = self.video_processor.preprocess(image, height=height, width=width)
            if image_tensor.dim() == 5:
                image_tensor = image_tensor.permute(0, 2, 1, 3, 4)
            elif image_tensor.dim() == 4:
                image_tensor = image_tensor.unsqueeze(2)
            else:
                raise ValueError(f"Unsupported conditioning image tensor shape: {tuple(image_tensor.shape)}")
            image_tensor = image_tensor.to(device=device, dtype=torch.float32)

        latents, cond_count = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
            image=image_tensor,
        )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dtype=self.transformer.dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                if cond_count > 0:
                    timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep[:, :cond_count] = 0

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=cond_count,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    B = noise_pred_text.shape[0]
                    positive = noise_pred_text.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    st_star = optimized_scale(positive, negative).view(B, 1, 1, 1, 1)
                    noise_pred = noise_pred_uncond * st_star + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond * st_star
                    )

                noise_pred = -noise_pred

                if cond_count > 0:
                    latents[:, :, cond_count:] = self.scheduler.step(
                        noise_pred[:, :, cond_count:], t, latents[:, :, cond_count:], return_dict=False
                    )[0]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    latents_for_callback = self._unpack_latents(latents)
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                            callback_kwargs[k] = latents_for_callback
                        elif k == "latents_packed":
                            callback_kwargs[k] = latents
                        elif k == "prompt_embeds":
                            callback_kwargs[k] = prompt_embeds
                        elif k == "negative_prompt_embeds":
                            callback_kwargs[k] = negative_prompt_embeds if self.do_classifier_free_guidance else None
                        else:
                            callback_kwargs[k] = locals().get(k, None)
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    cb_latents = callback_outputs.pop("latents", None)
                    if cb_latents is not None:
                        latents = self._pack_latents(cb_latents).to(device=latents.device, dtype=latents.dtype)
                    cb_latents_packed = callback_outputs.pop("latents_packed", None)
                    if cb_latents_packed is not None:
                        latents = cb_latents_packed.to(device=latents.device, dtype=latents.dtype)
                    if callback_outputs:
                        logger.debug(
                            "Callback returned additional tensors that were ignored: %s",
                            list(callback_outputs.keys()),
                        )

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            video = latents
        else:
            vae = self._require_vae()
            latents_for_decode = self._unpack_latents(latents).to(vae.dtype)
            video = vae.decode(latents_for_decode, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        return LongCatVideoPipelineOutput(videos=video)
