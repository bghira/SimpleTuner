import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from simpletuner.helpers.models.longcat_image import calculate_shift, prepare_pos_ids, retrieve_timesteps, split_quotation
from simpletuner.helpers.models.longcat_image.pipeline_output import LongCatImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    return int(width), int(height)


class LongCatImageEditPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    Pipeline for LongCat-Image editing / img2img generation.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor", "text_processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        text_processor: AutoProcessor,
        transformer,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            text_processor=text_processor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.image_processor_vl = text_processor.image_processor

        self.image_token = "<|image_pad|>"
        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 67
        self.prompt_template_encode_end_idx = 5
        self.default_sample_size = 128
        self.max_tokenizer_len = 512
        self.latent_channels = 16

    @torch.inference_mode()
    def encode_prompt(self, image, prompts, device, dtype):
        raw_vl_input = self.image_processor_vl(images=image, return_tensors="pt")
        pixel_values = raw_vl_input["pixel_values"]
        image_grid_thw = raw_vl_input["image_grid_thw"]

        prompts = [
            prompt.strip('"') if isinstance(prompt, str) and prompt.startswith('"') and prompt.endswith('"') else prompt
            for prompt in prompts
        ]
        all_tokens = []

        for clean_prompt_sub, matched in split_quotation(prompts[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)

        all_tokens = all_tokens[: self.max_tokenizer_len]
        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": [all_tokens]},
            max_length=self.max_tokenizer_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        text = self.prompt_template_encode_prefix

        merge_length = self.image_processor_vl.merge_size**2
        while self.image_token in text:
            num_image_tokens = image_grid_thw.prod() // merge_length
            text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", self.image_token)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat((prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1)

        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, self.prompt_template_encode_start_idx : -self.prompt_template_encode_end_idx, :]

        dtype_pos_ids = dtype
        if device.type == "mps" and dtype_pos_ids == torch.float64:
            dtype_pos_ids = torch.float32
        text_ids = prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(device, dtype=dtype_pos_ids)

        return prompt_embeds, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        prompt_length: Optional[int] = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        dtype_pos = torch.float64 if device.type != "mps" else torch.float32
        text_token_count = prompt_length if prompt_length is not None else self.max_tokenizer_len
        latent_image_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(text_token_count, text_token_count),
            height=height // 2,
            width=width // 2,
        ).to(device, dtype=dtype_pos)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device)
        latents = latents.to(dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        text_ids: Optional[torch.FloatTensor] = None,
        negative_text_ids: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        enable_cfg_renorm: Optional[bool] = True,
        cfg_renorm_min: Optional[float] = 0.0,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )
            pixel_step = self.vae_scale_factor * 2
            height = int(height / pixel_step) * pixel_step
            width = int(width / pixel_step) * pixel_step

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("A prompt or prompt_embeds must be provided.")

        device = self._execution_device
        prompt_length = prompt_embeds.shape[1] if prompt_embeds is not None else None

        if prompt_embeds is None:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = [negative_prompt] * num_images_per_prompt
            prompt = [prompt] * num_images_per_prompt if prompt is not None else [""]

            prompt_embeds, text_ids = self.encode_prompt(
                image=image,
                prompts=prompt,
                device=device,
                dtype=torch.float64,
            )
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                image=image,
                prompts=negative_prompt,
                device=device,
                dtype=torch.float64,
            )
            prompt_length = prompt_embeds.shape[1]
        else:
            if text_ids is None:
                text_ids = prepare_pos_ids(
                    modality_id=0,
                    type="text",
                    start=(0, 0),
                    num_token=prompt_length,
                ).to(device, dtype=torch.float64)
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_text_ids = negative_text_ids.to(device) if negative_text_ids is not None else text_ids
            prompt_length = prompt_embeds.shape[1]

        prompt_embeds_length = prompt_embeds.shape[1]

        prompt_embeds_dtype = prompt_embeds.dtype

        if image is None and image_latents is None:
            raise ValueError("`image` or `image_latents` must be provided for LongCat-Image editing.")

        if image_latents is None:
            image = self.image_processor.preprocess(image, height=height, width=width)
            image_latents = self.vae.encode(image.to(dtype=prompt_embeds_dtype)).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            image_latents = image_latents.to(device=self.device, dtype=prompt_embeds_dtype)

        if image_latents.shape[1] != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, found {image_latents.shape[1]}.")
        image_latents = image_latents.to(prompt_embeds.dtype)
        image_latents = self._pack_latents(
            image_latents,
            batch_size,
            self.latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        num_channels_latents = 16
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            prompt_length=prompt_length,
        )

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1] + image_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        guidance = None
        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                neg_prompt = negative_prompt
                if neg_prompt is None:
                    neg_prompt = [""] * batch_size
                elif isinstance(neg_prompt, str):
                    neg_prompt = [neg_prompt] * batch_size
                neg_dtype = prompt_embeds.dtype if prompt_embeds is not None else torch.float32
                if device.type == "mps" and neg_dtype == torch.float64:
                    neg_dtype = torch.float32
                negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                    image=image,
                    prompts=neg_prompt,
                    device=device,
                    dtype=neg_dtype,
                )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
        else:
            prompt_embeds = prompt_embeds.to(device)

        text_ids = text_ids.to(device) if text_ids is not None else None
        latent_image_ids = latent_image_ids.to(device)
        dtype_pos = torch.float64 if device.type != "mps" else torch.float32
        latent_image_ids_ref = prepare_pos_ids(
            modality_id=2,
            type="image",
            start=(prompt_embeds_length, prompt_embeds_length),
            height=height // self.vae_scale_factor // 2,
            width=width // self.vae_scale_factor // 2,
        ).to(device, dtype=dtype_pos)

        latent_image_ids = torch.cat([latent_image_ids, latent_image_ids_ref], dim=0)
        img_ids_cfg = (
            torch.cat([latent_image_ids, latent_image_ids], dim=0) if self.do_classifier_free_guidance else latent_image_ids
        )
        image_latents_input = (
            torch.cat([image_latents, image_latents], dim=0) if self.do_classifier_free_guidance else image_latents
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                latent_model_input = torch.cat([latent_model_input, image_latents_input], dim=1)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids_cfg,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred[:, : latents.shape[1]]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if enable_cfg_renorm:
                        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        noise_pred = noise_pred * scale

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    latents_for_callback = self._unpack_latents(
                        latents,
                        height,
                        width,
                        self.vae_scale_factor,
                    )
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                            callback_kwargs[k] = latents_for_callback
                        elif k == "latents_packed":
                            callback_kwargs[k] = latents
                        else:
                            callback_kwargs[k] = locals().get(k, None)
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    cb_latents = callback_outputs.pop("latents", None)
                    if cb_latents is not None:
                        if cb_latents.dim() == 4:
                            latents = self._pack_latents(
                                cb_latents,
                                batch_size=cb_latents.shape[0],
                                num_channels_latents=cb_latents.shape[1],
                                height=cb_latents.shape[2],
                                width=cb_latents.shape[3],
                            ).to(device=latents.device, dtype=latents.dtype)
                        elif cb_latents.dim() == 3:
                            latents = cb_latents.to(device=latents.device, dtype=latents.dtype)
                    if callback_outputs:
                        logger.debug(
                            "Callback returned additional tensors that were ignored: %s",
                            list(callback_outputs.keys()),
                        )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return LongCatImagePipelineOutput(images=image)
