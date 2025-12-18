# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import torch
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PreTrainedModel, Siglip2ImageProcessorFast, Siglip2VisionModel

from simpletuner.helpers.models.z_image.pipeline import ZImageLoraLoaderMixin

from .pipeline_output import ZImagePipelineOutput
from .transformer import ZImageOmniTransformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImageOmniPipeline

        >>> pipe = ZImageOmniPipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in PyTorch.
        >>> # (1) Use flash attention 2
        >>> # pipe.transformer.set_attention_backend("flash")
        >>> # (2) Use flash attention 3
        >>> # pipe.transformer.set_attention_backend("_flash_3")

        >>> prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        >>> image = pipe(
        ...     prompt,
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=9,
        ...     guidance_scale=0.0,
        ...     generator=torch.Generator("cuda").manual_seed(42),
        ... ).images[0]
        >>> image.save("zimage.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ZImageOmniPipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageOmniTransformer2DModel,
        siglip: Siglip2VisionModel,
        siglip_processor: Siglip2ImageProcessorFast,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
            siglip=siglip,
            siglip_processor=siglip_processor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        num_condition_images: int = 0,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            num_condition_images=num_condition_images,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                num_condition_images=num_condition_images,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
        num_condition_images: int = 0,
    ) -> List[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            if num_condition_images == 0:
                prompt[i] = ["<|im_start|>user\n" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n"]
            elif num_condition_images > 0:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
                prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|im_end|>"]
                prompt[i] = prompt_list

        flattened_prompt = []
        prompt_list_lengths = []

        for i in range(len(prompt)):
            prompt_list_lengths.append(len(prompt[i]))
            flattened_prompt.extend(prompt[i])

        text_inputs = self.tokenizer(
            flattened_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []
        start_idx = 0
        for i in range(len(prompt_list_lengths)):
            batch_embeddings = []
            end_idx = start_idx + prompt_list_lengths[i]
            for j in range(start_idx, end_idx):
                batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx = end_idx

        return embeddings_list

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
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    def prepare_image_latents(
        self,
        images: List[torch.Tensor],
        batch_size,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent = (
                self.vae.encode(image.bfloat16()).latent_dist.mode()[0] - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            image_latent = image_latent.unsqueeze(1).to(dtype)
            image_latents.append(image_latent)

        image_latents = [image_latents.copy() for _ in range(batch_size)]

        return image_latents

    def prepare_siglip_embeds(
        self,
        images: List[torch.Tensor],
        batch_size,
        device,
        dtype,
    ):
        siglip_embeds = []
        for image in images:
            siglip_inputs = self.siglip_processor(images=[image], return_tensors="pt").to(device)
            shape = siglip_inputs.spatial_shapes[0]
            hidden_state = self.siglip(**siglip_inputs).last_hidden_state
            _, _, c = hidden_state.shape
            hidden_state = hidden_state[:, : shape[0] * shape[1]]
            hidden_state = hidden_state.view(shape[0], shape[1], c)
            siglip_embeds.append(hidden_state.to(dtype))

        siglip_embeds = [siglip_embeds.copy() for _ in range(batch_size)]

        return siglip_embeds

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if image is not None and not isinstance(image, list):
            image = [image]
        num_condition_images = len(image) if image is not None else 0

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
                num_condition_images=num_condition_images,
            )

        condition_images = []
        resized_images = []
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    if height is not None and width is not None:
                        img = self.image_processor._resize_to_target_area(img, height * width)
                    else:
                        img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                resized_images.append(img)

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)

            if len(condition_images) > 0:
                height = height or image_height
                width = width or image_width
        else:
            height = height or 1024
            width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        condition_latents = self.prepare_image_latents(
            images=condition_images,
            batch_size=batch_size * num_images_per_prompt,
            device=device,
            dtype=torch.float32,
        )
        condition_latents = [[lat.to(self.transformer.dtype) for lat in lats] for lats in condition_latents]
        if self.do_classifier_free_guidance:
            negative_condition_latents = [[lat.clone() for lat in batch] for batch in condition_latents]

        condition_siglip_embeds = self.prepare_siglip_embeds(
            images=resized_images,
            batch_size=batch_size * num_images_per_prompt,
            device=device,
            dtype=torch.float32,
        )
        condition_siglip_embeds = [[se.to(self.transformer.dtype) for se in sels] for sels in condition_siglip_embeds]
        if self.do_classifier_free_guidance:
            negative_condition_siglip_embeds = [[se.clone() for se in batch] for batch in condition_siglip_embeds]

        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        condition_siglip_embeds = [None if sels == [] else sels + [None] for sels in condition_siglip_embeds]
        if self.do_classifier_free_guidance:
            negative_condition_siglip_embeds = [
                None if sels == [] else sels + [None] for sels in negative_condition_siglip_embeds
            ]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                t_norm = timestep[0].item()

                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    condition_latents_model_input = condition_latents + negative_condition_latents
                    condition_siglip_embeds_model_input = condition_siglip_embeds + negative_condition_siglip_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    condition_latents_model_input = condition_latents
                    condition_siglip_embeds_model_input = condition_siglip_embeds
                    timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    condition_latents_model_input,
                    condition_siglip_embeds_model_input,
                    return_dict=False,
                )[0]

                if apply_cfg:
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
