import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import math
import einops
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, HiDreamImageLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from helpers.models.hidream.schedule import FlowUniPCMultistepScheduler

from dataclasses import dataclass
from typing import List, Union
from diffusers.utils import BaseOutput

import numpy as np
import PIL.Image

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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


@dataclass
class HiDreamImagePipelineOutput(BaseOutput):
    """
    Output class for HiDreamImage pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class HiDreamImagePipeline(
    DiffusionPipeline, FromSingleFileMixin, HiDreamImageLoraLoaderMixin
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
        text_encoder_4: LlamaForCausalLM,
        tokenizer_4: PreTrainedTokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            text_encoder_4=text_encoder_4,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        # HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.default_sample_size = 128
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 text encoder embeddings for the given prompt.

        Args:
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            T5 embeddings tensor of shape [batch_size, seq_len, dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[
                    :,
                    min(max_sequence_length, self.tokenizer_3.model_max_length)
                    - 1 : -1,
                ]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(
            text_input_ids.to(device), attention_mask=attention_mask.to(device)
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get CLIP text encoder embeddings for the given prompt.

        Args:
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            CLIP embeddings tensor of shape [batch_size, embedding_dim]
        """
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, 128),
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 128 - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {128} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def _get_llama3_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get Llama text encoder embeddings for the given prompt.

        Args:
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            Llama embeddings tensor of shape [num_layers, batch_size, seq_len, dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_4.batch_decode(
                untruncated_ids[
                    :,
                    min(max_sequence_length, self.tokenizer_4.model_max_length)
                    - 1 : -1,
                ]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
        )

        # Get all hidden states (layers) and stack them for the transformer to select
        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        _, _, seq_len, dim = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            -1, batch_size * num_images_per_prompt, seq_len, dim
        )
        return prompt_embeds

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

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
        """
        Prepare latents for denoising.

        Args:
            batch_size: Batch size
            num_channels_latents: Number of channels in latents
            height: Image height
            width: Image width
            dtype: Data type for latents
            device: Device to place latents on
            generator: Random number generator
            latents: Optional existing latents to use

        Returns:
            Prepared latent tensors
        """
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)
        return latents

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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        """
        Encode prompt into model embeddings needed for transformer.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            device: Device to place embeddings on
            dtype: Data type for embeddings
            num_images_per_prompt: Number of images to generate per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            negative_prompt: Negative prompt for CLIP L/14
            negative_prompt_2: Negative prompt for CLIP G/14
            negative_prompt_3: Negative prompt for T5
            negative_prompt_4: Negative prompt for Llama
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            negative_t5_prompt_embeds: Pre-computed negative T5 prompt embeddings
            negative_llama_prompt_embeds: Pre-computed negative Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings
            max_sequence_length: Maximum sequence length for tokenization
            lora_scale: Scale for LoRA weights

        Returns:
            Tuple containing:
            - t5_prompt_embeds: T5 encoder embeddings
            - llama_prompt_embeds: Llama encoder embeddings
            - negative_t5_prompt_embeds: Negative T5 encoder embeddings (if using guidance)
            - negative_llama_prompt_embeds: Negative Llama encoder embeddings (if using guidance)
            - pooled_prompt_embeds: Pooled CLIP embeddings
            - negative_pooled_prompt_embeds: Negative pooled CLIP embeddings (if using guidance)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            # If no prompt is provided, determine batch size from embeddings
            if t5_prompt_embeds is not None:
                batch_size = t5_prompt_embeds.shape[0]
            elif llama_prompt_embeds is not None:
                # Handle based on expected shape format
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    batch_size = llama_prompt_embeds.shape[1]
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    batch_size = llama_prompt_embeds.shape[0]
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                raise ValueError(
                    "Either prompt or pre-computed embeddings must be provided"
                )

        # Check if we need to compute embeddings or use provided ones
        if (
            t5_prompt_embeds is None
            or llama_prompt_embeds is None
            or pooled_prompt_embeds is None
        ):
            t5_prompt_embeds, llama_prompt_embeds, pooled_prompt_embeds = (
                self._encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_3=prompt_3,
                    prompt_4=prompt_4,
                    device=device,
                    dtype=dtype,
                    num_images_per_prompt=num_images_per_prompt,
                    t5_prompt_embeds=t5_prompt_embeds,
                    llama_prompt_embeds=llama_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    max_sequence_length=max_sequence_length,
                )
            )

        # Handle negative embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            if (
                negative_t5_prompt_embeds is None
                or negative_llama_prompt_embeds is None
                or negative_pooled_prompt_embeds is None
            ):
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt
                negative_prompt_3 = negative_prompt_3 or negative_prompt
                negative_prompt_4 = negative_prompt_4 or negative_prompt

                # normalize str to list
                negative_prompt = (
                    batch_size * [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
                negative_prompt_2 = (
                    batch_size * [negative_prompt_2]
                    if isinstance(negative_prompt_2, str)
                    else negative_prompt_2
                )
                negative_prompt_3 = (
                    batch_size * [negative_prompt_3]
                    if isinstance(negative_prompt_3, str)
                    else negative_prompt_3
                )
                negative_prompt_4 = (
                    batch_size * [negative_prompt_4]
                    if isinstance(negative_prompt_4, str)
                    else negative_prompt_4
                )

                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )

                (
                    negative_t5_prompt_embeds,
                    negative_llama_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self._encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_3=negative_prompt_3,
                    prompt_4=negative_prompt_4,
                    device=device,
                    dtype=dtype,
                    num_images_per_prompt=num_images_per_prompt,
                    t5_prompt_embeds=negative_t5_prompt_embeds,
                    llama_prompt_embeds=negative_llama_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    max_sequence_length=max_sequence_length,
                )

        return (
            t5_prompt_embeds,
            llama_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
    ):
        """
        Internal method to encode prompts to embeddings for the model.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            device: Device to place embeddings on
            dtype: Data type for embeddings
            num_images_per_prompt: Number of images to generate per prompt
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            max_sequence_length: Maximum sequence length for tokenization

        Returns:
            Tuple containing:
            - t5_prompt_embeds: T5 encoder embeddings
            - llama_prompt_embeds: Llama encoder embeddings
            - pooled_prompt_embeds: Pooled CLIP embeddings
        """
        device = device or self._execution_device

        # Check if we need to compute any embeddings
        need_pooled = pooled_prompt_embeds is None
        need_t5 = t5_prompt_embeds is None
        need_llama = llama_prompt_embeds is None

        if need_pooled or need_t5 or need_llama:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            # Get CLIP embeddings for pooled embeddings if needed
            if need_pooled:
                # Get CLIP L/14 embeddings
                pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                    self.tokenizer,
                    self.text_encoder,
                    prompt=prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

                # Get CLIP G/14 embeddings
                pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                    self.tokenizer_2,
                    self.text_encoder_2,
                    prompt=prompt_2,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

                # Concatenate CLIP embeddings
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
                )

            # Get T5 embeddings if needed
            if need_t5:
                t5_prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

            # Get Llama embeddings if needed
            if need_llama:
                llama_prompt_embeds = self._get_llama3_prompt_embeds(
                    prompt=prompt_4,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        return t5_prompt_embeds, llama_prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ):
        """
        Generate images based on text prompts.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            sigmas: Optional custom sigmas for scheduler
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Negative prompt for CLIP L/14
            negative_prompt_2: Negative prompt for CLIP G/14
            negative_prompt_3: Negative prompt for T5
            negative_prompt_4: Negative prompt for Llama
            num_images_per_prompt: Number of images to generate per prompt
            generator: Random number generator
            latents: Optional existing latents to use
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            negative_t5_prompt_embeds: Pre-computed negative T5 prompt embeddings
            negative_llama_prompt_embeds: Pre-computed negative Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings
            output_type: Output type - "pil", "latent", or "pt"
            return_dict: Whether to return as a dictionary
            joint_attention_kwargs: Additional attention parameters
            callback_on_step_end: Callback after each denoising step
            callback_on_step_end_tensor_inputs: Tensor inputs to pass to callback
            max_sequence_length: Maximum sequence length for tokenization

        Returns:
            Generated images
        """
        # 1. Set up image dimensions and scales
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(
            height * scale // division * division
        )

        # 2. Set up parameters for generation
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 3. Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # Determine batch size from embeddings
            if t5_prompt_embeds is not None:
                batch_size = t5_prompt_embeds.shape[0]
            elif llama_prompt_embeds is not None:
                # Handle based on expected shape format
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    batch_size = llama_prompt_embeds.shape[1]
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    batch_size = llama_prompt_embeds.shape[0]
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                raise ValueError("Either prompt or embeddings must be provided")

        device = self.transformer.device

        # 4. Encode prompts
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            t5_prompt_embeds,
            llama_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            t5_prompt_embeds=t5_prompt_embeds,
            llama_prompt_embeds=llama_prompt_embeds,
            negative_t5_prompt_embeds=negative_t5_prompt_embeds,
            negative_llama_prompt_embeds=negative_llama_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 5. Prepare embeddings for guidance if needed
        if self.do_classifier_free_guidance:
            # Format embeddings for the transformer which expects separate inputs
            # Handle T5 embeddings (shape: [batch, seq_len, dim])
            if negative_t5_prompt_embeds is not None:
                t5_embeds_input = torch.cat(
                    [negative_t5_prompt_embeds, t5_prompt_embeds], dim=0
                )
            else:
                t5_embeds_input = t5_prompt_embeds

            # Handle Llama embeddings
            if negative_llama_prompt_embeds is not None:
                # The shape handling depends on the format of llama embeddings
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    llama_embeds_input = torch.cat(
                        [negative_llama_prompt_embeds, llama_prompt_embeds], dim=1
                    )
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    llama_embeds_input = torch.cat(
                        [negative_llama_prompt_embeds, llama_prompt_embeds], dim=0
                    )
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                llama_embeds_input = llama_prompt_embeds

            # Combine embeddings for passing to transformer
            pooled_embeds_input = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        else:
            # If not using guidance, use the embeddings directly
            t5_embeds_input = t5_prompt_embeds
            llama_embeds_input = llama_prompt_embeds
            pooled_embeds_input = pooled_prompt_embeds

        # 6. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_embeds_input.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare spatial data for non-square images
        if latents.shape[-2] != latents.shape[-1]:
            B, C, H, W = latents.shape
            pH, pW = (
                H // self.transformer.config.patch_size,
                W // self.transformer.config.patch_size,
            )

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(latents.device)
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device)
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None

        # 8. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(
                num_inference_steps, device=device, shift=math.exp(mu)
            )
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 9. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Reshape latents for transformer if needed
                if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                    B, C, H, W = latent_model_input.shape
                    patch_size = self.transformer.config.patch_size
                    pH, pW = H // patch_size, W // patch_size
                    out = torch.zeros(
                        (B, C, self.transformer.max_seq, patch_size * patch_size),
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device,
                    )
                    latent_model_input = einops.rearrange(
                        latent_model_input,
                        "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                        p1=patch_size,
                        p2=patch_size,
                    )
                    out[:, :, 0 : pH * pW] = latent_model_input
                    latent_model_input = out

                # Call transformer with the updated input format
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    t5_hidden_states=t5_embeds_input,
                    llama_hidden_states=llama_embeds_input,
                    pooled_embeds=pooled_embeds_input,
                    img_sizes=img_sizes,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    t5_embeds_input = callback_outputs.pop(
                        "t5_embeds_input", t5_embeds_input
                    )
                    llama_embeds_input = callback_outputs.pop(
                        "llama_embeds_input", llama_embeds_input
                    )
                    pooled_embeds_input = callback_outputs.pop(
                        "pooled_embeds_input", pooled_embeds_input
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 10. Post-processing
        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(
                latents.to(dtype=self.vae.dtype, device=self.vae.device),
                return_dict=False,
            )[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HiDreamImagePipelineOutput(images=image)
