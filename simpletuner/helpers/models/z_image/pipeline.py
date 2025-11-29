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
import math
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PreTrainedModel

from .pipeline_output import ZImagePipelineOutput
from .transformer import ZImageTransformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImagePipeline

        >>> pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(42)).images[0]
        >>> image.save("zimage.png")
        ```
"""

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True


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


# cfg zero* optimization from sd3
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot product
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


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


class ZImageLoraLoaderMixin(LoraBaseMixin):
    _lora_loadable_modules = ["transformer", "controlnet"]
    transformer_name = "transformer"
    controlnet_name = "controlnet"

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dict, _ = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        return state_dict

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name=None,
        hotswap: bool = False,
        **kwargs,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        transformer_state_dict = {}
        controlnet_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith(f"{self.controlnet_name}."):
                controlnet_state_dict[k] = v
            else:
                transformer_state_dict[k] = v

        if transformer_state_dict:
            self.load_lora_into_transformer(
                transformer_state_dict,
                transformer=(
                    getattr(self, self.transformer_name) if hasattr(self, self.transformer_name) else self.transformer
                ),
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
                prefix=kwargs.get("prefix", None),
            )

        if controlnet_state_dict and hasattr(self, self.controlnet_name):
            self.load_lora_into_controlnet(
                controlnet_state_dict,
                controlnet=getattr(self, self.controlnet_name),
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        prefix: Optional[str] = None,
    ):
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Normalize common prefixes from training scripts (e.g., diffusers uses "unet", some trainers use "diffusion_model")
        if any(k.startswith("diffusion_model.") for k in state_dict):
            state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
            prefix = None
            logger.info("Stripped 'diffusion_model.' prefix from LoRA keys for transformer loading.")
        elif any(k.startswith("transformer.") for k in state_dict):
            state_dict = {k.replace("transformer.", "", 1): v for k, v in state_dict.items()}
            prefix = None
            logger.info("Stripped 'transformer.' prefix from LoRA keys for transformer loading.")

        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            prefix=prefix,
        )

    @classmethod
    def load_lora_into_controlnet(
        cls,
        state_dict,
        controlnet,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
    ):
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        keys = list(state_dict.keys())
        controlnet_keys = [k for k in keys if k.startswith(cls.controlnet_name)]
        state_dict = {k.replace(f"{cls.controlnet_name}.", ""): v for k, v in state_dict.items() if k in controlnet_keys}

        if len(state_dict.keys()) > 0:
            logger.info(f"Loading {cls.controlnet_name}.")

            if hasattr(controlnet, "load_lora_adapter"):
                controlnet.load_lora_adapter(
                    state_dict,
                    network_alphas=None,
                    adapter_name=adapter_name,
                    _pipeline=_pipeline,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    hotswap=hotswap,
                )
            elif hasattr(controlnet, "load_attn_procs"):
                controlnet.load_attn_procs(
                    state_dict,
                    network_alphas=None,
                    adapter_name=adapter_name,
                    _pipeline=_pipeline,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
            else:
                raise AttributeError(
                    "ControlNet model does not have a method to load LoRA weights. "
                    "Please ensure your ControlNet model supports LoRA loading."
                )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        controlnet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
        controlnet_lora_adapter_metadata: Optional[dict] = None,
    ):
        state_dict = {}
        lora_adapter_metadata = {}

        if not (transformer_lora_layers or controlnet_lora_layers):
            raise ValueError("You must pass at least one of `transformer_lora_layers` or `controlnet_lora_layers`.")

        if transformer_lora_layers:
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

        if controlnet_lora_layers:
            state_dict.update(cls.pack_weights(controlnet_lora_layers, cls.controlnet_name))

        if transformer_lora_adapter_metadata:
            lora_adapter_metadata.update(cls.pack_weights(transformer_lora_adapter_metadata, cls.transformer_name))

        if controlnet_lora_adapter_metadata:
            lora_adapter_metadata.update(cls.pack_weights(controlnet_lora_adapter_metadata, cls.controlnet_name))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            lora_adapter_metadata=lora_adapter_metadata,
        )

    def fuse_lora(
        self,
        components: List[str] = ["transformer", "controlnet"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
        )

    def unfuse_lora(self, components: List[str] = ["transformer", "controlnet"], **kwargs):
        super().unfuse_lora(components=components, **kwargs)


class ZImagePipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
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
                dtype=dtype,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:

        assert num_images_per_prompt == 1
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
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

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

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

    @staticmethod
    def value_from_time_aware_config(config, t):
        if isinstance(config, (float, int, str)):
            return config
        elif isinstance(config, torch.Tensor):
            assert config.numel() == 1
            return config.item()
        elif isinstance(config, (tuple, list)):
            assert isinstance(config[0], (float, int, str))
            assert all([isinstance(x, (tuple, list, str)) for x in config[1:]])
            result = config[0]
            for thresh, val in config[1:]:
                if t >= thresh:
                    result = val
                else:
                    break
            return result
        else:
            raise ValueError(f"invalid time-aware config {config} of type {type(config)}")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
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
        skip_guidance_layers: Optional[List[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        use_cfg_zero_star: bool = True,
        use_zero_init: bool = True,
        zero_steps: int = 0,
        no_cfg_until_timestep: int = 0,
        cfg_end_timestep: Optional[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply configuration normalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                The truncation value for configuration.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.ZImagePipelineOutput`] instead of a
                plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*): Layer indices to skip when computing guidance. Guidance
                will be applied to the remaining layers only during the configured window.
            skip_layer_guidance_scale (`float`, *optional*): Scale for the skipped-layer guidance correction when
                `skip_guidance_layers` is provided.
            skip_layer_guidance_stop (`float`, *optional*): Fraction of inference after which skipped-layer guidance is
                disabled.
            skip_layer_guidance_start (`float`, *optional*): Fraction of inference before which skipped-layer guidance
                is disabled.
            use_cfg_zero_star (`bool`, *optional*): Whether to use the CFG Zero* optimization when applying guidance.
            use_zero_init (`bool`, *optional*): When `use_cfg_zero_star` is enabled, optionally zero the guidance for
                the first `zero_steps` steps.
            zero_steps (`int`, *optional*): Number of initial steps to zero out guided predictions when using CFG
                Zero*.
            no_cfg_until_timestep (`int`, *optional*): Skip classifier-free guidance until this (zero-indexed) step.
            cfg_end_timestep (`int`, *optional*): If provided, disable classifier-free guidance after this step.

        Examples:

        Returns:
            [`~pipelines.z_image.ZImagePipelineOutput`] or `tuple`:
            [`~pipelines.z_image.ZImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        height = height or 1024
        width = width or 1024
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(f"Z-Image requires height and width to be divisible by {vae_scale} (got {height}x{width}).")

        dtype = getattr(self.transformer, "dtype", torch.float32)
        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        # 2. Define call parameters
        # Determine batch size and handle precomputed embeddings.
        if prompt_embeds is not None:
            if isinstance(prompt_embeds, list):
                batch_size = len(prompt_embeds)
            else:
                batch_size = prompt_embeds.shape[0]
                prompt_embeds = list(prompt_embeds)
            prompt_embeds = [pe.to(device=device, dtype=dtype) for pe in prompt_embeds]

            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                # Synthesize unconditional embeddings for CFG when only positive embeds are supplied.
                negative_prompt_embeds = self._encode_prompt(
                    prompt=[""] * batch_size,
                    device=device,
                    dtype=dtype,
                    num_images_per_prompt=num_images_per_prompt,
                    prompt_embeds=None,
                    max_sequence_length=max_sequence_length,
                )
            elif negative_prompt_embeds is not None:
                if isinstance(negative_prompt_embeds, list):
                    pass
                else:
                    negative_prompt_embeds = list(negative_prompt_embeds)
                negative_prompt_embeds = [ne.to(device=device, dtype=dtype) for ne in negative_prompt_embeds]
        else:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")

            lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                dtype=dtype,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # Repeat embeddings for num_images_per_prompt, if needed.
        if num_images_per_prompt and num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [ne for ne in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = len(prompt_embeds)

        # 4. Prepare latent variables
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
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] / 2)

        # 5. Prepare timesteps
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

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    current_guidance_scale = self.value_from_time_aware_config(
                        (self.guidance_scale, (self._cfg_truncation, 0.0)), t_norm
                    )

                # Run CFG only if configured AND scale is non-zero
                within_cfg_window = i >= no_cfg_until_timestep and (cfg_end_timestep is None or i <= cfg_end_timestep)
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0 and within_cfg_window
                if apply_cfg and not negative_prompt_embeds:
                    raise ValueError(
                        "Classifier-free guidance is enabled but negative_prompt_embeds are missing. "
                        "Provide `negative_prompt` or `negative_prompt_embeds` when supplying precomputed embeddings."
                    )
                if apply_cfg and len(negative_prompt_embeds) != actual_batch_size:
                    raise ValueError(
                        f"Expected {actual_batch_size} negative_prompt_embeds for CFG but received "
                        f"{len(negative_prompt_embeds)}."
                    )

                if apply_cfg:
                    # Prepare inputs for CFG
                    latent_model_input = torch.cat([latents.to(dtype)] * 2)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = torch.cat([timestep] * 2)
                else:
                    latent_model_input = latents.to(dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                )[0]

                if apply_cfg:
                    # Perform CFG
                    pos_out = torch.stack([out.float() for out in model_out_list[:actual_batch_size]], dim=0)
                    neg_out = torch.stack([out.float() for out in model_out_list[actual_batch_size:]], dim=0)

                    if use_cfg_zero_star:
                        pos_flat = pos_out.view(actual_batch_size, -1)
                        neg_flat = neg_out.view(actual_batch_size, -1)
                        alpha = optimized_scale(pos_flat, neg_flat).view(actual_batch_size, *([1] * (pos_out.dim() - 1)))
                        if i <= zero_steps and use_zero_init:
                            guided = pos_out * 0.0
                        else:
                            guided = neg_out * alpha + current_guidance_scale * (pos_out - neg_out * alpha)
                    else:
                        guided = pos_out + current_guidance_scale * (pos_out - neg_out)

                    # Renormalization
                    if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos_out.view(actual_batch_size, -1), dim=1)
                        new_pos_norm = torch.linalg.vector_norm(guided.view(actual_batch_size, -1), dim=1)
                        max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                        scale_mask = new_pos_norm > max_new_norm
                        if torch.any(scale_mask):
                            scale = (max_new_norm / (new_pos_norm + 1e-8)).view(
                                actual_batch_size, *([1] * (guided.dim() - 1))
                            )
                            guided = torch.where(
                                scale_mask.view(actual_batch_size, *([1] * (guided.dim() - 1))), guided * scale, guided
                            )

                    noise_pred = guided
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                # Optional skipped-layer guidance correction
                should_skip_layers = (
                    apply_cfg
                    and skip_guidance_layers is not None
                    and len(skip_guidance_layers) > 0
                    and i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                )
                if should_skip_layers:
                    skip_latent_input = latents.to(dtype).unsqueeze(2)
                    skip_latent_list = list(skip_latent_input.unbind(dim=0))
                    skip_out = self.transformer(
                        skip_latent_list,
                        timestep,
                        prompt_embeds,
                        skip_layers=skip_guidance_layers,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                    )[0]
                    skip_pred = torch.stack([out.float() for out in skip_out], dim=0).squeeze(2)
                    skip_pred = -skip_pred
                    noise_pred = noise_pred + (pos_out.squeeze(2) - skip_pred) * self._skip_layer_guidance_scale

                # compute the previous noisy sample x_t -> x_t-1
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

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = latents.to(dtype)
        if output_type == "latent":
            image = latents

        else:
            latents = latents.to(dtype=self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
