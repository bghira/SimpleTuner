# Copyright 2024 AuraFlow Authors and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5Tokenizer, UMT5EncoderModel

from diffusers.loaders.lora_base import (  # noqa
    LORA_WEIGHT_NAME,
    LORA_WEIGHT_NAME_SAFE,
    LoraBaseMixin,
    _fetch_state_dict,
    _load_lora_into_text_encoder,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AuraFlowTransformer2DModel, AutoencoderKL
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AuraFlowPipeline

        >>> pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("aura_flow.png")
        ```
"""


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


# Function from SD3 for CFG Zero* optimization
def optimized_scale(positive_flat, negative_flat):
    # Compute the norm of the positive and negative prompts
    positive_norm = torch.norm(positive_flat, dim=1, keepdim=True)
    negative_norm = torch.norm(negative_flat, dim=1, keepdim=True)

    # Compute the cosine similarity between positive and negative prompts
    pos_dot_neg = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    similarity = pos_dot_neg / (positive_norm * negative_norm + 1e-8)

    # Compute the scaling factors
    alpha = torch.clamp(similarity, 0, 1)
    return alpha
import os
import PIL
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    get_submodule_by_name,
    is_bitsandbytes_available,
    is_gguf_available,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
)
TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"
from huggingface_hub.utils import validate_hf_hub_args
_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

class AuraFlowLoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`AuraFlowTransformer2DModel`] Specific to [`AuraFlowPipeline`].
    """

    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    @validate_hf_hub_args
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.lora_state_dict
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # transformer and text encoder or both.
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

        state_dict = _fetch_state_dict(
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

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.load_lora_weights
    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer` and
        `self.text_encoder`. All kwargs are forwarded to `self.lora_state_dict`. See
        [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] for more details on how the state
        dict is loaded into `self.transformer`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin.load_lora_into_transformer with SD3Transformer2DModel->AuraFlowTransformer2DModel
    def load_lora_into_transformer(
        cls, state_dict, transformer, adapter_name=None, _pipeline=None, low_cpu_mem_usage=False, hotswap: bool = False
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            transformer (`AuraFlowTransformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            hotswap : (`bool`, *optional*)
                Defaults to `False`. Whether to substitute an existing (LoRA) adapter with the newly loaded adapter
                in-place. This means that, instead of loading an additional adapter, this will take the existing
                adapter weights and replace them with the weights of the new adapter. This can be faster and more
                memory efficient. However, the main advantage of hotswapping is that when the model is compiled with
                torch.compile, loading the new adapter does not require recompilation of the model. When using
                hotswapping, the passed `adapter_name` should be the name of an already loaded adapter.

                If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need
                to call an additional method before loading the adapter:

                ```py
                pipeline = ...  # load diffusers pipeline
                max_rank = ...  # the highest rank among all LoRAs that you want to load
                # call *before* compiling and loading the LoRA adapter
                pipeline.enable_lora_hotswap(target_rank=max_rank)
                pipeline.load_lora_weights(file_name)
                # optionally compile the model now
                ```

                Note that hotswapping adapters of the text encoder is not yet supported. There are some further
                limitations to this technique, which are documented here:
                https://huggingface.co/docs/peft/main/en/package_reference/hotswap
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Load the layers corresponding to transformer.
        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        state_dict = {}

        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers`.")

        if transformer_lora_layers:
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.loaders.lora_pipeline.SanaLoraLoaderMixin.fuse_lora
    def fuse_lora(
        self,
        components: List[str] = ["transformer"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    # Copied from diffusers.loaders.lora_pipeline.SanaLoraLoaderMixin.unfuse_lora
    def unfuse_lora(self, components: List[str] = ["transformer", "text_encoder"], **kwargs):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
            unfuse_transformer (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
        """
        super().unfuse_lora(components=components, **kwargs)

class AuraFlowPipeline(DiffusionPipeline, AuraFlowLoraLoaderMixin):
    r"""
    Args:
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. AuraFlow uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [EleutherAI/pile-t5-xl](https://huggingface.co/EleutherAI/pile-t5-xl) variant.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        transformer ([`AuraFlowTransformer2DModel`]):
            Conditional Transformer (MMDiT and DiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKL,
        transformer: AuraFlowTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.patch_size = (
            2  # AuraFlow default patch size (similar to SD3's patch_size attribute)
        )
        self._interrupt = False  # Adding interrupt property for compatibility with SD3

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            )

        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            max_sequence_length (`int`, defaults to 256): Maximum sequence length to use for the prompt.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, AuraFlowLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        if device is None:
            device = self._execution_device
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        max_length = max_sequence_length
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            text_input_ids = text_inputs["input_ids"]
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because T5 can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            prompt_embeds = self.text_encoder(**text_inputs)[0]
            prompt_attention_mask = (
                text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
            )
            prompt_embeds = prompt_embeds * prompt_attention_mask

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.reshape(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens = (
                [negative_prompt] * batch_size
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
            negative_prompt_embeds = self.text_encoder(**uncond_input)[0]
            negative_prompt_attention_mask = (
                uncond_input["attention_mask"]
                .unsqueeze(-1)
                .expand(negative_prompt_embeds.shape)
            )
            negative_prompt_embeds = (
                negative_prompt_embeds * negative_prompt_attention_mask
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = negative_prompt_attention_mask.reshape(
                bs_embed, -1
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                num_images_per_prompt, 1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        if self.text_encoder is not None:
            if isinstance(self, AuraFlowLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def skip_layer_guidance_scale(self):
        return self._skip_layer_guidance_scale

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
        image=None,
        timestep=None,
        is_strength_max=True,
    ):
        """
        Prepare latents for diffusion. If image is provided, encode it using the VAE.

        Args:
            batch_size (int): Batch size
            num_channels_latents (int): Number of channels in latent space
            height (int): Height of the generated image
            width (int): Width of the generated image
            dtype (torch.dtype): Data type to use
            device (torch.device): Device to use
            generator (torch.Generator): Random number generator
            latents (torch.FloatTensor, optional): Precomputed latents to use
            image (Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]], optional): Image to use for img2img
            timestep (int, optional): Timestep for noise level when using img2img
            is_strength_max (bool, optional): Whether strength is max (1.0) for faster processing
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if image is None or is_strength_max:
            # Generate random latents if no image provided or strength is 1.0
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )

            # If image is provided but strength is 1.0, we still need to encode the image
            # but we'll apply full noise to it, so the result will be just noise
            if image is not None:
                image_latents = self._encode_vae_image(image, generator)
                image_latents = image_latents.to(device=device, dtype=dtype)
        else:
            # Encode image using VAE
            image_latents = self._encode_vae_image(image, generator)
            image_latents = image_latents.to(device=device, dtype=dtype)

            # Add noise to image latents based on timestep
            noise = randn_tensor(
                image_latents.shape, generator=generator, device=device, dtype=dtype
            )

            # Apply noise to latents according to timestep
            latents = self.scheduler.add_noise(image_latents, noise, timestep)

        return latents

    def _encode_vae_image(self, image, generator=None):
        """
        Encode an input image into latent space using the VAE.

        Args:
            image (Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]]): Input image to encode
            generator (torch.Generator, optional): Random number generator for reproducibility

        Returns:
            torch.FloatTensor: Encoded latents
        """
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.preprocess(image)

        # Move to same device as VAE
        image = image.to(device=self.vae.device, dtype=self.vae.dtype)

        # VAE encode
        image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

        # Scale and shift latents according to VAE configuration
        if hasattr(self.vae.config, "scaling_factor") and hasattr(
            self.vae.config, "shift_factor"
        ):
            image_latents = (
                image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        else:
            # Fallback for older VAE models
            image_latents = image_latents * 0.18215

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # New parameters from SD3
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: int = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        use_cfg_zero_star: Optional[bool] = True,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 0,
        # New img2img parameters
        image: Optional[
            Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]]
        ] = None,
        strength: Optional[float] = 1.0,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*): A list of integers that specify layers to skip during guidance.
                If not provided, all layers will be used for guidance. If provided, the guidance will only be applied
                to the layers specified in the list.
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers with
                a scale of `1`.
            skip_layer_guidance_stop (`float`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`.
            skip_layer_guidance_start (`float`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`.
            mu (`float`, *optional*): Parameter for certain schedulers that use dynamic shifting.
            use_cfg_zero_star (`bool`, *optional*): Whether to use the CFG Zero* optimization.
            use_zero_init (`bool`, *optional*): Whether to use zero initialization for the first few steps when using CFG Zero*.
            zero_steps (`int`, *optional*): Number of steps to use zero initialization when using CFG Zero*.
            image (`torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                Image or images to use for image-to-image generation. If provided, `strength` will be used to blend the
                noise with the image latents.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to transform the input image. Must be between 0 and 1.
                `strength` = 0 means the input image is returned as-is, while `strength` = 1 means the input image is
                completely transformed according to the prompt.
        """
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 1.1 Check image and strength inputs for img2img
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should be in [0.0, 1.0] but is {strength}"
            )

        if image is not None and not isinstance(
            image, (torch.Tensor, PIL.Image.Image, list)
        ):
            raise ValueError(
                f"image must be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        if image is not None and isinstance(image, list) and len(image) != batch_size:
            raise ValueError(
                f"If image is a list, it must have the same length as the batch size {batch_size}, but got {len(image)} images."
            )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._interrupt = False  # Initialize interrupt property

        # 2. Determine batch size.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.transformer.device
        lora_scale = (
            self.attention_kwargs.get("scale", None)
            if self.attention_kwargs is not None
            else None
        )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                # Store original prompt embeds for skip layer guidance
                original_prompt_embeds = prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        # Handle dynamic shifting if needed
        scheduler_kwargs = {}
        if (
            hasattr(self.scheduler.config, "use_dynamic_shifting")
            and self.scheduler.config.get("use_dynamic_shifting", None)
            and mu is None
        ):
            _, _, height, width = (
                latents.shape if latents is not None else (None, None, height, width)
            )
            # If dynamic shifting is supported, calculate mu based on image dimensions
            if hasattr(self.scheduler.config, "base_image_seq_len"):
                image_seq_len = (height // self.patch_size) * (width // self.patch_size)
                mu = self._calculate_shift(
                    image_seq_len,
                    self.scheduler.config.base_image_seq_len,
                    self.scheduler.config.max_image_seq_len,
                    self.scheduler.config.base_shift,
                    self.scheduler.config.max_shift,
                )
                scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
            sigmas=sigmas,
            **scheduler_kwargs,
        )

        # 5. Set timesteps for img2img if needed
        # Img2img: Modify timestep sequence based on strength
        if image is not None and strength < 1.0:
            # Get the original timestep to start from based on strength
            init_timestep = min(
                int(num_inference_steps * (1.0 - strength)), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)

            # Get the corresponding timestep value
            timesteps = timesteps[t_start:]
            num_inference_steps = len(timesteps)

            # Get starting timestep for prepare_latents
            timestep_for_latents = timesteps[0]
        else:
            timestep_for_latents = None

        # 6. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=image,
            timestep=timestep_for_latents,
            is_strength_max=strength >= 1.0,
        )

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )

                # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = torch.tensor([t / 1000]).expand(latent_model_input.shape[0])
                timestep = timestep.to(latents.device, dtype=latents.dtype)

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                    # attention_kwargs=self.attention_kwargs,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    if use_cfg_zero_star:
                        # CFG Zero* optimization from SD3
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_pred_uncond.view(batch_size, -1)

                        alpha = optimized_scale(positive_flat, negative_flat)
                        alpha = alpha.view(batch_size, 1, 1, 1)
                        alpha = alpha.to(positive_flat.dtype)

                        if (i <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_pred_uncond * alpha + guidance_scale * (
                                noise_pred_text - noise_pred_uncond * alpha
                            )
                    else:
                        # Standard CFG
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # Skip layer guidance (SLG) implementation
                    should_skip_layers = (
                        True
                        if skip_guidance_layers is not None
                        and i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )

                    if should_skip_layers:
                        # Only run inference on the original prompt with skip_layers
                        timestep_single = torch.tensor([t / 1000]).expand(
                            latents.shape[0]
                        )
                        timestep_single = timestep_single.to(
                            latents.device, dtype=latents.dtype
                        )

                        latent_model_input_single = latents

                        # Add skip_layers parameter to transformer call
                        noise_pred_skip_layers = self.transformer(
                            latent_model_input_single,
                            encoder_hidden_states=original_prompt_embeds,
                            timestep=timestep_single,
                            return_dict=False,
                            attention_kwargs=self.attention_kwargs,
                            skip_layers=skip_guidance_layers,
                        )[0]

                        # Apply skip layer guidance
                        noise_pred = (
                            noise_pred
                            + (noise_pred_text - noise_pred_skip_layers)
                            * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # Handle dtype mismatch that can occur on some platforms
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # Some platforms (eg. apple mps) misbehave due to a pytorch bug
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    if "negative_prompt_embeds" in callback_outputs:
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds"
                        )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            # Scale and shift latents according to VAE configuration
            if hasattr(self.vae.config, "scaling_factor") and hasattr(
                self.vae.config, "shift_factor"
            ) and getattr(self.vae.config, "shift_factor", None) is not None:
                latents = (
                    latents / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor
            else:
                # Fallback for older VAE models
                latents = latents / self.vae.config.scaling_factor

            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and getattr(
                self.vae.config, "force_upcast", False
            )
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            # Handle special SDPA attention types if available
            if hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa"):
                # we have SageAttention loaded. fallback to SDPA for decode.
                torch.nn.functional.scaled_dot_product_attention = (
                    torch.nn.functional.scaled_dot_product_attention_sdpa
                )

            image = self.vae.decode(
                latents.to(device=self.vae.device, dtype=self.vae.dtype),
                return_dict=False,
            )[0]

            if hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa"):
                # reenable SageAttention for training.
                torch.nn.functional.scaled_dot_product_attention = (
                    torch.nn.functional.scaled_dot_product_attention_sage
                )

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
