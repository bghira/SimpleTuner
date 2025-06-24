from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.controlnets.controlnet import (
    ControlNetConditioningEmbedding,
    zero_module,
)
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from helpers.models.hidream.transformer import (
    HiDreamImageSingleTransformerBlock,
    HiDreamImageTransformerBlock,
)
from helpers.models.hidream.pipeline import HiDreamImageLoraLoaderMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class HiDreamControlNetOutput(BaseOutput):
    """
    The output of [`HiDreamControlNetModel`].

    Args:
        controlnet_block_samples (`Tuple[torch.Tensor]`):
            A tuple of output tensors from transformer blocks.
        controlnet_single_block_samples (`Tuple[torch.Tensor]`):
            A tuple of output tensors from single transformer blocks.
    """

    controlnet_block_samples: Tuple[torch.Tensor]
    controlnet_single_block_samples: Tuple[torch.Tensor]


from helpers.models.hidream.transformer import (
    PatchEmbed,
    OutEmbed,
    TimestepEmbed,
    PooledEmbed,
    EmbedND,
    HiDreamImageTransformerBlock,
    HiDreamImageSingleTransformerBlock,
)


class HiDreamControlNetOutput:
    def __init__(self, controlnet_block_samples, controlnet_single_block_samples):
        self.controlnet_block_samples = controlnet_block_samples
        self.controlnet_single_block_samples = controlnet_single_block_samples


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class HiDreamControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 2048,
        guidance_embeds: bool = False,
        max_seq_length: int = 16384,
        conditioning_embedding_channels: Optional[int] = None,
        axes_dims_rope: Tuple[int, int] = (32, 32),
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        self.max_seq = max_seq_length
        self.config.patch_size = patch_size
        self.config.axes_dims_rope = axes_dims_rope

        patch_dim = in_channels * patch_size * patch_size
        self.x_embedder = PatchEmbed(patch_size, in_channels, self.inner_dim)
        self.controlnet_x_embedder = zero_module(nn.Linear(patch_dim, self.inner_dim))

        self.t5_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.llama_embedder = nn.Linear(joint_attention_dim, self.inner_dim)

        self.t_embedder = TimestepEmbed(self.inner_dim)
        self.p_embedder = PooledEmbed(pooled_projection_dim, self.inner_dim)
        self.pe_embedder = EmbedND(theta=10000, axes_dim=list(axes_dims_rope))

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.controlnet_blocks = nn.ModuleList(
            [
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
                for _ in range(num_layers)
            ]
        )

        self.controlnet_single_blocks = nn.ModuleList(
            [
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
                for _ in range(num_single_layers)
            ]
        )

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls,
        transformer,
        load_weights_from_transformer: bool = True,
        use_shared_modules: bool = True,
        num_layers: Optional[int] = None,
        num_single_layers: Optional[int] = None,
    ):
        config = dict(transformer.config)
        config["joint_attention_dim"] = 4096
        config["num_layers"] = num_layers if num_layers is not None else len(transformer.double_stream_blocks)
        config["num_single_layers"] = num_single_layers if num_single_layers is not None else len(transformer.single_stream_blocks)
        logger.info(f"ControlNet will have {config['num_layers']} double stream and {config['num_single_layers']} single stream layers.")
        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            if use_shared_modules:
                ### We'll just apply shared references since its frozen:
                controlnet.t_embedder = transformer.t_embedder
                controlnet.p_embedder = transformer.p_embedder
                controlnet.x_embedder = transformer.x_embedder
                controlnet.pe_embedder = transformer.pe_embedder
                controlnet.double_stream_blocks = transformer.double_stream_blocks
                controlnet.single_stream_blocks = transformer.single_stream_blocks
            else:
                ### If we were to deepcopy instead:
                controlnet.t_embedder.load_state_dict(transformer.t_embedder.state_dict())
                controlnet.p_embedder.load_state_dict(transformer.p_embedder.state_dict())
                controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())

                controlnet.double_stream_blocks.load_state_dict(
                    transformer.double_stream_blocks.state_dict(), strict=False
                )
                controlnet.single_stream_blocks.load_state_dict(
                    transformer.single_stream_blocks.state_dict(), strict=False
                )
        cp = transformer.caption_projection
        controlnet.t5_embedder = cp[-1]
        controlnet.llama_embedder = cp[0]

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timesteps: torch.Tensor,
        t5_hidden_states: torch.Tensor,
        llama_hidden_states: torch.Tensor,
        pooled_embeds: torch.Tensor,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        img_ids: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.FloatTensor], HiDreamControlNetOutput]:

        B, C, H, W = hidden_states.shape
        patch_size = self.config.patch_size
        pH, pW = H // patch_size, W // patch_size
        patch_dim = C * patch_size * patch_size

        def patchify(x):
            return einops.rearrange(
                x,
                "B C (H p1) (W p2) -> B (H W) (p1 p2 C)",
                p1=patch_size,
                p2=patch_size,
            )

        hidden_states = patchify(hidden_states)
        controlnet_cond = patchify(controlnet_cond)

        hidden_states = self.x_embedder(hidden_states) + self.controlnet_x_embedder(
            controlnet_cond
        )
        temb = self.t_embedder(timesteps, hidden_states.dtype) + self.p_embedder(
            pooled_embeds
        )

        t5_embeds = self.t5_embedder(t5_hidden_states)
        llama_embeds = self.llama_embedder(llama_hidden_states)
        if llama_hidden_states.dim() == 5:
            llama_hidden_states = llama_hidden_states.squeeze(2)  # [B, L, S, D]
            selected = [
                llama_hidden_states[:, i]
                for i in range(min(2, llama_hidden_states.shape[1]))
            ]
        elif llama_hidden_states.dim() == 4:
            selected = [
                llama_hidden_states[i]
                for i in range(min(2, llama_hidden_states.shape[0]))
            ]
        else:
            raise ValueError(
                f"Unsupported llama_hidden_states shape: {llama_hidden_states.shape}"
            )
        llama_embeds = [self.llama_embedder(x) for x in selected]
        llama_embeds = torch.cat(llama_embeds, dim=1)

        encoder_hidden_states = torch.cat([t5_embeds, llama_embeds], dim=1)

        txt_ids = torch.zeros(
            B, encoder_hidden_states.shape[1], 3, device=hidden_states.device
        )
        img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
        img_ids[..., 1] = torch.arange(pH).unsqueeze(1)
        img_ids[..., 2] = torch.arange(pW).unsqueeze(0)
        img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=B)
        ids = torch.cat([img_ids, txt_ids], dim=1)
        rope = self.pe_embedder(ids)

        block_samples = []
        for i, block in enumerate(self.double_stream_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states, None, encoder_hidden_states, temb, rope
            )
            block_samples.append(hidden_states)

        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        single_block_samples = []
        for block in self.single_stream_blocks:
            rope = self.pe_embedder(
                torch.zeros(B, hidden_states.shape[1], 3, device=hidden_states.device)
            )
            hidden_states = block(hidden_states, None, None, temb, rope)
            single_block_samples.append(hidden_states[:, : pH * pW])

        controlnet_block_samples = [
            m(s) * conditioning_scale
            for m, s in zip(self.controlnet_blocks, block_samples)
        ]
        controlnet_single_block_samples = [
            m(s) * conditioning_scale
            for m, s in zip(self.controlnet_single_blocks, single_block_samples)
        ]

        if not return_dict:
            return controlnet_block_samples, controlnet_single_block_samples

        return HiDreamControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


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

from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from dataclasses import dataclass
from typing import List, Union
from diffusers.utils import BaseOutput
from helpers.models.hidream.schedule import FlowUniPCMultistepScheduler
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from helpers.models.flux.pipeline import retrieve_latents

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


class HiDreamEmbedder(nn.Module):
    """Embedder module that matches HiDream transformer's structure"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.proj(x)


@dataclass
class HiDreamControlNetPipelineOutput(BaseOutput):
    """
    Output class for HiDream ControlNet pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class HiDreamControlNetPipeline(
    DiffusionPipeline, FromSingleFileMixin, HiDreamImageLoraLoaderMixin
):
    r"""
    The HiDream pipeline for text-to-image generation with ControlNet.

    Args:
        transformer ([`HiDreamImageTransformer2DModel`]):
            Conditional Transformer (MM-DiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            First CLIP text encoder (CLIP-L/14)
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the first CLIP text encoder
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            Second CLIP text encoder (CLIP-G/14)
        tokenizer_2 ([`CLIPTokenizer`]):
            Tokenizer for the second CLIP text encoder
        text_encoder_3 ([`T5EncoderModel`]):
            T5 XXL text encoder
        tokenizer_3 ([`T5Tokenizer`]):
            Tokenizer for T5 XXL
        text_encoder_4 ([`LlamaForCausalLM`]):
            Llama text encoder
        tokenizer_4 ([`PreTrainedTokenizerFast`]):
            Tokenizer for Llama
        controlnet ([`HiDreamControlNetModel`] or `List[HiDreamControlNetModel]`):
            Provides additional conditioning to the `transformer` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->transformer->vae"
    )
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "control_image"]

    def __init__(
        self,
        transformer: HiDreamImageTransformer2DModel,
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
        controlnet: HiDreamControlNetModel,
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
            controlnet=controlnet,
        )
        self.default_sample_size = None # automatically scale sample size by control input.
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        # HiDream latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.tokenizer_4.pad_token = getattr(
            self.tokenizer_4, "eos_token", "<|eot_id|>"
        )

    def _nearest_sample_size(self, h: int, w: int) -> tuple[int, int]:
        """
        Round an (h, w) RGB resolution *down* to the nearest values compatible with:
            • VAE down-sampling by `vae_scale_factor` (e.g. 8×)
            • 2 × 2 latent-patch packing (`patch_size = 2`)
        """
        div = self.vae_scale_factor * self.transformer.config.patch_size   # 8*2 = 16
        h = (h // div) * div
        w = (w // div) * div
        if h == 0 or w == 0:
            raise ValueError(
                f"Resolution below minimum {div}×{div} (got {h}×{w})."
            )
        return h, w

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Get T5 text encoder embeddings for the given prompt."""
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
        """Get CLIP text encoder embeddings for the given prompt."""
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
        """Get Llama text encoder embeddings for the given prompt."""
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
        """Encode prompts into embeddings for HiDream transformer."""
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, HiDreamImageLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if USE_PEFT_BACKEND:
                for encoder in [
                    self.text_encoder,
                    self.text_encoder_2,
                    self.text_encoder_3,
                    self.text_encoder_4,
                ]:
                    if encoder is not None:
                        scale_lora_layers(encoder, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = (
                t5_prompt_embeds.shape[0] if t5_prompt_embeds is not None else 1
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

        if self.text_encoder is not None:
            if isinstance(self, HiDreamImageLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                for encoder in [
                    self.text_encoder,
                    self.text_encoder_2,
                    self.text_encoder_3,
                    self.text_encoder_4,
                ]:
                    if encoder is not None:
                        unscale_lora_layers(encoder, lora_scale)

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
        """Internal method to encode prompts to embeddings for the model."""
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

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        negative_prompt_4=None,
        t5_prompt_embeds=None,
        negative_t5_prompt_embeds=None,
        llama_prompt_embeds=None,
        negative_llama_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if (
            height % (self.vae_scale_factor * 2) != 0
            or width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and t5_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {t5_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}"
            )

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
        """Prepare latents for denoising."""
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

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        """Prepare control image for conditioning."""
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        return image

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
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
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
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation (for CLIP L/14).
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to CLIP G/14.
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to T5 XXL.
            prompt_4 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to Llama.
            height (`int`, *optional*, defaults to self.default_sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.default_sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale for classifier-free guidance.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, etc.):
                The ControlNet input condition.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original transformer.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt for CLIP L/14.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The negative prompt for CLIP G/14.
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The negative prompt for T5 XXL.
            negative_prompt_4 (`str` or `List[str]`, *optional*):
                The negative prompt for Llama.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random number generator(s) for deterministic generation.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            t5_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated T5 embeddings.
            llama_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated Llama embeddings.
            negative_t5_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative T5 embeddings.
            negative_llama_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative Llama embeddings.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled CLIP embeddings.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled CLIP embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.hidream.HiDreamControlNetPipelineOutput`].
            joint_attention_kwargs (`dict`, *optional*):
                Additional kwargs for the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                Callback to be called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to 128):
                Maximum sequence length to use with the tokenizers.

        Returns:
            [`~pipelines.hidream.HiDreamControlNetPipelineOutput`] or `tuple`
        """
        # 1. Set up image dimensions and scales
        if height is None or width is None:
            if control_image is None:
                raise ValueError("height/width or control_image must be supplied")
            h0, w0 = control_image.shape[-2:]
        else:
            h0, w0 = height, width                       # explicit user request

        height, width = self._nearest_sample_size(h0, w0)            # final RGB dims

        pH = height // (self.vae_scale_factor * 2)        # latent tokens per side
        pW = width  // (self.vae_scale_factor * 2)
        if pH * pW > self.transformer.max_seq:
            raise ValueError(
                f"Resolution too large: needs {pH*pW} tokens, "
                f"but transformer.max_seq is {self.transformer.max_seq}"
            )

        # store so scheduler / helpers can see them
        self.latent_h, self.latent_w = pH, pW

        # Handle control guidance scheduling
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            # TODO: MultiControlNet support
            # mult = (
            #     len(self.controlnet.nets)
            #     if isinstance(self.controlnet, HiDreamMultiControlNetModel)
            #     else 1
            # )
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 2. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            prompt_4,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            t5_prompt_embeds=t5_prompt_embeds,
            negative_t5_prompt_embeds=negative_t5_prompt_embeds,
            llama_prompt_embeds=llama_prompt_embeds,
            negative_llama_prompt_embeds=negative_llama_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = (
                t5_prompt_embeds.shape[0] if t5_prompt_embeds is not None else 1
            )

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
            dtype=self.transformer.dtype,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 5. Prepare control image
        num_channels_latents = self.transformer.config.in_channels
        if isinstance(self.controlnet, HiDreamControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            height, width = control_image.shape[-2:]

            # Encode control image to latents
            control_image = retrieve_latents(
                self.vae.encode(control_image), generator=generator
            )
            control_image = (
                control_image - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor

        # TODO: MultiControlNet support
        # elif isinstance(self.controlnet, HiDreamMultiControlNetModel):
        #     control_images = []
        #     for i, control_image_ in enumerate(control_image):
        #         control_image_ = self.prepare_image(
        #             image=control_image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=self.vae.dtype,
        #             do_classifier_free_guidance=self.do_classifier_free_guidance,
        #         )
        #         height, width = control_image_.shape[-2:]

        #         # Encode control image to latents
        #         control_image_ = retrieve_latents(
        #             self.vae.encode(control_image_), generator=generator
        #         )
        #         control_image_ = (
        #             control_image_ - self.vae.config.shift_factor
        #         ) * self.vae.config.scaling_factor

        #         control_images.append(control_image_)

        #     control_image = control_images

        # 6. Prepare embeddings for guidance if needed
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

        # 7. Prepare latent variables
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

        # 8. Prepare spatial data for non-square images
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

        # 9. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(
                num_inference_steps, device=device, shift=math.exp(mu)
            )
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, UniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
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

        # 10. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0]
                if isinstance(self.controlnet, HiDreamControlNetModel)
                else keeps
            )

        # 11. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # Control image expansion for guidance
                if self.do_classifier_free_guidance:
                    if isinstance(control_image, list):
                        control_model_input = [
                            torch.cat([img] * 2) for img in control_image
                        ]
                    else:
                        control_model_input = torch.cat([control_image] * 2)
                else:
                    control_model_input = control_image

                # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Handle conditioning scale
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [
                        c * s
                        for c, s in zip(
                            controlnet_conditioning_scale, controlnet_keep[i]
                        )
                    ]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # Reshape latents for transformer if needed
                if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                    B, C, H, W = latent_model_input.shape
                    patch_size = self.transformer.config.patch_size
                    pH, pW = H // patch_size, W // patch_size

                    # Reshape main latents
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

                    # CRITICAL: Apply same reshaping to control input
                    if self.do_classifier_free_guidance:
                        if isinstance(control_image, list):
                            control_model_input = []
                            for img in control_image:
                                out_control = torch.zeros(
                                    (
                                        img.shape[0],
                                        img.shape[1],
                                        self.transformer.max_seq,
                                        patch_size * patch_size,
                                    ),
                                    dtype=img.dtype,
                                    device=img.device,
                                )
                                img_reshaped = einops.rearrange(
                                    img,
                                    "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                                    p1=patch_size,
                                    p2=patch_size,
                                )
                                out_control[:, :, 0 : pH * pW] = img_reshaped
                                control_model_input.append(out_control)
                        else:
                            out_control = torch.zeros(
                                (
                                    control_model_input.shape[0],
                                    control_model_input.shape[1],
                                    self.transformer.max_seq,
                                    patch_size * patch_size,
                                ),
                                dtype=control_model_input.dtype,
                                device=control_model_input.device,
                            )
                            control_reshaped = einops.rearrange(
                                control_model_input,
                                "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                                p1=patch_size,
                                p2=patch_size,
                            )
                            out_control[:, :, 0 : pH * pW] = control_reshaped
                            control_model_input = out_control
                    else:
                        # Same for non-CFG case
                        if isinstance(control_image, list):
                            # Handle list case
                            pass
                        else:
                            out_control = torch.zeros(
                                (
                                    control_image.shape[0],
                                    control_image.shape[1],
                                    self.transformer.max_seq,
                                    patch_size * patch_size,
                                ),
                                dtype=control_image.dtype,
                                device=control_image.device,
                            )
                            control_reshaped = einops.rearrange(
                                control_image,
                                "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                                p1=patch_size,
                                p2=patch_size,
                            )
                            out_control[:, :, 0 : pH * pW] = control_reshaped
                            control_model_input = control_image
                # ControlNet prediction
                controlnet_block_samples, controlnet_single_block_samples = (
                    self.controlnet(
                        hidden_states=latent_model_input,
                        controlnet_cond=control_model_input,
                        timesteps=timestep,
                        t5_hidden_states=t5_embeds_input,
                        llama_hidden_states=llama_embeds_input,
                        pooled_embeds=pooled_embeds_input,
                        img_sizes=img_sizes,
                        img_ids=img_ids,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )
                )

                # Transformer prediction with ControlNet conditioning
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    t5_hidden_states=t5_embeds_input,
                    llama_hidden_states=llama_embeds_input,
                    pooled_embeds=pooled_embeds_input,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    img_sizes=img_sizes,
                    img_ids=img_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred  # HiDream uses inverted velocity

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

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
                    t5_embeds_input = callback_outputs.pop(
                        "t5_embeds_input", t5_embeds_input
                    )
                    llama_embeds_input = callback_outputs.pop(
                        "llama_embeds_input", llama_embeds_input
                    )
                    pooled_embeds_input = callback_outputs.pop(
                        "pooled_embeds_input", pooled_embeds_input
                    )
                    control_image = callback_outputs.pop("control_image", control_image)

                # Call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 12. Post-processing
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

        return HiDreamControlNetPipelineOutput(images=image)
