# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AuraFlowAttnProcessor2_0,
    FusedAuraFlowAttnProcessor2_0,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormZero, FP32LayerNorm
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.utils import BaseOutput
from diffusers.models.controlnet import zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# Import or redefine the necessary AuraFlow components
from helpers.models.auraflow.transformer import (
    AuraFlowPatchEmbed,
    AuraFlowSingleTransformerBlock,
    AuraFlowJointTransformerBlock,
    AuraFlowPreFinalBlock,
    AuraFlowFeedForward,
)


@dataclass
class AuraFlowControlNetOutput(BaseOutput):
    """
    The output of [`AuraFlowControlNetModel`].

    Args:
        controlnet_block_samples (`tuple[torch.Tensor]`):
            A tuple of tensors that are added to the residuals of AuraFlow transformer blocks.
    """

    controlnet_block_samples: Tuple[torch.Tensor]


class AuraFlowControlNetModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    """
    A ControlNet model for AuraFlow.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods
    implemented for all models (downloading, saving, etc.).

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input.
        num_mmdit_layers (`int`, *optional*, defaults to 4): The number of MMDiT Transformer blocks to use.
        num_single_dit_layers (`int`, *optional*, defaults to 32):
            The number of single DiT Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 256): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 12): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*, defaults to 2048): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`, *optional*, defaults to 3072): Number of dimensions to use when projecting the `encoder_hidden_states`.
        out_channels (`int`, defaults to 4): Number of output channels.
        pos_embed_max_size (`int`, defaults to 1024): Maximum positions to embed from the image latents.
        num_layers (`int`, *optional*): The number of layers of transformer blocks to use. If not provided,
            defaults to num_mmdit_layers + num_single_dit_layers.
        extra_conditioning_channels (`int`, defaults to 0): Number of extra conditioning channels to add to the input.
            If non-zero, pos_embed_input will be initialized to handle the extra channels.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "AuraFlowJointTransformerBlock",
        "AuraFlowSingleTransformerBlock",
        "AuraFlowPatchEmbed",
    ]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 4,
        num_mmdit_layers: int = 4,
        num_single_dit_layers: int = 32,
        attention_head_dim: int = 256,
        num_attention_heads: int = 12,
        joint_attention_dim: int = 2048,
        caption_projection_dim: int = 3072,
        out_channels: int = 4,
        pos_embed_max_size: int = 1024,
        num_layers: Optional[int] = None,
        extra_conditioning_channels: int = 0,
    ):
        super().__init__()

        default_out_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else default_out_channels
        )
        self.inner_dim = num_attention_heads * attention_head_dim

        # If num_layers is specified, use it to limit the number of blocks
        if num_layers is not None:
            # Distribute layers between joint and single blocks proportionally
            total_layers = num_mmdit_layers + num_single_dit_layers
            mmdit_ratio = num_mmdit_layers / total_layers

            actual_num_mmdit_layers = int(num_layers * mmdit_ratio)
            actual_num_single_dit_layers = num_layers - actual_num_mmdit_layers
        else:
            actual_num_mmdit_layers = num_mmdit_layers
            actual_num_single_dit_layers = num_single_dit_layers

        self.num_layers = actual_num_mmdit_layers + actual_num_single_dit_layers

        # Standard components matching the base AuraFlow transformer
        self.pos_embed = AuraFlowPatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.context_embedder = nn.Linear(
            joint_attention_dim,
            caption_projection_dim,
            bias=False,
        )

        self.time_step_embed = Timesteps(
            num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True
        )
        self.time_step_proj = TimestepEmbedding(
            in_channels=256, time_embed_dim=self.inner_dim
        )

        # Joint transformer blocks
        self.joint_transformer_blocks = nn.ModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(actual_num_mmdit_layers)
            ]
        )

        # Single transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(actual_num_single_dit_layers)
            ]
        )

        # ControlNet specific components
        # Create zero-initialized linear layers for each transformer block
        self.controlnet_blocks = nn.ModuleList([])
        total_blocks = len(self.joint_transformer_blocks) + len(
            self.single_transformer_blocks
        )

        for _ in range(total_blocks):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        # Additional conditioning embedding
        if extra_conditioning_channels > 0:
            self.pos_embed_input = AuraFlowPatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels + extra_conditioning_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
            )
            self.pos_embed_input = zero_module(self.pos_embed_input)
        else:
            self.pos_embed_input = None

        # Register tokens (matching AuraFlow)
        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        self.gradient_checkpointing = False

    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use feed forward chunking.
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def disable_forward_chunking(self):
        """Disables the forward chunking if enabled."""

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # def _set_gradient_checkpointing(self, module, enable=False, **kwargs):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = enable

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: Optional[int] = None,
        num_extra_conditioning_channels: int = 0,
        load_weights_from_transformer: bool = True,
    ):
        """
        Create a ControlNet model from a pre-trained transformer model.

        Args:
            transformer: The transformer model to use as a base.
            num_layers: The number of transformer layers to use. If None, uses all layers.
            num_extra_conditioning_channels: Number of extra channels for controlnet conditioning.
            load_weights_from_transformer: Whether to load weights from the transformer.
        """
        # Extract config parameters from transformer
        config = transformer.config

        # Build config dict for controlnet initialization
        controlnet_config = {
            "sample_size": config.sample_size,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "num_mmdit_layers": config.num_mmdit_layers,
            "num_single_dit_layers": config.num_single_dit_layers,
            "attention_head_dim": config.attention_head_dim,
            "num_attention_heads": config.num_attention_heads,
            "joint_attention_dim": config.joint_attention_dim,
            "caption_projection_dim": config.caption_projection_dim,
            "out_channels": config.out_channels,
            "pos_embed_max_size": config.pos_embed_max_size,
        }

        # Update with controlnet-specific settings
        if num_layers is not None:
            controlnet_config["num_layers"] = num_layers
        controlnet_config["extra_conditioning_channels"] = (
            num_extra_conditioning_channels
        )

        # Create the controlnet model
        controlnet = cls(**controlnet_config)

        if load_weights_from_transformer:
            # Load weights from transformer
            # Handle different number of layers gracefully
            if hasattr(transformer, "pos_embed"):
                controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            if hasattr(transformer, "time_step_embed"):
                controlnet.time_step_embed.load_state_dict(
                    transformer.time_step_embed.state_dict()
                )
            if hasattr(transformer, "time_step_proj"):
                controlnet.time_step_proj.load_state_dict(
                    transformer.time_step_proj.state_dict()
                )
            if hasattr(transformer, "context_embedder"):
                controlnet.context_embedder.load_state_dict(
                    transformer.context_embedder.state_dict()
                )
            if hasattr(transformer, "register_tokens"):
                controlnet.register_tokens.data = (
                    transformer.register_tokens.data.clone()
                )

            # Load transformer blocks (up to the number of layers in controlnet)
            for i in range(len(controlnet.joint_transformer_blocks)):
                if i < len(transformer.joint_transformer_blocks):
                    controlnet.joint_transformer_blocks[i].load_state_dict(
                        transformer.joint_transformer_blocks[i].state_dict()
                    )

            for i in range(len(controlnet.single_transformer_blocks)):
                if i < len(transformer.single_transformer_blocks):
                    controlnet.single_transformer_blocks[i].load_state_dict(
                        transformer.single_transformer_blocks[i].state_dict()
                    )

            # Zero initialize the pos_embed_input if it exists
            if controlnet.pos_embed_input is not None:
                controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        return controlnet

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[AuraFlowControlNetOutput, Tuple]:
        """
        The [`AuraFlowControlNetModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, channel, height, width)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                Additional keyword arguments for the attention processors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~AuraFlowControlNetOutput`] instead of a plain tuple.

        Returns:
            If `return_dict` is True, an [`~AuraFlowControlNetOutput`] is returned, otherwise a `tuple` where the
            first element is the controlnet block samples.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
            attention_kwargs = {}

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Apply initial embeddings
        hidden_states = self.pos_embed(hidden_states)

        # Add controlnet conditioning if we have extra channels
        if self.pos_embed_input is not None:
            hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        # Time embedding
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)

        # Context embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [
                self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1),
                encoder_hidden_states,
            ],
            dim=1,
        )

        block_res_samples = []

        # Process through joint transformer blocks
        for block in self.joint_transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    attention_kwargs=attention_kwargs,
                )

            block_res_samples.append(hidden_states)

        # Process through single transformer blocks
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat(
                [encoder_hidden_states, hidden_states], dim=1
            )

            for block in self.single_transformer_blocks:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    combined_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        combined_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                else:
                    combined_hidden_states = block(
                        hidden_states=combined_hidden_states,
                        temb=temb,
                        attention_kwargs=attention_kwargs,
                    )

                # Extract the hidden states part (not the encoder hidden states)
                hidden_states = combined_hidden_states[:, encoder_seq_len:]
                block_res_samples.append(hidden_states)

        # Apply controlnet blocks and scaling
        controlnet_block_res_samples = []
        for block_res_sample, controlnet_block in zip(
            block_res_samples, self.controlnet_blocks
        ):
            block_res_sample = controlnet_block(block_res_sample)
            block_res_sample = block_res_sample * conditioning_scale
            controlnet_block_res_samples.append(block_res_sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return AuraFlowControlNetOutput(
            controlnet_block_samples=controlnet_block_res_samples
        )
