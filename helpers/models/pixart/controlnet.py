from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import PixArtTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import PeftAdapterMixin


class PixArtSigmaControlNetAdapterBlock(nn.Module):
    def __init__(
        self,
        block_index,
        # PixArt Sigma typically uses these configurations
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",  # Sigma uses the same norm type
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_type: Optional[str] = "default",
    ):
        super().__init__()

        self.block_index = block_index
        self.inner_dim = num_attention_heads * attention_head_dim

        # the first block has a zero before layer
        if self.block_index == 0:
            self.before_proj = nn.Linear(self.inner_dim, self.inner_dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.transformer_block = BasicTransformerBlock(
            self.inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )

        self.after_proj = nn.Linear(self.inner_dim, self.inner_dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def train(self, mode: bool = True):
        self.transformer_block.train(mode)

        if self.block_index == 0:
            self.before_proj.train(mode)

        self.after_proj.train(mode)

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.block_index == 0:
            controlnet_states = self.before_proj(controlnet_states)
            controlnet_states = hidden_states + controlnet_states

        controlnet_states_down = self.transformer_block(
            hidden_states=controlnet_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            class_labels=None,
        )

        controlnet_states_left = self.after_proj(controlnet_states_down)

        return controlnet_states_left, controlnet_states_down


class PixArtSigmaControlNetAdapterModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    # N=13, as specified in the paper https://arxiv.org/html/2401.05252v1/#S4 ControlNet-Transformer
    @register_to_config
    def __init__(
        self,
        num_layers: int = 13,
        # Additional Sigma-specific parameters if needed
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        cross_attention_dim: Optional[int] = 1152,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim

        self.controlnet_blocks = nn.ModuleList(
            [
                PixArtSigmaControlNetAdapterBlock(
                    block_index=i,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for i in range(num_layers)
            ]
        )

    @classmethod
    def from_transformer(
        cls, transformer: PixArtTransformer2DModel, num_layers: Optional[int] = None
    ):
        """
        Create a ControlNet adapter from an existing PixArt Sigma transformer.

        Args:
            transformer: The base PixArt Sigma transformer model
            num_layers: Number of layers to copy (defaults to 13 or transformer layers, whichever is smaller)
        """
        if num_layers is None:
            num_layers = min(13, len(transformer.transformer_blocks))

        # Extract configuration from the transformer
        config_args = {
            "num_layers": num_layers,
            "num_attention_heads": transformer.config.num_attention_heads,
            "attention_head_dim": transformer.config.attention_head_dim,
        }

        if hasattr(transformer.config, "cross_attention_dim"):
            config_args["cross_attention_dim"] = transformer.config.cross_attention_dim

        control_net = cls(**config_args)

        # Copy the specified number of blocks from the transformer
        for depth in range(num_layers):
            control_net.controlnet_blocks[depth].transformer_block.load_state_dict(
                transformer.transformer_blocks[depth].state_dict()
            )

        return control_net

    def train(self, mode: bool = True):
        for block in self.controlnet_blocks:
            block.train(mode)


class PixArtSigmaControlNetTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A wrapper model that combines PixArt Sigma transformer with ControlNet adapter.
    This follows the same pattern as the Alpha version but ensures compatibility with Sigma.
    """

    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
        controlnet: PixArtSigmaControlNetAdapterModel,
        blocks_num: Optional[int] = None,
        init_from_transformer: bool = False,
        training: bool = False,
    ):
        super().__init__()

        # Use controlnet's num_layers if blocks_num not specified
        self.blocks_num = (
            blocks_num if blocks_num is not None else controlnet.num_layers
        )
        self.gradient_checkpointing = False
        self.register_to_config(**transformer.config)
        self.training = training

        if init_from_transformer:
            # Initialize controlnet from transformer
            controlnet = PixArtSigmaControlNetAdapterModel.from_transformer(
                transformer, num_layers=self.blocks_num
            )

        self.transformer = transformer
        self.controlnet = controlnet

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency during training."""
        self.gradient_checkpointing = True
        self.transformer.gradient_checkpointing = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.transformer.config.patch_size,
            hidden_states.shape[-1] // self.transformer.config.patch_size,
        )
        hidden_states = self.transformer.pos_embed(hidden_states)

        timestep, embedded_timestep = self.transformer.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.transformer.caption_projection is not None:
            encoder_hidden_states = self.transformer.caption_projection(
                encoder_hidden_states
            )
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        controlnet_states_down = None
        if controlnet_cond is not None:
            controlnet_states_down = self.transformer.pos_embed(controlnet_cond)

        # 2. Blocks
        for block_index, block in enumerate(self.transformer.transformer_blocks):
            if (
                self.training
                and self.gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                # TODO: Implement gradient checkpointing support
                # For now, fall through to regular forward
                pass

            # the control nets are only used for blocks 1 to self.blocks_num
            if (
                block_index > 0
                and block_index <= self.blocks_num
                and controlnet_states_down is not None
            ):
                (
                    controlnet_states_left,
                    controlnet_states_down,
                ) = self.controlnet.controlnet_blocks[block_index - 1](
                    hidden_states=hidden_states,  # used only in the first block
                    controlnet_states=controlnet_states_down,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )

                hidden_states = hidden_states + controlnet_states_left

            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

        # 3. Output
        shift, scale = (
            self.transformer.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.transformer.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(
            hidden_states.device
        )
        hidden_states = self.transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.transformer.config.patch_size,
                self.transformer.config.patch_size,
                self.transformer.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.transformer.out_channels,
                height * self.transformer.config.patch_size,
                width * self.transformer.config.patch_size,
            )
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
