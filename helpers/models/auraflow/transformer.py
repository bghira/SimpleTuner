from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    is_torch_version,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class AuraFlowPatchEmbed(nn.Module):
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        pos_embed_max_size=None,
    ):
        super().__init__()

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, pos_embed_max_size, embed_dim) * 0.1
        )

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

    def pe_selection_index_based_on_dim(self, h, w):
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.pos_embed.shape[1])
        h_max, w_max = int(self.pos_embed_max_size**0.5), int(
            self.pos_embed_max_size**0.5
        )
        original_pe_indexes = original_pe_indexes.view(h_max, w_max)
        starth = h_max // 2 - h_p // 2
        endh = starth + h_p
        startw = w_max // 2 - w_p // 2
        endw = startw + w_p
        original_pe_indexes = original_pe_indexes[starth:endh, startw:endw]
        return original_pe_indexes.flatten()

    def forward(self, latent):
        batch_size, num_channels, height, width = latent.size()
        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        latent = self.proj(latent)
        pe_index = self.pe_selection_index_based_on_dim(height, width)
        return latent + self.pos_embed[:, pe_index]


class AuraFlowFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        final_hidden_dim = int(2 * hidden_dim / 3)
        final_hidden_dim = find_multiple(final_hidden_dim, 256)

        self.linear_1 = nn.Linear(dim, final_hidden_dim, bias=False)
        self.linear_2 = nn.Linear(dim, final_hidden_dim, bias=False)
        self.out_projection = nn.Linear(final_hidden_dim, dim, bias=False)

        # Add chunk feed forward capability
        self.chunk_size = None
        self.dim = 0

    # Add support for chunked feed forward for improved memory efficiency
    def set_chunk_feed_forward(self, chunk_size: Optional[int] = None, dim: int = 0):
        self.chunk_size = chunk_size
        self.dim = dim

    def _chunk_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation of chunked feed forward
        num_chunks = (
            x.shape[self.dim] // self.chunk_size
            if (x.shape[self.dim] % self.chunk_size == 0)
            else (x.shape[self.dim] // self.chunk_size) + 1
        )
        chunks = torch.chunk(x, num_chunks, dim=self.dim)
        output_chunks = []

        for chunk in chunks:
            chunk_output = F.silu(self.linear_1(chunk)) * self.linear_2(chunk)
            chunk_output = self.out_projection(chunk_output)
            output_chunks.append(chunk_output)

        return torch.cat(output_chunks, dim=self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None:
            return self._chunk_forward(x)

        x = F.silu(self.linear_1(x)) * self.linear_2(x)
        x = self.out_projection(x)
        return x


class AuraFlowPreFinalBlock(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=False
        )

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


@maybe_allow_in_graph
class AuraFlowSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)

    def set_chunk_feed_forward(self, chunk_size: Optional[int] = None, dim: int = 0):
        self.ff.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        residual = hidden_states
        attention_kwargs = attention_kwargs or {}

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        # Attention.
        attn_output = self.attn(hidden_states=norm_hidden_states, **attention_kwargs)

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = residual + hidden_states

        return hidden_states



@maybe_allow_in_graph
class AuraFlowJointTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")
        self.norm1_context = AdaLayerNormZero(
            dim, bias=False, norm_type="fp32_layer_norm"
        )

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            added_proj_bias=False,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
            context_pre_only=False,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff_context = AuraFlowFeedForward(dim, dim * 4)
        self.context_pre_only = False

    def set_chunk_feed_forward(self, chunk_size: Optional[int] = None, dim: int = 0):
        self.ff.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
        self.ff_context.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        residual = hidden_states
        residual_context = encoder_hidden_states
        attention_kwargs = attention_kwargs or {}

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = gate_mlp.unsqueeze(1) * self.ff(hidden_states)
        hidden_states = residual + hidden_states

        # Process attention outputs for the `encoder_hidden_states`.
        encoder_hidden_states = self.norm2_context(
            residual_context + c_gate_msa.unsqueeze(1) * context_attn_output
        )
        encoder_hidden_states = (
            encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(
            encoder_hidden_states
        )
        encoder_hidden_states = residual_context + encoder_hidden_states

        return encoder_hidden_states, hidden_states


class AuraFlowTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    """
    An updated AuraFlowTransformer2DModel with additional SD3 features like gradient checkpointing interval,
    skip layers, forward chunking, and ControlNet support.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_mmdit_layers (`int`, *optional*, defaults to 4): The number of layers of MMDiT Transformer blocks to use.
        num_single_dit_layers (`int`, *optional*, defaults to 4):
            The number of layers of Transformer blocks to use. These blocks use concatenated image and text
            representations.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        out_channels (`int`, defaults to 16): Number of output channels.
        pos_embed_max_size (`int`, defaults to 4096): Maximum positions to embed from the image latents.
    """

    _no_split_modules = [
        "AuraFlowJointTransformerBlock",
        "AuraFlowSingleTransformerBlock",
        "AuraFlowPatchEmbed",
    ]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _supports_gradient_checkpointing = True

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
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else default_out_channels
        )
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.pos_embed = AuraFlowPatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.context_embedder = nn.Linear(
            self.config.joint_attention_dim,
            self.config.caption_projection_dim,
            bias=False,
        )
        self.time_step_embed = Timesteps(
            num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True
        )
        self.time_step_proj = TimestepEmbedding(
            in_channels=256, time_embed_dim=self.inner_dim
        )

        self.joint_transformer_blocks = nn.ModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_dit_layers)
            ]
        )

        self.norm_out = AuraFlowPreFinalBlock(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
        )

        # https://arxiv.org/abs/2309.16588
        # prevents artifacts in the attention maps
        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        self.gradient_checkpointing = False
        self.gradient_checkpointing_interval = None

    def set_gradient_checkpointing_interval(self, interval: int):
        """
        Sets the interval for gradient checkpointing.

        Parameters:
            interval (`int`): The interval for gradient checkpointing.
        """
        self.gradient_checkpointing_interval = interval

    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use feed forward chunking.

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
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

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAuraFlowAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        block_controlnet_hidden_states: List = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The AuraFlowTransformer2DModel forward method with additional support for skip_layers and controlnet.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor`):
                Conditional embeddings for cross attention.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                Additional keyword arguments for the attention processors.
            block_controlnet_hidden_states (`list` of `torch.Tensor`, *optional*):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dict or tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.
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

        height, width = hidden_states.shape[-2:]

        # Apply patch embedding, timestep embedding, and project the caption embeddings.
        hidden_states = self.pos_embed(
            hidden_states
        )  # takes care of adding positional embeddings too.
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [
                self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1),
                encoder_hidden_states,
            ],
            dim=1,
        )

        # Total number of blocks for ControlNet integration
        total_blocks = len(self.joint_transformer_blocks) + len(
            self.single_transformer_blocks
        )

        # MMDiT blocks.
        for index_block, block in enumerate(self.joint_transformer_blocks):
            # Skip specified layers
            if skip_layers is not None and index_block in skip_layers:
                if block_controlnet_hidden_states is not None:
                    interval_control = total_blocks // len(
                        block_controlnet_hidden_states
                    )
                    hidden_states = (
                        hidden_states
                        + block_controlnet_hidden_states[
                            index_block // interval_control
                        ]
                    )
                continue

            if (
                self.training
                and self.gradient_checkpointing
                and (
                    self.gradient_checkpointing_interval is None
                    or index_block % self.gradient_checkpointing_interval == 0
                )
            ):

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

            # controlnet residual
            if block_controlnet_hidden_states is not None:
                interval_control = total_blocks // len(block_controlnet_hidden_states)
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[index_block // interval_control]
                )

        # Single DiT blocks that combine the `hidden_states` (image) and `encoder_hidden_states` (text)
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat(
                [encoder_hidden_states, hidden_states], dim=1
            )

            for index_block, block in enumerate(self.single_transformer_blocks):
                actual_index = len(self.joint_transformer_blocks) + index_block

                # Skip specified layers
                if skip_layers is not None and actual_index in skip_layers:
                    if block_controlnet_hidden_states is not None:
                        interval_control = total_blocks // len(
                            block_controlnet_hidden_states
                        )
                        combined_hidden_states = combined_hidden_states + torch.cat(
                            [
                                torch.zeros_like(encoder_hidden_states),
                                block_controlnet_hidden_states[
                                    actual_index // interval_control
                                ],
                            ],
                            dim=1,
                        )
                    continue

                if (
                    self.training
                    and self.gradient_checkpointing
                    and (
                        self.gradient_checkpointing_interval is None
                        or index_block % self.gradient_checkpointing_interval == 0
                    )
                ):

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

                # controlnet residual for single transformer blocks
                if block_controlnet_hidden_states is not None:
                    interval_control = total_blocks // len(
                        block_controlnet_hidden_states
                    )
                    # Apply controlnet only to the hidden_states part, not to encoder_hidden_states
                    controlnet_hidden = torch.cat(
                        [
                            torch.zeros_like(encoder_hidden_states),
                            block_controlnet_hidden_states[
                                actual_index // interval_control
                            ],
                        ],
                        dim=1,
                    )
                    combined_hidden_states = combined_hidden_states + controlnet_hidden

            hidden_states = combined_hidden_states[:, encoder_seq_len:]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        out_channels = self.config.out_channels
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
