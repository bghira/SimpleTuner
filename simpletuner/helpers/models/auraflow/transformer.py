from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
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
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.utils.patching import CallableDict, MutableModuleList, PatchableModule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor, image_tokens_start: int | None = None):
    if buffer is None:
        return
    if image_tokens_start is not None and hidden_states.dim() >= 3:
        buffer[key] = hidden_states[:, image_tokens_start:, ...]
    else:
        buffer[key] = hidden_states


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
        self.pos_embed = nn.Parameter(torch.randn(1, pos_embed_max_size, embed_dim) * 0.1)

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

    def pe_selection_index_based_on_dim(self, h, w):
        # select subset of positional embedding based on H, W, where H, W is size of latent
        # PE will be viewed as 2d-grid, and H/p x W/p of the PE will be selected
        # because original input are in flattened format, we have to flatten this 2d grid as well.
        h_p, w_p = h // self.patch_size, w // self.patch_size
        h_max, w_max = int(self.pos_embed_max_size**0.5), int(self.pos_embed_max_size**0.5)

        # Calculate the top-left corner indices for the centered patch grid
        starth = h_max // 2 - h_p // 2
        startw = w_max // 2 - w_p // 2

        # Generate the row and column indices for the desired patch grid
        rows = torch.arange(starth, starth + h_p, device=self.pos_embed.device)
        cols = torch.arange(startw, startw + w_p, device=self.pos_embed.device)

        # Create a 2D grid of indices
        row_indices, col_indices = torch.meshgrid(rows, cols, indexing="ij")

        # Convert the 2D grid indices to flattened 1D indices
        selected_indices = (row_indices * w_max + col_indices).flatten()

        return selected_indices

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

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.linear_1(x)) * self.linear_2(x)
        x = self.out_projection(x)
        return x

    # Add support for chunked feed forward for improved memory efficiency
    def set_chunk_feed_forward(self, chunk_size: Optional[int] = None, dim: int = 0):
        self.chunk_size = chunk_size
        self.dim = dim

    def _should_chunk(self, x: torch.Tensor) -> bool:
        if self.chunk_size is None or self.chunk_size <= 0:
            return False

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            raise ValueError(f"Invalid dimension {self.dim} for tensor with shape {x.shape}")

        dim_size = x.shape[dim]
        if self.chunk_size >= dim_size:
            return False

        # Avoid chunking very small tensors where overhead dominates runtime.
        if dim_size <= self.chunk_size * 4:
            return False

        return True

    def _chunk_forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.chunk_size, dim=self.dim)
        if len(chunks) == 1:
            return self._feed_forward(chunks[0])

        output_chunks = []
        for chunk in chunks:
            chunk_output = self._feed_forward(chunk)
            output_chunks.append(chunk_output)

        return torch.cat(output_chunks, dim=self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_chunk(x):
            return self._chunk_forward(x)

        return self._feed_forward(x)


class AuraFlowPreFinalBlock(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


@maybe_allow_in_graph
class AuraFlowSingleTransformerBlock(PatchableModule):
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
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

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
class AuraFlowJointTransformerBlock(PatchableModule):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")
        self.norm1_context = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

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
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
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
        encoder_hidden_states = self.norm2_context(residual_context + c_gate_msa.unsqueeze(1) * context_attn_output)
        encoder_hidden_states = encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(encoder_hidden_states)
        encoder_hidden_states = residual_context + encoder_hidden_states

        return encoder_hidden_states, hidden_states


class AuraFlowTransformer2DModel(PatchableModule, ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
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
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

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
        effective_out_channels = out_channels if out_channels is not None else default_out_channels
        self.register_to_config(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_mmdit_layers=num_mmdit_layers,
            num_single_dit_layers=num_single_dit_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            out_channels=effective_out_channels,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.out_channels = effective_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

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
        self.time_step_embed = Timesteps(num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True)
        self.time_step_proj = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        # Signed-time embedding for TwinFlow-style negative time handling.
        self.time_sign_embed = nn.Embedding(2, self.inner_dim)
        nn.init.zeros_(self.time_sign_embed.weight)

        self.joint_transformer_blocks = MutableModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        self.single_transformer_blocks = MutableModuleList(
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
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

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

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        """Set the TREAD router and routing configuration."""
        self._tread_router = router
        self._tread_routes = routes

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
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

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def disable_forward_chunking(self):
        """Disables the forward chunking if enabled."""

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        override = getattr(self, "_attn_processors_override", None)
        if override is not None:
            return override

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

        return CallableDict(processors)

    @attn_processors.setter
    def attn_processors(self, value):
        self._attn_processors_override = value

    @attn_processors.deleter
    def attn_processors(self):
        self._attn_processors_override = None

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
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
                    key = f"{name}.processor"
                    processor_value = processor.pop(key, None)
                    if processor_value is not None:
                        module.set_processor(processor_value)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

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
        timestep_sign: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        block_controlnet_hidden_states: List = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        hidden_states_buffer: Optional[dict] = None,
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
            force_keep_mask (`torch.Tensor`, *optional*):
                A mask tensor for TREAD routing indicating which tokens to force keep.
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
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        height, width = hidden_states.shape[-2:]

        # Apply patch embedding, timestep embedding, and project the caption embeddings.
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)
        if timestep_sign is not None:
            sign_idx = (timestep_sign.view(-1) < 0).long().to(device=hidden_states.device)
            temb = temb + self.time_sign_embed(sign_idx).to(dtype=temb.dtype, device=hidden_states.device)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [
                self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1),
                encoder_hidden_states,
            ],
            dim=1,
        )

        # TREAD initialization
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_tokens = None
        global_idx = 0

        # Handle negative route indices
        if routes:
            total_layers = len(self.joint_transformer_blocks) + len(self.single_transformer_blocks)

            def _to_pos(idx):
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **r,
                    "start_layer_idx": _to_pos(r["start_layer_idx"]),
                    "end_layer_idx": _to_pos(r["end_layer_idx"]),
                }
                for r in routes
            ]

        # Total number of blocks for ControlNet integration
        total_blocks = len(self.joint_transformer_blocks) + len(self.single_transformer_blocks)
        capture_idx = 0

        # MMDiT blocks.
        for index_block, block in enumerate(self.joint_transformer_blocks):
            # TREAD: START a route?
            if use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]
                tread_mask_info = router.get_mask(
                    hidden_states,
                    mask_ratio=mask_ratio,
                    force_keep=force_keep_mask,
                )
                saved_tokens = hidden_states.clone()
                hidden_states = router.start_route(hidden_states, tread_mask_info)
                routing_now = True

            # Skip specified layers
            if skip_layers is not None and index_block in skip_layers:
                if block_controlnet_hidden_states is not None:
                    interval_control = total_blocks // len(block_controlnet_hidden_states)
                    hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
                continue

            if (
                self.training
                and self.gradient_checkpointing
                and (self.gradient_checkpointing_interval is None or index_block % self.gradient_checkpointing_interval == 0)
            ):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
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
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

            _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", hidden_states)
            capture_idx += 1
            # TREAD: END the current route?
            if routing_now and global_idx == routes[route_ptr]["end_layer_idx"]:
                hidden_states = router.end_route(
                    hidden_states,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                routing_now = False
                route_ptr += 1

            global_idx += 1

        # Single DiT blocks that combine the `hidden_states` (image) and `encoder_hidden_states` (text)
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                actual_index = len(self.joint_transformer_blocks) + index_block

                # TREAD: START a route?
                if use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]:
                    mask_ratio = routes[route_ptr]["selection_ratio"]

                    # For single transformer blocks, we work with combined hidden states
                    # Extract image tokens for routing (encoder tokens remain unchanged)
                    img_tokens = combined_hidden_states[:, encoder_seq_len:]

                    fkm = force_keep_mask if force_keep_mask is not None else None
                    tread_mask_info = router.get_mask(
                        img_tokens,
                        mask_ratio=mask_ratio,
                        force_keep=fkm,
                    )
                    saved_tokens = img_tokens.clone()
                    img_tokens = router.start_route(img_tokens, tread_mask_info)

                    # Recombine with encoder tokens
                    combined_hidden_states = torch.cat(
                        [
                            combined_hidden_states[:, :encoder_seq_len],  # encoder tokens (unchanged)
                            img_tokens,  # routed image tokens
                        ],
                        dim=1,
                    )
                    routing_now = True

                # Skip specified layers
                if skip_layers is not None and actual_index in skip_layers:
                    if block_controlnet_hidden_states is not None:
                        interval_control = total_blocks // len(block_controlnet_hidden_states)
                        combined_hidden_states = combined_hidden_states + torch.cat(
                            [
                                torch.zeros_like(encoder_hidden_states),
                                block_controlnet_hidden_states[actual_index // interval_control],
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

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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
                    interval_control = total_blocks // len(block_controlnet_hidden_states)
                    # Apply controlnet only to the hidden_states part, not to encoder_hidden_states
                    controlnet_hidden = torch.cat(
                        [
                            torch.zeros_like(encoder_hidden_states),
                            block_controlnet_hidden_states[actual_index // interval_control],
                        ],
                        dim=1,
                    )
                    combined_hidden_states = combined_hidden_states + controlnet_hidden

                _store_hidden_state(
                    hidden_states_buffer,
                    f"layer_{capture_idx}",
                    combined_hidden_states,
                    image_tokens_start=encoder_seq_len,
                )
                capture_idx += 1
                # TREAD: END the current route?
                if routing_now and global_idx == routes[route_ptr]["end_layer_idx"]:
                    img_tokens_r = combined_hidden_states[:, encoder_seq_len:]
                    img_tokens = router.end_route(
                        img_tokens_r,
                        tread_mask_info,
                        original_x=saved_tokens,
                    )
                    combined_hidden_states = torch.cat(
                        [
                            combined_hidden_states[:, :encoder_seq_len],  # encoder tokens (unchanged)
                            img_tokens,  # restored image tokens
                        ],
                        dim=1,
                    )
                    routing_now = False
                    route_ptr += 1

                global_idx += 1

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
