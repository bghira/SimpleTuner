# Copyright 2024 Stability AI, The HuggingFace Team, The InstantX Team, and Terminus Research Group and 2025 bghira. All rights reserved.
#
# Originally licensed under the Apache License, Version 2.0 (the "License");
# Updated to "Affero GENERAL PUBLIC LICENSE Version 3, 19 November 2007" via extensive updates to attn_mask usage.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel as OriginalFluxTransformer2DModel
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import FluxPosEmbed
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

is_flash_attn_available = False
try:
    from flash_attn_interface import flash_attn_func

    is_flash_attn_available = True
except:
    pass

from simpletuner.helpers.models.flux.attention import FluxAttnProcessor3_0, FluxSingleAttnProcessor3_0
from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.utils.patching import CallableDict, MutableModuleList, PatchableModule


def _apply_rotary_emb_anyshape(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    """
    Same API as the original, but also works when `freqs_cis` is
    batched (cos, sin) each (B, S, D).
    """
    cos, sin = freqs_cis
    if cos.ndim == 3:  # (B, S, D)  ← new case
        cos = cos[:, None]  # (B, 1, S, D)
        sin = sin[:, None]
    else:  # (S, D)     ← old case
        cos = cos[None, None]  # (1, 1, S, D)
        sin = sin[None, None]

    cos, sin = cos.to(x.device), sin.to(x.device)

    if use_real:
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.to(x_rotated.device)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        if freqs_cis.shape[-1] != x_rotated.shape[-1]:
            freqs_cis = freqs_cis[..., : x_rotated.shape[-1]]
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention") or not callable(F.scaled_dot_product_attention):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = _apply_rotary_emb_anyshape(query, image_rotary_emb)
            key = _apply_rotary_emb_anyshape(key, image_rotary_emb)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (attention_mask > 0).bool()
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        return hidden_states


def expand_flux_attention_mask(
    hidden_states: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Expand a mask so that the image is included.
    """
    bsz = attn_mask.shape[0]
    assert bsz == hidden_states.shape[0]
    residual_seq_len = hidden_states.shape[1]
    mask_seq_len = attn_mask.shape[1]

    expanded_mask = torch.ones(bsz, residual_seq_len)
    expanded_mask[:, :mask_seq_len] = attn_mask

    return expanded_mask


@maybe_allow_in_graph
class FluxSingleTransformerBlock(PatchableModule):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        if attention_mask is not None:
            attention_mask = expand_flux_attention_mask(
                hidden_states,
                attention_mask,
            )

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=65504, neginf=-65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(PatchableModule):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError("The current PyTorch version does not support the `scaled_dot_product_attention` function.")
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        if attention_mask is not None:
            attention_mask = expand_flux_attention_mask(
                torch.cat([encoder_hidden_states, hidden_states], dim=1),
                attention_mask,
            )

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=65504, neginf=-65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(PatchableModule, ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    # Hint FSDP auto wrap policy to shard per transformer block.
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        self.register_to_config(
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = MutableModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = MutableModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        # optional interval for gradient checkpointing
        self.gradient_checkpointing_interval = None

    def set_gradient_checkpointing_interval(self, value: int):
        self.gradient_checkpointing_interval = value

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        self._tread_router = router
        self._tread_routes = routes

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
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

    @staticmethod
    def _route_rope(rope, info, keep_len: int, batch: int):
        """
        Apply the router's (ids_shuffle ➜ slice) transform to a rotary
        embedding.

        `rope` can be either
        • a tensor  of shape (S, D)                – single‑matrix form
        • a tuple   (cos, sin) each (S, D)         – Diffusers / SD form

        Returns
        -------
        If `rope` is a tensor  ➜  (B, keep_len, D) tensor
        If `rope` is a tuple   ➜  tuple of two tensors, each (B, keep_len, D)
        """

        def _route_one(r: torch.Tensor) -> torch.Tensor:
            rB = r.unsqueeze(0).expand(batch, -1, -1)  # (B, S, D)
            shuf = torch.take_along_dim(rB, info.ids_shuffle.unsqueeze(-1).expand_as(rB), dim=1)
            return shuf[:, :keep_len, :]  # (B, keep, D)

        if isinstance(rope, tuple):
            return tuple(_route_one(r) for r in rope)
        else:
            return _route_one(rope)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        controlnet_blocks_repeat: bool = False,
        force_keep_mask=None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # scale lora layers
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_len = encoder_hidden_states.shape[1]

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)

        image_rotary_emb = self.pos_embed(ids)

        # IP adapter
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # TREAD related
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()  # enable only while training
        route_ptr = 0  # which entry in `routes` we’re on
        routing_now = False  # are we inside a route?
        tread_mask_info = None
        saved_tokens = None  # copy of full‑seq image tokens
        global_idx = 0  # counts over *all* transformer layers
        current_rope = image_rotary_emb

        # TREAD: handle negative route index.
        if routes:
            total_layers = len(self.transformer_blocks) + len(self.single_transformer_blocks)

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

        for index_block, block in enumerate(self.transformer_blocks):
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

                # --- build RoPE with    [ text | routed‑image ]    rows -------------
                # 1. text part:  (B, txt_len, D) – identical for all samples
                text_rope_b = tuple(r[:txt_len].unsqueeze(0).expand(hidden_states.size(0), -1, -1) for r in image_rotary_emb)

                # 2. image part: apply the same shuffle+slice as the tokens
                image_only_rope = tuple(r[txt_len:] for r in image_rotary_emb)
                img_rope_r = self._route_rope(
                    image_only_rope,
                    tread_mask_info,
                    keep_len=hidden_states.size(1),  # S_keep
                    batch=hidden_states.size(0),
                )

                # concatenate text + image rope
                current_rope = tuple(torch.cat([tr, ir], dim=1) for tr, ir in zip(text_rope_b, img_rope_r))
            if (
                self.training
                and self.gradient_checkpointing
                and (self.gradient_checkpointing_interval is None or index_block % self.gradient_checkpointing_interval == 0)
            ):

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    current_rope,
                    attention_mask,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=current_rope,
                    attention_mask=attention_mask,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            # TREAD: END the current route?
            if routing_now and global_idx == routes[route_ptr]["end_layer_idx"]:
                hidden_states = router.end_route(
                    hidden_states,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                routing_now = False
                route_ptr += 1
                current_rope = image_rotary_emb

            global_idx += 1  # advance global layer counter

        # Flux places the text tokens in front of the image tokens in the
        # sequence.
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        txt_len = encoder_hidden_states.shape[1]

        for index_block, block in enumerate(self.single_transformer_blocks):
            # TREAD: START? (operate on *image* tokens only)
            if use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]

                text_tok = hidden_states[:, :txt_len, :]
                img_tok = hidden_states[:, txt_len:, :]

                fkm = force_keep_mask[:, txt_len:] if force_keep_mask is not None else None
                tread_mask_info = router.get_mask(img_tok, mask_ratio=mask_ratio, force_keep=fkm)
                saved_tokens = img_tok.clone()
                img_tok = router.start_route(img_tok, tread_mask_info)

                hidden_states = torch.cat([text_tok, img_tok], dim=1)
                routing_now = True

                # shrink the attention_mask, if one is provided
                if attention_mask is not None:
                    pad = torch.zeros(
                        attention_mask.size(0),
                        img_tok.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask[:, :txt_len], pad], dim=1)

                # Handle shrinking RoPE
                text_rope_b = tuple(r[:txt_len].unsqueeze(0).expand(img_tok.size(0), -1, -1) for r in image_rotary_emb)
                img_rope_r = self._route_rope(
                    tuple(r[txt_len:] for r in image_rotary_emb),
                    tread_mask_info,
                    keep_len=img_tok.size(1),
                    batch=img_tok.size(0),
                )
                current_rope = tuple(torch.cat([tr, ir], dim=1) for tr, ir in zip(text_rope_b, img_rope_r))

            if (
                self.training
                and self.gradient_checkpointing
                or (
                    self.gradient_checkpointing_interval is not None
                    and index_block % self.gradient_checkpointing_interval == 0
                )
            ):

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    current_rope,
                    attention_mask,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=current_rope,
                    attention_mask=attention_mask,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

            # TREAD: END current route?
            if routing_now and global_idx == routes[route_ptr]["end_layer_idx"]:
                text_tok = hidden_states[:, :txt_len, :]
                img_tok_r = hidden_states[:, txt_len:, :]

                img_tok = router.end_route(
                    img_tok_r,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                hidden_states = torch.cat([text_tok, img_tok], dim=1)

                routing_now = False
                route_ptr += 1

                if attention_mask is not None:
                    pad = torch.zeros(
                        attention_mask.size(0),
                        img_tok.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask[:, :txt_len], pad], dim=1)

                current_rope = image_rotary_emb

            global_idx += 1

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # unscale lora layers
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    dtype = torch.bfloat16
    bsz = 2
    img = torch.rand((bsz, 16, 64, 64)).to("cuda", dtype=dtype)
    timestep = torch.tensor([0.5, 0.5]).to("cuda", dtype=torch.float32)
    pooled = torch.rand(bsz, 768).to("cuda", dtype=dtype)
    text = torch.rand((bsz, 512, 4096)).to("cuda", dtype=dtype)
    attn_mask = torch.tensor([[1.0] * 384 + [0.0] * 128] * bsz).to("cuda", dtype=dtype)  # Last 128 positions are masked

    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def _prepare_latent_image_ids(batch_size, height, width, device="cuda", dtype=dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size,
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels,
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    txt_ids = torch.zeros(bsz, text.shape[1], 3).to(device="cuda", dtype=dtype)

    vae_scale_factor = 16
    height = 2 * (int(512) // vae_scale_factor)
    width = 2 * (int(512) // vae_scale_factor)
    img_ids = _prepare_latent_image_ids(bsz, height, width)
    img = _pack_latents(img, img.shape[0], 16, height, width)

    # Gotta go fast
    transformer = FluxTransformer2DModel.from_config(
        {
            "attention_head_dim": 128,
            "guidance_embeds": True,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 4,
            "num_single_layers": 8,
            "patch_size": 1,
            "pooled_projection_dim": 768,
        }
    ).to("cuda", dtype=dtype)

    guidance = torch.tensor([2.0], device="cuda")
    guidance = guidance.expand(bsz)

    with torch.no_grad():
        no_mask = transformer(
            img,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        mask = transformer(
            img,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            attention_mask=attn_mask,
        )

    assert torch.allclose(no_mask.sample, mask.sample) is False
    # attention masking test passed - outputs differ as expected

    # Test TREAD
    # ------------------------------------------------------------------
    # 0. dummy inputs (same as before)
    # ------------------------------------------------------------------
    dtype = torch.bfloat16
    bsz = 2
    img = torch.rand((bsz, 16, 64, 64), device="cuda", dtype=dtype)
    timestep = torch.tensor([0.5, 0.5], device="cuda")
    pooled = torch.rand(bsz, 768, device="cuda", dtype=dtype)
    text = torch.rand((bsz, 512, 4096), device="cuda", dtype=dtype)

    attn_mask = torch.tensor([[1.0] * 384 + [0.0] * 128] * bsz, device="cuda", dtype=dtype)

    # helpers -----------------------------------------------------------------
    def _pack_latents(latents, batch, C, H, W):
        latents = latents.view(batch, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch, (H // 2) * (W // 2), C * 4)

    def _latent_ids(batch, H, W):
        ids = torch.zeros(H // 2, W // 2, 3)
        ids[..., 1] += torch.arange(H // 2)[:, None]
        ids[..., 2] += torch.arange(W // 2)[None, :]
        ids = ids.unsqueeze(0).repeat(batch, 1, 1, 1)
        return ids.reshape(batch, -1, 3).to("cuda", dtype=dtype)

    txt_ids = torch.zeros(bsz, text.size(1), 3, device="cuda", dtype=dtype)
    scale = 16
    H = 2 * (512 // scale)
    W = 2 * (512 // scale)
    img_ids = _latent_ids(bsz, H, W)
    img = _pack_latents(img, bsz, 16, H, W)

    # ------------------------------------------------------------------
    # 1. build model & simple routing schedule
    # ------------------------------------------------------------------
    transformer = FluxTransformer2DModel.from_config(
        dict(
            attention_head_dim=128,
            guidance_embeds=True,
            in_channels=64,
            joint_attention_dim=4096,
            num_attention_heads=24,
            num_layers=4,
            num_single_layers=8,
            patch_size=1,
            pooled_projection_dim=768,
        )
    ).to("cuda", dtype=dtype)

    # route keeps 50 % of tokens between global layers 2 and 5
    routes = [dict(selection_ratio=0.5, start_layer_idx=2, end_layer_idx=5)]
    transformer.set_router(
        TREADRouter(device="cuda"),
        routes,
    )

    transformer.train()  # routing only works in train mode
    guidance = torch.full((bsz,), 2.0, device="cuda")

    # ------------------------------------------------------------------
    # 2. single debug step
    # ------------------------------------------------------------------
    try:
        out = transformer(
            img,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            attention_mask=attn_mask,
        )
        loss = out.sample.mean()
        loss.backward()

        # forward & backward passed
        pass

    except Exception as e:
        import traceback

        # crash during step
        traceback.print_exc()
