# Copyright 2025 Black Forest Labs, The HuggingFace Team and loadstone-rock.
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import FluxPosEmbed, PixArtAlphaTextProjection, Timesteps, get_timestep_embedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import CombinedTimestepLabelEmbeddings, FP32LayerNorm, RMSNorm
from diffusers.models.transformers.transformer_flux import FluxAttention, FluxAttnProcessor
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from simpletuner.helpers.training.tread import TREADRouter


class ChromaAdaLayerNormZeroPruned(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.flatten(1, 2).chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class ChromaAdaLayerNormZeroSinglePruned(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa = emb.flatten(1, 2).chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class ChromaAdaLayerNormContinuousPruned(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(emb.flatten(1, 2).to(x.dtype), 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class ChromaCombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, num_channels: int, out_dim: int):
        super().__init__()

        self.time_proj = Timesteps(num_channels=num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.guidance_proj = Timesteps(num_channels=num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.register_buffer(
            "mod_proj",
            get_timestep_embedding(
                torch.arange(out_dim) * 1000, 2 * num_channels, flip_sin_to_cos=True, downscale_freq_shift=0
            ),
            persistent=False,
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        mod_index_length = self.mod_proj.shape[0]
        batch_size = timestep.shape[0]

        timesteps_proj = self.time_proj(timestep).to(dtype=timestep.dtype)
        guidance_proj = self.guidance_proj(torch.tensor([0] * batch_size)).to(dtype=timestep.dtype, device=timestep.device)

        mod_proj = self.mod_proj.to(dtype=timesteps_proj.dtype, device=timesteps_proj.device).repeat(batch_size, 1, 1)
        timestep_guidance = torch.cat([timesteps_proj, guidance_proj], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        input_vec = torch.cat([timestep_guidance, mod_proj], dim=-1)
        return input_vec.to(timestep.dtype)


class ChromaApproximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [PixArtAlphaTextProjection(hidden_dim, hidden_dim, act_fn="silu") for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        return self.out_proj(x)


@maybe_allow_in_graph
class ChromaSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = ChromaAdaLayerNormZeroSinglePruned(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            from diffusers.models.attention_processor import FluxAttnProcessor2_0_NPU

            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor()

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class ChromaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = ChromaAdaLayerNormZeroPruned(dim)
        self.norm1_context = ChromaAdaLayerNormZeroPruned(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb_img, temb_txt = temb[:, :6], temb[:, 6:]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_img)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb_txt
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class ChromaTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux, modified for Chroma.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ChromaTransformerBlock", "ChromaSingleTransformerBlock"]
    _repeated_blocks = ["ChromaTransformerBlock", "ChromaSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),
        approximator_num_channels: int = 64,
        approximator_hidden_dim: int = 5120,
        approximator_layers: int = 5,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_text_embed = ChromaCombinedTimestepTextProjEmbeddings(
            num_channels=approximator_num_channels // 4,
            out_dim=3 * num_single_layers + 2 * 6 * num_layers + 2,
        )
        self.distilled_guidance_layer = ChromaApproximator(
            in_dim=approximator_num_channels,
            out_dim=self.inner_dim,
            hidden_dim=approximator_hidden_dim,
            n_layers=approximator_layers,
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                ChromaTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                ChromaSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = ChromaAdaLayerNormContinuousPruned(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        self._tread_router = router
        self._tread_routes = routes

    @staticmethod
    def _route_rope(rope, info, keep_len: int, batch: int):
        """
        Apply the router's shuffle and slice to rotary embeddings.

        Accepts either a tensor of shape (S, D) or a tuple (cos, sin).
        Returns a batched tensor (B, keep_len, D) or tuple of tensors.
        """

        def _route_one(r: torch.Tensor) -> torch.Tensor:
            rB = r.unsqueeze(0).expand(batch, -1, -1)
            shuf = torch.take_along_dim(rB, info.ids_shuffle.unsqueeze(-1).expand_as(rB), dim=1)
            return shuf[:, :keep_len, :]

        if isinstance(rope, tuple):
            return tuple(_route_one(r) for r in rope)
        return _route_one(rope)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        force_keep_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000

        input_vec = self.time_text_embed(timestep)
        pooled_temb = self.distilled_guidance_layer(input_vec)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        txt_len = encoder_hidden_states.shape[1]

        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_tokens = None
        global_idx = 0
        current_rope = image_rotary_emb

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
            img_offset = 3 * len(self.single_transformer_blocks)
            txt_offset = img_offset + 6 * len(self.transformer_blocks)
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            temb = torch.cat(
                (
                    pooled_temb[:, img_modulation : img_modulation + 6],
                    pooled_temb[:, text_modulation : text_modulation + 6],
                ),
                dim=1,
            )

            if use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]
                mask_for_stage = force_keep_mask
                if mask_for_stage is not None and mask_for_stage.shape[1] != hidden_states.shape[1]:
                    mask_for_stage = mask_for_stage[:, -hidden_states.shape[1] :]
                tread_mask_info = router.get_mask(
                    hidden_states,
                    mask_ratio=mask_ratio,
                    force_keep=mask_for_stage,
                )
                saved_tokens = hidden_states.clone()
                hidden_states = router.start_route(hidden_states, tread_mask_info)
                routing_now = True

                text_rope_b = tuple(r[:txt_len].unsqueeze(0).expand(hidden_states.size(0), -1, -1) for r in image_rotary_emb)
                image_only_rope = tuple(r[txt_len:] for r in image_rotary_emb)
                img_rope_r = self._route_rope(
                    image_only_rope,
                    tread_mask_info,
                    keep_len=hidden_states.size(1),
                    batch=hidden_states.size(0),
                )
                current_rope = tuple(torch.cat([tr, ir], dim=1) for tr, ir in zip(text_rope_b, img_rope_r))

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    current_rope,
                    attention_mask,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=current_rope,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            if routing_now and route_ptr < len(routes) and global_idx == routes[route_ptr]["end_layer_idx"]:
                hidden_states = router.end_route(
                    hidden_states,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                routing_now = False
                route_ptr += 1
                current_rope = image_rotary_emb

            global_idx += 1

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        txt_len = encoder_hidden_states.shape[1]

        for index_block, block in enumerate(self.single_transformer_blocks):
            start_idx = 3 * index_block
            temb = pooled_temb[:, start_idx : start_idx + 3]

            if use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]

                text_tok = hidden_states[:, :txt_len, :]
                img_tok = hidden_states[:, txt_len:, :]

                fkm = None
                if force_keep_mask is not None:
                    if force_keep_mask.shape[1] == img_tok.shape[1]:
                        fkm = force_keep_mask
                    elif force_keep_mask.shape[1] == txt_len + img_tok.shape[1]:
                        fkm = force_keep_mask[:, txt_len:]
                    else:
                        fkm = force_keep_mask[:, -img_tok.shape[1] :]
                tread_mask_info = router.get_mask(
                    img_tok,
                    mask_ratio=mask_ratio,
                    force_keep=fkm,
                )
                saved_tokens = img_tok.clone()
                img_tok = router.start_route(img_tok, tread_mask_info)

                hidden_states = torch.cat([text_tok, img_tok], dim=1)
                routing_now = True

                if attention_mask is not None:
                    pad = torch.zeros(
                        attention_mask.size(0),
                        img_tok.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask[:, :txt_len], pad], dim=1)

                text_rope_b = tuple(r[:txt_len].unsqueeze(0).expand(img_tok.size(0), -1, -1) for r in image_rotary_emb)
                img_rope_r = self._route_rope(
                    tuple(r[txt_len:] for r in image_rotary_emb),
                    tread_mask_info,
                    keep_len=img_tok.size(1),
                    batch=img_tok.size(0),
                )
                current_rope = tuple(torch.cat([tr, ir], dim=1) for tr, ir in zip(text_rope_b, img_rope_r))

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    current_rope,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=current_rope,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

            if routing_now and route_ptr < len(routes) and global_idx == routes[route_ptr]["end_layer_idx"]:
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

        temb = pooled_temb[:, -2:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
