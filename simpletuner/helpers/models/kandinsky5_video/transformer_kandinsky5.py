# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from torch import Tensor

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.qk_clip_logging import publish_attention_max_logits
from simpletuner.helpers.training.tread import TREADRouter

try:
    from diffusers.models.attention_dispatch import _CAN_USE_FLEX_ATTN, dispatch_attention_fn
except Exception:
    _CAN_USE_FLEX_ATTN = False

    def dispatch_attention_fn(attn, hidden_states, *args, **kwargs):
        # Fallback to direct module call when attention dispatch is unavailable.
        return attn(hidden_states, *args, **kwargs)


logger = logging.get_logger(__name__)


def _describe_tensor_debug(name: str, tensor: Optional[Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    try:
        return f"{name}(shape={tuple(tensor.shape)}, dim={tensor.dim()}, dtype={tensor.dtype}, device={tensor.device})"
    except Exception:
        return f"{name}=unavailable"


def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)
    return freqs


def fractal_flatten(x, rope, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = local_patching(x, shape, (1, pixel_size, pixel_size), dim=1)
        rope = local_patching(rope, shape, (1, pixel_size, pixel_size), dim=1)
        x = x.flatten(1, 2)
        rope = rope.flatten(1, 2)
    else:
        x = x.flatten(1, 3)
        rope = rope.flatten(1, 3)
    return x, rope


def fractal_unflatten(x, shape, block_mask=False):
    if block_mask:
        pixel_size = 8
        x = x.reshape(x.shape[0], -1, pixel_size**2, *x.shape[2:])
        x = local_merge(x, shape, (1, pixel_size, pixel_size), dim=1)
    else:
        x = x.reshape(*shape, *x.shape[2:])
    return x


def local_patching(x, shape, group_size, dim=0):
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x, shape, group_size, dim=0):
    batch_size, duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2 :],
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def nablaT_v2(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
):
    if _CAN_USE_FLEX_ATTN:
        from torch.nn.attention.flex_attention import BlockMask
    else:
        raise ValueError("Nabla attention is not supported with this version of PyTorch")

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()

    # Map estimation
    B, h, S, D = q.shape
    s1 = S // 64
    qa = q.reshape(B, h, s1, 64, D).mean(-2)
    ka = k.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
    map = qa @ ka

    map = torch.softmax(map / math.sqrt(D), dim=-1)
    # Map binarization
    vals, inds = map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    mask = torch.logical_or(mask, sta)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None)


class Kandinsky5TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.freqs = get_freqs(self.model_dim // 2, self.max_period)
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)
        # Signed-time embedding for TwinFlow-style negative time handling.
        self.time_sign_embed = nn.Embedding(2, time_dim)
        nn.init.zeros_(self.time_sign_embed.weight)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time, timestep_sign: Optional[torch.Tensor] = None):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        if timestep_sign is not None:
            sign_idx = (timestep_sign.view(-1) < 0).long().to(device=time_embed.device)
            time_embed = time_embed + self.time_sign_embed(sign_idx).to(dtype=time_embed.dtype, device=time_embed.device)
        return time_embed


class Kandinsky5TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class Kandinsky5VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        batch_size, duration, height, width, dim = x.shape
        x = (
            x.view(
                batch_size,
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class Kandinsky5RoPE1D(nn.Module):
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos):
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky5RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)):
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).repeat(batch_size, 1, height, width, 1),
                args_h.view(1, 1, height, 1, -1).repeat(batch_size, duration, 1, width, 1),
                args_w.view(1, 1, 1, width, -1).repeat(batch_size, duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky5Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))


class Kandinsky5AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, rotary_emb=None, sparse_params=None):
        def _describe_tensor(name: str, tensor: Optional[Tensor]) -> str:
            if tensor is None:
                return f"{name}=None"
            return f"{name}(shape={tuple(tensor.shape)}, dim={tensor.dim()}, dtype={tensor.dtype}, device={tensor.device})"

        def _describe_rotary(rope) -> str:
            if rope is None:
                return "rotary_emb=None"
            if isinstance(rope, (tuple, list)):
                parts = [_describe_tensor(f"rotary_emb[{idx}]", value) for idx, value in enumerate(rope)]
                return "; ".join(parts)
            return _describe_tensor("rotary_emb", rope)

        # query, key, value = self.get_qkv(x)
        query = attn.to_query(hidden_states)

        if encoder_hidden_states is not None:
            key = attn.to_key(encoder_hidden_states)
            value = attn.to_value(encoder_hidden_states)

            shape, cond_shape = query.shape[:-1], key.shape[:-1]
            query = query.reshape(*shape, attn.num_heads, -1)
            key = key.reshape(*cond_shape, attn.num_heads, -1)
            value = value.reshape(*cond_shape, attn.num_heads, -1)

        else:
            key = attn.to_key(hidden_states)
            value = attn.to_value(hidden_states)

            shape = query.shape[:-1]
            query = query.reshape(*shape, attn.num_heads, -1)
            key = key.reshape(*shape, attn.num_heads, -1)
            value = value.reshape(*shape, attn.num_heads, -1)

        # query, key = self.norm_qk(query, key)
        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)

        def apply_rotary(x, rope):
            x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
            x_out = (rope * x_).sum(dim=-1)
            return x_out.reshape(*x.shape).to(torch.bfloat16)

        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).type_as(query)
            key = apply_rotary(key, rotary_emb).type_as(key)

        if sparse_params is not None:
            attn_mask = nablaT_v2(
                query,
                key,
                sparse_params["sta_mask"],
                thr=sparse_params["P"],
            )

        else:
            attn_mask = None

        try:
            publish_attention_max_logits(
                query,
                key,
                attn_mask,
                getattr(attn, "to_query", None) and attn.to_query.weight,
                getattr(attn, "to_key", None) and attn.to_key.weight,
            )
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                backend=self._attention_backend,
            )
        except Exception:
            logger.error(
                "dispatch_attention_fn failed (backend=%s): %s; %s; %s; hidden_states=%s; encoder_hidden_states=%s; attn_mask=%s; %s",
                self._attention_backend,
                _describe_tensor("query", query),
                _describe_tensor("key", key),
                _describe_tensor("value", value),
                _describe_tensor("hidden_states", hidden_states),
                _describe_tensor("encoder_hidden_states", encoder_hidden_states),
                _describe_tensor("attn_mask", attn_mask),
                _describe_rotary(rotary_emb),
            )
            raise

        attn_output = attn_output.flatten(-2, -1)

        attn_out = attn.out_layer(attn_output)
        return attn_out


class Kandinsky5Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Kandinsky5AttnProcessor
    _available_processors = [
        Kandinsky5AttnProcessor,
    ]

    def __init__(self, num_channels, head_dim, processor=None):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        sparse_params: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_processor_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            sparse_params=sparse_params,
            rotary_emb=rotary_emb,
            **kwargs,
        )


class Kandinsky5FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class Kandinsky5OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky5Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed, text_embed, time_embed):
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)

        visual_embed = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0) + shift.float()[:, None, None]
        ).type_as(visual_embed)

        x = self.out_layer(visual_embed)

        batch_size, duration, height, width, _ = x.shape
        x = (
            x.view(
                batch_size,
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        return x


class Kandinsky5TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.text_modulation = Kandinsky5Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5Attention(model_dim, head_dim, processor=Kandinsky5AttnProcessor())

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        try:
            out = (self.self_attention_norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)
        except RuntimeError as err:
            debug_msg = "; ".join(
                [
                    "K5 text block modulation broadcast failed",
                    _describe_tensor_debug("x", x),
                    _describe_tensor_debug("shift", shift),
                    _describe_tensor_debug("scale", scale),
                    _describe_tensor_debug("gate", gate),
                    _describe_tensor_debug("time_embed", time_embed),
                    _describe_tensor_debug("rope", rope),
                ]
            )
            raise RuntimeError(debug_msg) from err
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (self.feed_forward_norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)
        out = self.feed_forward(out)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        return x


class Kandinsky5TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.visual_modulation = Kandinsky5Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5Attention(model_dim, head_dim, processor=Kandinsky5AttnProcessor())

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = Kandinsky5Attention(model_dim, head_dim, processor=Kandinsky5AttnProcessor())

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = (self.self_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            visual_embed
        )
        visual_out = self.self_attention(visual_out, rotary_emb=rope, sparse_params=sparse_params)
        visual_embed = (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = (self.cross_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            visual_embed
        )
        visual_out = self.cross_attention(visual_out, encoder_hidden_states=text_embed)
        visual_embed = (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = (self.feed_forward_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            visual_embed
        )
        visual_out = self.feed_forward(visual_out)
        visual_embed = (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)

        return visual_embed


class Kandinsky5Transformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    A 3D Diffusion Transformer model for video-like data.
    """

    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    _repeated_blocks = [
        "Kandinsky5TransformerEncoderBlock",
        "Kandinsky5TransformerDecoderBlock",
    ]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_type: str = "regular",
        attention_causal: bool = None,
        attention_local: bool = None,
        attention_glob: bool = None,
        attention_window: int = None,
        attention_P: float = None,
        attention_wT: int = None,
        attention_wW: int = None,
        attention_wH: int = None,
        attention_add_sta: bool = None,
        attention_method: str = None,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ):
        super().__init__()

        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.attention_type = attention_type

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim

        # Initialize embeddings
        self.time_embeddings = Kandinsky5TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = Kandinsky5TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = Kandinsky5TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = Kandinsky5VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        # Initialize positional embeddings
        self.text_rope_embeddings = Kandinsky5RoPE1D(head_dim)
        self.visual_rope_embeddings = Kandinsky5RoPE3D(axes_dims)

        # Initialize transformer blocks
        self.text_transformer_blocks = nn.ModuleList(
            [Kandinsky5TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_text_blocks)]
        )

        self.visual_transformer_blocks = nn.ModuleList(
            [Kandinsky5TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_visual_blocks)]
        )

        # Initialize output layer
        self.out_layer = Kandinsky5OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        self.gradient_checkpointing = False
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=num_text_blocks + num_visual_blocks,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        """Attach a TREAD router and route definitions."""
        self._tread_router = router
        self._tread_routes = routes

    @staticmethod
    def _route_rope(rope: torch.Tensor, info, keep_len: int) -> torch.Tensor:
        """
        Apply the router's shuffle/slice to the rotary embeddings so token positions stay aligned.
        """
        if rope is None:
            return rope

        if rope.dim() < 2:
            raise ValueError(f"Expected rotary embedding to have at least 2 dims (B, S, ...); got {rope.shape}")

        gather_idx = info.ids_shuffle.view(info.ids_shuffle.shape[0], info.ids_shuffle.shape[1], *([1] * (rope.dim() - 2)))
        gather_idx = gather_idx.expand_as(rope)
        routed = torch.take_along_dim(rope, gather_idx, dim=1)
        return routed[:, :keep_len]

    def forward(
        self,
        hidden_states: torch.Tensor,  # x
        encoder_hidden_states: torch.Tensor,  # text_embed
        timestep: torch.Tensor,  # time
        pooled_projections: torch.Tensor,  # pooled_text_embed
        visual_rope_pos: Tuple[int, int, int],
        text_rope_pos: torch.LongTensor,
        scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        force_keep_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        hidden_state_layer: Optional[int] = None,
        timestep_sign: Optional[torch.Tensor] = None,
    ) -> Union[Transformer2DModelOutput, torch.FloatTensor]:
        """
        Forward pass of the Kandinsky5 3D Transformer.

        Args:
            hidden_states (`torch.FloatTensor`): Input visual states
            encoder_hidden_states (`torch.FloatTensor`): Text embeddings
            timestep (`torch.Tensor` or `float` or `int`): Current timestep
            pooled_projections (`torch.FloatTensor`): Pooled text embeddings
            visual_rope_pos (`Tuple[int, int, int]`): Position for visual RoPE
            text_rope_pos (`torch.LongTensor`): Position for text RoPE
            scale_factor (`Tuple[float, float, float]`, optional): Scale factor for RoPE
            sparse_params (`Dict[str, Any]`, optional): Parameters for sparse attention
            force_keep_mask (`torch.Tensor`, *optional*): Boolean mask used by TREAD to prevent specific tokens from
                being dropped. Shape must match the routed token sequence length.
            return_dict (`bool`, optional): Whether to return a dictionary

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `torch.FloatTensor`: The output of the transformer
        """
        x = hidden_states
        text_embed = encoder_hidden_states
        time = timestep
        pooled_text_embed = pooled_projections
        batch_size = x.shape[0]

        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(
            time,
            timestep_sign=timestep_sign,
        )
        time_embed = time_embed + self.pooled_text_embeddings(pooled_text_embed)
        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_rope = text_rope.unsqueeze(dim=0)

        for text_transformer_block in self.text_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                text_embed = self._gradient_checkpointing_func(text_transformer_block, text_embed, time_embed, text_rope)
            else:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope)

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape, block_mask=to_fractal)

        grad_enabled = torch.is_grad_enabled()
        text_block_count = len(self.text_transformer_blocks)
        combined_blocks = list(self.text_transformer_blocks) + list(self.visual_transformer_blocks)
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(combined_blocks, visual_embed.device, grad_enabled)

        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_tokens = None
        current_rope = visual_rope

        if routes:
            total_layers = len(self.visual_transformer_blocks)

            def _to_pos(idx: int) -> int:
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **r,
                    "start_layer_idx": _to_pos(r["start_layer_idx"]),
                    "end_layer_idx": _to_pos(r["end_layer_idx"]),
                }
                for r in routes
            ]

        if use_routing and router is None:
            raise ValueError("TREAD routing requested but no router has been configured. Call set_router before training.")

        if use_routing and force_keep_mask is not None:
            if force_keep_mask.dim() > 2:
                force_keep_mask = force_keep_mask.view(force_keep_mask.shape[0], -1)
            expected = visual_embed.shape[1]
            if force_keep_mask.numel() == force_keep_mask.shape[0] * expected and force_keep_mask.shape[1] != expected:
                force_keep_mask = force_keep_mask.view(force_keep_mask.shape[0], expected)
            if force_keep_mask.shape[1] != expected:
                raise ValueError(
                    f"force_keep_mask has sequence length {force_keep_mask.shape[1]}, expected {expected} tokens."
                )
            force_keep_mask = force_keep_mask.to(device=visual_embed.device, dtype=torch.bool)

        captured_frame_hidden: Optional[torch.Tensor] = None

        for layer_idx, visual_transformer_block in enumerate(self.visual_transformer_blocks):
            global_idx = text_block_count + layer_idx
            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_in(visual_transformer_block, visual_embed.device)

            if use_routing and route_ptr < len(routes) and layer_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]
                tread_mask_info = router.get_mask(
                    visual_embed,
                    mask_ratio=mask_ratio,
                    force_keep=force_keep_mask,
                )
                saved_tokens = visual_embed.clone()
                visual_embed = router.start_route(visual_embed, tread_mask_info)
                current_rope = self._route_rope(visual_rope, tread_mask_info, keep_len=visual_embed.size(1))
                routing_now = True

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                visual_embed = self._gradient_checkpointing_func(
                    visual_transformer_block,
                    visual_embed,
                    text_embed,
                    time_embed,
                    current_rope,
                    sparse_params,
                )
            else:
                visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed, current_rope, sparse_params)

            if routing_now and layer_idx == routes[route_ptr]["end_layer_idx"]:
                visual_embed = router.end_route(
                    visual_embed,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                routing_now = False
                route_ptr += 1
                current_rope = visual_rope
            if (
                output_hidden_states
                and not routing_now
                and visual_embed.shape[1] == visual_shape[1] * visual_shape[2] * visual_shape[3]
                and (hidden_state_layer is None or layer_idx == hidden_state_layer)
            ):
                captured_frame_hidden = visual_embed.reshape(
                    batch_size,
                    visual_shape[1],
                    visual_shape[2] * visual_shape[3],
                    -1,
                )
                if hidden_state_layer is not None and layer_idx == hidden_state_layer:
                    output_hidden_states = False

            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_out(visual_transformer_block)

        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)

        if not return_dict:
            if captured_frame_hidden is None:
                return x
            return x, captured_frame_hidden

        result = Transformer2DModelOutput(sample=x)
        if captured_frame_hidden is not None:
            result.crepa_hidden_states = captured_frame_hidden
        return result
