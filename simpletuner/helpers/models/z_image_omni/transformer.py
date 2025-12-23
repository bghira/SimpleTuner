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

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.loaders import peft as diffusers_peft
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch.nn.utils.rnn import pad_sequence

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.qk_clip_logging import publish_attention_max_logits
from simpletuner.helpers.training.tread import TREADRouter

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

logger = logging.getLogger(__name__)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor, image_tokens_start: int | None = None):
    if buffer is None:
        return
    if image_tokens_start is not None and hidden_states.dim() >= 3:
        buffer[key] = hidden_states[:, image_tokens_start:, ...]
    else:
        buffer[key] = hidden_states


# Clamp NaN/Inf that can appear when running in float16. Ported from ComfyUI commit daaceac769a1355ab975758ede064317ea7514b4.
def clamp_fp16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


# Ensure diffusers knows how to scale adapters for this transformer type.
if "ZImageOmniTransformer2DModel" not in diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING:
    diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING["ZImageOmniTransformer2DModel"] = lambda model_cls, weights: weights


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256, enable_time_sign_embed: bool = False):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.time_sign_embed: Optional[nn.Embedding] = None
        if enable_time_sign_embed:
            self.time_sign_embed = nn.Embedding(2, out_size)
            nn.init.zeros_(self.time_sign_embed.weight)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t, timestep_sign: Optional[torch.Tensor] = None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable TwinFlow (or load a TwinFlow-compatible checkpoint) to use signed-timestep conditioning."
                )
            sign_idx = (timestep_sign.view(-1) < 0).long().to(device=t_emb.device)
            t_emb = t_emb + self.time_sign_embed(sign_idx).to(dtype=t_emb.dtype, device=t_emb.device)
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        publish_attention_max_logits(
            query,
            key,
            attention_mask,
            getattr(attn, "to_q", None) and attn.to_q.weight,
            getattr(attn, "to_k", None) and attn.to_k.weight,
        )
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            if noise_mask is not None and adaln_noisy is not None and adaln_clean is not None:
                batch_size, seq_len = x.shape[0], x.shape[1]

                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                noise_mask_expanded = noise_mask.unsqueeze(-1)
                scale_msa = torch.where(
                    noise_mask_expanded == 1,
                    scale_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    scale_msa_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                scale_mlp = torch.where(
                    noise_mask_expanded == 1,
                    scale_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    scale_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                gate_msa = torch.where(
                    noise_mask_expanded == 1,
                    gate_msa_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    gate_msa_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
                gate_mlp = torch.where(
                    noise_mask_expanded == 1,
                    gate_mlp_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                    gate_mlp_clean.unsqueeze(1).expand(-1, seq_len, -1),
                )
            else:
                assert adaln_input is not None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = clamp_fp16(
                self.attention(
                    self.attention_norm1(x) * scale_msa,
                    attention_mask=attn_mask,
                    freqs_cis=freqs_cis,
                )
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            x = x + gate_mlp * self.ffn_norm2(
                clamp_fp16(
                    self.feed_forward(self.ffn_norm1(x) * scale_mlp),
                )
            )
        else:
            attn_out = clamp_fp16(
                self.attention(
                    self.attention_norm1(x),
                    attention_mask=attn_mask,
                    freqs_cis=freqs_cis,
                )
            )
            x = x + self.attention_norm2(attn_out)

            x = x + self.ffn_norm2(
                clamp_fp16(
                    self.feed_forward(
                        self.ffn_norm1(x),
                    )
                )
            )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        if noise_mask is not None and c_noisy is not None and c_clean is not None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)

            noise_mask_expanded = noise_mask.unsqueeze(-1)
            scale = torch.where(
                noise_mask_expanded == 1,
                scale_noisy.unsqueeze(1).expand(-1, seq_len, -1),
                scale_clean.unsqueeze(1).expand(-1, seq_len, -1),
            )
        else:
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)
        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        max_ids = [int(ids[:, i].max().item()) if ids.numel() > 0 else -1 for i in range(len(self.axes_dims))]

        if self.freqs_cis is None:
            target_lens = [
                max(self.axes_lens[i], max_ids[i] + 1) if max_ids[i] >= 0 else self.axes_lens[i]
                for i in range(len(self.axes_dims))
            ]
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, target_lens, theta=self.theta)
        else:
            # Grow cached frequencies if incoming position ids exceed the precomputed range.
            for i, max_id in enumerate(max_ids):
                needed = max_id + 1
                if needed > self.freqs_cis[i].shape[0]:
                    new_len = max(self.axes_lens[i], needed)
                    self.freqs_cis[i] = self.precompute_freqs_cis(
                        [self.axes_dims[i]],
                        [new_len],
                        theta=self.theta,
                    )[0]

        if any(freqs_cis.device != device for freqs_cis in self.freqs_cis):
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        freqs_cis = torch.cat(result, dim=-1)
        target_dim = sum(self.axes_dims) // 2
        if freqs_cis.shape[-1] > target_dim:
            freqs_cis = freqs_cis[..., :target_dim]
        return freqs_cis


class ZImageOmniTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=1152,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        enable_time_sign_embed: bool = False,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.siglip_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    2000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(
            min(dim, ADALN_EMBED_DIM),
            mid_size=1024,
            enable_time_sign_embed=enable_time_sign_embed,
        )
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))
        self.siglip_embedder = nn.Sequential(
            RMSNorm(siglip_feat_dim, eps=norm_eps), nn.Linear(siglip_feat_dim, dim, bias=True)
        )
        self.siglip_feat_dim = siglip_feat_dim

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm) for layer_id in range(n_layers)]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=n_layers,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        self._tread_router = router
        self._tread_routes = routes

    def unpatchify(
        self,
        unified: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        x_pos_offsets,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(unified)
        assert len(size) == bsz

        x = []
        for i in range(bsz):
            x_item = []
            unified_x = unified[i][x_pos_offsets[i][0] : x_pos_offsets[i][1]]
            cu_len = 0
            for j in range(len(size[i])):
                if size[i][j] is None:
                    x_item.append(None)
                    ori_len = 0
                    pad_len = SEQ_MULTI_OF
                    cu_len += pad_len + ori_len
                else:
                    F, H, W = size[i][j]
                    ori_len = (F // pF) * (H // pH) * (W // pW)
                    pad_len = (-ori_len) % SEQ_MULTI_OF
                    x_item.append(
                        einops.rearrange(
                            unified_x[cu_len : cu_len + ori_len].view(
                                F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels
                            ),
                            "f h w pf ph pw c -> c (f pf) (h ph) (w pw)",
                        )
                    )
                    cu_len += ori_len + pad_len
            x.append(x_item[-1])
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_x,
        all_cap_feats,
        all_siglip_feats,
        patch_size: int,
        f_patch_size: int,
        images_noise_mask: List[List[int]],
    ):
        bsz = len(all_x)
        pH = pW = patch_size
        pF = f_patch_size
        device = all_x[0][-1].device
        dtype = all_x[0][-1].dtype
        sig_pad_dim = self.siglip_feat_dim

        all_x_padded = []
        all_x_size = []
        all_x_pos_ids = []
        all_x_pad_mask = []
        all_x_len = []
        all_x_noise_mask = []
        all_cap_padded_feats = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_len = []
        all_cap_noise_mask = []
        all_siglip_padded_feats = []
        all_siglip_pos_ids = []
        all_siglip_pad_mask = []
        all_siglip_len = []
        all_siglip_noise_mask = []

        for i in range(bsz):
            num_images = len(all_x[i])
            cap_padded_feats = []
            cap_item_cu_len = 1
            cap_start_pos = []
            cap_end_pos = []
            cap_padded_pos_ids = []
            cap_pad_mask = []
            cap_len = []
            cap_noise_mask = []
            for j, cap_item in enumerate(all_cap_feats[i]):
                cap_item_ori_len = len(cap_item)
                cap_item_padding_len = (-cap_item_ori_len) % SEQ_MULTI_OF
                cap_len.append(cap_item_ori_len + cap_item_padding_len)
                cap_item_padding_pos_ids = (
                    self.create_coordinate_grid(
                        size=(1, 1, 1),
                        start=(0, 0, 0),
                        device=device,
                    )
                    .flatten(0, 2)
                    .repeat(cap_item_padding_len, 1)
                )
                cap_start_pos.append(cap_item_cu_len)
                cap_item_ori_pos_ids = self.create_coordinate_grid(
                    size=(cap_item_ori_len, 1, 1),
                    start=(cap_item_cu_len, 0, 0),
                    device=device,
                ).flatten(0, 2)
                cap_padded_pos_ids.append(cap_item_ori_pos_ids)
                cap_padded_pos_ids.append(cap_item_padding_pos_ids)
                cap_item_cu_len += cap_item_ori_len
                cap_end_pos.append(cap_item_cu_len)
                cap_item_cu_len += 2
                cap_pad_mask.append(torch.zeros((cap_item_ori_len,), dtype=torch.bool, device=device))
                cap_pad_mask.append(torch.ones((cap_item_padding_len,), dtype=torch.bool, device=device))
                cap_item_padded_feat = torch.cat([cap_item, cap_item[-1:].repeat(cap_item_padding_len, 1)], dim=0)
                cap_padded_feats.append(cap_item_padded_feat)
                if j < len(images_noise_mask[i]):
                    cap_noise_mask.extend([images_noise_mask[i][j]] * (cap_item_ori_len + cap_item_padding_len))
                else:
                    cap_noise_mask.extend([1] * (cap_item_ori_len + cap_item_padding_len))

            all_cap_noise_mask.append(cap_noise_mask)
            cap_padded_pos_ids = torch.cat(cap_padded_pos_ids, dim=0)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            cap_pad_mask = torch.cat(cap_pad_mask, dim=0)
            all_cap_pad_mask.append(cap_pad_mask)
            all_cap_padded_feats.append(torch.cat(cap_padded_feats, dim=0))
            all_cap_len.append(cap_len)

            x_padded = []
            x_padded_pos_ids = []
            x_pad_mask = []
            x_len = []
            x_size = []
            x_noise_mask = []
            for j, x_item in enumerate(all_x[i]):
                if x_item is not None:
                    C, F, H, W = x_item.size()
                    x_size.append((F, H, W))
                    F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

                    x_item = x_item.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
                    x_item = einops.rearrange(x_item, "c f pf h ph w pw -> (f h w) (pf ph pw c)")

                    x_item_ori_len = len(x_item)
                    x_item_padding_len = (-x_item_ori_len) % SEQ_MULTI_OF
                    x_len.append(x_item_ori_len + x_item_padding_len)
                    x_item_padding_pos_ids = (
                        self.create_coordinate_grid(
                            size=(1, 1, 1),
                            start=(0, 0, 0),
                            device=device,
                        )
                        .flatten(0, 2)
                        .repeat(x_item_padding_len, 1)
                    )
                    x_item_ori_pos_ids = self.create_coordinate_grid(
                        size=(F_tokens, H_tokens, W_tokens), start=(cap_end_pos[j], 0, 0), device=device
                    ).flatten(0, 2)
                    x_padded_pos_ids.append(x_item_ori_pos_ids)
                    x_padded_pos_ids.append(x_item_padding_pos_ids)

                    x_pad_mask.append(torch.zeros((x_item_ori_len,), dtype=torch.bool, device=device))
                    x_pad_mask.append(torch.ones((x_item_padding_len,), dtype=torch.bool, device=device))
                    x_item_padded_feat = torch.cat([x_item, x_item[-1:].repeat(x_item_padding_len, 1)], dim=0)
                    x_padded.append(x_item_padded_feat)
                    x_noise_mask.extend([images_noise_mask[i][j]] * (x_item_ori_len + x_item_padding_len))
                else:
                    x_pad_dim = 64
                    x_item_ori_len = 0
                    x_item_padding_len = SEQ_MULTI_OF
                    x_size.append(None)
                    x_item_padding_pos_ids = (
                        self.create_coordinate_grid(
                            size=(1, 1, 1),
                            start=(0, 0, 0),
                            device=device,
                        )
                        .flatten(0, 2)
                        .repeat(x_item_padding_len, 1)
                    )
                    x_len.append(x_item_ori_len + x_item_padding_len)
                    x_padded_pos_ids.append(x_item_padding_pos_ids)
                    x_pad_mask.append(torch.ones((x_item_padding_len,), dtype=torch.bool, device=device))
                    x_padded.append(torch.zeros((x_item_padding_len, x_pad_dim), dtype=dtype, device=device))
                    x_noise_mask.extend([images_noise_mask[i][j]] * x_item_padding_len)

            all_x_noise_mask.append(x_noise_mask)
            all_x_size.append(x_size)
            x_padded_pos_ids = torch.cat(x_padded_pos_ids, dim=0)
            all_x_pos_ids.append(x_padded_pos_ids)
            x_pad_mask = torch.cat(x_pad_mask, dim=0)
            all_x_pad_mask.append(x_pad_mask)
            all_x_padded.append(torch.cat(x_padded, dim=0))
            all_x_len.append(x_len)

            sig_padded_feats = []
            sig_padded_pos_ids = []
            sig_pad_mask = []
            sig_len = []
            sig_noise_mask = []

            siglip_entries = all_siglip_feats[i]
            no_siglip_data = siglip_entries is None
            if siglip_entries is None:
                siglip_entries = [None for _ in range(num_images)]

            for j, sig_item in enumerate(siglip_entries):
                if sig_item is not None:
                    sig_H, sig_W, sig_C = sig_item.size()
                    sig_H_tokens, sig_W_tokens, sig_F_tokens = sig_H, sig_W, 1

                    sig_item = sig_item.view(sig_C, sig_F_tokens, 1, sig_H_tokens, 1, sig_W_tokens, 1)
                    sig_item = einops.rearrange(sig_item, "c f pf h ph w pw -> (f h w) (pf ph pw c)")

                    sig_item_ori_len = len(sig_item)
                    sig_item_padding_len = (-sig_item_ori_len) % SEQ_MULTI_OF
                    sig_len.append(sig_item_ori_len + sig_item_padding_len)
                    sig_item_padding_pos_ids = (
                        self.create_coordinate_grid(
                            size=(1, 1, 1),
                            start=(0, 0, 0),
                            device=device,
                        )
                        .flatten(0, 2)
                        .repeat(sig_item_padding_len, 1)
                    )
                    sig_item_ori_pos_ids = self.create_coordinate_grid(
                        size=(sig_F_tokens, sig_H_tokens, sig_W_tokens), start=(cap_end_pos[j] + 1, 0, 0), device=device
                    )
                    sig_item_ori_pos_ids[..., 1] = sig_item_ori_pos_ids[..., 1] / (sig_H_tokens - 1) * (x_size[j][1] - 1)
                    sig_item_ori_pos_ids[..., 2] = sig_item_ori_pos_ids[..., 2] / (sig_W_tokens - 1) * (x_size[j][2] - 1)
                    sig_item_ori_pos_ids = sig_item_ori_pos_ids.flatten(0, 2)
                    sig_padded_pos_ids.append(sig_item_ori_pos_ids)
                    sig_padded_pos_ids.append(sig_item_padding_pos_ids)

                    sig_pad_mask.append(torch.zeros((sig_item_ori_len,), dtype=torch.bool, device=device))
                    sig_pad_mask.append(torch.ones((sig_item_padding_len,), dtype=torch.bool, device=device))
                    sig_item_padded_feat = torch.cat([sig_item, sig_item[-1:].repeat(sig_item_padding_len, 1)], dim=0)
                    sig_padded_feats.append(sig_item_padded_feat)
                    sig_noise_mask.extend([images_noise_mask[i][j]] * (sig_item_ori_len + sig_item_padding_len))
                else:
                    sig_item_padding_len = 0 if no_siglip_data else SEQ_MULTI_OF
                    sig_len.append(sig_item_padding_len)
                    sig_item_padding_pos_ids = (
                        self.create_coordinate_grid(
                            size=(1, 1, 1),
                            start=(0, 0, 0),
                            device=device,
                        )
                        .flatten(0, 2)
                        .repeat(sig_item_padding_len, 1)
                    )
                    sig_padded_pos_ids.append(sig_item_padding_pos_ids)
                    sig_pad_mask.append(torch.ones((sig_item_padding_len,), dtype=torch.bool, device=device))
                    sig_padded_feats.append(torch.zeros((sig_item_padding_len, sig_pad_dim), dtype=dtype, device=device))
                    sig_noise_mask.extend([images_noise_mask[i][j]] * sig_item_padding_len)

            all_siglip_noise_mask.append(sig_noise_mask)
            all_siglip_pos_ids.append(
                torch.cat(sig_padded_pos_ids, dim=0)
                if len(sig_padded_pos_ids) > 0
                else torch.zeros((0, 3), dtype=torch.int32, device=device)
            )
            all_siglip_pad_mask.append(
                torch.cat(sig_pad_mask, dim=0)
                if len(sig_pad_mask) > 0
                else torch.zeros((0,), dtype=torch.bool, device=device)
            )
            all_siglip_padded_feats.append(
                torch.cat(sig_padded_feats, dim=0)
                if len(sig_padded_feats) > 0
                else torch.zeros((0, sig_pad_dim), dtype=dtype, device=device)
            )
            all_siglip_len.append(sig_len)

        all_x_pos_offsets = []
        for i in range(bsz):
            start = sum(all_cap_len[i])
            end = start + sum(all_x_len[i])
            all_x_pos_offsets.append((start, end))
            assert all_x_padded[i].shape[0] + all_cap_padded_feats[i].shape[0] == sum(all_cap_len[i]) + sum(
                all_x_len[i]
            ), f"Batch item {i}: x length {all_x_padded[i].shape[0]} + cap length {all_cap_padded_feats[i].shape[0]} != sum(all_cap_len[i]) + sum(all_x_len[i]) {sum(all_cap_len[i]) + sum(all_x_len[i])}"

        return (
            all_x_padded,
            all_cap_padded_feats,
            all_siglip_padded_feats,
            all_x_size,
            all_x_pos_ids,
            all_cap_pos_ids,
            all_siglip_pos_ids,
            all_x_pad_mask,
            all_cap_pad_mask,
            all_siglip_pad_mask,
            all_x_pos_offsets,
            all_x_noise_mask,
            all_cap_noise_mask,
            all_siglip_noise_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[List[torch.Tensor]],
        cond_latents: List[List[torch.Tensor]],
        siglip_feats: List[List[torch.Tensor]],
        patch_size=2,
        f_patch_size=1,
        timestep_sign: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        hidden_states_buffer: Optional[dict] = None,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = torch.cat([t, torch.ones_like(t, dtype=t.dtype, device=device)], dim=0)
        if timestep_sign is not None:
            timestep_sign = torch.cat(
                [timestep_sign, torch.ones_like(timestep_sign, dtype=timestep_sign.dtype, device=device)],
                dim=0,
            )
        t = t * self.t_scale
        t = self.t_embedder(t, timestep_sign=timestep_sign)

        t_noisy = t[:bsz]
        t_clean = t[bsz:]

        x = [cond_latents[i] + [x[i]] for i in range(bsz)]
        image_noise_mask = [[0] * (len(x[i]) - 1) + [1] for i in range(bsz)]

        (
            x,
            cap_feats,
            siglip_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            siglip_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
            siglip_inner_pad_mask,
            x_pos_offsets,
            x_noise_mask,
            cap_noise_mask,
            siglip_noise_mask,
        ) = self.patchify_and_embed(x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask)

        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_noise_mask_tensor = []
        for i in range(bsz):
            x_mask = torch.tensor(x_noise_mask[i], dtype=torch.long, device=device)
            x_noise_mask_tensor.append(x_mask)
        x_noise_mask_tensor = pad_sequence(x_noise_mask_tensor, batch_first=True, padding_value=0)
        x_noise_mask_tensor = x_noise_mask_tensor[:, : x.shape[1]]

        t_noisy_x = t_noisy.type_as(x)
        t_clean_x = t_clean.type_as(x)

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(
                    layer,
                    x,
                    x_attn_mask,
                    x_freqs_cis,
                    noise_mask=x_noise_mask_tensor,
                    adaln_noisy=t_noisy_x,
                    adaln_clean=t_clean_x,
                )
        else:
            for layer in self.noise_refiner:
                x = layer(
                    x,
                    x_attn_mask,
                    x_freqs_cis,
                    noise_mask=x_noise_mask_tensor,
                    adaln_noisy=t_noisy_x,
                    adaln_clean=t_clean_x,
                )

        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis)
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        siglip_present = any(feats is not None and torch.is_tensor(feats) and feats.shape[0] > 0 for feats in siglip_feats)
        if siglip_present:
            siglip_item_seqlens = [len(_) for _ in siglip_feats]
            siglip_max_item_seqlen = max(siglip_item_seqlens)

            siglip_feats = torch.cat(siglip_feats, dim=0)
            siglip_feats = self.siglip_embedder(siglip_feats)
            siglip_feats[torch.cat(siglip_inner_pad_mask)] = self.siglip_pad_token
            siglip_feats = list(siglip_feats.split(siglip_item_seqlens, dim=0))
            siglip_freqs_cis = list(
                self.rope_embedder(torch.cat(siglip_pos_ids, dim=0)).split([len(_) for _ in siglip_pos_ids], dim=0)
            )

            siglip_feats = pad_sequence(siglip_feats, batch_first=True, padding_value=0.0)
            siglip_freqs_cis = pad_sequence(siglip_freqs_cis, batch_first=True, padding_value=0.0)
            siglip_freqs_cis = siglip_freqs_cis[:, : siglip_feats.shape[1]]

            siglip_attn_mask = torch.zeros((bsz, siglip_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(siglip_item_seqlens):
                siglip_attn_mask[i, :seq_len] = 1

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for layer in self.siglip_refiner:
                    siglip_feats = self._gradient_checkpointing_func(layer, siglip_feats, siglip_attn_mask, siglip_freqs_cis)
            else:
                for layer in self.siglip_refiner:
                    siglip_feats = layer(siglip_feats, siglip_attn_mask, siglip_freqs_cis)

        unified = []
        unified_freqs_cis = []
        unified_noise_mask = []
        if siglip_present:
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                cap_len = cap_item_seqlens[i]
                siglip_len = siglip_item_seqlens[i]
                unified.append(torch.cat([cap_feats[i][:cap_len], x[i][:x_len], siglip_feats[i][:siglip_len]]))
                unified_freqs_cis.append(
                    torch.cat([cap_freqs_cis[i][:cap_len], x_freqs_cis[i][:x_len], siglip_freqs_cis[i][:siglip_len]])
                )
                unified_noise_mask.append(
                    torch.tensor(cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i], dtype=torch.long, device=device)
                )
            unified_item_seqlens = [a + b + c for a, b, c in zip(cap_item_seqlens, x_item_seqlens, siglip_item_seqlens)]
        else:
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                cap_len = cap_item_seqlens[i]
                unified.append(torch.cat([cap_feats[i][:cap_len], x[i][:x_len]]))
                unified_freqs_cis.append(torch.cat([cap_freqs_cis[i][:cap_len], x_freqs_cis[i][:x_len]]))
                unified_noise_mask.append(torch.tensor(cap_noise_mask[i] + x_noise_mask[i], dtype=torch.long, device=device))
            unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        unified_noise_mask_tensor = pad_sequence(unified_noise_mask, batch_first=True, padding_value=0)
        unified_noise_mask_tensor = unified_noise_mask_tensor[:, : unified.shape[1]]

        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        if use_routing and router is None:
            raise ValueError("TREAD routing requested but no router has been configured. Call set_router before training.")

        if routes:
            total_layers = len(self.layers)

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

        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_tokens = None
        saved_freqs = None
        saved_attn = None
        saved_noise_mask = None

        def apply_layer(layer_module, h, attn_mask, freqs, noise_mask):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                return self._gradient_checkpointing_func(
                    layer_module,
                    h,
                    attn_mask,
                    freqs,
                    noise_mask=noise_mask,
                    adaln_noisy=t_noisy_x,
                    adaln_clean=t_clean_x,
                )
            return layer_module(
                h,
                attn_mask,
                freqs,
                noise_mask=noise_mask,
                adaln_noisy=t_noisy_x,
                adaln_clean=t_clean_x,
            )

        skip_set = set(skip_layers) if skip_layers is not None else set()
        capture_idx = 0

        # Musubi block swap activation
        combined_blocks = list(self.layers)
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        grad_enabled = torch.is_grad_enabled()
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(combined_blocks, unified.device, grad_enabled)

        for idx, layer in enumerate(self.layers):
            if musubi_offload_active and musubi_manager.is_managed_block(idx):
                musubi_manager.stream_in(layer, unified.device)
            if use_routing and route_ptr < len(routes) and idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]
                force_keep = torch.zeros_like(unified_attn_mask)
                for b in range(bsz):
                    cap_len = cap_item_seqlens[b]
                    force_keep[b, :cap_len] = True  # keep captions
                    force_keep[b, unified_item_seqlens[b] :] = True  # keep padding tail
                if force_keep_mask is not None:
                    force_keep = force_keep | force_keep_mask
                tread_mask_info = router.get_mask(unified, mask_ratio=mask_ratio, force_keep=force_keep)
                saved_tokens = unified.clone()
                saved_freqs = unified_freqs_cis.clone()
                saved_attn = unified_attn_mask.clone()
                saved_noise_mask = unified_noise_mask_tensor.clone()
                unified = router.start_route(unified, tread_mask_info)
                unified_freqs_cis = router.start_route(unified_freqs_cis, tread_mask_info)
                unified_noise_mask_tensor = router.start_route(unified_noise_mask_tensor, tread_mask_info)
                unified_attn_mask = torch.ones((bsz, unified.shape[1]), dtype=torch.bool, device=unified.device)
                routing_now = True

            if idx in skip_set:
                layer_out = unified
            else:
                layer_out = apply_layer(layer, unified, unified_attn_mask, unified_freqs_cis, unified_noise_mask_tensor)
            unified = layer_out

            if hidden_states_buffer is not None:
                img_tokens = torch.zeros(
                    (bsz, x_max_item_seqlen, unified.shape[-1]),
                    dtype=unified.dtype,
                    device=unified.device,
                )
                for b in range(bsz):
                    cap_len = cap_item_seqlens[b]
                    x_len = x_item_seqlens[b]
                    img_tokens[b, :x_len] = unified[b, cap_len : cap_len + x_len]
                _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", img_tokens, image_tokens_start=0)
                capture_idx += 1

            if routing_now and route_ptr < len(routes) and idx == routes[route_ptr]["end_layer_idx"]:
                unified = router.end_route(unified, tread_mask_info, original_x=saved_tokens)
                unified_freqs_cis = router.end_route(unified_freqs_cis, tread_mask_info, original_x=saved_freqs)
                unified_noise_mask_tensor = router.end_route(
                    unified_noise_mask_tensor,
                    tread_mask_info,
                    original_x=saved_noise_mask,
                )
                unified_attn_mask = saved_attn
                routing_now = False
                route_ptr += 1

            if musubi_offload_active and musubi_manager.is_managed_block(idx):
                musubi_manager.stream_out(layer)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified,
            noise_mask=unified_noise_mask_tensor,
            c_noisy=t_noisy_x,
            c_clean=t_clean_x,
        )

        x = self.unpatchify(unified, x_size, patch_size, f_patch_size, x_pos_offsets)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)
