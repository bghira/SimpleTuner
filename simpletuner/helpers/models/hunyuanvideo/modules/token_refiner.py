# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .activation_layers import get_activation_layer
from .attention import attention
from .embed_layers import TextProjection, TimestepEmbedder
from .mlp_layers import MLP
from .modulate_layers import apply_gate
from .norm_layers import get_norm_layer


class IndividualTokenRefinerBlock(nn.Module):
    """
    A single block for token refinement with self-attention and MLP.

    Args:
        hidden_size: Hidden dimension size.
        heads_num: Number of attention heads.
        mlp_width_ratio: Expansion ratio for MLP hidden size.
        mlp_drop_rate: Dropout rate for MLP.
        act_type: Activation function type.
        qk_norm: Whether to use QK normalization.
        qk_norm_type: Type of QK normalization.
        qkv_bias: Whether to use bias in QKV projections.
        dtype: Optional torch dtype.
        device: Optional torch device.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for IndividualTokenRefinerBlock.

        Args:
            x: Input tensor of shape [B, L, C].
            c: Conditioning tensor of shape [B, C].
            attn_mask: Optional attention mask of shape [B, L].

        Returns:
            Refined tensor of shape [B, L, C].
        """
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        attn = attention(q, k, v, attn_mask=attn_mask)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class IndividualTokenRefiner(nn.Module):
    """
    Stacks multiple IndividualTokenRefinerBlock modules.

    Args:
        hidden_size: Hidden dimension size.
        heads_num: Number of attention heads.
        depth: Number of blocks.
        mlp_width_ratio: Expansion ratio for MLP hidden size.
        mlp_drop_rate: Dropout rate for MLP.
        act_type: Activation function type.
        qk_norm: Whether to use QK normalization.
        qk_norm_type: Type of QK normalization.
        qkv_bias: Whether to use bias in QKV projections.
        dtype: Optional torch dtype.
        device: Optional torch device.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for IndividualTokenRefiner.

        Args:
            x: Input tensor of shape [B, L, C].
            c: Conditioning tensor of shape [B, C].
            mask: Optional mask tensor of shape [B, L].

        Returns:
            Refined tensor of shape [B, L, C].
        """
        if mask is not None:
            mask = mask.clone().bool()
            mask[:, 0] = True  # Prevent attention weights from becoming NaN
        for block in self.blocks:
            x = block(x, c, mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    Single token refiner block for LLM text embedding refinement.

    Args:
        in_channels: Input feature dimension.
        hidden_size: Hidden dimension size.
        heads_num: Number of attention heads.
        depth: Number of blocks.
        mlp_width_ratio: Expansion ratio for MLP hidden size.
        mlp_drop_rate: Dropout rate for MLP.
        act_type: Activation function type.
        qk_norm: Whether to use QK normalization.
        qk_norm_type: Type of QK normalization.
        qkv_bias: Whether to use bias in QKV projections.
        dtype: Optional torch dtype.
        device: Optional torch device.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer, **factory_kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for SingleTokenRefiner.

        Args:
            x: Input tensor of shape [B, L, in_channels].
            t: Timestep tensor of shape [B].
            mask: Optional mask tensor of shape [B, L].

        Returns:
            Refined tensor of shape [B, L, hidden_size].
        """
        timestep_aware_representations = self.t_embedder(t)
        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [B, L, 1]
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask)
        return x
