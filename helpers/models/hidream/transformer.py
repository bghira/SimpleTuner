from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import einops
from einops import repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def batched_load_balancing_loss():
    aux_losses_arr = get_load_balancing_loss()
    alpha = aux_losses_arr[0][-1]
    Pi = torch.stack([ent[1] for ent in aux_losses_arr], dim=0)
    fi = torch.stack([ent[2] for ent in aux_losses_arr], dim=0)

    fi_list = all_gather(fi)
    fi = torch.stack(fi_list, 0).mean(0)

    aux_loss = (Pi * fi).sum(-1).mean() * alpha
    return aux_loss


import torch
from torch import nn
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import maybe_allow_in_graph
from typing import Optional
from typing import List
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, out_channels, bias=True
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, latent):
        latent = self.proj(latent)
        return latent


class PooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(
            in_channels=text_emb_dim, time_embed_dim=hidden_size
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pooled_embed):
        return self.pooled_embedder(pooled_embed)


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=hidden_size
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, timesteps, wdtype):
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb


class OutEmbed(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adaln_input):
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


try:
    from flash_attn_interface import flash_attn_func

    USE_FLASH_ATTN3 = True
except:
    from flash_attn import flash_attn_func

    USE_FLASH_ATTN3 = False


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if USE_FLASH_ATTN3:
        hidden_states = flash_attn_func(
            query, key, value, causal=False, deterministic=False
        )[0]
    else:
        hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
    hidden_states = hidden_states.flatten(-2)
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states


@maybe_allow_in_graph
class Attention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        linear_cls = nn.Linear
        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim)
        self.to_k = linear_cls(self.inner_dim, self.inner_dim)
        self.to_v = linear_cls(self.inner_dim, self.inner_dim)
        self.to_out = linear_cls(self.inner_dim, self.out_dim)
        self.q_rms_norm = nn.RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = nn.RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = linear_cls(query_dim, self.inner_dim)
            self.to_k_t = linear_cls(self.inner_dim, self.inner_dim)
            self.to_v_t = linear_cls(self.inner_dim, self.inner_dim)
            self.to_out_t = linear_cls(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        norm_image_tokens: torch.FloatTensor,
        image_tokens_masks: torch.FloatTensor = None,
        norm_text_tokens: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            image_tokens=norm_image_tokens,
            image_tokens_masks=image_tokens_masks,
            text_tokens=norm_text_tokens,
            rope=rope,
        )


class HiDreamAttnProcessor_flashattn:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = attention(query, key, value)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(
                hidden_states, [num_image_tokens, num_text_tokens], dim=1
            )
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_routed_experts=4,
        num_activated_experts=2,
        aux_loss_alpha=0.01,
    ):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        
        # Compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Expert-level computation auxiliary loss with gradient checkpointing
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # Sequence-level auxiliary loss with gradient checkpointing
                def create_seq_aux_loss_fn(scores_view, idx, device):
                    def compute_seq_aux_loss():
                        ce = torch.zeros(bsz, self.n_routed_experts, device=device)
                        ce.scatter_add_(
                            1,
                            idx,
                            torch.ones(bsz, seq_len * aux_topk, device=device),
                        ).div_(seq_len * aux_topk / self.n_routed_experts)
                        return ce, scores_view
                    return compute_seq_aux_loss
                
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                ce, scores_view = torch.utils.checkpoint.checkpoint(
                    create_seq_aux_loss_fn(scores_for_seq_aux, topk_idx_for_aux_loss, hidden_states.device),
                    **ckpt_kwargs
                )
                
                aux_loss = (ce * scores_view.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token-level auxiliary loss with gradient checkpointing
                def create_token_aux_loss_fn(scores_mean, idx, num_classes):
                    def compute_token_aux_loss():
                        mask_ce = F.one_hot(idx.view(-1), num_classes=num_classes)
                        ce = mask_ce.float().mean(0)
                        return ce, scores_mean
                    return compute_token_aux_loss
                
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                scores_mean = scores_for_aux.mean(0)
                
                ce, scores_mean = torch.utils.checkpoint.checkpoint(
                    create_token_aux_loss_fn(scores_mean, topk_idx_for_aux_loss, self.n_routed_experts),
                    **ckpt_kwargs
                )
                
                fi = ce * self.n_routed_experts
                aux_loss = (scores_mean * fi).sum() * self.alpha
                
                # Store for later use but detach to prevent memory leakage
                with torch.no_grad():
                    save_load_balancing_loss((aux_loss.detach(), scores_mean.detach(), fi.detach(), self.alpha))
        else:
            aux_loss = None
            
        return topk_idx, topk_weight, aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
    ):
        super().__init__()
        self.shared_experts = FeedForward(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim) for i in range(num_routed_experts)]
        )
        self.gate = MoEGate(
            embed_dim=dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype)
            # y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=False
        )

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


class BlockType:
    TransformerBlock = 1
    SingleTransformerBlock = 2


@maybe_allow_in_graph
class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor_flashattn(),
            single=True,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
            )
        else:
            self.ff_i = FeedForward(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = (
            self.adaLN_modulation(adaln_input)[:, None].chunk(6, dim=-1)
        )

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            rope=rope,
        )
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(dtype=wtype))
        image_tokens = ff_output_i + image_tokens
        return image_tokens


@maybe_allow_in_graph
class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 12 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm1_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor_flashattn(),
            single=False,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
            )
        else:
            self.ff_i = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.norm3_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_t = FeedForward(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(adaln_input)[:, None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope=rope,
        )

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens)
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens)
        image_tokens = ff_output_i + image_tokens
        text_tokens = ff_output_t + text_tokens
        return image_tokens, text_tokens


@maybe_allow_in_graph
class HiDreamImageBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        block_type: BlockType = BlockType.TransformerBlock,
    ):
        super().__init__()
        block_classes = {
            BlockType.TransformerBlock: HiDreamImageTransformerBlock,
            BlockType.SingleTransformerBlock: HiDreamImageSingleTransformerBlock,
        }
        self.block = block_classes[block_type](
            dim,
            num_attention_heads,
            attention_head_dim,
            num_routed_experts,
            num_activated_experts,
        )

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        return self.block(
            image_tokens,
            image_tokens_masks,
            text_tokens,
            adaln_input,
            rope,
        )


class HiDreamImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        self.llama_layers = llama_layers

        self.t_embedder = TimestepEmbed(self.inner_dim)
        self.p_embedder = PooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.inner_dim,
        )
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    num_routed_experts=num_routed_experts,
                    num_activated_experts=num_activated_experts,
                    block_type=BlockType.TransformerBlock,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    num_routed_experts=num_routed_experts,
                    num_activated_experts=num_activated_experts,
                    block_type=BlockType.SingleTransformerBlock,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.final_layer = OutEmbed(self.inner_dim, patch_size, self.out_channels)

        # Create projection layers for both T5 and Llama embeddings
        caption_channels = [caption_channels[1]] * (num_layers + num_single_layers) + [
            caption_channels[0]
        ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(
                TextProjection(in_features=caption_channel, hidden_size=self.inner_dim)
            )
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = (
            max_resolution[0] * max_resolution[1] // (patch_size * patch_size)
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Recursively enables or disables gradient checkpointing for all modules.

        Args:
            module: Module to set gradient checkpointing for
            value: Whether to enable (True) or disable (False) gradient checkpointing
        """
        if isinstance(
            module,
            (
                HiDreamImageBlock,
                Attention,
                FeedForward,
                MOEFeedForward,
            ),
        ):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value

        # Also set checkpointing for child modules that might not be directly accessible
        for child in module.children():
            self._set_gradient_checkpointing(child, value)

    def enable_gradient_checkpointing(self):
        """Enables gradient checkpointing for the model"""
        self.gradient_checkpointing = True
        self._set_gradient_checkpointing(self, True)

    def disable_gradient_checkpointing(self):
        """Disables gradient checkpointing for the model"""
        self.gradient_checkpointing = False
        self._set_gradient_checkpointing(self, False)

    def expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        return timesteps

    def unpatchify(
        self, x: torch.Tensor, img_sizes: List[Tuple[int, int]], is_training: bool
    ) -> List[torch.Tensor]:
        if 0 and is_training:
            x = einops.rearrange(
                x,
                "B S (p1 p2 C) -> B C S (p1 p2)",
                p1=self.config.patch_size,
                p2=self.config.patch_size,
            )
        else:
            x_arr = []
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                x_arr.append(
                    einops.rearrange(
                        x[i, : pH * pW].reshape(1, pH, pW, -1),
                        "B H W (p1 p2 C) -> B C (H p1) (W p2)",
                        p1=self.config.patch_size,
                        p2=self.config.patch_size,
                    )
                )
            x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, x, max_seq, img_sizes=None):
        pz2 = self.config.patch_size * self.config.patch_size
        if isinstance(x, torch.Tensor):
            B, C = x.shape[0], x.shape[1]
            device = x.device
            dtype = x.dtype
        else:
            B, C = len(x), x[0].shape[0]
            device = x[0].device
            dtype = x[0].dtype
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes):
                x_masks[i, 0 : img_size[0] * img_size[1]] = 1
            x = einops.rearrange(x, "B C S p -> B S (p C)", p=pz2)
        elif isinstance(x, torch.Tensor):
            pH, pW = (
                x.shape[-2] // self.config.patch_size,
                x.shape[-1] // self.config.patch_size,
            )
            x = einops.rearrange(
                x,
                "B C (H p1) (W p2) -> B (H W) (p1 p2 C)",
                p1=self.config.patch_size,
                p2=self.config.patch_size,
            )
            img_sizes = [[pH, pW]] * B
            x_masks = None
        else:
            raise NotImplementedError
        return x, x_masks, img_sizes

    def _extract_llama_layers(self, llama_hidden_states):
        """
        Extract specific layers from the provided Llama hidden states based on self.llama_layers.

        Args:
            llama_hidden_states: Tensor containing Llama hidden states

        Returns:
            List of extracted layer tensors
        """
        llama_shape = llama_hidden_states.shape
        extracted_layers = []

        # Process based on tensor shape
        if len(llama_shape) == 5:  # [batch, num_layers, 1, seq, dim]
            # Remove singleton dimension if present
            llama_hidden_states = llama_hidden_states.squeeze(2)
            for layer_idx in self.llama_layers:
                # Handle index being out of bounds by using modulo
                safe_idx = layer_idx % llama_shape[1]
                layer_emb = llama_hidden_states[:, safe_idx]
                extracted_layers.append(layer_emb)
        elif len(llama_shape) == 4:  # [num_layers, batch, seq, dim]
            for layer_idx in self.llama_layers:
                # Handle index being out of bounds by using modulo
                safe_idx = layer_idx % llama_shape[0]
                layer_emb = llama_hidden_states[safe_idx]
                extracted_layers.append(layer_emb)
        else:
            # Unsupported format, try to use as is but log warning
            logger.warning(f"Unexpected llama_hidden_states shape: {llama_shape}")
            # Handle as best we can
            if not isinstance(llama_hidden_states, list):
                extracted_layers = [llama_hidden_states]
            else:
                extracted_layers = llama_hidden_states

        return extracted_layers

    def _process_embeddings(
        self, t5_hidden_states, extracted_llama_states, batch_size, hidden_dim
    ):
        """
        Process T5 and Llama embeddings through projection layers.

        Args:
            t5_hidden_states: T5 encoder hidden states
            extracted_llama_states: List of extracted Llama states
            batch_size: Batch size
            hidden_dim: Hidden dimension for reshaping

        Returns:
            Tuple of (processed_t5_embeddings, processed_llama_embeddings)
        """
        # Apply gradient checkpointing to embedding processing if enabled
        if self.training and self.gradient_checkpointing:

            def create_custom_forward_t5(t5_states):
                def custom_forward(proj):
                    processed = proj(t5_states)
                    return processed.view(batch_size, -1, hidden_dim)

                return custom_forward

            def create_custom_forward_llama(llama_state, i):
                def custom_forward(proj):
                    processed = proj(llama_state)
                    return processed.view(batch_size, -1, hidden_dim)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            # Process T5 embeddings with checkpointing
            processed_t5_embeddings = torch.utils.checkpoint.checkpoint(
                create_custom_forward_t5(t5_hidden_states),
                self.caption_projection[-1],
                **ckpt_kwargs,
            )

            # Process Llama embeddings with checkpointing
            processed_llama_embeddings = []
            for i, llama_emb in enumerate(extracted_llama_states):
                if i < len(self.caption_projection) - 1:  # Reserve last one for T5
                    processed_emb = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_llama(llama_emb, i),
                        self.caption_projection[i],
                        **ckpt_kwargs,
                    )
                    processed_llama_embeddings.append(processed_emb)
        else:
            # Standard processing without checkpointing
            processed_t5_embeddings = self.caption_projection[-1](t5_hidden_states)
            processed_t5_embeddings = processed_t5_embeddings.view(
                batch_size, -1, hidden_dim
            )

            processed_llama_embeddings = []
            for i, llama_emb in enumerate(extracted_llama_states):
                if i < len(self.caption_projection) - 1:  # Reserve last one for T5
                    processed_emb = self.caption_projection[i](llama_emb)
                    processed_emb = processed_emb.view(batch_size, -1, hidden_dim)
                    processed_llama_embeddings.append(processed_emb)

        return processed_t5_embeddings, processed_llama_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        t5_hidden_states: torch.Tensor = None,
        llama_hidden_states: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        img_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass for the HiDreamImageTransformer2DModel.

        Args:
            hidden_states: Input latents
            timesteps: Current timestep
            t5_hidden_states: T5 encoder hidden states with shape [batch_size, seq_len, dim]
            llama_hidden_states: Llama hidden states with shape [batch_size, num_layers, 1, seq_len, dim] or
                                [num_layers, batch_size, seq_len, dim]
            pooled_embeds: Pooled embeddings from CLIP encoders
            img_sizes: List of image dimensions
            img_ids: Image positional IDs
            joint_attention_kwargs: Additional attention parameters
            return_dict: Whether to return as a dict

        Returns:
            Output sample and mask
        """
        # Handle LoRA scale if provided
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # Weight the lora layers by setting lora_scale for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing scale via joint_attention_kwargs when not using the PEFT backend is ineffective."
                )

        # Get batch size and data type
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 1. Process timesteps and pooled embeddings
        # Apply gradient checkpointing to this step if enabled
        if self.training and self.gradient_checkpointing:

            def create_custom_forward_timestep(timesteps):
                def custom_forward(t_embedder):
                    return t_embedder(timesteps, hidden_states_type)

                return custom_forward

            def create_custom_forward_pooled(pooled):
                def custom_forward(p_embedder):
                    return p_embedder(pooled)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            timesteps = self.expand_timesteps(
                timesteps, batch_size, hidden_states.device
            )
            print(f"t_embedder device: {self.t_embedder.device}")
            timesteps_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward_timestep(timesteps),
                self.t_embedder,
                **ckpt_kwargs,
            )

            pooled_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward_pooled(pooled_embeds),
                self.p_embedder,
                **ckpt_kwargs,
            )
            adaln_input = timesteps_emb + pooled_emb
        else:
            # Standard processing without checkpointing
            timesteps = self.expand_timesteps(
                timesteps, batch_size, hidden_states.device
            )
            print(f"t_embedder device: {self.t_embedder.device}")
            timesteps = self.t_embedder(timesteps, hidden_states_type)
            p_embedder = self.p_embedder(pooled_embeds)
            adaln_input = timesteps + p_embedder

        # 2. Process input hidden states (no checkpointing for patchify)
        hidden_states, image_tokens_masks, img_sizes = self.patchify(
            hidden_states, self.max_seq, img_sizes
        )

        # Apply checkpointing only to the embedding step, not patchify
        if self.training and self.gradient_checkpointing:

            def create_custom_forward_embed(patched_states):
                def custom_forward(embedder):
                    return embedder(patched_states)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward_embed(hidden_states),
                self.x_embedder,
                **ckpt_kwargs,
            )
        else:
            hidden_states = self.x_embedder(hidden_states)

        # Create image positional IDs if not provided
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = (
                img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            )
            img_ids[..., 2] = (
                img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            )
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

        # 3. Extract and process text embeddings
        extracted_llama_states = self._extract_llama_layers(llama_hidden_states)

        # Process the text embeddings with optional checkpointing
        if self.training and self.gradient_checkpointing:

            def create_custom_forward_t5(t5_states):
                def custom_forward(proj):
                    processed = proj(t5_states)
                    return processed.view(batch_size, -1, hidden_states.shape[-1])

                return custom_forward

            def create_custom_forward_llama(llama_state, i):
                def custom_forward(proj):
                    processed = proj(llama_state)
                    return processed.view(batch_size, -1, hidden_states.shape[-1])

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            # Process T5 embeddings with checkpointing
            processed_t5_embeddings = torch.utils.checkpoint.checkpoint(
                create_custom_forward_t5(t5_hidden_states),
                self.caption_projection[-1],
                **ckpt_kwargs,
            )

            # Process Llama embeddings with checkpointing
            processed_llama_embeddings = []
            for i, llama_emb in enumerate(extracted_llama_states):
                if i < len(self.caption_projection) - 1:  # Reserve last one for T5
                    processed_emb = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_llama(llama_emb, i),
                        self.caption_projection[i],
                        **ckpt_kwargs,
                    )
                    processed_llama_embeddings.append(processed_emb)
        else:
            # Standard processing without checkpointing
            processed_t5_embeddings = self.caption_projection[-1](t5_hidden_states)
            processed_t5_embeddings = processed_t5_embeddings.view(
                batch_size, -1, hidden_states.shape[-1]
            )

            processed_llama_embeddings = []
            for i, llama_emb in enumerate(extracted_llama_states):
                if i < len(self.caption_projection) - 1:  # Reserve last one for T5
                    processed_emb = self.caption_projection[i](llama_emb)
                    processed_emb = processed_emb.view(
                        batch_size, -1, hidden_states.shape[-1]
                    )
                    processed_llama_embeddings.append(processed_emb)

        # Ensure we have enough processed embeddings for all blocks
        total_blocks = self.config.num_layers + self.config.num_single_layers
        while len(processed_llama_embeddings) < total_blocks:
            # Cycle through existing embeddings if we don't have enough
            for i in range(len(processed_llama_embeddings)):
                if len(processed_llama_embeddings) < total_blocks:
                    processed_llama_embeddings.append(processed_llama_embeddings[i])
                else:
                    break

        # 4. Create positional encoding with optional checkpointing
        if self.training and self.gradient_checkpointing:

            def create_custom_forward_rope(ids):
                def custom_forward(pe_embedder):
                    return pe_embedder(ids)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            txt_ids = torch.zeros(
                batch_size,
                processed_t5_embeddings.shape[1]
                + processed_llama_embeddings[0].shape[1] * 2,
                3,
                device=img_ids.device,
                dtype=img_ids.dtype,
            )
            ids = torch.cat((img_ids, txt_ids), dim=1)

            rope = torch.utils.checkpoint.checkpoint(
                create_custom_forward_rope(ids), self.pe_embedder, **ckpt_kwargs
            )
        else:
            txt_ids = torch.zeros(
                batch_size,
                processed_t5_embeddings.shape[1]
                + processed_llama_embeddings[0].shape[1] * 2,
                3,
                device=img_ids.device,
                dtype=img_ids.dtype,
            )
            ids = torch.cat((img_ids, txt_ids), dim=1)
            rope = self.pe_embedder(ids)

        # 5. Process through transformer blocks
        block_id = 0

        # Prepare initial combined embeddings for first set of blocks
        initial_encoder_hidden_states = torch.cat(
            [
                processed_t5_embeddings,
                processed_llama_embeddings[-1 % len(processed_llama_embeddings)],
            ],
            dim=1,
        )
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]

        # Process through double stream blocks
        for bid, block in enumerate(self.double_stream_blocks):
            # Get the current Llama embedding for this block with safe indexing
            safe_idx = block_id % len(processed_llama_embeddings)
            cur_llama_embedding = processed_llama_embeddings[safe_idx]

            # Combine embeddings for this block
            cur_encoder_hidden_states = torch.cat(
                [initial_encoder_hidden_states, cur_llama_embedding],
                dim=1,
            )

            # Process through the block with optional gradient checkpointing
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, initial_encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        image_tokens_masks,
                        cur_encoder_hidden_states,
                        adaln_input,
                        rope,
                        **ckpt_kwargs,
                    )
                )
            else:
                hidden_states, initial_encoder_hidden_states = block(
                    image_tokens=hidden_states,
                    image_tokens_masks=image_tokens_masks,
                    text_tokens=cur_encoder_hidden_states,
                    adaln_input=adaln_input,
                    rope=rope,
                )

            # Keep consistent encoder states length
            initial_encoder_hidden_states = initial_encoder_hidden_states[
                :, :initial_encoder_hidden_states_seq_len
            ]
            block_id += 1

        # 6. Prepare for single stream blocks
        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]

        # Update attention masks for combined hidden states
        if image_tokens_masks is not None:
            attention_mask_ones = torch.ones(
                (
                    batch_size,
                    initial_encoder_hidden_states.shape[1]
                    + cur_llama_embedding.shape[1],
                ),
                device=image_tokens_masks.device,
                dtype=image_tokens_masks.dtype,
            )
            image_tokens_masks = torch.cat(
                [image_tokens_masks, attention_mask_ones], dim=1
            )

        # 7. Process through single stream blocks
        for bid, block in enumerate(self.single_stream_blocks):
            # Get the current Llama embedding for this block with safe indexing
            safe_idx = block_id % len(processed_llama_embeddings)
            cur_llama_embedding = processed_llama_embeddings[safe_idx]

            # Concatenate for processing
            hidden_states = torch.cat([hidden_states, cur_llama_embedding], dim=1)

            # Process through the block with optional gradient checkpointing
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    image_tokens_masks,
                    None,
                    adaln_input,
                    rope,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    image_tokens=hidden_states,
                    image_tokens_masks=image_tokens_masks,
                    text_tokens=None,
                    adaln_input=adaln_input,
                    rope=rope,
                )

            # Maintain consistent hidden state length
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        # 8. Final processing with optional checkpointing
        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]

        if self.training and self.gradient_checkpointing:

            def create_custom_forward_final(hidden_states):
                def custom_forward(final_layer):
                    return final_layer(hidden_states, adaln_input)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward_final(hidden_states),
                self.final_layer,
                **ckpt_kwargs,
            )

            # Don't checkpoint the unpatchify operation
            output = self.unpatchify(output, img_sizes, self.training)
        else:
            output = self.final_layer(hidden_states, adaln_input)
            output = self.unpatchify(output, img_sizes, self.training)

        # Update attention mask if needed
        if image_tokens_masks is not None:
            image_tokens_masks = image_tokens_masks[:, :image_tokens_seq_len]

        # 9. Unscale LoRA if needed
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        # 10. Return results
        if not return_dict:
            return (output, image_tokens_masks)
        return Transformer2DModelOutput(sample=output, mask=image_tokens_masks)
