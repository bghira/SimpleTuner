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

import math
from functools import lru_cache

import numpy as np
import torch
from einops import rearrange


def tile(x, canvas_thw, tile_thw, sp_size=1):
    r"""Rearrange tensor into tiles for block-based attention.

    Args:
        x: Input tensor with shape (b, head, s, d) where s = t * h * w
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
        sp_size: Spatial size parameter, defaults to 1

    Returns:
        Rearranged tensor organized by tiles
    """
    b, h, s, d = x.shape
    t, h, w = canvas_thw
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} == s:{s}"

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t / tile_t_dim)
    n_h = int(h / tile_h_dim)
    n_w = int(w / tile_w_dim)
    x = rearrange(x, "b head (sp t h w) d -> b head (t sp h w) d", sp=sp_size, t=t // sp_size, h=h, w=w)
    return rearrange(
        x,
        "b h (n_t ts_t n_h ts_h n_w ts_w) d -> b h (n_t n_h n_w ts_t ts_h ts_w) d",
        n_t=n_t,
        n_h=n_h,
        n_w=n_w,
        ts_t=tile_t_dim,
        ts_h=tile_h_dim,
        ts_w=tile_w_dim,
    )


def untile(x, canvas_thw, tile_thw, sp_size=1):
    r"""Reverse the tiling operation to restore original tensor layout.

    Args:
        x: Tiled tensor
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
        sp_size: Spatial size parameter, defaults to 1

    Returns:
        Restored tensor with original layout
    """
    t, h, w = canvas_thw

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t / tile_t_dim)
    n_h = int(h / tile_h_dim)
    n_w = int(w / tile_w_dim)

    x = rearrange(
        x,
        "b h (n_t n_h n_w ts_t ts_h ts_w) d -> b h (n_t ts_t n_h ts_h n_w ts_w) d",
        n_t=n_t,
        n_h=n_h,
        n_w=n_w,
        ts_t=tile_t_dim,
        ts_h=tile_h_dim,
        ts_w=tile_w_dim,
    )
    return rearrange(x, "b head (t sp h w) d -> b head (sp t h w) d", sp=sp_size, t=t // sp_size, h=h, w=w)


def get_tile_t_h_w(tile_id, tile_thw_dim):
    """Extract temporal, height, and width indices from a flattened tile ID."""
    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw_dim
    tile_t = tile_id // (tile_h_dim * tile_w_dim)
    tile_h = (tile_id % (tile_h_dim * tile_w_dim)) // tile_w_dim
    tile_w = tile_id % tile_w_dim

    return tile_t, tile_h, tile_w


def importance_sampling(q, k, topk, threshold=0.0, lambda_=0.9, adaptive_pool=None):
    r"""Select top-k blocks based on importance scores considering both similarity and redundancy.

    Args:
        q: Query tensor with shape (B, H, S, D)
        k: Key tensor with shape (B, H, K, D)
        topk: Number of top blocks to select
        threshold: Threshold parameter (not implemented)
        lambda_: Weight factor balancing similarity and redundancy
        adaptive_pool: Adaptive pooling parameter (unused)

    Returns:
        top_block_indices: Indices of selected blocks with shape (B, H, S, topk)
    """
    if threshold > 0.0:
        raise NotImplementedError("importance_sampling with threshold not implemented")

    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    gate_similarity = torch.einsum("bhsd,bhkd->bhsk", q, k)
    gate_similarity = (gate_similarity + 1.0) / 2.0
    gate_unique = torch.einsum("bhsd,bhkd->bhsk", k, k)
    gate_unique = (gate_unique + 1.0) / 2.0

    B, H, K_num, D = k.shape
    diag_indices = torch.arange(K_num, device=k.device)
    gate_unique[:, :, diag_indices, diag_indices] = torch.nan

    mean_redundancy = torch.nanmean(gate_unique, dim=-2, keepdim=True)

    importance_scores = lambda_ * gate_similarity - (1 - lambda_) * mean_redundancy

    topk = min(topk, importance_scores.size(-1))
    _, top_block_indices = importance_scores.topk(k=topk, dim=-1, sorted=False)
    return top_block_indices


def similarity_sampling(q, k, topk, threshold=0.0, block_num=None, adaptive_pool=None, temperature=0.01):
    r"""Select top-k blocks based on similarity scores between query and key averages.

    Args:
        q: Query tensor with shape (B, H, S, D)
        k: Key tensor with shape (B, H, K, D)
        topk: Number of top blocks to select
        threshold: Cumulative score threshold for dynamic topk selection
        block_num: Total number of blocks (unused)
        adaptive_pool: Adaptive pooling parameter (unused)
        temperature: Temperature scaling for softmax

    Returns:
        top_block_indices: Indices of selected blocks with shape (B, H, S, topk)
    """
    if threshold > 0.0:
        gate = torch.einsum("bhsd,bhkd->bhsk", q, k)
        gate = gate / temperature
        gate_ = torch.softmax(gate, dim=-1)
        sorted_gate, sorted_indices = torch.sort(gate_, dim=-1, descending=True)
        cum_scores = torch.cumsum(sorted_gate, dim=-1)
        above_threshold = cum_scores >= threshold
        has_any_above = above_threshold.any(dim=-1, keepdim=True)
        # Find the first position exceeding threshold, add 1 to get the number of blocks to select
        dynamic_topk = above_threshold.int().argmax(dim=-1, keepdim=True) + 1
        dynamic_topk = torch.where(has_any_above, dynamic_topk, torch.full_like(dynamic_topk, topk))
        # Generate fixed-shape indices and pad with top1 indices for parts not meeting threshold
        # Limit minimum k value
        dynamic_topk = torch.clamp(dynamic_topk, min=8, max=topk)
        indices = torch.arange(gate.size(-1), device=gate.device).expand(gate.size())
        mask = (indices < dynamic_topk).int()
        # Use top1 to pad to topk to ensure same size
        top_block_indices = torch.gather(sorted_indices, -1, indices) * mask + (1 - mask) * sorted_indices[..., 0:1]
    else:
        gate = torch.einsum("bhsd,bhkd->bhsk", q, k)
        topk = min(topk, gate.size(-1))
        _, top_block_indices = gate.topk(k=topk, dim=-1, sorted=False)
    return top_block_indices


def create_moba_3d_mask(
    q,
    k,
    canvas_thw,
    topk,
    tile_thw,
    kernel_thw,
    text_block_num=0,
    add_text_mask=False,
    threshold=0.0,
    lambda_=None,
    mask_share_within_head=True,
    q_block_avg_pool=True,
    adaptive_pool=None,
    sampling_type=None,
):
    r"""Create MOBA (Mixture of Block Attention) 3D mask for sparse attention.

    Args:
        q: Query tensor
        k: Key tensor
        canvas_thw: Canvas dimensions (t, h, w)
        topk: Number of top blocks to attend to
        tile_thw: Tile dimensions
        kernel_thw: Kernel dimensions
        text_block_num: Number of text blocks
        add_text_mask: Whether to add text mask
        threshold: Threshold for similarity sampling
        lambda_: Weight factor for importance sampling
        mask_share_within_head: Whether to share mask across heads
        q_block_avg_pool: Whether to apply average pooling to query blocks
        adaptive_pool: Adaptive pooling size
        sampling_type: Type of sampling ("similarity" or "importance")

    Returns:
        moba_3d_mask: 3D attention mask with shape (num_heads, block_num, block_num)
    """
    seq_len = q.size(2)
    block_size = np.prod(tile_thw)
    block_num = int(seq_len / block_size)

    image_shape = canvas_thw
    block_shape = tile_thw
    batch_size, num_heads, seq_len, head_dim = k.shape
    num_blocks_t = np.ceil(image_shape[0] / block_shape[0]).astype(int)
    num_blocks_h = np.ceil(image_shape[1] / block_shape[1]).astype(int)
    num_blocks_w = np.ceil(image_shape[2] / block_shape[2]).astype(int)

    def get_block_avg_feat(x: torch.Tensor, adaptive_pool=None, pooling_type="avg") -> torch.Tensor:
        x_block_means = x.view(
            batch_size,
            num_heads,
            num_blocks_t,
            num_blocks_h,
            num_blocks_w,
            block_shape[0],
            block_shape[1],
            block_shape[2],
            head_dim,
        )
        if adaptive_pool is not None:
            x_block_means = x_block_means.view(-1, block_shape[0], block_shape[1], block_shape[2], head_dim)
            x_block_means = x_block_means.permute(0, 4, 1, 2, 3)
            if pooling_type == "avg":
                global_avg = torch.nn.functional.adaptive_avg_pool3d(x_block_means, adaptive_pool)
                global_avg = global_avg.permute(0, 2, 3, 4, 1)
                global_avg = global_avg.reshape(
                    batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2]
                )
                x_block_means = global_avg
            elif pooling_type == "max":
                max_pool = torch.nn.functional.adaptive_max_pool3d(x_block_means, adaptive_pool)
                max_pool = max_pool.permute(0, 2, 3, 4, 1)
                max_pool = max_pool.reshape(
                    batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2]
                )
                x_block_means = max_pool
            elif pooling_type == "mix":
                global_avg = torch.nn.functional.adaptive_avg_pool3d(x_block_means, (1, 1, 1))
                global_avg = global_avg / (global_avg.norm(dim=1, keepdim=True) + 1e-8)
                global_avg = global_avg.permute(0, 2, 3, 4, 1)
                global_avg = global_avg.reshape(batch_size, num_heads, -1, head_dim)

                max_pool = torch.nn.functional.adaptive_max_pool3d(x_block_means, adaptive_pool)
                max_pool = max_pool / (max_pool.norm(dim=1, keepdim=True) + 1e-8)
                max_pool = max_pool.permute(0, 2, 3, 4, 1)
                max_pool = max_pool.reshape(
                    batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2]
                )
                x_block_means = torch.cat([max_pool, global_avg], dim=-1)
            else:
                raise ValueError(f"pooling_type={pooling_type} is not Supported")
        else:
            x_block_means = x_block_means.mean(dim=(-2, -3, -4)).view(batch_size, num_heads, -1, head_dim)
        return x_block_means

    if (sampling_type == "similarity" and threshold > 0.0) and adaptive_pool is None:
        adaptive_pool = (2, 2, 2)

    k_block_means = get_block_avg_feat(k, adaptive_pool)

    if q_block_avg_pool:
        q = get_block_avg_feat(q, adaptive_pool)
    q = q.type(torch.float32)
    k_block_means = k_block_means.type(torch.float32)  # float logit for better gate logit perception

    if mask_share_within_head:
        q = q.mean(dim=1, keepdim=True)
        k_block_means = k_block_means.mean(dim=1, keepdim=True)

    if sampling_type == "similarity":
        top_block_indices = similarity_sampling(
            q, k_block_means, topk, threshold, block_num=block_num, adaptive_pool=adaptive_pool
        )
    elif sampling_type == "importance":
        top_block_indices = importance_sampling(
            q, k_block_means, topk, threshold, lambda_=lambda_, adaptive_pool=adaptive_pool
        )
    else:
        raise NotImplementedError(f"sampling_type={sampling_type} is not Supported")

    q = q.type_as(k)
    # Keep head dimension to dynamically build mask
    assert top_block_indices.size(0) == 1, "top_block_indices batch size must be 1"
    top_block_indices = top_block_indices.squeeze(0)  # Remove batch dimension [h, s_block, topk]

    # Create 3D mask template (heads, block_num, block_num)
    gate_idx_mask = torch.zeros((top_block_indices.size(0), block_num, block_num), dtype=torch.bool, device=q.device)

    # Iterate over each attention head using dim=0
    for head_idx in range(top_block_indices.size(0)):
        # Build index matrix for each head
        head_mask = torch.zeros((block_num, block_num), dtype=torch.bool, device=q.device)
        # Fill indices for current head
        head_mask.scatter_(dim=-1, index=top_block_indices[head_idx], value=True)
        gate_idx_mask[head_idx] = head_mask

    if text_block_num > 0:
        pad_block_num = block_num + text_block_num
        moba_3d_mask = torch.full(
            (gate_idx_mask.size(0), pad_block_num, pad_block_num), False, dtype=torch.bool, device=gate_idx_mask.device
        )
        moba_3d_mask[:, :block_num, :block_num] = gate_idx_mask
        if add_text_mask:
            moba_3d_mask[:, :, -text_block_num:] = True
            moba_3d_mask[:, -text_block_num:, :] = True
    else:
        moba_3d_mask = gate_idx_mask

    return moba_3d_mask


@lru_cache(maxsize=4096)
def create_sta_3d_mask_optimize(canvas_thw, tile_thw, kernel_thw):
    r"""Create optimized STA (Spatio-Temporal Attention) 3D mask using vectorized operations.

    Args:
        canvas_thw: String representation of canvas dimensions "t_h_w"
        tile_thw: String representation of tile dimensions "t_h_w"
        kernel_thw: String representation of kernel dimensions "t_h_w"

    Returns:
        block_mask: Boolean mask tensor with shape (block_num, block_num)
    """
    canvas_thw = tuple(map(int, canvas_thw.split("_")))
    seq_len = np.prod(canvas_thw)
    tile_thw = tuple(map(int, tile_thw.split("_")))
    kernel_thw = tuple(map(int, kernel_thw.split("_")))

    kernel_t, kernel_h, kernel_w = kernel_thw

    block_size = np.prod(tile_thw)
    block_num = int(seq_len / block_size)

    block_mask = np.full((block_num, block_num), False, dtype=bool)
    tile_thw_num = (canvas_thw[0] // tile_thw[0], canvas_thw[1] // tile_thw[1], canvas_thw[2] // tile_thw[2])

    i_indices = np.arange(block_num)
    j_indices = np.arange(block_num)

    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="ij")

    q_t_tile = i_grid // (tile_thw_num[1] * tile_thw_num[2])
    q_h_tile = (i_grid % (tile_thw_num[1] * tile_thw_num[2])) // tile_thw_num[2]
    q_w_tile = i_grid % tile_thw_num[2]

    kv_t_tile = j_grid // (tile_thw_num[1] * tile_thw_num[2])
    kv_h_tile = (j_grid % (tile_thw_num[1] * tile_thw_num[2])) // tile_thw_num[2]
    kv_w_tile = j_grid % tile_thw_num[2]

    kernel_center_t = np.clip(q_t_tile, kernel_t // 2, (tile_thw_num[0] - 1) - kernel_t // 2)
    kernel_center_h = np.clip(q_h_tile, kernel_h // 2, (tile_thw_num[1] - 1) - kernel_h // 2)
    kernel_center_w = np.clip(q_w_tile, kernel_w // 2, (tile_thw_num[2] - 1) - kernel_w // 2)

    time_mask = np.abs(kernel_center_t - kv_t_tile) <= kernel_t // 2
    hori_mask = np.abs(kernel_center_h - kv_h_tile) <= kernel_h // 2
    vert_mask = np.abs(kernel_center_w - kv_w_tile) <= kernel_w // 2

    block_mask = time_mask & hori_mask & vert_mask

    block_mask = torch.tensor(block_mask, dtype=torch.bool)
    return block_mask


@torch.no_grad()
def create_sta_3d_mask(canvas_thw, tile_thw, kernel_thw, text_block_num=0):
    r"""Create STA (Spatio-Temporal Attention) 3D mask.

    Args:
        canvas_thw: Canvas dimensions (t, h, w)
        tile_thw: Tile dimensions
        kernel_thw: Kernel dimensions
        text_block_num: Number of text blocks to pad

    Returns:
        sta_mask: Boolean mask tensor with optional text block padding
    """
    block_mask = create_sta_3d_mask_optimize(
        "_".join([str(x) for x in canvas_thw]), "_".join([str(x) for x in tile_thw]), "_".join([str(x) for x in kernel_thw])
    )
    sta_mask = None
    block_num = block_mask.size(0)
    if text_block_num > 0:
        pad_block_num = block_num + text_block_num
        sta_mask = torch.full(
            (pad_block_num, pad_block_num),
            False,
            dtype=torch.bool,
        )
        sta_mask[:block_num, :block_num] = block_mask
        sta_mask[:, -text_block_num:] = True
        sta_mask[-text_block_num:, :] = True
    else:
        sta_mask = block_mask
    return sta_mask


@torch.no_grad()
def create_ssta_3d_mask(
    q,
    k,
    canvas_thw,
    topk,
    tile_thw,
    kernel_thw,
    text_block_num=0,
    threshold=0.0,
    lambda_=None,
    text_mask=None,
    mask_share_within_head=True,
    adaptive_pool=None,
    sampling_type=None,
):
    r"""Create SSTA (Sparse Spatio-Temporal Attention) 3D mask combining STA and MOBA masks.

    Args:
        q: Query tensor
        k: Key tensor
        canvas_thw: Canvas dimensions (t, h, w)
        topk: Number of top blocks to attend to
        tile_thw: Tile dimensions
        kernel_thw: Kernel dimensions
        text_block_num: Number of text blocks
        threshold: Threshold for similarity sampling
        lambda_: Weight factor for importance sampling
        text_mask: Optional text mask tensor
        mask_share_within_head: Whether to share mask across heads
        adaptive_pool: Adaptive pooling size
        sampling_type: Type of sampling ("similarity" or "importance")

    Returns:
        ssta_3d_mask: Combined 3D attention mask
    """
    sta_3d_mask = create_sta_3d_mask(canvas_thw, tile_thw, kernel_thw, text_block_num)
    sta_3d_mask = sta_3d_mask.to(q.device)

    moba_3d_mask = create_moba_3d_mask(
        q,
        k,
        canvas_thw,
        topk,
        tile_thw,
        kernel_thw,
        text_block_num,
        threshold=threshold,
        lambda_=lambda_,
        mask_share_within_head=mask_share_within_head,
        adaptive_pool=adaptive_pool,
        sampling_type=sampling_type,
    )
    ssta_3d_mask = torch.logical_or(sta_3d_mask.unsqueeze(0), moba_3d_mask)
    assert len(ssta_3d_mask.size()) == 3, "ssta_3d_mask should be 3D"

    # set text block mask
    if text_mask is not None:
        block_size = np.prod(tile_thw)
        seq_len = q.size(2)
        block_num = int(seq_len / block_size)
        text_mask_index = torch.ceil(text_mask.sum() / block_size).long()
        text_mask_index = text_mask_index.clamp(min=1).item()
        assert ssta_3d_mask.shape[-1] == ssta_3d_mask.shape[-2], "ssta_3d_mask should be square"

        pad_start_index = block_num + text_mask_index
        ssta_3d_mask[:, pad_start_index:, :] = False
        ssta_3d_mask[:, :, pad_start_index:] = False
        eye_mask = torch.eye(
            ssta_3d_mask.shape[1] - pad_start_index, dtype=torch.bool, device=ssta_3d_mask.device
        ).unsqueeze(0)
        ssta_3d_mask[:, pad_start_index:, pad_start_index:] = ssta_3d_mask[:, pad_start_index:, pad_start_index:] | eye_mask
    return ssta_3d_mask


def ssta_3d_attention(
    all_q,
    all_k,
    all_v,
    canvas_thw,
    topk=1,
    tile_thw=(6, 8, 8),
    kernel_thw=(1, 1, 1),
    text_len=0,
    sparse_type="ssta",
    threshold=0.0,
    lambda_=None,
    pad_type="zero",
    text_mask=None,
    mask_share_within_head=True,
    sampling_type=None,
    adaptive_pool=None,
):
    r"""Sparse Spatio-Temporal Attention (SSTA) 3D attention mechanism.

    Args:
        all_q: Query tensor with shape (B, H, S, D)
        all_k: Key tensor with shape (B, H, S, D)
        all_v: Value tensor with shape (B, H, S, D)
        canvas_thw: Canvas dimensions (t, h, w)
        topk: Number of top blocks to attend to
        tile_thw: Tile dimensions
        kernel_thw: Kernel dimensions
        text_len: Length of text sequence
        sparse_type: Type of sparse attention ('sta', 'block_attn', or 'ssta')
        threshold: Threshold for similarity sampling
        lambda_: Weight factor for importance sampling
        pad_type: Padding type ("zero" or "repeat")
        text_mask: Optional text mask tensor
        mask_share_within_head: Whether to share mask across heads
        sampling_type: Type of sampling ("similarity" or "importance")
        adaptive_pool: Adaptive pooling size

    Returns:
        tuple: (output tensor, sparse_ratio)
            - output: Attention output with shape (B, H, S, D)
            - sparse_ratio: Ratio of non-zero attention weights
    """
    try:
        from flex_block_attn import flex_block_attn_func
    except ImportError as e:
        raise Exception(
            "Could not load flex-block-attn. Please install the flex-block-attn package. You can install it via 'pip install flex-block-attn'."
        ) from e
    assert pad_type in ["zero", "repeat"]
    assert sampling_type in ["similarity", "importance"]
    assert (lambda_ is not None and sampling_type == "importance") or sampling_type == "similarity"

    if text_len > 0:
        image_q = all_q[:, :, :-text_len, :]
        image_k = all_k[:, :, :-text_len, :]
        image_v = all_v[:, :, :-text_len, :]

        text_q = all_q[:, :, -text_len:, :]
        text_k = all_k[:, :, -text_len:, :]
        text_v = all_v[:, :, -text_len:, :]
    else:
        image_q = all_q
        image_k = all_k
        image_v = all_v

    b, hd, s, d = image_q.shape
    t, h, w = canvas_thw
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} != s:{s}"
    tile_t, tile_h, tile_w = tile_thw
    block_size = np.prod(tile_thw)

    need_pad = False
    if t % tile_t != 0 or h % tile_h != 0 or w % tile_w != 0:
        need_pad = True
        pad_image_q = image_q.reshape(b, hd, t, h, w, d)
        pad_image_k = image_k.reshape(b, hd, t, h, w, d)
        pad_image_v = image_v.reshape(b, hd, t, h, w, d)

        pad_t = 0 if t % tile_t == 0 else tile_t - t % tile_t
        if pad_t > 0:
            t = t + pad_t
            repeat_q = pad_image_q[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            repeat_k = pad_image_k[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            repeat_v = pad_image_v[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=2)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=2)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=2)

        pad_h = 0 if h % tile_h == 0 else tile_h - h % tile_h
        if pad_h > 0:
            h = h + pad_h
            repeat_q = pad_image_q[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            repeat_k = pad_image_k[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            repeat_v = pad_image_v[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=3)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=3)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=3)

        pad_w = 0 if w % tile_w == 0 else tile_w - w % tile_w
        if pad_w > 0:
            w = w + pad_w
            repeat_q = pad_image_q[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            repeat_k = pad_image_k[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            repeat_v = pad_image_v[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=4)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=4)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=4)

        image_q = pad_image_q.reshape(b, hd, -1, d)
        image_k = pad_image_k.reshape(b, hd, -1, d)
        image_v = pad_image_v.reshape(b, hd, -1, d)

        canvas_thw = (t, h, w)

    need_pad_text = False
    text_block_num = math.ceil(text_len / block_size)
    text_target_size = text_block_num * block_size
    if text_len % block_size > 0:
        need_pad_text = True
        text_pad_size = text_target_size - text_len

        pad_text_q = text_q[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        pad_text_k = text_k[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        pad_text_v = text_v[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)

        text_q = torch.cat([text_q, pad_text_q], dim=2)
        text_k = torch.cat([text_k, pad_text_k], dim=2)
        text_v = torch.cat([text_v, pad_text_v], dim=2)

    image_q = tile(image_q, canvas_thw, tile_thw)
    image_k = tile(image_k, canvas_thw, tile_thw)
    image_v = tile(image_v, canvas_thw, tile_thw)

    if text_len > 0:
        q = torch.cat([image_q, text_q], dim=2)
        k = torch.cat([image_k, text_k], dim=2)
        v = torch.cat([image_v, text_v], dim=2)
    else:
        q = image_q
        k = image_k
        v = image_v

    if sparse_type == "sta":
        assert text_mask is None, "text_mask do not support in sta sparse_type"
        block_mask = create_sta_3d_mask(canvas_thw, tile_thw, kernel_thw, text_block_num).to(q.device)
        o = flex_block_attn_func(q, k, v, block_size, block_size, block_mask)
    elif sparse_type == "block_attn":
        image_q_list = torch.split(image_q, 1, dim=0)
        image_k_list = torch.split(image_k, 1, dim=0)

        mask_list = []
        for i in range(b):
            block_mask = create_moba_3d_mask(
                image_q_list[i],
                image_k_list[i],
                canvas_thw=canvas_thw,
                topk=topk,
                tile_thw=tile_thw,
                kernel_thw=kernel_thw,
                text_block_num=text_block_num,
                add_text_mask=True,
                lambda_=lambda_,
                threshold=threshold,
                mask_share_within_head=mask_share_within_head,
                adaptive_pool=adaptive_pool,
                sampling_type=sampling_type,
            )
            mask_list.append(block_mask)
        block_mask = torch.stack(mask_list, dim=0)
        o = flex_block_attn_func(q, k, v, block_size, block_size, block_mask)

    elif sparse_type == "ssta":
        image_q_list = torch.split(image_q, 1, dim=0)
        image_k_list = torch.split(image_k, 1, dim=0)

        mask_list = []
        for i in range(b):
            block_mask = create_ssta_3d_mask(
                image_q_list[i],
                image_k_list[i],
                canvas_thw=canvas_thw,
                tile_thw=tile_thw,
                kernel_thw=kernel_thw,
                text_block_num=text_block_num,
                topk=topk,
                threshold=threshold,
                lambda_=lambda_,
                text_mask=text_mask[i] if text_mask is not None else None,
                mask_share_within_head=mask_share_within_head,
                adaptive_pool=adaptive_pool,
                sampling_type=sampling_type,
            )
            mask_list.append(block_mask)
        block_mask = torch.stack(mask_list, dim=0)
        if mask_share_within_head:
            block_mask = block_mask.unsqueeze(1)  # [b, 1, s_block, s_block]
        o = flex_block_attn_func(q, k, v, block_size, block_size, block_mask)
    else:
        raise Exception(f"unsupported sparse_type:{sparse_type}")
    sparse_ratio = block_mask.float().mean().cpu().item()

    if text_len > 0:
        image_o = o[:, :, :-text_target_size, :]
        if need_pad_text:
            text_o = o[:, :, -text_target_size:-text_pad_size, :]
        else:
            text_o = o[:, :, -text_target_size:, :]
    else:
        image_o = o

    image_o = untile(image_o, canvas_thw, tile_thw)

    if need_pad:
        # Remove padding from output
        unpad_image_o = image_o.reshape(b, hd, t, h, w, d)
        if pad_t > 0:
            unpad_image_o = unpad_image_o[:, :, :-pad_t, :, :, :]
        if pad_h > 0:
            unpad_image_o = unpad_image_o[:, :, :, :-pad_h, :, :]
        if pad_w > 0:
            unpad_image_o = unpad_image_o[:, :, :, :, :-pad_w, :]
        image_o = unpad_image_o.reshape(b, hd, -1, d)

    if text_len > 0:
        o = torch.cat([image_o, text_o], dim=2)
    else:
        o = image_o

    return o, sparse_ratio
