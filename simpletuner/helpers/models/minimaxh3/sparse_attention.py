"""Experimental train-aware sparse attention for MiniMax-H3 video rows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

H3_SPARSE_ATTENTION_MODES = ("disabled", "moba3d")


def parse_h3_sparse_block_shape(
    value: str | tuple[int, int, int] | list[int],
) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
        try:
            shape = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"MiniMax-H3 sparse block shape must contain three integers, got {value!r}.") from exc
    else:
        shape = tuple(int(part) for part in value)
    if len(shape) != 3 or min(shape, default=0) <= 0:
        raise ValueError(f"MiniMax-H3 sparse block shape must contain three positive integers, got {value!r}.")
    if math.prod(shape) != 128:
        raise ValueError(
            "MiniMax-H3 FlexAttention sparse blocks currently require exactly 128 video tokens; "
            f"got {shape} ({math.prod(shape)} tokens)."
        )
    return shape


@dataclass(frozen=True)
class MiniMaxH3SparseAttentionConfig:
    mode: str = "disabled"
    block_shape: tuple[int, int, int] = (1, 8, 16)
    video_kv_fraction: float = 0.5
    share_across_heads: bool = False
    start_layer: int = 0

    def __post_init__(self) -> None:
        mode = str(self.mode or "disabled").strip().lower().replace("-", "_")
        aliases = {
            "none": "disabled",
            "full": "disabled",
            "dense": "disabled",
            "moba": "moba3d",
        }
        mode = aliases.get(mode, mode)
        if mode not in H3_SPARSE_ATTENTION_MODES:
            raise ValueError(
                f"MiniMax-H3 sparse attention mode must be one of {', '.join(H3_SPARSE_ATTENTION_MODES)}, "
                f"got {self.mode!r}."
            )
        fraction = float(self.video_kv_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("MiniMax-H3 sparse video KV fraction must be finite and in (0, 1].")
        start_layer = int(self.start_layer)
        if start_layer < 0:
            raise ValueError("MiniMax-H3 sparse start layer must be non-negative.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "block_shape", parse_h3_sparse_block_shape(self.block_shape))
        object.__setattr__(self, "video_kv_fraction", fraction)
        object.__setattr__(self, "share_across_heads", bool(self.share_across_heads))
        object.__setattr__(self, "start_layer", start_layer)

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


@dataclass(frozen=True)
class MiniMaxH3SparseAttentionLayout:
    target_start: int
    target_shape: tuple[int, int, int]
    trailing_padding: int = 0

    def validate(self, sequence_length: int) -> None:
        target_tokens = math.prod(self.target_shape)
        if (
            self.target_start < 0
            or self.trailing_padding < 0
            or self.target_start + target_tokens + self.trailing_padding != sequence_length
        ):
            raise ValueError(
                "MiniMax-H3 sparse attention requires the target-video lattice to be the final live packed segment; "
                f"got start={self.target_start}, shape={self.target_shape}, trailing_padding={self.trailing_padding}, "
                f"sequence={sequence_length}."
            )


@dataclass(frozen=True)
class _ReorderedLayout:
    prefix_tokens: int
    prefix_blocks: int
    target_shape: tuple[int, int, int]
    padded_target_shape: tuple[int, int, int]
    block_shape: tuple[int, int, int]
    target_blocks_shape: tuple[int, int, int]
    target_valid: torch.Tensor
    trailing_padding: int
    trailing_blocks: int

    @property
    def block_size(self) -> int:
        return math.prod(self.block_shape)

    @property
    def target_blocks(self) -> int:
        return math.prod(self.target_blocks_shape)

    @property
    def total_blocks(self) -> int:
        return self.prefix_blocks + self.target_blocks + self.trailing_blocks

    @property
    def sequence_length(self) -> int:
        return self.total_blocks * self.block_size


def _build_reordered_layout(
    layout: MiniMaxH3SparseAttentionLayout,
    block_shape: tuple[int, int, int],
    device: torch.device,
) -> _ReorderedLayout:
    block_size = math.prod(block_shape)
    prefix_blocks = math.ceil(layout.target_start / block_size)
    trailing_blocks = math.ceil(layout.trailing_padding / block_size)
    padded_target_shape = tuple(math.ceil(size / block) * block for size, block in zip(layout.target_shape, block_shape))
    target_blocks_shape = tuple(size // block for size, block in zip(padded_target_shape, block_shape))

    target_valid = torch.zeros(padded_target_shape, dtype=torch.bool, device=device)
    target_valid[
        : layout.target_shape[0],
        : layout.target_shape[1],
        : layout.target_shape[2],
    ] = True
    nt, nh, nw = target_blocks_shape
    bt, bh, bw = block_shape
    target_valid = target_valid.view(nt, bt, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, block_size)
    return _ReorderedLayout(
        prefix_tokens=layout.target_start,
        prefix_blocks=prefix_blocks,
        target_shape=layout.target_shape,
        padded_target_shape=padded_target_shape,
        block_shape=block_shape,
        target_blocks_shape=target_blocks_shape,
        target_valid=target_valid,
        trailing_padding=layout.trailing_padding,
        trailing_blocks=trailing_blocks,
    )


def _reorder_qkv(value: torch.Tensor, layout: _ReorderedLayout) -> torch.Tensor:
    batch, heads, _sequence, dim = value.shape
    block_size = layout.block_size
    prefix_padding = layout.prefix_blocks * block_size - layout.prefix_tokens
    prefix = F.pad(value[:, :, : layout.prefix_tokens], (0, 0, 0, prefix_padding))

    target_end = layout.prefix_tokens + math.prod(layout.target_shape)
    target = value[:, :, layout.prefix_tokens : target_end].reshape(batch, heads, *layout.target_shape, dim)
    pt = layout.padded_target_shape[0] - layout.target_shape[0]
    ph = layout.padded_target_shape[1] - layout.target_shape[1]
    pw = layout.padded_target_shape[2] - layout.target_shape[2]
    target = F.pad(target, (0, 0, 0, pw, 0, ph, 0, pt))
    nt, nh, nw = layout.target_blocks_shape
    bt, bh, bw = layout.block_shape
    target = (
        target.view(batch, heads, nt, bt, nh, bh, nw, bw, dim)
        .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
        .reshape(batch, heads, layout.target_blocks * block_size, dim)
    )
    parts = [prefix, target]
    if layout.trailing_padding:
        trailing = value[:, :, target_end:]
        trailing_padding = layout.trailing_blocks * block_size - layout.trailing_padding
        parts.append(F.pad(trailing, (0, 0, 0, trailing_padding)))
    return torch.cat(parts, dim=2)


def _restore_output(value: torch.Tensor, layout: _ReorderedLayout) -> torch.Tensor:
    batch, heads, _sequence, dim = value.shape
    block_size = layout.block_size
    prefix = value[:, :, : layout.prefix_tokens]
    target_start = layout.prefix_blocks * block_size
    target_end = target_start + layout.target_blocks * block_size
    target = value[:, :, target_start:target_end]
    nt, nh, nw = layout.target_blocks_shape
    bt, bh, bw = layout.block_shape
    target = (
        target.view(batch, heads, nt, nh, nw, bt, bh, bw, dim)
        .permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
        .reshape(batch, heads, *layout.padded_target_shape, dim)
    )
    target = target[
        :,
        :,
        : layout.target_shape[0],
        : layout.target_shape[1],
        : layout.target_shape[2],
    ].reshape(batch, heads, -1, dim)
    parts = [prefix, target]
    if layout.trailing_padding:
        trailing = value[:, :, target_end : target_end + layout.trailing_padding]
        parts.append(trailing)
    return torch.cat(parts, dim=2)


def _valid_rows(layout: _ReorderedLayout, device: torch.device) -> torch.Tensor:
    prefix = torch.arange(layout.prefix_blocks * layout.block_size, device=device) < layout.prefix_tokens
    parts = [prefix, layout.target_valid.reshape(-1)]
    if layout.trailing_blocks:
        parts.append(
            torch.zeros(
                layout.trailing_blocks * layout.block_size,
                dtype=torch.bool,
                device=device,
            )
        )
    return torch.cat(parts)


def _route_video_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    layout: _ReorderedLayout,
    config: MiniMaxH3SparseAttentionConfig,
    shared_head_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, _sequence, dim = query.shape
    block_size = layout.block_size
    target_block_start = layout.prefix_blocks
    target_block_end = target_block_start + layout.target_blocks
    target_token_start = target_block_start * block_size
    target_token_end = target_block_end * block_size
    target_query = query[:, :, target_token_start:target_token_end].reshape(
        batch, heads, layout.target_blocks, block_size, dim
    )
    target_key = key[:, :, target_token_start:target_token_end].reshape(batch, heads, layout.target_blocks, block_size, dim)
    valid = layout.target_valid.to(dtype=torch.float32)
    counts = valid.sum(-1).clamp_min(1.0).view(1, 1, layout.target_blocks, 1)

    with torch.no_grad():
        query_summary = (target_query.float() * valid.view(1, 1, layout.target_blocks, block_size, 1)).sum(-2) / counts
        key_summary = (target_key.float() * valid.view(1, 1, layout.target_blocks, block_size, 1)).sum(-2) / counts
        if config.share_across_heads:
            query_summary = query_summary.mean(1, keepdim=True)
            key_summary = key_summary.mean(1, keepdim=True)
            if shared_head_group is not None:
                import torch.distributed as dist

                world_size = dist.get_world_size(shared_head_group)
                dist.all_reduce(query_summary, group=shared_head_group)
                dist.all_reduce(key_summary, group=shared_head_group)
                query_summary.div_(world_size)
                key_summary.div_(world_size)
        scores = torch.matmul(query_summary, key_summary.transpose(-1, -2))
        selected_count = max(
            1,
            min(
                layout.target_blocks,
                math.ceil(config.video_kv_fraction * layout.target_blocks),
            ),
        )
        selected = torch.topk(scores, selected_count, dim=-1, sorted=False).indices
        if config.share_across_heads:
            selected = selected.expand(batch, heads, -1, -1)

    total_blocks = layout.total_blocks
    block_indices = torch.arange(total_blocks, device=query.device, dtype=torch.int32).view(1, 1, 1, -1)
    block_indices = block_indices.expand(batch, heads, total_blocks, total_blocks).clone()
    if selected_count < layout.target_blocks:
        prefix_indices = torch.arange(layout.prefix_blocks, device=query.device, dtype=torch.int32)
        prefix_indices = prefix_indices.view(1, 1, 1, -1).expand(batch, heads, layout.target_blocks, -1)
        selected = selected.to(torch.int32) + layout.prefix_blocks
        sparse_indices = torch.cat((prefix_indices, selected), dim=-1)
        block_indices[:, :, target_block_start:target_block_end, : sparse_indices.shape[-1]] = sparse_indices

    block_counts = torch.full(
        (batch, heads, total_blocks),
        total_blocks,
        dtype=torch.int32,
        device=query.device,
    )
    block_counts[:, :, target_block_start:target_block_end] = layout.prefix_blocks + selected_count
    return block_counts, block_indices


@torch.compiler.disable
def _flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
) -> torch.Tensor:
    from torch.nn.attention.flex_attention import flex_attention

    compiled = getattr(_flex_attention, "_compiled", None)
    if compiled is None:
        compiled = torch.compile(flex_attention, dynamic=False)
        _flex_attention._compiled = compiled
    return compiled(query, key, value, block_mask=block_mask)


def minimax_h3_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: MiniMaxH3SparseAttentionLayout,
    config: MiniMaxH3SparseAttentionConfig,
    shared_head_group=None,
) -> torch.Tensor:
    """Run H3 packed attention with only target-video-to-target-video edges sparsified."""
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("MiniMax-H3 sparse attention expects matching [batch, sequence, heads, dim] Q/K/V tensors.")
    layout.validate(query.shape[1])
    if query.device.type != "cuda":
        raise ValueError("MiniMax-H3 sparse attention currently requires CUDA FlexAttention.")

    from torch.nn.attention.flex_attention import BlockMask

    reordered_layout = _build_reordered_layout(layout, config.block_shape, query.device)
    query, key, value = (_reorder_qkv(tensor.permute(0, 2, 1, 3), reordered_layout) for tensor in (query, key, value))
    block_counts, block_indices = _route_video_blocks(
        query,
        key,
        reordered_layout,
        config,
        shared_head_group=shared_head_group,
    )
    valid_rows = _valid_rows(reordered_layout, query.device)

    def mask_mod(batch_index, head_index, query_index, key_index):
        return valid_rows[query_index] & valid_rows[key_index]

    block_mask = BlockMask.from_kv_blocks(
        block_counts,
        block_indices,
        BLOCK_SIZE=reordered_layout.block_size,
        mask_mod=mask_mod,
        seq_lengths=(
            reordered_layout.sequence_length,
            reordered_layout.sequence_length,
        ),
    )
    output = _flex_attention(query, key, value, block_mask)
    output = _restore_output(output, reordered_layout)
    return output.permute(0, 2, 1, 3)


def _ulysses_sequence_to_heads(value: torch.Tensor, process_group, world_size: int) -> torch.Tensor:
    """Exchange local sequence shards for a full sequence and local heads."""
    from torch.distributed.nn.functional import all_to_all_single

    batch, local_sequence, heads, dim = value.shape
    if heads % world_size != 0:
        raise ValueError(f"MiniMax-H3 Ulysses sparse attention requires {heads} heads to divide by CP size {world_size}.")
    local_heads = heads // world_size
    send = value.reshape(batch, local_sequence, world_size, local_heads, dim).permute(2, 1, 0, 3, 4).contiguous()
    received = all_to_all_single(torch.empty_like(send), send, group=process_group)
    return received.flatten(0, 1).permute(1, 0, 2, 3).contiguous()


def _ulysses_heads_to_sequence(value: torch.Tensor, process_group, world_size: int) -> torch.Tensor:
    """Restore full heads and the rank-local sequence shard."""
    from torch.distributed.nn.functional import all_to_all_single

    batch, sequence, local_heads, dim = value.shape
    if sequence % world_size != 0:
        raise ValueError(f"MiniMax-H3 Ulysses sparse sequence {sequence} does not divide by CP size {world_size}.")
    local_sequence = sequence // world_size
    send = value.reshape(batch, world_size, local_sequence, local_heads, dim).permute(1, 3, 0, 2, 4).contiguous()
    received = all_to_all_single(torch.empty_like(send), send, group=process_group)
    return received.flatten(0, 1).permute(1, 2, 0, 3).contiguous()


def minimax_h3_sparse_attention_ulysses(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: MiniMaxH3SparseAttentionLayout,
    config: MiniMaxH3SparseAttentionConfig,
    process_group,
) -> torch.Tensor:
    """Run H3 sparse attention over Ulysses-sharded packed rows."""
    import torch.distributed as dist

    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return minimax_h3_sparse_attention(query, key, value, layout=layout, config=config)
    query, key, value = (_ulysses_sequence_to_heads(tensor, process_group, world_size) for tensor in (query, key, value))
    output = minimax_h3_sparse_attention(
        query,
        key,
        value,
        layout=layout,
        config=config,
        shared_head_group=process_group,
    )
    return _ulysses_heads_to_sequence(output, process_group, world_size)


__all__ = [
    "H3_SPARSE_ATTENTION_MODES",
    "MiniMaxH3SparseAttentionConfig",
    "MiniMaxH3SparseAttentionLayout",
    "minimax_h3_sparse_attention",
    "minimax_h3_sparse_attention_ulysses",
    "parse_h3_sparse_block_shape",
]
