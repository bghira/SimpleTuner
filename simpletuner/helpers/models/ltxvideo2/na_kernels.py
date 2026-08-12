# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NA_SCORE_BUDGET = 2**25
NA_KV_STACK_BUDGET = 2**28


def _storage_bounds(x: torch.Tensor) -> tuple[int, int]:
    lo = hi = x.storage_offset()
    for size, stride in zip(x.shape, x.stride(), strict=True):
        extent = (size - 1) * stride
        lo += min(0, extent)
        hi += max(0, extent)
    element_size = x.element_size()
    return lo * element_size, (hi + 1) * element_size - 1


def _interleaved_disjoint(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.shape != y.shape or x.stride() != y.stride():
        return False
    strides = x.stride()
    if not strides or strides[-1] != 1:
        return False
    extent = 1
    i = len(strides)
    while i > 0 and strides[i - 1] == extent:
        extent *= x.shape[i - 1]
        i -= 1
    outer = [s for size, s in zip(x.shape[:i], strides[:i], strict=True) if size > 1]
    if not outer:
        return False
    period = min(outer)
    if period <= 0 or any(s % period != 0 for s in outer):
        return False
    delta = abs(x.storage_offset() - y.storage_offset())
    return delta >= extent and delta + extent <= period


def _tensors_overlap(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.numel() == 0 or y.numel() == 0 or x.device != y.device:
        return False
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    x0, x1 = _storage_bounds(x)
    y0, y1 = _storage_bounds(y)
    if max(x0, y0) > min(x1, y1):
        return False
    return not _interleaved_disjoint(x, y)


def _check_rope_inplace(*xs: torch.Tensor, readonly: tuple[torch.Tensor, ...] = ()) -> None:
    for x in (*xs, *readonly):
        if x.requires_grad:
            raise RuntimeError("in-place RoPE operations are inference-only and do not support autograd")

    for x in xs:
        required = 1
        dimensions = sorted((abs(stride), size) for size, stride in zip(x.shape, x.stride(), strict=True) if size > 1)
        for stride, size in dimensions:
            if stride < required:
                raise ValueError("in-place RoPE requires views without internal overlap")
            required = stride * size

    if len(xs) == 2 and _tensors_overlap(xs[0], xs[1]):
        raise ValueError("paired in-place RoPE requires non-overlapping input storage")
    for x in xs:
        for source in readonly:
            if _tensors_overlap(x, source):
                raise ValueError("in-place RoPE inputs must not overlap frequencies or scales")


def _trim_rope_freqs(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2 and freqs_cis.ndim > 2 and x.shape[2] != 1 and freqs_cis.shape[2] > x.shape[2]:
        return freqs_cis[:, :, : x.shape[2]]
    return freqs_cis


def _apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    freqs_cis = _trim_rope_freqs(x, freqs_cis)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def _rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    x_norm = F.rms_norm(
        x,
        (x.shape[-1],),
        weight=scale,
        eps=epsilon,
    )
    return _apply_rope1(x_norm, freqs_cis)


def rms_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    _check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    q.copy_(_rms_rope1(q, freqs_cis, q_scale, epsilon))
    k.copy_(_rms_rope1(k, freqs_cis, k_scale, epsilon))
    return q, k


def _window_bounds(length: int, kernel: int, causal: bool) -> tuple[list[int], list[int]]:
    starts = []
    ends = []
    if causal:
        for i in range(length):
            starts.append(max(0, i - kernel + 1))
            ends.append(i + 1)
    else:
        kernel = min(kernel, length)
        lo = length - kernel
        half = kernel // 2
        for i in range(length):
            s = min(max(i - half, 0), lo)
            starts.append(s)
            ends.append(s + kernel)
    return starts, ends


def _pick_tiles(dims: tuple[int, int, int], kernels: list[int]) -> list[int]:
    tiles = list(dims)

    def cost(ts: list[int]) -> int:
        nq = math.prod(ts)
        nk = math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims, strict=True))
        return nq * nk

    while cost(tiles) > NA_SCORE_BUDGET and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _group_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bools = []
    for starts, ends in rel_bounds:
        st = torch.tensor(starts, device=device)
        en = torch.tensor(ends, device=device)
        kj = torch.arange(int(en.max()), device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < en[:, None]))
    visible = (
        bools[0][:, None, None, :, None, None]
        & bools[1][None, :, None, None, :, None]
        & bools[2][None, None, :, None, None, :]
    )
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int | list[int] | tuple[int, int, int],
    is_causal: bool | list[bool] | tuple[bool, bool, bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3
    else:
        kernel_size = list(kernel_size)
    if is_causal is None:
        causal = [False, False, False]
    elif isinstance(is_causal, bool):
        causal = [is_causal] * 3
    else:
        causal = list(is_causal)
    if len(kernel_size) != 3:
        raise ValueError(f"na3d kernel_size must have 3 elements, got {len(kernel_size)}")
    if len(causal) != 3:
        raise ValueError(f"na3d is_causal must have 3 elements, got {len(causal)}")

    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    kernels = [k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, dims, strict=True)]
    if scale is None:
        scale = hd**-0.5
    device = q.device
    if scale != 1.0:
        q = q * scale

    bounds = [_window_bounds(d, k_, c) for d, k_, c in zip(dims, kernels, causal, strict=True)]
    tile_t, tile_h, tile_w = _pick_tiles(dims, [min(k_, d) for k_, d in zip(kernels, dims, strict=True)])

    groups = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (tuple(s - rt0 for s in bounds[0][0][t0:t1]), tuple(e - rt0 for e in bounds[0][1][t0:t1]))
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (tuple(s - rh0 for s in bounds[1][0][h0:h1]), tuple(e - rh0 for e in bounds[1][1][h0:h1]))
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(s - rw0 for s in bounds[2][0][w0:w1]),
                    tuple(e - rw0 for e in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                    )
                )

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        if device.type == "cuda":
            g_max = max(1, NA_KV_STACK_BUDGET // max(1, batch * nh * nk * hd * 2))
        else:
            g_max = 1
        qs0, _ = tiles[0]
        tq, th, tw = (qs0[0].stop - qs0[0].start, qs0[1].stop - qs0[1].start, qs0[2].stop - qs0[2].start)
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out
