# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""Analytical ranking of sm_100 kernel configurations.

``config.py`` decides which configurations are legal (tensor memory, shared memory,
ring deadlock freedom); this module predicts how fast each legal one runs on a
``Device`` and returns the best few. The predictions only have to rank: ``tune.py``
times the shortlist (``plan_topk``) on the card and keeps the winner.

Cost model, per CTA, in SM clock cycles (constants calibrated from in-kernel %clock
traces on B300, see docs/performance-notes.md):

* tensor cores: multiply-adds / ``mma_macs``, slowed when an MMA's shared-memory
  operand traffic exceeds ``smem_bw`` (both operands in smem at N = 64: 1.5x);
* softmax (forward) / P and dS (backward): a fixed per-tile cost plus a per-element
  cost per warpgroup, and the exponentials' MUFU throughput over all warpgroups;
* the forward's serial chain when the S ring has fewer than two slots per half (each
  half's next QK^T waits for its P V): softmax + O rescale + both halves' MMAs;
* L2 traffic of the streamed operands against ``l2_bw``;
* the backward's dQ reduce-adds (TMA, fp32) against a measured rate;
* recomputation (forward value split, backward output slices) and wave quantization.
"""

import dataclasses
import math

from simpletuner.helpers.training.kohakufa.device import Device
from simpletuner.helpers.training.kohakufa.sm100.config import (
    TILE,
    BackwardConfig,
    ForwardConfig,
    backward_configs,
    forward_configs,
)

HALF = 128

# calibration (B300 traces, D = 128): softmax step = 720 cycles of loads / max / alpha /
# P store plus ~5.9 cycles per score column per warpgroup; one O rescale of 64 columns
# ~300 cycles; backward P and dS warpgroups similar per element; a dQ reduce-add of
# 16 KB costs about 250 cycles of TMA / L2 time.
SOFTMAX_FIXED = 720.0
SOFTMAX_PER_COLUMN = 5.9
RESCALE_PER_CHUNK = 300.0
BWD_COMPUTE_FIXED = 500.0
BWD_COMPUTE_PER_COLUMN = 5.0
REDUCE_PER_16KB = 250.0
# second-order effects, as multipliers (they order otherwise equal candidates): narrow
# head-dim chunks (more MMA instructions and TMA boxes), a K / V ring under three tiles,
# single Q / output buffers, a backward ring under six chunks
NARROW_CHUNK = 0.15
SHALLOW_RING = 0.08
SINGLE_BUFFER = 0.03
# two query halves per CTA: serial chain length per S-ring depth (measured: 1 slot ~1.9x
# and 2 slots ~1.2x the 3-slot time at D = 64)
CHAIN_OVERLAP = {1: 1.6, 2: 1.04, 3: 0.84}
RING_CHUNKS = 6  # K / V chunks in flight that hide the TMA latency
L2_SHARE = 0.7  # the forward's K / V reads hit L2 lines other CTAs of the head fetched


@dataclasses.dataclass(frozen=True)
class Shape:
    """One attention call: batch x heads, query and key lengths, head dim, causal."""

    batch_heads: int
    seq_q: int
    seq_kv: int
    head_dim: int
    causal: bool = False

    def visible(self) -> float:
        """Fraction of the score matrix that is computed."""
        if not self.causal:
            return 1.0
        return min(1.0, (self.seq_q + 1) / (2 * self.seq_kv))


@dataclasses.dataclass(frozen=True)
class Estimate:
    config: object
    seconds: float
    limiter: str
    cycles_per_tile: float


def _mma_cycles(dev: Device, m: int, n: int, k: int, a_in_smem: bool) -> float:
    """One [m, n, k] tcgen05 MMA chain, derated when its smem operands outrun smem."""
    macs = m * n * k
    cycles = macs / dev.mma_macs
    per_k16 = (m * 16 * 2 if a_in_smem else 0) + n * 16 * 2
    demand = per_k16 / (m * n * 16 / dev.mma_macs)  # bytes per cycle
    return cycles * max(1.0, demand / dev.smem_bw)


def score_forward(cfg: ForwardConfig, shape: Shape, dev: Device) -> Estimate:
    d, bn, halves, nv_cols = shape.head_dim, cfg.block_n, cfg.halves, cfg.nv * cfg.dc
    rows = halves * HALF
    # per key tile, the whole CTA
    qk = halves * _mma_cycles(dev, HALF, bn, d, a_in_smem=True)
    pv = halves * _mma_cycles(dev, HALF, nv_cols, bn, a_in_smem=False)
    mma = qk + pv
    softmax_wg = SOFTMAX_FIXED + SOFTMAX_PER_COLUMN * bn
    ex2 = halves * HALF * bn / dev.ex2
    per_tile, limiter = max((mma, "mma"), (softmax_wg, "softmax"), (ex2, "mufu"))
    if halves == 2:  # each half's next S waits for its P V: the deeper the S ring, the
        # more of the chain overlaps the other half's work (fitted at D = 64 and 128)
        chain = (softmax_wg + RESCALE_PER_CHUNK * cfg.nv + mma) * CHAIN_OVERLAP[cfg.s_slots]
        if chain > per_tile:
            per_tile, limiter = chain, "chain"
    elif cfg.s_slots == 1:  # one half, one S slot: the next QK^T waits for this softmax
        chain = softmax_wg + mma
        if chain > per_tile:
            per_tile, limiter = chain, "chain"
    kv_bytes = (bn * d + bn * nv_cols) * 2
    l2 = L2_SHARE * kv_bytes / (dev.l2_bw * 1e9 / dev.sms / (dev.clock_ghz * 1e9))
    if l2 > per_tile:
        per_tile, limiter = l2, "l2"
    ring_chunks = cfg.kv_slots * cfg.kv_group  # [block_n, dc] chunks in flight
    per_tile *= _second_order(cfg.dc, ring_chunks / RING_CHUNKS, cfg.q_bufs, cfg.out_bufs)
    tiles = shape.batch_heads * math.ceil(shape.seq_q / rows) * cfg.vsplit
    key_tiles = math.ceil(shape.seq_kv / bn) * shape.visible()
    waves = math.ceil(tiles / dev.sms)
    cycles = waves * key_tiles * per_tile
    return Estimate(cfg, cycles / (dev.clock_ghz * 1e9), limiter, per_tile)


def _second_order(dc: int, ring: float, q_bufs: int, out_bufs: int) -> float:
    """Multiplier for effects the main terms ignore (``ring``: depth relative to the
    depth that hides the loads)."""
    factor = 1.0 + NARROW_CHUNK * (64 / dc - 1)
    factor *= 1.0 + SHALLOW_RING * max(0.0, 1.0 - ring) * 3
    factor *= 1.0 + SINGLE_BUFFER * ((q_bufs == 1) + (out_bufs == 1))
    return factor


def score_backward(cfg: BackwardConfig, shape: Shape, dev: Device) -> Estimate:
    d = shape.head_dim
    w = cfg.nw * cfg.dc  # output columns of one CTA
    # per query step, one CTA (128 keys x 128 queries)
    s_dp = 2 * _mma_cycles(dev, TILE, TILE, d, a_in_smem=True)
    whole = cfg.nw == cfg.nch and (w & (w - 1)) == 0  # one MMA of N = D per output
    n_out = w if whole else cfg.dc
    outs = cfg.nw if not whole else 1
    dv = outs * _mma_cycles(dev, TILE, n_out, TILE, a_in_smem=False)
    dk_dq = 2 * outs * _mma_cycles(dev, TILE, n_out, TILE, a_in_smem=True)
    mma = s_dp + dv + dk_dq
    groups = 2 if cfg.alias else 1  # ALIAS: two warpgroups each do P and dS of half
    per_column = BWD_COMPUTE_PER_COLUMN * (2 if cfg.alias else 1)
    compute = BWD_COMPUTE_FIXED + per_column * TILE / groups
    ex2 = TILE * TILE / dev.ex2
    reduce = REDUCE_PER_16KB * (TILE * w * 4) / 16384
    per_step, limiter = max((mma, "mma"), (compute, "compute"), (ex2, "mufu"), (reduce, "dq"))
    if cfg.alias:  # dP(i+1) waits for dQ(i): dS, dK / dQ and the readout are serial
        chain = compute + dk_dq + _mma_cycles(dev, TILE, TILE, d, a_in_smem=True)
        if chain > per_step:
            per_step, limiter = chain, "chain"
    if cfg.stream:
        streamed = (3 + (0 if cfg.k_res else 1)) * TILE * d + 2 * TILE * w
        l2 = streamed * 2 / (dev.l2_bw * 1e9 / dev.sms / (dev.clock_ghz * 1e9))
        if cfg.do_slots < 6:  # a shallow ring cannot hide the loads (measured)
            l2 *= 6 / cfg.do_slots
        if l2 > per_step:
            per_step, limiter = l2, "l2"
    ring = cfg.do_slots / 6 if cfg.stream else 1.0
    per_step *= _second_order(cfg.dc, ring, min(cfg.q_slots, 2), 2)
    tiles = shape.batch_heads * math.ceil(shape.seq_kv / TILE) * cfg.nsl
    steps = math.ceil(shape.seq_q / TILE) * shape.visible()
    waves = math.ceil(tiles / dev.sms)
    cycles = waves * steps * per_step
    return Estimate(cfg, cycles / (dev.clock_ghz * 1e9), limiter, per_step)


def plan_topk(kind: str, shape: Shape, dev: Device, elem_bytes: int = 2, topk: int = 4):
    """The ``topk`` best predicted legal configurations of ``kind`` ("forward" or
    "backward") for ``shape``, best first."""
    if kind == "forward":
        configs = set(forward_configs(shape.head_dim, elem_bytes))
        scored = [score_forward(c, shape, dev) for c in configs]
    else:
        configs = set(backward_configs(shape.head_dim, elem_bytes))
        scored = [score_backward(c, shape, dev) for c in configs]
    if not scored:
        raise ValueError(f"no legal {kind} configuration for head dim {shape.head_dim}")
    scored.sort(key=lambda e: e.seconds)
    return scored[:topk]
