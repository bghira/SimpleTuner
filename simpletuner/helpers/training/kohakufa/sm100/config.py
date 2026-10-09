# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""Kernel configurations for sm_100 and their resource legality.

A configuration fixes the compile-time shape of a kernel: the head-dim chunk, the S ring,
the Q / K / V buffering. It is legal when it fits the SM: 512 tensor-memory columns and
the shared memory one CTA may use. ``forward_configs`` enumerates the legal ones for a
head dim (the tuner scores and times them); ``forward_config`` is the default pick.
"""

import dataclasses

TMEM_COLUMNS = 512  # per SM (fp32 columns of 128 lanes)
SMEM_BYTES = 227 * 1024  # per CTA, dynamic shared memory
SMEM_RESERVED = 6 * 1024  # mbarriers, softmax statistics, alignment
HALF = 128  # query rows per softmax half (two halves per CTA)
CHUNKS = (64, 32, 16)  # head-dim chunks: a 16-bit TMA row of 128 B at most (128 B swizzle)


@dataclasses.dataclass(frozen=True)
class ForwardConfig:
    """``dc`` x ``nch`` = head dim; ``block_n`` keys per K / V tile; ``s_slots`` S tiles in
    tensor memory (how far QK^T runs ahead of P V); ``q_bufs`` Q buffers per half;
    ``kv_slots`` K / V ring slots of ``kv_group`` [block_n, dc] head-dim chunks each
    (``kv_group`` = nch: one tile per slot); ``halves`` 128-row query halves per CTA (1:
    one softmax warpgroup, so O of a wide head fits); ``vsplit`` CTAs per query tile,
    each computing 1 / vsplit of the output's columns (QK^T recomputed by each);
    ``out_bufs`` output staging buffers."""

    dc: int
    nch: int
    block_n: int
    s_slots: int
    q_bufs: int
    kv_slots: int
    kv_group: int
    halves: int = 2
    vsplit: int = 1
    out_bufs: int = 2

    @property
    def head_dim(self) -> int:
        return self.dc * self.nch

    @property
    def nv(self) -> int:
        return self.nch // self.vsplit  # head-dim chunks of O per CTA

    def tmem_columns(self) -> int:
        return self.s_slots * self.block_n + self.halves * self.nv * self.dc  # S ring + O

    def smem_bytes(self, elem_bytes: int = 2) -> int:
        q = self.halves * self.q_bufs * HALF * self.head_dim
        kv = self.kv_slots * self.kv_group * self.block_n * self.dc
        staging = self.out_bufs * HALF * self.dc
        return (q + kv + staging) * elem_bytes + SMEM_RESERVED

    def legal(self, elem_bytes: int = 2) -> bool:
        # two halves share K_j and V_j: a K tile's chunks wait in the ring for the second
        # half's QK^T while the previous V tile waits for its second P V; one half
        # releases every chunk after its MMA, so any ring streams
        ring = self.kv_slots * self.kv_group >= (2 * self.nch if self.halves == 2 else 1)
        return (
            self.tmem_columns() <= TMEM_COLUMNS
            and self.smem_bytes(elem_bytes) <= SMEM_BYTES
            and self.s_slots >= 1
            and self.nch % self.vsplit == 0
            and self.nch % self.kv_group == 0
            and self.nv % self.kv_group == 0
            and ring
        )

    def recompute(self) -> float:
        """MMA work relative to the work proper: QK^T is done by every value part."""
        return (self.vsplit + 1) / 2


def chunk_of(head_dim: int) -> int:
    """The largest supported chunk dividing ``head_dim`` (a multiple of 16)."""
    if head_dim % 16:
        raise ValueError(f"kernel head dim {head_dim}: a multiple of 16")
    return next(c for c in CHUNKS if head_dim % c == 0)


def forward_configs(head_dim: int, elem_bytes: int = 2):
    """Every legal forward configuration for ``head_dim`` (each ring depth once: the
    deepest that fits, and a 6-tile cap when deeper fits)."""
    if head_dim % 16:
        raise ValueError(f"kernel head dim {head_dim}: a multiple of 16")
    for dc in (c for c in CHUNKS if head_dim % c == 0):
        nch = head_dim // dc
        for block_n in (128, 64):
            for vsplit in (v for v in (1, 2, 4, 8) if nch % v == 0):
                for halves in (2, 1):
                    for s_slots in (3, 2, 1):
                        for q_bufs in (2, 1):
                            for out_bufs in (2, 1):
                                for group in sorted({nch, 2, 1}, reverse=True):
                                    yield from _ring_depths(
                                        dc, nch, block_n, s_slots, q_bufs, group, halves,
                                        vsplit, out_bufs, elem_bytes,
                                    )  # fmt: skip


def _ring_depths(dc, nch, block_n, s_slots, q_bufs, group, halves, vsplit, out_bufs, elem_bytes):
    cfgs = (
        ForwardConfig(dc, nch, block_n, s_slots, q_bufs, slots, group, halves, vsplit, out_bufs)
        for slots in range(32, 0, -1)
    )
    legal = [c for c in cfgs if c.legal(elem_bytes)]
    capped = [c for c in legal if c.kv_slots * c.kv_group <= 6 * nch]
    yield from {c for c in legal[:1] + capped[:1]}


def forward_config(head_dim: int, elem_bytes: int = 2, causal: bool = False, seq: int = 0) -> ForwardConfig:
    """The forward configuration: the tuned table's entry for this card, head dim,
    masking and sequence-length bucket (``tune.py``), else the planner's best prediction
    (``plan.py``), else ``heuristic_forward``. ``seq``: the longer of the query / key
    lengths (0: unknown, the long bucket)."""
    return _choose("forward", head_dim, elem_bytes, causal, bucket(seq)) or heuristic_forward(head_dim, elem_bytes)


def heuristic_forward(head_dim: int, elem_bytes: int = 2) -> ForwardConfig:
    """Default: an S ring of at least two (QK^T ahead of P V), least recomputation (no
    value split), two halves, the widest head-dim chunk, a K / V ring of at least three
    tiles, 128-key tiles, then deeper S ring, Q double buffering, output staging,
    whole-tile ring slots and the deepest ring up to 6 tiles."""
    configs = list(forward_configs(head_dim, elem_bytes))
    if not configs:
        raise ValueError(f"head dim {head_dim}: no forward configuration fits one SM")

    def preference(c):
        ring = min(c.kv_slots * c.kv_group, 6 * c.nch)
        return (
            c.s_slots < 2, c.vsplit, -c.halves, -c.dc, ring < 3 * c.nch, -c.block_n,
            -c.s_slots, -c.q_bufs, -c.out_bufs, -c.kv_group, -ring,
        )  # fmt: skip

    return min(configs, key=preference)


TILE = 128  # backward: keys per CTA tile = query rows per step


@dataclasses.dataclass(frozen=True)
class BackwardConfig:
    """``dc`` x ``nch`` = head dim. ``alias``: P^T shares the S^T columns and dQ the dP^T
    columns (FA4's layout; needed above D = 64, at the price of dP^T(i + 1) waiting for
    dQ(i) to be read out). ``kv_bufs`` K / V tiles (2: the next tile loads during this
    one); ``q_slots`` / ``do_slots`` Q (+ statistics) and dO buffers (the loads run that
    many steps ahead).

    ``stream`` (wide heads, ALIAS layout): ``nsl`` CTAs per key tile, each computing an
    output slice of ``nw`` head-dim chunks of dK, dV, dQ (S^T and dP^T recomputed by
    each); K resident (``k_res``: all of it, else the slice's chunks), the other operands
    streamed through a ring of ``do_slots`` chunks; ``q_slots`` statistics slots."""

    dc: int
    nch: int
    alias: bool
    kv_bufs: int
    q_slots: int
    do_slots: int
    p_regs: int = 0  # registers per thread of the P / dS warpgroups (0: by layout); the
    ds_regs: int = 0  # reduce warpgroup gets 4 x 128 - p - ds - small (it holds dQ(i): D)
    small_regs: int = 32  # the one-warp mma and load partitions
    reduce_cols: int = 0  # fp32 dQ columns per TMA reduce-add (0: min(DC, 32)) ...
    reduce_stages: int = 1  # ... through a ring of this many [TILE, reduce_cols] buffers
    stream: bool = False
    nsl: int = 1
    nw: int = 0  # 0: nch (no slicing)
    k_res: bool = True

    def __post_init__(self):
        if not self.nw:
            object.__setattr__(self, "nw", self.nch)
        if not self.reduce_cols:
            object.__setattr__(self, "reduce_cols", min(self.dc, 32))
        if not self.p_regs:  # without ALIAS the dS warpgroup holds dP^T (128 fp32)
            regs = (144, 144) if self.alias else (152, 200)
            object.__setattr__(self, "p_regs", regs[0])
            object.__setattr__(self, "ds_regs", regs[1])

    @property
    def head_dim(self) -> int:
        return self.dc * self.nch

    def tmem_columns(self) -> int:
        scores = 2 * TILE  # S^T, dP^T
        acc = 1 << (self.nw * self.dc - 1).bit_length()  # TMEM blocks are powers of two
        if self.alias:
            return scores + 2 * acc  # dV, dK (P^T in S^T, dQ in dP^T)
        return scores + TILE // 2 + 3 * acc  # P^T (16-bit), dV, dK, dQ

    def smem_bytes(self, elem_bytes: int = 2) -> int:
        if self.stream:  # resident K + the chunk ring
            k_chunks = self.nch if self.k_res else self.nw
            tiles = (k_chunks + self.do_slots) * TILE * self.dc
        else:
            tiles = (2 * self.kv_bufs + self.q_slots + self.do_slots) * TILE * self.head_dim
        ds = TILE * TILE  # dS^T
        staging = TILE * self.reduce_cols * 4 * self.reduce_stages
        stats = 3 * self.q_slots * TILE * 4
        return (tiles + ds) * elem_bytes + staging + stats + SMEM_RESERVED

    def legal(self, elem_bytes: int = 2) -> bool:
        # the aliased dQ (of the slice) lives in the 128 dP^T columns
        if self.alias and self.nw * self.dc > TILE:
            return False
        if self.stream and not (self.alias and self.nsl * self.nw >= self.nch):
            return False
        # S^T(i + 1) is issued before dK(i) releases Q(i)'s slot: one Q slot deadlocks
        q_ok = self.q_slots >= 2 or self.stream
        slots_ok = (self.do_slots >= 1 if self.stream else 1 <= self.do_slots <= self.q_slots) and q_ok
        return (
            self.tmem_columns() <= TMEM_COLUMNS
            and self.smem_bytes(elem_bytes) <= SMEM_BYTES
            and slots_ok
            and self.dc % self.reduce_cols == 0
            # the staging ring doubles as the [TILE, DC] 16-bit dK / dV store buffer
            and self.reduce_stages * self.reduce_cols * 4 >= self.dc * elem_bytes
        )


def backward_configs(head_dim: int, elem_bytes: int = 2):
    """Every legal backward configuration for ``head_dim``, in default preference order:
    no aliasing, then Q at least double-buffered (the loads run ahead), double-buffered
    K / V, deeper Q, then dO buffering; above D = 128 the streamed, sliced kernel."""
    if head_dim > TILE:
        yield from _stream_configs(head_dim, elem_bytes)
        return
    dc = chunk_of(head_dim)
    buffering = ((2, 3), (1, 3), (2, 2), (1, 2), (2, 1), (1, 1))  # (kv_bufs, q_slots)
    for alias in (False, True):
        for kv_bufs, q_slots in buffering:
            for do_slots in range(q_slots, 0, -1):
                cfg = BackwardConfig(dc, head_dim // dc, alias, kv_bufs, q_slots, do_slots)
                if cfg.legal(elem_bytes):
                    yield cfg


def _stream_configs(head_dim: int, elem_bytes: int):
    """STREAM configurations (128-wide output slices, the last may be narrower; each with
    its deepest ring <= 12) in preference order: 64-element chunks, fewest slices, a ring
    of at least 6 chunks (measured: a shallower one starves the MMAs), all of K resident,
    the deeper ring."""
    configs = []
    for dc in (c for c in CHUNKS if head_dim % c == 0):
        nch = head_dim // dc
        nw = min(TILE // dc, nch)
        nsl = -(-nch // nw)
        for k_res in (True, False):
            for stats in (3, 2):
                for ring in range(12, 1, -1):
                    cfg = BackwardConfig(dc, nch, True, 1, stats, ring, stream=True, nsl=nsl, nw=nw, k_res=k_res)
                    if cfg.legal(elem_bytes):
                        configs.append(cfg)
                        break
    yield from sorted(
        configs,
        key=lambda c: (-c.dc, c.nsl, c.do_slots < 6, not c.k_res, -c.do_slots, -c.q_slots),
    )


def backward_config(head_dim: int, elem_bytes: int = 2, causal: bool = False, seq: int = 0) -> BackwardConfig:
    """The backward configuration: tuned table, else planner, else ``heuristic_backward``."""
    return _choose("backward", head_dim, elem_bytes, causal, bucket(seq)) or heuristic_backward(head_dim, elem_bytes)


def heuristic_backward(head_dim: int, elem_bytes: int = 2) -> BackwardConfig:
    """The first legal configuration in ``backward_configs`` order."""
    for cfg in backward_configs(head_dim, elem_bytes):
        return cfg
    raise ValueError(f"head dim {head_dim}: no backward configuration fits one SM")


_CHOSEN: dict[tuple, object] = {}
BUCKETS = (("s512", 512), ("s2048", 2048))  # tuned length buckets (beyond: "long")


def bucket(seq: int) -> str:
    """The tuned length bucket of a call whose longer sequence has ``seq`` tokens."""
    for name, limit in BUCKETS:
        if 0 < seq <= limit:
            return name
    return "long"


def _choose(kind: str, head_dim: int, elem_bytes: int, causal: bool, length: str):
    """Tuned table entry for the current card, else the planner's best prediction
    (memoized; None without a CUDA device)."""
    k = (kind, head_dim, elem_bytes, causal, length)
    if k not in _CHOSEN:
        _CHOSEN[k] = _tuned_or_planned(kind, head_dim, elem_bytes, causal, length)
    return _CHOSEN[k]


def _tuned_or_planned(kind, head_dim, elem_bytes, causal, length):
    import torch

    if not torch.cuda.is_available():
        return None
    from simpletuner.helpers.training.kohakufa.device import Device
    from simpletuner.helpers.training.kohakufa.sm100 import plan, tune

    dev = Device.query()
    found = tune.lookup(dev.arch, kind, head_dim, elem_bytes, causal, length)
    if found is not None:
        return found
    seq = {"s512": 256, "s2048": 1024}.get(length, 8192)
    shape = plan.Shape(32 * 8192 // seq, seq, seq, head_dim, causal)
    return plan.plan_topk(kind, shape, dev, elem_bytes, topk=1)[0].config
