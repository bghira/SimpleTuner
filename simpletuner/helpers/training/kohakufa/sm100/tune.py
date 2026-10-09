# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""Time the planner's shortlist on the card and keep the winner.

``plan.py`` ranks every legal configuration; only a shortlist is ever run: the best
predicted configuration of each structural family (halves x value split x key tile x
head-dim chunk, or output slices x streaming for the backward), so a family the model
misjudges is still timed. Candidates are timed interleaved (every candidate once per
round, best of several rounds): a GPU's clock drifts as it heats, and timing one
candidate after the other would credit the first ones.

The result is a table keyed by (architecture, kind, head dim, operand bytes, causal,
length bucket: config.BUCKETS), saved as JSON. The short-sequence buckets are tuned on
lists of real call-site shapes, by their summed time. ``config.forward_config`` / ``backward_config`` read it (the shipped
tables are in ``tuned/``); shapes it does not cover use the planner's best prediction.
Tuning runs outside CUDA-graph capture and is never triggered implicitly.
"""

import dataclasses
import json
import os

import torch

from simpletuner.helpers.training.kohakufa.device import Device
from simpletuner.helpers.training.kohakufa.sm100 import plan as _plan
from simpletuner.helpers.training.kohakufa.sm100.config import BackwardConfig, ForwardConfig, bucket

_CACHE: dict[tuple, object] = {}
_TUNED: set = set()
EXHAUSTIVE = 24  # legal spaces up to this size are timed in full  # keys tuned in this process (``save`` writes only these)
_KINDS = {"forward": ForwardConfig, "backward": BackwardConfig}
TABLES = os.path.join(os.path.dirname(__file__), "tuned")


def key(arch: str, kind: str, head_dim: int, elem_bytes: int, causal: bool, length: str) -> tuple:
    return (arch, kind, head_dim, elem_bytes, bool(causal), length)


def lookup(arch: str, kind: str, head_dim: int, elem_bytes: int, causal: bool, length: str):
    """The tuned configuration, or None. Fallbacks: the other length buckets (nearest
    first), then the non-causal entry."""
    _load_shipped()
    order = ["s512", "s2048", "long"]
    nearest = sorted(order, key=lambda b: abs(order.index(b) - order.index(length)))
    for c in (causal, False) if causal else (causal,):
        for b in nearest:
            found = _CACHE.get(key(arch, kind, head_dim, elem_bytes, c, b))
            if found is not None:
                return found
    return None


def shortlist(kind: str, shape: "_plan.Shape", dev: Device, overall: int = 8, families: int = 8):
    """The ``overall`` best predicted configurations, plus the best predicted one of each
    of the ``families`` best structural families (so a family the model misjudges is
    still timed); a small legal space is returned whole."""
    ranked = _plan.plan_topk(kind, shape, dev, topk=10**6)
    if len(ranked) <= EXHAUSTIVE:  # a small space (the backward's): time all of it
        return [estimate.config for estimate in ranked]
    picked = [estimate.config for estimate in ranked[:overall]]
    seen = set()
    for estimate in ranked:
        family = _family(estimate.config)
        if family not in seen:
            seen.add(family)
            if estimate.config not in picked:
                picked.append(estimate.config)
            if len(seen) == families:
                break
    return picked


def _family(cfg) -> tuple:
    if isinstance(cfg, ForwardConfig):
        return (cfg.halves, cfg.vsplit, cfg.block_n, cfg.dc)
    return (cfg.alias, cfg.stream, cfg.nsl, cfg.dc, cfg.k_res)


def tune(kind: str, head_dim: int, elem_bytes: int = 2, causal: bool = False,
         seq: int = 8192, batch_heads: int = 32, dev: Device | None = None,
         rounds: int = 5, verbose: bool = False, shapes=None):  # fmt: skip
    """Time the shortlist for ``kind`` at ``head_dim`` and record the fastest
    configuration; returns it. The problem: [batch_heads, seq] (token causal if
    ``causal``), or ``shapes``, a list of (batch, heads, seq_q, seq_kv, block) whose
    summed time decides (``causal`` then means the shapes are masked)."""
    from simpletuner.helpers.training.kohakufa.sm100 import bwd, fwd  # the kernels (imported lazily: no import cycle)

    dev = dev or Device.query()
    if shapes is None:
        shapes = [(1, batch_heads, seq, seq, int(causal))]
    first = shapes[0]
    plan_shape = _plan.Shape(first[0] * first[1], first[2], first[3], head_dim, causal)
    candidates = shortlist(kind, plan_shape, dev)
    from simpletuner.helpers.training.kohakufa.sm100.config import heuristic_backward, heuristic_forward

    default = (heuristic_forward if kind == "forward" else heuristic_backward)(head_dim, elem_bytes)
    if default not in candidates:  # a tuned entry is never worse than the default
        candidates.append(default)
    problems = [_problem(kind, s, head_dim, elem_bytes) for s in shapes]
    runs = []
    for cfg in candidates:
        calls = [_call(kind, cfg, fwd, bwd, problem) for problem in problems]

        def run(calls=calls):
            for call in calls:
                call()

        try:
            run()
            torch.cuda.synchronize()
            runs.append((cfg, run))
        except Exception as error:  # noqa: BLE001 -- a candidate that fails to build is skipped
            if verbose:
                print(f"skip {cfg}: {str(error).splitlines()[-1][:120]}")
    if not runs:
        raise RuntimeError(f"no {kind} candidate for head dim {head_dim} ran")
    best = dict.fromkeys(range(len(runs)), float("inf"))
    for _ in range(rounds):  # interleaved: every candidate once per round
        for index, (_, run) in enumerate(runs):
            best[index] = min(best[index], _time(run))
    winner = runs[min(best, key=best.get)][0]
    if verbose:
        for index, (cfg, _) in enumerate(runs):
            mark = "*" if cfg == winner else " "
            print(f"{mark} {best[index]:8.3f} ms  {cfg}")
    length = bucket(max(max(s[2], s[3]) for s in shapes))
    k = key(dev.arch, kind, head_dim, elem_bytes, causal, length)
    _CACHE[k] = winner
    _TUNED.add(k)
    return winner


def _problem(kind, shape, head_dim, elem_bytes):
    """Inputs of one (batch, heads, seq_q, seq_kv, block) problem."""
    from simpletuner.helpers.training.kohakufa.sm100 import fwd

    batch, heads, seq_q, seq_kv, block = shape
    dtype = torch.float16 if elem_bytes == 2 else torch.float32
    gen = torch.Generator(device="cuda").manual_seed(0)

    def make(seq):
        x = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype, generator=gen)
        return x.transpose(1, 2)

    q, k, v = make(seq_q), make(seq_kv), make(seq_kv)
    scale = head_dim**-0.5
    if kind == "forward":
        return (q, k, v, scale, block)
    out, lse = fwd.attention_forward(q, k, v, scale, block)
    return (q, k, v, out, torch.randn_like(out), lse, scale, block)


def _call(kind, cfg, fwd, bwd, problem):
    if kind == "forward":
        return _bound(fwd.attention_forward, fwd, "forward_config", cfg, *problem)
    return _bound(bwd.attention_backward, bwd, "backward_config", cfg, *problem)


def _bound(fn, module, attr, cfg, *args):
    """``fn(*args)`` with ``module.attr`` (the config function) pinned to ``cfg``."""

    def run():
        saved = getattr(module, attr)
        setattr(module, attr, lambda *a, **kw: cfg)
        try:
            return fn(*args)
        finally:
            setattr(module, attr, saved)

    return run


def _time(fn, iters: int = 10) -> float:
    """Mean ms of ``iters`` calls replayed from one CUDA graph (no launch overhead: at
    the 10-50 us of short-sequence calls it would swamp the differences), eager if the
    call cannot be captured."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
        replay = graph.replay
    except Exception:  # noqa: BLE001 -- eager fallback
        torch.cuda.synchronize()

        def replay():
            for _ in range(iters):
                fn()

    replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def save(path: str) -> None:
    """Write the entries tuned in this process (not the shipped ones it loaded)."""
    rows = [{"key": list(k), "config": dataclasses.asdict(_CACHE[k])} for k in sorted(_TUNED)]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)


def load(path: str, override: bool = True) -> None:
    """Read a table; ``override=False`` keeps entries already present (the shipped
    tables must never replace what this process just tuned)."""
    with open(path) as fh:
        for row in json.load(fh):
            k = tuple(row["key"])
            if override or k not in _CACHE:
                _CACHE[k] = _KINDS[k[1]](**row["config"])


_SHIPPED = []


def _load_shipped() -> None:
    if _SHIPPED:
        return
    _SHIPPED.append(True)
    if os.path.isdir(TABLES):
        for name in sorted(os.listdir(TABLES)):
            if name.endswith(".json"):
                load(os.path.join(TABLES, name), override=False)


def clear() -> None:
    _CACHE.clear()
    _TUNED.clear()
    _SHIPPED.clear()
