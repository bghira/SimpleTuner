"""Attention backend shim for MageFlow's varlen attention calls.

Exports a single ``flash_attn_varlen_func`` with the FA2 calling convention.
FlashAttention backends are resolved through SimpleTuner's packed attention
dispatcher so Hub kernels such as ``kernels-community/flash-attn2`` work
without requiring users to build the ``flash-attn`` package manually.

Modules that previously did ``from flash_attn import flash_attn_varlen_func``
should import from here instead.
"""

from __future__ import annotations

from typing import Any, Callable

from simpletuner.helpers.training import attention_backend as simpletuner_attention_backend

_SDPA_ALIASES = {
    "diffusers",
    "native-math",
    "sdpa",
    "scaled-dot-product-attention",
    "scaled_dot_product_attention",
    "torch-sdpa",
    "torch_sdpa",
}
_PACKED_ALIAS_OVERRIDES = {
    "fa2": "flash2",
    "fa3": "flash3",
    "fa4": "flash4",
    "flash-attention-2": "flash-attn-2",
    "flash-attention-2-hub": "flash-attn-2-hub",
    "flash-attention-3": "flash-attn-3",
    "flash-attention-3-hub": "flash-attn-3-hub",
    "flash-attention-3-varlen": "flash-attn-3-varlen",
    "flash-attention-3-varlen-hub": "flash-attn-3-varlen-hub",
    "flash-attention-4": "flash-attn-4",
    "flash-attention-4-hub": "flash-attn-4-hub",
}

_BACKEND: str = "sdpa"
_RESOLVED_FN: Callable[..., Any] | None = None


def _normalize(name: str) -> str:
    n = name.lower().strip().replace("_", "-")
    n = _PACKED_ALIAS_OVERRIDES.get(n, n)
    if n in _SDPA_ALIASES:
        return "sdpa"
    if n in simpletuner_attention_backend._PACKED_BACKEND_ALIASES:
        return n
    raise ValueError(
        f"Unknown attention backend {name!r}; expected one of "
        f"{sorted(_SDPA_ALIASES | set(simpletuner_attention_backend._PACKED_BACKEND_ALIASES))}"
    )


def set_attn_backend(name: str) -> None:
    """Select the flash-attn backend used by ``flash_attn_varlen_func``.

    Safe to call multiple times; clears the cached resolution on change.
    """
    global _BACKEND, _RESOLVED_FN
    new = _normalize(name)
    if new != _BACKEND:
        _RESOLVED_FN = None
    _BACKEND = new


def _pop_qkv(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    if len(args) >= 3:
        return args[0], args[1], args[2], args[3:]

    q = kwargs.pop("q", kwargs.pop("query", None))
    k = kwargs.pop("k", kwargs.pop("key", None))
    v = kwargs.pop("v", kwargs.pop("value", None))
    if q is None or k is None or v is None:
        raise ValueError("flash_attn_varlen_func requires q, k, and v tensors.")
    return q, k, v, args


def _pop_varlen_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, int, int]:
    names = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
    for name, value in zip(names, args):
        kwargs.setdefault(name, value)

    missing = [name for name in names if kwargs.get(name) is None]
    if missing:
        raise ValueError(f"flash_attn_varlen_func requires {', '.join(missing)}.")
    return kwargs.pop("cu_seqlens_q"), kwargs.pop("cu_seqlens_k"), kwargs.pop("max_seqlen_q"), kwargs.pop("max_seqlen_k")


def _validate_unsupported_options(
    *,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    return_attn_probs: bool = False,
    block_table=None,
) -> None:
    if alibi_slopes is not None:
        raise NotImplementedError("MageFlow attention dispatch does not support alibi_slopes")
    if return_attn_probs:
        raise NotImplementedError("MageFlow attention dispatch does not support return_attn_probs")
    if softcap and softcap > 0:
        raise NotImplementedError("MageFlow attention dispatch does not support softcap")
    if window_size not in ((-1, -1), (None, None), (0, 0)):
        raise NotImplementedError(f"MageFlow attention dispatch does not support sliding window (got {window_size})")
    if block_table is not None:
        raise NotImplementedError("MageFlow attention dispatch does not support paged attention")


def _resolve_packed() -> Callable[..., Any]:
    backend = simpletuner_attention_backend.get_packed_attention_backend(_BACKEND)

    def _packed_wrapper(*args, **kwargs):
        kwargs = dict(kwargs)
        q, k, v, remaining_args = _pop_qkv(args, kwargs)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _pop_varlen_metadata(remaining_args, kwargs)
        _validate_unsupported_options(
            window_size=kwargs.pop("window_size", (-1, -1)),
            softcap=kwargs.pop("softcap", 0.0),
            alibi_slopes=kwargs.pop("alibi_slopes", None),
            return_attn_probs=bool(kwargs.pop("return_attn_probs", False)),
            block_table=kwargs.pop("block_table", None),
        )
        kwargs.pop("deterministic", None)
        return backend.varlen_unpacked(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=kwargs.pop("dropout_p", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=bool(kwargs.pop("causal", False)),
        )

    return _packed_wrapper


def _resolve_sdpa() -> Callable[..., Any]:
    """FA2 varlen → per-sequence torch.SDPA fallback.

    Use when flash-attn is unavailable (e.g. CUDA 13 has no prebuilt wheel
    and source build is brittle). Slower than FA2 (one SDPA dispatch per
    sequence), but functionally equivalent for the dense / causal / no-alibi
    paths mageflow actually uses. Window / softcap / alibi / paged-attn /
    return_attn_probs are not supported and will raise.
    """
    import torch
    import torch.nn.functional as F

    def _sdpa_wrapper(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p: float = 0.0,
        softmax_scale=None,
        causal: bool = False,
        window_size=(-1, -1),
        softcap: float = 0.0,
        alibi_slopes=None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        block_table=None,
        **_unused: Any,
    ):
        if dropout_p and dropout_p > 0:
            raise NotImplementedError("SDPA backend does not support dropout_p>0")
        if alibi_slopes is not None:
            raise NotImplementedError("SDPA backend does not support alibi_slopes")
        if return_attn_probs:
            raise NotImplementedError("SDPA backend does not support return_attn_probs")
        if softcap and softcap > 0:
            raise NotImplementedError("SDPA backend does not support softcap")
        if window_size not in ((-1, -1), (None, None), (0, 0)):
            raise NotImplementedError(f"SDPA backend does not support sliding window (got {window_size})")
        if block_table is not None:
            raise NotImplementedError("SDPA backend does not support paged attention")
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("SDPA backend requires cu_seqlens_q and cu_seqlens_k")

        # GQA: FA2 broadcasts k/v across query head groups natively; torch SDPA
        # does not (the q vs k head-dim mismatch is the AssertionError "tensor
        # a (32) must match tensor b (8) at non-singleton dimension 1" we'd see
        # otherwise). Repeat k/v along the head dim to match q before the loop.
        n_heads_q = q.shape[1]
        n_heads_kv = k.shape[1]
        if n_heads_q != n_heads_kv:
            if n_heads_q % n_heads_kv != 0:
                raise ValueError(
                    f"SDPA backend GQA expansion requires q heads ({n_heads_q}) "
                    f"to be divisible by k/v heads ({n_heads_kv})"
                )
            repeat = n_heads_q // n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # q/k/v: (total_tokens, nheads, head_dim). Dispatch SDPA per sequence,
        # then concat. Python-level loop is fine since nseq is small (one per
        # image in the pack) and image-gen latency is dominated by sampling.
        cu_q = cu_seqlens_q.tolist()
        cu_k = cu_seqlens_k.tolist()
        outs = []
        for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
            # (s, h, d) → (1, h, s, d)
            q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
            k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
            v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                q_i,
                k_i,
                v_i,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=softmax_scale,
            )
            # (1, h, s, d) → (s, h, d)
            outs.append(out_i.squeeze(0).transpose(0, 1))
        return torch.cat(outs, dim=0).contiguous()

    return _sdpa_wrapper


def _resolve() -> Callable[..., Any]:
    global _RESOLVED_FN
    if _RESOLVED_FN is None:
        if _BACKEND == "sdpa":
            _RESOLVED_FN = _resolve_sdpa()
        else:
            _RESOLVED_FN = _resolve_packed()
    return _RESOLVED_FN


def flash_attn_varlen_func(*args, **kwargs):
    return _resolve()(*args, **kwargs)


__all__ = ["flash_attn_varlen_func", "set_attn_backend"]
