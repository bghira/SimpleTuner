"""Qualified compact D64 key/value ownership for RTX 5090 attention."""

import cutlass.cute as cute
from cuda.bindings import driver as cuda

from .kernels import Attention


class D64Values(Attention):
    """Retain independent scaled low components with a smaller output fragment."""

    def __init__(self, fallback):
        assert type(fallback) is Attention and fallback.mode == "kv" and fallback.centered
        assert not fallback.masked and fallback.width == 64
        super().__init__(
            "kv",
            fallback.causal,
            False,
            columns=32,
            width=64,
            warps=4,
            centered=True,
            vector=fallback.vector,
            copy_cache_global=fallback.copy_cache_global,
            reload_center=fallback.reload_center,
            kv_column_warps=2,
            kv_column_statistics=True,
        )
        self.compensated_keys = self.compensated_values = False
        self.scaled_key_residual = self.scaled_value_residual = True
        self.combined_key_products = self.combined_value_products = False


class D64Square:
    """Keep the measured preset and both runtime length boundaries explicit."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.tuned = D64Values(fallback)
        self.limit = 4096 if fallback.causal else 512

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] == k.shape[2] and q.shape[2] >= 128 and q.shape[2] <= self.limit:
            self.tuned(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
