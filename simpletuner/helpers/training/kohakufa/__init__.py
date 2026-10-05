# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""KohakuFA: flash attention for Blackwell in Gluon, with a numerically careful softmax.

    from simpletuner.helpers.training.kohakufa import attention
    out = attention(q, k, v, causal=True)  # q [B, H, Sq, D], k / v [B, H, Skv, D]

See README.md for what is supported and docs/precision.md for the numerics.
"""

from simpletuner.helpers.training.kohakufa.api import attention, attention_varlen, varlen_plan
from simpletuner.helpers.training.kohakufa.mask import PackedMask, pack_mask

__all__ = ["PackedMask", "attention", "attention_varlen", "pack_mask", "varlen_plan"]
__version__ = "0.1.0"
