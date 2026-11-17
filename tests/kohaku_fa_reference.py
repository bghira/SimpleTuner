"""Independent backward equations conditioned on the actual rounded forward O."""

import torch

from scripts.benchmark_attention_precision import rounded_reference


@torch.no_grad()
def attention_reference(inputs, dout, mask=None, causal=False, scale=None, *, output=None):
    # Delta is defined from the implemented forward O, not an ideal FP64 O.
    # Recompute it eagerly when the test constructs its reference before capture;
    # the returned forward value still comes from independent equations.
    if output is None:
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        output = attention(*inputs, mask=mask, causal=causal, scale=scale)
    return rounded_reference(inputs, dout, mask, causal, scale, output=output)
