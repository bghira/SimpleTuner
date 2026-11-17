"""Eager and compiler-safe autograd for the local CuTe kernels."""

import torch


def forward_impl(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .native import extension

    return tuple(extension().forward(q, k, v, mask, scale, causal))


# The saved-output contract now includes sm_120. Version the operators so persistent
# AOTAutograd caches cannot reuse a backward graph from the uncentered sm_120 path.
forward = torch.library.custom_op("simpletuner_cute_attention::forward_v2", mutates_args=())(forward_impl)


@forward.register_fake
def forward_fake(q, k, v, mask, scale, causal):
    stats = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    blocks = torch.empty(
        (
            1 if mask is None or mask.stride(0) == 0 else q.shape[0],
            1 if mask is None or mask.stride(1) == 0 else q.shape[1],
            1 if mask is None or mask.stride(2) == 0 else (q.shape[2] + 63) // 64,
            1 if mask is None or mask.stride(3) == 0 else (k.shape[2] + 63) // 64,
        ),
        device=q.device,
        dtype=torch.int32,
    )
    return (torch.empty_like(q), stats, torch.empty_like(stats), blocks)


def backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    maximum: torch.Tensor,
    logsum: torch.Tensor,
    dout: torch.Tensor,
    blocks: torch.Tensor,
    out: torch.Tensor | None,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .native import extension

    return tuple(extension().backward(q, k, v, mask, maximum, logsum, dout, blocks, out, scale, causal))


backward = torch.library.custom_op("simpletuner_cute_attention::backward_v2", mutates_args=())(backward_impl)


@backward.register_fake
def backward_fake(q, k, v, mask, maximum, logsum, dout, blocks, out, scale, causal):
    return tuple(torch.empty_like(tensor) for tensor in (q, k, v))


def setup_context(ctx, inputs, output):
    q, k, v, mask, ctx.scale, ctx.causal = inputs
    out, maximum, logsum, blocks = output
    ctx.has_mask = mask is not None
    ctx.centered = torch.cuda.get_device_capability(q.device.index) in (
        (8, 0),
        (8, 6),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (12, 0),
        (12, 1),
    )
    ctx.save_for_backward(
        q, k, v, maximum, logsum, blocks, *(() if mask is None else (mask,)), *((out,) if ctx.centered else ())
    )

    ctx.mark_non_differentiable(*output[1:])


def autograd_backward(ctx, dout, dmaximum, dlogsum, dblocks):
    q, k, v, maximum, logsum, blocks, *extras = ctx.saved_tensors
    dq, dk, dv = backward(
        q,
        k,
        v,
        extras[0] if ctx.has_mask else None,
        maximum,
        logsum,
        dout,
        blocks,
        extras[-1] if ctx.centered else None,
        ctx.scale,
        ctx.causal,
    )
    return dq, dk, dv, None, None, None


forward.register_autograd(autograd_backward, setup_context=setup_context)


def attention(q, k, v, *, mask=None, causal=False, scale=None):
    if mask is not None:
        mask = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    arguments = (q, k, v, mask, q.shape[-1] ** -0.5 if scale is None else float(scale), causal)
    return forward(*arguments)[0]
