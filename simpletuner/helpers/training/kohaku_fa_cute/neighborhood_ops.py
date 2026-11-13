"""Autograd and fake implementations for explicit 2D neighborhood attention."""

import torch

# The serialized tuning runner can restore the previous autograd behavior as
# an explicit control. This never changes arithmetic or selects by device.
_MATERIALIZE_OUTPUT_GRADS = False


def _arguments(q, k, v, out, maximum, logsum, dout=None, gradients=None):
    # Unused operands satisfy the shared kernel signature with constant storage.
    # In particular there is no S*S mask, packed mask, or neighborhood table.
    mask = torch.empty(1, device=q.device, dtype=torch.bool)
    blocks = torch.empty(1, device=q.device, dtype=torch.int32)
    delta = torch.empty_like(maximum) if dout is not None else maximum
    delta_base = delta
    if dout is not None:
        from .neighborhood_kernels import requires_separate_delta

        if requires_separate_delta():
            # Query publishes a reconstruction correction to delta; OutputDelta
            # owns delta_base. Their lifetimes overlap until KV has read both.
            delta_base = torch.empty_like(maximum)
    dq, dk, dv = (q, k, v) if gradients is None else gradients
    return q, k, v, mask, out, maximum, logsum, q if dout is None else dout, delta, dq, dk, dv, blocks, delta_base


@torch.library.custom_op("simpletuner_cute_attention::neighborhood_forward", mutates_args=())
def forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, height: int, width: int, window_h: int, window_w: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .neighborhood_kernels import launch

    out, maximum, logsum = forward_fake(q, k, v, height, width, window_h, window_w, scale)
    launch("forward", _arguments(q, k, v, out, maximum, logsum), (height, width), (window_h, window_w), scale)
    return out, maximum, logsum


@forward.register_fake
def forward_fake(q, k, v, height, width, window_h, window_w, scale):
    stats = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    return torch.empty_like(q), stats, torch.empty_like(stats)


@torch.library.custom_op("simpletuner_cute_attention::neighborhood_backward", mutates_args=())
def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    maximum: torch.Tensor,
    logsum: torch.Tensor,
    dout: torch.Tensor,
    height: int,
    width: int,
    window_h: int,
    window_w: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .neighborhood_kernels import launch

    # Saved-tensor hooks can also return logical views of the saved operands.
    q, k, v, out, maximum, logsum, dout = (
        tensor.resolve_conj().resolve_neg() for tensor in (q, k, v, out, maximum, logsum, dout)
    )
    gradients = tuple(torch.empty_like(t) for t in (q, k, v))
    launch(
        "backward",
        _arguments(q, k, v, out, maximum, logsum, dout, gradients),
        (height, width),
        (window_h, window_w),
        scale,
    )
    return gradients


@backward.register_fake
def backward_fake(q, k, v, out, maximum, logsum, dout, height, width, window_h, window_w, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v))


def setup_context(ctx, inputs, output):
    q, k, v, *ctx.parameters = inputs
    out, maximum, logsum = output
    ctx.save_for_backward(q, k, v, out, maximum, logsum)
    ctx.mark_non_differentiable(maximum, logsum)
    # Only O is differentiable. Zero gradients for the two saved statistics
    # launch fill kernels that backward never reads.
    ctx.set_materialize_grads(_MATERIALIZE_OUTPUT_GRADS)


def autograd_backward(ctx, dout, dmaximum, dlogsum):
    if dout is None:
        return (None,) * 8
    return *backward(*ctx.saved_tensors, dout, *ctx.parameters), None, None, None, None, None


forward.register_autograd(autograd_backward, setup_context=setup_context)
