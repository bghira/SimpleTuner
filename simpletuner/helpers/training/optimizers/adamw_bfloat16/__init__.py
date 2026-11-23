"""
Different versions appeared,
they have identical interface, but sutiable for different scenarios.
"""

__version__ = "0.2.0"

__all__ = ["AdamW_BF16"]

"""
This implementation uses torch.compile to speed up,
should be suitable for different backends.
"""

import random

import torch
from torch.optim.optimizer import Optimizer

from .stochastic import add_stochastic_, addcdiv_stochastic_


class AdamWBF16(Optimizer):
    decay_threshold = 5e-3

    def __init__(
        self,
        params,
        *,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        mup=False,
    ):
        """
        Implements AdamW optimization specifically for bfloat16 models.
        No other dtype is supported.
        Compatible with cuda graphs.
        Uses delayed accumulation for decays and compensated summation for Adam steps.
        Uses only one additional bfloat16 weight for keeping correction.
        Do not use schedulers - those can't affect cuda graphs.
        Set mup=True after assigning µP base shapes to the model.
        """
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay, lr=lr, mup=mup)
        super().__init__(params, defaults)
        self._rounding_generators = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        self._rounding_generators = {}

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if not group.get("mup", False) or not group["params"]:
            return

        # PyTorch has validated the group and normalized named parameters and iterables.
        self.param_groups.pop()
        by_width = {}
        for index, param in enumerate(group["params"]):
            shape = getattr(param, "infshape", None)
            if shape is None:
                raise ValueError("adamw_bf16 with mup=True requires mup.set_base_shapes() on every optimized parameter.")
            infinite_dims = shape.ninf()
            if infinite_dims > 2:
                raise ValueError("µP AdamW supports at most two infinite dimensions per parameter.")
            width = shape.width_mult() if infinite_dims == 2 else 1
            by_width.setdefault(width, []).append(index)

        for width, indices in by_width.items():
            scaled = dict(group)
            scaled["params"] = [group["params"][i] for i in indices]
            if "param_names" in group:
                scaled["param_names"] = [group["param_names"][i] for i in indices]
            scaled["lr"] = group["lr"] / width
            if "initial_lr" in group:
                scaled["initial_lr"] = group["initial_lr"] / width
            # Keep the decoupled decay per step (lr * weight_decay) independent of width.
            scaled["weight_decay"] = group["weight_decay"] * width
            self.param_groups.append(scaled)

    @torch.no_grad()
    def step(self, zero_grad: bool = False):
        """Performs a single optimization step."""
        param_offset = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param_index, p in enumerate(group["params"], start=param_offset):
                if p.grad is not None:
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                        state["step"] = 0.0
                        # Create state on gradient's device to support RamTorch (weights on CPU, grads on GPU)
                        grad_device = p.grad.device
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format, device=grad_device)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format, device=grad_device)
                        # accumulated shift that should be added to p, but wasn't because of truncation
                        # true value is p + shift
                        state["shift"] = torch.zeros_like(p, memory_format=torch.preserve_format, device=grad_device)
                        # using decay at each step will work only for float32, so we just remember how much owe to decay
                        # and decay once in n iterations
                        # Each weight has its own starting point to avoid simultaneous updates in all weights
                        state["accumulated_decay"] = random.Random(param_index).random() * self.decay_threshold

                    grad = p.grad
                    grad_device = grad.device

                    # Migrate state to gradient device if needed (handles checkpoint resume)
                    if state["exp_avg"].device != grad_device:
                        state["exp_avg"] = state["exp_avg"].to(grad_device)
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad_device)
                        state["shift"] = state["shift"].to(grad_device)

                    state["step"] += 1
                    lr = group["lr"]

                    state["accumulated_decay"] += group["weight_decay"] * lr
                    accum_decay = state["accumulated_decay"]
                    decay_this_iteration = (accum_decay > self.decay_threshold) * accum_decay
                    state["accumulated_decay"] -= decay_this_iteration

                    # Replicas share parameter order and checkpointed update counts, even when
                    # training noise, dropout, and the local CUDA device index differ.
                    seed = (int(state["step"]) * 0x9E3779B97F4A7C15 + param_index) & 0xFFFFFFFFFFFFFFFF
                    grad_rng = self._rounding_generator(grad_device, seed)
                    weight_rng = grad_rng if p.device == grad_device else self._rounding_generator(p.device, seed)
                    _make_step(
                        grad,
                        p,
                        state["shift"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        beta1=beta1,
                        beta2=beta2,
                        step=state["step"],
                        lr=lr,
                        eps=group["eps"],
                        decay_this_iteration=decay_this_iteration,
                        zero_grad=zero_grad,
                        grad_rng=grad_rng,
                        weight_rng=weight_rng,
                    )
            param_offset += len(group["params"])

    def _rounding_generator(self, device, seed):
        if device not in self._rounding_generators:
            self._rounding_generators[device] = torch.Generator(device=device)
        return self._rounding_generators[device].manual_seed(seed)


def _make_step(
    grad,
    p,
    shift,
    exp_avg,
    exp_avg_sq,
    beta1: float,
    beta2: float,
    step: float,
    lr: float,
    eps: float,
    decay_this_iteration: float,
    zero_grad: bool,
    grad_rng: torch.Generator,
    weight_rng: torch.Generator,
):
    # Handle RamTorch case where p is on CPU but state/grads are on GPU
    p_device = p.device
    state_device = shift.device

    # Originally:
    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg.mul_(beta1)
    add_stochastic_(exp_avg, grad, alpha=1 - beta1, generator=grad_rng)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

    denom_correction = (1 - beta2**step) ** 0.5

    # Originally:
    # shift.addcdiv_(
    #     exp_avg,
    #     exp_avg_sq.sqrt().add_(eps, alpha=1),
    #     value=-lr * denom_correction,
    # )

    addcdiv_stochastic_(
        shift,
        exp_avg,
        exp_avg_sq.sqrt().add_(eps, alpha=1),
        value=-lr * denom_correction,
        generator=grad_rng,
    )

    # For RamTorch: move shift to parameter device, update p, then compute error
    if p_device != state_device:
        # Move shift to CPU for parameter update
        shift_on_p = shift.to(p_device)
        buffer = p.clone()
        add_stochastic_(p, shift_on_p, generator=weight_rng)
        # Compute truncation error and move back to state device
        error = buffer.sub_(p).to(state_device)
        add_stochastic_(shift, error, generator=grad_rng)
        if decay_this_iteration > 0:
            shift.add_(p.to(state_device), alpha=-decay_this_iteration)
    else:
        buffer = p.clone()
        # Originally:
        # p.add_(shift)
        add_stochastic_(p, shift, generator=weight_rng)

        # Originally:
        # shift.add_(buffer.sub_(p))
        add_stochastic_(shift, buffer.sub_(p), generator=grad_rng)

        if decay_this_iteration > 0:
            shift.add_(p, alpha=-decay_this_iteration)
            # Do NOT do this, it will cause the model to become unstable.
            # add_stochastic_(shift, p, alpha=-decay_this_iteration)

    if zero_grad:
        grad.zero_()
