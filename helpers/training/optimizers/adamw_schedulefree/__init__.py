import torch
from torch.optim.optimizer import Optimizer
import math
from typing import Iterable
from helpers.training.state_tracker import StateTracker


class AdamWScheduleFreeKahan(Optimizer):
    """AdamW optimizer with schedule-free adjustments and Kahan summation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients for gradient and squared gradient moving averages (default: (0.9, 0.999)).
        eps: Added to denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 1e-2).
        warmup_steps: Number of steps to warm up the learning rate (default: 0).
        kahan_sum: Enables Kahan summation for more accurate parameter updates when training in low precision.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        warmup_steps: int = 0,
        kahan_sum: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            kahan_sum=kahan_sum,
        )
        super(AdamWScheduleFreeKahan, self).__init__(params, defaults)
        self.k = 0
        self.lr_max = -1.0
        self.last_lr = -1.0
        self.weight_sum = 0.0

    def _initialize_state(self, state, p):
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if self.defaults["kahan_sum"]:
                state["kahan_comp"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

    def eval(self):
        for group in self.param_groups:
            train_mode = group.get("train_mode", True)
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to x
                        p.data.lerp_(
                            end=state["z"].to(p.data.device), weight=1 - 1 / beta1
                        )
                group["train_mode"] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group.get("train_mode", False)
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to y
                        p.data.lerp_(end=state["z"].to(p.data.device), weight=1 - beta1)
                group["train_mode"] = True

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            warmup_steps = group.get("warmup_steps", 0)
            kahan_sum = group["kahan_sum"]

            k = self.k

            # Adjust learning rate with warmup
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            adjusted_lr = lr * sched * (bias_correction2**0.5)
            self.lr_max = max(adjusted_lr, self.lr_max)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                self._initialize_state(state, p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if kahan_sum:
                    kahan_comp = state["kahan_comp"]
                    grad.add_(kahan_comp)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)

                step_size = adjusted_lr / (bias_correction2**0.5)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay)

                # Kahan summation to improve precision
                step = exp_avg / denom
                p.data.add_(-step_size * step)

                if kahan_sum:
                    buffer = p.data.add(-step_size * step)
                    kahan_comp.copy_(p.data.sub(buffer).add(buffer.sub_(p.data)))

            self.k += 1
            self.last_lr = adjusted_lr
            StateTracker.set_last_lr(adjusted_lr)

        return loss

    def get_last_lr(self):
        return self.last_lr
