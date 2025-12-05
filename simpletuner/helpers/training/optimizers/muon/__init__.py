"""Muon optimizer with optional QK-Clip integration."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import Tensor

from simpletuner.helpers.training.optimizers.adamw_bfloat16.stochastic import add_stochastic_

__all__ = ["MuonClip"]

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    use_cans: bool = False,
    cans_a_bound: float = 1e-4,
) -> Tensor:
    """Compute an orthogonalized update using Newton-Schulz iteration."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.dim() != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if not use_cans and len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must be a tuple of exactly 3 values")

    transposed = grad.size(-2) > grad.size(-1)
    X = grad.mT if transposed else grad

    working_dtype = torch.float32 if grad.dtype != torch.float64 else grad.dtype
    X = X.to(working_dtype)
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_(min=eps))

    if use_cans:
        lower_bound = cans_a_bound
        upper_bound = 1.0
        inv_3 = 1.0 / 3.0

        n = X.size(0)
        A = torch.empty(n, n, dtype=X.dtype, device=X.device)
        AX = torch.empty_like(X)

        for _ in range(ns_steps):
            a_bound, b_bound = lower_bound, upper_bound

            a_sq = a_bound * a_bound
            b_sq = b_bound * b_bound
            ab = a_bound * b_bound

            e_sq = (a_sq + ab + b_sq) * inv_3
            e_pow_1_5 = e_sq * math.sqrt(e_sq)

            common_den_part = 2.0 * e_pow_1_5
            ab_part = a_sq * b_bound + b_sq * a_bound
            alpha_den = common_den_part + ab_part
            alpha = 6.0 / alpha_den

            c1 = alpha * e_sq
            c3 = -alpha * inv_3

            torch.mm(X, X.mT, out=A)
            torch.mm(A, X, out=AX)
            X.mul_(c1).add_(AX, alpha=c3)

            eps_val = (common_den_part - ab_part) / alpha_den
            lower_bound = 1.0 - eps_val
            upper_bound = 1.0 + eps_val
    else:
        a, b, c = ns_coefficients

        n = X.size(0)
        A = torch.empty(n, n, dtype=X.dtype, device=X.device)
        B = torch.empty(n, n, dtype=X.dtype, device=X.device)

        for _ in range(ns_steps):
            torch.mm(X, X.mT, out=A)
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
            torch.addmm(X, B, X, beta=a, out=X)

    if transposed:
        X = X.mT

    return X.to(grad.dtype)


def _unnmf(row_col: tuple[Tensor, Tensor], out: Optional[Tensor] = None) -> torch.Tensor:
    if out is not None:
        return torch.outer(row_col[0], row_col[1], out=out)
    return torch.outer(row_col[0], row_col[1])


def _nnmf(matrix: torch.Tensor, out: tuple[Tensor, Tensor]) -> None:
    shape = matrix.shape
    torch.sum(matrix, dim=1, out=out[0])
    torch.sum(matrix, dim=0, out=out[1])

    if shape[0] < shape[1]:
        scale = out[0].sum()
        if scale != 0:
            out[0].div_(scale)
    else:
        scale = out[1].sum()
        if scale != 0:
            out[1].div_(scale)


_BIT_MASKS = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)


@torch.no_grad()
def _pack_bools(tensor: torch.Tensor) -> torch.Tensor:
    """Pack boolean tensor into uint8 tensor (8 bools per uint8)."""
    n, m = tensor.shape
    packed_m = (m + 7) // 8

    if m % 8 != 0:
        padded_tensor = torch.nn.functional.pad(tensor, (0, packed_m * 8 - m), "constant", 0).to(torch.uint8)
    else:
        padded_tensor = tensor.to(torch.uint8)

    bit_masks = _BIT_MASKS.to(tensor.device)
    reshaped = padded_tensor.reshape(n, packed_m, 8)
    packed = (reshaped * bit_masks).sum(dim=2, dtype=torch.uint8)

    return packed


@torch.no_grad()
def _unpack_bools(packed_tensor: torch.Tensor, original_m: int) -> torch.Tensor:
    """Unpack uint8 tensor back to boolean tensor."""
    n, _ = packed_tensor.shape
    bit_masks = _BIT_MASKS.to(packed_tensor.device).view(1, 1, 8)
    unpacked = ((packed_tensor.unsqueeze(2) & bit_masks) != 0).reshape(n, -1)[:, :original_m]
    return unpacked


def _get_effective_shape(numel: int) -> tuple[int, int]:
    if numel <= 0:
        return (0, 0)
    for i in reversed(range(1, int(numel**0.5) + 1)):
        if numel % i == 0:
            return (numel // i, i)
    return (numel, 1)


class MuonClip(torch.optim.Optimizer):
    """
    Muon optimizer with optional QK-Clip integration.

    Args mirror the PyTorch Muon implementation; QK-Clip is enabled by passing
    per-head `attention_max_logits` to `step`.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        ns_steps: int = DEFAULT_NS_STEPS,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        rms_scale_factor: float = 0.2,
        use_smmf: bool = False,
        vector_reshape: bool = False,
        stochastic_rounding: bool = True,
        use_cans: bool = False,
        cans_a_bound: float = 1e-4,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps >= 100:
            raise ValueError("Number of steps must be less than 100 for computational efficiency")
        if not use_cans and len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a tuple of exactly 3 values")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            rms_scale_factor=rms_scale_factor,
            use_smmf=use_smmf,
            vector_reshape=vector_reshape,
            use_cans=use_cans,
            cans_a_bound=cans_a_bound,
        )
        super().__init__(params, defaults)

        self.stochastic_rounding = stochastic_rounding
        self._param_to_name: Dict[int, str] = {}

    @torch.no_grad()
    def step(self, closure=None, attention_max_logits: Optional[Dict[str, torch.Tensor]] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
            attention_max_logits: Optional dict mapping parameter names to max logit values per head.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            rms_scale = group["rms_scale_factor"]
            use_smmf = group["use_smmf"]
            vector_reshape = group["vector_reshape"]
            use_cans = group["use_cans"]
            cans_a_bound = group["cans_a_bound"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                use_factored = use_smmf and (grad.dim() >= 2 or (grad.dim() == 1 and vector_reshape))

                if len(state) == 0:
                    if use_factored:
                        if grad.dim() == 1:
                            state["effective_shape"] = _get_effective_shape(grad.numel())
                        else:
                            state["effective_shape"] = (grad.shape[0], grad[0].numel())

                        d1, d2 = state["effective_shape"]
                        state["mu_row"] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                        state["mu_col"] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                        packed_d2 = (d2 + 7) // 8
                        state["sign_buf"] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                        state["mt_buf"] = torch.empty((d1, d2), device=p.device, dtype=torch.float32)
                        state["factored"] = True
                    else:
                        state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        state["factored"] = False

                if grad.dim() == 2 or (grad.dim() > 2 and use_factored):
                    original_shape = grad.shape

                    if state.get("factored", False):
                        d1, d2 = state["effective_shape"]
                        mt_buf = state["mt_buf"]

                        _unnmf((state["mu_row"], state["mu_col"]), out=mt_buf)
                        unpacked_sign = _unpack_bools(state["sign_buf"], original_m=d2)
                        torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)

                        mt_buf.lerp_(grad.reshape(d1, d2), 1 - momentum)

                        ortho_update = _zeropower_via_newtonschulz(
                            mt_buf, ns_coefficients, ns_steps, eps, use_cans, cans_a_bound
                        )

                        scale_factor = math.sqrt(max(d1, d2)) * rms_scale
                        ortho_update.mul_(scale_factor)

                        state["sign_buf"] = _pack_bools(mt_buf > 0)
                        _nnmf(mt_buf.abs(), out=(state["mu_row"], state["mu_col"]))

                        ortho_update = ortho_update.reshape(original_shape)
                    else:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.lerp_(grad, 1 - momentum)

                        update_2d = momentum_buffer.reshape(original_shape[0], -1)
                        ortho_update = _zeropower_via_newtonschulz(
                            update_2d, ns_coefficients, ns_steps, eps, use_cans, cans_a_bound
                        )

                        scale_factor = math.sqrt(max(update_2d.shape[0], update_2d.shape[1])) * rms_scale
                        ortho_update.mul_(scale_factor)
                        ortho_update = ortho_update.reshape(original_shape)

                    if weight_decay > 0:
                        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                            add_stochastic_(p.data, p.data, alpha=-lr * weight_decay)
                        else:
                            p.add_(p, alpha=-lr * weight_decay)

                    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                        add_stochastic_(p.data, ortho_update, alpha=-lr)
                    else:
                        p.add_(ortho_update, alpha=-lr)

                else:
                    if state.get("factored", False):
                        d1, d2 = state["effective_shape"]
                        mt_buf = state["mt_buf"]

                        _unnmf((state["mu_row"], state["mu_col"]), out=mt_buf)
                        unpacked_sign = _unpack_bools(state["sign_buf"], original_m=d2)
                        torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)

                        mt_buf.lerp_(grad.reshape(d1, d2), 1 - momentum)
                        update = mt_buf.reshape(grad.shape)

                        state["sign_buf"] = _pack_bools(mt_buf > 0)
                        _nnmf(mt_buf.abs(), out=(state["mu_row"], state["mu_col"]))
                    else:
                        update = state["momentum_buffer"]
                        update.lerp_(grad, 1 - momentum)

                    if weight_decay > 0:
                        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                            add_stochastic_(p.data, p.data, alpha=-lr * weight_decay)
                        else:
                            p.add_(p, alpha=-lr * weight_decay)

                    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                        add_stochastic_(p.data, update, alpha=-lr)
                    else:
                        p.add_(update, alpha=-lr)

        if attention_max_logits is not None:
            self._apply_qk_clip(attention_max_logits)

        return loss

    @torch.no_grad()
    def _apply_qk_clip(self, attention_max_logits: Dict[str, torch.Tensor]) -> None:
        tau = self.defaults["qk_clip_threshold"]
        alpha = self.defaults["qk_clip_alpha"]

        for group in self.param_groups:
            for p in group["params"]:
                param_name = self._get_param_name(p)

                if not param_name or param_name not in attention_max_logits:
                    continue

                max_logits = attention_max_logits[param_name]
                if max_logits.device != p.device:
                    max_logits = max_logits.to(device=p.device, dtype=p.dtype)

                gamma = (tau / max_logits).clamp_(max=1.0)
                needs_clip = gamma < 1.0

                if not needs_clip.any():
                    continue

                if "Wq_c" in param_name or "to_q" in param_name or "q_proj" in param_name or ".wq" in param_name:
                    scale = gamma.pow(alpha)
                    self._scale_attention_heads(p, scale, needs_clip)
                elif "Wk_c" in param_name or "to_k" in param_name or "k_proj" in param_name or ".wk" in param_name:
                    scale = gamma.pow(1.0 - alpha)
                    self._scale_attention_heads(p, scale, needs_clip)
                elif "Wq_r" in param_name:
                    self._scale_attention_heads(p, gamma, needs_clip)
                elif "Wk_r" in param_name:
                    continue

    @staticmethod
    def _scale_attention_heads(param: torch.Tensor, scale_factors: torch.Tensor, mask: torch.Tensor) -> None:
        """Scale specific attention heads in a parameter tensor."""
        num_heads = scale_factors.shape[0]

        if param.dim() == 2:
            head_dim = param.shape[0] // num_heads

            for h in range(num_heads):
                if mask[h]:
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    param[start_idx:end_idx].mul_(scale_factors[h])

        elif param.dim() == 3:
            for h in range(num_heads):
                if mask[h]:
                    param[h].mul_(scale_factors[h])

    def _get_param_name(self, param: torch.Tensor) -> str:
        return self._param_to_name.get(id(param), "")

    def register_attention_params(self, param_name_mapping: Dict[str, torch.nn.Parameter]) -> None:
        self._param_to_name.update({id(param): name for name, param in param_name_mapping.items()})

    def register_attention_params_from_model(
        self,
        model: torch.nn.Module,
        name_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if model is None:
            return

        if name_filter is None:
            name_filter = lambda n: ("attn" in n.lower() or "attention" in n.lower()) and (  # noqa: E731
                "q" in n.lower() or "k" in n.lower()
            )

        mapping: Dict[str, torch.nn.Parameter] = {}
        for name, param in model.named_parameters():
            if param is None:
                continue
            if name_filter(name):
                mapping[name] = param

        if mapping:
            self.register_attention_params(mapping)

    def state_dict(self) -> Dict[str, object]:
        base = super().state_dict()
        param_names = {}
        for group_idx, group in enumerate(self.param_groups):
            names = [self._param_to_name.get(id(p), "") for p in group.get("params", [])]
            param_names[group_idx] = names
        base["param_names"] = param_names
        return base

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        param_names = state_dict.pop("param_names", None)
        super().load_state_dict(state_dict)
        if param_names:
            for group_idx, names in param_names.items():
                if group_idx >= len(self.param_groups):
                    continue
                params = self.param_groups[group_idx].get("params", [])
                for param, name in zip(params, names):
                    if name:
                        self._param_to_name[id(param)] = name
