from __future__ import annotations

import json
import math
from collections.abc import Sequence
from numbers import Real
from typing import Optional

import torch
from torch.distributions import Distribution, constraints


def parse_cubic_spline_weights(raw_value) -> Optional[tuple[float, ...]]:
    """Parse configured density knots while preserving None versus an empty uniform schedule."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped or stripped.lower() == "none":
            return None
        try:
            raw_value = json.loads(stripped)
        except json.JSONDecodeError:
            segments = [segment.strip() for segment in stripped.replace(";", ",").split(",")]
            if any(not segment for segment in segments):
                raise ValueError("flow_cubic_schedule_weights contains an empty value.")
            raw_value = segments

    if torch.is_tensor(raw_value):
        raw_value = raw_value.detach().cpu().flatten().tolist()
    elif hasattr(raw_value, "tolist") and not isinstance(raw_value, (str, bytes)):
        raw_value = raw_value.tolist()
    if isinstance(raw_value, Real):
        raw_value = [raw_value]
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError("flow_cubic_schedule_weights must be a JSON array or comma-separated list of numbers.")

    weights = []
    for index, raw_weight in enumerate(raw_value):
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flow_cubic_schedule_weights[{index}] must be numeric.") from exc
        if not math.isfinite(weight):
            raise ValueError(f"flow_cubic_schedule_weights[{index}] must be finite.")
        if weight < 0:
            raise ValueError(f"flow_cubic_schedule_weights[{index}] must be non-negative.")
        weights.append(weight)

    if len(weights) > 1 and not any(weight > 0 for weight in weights):
        raise ValueError("flow_cubic_schedule_weights must contain a positive value when two or more knots are used.")
    return tuple(weights)


class CubicSplineDistribution(Distribution):
    """A normalized PDF through equally spaced non-negative density knots."""

    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(
        self,
        weights: Sequence[float],
        *,
        device: torch.device | str = "cpu",
        resolution: int = 4097,
        validate_args: Optional[bool] = None,
    ):
        parsed = parse_cubic_spline_weights(weights)
        if parsed is None:
            raise ValueError("CubicSplineDistribution requires an explicit weight sequence.")
        if resolution < 257:
            raise ValueError("CubicSplineDistribution resolution must be at least 257.")

        self.weights = parsed
        self.resolution = int(resolution)
        self.device = torch.device(device)
        self.is_uniform = len(parsed) <= 1
        self.grid_x: Optional[torch.Tensor] = None
        self.pdf_grid: Optional[torch.Tensor] = None
        self.cdf_grid: Optional[torch.Tensor] = None
        if not self.is_uniform:
            self._build_grid()
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size(), validate_args=validate_args)

    @staticmethod
    def _pchip_slopes(values: torch.Tensor, spacing: float) -> torch.Tensor:
        deltas = (values[1:] - values[:-1]) / spacing
        slopes = torch.zeros_like(values)
        if values.numel() == 2:
            slopes[:] = deltas[0]
            return slopes

        previous = deltas[:-1]
        following = deltas[1:]
        same_direction = previous * following > 0
        denominator = torch.where(same_direction, previous + following, torch.ones_like(previous))
        slopes[1:-1] = torch.where(
            same_direction,
            2.0 * previous * following / denominator,
            torch.zeros_like(previous),
        )

        first = (3.0 * deltas[0] - deltas[1]) / 2.0
        first = torch.where(torch.sign(first) != torch.sign(deltas[0]), torch.zeros_like(first), first)
        first = torch.where(
            (torch.sign(deltas[0]) != torch.sign(deltas[1])) & (first.abs() > 3.0 * deltas[0].abs()),
            3.0 * deltas[0],
            first,
        )
        last = (3.0 * deltas[-1] - deltas[-2]) / 2.0
        last = torch.where(torch.sign(last) != torch.sign(deltas[-1]), torch.zeros_like(last), last)
        last = torch.where(
            (torch.sign(deltas[-1]) != torch.sign(deltas[-2])) & (last.abs() > 3.0 * deltas[-1].abs()),
            3.0 * deltas[-1],
            last,
        )
        slopes[0] = first
        slopes[-1] = last
        return slopes

    def _build_grid(self) -> None:
        values = torch.tensor(self.weights, device=self.device, dtype=torch.float32)
        knot_count = values.numel()
        spacing = 1.0 / (knot_count - 1)
        positions = torch.linspace(0.0, knot_count - 1, self.resolution, device=self.device, dtype=torch.float32)
        segment = positions.floor().long().clamp(max=knot_count - 2)
        local = positions - segment

        if knot_count == 2:
            pdf = torch.lerp(values[0].expand_as(local), values[1].expand_as(local), local)
        else:
            slopes = self._pchip_slopes(values, spacing)
            local_squared = local.square()
            local_cubed = local_squared * local
            h00 = 2.0 * local_cubed - 3.0 * local_squared + 1.0
            h10 = local_cubed - 2.0 * local_squared + local
            h01 = -2.0 * local_cubed + 3.0 * local_squared
            h11 = local_cubed - local_squared
            pdf = (
                h00 * values[segment]
                + h10 * spacing * slopes[segment]
                + h01 * values[segment + 1]
                + h11 * spacing * slopes[segment + 1]
            )

        pdf = pdf.clamp_min(0.0)
        grid_x = torch.linspace(0.0, 1.0, self.resolution, device=self.device, dtype=torch.float32)
        dx = 1.0 / (self.resolution - 1)
        interval_areas = (pdf[:-1] + pdf[1:]) * (0.5 * dx)
        total_area = interval_areas.sum()
        if not torch.isfinite(total_area) or total_area <= 0:
            raise ValueError("Cubic spline density has zero or non-finite area.")

        self.grid_x = grid_x
        self.pdf_grid = pdf / total_area
        self.cdf_grid = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(interval_areas, dim=0)])
        self.cdf_grid = self.cdf_grid / self.cdf_grid[-1]

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = torch.Size(sample_shape)
        uniform = torch.rand(shape, device=self.device, dtype=torch.float32)
        if self.is_uniform:
            return uniform

        indices = torch.searchsorted(self.cdf_grid, uniform).clamp(1, self.resolution - 1)
        lower = indices - 1
        cdf_lower = self.cdf_grid[lower]
        cdf_upper = self.cdf_grid[indices]
        denominator = (cdf_upper - cdf_lower).clamp_min(torch.finfo(torch.float32).eps)
        fraction = (uniform - cdf_lower) / denominator
        return torch.lerp(self.grid_x[lower], self.grid_x[indices], fraction)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if self.is_uniform:
            result = torch.zeros_like(value)
        else:
            scaled = value.clamp(0.0, 1.0) * (self.resolution - 1)
            lower = scaled.floor().long().clamp(0, self.resolution - 1)
            upper = (lower + 1).clamp(max=self.resolution - 1)
            density = torch.lerp(self.pdf_grid[lower], self.pdf_grid[upper], scaled - lower)
            result = density.log()
        outside = (value < 0.0) | (value > 1.0)
        return torch.where(outside, torch.full_like(result, -torch.inf), result)
