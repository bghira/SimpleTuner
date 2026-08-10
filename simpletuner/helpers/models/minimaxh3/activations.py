import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMaxH3SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, gate_first: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.gate_first = bool(gate_first)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        left, right = self.proj(hidden_states).chunk(2, dim=-1)
        if self.gate_first:
            return F.silu(left) * right
        return left * F.silu(right)


class MiniMaxH3FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        gate_first: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.ModuleList(
            [
                MiniMaxH3SwiGLU(dim, inner_dim, bias=bias, gate_first=gate_first),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
