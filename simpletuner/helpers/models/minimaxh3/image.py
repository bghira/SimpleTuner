"""Released MiniMax-H3-Image checkpoint and PDD sampling contract."""

import torch

from .scheduler import MiniMaxH3Scheduler

MINIMAX_H3_IMAGE_REPO = "MiniMaxAI/MiniMax-H3-Image"
MINIMAX_H3_IMAGE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)


class MiniMaxH3ImageScheduler(MiniMaxH3Scheduler):
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
        steps = 6 if num_inference_steps is None else num_inference_steps
        if not isinstance(steps, int) or steps < 6:
            raise ValueError("MiniMax-H3-Image requires an integer inference step count of at least six.")
        if sigmas is not None and (steps != 6 or tuple(float(value) for value in sigmas) != MINIMAX_H3_IMAGE_SIGMAS):
            raise ValueError("MiniMax-H3-Image requires its released PDD sigma schedule.")
        if steps == 6:
            timesteps = torch.tensor([1 - sigma for sigma in MINIMAX_H3_IMAGE_SIGMAS[:-1]], dtype=torch.float32)
            sigmas = torch.cat([1 - timesteps, torch.zeros(1)])
        else:
            grid = torch.tensor(MINIMAX_H3_IMAGE_SIGMAS, dtype=torch.float64)
            positions = torch.linspace(0, 6, steps + 1, dtype=torch.float64)
            lower = positions.floor().long().clamp(max=5)
            sigmas = torch.lerp(grid[lower], grid[lower + 1], positions - lower).float()
        self.pdd_step_indices = [
            min(5, min(range(7), key=lambda index: abs(MINIMAX_H3_IMAGE_SIGMAS[index] - sigma)))
            for sigma in sigmas[:-1].tolist()
        ]
        return super().set_timesteps(device=device, sigmas=sigmas)
