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
        if num_inference_steps not in (None, 6):
            raise ValueError("MiniMax-H3-Image requires six inference steps.")
        if sigmas is not None and tuple(float(value) for value in sigmas) != MINIMAX_H3_IMAGE_SIGMAS:
            raise ValueError("MiniMax-H3-Image requires its released PDD sigma schedule.")
        timesteps = torch.tensor([1 - sigma for sigma in MINIMAX_H3_IMAGE_SIGMAS[:-1]], dtype=torch.float32)
        return super().set_timesteps(device=device, sigmas=torch.cat([1 - timesteps, torch.zeros(1)]))
