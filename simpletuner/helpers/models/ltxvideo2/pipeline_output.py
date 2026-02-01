from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.utils import BaseOutput


@dataclass
class LTX2PipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, List[List[PIL.Image.Image]], or `None`):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`. `None` when `audio_only=True`.
        audio (`torch.Tensor`, `np.ndarray`):
            TODO
    """

    frames: Optional[torch.Tensor]
    audio: torch.Tensor
