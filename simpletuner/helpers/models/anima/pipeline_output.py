# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/pipeline_output.py
# Adapted for SimpleTuner local imports.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch
from diffusers.utils import BaseOutput


@dataclass
class AnimaPipelineOutput(BaseOutput):
    """Output class for Anima image generation pipelines."""

    images: list[PIL.Image.Image] | np.ndarray | torch.Tensor
