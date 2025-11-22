# This was MIT-licensed by Kandinsky Lab; now AGPL-3.0-or-later, SimpleTuner (c) bghira
from dataclasses import dataclass

import torch
from diffusers.utils import BaseOutput


@dataclass
class KandinskyPipelineOutput(BaseOutput):
    """
    Output class for Kandinsky pipelines.
    """

    frames: torch.Tensor
