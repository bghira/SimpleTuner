from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
from diffusers.utils import BaseOutput


@dataclass
class LongCatImagePipelineOutput(BaseOutput):
    """
    Output class for LongCat-Image pipelines.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
