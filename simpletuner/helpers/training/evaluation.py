import logging
import os
from functools import partial

import numpy as np
import torch
from PIL import Image

try:
    from torchmetrics.functional.multimodal import clip_score

    _clip_available = True
except Exception:  # pragma: no cover - optional dependency
    clip_score = None  # type: ignore[assignment]
    _clip_available = False

try:
    from torchvision import transforms

    _transforms_available = True
except Exception:  # pragma: no cover - optional dependency
    transforms = None  # type: ignore[assignment]
    _transforms_available = False

from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("ModelEvaluator")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

model_evaluator_map = {
    "clip": "CLIPModelEvaluator",
}


class ModelEvaluator:
    def __init__(self, pretrained_model_name_or_path):
        raise NotImplementedError("Subclasses is incomplete, no __init__ method was found.")

    def evaluate(self, images, prompts):
        raise NotImplementedError("Subclasses should implement the evaluate() method.")

    @staticmethod
    def from_config(args):
        """Instantiate a ModelEvaluator from the training config, if set to do so."""
        accelerator = StateTracker.get_accelerator()
        if accelerator is None or not getattr(accelerator, "is_main_process", True):
            return None
        evaluation_type = getattr(args, "evaluation_type", None)
        if evaluation_type is not None and evaluation_type.lower() != "" and evaluation_type.lower() != "none":
            if not (_clip_available and _transforms_available):
                raise RuntimeError("CLIP evaluation requires torchmetrics[vision] and torchvision to be installed.")
            model_evaluator = model_evaluator_map[evaluation_type]
            pretrained_model_path = getattr(args, "pretrained_evaluation_model_name_or_path", None)
            return globals()[model_evaluator](pretrained_model_path)

        return None


class CLIPModelEvaluator(ModelEvaluator):
    def __init__(self, pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"):
        if not (_clip_available and _transforms_available):
            raise RuntimeError("CLIP evaluation requires torchmetrics[vision] and torchvision to be installed.")
        self.clip_score_fn = partial(clip_score, model_name_or_path=pretrained_model_name_or_path)
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def evaluate(self, images, prompts):
        # Preprocess images
        images_tensor = torch.stack([self.preprocess(img) * 255 for img in images])
        # Compute CLIP scores
        result = self.clip_score_fn(images_tensor, prompts).detach().cpu()

        return result
