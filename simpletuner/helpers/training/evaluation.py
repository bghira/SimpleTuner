import logging
import os
from functools import partial

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms

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
        if not StateTracker.get_accelerator().is_main_process:
            return None
        if (
            args.evaluation_type is not None
            and args.evaluation_type.lower() != ""
            and args.evaluation_type.lower() != "none"
        ):
            model_evaluator = model_evaluator_map[args.evaluation_type]
            return globals()[model_evaluator](args.pretrained_evaluation_model_name_or_path)

        return None


class CLIPModelEvaluator(ModelEvaluator):
    def __init__(self, pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"):
        self.clip_score_fn = partial(clip_score, model_name_or_path=pretrained_model_name_or_path)
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def evaluate(self, images, prompts):
        # Preprocess images
        images_tensor = torch.stack([self.preprocess(img) * 255 for img in images])
        # Compute CLIP scores
        result = self.clip_score_fn(images_tensor, prompts).detach().cpu()

        return result
