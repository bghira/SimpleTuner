from torch.utils.data import Dataset
from pathlib import Path
from PIL.ImageOps import exif_transpose
from .state_tracker import StateTracker
from PIL import Image
import json, logging, os, multiprocessing
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager, Value, Lock, Process, Queue
import numpy as np
from itertools import repeat
from ctypes import c_int

from helpers.multiaspect.image import MultiaspectImage
from helpers.multiaspect.bucket import BucketManager
from helpers.prompts import PromptHandler

logger = logging.getLogger("MultiAspectDataset")
logger.setLevel(os.environ.get('SIMPLETUNER_LOG_LEVEL', 'WARNING'))
from concurrent.futures import ThreadPoolExecutor
import threading

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel('WARNING')
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel('WARNING')

multiprocessing.set_start_method("fork")


class MultiAspectDataset(Dataset):
    """
    A multi-aspect dataset requires special consideration and handling.
    This class implements bucketed data loading for precomputed text embeddings.
    This class does not do any image transforms, as those are handled by VAECache.
    """

    def __init__(
        self,
        instance_data_root,
        accelerator,
        bucket_manager: BucketManager,
        instance_prompt: str = None,
        tokenizer=None,
        aspect_ratio_buckets=[1.0, 1.5, 0.67, 0.75, 1.78],
        size=1024,
        center_crop=False,
        print_names=False,
        use_captions=True,
        prepend_instance_prompt=False,
        use_original_images=False,
        caption_dropout_interval: int = 0,
        use_precomputed_token_ids: bool = True,
        debug_dataset_loader: bool = False,
        caption_strategy: str = "filename",
    ):
        self.prepend_instance_prompt = prepend_instance_prompt
        self.bucket_manager = bucket_manager
        self.use_captions = use_captions
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.print_names = print_names
        self.debug_dataset_loader = debug_dataset_loader
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )
        self.instance_prompt = instance_prompt
        self.aspect_ratio_buckets = aspect_ratio_buckets
        self.use_original_images = use_original_images
        self.accelerator = accelerator
        self.caption_dropout_interval = caption_dropout_interval
        self.caption_loop_count = 0
        self.caption_strategy = caption_strategy
        self.use_precomputed_token_ids = use_precomputed_token_ids
        if not use_original_images:
            logger.debug(f"Building transformations.")
            self.image_transforms = MultiaspectImage.get_image_transforms()

    def __len__(self):
        return len(self.bucket_manager)

    def __getitem__(self, image_path):
        logger.debug(f"Running __getitem__ for {image_path} inside Dataloader.")
        example = {"instance_images_path": image_path}
        if self.print_names:
            logger.info(f"Open image: {image_path}")
        
        # Images might fail to load. If so, it is better to just be the bearer of bad news.
        try:
            instance_image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Encountered error opening image: {e}")
            raise e

        # Apply EXIF and colour channel modifications.
        instance_image = MultiaspectImage.prepare_image(instance_image)

        # We return the actual Image object, so that the collate function can encode it, if needed.
        # It also makes it easier to discover the image width/height. And, I am lazy.
        example["instance_images"] = instance_image

        # Use the magic prompt handler to retrieve the captions.
        example["instance_prompt_text"] = PromptHandler.magic_prompt(
            image_path=image_path,
            caption_strategy=self.caption_strategy,
            use_captions=self.use_captions,
            prepend_instance_prompt=self.prepend_instance_prompt
        )

        return example