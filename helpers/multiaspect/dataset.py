from torch.utils.data import Dataset
from pathlib import Path
import logging, os

from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.multiaspect.bucket import BucketManager
from helpers.prompts import PromptHandler

logger = logging.getLogger("MultiAspectDataset")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


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
        data_backend: BaseDataBackend,
        instance_prompt: str = None,
        tokenizer=None,
        aspect_ratio_buckets=[1.0, 1.5, 0.67, 0.75, 1.78],
        size=1024,
        print_names=False,
        use_captions=True,
        prepend_instance_prompt=False,
        use_original_images=False,
        caption_dropout_interval: int = 0,
        use_precomputed_token_ids: bool = True,
        debug_dataset_loader: bool = False,
        caption_strategy: str = "filename",
        return_tensor: bool = False,
        size_type: str = "pixel",
    ):
        self.prepend_instance_prompt = prepend_instance_prompt
        self.bucket_manager = bucket_manager
        self.data_backend = data_backend
        self.use_captions = use_captions
        self.size = size
        self.size_type = size_type
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
        logger.debug(f"Building transformations.")
        self.image_transforms = MultiaspectImage.get_image_transforms()
        self.return_tensor = return_tensor

    def __len__(self):
        return len(self.bucket_manager)

    def __getitem__(self, image_tuple):
        output_data = []
        for sample in image_tuple:
            image_path = sample["image_path"]
            logger.debug(f"Running __getitem__ for {image_path} inside Dataloader.")
            image_metadata = self.bucket_manager.get_metadata_by_filepath(image_path)
            image_metadata["image_path"] = image_path

            if (
                image_metadata["original_size"] is None
                or image_metadata["target_size"] is None
            ):
                raise Exception(
                    f"Metadata was unavailable for image: {image_path}. Ensure --skip_file_discovery=metadata is not set."
                    f" Metadata: {self.bucket_manager.get_metadata_by_filepath(image_path)}"
                )

            if self.print_names:
                logger.info(f"Dataset is now using image: {image_path}")

            # Use the magic prompt handler to retrieve the captions.
            image_metadata["instance_prompt_text"] = PromptHandler.magic_prompt(
                data_backend=self.data_backend,
                image_path=image_path,
                caption_strategy=self.caption_strategy,
                use_captions=self.use_captions,
                prepend_instance_prompt=self.prepend_instance_prompt,
            )
            output_data.append(image_metadata)

        return output_data
