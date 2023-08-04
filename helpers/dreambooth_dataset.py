from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from .state_tracker import StateTracker
from PIL import Image
import json, logging, os
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from concurrent.futures import ThreadPoolExecutor
import threading


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
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
    ):
        self.prepend_instance_prompt = prepend_instance_prompt
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
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.aspect_ratio_buckets = aspect_ratio_buckets
        self.use_original_images = use_original_images
        self.aspect_ratio_bucket_indices = self.assign_to_buckets()
        self.caption_dropout_interval = caption_dropout_interval
        self.caption_loop_count = 0
        self.use_precomputed_token_ids = use_precomputed_token_ids
        if len(self.aspect_ratio_bucket_indices) > 0:
            logger.debug(f"Updating cache...")
            self.update_cache()
        if not use_original_images:
            logger.debug(f"Building transformations.")
            self.image_transforms = self._get_image_transforms()

    def _get_image_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.size)
                if self.center_crop
                else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def update_cache(self, base_dir=None, max_workers=10):
        """Update the aspect_ratio_bucket_indices based on the current state of the file system."""

        if base_dir is None:
            base_dir = self.instance_data_root
        else:
            base_dir = Path(base_dir)
            if not base_dir.exists():
                raise ValueError(f"Directory {base_dir} does not exist.")

        new_file_paths = [
            str(path)
            for path in base_dir.iterdir()
            if path not in self.instance_images_path
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._add_file_to_cache, new_file_paths))

        # Update the instance_images_path to include the new images
        self.instance_images_path += new_file_paths

        # Update the total number of instance images
        self.num_instance_images = len(self.instance_images_path)

        # Save updated aspect_ratio_bucket_indices to the cache file
        cache_file = self.instance_data_root / "aspect_ratio_bucket_indices.json"
        with cache_file.open("w") as f:
            json.dump(self.aspect_ratio_bucket_indices, f)

    def _add_file_to_cache(self, file_path):
        """Add a single file to the cache (thread-safe)."""
        try:
            with Image.open(file_path) as image:
                # Apply EXIF transforms
                image = exif_transpose(image)
                aspect_ratio = round(
                    image.width / image.height, 3
                )  # Round to avoid excessive unique buckets

                with threading.Lock():
                    # Create a new bucket if it doesn't exist
                    if str(aspect_ratio) not in self.aspect_ratio_bucket_indices:
                        self.aspect_ratio_bucket_indices[str(aspect_ratio)] = []
                    self.aspect_ratio_bucket_indices[str(aspect_ratio)].append(
                        file_path
                    )
        except Exception as e:
            logging.error(f"Error processing image {file_path}.")
            logging.error(e)

    def load_aspect_ratio_bucket_indices(self, cache_file):
        logging.info("Loading aspect ratio bucket indices from cache file.")
        with cache_file.open("r") as f:
            try:
                aspect_ratio_bucket_indices = json.load(f)
            except:
                logging.warn(
                    f"Could not load aspect ratio bucket indices from {cache_file}. Creating a new one!"
                )
                aspect_ratio_bucket_indices = {}
        return aspect_ratio_bucket_indices

    def compute_aspect_ratio_bucket_indices(self, cache_file):
        logging.warning("Computing aspect ratio bucket indices.")
        aspect_ratio_bucket_indices = {}

        def rglob_follow_symlinks(path: Path, pattern: str):
            for p in path.glob(pattern):
                yield p
            for p in path.iterdir():
                if p.is_dir() and not p.is_symlink():
                    yield from rglob_follow_symlinks(p, pattern)
                elif p.is_symlink():
                    real_path = Path(os.readlink(p))
                    if real_path.is_dir():
                        yield from rglob_follow_symlinks(real_path, pattern)

        all_image_files = list(
            rglob_follow_symlinks(Path(self.instance_data_root), "*.[jJpP][pPnN][gG]")
        )

        for image_path in tqdm(all_image_files, desc="Assigning to buckets"):
            try:
                image_path_str = str(image_path)
                image = Image.open(image_path_str)
                # Apply EXIF transforms
                image = exif_transpose(image)
                aspect_ratio = round(
                    image.width / image.height, 3
                )  # Round to avoid excessive unique buckets
                # Create a new bucket if it doesn't exist
                if str(aspect_ratio) not in aspect_ratio_bucket_indices:
                    aspect_ratio_bucket_indices[str(aspect_ratio)] = []
                aspect_ratio_bucket_indices[str(aspect_ratio)].append(image_path_str)
                with cache_file.open("w") as f:
                    json.dump(aspect_ratio_bucket_indices, f)
            except Exception as e:
                logging.error(f"Error processing image {image_path_str}.")
                logging.error(e)
                continue
            finally:
                if "image" in locals():
                    image.close()
        return aspect_ratio_bucket_indices

    def assign_to_buckets(self):
        cache_file = self.instance_data_root / "aspect_ratio_bucket_indices.json"
        output = None
        if cache_file.exists():
            output = self.load_aspect_ratio_bucket_indices(cache_file)
        if output is not None and len(output) > 0:
            return output
        return self.compute_aspect_ratio_bucket_indices(cache_file)

    def __len__(self):
        return self._length

    def get_all_captions(self):
        captions = []
        for image_path in self.instance_images_path:
            caption = self._prepare_instance_prompt(image_path)
            captions.append(caption)
        return captions

    def _prepare_instance_prompt(self, image_path):
        instance_prompt = self.instance_prompt
        if self.use_captions:
            instance_prompt = Path(image_path).stem
            # Remove underscores and swap with spaces:
            instance_prompt = instance_prompt.replace("_", " ")
            instance_prompt = instance_prompt.split("upscaled by")[0]
            instance_prompt = instance_prompt.split("upscaled beta")[0]
            if self.prepend_instance_prompt:
                instance_prompt = self.instance_prompt + " " + instance_prompt
        if self.print_names:
            logger.debug(f"Prompt: {instance_prompt}")
        return instance_prompt

    def caption_loop_interval_bump(self):
        self.caption_loop_count += 1
        if self.caption_loop_count > 100:
            self.caption_loop_count = 0

    def __getitem__(self, image_path):
        if self.debug_dataset_loader:
            logger.debug(f"Running __getitem__ for {image_path} inside Dataloader.")
        if not StateTracker.status_training():
            if self.debug_dataset_loader:
                logger.warning(f"Skipping getitem because we are not yet training.")
            return None
        example = {}
        if self.print_names and self.debug_dataset_loader:
            logger.debug(f"Open image: {image_path}")
        instance_image = Image.open(image_path)
        # Apply EXIF transformations.
        instance_image = exif_transpose(instance_image)
        instance_prompt = self._prepare_instance_prompt(image_path)
        if not instance_image.mode == "RGB" and StateTracker.status_training():
            instance_image = instance_image.convert("RGB")
        if StateTracker.status_training():
            logger.debug(f"Resizing sample to {self.size}")
            example["instance_images"] = self._resize_for_condition_image(
                instance_image, self.size
            )
        else:
            example["instance_images"] = instance_image
        if not self.use_original_images and StateTracker.status_training():
            example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = None
        if StateTracker.status_training():
            if self.caption_dropout_interval > 0:
                if self.caption_loop_count % self.caption_dropout_interval == 0:
                    if self.debug_dataset_loader:
                        logger.debug(
                            f"Caption dropout, removing caption: {instance_prompt}"
                        )
                    instance_prompt = ""
                self.caption_loop_interval_bump()
            if not self.use_precomputed_token_ids:
                example["instance_prompt_ids"] = self.tokenizer(
                    instance_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids
        example["instance_prompt_text"] = instance_prompt
        if self.debug_dataset_loader:
            logger.debug(
                f"Returning from __getitem__ for {image_path} inside Dataloader."
            )
        return example

    def _resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        aspect_ratio = round(W / H, 3)
        msg = f"Inspecting image of aspect {aspect_ratio} and size {W}x{H} to "
        if W < H:
            W = resolution
            H = int(resolution / aspect_ratio)  # Calculate the new height
        elif H < W:
            H = resolution
            W = int(resolution * aspect_ratio)  # Calculate the new width
        if W == H:
            W = resolution
            H = resolution
        msg = f"{msg} {W}x{H}."
        if self.debug_dataset_loader:
            logger.debug(msg)
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img
