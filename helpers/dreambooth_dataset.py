from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from .state_tracker import StateTracker
from PIL import Image
import json, logging
from tqdm import tqdm


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        aspect_ratio_buckets=[1.0, 1.5, 0.67, 0.75, 1.78],
        size=768,
        center_crop=False,
        print_names=False,
        use_captions=True,
        prepend_instance_prompt=False,
        use_original_images=False,
    ):
        self.prepend_instance_prompt = prepend_instance_prompt
        self.use_captions = use_captions
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.print_names = print_names
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
        if not use_original_images:
            logging.debug(f"Building transformations.")
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

    def load_aspect_ratio_bucket_indices(self, cache_file):
        logging.info("Loading aspect ratio bucket indices from cache file.")
        with cache_file.open("r") as f:
            aspect_ratio_bucket_indices = json.load(f)
            for bucket in aspect_ratio_bucket_indices:
                bucket[:] = [
                    image_file for image_file in bucket if Path(image_file).exists()
                ]
        with cache_file.open("w") as f:
            json.dump(aspect_ratio_bucket_indices, f)
        return aspect_ratio_bucket_indices

    def compute_aspect_ratio_bucket_indices(self, cache_file):
        logging.warning("Computing aspect ratio bucket indices.")
        aspect_ratio_bucket_indices = {}
        for i in tqdm(
            range(len(self.instance_images_path)), desc="Assigning to buckets"
        ):
            try:
                image_path = str(self.instance_images_path[i])
                image = Image.open(image_path)
                # Apply EXIF transforms
                image = exif_transpose(image)
                aspect_ratio = round(
                    image.width / image.height, 3
                )  # Round to avoid excessive unique buckets
                # Create a new bucket if it doesn't exist
                if str(aspect_ratio) not in aspect_ratio_bucket_indices:
                    aspect_ratio_bucket_indices[str(aspect_ratio)] = []
                aspect_ratio_bucket_indices[str(aspect_ratio)].append(image_path)
                with cache_file.open("w") as f:
                    json.dump(aspect_ratio_bucket_indices, f)
            except Exception as e:
                logging.error(f"Error processing image {image_path}.")
                logging.error(e)
                continue
            finally:
                image.close()
        return aspect_ratio_bucket_indices

    def assign_to_buckets(self):
        cache_file = self.instance_data_root / "aspect_ratio_bucket_indices.json"
        if cache_file.exists():
            return self.load_aspect_ratio_bucket_indices(cache_file)
        else:
            return self.compute_aspect_ratio_bucket_indices(cache_file)

    def __len__(self):
        return self._length

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
            logging.debug(f"Prompt: {instance_prompt}")
        return instance_prompt

    def __getitem__(self, image_path):
        example = {}
        if self.print_names:
            logging.debug(f"Open image: {image_path}")
        instance_image = Image.open(image_path)
        # Apply EXIF transformations.
        if StateTracker.status_training():
            instance_image = exif_transpose(instance_image)
        instance_prompt = self._prepare_instance_prompt(image_path)
        if not instance_image.mode == "RGB" and StateTracker.status_training():
            instance_image = instance_image.convert("RGB")
        if StateTracker.status_training():
            example["instance_images"] = self._resize_for_condition_image(
                instance_image, self.size
            )
        else:
            example["instance_images"] = instance_image
        if not self.use_original_images and StateTracker.status_training():
            example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = None
        if StateTracker.status_training():
            example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
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
        if W == resolution and H == resolution:
            logging.debug(f"Returning square image of size {resolution}x{resolution}")
            return input_image
        if W == H:
            W = resolution
            H = resolution
        msg = f"{msg} {W}x{H}."
        logging.debug(msg)
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img
