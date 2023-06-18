from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
import json
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
        aspect_ratio_buckets=[ 1.0, 1.5, 0.67, 0.75, 1.78 ],
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
            print(f'Building transformations.')
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    def assign_to_buckets(self):
        cache_file = self.instance_data_root / "aspect_ratio_bucket_indices.json"
        if cache_file.exists():
            print("Loading aspect ratio bucket indices from cache file.")
            with cache_file.open("r") as f:
                self.aspect_ratio_bucket_indices = json.load(f)
        else:
            print("Computing aspect ratio bucket indices.")
            self.aspect_ratio_bucket_indices = {}
            for i in tqdm(range(len(self.instance_images_path)), desc="Assigning to buckets"):
                try:
                    image_path = self.instance_images_path[i]
                    image = Image.open(image_path)
                    image = self._resize_for_condition_image(image, self.size)
                    aspect_ratio = image.width / image.height
                    aspect_ratio = round(aspect_ratio, 2) # Round to 2 decimal places to avoid excessive unique buckets

                    # Create a new bucket if it doesn't exist
                    if str(aspect_ratio) not in self.aspect_ratio_bucket_indices:
                        self.aspect_ratio_bucket_indices[str(aspect_ratio)] = []

                    self.aspect_ratio_bucket_indices[str(aspect_ratio)].append(i)
                    with cache_file.open("w") as f:
                        json.dump(self.aspect_ratio_bucket_indices, f)
                except Exception as e:
                    print(f'Error processing image {image_path}.')
                    print(e)
                    continue
        return self.aspect_ratio_bucket_indices


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if self.print_names:
            print(f'\nOpen image: {self.instance_images_path[index % self.num_instance_images]}')
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_prompt = self.instance_prompt
        if self.use_captions:
            instance_prompt = self.instance_images_path[
                index % self.num_instance_images
            ].stem
            # Remove underscores and swap with spaces:
            instance_prompt = instance_prompt.replace("_", " ")
            instance_prompt = instance_prompt.split("upscaled by")[0]
            instance_prompt = instance_prompt.split("upscaled beta")[0]
            if self.prepend_instance_prompt:
                instance_prompt = self.instance_prompt + " " + instance_prompt
        if self.print_names:
            print(f'Prompt: {instance_prompt}')
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self._resize_for_condition_image(instance_image, self.size)
        if not self.use_original_images:
            example["instance_images"] = self.image_transforms(instance_image)
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
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img