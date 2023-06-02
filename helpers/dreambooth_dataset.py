from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image

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
        size=768,
        center_crop=False,
        print_names=False,
        use_captions=True,
        prepend_instance_prompt=False
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

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if self.print_names:
            print(f'Open image: {self.instance_images_path[index % self.num_instance_images]}')
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

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example
