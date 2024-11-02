import os
import re
from typing import Dict, List
import json

import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
)


class OmniGenCollator:
    def __init__(self, pad_token_id=2):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        print(f"features: {features}")
        input_ids = [f[0]["input_ids"] for f in features]
        attention_masks = []
        max_length = max(len(ids) for ids in input_ids)

        # Pad input_ids and create attention masks
        padded_input_ids = []
        for ids in input_ids:
            pad_length = max_length - len(ids)
            padded_ids = [self.pad_token_id] * pad_length + ids
            attention_mask = [0] * pad_length + [1] * len(ids)
            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        padded_input_ids = torch.tensor(padded_input_ids)
        attention_masks = torch.tensor(attention_masks)

        # Handle pixel values
        pixel_values = [
            f[0]["pixel_values"] for f in features if f[0]["pixel_values"] is not None
        ]
        if pixel_values:
            pixel_values = [pv for sublist in pixel_values for pv in sublist]
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = None

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_masks,
            "pixel_values": pixel_values,
            # Include other necessary fields
        }


class OmniGenTrainingProcessor:
    def __init__(self, text_tokenizer, max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: crop_arr(pil_image, max_image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self.collator = OmniGenCollator()

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv("HF_HUB_CACHE")
            model_name = snapshot_download(
                repo_id=model_name, cache_dir=cache_folder, allow_patterns="*.json"
            )
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(text_tokenizer)

    def process_image(self, image):
        image = Image.open(image).convert("RGB")
        return self.image_transform(image)

    def process_multi_modal_prompt(self, text, input_images):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {
                "input_ids": model_inputs.input_ids,
                "pixel_values": None,
                "image_sizes": None,
            }

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [
            self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)
        ]

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(
            range(1, len(unique_image_ids) + 1)
        ), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(
            input_images
        ), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"

        input_images = [input_images[x - 1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) - 1:
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) * input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx + size])
                all_input_ids.extend([0] * size)

        return {
            "input_ids": all_input_ids,
            "pixel_values": input_images,
            "image_sizes": img_inx,
        }

    def add_prefix_instruction(self, prompt):
        user_prompt = "<|user|>\n"
        generation_prompt = (
            "Generate an image according to the following instructions\n"
        )
        assistant_prompt = "<|assistant|>\n<|diffusion|>"
        prompt_suffix = "<|end|>\n"
        prompt = (
            f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"
        )
        return prompt

    def __call__(
        self,
        instructions: List[str],
        input_images: List[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
    ) -> Dict:

        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction

            mllm_input = self.process_multi_modal_prompt(
                cur_instruction, cur_input_images
            )

            input_data.append((mllm_input, [height, width]))

        return self.collator(input_data)
