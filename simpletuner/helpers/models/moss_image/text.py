"""Unpadded Qwen3.5 text conditioning for MOSS-Image."""

import torch
from transformers import AutoImageProcessor

from simpletuner.helpers.models.prompt_upsampling import resize_reference_images
from simpletuner.helpers.models.text import collate_embeddings

MAX_TEXT_LENGTH = 4096
VISION_PROCESSOR_ID = "Qwen/Qwen3.5-9B"


def load_image_processor(**kwargs):
    return AutoImageProcessor.from_pretrained(
        VISION_PROCESSOR_ID, size={"shortest_edge": 32 * 32, "longest_edge": 384 * 384}, **kwargs
    )


@torch.no_grad()
def encode_prompts(
    tokenizer, text_encoder, prompts, device, max_length=MAX_TEXT_LENGTH, *, reference_images=None, image_processor=None
):
    if not 1 <= max_length <= MAX_TEXT_LENGTH:
        raise ValueError("MOSS-Image tokenizer length must be between 1 and 4096.")
    entries = []
    if reference_images is not None and len(reference_images) != len(prompts):
        raise ValueError("Reference image sets must match the prompt batch size.")
    for index, prompt in enumerate(prompts):
        images = reference_images[index] if reference_images is not None else ()
        vision_inputs = {}
        if images:
            if image_processor is None:
                raise ValueError("MOSS-Image reference text encoding requires a Qwen3.5 image processor.")
            images = [resize_reference_images([image], max_total_pixels=384 * 384)[0] for image in images]
            processed = image_processor(images=images, return_tensors="pt")
            counts = (processed.image_grid_thw.prod(-1) // image_processor.merge_size**2).tolist()
            prefix = "".join(
                f"Picture {number}: <|vision_start|>" + "<|image_pad|>" * count + "<|vision_end|>"
                for number, count in enumerate(counts, start=1)
            )
            prompt = prefix + prompt
            vision_inputs = {
                "pixel_values": processed.pixel_values.to(device=device, dtype=text_encoder.dtype),
                "image_grid_thw": processed.image_grid_thw.to(device),
            }
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=not bool(images), max_length=max_length).to(device)
        if images:
            if inputs.input_ids.shape[1] > max_length:
                raise ValueError(
                    "MOSS-Image reference images and caption exceed tokenizer_max_length; shorten the caption or use fewer references."
                )
            token_types = (inputs.input_ids == text_encoder.config.image_token_id).int()
            if int(token_types.sum()) != sum(counts):
                raise ValueError("MOSS-Image vision placeholders must match the processed reference image grids.")
            vision_inputs["mm_token_type_ids"] = token_types
        output = text_encoder(**inputs, **vision_inputs, use_cache=False, return_dict=True)
        entries.append(
            {
                "prompt_embeds": output.last_hidden_state[0],
                "attention_masks": inputs.attention_mask[0].bool(),
            }
        )
    result = collate_embeddings(entries)
    return result["prompt_embeds"], result["attention_masks"]
