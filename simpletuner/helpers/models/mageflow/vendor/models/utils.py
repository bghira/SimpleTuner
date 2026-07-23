import json
import math
import os

import torch
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from safetensors.torch import load_file as load_sft
from torch import Tensor

from .mage_flow import MageFlow, MageFlowParams


def get_noise(
    num_samples: int,
    channel: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    # MageVAE: 16x downsample, no patch packing
    return torch.randn(
        num_samples,
        channel,
        math.ceil(height / 16),
        math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    # MageVAE: [B, H*W, C] -> [B, C, H, W], no patch unpacking
    return rearrange(
        x,
        "b (h w) c -> b c h w",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
    )


PROMPT_TEMPLATE = {
    "default": {"template": "{}", "start_idx": 0},
    "default-nonthinking": {"template": "{}<think>\n\n</think>\n\n", "start_idx": 0},
    "mage-flow": {
        "template": (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "start_idx": 34,
    },
    "mage-flow-edit": {
        "template": (
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
            " objects, background), then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the original "
            "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "start_idx": 64,
    },
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        logger.warning("\n" + "-" * 79 + "\n")
        logger.warning(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        logger.warning(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def correct_model_weight(state_dict):
    result = {}
    for key in state_dict.keys():
        if "_orig_mod." in key:
            result[key[10:]] = state_dict[key]
        else:
            result[key] = state_dict[key]
    return result


def load_hf_style_weight(pretrain_path, device):
    index_path = os.path.join(pretrain_path, "diffusion_pytorch_model.safetensors.index.json")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    sd = {}
    loaded_shards = set()

    for shard_file in weight_map.values():
        if shard_file in loaded_shards:
            continue
        shard_path = os.path.join(pretrain_path, shard_file)
        shard_sd = load_file(shard_path, device="cpu")
        sd.update(shard_sd)
        loaded_shards.add(shard_file)

    return sd


def load_model_weight(model, pretrain_path, device="cpu"):
    if os.path.exists(pretrain_path):
        logger.info(f"Loading checkpoint from {pretrain_path}")
        try:
            if pretrain_path.endswith("safetensors"):
                sd = load_sft(pretrain_path, device="cpu")
            elif os.path.exists(os.path.join(pretrain_path, "diffusion_pytorch_model.safetensors.index.json")):
                sd = load_hf_style_weight(pretrain_path, device)
            else:
                sd = torch.load(pretrain_path, map_location="cpu")

            sd = correct_model_weight(sd)
            sd = optionally_expand_state_dict(model, sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            print_load_warning(missing, unexpected)
            return True
        except Exception as e:
            logger.info(f"CANNOT Load {pretrain_path}, because {e}")
            return False
    return False


def load_model(dit_structure: dict, pretrain_path: str | None = None):
    logger.info("Init DiT model")

    # If name is a dict, we assume it contains the parameters directly
    # We need to determine the model class based on some heuristic or just default to MageFlow/Flux
    # For now, let's assume it's MageFlow if time_type is present, or check other fields
    params = MageFlowParams(**dit_structure)
    # Default to MageFlow for now as per user context, or we could add a 'model_type' field to the dict
    # The user mentioned "model structure option", implying we are configuring the structure.
    # Let's assume MageFlow for this refactor as the user was using qwen-image-tiny-wo-textemb
    model = MageFlow(params)

    # logger.info(f"Loading {name if isinstance(name, str) else 'custom config'} checkpoint from {pretrain_path}")
    if pretrain_path is not None:
        load_model_weight(model, pretrain_path, device="cpu")
    # if isinstance(name, str) and configs[name].lora_path is not None:
    #     logger.info("Loading LoRA")
    #     lora_sd = load_sft(configs[name].lora_path, device="cpu")
    #     # loading the lora params + overwriting scale values in the norms
    #     missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
    #     print_load_warning(missing, unexpected)
    return model


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                logger.info(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape "
                    f"{param.shape}."
                )
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict
