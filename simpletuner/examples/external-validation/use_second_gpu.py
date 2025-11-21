#!/usr/bin/env python3
"""
Example post-upload/validation hook that runs a Flux LoRA on a second GPU (cuda:1 by default).

Intended invocation with SimpleTuner placeholders:
  --post_upload_script="simpletuner/examples/external-validation/use_second_gpu.py \
    --local {local_checkpoint_path} \
    --hub_model_id {huggingface_path} \
    --model_family {model_family} \
    --prompt 'your prompt here'"

Notes:
- Requires `pip install diffusers torch`.
- Skips execution when model_family is not Flux to avoid accidental mismatches.
- If cuda:1 is unavailable, falls back to cuda:0, then MPS, then CPU.
- Treats `--hub_model_id` as the LoRA weights path. If omitted, runs the base model only.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch
from diffusers import DiffusionPipeline

logger = logging.getLogger("use_second_gpu")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _pick_device(preferred_index: int) -> torch.device:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        target = preferred_index if preferred_index < device_count else 0
        return torch.device(f"cuda:{target}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Run Flux LoRA inference on a secondary GPU.")
    parser.add_argument("--local", "--local_checkpoint_path", dest="local", default="", help="Local checkpoint path (optional)")
    parser.add_argument("--hub_model_id", default="", help="Hugging Face repo ID for LoRA weights (huggingface_path placeholder)")
    parser.add_argument("--model_family", default="", help="SimpleTuner model_family placeholder")
    parser.add_argument("--prompt", default="julie, in photograph style", help="Prompt to generate")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev", help="Base model ID to load")
    parser.add_argument("--device_index", type=int, default=1, help="Preferred CUDA device index (defaults to cuda:1)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1641421826)
    parser.add_argument("--output", default="external_validation.png", help="Where to save the generated image")
    args = parser.parse_args()

    normalized_family = (args.model_family or "").strip().lower()
    if "flux" not in normalized_family:
        logger.info("Skipping hook because model_family is not flux (got %s).", args.model_family)
        return

    device = _pick_device(args.device_index)
    logger.info("Using device: %s", device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)

    lora_path: Optional[str] = args.hub_model_id or args.local or None
    if lora_path:
        logger.info("Loading LoRA weights from %s", lora_path)
        if args.lora_type.lower() == "lycoris":
            try:
                from lycoris import create_lycoris_from_weights
            except ImportError:
                logger.error("lycoris is not installed; cannot load LyCORIS adapter.")
                sys.exit(1)
            wrapper, _ = create_lycoris_from_weights(1.0, lora_path, pipe.transformer)
            wrapper.merge_to()
        else:
            pipe.load_lora_weights(lora_path)

    pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    logger.info("Running inference...")
    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
    )
    image = result.images[0]
    image.save(args.output, format="PNG")
    logger.info("Saved image to %s", args.output)


if __name__ == "__main__":
    main()
