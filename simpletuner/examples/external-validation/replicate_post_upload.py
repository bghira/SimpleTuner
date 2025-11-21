#!/usr/bin/env python3
"""
Example post-upload hook for SimpleTuner that triggers a Replicate inference
after publishing completes. Intended to be invoked via:

  --post_upload_script="simpletuner/examples/external-validation/replicate_post_upload.py \
    --remote {remote_checkpoint_path} \
    --model_family {model_family} \
    --model_type {model_type} \
    --lora_type {lora_type} \
    --hub_model_id {huggingface_path}"

Notes:
- Requires `pip install replicate` and `REPLICATE_API_TOKEN` set in the environment.
- This script does not push any results back into SimpleTuner; emit tracker updates yourself.
- Adjust the Replicate model ID and inputs to suit your target model.
"""

import argparse
import logging
import os
import sys
from typing import Optional

try:
    import replicate
except ImportError:
    replicate = None

logger = logging.getLogger("replicate_post_upload")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_lora_weights_uri(hub_model_id: Optional[str], remote_checkpoint_path: Optional[str]) -> str:
    if hub_model_id:
        return f"huggingface.co/{hub_model_id}"
    if remote_checkpoint_path:
        return remote_checkpoint_path
    return ""


def _default_model_for_family(model_family: str) -> str:
    normalized = (model_family or "").strip().lower()
    if normalized.startswith("flux"):
        return "black-forest-labs/flux-dev-lora"
    return "black-forest-labs/flux-dev-lora"  # safe default


def main():
    parser = argparse.ArgumentParser(description="Trigger Replicate inference after a SimpleTuner upload.")
    parser.add_argument(
        "--remote", "--remote_checkpoint_path", dest="remote", default="", help="Remote URI returned by publisher"
    )
    parser.add_argument("--model_family", default="", help="SimpleTuner model_family placeholder")
    parser.add_argument("--model_type", default="", help="SimpleTuner model_type placeholder")
    parser.add_argument("--lora_type", default="", help="SimpleTuner lora_type placeholder")
    parser.add_argument("--hub_model_id", default="", help="Hugging Face repo ID (huggingface_path placeholder)")
    parser.add_argument(
        "--prompt",
        default="a bacon cheeseburger in the style of TOK a trtcrd, tarot style",
        help="Prompt to run on Replicate",
    )
    args = parser.parse_args()

    if replicate is None:
        logger.error("Replicate SDK is not installed. pip install replicate to use this hook.")
        sys.exit(1)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        logger.error("Missing REPLICATE_API_TOKEN environment variable.")
        sys.exit(1)

    lora_weights = _build_lora_weights_uri(args.hub_model_id, args.remote)
    if not lora_weights:
        logger.warning("No hub_model_id or remote checkpoint provided; using model defaults.")

    model_id = _default_model_for_family(args.model_family)
    logger.info("Calling Replicate model %s with LoRA weights %s", model_id, lora_weights or "<none>")

    try:
        output = replicate.run(
            model_id,
            input={
                "prompt": args.prompt,
                "go_fast": True,
                "guidance": 3,
                "lora_scale": 1,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "lora_weights": lora_weights,
                "output_format": "webp",
                "output_quality": 80,
                "prompt_strength": 0.8,
                "num_inference_steps": 28,
            },
        )
    except Exception as exc:  # pragma: no cover - network/& external service
        logger.error("Replicate call failed: %s", exc)
        sys.exit(1)

    if not output:
        logger.warning("Replicate returned no outputs.")
        return

    try:
        url = output[0].url()
    except Exception:
        url = str(output[0])
    logger.info("Replicate output URL: %s", url)


if __name__ == "__main__":
    main()
