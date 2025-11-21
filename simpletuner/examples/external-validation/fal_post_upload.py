#!/usr/bin/env python3
"""
Example post-upload hook for SimpleTuner that triggers a fal.ai Flux LoRA inference
after publishing completes. Intended to be invoked via:

  --post_upload_script="simpletuner/examples/external-validation/fal_post_upload.py \
    --remote {remote_checkpoint_path} \
    --model_family {model_family} \
    --model_type {model_type} \
    --lora_type {lora_type} \
    --hub_model_id {huggingface_path}"

Notes:
- Requires `pip install httpx` and `FAL_KEY` set in the environment.
- Targets the Flux LoRA endpoint (`fal-ai/flux-lora`) and skips execution for non-Flux model_family.
- This script polls fal.ai's queue API and prints the resulting image URL(s).
- SimpleTuner does not ingest results—emit tracker updates yourself.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("fal_post_upload")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_lora_path(hub_model_id: Optional[str], remote_checkpoint_path: Optional[str]) -> str:
    if hub_model_id:
        return f"huggingface.co/{hub_model_id}"
    if remote_checkpoint_path:
        return remote_checkpoint_path
    return ""


def _prepare_input(prompt: str, lora_path: str) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "loras": [{"path": lora_path, "scale": 1}] if lora_path else [],
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "output_format": "jpeg",
        "image_size": "landscape_4_3",
        "enable_safety_checker": True,
    }


def submit_job(api_key: str, model_id: str, payload: Dict[str, Any]) -> str:
    url = f"https://queue.fal.run/{model_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Key {api_key}",
    }
    resp = httpx.post(url, headers=headers, json={"input": payload})
    resp.raise_for_status()
    data = resp.json()
    request_id = data.get("request_id") or data.get("requestId")
    if not request_id:
        raise RuntimeError(f"fal.ai response missing request_id: {data}")
    return request_id


def poll_result(api_key: str, model_id: str, request_id: str, timeout: int = 180, interval: float = 2.0) -> Optional[str]:
    status_url = f"https://queue.fal.run/{model_id}/requests/{request_id}"
    result_url = f"{status_url}/result"
    headers = {"Authorization": f"Key {api_key}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(status_url, headers=headers)
        if resp.status_code == 404:
            time.sleep(interval)
            continue
        resp.raise_for_status()
        status_payload = resp.json()
        status = (status_payload.get("status") or "").lower()
        if status in {"succeeded", "completed", "finished"}:
            break
        if status in {"failed", "error", "canceled"}:
            raise RuntimeError(f"fal.ai job failed: {status_payload}")
        time.sleep(interval)
    else:
        raise TimeoutError(f"fal.ai job {request_id} did not complete within {timeout}s.")

    resp = httpx.get(result_url, headers=headers)
    resp.raise_for_status()
    result_payload = resp.json()
    outputs = result_payload.get("images") or result_payload.get("output") or []
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            return first.get("url") or first.get("uri") or str(first)
        return str(first)
    return None


def main():
    parser = argparse.ArgumentParser(description="Trigger fal.ai Flux LoRA inference after a SimpleTuner upload.")
    parser.add_argument(
        "--remote", "--remote_checkpoint_path", dest="remote", default="", help="Remote URI returned by publisher"
    )
    parser.add_argument("--model_family", default="", help="SimpleTuner model_family placeholder")
    parser.add_argument("--model_type", default="", help="SimpleTuner model_type placeholder")
    parser.add_argument("--lora_type", default="", help="SimpleTuner lora_type placeholder")
    parser.add_argument("--hub_model_id", default="", help="Hugging Face repo ID (huggingface_path placeholder)")
    parser.add_argument(
        "--prompt", default="Extreme close-up of a single tiger eye, direct frontal view...", help="Prompt to run on fal.ai"
    )
    parser.add_argument("--model_id", default="fal-ai/flux-lora", help="fal.ai model endpoint to call")
    parser.add_argument("--timeout", type=int, default=180, help="Polling timeout in seconds")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()

    normalized_family = (args.model_family or "").strip().lower()
    if "flux" not in normalized_family:
        logger.info("Skipping fal.ai hook because model_family is not flux (got %s).", args.model_family)
        return

    api_key = os.environ.get("FAL_KEY")
    if not api_key:
        logger.error("Missing FAL_KEY environment variable.")
        sys.exit(1)

    lora_path = _build_lora_path(args.hub_model_id, args.remote)
    payload = _prepare_input(prompt=args.prompt, lora_path=lora_path)

    try:
        request_id = submit_job(api_key, args.model_id, payload)
        logger.info("fal.ai request submitted: %s", request_id)
        result_url = poll_result(api_key, args.model_id, request_id, timeout=args.timeout, interval=args.interval)
    except Exception as exc:  # pragma: no cover - network/external service
        logger.error("fal.ai call failed: %s", exc)
        sys.exit(1)

    if result_url:
        logger.info("fal.ai output URL: %s", result_url)
    else:
        logger.info("fal.ai job completed with no output URL.")


if __name__ == "__main__":
    main()
