#!/usr/bin/env python3
"""
Example post-upload hook for SimpleTuner that triggers a WaveSpeed inference
after publishing completes. Intended to be invoked via:

  --post_upload_script="simpletuner/examples/external-validation/wavespeed_post_upload.py \
    --remote {remote_checkpoint_path} \
    --model_family {model_family} \
    --model_type {model_type} \
    --lora_type {lora_type} \
    --hub_model_id {huggingface_path}"

Notes:
- Requires `pip install httpx` and `WAVESPEED_API_KEY` set in the environment.
- This script polls WaveSpeed for completion and prints the resulting URL(s).
- SimpleTuner does not ingest results—emit tracker updates yourself.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("wavespeed_post_upload")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_lora_path(hub_model_id: Optional[str], remote_checkpoint_path: Optional[str]) -> str:
    if hub_model_id:
        return hub_model_id
    if remote_checkpoint_path:
        return remote_checkpoint_path
    return ""


def _request_payload(prompt: str, lora_path: str) -> Dict[str, Any]:
    return {
        "enable_base64_output": False,
        "guidance_scale": 3.5,
        "image": "",
        "loras": [{"path": lora_path, "scale": 1}] if lora_path else [],
        "num_images": 1,
        "num_inference_steps": 28,
        "output_format": "jpeg",
        "prompt": prompt,
        "seed": -1,
        "size": "1024*1024",
        "strength": 0.8,
    }


def submit_job(api_key: str, model_id: str, payload: Dict[str, Any]) -> str:
    url = f"https://api.wavespeed.ai/api/v3/{model_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = httpx.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    request_id = data.get("requestId")
    if not request_id:
        raise RuntimeError(f"WaveSpeed response missing requestId: {data}")
    return request_id


def poll_result(api_key: str, request_id: str, timeout: int = 120, interval: float = 2.0) -> Optional[str]:
    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(url, headers=headers)
        if resp.status_code == 404:
            time.sleep(interval)
            continue
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").lower()
        if status in {"succeeded", "completed", "finished"}:
            outputs = data.get("output") or data.get("result") or []
            if isinstance(outputs, list) and outputs:
                return str(outputs[0])
            return None
        if status in {"failed", "error", "canceled"}:
            raise RuntimeError(f"WaveSpeed job failed: {data}")
        time.sleep(interval)
    raise TimeoutError(f"WaveSpeed job {request_id} did not complete within {timeout}s.")


def main():
    parser = argparse.ArgumentParser(description="Trigger WaveSpeed inference after a SimpleTuner upload.")
    parser.add_argument(
        "--remote", "--remote_checkpoint_path", dest="remote", default="", help="Remote URI returned by publisher"
    )
    parser.add_argument("--model_family", default="", help="SimpleTuner model_family placeholder")
    parser.add_argument("--model_type", default="", help="SimpleTuner model_type placeholder")
    parser.add_argument("--lora_type", default="", help="SimpleTuner lora_type placeholder")
    parser.add_argument("--hub_model_id", default="", help="Hugging Face repo ID (huggingface_path placeholder)")
    parser.add_argument(
        "--prompt",
        default="Super Realism, High-resolution photograph, woman, UHD, photorealistic, shot on a Sony A7III --chaos 20 --ar 1:2 --style raw --stylize 250",
        help="Prompt to run on WaveSpeed",
    )
    parser.add_argument(
        "--model_id", default="wavespeed-ai/flux-dev-lora-ultra-fast", help="WaveSpeed model endpoint to call"
    )
    parser.add_argument("--timeout", type=int, default=120, help="Polling timeout in seconds")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()

    api_key = os.environ.get("WAVESPEED_API_KEY")
    if not api_key:
        logger.error("Missing WAVESPEED_API_KEY environment variable.")
        sys.exit(1)

    lora_path = _build_lora_path(args.hub_model_id, args.remote)
    payload = _request_payload(prompt=args.prompt, lora_path=lora_path)

    try:
        request_id = submit_job(api_key, args.model_id, payload)
        logger.info("WaveSpeed request submitted: %s", request_id)
        result_url = poll_result(api_key, request_id, timeout=args.timeout, interval=args.interval)
    except Exception as exc:  # pragma: no cover - network/external service
        logger.error("WaveSpeed call failed: %s", exc)
        sys.exit(1)

    if result_url:
        logger.info("WaveSpeed output URL: %s", result_url)
    else:
        logger.info("WaveSpeed job completed with no output URL.")


if __name__ == "__main__":
    main()
