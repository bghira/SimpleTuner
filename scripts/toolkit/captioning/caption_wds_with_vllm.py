#!/usr/bin/env python3
"""
Multi-GPU, multi-process captioner with true vLLM batching and shard-level tracking.
- Tracks both individual keys and complete shards for efficient restarts
- One Python process per GPU, each with batched generate()
- Better error handling with per-shard isolation
- Supports partial shard recovery

Run:
  python caption_wds_with_vllm.py \
    --gpus 0,1,2,3 \
    --output-dir captioner_output/captions \
    --checkpoint-dir captioner_output/checkpoints \
    --batch-size 8 \
    --coalesce-ms 30 \
    --max-inflight-per-gpu 256
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "64"
import io
import json
import time
import queue
import logging
import signal
import shlex
import sys
import argparse
import traceback
import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional
from multiprocessing import Process, Queue, Event, set_start_method
from PIL import Image

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from tqdm import tqdm

# optional JXL load support
try:
    import pillow_jxl  # noqa: F401
except ModuleNotFoundError:
    pass

# ---------------- Configuration ----------------
MAX_MODEL_LEN = 16384
LIMIT_IMAGES_PER_PROMPT = 1

LOG = logging.getLogger("caption-mproc")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------- Checkpoint Manager ----------------
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "checkpoint.json"
        self.shard_path = self.dir / "shards.json"

    def save(
        self, processed: set, stats: dict, completed_shards: set, partial_shards: dict
    ):
        """Save both key-level and shard-level progress"""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "processed_keys": list(processed),
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

        # Save shard progress
        shard_tmp = self.shard_path.with_suffix(".tmp")
        shard_tmp.write_text(
            json.dumps(
                {
                    "completed_shards": list(completed_shards),
                    "partial_shards": partial_shards,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        shard_tmp.replace(self.shard_path)

    def load(self) -> Tuple[set, dict, set, dict]:
        """Load both key and shard progress"""
        processed = set()
        stats = {"processed": 0, "errors": 0, "skipped": 0}
        completed_shards = set()
        partial_shards = {}

        if self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            processed = set(d.get("processed_keys", []))
            stats = d.get("stats", {"processed": 0, "errors": 0, "skipped": 0})

        if self.shard_path.exists():
            d = json.loads(self.shard_path.read_text(encoding="utf-8"))
            completed_shards = set(d.get("completed_shards", []))
            partial_shards = d.get("partial_shards", {})

        return processed, stats, completed_shards, partial_shards


# ---------------- Caption Utils ----------------
class CaptionUtils:
    @staticmethod
    def clean_caption(c: str) -> str:
        if not c:
            return ""
        generic = [
            "in this image we can see ",
            "this image shows ",
            "the image depicts ",
            "the image features ",
            "this is an image of ",
            "the image contains ",
            "the picture shows ",
            "we can see ",
            "there is ",
            "there are ",
        ]
        low = c.lower()
        for p in generic:
            if low.startswith(p):
                c = c[len(p) :]
                if c:
                    c = c[0].upper() + c[1:]
                break
        if c.lower().startswith(("a ", "an ")):
            parts = c.split(maxsplit=1)
            if len(parts) > 1 and not parts[1][0].isupper():
                c = parts[1]
                c = c[0].upper() + c[1:]
        c = " ".join(c.split())
        if c and c[-1] not in ".!?":
            c += "."
        return c

    @classmethod
    def combine(cls, descs: List[str]) -> str:
        if not descs:
            return ""
        filtered = []
        heads = [
            "in this image we can see",
            "this image shows",
            "the image depicts",
            "a cartoon",
            "a drawing",
            "an illustration",
        ]
        for d in descs:
            if not d:
                continue
            dl = d.lower().strip()
            if any(dl.startswith(h) and len(dl.split()) < 8 for h in heads):
                continue
            if len(d) > 10:
                filtered.append(d)
        if not filtered:
            filtered = [max(descs, key=len, default="")]
        main = cls.clean_caption(max(filtered, key=len))
        parts = [main]
        seen = set(main.lower().split())

        buckets = {
            "characters": [
                "character",
                "person",
                "animal",
                "anthro",
                "wearing",
                "dressed",
            ],
            "actions": ["doing", "action", "playing", "running", "sitting", "standing"],
            "settings": [
                "room",
                "outdoor",
                "indoor",
                "setting",
                "background",
                "environment",
            ],
            "styles": ["style", "art", "drawn", "sketch", "painted", "digital"],
            "moods": [
                "mood",
                "emotion",
                "feeling",
                "atmosphere",
                "happy",
                "sad",
                "angry",
            ],
        }

        def buck(t: str) -> str:
            tl = t.lower()
            for k, v in buckets.items():
                if any(w in tl for w in v):
                    return k
            return "details"

        byb: Dict[str, List[str]] = {}
        for d in filtered:
            byb.setdefault(buck(d), []).append(d)

        for k in ["characters", "actions", "settings", "moods", "styles", "details"]:
            if k in byb and byb[k]:
                d = byb[k][0]
                words = d.lower().split()
                if len([w for w in words if w not in seen and len(w) > 3]) > 3:
                    clean = cls.clean_caption(d)
                    if clean and clean not in parts:
                        parts.append(clean)
                        seen.update(words)

        # Return each part as a separate line
        return "\n".join(parts)


# ---------------- GPU Worker Process ----------------
def gpu_worker(
    gpu_id: int,
    in_q: Queue,
    out_q: Queue,
    stop_ev: Event,
    precision: str,
    batch_size: int,
    coalesce_ms: int,
    pretrained_model_name_or_path: str,
    max_retries: int = 3,  # Add max retries parameter
):
    # Isolate the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from vllm import LLM, SamplingParams
    from PIL import Image
    from transformers import AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_fast=True
    )
    # optional: match qwen’s recommended pixel budget
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    LOG.info(
        f"[gpu {gpu_id}] Loading model {pretrained_model_name_or_path} ({precision})"
    )
    dtype = "float16" if precision == "fp16" else "bfloat16"

    # Configure based on precision
    if precision == "awq":
        llm = LLM(
            model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            quantization="awq",
            gpu_memory_utilization=0.95,
            dtype="float16",
            limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
            disable_mm_preprocessor_cache=True,
        )
    elif precision == "fp8":
        llm = LLM(
            model=pretrained_model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.92,
            dtype="bfloat16",
            quantization="fp8",
            limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
            disable_mm_preprocessor_cache=True,
        )
    else:
        llm = LLM(
            model=pretrained_model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=512,
            max_num_batched_tokens=MAX_MODEL_LEN,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.92,
            dtype=dtype,
            limit_mm_per_prompt={"image": LIMIT_IMAGES_PER_PROMPT},
            disable_mm_preprocessor_cache=True,
        )

    # Different prompt variations for retries
    prompt_sets = [
        [
            "describe in detail without speculating. don't write anything except the caption.",
        ],
        [
            "provide a detailed description of the visual content in this image.",
        ],
        [
            "what is shown in this image? provide a comprehensive description.",
        ],
    ]

    def get_sampling_params(temperature=0.7):
        return SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=256,
            stop=[
                "<|end|>",
                "<|endoftext|>",
                "<|im_end|>",
                "<|end_of_text|>",
                "|assistant|",
                "<|assistant_end|>",
            ],
            stop_token_ids=[151643, 151645],
            repetition_penalty=1.05,
            frequency_penalty=0.1,
            skip_special_tokens=True,
        )

    def build_inputs(pil_img: Image.Image, question: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # if you want qwen’s video/image normalization path:
        image_inputs, _ = process_vision_info(messages)  # returns processed images
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        return {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {"image": image_inputs},  # list of PILs/arrays
        }

    def make_req(img: Image.Image, q: str) -> Dict[str, Any]:
        return {
            "prompt": f"<|user|>\n<|image_pad|>\n{q}<|end|>\n<|assistant|>",
            "multi_modal_data": {"image": [img]},
        }

    def clean_output(text: str) -> str:
        """Clean any remaining tokens that slipped through"""
        if not text:
            return ""

        end_tokens = [
            "<|end|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|assistant>",
            "|",
            ".<",
            "</",
            "I'm sorry",
            "The content is inappropriate",
            "I apologize",
            "The image does not",
            "If you",
        ]
        cleaned = text
        for token in end_tokens:
            if token in cleaned:
                cleaned = cleaned.split(token)[0]

        if "I'm sorry" in cleaned or "I cannot" in cleaned:
            parts = cleaned.split("I'm sorry")[0].split("I cannot")[0]
            if len(parts.strip()) > 50:
                cleaned = parts

        return cleaned.strip()

    def is_refusal(text: str) -> bool:
        """Check if the output is likely a refusal or too short"""
        if not text or len(text) < 20:
            return True

        refusal_patterns = [
            "i'm sorry",
            "i cannot",
            "i apologize",
            "inappropriate",
            "i can't",
            "unable to",
            "not able to",
            "refuse to",
            "digital artwork",
            "stylized",  # Also catch generic fallbacks
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in refusal_patterns)

    buf: List[Tuple[str, str, bytes]] = []
    last = time.monotonic()

    def flush():
        nonlocal buf, last
        if not buf:
            return

        LOG.info(f"[gpu {gpu_id}] flushing batch of {len(buf)} images")

        retry_items = []
        completed_items = []

        # run up to max_retries; each retry uses a different prompt set & temp
        for retry_attempt in range(max_retries):
            items_to_process = buf if retry_attempt == 0 else retry_items
            retry_items = []  # reset for the next loop

            if not items_to_process:
                break

            if retry_attempt > 0:
                LOG.info(
                    f"[gpu {gpu_id}] retry {retry_attempt} for {len(items_to_process)} images"
                )

            # choose prompt variants for this round
            prompts = prompt_sets[min(retry_attempt, len(prompt_sets) - 1)]

            # build a single batched call to vllm
            reqs = []
            index = []

            for k, u, jb in items_to_process:
                try:
                    im = Image.open(io.BytesIO(jb)).convert("RGB")
                    # keep image work cheap + consistent
                    im.thumbnail((512, 512), Image.BILINEAR)

                    # two prompts per image to improve robustness
                    for q in prompts:
                        reqs.append(build_inputs(im, q))

                    index.append((k, u, jb))
                except Exception as e:
                    LOG.error(f"[gpu {gpu_id}] failed to decode image {k}: {e}")
                    out_q.put(("err", k, u, f"decode-failed: {e}"))

            if not reqs:
                continue

            # slightly raise temperature per attempt
            temperature = 0.7 + (0.1 * retry_attempt)
            sampling = get_sampling_params(temperature)

            try:
                outs = llm.generate(reqs, sampling)
            except Exception as e:
                LOG.error(f"[gpu {gpu_id}] batch generation failed: {e}")
                # mark all as errored for this attempt (we won't retry the whole batch again here)
                for k, u, _ in items_to_process:
                    out_q.put(("err", k, u, f"batch-failed: {e}"))
                continue

            # collect texts in order; vllm returns one result per req
            texts = [
                (clean_output(o.outputs[0].text) if o.outputs else "") for o in outs
            ]

            # map back: 2 requests per image
            for i, (k, u, jb) in enumerate(index):
                d1 = texts[2 * i] if 2 * i < len(texts) else ""
                d2 = texts[2 * i + 1] if (2 * i + 1) < len(texts) else ""

                # if both look like refusals/too-short, consider retrying
                if is_refusal(d1) and is_refusal(d2):
                    if retry_attempt < max_retries - 1:
                        retry_items.append((k, u, jb))
                        LOG.debug(
                            f"[gpu {gpu_id}] {k} queued for retry (attempt {retry_attempt + 1})"
                        )
                    else:
                        out_q.put(("err", k, u, "all attempts refused/too-short"))
                    continue

                # pick valid ones, combine for richness, then final clean
                valids = [t for t in (d1, d2) if not is_refusal(t)]
                combined = (
                    CaptionUtils.combine(valids)
                    if valids
                    else max([d1, d2], key=len, default="")
                )
                cap = CaptionUtils.clean_caption(combined)

                # if still junk and we have retries left, retry; else emit
                if is_refusal(cap) and retry_attempt < max_retries - 1:
                    retry_items.append((k, u, jb))
                else:
                    out_q.put(("ok", k, u, cap))
                    completed_items.append(k)

        LOG.info(
            f"[gpu {gpu_id}] batch complete: {len(completed_items)} ok, "
            f"{len(retry_items)} failed all retries"
        )

        buf = []
        last = time.monotonic()

    LOG.info(f"[gpu {gpu_id}] Ready, entering work loop")

    while not stop_ev.is_set():
        now = time.monotonic()
        timeout = max(0.0, (coalesce_ms / 1000.0) - (now - last))

        try:
            item = in_q.get(timeout=timeout if buf else 0.1)
            if item == "STOP":
                break
            buf.append(item)
            if len(buf) >= batch_size:
                flush()
        except queue.Empty:
            if buf and (time.monotonic() - last) * 1000.0 >= coalesce_ms:
                flush()

    # Final flush
    try:
        flush()
    except Exception:
        pass
    LOG.info(f"[gpu {gpu_id}] Exiting")


# ---------------- Dataset Functions ----------------
def create_shard_list(dataset_repo_path: str) -> List[str]:
    """Get list of all shard URLs"""
    LOG.info("Getting shard list from HuggingFace...")
    token = get_token()
    if not token:
        raise ValueError("No Hugging Face token found; run `huggingface-cli login`")

    fs = HfFileSystem()
    files = [
        fs.resolve_path(p)
        for p in fs.glob(f"hf://datasets/{dataset_repo_path}/**/*.tar")
    ]
    urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in files]
    LOG.info(f"Found {len(urls)} shards")
    return sorted(urls)  # Sort for consistent ordering


def process_shard(shard_url: str, token: str, processed_keys: set) -> wds.DataPipeline:
    """Create dataset for a single shard with proper error handling"""
    # Use individual curl command for better error handling
    url_cmd = f"pipe:curl -s -L -H 'Authorization:Bearer {shlex.quote(token)}' {shlex.quote(shard_url)} || true"

    ds = wds.DataPipeline(
        wds.SimpleShardList(url_cmd),
        wds.tarfile_to_samples(),
        wds.to_tuple("__key__", "__url__", "jpg;png;jpeg;webp"),
        # Filter out already processed keys at the WebDataset level
        wds.select(lambda x: x[0] not in processed_keys),
    )
    return ds


def _drain_outputs(
    outq: Queue,
    outdir: str,
    processed: set,
    stats: dict,
    pbar: Optional[tqdm],
    drain_all: bool = False,
):
    """Write finished captions; non-blocking unless drain_all."""
    count = 0
    while True:
        try:
            item = outq.get(timeout=0.0 if not drain_all else 0.5)
        except queue.Empty:
            break

        status, key, url, payload = item
        try:
            if status == "ok" and payload:
                shard = Path(url).stem
                d = Path(outdir) / shard
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{key}.txt").write_text(payload, encoding="utf-8")
                processed.add(key)
                stats["processed"] = stats.get("processed", 0) + 1
            else:
                stats["errors"] = stats.get("errors", 0) + 1
        except Exception as e:
            LOG.error(f"Write failed for {key}: {e}")
            stats["errors"] = stats.get("errors", 0) + 1
        finally:
            if pbar:
                pbar.update(1)
            count += 1

    if count > 0:
        LOG.debug(f"Drained {count} outputs")


# ---------------- Main Orchestration ----------------
def main():
    set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU IDs; default = all visible",
    )
    ap.add_argument(
        "--precision", choices=["fp16", "bf16", "fp8", "awq"], default="fp16"
    )
    ap.add_argument(
        "--batch-size", type=int, default=8, help="Target batch images per GPU flush"
    )
    ap.add_argument(
        "--coalesce-ms",
        type=int,
        default=30,
        help="Microbatch coalesce window in milliseconds",
    )
    ap.add_argument(
        "--max-inflight-per-gpu",
        type=int,
        default=256,
        help="Max queue size to avoid OOM",
    )
    ap.add_argument("--output-dir", type=str, default="paligemma_captions")
    ap.add_argument("--checkpoint-dir", type=str, default="paligemma_checkpoints")
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=200,
        help="Save checkpoint every N items",
    )
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=False,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    args = ap.parse_args()

    # Pick GPUs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception:
            n = int(os.environ.get("CUDA_DEVICE_COUNT", "0") or "0")
        gpu_ids = list(range(n))

    if not gpu_ids:
        raise RuntimeError("No GPUs found or specified")

    LOG.info(f"Using GPUs: {gpu_ids}")

    # Setup directories and checkpointing
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(args.checkpoint_dir)
    processed, stats, completed_shards, partial_shards = ckpt.load()
    LOG.info(
        f"Resuming with {len(processed)} processed keys, "
        f"{len(completed_shards)} completed shards"
    )

    # Signal handlers for clean shutdown
    shutdown_flag = Event()

    def _sig_handler(signum, frame):
        LOG.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_flag.set()
        ckpt.save(processed, stats, completed_shards, partial_shards)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Spawn worker processes
    inqs: Dict[int, Queue] = {}
    outq: Queue = Queue()
    stop = Event()
    procs: List[Process] = []

    for gid in gpu_ids:
        iq = Queue(maxsize=args.max_inflight_per_gpu)
        inqs[gid] = iq
        p = Process(
            target=gpu_worker,
            args=(
                gid,
                iq,
                outq,
                stop,
                args.precision,
                args.batch_size,
                args.coalesce_ms,
                args.pretrained_model_name_or_path,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)
        LOG.info(f"Spawned worker on GPU {gid} (pid={p.pid})")

    # Get shard list and filter completed ones
    token = get_token()
    all_shards = create_shard_list(dataset_repo_path=args.dataset)
    remaining_shards = [s for s in all_shards if Path(s).stem not in completed_shards]

    LOG.info(
        f"Total shards: {len(all_shards)}, Completed: {len(completed_shards)}, "
        f"Remaining: {len(remaining_shards)}"
    )

    # Process each shard
    for shard_idx, shard_url in enumerate(remaining_shards):
        if shutdown_flag.is_set():
            break

        shard_name = Path(shard_url).stem
        shard_processed = set()
        shard_errors = 0

        # Check if we have partial progress for this shard
        if shard_name in partial_shards:
            shard_keys = set(partial_shards[shard_name].get("keys", []))
            processed.update(shard_keys)
            LOG.info(f"Resuming shard {shard_name} with {len(shard_keys)} already done")

        try:
            LOG.info(
                f"Processing shard {shard_name} ({shard_idx+1}/{len(remaining_shards)})"
            )

            # Create dataset for this specific shard
            ds = process_shard(shard_url, token, processed)

            # Count items for progress bar (optional, can be slow)
            shard_items = []
            try:
                for item in ds:
                    shard_items.append(item)
            except Exception as e:
                LOG.error(f"Failed to load shard {shard_name}: {e}")
                continue

            if not shard_items:
                LOG.info(
                    f"Shard {shard_name} is empty or fully processed, marking complete"
                )
                completed_shards.add(shard_name)
                if shard_name in partial_shards:
                    del partial_shards[shard_name]
                ckpt.save(processed, stats, completed_shards, partial_shards)
                continue

            LOG.info(f"Shard {shard_name} has {len(shard_items)} items to process")

            # Process samples from this shard
            rr = 0
            with tqdm(
                total=len(shard_items),
                desc=f"Shard {shard_name}",
                position=1,
                leave=False,
            ) as pbar:
                for key, url, image_data in shard_items:
                    if shutdown_flag.is_set():
                        break

                    try:
                        # Convert image data to PIL if needed
                        if isinstance(image_data, bytes):
                            image = Image.open(io.BytesIO(image_data)).convert("RGB")
                        elif isinstance(image_data, Image.Image):
                            try:
                                image = Image.open(io.BytesIO(image_data)).convert(
                                    "RGB"
                                )
                            except Exception as e:
                                LOG.warning(
                                    f"Failed to open/convert image bytes for {key}: {e}"
                                )
                                shard_errors += 1
                                stats["errors"] = stats.get("errors", 0) + 1
                                pbar.update(1)
                                continue
                        elif isinstance(image_data, Image.Image):
                            try:
                                image = image_data.convert("RGB")
                            except Exception as e:
                                LOG.warning(
                                    f"Failed to convert PIL Image for {key}: {e}"
                                )
                                shard_errors += 1
                                stats["errors"] = stats.get("errors", 0) + 1
                                pbar.update(1)
                                continue
                        else:
                            LOG.warning(
                                f"Unexpected image type for {key}: {type(image_data)}"
                            )
                            shard_errors += 1
                            stats["errors"] = stats.get("errors", 0) + 1
                            pbar.update(1)
                            continue

                        # Compress to JPEG
                        buf = io.BytesIO()
                        image.save(buf, format="JPEG", quality=92, optimize=True)
                        jpeg_bytes = buf.getvalue()

                        # Round-robin to GPUs
                        gid = gpu_ids[rr]
                        rr = (rr + 1) % len(gpu_ids)

                        # Send to worker with backpressure handling
                        while not shutdown_flag.is_set():
                            try:
                                inqs[gid].put((key, url, jpeg_bytes), timeout=0.1)
                                break
                            except queue.Full:
                                # Drain outputs while waiting
                                _drain_outputs(
                                    outq, args.output_dir, processed, stats, pbar
                                )

                        shard_processed.add(key)

                        # Periodic checkpoint
                        if len(shard_processed) % args.checkpoint_interval == 0:
                            partial_shards[shard_name] = {"keys": list(shard_processed)}
                            ckpt.save(
                                processed, stats, completed_shards, partial_shards
                            )
                            _drain_outputs(
                                outq, args.output_dir, processed, stats, pbar
                            )

                    except Exception as e:
                        LOG.error(f"Error processing {key}: {e}")
                        shard_errors += 1
                        stats["errors"] = stats.get("errors", 0) + 1
                        pbar.update(1)

            # Wait for shard to complete
            LOG.info(f"Waiting for GPU workers to finish shard {shard_name}...")
            time.sleep(2)  # Allow workers to finish
            _drain_outputs(
                outq, args.output_dir, processed, stats, None, drain_all=True
            )

            # Mark shard as complete
            completed_shards.add(shard_name)
            if shard_name in partial_shards:
                del partial_shards[shard_name]

            LOG.info(
                f"Completed shard {shard_name}: {len(shard_processed)} processed, "
                f"{shard_errors} errors"
            )
            ckpt.save(processed, stats, completed_shards, partial_shards)

        except Exception as e:
            LOG.error(f"Failed to process shard {shard_name}: {e}")
            traceback.print_exc()
            # Save partial progress
            if shard_processed:
                partial_shards[shard_name] = {"keys": list(shard_processed)}
                ckpt.save(processed, stats, completed_shards, partial_shards)

    # Final drain
    LOG.info("All shards queued, waiting for final outputs...")
    time.sleep(5)
    _drain_outputs(outq, args.output_dir, processed, stats, None, drain_all=True)

    # Shutdown workers
    LOG.info("Shutting down GPU workers...")
    stop.set()
    for gid, q in inqs.items():
        try:
            q.put("STOP", timeout=0.1)
        except Exception:
            pass

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            LOG.warning(f"Worker {p.pid} did not exit cleanly, terminating...")
            p.terminate()
            p.join(timeout=5)

    # Final save
    ckpt.save(processed, stats, completed_shards, partial_shards)

    # Print summary
    LOG.info("=" * 60)
    LOG.info(f"COMPLETE: GPUs={gpu_ids}")
    LOG.info(f"Processed: {stats.get('processed', 0)}")
    LOG.info(f"Errors: {stats.get('errors', 0)}")
    LOG.info(f"Completed shards: {len(completed_shards)}/{len(all_shards)}")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
