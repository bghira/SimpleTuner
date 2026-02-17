"""BboxGenerator: auto-detect objects via Florence-2 and write .bbox sidecar files for the grounding pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Union
from unittest.mock import patch

from simpletuner.helpers.training import image_file_extensions

logger = logging.getLogger(__name__)

# Image-only extensions (exclude video).
_IMAGE_ONLY_EXTENSIONS: set[str] = image_file_extensions - {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
    "flv",
    "wmv",
    "m4v",
    "mpeg",
    "mpg",
    "3gp",
    "ogv",
}


def _fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Workaround for Florence-2 flash_attn import on systems without it."""
    from transformers.dynamic_module_utils import get_imports

    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class BboxGenerator:
    """Run Florence-2 detection and write ``.bbox`` sidecars next to source images.

    When *labels* are provided, uses ``<OPEN_VOCABULARY_DETECTION>`` with the
    labels as a text prompt.  When *labels* are empty, uses ``<CAPTION>`` followed
    by ``<CAPTION_TO_PHRASE_GROUNDING>`` to automatically caption and ground
    entities without any predefined classes.

    Designed to be invoked once during dataset setup (inside
    ``_inject_grounding_configs``), then discarded.  The model is lazy-loaded
    on first use and explicitly freed after :meth:`generate` completes.
    """

    DEFAULT_MODEL = "microsoft/Florence-2-large"

    def __init__(self, config: dict[str, Any], accelerator):
        self.model_name: str = config.get("model", self.DEFAULT_MODEL)
        raw_labels = config.get("labels", [])
        if isinstance(raw_labels, str):
            raw_labels = [part.strip() for part in raw_labels.split(",") if part.strip()]
        self.labels: list[str] = raw_labels
        self.batch_size: int = int(config.get("batch_size", 4))
        self.accelerator = accelerator
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, instance_data_dir: str) -> int:
        """Detect objects in every image under *instance_data_dir* and write ``.bbox`` files.

        Returns the number of ``.bbox`` files written.
        """
        data_dir = Path(instance_data_dir)
        if not data_dir.is_dir():
            raise ValueError(f"instance_data_dir is not a directory: {instance_data_dir}")

        image_paths = self._collect_image_paths(data_dir)
        pending = [p for p in image_paths if not p.with_suffix(".bbox").exists()]

        if not pending:
            logger.info("BboxGenerator: all images already have .bbox files, nothing to do.")
            return 0

        logger.info(
            f"BboxGenerator: {len(pending)} images need .bbox files " f"({len(image_paths) - len(pending)} already done)."
        )

        self._load_model()
        written = 0
        try:
            for batch_start in range(0, len(pending), self.batch_size):
                batch_paths = pending[batch_start : batch_start + self.batch_size]
                batch_results = self._detect_batch(batch_paths)
                for img_path, detections in zip(batch_paths, batch_results):
                    self._write_bbox_file(img_path, detections)
                    written += 1
        finally:
            self._unload_model()

        logger.info(f"BboxGenerator: wrote {written} .bbox files.")
        return written

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _collect_image_paths(self, data_dir: Path) -> list[Path]:
        paths: list[Path] = []
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix.lstrip(".").lower() in _IMAGE_ONLY_EXTENSIONS:
                paths.append(p)
        return paths

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self.accelerator.device
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype)
                .eval()
                .to(device)
            )
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        logger.info(f"BboxGenerator: loaded Florence-2 model {self.model_name!r} on {device}.")

    def _unload_model(self):
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_florence2(self, image, task: str, text_input: str | None = None) -> dict:
        """Run a single Florence-2 inference and return the parsed result."""
        import torch

        prompt = task if text_input is None else task + text_input
        device = self._model.device
        dtype = next(self._model.parameters()).dtype
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))

    def _detect_batch(self, batch_paths: list[Path]) -> list[list[dict]]:
        """Run detection on a batch of images and return normalised XYXY results."""
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in batch_paths]
        results: list[list[dict]] = []

        for img in images:
            if self.labels:
                raw = self._detect_open_vocabulary(img)
            else:
                raw = self._detect_caption_grounding(img)
            detections = self._postprocess(raw, img.width, img.height)
            results.append(detections)

        return results

    def _detect_open_vocabulary(self, image) -> dict:
        """Use ``<OPEN_VOCABULARY_DETECTION>`` with user-provided labels."""
        text_prompt = ", ".join(self.labels)
        result = self._run_florence2(image, "<OPEN_VOCABULARY_DETECTION>", text_prompt)
        return result.get("<OPEN_VOCABULARY_DETECTION>", {"bboxes": [], "labels": []})

    def _detect_caption_grounding(self, image) -> dict:
        """Use ``<CAPTION>`` then ``<CAPTION_TO_PHRASE_GROUNDING>`` for automatic detection."""
        caption_result = self._run_florence2(image, "<CAPTION>")
        caption = caption_result.get("<CAPTION>", "")
        if not caption:
            return {"bboxes": [], "labels": []}
        grounding_result = self._run_florence2(image, "<CAPTION_TO_PHRASE_GROUNDING>", caption)
        return grounding_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []})

    @staticmethod
    def _postprocess(raw: dict, img_w: int, img_h: int) -> list[dict]:
        """Normalise pixel coords to 0-1 XYXY and discard degenerate boxes."""
        bboxes = raw.get("bboxes", [])
        labels = raw.get("labels", [])
        entities: list[dict] = []
        for bbox, label in zip(bboxes, labels):
            if len(bbox) != 4:
                continue
            x1 = bbox[0] / img_w
            y1 = bbox[1] / img_h
            x2 = bbox[2] / img_w
            y2 = bbox[3] / img_h
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x1 >= x2 or y1 >= y2:
                continue
            entities.append({"label": label, "bbox": [x1, y1, x2, y2]})
        return entities

    @staticmethod
    def _write_bbox_file(img_path: Path, detections: list[dict]):
        bbox_path = img_path.with_suffix(".bbox")
        bbox_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
