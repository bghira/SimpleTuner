"""Transformers-based NSFW image classification helpers."""

from __future__ import annotations

import gc
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_NSFW_SCORE_THRESHOLD = 0.5
DEFAULT_NSFW_CHECK_MODELS = (
    "Falconsai/nsfw_image_detection:threshold=0.5",
    "AdamCodd/vit-base-nsfw-detector:threshold=0.5",
)
DEFAULT_NSFW_CHECK_MODELS_CSV = ",".join(DEFAULT_NSFW_CHECK_MODELS)

NSFW_LABEL_HINTS = ("nsfw", "unsafe", "porn", "hentai", "sexy", "explicit", "adult")
SFW_LABEL_HINTS = ("sfw", "safe", "neutral", "normal", "drawing")


@dataclass(frozen=True)
class NsfwModelSpec:
    model_id: str
    threshold: float = DEFAULT_NSFW_SCORE_THRESHOLD

    @property
    def key(self) -> str:
        return self.model_id


def _normalize_label(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")


def _label_matches(label: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize_label(label)
    parts = {part for part in normalized.split("_") if part}
    for hint in hints:
        if hint in parts:
            return True
        if hint in {"sfw", "safe"}:
            continue
        if hint in normalized:
            return True
    return False


def score_sum(scores: list[dict[str, Any]], hints: tuple[str, ...]) -> float | None:
    matched = [float(item["score"]) for item in scores if _label_matches(str(item["label"]), hints)]
    if not matched:
        return None
    return float(sum(matched))


def _parse_threshold(model_id: str, threshold_text: str) -> NsfwModelSpec:
    try:
        threshold = float(threshold_text)
    except ValueError as exc:
        raise ValueError(f"Invalid NSFW threshold for model {model_id!r}: {threshold_text!r}") from exc
    if threshold < 0 or threshold > 1:
        raise ValueError(f"NSFW threshold for model {model_id!r} must be between 0 and 1.")
    return NsfwModelSpec(model_id=model_id, threshold=threshold)


def parse_nsfw_model_specs(raw_models: str | Iterable[str] | None) -> list[NsfwModelSpec]:
    if raw_models is None:
        return []
    if isinstance(raw_models, str):
        raw_items = raw_models.split(",")
    else:
        raw_items = list(raw_models)

    specs: list[NsfwModelSpec] = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue

        match = re.fullmatch(r"(?P<model_id>.+):threshold=(?P<threshold>[0-9]*\.?[0-9]+)", item)
        if match:
            specs.append(_parse_threshold(match.group("model_id").strip(), match.group("threshold")))
            continue

        specs.append(NsfwModelSpec(model_id=item, threshold=DEFAULT_NSFW_SCORE_THRESHOLD))

    return specs


def _csv_tokens(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    return {str(item).strip().lower() for item in raw_items if str(item).strip()}


def csv_option_allows(value: str | Iterable[str] | None, candidate: str) -> bool:
    tokens = _csv_tokens(value)
    return "all" in tokens or candidate.lower() in tokens


def _to_numpy(sample: Any) -> np.ndarray | None:
    if sample is None:
        return None
    if isinstance(sample, Image.Image):
        return np.asarray(sample.convert("RGB"))
    if torch.is_tensor(sample):
        return sample.detach().cpu().numpy()
    if isinstance(sample, np.ndarray):
        return sample
    return None


def _array_to_pil(array: np.ndarray) -> Image.Image:
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.ndim == 2:
        pass
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array.squeeze(-1)
    elif array.ndim != 3:
        raise ValueError(f"Unsupported image array shape for NSFW classification: {array.shape}")

    if array.dtype.kind in {"f", "d"}:
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    return image.convert("RGB")


def _select_frame_indices(total_frames: int, frame_count: int, selection: str) -> list[int]:
    if total_frames <= 0:
        return []
    frame_count = max(1, min(int(frame_count), total_frames))
    selection = str(selection or "uniform").lower()

    if selection == "first":
        return list(range(frame_count))
    if selection == "middle":
        start = max(0, (total_frames - frame_count) // 2)
        return list(range(start, start + frame_count))
    if selection != "uniform":
        raise ValueError(f"Unknown NSFW video frame selection mode: {selection}")
    if frame_count == 1:
        return [total_frames // 2]

    indices = [round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)]
    return sorted(dict.fromkeys(indices))


def extract_classifier_frames(sample: Any, frame_count: int = 3, selection: str = "uniform") -> list[Image.Image]:
    if isinstance(sample, Image.Image):
        return [sample.convert("RGB")]

    if isinstance(sample, (list, tuple)):
        indices = _select_frame_indices(len(sample), frame_count, selection)
        frames = []
        for index in indices:
            frame = sample[index]
            if isinstance(frame, Image.Image):
                frames.append(frame.convert("RGB"))
            else:
                array = _to_numpy(frame)
                if array is not None:
                    frames.append(_array_to_pil(array))
        return frames

    array = _to_numpy(sample)
    if array is None:
        raise ValueError(f"Unsupported sample type for NSFW classification: {type(sample)}")
    if array.ndim == 5:
        array = array[0]
    if array.ndim == 4:
        indices = _select_frame_indices(array.shape[0], frame_count, selection)
        return [_array_to_pil(array[index]) for index in indices]
    return [_array_to_pil(array)]


class NsfwClassifierModelStore:
    """Load and reuse Hugging Face Transformers image classifiers."""

    def __init__(
        self,
        *,
        model_specs: list[NsfwModelSpec],
        min_votes: int,
        video_frame_count: int = 3,
        video_frame_selection: str = "uniform",
        video_min_flagged_frames: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        if not model_specs:
            raise ValueError("NSFW checks are enabled but no classifier models were configured.")
        if min_votes < 1:
            raise ValueError("nsfw_check_min_votes must be at least 1.")
        if min_votes > len(model_specs):
            raise ValueError("nsfw_check_min_votes cannot be larger than the number of configured NSFW classifiers.")
        self.video_frame_count = int(video_frame_count)
        self.video_min_flagged_frames = int(video_min_flagged_frames)
        if self.video_frame_count < 1:
            raise ValueError("nsfw_check_video_frame_count must be at least 1.")
        if self.video_min_flagged_frames < 1:
            raise ValueError("nsfw_check_video_min_flagged_frames must be at least 1.")
        if self.video_min_flagged_frames > self.video_frame_count:
            raise ValueError("nsfw_check_video_min_flagged_frames cannot exceed nsfw_check_video_frame_count.")

        self.model_specs = model_specs
        self.min_votes = min_votes
        self.video_frame_selection = str(video_frame_selection or "uniform").lower()
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._models: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._models.clear()
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_id: str) -> tuple[Any, Any]:
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]

            from transformers import AutoImageProcessor, AutoModelForImageClassification

            try:
                processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=False)
                model = AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load NSFW classifier {model_id!r} with Hugging Face Transformers. "
                    "Only standard Transformers image-classification models are supported."
                ) from exc

            model.eval()
            model.to(self.device)
            self._models[model_id] = (processor, model)
            return processor, model

    def _classify_frame_with_model(self, image: Image.Image, model_spec: NsfwModelSpec) -> dict[str, Any]:
        started_at = time.perf_counter()
        processor, model = self._load_model(model_spec.model_id)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
        scores = []
        for index, score in enumerate(probabilities.tolist()):
            label = id2label.get(index, id2label.get(str(index), str(index)))
            scores.append({"label": str(label), "score": float(score)})
        scores.sort(key=lambda item: item["score"], reverse=True)

        nsfw_score = score_sum(scores, NSFW_LABEL_HINTS)
        if nsfw_score is None:
            raise ValueError(f"NSFW classifier {model_spec.model_id!r} did not return any recognizable unsafe labels.")
        sfw_score = score_sum(scores, SFW_LABEL_HINTS)
        top = scores[0] if scores else {"label": None, "score": math.nan}
        return {
            "key": model_spec.key,
            "model_id": model_spec.model_id,
            "threshold": model_spec.threshold,
            "top_label": top["label"],
            "top_score": round(float(top["score"]), 6),
            "nsfw_score": round(float(nsfw_score), 6),
            "sfw_score": None if sfw_score is None else round(float(sfw_score), 6),
            "verdict": "nsfw" if nsfw_score >= model_spec.threshold else "sfw",
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }

    def classify_image(self, image: Image.Image) -> dict[str, Any]:
        with self._lock:
            classifier_results = [self._classify_frame_with_model(image, model_spec) for model_spec in self.model_specs]
        nsfw_votes = sum(1 for result in classifier_results if result["verdict"] == "nsfw")
        known_votes = len(classifier_results)
        return {
            "classifiers": classifier_results,
            "summary": {
                "count": len(classifier_results),
                "known_verdicts": known_votes,
                "nsfw_votes": nsfw_votes,
                "sfw_votes": known_votes - nsfw_votes,
                "majority_verdict": "nsfw" if nsfw_votes >= self.min_votes else "sfw",
            },
        }

    def classify_sample(self, sample: Any, *, filepath: str | None = None) -> dict[str, Any]:
        frames = extract_classifier_frames(
            sample,
            frame_count=self.video_frame_count,
            selection=self.video_frame_selection,
        )
        if not frames:
            raise ValueError(f"No frames available for NSFW classification: {filepath}")

        frame_results = []
        flagged_frames = 0
        for index, frame in enumerate(frames):
            frame_result = self.classify_image(frame)
            frame_result["frame_index"] = index
            if frame_result["summary"]["nsfw_votes"] >= self.min_votes:
                flagged_frames += 1
            frame_results.append(frame_result)

        rejected = flagged_frames >= self.video_min_flagged_frames
        return {
            "filepath": filepath,
            "frames_scanned": len(frames),
            "frame_results": frame_results,
            "summary": {
                "flagged_frames": flagged_frames,
                "video_min_flagged_frames": self.video_min_flagged_frames,
                "rejected": rejected,
            },
        }
