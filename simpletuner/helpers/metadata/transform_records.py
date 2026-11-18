"""The geometry decided for a sample, how it is encoded, and the recipe of settings it follows from."""

import json
from hashlib import sha256

from simpletuner.helpers.data_backend.dataset_runtime import RUNTIME_MUTABLE_KEYS, recrops_each_epoch
from simpletuner.helpers.training.state_tracker import StateTracker

DECIDED = ("aspect_ratio", "target_size", "intermediary_size", "crop_coordinates")
_PAIRS = DECIDED[1:]


def encode_transform(record: dict) -> bytes:
    return json.dumps([record[key] for key in DECIDED], separators=(",", ":")).encode("utf-8")


def decode_transform(value: bytes) -> dict:
    aspect_ratio, *pairs = json.loads(value)
    return {"aspect_ratio": aspect_ratio, **{key: None if pair is None else tuple(pair) for key, pair in zip(_PAIRS, pairs)}}


def settings_digest(metadata_backend, *, fingerprint=None, units=None) -> str:
    """What a dataset's decisions follow from: its settings, the model's canvas and the source they were made over."""
    args, model = StateTracker.get_args(), StateTracker.get_model()
    config = StateTracker.get_data_backend_config(metadata_backend.id)
    decided_by = {
        "config": {key: value for key, value in config.items() if key not in RUNTIME_MUTABLE_KEYS},
        "resolution": [metadata_backend.resolution, metadata_backend.resolution_type],
        "limits": [
            metadata_backend.minimum_image_size,
            metadata_backend.minimum_aspect_ratio,
            metadata_backend.maximum_aspect_ratio,
        ],
        "alignment": getattr(args, "aspect_bucket_alignment", None),
        "rounding": getattr(args, "aspect_bucket_rounding", None),
        "canvas": getattr(model, "MAXIMUM_CANVAS_SIZE", None),
        "shards": units,
        "fingerprint": fingerprint,
    }
    return sha256(json.dumps(decided_by, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def transform_recipe(digest: str, epoch: int, dataset_id: str) -> str:
    """A dataset that re-crops each epoch decides again in each one, so the epoch is part of its recipe."""
    if recrops_each_epoch(StateTracker.get_data_backend_config(dataset_id), StateTracker.get_args()):
        return f"{digest}-epoch-{int(epoch)}"
    return digest
