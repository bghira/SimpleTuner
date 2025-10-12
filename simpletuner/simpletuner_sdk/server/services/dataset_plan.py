"""Dataset plan persistence and validation helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import (
    BackendBlueprint,
    find_blueprint,
    get_dataset_blueprints,
)

_DATASET_PLAN_ENV = "SIMPLETUNER_DATASET_PLAN_PATH"
_DEFAULT_PLAN_PATH = Path("config/multidatabackend.json")


class ValidationMessage(BaseModel):
    field: str
    message: str
    level: Literal["error", "warning", "info"]
    suggestion: Optional[str] = None


class DatasetPlanStore:
    """Small helper for reading and writing dataset plan files."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            self.path = self._resolve_default_path()

    @staticmethod
    def _resolve_default_path() -> Path:
        env_path = os.environ.get(_DATASET_PLAN_ENV)
        if env_path:
            return Path(env_path)
        return _DEFAULT_PLAN_PATH

    def load(self) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """Load the dataset plan from disk."""
        path = self.path
        if not path.exists():
            return [], "default", None

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"dataset plan file '{path}' contains invalid JSON: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"dataset plan file '{path}' must contain a JSON array.")

        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return data, "disk", updated_at

    def save(self, datasets: List[Dict[str, Any]]) -> str:
        """Persist the dataset plan to disk and return the update timestamp."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(datasets, handle, indent=2, sort_keys=True)
            handle.write("\n")

        temp_path.replace(path)
        return datetime.now(timezone.utc).isoformat()


def _normalise_identifier(dataset: Dict[str, Any]) -> str:
    dataset_id = dataset.get("id")
    if isinstance(dataset_id, str) and dataset_id.strip():
        return dataset_id.strip()
    dataset_type = dataset.get("dataset_type")
    backend_type = dataset.get("type")
    parts = [str(part) for part in (backend_type, dataset_type) if part]
    return "|".join(parts) or "dataset"


def compute_validations(
    datasets: List[Dict[str, Any]],
    blueprints: Optional[List[BackendBlueprint]] = None,
    model_family: Optional[str] = None,
) -> List[ValidationMessage]:
    """Perform lightweight validation mirroring the UI logic."""
    validations: List[ValidationMessage] = []

    if not datasets:
        validations.append(
            ValidationMessage(
                field="datasets",
                message="add at least one dataset before exporting",
                level="info",
            )
        )
        return validations

    id_counts: Dict[str, int] = {}
    for dataset in datasets:
        dataset_id = str(dataset.get("id", "")).strip()
        if not dataset_id:
            validations.append(
                ValidationMessage(
                    field=_normalise_identifier(dataset),
                    message="dataset id is required",
                    level="error",
                )
            )
        else:
            id_counts[dataset_id] = id_counts.get(dataset_id, 0) + 1

        backend_type = str(dataset.get("type", "")).strip()
        if not backend_type:
            validations.append(
                ValidationMessage(
                    field=f"{_normalise_identifier(dataset)}.type",
                    message="backend type is required",
                    level="error",
                )
            )

        dataset_type = str(dataset.get("dataset_type", "")).strip()
        if not dataset_type:
            validations.append(
                ValidationMessage(
                    field=f"{_normalise_identifier(dataset)}.dataset_type",
                    message="dataset_type is required",
                    level="error",
                )
            )

    for dataset_id, count in id_counts.items():
        if count > 1:
            validations.append(
                ValidationMessage(
                    field=dataset_id,
                    message="dataset ids must be unique",
                    level="error",
                )
            )

    # Check if this is a video-only model
    is_video_model = False
    if model_family:
        try:
            from simpletuner.helpers.models.common import VideoModelFoundation
            from simpletuner.helpers.models.registry import ModelRegistry

            model_cls = ModelRegistry.get(model_family)
            if model_cls:
                is_video_model = issubclass(model_cls, VideoModelFoundation)
        except Exception:
            pass

    # For video models, require at least one video dataset
    # For image models, require at least one image dataset
    image_count = sum(1 for dataset in datasets if dataset.get("dataset_type") == "image")
    video_count = sum(1 for dataset in datasets if dataset.get("dataset_type") == "video")

    if is_video_model:
        if video_count == 0:
            validations.append(
                ValidationMessage(
                    field="datasets",
                    message="at least one video dataset is required for video models",
                    level="error",
                )
            )
    else:
        if image_count == 0:
            validations.append(
                ValidationMessage(
                    field="datasets",
                    message="at least one image dataset is required",
                    level="error",
                )
            )

    text_embed_datasets = [dataset for dataset in datasets if dataset.get("dataset_type") == "text_embeds"]
    if not text_embed_datasets:
        validations.append(
            ValidationMessage(
                field="text_embeds",
                message="text embed datasets are required for caption guidance",
                level="error",
            )
        )
    else:
        default_count = sum(1 for dataset in text_embed_datasets if bool(dataset.get("default")))
        if default_count == 0:
            validations.append(
                ValidationMessage(
                    field="text_embeds",
                    message="mark one text embed dataset as default",
                    level="error",
                )
            )
        elif default_count > 1:
            validations.append(
                ValidationMessage(
                    field="text_embeds",
                    message="only one text embed dataset can be default",
                    level="error",
                )
            )

    # Check for orphaned text_embeds and image_embeds references
    text_embed_ids = {dataset.get("id") for dataset in datasets if dataset.get("dataset_type") == "text_embeds"}
    image_embed_ids = {dataset.get("id") for dataset in datasets if dataset.get("dataset_type") == "image_embeds"}

    for dataset in datasets:
        dataset_id = dataset.get("id", "unknown")

        # Check text_embeds reference
        text_embeds_ref = dataset.get("text_embeds")
        if text_embeds_ref and text_embeds_ref not in text_embed_ids:
            validations.append(
                ValidationMessage(
                    field=f"{_normalise_identifier(dataset)}.text_embeds",
                    message=f"references non-existent text_embeds dataset '{text_embeds_ref}'",
                    level="error",
                )
            )

        # Check image_embeds reference
        image_embeds_ref = dataset.get("image_embeds")
        if image_embeds_ref and image_embeds_ref not in image_embed_ids:
            validations.append(
                ValidationMessage(
                    field=f"{_normalise_identifier(dataset)}.image_embeds",
                    message=f"references non-existent image_embeds dataset '{image_embeds_ref}'",
                    level="error",
                )
            )

    if blueprints is None:
        blueprints = get_dataset_blueprints()

    for dataset in datasets:
        backend_type = str(dataset.get("type", "")).strip()
        dataset_type = str(dataset.get("dataset_type", "")).strip()
        if not backend_type or not dataset_type:
            continue

        blueprint = find_blueprint(backend_type, dataset_type)
        if blueprint is None:
            validations.append(
                ValidationMessage(
                    field=_normalise_identifier(dataset),
                    message="blueprint data missing for this dataset",
                    level="warning",
                )
            )
            continue

        for field in blueprint.fields:
            if not field.required:
                continue
            value = dataset.get(field.id)
            if value is None:
                validations.append(
                    ValidationMessage(
                        field=f"{_normalise_identifier(dataset)}.{field.id}",
                        message=f"{field.label.lower()} is required",
                        level="error",
                    )
                )
                continue
            if isinstance(value, str) and not value.strip():
                validations.append(
                    ValidationMessage(
                        field=f"{_normalise_identifier(dataset)}.{field.id}",
                        message=f"{field.label.lower()} is required",
                        level="error",
                    )
                )

    return validations
