"""Dataset plan persistence and validation helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.distillation.requirements import (
    EMPTY_PROFILE,
    DistillerRequirementProfile,
    describe_requirement_groups,
    evaluate_requirement_profile,
)
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
    model_flavour: Optional[str] = None,
    distillation_method: Optional[str] = None,
) -> List[ValidationMessage]:
    """Perform lightweight validation mirroring the UI logic."""
    validations: List[ValidationMessage] = []
    distiller_profile = _resolve_distiller_profile(distillation_method)
    relax_training_requirement = _relaxes_training_requirement(distiller_profile)
    requirement_eval = None

    if not datasets:
        validations.append(
            ValidationMessage(
                field="datasets",
                message="add at least one dataset before exporting",
                level="info",
            )
        )
        return validations

    if distiller_profile:
        requirement_eval = evaluate_requirement_profile(distiller_profile, datasets)
        if not requirement_eval.fulfilled:
            method_label = (distillation_method or "distiller").replace("_", " ")
            validations.append(
                ValidationMessage(
                    field="datasets",
                    message=f"{method_label} requires datasets matching "
                    f"{describe_requirement_groups(requirement_eval.missing_requirements)}",
                    level="error",
                )
            )

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
        normalized_type = _dataset_type(dataset)
        if normalized_type is DatasetType.CAPTION and backend_type.lower() == "csv":
            validations.append(
                ValidationMessage(
                    field=_normalise_identifier(dataset),
                    message="Caption datasets cannot use CSV backends; select local, AWS, parquet, or Hugging Face storage instead.",
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
    requires_strict_video_inputs = False
    supports_audio_only = False
    if model_family:
        try:
            from simpletuner.helpers.models.common import VideoModelFoundation
            from simpletuner.helpers.models.registry import ModelRegistry

            model_cls = ModelRegistry.get(model_family)
            if model_cls:
                # Resolve lazy model class if needed
                if hasattr(model_cls, "get_real_class"):
                    actual_cls = model_cls.get_real_class()
                else:
                    actual_cls = model_cls
                is_video_model = issubclass(actual_cls, VideoModelFoundation)
                if hasattr(actual_cls, "is_strict_i2v_flavour") and callable(actual_cls.is_strict_i2v_flavour):
                    requires_strict_video_inputs = bool(actual_cls.is_strict_i2v_flavour(model_flavour))
                if hasattr(actual_cls, "supports_audio_only_training") and callable(actual_cls.supports_audio_only_training):
                    supports_audio_only = bool(actual_cls.supports_audio_only_training())
        except Exception:
            pass

    # For video models, require at least one video dataset
    # For image models, require at least one image dataset
    image_count = sum(1 for dataset in datasets if _dataset_type(dataset) is DatasetType.IMAGE)
    video_count = sum(1 for dataset in datasets if _dataset_type(dataset) is DatasetType.VIDEO)
    audio_count = sum(1 for dataset in datasets if _dataset_type(dataset) is DatasetType.AUDIO)

    # Check for audio-only mode:
    # 1. Explicit: audio datasets with audio.audio_only: true
    # 2. Implicit: model supports audio-only AND has audio datasets AND no video/image datasets
    explicit_audio_only_count = sum(
        1
        for dataset in datasets
        if _dataset_type(dataset) is DatasetType.AUDIO and dataset.get("audio", {}).get("audio_only", False)
    )
    implicit_audio_only = supports_audio_only and audio_count > 0 and video_count == 0 and image_count == 0
    has_valid_audio_only = supports_audio_only and (explicit_audio_only_count > 0 or implicit_audio_only)

    if str(model_family).lower() == "ace_step":
        if audio_count == 0 and not relax_training_requirement:
            validations.append(
                ValidationMessage(
                    field="datasets",
                    message="at least one audio dataset is required for ACE-Step models",
                    level="error",
                )
            )
    elif is_video_model and not relax_training_requirement:
        if video_count == 0:
            if requires_strict_video_inputs:
                validations.append(
                    ValidationMessage(
                        field="datasets",
                        message="strict image-to-video flavours require at least one video dataset",
                        level="error",
                    )
                )
            elif image_count == 0 and not has_valid_audio_only:
                validations.append(
                    ValidationMessage(
                        field="datasets",
                        message="add at least one image or video dataset for this model",
                        level="error",
                    )
                )
    elif not relax_training_requirement:
        if image_count == 0:
            validations.append(
                ValidationMessage(
                    field="datasets",
                    message="at least one image dataset is required",
                    level="error",
                )
            )

    text_embed_datasets = [dataset for dataset in datasets if _dataset_type(dataset) is DatasetType.TEXT_EMBEDS]
    if not text_embed_datasets:
        validations.append(
            ValidationMessage(
                field="text_embeds",
                message="no text embed dataset provided - a default cache dataset will be created automatically",
                level="info",
            )
        )
    else:
        default_count = sum(1 for dataset in text_embed_datasets if bool(dataset.get("default")))
        if default_count == 0:
            validations.append(
                ValidationMessage(
                    field="text_embeds",
                    message="no default text embed dataset set - the first entry will be used automatically",
                    level="info",
                )
            )
        elif default_count > 1:
            validations.append(
                ValidationMessage(
                    field="text_embeds",
                    message="multiple text embed datasets marked default - only one is allowed",
                    level="error",
                )
            )

    # Check for orphaned text_embeds and image_embeds references
    text_embed_ids = {dataset.get("id") for dataset in datasets if _dataset_type(dataset) is DatasetType.TEXT_EMBEDS}
    image_embed_ids = {dataset.get("id") for dataset in datasets if _dataset_type(dataset) is DatasetType.IMAGE_EMBEDS}
    conditioning_ids = {dataset.get("id") for dataset in datasets if _dataset_type(dataset) is DatasetType.CONDITIONING}
    image_video_datasets = [
        dataset for dataset in datasets if _dataset_type(dataset) in {DatasetType.IMAGE, DatasetType.VIDEO}
    ]

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

    # Conditioning linkage warnings
    conditioning_links: set[str] = set()
    for dataset in image_video_datasets:
        raw_links = dataset.get("conditioning_data") or []
        if isinstance(raw_links, str):
            raw_links = [link.strip() for link in raw_links.split(",") if link.strip()]
        if isinstance(raw_links, list):
            conditioning_links.update(str(link) for link in raw_links if link)

    if conditioning_ids and image_video_datasets and not (conditioning_links & conditioning_ids):
        validations.append(
            ValidationMessage(
                field="conditioning",
                message="conditioning dataset present but not linked to any image/video dataset",
                level="warning",
            )
        )
    if conditioning_ids and not image_video_datasets:
        validations.append(
            ValidationMessage(
                field="conditioning",
                message="conditioning dataset configured without any image/video dataset",
                level="warning",
            )
        )

    # Edit model guidance
    model_family_lc = str(model_family or "").lower()
    model_flavour_lc = str(model_flavour or "").lower()
    is_edit_model = (model_family_lc == "qwen_image" and "edit" in model_flavour_lc) or (
        model_family_lc == "flux" and "kontext" in model_flavour_lc
    )
    if is_edit_model and not conditioning_ids:
        validations.append(
            ValidationMessage(
                field="conditioning",
                message="edit model detected but no conditioning dataset configured",
                level="warning",
            )
        )

    # Check that at least one training dataset is immediately available
    # (not delayed by start_epoch or start_step scheduling)
    training_datasets = [
        dataset
        for dataset in datasets
        if _dataset_type(dataset) in {DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.AUDIO}
        and not dataset.get("disabled")
    ]
    if training_datasets:
        immediate_datasets = [dataset for dataset in training_datasets if _is_immediately_available(dataset)]
        if not immediate_datasets:
            validations.append(
                ValidationMessage(
                    field="scheduling",
                    message="at least one dataset must be available from the start of training (start_epoch <= 1 and start_step <= 0)",
                    level="error",
                    suggestion="Set one dataset to use 'Immediate' scheduling mode or set start_epoch=1 and start_step=0",
                )
            )

        # Validate end_epoch/end_step scheduling
        for dataset in training_datasets:
            dataset_id = dataset.get("id", "unknown")
            start_epoch = dataset.get("start_epoch")
            start_step = dataset.get("start_step")
            end_epoch = dataset.get("end_epoch")
            end_step = dataset.get("end_step")

            # Validate end_epoch >= start_epoch when both are set
            if end_epoch is not None and start_epoch is not None:
                try:
                    end_e = int(end_epoch)
                    start_e = int(start_epoch) if start_epoch else 1
                    if end_e > 0 and end_e < start_e:
                        validations.append(
                            ValidationMessage(
                                field=f"{_normalise_identifier(dataset)}.scheduling",
                                message=f"end_epoch ({end_e}) must be >= start_epoch ({start_e})",
                                level="error",
                            )
                        )
                except (TypeError, ValueError):
                    pass

            # Validate end_step >= start_step when both are set
            if end_step is not None and start_step is not None:
                try:
                    end_s = int(end_step)
                    start_s = int(start_step) if start_step else 0
                    if end_s > 0 and end_s < start_s:
                        validations.append(
                            ValidationMessage(
                                field=f"{_normalise_identifier(dataset)}.scheduling",
                                message=f"end_step ({end_s}) must be >= start_step ({start_s})",
                                level="error",
                            )
                        )
                except (TypeError, ValueError):
                    pass

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


def _dataset_type(dataset: Dict[str, Any]) -> Optional[DatasetType]:
    """Best-effort conversion of raw dataset_type values into DatasetType enums."""
    value = dataset.get("dataset_type")
    if value is None:
        return None
    try:
        return DatasetType.from_value(value)
    except ValueError:
        return None


def _is_immediately_available(dataset: Dict[str, Any]) -> bool:
    """Check if a dataset is available from the start of training."""
    start_epoch = dataset.get("start_epoch")
    start_step = dataset.get("start_step")

    # Normalize start_epoch: not set or <= 1 means available from epoch 1
    epoch_ok = start_epoch is None or (isinstance(start_epoch, (int, float)) and start_epoch <= 1)

    # Normalize start_step: not set or <= 0 means available from step 0
    step_ok = start_step is None or (isinstance(start_step, (int, float)) and start_step <= 0)

    return epoch_ok and step_ok


def _resolve_distiller_profile(distillation_method: Optional[str]) -> DistillerRequirementProfile:
    if not distillation_method:
        return EMPTY_PROFILE
    return DistillationRegistry.get_requirement_profile(distillation_method)


def _relaxes_training_requirement(profile: DistillerRequirementProfile) -> bool:
    if not profile:
        return False
    if not profile.is_data_generator:
        return False
    return not any(profile.requires_dataset_type(dataset_type) for dataset_type in (DatasetType.IMAGE, DatasetType.VIDEO))
