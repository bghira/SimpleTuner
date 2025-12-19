"""Dataset blueprint and plan routes exposed to the Web UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import get_dataset_blueprints
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.dataset_caption_service import (
    CaptionStatus,
    CaptionWriteResult,
    DatasetCaptionService,
    ThumbnailInfo,
)
from simpletuner.simpletuner_sdk.server.services.dataset_connection_service import (
    DatasetConnectionError,
    DatasetConnectionService,
)
from simpletuner.simpletuner_sdk.server.services.dataset_plan import DatasetPlanStore, ValidationMessage, compute_validations
from simpletuner.simpletuner_sdk.server.services.dataset_upload_service import (
    DatasetUploadService,
    FolderCreateResult,
    UploadResult,
)
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetConnectionRequest(BaseModel):
    dataset: Dict[str, Any]
    configs_dir: Optional[str] = None


_connection_service = DatasetConnectionService()


class DatasetPlanEntry(BaseModel):
    id: str
    type: str
    dataset_type: str

    model_config = ConfigDict(extra="allow")


class DatasetPlanPayload(BaseModel):
    datasets: List[DatasetPlanEntry]
    createBackup: Optional[bool] = False


class DatasetPlanResponse(BaseModel):
    datasets: List[Dict[str, Any]]
    validations: List[ValidationMessage]
    source: Literal["default", "disk"]
    updated_at: Optional[str] = None
    backupPath: Optional[str] = None


def _ensure_text_embed_dataset(
    datasets: List[Dict[str, Any]], *, model_family: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Guarantee at least one text_embeds dataset exists and a single default is selected."""
    seen_ids = {str(ds.get("id", "")) for ds in datasets if ds.get("id")}
    text_embeds = [ds for ds in datasets if str(ds.get("dataset_type", "")).lower() == "text_embeds"]
    enabled_text_embeds = [ds for ds in text_embeds if not ds.get("disabled") and not ds.get("disable")]

    if not enabled_text_embeds:
        base_id = "text-embeds"
        candidate = base_id
        suffix = 1
        while candidate in seen_ids:
            candidate = f"{base_id}-{suffix}"
            suffix += 1
        cache_dir = f"{{output_dir}}/cache/text/{model_family or 'base'}"
        datasets.append(
            {
                "id": candidate,
                "type": "local",
                "dataset_type": "text_embeds",
                "cache_dir": cache_dir,
                "default": True,
                "write_batch_size": 128,
            }
        )
        text_embeds = [ds for ds in datasets if str(ds.get("dataset_type", "")).lower() == "text_embeds"]
        enabled_text_embeds = [ds for ds in text_embeds if not ds.get("disabled") and not ds.get("disable")]

    # Ensure only one default flag
    default_assigned = False
    for ds in text_embeds:
        if bool(ds.get("default")) and not default_assigned:
            ds["default"] = True
            default_assigned = True
        else:
            ds["default"] = False
    if not default_assigned and enabled_text_embeds:
        enabled_text_embeds[0]["default"] = True

    return datasets


def _store() -> DatasetPlanStore:
    """Create a dataset plan store using current environment settings."""
    env_path = os.environ.get("SIMPLETUNER_DATASET_PLAN_PATH")
    if env_path:
        try:
            return DatasetPlanStore(path=Path(env_path).expanduser())
        except Exception:
            return DatasetPlanStore(path=Path(env_path))

    # Try to get data backend config from active configuration
    try:
        defaults = WebUIStateStore().load_defaults()
        config_store = ConfigStore(config_dir=defaults.configs_dir) if defaults.configs_dir else ConfigStore()
        active_config_name = config_store.get_active_config()

        if active_config_name:
            config_data, _ = config_store.load_config(active_config_name)
            # Try with and without -- prefix
            data_backend_config = config_data.get("data_backend_config") or config_data.get("--data_backend_config")

            if data_backend_config:
                # Resolve the path relative to config directory
                resolved_path = resolve_config_path(
                    data_backend_config, config_dir=config_store.config_dir, check_cwd_first=True
                )
                if resolved_path:
                    return DatasetPlanStore(path=resolved_path)
    except Exception:
        pass  # Fall back to default behavior

    try:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            return DatasetPlanStore(path=Path(defaults.configs_dir).expanduser() / "multidatabackend.json")
    except Exception:
        pass

    return DatasetPlanStore()


def _get_active_resolution() -> int:
    """Get resolution from active config, defaulting to 1024."""
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        config_blob = active_config.get("config", {})
        resolution = config_blob.get("resolution") or config_blob.get("--resolution")
        if resolution is not None:
            return int(resolution)
    except Exception:
        pass
    return 1024


def _apply_resolution_defaults(blueprints: List[Dict[str, Any]], resolution: int) -> List[Dict[str, Any]]:
    """Apply resolution-based defaults to blueprint fields."""
    for blueprint in blueprints:
        # Update defaults dict
        if "defaults" in blueprint:
            defaults = blueprint["defaults"]
            if "resolution" in defaults:
                defaults["resolution"] = resolution

        # Update field defaults
        for field in blueprint.get("fields", []):
            field_id = field.get("id")
            if field_id == "resolution":
                field["defaultValue"] = resolution
            elif field_id in ("minimum_image_size", "maximum_image_size", "target_downsample_size"):
                field["defaultValue"] = resolution

    return blueprints


@router.get("/blueprints")
async def list_blueprints() -> Dict[str, Any]:
    """Return blueprint metadata for all supported dataset backends."""
    blueprints = [blueprint.model_dump() for blueprint in get_dataset_blueprints()]
    resolution = _get_active_resolution()
    blueprints = _apply_resolution_defaults(blueprints, resolution)
    return {"blueprints": blueprints, "warnings": [], "source": "remote"}


@router.get("/plan", response_model=DatasetPlanResponse)
async def get_dataset_plan() -> DatasetPlanResponse:
    """Retrieve the currently saved dataset plan."""
    store = _store()
    try:
        datasets, source, updated_at = store.load()
    except ValueError as exc:
        validation = ValidationMessage(field="datasets", message=str(exc), level="error")
        return DatasetPlanResponse(datasets=[], validations=[validation], source="default", updated_at=None)

    # Get model_family from active config
    model_family = None
    model_flavour = None
    distillation_method = None
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        config_blob = active_config["config"]
        model_family = config_blob.get("model_family") or config_blob.get("--model_family")
        model_flavour = config_blob.get("model_flavour") or config_blob.get("--model_flavour")
        distillation_method = config_blob.get("distillation_method") or config_blob.get("--distillation_method")
    except Exception:
        pass

    validations = compute_validations(
        datasets,
        get_dataset_blueprints(),
        model_family=model_family,
        model_flavour=model_flavour,
        distillation_method=distillation_method,
    )
    return DatasetPlanResponse(
        datasets=datasets,
        validations=validations,
        source=source,
        updated_at=updated_at,
    )


@router.post("/test-connection")
async def test_dataset_connection(request: DatasetConnectionRequest) -> Dict[str, Any]:
    """Run a lightweight connection test for a dataset payload."""

    try:
        return _connection_service.test_connection(request.dataset, request.configs_dir)
    except DatasetConnectionError as exc:
        detail: Dict[str, Any] = {"message": exc.message}
        if exc.backend:
            detail["backend"] = exc.backend
        raise HTTPException(status_code=exc.status_code, detail=detail)


def _persist_plan(payload: DatasetPlanPayload) -> DatasetPlanResponse:
    datasets: List[Dict[str, Any]] = [entry.model_dump(exclude_none=True) for entry in payload.datasets]

    # Get model_family from active config
    model_family = None
    model_flavour = None
    distillation_method = None
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        config_blob = active_config["config"]
        model_family = config_blob.get("model_family") or config_blob.get("--model_family")
        model_flavour = config_blob.get("model_flavour") or config_blob.get("--model_flavour")
        distillation_method = config_blob.get("distillation_method") or config_blob.get("--distillation_method")
    except Exception:
        pass

    datasets = _ensure_text_embed_dataset(datasets, model_family=model_family)

    validations = compute_validations(
        datasets,
        get_dataset_blueprints(),
        model_family=model_family,
        model_flavour=model_flavour,
        distillation_method=distillation_method,
    )
    errors = [message for message in validations if message.level == "error"]
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "dataset plan failed validation",
                "validations": [message.model_dump() for message in validations],
            },
        )

    store = _store()
    backup_path = None

    # Create backup if requested
    if payload.createBackup:
        try:
            import shutil
            from datetime import datetime

            if store.path.exists():
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_filename = f"{store.path.name}.backup-{timestamp}"
                backup_path = store.path.parent / backup_filename
                shutil.copy2(store.path, backup_path)
        except Exception as e:
            # Log but don't fail the save operation
            print(f"Warning: Failed to create backup: {e}")

    try:
        updated_at = store.save(datasets)
    except OSError as exc:  # pragma: no cover - defensive, rarely triggered in tests
        raise HTTPException(status_code=500, detail=f"failed to persist dataset plan: {exc}") from exc

    APIState.set_state("dataset_plan", datasets)
    APIState.set_state("dataset_plan_updated_at", updated_at)

    return DatasetPlanResponse(
        datasets=datasets,
        validations=validations,
        source="disk",
        updated_at=updated_at,
        backupPath=str(backup_path) if backup_path else None,
    )


@router.post("/plan", response_model=DatasetPlanResponse)
async def create_dataset_plan(payload: DatasetPlanPayload) -> DatasetPlanResponse:
    """Create or replace the dataset plan configuration."""
    return _persist_plan(payload)


@router.patch("/plan", response_model=DatasetPlanResponse)
async def update_dataset_plan(payload: DatasetPlanPayload) -> DatasetPlanResponse:
    """Update the dataset plan configuration."""
    return _persist_plan(payload)


# File Browser and Dataset Detection Endpoints


def _resolve_datasets_dir_and_validate_path(
    path: Optional[str] = None,
) -> Tuple[Path, Optional[str], bool]:
    """
    Resolve datasets_dir from WebUI state and validate the requested path.

    Returns:
        Tuple of (validated_path_obj, datasets_dir, allow_outside)

    Raises:
        HTTPException: If path validation fails or access is denied
    """
    # Load resolved defaults from WebUIStateStore (includes fallbacks)
    webui_state = WebUIStateStore()
    defaults_bundle = webui_state.get_defaults_bundle()
    resolved = defaults_bundle["resolved"]

    # Onboarding values take precedence when present. The onboarding step is named
    # "default_datasets_dir" in the flow; fall back to defaults otherwise.
    onboarding = webui_state.load_onboarding()
    datasets_dir = None
    onboarding_step = onboarding.steps.get("default_datasets_dir")
    if onboarding_step and onboarding_step.value:
        datasets_dir = onboarding_step.value
    else:
        datasets_dir = resolved.get("datasets_dir")
        if not datasets_dir:
            datasets_dir = defaults_bundle["fallbacks"].get("datasets_dir")

    allow_outside = bool(resolved.get("allow_dataset_paths_outside_dir", False))

    # Use provided path or fall back to resolved datasets_dir (which includes fallback)
    if path is None:
        path = datasets_dir

    path_obj = Path(path).resolve()

    # Enforce datasets_dir root unless override is enabled
    if not allow_outside and datasets_dir:
        datasets_root = Path(datasets_dir).resolve()
        try:
            path_obj.relative_to(datasets_root)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Access denied: path is outside configured datasets directory. "
                    "Enable 'allow_dataset_paths_outside_dir' to browse other locations."
                ),
            )

    if not path_obj.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory does not exist: {path_obj}",
        )

    if not path_obj.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {path_obj}",
        )

    return path_obj, datasets_dir, allow_outside


@router.get("/browse")
async def browse_directories(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Browse directories at the given path and detect existing datasets.

    Returns a list of directories with metadata about whether they contain
    existing SimpleTuner datasets (detected via aspect bucket JSON files).

    If no path is provided, uses the configured datasets_dir from defaults.
    Enforces datasets_dir root unless allow_dataset_paths_outside_dir is enabled.
    """
    try:
        path_obj, datasets_dir, allow_outside = _resolve_datasets_dir_and_validate_path(path)

        directories = []

        for item in sorted(path_obj.iterdir()):
            if not item.is_dir():
                continue

            dir_info = {
                "name": item.name,
                "path": str(item),
                "hasDataset": False,
                "datasetId": None,
                "fileCount": None,
            }

            # Check for aspect bucket metadata files
            bucket_files = list(item.glob("aspect_ratio_bucket_indices_*.json"))
            if bucket_files:
                # Extract dataset ID from filename
                # Format: aspect_ratio_bucket_indices_{dataset_id}.json
                bucket_file = bucket_files[0]
                filename = bucket_file.stem  # Remove .json
                dataset_id = filename.replace("aspect_ratio_bucket_indices_", "")

                dir_info["hasDataset"] = True
                dir_info["datasetId"] = dataset_id

                # Count files in the directory (excluding json metadata)
                all_files = [f for f in item.iterdir() if f.is_file() and f.suffix != ".json"]
                dir_info["fileCount"] = len(all_files)

            directories.append(dir_info)

        # Calculate parent path in a platform-neutral way
        parent_path = str(path_obj.parent) if path_obj.parent != path_obj else None

        # Check if we can go up (don't allow going above datasets_dir unless override)
        can_go_up = parent_path is not None
        if can_go_up and not allow_outside and datasets_dir:
            datasets_root = Path(datasets_dir).resolve()
            try:
                Path(parent_path).resolve().relative_to(datasets_root)
            except ValueError:
                can_go_up = False
                parent_path = None

        return {
            "directories": directories,
            "currentPath": str(path_obj),
            "parentPath": parent_path,
            "canGoUp": can_go_up,
        }

    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied accessing path: {path_obj}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error browsing directories: {str(e)}",
        )


@router.get("/detect")
async def detect_dataset(path: str) -> Dict[str, Any]:
    """
    Detect if a directory contains an existing SimpleTuner dataset by reading
    the aspect bucket metadata files and returning the configuration.

    Enforces datasets_dir root unless allow_dataset_paths_outside_dir is enabled.
    """
    try:
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(path)

        # Look for aspect bucket indices file
        bucket_files = list(path_obj.glob("aspect_ratio_bucket_indices_*.json"))
        if not bucket_files:
            return {"hasDataset": False, "path": str(path_obj)}

        bucket_file = bucket_files[0]
        filename = bucket_file.stem
        dataset_id = filename.replace("aspect_ratio_bucket_indices_", "")

        # Read the bucket indices file to get config
        with open(bucket_file, "r") as f:
            bucket_data = json.load(f)

        config = bucket_data.get("config", {})

        # Count total files from aspect ratio buckets
        aspect_buckets = bucket_data.get("aspect_ratio_bucket_indices", {})
        total_files = sum(len(files) for files in aspect_buckets.values())

        return {
            "hasDataset": True,
            "datasetId": dataset_id,
            "path": str(path_obj),
            "config": config,
            "totalFiles": total_files,
            "aspectRatios": list(aspect_buckets.keys()),
        }

    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied accessing path: {path}",
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing dataset metadata: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting dataset: {str(e)}",
        )


# Folder Creation, Upload, and Caption Endpoints


def _get_upload_service() -> DatasetUploadService:
    """Get a DatasetUploadService configured with current settings."""
    webui_state = WebUIStateStore()
    defaults_bundle = webui_state.get_defaults_bundle()
    resolved = defaults_bundle["resolved"]

    # Get datasets_dir from onboarding or defaults
    onboarding = webui_state.load_onboarding()
    datasets_dir = None
    onboarding_step = onboarding.steps.get("default_datasets_dir")
    if onboarding_step and onboarding_step.value:
        datasets_dir = onboarding_step.value
    else:
        datasets_dir = resolved.get("datasets_dir")
        if not datasets_dir:
            datasets_dir = defaults_bundle["fallbacks"].get("datasets_dir")

    allow_outside = bool(resolved.get("allow_dataset_paths_outside_dir", False))
    datasets_path = Path(datasets_dir).resolve() if datasets_dir else None

    return DatasetUploadService(datasets_dir=datasets_path, allow_outside=allow_outside)


def _get_caption_service() -> DatasetCaptionService:
    """Get a DatasetCaptionService configured with current settings."""
    webui_state = WebUIStateStore()
    defaults_bundle = webui_state.get_defaults_bundle()
    resolved = defaults_bundle["resolved"]

    # Get datasets_dir from onboarding or defaults
    onboarding = webui_state.load_onboarding()
    datasets_dir = None
    onboarding_step = onboarding.steps.get("default_datasets_dir")
    if onboarding_step and onboarding_step.value:
        datasets_dir = onboarding_step.value
    else:
        datasets_dir = resolved.get("datasets_dir")
        if not datasets_dir:
            datasets_dir = defaults_bundle["fallbacks"].get("datasets_dir")

    allow_outside = bool(resolved.get("allow_dataset_paths_outside_dir", False))
    datasets_path = Path(datasets_dir).resolve() if datasets_dir else None

    return DatasetCaptionService(datasets_dir=datasets_path, allow_outside=allow_outside)


@router.post("/folders", response_model=FolderCreateResult)
async def create_folder(
    parent_path: str = Form(...),
    folder_name: str = Form(...),
) -> FolderCreateResult:
    """Create a new folder in the given parent directory.

    Args:
        parent_path: Absolute path to the parent directory.
        folder_name: Name for the new folder.

    Returns:
        FolderCreateResult with path and success status.
    """
    try:
        # Validate parent path exists and is within bounds
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(parent_path)

        service = _get_upload_service()
        result = service.create_folder(path_obj, folder_name)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error or "Failed to create folder",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating folder: {str(e)}",
        )


@router.post("/upload", response_model=UploadResult)
async def upload_files(
    target_path: str = Form(...),
    files: List[UploadFile] = File(...),
) -> UploadResult:
    """Upload files (images, txt, parquet, jsonl, csv) to target directory.

    Args:
        target_path: Absolute path to the target directory.
        files: List of files to upload.

    Returns:
        UploadResult with counts of uploaded/skipped files.
    """
    try:
        # Validate target path exists and is within bounds
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(target_path)

        service = _get_upload_service()
        result = await service.handle_file_uploads(files, path_obj)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}",
        )


@router.post("/upload/zip", response_model=UploadResult)
async def upload_zip(
    target_path: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResult:
    """Upload and extract a zip file to target directory.

    Args:
        target_path: Absolute path to the target directory.
        file: ZIP file to extract.

    Returns:
        UploadResult with counts of extracted files.
    """
    try:
        # Validate target path exists and is within bounds
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(target_path)

        # Validate it's a zip file
        if not file.filename or not file.filename.lower().endswith(".zip"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a ZIP archive",
            )

        service = _get_upload_service()
        result = await service.handle_zip_upload(file, path_obj)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting ZIP: {str(e)}",
        )


@router.get("/captions/status", response_model=CaptionStatus)
async def get_caption_status(path: str) -> CaptionStatus:
    """Analyze caption coverage in a directory.

    Args:
        path: Absolute path to the directory to analyze.

    Returns:
        CaptionStatus with coverage statistics.
    """
    try:
        # Validate path exists and is within bounds
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(path)

        service = _get_caption_service()
        return service.get_caption_status(path_obj)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing captions: {str(e)}",
        )


@router.get("/captions/thumbnails", response_model=List[ThumbnailInfo])
async def get_thumbnails(
    path: str,
    limit: int = 50,
    offset: int = 0,
) -> List[ThumbnailInfo]:
    """Get thumbnails for images missing captions in a directory.

    Args:
        path: Absolute path to the directory.
        limit: Maximum number of thumbnails to return.
        offset: Number of images to skip (for pagination).

    Returns:
        List of ThumbnailInfo with base64 encoded images.
    """
    try:
        # Validate path exists and is within bounds
        path_obj, _, _ = _resolve_datasets_dir_and_validate_path(path)

        service = _get_caption_service()

        # First get the caption status to find images without captions
        status_result = service.get_caption_status(path_obj)
        image_paths = [Path(img.path) for img in status_result.images_without_captions]

        # Generate thumbnails
        return service.get_thumbnails(image_paths, limit=limit, offset=offset)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating thumbnails: {str(e)}",
        )


class CaptionWriteRequest(BaseModel):
    """Request body for writing captions."""

    captions: Dict[str, str]


@router.post("/captions", response_model=CaptionWriteResult)
async def save_captions(
    request: CaptionWriteRequest = Body(...),
) -> CaptionWriteResult:
    """Create .txt caption files for the given images.

    Args:
        request: CaptionWriteRequest with dict mapping image paths to caption text.

    Returns:
        CaptionWriteResult with count of files written.
    """
    try:
        service = _get_caption_service()
        result = service.create_captions(request.captions)

        if not result.success and result.files_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to write any caption files",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error writing captions: {str(e)}",
        )


# HTMX Individual Dataset CRUD Operations


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Get a single dataset by ID."""
    store = _store()
    try:
        datasets, _, _ = store.load()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Dataset plan not found: {exc}")

    dataset = next((d for d in datasets if d.get("id") == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    return dataset


@router.post("/")
async def create_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new dataset."""
    store = _store()
    try:
        datasets, _, _ = store.load()
    except ValueError:
        datasets = []

    # Validate dataset ID is unique
    if any(d.get("id") == dataset.get("id") for d in datasets):
        raise HTTPException(status_code=400, detail={"message": f"Dataset ID '{dataset.get('id')}' already exists"})

    # Add dataset to plan
    datasets.append(dataset)

    # Get model_family from active config
    model_family = None
    model_flavour = None
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        config_blob = active_config["config"]
        model_family = config_blob.get("model_family") or config_blob.get("--model_family")
        model_flavour = config_blob.get("model_flavour") or config_blob.get("--model_flavour")
    except Exception:
        pass

    # Validate the updated plan
    validations = compute_validations(
        datasets,
        get_dataset_blueprints(),
        model_family=model_family,
        model_flavour=model_flavour,
    )
    errors = [v for v in validations if v.level == "error"]

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Dataset validation failed",
                "validation_errors": {v.field: v.message for v in errors if v.field},
            },
        )

    # Save updated plan
    try:
        store.save(datasets)
        APIState.set_state("dataset_plan", datasets)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save dataset: {exc}")

    return dataset


@router.put("/{dataset_id}")
async def update_dataset(dataset_id: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing dataset."""
    store = _store()
    try:
        datasets, _, _ = store.load()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Dataset plan not found: {exc}")

    # Find and update the dataset
    dataset_index = next((i for i, d in enumerate(datasets) if d.get("id") == dataset_id), None)
    if dataset_index is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Ensure the ID matches
    dataset["id"] = dataset_id
    datasets[dataset_index] = dataset

    # Get model_family from active config
    model_family = None
    model_flavour = None
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        config_blob = active_config["config"]
        model_family = config_blob.get("model_family") or config_blob.get("--model_family")
        model_flavour = config_blob.get("model_flavour") or config_blob.get("--model_flavour")
    except Exception:
        pass

    # Validate the updated plan
    validations = compute_validations(
        datasets,
        get_dataset_blueprints(),
        model_family=model_family,
        model_flavour=model_flavour,
    )
    errors = [v for v in validations if v.level == "error"]

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Dataset validation failed",
                "validation_errors": {v.field: v.message for v in errors if v.field},
            },
        )

    # Save updated plan
    try:
        store.save(datasets)
        APIState.set_state("dataset_plan", datasets)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save dataset: {exc}")

    return dataset


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    """Delete a dataset."""
    store = _store()
    try:
        datasets, _, _ = store.load()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Dataset plan not found: {exc}")

    # Find and remove the dataset
    original_count = len(datasets)
    datasets = [d for d in datasets if d.get("id") != dataset_id]

    if len(datasets) == original_count:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Save updated plan
    try:
        store.save(datasets)
        APIState.set_state("dataset_plan", datasets)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save datasets: {exc}")

    return {"message": f"Dataset {dataset_id} deleted successfully"}


# Field Validation Endpoints for HTMX


@router.post("/validate/{field_name}", response_class=HTMLResponse)
async def validate_field(field_name: str, value: str = "") -> str:
    """Validate a single field and return HTML error fragment."""
    error_html = ""

    # Basic validation rules
    if field_name == "id":
        if not value or not value.strip():
            error_html = "Dataset ID is required"
        elif not value.replace("-", "").replace("_", "").isalnum():
            error_html = "Dataset ID must contain only letters, numbers, hyphens, and underscores"
        elif len(value) > 50:
            error_html = "Dataset ID must be 50 characters or less"

    elif field_name == "instance_data_dir":
        if not value or not value.strip():
            error_html = "Instance data directory is required"
        elif not value.startswith("/"):
            error_html = "Path must be absolute (start with /)"

    elif field_name == "resolution":
        try:
            res = int(value) if value else 0
            if res < 256:
                error_html = "Resolution must be at least 256"
            elif res > 4096:
                error_html = "Resolution must not exceed 4096"
            elif res % 64 != 0:
                error_html = "Resolution must be divisible by 64"
        except ValueError:
            error_html = "Resolution must be a valid number"

    elif field_name == "probability":
        try:
            prob = float(value) if value else 0
            if prob < 0 or prob > 1:
                error_html = "Probability must be between 0 and 1"
        except ValueError:
            error_html = "Probability must be a valid number"

    elif field_name == "repeats":
        try:
            repeats = int(value) if value else 0
            if repeats < 0:
                error_html = "Repeats must be 0 or greater"
        except ValueError:
            error_html = "Repeats must be a valid number"

    return error_html
