"""Dataset blueprint and plan routes exposed to the Web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import get_dataset_blueprints
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.dataset_connection_service import (
    DatasetConnectionError,
    DatasetConnectionService,
)
from simpletuner.simpletuner_sdk.server.services.dataset_plan import DatasetPlanStore, ValidationMessage, compute_validations
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
    except Exception as e:
        pass  # Fall back to default behavior

    try:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            return DatasetPlanStore(path=Path(defaults.configs_dir).expanduser() / "multidatabackend.json")
    except Exception:
        pass

    return DatasetPlanStore()


@router.get("/blueprints")
async def list_blueprints() -> Dict[str, Any]:
    """Return blueprint metadata for all supported dataset backends."""
    blueprints = [blueprint.model_dump() for blueprint in get_dataset_blueprints()]
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
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        model_family = active_config["config"].get("model_family") or active_config["config"].get("--model_family")
    except Exception:
        pass

    validations = compute_validations(datasets, get_dataset_blueprints(), model_family=model_family)
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
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        model_family = active_config["config"].get("model_family") or active_config["config"].get("--model_family")
    except Exception:
        pass

    validations = compute_validations(datasets, get_dataset_blueprints(), model_family=model_family)
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
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        model_family = active_config["config"].get("model_family") or active_config["config"].get("--model_family")
    except Exception:
        pass

    # Validate the updated plan
    validations = compute_validations(datasets, get_dataset_blueprints(), model_family=model_family)
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
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        configs_service = ConfigsService()
        active_config = configs_service.get_active_config()
        model_family = active_config.get("model_family") or active_config.get("--model_family")
    except Exception:
        pass

    # Validate the updated plan
    validations = compute_validations(datasets, get_dataset_blueprints(), model_family=model_family)
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
