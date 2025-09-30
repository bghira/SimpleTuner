"""Model information routes for SimpleTuner server."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from simpletuner.helpers.configuration.cmd_args import get_model_flavour_choices as _get_model_flavour_choices
from simpletuner.simpletuner_sdk.server.services.models_service import MODELS_SERVICE, ModelRegistry, ModelServiceError

_MODEL_FAMILY_SENTINEL = ModelRegistry.model_families()
model_families = _MODEL_FAMILY_SENTINEL


def get_model_flavour_choices(model_family: str) -> Iterable[str]:
    if _get_model_flavour_choices:
        try:
            return _get_model_flavour_choices(model_family) or []
        except Exception:  # pragma: no cover - defensive
            return []
    result = MODELS_SERVICE.get_model_flavours(model_family)
    return result.get("flavours", [])


router = APIRouter()


def _call_service(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ModelServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/api/models")
@router.get("/models")
async def get_model_families():
    """Return all available model families."""
    if model_families is not _MODEL_FAMILY_SENTINEL:
        try:
            keys = list(model_families.keys())  # type: ignore[attr-defined]
        except AttributeError:
            keys = list(model_families)  # type: ignore[arg-type]
        return {"families": keys}
    return _call_service(MODELS_SERVICE.list_families)


@router.get("/api/models/{model_family}")
@router.get("/models/{model_family}")
async def get_model_details(model_family: str):
    """Return metadata for a specific model family."""
    return _call_service(MODELS_SERVICE.get_model_details, model_family)


@router.get("/api/models/{model_family}/flavours")
@router.get("/models/{model_family}/flavours")
async def get_model_flavours(model_family: str):
    """Return the flavours for a specific model family."""
    if model_families is not _MODEL_FAMILY_SENTINEL:
        if model_family not in model_families:  # type: ignore[operator]
            raise HTTPException(status_code=404, detail=f"Model family '{model_family}' not found")
        return {"flavours": list(get_model_flavour_choices(model_family))}
    result = _call_service(MODELS_SERVICE.get_model_flavours, model_family)
    return {"flavours": result.get("flavours", [])}


@router.get("/api/models/{model_family}/flavours-select", response_class=HTMLResponse)
@router.get("/models/{model_family}/flavours-select", response_class=HTMLResponse)
async def get_model_flavours_html(model_family: str, current_value: str = ""):
    """Return flavours as HTML <option> elements for HTMX usage."""
    if model_family == "loading":
        return """
        <option value="">Loading flavours...</option>
        """

    if model_families is not _MODEL_FAMILY_SENTINEL and model_family not in model_families:  # type: ignore[operator]
        return """
        <option value="">Select a valid model family</option>
        """

    try:
        if model_families is not _MODEL_FAMILY_SENTINEL:
            result = {"flavours": list(get_model_flavour_choices(model_family))}
        else:
            result = MODELS_SERVICE.get_model_flavours(model_family)
    except ModelServiceError as exc:
        if exc.status_code == 404:
            return """
            <option value="">Select a model family first</option>
            """
        return """
        <option value="">Error loading flavours</option>
        """

    flavours = result.get("flavours", [])
    if not flavours:
        return """
        <option value="">No flavours available</option>
        """

    options = ['<option value="">Default</option>']
    for flavour in flavours:
        selected = "selected" if flavour == current_value else ""
        options.append(f'<option value="{flavour}" {selected}>{flavour}</option>')

    return "\n".join(options)


@router.post("/api/models/requirements")
@router.post("/models/requirements")
async def evaluate_model_requirements(payload: Dict[str, Any]):
    """Evaluate conditioning requirements for a model configuration."""

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    model_family = payload.get("model_family")
    model_config = payload.get("config") or {}
    metadata = payload.get("metadata") or {}

    return _call_service(MODELS_SERVICE.evaluate_requirements, model_family, model_config, metadata)
