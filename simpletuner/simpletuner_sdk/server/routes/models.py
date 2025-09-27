"""Model information routes for SimpleTuner server."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from simpletuner.simpletuner_sdk.server.services.models_service import (
    MODELS_SERVICE,
    ModelServiceError,
)

router = APIRouter(prefix="/api/models")


def _call_service(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ModelServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("")
async def get_model_families():
    """Return all available model families."""
    return _call_service(MODELS_SERVICE.list_families)


@router.get("/{model_family}")
async def get_model_details(model_family: str):
    """Return metadata for a specific model family."""
    return _call_service(MODELS_SERVICE.get_model_details, model_family)


@router.get("/{model_family}/flavours")
async def get_model_flavours(model_family: str):
    """Return the flavours for a specific model family."""
    result = _call_service(MODELS_SERVICE.get_model_flavours, model_family)
    return {"flavours": result.get("flavours", [])}


@router.get("/{model_family}/flavours-select", response_class=HTMLResponse)
async def get_model_flavours_html(model_family: str, current_value: str = ""):
    """Return flavours as HTML <option> elements for HTMX usage."""
    if model_family == "loading":
        return """
        <option value="">Loading flavours...</option>
        """

    try:
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
        selected = 'selected' if flavour == current_value else ''
        options.append(f'<option value="{flavour}" {selected}>{flavour}</option>')

    return "\n".join(options)
