"""Model information routes for SimpleTuner server."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger("ModelRoutes")

router = APIRouter(prefix="/api/models")


@router.get("")
async def get_model_families():
    """Return all available model families."""
    try:
        from simpletuner.helpers.models.all import model_families
    except ImportError as exc:
        logger.error("Failed to import model families: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load model families")

    return {"families": list(model_families.keys())}


@router.get("/{model_family}/flavours")
async def get_model_flavours(model_family: str):
    """Return the flavours for a specific model family."""
    try:
        from simpletuner.helpers.models.all import get_model_flavour_choices, model_families
    except ImportError as exc:
        logger.error("Failed to import model utilities: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load model information")

    if model_family not in model_families:
        raise HTTPException(status_code=404, detail=f"Model family '{model_family}' not found")

    flavours = list(get_model_flavour_choices(model_family))
    return {"flavours": flavours}


@router.get("/{model_family}/flavours-select", response_class=HTMLResponse)
async def get_model_flavours_html(model_family: str, current_value: str = ""):
    """Return flavours as HTML <option> elements for HTMX usage."""
    try:
        from simpletuner.helpers.models.all import get_model_flavour_choices, model_families
    except ImportError as exc:
        logger.error("Failed to import model utilities: %s", exc)
        return """
        <option value="">Error loading flavours</option>
        """

    if model_family == "loading":
        return """
        <option value="">Loading flavours...</option>
        """

    if model_family not in model_families:
        return """
        <option value="">Select a model family first</option>
        """

    flavours = get_model_flavour_choices(model_family)

    options = ['<option value="">Default</option>']
    for flavour in flavours:
        selected = 'selected' if flavour == current_value else ''
        options.append(f'<option value="{flavour}" {selected}>{flavour}</option>')

    return "\n".join(options)
