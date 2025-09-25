"""
Model information routes for SimpleTuner server.
Provides model families and flavours information.
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("ModelRoutes")

router = APIRouter(prefix="/api/models")


@router.get("")
async def get_model_families():
    """Get all available model families."""
    try:
        from simpletuner.helpers.models.all import model_families

        return {"families": list(model_families.keys())}
    except ImportError as e:
        logger.error(f"Failed to import model families: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model families")


@router.get("/{model_family}/flavours")
async def get_model_flavours(model_family: str):
    """Get available flavours for a specific model family."""
    try:
        from simpletuner.helpers.models.all import get_model_flavour_choices, model_families

        if model_family not in model_families:
            raise HTTPException(status_code=404, detail=f"Model family '{model_family}' not found")

        flavours = get_model_flavour_choices(model_family)
        return {"flavours": list(flavours)}

    except ImportError as e:
        logger.error(f"Failed to import model utilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model information")
