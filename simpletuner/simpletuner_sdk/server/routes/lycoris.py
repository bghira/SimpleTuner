"""LyCORIS metadata routes for powering the WebUI builder."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from simpletuner.simpletuner_sdk.server.services.lycoris_builder_service import LYCORIS_BUILDER_SERVICE

router = APIRouter(prefix="/api/lycoris", tags=["lycoris"])


@router.get("/metadata")
async def get_lycoris_metadata(force_refresh: bool = False) -> Dict[str, Any]:
    """Expose LyCORIS algorithm/preset metadata for the UI."""
    return LYCORIS_BUILDER_SERVICE.get_metadata(force_refresh=force_refresh)
