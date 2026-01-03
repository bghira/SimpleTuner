"""LyCORIS metadata routes for powering the WebUI builder."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.lycoris_builder_service import LYCORIS_BUILDER_SERVICE

router = APIRouter(prefix="/api/lycoris", tags=["lycoris"])


@router.get("/metadata")
async def get_lycoris_metadata(force_refresh: bool = False, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Expose LyCORIS algorithm/preset metadata for the UI."""
    return LYCORIS_BUILDER_SERVICE.get_metadata(force_refresh=force_refresh)
