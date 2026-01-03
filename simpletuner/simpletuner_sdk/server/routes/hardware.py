"""Hardware discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.hardware_service import detect_gpu_inventory, detect_system_memory

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("/gpus")
async def get_gpu_inventory(_user: User = Depends(get_current_user)) -> dict:
    """Return the detected GPU inventory."""

    return detect_gpu_inventory()


@router.get("/memory")
async def get_system_memory(_user: User = Depends(get_current_user)) -> dict:
    """Return system memory information."""

    return detect_system_memory()
