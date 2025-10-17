"""System status API endpoints."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from simpletuner.simpletuner_sdk.server.services.maintenance_service import MAINTENANCE_SERVICE, MaintenanceServiceError
from simpletuner.simpletuner_sdk.server.services.system_status_service import SystemStatusService

router = APIRouter(prefix="/api/system", tags=["system"])
_service = SystemStatusService()


@router.get("/status")
async def get_system_status() -> dict:
    """Return current system load, memory usage, and GPU utilisation."""
    return _service.get_status()


@router.post("/maintenance/clear-fsdp-block-cache")
async def clear_fsdp_block_cache() -> Dict[str, Any]:
    """Clear cached FSDP detection metadata."""
    try:
        return MAINTENANCE_SERVICE.clear_fsdp_block_cache()
    except MaintenanceServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post("/maintenance/clear-deepspeed-offload")
async def clear_deepspeed_offload(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Delete DeepSpeed NVMe offload cache directories."""
    config_name = None
    if isinstance(payload, dict):
        config_name = payload.get("config") or payload.get("config_name")
    try:
        return MAINTENANCE_SERVICE.clear_deepspeed_offload_cache(config_name=config_name)
    except MaintenanceServiceError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
