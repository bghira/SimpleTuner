"""System status API endpoints."""

from fastapi import APIRouter

from simpletuner.simpletuner_sdk.server.services.system_status_service import SystemStatusService

router = APIRouter(prefix="/api/system", tags=["system"])
_service = SystemStatusService()


@router.get("/status")
async def get_system_status() -> dict:
    """Return current system load, memory usage, and GPU utilisation."""
    return _service.get_status()
