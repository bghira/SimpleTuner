"""API routes for caption filter management."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.caption_filters_service import CAPTION_FILTERS_SERVICE, CaptionFilterError

router = APIRouter(prefix="/api/caption-filters", tags=["caption-filters"])


class CaptionFilterRequest(BaseModel):
    """Payload for creating a caption filter."""

    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    entries: List[str] = Field(default_factory=list)


class CaptionFilterUpdateRequest(BaseModel):
    """Payload for updating an existing caption filter."""

    name: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    entries: Optional[List[str]] = None


class CaptionFilterTestRequest(BaseModel):
    """Payload for testing caption filter behaviour."""

    entries: List[str] = Field(default_factory=list)
    sample: str


def _call_service(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except CaptionFilterError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("/")
async def list_caption_filters() -> Dict[str, Any]:
    """Return available caption filters."""

    filters = [record.to_public_dict() for record in CAPTION_FILTERS_SERVICE.list_filters()]
    return {"filters": filters, "count": len(filters)}


@router.post("/test")
async def test_caption_filter(request: CaptionFilterTestRequest) -> Dict[str, str]:
    """Apply filters to the provided sample without persisting."""

    output = CAPTION_FILTERS_SERVICE.test_entries(request.entries, request.sample)
    return {"output": output}


@router.get("/{name}")
async def get_caption_filter(name: str) -> Dict[str, Any]:
    """Retrieve a specific caption filter definition."""

    record = _call_service(CAPTION_FILTERS_SERVICE.get_filter, name)
    return {"filter": record.to_public_dict()}


@router.post("/")
async def create_caption_filter(request: CaptionFilterRequest) -> Dict[str, Any]:
    """Create a new caption filter."""

    record = _call_service(CAPTION_FILTERS_SERVICE.create_filter, request.model_dump())
    return {"filter": record.to_public_dict()}


@router.put("/{name}")
async def update_caption_filter(name: str, request: CaptionFilterUpdateRequest) -> Dict[str, Any]:
    """Update an existing caption filter."""

    payload = request.model_dump(exclude_unset=True)
    record = _call_service(CAPTION_FILTERS_SERVICE.update_filter, name, payload)
    return {"filter": record.to_public_dict()}


@router.delete("/{name}")
async def delete_caption_filter(name: str) -> Dict[str, Any]:
    """Delete a caption filter."""

    _call_service(CAPTION_FILTERS_SERVICE.delete_filter, name)
    return {"status": "deleted", "name": name}
