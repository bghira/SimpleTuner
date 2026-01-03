"""Routes for managing validation prompt libraries."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.prompt_library_service import PromptLibraryError, PromptLibraryService

router = APIRouter(prefix="/api/prompt-libraries", tags=["prompt_libraries"])


class PromptLibraryEntryModel(BaseModel):
    prompt: str
    adapter_strength: Optional[float] = None


class PromptLibraryPayload(BaseModel):
    entries: Dict[str, Union[str, PromptLibraryEntryModel]]
    previous_filename: Optional[str] = None


def _call_service(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except PromptLibraryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def _get_service() -> PromptLibraryService:
    return PromptLibraryService()


@router.get("/")
async def list_prompt_libraries(_user: User = Depends(get_current_user)) -> Dict[str, object]:
    service = _get_service()
    records = _call_service(service.list_libraries)
    return {"libraries": [asdict(record) for record in records], "count": len(records)}


@router.get("/{filename}")
async def get_prompt_library(filename: str, _user: User = Depends(get_current_user)) -> Dict[str, object]:
    service = _get_service()
    result = _call_service(service.read_library, filename)
    return {"entries": result["entries"], "library": asdict(result["library"])}


@router.put("/{filename}")
async def save_prompt_library(
    filename: str, payload: PromptLibraryPayload, _user: User = Depends(get_current_user)
) -> Dict[str, object]:
    service = _get_service()
    entries: Dict[str, object] = {}
    for key, value in payload.entries.items():
        if isinstance(value, PromptLibraryEntryModel):
            entries[key] = value.model_dump(exclude_none=True)
        else:
            entries[key] = value
    result = _call_service(service.save_library, filename, entries, payload.previous_filename)
    return {"entries": result["entries"], "library": asdict(result["library"])}
