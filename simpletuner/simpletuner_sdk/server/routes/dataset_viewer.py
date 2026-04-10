"""Dataset viewer and scan routes - browsing, thumbnails, caption validation,
filtering analysis, conditioning visualization, media preview, standalone scanning,
and cache operations (text embeds, VAE, conditioning)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, conlist

from simpletuner.simpletuner_sdk.server.services.cache_job_service import get_cache_service
from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import get_scan_service
from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import (
    AudioPreview,
    CaptionValidationResult,
    DatasetFilePage,
    DatasetGraph,
    DatasetSummary,
    DatasetViewerService,
    FilteredFilesReport,
    OrphanReport,
    VideoPreview,
    ViewerThumbnailInfo,
)

router = APIRouter(prefix="/api/datasets", tags=["dataset_viewer"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_viewer_service() -> DatasetViewerService:
    return DatasetViewerService()


def _load_plan() -> List[Dict[str, Any]]:
    """Load the current dataset plan via the same store as the datasets module."""
    from .datasets import _store

    store = _store()
    datasets, _, _ = store.load()
    return datasets


def _get_dataset_config_by_id(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Lookup a single dataset config by ID from the current plan."""
    try:
        datasets = _load_plan()
    except ValueError:
        return None
    return next((d for d in datasets if d.get("id") == dataset_id), None)


def _require_dataset_config(dataset_id: str) -> Dict[str, Any]:
    """Return the config or raise 404."""
    config = _get_dataset_config_by_id(dataset_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found in plan")
    return config


def _get_global_config() -> Dict[str, Any]:
    """Get global training config values relevant to scanning."""
    try:
        from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService

        return ConfigsService().get_active_config().get("config", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Viewer endpoints (Stage 1)
# ---------------------------------------------------------------------------


@router.get("/viewer/summary", response_model=DatasetSummary)
async def get_viewer_summary(dataset_id: str, _user: User = Depends(get_current_user)) -> DatasetSummary:
    """Summary of a dataset's cached bucket data."""
    return _get_viewer_service().get_dataset_summary(_require_dataset_config(dataset_id))


@router.get("/viewer/summaries")
async def get_viewer_summaries(
    _user: User = Depends(get_current_user),
) -> List[DatasetSummary]:
    """Summaries for all browsable datasets in the current plan."""
    try:
        datasets = _load_plan()
    except ValueError:
        return []

    service = _get_viewer_service()
    return [
        service.get_dataset_summary(ds)
        for ds in datasets
        if str(ds.get("dataset_type", "")).lower() not in ("text_embeds", "image_embeds")
    ]


@router.get("/viewer/files", response_model=DatasetFilePage)
async def get_viewer_files(
    dataset_id: str,
    bucket_key: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _user: User = Depends(get_current_user),
) -> DatasetFilePage:
    """Paginated file listing from a dataset's bucket cache."""
    return _get_viewer_service().get_dataset_files(
        _require_dataset_config(dataset_id), bucket_key=bucket_key, limit=limit, offset=offset
    )


@router.get("/viewer/file-metadata")
async def get_viewer_file_metadata(
    dataset_id: str, file_path: str, _user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Per-file metadata from the metadata cache."""
    meta = _get_viewer_service().get_file_metadata(_require_dataset_config(dataset_id), file_path)
    if meta is None:
        return {"path": file_path, "found": False}
    return {"found": True, **meta.model_dump()}


@router.get("/viewer/thumbnails", response_model=List[ViewerThumbnailInfo])
async def get_viewer_thumbnails(
    dataset_id: str,
    bucket_key: Optional[str] = None,
    limit: int = 24,
    offset: int = 0,
    _user: User = Depends(get_current_user),
) -> List[ViewerThumbnailInfo]:
    """Thumbnails for files in a dataset bucket."""
    return _get_viewer_service().get_dataset_thumbnails(
        _require_dataset_config(dataset_id), bucket_key=bucket_key, limit=limit, offset=offset
    )


@router.get("/viewer/preview")
async def get_viewer_preview(
    dataset_id: str,
    file_path: str,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Preview images for the file detail modal.

    Returns original, intermediary (resized), and cropped images to
    simulate the TrainingSample.prepare() pipeline.
    """
    service = _get_viewer_service()
    config = _require_dataset_config(dataset_id)

    # Load metadata so the preview can simulate the crop pipeline
    meta = service.get_file_metadata(config, file_path)
    meta_dict = meta.model_dump() if meta else None

    result = service.get_image_preview(config, file_path, metadata=meta_dict)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not available")
    return result


class CropCoordinatesUpdate(BaseModel):
    dataset_id: str
    file_path: str
    crop_coordinates: conlist(int, min_length=2, max_length=2)


@router.patch("/viewer/crop-coordinates")
async def update_viewer_crop_coordinates(
    request: CropCoordinatesUpdate, _user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update crop coordinates for a file and return a re-cropped preview."""
    service = _get_viewer_service()
    config = _require_dataset_config(request.dataset_id)

    if not service.update_crop_coordinates(config, request.file_path, request.crop_coordinates):
        raise HTTPException(status_code=404, detail="Metadata entry not found")

    # Re-generate the cropped preview with the new coordinates
    meta = service.get_file_metadata(config, request.file_path)
    meta_dict = meta.model_dump() if meta else None
    result = service.get_image_preview(config, request.file_path, metadata=meta_dict)

    return {
        "saved": True,
        "crop_coordinates": request.crop_coordinates,
        "cropped": result.get("cropped") if result else None,
    }


class BboxEntitiesUpdate(BaseModel):
    dataset_id: str
    file_path: str
    bbox_entities: Optional[List[Dict[str, Any]]] = None


@router.patch("/viewer/bbox-entities")
async def update_viewer_bbox_entities(
    request: BboxEntitiesUpdate, _user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update bounding box entities for a file in the metadata cache."""
    service = _get_viewer_service()
    config = _require_dataset_config(request.dataset_id)

    if not service.update_bbox_entities(config, request.file_path, request.bbox_entities):
        raise HTTPException(status_code=404, detail="Metadata entry not found")

    return {
        "saved": True,
        "bbox_entities": request.bbox_entities,
    }


class SingleFileAction(BaseModel):
    dataset_id: str
    file_path: str


@router.post("/viewer/rebuild-metadata")
async def rebuild_file_metadata(request: SingleFileAction, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Recalculate metadata for a single file using current dataset config."""
    service = _get_viewer_service()
    config = _require_dataset_config(request.dataset_id)
    global_config = _get_global_config()

    result = service.rebuild_file_metadata(config, global_config, request.file_path)
    if result is None:
        raise HTTPException(status_code=404, detail="Could not rebuild metadata for this file")

    return {"rebuilt": True, "metadata": result}


@router.post("/viewer/delete-vae-cache")
async def delete_vae_cache_file(request: SingleFileAction, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Delete the VAE cache file for a single image/video."""
    service = _get_viewer_service()
    config = _require_dataset_config(request.dataset_id)
    return service.delete_vae_cache_file(config, request.file_path)


# ---------------------------------------------------------------------------
# Caption validation (Stage 3)
# ---------------------------------------------------------------------------


@router.get("/viewer/caption-status", response_model=CaptionValidationResult)
async def get_viewer_caption_status(dataset_id: str, _user: User = Depends(get_current_user)) -> CaptionValidationResult:
    """Validate caption coverage for a dataset using its caption strategy."""
    return _get_viewer_service().get_caption_status(_require_dataset_config(dataset_id))


# ---------------------------------------------------------------------------
# Size filter analysis (Stage 4)
# ---------------------------------------------------------------------------


@router.get("/viewer/filter-summary")
async def get_viewer_filter_summary(dataset_id: str, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Filtering statistics from the bucket cache."""
    return _get_viewer_service().get_filtering_summary(_require_dataset_config(dataset_id))


@router.get("/viewer/filtered", response_model=FilteredFilesReport)
async def get_viewer_filtered_files(
    dataset_id: str,
    reason: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _user: User = Depends(get_current_user),
) -> FilteredFilesReport:
    """Files that were filtered out by size/aspect ratio settings."""
    return _get_viewer_service().get_filtered_files(
        _require_dataset_config(dataset_id), reason=reason, limit=limit, offset=offset
    )


# ---------------------------------------------------------------------------
# Conditioning dataset visualization (Stage 5)
# ---------------------------------------------------------------------------


@router.get("/viewer/graph", response_model=DatasetGraph)
async def get_viewer_graph(_user: User = Depends(get_current_user)) -> DatasetGraph:
    """Dependency graph of all datasets."""
    try:
        datasets = _load_plan()
    except ValueError:
        return DatasetGraph()
    return _get_viewer_service().get_dataset_graph(datasets)


@router.get("/viewer/conditioning-pairs")
async def get_viewer_conditioning_pairs(
    source_id: str,
    conditioning_id: str,
    limit: int = 12,
    offset: int = 0,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Matched conditioning pairs between source and conditioning datasets."""
    return _get_viewer_service().get_conditioning_pairs(
        _require_dataset_config(source_id),
        _require_dataset_config(conditioning_id),
        limit=limit,
        offset=offset,
    )


@router.get("/viewer/conditioning-match")
async def get_viewer_conditioning_match(
    dataset_id: str,
    file_path: str,
    _user: User = Depends(get_current_user),
) -> Optional[Dict[str, Any]]:
    """Find the conditioning image matching a source file."""
    try:
        datasets = _load_plan()
    except ValueError:
        return None
    config = _require_dataset_config(dataset_id)
    return _get_viewer_service().get_conditioning_file_match(config, datasets, file_path)


@router.get("/viewer/conditioning-orphans", response_model=OrphanReport)
async def get_viewer_conditioning_orphans(
    source_id: str, conditioning_id: str, _user: User = Depends(get_current_user)
) -> OrphanReport:
    """Orphaned files between source and conditioning datasets."""
    return _get_viewer_service().get_orphaned_files(
        _require_dataset_config(source_id),
        _require_dataset_config(conditioning_id),
    )


# ---------------------------------------------------------------------------
# Audio/Video preview (Stage 6)
# ---------------------------------------------------------------------------


@router.get("/viewer/video-preview", response_model=VideoPreview)
async def get_viewer_video_preview(
    dataset_id: str, file_path: str, frame_count: int = 4, _user: User = Depends(get_current_user)
) -> VideoPreview:
    """Video preview with extracted frames."""
    return _get_viewer_service().get_video_preview(_require_dataset_config(dataset_id), file_path, frame_count=frame_count)


@router.get("/viewer/audio-preview", response_model=AudioPreview)
async def get_viewer_audio_preview(dataset_id: str, file_path: str, _user: User = Depends(get_current_user)) -> AudioPreview:
    """Audio preview with waveform."""
    return _get_viewer_service().get_audio_preview(_require_dataset_config(dataset_id), file_path)


# ---------------------------------------------------------------------------
# Standalone scan (Stage 2)
# ---------------------------------------------------------------------------


class ScanRequest(BaseModel):
    dataset_id: str
    force_rescan: bool = False
    clear_vae_cache: bool = False
    clear_conditioning_cache: bool = False


class ScanAllRequest(BaseModel):
    force_rescan: bool = False


@router.post("/scan")
async def start_dataset_scan(request: ScanRequest, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Start a metadata scan for a single dataset."""
    config = _require_dataset_config(request.dataset_id)
    global_config = _get_global_config()
    try:
        job_id = get_scan_service().scan_dataset(
            request.dataset_id,
            config,
            global_config,
            force_rescan=request.force_rescan,
            clear_vae_cache=request.clear_vae_cache,
            clear_conditioning_cache=request.clear_conditioning_cache,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job_id, "dataset_id": request.dataset_id}


@router.post("/scan/all")
async def start_scan_all(request: ScanAllRequest, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Queue all browsable datasets for sequential scanning."""
    try:
        datasets = _load_plan()
    except ValueError:
        raise HTTPException(status_code=404, detail="No dataset plan found")

    browsable = [ds for ds in datasets if str(ds.get("dataset_type", "")).lower() not in ("text_embeds", "image_embeds")]
    if not browsable:
        raise HTTPException(status_code=400, detail="No scannable datasets in plan")

    try:
        queue_id = get_scan_service().scan_all(browsable, _get_global_config(), force_rescan=request.force_rescan)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"queue_id": queue_id, "dataset_count": len(browsable)}


@router.get("/scan/active")
async def get_active_scan(
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Current active scan status, if any. Returns {active: false} when idle."""
    result = get_scan_service().get_active_status()
    if result is None:
        return {"active": False}
    return {"active": True, **result}


@router.get("/scan/status")
async def get_scan_status_endpoint(job_id: str, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Status of a scan job."""
    result = get_scan_service().get_scan_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scan job '{job_id}' not found")
    return result


@router.get("/scan/queue-status")
async def get_scan_queue_status(queue_id: str, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Status of a scan queue."""
    result = get_scan_service().get_queue_status(queue_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scan queue '{queue_id}' not found")
    return result


@router.post("/scan/cancel")
async def cancel_scan(_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Cancel the active scan and remaining queue."""
    return {"cancelled": get_scan_service().cancel_scan()}


# ---------------------------------------------------------------------------
# Cache operations (text embeds, VAE, conditioning)
# ---------------------------------------------------------------------------


class CacheStartRequest(BaseModel):
    dataset_id: str
    cache_type: Literal["text_embeds", "vae", "conditioning"]


@router.get("/cache/capabilities")
async def get_cache_capabilities(_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Available cache types based on model family and dataset configuration."""
    global_config = _get_global_config()
    model_family = global_config.get("model_family", "")
    try:
        datasets = _load_plan()
    except ValueError:
        datasets = []

    from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService

    return CacheJobService.get_capabilities(model_family, datasets, global_config)


@router.post("/cache/start")
async def start_cache_job(request: CacheStartRequest, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Start a background cache job for a dataset."""
    config = _require_dataset_config(request.dataset_id)
    global_config = _get_global_config()
    try:
        job_id = get_cache_service().start_cache_job(
            request.dataset_id,
            request.cache_type,
            config,
            global_config,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job_id, "dataset_id": request.dataset_id, "cache_type": request.cache_type}


@router.get("/cache/active")
async def get_active_cache_job(_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Current active cache job status, if any."""
    result = get_cache_service().get_active_status()
    if result is None:
        return {"active": False}
    return {"active": True, **result}


@router.get("/cache/status")
async def get_cache_job_status(job_id: str, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Status of a cache job."""
    result = get_cache_service().get_job_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Cache job '{job_id}' not found")
    return result


@router.post("/cache/cancel")
async def cancel_cache_job(_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Cancel the active cache job."""
    return {"cancelled": get_cache_service().cancel()}
