"""API routes for theme management."""

import mimetypes
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..services.theme_service import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_SOUND_EXTENSIONS, ThemeService

router = APIRouter(prefix="/api/themes", tags=["themes"])

# MIME types for theme assets
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}

SOUND_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


@router.get("", response_model=List[Dict])
async def list_themes() -> List[Dict]:
    """List all available themes."""
    service = ThemeService.get_instance()
    return service.list_for_ui()


@router.get("/{theme_id}")
async def get_theme(theme_id: str) -> Dict:
    """Get theme metadata by ID."""
    service = ThemeService.get_instance()
    theme = service.get_theme(theme_id)
    if theme is None:
        raise HTTPException(status_code=404, detail=f"Theme '{theme_id}' not found")
    return {
        "id": theme.id,
        "name": theme.name,
        "description": theme.description,
        "author": theme.author,
        "source": theme.source,
        "has_css": theme.css_path is not None and theme.css_path.exists(),
        "has_assets": bool(theme.assets.images or theme.assets.sounds),
    }


@router.get("/{theme_id}/manifest")
async def get_theme_manifest(theme_id: str) -> Dict:
    """Get theme manifest with asset URLs.

    Returns theme metadata along with URLs for all declared assets.
    JavaScript can use this to configure sounds and images dynamically.
    """
    service = ThemeService.get_instance()
    manifest = service.get_theme_manifest(theme_id)

    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Theme '{theme_id}' not found")

    return manifest


@router.get("/{theme_id}/theme.css")
async def get_theme_css(theme_id: str) -> FileResponse:
    """Serve theme CSS file."""
    service = ThemeService.get_instance()
    theme = service.get_theme(theme_id)

    if theme is None:
        raise HTTPException(status_code=404, detail=f"Theme '{theme_id}' not found")

    if theme.css_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Theme '{theme_id}' is a built-in theme with no separate CSS file",
        )

    if not theme.css_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Theme CSS file not found at {theme.css_path}",
        )

    return FileResponse(
        theme.css_path,
        media_type="text/css",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{theme_id}/assets/images/{asset_name}")
async def get_theme_image(theme_id: str, asset_name: str) -> FileResponse:
    """Serve a theme image asset.

    Args:
        theme_id: The theme identifier
        asset_name: Asset name as declared in theme manifest (without extension)

    Security:
        - Asset name validated against safe pattern
        - Path traversal prevented
        - File extension whitelisted
        - Asset must be declared in theme manifest
    """
    service = ThemeService.get_instance()

    # Get the sanitized asset path
    asset_path = service.get_asset_path(theme_id, "images", asset_name)

    if asset_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image asset '{asset_name}' not found for theme '{theme_id}'",
        )

    # Determine MIME type
    suffix = asset_path.suffix.lower()
    mime_type = IMAGE_MIME_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(
        asset_path,
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache images for 24 hours
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{theme_id}/assets/sounds/{asset_name}")
async def get_theme_sound(theme_id: str, asset_name: str) -> FileResponse:
    """Serve a theme sound asset.

    Args:
        theme_id: The theme identifier
        asset_name: Asset name as declared in theme manifest (without extension)

    Security:
        - Asset name validated against safe pattern
        - Path traversal prevented
        - File extension whitelisted
        - Asset must be declared in theme manifest
    """
    service = ThemeService.get_instance()

    # Get the sanitized asset path
    asset_path = service.get_asset_path(theme_id, "sounds", asset_name)

    if asset_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sound asset '{asset_name}' not found for theme '{theme_id}'",
        )

    # Determine MIME type
    suffix = asset_path.suffix.lower()
    mime_type = SOUND_MIME_TYPES.get(suffix, "audio/wav")

    return FileResponse(
        asset_path,
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache sounds for 24 hours
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{theme_id}/assets")
async def list_theme_assets(theme_id: str) -> Dict:
    """List all assets declared by a theme.

    Returns lists of asset names for images and sounds.
    """
    service = ThemeService.get_instance()
    assets = service.list_theme_assets(theme_id)

    if assets is None:
        raise HTTPException(status_code=404, detail=f"Theme '{theme_id}' not found")

    return assets


@router.post("/refresh")
async def refresh_themes() -> Dict:
    """Refresh the theme cache and return updated list."""
    service = ThemeService.get_instance()
    service.invalidate_cache()
    return {"status": "ok", "themes": service.list_for_ui()}
