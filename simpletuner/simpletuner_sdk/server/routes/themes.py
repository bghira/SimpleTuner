"""API routes for theme management."""

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..services.theme_service import ThemeService

router = APIRouter(prefix="/api/themes", tags=["themes"])


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
    }


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


@router.post("/refresh")
async def refresh_themes() -> Dict:
    """Refresh the theme cache and return updated list."""
    service = ThemeService.get_instance()
    service.invalidate_cache()
    return {"status": "ok", "themes": service.list_for_ui()}
