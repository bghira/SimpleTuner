"""Refactored Web UI routes using service layer and shared dependencies."""

from __future__ import annotations

import os
import logging

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..dependencies.common import (
    get_config_store,
    get_tab_render_data,
    TabRenderData,
)
from ..services.tab_service import TabService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["web"])

# Get template directory from environment
template_dir = os.environ.get("TEMPLATE_DIR", "templates")
templates = Jinja2Templates(directory=template_dir)

# Initialize services
tab_service = TabService(templates)


@router.get("/trainer", response_class=HTMLResponse)
async def trainer_page(
    request: Request,
    config_store = Depends(get_config_store),
):
    """Main trainer page."""
    # Get available tabs
    tabs = tab_service.get_all_tabs()

    configs = config_store.list_configs()
    active_config = config_store.get_active_config()

    context = {
        "request": request,
        "page_title": "SimpleTuner Training Interface",
        "tabs": tabs,
        "configs": [c.to_dict() if not isinstance(c, dict) else c for c in configs],
        "active_config": active_config,
    }

    return templates.TemplateResponse("trainer_htmx.html", context)


@router.get("/trainer/tabs/{tab_name}", response_class=HTMLResponse)
async def render_tab(
    request: Request,
    tab_name: str,
    tab_data: TabRenderData = Depends(get_tab_render_data),
):
    """Unified tab handler using TabService.

    This single endpoint replaces all individual tab handlers
    by using the TabService to handle tab-specific logic.
    """
    logger.debug(f"=== RENDERING TAB: {tab_name} ===")

    try:
        # Render tab using TabService
        return await tab_service.render_tab(
            request=request,
            tab_name=tab_name,
            fields=tab_data.fields,
            config_values=tab_data.config_values,
            sections=tab_data.sections,
            raw_config=tab_data.raw_config,
            webui_defaults=tab_data.webui_defaults,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering tab '{tab_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render tab: {str(e)}"
        )


# Utility endpoints
@router.get("/trainer/config-selector", response_class=HTMLResponse)
async def config_selector(
    request: Request,
    config_store = Depends(get_config_store),
):
    """Config selector fragment for HTMX."""
    configs = config_store.list_configs()
    active_config = config_store.get_active_config()

    context = {
        "request": request,
        "configs": [c.to_dict() for c in configs],
        "active_config": active_config,
    }

    return templates.TemplateResponse("fragments/config_selector.html", context)


@router.get("/trainer/tab-list", response_class=HTMLResponse)
async def tab_list(request: Request):
    """Tab list fragment for HTMX."""
    tabs = tab_service.get_all_tabs()

    context = {
        "request": request,
        "tabs": tabs,
    }

    return templates.TemplateResponse("fragments/tab_list.html", context)


# Backward compatibility redirects
@router.get("/trainer/tabs/basic", response_class=HTMLResponse)
async def basic_tab_redirect(request: Request):
    """Redirect old basic tab URL to new unified handler."""
    return await render_tab(request, "basic")


@router.get("/trainer/tabs/model", response_class=HTMLResponse)
async def model_tab_redirect(request: Request):
    """Redirect old model tab URL to new unified handler."""
    return await render_tab(request, "model")


@router.get("/trainer/tabs/training", response_class=HTMLResponse)
async def training_tab_redirect(request: Request):
    """Redirect old training tab URL to new unified handler."""
    return await render_tab(request, "training")


@router.get("/trainer/tabs/advanced", response_class=HTMLResponse)
async def advanced_tab_redirect(request: Request):
    """Redirect old advanced tab URL to new unified handler."""
    return await render_tab(request, "advanced")


@router.get("/trainer/tabs/datasets", response_class=HTMLResponse)
async def datasets_tab_redirect(request: Request):
    """Redirect old datasets tab URL to new unified handler."""
    return await render_tab(request, "datasets")


@router.get("/trainer/tabs/environments", response_class=HTMLResponse)
async def environments_tab_redirect(request: Request):
    """Redirect old environments tab URL to new unified handler."""
    return await render_tab(request, "environments")


@router.get("/trainer/tabs/validation", response_class=HTMLResponse)
async def validation_tab_redirect(request: Request):
    """Redirect old validation tab URL to new unified handler."""
    return await render_tab(request, "validation")
