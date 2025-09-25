"""Refactored Web UI routes using service layer and shared dependencies."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..dependencies.common import (
    get_config_data,
    get_webui_defaults,
    get_field_registry,
    get_tab_fields,
    TabFieldsDependency,
)
from ..services.tab_service import TabService
from ..services.field_service import FieldService, FieldFormat
from ..services.config_store import ConfigStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["web"])

# Get template directory from environment
template_dir = os.environ.get("TEMPLATE_DIR", "templates")
templates = Jinja2Templates(directory=template_dir)

# Initialize services
tab_service = TabService(templates)
field_service = FieldService()


@router.get("/trainer", response_class=HTMLResponse)
async def trainer_page(request: Request):
    """Main trainer page."""
    # Get available tabs
    tabs = tab_service.get_all_tabs()

    # Get config store for metadata
    config_store = ConfigStore()
    configs = config_store.list_configs()
    active_config = config_store.get_active_config()

    context = {
        "request": request,
        "page_title": "SimpleTuner Training Interface",
        "tabs": tabs,
        "configs": [c.to_dict() for c in configs],
        "active_config": active_config,
    }

    return templates.TemplateResponse("web/trainer_htmx.html", context)


@router.get("/trainer/tabs/{tab_name}", response_class=HTMLResponse)
async def render_tab(
    request: Request,
    tab_name: str,
    field_registry = Depends(get_field_registry),
    config_data: Dict[str, Any] = Depends(get_config_data),
    webui_defaults: Dict[str, Any] = Depends(get_webui_defaults)
):
    """Unified tab handler using TabService.

    This single endpoint replaces all individual tab handlers
    by using the TabService to handle tab-specific logic.
    """
    logger.debug(f"=== RENDERING TAB: {tab_name} ===")

    try:
        # Get fields for the tab
        tab_fields = field_registry.get_fields_for_tab(tab_name)
        logger.debug(f"Retrieved {len(tab_fields)} fields for '{tab_name}' tab")

        # Build config values including webui defaults
        config_values = {}
        for field in tab_fields:
            if field.name == "output_dir" and tab_name == "basic":
                # Special handling for output_dir in basic tab
                config_values[field.name] = config_data.get(
                    field.name,
                    webui_defaults.get("output_dir", "")
                )
            else:
                config_values[field.name] = config_data.get(
                    field.name,
                    field.default_value
                )

        # Add webui-specific values for basic tab
        if tab_name == "basic":
            config_values["configs_dir"] = webui_defaults.get("configs_dir", "")
            config_values["job_id"] = config_data.get("job_id", "")

        # Convert fields to template format
        formatted_fields = field_service.convert_fields(
            tab_fields,
            FieldFormat.TEMPLATE,
            config_values
        )

        # Get sections if available
        sections = field_registry.get_sections_for_tab(tab_name)

        # Render tab using TabService
        return await tab_service.render_tab(
            request=request,
            tab_name=tab_name,
            fields=formatted_fields,
            config_values=config_values,
            sections=sections
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering tab '{tab_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render tab: {str(e)}"
        )


# Alternative approach using dependency injection for fields
@router.get("/trainer/tabs/v2/{tab_name}", response_class=HTMLResponse)
async def render_tab_v2(
    request: Request,
    tab_name: str,
    fields: list = Depends(lambda tab=tab_name: get_tab_fields(tab)),
    config_data: Dict[str, Any] = Depends(get_config_data),
    webui_defaults: Dict[str, Any] = Depends(get_webui_defaults)
):
    """Alternative tab handler using field dependency injection.

    This approach uses the TabFieldsDependency to get pre-formatted fields.
    """
    logger.debug(f"=== RENDERING TAB V2: {tab_name} ===")

    try:
        # Get sections
        field_registry = await get_field_registry()
        sections = field_registry.get_sections_for_tab(tab_name)

        # Build config values for tab context
        config_values = config_data.copy()
        if tab_name == "basic":
            config_values.update({
                "configs_dir": webui_defaults.get("configs_dir", ""),
                "output_dir": webui_defaults.get("output_dir", config_values.get("output_dir", "")),
                "job_id": config_values.get("job_id", "")
            })

        # Render tab
        return await tab_service.render_tab(
            request=request,
            tab_name=tab_name,
            fields=fields,
            config_values=config_values,
            sections=sections
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering tab v2 '{tab_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render tab: {str(e)}"
        )


# Utility endpoints
@router.get("/trainer/config-selector", response_class=HTMLResponse)
async def config_selector(request: Request):
    """Config selector fragment for HTMX."""
    config_store = ConfigStore()
    configs = config_store.list_configs()
    active_config = config_store.get_active_config()

    context = {
        "request": request,
        "configs": [c.to_dict() for c in configs],
        "active_config": active_config,
    }

    return templates.TemplateResponse("web/fragments/config_selector.html", context)


@router.get("/trainer/tab-list", response_class=HTMLResponse)
async def tab_list(request: Request):
    """Tab list fragment for HTMX."""
    tabs = tab_service.get_all_tabs()

    context = {
        "request": request,
        "tabs": tabs,
    }

    return templates.TemplateResponse("web/fragments/tab_list.html", context)


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