"""Web UI routes for HTMX enhanced trainer interface."""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from simpletuner.helpers.models.all import model_families
from simpletuner.simpletuner_sdk.server.services.field_registry import field_registry

router = APIRouter(prefix="/web", tags=["web"])

# Get template directory from environment
template_dir = os.environ.get("TEMPLATE_DIR", "templates")
templates = Jinja2Templates(directory=template_dir)


def get_model_family_label(model_key: str) -> str:
    """Generate a human-readable label for a model family key."""
    labels = {
        "sd1x": "Stable Diffusion 1.x",
        "sd2x": "Stable Diffusion 2.x",
        "sd3": "Stable Diffusion 3",
        "deepfloyd": "DeepFloyd IF",
        "sana": "Sana",
        "sdxl": "Stable Diffusion XL",
        "kolors": "Kolors",
        "flux": "Flux",
        "wan": "Wan",
        "ltxvideo": "LTX Video",
        "pixart_sigma": "PixArt-Σ",
        "omnigen": "OmniGen",
        "hidream": "HiDream",
        "auraflow": "AuraFlow",
        "lumina2": "Lumina 2",
        "cosmos2image": "Cosmos2Image",
        "qwen_image": "Qwen Image"
    }
    return labels.get(model_key, model_key.upper())


def _convert_field_to_template_format(field, config_values) -> Dict[str, Any]:
    """Convert a ConfigField to the format expected by the template."""
    field_dict = {
        "id": field.name,
        "name": field.arg_name,
        "label": field.ui_label,
        "type": field.field_type.value.lower(),
        "value": config_values.get(field.name, field.default_value),
        "description": field.help_text,
    }

    # Add cmd_args help for detailed tooltip
    if hasattr(field, 'cmd_args_help') and field.cmd_args_help:
        field_dict["cmd_args_help"] = field.cmd_args_help

    # Add tooltip text (prefer cmd_args_help over general tooltip)
    if hasattr(field, 'cmd_args_help') and field.cmd_args_help:
        field_dict["tooltip"] = field.cmd_args_help
    elif field.tooltip:
        field_dict["tooltip"] = field.tooltip

    # Add field-specific attributes
    if field.placeholder:
        field_dict["placeholder"] = field.placeholder

    if field.field_type.value == "NUMBER":
        # Add number-specific attributes
        for rule in field.validation_rules:
            if rule.rule_type.value == "min":
                field_dict["min"] = rule.value
            elif rule.rule_type.value == "max":
                field_dict["max"] = rule.value

    if field.field_type.value == "SELECT":
        field_dict["options"] = field.choices or []

    if field.field_type.value == "TEXTAREA":
        field_dict["rows"] = getattr(field, "rows", 3)

    return field_dict


def _get_fields_for_section(tab_name: str, section_name: str, config_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get fields for a specific section from the field registry."""
    fields = field_registry.get_fields_for_tab(tab_name)
    section_fields = [f for f in fields if f.section == section_name]

    # Convert to template format
    return [_convert_field_to_template_format(field, config_values) for field in section_fields]


def _load_active_config() -> Dict[str, Any]:
    """Load the active configuration values."""
    from ..services.config_store import ConfigStore
    from ..services.webui_state import WebUIStateStore

    try:
        # Get config store with user defaults
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        if defaults.configs_dir:
            store = ConfigStore(config_dir=defaults.configs_dir)
        else:
            store = ConfigStore()

        # Load active config
        active_config = store.get_active_config()
        if active_config:
            config_data, _ = store.load_config(active_config)
            return config_data
    except Exception:
        pass

    return {}


def _get_config_value(config_data: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Get config value, checking both modern (no prefix) and legacy (-- prefix) formats.

    Args:
        config_data: The configuration dictionary
        key: The key to look for (without -- prefix)
        default: Default value if not found

    Returns:
        The config value if found, otherwise the default
    """
    # First try modern format without --
    if key in config_data:
        return config_data[key]

    # Then try legacy format with --
    legacy_key = f"--{key}"
    if legacy_key in config_data:
        return config_data[legacy_key]

    return default


@router.get("/", response_class=HTMLResponse)
async def web_home(request: Request):
    """Redirect to trainer page."""
    return templates.TemplateResponse("trainer_htmx.html", {"request": request, "title": "SimpleTuner Training Studio"})


@router.get("/trainer", response_class=HTMLResponse)
async def trainer_page(request: Request):
    """Enhanced trainer page with HTMX."""
    return templates.TemplateResponse("trainer_htmx.html", {"request": request, "title": "SimpleTuner Training Studio"})


@router.get("/trainer/tabs/basic", response_class=HTMLResponse)
async def basic_config_tab(request: Request):
    """Basic configuration tab content."""
    # Load WebUI defaults
    from ..services.webui_state import WebUIStateStore

    webui_defaults = {}
    try:
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        webui_defaults = {
            "configs_dir": defaults.configs_dir or "Not configured",
            "output_dir": defaults.output_dir or "Not configured",
        }
    except Exception:
        webui_defaults = {"configs_dir": "Not configured", "output_dir": "Not configured"}

    # Load active config values
    config_data = _load_active_config()

    # Get all fields from the field registry for the basic tab
    tab_fields = field_registry.get_fields_for_tab("basic")

    # Extract all field names for config loading
    config_values = {}
    for field in tab_fields:
        if field.name == "configs_dir":
            config_values[field.name] = webui_defaults.get("configs_dir", "")
        elif field.name == "output_dir":
            config_values[field.name] = _get_config_value(config_data, field.name, webui_defaults.get("output_dir", ""))
        else:
            config_values[field.name] = _get_config_value(config_data, field.name, field.default_value)

    # Build all fields into template format
    all_fields = [_convert_field_to_template_format(field, config_values) for field in tab_fields]

    context = {
        "request": request,
        "section": {
            "id": "basic-config",
            "title": "Basic Configuration",
            "icon": "fas fa-cog",
            "description": "Essential settings to get started with training",
            "expanded": True,
            "fields": all_fields,
            "actions": [
                {
                    "label": "Save Changes",
                    "class": "btn-primary",
                    "type": "button",
                    "icon": "fas fa-save",
                    "hx_post": "/api/training/config",
                    "hx_include": "#trainer-form",
                    "hx_indicator": "#save-spinner",
                }
            ],
        },
    }
    return templates.TemplateResponse("tabs/basic_config_multi.html", context)


@router.get("/trainer/tabs/model", response_class=HTMLResponse)
async def model_config_tab(request: Request):
    """Model configuration tab content."""
    # Load active config values
    config_data = _load_active_config()

    # Get all fields from the field registry for the model tab
    tab_fields = field_registry.get_fields_for_tab("model")

    # Extract all field names for config loading
    config_values = {}
    for field in tab_fields:
        config_values[field.name] = _get_config_value(config_data, field.name, field.default_value)

    # Build all fields into template format
    all_fields = [_convert_field_to_template_format(field, config_values) for field in tab_fields]

    # Handle special cases for model_family options
    for field_dict in all_fields:
        if field_dict["id"] == "model_family" and "options" in field_dict:
            # Update options to use get_model_family_label for display
            field_dict["options"] = [
                {"value": opt["value"], "label": get_model_family_label(opt["value"])}
                for opt in field_dict["options"]
            ]
        elif field_dict["id"] == "lora_alpha":
            # Always set lora_alpha to match lora_rank value
            field_dict["value"] = config_values.get("lora_rank", "16")
            field_dict["disabled"] = True

    context = {
        "request": request,
        "section": {
            "id": "model-config",
            "title": "Model Configuration",
            "icon": "fas fa-brain",
            "description": "Model architecture and LoRA settings",
            "expanded": True,
            "fields": all_fields,
        },
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/training", response_class=HTMLResponse)
async def training_config_tab(request: Request):
    """Training configuration tab content."""
    # Load active config values
    config_data = _load_active_config()

    # Get all fields from the field registry for the training tab
    tab_fields = field_registry.get_fields_for_tab("training")

    # Extract all field names for config loading
    config_values = {}
    for field in tab_fields:
        config_values[field.name] = _get_config_value(config_data, field.name, field.default_value)

    # Build all fields into template format
    all_fields = [_convert_field_to_template_format(field, config_values) for field in tab_fields]

    # Handle special Alpine.js bindings for epochs/steps mutual exclusion
    for field_dict in all_fields:
        if field_dict["id"] == "num_train_epochs":
            field_dict["x_model"] = "numTrainEpochs"
            field_dict["x_bind_disabled"] = "maxTrainSteps != 0"
            field_dict["x_on_input"] = "handleEpochsChange()"
        elif field_dict["id"] == "max_train_steps":
            field_dict["x_model"] = "maxTrainSteps"
            field_dict["x_bind_disabled"] = "numTrainEpochs != 0"
            field_dict["x_on_input"] = "handleMaxStepsChange()"

    context = {
        "request": request,
        "section": {
            "id": "training-config",
            "title": "Training Parameters",
            "icon": "fas fa-graduation-cap",
            "description": "Core training hyperparameters",
            "expanded": True,
            "fields": all_fields,
        },
    }
    # Add config_values to the context so the template can access them
    context["config_values"] = config_values
    return templates.TemplateResponse("training_config_section.html", context)


@router.get("/trainer/tabs/advanced", response_class=HTMLResponse)
async def advanced_config_tab(request: Request):
    """Advanced configuration tab content."""
    # Load active config values
    config_data = _load_active_config()

    # Get all fields from the field registry for the advanced tab
    tab_fields = field_registry.get_fields_for_tab("advanced")

    # Extract all field names for config loading
    config_values = {}
    for field in tab_fields:
        config_values[field.name] = _get_config_value(config_data, field.name, field.default_value)

    # Build all fields into template format
    all_fields = [_convert_field_to_template_format(field, config_values) for field in tab_fields]

    context = {
        "request": request,
        "section": {
            "id": "advanced-config",
            "title": "Advanced Options",
            "icon": "fas fa-sliders-h",
            "description": "Advanced training and optimization settings",
            "expanded": True,
            "fields": all_fields,
        },
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/datasets", response_class=HTMLResponse)
async def datasets_tab(request: Request):
    """Dataset configuration tab content."""
    import json
    from pathlib import Path
    from ..services.dataset_plan import DatasetPlanStore
    from ..utils.paths import resolve_config_path

    datasets = []

    try:
        # First, try to load from active configuration's data backend config
        config_data = _load_active_config()
        data_backend_config_path = _get_config_value(config_data, "data_backend_config", None)

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"datasets_tab: data_backend_config_path = {data_backend_config_path}")

        if data_backend_config_path:
            # Resolve the path using our utility function
            # Get config directory from webui state
            from ..services.webui_state import WebUIStateStore
            webui_state = WebUIStateStore()
            defaults = webui_state.load_defaults()
            config_dir = defaults.configs_dir if defaults.configs_dir else None

            resolved_path = resolve_config_path(
                data_backend_config_path,
                config_dir=config_dir,
                check_cwd_first=True
            )

            logger.info(f"datasets_tab: resolved_path = {resolved_path}")

            if resolved_path and resolved_path.exists():
                try:
                    with open(resolved_path, 'r') as f:
                        datasets = json.load(f)
                        if not isinstance(datasets, list):
                            datasets = []
                        logger.info(f"datasets_tab: loaded {len(datasets)} datasets from {resolved_path}")
                except (json.JSONDecodeError, IOError) as e:
                    # Fall back to default behavior if we can't read the file
                    pass

        # Note: We don't fall back to global store if datasets is empty
        # An empty dataset list is a valid configuration for an environment
        logger.info(f"datasets_tab: Finished loading, have {len(datasets)} datasets")

    except Exception as e:
        # Fall back to default behavior on any error
        logger.error(f"datasets_tab: Exception loading from config - {str(e)}")
        try:
            store = DatasetPlanStore()
            datasets, _, _ = store.load()
            logger.warning(f"datasets_tab: Fell back to global store, loaded {len(datasets)} datasets")
        except ValueError:
            datasets = []

    context = {
        "request": request,
        "datasets": datasets,
        # Add default config for compatibility with trainer_dataloader_section.html
        "default_config": {
            "model_families": [
                {"value": key, "label": get_model_family_label(key), "selected": key == "sdxl"}
                for key in sorted(model_families.keys())
            ]
        },
    }

    return templates.TemplateResponse("datasets_tab.html", context)


@router.get("/trainer/tabs/environments", response_class=HTMLResponse)
async def environments_tab(request: Request):
    """Environments configuration management tab."""
    return templates.TemplateResponse("environments_tab.html", {"request": request})


@router.get("/trainer/tabs/validation", response_class=HTMLResponse)
async def validation_config_tab(request: Request):
    """Validation configuration tab content."""
    # Load active config values
    config_data = _load_active_config()

    # Get all fields from the field registry for the validation tab
    tab_fields = field_registry.get_fields_for_tab("validation")

    # Extract all field names for config loading
    config_values = {}
    for field in tab_fields:
        config_values[field.name] = _get_config_value(config_data, field.name, field.default_value)

    # Get sections for this tab
    sections = field_registry.get_sections_for_tab("validation")

    # Build all fields into a single section (for now)
    all_fields = [_convert_field_to_template_format(field, config_values) for field in tab_fields]

    context = {
        "request": request,
        "section": {
            "id": "validation-config",
            "title": "Validation & Output",
            "icon": "fas fa-check-circle",
            "description": "Validation settings and model output configuration",
            "expanded": True,
            "fields": all_fields,
            "actions": [
                {
                    "label": "Save Configuration",
                    "type": "button",
                    "class": "btn btn-primary",
                    "hx_post": "/api/configs/update-validation",
                    "hx_include": "#validation-config",
                    "hx_indicator": "#save-indicator"
                }
            ]
        }
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/datasets/new", response_class=HTMLResponse)
async def new_dataset_modal(request: Request):
    """New dataset modal content."""
    context = {
        "request": request,
        "dataset": {
            "id": "",
            "name": "",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": "",
            "resolution": 1024,
            "probability": 1.0,
        },
    }

    return templates.TemplateResponse("partials/dataset_card.html", context)
