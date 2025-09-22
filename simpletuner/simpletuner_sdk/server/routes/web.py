"""Web UI routes for HTMX enhanced trainer interface."""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/web", tags=["web"])

# Get template directory from environment
template_dir = os.environ.get("TEMPLATE_DIR", "templates")
templates = Jinja2Templates(directory=template_dir)


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
    config_values = {
        "configs_dir": webui_defaults.get("configs_dir", ""),
        "model_name": config_data.get("--job_id", ""),
        "output_dir": config_data.get("--output_dir", webui_defaults.get("output_dir", "")),
        "pretrained_model_name_or_path": config_data.get("--pretrained_model_name_or_path", ""),
    }

    context = {
        "request": request,
        "section": {
            "id": "basic-config",
            "title": "Basic Configuration",
            "icon": "fas fa-cog",
            "description": "Essential settings to get started with training",
            "expanded": True,
            "fields": [
                {
                    "id": "configs_dir",
                    "name": "configs_dir",
                    "label": "Configurations Directory",
                    "type": "text",
                    "placeholder": "/path/to/configs",
                    "required": True,
                    "value": config_values.get("configs_dir", ""),
                    "description": "Directory where training configurations are stored",
                },
                {
                    "id": "model_name",
                    "name": "--job_id",
                    "label": "Model Name",
                    "type": "text",
                    "placeholder": "my-awesome-model",
                    "required": True,
                    "value": config_values.get("model_name", ""),
                    "description": "Name for your trained model",
                },
                {
                    "id": "output_dir",
                    "name": "--output_dir",
                    "label": "Output Directory",
                    "type": "text",
                    "placeholder": "/path/to/output",
                    "required": True,
                    "value": config_values.get("output_dir", ""),
                    "description": "Where to save the trained model",
                },
                {
                    "id": "pretrained_model_name_or_path",
                    "name": "--pretrained_model_name_or_path",
                    "label": "Base Model Path",
                    "type": "text",
                    "placeholder": "black-forest-labs/FLUX.1-dev",
                    "required": True,
                    "value": config_values.get("pretrained_model_name_or_path", ""),
                    "description": "Hugging Face model or local path",
                },
            ],
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
    config_values = {
        "model_family": config_data.get("--model_family", ""),
        "lora_rank": config_data.get("--lora_rank", "16"),
        "lora_alpha": config_data.get("--lora_alpha", "16"),
    }

    context = {
        "request": request,
        "section": {
            "id": "model-config",
            "title": "Model Configuration",
            "icon": "fas fa-brain",
            "description": "Model architecture and LoRA settings",
            "expanded": True,
            "fields": [
                {
                    "id": "model_family",
                    "name": "--model_family",
                    "label": "Model Family",
                    "type": "select",
                    "required": True,
                    "value": config_values.get("model_family", ""),
                    "options": [
                        {"value": "flux", "label": "Flux"},
                        {"value": "sd3", "label": "Stable Diffusion 3"},
                        {"value": "sdxl", "label": "Stable Diffusion XL"},
                        {"value": "sd", "label": "Stable Diffusion 1.5"},
                    ],
                },
                {
                    "id": "lora_rank",
                    "name": "--lora_rank",
                    "label": "LoRA Rank",
                    "type": "number",
                    "value": config_values.get("lora_rank", "16"),
                    "min": 1,
                    "max": 256,
                    "description": "Rank of LoRA matrices",
                },
                {
                    "id": "lora_alpha",
                    "name": "--lora_alpha",
                    "label": "LoRA Alpha",
                    "type": "number",
                    "value": config_values.get("lora_alpha", "16"),
                    "min": 1,
                    "description": "Scaling factor for LoRA",
                },
            ],
        },
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/training", response_class=HTMLResponse)
async def training_config_tab(request: Request):
    """Training configuration tab content."""
    # Load active config values
    config_data = _load_active_config()
    config_values = {
        "learning_rate": config_data.get("--learning_rate", "0.0001"),
        "train_batch_size": config_data.get("--train_batch_size", "1"),
        "num_train_epochs": config_data.get("--num_train_epochs", "10"),
        "mixed_precision": config_data.get("--mixed_precision", "bf16"),
    }

    context = {
        "request": request,
        "section": {
            "id": "training-config",
            "title": "Training Parameters",
            "icon": "fas fa-graduation-cap",
            "description": "Core training hyperparameters",
            "expanded": True,
            "fields": [
                {
                    "id": "learning_rate",
                    "name": "--learning_rate",
                    "label": "Learning Rate",
                    "type": "number",
                    "step": 0.000001,
                    "value": config_values.get("learning_rate", "0.0001"),
                    "min": 0,
                    "description": "Base learning rate for training",
                },
                {
                    "id": "train_batch_size",
                    "name": "--train_batch_size",
                    "label": "Batch Size",
                    "type": "number",
                    "value": config_values.get("train_batch_size", "1"),
                    "min": 1,
                    "description": "Number of samples per batch",
                },
                {
                    "id": "num_train_epochs",
                    "name": "--num_train_epochs",
                    "label": "Number of Epochs",
                    "type": "number",
                    "value": config_values.get("num_train_epochs", "10"),
                    "min": 1,
                    "description": "Number of training epochs",
                },
                {
                    "id": "mixed_precision",
                    "name": "--mixed_precision",
                    "label": "Mixed Precision",
                    "type": "select",
                    "value": config_values.get("mixed_precision", "bf16"),
                    "options": [
                        {"value": "no", "label": "No (FP32)"},
                        {"value": "fp16", "label": "FP16"},
                        {"value": "bf16", "label": "BF16 (Recommended)"},
                    ],
                },
            ],
        },
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/advanced", response_class=HTMLResponse)
async def advanced_config_tab(request: Request):
    """Advanced configuration tab content."""
    # Load active config values
    config_data = _load_active_config()
    config_values = {
        "gradient_accumulation_steps": config_data.get("--gradient_accumulation_steps", "1"),
        "checkpointing_steps": config_data.get("--checkpointing_steps", "500"),
        "validation_steps": config_data.get("--validation_steps", "100"),
        "seed": config_data.get("--seed", "42"),
    }

    context = {
        "request": request,
        "section": {
            "id": "advanced-config",
            "title": "Advanced Options",
            "icon": "fas fa-sliders-h",
            "description": "Advanced training and optimization settings",
            "expanded": True,
            "fields": [
                {
                    "id": "gradient_accumulation_steps",
                    "name": "--gradient_accumulation_steps",
                    "label": "Gradient Accumulation Steps",
                    "type": "number",
                    "value": config_values.get("gradient_accumulation_steps", "1"),
                    "min": 1,
                    "description": "Steps to accumulate gradients",
                },
                {
                    "id": "checkpointing_steps",
                    "name": "--checkpointing_steps",
                    "label": "Checkpoint Every N Steps",
                    "type": "number",
                    "value": config_values.get("checkpointing_steps", "500"),
                    "min": 1,
                    "description": "How often to save checkpoints",
                },
                {
                    "id": "validation_steps",
                    "name": "--validation_steps",
                    "label": "Validation Every N Steps",
                    "type": "number",
                    "value": config_values.get("validation_steps", "100"),
                    "min": 1,
                    "description": "How often to run validation",
                },
                {
                    "id": "seed",
                    "name": "--seed",
                    "label": "Random Seed",
                    "type": "number",
                    "value": config_values.get("seed", "42"),
                    "min": 0,
                    "description": "Random seed for reproducibility",
                },
            ],
        },
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/datasets", response_class=HTMLResponse)
async def datasets_tab(request: Request):
    """Dataset configuration tab content."""
    # Load existing datasets from the dataset plan
    from ..services.dataset_plan import DatasetPlanStore

    try:
        store = DatasetPlanStore()
        datasets, _, _ = store.load()
    except ValueError:
        datasets = []

    context = {
        "request": request,
        "datasets": datasets,
        # Add default config for compatibility with trainer_dataloader_section.html
        "default_config": {
            "model_families": [
                {"value": "flux", "label": "Flux", "selected": False},
                {"value": "sd3", "label": "Stable Diffusion 3", "selected": False},
                {"value": "sdxl", "label": "Stable Diffusion XL", "selected": True},
                {"value": "sd", "label": "Stable Diffusion 1.5", "selected": False},
            ]
        },
    }

    return templates.TemplateResponse("datasets_tab.html", context)


@router.get("/trainer/tabs/environments", response_class=HTMLResponse)
async def environments_tab(request: Request):
    """Environments configuration management tab."""
    return templates.TemplateResponse("environments_tab.html", {"request": request})


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
