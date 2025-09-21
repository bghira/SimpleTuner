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


@router.get("/", response_class=HTMLResponse)
async def web_home(request: Request):
    """Redirect to trainer page."""
    return templates.TemplateResponse("trainer_htmx.html", {
        "request": request,
        "title": "SimpleTuner Training Studio"
    })


@router.get("/trainer", response_class=HTMLResponse)
async def trainer_page(request: Request):
    """Enhanced trainer page with HTMX."""
    return templates.TemplateResponse("trainer_htmx.html", {
        "request": request,
        "title": "SimpleTuner Training Studio"
    })


@router.get("/trainer/tabs/basic", response_class=HTMLResponse)
async def basic_config_tab(request: Request):
    """Basic configuration tab content."""
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
                    "id": "model_name",
                    "label": "Model Name",
                    "type": "text",
                    "placeholder": "my-awesome-model",
                    "required": True,
                    "description": "Name for your trained model"
                },
                {
                    "id": "output_dir",
                    "label": "Output Directory",
                    "type": "text",
                    "placeholder": "/path/to/output",
                    "required": True,
                    "description": "Where to save the trained model"
                },
                {
                    "id": "pretrained_model_name_or_path",
                    "label": "Base Model Path",
                    "type": "text",
                    "placeholder": "black-forest-labs/FLUX.1-dev",
                    "required": True,
                    "description": "Hugging Face model or local path"
                }
            ]
        }
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/model", response_class=HTMLResponse)
async def model_config_tab(request: Request):
    """Model configuration tab content."""
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
                    "label": "Model Family",
                    "type": "select",
                    "required": True,
                    "options": [
                        {"value": "flux", "label": "Flux"},
                        {"value": "sd3", "label": "Stable Diffusion 3"},
                        {"value": "sdxl", "label": "Stable Diffusion XL"},
                        {"value": "sd", "label": "Stable Diffusion 1.5"}
                    ]
                },
                {
                    "id": "lora_rank",
                    "label": "LoRA Rank",
                    "type": "number",
                    "value": "16",
                    "min": 1,
                    "max": 256,
                    "description": "Rank of LoRA matrices"
                },
                {
                    "id": "lora_alpha",
                    "label": "LoRA Alpha",
                    "type": "number",
                    "value": "16",
                    "min": 1,
                    "description": "Scaling factor for LoRA"
                }
            ]
        }
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/training", response_class=HTMLResponse)
async def training_config_tab(request: Request):
    """Training configuration tab content."""
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
                    "label": "Learning Rate",
                    "type": "number",
                    "step": 0.000001,
                    "value": "0.0001",
                    "min": 0,
                    "description": "Base learning rate for training"
                },
                {
                    "id": "train_batch_size",
                    "label": "Batch Size",
                    "type": "number",
                    "value": "1",
                    "min": 1,
                    "description": "Number of samples per batch"
                },
                {
                    "id": "num_train_epochs",
                    "label": "Number of Epochs",
                    "type": "number",
                    "value": "10",
                    "min": 1,
                    "description": "Number of training epochs"
                },
                {
                    "id": "mixed_precision",
                    "label": "Mixed Precision",
                    "type": "select",
                    "value": "bf16",
                    "options": [
                        {"value": "no", "label": "No (FP32)"},
                        {"value": "fp16", "label": "FP16"},
                        {"value": "bf16", "label": "BF16 (Recommended)"}
                    ]
                }
            ]
        }
    }
    return templates.TemplateResponse("partials/form_section.html", context)


@router.get("/trainer/tabs/advanced", response_class=HTMLResponse)
async def advanced_config_tab(request: Request):
    """Advanced configuration tab content."""
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
                    "label": "Gradient Accumulation Steps",
                    "type": "number",
                    "value": "1",
                    "min": 1,
                    "description": "Steps to accumulate gradients"
                },
                {
                    "id": "checkpointing_steps",
                    "label": "Checkpoint Every N Steps",
                    "type": "number",
                    "value": "500",
                    "min": 1,
                    "description": "How often to save checkpoints"
                },
                {
                    "id": "validation_steps",
                    "label": "Validation Every N Steps",
                    "type": "number",
                    "value": "100",
                    "min": 1,
                    "description": "How often to run validation"
                },
                {
                    "id": "seed",
                    "label": "Random Seed",
                    "type": "number",
                    "value": "42",
                    "min": 0,
                    "description": "Random seed for reproducibility"
                }
            ]
        }
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
                {"value": "sd", "label": "Stable Diffusion 1.5", "selected": False}
            ]
        }
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
            "probability": 1.0
        }
    }

    return templates.TemplateResponse("partials/dataset_card.html", context)