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
        "model_type": config_data.get("--model_type", "lora"),
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
                    "id": "model_type",
                    "name": "--model_type",
                    "label": "Model Type",
                    "type": "select",
                    "required": True,
                    "value": config_values.get("model_type", "lora"),
                    "options": [
                        {"value": "full", "label": "Full Model Training"},
                        {"value": "lora", "label": "LoRA (Low-Rank Adaptation)"},
                    ],
                    "description": "Choose between full model fine-tuning or LoRA training",
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
        "resolution": config_data.get("--resolution", "1024"),
        "gradient_checkpointing": config_data.get("--gradient_checkpointing", False),
        "optimizer": config_data.get("--optimizer", "adamw"),
        "lr_scheduler": config_data.get("--lr_scheduler", "cosine"),
        "lr_warmup_steps": config_data.get("--lr_warmup_steps", "0"),
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
                {
                    "id": "resolution",
                    "name": "--resolution",
                    "label": "Resolution",
                    "type": "number",
                    "value": config_values.get("resolution", "1024"),
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "description": "Training resolution (must be divisible by 64)",
                },
                {
                    "id": "gradient_checkpointing",
                    "name": "--gradient_checkpointing",
                    "label": "Gradient Checkpointing",
                    "type": "checkbox",
                    "value": config_values.get("gradient_checkpointing", False),
                    "description": "Trade compute for memory, enabling larger batch sizes",
                },
                {
                    "id": "optimizer",
                    "name": "--optimizer",
                    "label": "Optimizer",
                    "type": "select",
                    "value": config_values.get("optimizer", "adamw"),
                    "options": [
                        {"value": "adamw", "label": "AdamW (Recommended)"},
                        {"value": "adam", "label": "Adam"},
                        {"value": "sgd", "label": "SGD"},
                        {"value": "adafactor", "label": "Adafactor"},
                        {"value": "adamw_bnb_8bit", "label": "AdamW 8-bit (Low VRAM)"},
                    ],
                },
                {
                    "id": "lr_scheduler",
                    "name": "--lr_scheduler",
                    "label": "Learning Rate Scheduler",
                    "type": "select",
                    "value": config_values.get("lr_scheduler", "cosine"),
                    "options": [
                        {"value": "constant", "label": "Constant"},
                        {"value": "linear", "label": "Linear"},
                        {"value": "cosine", "label": "Cosine (Recommended)"},
                        {"value": "cosine_with_restarts", "label": "Cosine with Restarts"},
                        {"value": "polynomial", "label": "Polynomial"},
                    ],
                },
                {
                    "id": "lr_warmup_steps",
                    "name": "--lr_warmup_steps",
                    "label": "LR Warmup Steps",
                    "type": "number",
                    "value": config_values.get("lr_warmup_steps", "0"),
                    "min": 0,
                    "description": "Number of warmup steps for learning rate",
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
        "max_grad_norm": config_data.get("--max_grad_norm", "1.0"),
        "train_text_encoder": config_data.get("--train_text_encoder", False),
        "enable_xformers_memory_efficient_attention": config_data.get("--enable_xformers_memory_efficient_attention", True),
        "use_ema": config_data.get("--use_ema", False),
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
                {
                    "id": "max_grad_norm",
                    "name": "--max_grad_norm",
                    "label": "Max Gradient Norm",
                    "type": "number",
                    "value": config_values.get("max_grad_norm", "1.0"),
                    "min": 0,
                    "step": 0.1,
                    "description": "Maximum gradient norm for clipping",
                },
                {
                    "id": "train_text_encoder",
                    "name": "--train_text_encoder",
                    "label": "Train Text Encoder",
                    "type": "checkbox",
                    "value": config_values.get("train_text_encoder", False),
                    "description": "Also train the text encoder (increases VRAM usage)",
                },
                {
                    "id": "enable_xformers_memory_efficient_attention",
                    "name": "--enable_xformers_memory_efficient_attention",
                    "label": "Enable xFormers",
                    "type": "checkbox",
                    "value": config_values.get("enable_xformers_memory_efficient_attention", True),
                    "description": "Use memory-efficient attention (recommended)",
                },
                {
                    "id": "use_ema",
                    "name": "--use_ema",
                    "label": "Use EMA",
                    "type": "checkbox",
                    "value": config_values.get("use_ema", False),
                    "description": "Use Exponential Moving Average for model weights",
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


@router.get("/trainer/tabs/validation", response_class=HTMLResponse)
async def validation_config_tab(request: Request):
    """Validation configuration tab content."""
    # Load active config values
    config_data = _load_active_config()
    config_values = {
        "validation_prompt": config_data.get("--validation_prompt", ""),
        "validation_steps": config_data.get("--validation_steps", "100"),
        "num_validation_images": config_data.get("--num_validation_images", "4"),
        "validation_guidance_scale": config_data.get("--validation_guidance_scale", "7.5"),
        "validation_guidance_rescale": config_data.get("--validation_guidance_rescale", "0.0"),
        "validation_num_inference_steps": config_data.get("--validation_num_inference_steps", "20"),
        "validation_resolution": config_data.get("--validation_resolution", ""),
        "validation_negative_prompt": config_data.get("--validation_negative_prompt", ""),
        "push_to_hub": config_data.get("--push_to_hub", False),
        "hub_model_id": config_data.get("--hub_model_id", ""),
        "webhook_config": config_data.get("--webhook_config", ""),
    }

    context = {
        "request": request,
        "section": {
            "id": "validation-config",
            "title": "Validation & Output",
            "icon": "fas fa-check-circle",
            "description": "Validation settings and model output configuration",
            "expanded": True,
            "fields": [
                {
                    "id": "validation_prompt",
                    "name": "--validation_prompt",
                    "label": "Validation Prompt",
                    "type": "textarea",
                    "value": config_values.get("validation_prompt", ""),
                    "description": "Prompt to use for generating validation images",
                    "rows": 3,
                },
                {
                    "id": "validation_steps",
                    "name": "--validation_steps",
                    "label": "Validation Every N Steps",
                    "type": "number",
                    "value": config_values.get("validation_steps", "100"),
                    "min": 1,
                    "description": "How often to generate validation images",
                },
                {
                    "id": "num_validation_images",
                    "name": "--num_validation_images",
                    "label": "Number of Validation Images",
                    "type": "number",
                    "value": config_values.get("num_validation_images", "4"),
                    "min": 1,
                    "max": 16,
                    "description": "How many images to generate for validation",
                },
                {
                    "id": "validation_guidance_scale",
                    "name": "--validation_guidance_scale",
                    "label": "Validation Guidance Scale",
                    "type": "number",
                    "value": config_values.get("validation_guidance_scale", "7.5"),
                    "min": 0,
                    "max": 20,
                    "step": 0.5,
                    "description": "CFG scale for validation images",
                },
                {
                    "id": "validation_num_inference_steps",
                    "name": "--validation_num_inference_steps",
                    "label": "Validation Inference Steps",
                    "type": "number",
                    "value": config_values.get("validation_num_inference_steps", "20"),
                    "min": 1,
                    "max": 100,
                    "description": "Number of denoising steps for validation",
                },
                {
                    "id": "validation_negative_prompt",
                    "name": "--validation_negative_prompt",
                    "label": "Validation Negative Prompt",
                    "type": "textarea",
                    "value": config_values.get("validation_negative_prompt", ""),
                    "description": "Negative prompt for validation images",
                    "rows": 2,
                },
                {
                    "id": "push_to_hub",
                    "name": "--push_to_hub",
                    "label": "Push to Hugging Face Hub",
                    "type": "checkbox",
                    "value": config_values.get("push_to_hub", False),
                    "description": "Automatically upload model to Hugging Face Hub",
                },
                {
                    "id": "hub_model_id",
                    "name": "--hub_model_id",
                    "label": "Hub Model ID",
                    "type": "text",
                    "value": config_values.get("hub_model_id", ""),
                    "placeholder": "username/model-name",
                    "description": "Hugging Face Hub repository name",
                },
                {
                    "id": "webhook_config",
                    "name": "--webhook_config",
                    "label": "Webhook Configuration",
                    "type": "text",
                    "value": config_values.get("webhook_config", ""),
                    "placeholder": "/path/to/webhook.json",
                    "description": "Path to webhook configuration file",
                },
            ],
        },
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
