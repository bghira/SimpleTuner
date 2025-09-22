"""Field validation routes for HTMX forms."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/api/validate", tags=["validation"])


@router.post("/{field_name}", response_class=HTMLResponse)
async def validate_field(field_name: str, value: str = Form("")) -> str:
    """Validate a single field and return HTML error fragment."""
    error_html = ""

    # General field validations
    if field_name in ["model_name", "output_dir", "pretrained_model_name_or_path"]:
        if not value or not value.strip():
            error_html = f"{field_name.replace('_', ' ').title()} is required"
        elif len(value) > 200:
            error_html = f"{field_name.replace('_', ' ').title()} must be 200 characters or less"

    elif field_name in ["learning_rate", "lr_scheduler_eta_min"]:
        try:
            lr = float(value) if value else 0
            if lr <= 0:
                error_html = "Learning rate must be greater than 0"
            elif lr > 1:
                error_html = "Learning rate should typically be less than 1"
        except ValueError:
            error_html = "Learning rate must be a valid number"

    elif field_name in ["num_train_epochs", "max_train_steps"]:
        try:
            epochs = int(value) if value else 0
            if epochs <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name in ["train_batch_size", "gradient_accumulation_steps"]:
        try:
            batch_size = int(value) if value else 0
            if batch_size <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
            elif batch_size > 128:
                error_html = f"{field_name.replace('_', ' ').title()} should typically be 128 or less"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name == "resolution":
        try:
            res = int(value) if value else 0
            if res < 256:
                error_html = "Resolution must be at least 256"
            elif res > 4096:
                error_html = "Resolution must not exceed 4096"
            elif res % 64 != 0:
                error_html = "Resolution must be divisible by 64"
        except ValueError:
            error_html = "Resolution must be a valid number"

    elif field_name in ["output_dir", "logging_dir", "instance_data_dir"]:
        if not value or not value.strip():
            error_html = f"{field_name.replace('_', ' ').title()} is required"
        elif not os.path.isabs(value) and not ":" in value:
            error_html = "Path must be absolute (start with /) or include drive letter on Windows"

    elif field_name == "webhook_url":
        if value and not re.match(r"^https?://", value):
            error_html = "Webhook URL must start with http:// or https://"

    elif field_name in ["lora_rank", "lora_alpha"]:
        try:
            rank = int(value) if value else 0
            if rank <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
            elif rank > 512:
                error_html = f"{field_name.replace('_', ' ').title()} should typically be 512 or less"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name == "mixed_precision":
        if value not in ["no", "fp16", "bf16"]:
            error_html = "Mixed precision must be 'no', 'fp16', or 'bf16'"

    elif field_name == "lr_scheduler":
        valid_schedulers = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ]
        if value and value not in valid_schedulers:
            error_html = f"Learning rate scheduler must be one of: {', '.join(valid_schedulers)}"

    elif field_name == "optimizer":
        valid_optimizers = ["adamw", "adam", "sgd", "adafactor", "adamw_bnb_8bit"]
        if value and value not in valid_optimizers:
            error_html = f"Optimizer must be one of: {', '.join(valid_optimizers)}"

    return error_html
