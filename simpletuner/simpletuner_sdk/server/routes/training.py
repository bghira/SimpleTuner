"""Training control routes for HTMX interface."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk import process_keeper

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingConfig(BaseModel):
    """Training configuration model."""
    model_name: str
    output_dir: str
    pretrained_model_name_or_path: str
    learning_rate: float = 0.0001
    train_batch_size: int = 1
    num_train_epochs: int = 10
    mixed_precision: str = "bf16"


@router.post("/validate", response_class=HTMLResponse)
async def validate_config(request: Request):
    """Validate training configuration and return HTML feedback."""
    form_data = await request.form()

    # Convert form data to config dict with -- prefixes
    config_dict = {}
    for key, value in form_data.items():
        if key.startswith("--"):
            config_dict[key] = value
        else:
            # Add -- prefix if not present
            config_dict[f"--{key}"] = value

    # Use ConfigStore validation
    store = ConfigStore()
    validation = store.validate_config(config_dict)

    errors = validation.errors
    warnings = validation.warnings
    suggestions = validation.suggestions

    # Generate HTML response
    if errors:
        html = f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Validation Failed</h6>
            <ul class="mb-0">
                {''.join(f'<li>{error}</li>' for error in errors)}
            </ul>
        </div>
        """
    elif warnings or suggestions:
        html = '<div class="alert alert-warning">'
        html += '<h6><i class="fas fa-exclamation-circle"></i> Validation Warnings</h6>'
        if warnings:
            html += '<ul class="mb-2">'
            html += ''.join(f'<li>{warning}</li>' for warning in warnings)
            html += '</ul>'
        if suggestions:
            html += '<h6>Suggestions:</h6>'
            html += '<ul class="mb-0">'
            html += ''.join(f'<li>{suggestion}</li>' for suggestion in suggestions)
            html += '</ul>'
        html += '<small class="text-muted">You can proceed but review these items.</small>'
        html += '</div>'
    else:
        html = """
        <div class="alert alert-success">
            <h6><i class="fas fa-check-circle"></i> Configuration Valid</h6>
            <p class="mb-0">All settings look good! Ready to start training.</p>
        </div>
        """

    return html


@router.post("/config", response_class=HTMLResponse)
async def save_config(request: Request):
    """Save training configuration."""
    form_data = await request.form()

    # Convert form data to config dict with -- prefixes
    config_dict = {}
    for key, value in form_data.items():
        if key.startswith("--"):
            config_dict[key] = value
        else:
            # Add -- prefix if not present
            config_dict[f"--{key}"] = value

    # Save to active config in ConfigStore
    store = ConfigStore()
    active_config = store.get_active_config()

    try:
        # Load existing metadata if present
        try:
            _, metadata = store.load_config(active_config)
        except:
            metadata = None

        # Save updated config
        store.save_config(active_config, config_dict, metadata, overwrite=True)

        # Also store in API state for compatibility
        APIState.set_state("training_config", config_dict)

        return """
        <div class="text-success">
            <i class="fas fa-check"></i> Configuration saved
        </div>
        """
    except Exception as e:
        return f"""
        <div class="text-danger">
            <i class="fas fa-exclamation-triangle"></i> Failed to save: {str(e)}
        </div>
        """


@router.post("/start", response_class=HTMLResponse)
async def start_training(request: Request):
    """Start training with current configuration."""
    form_data = await request.form()

    # Convert form data to config dict with -- prefixes
    config_dict = {}
    for key, value in form_data.items():
        if key.startswith("--"):
            config_dict[key] = value
        else:
            # Add -- prefix if not present
            config_dict[f"--{key}"] = value

    # Validate using ConfigStore
    store = ConfigStore()
    validation = store.validate_config(config_dict)

    if not validation.is_valid:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Cannot Start Training</h6>
            <ul class="mb-0">
                {''.join(f'<li>{error}</li>' for error in validation.errors)}
            </ul>
        </div>
        """

    # Save config to active config
    store = ConfigStore()
    active_config = store.get_active_config()

    try:
        # Load existing metadata if present
        try:
            _, metadata = store.load_config(active_config)
        except:
            metadata = None

        # Save updated config
        store.save_config(active_config, config_dict, metadata, overwrite=True)

        # Store config in API state
        APIState.set_state("training_config", config_dict)
        APIState.set_state("training_status", "starting")

        # Start the training process using process keeper functions
        import uuid
        job_id = str(uuid.uuid4())[:8]

        # Submit training job
        from simpletuner.helpers.training.trainer import Trainer
        process = process_keeper.submit_job(job_id, Trainer, config_dict)

        # Store job ID
        APIState.set_state("current_job_id", job_id)

        return f"""
        <div class="alert alert-info">
            <h6><i class="fas fa-cog fa-spin"></i> Training Starting</h6>
            <p>Your training job is being initialized.</p>
            <p class="mb-0"><small>Job ID: {job_id}</small></p>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Failed to Start Training</h6>
            <p>{str(e)}</p>
        </div>
        """


@router.post("/stop")
async def stop_training():
    """Stop current training."""
    try:
        job_id = APIState.get_state("current_job_id")
        if job_id:
            # Terminate the process
            success = process_keeper.terminate_process(job_id)
            APIState.set_state("training_status", "stopped")
            return {"message": f"Training job {job_id} stop requested"}
        else:
            return {"message": "No active training job to stop"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    status = APIState.get_state("training_status", "idle")
    config = APIState.get_state("training_config", {})
    job_id = APIState.get_state("current_job_id")

    # Get detailed job status if available
    job_info = None
    if job_id:
        try:
            # Get process status
            job_status = process_keeper.get_process_status(job_id)
            job_info = {"status": job_status}
        except:
            pass

    return {
        "status": status,
        "config": config,
        "job_id": job_id,
        "job_info": job_info
    }


