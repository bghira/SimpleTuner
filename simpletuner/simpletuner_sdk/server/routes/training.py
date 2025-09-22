"""Training control routes for HTMX interface."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore, ConfigMetadata
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
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


def _get_config_store() -> ConfigStore:
    """Get the configuration store instance with user defaults if available."""
    try:
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        if defaults.configs_dir:
            return ConfigStore(config_dir=defaults.configs_dir)
    except Exception:
        pass
    return ConfigStore()


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
    store = _get_config_store()
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
            html += "".join(f"<li>{warning}</li>" for warning in warnings)
            html += "</ul>"
        if suggestions:
            html += "<h6>Suggestions:</h6>"
            html += '<ul class="mb-0">'
            html += "".join(f"<li>{suggestion}</li>" for suggestion in suggestions)
            html += "</ul>"
        html += '<small class="text-muted">You can proceed but review these items.</small>'
        html += "</div>"
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

    # Separate WebUI settings from training config
    webui_settings = {}
    config_dict = {}

    for key, value in form_data.items():
        if key == "configs_dir":
            webui_settings["configs_dir"] = value
        elif key.startswith("--"):
            config_dict[key] = value
        else:
            # Add -- prefix if not present
            config_dict[f"--{key}"] = value

    try:
        # Save WebUI settings if present
        if webui_settings:
            state_store = WebUIStateStore()
            defaults = state_store.load_defaults()
            for key, value in webui_settings.items():
                setattr(defaults, key, value)
            state_store.save_defaults(defaults)

        # Save training config only if we have parameters
        if config_dict:
            store = _get_config_store()
            active_config = store.get_active_config()

            # If no active config, use default
            if not active_config:
                active_config = "default"

            # Check if config exists, create metadata if needed
            try:
                # Load existing metadata if present
                _, metadata = store.load_config(active_config)
                metadata.modified_at = datetime.now().isoformat()
            except:
                # Create new metadata for new config
                metadata = ConfigMetadata(
                    description="Default configuration",
                    created_at=datetime.now().isoformat(),
                    modified_at=datetime.now().isoformat()
                )

            # Save updated config
            store.save_config(active_config, config_dict, metadata, overwrite=True)

            # Set as active config if it wasn't already
            if not store.get_active_config():
                store.set_active_config(active_config)

            # Also store in API state for compatibility
            APIState.set_state("training_config", config_dict)

        return """
        <div class="text-success" x-data="{ show: true }" x-show="show" x-init="setTimeout(() => show = false, 3000)" x-transition.opacity.duration.500ms>
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
    store = _get_config_store()
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
    store = _get_config_store()
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

        process_keeper.submit_job(job_id, Trainer, config_dict)

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
            process_keeper.terminate_process(job_id)
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
        except Exception:
            # Process keeper might not have this job
            pass

    return {"status": status, "config": config, "job_id": job_id, "job_info": job_info}
