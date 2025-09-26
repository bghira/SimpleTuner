"""Training control routes for HTMX interface."""

from __future__ import annotations

import os
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore, ConfigMetadata
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk import process_keeper
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.field_registry import FieldType

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


def _convert_value_by_type(value: str, field_type: FieldType) -> any:
    """Convert a string value to the appropriate type based on field type.

    Args:
        value: The string value to convert
        field_type: The field type from the field registry

    Returns:
        The converted value with the correct type
    """
    if not value and value != "0":
        return value

    if field_type == FieldType.NUMBER:
        # Try to convert to int first, then float
        try:
            # Check if it's a float
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value
    elif field_type == FieldType.CHECKBOX:
        # Convert checkbox values to boolean
        return value.lower() in ('true', '1', 'yes', 'on')
    else:
        # TEXT, SELECT, TEXTAREA, FILE, etc. remain as strings
        return value


def _normalize_form_to_config(form_data: dict, directory_fields: list = None) -> dict:
    """Convert form data to config dict with -- prefixes and proper type conversion.

    Args:
        form_data: The form data dictionary from the request
        directory_fields: Optional list of fields that should have path expansion applied

    Returns:
        Dictionary with normalized config keys (all having -- prefix) and proper types
    """
    config_dict = {}
    # Fields that should be excluded from config dict
    excluded_fields = {"configs_dir"}

    # Get field registry to determine types
    field_types = {}
    for field in lazy_field_registry.get_all_fields():
        field_types[field.arg_name] = field.field_type

    # Numeric fields that should always be included, even if "0"
    numeric_fields = ["--num_train_epochs", "--max_train_steps", "--lr_warmup_steps",
                      "--gradient_accumulation_steps", "--train_batch_size"]

    # Fields that should always be included even if empty
    always_include_fields = ["--model_flavour", "--optimizer_config"]

    for key, value in form_data.items():
        # Skip excluded fields
        if key in excluded_fields:
            continue

        # Ensure -- prefix
        config_key = key if key.startswith("--") else f"--{key}"

        # For numeric fields, always include the value (even "0")
        if config_key in numeric_fields:
            # Always include numeric fields, even if the value is "0" or empty
            if not value:
                value = "0"
        elif config_key in always_include_fields:
            # Always include these fields even if empty
            if not value:
                value = ""
        elif not value:
            # Skip empty values for other fields
            continue

        # Apply directory path expansion if needed
        if directory_fields and config_key in directory_fields and value:
            config_dict[config_key] = os.path.abspath(os.path.expanduser(value))
        else:
            # Convert value to appropriate type
            field_type = field_types.get(config_key, FieldType.TEXT)
            config_dict[config_key] = _convert_value_by_type(value, field_type)

    if "--i_know_what_i_am_doing" not in config_dict:
        config_dict["--i_know_what_i_am_doing"] = False

    return config_dict


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


def _get_all_field_defaults() -> dict:
    """Get all default values from field registry with proper types.

    Returns:
        Dictionary with all field defaults with -- prefixes and proper types
    """
    defaults = {}

    for field in lazy_field_registry.get_all_fields():
        if field.default_value is not None:
            # Convert default value to proper type
            defaults[field.arg_name] = _convert_value_by_type(
                str(field.default_value) if not isinstance(field.default_value, bool) else str(field.default_value).lower(),
                field.field_type
            )

    return defaults


@router.post("/validate", response_class=HTMLResponse)
async def validate_config(request: Request):
    """Validate training configuration and return HTML feedback."""
    form_data = await request.form()

    # Convert form data to config dict with -- prefixes
    config_dict = _normalize_form_to_config(dict(form_data))

    # Get all field defaults and merge with form data for complete validation
    all_defaults = _get_all_field_defaults()
    complete_config = {**all_defaults, **config_dict}

    # Use ConfigStore validation on the complete config
    store = _get_config_store()
    validation = store.validate_config(complete_config)

    errors = list(validation.errors) if validation.errors else []
    warnings = list(validation.warnings) if validation.warnings else []
    suggestions = list(validation.suggestions) if validation.suggestions else []

    # Add custom validation for mutual exclusivity
    num_epochs = config_dict.get("--num_train_epochs", "10")
    max_steps = config_dict.get("--max_train_steps", "0")

    try:
        epochs_val = int(num_epochs) if num_epochs else 0
        steps_val = int(max_steps) if max_steps else 0

        if epochs_val == 0 and steps_val == 0:
            errors.append("Either num_train_epochs or max_train_steps must be greater than 0. You cannot set both to 0.")

        if epochs_val > 0 and steps_val > 0:
            errors.append("num_train_epochs and max_train_steps cannot both be set. Set one of them to 0.")
    except ValueError:
        errors.append("Invalid value for num_train_epochs or max_train_steps. Must be numeric.")

    lr_scheduler = config_dict.get("--lr_scheduler", complete_config.get("--lr_scheduler", ""))
    warmup_raw = config_dict.get("--lr_warmup_steps", complete_config.get("--lr_warmup_steps", 0))
    try:
        warmup_val = int(warmup_raw) if warmup_raw else 0
        if lr_scheduler == "constant" and warmup_val > 0:
            errors.append("Warmup steps are not supported with the 'constant' learning rate scheduler. Use 'constant_with_warmup' or set warmup steps to 0.")
    except ValueError:
        errors.append("Warmup steps must be a whole number.")

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


@router.post("/configuration/check", response_class=HTMLResponse)
async def configuration_check(request: Request):
    """Compatibility endpoint for legacy clients expecting configuration/check."""
    return await validate_config(request)


@router.post("/config", response_class=HTMLResponse)
async def save_config(request: Request):
    """Save training configuration."""
    form_data = await request.form()

    # Separate WebUI settings and save options from training config
    webui_settings = {}
    save_options = {}

    # Extract WebUI-specific settings and save options
    form_dict = dict(form_data)
    if "configs_dir" in form_dict:
        webui_settings["configs_dir"] = os.path.abspath(os.path.expanduser(form_dict["configs_dir"])) if form_dict["configs_dir"] else form_dict["configs_dir"]

    # Extract save options
    if "preserve_defaults" in form_dict:
        save_options["preserve_defaults"] = form_dict["preserve_defaults"] == "true"
        del form_dict["preserve_defaults"]
    if "create_backup" in form_dict:
        save_options["create_backup"] = form_dict["create_backup"] == "true"
        del form_dict["create_backup"]

    # Directory fields that need path expansion
    directory_fields = ["--output_dir", "--instance_data_dir", "--logging_dir"]

    # Convert form data to config dict with path expansion for directory fields
    config_dict = _normalize_form_to_config(form_dict, directory_fields)

    # Debug log what we're saving
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Saving config_dict: {config_dict}")
    logger.info(f"num_train_epochs value: {config_dict.get('--num_train_epochs', 'NOT FOUND')}")
    logger.info(f"max_train_steps value: {config_dict.get('--max_train_steps', 'NOT FOUND')}")

    # Prepare config store and active config information
    store = _get_config_store()
    active_config = store.get_active_config()

    # Load existing configuration (if any) so values from other tabs are preserved
    existing_config_cli = {}
    if active_config:
        try:
            existing_config_data, _ = store.load_config(active_config)
            if isinstance(existing_config_data, dict):
                for key, value in existing_config_data.items():
                    cli_key = key if key.startswith("--") else f"--{key}"
                    existing_config_cli[cli_key] = value
        except FileNotFoundError:
            pass
        except ValueError:
            # Malformed config should not block saving new values
            pass

    try:
        # Save WebUI settings if present
        if webui_settings:
            state_store = WebUIStateStore()
            defaults = state_store.load_defaults()
            for key, value in webui_settings.items():
                setattr(defaults, key, value)
            state_store.save_defaults(defaults)

        # Get all field defaults and merge with existing config plus form data
        all_defaults = _get_all_field_defaults()
        # Merge order: defaults < existing config < submitted form values
        complete_config = {**all_defaults, **existing_config_cli, **config_dict}

        # Process the config for saving
        save_config = {}
        for key, value in complete_config.items():
            # Remove "--" prefix from keys
            clean_key = key[2:] if key.startswith("--") else key

            # If preserve_defaults is true, only include non-default values
            if save_options.get("preserve_defaults", False):
                # Compare with default value
                default_value = all_defaults.get(key)
                if value != default_value:
                    save_config[clean_key] = value
            else:
                save_config[clean_key] = value

        # Save training config
        # If no active config, use default
        if not active_config:
            active_config = "default"

        # Handle backup option
        if save_options.get("create_backup", False):
            config_path = store._get_config_path(active_config)
            if config_path.exists():
                # Create backup with timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = config_path.with_suffix(f".json.backup-{timestamp}")
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Created backup at {backup_path}")

        # Use the new save_trainer_config method for flat JSON format
        store.save_trainer_config(active_config, save_config, overwrite=True)

        # Set as active config if it wasn't already
        if not store.get_active_config():
            store.set_active_config(active_config)

        # Also store in API state for compatibility (use complete config with -- prefixes for internal use)
        APIState.set_state("training_config", complete_config)

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
    config_dict = _normalize_form_to_config(dict(form_data))

    # Get all field defaults and merge with form data for complete validation
    all_defaults = _get_all_field_defaults()
    complete_config = {**all_defaults, **config_dict}

    # Validate using ConfigStore on the complete config
    store = _get_config_store()
    validation = store.validate_config(complete_config)

    # Add custom validation for mutual exclusivity
    errors = list(validation.errors) if validation.errors else []

    num_epochs = config_dict.get("--num_train_epochs", "10")
    max_steps = config_dict.get("--max_train_steps", "0")

    try:
        epochs_val = int(num_epochs) if num_epochs else 0
        steps_val = int(max_steps) if max_steps else 0

        # Check if both are zero
        if epochs_val == 0 and steps_val == 0:
            errors.append("Either num_train_epochs or max_train_steps must be greater than 0. You cannot set both to 0.")

        # Don't error if one is 0 - this is the valid use case
        # Only error if user is trying to use both methods simultaneously
        # (No check needed here - it's valid to have one at 0 and one > 0)
    except ValueError:
        errors.append("Invalid value for num_train_epochs or max_train_steps. Must be numeric.")

    if not validation.is_valid or errors:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Cannot Start Training</h6>
            <ul class="mb-0">
                {''.join(f'<li>{error}</li>' for error in (errors if errors else validation.errors))}
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


@router.post("/configuration/run", response_class=HTMLResponse)
async def configuration_run(request: Request):
    """Compatibility endpoint mapping to the legacy configuration/run URL."""
    return await start_training(request)


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


@router.post("/cancel", response_class=HTMLResponse)
async def cancel_training(request: Request):
    """Cancel current training and return HTML response for HTMX."""
    try:
        form_data = await request.form()
        job_id = form_data.get("job_id", APIState.get_state("current_job_id"))

        if job_id:
            # Terminate the process
            process_keeper.terminate_process(job_id)
            APIState.set_state("training_status", "cancelled")
            APIState.set_state("current_job_id", None)

            return f"""
            <div class="alert alert-warning">
                <h6><i class="fas fa-hand-paper"></i> Training Cancelled</h6>
                <p>Training job has been cancelled successfully.</p>
                <p class="mb-0"><small>Job ID: {job_id}</small></p>
            </div>
            """
        else:
            return f"""
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle"></i> No Active Training</h6>
                <p>There is no active training job to cancel.</p>
            </div>
            """
    except Exception as e:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Failed to Cancel Training</h6>
            <p>{str(e)}</p>
        </div>
        """


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


@router.get("/events")
async def get_training_events(since_index: int = 0):
    """Get training events since a given index."""
    job_id = APIState.get_state("current_job_id")

    if not job_id:
        return {"events": [], "job_id": None}

    try:
        events = process_keeper.get_process_events(job_id, since_index)
        return {"events": events, "job_id": job_id, "next_index": since_index + len(events)}
    except Exception as e:
        # Job might not exist
        return {"events": [], "job_id": job_id, "error": str(e)}


@router.websocket("/events/stream")
async def stream_training_events(websocket: WebSocket):
    """Stream training events via WebSocket."""
    await websocket.accept()

    try:
        last_index = 0
        while True:
            job_id = APIState.get_state("current_job_id")

            if job_id:
                try:
                    # Get new events since last index
                    events = process_keeper.get_process_events(job_id, last_index)

                    if events:
                        # Send each event
                        for event in events:
                            await websocket.send_json({
                                "job_id": job_id,
                                "event": event
                            })

                        last_index += len(events)
                except Exception as e:
                    # Send error but continue streaming
                    await websocket.send_json({
                        "error": str(e),
                        "job_id": job_id
                    })

            # Wait a bit before checking for new events
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Send error and close
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        await websocket.close()


@router.get("/checkpoints")
async def list_checkpoints(output_dir: str = None):
    """List available checkpoints in the output directory.

    Args:
        output_dir: Output directory path. If not provided, uses current config.

    Returns:
        List of checkpoint information
    """
    try:
        # If no output_dir provided, try to get from current config
        if not output_dir:
            config = APIState.get_state("training_config", {})
            output_dir = config.get("--output_dir")

        if not output_dir:
            return {"error": "No output directory specified", "checkpoints": []}

        # Initialize CheckpointManager
        checkpoint_manager = CheckpointManager(output_dir)

        # Get list of checkpoints with metadata
        checkpoints = checkpoint_manager.list_checkpoints(include_metadata=True)

        # Add latest flag to the most recent checkpoint
        if checkpoints:
            checkpoints[0]["is_latest"] = True

        return {
            "output_dir": output_dir,
            "checkpoints": checkpoints,
            "total": len(checkpoints)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints/validate/{checkpoint_name}")
async def validate_checkpoint(checkpoint_name: str, output_dir: str = None):
    """Validate a specific checkpoint for resuming training.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "checkpoint-1000")
        output_dir: Output directory path. If not provided, uses current config.

    Returns:
        Validation result with status and message
    """
    try:
        # If no output_dir provided, try to get from current config
        if not output_dir:
            config = APIState.get_state("training_config", {})
            output_dir = config.get("--output_dir")

        if not output_dir:
            return {"valid": False, "message": "No output directory specified"}

        # Initialize CheckpointManager
        checkpoint_manager = CheckpointManager(output_dir)

        # Validate the checkpoint
        is_valid, error_message = checkpoint_manager.validate_checkpoint(checkpoint_name)

        return {
            "checkpoint": checkpoint_name,
            "valid": is_valid,
            "message": error_message or "Checkpoint is valid and ready for resuming"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
