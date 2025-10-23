"""Training control routes for HTMX interface."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

try:  # pragma: no cover - optional dependency
    from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
except Exception:  # websockets may not be installed in all environments
    ConnectionClosed = ConnectionClosedOK = RuntimeError
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.simpletuner_sdk import process_keeper
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services import training_service


def _json_default(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


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

    bundle = training_service.build_config_bundle(form_data)
    validation_result = training_service.validate_training_config(
        bundle.store,
        bundle.complete_config,
        bundle.config_dict,
        strict_epoch_exclusivity=True,
    )

    errors = validation_result.errors
    warnings = validation_result.warnings
    suggestions = validation_result.suggestions

    if errors:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Validation Failed</h6>
            <ul class="mb-0">
                {''.join(f'<li>{error}</li>' for error in errors)}
            </ul>
        </div>
        """

    if warnings or suggestions:
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
        return html

    return """
    <div class="alert alert-success">
        <h6><i class="fas fa-check-circle"></i> Configuration Valid</h6>
        <p class="mb-0">All settings look good! Ready to start training.</p>
    </div>
    """


@router.post("/configuration/check", response_class=HTMLResponse)
async def configuration_check(request: Request):
    """Compatibility endpoint for legacy clients expecting configuration/check."""
    return await validate_config(request)


@router.post("/config", response_class=HTMLResponse)
async def save_config(request: Request):
    """Save training configuration."""
    form_data = await request.form()

    try:
        bundle = training_service.build_config_bundle(form_data)
        training_service.persist_config_bundle(bundle)
        return """
        <div class="text-success" x-data="{ show: true }" x-show="show" x-init="setTimeout(() => show = false, 3000)" x-transition.opacity.duration.500ms>
            <i class="fas fa-check"></i> Configuration saved
        </div>
        """
    except Exception as exc:
        return f"""
        <div class="text-danger">
            <i class="fas fa-exclamation-triangle"></i> Failed to save: {str(exc)}
        </div>
        """


@router.post("/start", response_class=HTMLResponse)
async def start_training(request: Request):
    """Start training with current configuration."""
    form_data = await request.form()
    bundle = training_service.build_config_bundle(form_data)

    validation_result = training_service.validate_training_config(
        bundle.store,
        bundle.complete_config,
        bundle.config_dict,
    )

    errors = list(validation_result.errors)
    if not validation_result.raw_validation.is_valid and not errors:
        errors.append("Configuration validation failed. Please review your settings.")

    if errors:
        error_list = "".join(f"<li>{error}</li>" for error in errors)
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Cannot Start Training</h6>
            <ul class="mb-0">
                {error_list}
            </ul>
        </div>
        """

    try:
        training_service.persist_config_bundle(bundle)
        job_id = training_service.start_training_job(bundle.complete_config)

        return f"""
        <div class="alert alert-info">
            <h6><i class="fas fa-cog fa-spin"></i> Training Starting</h6>
            <p>Your training job is being initialized.</p>
            <p class="mb-0"><small>Job ID: {job_id}</small></p>
        </div>
        """
    except Exception as exc:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Failed to Start Training</h6>
            <p>{str(exc)}</p>
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
        if training_service.terminate_training_job(job_id, status="stopped", clear_job_id=False):
            return {"message": f"Training job {job_id} stop requested"}
        return {"message": "No active training job to stop"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cancel", response_class=HTMLResponse)
async def cancel_training(request: Request):
    """Cancel current training and return HTML response for HTMX."""
    try:
        form_data = await request.form()
        job_id = form_data.get("job_id", APIState.get_state("current_job_id"))

        if training_service.terminate_training_job(job_id, status="cancelled", clear_job_id=True):
            return f"""
            <div class="alert alert-warning">
                <h6><i class="fas fa-hand-paper"></i> Training Cancelled</h6>
                <p>Training job has been cancelled successfully.</p>
                <p class="mb-0"><small>Job ID: {job_id}</small></p>
            </div>
            """
        return f"""
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle"></i> No Active Training</h6>
                <p>There is no active training job to cancel.</p>
            </div>
            """
    except Exception as exc:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Failed to Cancel Training</h6>
            <p>{str(exc)}</p>
        </div>
        """


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    status = APIState.get_state("training_status") or "idle"
    config = APIState.get_state("training_config") or {}
    job_id = APIState.get_state("current_job_id")
    progress = APIState.get_state("training_progress") or None

    # Get detailed job status if available
    job_info = None
    if job_id:
        try:
            # Get process status
            job_status = str(process_keeper.get_process_status(job_id) or "").strip()
            if job_status:
                job_info = {"status": job_status}

                normalized_status = job_status.lower()
                mapped_status = None

                if normalized_status in {"failed", "crashed"}:
                    mapped_status = "error"
                elif normalized_status == "completed":
                    mapped_status = "completed"
                elif normalized_status == "terminated":
                    mapped_status = "cancelled"
                elif normalized_status == "running":
                    mapped_status = "running"
                elif normalized_status in {"pending", "starting"}:
                    mapped_status = "starting"

                if mapped_status and mapped_status != status:
                    status = mapped_status
                    APIState.set_state("training_status", status)
                    if mapped_status in {"error", "cancelled"}:
                        APIState.set_state("training_progress", None)
                    if mapped_status in {"error", "cancelled", "completed"}:
                        APIState.set_state("training_startup_stages", {})
                        APIState.set_state("current_job_id", None)
                        job_id = None
                    if mapped_status == "completed":
                        progress_state = APIState.get_state("training_progress") or {}
                        if isinstance(progress_state, Mapping):
                            progress_state = dict(progress_state)
                            progress_state["percent"] = 100
                            APIState.set_state("training_progress", progress_state)
            else:
                job_info = {"status": job_status or "unknown"}
        except Exception:
            # Process keeper might not have this job
            pass

    raw_startup_stages = APIState.get_state("training_startup_stages") or {}
    startup_stages = (
        {key: value for key, value in raw_startup_stages.items() if isinstance(value, Mapping)}
        if isinstance(raw_startup_stages, Mapping)
        else {}
    )

    if startup_stages:
        stage_statuses = {
            str(stage.get("status", "")).lower() for stage in startup_stages.values() if isinstance(stage, Mapping)
        }
        if any(status_name in {"failed", "error"} for status_name in stage_statuses):
            status = "error"
        elif status not in {"failed", "error", "fatal", "cancelled", "stopped", "completed", "success", "running"}:
            status = "starting"

    return {
        "status": status,
        "config": config,
        "job_id": job_id,
        "job_info": job_info,
        "progress": progress,
        "startup_progress": startup_stages,
    }


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
                            try:
                                serialisable_event = json.loads(json.dumps(event, default=_json_default))
                                await websocket.send_json({"job_id": job_id, "event": serialisable_event})
                            except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedOK, RuntimeError):
                                return

                        last_index += len(events)
                except Exception as e:
                    # Send error but continue streaming
                    try:
                        await websocket.send_json({"error": str(e), "job_id": job_id})
                    except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedOK, RuntimeError):
                        return

            # Wait a bit before checking for new events
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except (ConnectionClosed, ConnectionClosedOK):
        # Underlying websocket library already closed
        pass
    except asyncio.CancelledError:
        # Event loop shutting down – exit quietly without logging stack traces
        try:
            await websocket.close()
        except Exception:
            pass
        return
    except Exception as e:
        # Send error and close
        try:
            await websocket.send_json({"error": str(e)})
        except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedOK, RuntimeError):
            pass
        try:
            await websocket.close()
        except Exception:
            pass


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

        return {"output_dir": output_dir, "checkpoints": checkpoints, "total": len(checkpoints)}
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
            "message": error_message or "Checkpoint is valid and ready for resuming",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
