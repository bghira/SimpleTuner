"""Training control routes for HTMX interface."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from html import escape
from typing import Any, Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

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
from simpletuner.simpletuner_sdk.server.services.callback_service import get_default_callback_service
from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user, get_current_user_ws
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

from ..services.worker_groups import launch_inventory as _launch_workers
from ..services.worker_groups import resolve_worker_group

logger = logging.getLogger(__name__)


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
async def validate_config(request: Request, _user: User = Depends(get_current_user)):
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
async def configuration_check(request: Request, _user: User = Depends(get_current_user)):
    """Compatibility endpoint for legacy clients expecting configuration/check."""
    return await validate_config(request)


@router.post("/config", response_class=HTMLResponse)
async def save_config(request: Request, _user: User = Depends(get_current_user)):
    """Save training configuration."""
    form_data = await request.form()

    try:
        bundle = training_service.build_config_bundle(form_data)
        if bundle.default_worker_ids:
            resolve_worker_group(bundle.default_worker_ids, await _launch_workers(_user))
        training_service.persist_config_bundle(bundle)
        return """
        <div class="text-success" x-data="{ show: true }" x-show="show" x-init="setTimeout(() => show = false, 3000)" x-transition.opacity.duration.500ms>
            <i class="fas fa-check"></i> Configuration saved
        </div>
        """
    except ValueError as exc:
        return HTMLResponse(f'<div class="text-danger">{escape(str(exc))}</div>', status_code=422)
    except Exception as exc:
        return HTMLResponse(f'<div class="text-danger">Failed to save: {escape(str(exc))}</div>', status_code=500)


@router.get("/launch-workers")
async def list_launch_workers(
    _user: User = Depends(get_current_user), include_offline: bool = False, environment: str | None = None
):
    from .workers import is_worker_connected

    workers = [
        {
            "id": worker.worker_id,
            "name": worker.name,
            "gpu": worker.gpu_info.get("name"),
            "status": worker.status.value if is_worker_connected(worker.worker_id) else "offline",
            "multinode_gpus": worker.multinode_gpus,
            "address": worker.gpu_info.get("distributed_address"),
        }
        for worker in await _launch_workers(_user, include_offline=include_offline)
    ]
    if environment is None:
        return workers
    store = training_service.get_config_store()
    if environment != (store.get_active_config() or "default"):
        raise HTTPException(409, "The active environment changed. Reload this page.")
    try:
        _, metadata = store.load_config(environment)
        ids = metadata.default_worker_ids
    except FileNotFoundError:
        ids = []
    return {"workers": workers, "default_worker_ids": ids}


@router.post("/start", response_class=HTMLResponse)
async def start_training(request: Request, _user: User = Depends(get_current_user)):
    """Start training with current configuration."""
    destination = request.headers.get("X-Training-Destination", "environment")
    form_data = await request.form()
    bundle = training_service.build_config_bundle(form_data)
    environment = request.headers.get("X-Training-Environment")
    if environment and environment != bundle.active_config:
        return HTMLResponse("The active environment changed. Retry from the current environment.", status_code=409)

    selected = []
    if destination in ("environment", "saved-group", "local-only"):
        try:
            _, metadata = bundle.store.load_config(bundle.active_config)
            ids = metadata.default_worker_ids
        except FileNotFoundError:
            ids = []
        if destination == "saved-group" and not ids:
            return '<div class="alert alert-warning">The default workers changed. Run again to choose a destination.</div>'
        if destination == "local-only" and not ids and await _launch_workers(_user):
            return '<div class="alert alert-warning">A worker just connected. Run again to choose a destination.</div>'
    else:
        ids = [] if destination == "local" else destination.split(",")
    if ids:
        try:
            selected = resolve_worker_group(ids, await _launch_workers(_user))
        except ValueError as exc:
            return f'<div class="alert alert-danger">{escape(str(exc))}</div>'

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
        if selected:
            from .queue import submit_selected_worker_job

            config = {key.removeprefix("--"): value for key, value in bundle.complete_config.items()}
            # The local launcher injects a loopback webhook into the form snapshot.
            config["webhook_config"] = bundle.save_config.get("webhook_config", [])
            result = await submit_selected_worker_job(bundle.active_config, config, bundle.store, _user, ids)
            if not result.success:
                return f'<div class="alert alert-danger">{escape(result.error)}</div>'
            state = "queued for" if result.status == "queued" else "submitted to"
            names = ", ".join(escape(w.name) for w in selected)
            return (
                f'<div class="alert alert-info">Training {state} <strong>{names}</strong>.'
                f"<br><small>Job ID: {escape(result.job_id)}</small></div>"
            )

        result = training_service.start_training_job(
            bundle.complete_config,
            env_name=bundle.active_config,
            user_id=_user.id,
            org_id=_user.org_id,
        )

        if result.status == "rejected":
            return f"""
            <div class="alert alert-warning">
                <h6><i class="fas fa-exclamation-triangle"></i> Training Rejected</h6>
                <p>{result.reason or "Required GPUs unavailable"}</p>
            </div>
            """

        if result.status == "queued":
            return f"""
            <div class="alert alert-info">
                <h6><i class="fas fa-clock"></i> Training Queued</h6>
                <p>Your job is queued on Local at position {result.queue_position}.</p>
                <p class="mb-0"><small>Job ID: {result.job_id}</small></p>
            </div>
            """

        return f"""
        <div class="alert alert-info" data-local-training-started>
            <h6><i class="fas fa-cog fa-spin"></i> Training Starting</h6>
            <p>Your training job is being initialized.</p>
            <p class="mb-0"><small>Job ID: {result.job_id}</small></p>
        </div>
        """
    except Exception as exc:
        return f"""
        <div class="alert alert-danger">
            <h6><i class="fas fa-exclamation-triangle"></i> Failed to Start Training</h6>
            <p>{escape(str(exc))}</p>
        </div>
        """


@router.post("/configuration/run", response_class=HTMLResponse)
async def configuration_run(request: Request, _user: User = Depends(get_current_user)):
    """Compatibility endpoint mapping to the legacy configuration/run URL."""
    return await start_training(request, _user)


@router.post("/stop")
async def stop_training(_user: User = Depends(get_current_user)):
    """Stop current training."""
    try:
        job_id = APIState.get_state("current_job_id")
        if training_service.terminate_training_job(job_id, status="stopped", clear_job_id=False):
            return {"message": f"Training job {job_id} stop requested"}
        return {"message": "No active training job to stop"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _manual_targets(user: User) -> list[dict]:
    from ..models.worker import WorkerStatus
    from ..services.cloud.storage.job_repository import get_job_repository
    from ..services.worker_repository import get_worker_repository
    from .workers import manual_control_workers, worker_streams

    jobs = get_job_repository()
    workers = get_worker_repository()
    targets = []
    can_manage_all = user.has_permission("job.cancel.all")
    can_manage_own = user.has_permission("job.submit")
    candidates = [
        (job_id, None)
        for job_id, process in process_keeper.list_processes(target=training_service.run_trainer_job).items()
        if process["alive"] and process["status"] == "running"
    ]
    for worker_id in tuple(worker_streams):
        worker = await workers.get_worker(worker_id)
        if worker and worker.current_job_id and worker.status in (WorkerStatus.BUSY, WorkerStatus.DRAINING):
            candidates.append((worker.current_job_id, worker))
    for job_id, worker in candidates:
        job = await jobs.get(job_id)
        if not can_manage_all and not (can_manage_own and job and job.user_id == user.id):
            continue
        if job and job.is_terminal:
            continue
        if worker and (
            not job
            or job.status not in {"starting", "running", "training"}
            or (job.metadata or {}).get("worker_id") != worker.worker_id
        ):
            continue
        targets.append(
            {
                "job_id": job_id,
                "worker_id": worker.worker_id if worker else None,
                "name": worker.name if worker else "Local",
                "config_name": job.config_name if job else None,
                "gpu": (worker.gpu_info or {}).get("name") if worker else None,
                "available": not worker or worker.worker_id in manual_control_workers,
                "is_current": worker is None and job_id == APIState.get_state("current_job_id"),
            }
        )
    return targets


@router.get("/control-targets")
async def list_manual_targets(_user: User = Depends(get_current_user)):
    return await _manual_targets(_user)


async def _trigger_manual_action(
    action: Literal["checkpoint", "validation"], job_id: str | None, worker_id: str | None, user: User
):
    from .workers import manual_control_workers, push_to_worker

    selected = job_id if job_id is not None else APIState.get_state("current_job_id")
    if not selected:
        raise HTTPException(status_code=400, detail="No active training job")
    targets = await _manual_targets(user)
    target = next((item for item in targets if item["job_id"] == selected and item["worker_id"] == worker_id), None)
    if target is None:
        raise HTTPException(status_code=409, detail="This training job is no longer available.")
    if worker_id:
        if worker_id not in manual_control_workers:
            raise HTTPException(status_code=409, detail="Update the worker agent to enable manual actions.")
        if not await push_to_worker(worker_id, {"type": "job_manual_action", "job_id": selected, "action": action}):
            raise HTTPException(status_code=409, detail="The training worker disconnected.")
    else:
        request_action = (
            training_service.request_manual_checkpoint
            if action == "checkpoint"
            else training_service.request_manual_validation
        )
        try:
            request_action(selected)
        except Exception:
            if process_keeper.get_process_status(selected) != "running":
                raise HTTPException(status_code=409, detail="The selected training process has stopped.")
            raise
    return {"job_id": selected, "message": f"{action.capitalize()} queued for {target['name']} at the next gradient sync."}


@router.post("/validation/run")
async def trigger_manual_validation(
    job_id: str | None = None, worker_id: str | None = None, _user: User = Depends(get_current_user)
):
    return await _trigger_manual_action("validation", job_id, worker_id, _user)


@router.post("/checkpoint/run")
async def trigger_manual_checkpoint(
    job_id: str | None = None, worker_id: str | None = None, _user: User = Depends(get_current_user)
):
    return await _trigger_manual_action("checkpoint", job_id, worker_id, _user)


@router.post("/cancel", response_class=HTMLResponse)
async def cancel_training(request: Request, _user: User = Depends(get_current_user)):
    """Cancel current training and return HTML response for HTMX."""
    try:
        form_data = await request.form()
        job_id = form_data.get("job_id", APIState.get_state("current_job_id"))

        local_ids = {target["job_id"] for target in await _manual_targets(_user) if target["worker_id"] is None}
        if job_id in local_ids and training_service.terminate_training_job(job_id, status="cancelled", clear_job_id=True):
            return f"""
            <div class="alert alert-warning" data-local-training-cancelled>
                <h6><i class="fas fa-hand-paper"></i> Training Cancelled</h6>
                <p>Training job has been cancelled successfully.</p>
                <p class="mb-0"><small>Job ID: {escape(job_id)}</small></p>
            </div>
            """
        return f"""
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle"></i> No Active Training</h6>
                <p>The selected local training job is no longer active.</p>
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
async def get_training_status(_user: User = Depends(get_current_user)):
    """Get current training status."""
    status = APIState.get_state("training_status") or "idle"
    config = APIState.get_state("training_config") or {}
    job_id = APIState.get_state("current_job_id")
    active_job_id = job_id
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

                if normalized_status in {"failed", "error", "fatal", "crashed"}:
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
                        logger.info(
                            "Clearing current_job_id=%s due to terminal status=%s (process_status=%s)",
                            job_id,
                            mapped_status,
                            normalized_status,
                        )
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

        if job_info and status == "error":
            try:
                events = process_keeper.get_process_events(active_job_id) if active_job_id else []
            except Exception:
                events = []

            latest_error: dict | None = None
            for event in reversed(events):
                if not isinstance(event, Mapping):
                    continue
                event_type = str(event.get("type") or "").lower()
                if event_type == "error":
                    latest_error = event
                    break
                if event_type == "state":
                    data = event.get("data") or {}
                    state_text = str(data.get("status") or "").lower()
                    if state_text in {"failed", "error", "fatal"}:
                        latest_error = {"data": data}
                        break

            if latest_error and isinstance(latest_error, Mapping):
                error_payload = latest_error.get("data") if isinstance(latest_error.get("data"), Mapping) else {}
                message = str(error_payload.get("message") or error_payload.get("error") or "").strip()
                job_info["status_detail"] = "error_with_details"
                job_info["error"] = dict(error_payload)
                if message:
                    job_info["message"] = message
            else:
                job_info["status_detail"] = "crashed_without_details"

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
async def get_training_events(request: Request, since_index: int = 0, _user: User = Depends(get_current_user)):
    """Get training events since a given index."""
    job_id = APIState.get_state("current_job_id")

    result = {"events": [], "job_id": job_id}
    if request.app.state.mode.value == "trainer":
        _, result["training_runs"] = get_default_callback_service().training_runs_since(-1, _user)
    if not job_id:
        return result

    try:
        events = process_keeper.get_process_events(job_id, since_index)
        return {**result, "events": events, "next_index": since_index + len(events)}
    except Exception as e:
        # Job might not exist
        return {**result, "error": str(e)}


@router.websocket("/events/stream")
async def stream_training_events(websocket: WebSocket, _user: User = Depends(get_current_user_ws)):
    """Stream training events via WebSocket."""
    await websocket.accept()

    try:
        last_index = 0
        runs_revision = -1
        while True:
            if websocket.app.state.mode.value == "trainer":
                runs_revision, runs = get_default_callback_service().training_runs_since(runs_revision, _user)
                if runs is not None:
                    await websocket.send_json({"training_runs": runs})

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
async def list_checkpoints(output_dir: str = None, _user: User = Depends(get_current_user)):
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
async def validate_checkpoint(checkpoint_name: str, output_dir: str = None, _user: User = Depends(get_current_user)):
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
