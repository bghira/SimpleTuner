"""Pre-submission utility endpoints for cloud training.

Provides endpoints for cost estimation, hardware options, and config listing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from ._shared import (
    AvailableConfigsResponse,
    CostEstimateResponse,
    HardwareOption,
    HardwareOptionsResponse,
    PreSubmitCheckResponse,
    get_active_config,
    get_job_store,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/pre-submit-check", response_model=PreSubmitCheckResponse)
async def pre_submit_check() -> PreSubmitCheckResponse:
    """Check git status and get tracker_run_name for the pre-submit modal."""
    from ...services.git_config_service import GIT_CONFIG_SERVICE

    result = PreSubmitCheckResponse()

    try:
        head_info = GIT_CONFIG_SERVICE.get_head_info(config_type="model")
        result.repo_present = head_info.repo_present
        result.git_available = True

        if head_info.repo_present:
            result.current_commit = head_info.commit
            result.current_abbrev = head_info.abbrev
            result.current_branch = head_info.branch
            result.is_dirty = head_info.is_dirty
            result.dirty_count = head_info.dirty_count
            result.dirty_paths = head_info.dirty_paths or []
    except Exception as exc:
        logger.warning("Failed to get git status for pre-submit: %s", exc)
        result.git_available = False

    try:
        config = await get_active_config()
        if config:
            result.tracker_run_name = config.get("tracker_run_name") or config.get("--tracker_run_name")
            result.config_name = config.get("_name") or config.get("--output_dir", "").split("/")[-1]
    except Exception as exc:
        logger.warning("Failed to get config for pre-submit: %s", exc)

    return result


@router.get("/available-configs", response_model=AvailableConfigsResponse)
async def list_available_configs() -> AvailableConfigsResponse:
    """List available configurations for cloud job submission."""
    from ...services.config_store import ConfigStore
    from ...services.webui_state import WebUIStateStore

    try:
        defaults = WebUIStateStore().load_defaults()
        if not defaults.configs_dir:
            return AvailableConfigsResponse()

        store = ConfigStore(
            config_dir=Path(defaults.configs_dir).expanduser(),
            config_type="model",
        )
        config_list = store.list_configs()

        configs = []
        for config_data in config_list:
            config_name = config_data.get("name") if isinstance(config_data, dict) else str(config_data)
            if config_name:
                configs.append(
                    {
                        "name": config_name,
                        "is_active": config_name == defaults.active_config,
                    }
                )

        return AvailableConfigsResponse(configs=configs, active_config=defaults.active_config)
    except Exception as exc:
        logger.warning("Failed to list configs: %s", exc)
        return AvailableConfigsResponse()


@router.post("/cost-estimate", response_model=CostEstimateResponse)
async def get_cost_estimate(
    dataloader_config: List[Dict[str, Any]] = [],
) -> CostEstimateResponse:
    """Estimate cost for a training job based on historical data."""
    from ...services.cloud.cloud_upload_service import CloudUploadService
    from ...services.cloud.replicate_client import get_default_hardware_cost_per_hour

    store = get_job_store()

    upload_service = CloudUploadService()
    data_size_bytes = upload_service.estimate_upload_size(dataloader_config)
    data_size_mb = round(data_size_bytes / (1024 * 1024), 2) if data_size_bytes > 0 else 0

    summary = await store.get_metrics_summary(days=90)
    avg_duration = summary.get("avg_job_duration_seconds")

    cost_per_hour = get_default_hardware_cost_per_hour(store)

    if avg_duration is None:
        return CostEstimateResponse(
            has_estimate=False,
            hardware_cost_per_hour=round(cost_per_hour, 2),
            data_size_mb=data_size_mb,
            message="Not enough completed jobs to estimate cost. Run a few jobs first.",
        )

    duration_hours = avg_duration / 3600
    estimated_cost = duration_hours * cost_per_hour

    return CostEstimateResponse(
        has_estimate=True,
        estimated_cost_usd=round(estimated_cost, 2),
        avg_duration_seconds=round(avg_duration, 0),
        hardware_cost_per_hour=round(cost_per_hour, 2),
        data_size_mb=data_size_mb,
        message=f"Based on average of {round(avg_duration / 60, 1)} min from recent jobs",
    )


@router.get("/hardware-options", response_model=HardwareOptionsResponse)
async def get_hardware_options() -> HardwareOptionsResponse:
    """Get available hardware options for cloud training."""
    from ...services.cloud.replicate_client import get_hardware_info_async

    store = get_job_store()
    hardware_info = await get_hardware_info_async(store)

    hardware = []
    for hw_id, hw_info in hardware_info.items():
        cost_per_sec = hw_info.get("cost_per_second", 0.001)
        hardware.append(
            HardwareOption(
                id=hw_id,
                name=hw_info.get("name", hw_id),
                cost_per_second=cost_per_sec,
                cost_per_hour=round(cost_per_sec * 3600, 2),
            )
        )

    return HardwareOptionsResponse(
        hardware=hardware,
        default_hardware="gpu-l40s",
        message="Hardware is determined by the model endpoint. Use 'Model Version' selector to switch endpoints.",
    )
