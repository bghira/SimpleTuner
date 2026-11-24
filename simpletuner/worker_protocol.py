"""Payloads and limits shared by workers and the orchestrator."""

from pydantic import BaseModel, ConfigDict, ValidationError

WORKER_LOG_TAIL_SIZE = 50_000
WORKER_HEARTBEAT_SECONDS = 30
WORKER_JOB_PROVIDERS = {"worker", "kubeflow"}
WORKER_JOB_ACTIVE_STATUSES = {"starting", "training", "running"}


def bundled_dataloader(metadata: dict) -> list[dict] | None:
    return metadata.get("dataloader", (metadata.get("config") or {}).get("dataloader_config"))


class WorkerDatasetSummary(BaseModel):
    """Dataset settings safe to include in the remote job overview."""

    model_config = ConfigDict(allow_inf_nan=False)

    id: str
    type: str | None = None
    dataset_type: str = "image"
    disabled: bool = False
    resolution: float | int | None = None
    resolution_type: str | None = None
    crop: bool | None = None
    crop_aspect: str | None = None
    crop_style: str | None = None
    caption_strategy: str | None = None
    repeats: int | None = None


def summarize_worker_datasets(config: list[dict]) -> list[dict]:
    summaries = []
    for dataset in config:
        if not isinstance(dataset, dict):
            continue
        try:
            summary = WorkerDatasetSummary.model_validate(
                {
                    **dataset,
                    "dataset_type": dataset.get("dataset_type") or "image",
                    "disabled": bool(dataset.get("disabled") or dataset.get("disable")),
                }
            )
        except ValidationError:
            # A report must not interfere with validation of the training configuration.
            continue
        summaries.append(summary.model_dump(exclude_none=True))
    return summaries
