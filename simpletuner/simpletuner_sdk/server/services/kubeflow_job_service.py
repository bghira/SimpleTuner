"""Lifecycle service for Kubeflow-backed SimpleTuner jobs."""

from __future__ import annotations

import asyncio
import copy
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from ..models.worker import WorkerType
from .cloud.base import CloudJobStatus, JobType, UnifiedJob
from .kubeflow import (
    KUBEFLOW_PROVIDER,
    LOCAL_UPLOAD_BUCKET,
    KubeflowPhase,
    KubeflowResources,
    KubeflowSettings,
    KubeflowWorkerProvisioner,
)
from .worker_credentials import create_worker_credentials

logger = logging.getLogger(__name__)


class KubeflowJobService:
    """Coordinate SimpleTuner jobs with one-shot Kubeflow Workers."""

    def __init__(
        self,
        *,
        settings: KubeflowSettings,
        provisioner: KubeflowWorkerProvisioner,
        job_store: Any,
        worker_repository: Any,
    ) -> None:
        """Initialize the lifecycle service.

        Args:
            settings: Validated Kubeflow configuration.
            provisioner: Kubernetes resource provisioner.
            job_store: Unified SimpleTuner job store.
            worker_repository: Worker persistence repository.
        """
        settings.validate()
        self.settings = settings
        self.provisioner = provisioner
        self.job_store = job_store
        self.worker_repository = worker_repository
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def submit(
        self,
        *,
        config_name: str,
        config: dict[str, Any],
        user_id: Optional[int],
    ) -> UnifiedJob:
        """Submit one config as a queued single-GPU TrainJob.

        Args:
            config_name: Stored SimpleTuner configuration name.
            config: Fully materialized training configuration.
            user_id: Submitting user identifier.

        Returns:
            Persisted queued job.
        """
        suffix = uuid.uuid4().hex[:12]
        job_id = f"kjob-{suffix}"
        worker_id = f"worker-{suffix}"
        credentials = await create_worker_credentials(
            self.worker_repository,
            worker_id=worker_id,
            name=worker_id,
            user_id=user_id or 0,
            worker_type=WorkerType.EPHEMERAL,
            provider=KUBEFLOW_PROVIDER,
            current_job_id=job_id,
        )
        now = datetime.now(timezone.utc).isoformat()
        upload_token = secrets.token_urlsafe(32)
        job_config = self._with_local_publishing_config(config, job_id, upload_token)
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.LOCAL,
            provider=KUBEFLOW_PROVIDER,
            status=CloudJobStatus.QUEUED.value,
            config_name=config_name,
            created_at=now,
            queued_at=now,
            user_id=user_id,
            num_processes=1,
            upload_token=upload_token,
            output_url=f"/api/cloud/storage/{LOCAL_UPLOAD_BUCKET}/{job_id}",
            metadata={
                "config": job_config,
                "target": KUBEFLOW_PROVIDER,
                "worker_id": worker_id,
                "provisioner": KUBEFLOW_PROVIDER,
                "infrastructure_phase": KubeflowPhase.WAITING.value,
            },
        )
        await self.job_store.add_job(job)

        try:
            resources = await self.provisioner.create(
                job_id=job_id,
                worker_id=worker_id,
                worker_token=credentials.token,
            )
        except Exception as exc:
            await self.worker_repository.delete_worker(worker_id)
            await self.job_store.update_job(
                job_id,
                {
                    "status": CloudJobStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": f"Kubeflow TrainJob creation failed: {exc}",
                },
            )
            raise

        job.metadata["kubernetes"] = resources.to_dict()
        await self.job_store.update_job(job_id, {"metadata": job.metadata})
        return job

    def _with_local_publishing_config(
        self,
        config: dict[str, Any],
        job_id: str,
        upload_token: str,
    ) -> dict[str, Any]:
        """Append the Server's existing S3-compatible output destination.

        The resulting training configuration is consumed by the normal
        SimpleTuner publishing path. The Worker remains unaware of Kubeflow.

        Args:
            config: User training configuration.
            job_id: Job identifier used as the remote object prefix.
            upload_token: Per-job credential accepted by the Server.

        Returns:
            A copied configuration containing the central publishing target.

        Raises:
            ValueError: If an existing publishing configuration has an
                unsupported representation.
        """
        prepared = copy.deepcopy(config)
        existing = prepared.get("publishing_config")
        if existing in (None, "", [], {}, "None"):
            publishing: list[dict[str, Any]] = []
        elif isinstance(existing, dict):
            publishing = [existing]
        elif isinstance(existing, list):
            publishing = list(existing)
        else:
            raise ValueError("Kubeflow jobs require publishing_config to be a mapping or list")

        endpoint = f"{str(self.settings.orchestrator_url).rstrip('/')}/api/cloud/storage"
        publishing.append(
            {
                "provider": "s3",
                "bucket": LOCAL_UPLOAD_BUCKET,
                "endpoint_url": endpoint,
                "access_key": "local",
                "secret_key": upload_token,
                "base_path": job_id,
                "use_ssl": urlparse(endpoint).scheme == "https",
                "request_headers": {"X-SimpleTuner-Secret": upload_token},
                "force_single_part": True,
                "required": True,
            }
        )
        prepared["publishing_config"] = publishing
        return prepared


    @staticmethod
    def artifacts_received(job: UnifiedJob) -> bool:
        """Check whether the Server has received a LoRA weight artifact.

        Args:
            job: Kubeflow job whose upload metadata should be inspected.

        Returns:
            True when at least one centrally registered safetensors file exists.
        """
        metadata = getattr(job, "metadata", None) or {}
        upload_state = metadata.get("artifact_upload") or {}
        received_files = upload_state.get("received_files") or []
        return any(
            str(object_path).lower().endswith(".safetensors")
            for object_path in received_files
        )

    async def _archive_logs(self, job: UnifiedJob) -> dict[str, Any]:
        """Persist the ephemeral Worker log before Kubernetes cleanup.

        Args:
            job: Kubeflow job whose Worker Pod is still available.

        Returns:
            Updated job metadata, including the archived log path when present.
        """
        metadata = dict(job.metadata or {})
        try:
            logs = await self.get_logs(job)
        except Exception as exc:
            logger.warning("Could not read Kubeflow logs for %s: %s", job.job_id, exc)
            return metadata
        if not isinstance(logs, str) or not logs:
            return metadata

        from ..routes.cloud.helpers import get_local_upload_dir

        relative_path = f"{LOCAL_UPLOAD_BUCKET}/{job.job_id}/worker.log"
        log_path = get_local_upload_dir() / relative_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(logs, encoding="utf-8")

        upload_state = dict(metadata.get("artifact_upload") or {})
        received_files = list(upload_state.get("received_files") or [])
        if relative_path not in received_files:
            received_files.append(relative_path)
        upload_state["received_files"] = received_files
        metadata["artifact_upload"] = upload_state
        metadata["worker_log"] = relative_path
        await self.job_store.update_job(job.job_id, {"metadata": metadata})
        return metadata

    async def get_logs(self, job: UnifiedJob) -> str:
        """Read current Kubernetes logs through the Server adapter.

        Args:
            job: Kubeflow job containing TrainJob resource metadata.

        Returns:
            Current Worker Pod log text.
        """
        resources = self._resources_for_job(job)
        return await self.provisioner.get_logs(resources.trainjob_name)

    async def get_managed_job(self, job_id: str) -> Optional[UnifiedJob]:
        """Return a job only when this service owns its runtime lifecycle.

        Args:
            job_id: SimpleTuner job identifier.

        Returns:
            The Kubeflow-backed job, or None for an unknown or differently
            provisioned job.
        """
        job = await self.job_store.get_job(job_id)
        if job is None or job.provider != KUBEFLOW_PROVIDER:
            return None
        return job

    async def cancel(self, job_id: str) -> None:
        """Cancel a Kubeflow job and release all ephemeral resources.

        Args:
            job_id: SimpleTuner job identifier.

        Raises:
            ValueError: If the job does not exist or lacks resource metadata.
        """
        job = await self.get_managed_job(job_id)
        if job is None:
            raise ValueError(f"Kubeflow-managed job not found: {job_id}")

        worker_id = job.metadata.get("worker_id")
        if worker_id:
            from ..routes.workers import is_worker_connected, push_to_worker

            if is_worker_connected(worker_id):
                await push_to_worker(worker_id, {"type": "job_cancel", "job_id": job_id})

        resources = self._resources_for_job(job)
        metadata = await self._archive_logs(job)
        metadata["infrastructure_phase"] = CloudJobStatus.CANCELLED.value
        await self.provisioner.delete(resources)
        if worker_id:
            await self.worker_repository.delete_worker(worker_id)
        await self.job_store.update_job(
            job_id,
            {
                "status": CloudJobStatus.CANCELLED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
            },
        )

    async def finalize(self, job_id: str) -> None:
        """Release resources after a Worker reports a terminal state.

        Args:
            job_id: Completed, failed, or cancelled job identifier.
        """
        job = await self.job_store.get_job(job_id)
        if job is None:
            return
        metadata = await self._archive_logs(job)
        metadata["infrastructure_phase"] = job.status
        await self.job_store.update_job(job_id, {"metadata": metadata})
        await self.provisioner.delete(self._resources_for_job(job))
        worker_id = job.metadata.get("worker_id")
        if worker_id:
            await self.worker_repository.delete_worker(worker_id)

    async def reconcile_once(self) -> None:
        """Synchronize active TrainJob infrastructure state into job metadata."""
        jobs = await self.job_store.list_jobs(limit=1000, provider=KUBEFLOW_PROVIDER)
        for job in jobs:
            if job.is_terminal or not job.metadata.get("kubernetes"):
                continue
            resources = self._resources_for_job(job)
            phase = await self.provisioner.get_phase(resources.trainjob_name)
            metadata = dict(job.metadata)
            metadata["infrastructure_phase"] = phase.value
            updates: dict[str, Any] = {"metadata": metadata}
            terminal = False
            if phase in {KubeflowPhase.FAILED, KubeflowPhase.MISSING}:
                terminal = True
                updates.update(
                    {
                        "status": CloudJobStatus.FAILED.value,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "error_message": "Kubeflow TrainJob failed before Worker completion",
                    }
                )
            elif phase == KubeflowPhase.COMPLETED:
                terminal = True
                terminal_status = (
                    CloudJobStatus.COMPLETED.value
                    if self.artifacts_received(job)
                    else CloudJobStatus.FAILED.value
                )
                updates.update(
                    {
                        "status": terminal_status,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if terminal_status == CloudJobStatus.FAILED.value:
                    updates["error_message"] = (
                        "Kubeflow Worker exited without confirmed central artifact upload"
                    )

            if terminal:
                archived_metadata = await self._archive_logs(job)
                archived_metadata["infrastructure_phase"] = phase.value
                updates["metadata"] = archived_metadata
                await self.provisioner.delete(resources)
                worker_id = job.metadata.get("worker_id")
                if worker_id:
                    await self.worker_repository.delete_worker(worker_id)
            await self.job_store.update_job(job.job_id, updates)

    async def start(self) -> None:
        """Start periodic TrainJob reconciliation."""
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._reconcile_loop())

    async def stop(self) -> None:
        """Stop periodic TrainJob reconciliation."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _reconcile_loop(self) -> None:
        """Run reconciliation until service shutdown."""
        while not self._stop_event.is_set():
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("Kubeflow job reconciliation failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    @staticmethod
    def _resources_for_job(job: UnifiedJob) -> KubeflowResources:
        """Load Kubernetes resource references from a job.

        Args:
            job: Unified job containing Kubernetes metadata.

        Returns:
            Kubernetes resource references.

        Raises:
            ValueError: If resource metadata is absent.
        """
        values = job.metadata.get("kubernetes")
        if not values:
            raise ValueError(f"Job {job.job_id} has no Kubernetes resource metadata")
        return KubeflowResources.from_dict(values)


_kubeflow_job_service: Optional[KubeflowJobService] = None


def get_kubeflow_job_service() -> Optional[KubeflowJobService]:
    """Return the initialized Kubeflow job service, if enabled."""
    return _kubeflow_job_service


async def initialize_kubeflow_job_service(
    settings: Optional[KubeflowSettings] = None,
) -> Optional[KubeflowJobService]:
    """Initialize the singleton Kubeflow job service when enabled.

    Args:
        settings: Optional explicit settings for startup and tests.

    Returns:
        Running service, or None when Kubeflow integration is disabled.
    """
    global _kubeflow_job_service
    resolved = settings or KubeflowSettings.from_env()
    if not resolved.enabled:
        return None
    if _kubeflow_job_service is None:
        from .cloud.container import get_job_store
        from .worker_repository import get_worker_repository

        _kubeflow_job_service = KubeflowJobService(
            settings=resolved,
            provisioner=KubeflowWorkerProvisioner(resolved),
            job_store=get_job_store(),
            worker_repository=get_worker_repository(),
        )
        await _kubeflow_job_service.start()
    return _kubeflow_job_service


async def shutdown_kubeflow_job_service() -> None:
    """Stop and clear the singleton Kubeflow job service."""
    global _kubeflow_job_service
    if _kubeflow_job_service is not None:
        await _kubeflow_job_service.stop()
        _kubeflow_job_service = None
