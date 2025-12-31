"""Concrete command implementations for job operations.

Each command encapsulates all data and logic for a specific job operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import Command, CommandContext, CommandResult

if TYPE_CHECKING:
    from ..auth.models import User

logger = logging.getLogger(__name__)


@dataclass
class SubmitJobData:
    """Data for job submission."""

    job_id: str
    status: str
    provider: str
    config_name: Optional[str] = None
    data_uploaded: bool = False
    output_url: Optional[str] = None
    idempotent_hit: bool = False
    cost_limit_warning: Optional[str] = None


class SubmitJobCommand(Command[SubmitJobData]):
    """Command to submit a training job to a cloud provider.

    Encapsulates all submission logic including:
    - Config loading and validation
    - Quota and cost limit checks
    - Data upload
    - Provider submission
    - Job record creation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataloader_config: List[Dict[str, Any]],
        provider: str = "replicate",
        config_name: Optional[str] = None,
        webhook_url: Optional[str] = None,
        tracker_run_name: Optional[str] = None,
        snapshot_name: Optional[str] = None,
        snapshot_message: Optional[str] = None,
        upload_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        user: Optional["User"] = None,
    ):
        super().__init__()
        self.config = config
        self.dataloader_config = dataloader_config
        self.provider = provider
        self.config_name = config_name
        self.webhook_url = webhook_url
        self.tracker_run_name = tracker_run_name
        self.snapshot_name = snapshot_name
        self.snapshot_message = snapshot_message
        self.upload_id = upload_id
        self.idempotency_key = idempotency_key
        self.user = user

        # For rollback
        self._created_job_id: Optional[str] = None
        # For quota reservation cleanup
        self._reservation_id: Optional[str] = None
        # For response
        self._cost_limit_warning: Optional[str] = None
        self._quota_warnings: List[str] = []

    @classmethod
    async def from_request(
        cls,
        config: Optional[Dict[str, Any]],
        dataloader_config: Optional[List[Dict[str, Any]]],
        config_name_to_load: Optional[str],
        provider: str,
        user: Optional["User"] = None,
        webhook_url: Optional[str] = None,
        tracker_run_name: Optional[str] = None,
        snapshot_name: Optional[str] = None,
        snapshot_message: Optional[str] = None,
        upload_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        config_name: Optional[str] = None,
    ) -> "SubmitJobCommand":
        """Factory method to create command from request, loading config if needed."""
        final_config = config
        final_dataloader = dataloader_config
        final_config_name = config_name

        if config_name_to_load:
            from ...config_store import ConfigStore
            from ...webui_state import WebUIStateStore

            defaults = WebUIStateStore().load_defaults()
            if not defaults.configs_dir:
                raise ValueError("No configs directory configured")

            config_store = ConfigStore(
                config_dir=Path(defaults.configs_dir).expanduser(),
                config_type="model",
            )
            loaded_config, _ = config_store.load_config(config_name_to_load)
            final_config = loaded_config
            final_config_name = config_name_to_load

            if not final_dataloader:
                dataloader_store = ConfigStore(
                    config_dir=Path(defaults.configs_dir).expanduser(),
                    config_type="dataloader",
                )
                try:
                    loaded_dataloader, _ = dataloader_store.load_config(config_name_to_load)
                    final_dataloader = loaded_dataloader.get("datasets", [])
                except Exception:
                    pass

        return cls(
            config=final_config or {},
            dataloader_config=final_dataloader or [],
            provider=provider,
            config_name=final_config_name,
            webhook_url=webhook_url,
            tracker_run_name=tracker_run_name,
            snapshot_name=snapshot_name,
            snapshot_message=snapshot_message,
            upload_id=upload_id,
            idempotency_key=idempotency_key,
            user=user,
        )

    async def validate(self, ctx: CommandContext) -> Optional[str]:
        """Validate submission parameters including permissions, quotas, and cost limits."""
        if not self.config:
            return "No configuration provided"

        if not ctx.has_permission("job.submit"):
            return "Permission denied: job.submit"

        # Resource-based access check
        if self.user and self.config_name:
            from ..auth.models import ResourceType

            allowed, reason = self.user.can_access_resource(
                ResourceType.CONFIG, self.config_name, permission_context="job.submit"
            )
            if not allowed:
                return f"Access denied for config '{self.config_name}': {reason}"

        # Cost limit check
        if ctx.job_store:
            cost_error = await self._check_cost_limit(ctx)
            if cost_error:
                return cost_error

        return None

    async def _check_cost_limit(self, ctx: CommandContext) -> Optional[str]:
        """Check cost limits. Returns error message if blocked, None otherwise."""
        try:
            config = await ctx.job_store.get_provider_config(self.provider)

            if not config.get("cost_limit_enabled", False):
                return None

            limit_amount = config.get("cost_limit_amount")
            if not limit_amount or limit_amount <= 0:
                return None

            period = config.get("cost_limit_period", "monthly")
            action = config.get("cost_limit_action", "warn")

            # Get period in days
            period_days = {"daily": 1, "weekly": 7, "monthly": 30, "yearly": 365}.get(period, 30)

            summary = await ctx.job_store.get_metrics_summary(days=period_days)
            current_spend = summary.get("total_cost_usd", 0.0)

            is_exceeded = current_spend >= limit_amount
            percent_used = (current_spend / limit_amount) * 100 if limit_amount > 0 else 0
            is_warning = percent_used >= 80 and not is_exceeded

            if is_exceeded and action == "block":
                return f"Cost limit exceeded: ${current_spend:.2f} spent " f"of ${limit_amount:.2f} {period} limit."
            elif is_warning or (is_exceeded and action == "warn"):
                self._cost_limit_warning = (
                    f"Approaching limit: ${current_spend:.2f} / ${limit_amount:.2f} ({period})"
                    if is_warning
                    else f"Spending limit exceeded: ${current_spend:.2f} / ${limit_amount:.2f} ({period})"
                )
        except Exception as exc:
            logger.debug("Cost limit check failed (continuing): %s", exc)

        return None

    async def _check_idempotency(self, ctx: CommandContext) -> Optional[SubmitJobData]:
        """Check idempotency key. Returns existing job data if hit, None otherwise."""
        if not self.idempotency_key:
            return None

        try:
            from ..async_job_store import AsyncJobStore

            async_store = await AsyncJobStore.get_instance()
            existing_job_id = await async_store.check_idempotency_key(
                self.idempotency_key,
                user_id=ctx.user_id,
            )
            if existing_job_id:
                job = await ctx.job_store.get_job(existing_job_id)
                if job:
                    logger.info("Idempotency key hit - returning existing job %s", existing_job_id)
                    return SubmitJobData(
                        job_id=existing_job_id,
                        status=job.status,
                        provider=self.provider,
                        config_name=job.config_name,
                        idempotent_hit=True,
                    )
        except Exception as exc:
            logger.warning("Idempotency check failed (continuing): %s", exc)

        return None

    async def _reserve_quota(self, ctx: CommandContext) -> Optional[str]:
        """Reserve quota slot. Returns error message if quota exceeded, None otherwise."""
        if not self.user or not ctx.user_store:
            return None

        try:
            from ..async_job_store import AsyncJobStore
            from ..auth import QuotaChecker

            async_store = await AsyncJobStore.get_instance()
            quotas = await ctx.user_store.get_quotas(self.user.id)
            max_concurrent = quotas.get("max_concurrent_jobs", 5)

            self._reservation_id = await async_store.reserve_job_slot(
                user_id=self.user.id,
                max_concurrent=max_concurrent,
                ttl_seconds=300,
            )

            if self._reservation_id is None:
                return f"Quota exceeded: Maximum {max_concurrent} concurrent jobs allowed"

            quota_checker = QuotaChecker(ctx.job_store, ctx.user_store)
            _, quota_statuses = await quota_checker.can_submit_job(self.user.id)
            for qs in quota_statuses:
                if qs.message and "concurrent" not in qs.message.lower():
                    if qs.is_exceeded and qs.action.value == "block":
                        await self._release_reservation()
                        return f"Quota exceeded: {qs.message}"
                    self._quota_warnings.append(qs.message)

        except Exception as exc:
            logger.error("Quota reservation failed: %s", exc)
            return "Unable to verify quota. Please try again later."

        return None

    async def _release_reservation(self) -> None:
        """Release quota reservation if held."""
        if self._reservation_id:
            try:
                from ..async_job_store import AsyncJobStore

                async_store = await AsyncJobStore.get_instance()
                await async_store.release_reservation(self._reservation_id)
            except Exception as exc:
                logger.warning("Failed to release reservation: %s", exc)
            finally:
                self._reservation_id = None

    async def _consume_reservation(self) -> None:
        """Mark reservation as consumed (job submitted successfully)."""
        if self._reservation_id:
            try:
                from ..async_job_store import AsyncJobStore

                async_store = await AsyncJobStore.get_instance()
                await async_store.consume_reservation(self._reservation_id)
                self._reservation_id = None
            except Exception as exc:
                logger.warning("Failed to consume reservation: %s", exc)

    async def _store_idempotency_key(self, job_id: str, ctx: CommandContext) -> None:
        """Store idempotency key for future lookups."""
        if self.idempotency_key and job_id:
            try:
                from ..async_job_store import AsyncJobStore

                async_store = await AsyncJobStore.get_instance()
                await async_store.store_idempotency_key(
                    key=self.idempotency_key,
                    job_id=job_id,
                    user_id=ctx.user_id,
                    ttl_hours=24,
                )
            except Exception as exc:
                logger.warning("Failed to store idempotency key: %s", exc)

    async def execute(self, ctx: CommandContext) -> CommandResult[SubmitJobData]:
        """Execute job submission."""
        from ..job_submission import JobSubmissionService, SubmissionContext

        if not ctx.job_store:
            return CommandResult(
                success=False,
                error="Job store not available",
                error_code="MISSING_DEPENDENCY",
            )

        # Check idempotency first
        existing = await self._check_idempotency(ctx)
        if existing:
            return CommandResult(success=True, data=existing)

        # Reserve quota
        quota_error = await self._reserve_quota(ctx)
        if quota_error:
            return CommandResult(
                success=False,
                error=quota_error,
                error_code="QUOTA_EXCEEDED",
            )

        try:
            # Build submission context
            submission_ctx = SubmissionContext(
                config=self.config,
                dataloader_config=self.dataloader_config or [],
                config_name=self.config_name,
                provider=self.provider,
                user_id=ctx.user_id,
                client_ip=ctx.client_ip,
                webhook_url=self.webhook_url,
                tracker_run_name=self.tracker_run_name,
                snapshot_name=self.snapshot_name,
                snapshot_message=self.snapshot_message,
                upload_id=self.upload_id,
                quota_warnings=self._quota_warnings,
            )

            # Execute via service
            service = JobSubmissionService(ctx.job_store)
            result = await service.submit(submission_ctx)

            if not result.success:
                return CommandResult(
                    success=False,
                    error=result.error,
                    error_code="SUBMISSION_FAILED",
                )

            self._created_job_id = result.job_id

            # Post-submit bookkeeping
            await self._store_idempotency_key(result.job_id, ctx)
            await self._consume_reservation()

            # Audit
            ctx.log_audit(
                action="job.submitted",
                job_id=result.job_id,
                provider=self.provider,
                config_name=self.config_name,
                details={
                    "data_uploaded": result.data_uploaded,
                    "tracker_run_name": self.tracker_run_name,
                },
            )

            return CommandResult(
                success=True,
                data=SubmitJobData(
                    job_id=result.job_id,
                    status=result.status,
                    provider=self.provider,
                    config_name=self.config_name,
                    data_uploaded=result.data_uploaded,
                    cost_limit_warning=self._cost_limit_warning,
                ),
                warnings=self._quota_warnings,
            )
        finally:
            # Always release reservation if not consumed
            await self._release_reservation()

    @property
    def is_reversible(self) -> bool:
        return True

    async def rollback(self, ctx: CommandContext) -> bool:
        """Cancel the submitted job if possible."""
        if not self._created_job_id or not ctx.job_store:
            return False

        try:
            from ..base import CloudJobStatus
            from ..factory import ProviderFactory

            job = await ctx.job_store.get_job(self._created_job_id)
            if job and job.provider:
                client = ProviderFactory.get_provider(job.provider)
                await client.cancel_job(self._created_job_id)

            from datetime import datetime, timezone

            await ctx.job_store.update_job(
                self._created_job_id,
                {
                    "status": CloudJobStatus.CANCELLED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except Exception as exc:
            logger.error("Rollback failed for job %s: %s", self._created_job_id, exc)
            return False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "provider": self.provider,
                "config_name": self.config_name,
                "has_config": bool(self.config),
                "dataloader_count": len(self.dataloader_config) if self.dataloader_config else 0,
            }
        )
        return base


@dataclass
class CancelJobData:
    """Data returned from job cancellation."""

    job_id: str
    previous_status: str
    new_status: str


class CancelJobCommand(Command[CancelJobData]):
    """Command to cancel a running job."""

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        self._previous_status: Optional[str] = None

    async def validate(self, ctx: CommandContext) -> Optional[str]:
        """Validate cancellation is allowed."""
        if not ctx.job_store:
            return "Job store not available"

        job = await ctx.job_store.get_job(self.job_id)
        if not job:
            return f"Job not found: {self.job_id}"

        # Permission check
        is_own_job = job.user_id == ctx.user_id
        can_cancel_all = ctx.has_permission("job.cancel.all")
        can_cancel_own = ctx.has_permission("job.cancel.own")

        if is_own_job and not can_cancel_own and not can_cancel_all:
            return "Permission denied: job.cancel.own"
        if not is_own_job and not can_cancel_all:
            return "Permission denied: job.cancel.all"

        # State check
        from ..base import CloudJobStatus

        cancellable = [
            CloudJobStatus.PENDING.value,
            CloudJobStatus.UPLOADING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        ]
        if job.status not in cancellable:
            return f"Cannot cancel job in state '{job.status}'"

        return None

    async def execute(self, ctx: CommandContext) -> CommandResult[CancelJobData]:
        """Execute job cancellation."""
        from ..base import CloudJobStatus, JobType
        from ..factory import ProviderFactory

        job = await ctx.job_store.get_job(self.job_id)
        self._previous_status = job.status

        # Cancel with provider
        if job.job_type == JobType.CLOUD and job.provider:
            try:
                client = ProviderFactory.get_provider(job.provider)
                success = await client.cancel_job(self.job_id)
                if not success:
                    return CommandResult(
                        success=False,
                        error=f"Provider failed to cancel job {self.job_id}",
                        error_code="PROVIDER_ERROR",
                    )
            except ValueError:
                return CommandResult(
                    success=False,
                    error=f"Unknown provider: {job.provider}",
                    error_code="INVALID_PROVIDER",
                )
            except Exception as exc:
                logger.error("Error cancelling cloud job %s: %s", self.job_id, exc)
                return CommandResult(
                    success=False,
                    error=f"Error cancelling job: {exc}",
                    error_code="PROVIDER_ERROR",
                )

        elif job.job_type == JobType.LOCAL:
            try:
                from simpletuner.simpletuner_sdk import process_keeper

                process_keeper.terminate_process(self.job_id)
            except Exception as exc:
                logger.warning("Could not terminate local process %s: %s", self.job_id, exc)

        # Update status and set completion time
        from datetime import datetime, timezone

        await ctx.job_store.update_job(
            self.job_id,
            {
                "status": CloudJobStatus.CANCELLED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        ctx.log_audit(
            action="job.cancelled",
            job_id=self.job_id,
            provider=job.provider,
            config_name=job.config_name,
        )

        return CommandResult(
            success=True,
            data=CancelJobData(
                job_id=self.job_id,
                previous_status=self._previous_status,
                new_status=CloudJobStatus.CANCELLED.value,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["job_id"] = self.job_id
        return base


@dataclass
class DeleteJobData:
    """Data returned from job deletion."""

    job_id: str
    deleted: bool


class DeleteJobCommand(Command[DeleteJobData]):
    """Command to delete a job from history."""

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        self._deleted_job: Optional[Dict[str, Any]] = None

    async def validate(self, ctx: CommandContext) -> Optional[str]:
        """Validate deletion is allowed."""
        if not ctx.job_store:
            return "Job store not available"

        job = await ctx.job_store.get_job(self.job_id)
        if not job:
            return f"Job not found: {self.job_id}"

        # Permission check
        is_own_job = job.user_id == ctx.user_id
        can_delete_all = ctx.has_permission("job.delete.all")
        can_delete_own = ctx.has_permission("job.delete.own")

        if is_own_job and not can_delete_own and not can_delete_all:
            return "Permission denied: job.delete.own"
        if not is_own_job and not can_delete_all:
            return "Permission denied: job.delete.all"

        # State check
        from ..base import CloudJobStatus

        terminal = {
            CloudJobStatus.COMPLETED.value,
            CloudJobStatus.FAILED.value,
            CloudJobStatus.CANCELLED.value,
        }
        if job.status not in terminal:
            return f"Cannot delete job in state '{job.status}'. Cancel it first."

        return None

    async def execute(self, ctx: CommandContext) -> CommandResult[DeleteJobData]:
        """Execute job deletion."""
        job = await ctx.job_store.get_job(self.job_id)
        self._deleted_job = job.to_dict() if job else None

        success = await ctx.job_store.delete_job(self.job_id)

        if not success:
            return CommandResult(
                success=False,
                error="Failed to delete job",
                error_code="DELETE_FAILED",
            )

        ctx.log_audit(
            action="job.deleted",
            job_id=self.job_id,
            provider=job.provider if job else None,
            config_name=job.config_name if job else None,
        )

        return CommandResult(
            success=True,
            data=DeleteJobData(job_id=self.job_id, deleted=True),
        )

    @property
    def is_reversible(self) -> bool:
        return True

    async def rollback(self, ctx: CommandContext) -> bool:
        """Restore the deleted job."""
        if not self._deleted_job or not ctx.job_store:
            return False

        try:
            from ..base import UnifiedJob

            job = UnifiedJob.from_dict(self._deleted_job)
            await ctx.job_store.add_job(job)
            return True
        except Exception as exc:
            logger.error("Rollback failed for deleted job: %s", exc)
            return False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["job_id"] = self.job_id
        return base


@dataclass
class SyncJobsData:
    """Data returned from job sync."""

    provider: str
    new_jobs: int
    updated_jobs: int


class SyncJobsCommand(Command[SyncJobsData]):
    """Command to sync jobs from cloud providers."""

    def __init__(self, provider: str = "replicate", sync_active: bool = True):
        super().__init__()
        self.provider = provider
        self.sync_active = sync_active

    async def execute(self, ctx: CommandContext) -> CommandResult[SyncJobsData]:
        """Execute job sync."""
        from ..job_sync import sync_active_job_statuses, sync_replicate_jobs

        if not ctx.job_store:
            return CommandResult(
                success=False,
                error="Job store not available",
                error_code="MISSING_DEPENDENCY",
            )

        new_count = 0
        updated_count = 0

        if self.provider == "replicate":
            new_count, updated_count = await sync_replicate_jobs(ctx.job_store)

        if self.sync_active:
            active_updated = await sync_active_job_statuses(ctx.job_store)
            updated_count += active_updated

        return CommandResult(
            success=True,
            data=SyncJobsData(
                provider=self.provider,
                new_jobs=new_count,
                updated_jobs=updated_count,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "provider": self.provider,
                "sync_active": self.sync_active,
            }
        )
        return base
