"""Job submission service for cloud training.

Handles the business logic of submitting jobs to cloud providers.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import UnifiedJob
from .factory import ProviderFactory

if TYPE_CHECKING:
    from .async_job_store import AsyncJobStore

logger = logging.getLogger(__name__)


@dataclass
class SubmissionContext:
    """Context for a job submission request."""

    config: Dict[str, Any]
    dataloader_config: List[Dict[str, Any]]
    config_name: Optional[str] = None
    provider: str = "replicate"

    # User context
    user_id: Optional[int] = None
    client_ip: Optional[str] = None

    # Options
    webhook_url: Optional[str] = None
    tracker_run_name: Optional[str] = None
    snapshot_name: Optional[str] = None
    snapshot_message: Optional[str] = None
    upload_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    # Quota tracking
    reservation_id: Optional[str] = None
    quota_warnings: List[str] = field(default_factory=list)


@dataclass
class SubmissionResult:
    """Result of a job submission."""

    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    data_uploaded: bool = False
    cost_limit_warning: Optional[str] = None
    quota_warnings: List[str] = field(default_factory=list)
    idempotent_hit: bool = False


@dataclass
class _UploadResult:
    """Internal result of data upload handling."""

    uploaded: bool = False
    data_url: Optional[str] = None
    error: Optional[str] = None


class JobSubmissionService:
    """Service for submitting training jobs to cloud providers."""

    def __init__(self, store: "AsyncJobStore"):
        self.store = store

    async def submit(
        self,
        ctx: SubmissionContext,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> SubmissionResult:
        """Submit a training job to a cloud provider.

        Args:
            ctx: Submission context with all required parameters
            progress_callback: Optional callback for upload progress

        Returns:
            SubmissionResult with success status and job details
        """
        from .circuit_breaker import get_circuit_breaker

        # Get provider client
        try:
            client = ProviderFactory.get_provider(ctx.provider)
        except ValueError:
            return SubmissionResult(success=False, error=f"Unknown provider: {ctx.provider}")

        # Check circuit breaker
        circuit = None
        try:
            circuit = await get_circuit_breaker(ctx.provider)
            if not await circuit.can_execute():
                return SubmissionResult(
                    success=False,
                    error=f"Provider '{ctx.provider}' is temporarily unavailable. Please try again later.",
                )
        except Exception as exc:
            logger.debug("Circuit breaker check failed (continuing): %s", exc)

        # Build snapshot metadata
        snapshot_metadata = await self._build_snapshot_metadata(ctx)

        # Handle data upload if needed
        upload_id = ctx.upload_id or str(uuid.uuid4())
        upload_result = await self._handle_data_upload(ctx, upload_id, progress_callback)
        if upload_result.error:
            return SubmissionResult(success=False, error=upload_result.error)
        data_url = upload_result.data_url
        data_uploaded = upload_result.uploaded

        # Prepare config for submission
        config = dict(ctx.config)
        local_output_url, upload_token = self._setup_publishing_config(config, ctx.webhook_url)

        # Get HF token if pushing to hub
        hf_token = self._get_hf_token_if_needed(config)
        hub_model_id = config.get("hub_model_id") or config.get("--hub_model_id")

        # Load lycoris config if specified
        lycoris_config_data = self._load_lycoris_config(config)

        # Apply provider-specific settings
        provider_config = await self.store.get_provider_config(ctx.provider)
        if ctx.provider == "replicate" and hasattr(client, "_version"):
            version_override = provider_config.get("version_override")
            if version_override:
                client._version = version_override

        # Submit to provider
        try:
            cloud_job = await client.run_job(
                config=config,
                dataloader=ctx.dataloader_config,
                data_archive_url=data_url,
                webhook_url=ctx.webhook_url,
                hf_token=hf_token,
                hub_model_id=hub_model_id,
                lycoris_config=lycoris_config_data,
            )
            if circuit:
                await circuit.record_success()
        except Exception as provider_exc:
            if circuit:
                await circuit.record_failure(provider_exc)
            logger.error("Failed to submit job to %s: %s", ctx.provider, provider_exc, exc_info=True)
            return SubmissionResult(success=False, error=str(provider_exc))

        # Build unified job and store it
        unified_job = UnifiedJob.from_cloud_job(cloud_job)
        unified_job.config_name = ctx.config_name

        if ctx.user_id:
            unified_job.user_id = ctx.user_id
        if upload_token:
            unified_job.upload_token = upload_token
        if local_output_url:
            unified_job.output_url = local_output_url
        elif hub_model_id:
            unified_job.output_url = f"https://huggingface.co/{hub_model_id}"
        if snapshot_metadata:
            unified_job.metadata["snapshot"] = snapshot_metadata
        if ctx.tracker_run_name:
            unified_job.metadata["tracker_run_name"] = ctx.tracker_run_name

        await self.store.add_job(unified_job)

        # Enqueue the job for processing
        await self._enqueue_job(
            job_id=cloud_job.job_id,
            user_id=ctx.user_id,
            provider=ctx.provider,
            config_name=ctx.config_name,
        )

        logger.info("Submitted job to %s: %s", ctx.provider, cloud_job.job_id)

        return SubmissionResult(
            success=True,
            job_id=cloud_job.job_id,
            status=cloud_job.status.value,
            data_uploaded=data_uploaded,
            quota_warnings=ctx.quota_warnings,
        )

    async def _handle_data_upload(
        self,
        ctx: SubmissionContext,
        upload_id: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> _UploadResult:
        """Handle local data packaging and upload if needed.

        Args:
            ctx: Submission context
            upload_id: Unique ID for this upload
            progress_callback: Optional callback for progress updates

        Returns:
            _UploadResult with upload status, URL, or error
        """
        from .cloud_upload_service import CloudUploadService

        upload_service = CloudUploadService()

        if not upload_service.has_local_data(ctx.dataloader_config):
            return _UploadResult(uploaded=False, data_url=None)

        logger.info("Packaging and uploading local data...")

        def detailed_progress_callback(stage: str, current: int, total: int, message: str) -> None:
            self.store.update_upload_progress(
                upload_id=upload_id, stage=stage, current=current, total=total, message=message
            )
            if progress_callback:
                progress_callback(stage, current, total, message)

        try:
            self.store.update_upload_progress(
                upload_id=upload_id, stage="scanning", current=0, total=1, message="Scanning files..."
            )
            data_url = await upload_service.package_and_upload(
                ctx.dataloader_config, detailed_progress_callback=detailed_progress_callback
            )
            logger.info("Data uploaded successfully: %s", data_url)
            self.store.update_upload_progress(
                upload_id=upload_id, stage="complete", current=100, total=100, message="Upload complete", done=True
            )
            return _UploadResult(uploaded=True, data_url=data_url)

        except Exception as upload_exc:
            logger.error("Failed to upload data: %s", upload_exc)
            self.store.update_upload_progress(
                upload_id=upload_id, stage="error", current=0, total=100, error=str(upload_exc)
            )
            return _UploadResult(uploaded=False, error=f"Failed to upload data: {upload_exc}")

    async def _enqueue_job(
        self,
        job_id: str,
        user_id: Optional[int],
        provider: str,
        config_name: Optional[str],
    ) -> None:
        """Add a submitted job to the queue for tracking and scheduling.

        The queue tracks job status, handles concurrency limits, and enables
        scheduling features even in single-user mode (e.g., overnight runs).
        """
        try:
            from .background_tasks import get_queue_scheduler

            scheduler = get_queue_scheduler()
            if scheduler is None:
                # Queue not started yet (e.g., during early startup)
                logger.debug("Queue scheduler not available, skipping enqueue for %s", job_id)
                return

            # Get user levels for priority determination
            user_levels = None
            if user_id:
                try:
                    from .auth.user_store import UserStore

                    user_store = UserStore()
                    user = await user_store.get_user(user_id)
                    if user:
                        user_levels = [level.name for level in user.levels]
                except Exception:
                    pass

            await scheduler.enqueue_job(
                job_id=job_id,
                user_id=user_id,
                provider=provider,
                config_name=config_name,
                user_levels=user_levels,
            )
            logger.debug("Enqueued job %s", job_id)

        except Exception as exc:
            # Don't fail the submission if enqueue fails
            logger.warning("Failed to enqueue job %s: %s", job_id, exc)

    async def _build_snapshot_metadata(self, ctx: SubmissionContext) -> Dict[str, Any]:
        """Build git snapshot metadata for the job."""
        from ..git_config_service import GIT_CONFIG_SERVICE, GitConfigError

        snapshot_metadata: Dict[str, Any] = {}
        try:
            head_info = GIT_CONFIG_SERVICE.get_head_info(config_type="model")
            if head_info.repo_present:
                if head_info.is_dirty:
                    snapshot_name = ctx.snapshot_name or ctx.tracker_run_name
                    auto_generated = not snapshot_name
                    if auto_generated:
                        snapshot_name = f"env-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                    commit_message = ctx.snapshot_message or f"pre-submit: {snapshot_name}"

                    try:
                        commit_result = GIT_CONFIG_SERVICE.snapshot_directory(
                            message=commit_message,
                            config_type="model",
                            include_untracked=True,
                        )
                        snapshot_metadata = {
                            "commit": commit_result.get("commit"),
                            "abbrev": commit_result.get("abbrev"),
                            "name": snapshot_name,
                            "auto_generated": auto_generated,
                            "message": commit_message,
                            "branch": head_info.branch,
                            "was_dirty": commit_result.get("was_dirty", True),
                        }
                        logger.info("Created pre-submit snapshot: %s", commit_result.get("abbrev"))
                    except GitConfigError as exc:
                        logger.warning("Failed to create snapshot: %s", exc.message)
                        snapshot_metadata = {"error": exc.message, "branch": head_info.branch}
                else:
                    snapshot_metadata = {
                        "commit": head_info.commit,
                        "abbrev": head_info.abbrev,
                        "name": ctx.snapshot_name or ctx.tracker_run_name,
                        "branch": head_info.branch,
                        "was_dirty": False,
                    }
        except Exception as exc:
            logger.warning("Git snapshot check failed: %s", exc)

        return snapshot_metadata

    def _setup_publishing_config(
        self, config: Dict[str, Any], webhook_url: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Setup publishing config based on webhook URL.

        Returns:
            Tuple of (local_output_url, upload_token)
        """
        from pathlib import Path

        from ..webui_state import WebUIStateStore

        hub_model_id = config.get("hub_model_id") or config.get("--hub_model_id")
        push_to_hub = config.get("push_to_hub") or config.get("--push_to_hub", False)
        existing_publishing_config = config.get("publishing_config") or config.get("--publishing_config")
        has_publishing = push_to_hub or existing_publishing_config

        local_output_url = None
        upload_token = None

        if webhook_url and not has_publishing:
            import secrets as secrets_mod
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(webhook_url)
            s3_endpoint = urlunparse((parsed.scheme, parsed.netloc, "/api/cloud/storage", "", "", ""))
            upload_token = secrets_mod.token_urlsafe(32)

            local_publishing_config = [
                {
                    "provider": "s3",
                    "bucket": "outputs",
                    "endpoint_url": s3_endpoint,
                    "access_key": "local",
                    "secret_key": upload_token,
                    "use_ssl": parsed.scheme == "https",
                }
            ]

            config["publishing_config"] = local_publishing_config
            logger.info("Auto-injected local publishing config with per-job token: %s", s3_endpoint)

            try:
                defaults = WebUIStateStore().load_defaults()
                if defaults.cloud_upload_dir:
                    upload_dir = Path(defaults.cloud_upload_dir).expanduser()
                    local_output_url = str(upload_dir / "outputs")
            except Exception:
                pass

        return local_output_url, upload_token

    def _get_hf_token_if_needed(self, config: Dict[str, Any]) -> Optional[str]:
        """Get HF token if pushing to hub."""
        import os

        push_to_hub = config.get("push_to_hub") or config.get("--push_to_hub", False)
        if not push_to_hub:
            return None

        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _load_lycoris_config(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load lycoris config if specified."""
        from ..webui_state import WebUIStateStore

        lycoris_config_path = config.get("lycoris_config") or config.get("--lycoris_config")
        if not lycoris_config_path:
            return None

        try:
            lycoris_defaults = WebUIStateStore().load_defaults()
            lycoris_path = Path(lycoris_config_path).expanduser()
            if not lycoris_path.is_absolute() and lycoris_defaults.configs_dir:
                lycoris_path = Path(lycoris_defaults.configs_dir).expanduser() / lycoris_config_path
            if lycoris_path.exists():
                with open(lycoris_path, "r") as f:
                    data = json.load(f)
                logger.info("Loaded lycoris config from: %s", lycoris_path)
                return data
            else:
                logger.warning("Lycoris config file not found: %s", lycoris_config_path)
        except Exception as exc:
            logger.warning("Failed to load lycoris config: %s", exc)

        return None
