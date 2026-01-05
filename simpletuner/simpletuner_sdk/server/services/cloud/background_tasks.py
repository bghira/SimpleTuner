"""Background tasks for cloud job management.

Provides automatic job status polling without requiring webhooks,
suitable for deployments behind firewalls or without public endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .queue import QueueScheduler

logger = logging.getLogger(__name__)

# Configurable intervals (seconds) via environment variables
APPROVAL_CHECK_INTERVAL = int(os.environ.get("SIMPLETUNER_APPROVAL_CHECK_INTERVAL", "300"))
SESSION_CLEANUP_INTERVAL = int(os.environ.get("SIMPLETUNER_SESSION_CLEANUP_INTERVAL", "900"))

# Singleton task manager
_task_manager: Optional["BackgroundTaskManager"] = None


class BackgroundTaskManager:
    """Manages background tasks for cloud operations.

    Tasks:
    - Job status polling: Periodically syncs active job statuses from providers
    - Queue processing: Dispatches queued jobs based on priority and concurrency
    - Approval expiration: Auto-expires and rejects pending approvals past deadline
    """

    def __init__(self):
        self._job_polling_task: Optional[asyncio.Task] = None
        self._queue_task: Optional[asyncio.Task] = None
        self._approval_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._running = False
        self._scheduler: Optional["QueueScheduler"] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start all configured background tasks."""
        current_loop = asyncio.get_running_loop()

        # Check if we're in a different event loop than before (common in tests)
        if self._running and self._event_loop and self._event_loop is not current_loop:
            logger.warning("Event loop changed while running, resetting state")
            self._reset_state()

        if self._running:
            logger.warning("Background task manager already running")
            return

        self._running = True
        self._event_loop = current_loop
        self._stop_event = asyncio.Event()  # Create new Event on current loop

        # Start job polling if configured
        await self._start_polling_if_configured()

        # Always start approval expiration task
        self._approval_task = asyncio.create_task(self._approval_expiration_loop(APPROVAL_CHECK_INTERVAL))
        logger.info("Approval expiration task started (interval=%ds)", APPROVAL_CHECK_INTERVAL)

        # Start session and OAuth state cleanup task
        self._cleanup_task = asyncio.create_task(self._session_cleanup_loop(SESSION_CLEANUP_INTERVAL))
        logger.info("Session/OAuth cleanup task started (interval=%ds)", SESSION_CLEANUP_INTERVAL)

        # Always start queue processing (handles job scheduling and dispatch)
        await self._start_queue_processing()
        logger.info("Queue processing started")

    @property
    def is_polling_active(self) -> bool:
        """Check if job status polling is currently active."""
        return self._job_polling_task is not None and not self._job_polling_task.done()

    async def refresh_polling_state(self) -> bool:
        """Refresh polling state based on current configuration.

        Returns:
            True if polling is now active, False otherwise.
        """
        # Stop existing task if running
        if self._job_polling_task:
            self._job_polling_task.cancel()
            try:
                await self._job_polling_task
            except asyncio.CancelledError:
                pass
            self._job_polling_task = None
            logger.info("Stopped existing job polling task for refresh")

        if not self._running:
            return False

        # Start again based on config
        await self._start_polling_if_configured()
        return self.is_polling_active

    async def _start_polling_if_configured(self) -> None:
        """Start polling if configured in enterprise config or WebUI settings."""
        if self._job_polling_task:
            return

        # 1. Check enterprise config (highest priority)
        try:
            from ...config.enterprise import get_enterprise_config

            config = get_enterprise_config()

            if config.job_polling_enabled:
                interval = config.job_polling_interval
                logger.info("Starting job status polling (Enterprise Config, interval: %ds)", interval)
                self._job_polling_task = asyncio.create_task(self._job_polling_loop(interval))
                return
        except ImportError:
            logger.debug("Enterprise config not available, using defaults")
        except Exception as e:
            logger.warning("Error loading enterprise config: %s", e)

        # 2. Check WebUI settings (user preference)
        try:
            from ...services.webui_state import WebUIStateStore

            defaults = WebUIStateStore().load_defaults()
            polling_enabled = defaults.cloud_job_polling_enabled

            if polling_enabled is True:
                logger.info("Starting job status polling (WebUI Setting, interval: 30s)")
                self._job_polling_task = asyncio.create_task(self._job_polling_loop(30))
                return
            elif polling_enabled is False:
                logger.info("Job status polling is explicitly disabled in WebUI settings")
                return
        except Exception as e:
            logger.warning("Error loading WebUI settings for polling: %s", e)

        # 3. Default: start polling if webhooks are not configured (auto-detect)
        if await self._should_auto_enable_polling():
            logger.info("Auto-enabling job polling (no webhook configured)")
            self._job_polling_task = asyncio.create_task(self._job_polling_loop(30))  # Default 30 second interval

    async def _should_auto_enable_polling(self) -> bool:
        """Check if polling should be auto-enabled."""
        try:
            from .provider_config import get_provider_config_store

            store = get_provider_config_store()
            config = await store.get_config("replicate")
            # If no webhook configured, enable polling
            return not config.get("webhook_url")
        except Exception:
            return True  # Default to polling if we can't check

    async def _start_queue_processing(self) -> None:
        """Start the queue scheduler for processing queued jobs."""
        try:
            from .async_job_store import AsyncJobStore
            from .queue import QueueScheduler, get_dispatcher

            # Get or create scheduler
            job_store = await AsyncJobStore.get_instance()
            dispatcher = get_dispatcher(job_store)

            self._scheduler = QueueScheduler()
            self._scheduler.set_dispatch_callback(dispatcher.dispatch)

            # Start background processing (checks queue every 5 seconds)
            await self._scheduler.start_background_processing(interval_seconds=5.0)
            logger.info("Queue scheduler started with 5s processing interval")

        except Exception as exc:
            logger.error("Failed to start queue processing: %s", exc, exc_info=True)
            self._scheduler = None

    async def _stop_queue_processing(self) -> None:
        """Stop the queue scheduler."""
        if self._scheduler:
            try:
                await self._scheduler.stop_background_processing()
                logger.info("Queue scheduler stopped")
            except Exception as exc:
                logger.warning("Error stopping queue scheduler: %s", exc)
            self._scheduler = None

    def get_scheduler(self) -> Optional["QueueScheduler"]:
        """Get the queue scheduler instance."""
        return self._scheduler

    async def stop(self) -> None:
        """Stop all background tasks."""
        if not self._running:
            return

        logger.info("Stopping background tasks...")

        # Check for event loop mismatch (common in test scenarios)
        current_loop = asyncio.get_running_loop()
        if self._event_loop and self._event_loop is not current_loop:
            logger.warning("Event loop mismatch during stop, resetting state")
            self._reset_state()
            return

        self._stop_event.set()

        await self._cancel_task(self._job_polling_task)
        self._job_polling_task = None

        await self._cancel_task(self._approval_task)
        self._approval_task = None

        await self._cancel_task(self._cleanup_task)
        self._cleanup_task = None

        # Stop queue processing
        try:
            await self._stop_queue_processing()
        except RuntimeError:
            pass  # Event loop mismatch

        self._running = False
        self._event_loop = None
        logger.info("Background tasks stopped")

    async def _cancel_task(self, task: Optional[asyncio.Task]) -> None:
        """Cancel a task gracefully, handling event loop mismatches."""
        if not task:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, RuntimeError):
            pass

    def _reset_state(self) -> None:
        """Reset all task state (used on event loop mismatch)."""
        self._job_polling_task = None
        self._approval_task = None
        self._cleanup_task = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._event_loop = None

    async def _job_polling_loop(self, interval_seconds: int) -> None:
        """Background loop for polling job statuses."""
        from ..cloud import CloudJobStatus, JobType
        from .container import get_job_store

        logger.info("Job polling loop started (interval: %ds)", interval_seconds)

        while not self._stop_event.is_set():
            try:
                await self._sync_active_jobs()
            except Exception as exc:
                logger.error("Error in job polling: %s", exc, exc_info=True)

            # Wait for interval or stop event
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval_seconds,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

        logger.info("Job polling loop stopped")

    async def _sync_active_jobs(self) -> int:
        """Sync status of active cloud jobs. Returns count synced."""
        from ..cloud import CloudJobStatus, JobType
        from .async_job_store import AsyncJobStore
        from .factory import ProviderFactory
        from .job_sync import _get_external_job_id

        store = await AsyncJobStore.get_instance()

        active_statuses = {
            CloudJobStatus.PENDING.value,
            CloudJobStatus.UPLOADING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        }

        all_jobs = await store.list_jobs(limit=100)
        active_cloud_jobs = [j for j in all_jobs if j.job_type == JobType.CLOUD and j.status in active_statuses]

        if not active_cloud_jobs:
            return 0

        synced = 0
        jobs_by_provider = {}
        for job in active_cloud_jobs:
            if job.provider:
                jobs_by_provider.setdefault(job.provider, []).append(job)

        for provider_name, jobs in jobs_by_provider.items():
            try:
                client = ProviderFactory.get_provider(provider_name)
                for job in jobs:
                    # Skip jobs without valid external IDs (e.g., test jobs)
                    external_id = _get_external_job_id(job)
                    if not external_id:
                        logger.debug("Skipping sync for job %s - no valid external ID", job.job_id)
                        continue

                    try:
                        cloud_status = await client.get_job_status(external_id)
                        new_status = cloud_status.status.value
                        old_status = job.status

                        await store.update_job(
                            job.job_id,
                            {
                                "status": new_status,
                                "started_at": cloud_status.started_at,
                                "completed_at": cloud_status.completed_at,
                                "cost_usd": cloud_status.cost_usd,
                                "error_message": cloud_status.error_message,
                            },
                        )
                        synced += 1

                        # Update queue on terminal status changes
                        if new_status != old_status:
                            await self._update_queue_on_status_change(
                                job.job_id,
                                new_status,
                                cloud_status.error_message,
                                cloud_status.cost_usd,
                            )

                        # Emit SSE event if status changed
                        if cloud_status.status.value != job.status:
                            await self._emit_status_change(job.job_id, cloud_status.status.value)

                    except Exception as exc:
                        logger.debug("Failed to sync job %s: %s", job.job_id, exc)
            except ValueError:
                logger.warning("Unknown provider: %s", provider_name)
            except Exception as exc:
                logger.error("Failed to sync jobs for provider %s: %s", provider_name, exc)

        if synced > 0:
            logger.debug("Synced %d active jobs", synced)

        return synced

    async def _emit_status_change(self, job_id: str, new_status: str) -> None:
        """Emit SSE event for job status change."""
        try:
            from ..sse_manager import get_sse_manager

            manager = get_sse_manager()
            await manager.broadcast(
                {
                    "type": "job_status_changed",
                    "job_id": job_id,
                    "status": new_status,
                }
            )
        except Exception:
            pass  # SSE is optional

    async def _update_queue_on_status_change(
        self,
        job_id: str,
        new_status: str,
        error_message: Optional[str],
        cost_usd: Optional[float],
    ) -> None:
        """Update queue entry when job status changes.

        This keeps the queue in sync with job status from the provider.
        """
        from ..cloud import CloudJobStatus

        terminal_statuses = {
            CloudJobStatus.COMPLETED.value,
            CloudJobStatus.FAILED.value,
            CloudJobStatus.CANCELLED.value,
        }

        if new_status not in terminal_statuses:
            return

        try:
            from .queue import get_dispatcher

            dispatcher = get_dispatcher()

            if new_status == CloudJobStatus.COMPLETED.value:
                await dispatcher.on_job_completed(job_id, cost_usd)
            elif new_status in (CloudJobStatus.FAILED.value, CloudJobStatus.CANCELLED.value):
                await dispatcher.on_job_failed(job_id, error_message or "Job failed")

        except Exception as exc:
            logger.debug("Failed to update queue on status change: %s", exc)

    async def _approval_expiration_loop(self, interval_seconds: int) -> None:
        """Background loop for expiring pending approvals."""
        logger.info("Approval expiration loop started (interval: %ds)", interval_seconds)

        while not self._stop_event.is_set():
            try:
                await self._expire_and_reject_approvals()
            except Exception as exc:
                logger.error("Error in approval expiration: %s", exc, exc_info=True)

            # Wait for interval or stop event
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval_seconds,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

        logger.info("Approval expiration loop stopped")

    async def _expire_and_reject_approvals(self) -> int:
        """Expire old approvals and reject their associated jobs."""
        try:
            from .approval import ApprovalStore
            from .container import get_job_store
            from .queue import QueueStore

            approval_store = ApprovalStore()

            # Get pending approvals that will be expired
            pending = await approval_store.list_requests(pending_only=True)
            expired_job_ids = [r.job_id for r in pending if r.is_expired]

            if not expired_job_ids:
                return 0

            # Expire the approval requests
            expired_count = await approval_store.expire_old_requests()

            if expired_count > 0:
                # Also reject the jobs in the queue
                try:
                    from .queue import get_queue_adapter

                    queue_adapter = get_queue_adapter()
                    job_store = get_job_store()

                    for job_id in expired_job_ids:
                        # Update job status (via unified repository)
                        try:
                            await queue_adapter.mark_failed_by_job_id(
                                job_id, "Approval request expired (auto-rejected after timeout)"
                            )
                        except Exception:
                            pass

                        # Also update via job_store for backwards compat
                        try:
                            await job_store.update_job(
                                job_id, {"status": "cancelled", "error_message": "Approval request expired after 24 hours"}
                            )
                        except Exception:
                            pass

                    logger.info("Expired %d approval requests and rejected associated jobs", expired_count)

                    # Emit SSE notification
                    await self._emit_approval_expired(expired_job_ids)

                except Exception as exc:
                    logger.warning("Failed to reject expired approval jobs: %s", exc)

            return expired_count

        except ImportError:
            # Approval system not available
            return 0
        except Exception as exc:
            logger.debug("Approval expiration check failed: %s", exc)
            return 0

    async def _emit_approval_expired(self, job_ids: list) -> None:
        """Emit SSE events for expired approvals."""
        try:
            from ..sse_manager import get_sse_manager

            manager = get_sse_manager()
            for job_id in job_ids:
                await manager.broadcast(
                    {
                        "type": "approval_expired",
                        "job_id": job_id,
                        "message": "Approval request expired and job was auto-rejected",
                    }
                )
        except Exception:
            pass

    async def _session_cleanup_loop(self, interval_seconds: int) -> None:
        """Background loop for cleaning up expired sessions and OAuth states."""
        logger.info("Session/OAuth cleanup loop started (interval: %ds)", interval_seconds)

        while not self._stop_event.is_set():
            try:
                await self._cleanup_expired_auth_data()
            except Exception as exc:
                logger.error("Error in session cleanup: %s", exc, exc_info=True)

            # Wait for interval or stop event
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval_seconds,
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

        logger.info("Session/OAuth cleanup loop stopped")

    async def _cleanup_expired_auth_data(self) -> None:
        """Clean up expired sessions and OAuth states."""
        try:
            from .auth.user_store import UserStore

            user_store = UserStore()

            # Cleanup expired sessions
            sessions_deleted = await user_store.cleanup_expired_sessions()
            if sessions_deleted > 0:
                logger.debug("Cleaned up %d expired sessions", sessions_deleted)

            # Cleanup expired OAuth states
            oauth_deleted = await user_store.cleanup_expired_oauth_states()
            if oauth_deleted > 0:
                logger.debug("Cleaned up %d expired OAuth states", oauth_deleted)

        except ImportError:
            pass  # Auth system not available
        except Exception as exc:
            logger.debug("Auth cleanup failed: %s", exc)


def get_task_manager() -> BackgroundTaskManager:
    """Get the singleton background task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


async def start_background_tasks() -> None:
    """Start background tasks (call from app lifespan)."""
    manager = get_task_manager()
    await manager.start()


async def stop_background_tasks() -> None:
    """Stop background tasks (call from app lifespan)."""
    manager = get_task_manager()
    await manager.stop()


def get_queue_scheduler() -> Optional["QueueScheduler"]:
    """Get the queue scheduler from the background task manager.

    Returns:
        The QueueScheduler instance, or None if not running.
    """
    manager = get_task_manager()
    return manager.get_scheduler()
