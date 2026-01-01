"""Cloud training API routes.

This package provides endpoints for cloud training job management,
provider configuration, and cloud-specific functionality.

NOTE: Many routes have been moved to the top-level routes package
as they are global concepts not specific to cloud providers:
- auth -> /api/auth
- external_auth -> /api/auth/external
- users -> /api/users
- orgs -> /api/orgs
- audit -> /api/audit
- approvals -> /api/approvals
- queue -> /api/queue
- quotas -> /api/quotas
- metrics -> /api/metrics
- webhooks -> /api/webhooks

Cloud-specific routes remaining here:
- jobs - Cloud job submission and management
- job_utils - Job utilities (configs, hardware, templates)
- providers - Cloud provider configuration
- storage - S3-compatible storage (renamed from s3)
- settings - Cloud-specific settings
- htmx - UI components for cloud features
- notifications - Notification channels (still here for now)
- metrics_config - Metrics configuration
"""

from fastapi import APIRouter

# Re-export commonly used items for backwards compatibility
from ._shared import CloudJobStatus, JobStore, JobType, UnifiedJob, get_job_store
from .htmx import router as htmx_router
from .job_utils import router as job_utils_router
from .jobs import router as jobs_router
from .metrics import router as cloud_metrics_router
from .metrics_config import router as metrics_config_router
from .notifications import router as notifications_router
from .providers import router as providers_router
from .settings import router as settings_router
from .storage import router as storage_router

# Create the main router that combines all cloud-specific sub-routers
router = APIRouter(prefix="/api/cloud", tags=["cloud"])

# Include cloud-specific sub-routers
router.include_router(jobs_router)
router.include_router(job_utils_router)
router.include_router(cloud_metrics_router)  # Cloud metrics (cost-limit, billing, etc.)
router.include_router(metrics_config_router, prefix="/metrics/config")
router.include_router(notifications_router)
router.include_router(providers_router)
router.include_router(settings_router)
router.include_router(htmx_router)
router.include_router(storage_router)

# Note: The following routers have been moved to top-level routes package:
# - auth_router -> routes/auth.py (/api/auth)
# - external_auth_router -> routes/external_auth.py (/api/auth/external)
# - users_router -> routes/users.py (/api/users)
# - orgs_router -> routes/orgs.py (/api/orgs)
# - audit_router -> routes/audit.py (/api/audit)
# - approval_router -> routes/approvals.py (/api/approvals)
# - queue_router -> routes/queue.py (/api/queue)
# - quotas_router -> routes/quotas.py (/api/quotas)
# - metrics_router -> routes/metrics.py (/api/metrics)
# - webhooks_router -> routes/webhooks.py (/api/webhooks)
# - s3_router -> routes/cloud/storage.py (renamed, kept in cloud)

__all__ = [
    "router",
    "CloudJobStatus",
    "JobStore",
    "JobType",
    "UnifiedJob",
    "get_job_store",
]
