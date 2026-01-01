"""Routes for SimpleTuner server.

This module provides both the top-level global routes and the cloud-specific routes.
Routes are organized as follows:

Global routes (under /api/):
- /api/auth - Authentication (login, logout, register, API keys)
- /api/auth/external - External auth providers (OIDC, LDAP)
- /api/users - User management
- /api/orgs - Organization and team management
- /api/audit - Audit logging
- /api/approvals - Approval workflows
- /api/queue - Job queue management
- /api/quotas - Quota management
- /api/metrics - Metrics and monitoring
- /api/webhooks - Webhook handling
- /api/backup - Backup and restore
- /api/database - Database health

Cloud-specific routes (under /api/cloud/):
- /api/cloud/jobs - Cloud job submission
- /api/cloud/providers - Provider configuration
- /api/cloud/storage - Cloud storage (S3-compatible)
- /api/cloud/settings - Cloud settings
- /api/cloud/htmx - UI components
"""

from fastapi import APIRouter

# Legacy exports for compatibility
from . import webui_state  # noqa: F401
from .approvals import router as approvals_router
from .audit import router as audit_router
from .auth import router as auth_router
from .backup import router as backup_router
from .database import router as database_router
from .external_auth import router as external_auth_router
from .metrics import router as metrics_router
from .orgs import router as orgs_router
from .queue import router as queue_router
from .quotas import router as quotas_router

# Import global route modules
from .users import router as users_router
from .webhooks import router as webhooks_router

# Create the main global router that combines all top-level routes
# These routes are global concepts not specific to cloud providers
global_router = APIRouter()

# Include all global routes (they already have /api/* prefixes)
global_router.include_router(auth_router)
global_router.include_router(external_auth_router)
global_router.include_router(users_router)
global_router.include_router(orgs_router)
global_router.include_router(audit_router)
global_router.include_router(approvals_router)
global_router.include_router(queue_router)
global_router.include_router(quotas_router)
global_router.include_router(metrics_router)
global_router.include_router(webhooks_router)
global_router.include_router(backup_router)
global_router.include_router(database_router)


def get_all_routers():
    """Get all routers for the application.

    Returns a list of tuples (router, prefix, tags) to be included in the FastAPI app.
    """
    from .cloud import router as cloud_router

    return [
        (global_router, "", []),  # Global routes already have /api/* prefixes
        (cloud_router, "", []),  # Cloud routes have /api/cloud prefix
    ]


__all__ = [
    "webui_state",
    "global_router",
    "get_all_routers",
    "auth_router",
    "external_auth_router",
    "users_router",
    "orgs_router",
    "audit_router",
    "approvals_router",
    "queue_router",
    "quotas_router",
    "metrics_router",
    "webhooks_router",
    "backup_router",
    "database_router",
]
