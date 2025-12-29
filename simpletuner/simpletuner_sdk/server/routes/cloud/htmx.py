"""HTMX endpoints for cloud feature.

These endpoints return server-rendered HTML fragments for HTMX integration.

Architecture Note
-----------------
The cloud dashboard primarily uses Alpine.js + JSON APIs because:
- Job cards need reactive selection state, keyboard navigation, hover effects
- Stats/queue data updates atomically with job data (single fetch)
- Interactive elements (sliders, buttons) require Alpine.js state binding

HTMX is reserved for components that:
1. Don't need client-side interactivity
2. Benefit from intersection-based lazy loading
3. Require server-side processing with HTML responses

The following use JSON APIs (in settings.py) + Alpine.js state:
- System status: /api/cloud/system-status + systemStatus Alpine state
- Publishing status: /api/cloud/publishing-status + publishingStatus Alpine state
- Job cards, stats, queue panel - all Alpine.js by design

Currently this module is a placeholder for future HTMX endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/htmx", tags=["htmx"])

# No active HTMX endpoints currently.
# System status and publishing status use JSON APIs + Alpine.js state management.
# See settings.py for /system-status and /publishing-status JSON endpoints.
