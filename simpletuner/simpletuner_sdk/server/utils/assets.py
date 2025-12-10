"""Helpers for static asset versioning."""

from __future__ import annotations

import os
import time

# Single version string used to bust client caches for static assets.
_ASSET_VERSION = os.environ.get("SIMPLETUNER_ASSET_VERSION") or os.environ.get("ASSET_VERSION") or str(int(time.time()))


def get_asset_version() -> str:
    """Return the server-wide asset version token."""
    return _ASSET_VERSION
