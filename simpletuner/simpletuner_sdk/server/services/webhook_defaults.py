"""Shared webhook defaults for trainer services."""

from __future__ import annotations

import os


DEFAULT_CALLBACK_URL = os.environ.get("SIMPLETUNER_WEBHOOK_CALLBACK_URL", "http://localhost:8001/callback")

DEFAULT_WEBHOOK_CONFIG = {
    "webhook_type": "raw",
    "callback_url": DEFAULT_CALLBACK_URL,
}

__all__ = [
    "DEFAULT_CALLBACK_URL",
    "DEFAULT_WEBHOOK_CONFIG",
]

