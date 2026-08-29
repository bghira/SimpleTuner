"""Runtime setup for SimpleTuner's RVC integration."""

from __future__ import annotations

import os


def configure_rvc_runtime() -> None:
    os.environ.setdefault("FAISS_OPT_LEVEL", "generic")
