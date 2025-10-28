"""Version metadata API endpoint."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter

import simpletuner
from simpletuner.simpletuner_sdk.server.utils.paths import get_simpletuner_root

router = APIRouter(prefix="/api", tags=["version"])


def _run_git_command(args: list[str], cwd: Path) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


@functools.lru_cache(maxsize=1)
def _compute_version_payload() -> Dict[str, Any]:
    """Gather version information, caching the result for future calls."""
    version = getattr(simpletuner, "__version__", None)
    major: Optional[int] = None
    if isinstance(version, str):
        parts = version.split(".")
        if parts:
            try:
                major = int(parts[0])
            except ValueError:
                major = None

    root = get_simpletuner_root()
    git_install = False
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None

    git_dir = root / ".git"
    if git_dir.exists() and git_dir.is_dir():
        git_install = True
        git_commit = _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        status_output = _run_git_command(["git", "status", "--porcelain"], cwd=root)
        git_dirty = bool(status_output.strip()) if isinstance(status_output, str) else None

    return {
        "version": version,
        "major": major,
        "git_install": git_install,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


@router.get("/version")
async def get_version() -> Dict[str, Any]:
    """Return SimpleTuner version metadata."""
    # Copy the cached payload to ensure callers can't mutate shared state.
    payload = dict(_compute_version_payload())
    return payload
