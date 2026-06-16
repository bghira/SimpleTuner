"""Import path setup for running prompt2effect scripts directly."""

from __future__ import annotations

import sys
from pathlib import Path

PROMPT2EFFECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROMPT2EFFECT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

for path in (REPO_ROOT, SCRIPTS_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
