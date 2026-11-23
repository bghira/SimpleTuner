"""Text helpers for log output."""

import re

_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from text."""
    return _ANSI_COLOR.sub("", text)
