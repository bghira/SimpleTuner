"""Helpers for building documentation URLs."""

from __future__ import annotations

import os
import re
from typing import Optional

from fastapi.templating import Jinja2Templates

DEFAULT_DOCS_BASE_URL = "https://bghira.github.io/SimpleTuner/"
DOCS_BASE_ENV = "SIMPLETUNER_DOCS_BASE_URL"
GITHUB_DOCS_PREFIX = "https://github.com/bghira/SimpleTuner/blob/main/documentation/"


def _docs_base_url() -> str:
    base = os.environ.get(DOCS_BASE_ENV, DEFAULT_DOCS_BASE_URL).strip()
    if not base:
        base = DEFAULT_DOCS_BASE_URL
    if not base.endswith("/"):
        base += "/"
    return base


def _normalize_anchor(anchor: str) -> str:
    anchor = anchor.lstrip("#")
    if not anchor:
        return ""
    return re.sub(r"-{2,}", "-", anchor)


def docs_url(doc_ref: Optional[str]) -> str:
    if not doc_ref:
        return _docs_base_url()

    raw = str(doc_ref)
    if raw.startswith(("http://", "https://")):
        if raw.startswith(GITHUB_DOCS_PREFIX):
            raw = raw[len(GITHUB_DOCS_PREFIX) :]
        else:
            return raw

    raw = raw.lstrip("/")
    if raw.startswith("documentation/"):
        raw = raw[len("documentation/") :]

    path, _, anchor = raw.partition("#")
    path = path.strip()
    if path.lower().endswith(".md"):
        path = path[:-3]
    if path.endswith("/index"):
        path = path[: -len("/index")]
    elif path == "index":
        path = ""

    base = _docs_base_url()
    if path:
        url = f"{base}{path}/" if not path.endswith("/") else f"{base}{path}"
    else:
        url = base

    if anchor:
        normalized = _normalize_anchor(anchor)
        if normalized:
            url = f"{url}#{normalized}"
    return url


def setup_docs_helpers(templates: Jinja2Templates) -> None:
    templates.env.globals["docs_url"] = docs_url
