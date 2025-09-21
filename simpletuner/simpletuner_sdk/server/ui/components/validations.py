"""Reusable validation rendering helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from nicegui import ui

from simpletuner.simpletuner_sdk.server.services.dataset_plan import ValidationMessage


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "—"
    return str(value)


def render_validations(container: ui.column, validations: Iterable[ValidationMessage]) -> None:
    container.clear()
    validations = list(validations)
    if not validations:
        ui.label("no validation messages").classes("text-sm text-slate-400").props("style='margin-top:0'")
        return
    level_classes = {"error": "text-red-400", "warning": "text-amber-300", "info": "text-slate-300"}
    for message in validations:
        level = message.level
        text_class = level_classes.get(level, "text-slate-300")
        with container.row().classes("items-start gap-3"):
            ui.label(level).classes(f"uppercase text-xs tracking-[0.3em] {text_class}")
            details = f"{message.field}: {message.message}"
            if message.suggestion:
                details = f"{details} ({message.suggestion})"
            ui.label(details).classes("text-sm text-slate-200")


def build_dataset_rows(dataset: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    rows = []
    for key, value in dataset.items():
        if key in {"id", "type", "dataset_type"}:
            continue
        rows.append((key, format_value(value)))
    return tuple(rows)
