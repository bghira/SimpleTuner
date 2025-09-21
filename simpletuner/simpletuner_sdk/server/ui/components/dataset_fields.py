"""Dynamic dataset field components."""
from __future__ import annotations

from typing import Any

from nicegui import ui

from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import DatasetField


def create_field_component(field: DatasetField, value: Any):
    label = f"{field.label}{' *' if field.required else ''}"
    with ui.column().classes("gap-1 w-full"):
        if field.type == "text":
            component = ui.input(label=label, value=str(value) if value is not None else "", placeholder=field.placeholder or "")
        elif field.type == "textarea":
            component = ui.textarea(label=label, value=str(value) if value is not None else "", placeholder=field.placeholder or "")
        elif field.type == "number":
            component = ui.number(
                label=label,
                value=value,
                min=field.min,
                max=field.max,
                step=field.step,
            )
        elif field.type == "select":
            options = {option.value: option.label for option in field.options or []}
            component = ui.select(options, value=value if value is not None else field.defaultValue, label=label)
        elif field.type == "toggle":
            component = ui.switch(label, value=bool(value) if value is not None else bool(field.defaultValue))
        else:  # fallback to text
            component = ui.input(label=label, value=str(value) if value is not None else "")
        description = field.description
        if description:
            ui.label(description).classes("text-xs text-slate-400")
        if field.advanced:
            ui.label("advanced").classes("text-[10px] uppercase tracking-[0.4em] text-amber-200")
        return component
