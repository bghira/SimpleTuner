"""Navigation helpers shared by NiceGUI pages."""
from __future__ import annotations

from nicegui import ui


def render_nav(active: str) -> None:
    """Render the persistent header with navigation links."""
    with ui.header().classes("items-center justify-between px-6 py-2 bg-slate-900 text-white"):
        ui.label("SimpleTuner UI").classes("font-semibold")
        with ui.row().classes("gap-4 text-sm"):
            link("builder", "/web/datasets", active)
            link("trainer", "/web/trainer", active)
            link("training", "/web/training", active)
            link("settings", "/web/settings", active)


def link(label: str, path: str, active: str) -> None:
    text_classes = "text-white" if active == label else "text-slate-400"
    ui.link(label, path).classes(text_classes)
