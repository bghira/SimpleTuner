"""Landing page for the NiceGUI interface."""
from __future__ import annotations

from nicegui import ui

from ..layouts.navigation import render_nav


@ui.page("/")
def index_page() -> None:
    render_nav("home")
    with ui.column().classes("max-w-3xl mx-auto mt-12 gap-4 text-slate-100"):
        ui.label("Welcome to SimpleTuner NiceGUI").classes("text-2xl font-semibold")
        ui.label(
            "Use the dataset builder to manage multidatabackend.json and review the trainer plan overview for saved data."
        ).classes("text-sm text-slate-300")
        with ui.row().classes("gap-4"):
            ui.link("open dataset builder", "/datasets").props("color=primary")
            ui.link("open trainer control", "/web/trainer").props("color=secondary")
            ui.link("view trainer overview", "/web/trainer").props("color=secondary")
