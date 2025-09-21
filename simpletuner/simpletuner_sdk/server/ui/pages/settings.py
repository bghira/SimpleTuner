"""Placeholder settings page until full migration is complete."""
from __future__ import annotations

from nicegui import ui

from ..layouts.navigation import render_nav


@ui.page("/settings")
def settings_page() -> None:
    render_nav("settings")
    with ui.column().classes("max-w-3xl mx-auto mt-8 gap-4 text-slate-100"):
        ui.label("Settings coming soon").classes("text-xl font-semibold")
        ui.label("Legacy controls will be ported to NiceGUI in a subsequent phase.").classes("text-sm text-slate-400")
