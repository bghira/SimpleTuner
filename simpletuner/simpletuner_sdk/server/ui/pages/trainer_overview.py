"""Dataset plan overview page."""
from __future__ import annotations

import asyncio

from nicegui import ui

from ..components.validations import build_dataset_rows, render_validations
from ..layouts.navigation import render_nav
from ..services.http import fetch_json
from ..state.plan_controller import PlanController


@ui.page("/trainer")
async def trainer_overview_page() -> None:
    render_nav("trainer")
    controller = PlanController()

    page_container = ui.column().classes("max-w-5xl mx-auto mt-6 gap-5")
    with page_container:
        summary_label = ui.label().classes("text-sm text-slate-300")
        dataset_container = ui.column().classes("gap-3")
        validations_container = ui.column().classes("gap-2")
        refresh_button = ui.button("refresh").props("color=primary")

    def render_plan() -> None:
        count = len(controller.datasets)
        timestamp = controller.updated_at or "never"
        summary_label.set_text(f"{count} datasets • source: {controller.source} • last saved: {timestamp}")
        dataset_container.clear()
        if not controller.datasets:
            ui.label("no datasets saved yet").classes("text-sm text-slate-400")
            render_validations(validations_container, controller.validations)
            return
        for dataset in controller.datasets:
            with ui.card().classes("bg-slate-900 text-slate-100 border border-slate-800"):
                ui.label(dataset.get("id", "dataset")).classes("text-base font-semibold")
                ui.label(
                    f"{dataset.get('type', 'unknown')} • {dataset.get('dataset_type', 'unknown')}"
                ).classes("text-xs text-slate-400")
                with ui.column().classes("mt-3 gap-2"):
                    for key, value in build_dataset_rows(dataset):
                        with ui.row().classes("justify-between text-sm"):
                            ui.label(key).classes("text-slate-300")
                            ui.label(value).classes("text-slate-100")
        render_validations(validations_container, controller.validations)

    async def refresh_from_api() -> None:
        try:
            result = await fetch_json("/api/datasets/plan")
        except Exception as exc:
            ui.notify(f"failed to reach dataset plan api: {exc}", color="negative")
            controller.reload()
            render_plan()
            return

        if not result.get("ok", False):
            status = result.get("status")
            ui.notify(f"plan request failed (status {status})", color="negative")
            controller.recompute_validations()
            render_plan()
            return

        data = result.get("data")
        if isinstance(data, dict):
            controller.apply_plan_response(data)
        render_plan()

    refresh_button.on_click(refresh_from_api)
    render_plan()
    ui.timer(0.1, lambda: asyncio.create_task(refresh_from_api()), once=True)
