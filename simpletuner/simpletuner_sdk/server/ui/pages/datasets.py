"""Dataset builder NiceGUI page."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from nicegui import events, ui

from ..components.dataset_fields import create_field_component
from ..components.validations import build_dataset_rows, render_validations
from ..layouts.navigation import render_nav
from ..services.http import fetch_json
from ..state.plan_controller import BlueprintOption, PlanController
from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import BackendBlueprint


@ui.page("/datasets")
async def dataset_builder_page() -> None:
    render_nav("builder")
    controller = PlanController()

    page_container = ui.column().classes("max-w-5xl mx-auto mt-6 gap-6")
    with page_container:
        summary_label = ui.label().classes("text-sm text-slate-300")
        buttons_row = ui.row().classes("gap-3")
        dataset_container = ui.column().classes("gap-3")
        validations_container = ui.column().classes("gap-2")

    dataset_dialog = ui.dialog()
    with dataset_dialog, ui.card().classes("max-w-3xl w-[640px] bg-slate-900 text-slate-100 gap-4"):
        dialog_title = ui.label("edit dataset").classes("text-lg font-semibold")
        blueprint_select = ui.select(
            {key: option.label for key, option in controller.blueprint_options.items()},
            label="dataset blueprint",
        ).classes("w-full")
        dataset_id_input = ui.input("dataset id").props("dense").classes("w-full")
        dataset_name_input = ui.input("display name (optional)").props("dense").classes("w-full")
        fields_container = ui.column().classes("gap-3 max-h-[420px] overflow-y-auto pr-1")
        with ui.row().classes("justify-end gap-2"):
            ui.button("cancel", on_click=dataset_dialog.close).props("color=secondary")
            save_button = ui.button("save").props("color=primary")

    dialog_index: Optional[int] = None
    dialog_option: Optional[BlueprintOption] = None
    field_components: Dict[str, Any] = {}
    dialog_dataset: Dict[str, Any] = {}

    def refresh_summary() -> None:
        count = len(controller.datasets)
        timestamp = controller.updated_at or "never"
        summary_label.set_text(f"{count} datasets • source: {controller.source} • last saved: {timestamp}")

    def refresh_validations() -> None:
        render_validations(validations_container, controller.validations)

    def refresh_datasets() -> None:
        dataset_container.clear()
        if not controller.datasets:
            ui.label("no datasets configured yet").classes("text-sm text-slate-400")
            return
        for index, dataset in enumerate(controller.datasets):
            option_key = controller.option_value_for_dataset(dataset)
            option = controller.blueprint_options.get(option_key or "")
            title = dataset.get("id") or f"dataset {index + 1}"
            subtitle = (
                f"{dataset.get('type', 'unknown')} • {dataset.get('dataset_type', 'unknown')}"
                if option is None
                else f"{option.blueprint.label} • {option.dataset_type}"
            )
            with ui.expansion(title, value=False).classes("bg-slate-900 text-slate-100 border border-slate-800 rounded-xl"):
                ui.label(subtitle).classes("text-xs text-slate-400")
                with ui.column().classes("mt-3 gap-2"):
                    for key, value in build_dataset_rows(dataset):
                        with ui.row().classes("justify-between text-sm"):
                            ui.label(key).classes("text-slate-300")
                            ui.label(value).classes("text-slate-100")
                with ui.row().classes("mt-3 gap-2"):
                    ui.button("edit", on_click=lambda i=index: open_dataset_dialog(i)).props("color=primary")
                    ui.button(
                        "remove",
                        on_click=lambda i=index: remove_dataset(i),
                    ).props("color=negative")

    def refresh_all() -> None:
        controller.recompute_validations()
        refresh_summary()
        refresh_datasets()
        refresh_validations()

    def merge_defaults(option: BlueprintOption, data: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**option.blueprint.defaults}
        merged.update({k: v for k, v in data.items() if v is not None})
        merged["type"] = option.blueprint.backendType
        merged["dataset_type"] = option.dataset_type
        if not merged.get("id"):
            merged["id"] = controller.suggest_id(option.key)
        return merged

    def rebuild_fields(option: Optional[BlueprintOption]) -> None:
        nonlocal field_components, dialog_dataset, dialog_option
        dialog_option = option
        field_components = {}
        fields_container.clear()
        if option is None:
            ui.label("blueprint metadata unavailable").classes("text-sm text-red-400")
            return
        dialog_dataset = merge_defaults(option, dialog_dataset)
        dataset_id_input.value = str(dialog_dataset.get("id", ""))
        dataset_name_input.value = str(dialog_dataset.get("name", "")) if dialog_dataset.get("name") else ""
        with fields_container:
            for field in option.blueprint.fields:
                value = dialog_dataset.get(field.id, option.blueprint.defaults.get(field.id, field.defaultValue))
                component = create_field_component(field, value)
                field_components[field.id] = component

    def handle_blueprint_change(event: events.ValueChangeEventArguments) -> None:
        key = event.value
        rebuild_fields(controller.blueprint_options.get(key))

    async def handle_save() -> None:
        if dialog_option is None:
            ui.notify("select a blueprint", color="negative")
            return
        dataset_id = (dataset_id_input.value or "").strip()
        if not dataset_id:
            ui.notify("dataset id is required", color="negative")
            return
        record: Dict[str, Any] = {
            "id": dataset_id,
            "dataset_type": dialog_option.dataset_type,
            "type": dialog_option.blueprint.backendType,
        }
        name_value = (dataset_name_input.value or "").strip()
        if name_value:
            record["name"] = name_value
        for field_id, component in field_components.items():
            value = component.value
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    record[field_id] = trimmed
            elif value is not None:
                record[field_id] = value
        if dialog_index is not None:
            controller.datasets[dialog_index] = record
        else:
            controller.datasets.append(record)
        controller.recompute_validations()
        refresh_summary()
        refresh_datasets()
        refresh_validations()
        dataset_dialog.close()
        ui.notify("dataset updated", color="positive")

    def remove_dataset(index: int) -> None:
        controller.datasets.pop(index)
        refresh_all()
        ui.notify("dataset removed", color="positive")

    def open_dataset_dialog(index: Optional[int] = None) -> None:
        nonlocal dialog_index, dialog_dataset
        dialog_index = index
        dialog_dataset = controller.datasets[index] if index is not None else {}
        blueprint_key = controller.option_value_for_dataset(dialog_dataset) if index is not None else None
        if blueprint_key is None and controller.blueprint_options:
            blueprint_key = next(iter(controller.blueprint_options.keys()))
        blueprint_select.value = blueprint_key
        dialog_title.set_text("edit dataset" if index is not None else "add dataset")
        rebuild_fields(controller.blueprint_options.get(blueprint_key))
        dataset_dialog.open()

    blueprint_select.on_value_change(handle_blueprint_change)
    save_button.on_click(handle_save)

    async def persist_plan() -> None:
        payload = {"datasets": controller.datasets}
        try:
            result = await fetch_json("/api/datasets/plan", "POST", payload)
        except Exception as exc:
            ui.notify(f"failed to reach dataset plan api: {exc}", color="negative")
            return

        if not result.get("ok", False):
            data = result.get("data")
            if isinstance(data, dict):
                detail = data.get("detail")
                validations = detail.get("validations") if isinstance(detail, dict) else None
                if validations:
                    controller.set_validations(validations)
                    refresh_validations()
                    ui.notify("dataset plan failed server validation", color="negative")
                    return
            status = result.get("status")
            ui.notify(f"plan save failed (status {status})", color="negative")
            return

        data = result.get("data")
        if isinstance(data, dict):
            controller.apply_plan_response(data)
        refresh_all()
        ui.notify("dataset plan saved", color="positive")

    async def reload_plan() -> None:
        try:
            result = await fetch_json("/api/datasets/plan")
        except Exception as exc:
            ui.notify(f"failed to reach dataset plan api: {exc}", color="negative")
            return

        if not result.get("ok", False):
            status = result.get("status")
            ui.notify(f"plan reload failed (status {status})", color="negative")
            return

        data = result.get("data")
        if isinstance(data, dict):
            controller.apply_plan_response(data)
        refresh_all()
        ui.notify("dataset plan reloaded", color="positive")

    async def initialise_from_api() -> None:
        try:
            blueprint_result = await fetch_json("/api/datasets/blueprints")
            if blueprint_result.get("ok", False):
                blueprint_payload = blueprint_result.get("data")
                if isinstance(blueprint_payload, dict):
                    remote_blueprints = [
                        BackendBlueprint.model_validate(item)
                        for item in blueprint_payload.get("blueprints", [])
                    ]
                    if remote_blueprints:
                        controller.replace_blueprints(remote_blueprints)
                        blueprint_select.options = {
                            key: option.label for key, option in controller.blueprint_options.items()
                        }
                        blueprint_select.update()

            plan_result = await fetch_json("/api/datasets/plan")
            if plan_result.get("ok", False):
                data = plan_result.get("data")
                if isinstance(data, dict):
                    controller.apply_plan_response(data)
        except Exception:
            ui.notify("dataset api unavailable, falling back to cached plan", color="warning")
            controller.reload()

        refresh_all()

    buttons_row.clear()
    buttons_row.classes("gap-3")
    with buttons_row:
        ui.button("add dataset", on_click=lambda: open_dataset_dialog()).props("color=primary")
        ui.button("save plan", on_click=persist_plan).props("color=primary")
        ui.button("reload", on_click=reload_plan).props("color=secondary")

    refresh_all()
    ui.timer(0.1, lambda: asyncio.create_task(initialise_from_api()), once=True)
