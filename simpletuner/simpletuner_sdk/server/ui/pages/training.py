"""Trainer control NiceGUI page styled like the legacy WebUI."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from nicegui import events, ui

from simpletuner.helpers.models.all import get_model_flavour_choices, model_families

from ..components.validations import build_dataset_rows
from ..layouts.navigation import render_nav
from ..services.http import fetch_json
from ..state.plan_controller import PlanController
from ..theme import ensure_legacy_theme


def _select_options(options: List[Dict[str, Any]]) -> Dict[str, str]:
    return {item["value"]: item.get("label", item["value"]) for item in options}


def _default_selected(options: List[Dict[str, Any]], fallback: Optional[str] = None) -> Optional[str]:
    for option in options:
        if option.get("selected"):
            return option["value"]
    return fallback or (options[0]["value"] if options else None)


def _default_webhooks(callback_url: str) -> str:
    payload = {
        "webhook_type": "raw",
        "callback_url": callback_url,
    }
    return json.dumps(payload, indent=4)


def _dataset_plan_json(controller: PlanController) -> str:
    if not controller.datasets:
        return "[]"
    return json.dumps(controller.datasets, indent=4)


def load_default_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    try:
        from simpletuner.helpers.configuration.cmd_args import get_default_config as get_cmd_default_config  # type: ignore

        try:
            defaults = get_cmd_default_config()
        except Exception:
            defaults = {}
    except Exception:
        defaults = {}

    template_defaults = {
        "job_id": f"training_run_{int(time.time())}",
        "data_backend_config": "config/multidatabackend.json",
        "resume_from_checkpoint": "",
        "output_dir": "output",
        "seed": 42,
        "model_type": "lora",
        "lora_type": "standard",
        "lycoris_config": "config/lycoris_config.json",
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "flux_lora_target": "all",
        "use_dora": False,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 4e-7,
        "num_train_epochs": 10,
        "max_train_steps": 0,
        "loss_type": "l2",
        "prediction_type": "",
        "mixed_precision": "bf16",
        "noise_offset": 0.1,
        "input_perturbation": 0.0,
        "snr_gamma": 5.0,
        "flow_sigmoid_scale": 1.0,
        "flow_schedule_shift": 3.0,
        "flux_guidance_mode": "constant",
        "flux_guidance_value": 1.0,
        "ema_decay": 0.995,
        "lr_scheduler": "polynomial",
        "lr_scheduler_num_cycles": 1,
    }

    for key, value in template_defaults.items():
        defaults.setdefault(key, value)

    return defaults


def model_type_options() -> List[Dict[str, Any]]:
    return [
        {"value": "full", "label": "Full Fine-tune"},
        {"value": "lora", "label": "LoRA", "selected": True},
    ]


def model_family_options() -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    for key, family in model_families.items():
        label = getattr(family, "NAME", key.replace("_", " ").title())
        options.append({"value": key, "label": label, "selected": key == "flux"})
    return options


def lr_scheduler_options() -> List[Dict[str, Any]]:
    return [
        {"value": "constant", "label": "Constant"},
        {"value": "constant_with_warmup", "label": "Constant with Warmup"},
        {"value": "cosine", "label": "Cosine"},
        {"value": "cosine_with_restarts", "label": "Cosine with Restarts"},
        {"value": "polynomial", "label": "Polynomial", "selected": True},
        {"value": "linear", "label": "Linear"},
        {"value": "sine", "label": "Sine"},
    ]


@ui.page("/training")
async def training_control_page() -> None:
    ensure_legacy_theme()
    render_nav("training")

    defaults = load_default_config()
    model_types = model_type_options()
    families = model_family_options()
    schedulers = lr_scheduler_options()

    default_family = _default_selected(families)
    default_model_type = _default_selected(model_types)
    default_scheduler = _default_selected(schedulers)

    trainer_inputs: Dict[str, Callable[[], Optional[str]]] = {}

    def register_input(
        cli_key: str,
        component: Any,
        *,
        transform: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        def getter() -> Optional[str]:
            value = component.value
            if transform is not None:
                return transform(value)
            if isinstance(value, str):
                trimmed = value.strip()
                return trimmed or None
            if value is None:
                return None
            return str(value)

        trainer_inputs[cli_key] = getter

    def register_checkbox(cli_key: str, component: Any) -> None:
        def getter() -> Optional[str]:
            return "true" if component.value else None

        trainer_inputs[cli_key] = getter

    def create_text_field(
        parent: ui.element,
        field_id: str,
        label: str,
        default: str = "",
        *,
        placeholder: str = "",
        cli_key: Optional[str] = None,
    ) -> Any:
        with parent:
            wrapper = ui.element('div').classes('mb-3')
            with wrapper:
                ui.element('label').props(f'for={field_id}').classes('form-label').text(label)
                field = ui.input(value=default, placeholder=placeholder)
                field.props(f'id={field_id}', 'clearable=false', 'dense=true')
                field.classes('legacy-input')
                field.props('input-class=form-control')
        if cli_key:
            register_input(cli_key, field)
        return field

    def create_number_field(
        parent: ui.element,
        field_id: str,
        label: str,
        default: float,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        cli_key: Optional[str] = None,
        transform: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> Any:
        with parent:
            wrapper = ui.element('div').classes('mb-3')
            with wrapper:
                ui.element('label').props(f'for={field_id}').classes('form-label').text(label)
                field = ui.number(value=default, min=min_value, max=max_value, step=step)
                field.props(f'id={field_id}', 'dense=true')
                field.classes('legacy-input')
        if cli_key:
            register_input(cli_key, field, transform=transform)
        return field

    def create_select_field(
        parent: ui.element,
        field_id: str,
        label: str,
        options: Dict[str, str],
        default: Optional[str] = None,
        *,
        cli_key: Optional[str] = None,
    ) -> Any:
        with parent:
            wrapper = ui.element('div').classes('mb-3')
            with wrapper:
                ui.element('label').props(f'for={field_id}').classes('form-label').text(label)
                field = ui.select(options, value=default)
                field.props(f'id={field_id}', 'dense=true', 'clearable=false', 'popup-content-class=legacy-select-popup')
                field.classes('legacy-input')
        if cli_key:
            register_input(cli_key, field)
        return field

    def create_checkbox_field(
        parent: ui.element,
        field_id: str,
        label: str,
        default: bool = False,
        *,
        cli_key: Optional[str] = None,
    ) -> Any:
        with parent:
            wrapper = ui.element('div').classes('form-check form-switch mb-3 legacy-form-check')
            with wrapper:
                field = ui.checkbox(label, value=default)
                field.props(f'id={field_id}')
                field.classes('legacy-checkbox')
        if cli_key:
            register_checkbox(cli_key, field)
        return field

    tab_definitions = [
        ("basic", "Basic Config", "fas fa-cog"),
        ("model", "Model Settings", "fas fa-brain"),
        ("training", "Training", "fas fa-graduation-cap"),
        ("advanced", "Advanced", "fas fa-sliders-h"),
        ("data", "Dataloader", "fas fa-database"),
    ]

    tab_buttons: Dict[str, Any] = {}
    tab_panels: Dict[str, Any] = {}
    active_tab = {"value": "basic"}

    def set_tab(name: str) -> None:
        active_tab["value"] = name
        for key, btn in tab_buttons.items():
            if key == name:
                btn.classes(add='active')
            else:
                btn.classes(remove='active')
        for key, panel in tab_panels.items():
            if key == name:
                panel.classes(add='active')
            else:
                panel.classes(remove='active')

    root = ui.element('div').classes('dashboard-wrapper')

    # Sidebar and navigation
    with root:
        toggle_primary = ui.element('button').classes('sidebar-toggle d-lg-none').props('type=button')
        with toggle_primary:
            ui.html('<i class="fas fa-bars"></i>')

        sidebar = ui.element('aside').classes('sidebar').props('id=sidebar')
        with sidebar:
            ui.element('div').classes('sidebar-header').html('<h1>🐈 SimpleTuner</h1>')
            with ui.element('nav').classes('sidebar-nav'):
                with ui.element('div').classes('nav-section'):
                    ui.element('div').classes('nav-section-title').text('Configuration')
                    for name, label, icon in tab_definitions:
                        button = ui.element('button').classes('tab-btn').props(f'data-tab={name} type=button')
                        if name == active_tab['value']:
                            button.classes(add='active')
                        with button:
                            ui.html(f'<i class="{icon}"></i> {label}')
                        button.on('click', lambda _, tab=name: set_tab(tab))
                        tab_buttons[name] = button

        # Main content
        main = ui.element('main').classes('main-content')
        with main:
            with ui.element('div').classes('topbar'):
                toggle_secondary = ui.element('button').classes('sidebar-toggle d-lg-none').props('type=button')
                with toggle_secondary:
                    ui.html('<i class="fas fa-bars"></i>')
                ui.element('div').classes('topbar-title').text('Training Configuration')
                with ui.element('div').classes('header-actions'):
                    validate_button = ui.element('button').classes('btn btn-sm btn-primary').props('id=validateBtn type=button')
                    with validate_button:
                        ui.html('<i class="fas fa-check-circle"></i> <span class="btn-text">Validate</span>')
                    run_button = ui.element('button').classes('btn btn-sm btn-success').props('id=runBtn type=button')
                    with run_button:
                        ui.html('<i class="fas fa-play"></i> <span>Start Training</span>')
                    cancel_button = ui.element('button').classes('btn btn-sm btn-danger').props('id=cancelBtn type=button')
                    with cancel_button:
                        ui.html('<i class="fas fa-stop"></i> <span>Cancel</span>')
                connection_status = ui.label('Disconnected').classes('connection-status').props('id=connectionStatus')

            with ui.element('div').classes('content-wrapper'):
                form = ui.element('form').props('id=configForm')
                with form:
                    status_container = ui.column().classes('status-container gap-2').props('id=statusContainer')
                    tab_wrapper = ui.element('div').classes('tab-content-wrapper')

    # Tab panels and configuration content
    with tab_wrapper:
        for name, label, icon in tab_definitions:
            panel = ui.element('div').classes('tab-content').props(f'id={name}-tab')
            if name == active_tab['value']:
                panel.classes(add='active')
            tab_panels[name] = panel

    # Basic tab content
    with tab_panels['basic']:
        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-play-circle"></i> Basic Configuration</h5>')
            grid = ui.element('div').classes('form-grid-2')
            job_id_input = create_text_field(grid, 'job_id', 'Job ID', defaults.get('job_id', ''))
            resume_input = create_text_field(
                grid,
                'resume_from_checkpoint',
                'Resume From Checkpoint',
                defaults.get('resume_from_checkpoint', ''),
                cli_key='--resume_from_checkpoint',
            )
            data_backend_input = create_text_field(
                ui.element('div').classes('form-grid-1'),
                'data_backend_config',
                'Data Backend Config',
                defaults.get('data_backend_config', 'config/multidatabackend.json'),
                cli_key='--data_backend_config',
            )
            output_dir_input = create_text_field(
                ui.element('div').classes('form-grid-1'),
                'output_dir',
                'Output Directory',
                defaults.get('output_dir', 'output'),
                cli_key='--output_dir',
            )
            seed_input = create_number_field(
                ui.element('div').classes('form-grid-1'),
                'seed',
                'Random Seed',
                defaults.get('seed', 42),
                min_value=0,
                step=1,
                cli_key='--seed',
                transform=lambda value: str(int(value)) if value is not None else None,
            )

    # Model tab content
    with tab_panels['model']:
        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-cube"></i> Model Architecture</h5>')
            grid_model = ui.element('div').classes('form-grid-2')
            model_family_select = create_select_field(
                grid_model,
                'model_family',
                'Model Family',
                _select_options(families),
                default_family,
                cli_key='--model_family',
            )
            model_type_select = create_select_field(
                grid_model,
                'model_type',
                'Model Type',
                _select_options(model_types),
                default_model_type,
                cli_key='--model_type',
            )
            grid_model_path = ui.element('div').classes('form-grid-1')
            model_path_input = create_text_field(
                grid_model_path,
                'model_path',
                'Path for pretrained model (optional, local or huggingface)',
                defaults.get('pretrained_model_name_or_path', ''),
                cli_key='--pretrained_model_name_or_path',
            )
            model_flavour_select = create_select_field(
                ui.element('div').classes('form-grid-1'),
                'model_flavour',
                'Model Flavour',
                {"": "Default"},
                '',
                cli_key='--model_flavour',
            )

        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-layer-group"></i> LoRA Configuration</h5>')
            grid_lora = ui.element('div').classes('form-grid-2')
            lora_type_select = create_select_field(
                grid_lora,
                'lora_type',
                'LoRA Type',
                {"standard": "PEFT LoRA", "lycoris": "LyCORIS LoKr"},
                defaults.get('lora_type', 'standard'),
                cli_key='--lora_type',
            )
            lycoris_config_input = create_text_field(
                grid_lora,
                'lycoris_config',
                'LyCORIS Config Path',
                defaults.get('lycoris_config', 'config/lycoris_config.json'),
                cli_key='--lycoris_config',
            )
            grid_lora_numbers = ui.element('div').classes('form-grid-3')
            lora_rank_input = create_number_field(
                grid_lora_numbers,
                'lora_rank',
                'LoRA Rank',
                defaults.get('lora_rank', 16),
                min_value=1,
                max_value=256,
                step=1,
                cli_key='--lora_rank',
                transform=lambda value: str(int(value)) if value is not None else None,
            )
            lora_alpha_input = create_number_field(
                grid_lora_numbers,
                'lora_alpha',
                'LoRA Alpha',
                defaults.get('lora_alpha', 16),
                min_value=1,
                step=1,
                cli_key='--lora_alpha',
                transform=lambda value: str(int(value)) if value is not None else None,
            )
            lora_dropout_input = create_number_field(
                grid_lora_numbers,
                'lora_dropout',
                'LoRA Dropout',
                defaults.get('lora_dropout', 0.1),
                min_value=0,
                max_value=1,
                step=0.01,
                cli_key='--lora_dropout',
            )
            grid_flux = ui.element('div').classes('form-grid-2')
            flux_target_select = create_select_field(
                grid_flux,
                'flux_lora_target',
                'Flux LoRA Target',
                {
                    "mmdit": "MMDiT",
                    "context": "Context",
                    "context+ffs": "Context + FFS",
                    "all": "All (Recommended)",
                    "all+ffs": "All + FFS",
                    "all+ffs+embedder": "All + FFS + Embedder",
                    "all+ffs+embedder+controlnet": "All + FFS + Embedder + ControlNet",
                    "ai-toolkit": "AI Toolkit",
                    "tiny": "Tiny",
                    "nano": "Nano",
                    "controlnet": "ControlNet Only",
                },
                defaults.get('flux_lora_target', 'all'),
                cli_key='--flux_lora_target',
            )
            use_dora_checkbox = create_checkbox_field(
                grid_flux,
                'use_dora',
                'Use DoRA (Weight-Decomposed LoRA)',
                defaults.get('use_dora', False),
                cli_key='--use_dora',
            )

    # Training tab content
    with tab_panels['training']:
        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-tachometer-alt"></i> Training Parameters</h5>')
            grid_training = ui.element('div').classes('form-grid-3')
            batch_size_input = create_number_field(
                grid_training,
                'train_batch_size',
                'Batch Size',
                defaults.get('train_batch_size', 1),
                min_value=1,
                step=1,
                cli_key='--train_batch_size',
                transform=lambda value: str(int(value)) if value is not None else None,
            )
            grad_accum_input = create_number_field(
                grid_training,
                'gradient_accumulation_steps',
                'Gradient Accumulation',
                defaults.get('gradient_accumulation_steps', 1),
                min_value=1,
                step=1,
                cli_key='--gradient_accumulation_steps',
                transform=lambda value: str(int(value)) if value is not None else None,
            )
            learning_rate_input = create_number_field(
                grid_training,
                'learning_rate',
                'Learning Rate',
                defaults.get('learning_rate', 4e-7),
                min_value=0,
                step=1e-6,
                cli_key='--learning_rate',
            )
            grid_epochs = ui.element('div').classes('form-grid-2')
            epochs_input = create_number_field(
                grid_epochs,
                'num_train_epochs',
                'Number of Epochs',
                defaults.get('num_train_epochs', 10),
                min_value=0,
                step=1,
                cli_key='--num_train_epochs',
                transform=lambda value: str(int(value)) if value is not None else None,
            )
            max_steps_input = create_number_field(
                grid_epochs,
                'max_train_steps',
                'Max Train Steps (0 = use epochs)',
                defaults.get('max_train_steps', 0),
                min_value=0,
                step=1,
                cli_key='--max_train_steps',
                transform=lambda value: str(int(value)) if value is not None else None,
            )

        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-cogs"></i> Training Settings</h5>')
            grid_settings = ui.element('div').classes('form-grid-3')
            loss_select = create_select_field(
                grid_settings,
                'loss_type',
                'Loss Type',
                {"l2": "L2 (MSE)", "huber": "Huber", "smooth_l1": "Smooth L1"},
                defaults.get('loss_type', 'l2'),
                cli_key='--loss_type',
            )
            prediction_select = create_select_field(
                grid_settings,
                'prediction_type',
                'Prediction Type',
                {"": "Auto-detect", "epsilon": "Epsilon", "v_prediction": "V-Prediction", "sample": "Sample", "flow_matching": "Flow Matching"},
                defaults.get('prediction_type', ''),
                cli_key='--prediction_type',
            )
            precision_select = create_select_field(
                grid_settings,
                'mixed_precision',
                'Mixed Precision',
                {"no": "No (FP32)", "fp16": "FP16", "bf16": "BF16", "fp8": "FP8 (Experimental)"},
                defaults.get('mixed_precision', 'bf16'),
                cli_key='--mixed_precision',
            )
            grid_misc = ui.element('div').classes('form-grid-3')
            noise_offset_input = create_number_field(
                grid_misc,
                'noise_offset',
                'Noise Offset',
                defaults.get('noise_offset', 0.1),
                min_value=0,
                max_value=1,
                step=0.01,
                cli_key='--noise_offset',
            )
            input_perturb_input = create_number_field(
                grid_misc,
                'input_perturbation',
                'Input Perturbation',
                defaults.get('input_perturbation', 0.0),
                min_value=0,
                max_value=1,
                step=0.01,
                cli_key='--input_perturbation',
            )
            snr_gamma_input = create_number_field(
                grid_misc,
                'snr_gamma',
                'SNR Gamma',
                defaults.get('snr_gamma', 5.0),
                min_value=0,
                step=0.1,
                cli_key='--snr_gamma',
            )
            grid_flow = ui.element('div').classes('form-grid-3')
            flow_sigmoid_input = create_number_field(
                grid_flow,
                'flow_sigmoid_scale',
                'Flow Sigmoid Scale',
                defaults.get('flow_sigmoid_scale', 1.0),
                min_value=0,
                step=0.1,
                cli_key='--flow_sigmoid_scale',
            )
            flow_shift_input = create_number_field(
                grid_flow,
                'flow_schedule_shift',
                'Flow Schedule Shift',
                defaults.get('flow_schedule_shift', 3.0),
                min_value=0,
                step=0.1,
                cli_key='--flow_schedule_shift',
            )
            flux_guidance_mode_select = create_select_field(
                grid_flow,
                'flux_guidance_mode',
                'Flux Guidance Mode',
                {"constant": "Constant", "random-range": "Random Range"},
                defaults.get('flux_guidance_mode', 'constant'),
                cli_key='--flux_guidance_mode',
            )
            grid_guidance = ui.element('div').classes('form-grid-2')
            flux_guidance_value_input = create_number_field(
                grid_guidance,
                'flux_guidance_value',
                'Flux Guidance Value',
                defaults.get('flux_guidance_value', 1.0),
                min_value=0,
                step=0.1,
                cli_key='--flux_guidance_value',
            )
            ema_decay_input = create_number_field(
                grid_guidance,
                'ema_decay',
                'EMA Decay',
                defaults.get('ema_decay', 0.995),
                min_value=0,
                max_value=1,
                step=0.001,
                cli_key='--ema_decay',
            )
            scheduler_select = create_select_field(
                ui.element('div').classes('form-grid-1'),
                'lr_scheduler',
                'LR Scheduler',
                _select_options(schedulers),
                default_scheduler,
                cli_key='--lr_scheduler',
            )
            scheduler_cycles_input = create_number_field(
                ui.element('div').classes('form-grid-1'),
                'lr_scheduler_num_cycles',
                'LR Scheduler Cycles',
                defaults.get('lr_scheduler_num_cycles', 1),
                min_value=0,
                step=1,
                cli_key='--lr_scheduler_num_cycles',
                transform=lambda value: str(int(value)) if value is not None else None,
            )

    # Advanced tab content
    with tab_panels['advanced']:
        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-tools"></i> Additional Arguments</h5>')
            ui.html('<p class="text-sm text-slate-400">Enter extra CLI arguments, one per line (e.g. --adam_beta1=0.9).</p>')
            additional_args_input = ui.textarea().classes('legacy-input').props(
                'rows=6',
                'id=additional_args',
            )

    # Data tab content
    plan_controller = PlanController()
    with tab_panels['data']:
        with ui.element('div').classes('config-card'):
            ui.html('<h5 class="card-title"><i class="fas fa-layer-group"></i> Dataloader Configuration</h5>')
            dataset_summary = ui.column().classes('gap-2')
            if plan_controller.datasets:
                for dataset in plan_controller.datasets:
                    with dataset_summary:
                        card = ui.element('div').classes('dataset-summary-card')
                        with card:
                            dataset_id = html.escape(str(dataset.get('id', 'dataset')))
                            dataset_type = html.escape(str(dataset.get('dataset_type', 'unknown')))
                            dataset_backend = html.escape(str(dataset.get('type', 'unknown')))
                            ui.html(
                                f"<div class='dataset-header'><span class='dataset-id'>{dataset_id}</span>"
                                f"<span class='dataset-meta'>{dataset_backend} • {dataset_type}</span></div>"
                            )
                            for key, value in build_dataset_rows(dataset):
                                key_html = html.escape(str(key))
                                value_html = html.escape(str(value))
                                ui.html(f"<div class='dataset-row'><span>{key_html}</span><span>{value_html}</span></div>")
            else:
                dataset_summary.clear()
                with dataset_summary:
                    ui.html('<p class="text-sm text-slate-400">No datasets configured yet. Use the builder to add one.</p>')
            with ui.element('div').classes('mb-3 mt-3 d-flex gap-2'):
                builder_button = ui.element('button').classes('btn btn-sm btn-primary').props('type=button')
                builder_button.on('click', lambda _: ui.link('/datasets'))
                with builder_button:
                    ui.html('<i class="fas fa-database"></i> Open Dataset Builder')
                reload_plan_button = ui.element('button').classes('btn btn-sm btn-secondary').props('type=button id=reloadPlanBtn')
                with reload_plan_button:
                    ui.html('<i class="fas fa-sync-alt"></i> Reload Plan')
            dataloader_input = ui.textarea(value=_dataset_plan_json(plan_controller)).classes('legacy-input').props(
                'rows=12',
                'id=dataloader_config',
            )
            webhooks_input = ui.textarea(value=_default_webhooks("http://localhost:8001/callback")).classes('legacy-input mt-3').props(
                'rows=6',
                'id=webhooks_config',
            )

    # Training progress and event display sections
    with main:
        with ui.element('div').classes('training-progress-section'):
            ui.html('<h5 class="card-title"><i class="fas fa-chart-line"></i> Training Progress</h5>')
            training_progress = ui.element('div').props('id=trainingProgress')
            progress_bars = ui.element('div').props('id=progressBars')
        with ui.element('div').classes('event-display'):
            ui.html('<h5 class="card-title"><i class="fas fa-terminal"></i> Training Events<div id="eventStatus"></div></h5>')
            event_container = ui.column().classes('event-list gap-2').props('id=eventList')

    toast_container = ui.element('div').classes('toast-container position-fixed bottom-0 end-0 p-3').props(
        'id=simpletuner-toasts'
    )

    tab_wrapper.update()

    ui.timer(0.1, lambda: asyncio.create_task(ui.run_javascript('window.__simpletunerInitSidebar && window.__simpletunerInitSidebar();')), once=True)

    event_items: List[Any] = []
    status_items: List[Any] = []

    async def show_toast(message: str, level: str = 'info') -> None:
        variant_map = {
            'success': 'success',
            'error': 'danger',
            'warning': 'warning',
            'info': 'info',
        }
        variant = variant_map.get(level, 'info')
        await ui.run_javascript(
            """
            (function(message, variant){
                const container = document.getElementById('simpletuner-toasts');
                if (!container || typeof bootstrap === 'undefined') { return; }
                const toastEl = document.createElement('div');
                toastEl.className = `toast text-white bg-${variant}`;
                toastEl.setAttribute('role', 'alert');
                toastEl.setAttribute('aria-live', 'assertive');
                toastEl.setAttribute('aria-atomic', 'true');
                toastEl.innerHTML = `<div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>`;
                container.appendChild(toastEl);
                const toast = new bootstrap.Toast(toastEl, { delay: 5000 });
                toast.show();
                toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
            })(arguments[0], arguments[1]);
            """,
            message,
            variant,
        )

    def add_status(message: str, level: str = "info") -> None:
        colors = {
            "info": "status-message status-info",
            "success": "status-message status-success",
            "error": "status-message status-error",
            "warning": "status-message status-warning",
        }
        classes = colors.get(level, colors['info'])
        escaped_message = html.escape(message)
        with status_container:
            item = ui.html(f'<div class="{classes}">{escaped_message}</div>')
            status_items.append(item)
        while len(status_items) > 5:
            old = status_items.pop(0)
            old.remove()

    async def set_button_loading(button_id: str, loading: bool, label_html: str) -> None:
        await ui.run_javascript(
            """
            (function(btnId, loading, labelHtml){
                const btn = document.getElementById(btnId);
                if (!btn) return;
                if (loading) {
                    btn.setAttribute('disabled', 'disabled');
                    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${labelHtml}`;
                } else {
                    btn.removeAttribute('disabled');
                    btn.innerHTML = labelHtml;
                }
            })(arguments[0], arguments[1], arguments[2]);
            """,
            button_id,
            loading,
            label_html,
        )

    async def set_connection_status(connected: bool, message: Optional[str] = None) -> None:
        text = message if message else ("Connected" if connected else "Disconnected")
        color_class = "text-success" if connected else "text-warning"
        await ui.run_javascript(
            """
            (function(content, colorClass){
                const target = document.getElementById('connectionStatus');
                if (!target) return;
                target.className = `connection-status ${colorClass}`;
                target.textContent = content;
            })(arguments[0], arguments[1]);
            """,
            text,
            color_class,
        )

    async def build_payload() -> Optional[Dict[str, Any]]:
        job_id = (job_id_input.value or "").strip()
        if not job_id:
            await show_toast("Job ID is required", "error")
            add_status("Job ID is required", "error")
            set_tab('basic')
            return None
        trainer_config: Dict[str, Any] = {}
        for key, getter in trainer_inputs.items():
            value = getter()
            if value is not None and value != "":
                trainer_config[key] = value
        extra_lines = (additional_args_input.value or "").splitlines()
        for line in extra_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith('--'):
                stripped = '--' + stripped
            parts = stripped.split('=', 1)
            if len(parts) == 1:
                trainer_config[parts[0]] = 'true'
            else:
                trainer_config[parts[0]] = parts[1]
        try:
            dataloader_config = json.loads(dataloader_input.value or '[]')
            if not isinstance(dataloader_config, list):
                raise ValueError('dataloader_config must be a JSON list')
        except Exception as exc:
            await show_toast(f"Invalid dataloader JSON: {exc}", "error")
            add_status(f"Invalid dataloader JSON: {exc}", "error")
            set_tab('data')
            return None
        try:
            webhooks_config = json.loads(webhooks_input.value or '{}')
            if not isinstance(webhooks_config, dict):
                raise ValueError('webhooks_config must be a JSON object')
        except Exception as exc:
            await show_toast(f"Invalid webhook JSON: {exc}", "error")
            add_status(f"Invalid webhook JSON: {exc}", "error")
            set_tab('data')
            return None
        return {
            "trainer_config": trainer_config,
            "dataloader_config": dataloader_config,
            "webhooks_config": webhooks_config,
            "job_id": job_id,
        }

    async def submit(endpoint: str, button_id: str, label_html: str, success_message: str) -> None:
        payload = await build_payload()
        if payload is None:
            return
        await set_button_loading(button_id, True, label_html)
        try:
            result = await fetch_json(endpoint, "POST", payload)
        except Exception as exc:
            await show_toast(f"Request failed: {exc}", "error")
            add_status(f"{endpoint} failed: {exc}", "error")
            await set_button_loading(button_id, False, label_html)
            return
        await set_button_loading(button_id, False, label_html)
        if not result.get("ok", False):
            status_code = result.get("status")
            await show_toast(f"Request failed (status {status_code})", "error")
            add_status(f"{endpoint} failed (status {status_code})", "error")
            return
        data = result.get("data") or {}
        detail = data.get("detail") if isinstance(data, dict) else None
        if detail:
            await show_toast(str(detail), "error")
            add_status(str(detail), "error")
            return
        response_message = data.get("result") if isinstance(data, dict) else success_message
        await show_toast(response_message, "success")
        add_status(response_message, "success")

    validate_button.on(
        'click',
        lambda _: asyncio.create_task(
            submit('/training/configuration/check', 'validateBtn', '<i class="fas fa-check-circle"></i> <span class="btn-text">Validate</span>', 'Configuration validated')
        ),
    )
    run_button.on(
        'click',
        lambda _: asyncio.create_task(
            submit('/training/configuration/run', 'runBtn', '<i class="fas fa-play"></i> <span>Start Training</span>', 'Training started')
        ),
    )

    async def cancel_job(_: events.ClickEventArguments) -> None:
        job_id = (job_id_input.value or "").strip()
        if not job_id:
            await show_toast("Provide a job ID to cancel", "warning")
            add_status("Provide a job ID to cancel", "warning")
            set_tab('basic')
            return
        await set_button_loading('cancelBtn', True, '<i class="fas fa-stop"></i> <span>Cancel</span>')
        try:
            result = await fetch_json('/training/cancel', 'POST', {"job_id": job_id})
        except Exception as exc:
            await show_toast(f"Cancel failed: {exc}", "error")
            add_status(f"Cancel request failed: {exc}", "error")
            await set_button_loading('cancelBtn', False, '<i class="fas fa-stop"></i> <span>Cancel</span>')
            return
        await set_button_loading('cancelBtn', False, '<i class="fas fa-stop"></i> <span>Cancel</span>')
        if not result.get("ok", False):
            await show_toast('Cancel failed', 'error')
            add_status('Cancel request failed', 'error')
            return
        data = result.get("data") or {}
        message = data.get("result") if isinstance(data, dict) else "Cancellation requested"
        await show_toast(message, "warning")
        add_status(message, "warning")

    cancel_button.on('click', cancel_job)

    async def load_model_flavours(family: Optional[str]) -> None:
        options: Dict[str, str] = {"": "Default"}
        if family:
            try:
                result = await fetch_json(f"/models/{family}/flavours")
                if result.get("ok", False):
                    data = result.get("data") or {}
                    flavours = data.get("flavours") or []
                else:
                    flavours = list(get_model_flavour_choices(family) or [])
            except Exception:
                flavours = list(get_model_flavour_choices(family) or [])
            for flavour in flavours:
                options[flavour] = flavour
        model_flavour_select.options = options
        model_flavour_select.value = ""
        model_flavour_select.update()

    await load_model_flavours(default_family)

    model_family_select.on_value_change(lambda e: asyncio.create_task(load_model_flavours(e.value)))

    async def reload_plan_from_api() -> None:
        try:
            result = await fetch_json('/api/datasets/plan')
        except Exception as exc:
            await show_toast(f'Failed to reach dataset plan API: {exc}', 'error')
            add_status(f'Dataset plan reload failed: {exc}', 'error')
            return
        if not result.get("ok", False):
            status_code = result.get("status")
            await show_toast(f"Plan request failed (status {status_code})", "error")
            add_status(f"Plan request failed (status {status_code})", "error")
            return
        data = result.get("data") or {}
        datasets = data.get("datasets") or []
        dataloader_input.value = json.dumps(datasets, indent=4)
        dataloader_input.update()
        plan_controller.apply_plan_response(data)
        dataset_summary.clear()
        if plan_controller.datasets:
            for dataset in plan_controller.datasets:
                with dataset_summary:
                    card = ui.element('div').classes('dataset-summary-card')
                    with card:
                        dataset_id = html.escape(str(dataset.get('id', 'dataset')))
                        dataset_type = html.escape(str(dataset.get('dataset_type', 'unknown')))
                        dataset_backend = html.escape(str(dataset.get('type', 'unknown')))
                        ui.html(
                            f"<div class='dataset-header'><span class='dataset-id'>{dataset_id}</span>"
                            f"<span class='dataset-meta'>{dataset_backend} • {dataset_type}</span></div>"
                        )
                        for key, value in build_dataset_rows(dataset):
                            key_html = html.escape(str(key))
                            value_html = html.escape(str(value))
                            ui.html(f"<div class='dataset-row'><span>{key_html}</span><span>{value_html}</span></div>")
        else:
            dataset_summary.clear()
            with dataset_summary:
                ui.html('<p class="text-sm text-slate-400">No datasets configured yet. Use the builder to add one.</p>')
        await show_toast('Dataset plan reloaded', 'success')
        add_status('Dataset plan reloaded', 'success')

    reload_plan_button.on('click', lambda _: asyncio.create_task(reload_plan_from_api()))

    last_event_index = 0
    polling_lock = asyncio.Lock()

    async def append_events(events_payload: List[Dict[str, Any]]) -> None:
        if not events_payload:
            return
        severity_class = {
            'error': 'event-item-error',
            'fatal_error': 'event-item-error',
            'exit': 'event-item-exit',
            'train': 'event-item-train',
            'info': 'event-item-info',
        }
        for entry in events_payload:
            message_type = entry.get('message_type', 'info')
            message = entry.get('message') or json.dumps(entry, indent=2)
            css = severity_class.get(message_type, 'event-item-default')
            with event_container:
                item = ui.element('div').classes(f'event-item {css}')
                item.text(f"[{message_type}] {message}")
                event_items.append(item)
        while len(event_items) > 200:
            old = event_items.pop(0)
            old.remove()
        await ui.run_javascript(
            """
            (function(){
                const container = document.getElementById('eventList');
                if (container) { container.scrollTop = container.scrollHeight; }
            })();
            """
        )

    async def poll_events() -> None:
        nonlocal last_event_index
        if polling_lock.locked():
            return
        async with polling_lock:
            try:
                result = await fetch_json(f"/broadcast?last_event_index={last_event_index}")
            except Exception as exc:
                await set_connection_status(False, f"Disconnected: {exc}")
                return
            if not result.get("ok", False):
                await set_connection_status(False, f"HTTP {result.get('status')}")
                return
            data = result.get("data") or {}
            events_payload = data.get('events') or []
            await append_events(events_payload)
            next_index = data.get('next_index')
            if isinstance(next_index, int):
                last_event_index = next_index
            await set_connection_status(True)

    poll_timer = ui.timer(1.0, lambda: asyncio.create_task(poll_events()))

    client = ui.get_client()

    @client.on_disconnect
    def _cleanup() -> None:
        poll_timer.cancel()
