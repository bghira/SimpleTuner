# simpletuner_sdk.interface.py
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from simpletuner.helpers.configuration.cmd_args import get_default_config as get_cmd_default_config

# Import the proper configuration modules
from simpletuner.helpers.models.all import get_all_model_flavours, model_families
from simpletuner.helpers.training.optimizer_param import optimizer_choices


class WebInterface:
    def __init__(self):
        self.router = APIRouter(prefix="/web")
        self.templates = Jinja2Templates(directory="templates")
        self.setup_routes()
        self.setup_template_functions()

    def setup_routes(self):
        """Setup all web routes"""
        # Main trainer interface
        self.router.add_api_route("/", self.redirect_to_trainer, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route(
            "/training",
            self.get_trainer_page,
            methods=["GET"],
            response_class=HTMLResponse,
        )

        # Additional pages if needed
        self.router.add_api_route(
            "/dashboard",
            self.get_dashboard,
            methods=["GET"],
            response_class=HTMLResponse,
        )
        self.router.add_api_route("/settings", self.get_settings, methods=["GET"], response_class=HTMLResponse)

    def setup_template_functions(self):
        """Add custom functions to Jinja2 environment"""

        # Add url_for function for static files
        def url_for(endpoint: str, **values):
            if endpoint == "static":
                filename = values.get("filename", "")
                return f"/static/{filename}"
            return f"/{endpoint}"

        # Add custom filters or functions
        self.templates.env.globals.update(
            url_for=url_for,
            render_field=self.render_field,
            render_select=self.render_select,
            render_checkbox=self.render_checkbox,
        )

    def render_field(
        self,
        name: str,
        label: str,
        type: str = "text",
        value: str = "",
        placeholder: str = "",
        extra_classes: str = "",
    ) -> str:
        """Render a form field"""
        return f"""
        <label for="{name}" class="form-label">{label}</label>
        <input type="{type}" class="form-control {extra_classes}"
               id="{name}" name="{name}" value="{value}"
               {'placeholder="' + placeholder + '"' if placeholder else ''}>
        """

    def render_select(
        self,
        name: str,
        label: str,
        options: List[Dict[str, str]],
        selected: str = None,
        extra_classes: str = "",
    ) -> str:
        """Render a select field"""
        options_html = ""
        for option in options:
            selected_attr = "selected" if option.get("value") == selected else ""
            options_html += f'<option value="{option["value"]}" {selected_attr}>{option["label"]}</option>\n'

        return f"""
        <label for="{name}" class="form-label">{label}</label>
        <select class="form-select {extra_classes}" id="{name}" name="{name}">
            {options_html}
        </select>
        """

    def render_checkbox(self, name: str, label: str, checked: bool = False, extra_classes: str = "") -> str:
        """Render a checkbox field"""
        checked_attr = "checked" if checked else ""
        return f"""
        <div class="form-check {extra_classes}">
            <input type="checkbox" class="form-check-input"
                   id="{name}" name="{name}" {checked_attr}>
            <label class="form-check-label" for="{name}">{label}</label>
        </div>
        """

    def get_model_types(self) -> List[Dict[str, Any]]:
        """Get available model types dynamically"""
        # Standard model types that are always available
        model_types = [
            {"value": "full", "label": "Full Fine-tune"},
            {"value": "lora", "label": "LoRA", "selected": True},
        ]

        # Add model-specific types if available
        # This could be extended based on the selected model family
        return model_types

    def get_model_families_options(self) -> List[Dict[str, Any]]:
        """Get model families from the actual configuration"""
        options = []
        for family_key, family_info in model_families.items():
            label = family_info.NAME
            is_default = family_key == "flux"  # Set default as needed
            options.append({"value": family_key, "label": label, "selected": is_default})
        return options

    def get_model_paths_options(self) -> List[Dict[str, Any]]:
        """Get model paths dynamically from model classes"""
        all_paths = []

        for family_key, model_class in model_families.items():
            # Check if the model class has HUGGINGFACE_PATHS attribute
            if hasattr(model_class, "HUGGINGFACE_PATHS"):
                huggingface_paths = model_class.HUGGINGFACE_PATHS
                default_flavour = getattr(model_class, "DEFAULT_MODEL_FLAVOUR", None)

                # Get display name for the model family
                display_name = getattr(model_class, "display_name", family_key.replace("_", " ").title())

                # Add each path from the model's HUGGINGFACE_PATHS
                for flavour, path in huggingface_paths.items():
                    # Create a descriptive label
                    if len(huggingface_paths) > 1:
                        label = f"{display_name} {flavour}"
                    else:
                        label = display_name

                    option = {"value": path, "label": label, "family": family_key}

                    # Mark as selected if it's the default flavour for this model
                    if default_flavour and flavour == default_flavour:
                        option["selected"] = True

                    all_paths.append(option)

        # Sort by label for better UX
        return sorted(all_paths, key=lambda x: x["label"])

    def get_optimizer_options(self) -> List[Dict[str, Any]]:
        """Get optimisers from the actual optimizer_choices configuration"""
        # Return as grouped structure for optgroup rendering
        categories = {}

        for opt_key, opt_info in optimizer_choices.items():
            # Create a human-readable label
            label = opt_key

            # Categorize optimisers (using British spelling in UI)
            if opt_key.startswith("ao-"):
                category = "TorchAO (Quantised)"
                label = opt_key.replace("ao-", "").upper()
            elif opt_key.startswith("bnb-"):
                category = "BitsAndBytes (8-bit)"
                label = opt_key.replace("bnb-", "").upper()
            elif opt_key.startswith("optimi-"):
                category = "Optimi Library"
                label = opt_key.replace("optimi-", "").title()
            elif "schedulefree" in opt_key:
                category = "Schedule-Free"
                label = opt_key.replace("_schedulefree", "").title() + " (Schedule-Free)"
            elif opt_key in ["prodigy", "soap", "adalomo", "adopt"]:
                category = "Advanced Optimisers"
                label = opt_key.upper()
            else:
                category = "Standard Optimisers"
                label = opt_key.replace("_", " ").replace("-", " ").title()

            # Add precision info if relevant
            precision = opt_info.get("precision", "any")
            if precision != "any" and precision not in label.lower():
                label += f" ({precision})"

            if category not in categories:
                categories[category] = []

            categories[category].append({"value": opt_key, "label": label})

        # Return grouped structure for proper optgroup rendering
        result = []
        for category in sorted(categories.keys()):
            result.append({"group": category, "options": sorted(categories[category], key=lambda x: x["label"])})

        return result

    def get_lr_scheduler_options(self) -> List[Dict[str, Any]]:
        """Get learning rate schedulers"""
        # If there's a function to get LR scheduler choices, use it
        # Otherwise, use the standard options
        schedulers = [
            {"value": "constant", "label": "Constant"},
            {"value": "constant_with_warmup", "label": "Constant with Warmup"},
            {"value": "cosine", "label": "Cosine"},
            {"value": "cosine_with_restarts", "label": "Cosine with Restarts"},
            {"value": "polynomial", "label": "Polynomial", "selected": True},
            {"value": "linear", "label": "Linear"},
            {"value": "sine", "label": "Sine"},
        ]
        return schedulers

    async def redirect_to_trainer(self, request: Request):
        """Redirect root web path to the primary trainer page."""
        return RedirectResponse(url="/web/trainer", status_code=307)

    async def get_trainer_page(self, request: Request):
        """Serve the HTMX-based trainer interface for legacy links."""
        return self.templates.TemplateResponse(
            request,
            "trainer_htmx.html",
            {
                "request": request,
                "title": "SimpleTuner Training Studio",
            },
        )

    def get_trainer_context(self, request: Request) -> Dict[str, Any]:
        """Prepare context data for trainer template with dynamic configuration"""
        return {
            "request": request,
            "model_types": self.get_model_types(),
            "model_families": self.get_model_families_options(),
            "model_paths": self.get_model_paths_options(),
            "optimizers": self.get_optimizer_options(),
            "lr_schedulers": self.get_lr_scheduler_options(),
            "default_config": self.get_default_config(),
            # Add model flavours if needed
            "model_flavours": (get_all_model_flavours() if hasattr(self, "get_all_model_flavours") else []),
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values from actual cmd_args"""
        # Get the real defaults from cmd_args
        defaults = get_cmd_default_config()

        # Override a few UI-specific defaults
        import random

        defaults["job_id"] = f"training_run_{random.randint(1000, 9999)}"
        defaults["data_backend_config"] = "config/multidatabackend.json"

        # Ensure all template-expected values have sensible defaults
        # These are fallbacks if cmd_args doesn't provide them
        template_defaults = {
            "model_type": "full",  # Default to full training
            "resume_from_checkpoint": "latest",
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 4e-7,
            "num_train_epochs": 10,
            "max_train_steps": 0,
            "lr_warmup_steps": 100,
            "snr_gamma": 5.0,
            "checkpointing_steps": 500,
            "validation_steps": 100,
            "seed": 42,
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "noise_offset": 0.1,
            "input_perturbation": 0.0,
            "flow_sigmoid_scale": 1.0,
            "flow_schedule_shift": 3.0,
            "flux_guidance_value": 1.0,
            "ema_decay": 0.995,
        }

        # Merge with template defaults (cmd_args takes precedence)
        for key, value in template_defaults.items():
            if key not in defaults or defaults[key] is None:
                defaults[key] = value

        return defaults

    async def get_dashboard(self, request: Request):
        """Redirect dashboard requests to the NiceGUI overview."""
        return RedirectResponse(url="/web/trainer", status_code=307)

    async def get_settings(self, request: Request):
        """Redirect settings requests to the NiceGUI page."""
        return RedirectResponse(url="/web/settings", status_code=307)

    def get_fallback_response(self, message: str = None) -> HTMLResponse:
        """Get fallback HTML response"""
        default_message = message or "Welcome to SimpleTuner. This installation does not include a compatible web interface."

        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SimpleTuner</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                        background-color: #f4f4f9;
                        margin: 0;
                        padding: 20px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                    }}
                    .container {{
                        max-width: 600px;
                        background: white;
                        padding: 2rem;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                        text-align: center;
                    }}
                    h1 {{
                        color: #333;
                        margin-bottom: 1rem;
                    }}
                    p {{
                        color: #666;
                        line-height: 1.6;
                    }}
                    .btn {{
                        display: inline-block;
                        margin-top: 1rem;
                        padding: 0.75rem 1.5rem;
                        background-color: #007bff;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                        transition: background-color 0.3s;
                    }}
                    .btn:hover {{
                        background-color: #0056b3;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>SimpleTuner</h1>
                    <p>{default_message}</p>
                    <a href="/docs" class="btn">View API Documentation</a>
                </div>
            </body>
            </html>
            """,
            status_code=200,
        )

    def get_error_response(self, error_message: str) -> HTMLResponse:
        """Get error HTML response"""
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SimpleTuner - Error</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                        background-color: #f4f4f9;
                        margin: 0;
                        padding: 20px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                    }}
                    .error-container {{
                        max-width: 600px;
                        background: white;
                        padding: 2rem;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        border-left: 4px solid #dc3545;
                    }}
                    h1 {{
                        color: #dc3545;
                        margin-bottom: 1rem;
                    }}
                    .error-message {{
                        color: #666;
                        line-height: 1.6;
                        background: #f8f9fa;
                        padding: 1rem;
                        border-radius: 4px;
                        margin: 1rem 0;
                        font-family: monospace;
                        font-size: 0.9rem;
                    }}
                </style>
            </head>
            <body>
                <div class="error-container">
                    <h1>Error Loading Page</h1>
                    <div class="error-message">{error_message}</div>
                    <p>Please check your templates directory and ensure all required files are present.</p>
                </div>
            </body>
            </html>
            """,
            status_code=500,
        )
