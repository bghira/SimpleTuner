"""Service for handling tab rendering logic.

This service centralizes all tab-related business logic, including
field organization, template rendering, and tab-specific customizations.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..services.webui_state import WebUIStateStore
from ..utils.assets import get_asset_version
from .custom_section_service import CUSTOM_SECTION_SERVICE
from .dataset_service import build_data_backend_choices
from .hardware_service import detect_gpu_inventory
from .prompt_library_service import PromptLibraryService

logger = logging.getLogger(__name__)


class TabType(str, Enum):
    """Available tab types in the trainer interface."""

    BASIC = "basic"
    HARDWARE = "hardware"
    MODEL = "model"
    TRAINING = "training"
    DATASETS = "datasets"
    ENVIRONMENTS = "environments"
    GIT_MIRROR = "git_mirror"
    VALIDATION = "validation"
    PUBLISHING = "publishing"
    CHECKPOINTS = "checkpoints"
    JOB_QUEUE = "job_queue"
    CLOUD = "cloud"
    UI_SETTINGS = "ui_settings"
    ADMIN = "admin"
    USERS = "users"  # Alias for admin (URL uses #users)
    AUDIT = "audit"
    METRICS = "metrics"
    NOTIFICATIONS = "notifications"
    QUOTAS = "quotas"


@dataclass
class TabConfig:
    """Configuration for a single tab."""

    id: str
    title: str
    icon: str
    template: str
    description: Optional[str] = None
    extra_context_handler: Optional[callable] = None


class TabService:
    """Service for managing tab rendering and field organization."""

    def __init__(self, templates: Jinja2Templates):
        """Initialize tab service.

        Args:
            templates: Jinja2 templates instance
        """
        self.templates = templates
        self._tab_configs = self._initialize_tab_configs()

    def _initialize_tab_configs(self) -> Dict[str, TabConfig]:
        """Initialize tab configurations."""
        return {
            TabType.BASIC: TabConfig(
                id="basic-config",
                title="Basic",
                icon="fas fa-cog",
                template="form_tab.html",
                description="Essential settings to get started",
                extra_context_handler=None,
            ),
            TabType.HARDWARE: TabConfig(
                id="hardware-config",
                title="Hardware",
                icon="fas fa-microchip",
                template="form_tab.html",
                description="Accelerate launch, distributed compute, and worker tuning",
                extra_context_handler=self._hardware_tab_context,
            ),
            TabType.MODEL: TabConfig(
                id="model-config",
                title="Model",
                icon="fas fa-brain",
                template="form_tab.html",
                description="Model architecture and settings",
                extra_context_handler=None,
            ),
            TabType.TRAINING: TabConfig(
                id="training-config",
                title="Training",
                icon="fas fa-graduation-cap",
                template="form_tab.html",
                description="Training parameters and optimization",
                extra_context_handler=None,
            ),
            TabType.DATASETS: TabConfig(
                id="datasets-config",
                title="Dataset",
                icon="fas fa-database",
                template="datasets_tab.html",
                description="Dataset loading and preprocessing",
                extra_context_handler=self._datasets_tab_context,
            ),
            TabType.VALIDATION: TabConfig(
                id="validation-config",
                title="Validation",
                icon="fas fa-check-circle",
                template="form_tab.html",
                description="Configure visual validation during training",
                extra_context_handler=self._validation_tab_context,
            ),
            TabType.PUBLISHING: TabConfig(
                id="publishing",
                title="Publishing",
                icon="fas fa-cloud-upload-alt",
                template="form_tab.html",
                description="Configure HuggingFace Hub publishing",
                extra_context_handler=self._publishing_tab_context,
            ),
            TabType.CHECKPOINTS: TabConfig(
                id="checkpoints",
                title="Checkpoints",
                icon="fas fa-save",
                template="checkpoints_tab.html",
                description="Browse and manage training checkpoints",
                extra_context_handler=self._checkpoints_tab_context,
            ),
            TabType.JOB_QUEUE: TabConfig(
                id="job-queue",
                title="Job Queue",
                icon="fas fa-tasks",
                template="job_queue_tab.html",
                description="View and manage all training jobs",
                extra_context_handler=None,
            ),
            TabType.ENVIRONMENTS: TabConfig(
                id="environments-config",
                title="Environment",
                icon="fas fa-server",
                template="environments_tab.html",
                description="Environment and compute settings",
                extra_context_handler=self._environments_tab_context,
            ),
            TabType.GIT_MIRROR: TabConfig(
                id="git-mirror",
                title="Git Mirror",
                icon="fas fa-code-branch",
                template="git_mirror_tab.html",
                description="Versioning and sync for configuration files",
                extra_context_handler=None,
            ),
            TabType.UI_SETTINGS: TabConfig(
                id="ui-settings",
                title="UI Settings",
                icon="fas fa-sliders",
                template="ui_settings_tab.html",
                description="Adjust WebUI preferences and behaviour",
                extra_context_handler=self._ui_settings_tab_context,
            ),
            TabType.CLOUD: TabConfig(
                id="cloud-dashboard",
                title="Cloud",
                icon="fas fa-cloud",
                template="cloud_tab.html",
                description="Manage local and cloud training jobs",
                extra_context_handler=self._cloud_tab_context,
            ),
            TabType.ADMIN: TabConfig(
                id="admin-panel",
                title="Administration",
                icon="fas fa-users-cog",
                template="admin_tab.html",
                description="User management and access control",
                extra_context_handler=None,
            ),
            TabType.AUDIT: TabConfig(
                id="audit-log",
                title="Audit Log",
                icon="fas fa-scroll",
                template="audit_tab.html",
                description="Security events and system activity",
                extra_context_handler=None,
            ),
            TabType.METRICS: TabConfig(
                id="metrics-panel",
                title="Metrics",
                icon="fas fa-chart-line",
                template="metrics_tab.html",
                description="Prometheus metrics export configuration",
                extra_context_handler=self._metrics_tab_context,
            ),
            TabType.USERS: TabConfig(
                id="admin-panel",
                title="Manage Users",
                icon="fas fa-users-cog",
                template="admin_tab.html",
                description="User management and access control",
                extra_context_handler=None,
            ),
            TabType.NOTIFICATIONS: TabConfig(
                id="notifications-panel",
                title="Notifications",
                icon="fas fa-bell",
                template="notifications_tab.html",
                description="Notification channels and event routing",
                extra_context_handler=None,
            ),
            TabType.QUOTAS: TabConfig(
                id="quotas-panel",
                title="Quotas",
                icon="fas fa-tachometer-alt",
                template="quotas_tab.html",
                description="Spending limits and job quotas",
                extra_context_handler=None,
            ),
        }

    def get_tab_config(self, tab_name: str) -> TabConfig:
        """Get configuration for a specific tab.

        Args:
            tab_name: Name of the tab

        Returns:
            TabConfig instance

        Raises:
            HTTPException: If tab not found
        """
        try:
            tab_type = TabType(tab_name)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        if tab_type == TabType.GIT_MIRROR and not self._git_mirror_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        if tab_type == TabType.CLOUD and not self._cloud_tab_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        if tab_type == TabType.ADMIN and not self._admin_tab_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        # These tabs follow the same access rules as admin tab
        if (
            tab_type in (TabType.USERS, TabType.AUDIT, TabType.NOTIFICATIONS, TabType.QUOTAS)
            and not self._admin_tab_enabled()
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        if tab_type == TabType.METRICS and not self._metrics_tab_enabled():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        return self._tab_configs.get(tab_type)

    async def render_tab(
        self,
        request: Request,
        tab_name: str,
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any],
        sections: Optional[List[Dict[str, Any]]] = None,
        raw_config: Optional[Dict[str, Any]] = None,
        webui_defaults: Optional[Dict[str, Any]] = None,
    ) -> HTMLResponse:
        """Render a tab with the provided fields and configuration."""

        tab_config = self.get_tab_config(tab_name)

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(value, (int, float)):
                return value != 0
            return False

        raw_config = raw_config or {}
        danger_mode_enabled = (
            _coerce_bool(config_values.get("i_know_what_i_am_doing"))
            or _coerce_bool(config_values.get("--i_know_what_i_am_doing"))
            or _coerce_bool(raw_config.get("i_know_what_i_am_doing"))
            or _coerce_bool(raw_config.get("--i_know_what_i_am_doing"))
        )

        asset_version = (
            webui_defaults.get("asset_version") if isinstance(webui_defaults, dict) else None
        ) or get_asset_version()

        context = {
            "request": request,
            "tab_name": tab_name,
            "tab_config": {
                "id": tab_config.id,
                "title": tab_config.title,
                "icon": tab_config.icon,
                "description": tab_config.description,
            },
            "section": {
                "id": tab_config.id,
                "title": tab_config.title,
                "icon": tab_config.icon,
                "description": tab_config.description,
            },
            "fields": fields,
            "config_values": config_values,
            "raw_config": raw_config,
            "webui_defaults": webui_defaults or {},
            "danger_mode_enabled": danger_mode_enabled,
            "asset_version": asset_version,
        }

        # Add sections if provided
        if sections:
            context["sections"] = sections

        # Apply tab-specific context modifications
        if tab_config.extra_context_handler:
            context = tab_config.extra_context_handler(context, fields, config_values)

        # Render template
        return self.templates.TemplateResponse(request=context["request"], name=tab_config.template, context=context)

    # Tab-specific context handlers
    def _validation_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure validation tab renders its configuration fields."""
        if not context.get("sections"):
            context["sections"] = [
                {
                    "id": context["tab_config"]["id"],
                    "title": context["tab_config"]["title"],
                    "icon": context["tab_config"].get("icon"),
                    "description": context["tab_config"].get("description"),
                }
            ]
        try:
            libraries = PromptLibraryService().list_libraries()
            context["prompt_libraries"] = [asdict(record) for record in libraries]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to load prompt libraries for validation tab: %s", exc)
            context["prompt_libraries"] = []
        return context

    def _publishing_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for publishing tab using standard form template."""
        field_sections = context.get("sections", [])
        context["sections"] = CUSTOM_SECTION_SERVICE.merge_custom_sections_with_field_sections(
            tab="publishing", field_sections=field_sections
        )
        return context

    def _checkpoints_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for checkpoints tab."""
        context["checkpoints_total_limit"] = config_values.get("checkpoints_total_limit", 10)
        context["output_dir"] = config_values.get("output_dir", "output")
        return context

    def _ui_settings_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for UI settings tab."""
        store = WebUIStateStore()
        bundle = store.get_defaults_bundle()

        context["ui_settings"] = {
            "defaults": bundle["resolved"],
            "raw_defaults": bundle["raw"],
            "fallbacks": bundle["fallbacks"],
            "themes": [
                {"value": "dark", "label": "Dark", "description": "Classic SimpleTuner palette"},
                {"value": "tron", "label": "Tron Prototype", "description": "Experimental neon styling"},
            ],
            "event_interval_options": [3, 5, 10, 15, 30, 60],
        }
        return context

    def _hardware_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust hardware tab based on detected capabilities."""

        inventory = detect_gpu_inventory()
        capabilities = inventory.get("capabilities", {}) or {}

        context["hardware_capabilities"] = capabilities
        context["gpu_inventory"] = inventory

        def _filter_by_prefix(prefixes: List[str]) -> None:
            if not prefixes:
                return
            fields[:] = [field for field in fields if not any(field.get("id", "").startswith(prefix) for prefix in prefixes)]

        if not capabilities.get("supports_deepspeed", False):
            _filter_by_prefix(["deepspeed_"])

        if not capabilities.get("supports_fsdp", False):
            _filter_by_prefix(["fsdp_"])

        sections = context.get("sections")
        if sections:
            filtered_sections: List[Dict[str, Any]] = []
            for section in sections:
                section_id = section.get("id")
                if section_id and any(field.get("section_id") == section_id for field in fields):
                    filtered_sections.append(section)
            context["sections"] = filtered_sections

        return context

    def _datasets_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for datasets tab."""
        # This tab uses a different template structure
        # Add data backend choices if available
        try:
            from .dataset_service import normalize_dataset_config_value

            context["data_backend_choices"] = build_data_backend_choices()

            # Get the current selected value from raw_config since config_values might be filtered
            raw_config = context.get("raw_config", {})
            field_value = (
                raw_config.get("data_backend_config")
                or raw_config.get("--data_backend_config")
                or config_values.get("data_backend_config")
                or config_values.get("--data_backend_config")
                or ""
            )

            # Normalize the value to match what the selector uses
            if field_value:
                field_value = normalize_dataset_config_value(field_value)

            # Find the selected option to populate the label
            selected_option = next(
                (opt for opt in context["data_backend_choices"] if opt.get("value") == field_value),
                None,
            )

            context["selected_value"] = field_value
            context["selected_env"] = selected_option.get("environment") if selected_option else ""
            context["selected_path"] = selected_option.get("path") if selected_option else ""

        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Could not build data backend choices: %s", exc)
            context["data_backend_choices"] = []
            context["selected_value"] = ""
            context["selected_env"] = ""
            context["selected_path"] = ""

        return context

    def _environments_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for environments tab."""
        # Add environment-specific context
        context["available_accelerators"] = ["cuda", "mps", "cpu"]
        try:
            libraries = PromptLibraryService().list_libraries()
            context["prompt_libraries"] = [asdict(record) for record in libraries]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to load prompt libraries for environments tab: %s", exc)
            context["prompt_libraries"] = []
        return context

    def _git_mirror_enabled(self) -> bool:
        try:
            defaults = WebUIStateStore().load_defaults()
            return bool(getattr(defaults, "git_mirror_enabled", False))
        except Exception as exc:
            logger.debug("Failed to evaluate git mirror enabled flag: %s", exc, exc_info=True)
            return False

    def _cloud_tab_enabled(self) -> bool:
        try:
            defaults = WebUIStateStore().load_defaults()
            return bool(getattr(defaults, "cloud_tab_enabled", True))
        except Exception as exc:
            logger.debug("Failed to evaluate cloud tab enabled flag: %s", exc, exc_info=True)
            return True

    def _admin_tab_enabled(self) -> bool:
        """Check if the admin tab should be shown.

        Returns True if:
        - admin_tab_enabled is True in settings, OR
        - Auth is in use (users configured or external auth providers)
        """
        try:
            defaults = WebUIStateStore().load_defaults()
            setting_enabled = bool(getattr(defaults, "admin_tab_enabled", True))

            # If setting says enabled, show it
            if setting_enabled:
                return True

            # If setting says disabled, check if auth is actually in use
            # If auth is in use, we MUST show the admin tab regardless of setting
            if self._is_auth_in_use():
                return True

            return False
        except Exception as exc:
            logger.debug("Failed to evaluate admin tab enabled flag: %s", exc, exc_info=True)
            return True

    def _is_auth_in_use(self) -> bool:
        """Check if user authentication is configured and in use."""
        try:
            from .cloud.auth.user_store import UserStore

            user_store = UserStore()
            users = user_store.list_users()
            # If there are any non-default users, auth is in use
            if len(users) > 1:
                return True
            # Check if any external auth providers are configured
            from .cloud.auth.external_auth import ExternalAuthManager

            manager = ExternalAuthManager()
            providers = manager.list_providers()
            if providers:
                return True
            return False
        except Exception as exc:
            logger.debug("Failed to check auth usage: %s", exc)
            return False

    def _cloud_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for cloud tab."""
        store = WebUIStateStore()
        defaults = store.load_defaults()
        context["cloud_settings"] = {
            "webhook_url": getattr(defaults, "cloud_webhook_url", None),
        }
        return context

    def _metrics_tab_enabled(self) -> bool:
        """Check if the metrics tab should be shown."""
        try:
            defaults = WebUIStateStore().load_defaults()
            return bool(getattr(defaults, "metrics_tab_enabled", True))
        except Exception as exc:
            logger.debug("Failed to evaluate metrics tab enabled flag: %s", exc, exc_info=True)
            return True

    def _metrics_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for metrics tab."""
        store = WebUIStateStore()
        defaults = store.load_defaults()
        context["metrics_settings"] = {
            "prometheus_enabled": getattr(defaults, "metrics_prometheus_enabled", False),
            "prometheus_categories": getattr(defaults, "metrics_prometheus_categories", ["jobs", "http"]),
            "tensorboard_enabled": getattr(defaults, "metrics_tensorboard_enabled", False),
            "dismissed_hints": getattr(defaults, "metrics_dismissed_hints", []),
        }
        return context

    def get_all_tabs(self) -> List[Dict[str, str]]:
        """Get information about all available tabs.

        Returns:
            List of tab info dictionaries
        """
        ordered = [
            TabType.BASIC,
            TabType.HARDWARE,
            TabType.MODEL,
            TabType.TRAINING,
            TabType.DATASETS,
            TabType.VALIDATION,
            TabType.PUBLISHING,
            TabType.CHECKPOINTS,
            TabType.JOB_QUEUE,
            TabType.ENVIRONMENTS,
            TabType.GIT_MIRROR,
            TabType.CLOUD,
            TabType.UI_SETTINGS,
        ]
        include_git = self._git_mirror_enabled()
        include_cloud = self._cloud_tab_enabled()
        tabs: List[Dict[str, str]] = []
        for tab_type in ordered:
            if tab_type == TabType.GIT_MIRROR and not include_git:
                continue
            if tab_type == TabType.CLOUD and not include_cloud:
                continue
            config = self._tab_configs.get(tab_type)
            if not config:
                continue
            tabs.append(
                {
                    "id": config.id,
                    "name": tab_type.value,
                    "title": config.title,
                    "icon": config.icon,
                    "description": config.description,
                }
            )
        return tabs
