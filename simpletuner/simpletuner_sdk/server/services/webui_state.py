"""Web UI state persistence and onboarding management."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from simpletuner.simpletuner_sdk.server.utils.assets import get_asset_version

_WEBUI_CONFIG_ENV = "SIMPLETUNER_WEB_UI_CONFIG"
_XDG_HOME_ENV = "XDG_HOME"
_XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"

_ALLOWED_ACCELERATE_CLI_KEYS = {
    "--num_processes",
    "--num_machines",
    "--main_process_ip",
    "--main_process_port",
    "--machine_rank",
    "--same_network",
    "--accelerate_extra_args",
    "--dynamo_backend",
}
_ALLOWED_ACCELERATE_META_KEYS = {"mode", "device_ids", "manual_count"}
_INT_ACCELERATE_KEYS = {
    "--num_processes",
    "--num_machines",
    "--main_process_port",
    "--machine_rank",
}
_BOOL_ACCELERATE_KEYS = {"--same_network"}


def _normalise_accelerate_overrides(raw: Any) -> Dict[str, Any]:
    """Convert persisted accelerate overrides into a cleaned mapping."""

    if not raw:
        return {}

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        payload = raw
    else:
        return {}

    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, ""):
            continue

        if not isinstance(key, str):
            key = str(key)
        normalized_key = key.strip()
        if not normalized_key:
            continue

        if normalized_key.startswith("--"):
            cli_key = normalized_key
        else:
            cli_key = f"--{normalized_key.lstrip('-')}"

        if cli_key in _ALLOWED_ACCELERATE_CLI_KEYS:
            if cli_key in _INT_ACCELERATE_KEYS:
                try:
                    cleaned[cli_key] = int(value)
                except (TypeError, ValueError):
                    continue
                continue

            if cli_key in _BOOL_ACCELERATE_KEYS:
                if isinstance(value, bool):
                    cleaned[cli_key] = value
                elif isinstance(value, str):
                    cleaned[cli_key] = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    cleaned[cli_key] = bool(value)
                continue

            cleaned[cli_key] = str(value).strip()
            continue

        if normalized_key in _ALLOWED_ACCELERATE_META_KEYS:
            if normalized_key == "mode":
                if isinstance(value, str):
                    normalized_value = value.strip().lower()
                    if normalized_value in {"auto", "manual", "disabled", "hardware"}:
                        cleaned[normalized_key] = normalized_value
                continue

            if normalized_key == "manual_count":
                try:
                    cleaned[normalized_key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
                continue

            if normalized_key == "device_ids":
                parsed: list[int] = []
                if isinstance(value, str):
                    tokens = [token.strip() for token in value.split(",") if token.strip()]
                    for token in tokens:
                        try:
                            parsed.append(int(token))
                        except ValueError:
                            continue
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        try:
                            parsed.append(int(item))
                        except (TypeError, ValueError):
                            continue
                if parsed:
                    seen = set()
                    ordered: list[int] = []
                    for device_id in parsed:
                        if device_id in seen:
                            continue
                        seen.add(device_id)
                        ordered.append(device_id)
                    cleaned[normalized_key] = ordered
                continue

    if cleaned.get("mode") == "hardware":
        cleaned.pop("--num_processes", None)

    return cleaned


@dataclass
class OnboardingStepState:
    """State for a single onboarding step."""

    completed_version: int = 0
    completed_at: Optional[str] = None
    value: Optional[str] = None


@dataclass
class WebUIOnboardingState:
    """Collection of onboarding step states."""

    steps: Dict[str, OnboardingStepState] = field(default_factory=dict)


@dataclass
class WebUIDefaults:
    """Persistent defaults used by the Web UI."""

    output_dir: Optional[str] = None
    configs_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    active_config: Optional[str] = None
    theme: str = "dark"
    event_polling_interval: int = 5
    event_stream_enabled: bool = True
    auto_preserve_defaults: bool = True
    allow_dataset_paths_outside_dir: bool = False
    show_documentation_links: bool = True
    accelerate_overrides: Dict[str, Any] = field(default_factory=dict)
    git_mirror_enabled: bool = False
    git_remote: Optional[str] = None
    git_branch: Optional[str] = None
    git_auto_commit: bool = False
    git_require_clean: bool = False
    git_push_on_snapshot: bool = False
    git_include_untracked: bool = False
    sync_onboarding_defaults: bool = False
    onboarding_sync_opt_out: list[str] = field(default_factory=list)
    cloud_tab_enabled: bool = True
    cloud_webhook_url: Optional[str] = None
    cloud_outputs_dir: Optional[str] = None
    cloud_dataloader_hint_dismissed: bool = False
    cloud_git_hint_dismissed: bool = False
    # Data consent for cloud uploads: "ask" (default), "allow", or "deny"
    cloud_data_consent: str = "ask"
    # Polling preference: None (auto), True (force enable), False (force disable)
    cloud_job_polling_enabled: Optional[bool] = None
    # Administration section (user management) - can be disabled if not using multi-user auth
    admin_tab_enabled: bool = True
    # List of dismissed admin hints (overview, users, levels, rules, quotas, approvals, auth, orgs)
    admin_dismissed_hints: list[str] = field(default_factory=list)
    # Metrics tab settings
    metrics_tab_enabled: bool = True
    # Prometheus export: disabled by default until user enables
    metrics_prometheus_enabled: bool = False
    # Categories to export: jobs, http, rate_limits, approvals, audit, health, circuit_breakers, provider
    metrics_prometheus_categories: list[str] = field(default_factory=lambda: ["jobs", "http"])
    # Tensorboard (placeholder, always False for now)
    metrics_tensorboard_enabled: bool = False
    # Dismissed metrics hints (for hero CTA)
    metrics_dismissed_hints: list[str] = field(default_factory=list)
    # Credential security settings
    # Threshold (in days) after which credentials are flagged as "stale"
    credential_rotation_threshold_days: int = 90
    # Enable early warning for credentials approaching the stale threshold
    credential_early_warning_enabled: bool = False
    # Early warning triggers at this % of threshold (e.g., 75 = warn at 67.5 days if threshold is 90)
    credential_early_warning_percent: int = 75
    # Track if credential security has been configured (for enterprise onboarding)
    credential_security_configured: bool = False
    credential_security_skipped: bool = False
    # UI sound settings
    sounds_enabled: bool = True
    sounds_volume: int = 50  # 0-100
    # Per-category sound toggles
    sounds_success_enabled: bool = True
    sounds_error_enabled: bool = True
    sounds_warning_enabled: bool = True
    sounds_info_enabled: bool = True
    # Easter egg: retro hover sound (opt-in, default off)
    sounds_retro_hover_enabled: bool = False
    # Audit export configuration for SIEM integration
    audit_export_format: str = "json"  # json or csv
    audit_export_webhook_url: Optional[str] = None
    audit_export_auth_token: Optional[str] = None
    audit_export_security_only: bool = False
    # Public registration settings
    # Whether public registration is enabled (admin can always create users)
    public_registration_enabled: bool = False
    # Default permission level for newly registered users
    public_registration_default_level: str = "researcher"
    # Local GPU concurrency settings
    # Maximum number of GPUs that can be in use by local jobs (None = all available)
    local_gpu_max_concurrent: Optional[int] = None
    # Maximum number of local jobs that can run simultaneously
    local_job_max_concurrent: int = 1


@dataclass
class WebUIState:
    """Aggregate of UI defaults and onboarding state."""

    defaults: WebUIDefaults = field(default_factory=WebUIDefaults)
    onboarding: WebUIOnboardingState = field(default_factory=WebUIOnboardingState)


class WebUIStateStore:
    """Persist Web UI state across sessions."""

    def __init__(self, base_dir: Optional[Path | str] = None):
        if base_dir is not None:
            self.base_dir = Path(base_dir).expanduser()
        else:
            self.base_dir = self._resolve_base_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_base_dir(self) -> Path:
        override = os.environ.get(_WEBUI_CONFIG_ENV)
        if override:
            return Path(override).expanduser()

        base_candidate = os.environ.get(_XDG_HOME_ENV) or os.environ.get(_XDG_CONFIG_HOME_ENV)
        if base_candidate:
            root = Path(base_candidate).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return root / "webui"

        candidate_roots = []
        if Path("/workspace").exists():
            candidate_roots.append(Path("/workspace/simpletuner"))
        if Path("/notebooks").exists():
            candidate_roots.append(Path("/notebooks/simpletuner"))
        candidate_roots.append(Path.home() / ".simpletuner")

        for root in candidate_roots:
            webui_dir = root / "webui"
            if webui_dir.exists():
                return webui_dir

        # Nothing pre-existing; create the first preferred root and return its webui directory
        preferred_root = candidate_roots[0]
        preferred_root.mkdir(parents=True, exist_ok=True)
        webui_dir = preferred_root / "webui"
        webui_dir.mkdir(parents=True, exist_ok=True)
        return webui_dir

    def _category_path(self, category: str) -> Path:
        safe_name = category.replace("/", "_")
        return self.base_dir / f"{safe_name}.json"

    def _read_json(self, category: str) -> Dict[str, object]:
        path = self._category_path(category)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"Failed to read web UI state '{category}': {error}") from error

    def _write_json(self, category: str, data: Dict[str, object]) -> None:
        path = self._category_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name,
            suffix=".tmp",
        ) as tmp_handle:
            json.dump(data, tmp_handle, indent=2, sort_keys=True)
            tmp_handle.write("\n")
            tmp_path = Path(tmp_handle.name)

        try:
            tmp_path.replace(path)
        except FileNotFoundError:
            # Another writer may have already moved the temporary file; fall back to direct write
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load_state(self) -> WebUIState:
        return WebUIState(
            defaults=self.load_defaults(),
            onboarding=self.load_onboarding(),
        )

    def load_defaults(self) -> WebUIDefaults:
        payload = self._read_json("defaults")
        if not payload:
            return WebUIDefaults()
        data: Dict[str, Any] = {}
        for key in WebUIDefaults().__dict__.keys():
            if key == "accelerate_overrides":
                data[key] = _normalise_accelerate_overrides(payload.get(key))
            else:
                data[key] = payload.get(key)
        defaults = WebUIDefaults(**data)

        # Normalise theme selection
        theme = (defaults.theme or "dark").lower().strip()
        defaults.theme = theme if theme in {"dark", "tron"} else "dark"

        # Ensure polling interval is a positive integer
        try:
            interval = int(defaults.event_polling_interval)
        except (TypeError, ValueError):
            interval = 5
        defaults.event_polling_interval = max(1, interval)

        # Normalise event stream toggle
        stream_value = defaults.event_stream_enabled
        if isinstance(stream_value, str):
            defaults.event_stream_enabled = stream_value.strip().lower() in {"1", "true", "yes", "on"}
        elif stream_value is None:
            defaults.event_stream_enabled = True
        else:
            defaults.event_stream_enabled = bool(stream_value)

        # Normalise auto preserve defaults toggle
        sentinel = object()
        auto_preserve_value = payload.get("auto_preserve_defaults", sentinel)
        if auto_preserve_value is sentinel:
            defaults.auto_preserve_defaults = True
        elif isinstance(auto_preserve_value, str):
            normalized = auto_preserve_value.strip().lower()
            defaults.auto_preserve_defaults = normalized not in {"0", "false", "no", "off"}
        elif auto_preserve_value is None:
            defaults.auto_preserve_defaults = True
        else:
            defaults.auto_preserve_defaults = bool(auto_preserve_value)

        defaults.accelerate_overrides = _normalise_accelerate_overrides(defaults.accelerate_overrides)

        # Normalise allow_dataset_paths_outside_dir toggle
        allow_paths_value = payload.get("allow_dataset_paths_outside_dir", False)
        if isinstance(allow_paths_value, str):
            defaults.allow_dataset_paths_outside_dir = allow_paths_value.strip().lower() in {"1", "true", "yes", "on"}
        elif allow_paths_value is None:
            defaults.allow_dataset_paths_outside_dir = False
        else:
            defaults.allow_dataset_paths_outside_dir = bool(allow_paths_value)

        # Normalise show_documentation_links toggle
        sentinel = object()
        show_docs_value = payload.get("show_documentation_links", sentinel)
        if show_docs_value is sentinel:
            defaults.show_documentation_links = True
        elif isinstance(show_docs_value, str):
            defaults.show_documentation_links = show_docs_value.strip().lower() not in {"0", "false", "no", "off"}
        elif show_docs_value is None:
            defaults.show_documentation_links = True
        else:
            defaults.show_documentation_links = bool(show_docs_value)

        # Normalise git preferences
        defaults.git_mirror_enabled = bool(payload.get("git_mirror_enabled", False))
        defaults.git_auto_commit = bool(payload.get("git_auto_commit", False))
        defaults.git_require_clean = bool(payload.get("git_require_clean", False))
        defaults.git_push_on_snapshot = bool(payload.get("git_push_on_snapshot", False))
        defaults.git_include_untracked = bool(payload.get("git_include_untracked", False))
        defaults.sync_onboarding_defaults = bool(payload.get("sync_onboarding_defaults", False))
        opt_out_raw = payload.get("onboarding_sync_opt_out", [])
        if isinstance(opt_out_raw, str):
            opt_out_raw = [opt_out_raw]
        if isinstance(opt_out_raw, (list, tuple, set)):
            cleaned = []
            for item in opt_out_raw:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if candidate and candidate not in cleaned:
                    cleaned.append(candidate)
            defaults.onboarding_sync_opt_out = cleaned
        else:
            defaults.onboarding_sync_opt_out = []

        git_remote = payload.get("git_remote")
        defaults.git_remote = git_remote.strip() if isinstance(git_remote, str) and git_remote.strip() else None
        git_branch = payload.get("git_branch")
        defaults.git_branch = git_branch.strip() if isinstance(git_branch, str) and git_branch.strip() else None

        # Normalise cloud hint dismissal booleans
        defaults.cloud_dataloader_hint_dismissed = bool(payload.get("cloud_dataloader_hint_dismissed", False))
        defaults.cloud_git_hint_dismissed = bool(payload.get("cloud_git_hint_dismissed", False))

        # Normalise cloud data consent
        consent_value = payload.get("cloud_data_consent")
        if isinstance(consent_value, str) and consent_value in {"ask", "allow", "deny"}:
            defaults.cloud_data_consent = consent_value
        else:
            defaults.cloud_data_consent = "ask"

        # Normalise cloud job polling enabled
        polling_value = payload.get("cloud_job_polling_enabled")
        if polling_value is None:
            defaults.cloud_job_polling_enabled = None
        elif isinstance(polling_value, bool):
            defaults.cloud_job_polling_enabled = polling_value
        elif isinstance(polling_value, str):
            defaults.cloud_job_polling_enabled = polling_value.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.cloud_job_polling_enabled = None

        # Normalise admin tab enabled (default True, but can be disabled if no auth configured)
        admin_tab_value = payload.get("admin_tab_enabled")
        if admin_tab_value is None:
            defaults.admin_tab_enabled = True
        else:
            defaults.admin_tab_enabled = bool(admin_tab_value)

        # Load admin dismissed hints
        admin_hints = payload.get("admin_dismissed_hints")
        if isinstance(admin_hints, list):
            defaults.admin_dismissed_hints = [h for h in admin_hints if isinstance(h, str)]
        else:
            defaults.admin_dismissed_hints = []

        # Normalise metrics tab enabled
        metrics_tab_value = payload.get("metrics_tab_enabled")
        if metrics_tab_value is None:
            defaults.metrics_tab_enabled = True
        else:
            defaults.metrics_tab_enabled = bool(metrics_tab_value)

        # Normalise Prometheus export enabled
        prometheus_enabled = payload.get("metrics_prometheus_enabled")
        if prometheus_enabled is None:
            defaults.metrics_prometheus_enabled = False
        elif isinstance(prometheus_enabled, bool):
            defaults.metrics_prometheus_enabled = prometheus_enabled
        elif isinstance(prometheus_enabled, str):
            defaults.metrics_prometheus_enabled = prometheus_enabled.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.metrics_prometheus_enabled = bool(prometheus_enabled)

        # Normalise Prometheus categories
        valid_categories = {"jobs", "http", "rate_limits", "approvals", "audit", "health", "circuit_breakers", "provider"}
        prometheus_categories = payload.get("metrics_prometheus_categories")
        if isinstance(prometheus_categories, list):
            defaults.metrics_prometheus_categories = [
                c for c in prometheus_categories if isinstance(c, str) and c in valid_categories
            ]
        else:
            defaults.metrics_prometheus_categories = ["jobs", "http"]

        # Tensorboard is always False for now (placeholder)
        defaults.metrics_tensorboard_enabled = False

        # Load metrics dismissed hints
        metrics_hints = payload.get("metrics_dismissed_hints")
        if isinstance(metrics_hints, list):
            defaults.metrics_dismissed_hints = [h for h in metrics_hints if isinstance(h, str)]
        else:
            defaults.metrics_dismissed_hints = []

        # Normalise credential rotation threshold (must be positive integer, 30-365 days)
        rotation_threshold = payload.get("credential_rotation_threshold_days")
        if rotation_threshold is not None:
            try:
                threshold_int = int(rotation_threshold)
                defaults.credential_rotation_threshold_days = max(30, min(365, threshold_int))
            except (TypeError, ValueError):
                defaults.credential_rotation_threshold_days = 90
        else:
            defaults.credential_rotation_threshold_days = 90

        # Normalise early warning enabled flag
        early_warning_enabled = payload.get("credential_early_warning_enabled")
        if early_warning_enabled is None:
            defaults.credential_early_warning_enabled = False
        elif isinstance(early_warning_enabled, bool):
            defaults.credential_early_warning_enabled = early_warning_enabled
        elif isinstance(early_warning_enabled, str):
            defaults.credential_early_warning_enabled = early_warning_enabled.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.credential_early_warning_enabled = bool(early_warning_enabled)

        # Normalise early warning percent (50-95% range)
        early_warning_percent = payload.get("credential_early_warning_percent")
        if early_warning_percent is not None:
            try:
                pct = int(early_warning_percent)
                defaults.credential_early_warning_percent = max(50, min(95, pct))
            except (TypeError, ValueError):
                defaults.credential_early_warning_percent = 75
        else:
            defaults.credential_early_warning_percent = 75

        # Normalise credential security onboarding flags
        cred_configured = payload.get("credential_security_configured")
        if isinstance(cred_configured, bool):
            defaults.credential_security_configured = cred_configured
        elif isinstance(cred_configured, str):
            defaults.credential_security_configured = cred_configured.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.credential_security_configured = False

        cred_skipped = payload.get("credential_security_skipped")
        if isinstance(cred_skipped, bool):
            defaults.credential_security_skipped = cred_skipped
        elif isinstance(cred_skipped, str):
            defaults.credential_security_skipped = cred_skipped.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.credential_security_skipped = False

        # Normalise UI sound settings
        sounds_enabled = payload.get("sounds_enabled")
        if sounds_enabled is None:
            defaults.sounds_enabled = True
        elif isinstance(sounds_enabled, bool):
            defaults.sounds_enabled = sounds_enabled
        elif isinstance(sounds_enabled, str):
            defaults.sounds_enabled = sounds_enabled.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.sounds_enabled = bool(sounds_enabled)

        # Normalise volume (0-100)
        sounds_volume = payload.get("sounds_volume")
        if sounds_volume is not None:
            try:
                vol = int(sounds_volume)
                defaults.sounds_volume = max(0, min(100, vol))
            except (TypeError, ValueError):
                defaults.sounds_volume = 50
        else:
            defaults.sounds_volume = 50

        # Normalise per-category sound toggles
        for category in ("success", "error", "warning", "info"):
            key = f"sounds_{category}_enabled"
            value = payload.get(key)
            if value is None:
                setattr(defaults, key, True)
            elif isinstance(value, bool):
                setattr(defaults, key, value)
            elif isinstance(value, str):
                setattr(defaults, key, value.lower() in {"true", "1", "yes", "on"})
            else:
                setattr(defaults, key, bool(value))

        # Normalise retro hover sound (opt-in, default off)
        retro_hover = payload.get("sounds_retro_hover_enabled")
        if retro_hover is None:
            defaults.sounds_retro_hover_enabled = False
        elif isinstance(retro_hover, bool):
            defaults.sounds_retro_hover_enabled = retro_hover
        elif isinstance(retro_hover, str):
            defaults.sounds_retro_hover_enabled = retro_hover.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.sounds_retro_hover_enabled = bool(retro_hover)

        # Normalise audit export settings
        export_format = payload.get("audit_export_format")
        if isinstance(export_format, str) and export_format in {"json", "csv"}:
            defaults.audit_export_format = export_format
        else:
            defaults.audit_export_format = "json"

        webhook_url = payload.get("audit_export_webhook_url")
        defaults.audit_export_webhook_url = (
            webhook_url.strip() if isinstance(webhook_url, str) and webhook_url.strip() else None
        )

        auth_token = payload.get("audit_export_auth_token")
        defaults.audit_export_auth_token = auth_token.strip() if isinstance(auth_token, str) and auth_token.strip() else None

        security_only = payload.get("audit_export_security_only")
        if security_only is None:
            defaults.audit_export_security_only = False
        elif isinstance(security_only, bool):
            defaults.audit_export_security_only = security_only
        elif isinstance(security_only, str):
            defaults.audit_export_security_only = security_only.lower() in {"true", "1", "yes", "on"}
        else:
            defaults.audit_export_security_only = bool(security_only)

        # Normalise local GPU concurrency settings
        local_gpu_max = payload.get("local_gpu_max_concurrent")
        if local_gpu_max is None:
            defaults.local_gpu_max_concurrent = None
        else:
            try:
                val = int(local_gpu_max)
                defaults.local_gpu_max_concurrent = max(1, val) if val > 0 else None
            except (TypeError, ValueError):
                defaults.local_gpu_max_concurrent = None

        local_job_max = payload.get("local_job_max_concurrent")
        if local_job_max is None:
            defaults.local_job_max_concurrent = 1
        else:
            try:
                defaults.local_job_max_concurrent = max(1, int(local_job_max))
            except (TypeError, ValueError):
                defaults.local_job_max_concurrent = 1

        defaults.active_config = self._validate_active_config(defaults.active_config, defaults.configs_dir)
        return defaults

    def _validate_active_config(self, active_config: Optional[str], configs_dir: Optional[str]) -> Optional[str]:
        if not active_config:
            return None

        candidate = active_config.strip()
        if not candidate:
            return None

        configs_path = Path(configs_dir).expanduser() if configs_dir else None

        if configs_path and (configs_path / candidate / "config.json").exists():
            return candidate

        # Lazy import to avoid circular dependency
        from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore

        config_store = ConfigStore(config_dir=configs_dir, config_type="model")
        try:
            config_store.load_config(candidate)
            return candidate
        except FileNotFoundError:
            return None

    def save_defaults(self, defaults: WebUIDefaults) -> WebUIDefaults:
        self._write_json("defaults", asdict(defaults))
        return defaults

    def _fallback_paths(self) -> Dict[str, str]:
        root_dir = self.base_dir.parent
        root_dir.mkdir(parents=True, exist_ok=True)
        return {
            "configs_dir": str(root_dir / "config"),
            "output_dir": str(root_dir / "output"),
            "datasets_dir": str(root_dir / "datasets"),
        }

    def resolve_defaults(self, defaults: WebUIDefaults) -> Dict[str, Any]:
        raw = asdict(defaults)
        resolved = raw.copy()
        fallbacks = self._fallback_paths()

        configs_dir = resolved.get("configs_dir")
        if configs_dir:
            resolved["configs_dir__source"] = "stored"
        else:
            resolved["configs_dir"] = fallbacks["configs_dir"]
            resolved["configs_dir__source"] = "fallback"

        output_dir = resolved.get("output_dir")
        if output_dir:
            resolved["output_dir__source"] = "stored"
        else:
            resolved["output_dir"] = fallbacks["output_dir"]
            resolved["output_dir__source"] = "fallback"

        datasets_dir = resolved.get("datasets_dir")
        if datasets_dir:
            resolved["datasets_dir__source"] = "stored"
        else:
            resolved["datasets_dir"] = fallbacks["datasets_dir"]
            resolved["datasets_dir__source"] = "fallback"

        # Provide a cache-busting token for static assets
        resolved["asset_version"] = get_asset_version()

        return {
            "raw": raw,
            "resolved": resolved,
            "fallbacks": fallbacks,
        }

    def get_defaults_bundle(self) -> Dict[str, Dict[str, Any]]:
        defaults = self.load_defaults()
        return self.resolve_defaults(defaults)

    def load_onboarding(self) -> WebUIOnboardingState:
        payload = self._read_json("onboarding")
        steps: Dict[str, OnboardingStepState] = {}
        raw_steps = payload.get("steps", {}) if isinstance(payload, dict) else {}
        for step_id, step_payload in raw_steps.items():
            if not isinstance(step_payload, dict):
                continue
            steps[step_id] = OnboardingStepState(
                completed_version=int(step_payload.get("completed_version", 0) or 0),
                completed_at=step_payload.get("completed_at"),
                value=step_payload.get("value"),
            )
        return WebUIOnboardingState(steps=steps)

    def save_onboarding(self, onboarding: WebUIOnboardingState) -> WebUIOnboardingState:
        payload = {"steps": {step_id: asdict(step_state) for step_id, step_state in onboarding.steps.items()}}
        self._write_json("onboarding", payload)
        return onboarding

    def record_onboarding_step(
        self,
        step_id: str,
        version: int,
        value: Optional[str] = None,
    ) -> OnboardingStepState:
        onboarding = self.load_onboarding()
        step_state = onboarding.steps.get(step_id, OnboardingStepState())
        if version > step_state.completed_version:
            step_state.completed_version = version
            step_state.completed_at = datetime.now(timezone.utc).isoformat()
        step_state.value = value
        onboarding.steps[step_id] = step_state
        self.save_onboarding(onboarding)
        return step_state

    def load_ui_state(self) -> Dict[str, Any]:
        """Load UI element states (collapsed sections, etc.)."""
        return self._read_json("ui_state")

    def save_ui_state(self, state: Dict[str, Any]) -> None:
        """Save UI element states (collapsed sections, etc.)."""
        self._write_json("ui_state", state)

    def get_collapsed_sections(self, tab_name: str) -> Dict[str, bool]:
        """Get collapsed state for sections in a specific tab."""
        ui_state = self.load_ui_state()
        collapsed = ui_state.get("collapsed_sections", {})
        return collapsed.get(tab_name, {})

    def save_collapsed_sections(self, tab_name: str, sections: Dict[str, bool]) -> None:
        """Save collapsed state for sections in a specific tab."""
        ui_state = self.load_ui_state()
        if "collapsed_sections" not in ui_state:
            ui_state["collapsed_sections"] = {}
        ui_state["collapsed_sections"][tab_name] = sections
        self.save_ui_state(ui_state)
