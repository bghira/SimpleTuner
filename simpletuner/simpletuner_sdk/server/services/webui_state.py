"""Web UI state persistence and onboarding management."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
                    if normalized_value in {"auto", "manual", "disabled"}:
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
    accelerate_overrides: Dict[str, Any] = field(default_factory=dict)


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

        base_candidate = os.environ.get(_XDG_HOME_ENV) or os.environ.get(_XDG_CONFIG_HOME_ENV) or str(Path.home())
        return Path(base_candidate).expanduser() / ".simpletuner" / "webui"

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
        home_dir = Path.home()
        return {
            "configs_dir": str(home_dir / ".simpletuner" / "configs"),
            "output_dir": str(home_dir / ".simpletuner" / "output"),
            "datasets_dir": str(home_dir / ".simpletuner" / "datasets"),
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
