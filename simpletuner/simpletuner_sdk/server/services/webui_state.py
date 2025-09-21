"""Web UI state persistence and onboarding management."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_WEBUI_CONFIG_ENV = "SIMPLETUNER_WEB_UI_CONFIG"
_XDG_HOME_ENV = "XDG_HOME"
_XDG_CONFIG_HOME_ENV = "XDG_CONFIG_HOME"


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

        base_candidate = (
            os.environ.get(_XDG_HOME_ENV)
            or os.environ.get(_XDG_CONFIG_HOME_ENV)
            or str(Path.home())
        )
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
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)

    def load_state(self) -> WebUIState:
        return WebUIState(
            defaults=self.load_defaults(),
            onboarding=self.load_onboarding(),
        )

    def load_defaults(self) -> WebUIDefaults:
        payload = self._read_json("defaults")
        if not payload:
            return WebUIDefaults()
        return WebUIDefaults(**{key: payload.get(key) for key in WebUIDefaults().__dict__.keys()})

    def save_defaults(self, defaults: WebUIDefaults) -> WebUIDefaults:
        self._write_json("defaults", asdict(defaults))
        return defaults

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
        payload = {
            "steps": {
                step_id: asdict(step_state)
                for step_id, step_state in onboarding.steps.items()
            }
        }
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
