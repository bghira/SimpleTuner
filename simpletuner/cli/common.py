"""
Shared utilities for SimpleTuner CLI.

Contains config discovery, process management, version helpers, and formatting utilities.
"""

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import List, Optional

CONFIG_FILENAMES = {
    "json": "config.json",
    "toml": "config.toml",
    "env": "config.env",
}


def get_version() -> str:
    """Get SimpleTuner version without importing heavy dependencies."""
    try:
        from importlib.metadata import version

        return version("simpletuner")
    except Exception:
        pass

    # Fallback: read from __init__.py directly
    try:
        import ast
        from pathlib import Path

        init_file = Path(__file__).parent.parent / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            for node in ast.parse(content).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__version__":
                            if isinstance(node.value, ast.Constant):
                                return str(node.value.value)
    except Exception:
        pass

    return "unknown"


# --- Config Discovery ---


def _find_webui_state_file(filename: str) -> Optional[Path]:
    candidates: list[Path] = []

    override = os.environ.get("SIMPLETUNER_WEB_UI_CONFIG")
    if override:
        candidates.append(Path(override).expanduser() / filename)

    base_candidate = os.environ.get("XDG_HOME") or os.environ.get("XDG_CONFIG_HOME")
    if base_candidate:
        candidates.append(Path(base_candidate).expanduser() / "webui" / filename)

    for root in (Path("/workspace/simpletuner"), Path("/notebooks/simpletuner"), Path.home() / ".simpletuner"):
        candidates.append(root / "webui" / filename)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return None


def _get_webui_configs_dir() -> Optional[Path]:
    defaults_path = _find_webui_state_file("defaults.json")
    if defaults_path:
        with defaults_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        configs_dir = payload.get("configs_dir")
        if configs_dir:
            return Path(str(configs_dir)).expanduser()

    onboarding_path = _find_webui_state_file("onboarding.json")
    if onboarding_path:
        with onboarding_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        steps = payload.get("steps", {})
        if isinstance(steps, dict):
            step_payload = steps.get("default_configs_dir", {})
            if isinstance(step_payload, dict):
                configs_dir = step_payload.get("value")
                if configs_dir:
                    return Path(str(configs_dir)).expanduser()

    return None


def find_config_file() -> Optional[str]:
    """Find config file in current directory or config/ subdirectory."""
    for config_name in ["config.json", "config.toml", "config.env"]:
        if os.path.exists(config_name):
            return config_name

    config_dir = Path("config")
    if config_dir.exists():
        for config_name in ["config.json", "config.toml", "config.env"]:
            config_path = config_dir / config_name
            if config_path.exists():
                return str(config_path)

    return None


def _extract_cli_override(arguments: List[str], option_names: tuple[str, ...]) -> Optional[str]:
    """Extract the value for a CLI override like --config_backend=value."""
    for arg in arguments:
        for name in option_names:
            prefix = f"--{name}"
            if arg.startswith(prefix):
                if len(arg) == len(prefix):
                    return None
                if arg[len(prefix)] == "=":
                    return arg[len(prefix) + 1 :]
    return None


def _candidate_config_paths(env: str, backend_override: Optional[str], config_path_override: Optional[str]) -> List[Path]:
    """List possible configuration files for an explicit environment."""
    backend = (backend_override or "").lower() or None
    if backend and backend not in CONFIG_FILENAMES:
        return []

    filenames = [CONFIG_FILENAMES[backend]] if backend else list(CONFIG_FILENAMES.values())

    candidates: list[Path] = []

    def _add(path: Path) -> None:
        expanded = path.expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

    if config_path_override:
        override = Path(config_path_override).expanduser()
        if override.suffix:
            _add(override)
        else:
            for name in filenames:
                _add(override / name)
                suffix = Path(name).suffix
                if suffix:
                    _add(override.with_suffix(suffix))
        return candidates

    env_path = Path(env).expanduser()
    if env_path.suffix:
        _add(env_path)
        if not env_path.is_absolute():
            _add(Path.cwd() / env_path)
        return candidates

    search_roots: List[Path] = []

    if env_path.is_absolute():
        search_roots.append(env_path)
    else:
        search_roots.append(env_path)
        search_roots.append(Path.cwd() / env_path)
        search_roots.append(Path("config") / env_path)

        home_configs = Path.home() / "config"
        search_roots.append(home_configs / env_path)

        config_dir_override = os.environ.get("SIMPLETUNER_CONFIG_DIR")
        if config_dir_override:
            search_roots.append(Path(config_dir_override).expanduser() / env_path)

        if env_path.parts and env_path.parts[0] == "examples":
            package_root = Path(__file__).resolve().parent.parent
            search_roots.append(package_root / env_path)

    for root in search_roots:
        for name in filenames:
            _add(root / name)

    return candidates


def _validate_environment_config(env: str, backend_override: Optional[str], config_path_override: Optional[str]) -> None:
    """Ensure an explicit environment points to an existing configuration file."""
    if not env or env == "default":
        return

    backend = backend_override.lower() if backend_override else None
    if backend == "cmd":
        return

    candidate_paths = _candidate_config_paths(env, backend, config_path_override)
    existing = [path for path in candidate_paths if path.is_file()]

    if existing:
        return

    if not candidate_paths:
        raise FileNotFoundError(f"No configuration candidates were produced for environment '{env}'.")

    checked = "\n  - ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Configuration for environment '{env}' not found. Checked:\n  - {checked}")


# --- Process Management ---


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate process group with SIGTERM."""
    if os.name != "nt" and getattr(process, "pid", None):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return
        except (ProcessLookupError, PermissionError):
            pass
    process.terminate()


def _kill_process_group(process: subprocess.Popen) -> None:
    """Force kill process group with SIGKILL."""
    if os.name != "nt" and getattr(process, "pid", None):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    process.kill()


# --- Formatting ---


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable format."""
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_cost(cost: Optional[float]) -> str:
    """Format cost in USD."""
    if cost is None or cost == 0:
        return "-"
    return f"${cost:.2f}"
