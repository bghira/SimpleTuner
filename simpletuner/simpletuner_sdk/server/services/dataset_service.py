"""Dataset-related helpers shared across routes and services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config_store import ConfigStore
from .dataset_plan import DatasetPlanStore
from .webui_state import WebUIStateStore
from ..utils.paths import get_config_directory, get_simpletuner_root, resolve_config_path


def _format_dataset_path(path: Path) -> str:
    """Return a dataset path string relative to the project root when possible."""
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(get_simpletuner_root()))
    except ValueError:
        return str(resolved)


def build_data_backend_choices() -> List[Dict[str, str]]:
    """Collect available dataset configuration candidates for selection widgets."""

    entries: List[Dict[str, str]] = []
    seen_values: Set[str] = set()

    config_roots: Set[Path] = set()

    def _add_config_root(raw_path: Optional[Path]) -> None:
        if not raw_path:
            return
        try:
            candidate = Path(raw_path).expanduser().resolve(strict=False)
        except Exception:
            candidate = Path(raw_path).expanduser()
        config_roots.add(candidate)

    def _relative_to_config_roots(path: Path) -> str:
        for root in sorted(config_roots, key=lambda p: len(str(p)), reverse=True):
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        try:
            default_root = Path(get_config_directory()).expanduser().resolve(strict=False)
            return str(path.relative_to(default_root))
        except Exception:
            pass
        try:
            project_root = Path(get_simpletuner_root()).expanduser().resolve(strict=False)
            return str(path.relative_to(project_root))
        except Exception:
            pass
        return path.name

    def _extract_environment_name(path: Path) -> Optional[str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                metadata = data.get("_metadata")
                if isinstance(metadata, dict):
                    name = metadata.get("name")
                    if isinstance(name, str) and name.strip():
                        return name.strip()
        except Exception:
            return None
        return None

    def _register_path(
        path: Path,
        env_name: Optional[str] = None,
    ) -> None:
        if not path:
            return
        resolved = path.expanduser().resolve(strict=False)
        value = _format_dataset_path(resolved)
        if value in seen_values:
            return

        rel_path = _relative_to_config_roots(resolved)
        rel_path_path = Path(rel_path)
        parent_segment = rel_path_path.parent if rel_path_path.parent != Path('.') else None

        environment = env_name or _extract_environment_name(resolved) or resolved.stem
        if not env_name and parent_segment:
            environment = parent_segment.name

        entries.append(
            {
                "value": value,
                "environment": environment,
                "path": rel_path,
            }
        )
        seen_values.add(value)

    candidate_dirs: Set[Path] = set()

    def _add_candidate_dir(raw_path: Optional[Path]) -> None:
        if not raw_path:
            return
        try:
            candidate = Path(raw_path).expanduser()
        except Exception:
            return

        try:
            resolved_candidate = candidate.resolve(strict=False)
        except Exception:
            resolved_candidate = candidate

        candidate_dirs.add(resolved_candidate)
        _add_config_root(resolved_candidate)

    # Include user-configured directories from WebUI defaults
    try:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            base = Path(defaults.configs_dir).expanduser()
            _add_candidate_dir(base)
            _add_candidate_dir(base / "dataloaders")
    except Exception:
        pass

    # Include ConfigStore-managed dataloader configs and directory
    try:
        dataloader_store = ConfigStore(config_type="dataloader")
        _add_candidate_dir(Path(dataloader_store.config_dir))
        for metadata in dataloader_store.list_configs():
            name = metadata.get("name")
            if not name:
                continue
            try:
                path = dataloader_store._get_config_path(name)
            except Exception:
                continue
            if not path or not path.exists():
                continue

            _register_path(path, env_name=str(name))
    except Exception:
        pass

    # Include dataset plan store path (global plan)
    try:
        dataset_plan_store = DatasetPlanStore()
        _add_candidate_dir(dataset_plan_store.path.parent)
        if dataset_plan_store.path.exists():
            _register_path(dataset_plan_store.path)
    except Exception:
        pass

    # Include paths referenced by active configuration
    try:
        model_store = ConfigStore()
        _add_candidate_dir(Path(model_store.config_dir))
        active_name = model_store.get_active_config()
        if active_name:
            config, _ = model_store.load_config(active_name)
            backend_path = config.get("--data_backend_config") or config.get("data_backend_config")
            if backend_path:
                resolved = resolve_config_path(backend_path, config_dir=model_store.config_dir, check_cwd_first=True)
                if resolved and resolved.exists():
                    _register_path(resolved)
                    _add_candidate_dir(resolved.parent)
    except Exception:
        pass

    # Always include the default config directory under the SimpleTuner installation
    try:
        _add_candidate_dir(Path(get_config_directory()))
    except Exception:
        pass

    # Discover dataset configs within the candidate directories
    for directory in sorted(candidate_dirs):
        if not directory or not directory.exists():
            continue
        try:
            for path in directory.glob("**/multidatabackend*.json"):
                _register_path(path)
        except Exception:
            continue

    if not entries:
        return []

    max_env_len = max(len(entry["environment"]) for entry in entries)
    max_path_len = max(len(entry["path"]) for entry in entries)

    for entry in entries:
        entry["label"] = f"{entry['environment']:<{max_env_len}} | {entry['path']}"

    entries.sort(key=lambda item: item["environment"].lower())
    return entries
